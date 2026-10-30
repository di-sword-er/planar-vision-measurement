"""两道视觉题的图像处理核心。

处理流程
--------
标记检测 -> 单应 -> 桌面/螺栓几何 -> 最小二乘精化

全部计算都在四个彩色标记所定义的毫米平面上完成；唯一的"全局参数"是本文件底部的
两个渲染偏差常数（在 144 张带标注的合成样本上拟合，见 REPORT.md 第 6 章）。
"""
from __future__ import annotations

import numpy as np
import cv2

# ---------------------------------------------------------------- 常量
# 四个印刷标记的精确 BGR（从发放包图像上实测得到）
PALETTE_BGR = {
    "red": (40, 45, 220),
    "green": (45, 190, 40),
    "blue": (220, 70, 35),
    "yellow": (25, 210, 235),
}
# HSV 色相窗口（OpenCV 的 H 取 0..180），只在调色板匹配失败时兜底使用
HUE_WINDOW = {"red": [(0, 10), (170, 180)], "green": [(40, 80)],
              "blue": [(100, 135)], "yellow": [(18, 38)]}

SQUARE_SIDE_MM = 550.0
DISK_RADIUS_MM = 300.0

# 渲染偏差标定（加在原始测量值上）。光栅化后的轮廓比生成器的名义尺寸
# 系统性偏长、偏细；两个偏置在 train+val 共 144 张带标注样本上拟合
# （见 REPORT.md 第 6.1 节）。
LENGTH_BIAS_MM = -0.28
DIAMETER_BIAS_MM = 0.05

# 轮廓边落在"螺栓填充色"与"板色"之间的位置，取两者间距的百分比。
# 渲染器在真实轮廓外侧还画了一圈浅色描边，所以边界并不在中点：取 0.44
# 把阈值压到描边之下。该系数在 128 张带标注螺栓图上标定，0.42~0.46 之间
# IoU 都是 0.9990，是个平台而不是尖峰；曝光变化时依然成立（见报告 6.5 节）。
SILHOUETTE_EDGE_FRACTION = 0.44

MARKER_AREA_RANGE = (40, 900)


# -------------------------------------------------------------------- 读写
def imread(path, flags=cv2.IMREAD_COLOR):
    """兼容非 ASCII 路径的 cv2.imread（Windows 上直接用 imread 会读失败）。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite(path, img, quality=None):
    """默认写 PNG；后缀是 .jpg 时写 JPEG（用于照片类叠加图）。"""
    path = str(path)
    if quality is not None or path.lower().endswith((".jpg", ".jpeg")):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality or 88)])
    else:
        ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok


# --------------------------------------------------------------- 标记检测
def _marker_mask(img, color, mode="palette"):
    if mode == "palette":
        target = np.array(PALETTE_BGR[color], np.int16)
        return ((np.abs(img.astype(np.int16) - target).sum(2)) < 60).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    win = np.zeros(h.shape, bool)
    for lo, hi in HUE_WINDOW[color]:
        win |= (h >= lo) & (h <= hi)
    # 相对门限，保证兜底路径在曝光变化下仍然可用
    smax = float(np.percentile(s, 99.5))
    vmax = float(np.percentile(v, 99.5))
    return ((s > max(90.0, 0.55 * smax)) & (v > max(70.0, 0.45 * vmax)) & win).astype(np.uint8)


def detect_markers(img):
    """返回 {颜色: (u, v)}，即四个印刷标记的像素中心。

    两遍尝试：(1) 精确调色板匹配，程序生成的合成图走这一遍；
    (2) 通用 HSV + 形状匹配，其余情况兜底。每一遍都按颜色取最像圆形的连通域，
    再校验四个质心是否构成合法的凸四边形。
    """
    for mode in ("palette", "hsv"):
        try:
            found = _markers_once(img, mode)
        except RuntimeError:
            continue
        if len(found) == 4 and _validate_quad(found):
            return found
    raise RuntimeError("four markers not found / not consistent")


def _markers_once(img, mode):
    out = {}
    for color in PALETTE_BGR:
        if mode == "palette":
            m = _marker_mask(img, color, "palette")
        else:
            m = _marker_mask(img, color, "hsv")
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        best, best_score = None, -1.0
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            w = st[i, cv2.CC_STAT_WIDTH]
            h = st[i, cv2.CC_STAT_HEIGHT]
            if not (MARKER_AREA_RANGE[0] <= area <= MARKER_AREA_RANGE[1]):
                continue
            fill = area / float(max(w * h, 1))
            aspect = min(w, h) / float(max(w, h))
            score = fill * aspect * aspect
            if score > best_score:
                best, best_score = (float(ce[i][0]), float(ce[i][1])), score
        if best is None:
            raise RuntimeError(f"marker {color} missing")
        out[color] = best
    return out


def _validate_quad(found):
    """把四个质心排序，并要求它们构成凸的、近似正方形的四边形。"""
    pts = np.array([found[c] for c in ("red", "green", "blue", "yellow")], float)
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    q = pts[order]
    area = cv2.contourArea(q.astype(np.float32))
    if area < 500:
        return False
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    return sides.max() / max(sides.min(), 1e-6) < 4.0


# ------------------------------------------------------------ 单应矩阵
def _normaliser(pts):
    c = pts.mean(0)
    s = np.sqrt(2.0) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]]), (pts - c) * s


def homography(src, dst):
    """归一化 DLT。n=4 时为精确解，n>4 时自动变成最小二乘。"""
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    Ts, s = _normaliser(src)
    Td, d = _normaliser(dst)
    A = np.zeros((2 * len(s), 9))
    for i, ((x, y), (X, Y)) in enumerate(zip(s, d)):
        A[2 * i] = [-x, -y, -1, 0, 0, 0, X * x, X * y, X]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]
    _, _, Vt = np.linalg.svd(A)
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_h(H, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def refine_homography(H0, px, mm, extra=None, radial=None, w_extra=1.0, iters=12):
    """对 H 的 8 个自由参数做高斯-牛顿迭代。

    extra  : (图像点, 毫米点) 额外的一一对应点（方桌四个角点）
    radial : 必须落在半径 DISK_RADIUS_MM 的圆上的图像点（圆盘边界）
    """
    h = (H0 / H0[2, 2]).ravel()[:8].copy()

    def build(h8):
        return np.append(h8, 1.0).reshape(3, 3)

    def resid(h8):
        H = build(h8)
        parts = [(apply_h(H, px) - mm).ravel()]
        if extra is not None and len(extra[0]):
            parts.append(w_extra * (apply_h(H, extra[0]) - extra[1]).ravel())
        if radial is not None and len(radial):
            parts.append(w_extra * (np.linalg.norm(apply_h(H, radial), axis=1) - DISK_RADIUS_MM))
        return np.concatenate(parts)

    r = resid(h)
    lam = 1e-6
    for _ in range(iters):
        J = np.zeros((r.size, 8))
        for k in range(8):
            dh = np.zeros(8)
            step = 1e-6 * (abs(h[k]) + 1e-3)
            dh[k] = step
            J[:, k] = (resid(h + dh) - r) / step
        dx = np.linalg.solve(J.T @ J + lam * np.eye(8), -(J.T @ r))
        h = h + dx
        r = resid(h)
        if np.linalg.norm(dx) < 1e-12:
            break
    return build(h)


# ------------------------------------------------------------- 几何工具
def fill_holes(mask):
    pad = np.pad((mask > 0).astype(np.uint8), 1)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    return (pad | (flood == 0).astype(np.uint8))[1:-1, 1:-1]


def largest_component(binary):
    n, lab, st, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if n < 2:
        raise RuntimeError("empty binary image")
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8)


def surface_mask(img):
    """取与图像边缘颜色差异最大的连通域（即桌面 / 标定板）。"""
    h, w = img.shape[:2]
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, 3)
    bg = np.median(border, 0).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    bgl = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
    dist = np.linalg.norm(lab - bgl, axis=2)
    m = (dist > 18).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fill_holes(largest_component(m))


def _tls_line(pts):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c)
    d = Vt[0]
    n = np.array([-d[1], d[0]])
    return n, float(n @ c)


def quad_corners(mask, iters=4):
    """桌面四边形的亚像素角点。

    轮廓没有抗锯齿，所以每条边用"穿过数百个阶梯边界点的总体最小二乘直线"
    来定位；平均之后可以消掉 ±0.5 px 的光栅化噪声。
    """
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    hull = cv2.convexHull(cont.astype(np.int32)).reshape(-1, 2).astype(float)
    peri = cv2.arcLength(hull.astype(np.int32), True)
    v = None
    for eps in (0.02, 0.04, 0.08):
        ap = cv2.approxPolyDP(hull.astype(np.int32), eps * peri, True).reshape(-1, 2).astype(float)
        if len(ap) == 4:
            v = ap
            break
    if v is None:
        raise RuntimeError("surface is not a quadrilateral")
    for _ in range(iters):
        lines = []
        for k in range(4):
            a, b = v[k], v[(k + 1) % 4]
            d = b - a
            d /= max(np.linalg.norm(d), 1e-9)
            n = np.array([-d[1], d[0]])
            sel = np.abs((hull - a) @ n) < 2.5
            lines.append(_tls_line(hull[sel]) if sel.sum() > 10 else _tls_line(np.array([a, b])))
        new = []
        for k in range(4):
            (n1, b1), (n2, b2) = lines[k - 1], lines[k]
            A = np.array([n1, n2])
            if abs(np.linalg.det(A)) < 1e-9:
                new.append(v[k])
                continue
            new.append(np.linalg.solve(A, np.array([b1, b2])))
        v = np.array(new)
    return v


def boundary_points(mask, n=400):
    """最大连通域的等间隔重采样外边界。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(cont) > n:
        idx = np.linspace(0, len(cont) - 1, n).round().astype(int)
        cont = cont[idx]
    return cont


def fit_ellipse(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea)
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(cont.astype(np.float32))
    return (float(cx), float(cy)), (d1 / 2.0, d2 / 2.0), float(ang)


def ellipse_mask(shape, cen, semi, ang, n=2000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th = np.deg2rad(ang)
    x, y = semi[0] * np.cos(t), semi[1] * np.sin(t)
    u = cen[0] + x * np.cos(th) - y * np.sin(th)
    v = cen[1] + x * np.sin(th) + y * np.cos(th)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.c_[u, v]).astype(np.int32)], 1)
    return m


def polygon_mask(shape, pts, shift=6):
    m = np.zeros(shape[:2], np.uint8)
    q = np.round(np.asarray(pts, float) * (1 << shift)).astype(np.int32)
    cv2.fillPoly(m, [q], 1, shift=shift)
    return m


# ---------------------------------------------------------------- 螺栓
def _otsu(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _bimodal_mid(values, min_class=50):
    """取两个主要灰度级之间的中点作为阈值。

    只用 Otsu 会随噪声漂移；取两类中位数的中点正好落在真实边缘上（也就是亚像素
    边缘应当在的位置），并且只随曝光移动、不随噪声移动。
    """
    t0 = _otsu(values)
    lo = values[values <= t0]
    hi = values[values > t0]
    if lo.size < min_class or hi.size < min_class:
        return t0
    return 0.5 * (float(np.median(lo)) + float(np.median(hi)))


def board_mask(img):
    """亮板：在"背景"与"板"两个众数之间取中点作为阈值。"""
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = (v > _bimodal_mid(v.ravel())).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fill_holes(largest_component(m))


def bolt_mask(img):
    """螺栓掩膜。

    主路径（已标定的标定板图像）：在亮板内部，螺栓属于暗类 —— 两个阈值都由图像
    自身的灰度分布导出，所以分割对曝光不敏感，也不依赖发放包的绝对灰度。
    兜底路径（无标定的照片）：取与图像边缘颜色差异最大的连通域。
    """
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = np.zeros(img.shape[:2], np.uint8)
    try:
        board = board_mask(img)
        vals = v[board > 0].ravel()
        if vals.size > 500:
            t0 = _otsu(vals)
            lo, hi = vals[vals <= t0], vals[vals > t0]
            if lo.size > 50 and hi.size > 50:
                t = float(np.median(lo)) + SILHOUETTE_EDGE_FRACTION * (
                    float(np.median(hi)) - float(np.median(lo)))
            else:
                t = t0
            cand = cv2.morphologyEx(((v < t) & (board > 0)).astype(np.uint8),
                                    cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = fill_holes(largest_component(cand))
    except (RuntimeError, cv2.error):
        m = np.zeros(img.shape[:2], np.uint8)
    if not (200 <= int(m.sum()) <= 0.60 * m.size) or _aspect(m) < 1.5:
        m = fill_holes(largest_component(_border_distance_mask(img)))
    if int(m.sum()) < 200:
        raise RuntimeError("bolt silhouette too small")
    if _aspect(m) < 1.35:
        raise RuntimeError(f"silhouette not elongated (aspect {_aspect(m):.2f})")
    return m


def _aspect(mask):
    """最小外接矩形长边与短边之比（1.0 表示各向同性团块）。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 1.0
    (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    return max(w, h) / max(min(w, h), 1e-6)


def _border_distance_mask(img, tau=25):
    h, w = img.shape[:2]
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, 3)
    bg = np.median(border, 0).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    bgl = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
    m = (np.linalg.norm(lab - bgl, axis=2) > tau).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def measure_bolt(mask, H):
    """总长 L（含头部）与杆部直径 d，单位毫米。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    cmm = apply_h(H, cont.astype(float))

    # 1) 主轴初值：最小外接矩形的长边方向
    (_, _), (w, h), ang = cv2.minAreaRect(cont)
    ang = ang if w >= h else ang + 90.0
    best = None
    for a in np.arange(ang - 30.0, ang + 30.0, 0.2):
        t = np.deg2rad(a)
        u = np.array([np.cos(t), np.sin(t)])
        n = np.array([-u[1], u[0]])
        box = np.ptp(cmm @ u) * np.ptp(cmm @ n)
        if best is None or box < best[0]:
            best = (box, u, n)
    _, u, n = best

    # 2) 用杆部两条长边精化主轴，再在这两条线之间量 d
    d = float("nan")
    head_low = True
    for _ in range(6):
        t = cmm @ u
        s = cmm @ n
        L = t.max() - t.min()
        smid = np.median(s)
        low = np.percentile(np.abs(s[t < t.min() + 0.22 * L] - smid), 95)
        high = np.percentile(np.abs(s[t > t.max() - 0.22 * L] - smid), 95)
        head_low = bool(low > high)                      # 头部是比较宽的那一端
        a0, a1 = ((t.min() + 0.38 * L, t.min() + 0.94 * L) if head_low
                  else (t.max() - 0.94 * L, t.max() - 0.38 * L))
        sel = (t > a0) & (t < a1)
        if sel.sum() < 20:
            raise RuntimeError("shank band too small")
        sides = []
        for grp in (sel & (s > np.median(s[sel])), sel & (s <= np.median(s[sel]))):
            pts = cmm[grp]
            if len(pts) < 8:
                raise RuntimeError("shank side too short")
            c = pts.mean(0)
            _, _, Vt = np.linalg.svd(pts - c)
            nn = np.array([-Vt[0, 1], Vt[0, 0]])
            bb = float(nn @ c)
            if nn @ n < 0:
                nn, bb = -nn, -bb
            sides.append((nn, bb, c))
        nn = sides[0][0] + sides[1][0]
        nn /= np.linalg.norm(nn)
        b1 = sides[0][1] + (nn - sides[0][0]) @ sides[0][2]
        b2 = sides[1][1] + (nn - sides[1][0]) @ sides[1][2]
        d = abs(b1 - b2)
        un = np.array([nn[1], -nn[0]])
        if un @ u < 0:
            un = -un
        u, n = un, nn

    L = float(np.ptp(cmm @ u))
    return dict(L=L + LENGTH_BIAS_MM, d=d + DIAMETER_BIAS_MM,
                axis_deg=float(np.rad2deg(np.arctan2(u[1], u[0]))),
                head_at_low_end=head_low, axis_px=u, normal_px=n)
