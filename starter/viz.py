"""报告用的定性图。

    python viz.py --data data --split val --out figures [--lang zh]

每道题输出一张叠加图：拟合的模型、测量几何、以及查询点的映射结果；
另外拼一张拼接图。
"""
from __future__ import annotations

import argparse
import json
import sys

from ioutil import make_stdout_safe
from pathlib import Path

import cv2
import numpy as np

import cn
import vision_lib as V
from solution import solve_task1, solve_task2

LANG = "en"          # 由 --lang 设置

_ZH = {
    "square": "方桌", "disk": "圆盘",
    "title": "{sid}  {shape}  轮廓与真值对照",
    "centre": "平面圆心投影 (0,0)",
    "q": "({u:.0f},{v:.0f}) → ({x:.1f},{y:.1f}) mm",
    "t2": "{sid}   总长 L={L:.2f} mm   杆径 d={d:.2f} mm",
    "t2e": "   ΔL={dl:+.2f}  Δd={dd:+.2f} mm",
    "d_lab": "杆径 d={d:.2f} mm",
    "head": "头端", "tail": "尾端",
}
_EN = {
    "square": "square", "disk": "disk",
    "title": "{sid}  {shape}  mask vs truth",
    "centre": "projected centre (0,0)",
    "q": "({u:.0f},{v:.0f}) -> ({x:.1f},{y:.1f})mm",
    "t2": "{sid}   L={L:.2f} mm   d={d:.2f} mm",
    "t2e": "   dL={dl:+.2f}  dD={dd:+.2f} mm",
    "d_lab": "d={d:.2f} mm",
    "head": "L0", "tail": "L1",
}


def T(key, **kw):
    d = _ZH if LANG == "zh" else _EN
    return d[key].format(**kw)

COL = {"red": (60, 60, 235), "green": (60, 190, 60), "blue": (235, 120, 60),
       "yellow": (60, 220, 240)}


def put(img, text, org, col=(255, 255, 255), scale=0.55, box=False):
    if LANG == "zh" and cn.available():
        cn.text(img, text, org, col, max(int(scale * 34), 12), plate=box)
        return
    if box:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = org
        cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + base + 4), (25, 25, 25), -1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def fig_task1(img, meta, res, lab=None):
    vis = img.copy()
    mask = res["mask_array"].astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)

    H = np.array(res["H_img_to_mm"])
    Hi = np.linalg.inv(H)
    # 平面内坐标轴：X 向右（单位 mm），Y 向下（单位 mm）
    for vec, col, name in (([1, 0], (0, 0, 255), "X"), ([0, 1], (0, 200, 0), "Y")):
        p0 = V.apply_h(Hi, [[0.0, 0.0]])[0]
        p1 = V.apply_h(Hi, [np.array(vec, float) * 120.0])[0]
        a, b = tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int))
        cv2.arrowedLine(vis, a, b, col, 2, cv2.LINE_AA, tipLength=0.15)
        put(vis, name, (b[0] + 4, b[1] - 4), col)

    for m in meta["markers"]:
        u, v = _marker_px(img, m)
        cv2.drawMarker(vis, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        put(vis, f"{int(m['xy_mm'][0])},{int(m['xy_mm'][1])}", (int(u) + 10, int(v) - 8),
            COL[m["color"]], 0.5)

    q = np.array(meta.get("queries_px", []), float)
    if len(q):
        mm = V.apply_h(H, q)
        for k, ((u, v), (x, y)) in enumerate(zip(q, mm)):
            cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 0, 255), -1, cv2.LINE_AA)
            if k >= 3:
                continue
            put(vis, T("q", u=u, v=v, x=x, y=y), (int(u) + 6, int(v) + 6), (255, 0, 255), 0.40)
    if res.get("corners_px"):
        for i, (u, v) in enumerate(res["corners_px"]):
            cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 128, 255), 2, cv2.LINE_AA)
            put(vis, f"P{i}", (int(u) + 8, int(v) - 8), (0, 128, 255))
    else:
        c = res["center_px"]
        cv2.drawMarker(vis, (int(round(c[0])), int(round(c[1]))), (0, 128, 255),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        put(vis, T("centre"), (int(c[0]) + 10, int(c[1]) - 10), (0, 128, 255))
    title = T("title", sid=meta["sample_id"], shape=T(res["shape"]))
    put(vis, title, (14, 32), (255, 255, 255), 0.7, box=True)
    return vis


def _marker_px(img, m):
    """按 solution.py 的方式还原标记位置。"""
    return m.get("_px", (0, 0))


def fig_task2(img, meta, res, lab=None):
    vis = img.copy()
    mask = res["mask_array"].astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)

    H = np.array(res["H_img_to_mm"])
    Hi = np.linalg.inv(H)
    cont = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    cmm = V.apply_h(H, cont)
    ax = np.array(res["axis_px"], float)
    ax /= np.linalg.norm(ax)
    t = cmm @ ax
    # 沿螺栓画出的主轴直线（图像坐标）
    tmin, tmax = t.min(), t.max()
    nvec = np.array([-ax[1], ax[0]])
    off = np.median(cmm @ nvec) * nvec          # 让画出来的主轴落在螺栓上
    a_mm = ax * (tmin - 6.0) + off
    b_mm = ax * (tmax + 6.0) + off
    a_px = V.apply_h(Hi, [a_mm])[0]
    b_px = V.apply_h(Hi, [b_mm])[0]
    cv2.line(vis, tuple(np.round(a_px).astype(int)), tuple(np.round(b_px).astype(int)),
             (0, 0, 255), 2, cv2.LINE_AA)
    head_name, tail_name = ((T("head"), T("tail")) if res["head_at_low_end"]
                            else (T("tail"), T("head")))
    for p, name in ((a_px, head_name), (b_px, tail_name)):
        cv2.drawMarker(vis, tuple(np.round(p).astype(int)), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 2)
        put(vis, name, (int(p[0]) + 8, int(p[1]) + 6), (0, 0, 255))

    # 直径测量位置：取杆部中段
    n = np.array([-ax[1], ax[0]])
    s = cmm @ n
    tmid = 0.5 * (tmin + tmax) + (0.22 * (tmax - tmin) * (1 if res["head_at_low_end"] else -1))
    sel = np.abs(t - tmid) < 0.06 * (tmax - tmin)
    if sel.sum() >= 2:
        for sgn in (s[sel].min(), s[sel].max()):
            p = V.apply_h(Hi, [ax * tmid + n * sgn])[0]
            cv2.circle(vis, tuple(np.round(p).astype(int)), 5, (255, 0, 0), 2, cv2.LINE_AA)
        p1 = V.apply_h(Hi, [ax * tmid + n * s[sel].min()])[0]
        p2 = V.apply_h(Hi, [ax * tmid + n * s[sel].max()])[0]
        cv2.line(vis, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                 (255, 0, 0), 2, cv2.LINE_AA)
        put(vis, T("d_lab", d=res["d_mm"]), (int(p1[0]) + 8, int((p1[1] + p2[1]) / 2)),
            (255, 0, 0), 0.5)
    txt = T("t2", sid=meta["sample_id"], L=res["L_mm"], d=res["d_mm"])
    if lab is not None:
        txt += T("t2e", dl=res["L_mm"] - lab["L_mm"], dd=res["d_mm"] - lab["d_mm"])
    put(vis, txt, (14, 32), (255, 255, 255), 0.62, box=True)
    return vis


def contact_sheet(images, cols=2, pad=8):
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 30, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet[pad + r * (h + pad):pad + r * (h + pad) + im.shape[0],
              pad + c * (w + pad):pad + c * (w + pad) + im.shape[1]] = im
    return sheet


def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../pack/data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--samples", default="")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    data = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sheets = {}
    for task, solve, figf in (("task1", solve_task1, fig_task1),
                              ("task2", solve_task2, fig_task2)):
        root = data / task / a.split
        imgs = []
        files = sorted((root / "inputs").glob("*.json"))
        if a.samples:
            want = set(a.samples.split(","))
            files = [f for f in files if f.stem in want]
        else:
            files = files[:a.limit]
        for ip in files:
            meta = json.loads(ip.read_text(encoding="utf-8"))
            img = V.imread(root / meta["image"])
            mk = V.detect_markers(img)
            for m in meta["markers"]:
                m["_px"] = mk[m["color"]]
            res = solve(img, meta)
            labp = root / "labels" / ip.name
            lab = json.loads(labp.read_text(encoding="utf-8")) if labp.exists() else None
            vis = figf(img, meta, res, lab)
            V.imwrite(out / f"{task}_{meta['sample_id']}.png", vis)
            imgs.append(vis)
        sheets[task] = imgs
        print(f"{task}: {len(imgs)} figures")

    for task, imgs in sheets.items():
        half = imgs[:4]
        if half:
            small = [cv2.resize(i, None, fx=0.6, fy=0.6) for i in half]
            sheet = contact_sheet(small, cols=2)
            if sheet is not None:
                V.imwrite(out / f"sheet_{task}.png", sheet)


if __name__ == "__main__":
    main()
