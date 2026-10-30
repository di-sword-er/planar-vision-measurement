"""扩展（进阶）：真实项目数据上的分割演示。

    python real_demo.py --data data --out real_figures

两个互不相关的演示，都不主张任何毫米真值：

  * 真实全图   —— 找到彩色桌垫（方桌）或圆盘并勾出轮廓
  * 真实螺栓裁剪图 —— 分割螺栓轮廓，并与既有审核掩膜比对（仅作轮廓参考）

每条结果都带状态，成功与失败按与正式题相同的方式计数。
"""
from __future__ import annotations

import argparse
import json
import sys

from ioutil import make_stdout_safe
from pathlib import Path

import cv2
import numpy as np

import vision_lib as V


# ------------------------------------------------------------------ 真实全图
def find_target(img):
    """对最大的强色彩区域返回 (掩膜, 形状, 附加信息)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    m = ((s > 90) & (v > 55)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        raise RuntimeError("no coloured region")
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
    comp = order[0]
    area = int(st[comp, cv2.CC_STAT_AREA])
    if area < 0.01 * img.shape[0] * img.shape[1]:
        raise RuntimeError(f"coloured region too small ({area} px)")
    region = V.fill_holes((lab == comp).astype(np.uint8))
    how_many = int(sum(st[i, cv2.CC_STAT_AREA] > 0.01 * img.shape[0] * img.shape[1]
                       for i in order))

    # 是四边形还是椭圆？数凸包上的角点 —— 用面积比不可靠，因为颜色区域
    # 本身被腐蚀过一圈，面积比会被带偏。
    # 用面积比不可靠：颜色区域本身被腐蚀过一圈，面积比会被带偏。
    cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(cont)
    peri = cv2.arcLength(hull, True)
    ap = cv2.approxPolyDP(hull, 0.02 * peri, True).reshape(-1, 2).astype(float)
    (_, _), (w, h), ang = cv2.minAreaRect(cont)
    is_quad = False
    if len(ap) == 4:
        angs = []
        for k in range(4):
            u = ap[k - 1] - ap[k]
            v = ap[(k + 1) % 4] - ap[k]
            u /= max(np.linalg.norm(u), 1e-9)
            v /= max(np.linalg.norm(v), 1e-9)
            angs.append(np.rad2deg(np.arccos(np.clip(u @ v, -1, 1))))
        is_quad = all(55 <= a <= 125 for a in angs)

    if is_quad:
        corners = V.quad_corners(region)
        mask = V.polygon_mask(img.shape, corners)
        # 合理性检查：凸，且大致是方的
        q = corners.astype(np.float32)
        if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)):
            raise RuntimeError("quadrilateral not convex")
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        ratio = sides.max() / max(sides.min(), 1e-6)
        if ratio > 1.6:
            raise RuntimeError(f"side ratio {ratio:.2f} not square-like")
        return mask, "square", dict(corners=corners.tolist(), side_ratio=float(ratio),
                                    n_targets=how_many, area_px=area)
    cen, semi, ang = V.fit_ellipse(region)
    mask = V.ellipse_mask(img.shape, cen, semi, ang)
    ratio = max(semi) / max(min(semi), 1e-6)
    if ratio > 3.5:
        raise RuntimeError(f"ellipse axis ratio {ratio:.2f} not disc-like")
    return mask, "disk", dict(center_px=list(cen), semi_axes_px=list(semi),
                              axis_ratio=float(ratio), n_targets=how_many,
                              area_px=area)


def demo_frames(data, out, limit=None):
    root = data / "real_project" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], []
    for p in sorted(root.glob("*.jpg"))[:limit]:
        img = V.imread(p)
        rec = dict(sample_id=p.stem, status="failed")
        vis = img.copy()
        try:
            mask, shape, meta = find_target(img)
            rec.update(status="ok", shape=shape, **meta)
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (0, 255, 255), 4)
            rec["mask_px"] = int((mask > 0).sum())
            cv2.putText(vis, f"{p.stem}  {shape}  "
                             f"targets_in_frame={meta['n_targets']}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(vis, f"{p.stem}  {shape}  "
                             f"targets_in_frame={meta['n_targets']}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception as exc:                       # noqa: BLE001 - 记失败但不中断
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            cv2.putText(vis, f"{p.stem}  FAILED: {exc}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
        V.imwrite(out / f"frame_{p.stem}.jpg", cv2.resize(vis, None, fx=0.5, fy=0.5))
        rows.append(rec)
    (out / "frames_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in rows if r["status"] == "ok"]
    sq = sum(r["shape"] == "square" for r in ok)
    print(f"real frames: {len(ok)}/{len(rows)} ok  (square {sq}, disk {len(ok) - sq})")
    for r in rows:
        if r["status"] != "ok":
            print("   fail", r["sample_id"], r.get("reason"))
    return rows


# ------------------------------------------------------------- 螺栓裁剪图
def crop_bolt_mask(img, tau=25):
    """无标定路径：裁剪图里既没有标定板也没有标记，只剩"到图像边缘色的距离"。
    这里故意不复用评分路径 —— 它是"没有标定时能做到什么程度"的诚实地板。"""
    return V.fill_holes(V.largest_component(V._border_distance_mask(img, tau)))


def demo_crops(data, out):
    root = data / "real_project" / "bolt_crops"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((root / "images").glob("*.png")):
        img = V.imread(p)
        gt = V.imread(root / "masks" / p.name, 0)
        rec = dict(sample_id=p.stem, status="failed")
        try:
            m = crop_bolt_mask(img)
            inter = int(((m > 0) & (gt > 0)).sum())
            union = int(((m > 0) | (gt > 0)).sum())
            rec.update(status="ok", iou=inter / max(union, 1))
        except Exception as exc:                       # noqa: BLE001
            rec["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
    ious = [r["iou"] for r in rows if r["status"] == "ok"]
    a = np.array(ious)
    summary = dict(
        n=len(rows), ok=len(ious), failed=len(rows) - len(ious),
        mean_iou=float(a.mean()), median_iou=float(np.median(a)),
        frac_above_0_9=float((a > 0.9).mean()), frac_below_0_5=float((a < 0.5).mean()),
        note="audit masks are a contour reference only; no metric truth exists",
    )
    (out / "crops_results.json").write_text(
        json.dumps(dict(summary=summary, per_sample=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("real bolt crops:", json.dumps(summary, ensure_ascii=False))

    rows.sort(key=lambda r: -r.get("iou", -1))
    for tag, group in (("best", rows[:6]), ("worst", rows[-6:])):
        tiles = []
        for r in group:
            p = root / "images" / f"{r['sample_id']}.png"
            img = V.imread(p)
            gt = V.imread(root / "masks" / f"{r['sample_id']}.png", 0)
            vis = img.copy()
            try:
                m = crop_bolt_mask(img)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cs, -1, (0, 255, 0), 2)
            except Exception:                          # noqa: BLE001 - 记失败，不中断
                pass
            cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (255, 0, 255), 1)
            lab = f"{r['sample_id'][-3:]}  IoU={r.get('iou', 0):.3f}"
            cv2.putText(vis, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(vis, (240, 240)))
        if tiles:
            sheet = np.full((240, 6 * 244, 3), 30, np.uint8)
            for i, t in enumerate(tiles):
                sheet[:, i * 244:i * 244 + 240] = t
            V.imwrite(out / f"crops_{tag}.jpg", sheet)
    return summary


def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../pack/data")
    ap.add_argument("--out", default="real_figures")
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out)
    demo_frames(data, out / "frames")
    demo_crops(data, out / "crops")


if __name__ == "__main__":
    main()
