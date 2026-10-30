"""误差最大的样本，以及逐例的误差归因。

    python error_cases.py --data data --split val --out figures

题目要求"展示至少 3 个误差较大的样本并分析原因"。由于本方案的验证集误差全在
亚毫米量级，没有离群样本，更诚实的做法是把误差拆开：

  * 总长误差 =（主轴角度）+（光栅化 / 渲染偏差）
  * 杆径误差 =（主轴角度）+（光栅化）
  拆分办法：用生成器给定的真实主轴方向，重跑同一套测量。
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


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def measure_with_axis(mask, H, u_dir):
    """同一套 L/d 测量，但把主轴方向固定为给定值。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    cmm = V.apply_h(H, cont)
    t = cmm @ u_dir
    n = np.array([-u_dir[1], u_dir[0]])
    s = cmm @ n
    L = float(t.max() - t.min())
    lo = np.percentile(np.abs(s[t < t.min() + 0.22 * L] - np.median(s)), 95)
    hi = np.percentile(np.abs(s[t > t.max() - 0.22 * L] - np.median(s)), 95)
    head_low = lo > hi
    a0, a1 = ((t.min() + 0.38 * L, t.min() + 0.94 * L) if head_low
              else (t.max() - 0.94 * L, t.max() - 0.38 * L))
    sel = (t > a0) & (t < a1)
    smid = np.median(s[sel])
    lines = []
    for grp in (sel & (s > smid), sel & (s <= smid)):
        pts = cmm[grp]
        c = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - c)
        nn = np.array([-Vt[0, 1], Vt[0, 0]])
        bb = float(nn @ c)
        if nn @ n < 0:
            nn, bb = -nn, -bb
        lines.append((nn, bb, c))
    nn = lines[0][0] + lines[1][0]
    nn /= np.linalg.norm(nn)
    b1 = lines[0][1] + (nn - lines[0][0]) @ lines[0][2]
    b2 = lines[1][1] + (nn - lines[1][0]) @ lines[1][2]
    return L, abs(b1 - b2)


def task2_cases(data, split, out, topn=3):
    root = Path(data) / "task2" / split
    rows = []
    for ip in sorted((root / "inputs").glob("*.json")):
        meta = json.loads(ip.read_text(encoding="utf-8"))
        lab = json.loads((root / "labels" / ip.name).read_text(encoding="utf-8"))
        img = V.imread(root / meta["image"])
        res = solve_task2(img, meta)
        H = np.array(res["H_img_to_mm"])
        u_ours = np.array(res["axis_px"], float)
        u_ours /= np.linalg.norm(u_ours)
        u_gt = np.array([np.cos(np.deg2rad(lab["axis_deg"])),
                         np.sin(np.deg2rad(lab["axis_deg"]))])
        L_gtaxis, d_gtaxis = measure_with_axis(res["mask_array"], H, u_gt)
        L_gtaxis += V.LENGTH_BIAS_MM
        d_gtaxis += V.DIAMETER_BIAS_MM
        ang_err = float(np.rad2deg(np.arccos(np.clip(abs(u_ours @ u_gt), -1, 1))))
        rows.append(dict(sid=meta["sample_id"],
                         L=res["L_mm"], d=res["d_mm"],
                         L_err=res["L_mm"] - lab["L_mm"], d_err=res["d_mm"] - lab["d_mm"],
                         L_err_axis=res["L_mm"] - L_gtaxis,
                         L_err_rest=L_gtaxis - lab["L_mm"],
                         d_err_axis=res["d_mm"] - d_gtaxis,
                         d_err_rest=d_gtaxis - lab["d_mm"],
                         axis_err_deg=ang_err, label=lab, meta=meta, img=img, res=res))
    rows.sort(key=lambda r: -max(abs(r["L_err"]), abs(r["d_err"]) / 3.0))
    picked = rows[:topn]
    print(f"{'sample':12s} {'L_err':>7s} {' axis':>7s} {' rest':>7s} | "
          f"{'d_err':>7s} {' axis':>7s} {' rest':>7s} | {'axis°':>6s}")
    for r in picked:
        print(f"{r['sid']:12s} {r['L_err']:+7.3f} {r['L_err_axis']:+7.3f} {r['L_err_rest']:+7.3f} | "
              f"{r['d_err']:+7.3f} {r['d_err_axis']:+7.3f} {r['d_err_rest']:+7.3f} | "
              f"{r['axis_err_deg']:6.2f}")

    tiles = []
    for r in picked:
        vis = V.imread(Path(data) / "task2" / split / r["meta"]["image"]).copy()
        H = np.array(r["res"]["H_img_to_mm"])
        cs, _ = cv2.findContours(r["res"]["mask_array"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
        _banner(vis, f"{r['sid']}   ΔL={r['L_err']:+.3f}   Δd={r['d_err']:+.3f} mm"
                     f"   主轴误差 {r['axis_err_deg']:.2f}°")
        tiles.append(cv2.resize(vis, (640, 480)))
    sheet = np.full((480, 644 * len(tiles), 3), 25, np.uint8)
    for i, t in enumerate(tiles):
        sheet[:, i * 644:i * 644 + 640] = t
    out.mkdir(parents=True, exist_ok=True)
    V.imwrite(out / "task2_worst_cases.jpg", sheet)

    clean = [{k: v for k, v in r.items() if k not in ("label", "meta", "img", "res")}
             for r in picked]
    (out / "task2_worst_cases.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return picked


def task1_cases(data, split, out, topn=3):
    root = Path(data) / "task1" / split
    rows = []
    for ip in sorted((root / "inputs").glob("*.json")):
        meta = json.loads(ip.read_text(encoding="utf-8"))
        lab = json.loads((root / "labels" / ip.name).read_text(encoding="utf-8"))
        img = V.imread(root / meta["image"])
        res = solve_task1(img, meta)
        q = np.array(meta["queries_px"], float)
        e = np.linalg.norm(V.apply_h(np.array(res["H_img_to_mm"]), q)
                           - np.array(lab["queries_mm"]), axis=1)
        rows.append(dict(sid=meta["sample_id"], shape=res["shape"],
                         pt=float(e.mean()), pt_max=float(e.max()),
                         iou=iou(res["mask_array"], V.imread(root / "labels" / lab["mask"], 0)),
                         img=img, meta=meta, res=res))
    rows.sort(key=lambda r: -r["pt"])
    picked = rows[:topn]
    print(f"\n{'sample':22s} {'shape':7s} {'ptMAE':>7s} {'ptMax':>7s} {'IoU':>7s}")
    for r in picked:
        print(f"{r['sid']:22s} {r['shape']:7s} {r['pt']:7.3f} {r['pt_max']:7.3f} {r['iou']:7.4f}")
    tiles = []
    for r in picked:
        vis = r["img"].copy()
        cs, _ = cv2.findContours(r["res"]["mask_array"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
        _banner(vis, f"{r['sid']}   点误差 {r['pt']:.3f} mm   IoU={r['iou']:.4f}")
        tiles.append(cv2.resize(vis, (640, 480)))
    sheet = np.full((480, 644 * len(tiles), 3), 25, np.uint8)
    for i, t in enumerate(tiles):
        sheet[:, i * 644:i * 644 + 640] = t
    out.mkdir(parents=True, exist_ok=True)
    V.imwrite(out / "task1_worst_cases.jpg", sheet)
    return picked


def _banner(img, text):
    cn.banner(img, text, (0, 255, 255), 20)


def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../pack/data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    task1_cases(a.data, a.split, Path(a.out))
    task2_cases(a.data, a.split, Path(a.out))


if __name__ == "__main__":
    main()
