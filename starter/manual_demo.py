"""手动点选 + 失败处理演示。

    python manual_demo.py --data data --out figures        # 遮挡 -> 自动失败 -> 手动路径恢复
    python manual_demo.py --data data --picture x.png --meta y.json --interactive

默认 demo 把**自动检测到的四个中心**传给 solve_task1_manual，用来验证“手动路径与自动路径
共用同一套下游几何”，不等于真人点击精度；真人鼠标点选请用 `--interactive`。
题目原文允许“从手动点选开始，再改为自动检测”；本文件是手动的那一半，
同时留下“自动那一半会明确失败、而不是悄悄给错值”的证据。
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
from solution import solve_task1, solve_task1_manual


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else np.count_nonzero(a & b) / u


def occlude_marker(img, u, v, radius=26):
    """用该标记周围的桌面颜色把标记涂掉。"""
    out = img.copy()
    fill = np.median(img[max(int(v) - radius - 12, 0):int(v) + radius + 12,
                         max(int(u) - radius - 12, 0):int(u) + radius + 12].reshape(-1, 3), 0)
    cv2.circle(out, (int(round(u)), int(round(v))), radius, fill.astype(np.uint8).tolist(), -1,
               cv2.LINE_AA)
    return out


def demo(data, out, sample="surface_square_036"):
    root = Path(data) / "task1" / "val"
    meta = json.loads((root / "inputs" / f"{sample}.json").read_text(encoding="utf-8"))
    lab = json.loads((root / "labels" / f"{sample}.json").read_text(encoding="utf-8"))
    gt = V.imread(root / "labels" / lab["mask"], 0)
    img = V.imread(root / meta["image"])

    # 1) 原图：自动检测正常
    px = V.detect_markers(img)
    ref = solve_task1(img, meta)
    q = np.array(meta["queries_px"], float)
    err_auto = float(np.linalg.norm(V.apply_h(np.array(ref["H_img_to_mm"]), q)
                                    - np.array(lab["queries_mm"]), axis=1).mean())

    # 2) 涂掉红色标记 —— 自动路径必须拒绝给出结果
    red = px["red"]
    bad = occlude_marker(img, red[0], red[1])
    auto_status, auto_reason = "ok", ""
    try:
        r2 = solve_task1(bad, meta)
        err_bad = float(np.linalg.norm(V.apply_h(np.array(r2["H_img_to_mm"]), q)
                                       - np.array(lab["queries_mm"]), axis=1).mean())
    except Exception as exc:                       # noqa: BLE001 - 记下失败原因
        auto_status = "failed"
        auto_reason = f"{type(exc).__name__}: {exc}"
        err_bad = float("nan")

    # 3) 在同一张受损图上走手动路径；默认用自动中心模拟点选（不是真人点击精度）
    man = solve_task1_manual(bad, meta, px)
    err_man = float(np.linalg.norm(V.apply_h(np.array(man["H_img_to_mm"]), q)
                                   - np.array(lab["queries_mm"]), axis=1).mean())

    print(f"[{sample}] auto(clean) pt={err_auto:.3f} mm | "
          f"auto(occluded) = {auto_status} {auto_reason} | "
          f"manual-path(simulated points) pt={err_man:.3f} mm")
    print(f"  manual-path IoU vs truth = {iou(man['mask_array'], gt):.4f}")

    # 4) 三联图
    h, w = img.shape[:2]
    panels = []
    a = img.copy()
    for c, (u, v) in px.items():
        cv2.circle(a, (int(u), int(v)), 10, (255, 255, 255), 2, cv2.LINE_AA)
    _label(a, f"(a) 自动检测成功，点误差 {err_auto:.3f} mm")
    panels.append(a)

    b = bad.copy()
    cv2.circle(b, (int(red[0]), int(red[1])), 30, (0, 0, 255), 2, cv2.LINE_AA)
    _label(b, "(b) 涂掉红标记 → 自动路径显式失败", (0, 0, 255))
    cv2.putText(b, auto_reason[:52], (18, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(b, auto_reason[:52], (18, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)
    panels.append(b)

    c_ = bad.copy()
    for name, (u, v) in px.items():
        cv2.drawMarker(c_, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cs, _ = cv2.findContours(man["mask_array"].astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(c_, cs, -1, (0, 255, 255), 2)
    _label(c_, f"(c) 手动路径恢复（自动中心模拟点选），点误差 {err_man:.3f} mm")
    panels.append(c_)

    sheet = np.full((h, w * 3 + 16, 3), 25, np.uint8)
    for i, p in enumerate(panels):
        sheet[:, i * (w + 8):i * (w + 8) + w] = p
    small = cv2.resize(sheet, None, fx=0.55, fy=0.55)
    out.mkdir(parents=True, exist_ok=True)
    V.imwrite(out / "failure_manual_recovery.jpg", small)
    print("figure ->", out / "failure_manual_recovery.jpg")

    return dict(sample=sample, auto_clean_pt_mm=err_auto, auto_occluded=auto_status,
                auto_reason=auto_reason, manual_pt_mm=err_man,
                manual_iou=iou(man["mask_array"], gt))


def _label(img, text, col=(0, 255, 255)):
    cn.banner(img, text, col, 22)


def interactive(path, meta_path):
    img = V.imread(path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    clicks = []
    names = [m["color"] for m in meta["markers"]]
    win = "click red, green, blue, yellow (ESC to abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x), float(y)))

    cv2.setMouseCallback(win, on_click)
    while True:
        vis = img.copy()
        for i, (u, v) in enumerate(clicks):
            cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, names[i], (int(u) + 12, int(v) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(clicks) == 4:
            cv2.waitKey(300)
            break
    cv2.destroyAllWindows()
    if len(clicks) != 4:
        raise SystemExit("aborted")
    res = solve_task1_manual(img, meta, dict(zip(names, clicks)))
    print("H_img_to_mm =", np.round(res["H_img_to_mm"], 6).tolist())
    print("corners/centre =", res.get("corners_px") or res.get("center_px"))


def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--sample", default="surface_square_036")
    ap.add_argument("--picture")
    ap.add_argument("--meta")
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args()
    if a.interactive:
        interactive(a.picture, a.meta)
        return
    res = demo(a.data, Path(a.out), a.sample)
    (Path(a.out) / "manual_demo.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
