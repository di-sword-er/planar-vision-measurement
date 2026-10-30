"""两道题的接口实现。

运行器只调用 `solve_task1` / `solve_task2`。推理阶段只允许读 `images/` 和
`inputs/`，不会打开任何标签文件。

自动标记检测可以换成手动点选（题目原文允许"从手动点选开始，再改为自动检测"），
见 `solve_task1_manual` / `solve_task2_manual`。四个标记点之后的所有几何计算与
自动路径完全共用，不存在第二套代码。
"""
from __future__ import annotations

import cv2
import numpy as np

import vision_lib as V


# --------------------------------------------------------------- 题一核心
def _task1_core(img, meta, markers):
    """从四个标记像素开始的一切计算。`markers` 为 {颜色: (u, v)}。"""
    shape = meta.get("shape_hint", "square")
    mm = np.array([m["xy_mm"] for m in meta["markers"]], float)
    px = np.array([markers[m["color"]] for m in meta["markers"]], float)
    H = V.homography(px, mm)

    fg = V.surface_mask(img)
    if fg.sum() < 2000:
        raise RuntimeError("surface region too small")

    out = {"shape": shape}
    if shape == "square":
        corners = V.quad_corners(fg)
        mmc = V.apply_h(H, corners)
        side = V.SQUARE_SIDE_MM
        ideal = np.array([[0, 0], [side, 0], [side, side], [0, side]], float)
        idx = [int(np.argmin(np.linalg.norm(mmc - p, axis=1))) for p in ideal]
        if len(set(idx)) != 4:
            raise RuntimeError("corner/P0..P3 assignment is ambiguous")
        corners = corners[idx]
        mask = V.polygon_mask(img.shape, corners)
        H = V.refine_homography(H, px, mm, extra=(corners, ideal), w_extra=1.0)
        out["corners_px"] = corners.tolist()
        out["outline_px"] = corners.tolist()
    else:
        cen, semi, ang = V.fit_ellipse(fg)
        mask = V.ellipse_mask(img.shape, cen, semi, ang)
        boundary = V.boundary_points(mask, 300)
        try:
            H = V.refine_homography(H, px, mm, radial=boundary, w_extra=1.0)
        except np.linalg.LinAlgError:
            pass                                    # 精化失败时保留仅标记解，仍然输出
        centre = V.apply_h(np.linalg.inv(H), [[0.0, 0.0]])[0]
        out["center_px"] = centre.tolist()
        out["outline_px"] = V.boundary_points(mask, 120).tolist()

    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise RuntimeError("invalid homography (non-finite or rank-deficient)")
    out["mask_array"] = mask
    out["H_img_to_mm"] = np.asarray(H, float).tolist()
    return out


def solve_task1(image_bgr, meta):
    """分割桌面 / 圆盘并建立 H_img_to_mm（标记由程序自动检测）。

    返回 {'mask_array', 'shape', 'H_img_to_mm', 'corners_px'|'center_px',
          'outline_px'}；运行器会去掉 mask_array 再落盘。
    """
    return _task1_core(image_bgr, meta, V.detect_markers(image_bgr))


def solve_task1_manual(image_bgr, meta, markers):
    """同样的契约，但四个标记像素由人工给出。

    `markers` 为 {颜色: (u, v)}，必须包含红 / 绿 / 蓝 / 黄。用于某个标记被遮挡、
    或者渲染器调色板发生变化的情况。
    """
    missing = [c for c in ("red", "green", "blue", "yellow") if c not in markers]
    if missing:
        raise RuntimeError(f"manual points missing: {missing}")
    return _task1_core(image_bgr, meta, {k: tuple(map(float, v)) for k, v in markers.items()})


# --------------------------------------------------------------- 题二核心
def _task2_core(img, meta, markers):
    mm = np.array([m["xy_mm"] for m in meta["markers"]], float)
    px = np.array([markers[m["color"]] for m in meta["markers"]], float)
    H = V.homography(px, mm)
    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise RuntimeError("invalid homography (non-finite or rank-deficient)")

    mask = V.bolt_mask(img)
    res = V.measure_bolt(mask, H)
    if not np.isfinite(res["L"]) or not np.isfinite(res["d"]):
        raise RuntimeError("non-finite measurement")
    if res["L"] <= 0 or res["d"] <= 0 or res["d"] > res["L"]:
        raise RuntimeError(f"implausible measurement L={res['L']:.2f} d={res['d']:.2f}")

    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=cv2.contourArea)
    cmm = V.apply_h(H, cont.reshape(-1, 2).astype(float))
    return dict(mask_array=mask, L_mm=float(res["L"]), d_mm=float(res["d"]),
                H_img_to_mm=np.asarray(H, float).tolist(),
                center_mm=np.round(cmm.mean(0), 3).tolist(),
                axis_deg=float(res["axis_deg"]),
                axis_px=[float(res["axis_px"][0]), float(res["axis_px"][1])],
                head_at_low_end=bool(res["head_at_low_end"]))


def solve_task2(image_bgr, meta):
    """分割单个螺栓，并测量总长 L 与杆部直径 d。"""
    return _task2_core(image_bgr, meta, V.detect_markers(image_bgr))


def solve_task2_manual(image_bgr, meta, markers):
    """同样的契约，标记像素由人工给出。"""
    missing = [c for c in ("red", "green", "blue", "yellow") if c not in markers]
    if missing:
        raise RuntimeError(f"manual points missing: {missing}")
    return _task2_core(image_bgr, meta, {k: tuple(map(float, v)) for k, v in markers.items()})
