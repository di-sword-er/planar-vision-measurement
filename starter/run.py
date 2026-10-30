"""运行器：python run.py --data data --split val --out predictions

与官方模板 starter/run.py 的契约完全一致（results.json 结构相同），但额外兼容
非 ASCII 路径，并对单个样本的失败做了容错：一张图异常只写一条 failed 记录，
不会中断整批。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solution import solve_task1, solve_task2
import vision_lib as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--tasks", default="task1,task2")
    args = ap.parse_args()

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task, solve in (("task1", solve_task1), ("task2", solve_task2)):
        if task not in wanted:
            continue
        root = Path(args.data) / task / args.split
        out = Path(args.out) / task
        (out / "masks").mkdir(parents=True, exist_ok=True)
        rows = []
        for path in sorted((root / "inputs").glob("*.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            img = V.imread(root / meta["image"])
            t0 = time.perf_counter()
            row = dict(sample_id=meta["sample_id"], status="failed")
            try:
                if img is None:
                    raise FileNotFoundError(meta["image"])
                result = solve(img, meta)
                mask = result.pop("mask_array")
                if not isinstance(mask, np.ndarray) or mask.shape != img.shape[:2]:
                    raise ValueError("mask_array must have shape H,W")
                mask = np.where(mask > 0, 255, 0).astype(np.uint8)
                rel = f"masks/{meta['sample_id']}.png"
                V.imwrite(out / rel, mask)
                row.update(result)
                row.update(status="ok", mask=rel)
            except Exception as exc:                      # noqa: BLE001 - 写失败记录，绝不让整批中断
                row["reason"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_ms"] = round(1000 * (time.perf_counter() - t0), 3)
            rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        ok = sum(r["status"] == "ok" for r in rows)
        print(f"{task}: {len(rows)} samples, {ok} ok")


if __name__ == "__main__":
    main()
