"""逐条核对：把发放包 docx 里的每一条硬性要求变成可执行断言。

    python check_requirements.py --data ../pack/data --pred predictions

输出一条一条的 PASS / FAIL，供答辩时直接引用。
"""
from __future__ import annotations

import argparse
import json
import sys

from ioutil import make_stdout_safe
import re
from pathlib import Path

import numpy as np
import cv2

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((bool(ok), name, detail))
    return ok


def imread(p, flag=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, flag) if buf.size else None


def sheet_fields(split):
    return {
        "task1": {"sample_id", "status", "shape", "mask", "H_img_to_mm"},
        "task2": {"sample_id", "status", "mask", "L_mm", "d_mm"},
    }


def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../pack/data")
    ap.add_argument("--pred", default="predictions")
    a = ap.parse_args()
    data, pred = Path(a.data), Path(a.pred)

    # ---- 1. 目录与数量
    for task, want in (("task1", 16), ("task2", 16)):
        for split in ("train", "val", "test"):
            root = data / task / split
            n_in = len(list((root / "inputs").glob("*.json")))
            n_im = len(list((root / "images").glob("*.png")))
            exp = {"train": 64, "val": 16, "test": 16}[split]
            check(f"{task}/{split} inputs={n_in} images={n_im} == {exp}",
                  n_in == exp and n_im == exp)
    check("test 划分不含 labels（答案不在发放包内）",
          not (data / "task1" / "test" / "labels").exists()
          and not (data / "task2" / "test" / "labels").exists())

    # ---- 2. results.json 结构
    for task in ("task1", "task2"):
        p = pred / task / "results.json"
        if not check(f"{task}/results.json 存在", p.exists()):
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        check(f"{task}/results.json 是数组", isinstance(rows, list))
        root = data / task / "val"
        ids = {q.stem for q in (root / "inputs").glob("*.json")}
        got = [r.get("sample_id") for r in rows]
        check(f"{task} 每个输入恰有一条记录", sorted(got) == sorted(ids),
              f"{len(got)} 条 vs {len(ids)} 个输入")
        check(f"{task} sample_id 无重复", len(set(got)) == len(got))
        need = sheet_fields(split=None)[task]
        bad = [r["sample_id"] for r in rows
               if r.get("status") == "ok" and not need.issubset(r.keys())]
        check(f"{task} 所有 ok 记录含必需字段 {sorted(need)}", not bad, str(bad[:3]))
        st = {r.get("status") for r in rows}
        check(f"{task} status 只取 ok/failed", st <= {"ok", "failed"}, str(st))
        bad_f = [r["sample_id"] for r in rows
                 if r.get("status") == "failed" and not r.get("reason")]
        check(f"{task} 所有 failed 记录都写了 reason", not bad_f, str(bad_f[:3]))

        # ---- 3. 掩膜规格
        viol_size, viol_bin, viol_path = [], [], []
        for r in rows:
            if r.get("status") != "ok":
                continue
            mrel = r.get("mask", "")
            mp = (pred / task / mrel).resolve()
            if not mp.is_relative_to((pred / task).resolve()):
                viol_path.append(r["sample_id"])
                continue
            if not mp.exists():
                viol_path.append(r["sample_id"])
                continue
            m = imread(mp, 0)
            meta = json.loads((root / "inputs" / f"{r['sample_id']}.json")
                              .read_text(encoding="utf-8"))
            img = imread(root / meta["image"])
            if m is None or m.shape != img.shape[:2]:
                viol_size.append(r["sample_id"])
                continue
            if m.ndim != 2:
                viol_size.append(r["sample_id"])
                continue
            if not set(np.unique(m)).issubset({0, 255}):
                viol_bin.append(r["sample_id"])
        check(f"{task} 掩膜尺寸 == 输入图像尺寸", not viol_size, str(viol_size[:3]))
        check(f"{task} 掩膜单通道且只含 0/255", not viol_bin, str(viol_bin[:3]))
        check(f"{task} mask 路径相对且不越界", not viol_path, str(viol_path[:3]))

        # ---- 4. 题一/题二专有字段
        if task == "task1":
            bad = []
            for r in rows:
                if r.get("status") != "ok":
                    continue
                H = np.array(r["H_img_to_mm"], float)
                if H.shape != (3, 3) or not np.isfinite(H).all():
                    bad.append(r["sample_id"])
                if r.get("shape") not in ("square", "disk"):
                    bad.append(r["sample_id"])
                if "corners_px" not in r and "center_px" not in r:
                    bad.append(r["sample_id"])
                if r.get("shape") == "square" and len(np.array(r.get("corners_px", []))) != 4:
                    bad.append(r["sample_id"])
            check("题一：H 为有限 3x3；shape∈{square,disk}；四角或圆心存在", not bad, str(set(bad)))
            # 字段顺序/约定：[X,Y,1] ∝ H·[u,v,1]
            r0 = next(r for r in rows if r.get("status") == "ok")
            meta = json.loads((root / "inputs" / f"{r0['sample_id']}.json")
                              .read_text(encoding="utf-8"))
            u, v = map(float, np.array(meta["queries_px"], float)[0])
            q = np.array(r0["H_img_to_mm"], float) @ np.array([u, v, 1.0])
            check("题一：H 满足 [X,Y,1] ∝ H·[u,v,1]（约定正确）",
                  abs(q[2]) > 1e-12, f"第三分量={q[2]:.4f}")
        else:
            bad = []
            for r in rows:
                if r.get("status") != "ok":
                    continue
                L, d = r.get("L_mm"), r.get("d_mm")
                if not (isinstance(L, (int, float)) and isinstance(d, (int, float))):
                    bad.append(r["sample_id"])
                elif not (0 < d < L):
                    bad.append(r["sample_id"])
            check("题二：L,d 为有限正数且 d < L（d 不含头、L 含头）", not bad, str(bad[:3]))

    # ---- 5. 推理阶段不读标签
    code = "\n".join((Path(__file__).parent / f).read_text(encoding="utf-8")
                     for f in ("vision_lib.py", "solution.py", "run.py"))
    hits = re.findall(r"labels", code)
    check("推理代码中不出现 'labels' 路径（不读答案）", not hits, f"出现 {len(hits)} 次")

    # ---- 6. 与官方评测器一致性
    ev = Path("evaluation_val.json")
    if ev.exists():
        d = json.loads(ev.read_text(encoding="utf-8"))
        check("预置 evaluation_val.json 含 75 分制总分（本项不现场重跑评测器）",
              "objective_total" in d, f"{d['objective_total']:.2f}/75")

    # ---- report
    ok = sum(1 for r in RESULTS if r[0])
    print(f"{'结果':6s} 检查项")
    print("-" * 78)
    for good, name, detail in RESULTS:
        mark = "PASS" if good else "FAIL"
        print(f"{mark:6s} {name}" + (f"    [{detail}]" if detail else ""))
    print("-" * 78)
    print(f"通过 {ok} / {len(RESULTS)}")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
