"""扩展：完整工业实例分割数据集上的使用与评估。

    python ext_industrial.py --data <数据集根目录> --out real_figures/industrial

一次跑完三件事：
  1. EDA            —— 类别分布、每图实例数、来源构成
  2. 规则基线       —— 一个手工设计的颜色规则，逐个 COCO 实例评分
  3. 来源核查       —— 与比赛包真实全图的重叠情况

这里的结果不参与评分流水线，它的作用是量化"为什么真实场景需要学习方法"。
"""
from __future__ import annotations

import argparse
import json
import sys

from ioutil import make_stdout_safe
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import vision_lib as V


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flag)


def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(p).lower().endswith(".jpg"):
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(p))
    else:
        cv2.imencode(".png", img)[1].tofile(str(p))


# ------------------------------------------------------------------ 工具函数
def load_coco(root, split):
    return json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))


def load_provenance(root):
    prov = {}
    with open(root / "annotations" / "accepted_provenance.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prov[(r["image_uid"], r["target_uid"])] = r
    return prov


def gt_instances(root, split, im, anns, prov):
    """把每个标注实例还原成整图大小的布尔掩膜。"""
    out = []
    h, w = im["height"], im["width"]
    for an in anns:
        r = prov.get((im["image_uid"], an["instance_uid"]))
        if r is None:
            continue
        x0, y0, _, _ = r["crop"]
        m = imread(root / "masks" / split / im["image_uid"] / f"{an['instance_uid']}.png", 0)
        if m is None:
            continue
        full = np.zeros((h, w), bool)
        hh, ww = min(m.shape[0], h - y0), min(m.shape[1], w - x0)
        if hh <= 0 or ww <= 0:
            continue
        full[y0:y0 + hh, x0:x0 + ww] = m[:hh, :ww] > 0
        out.append(full)
    return out


def mat_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 55)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    if m.sum() < 5000:
        return np.zeros(img.shape[:2], np.uint8)
    return V.fill_holes(V.largest_component(m))


def rule_parts(img, tau=28, min_area=250):
    """手工设计的基线：在彩色桌垫内部，凡与桌垫颜色差异大的连通域即一个零件。"""
    mat = mat_mask(img)
    if mat.sum() < 5000:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    med = np.median(img[mat > 0].reshape(-1, 3), 0).astype(np.uint8)
    medl = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
    d = np.linalg.norm(lab - medl, axis=2)
    m = ((d > tau) & (mat > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[cc == i] = 1
    return keep


# --------------------------------------------------------------------- 主流程
def main():
    make_stdout_safe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (contains annotations/ images/ masks/)")
    ap.add_argument("--out", default="real_figures/industrial")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 从标注统计 EDA
    eda = {}
    for split in ("train", "val"):
        c = load_coco(root, split)
        cats = {x["id"]: x["name"] for x in c["categories"]}
        per_img = Counter(an["image_id"] for an in c["annotations"])
        cls = Counter(cats[an["category_id"]] for an in c["annotations"])
        sz = Counter((im["width"], im["height"]) for im in c["images"])
        pub = sum(1 for im in c["images"] if "/public" in im["source_path"].lower())
        eda[split] = dict(images=len(c["images"]), instances=len(c["annotations"]),
                          per_class=dict(cls), sizes={f"{w}x{h}": n for (w, h), n in sz.items()},
                          public_images=pub, local_images=len(c["images"]) - pub,
                          inst_per_image=dict(min=min(per_img.values()),
                                              max=max(per_img.values()),
                                              mean=round(float(np.mean(list(per_img.values()))), 2),
                                              median=float(np.median(list(per_img.values())))))
    (out / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8")
    for s, e in eda.items():
        print(f"{s}: {e['images']} images, {e['instances']} instances, {e['per_class']}, "
              f"public {e['public_images']}, inst/img median {e['inst_per_image']['median']:.0f}")

    # ---- 规则基线评估
    prov = load_provenance(root)
    c = load_coco(root, a.split)
    anns_by_img = defaultdict(list)
    for an in c["annotations"]:
        anns_by_img[an["image_id"]].append(an)
    files = c["images"][:a.limit] if a.limit else c["images"]

    best_ious, n_gt, n_pred, hit = [], 0, 0, 0
    no_mat = 0
    gallery = []
    for im in files:
        img = imread(root / im["file_name"])
        if img is None:
            continue
        gts = gt_instances(root, a.split, im, anns_by_img.get(im["id"], []), prov)
        if not gts:
            continue
        pred = rule_parts(img)
        n_gt += len(gts)
        if pred is None:
            no_mat += len(gts)
            best_ious.extend([0.0] * len(gts))
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats(pred, 8)
        comps = [cc == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 250]
        n_pred += len(comps)
        used = set()
        img_hits = 0
        for g in gts:
            best, bi = 0.0, -1
            for j, p in enumerate(comps):
                if j in used:
                    continue
                inter = np.count_nonzero(g & p)
                if inter == 0:
                    continue
                iou = inter / np.count_nonzero(g | p)
                if iou > best:
                    best, bi = iou, j
            best_ious.append(best)
            if best >= 0.5:
                hit += 1
                img_hits += 1
                used.add(bi)
        if len(gallery) < 40:
            gallery.append((im["image_uid"], len(gts), len(comps),
                            img_hits / len(gts), img, gts, comps))

    bi = np.array(best_ious)
    res = dict(split=a.split, images_evaluated=len(files), gt_instances=n_gt,
               predicted_instances=n_pred,
               recall_at_05=hit / max(n_gt, 1),
               mean_best_iou=float(bi.mean()), median_best_iou=float(np.median(bi)),
               frac_ge={t: float((bi >= t).mean()) for t in (0.3, 0.5, 0.75, 0.9)},
               instances_without_mat=no_mat,
               note="hand-designed colour rule, no learning, no calibration")
    (out / "rule_baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"rule baseline: gt={n_gt} pred={n_pred} recall@0.5={res['recall_at_05']:.3f} "
          f"mean best IoU={res['mean_best_iou']:.3f}")

    # ---- 出图
    panels = []
    for uid, ng, npred, rec, img, gts, comps in gallery:
        if len(panels) >= 6:
            break
        vis = img.copy()
        U = np.zeros(img.shape[:2], bool)
        for g in gts:
            U |= g
        vis[U] = (0.55 * vis[U] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
        for p in comps:
            cs, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 255, 0), 3)
        txt = f"gt={ng} pred={npred} recall={rec:.2f}"
        cv2.putText(vis, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(vis, (640, 360)))
    if panels:
        rows = (len(panels) + 1) // 2
        sheet = np.full((rows * 368, 2 * 648, 3), 25, np.uint8)
        for i, p in enumerate(panels):
            r, cc2 = divmod(i, 2)
            sheet[r * 368:r * 368 + 360, cc2 * 648:cc2 * 648 + 640] = p
        imwrite(out / "rule_baseline_examples.jpg", sheet)

    print("figures ->", out)


if __name__ == "__main__":
    main()
