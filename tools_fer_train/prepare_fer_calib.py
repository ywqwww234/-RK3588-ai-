#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备 FER 校准图片 + dataset.txt（不量化模式下仍用于 accuracy_analysis）。"""

from __future__ import annotations

import argparse
import os
import random

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "fer_labeled", "train"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "fer_calib"))
    ap.add_argument("--txt", default=os.path.join(ROOT, "data", "dataset_fer.txt"))
    ap.add_argument("--max-per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    lines = []
    total = 0

    for cname in CLASSES:
        src_dir = os.path.join(args.src, cname)
        if not os.path.isdir(src_dir):
            print(f"  [SKIP] {cname}: no dir")
            continue
        files = [f for f in os.listdir(src_dir)
                 if os.path.splitext(f.lower())[1] in IMG_EXT]
        if not files:
            print(f"  [SKIP] {cname}: 0 images")
            continue

        take = min(args.max_per_class, len(files))
        picked = random.sample(files, take)

        for fn in picked:
            img = cv2.imread(os.path.join(src_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # 只做 resize 到 64x64，不做归一化 (RKNN mean/std 配置负责)
            img = cv2.resize(img, (64, 64))
            save_name = f"{cname}_{fn}"
            save_path = os.path.join(args.out, save_name)
            cv2.imwrite(save_path, img)
            lines.append(save_path + "\n")
            total += 1

        print(f"  {cname}: {take} of {len(files)}")

    with open(args.txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[DONE] {total} images -> {args.out}")
    print(f"[DONE] dataset.txt -> {args.txt}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
