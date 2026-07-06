#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 train/<class>/ 随机划出 ratio 到 val/<class>/（移动文件）。"""

from __future__ import annotations

import argparse
import os
import random
import shutil

CLASSES = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fer_labeled", help="含 train/ val/ 的根目录")
    ap.add_argument("--ratio", type=float, default=0.15, help="划入 val 的比例，如 0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-val", type=int, default=5, help="每类至少划到 val 的张数（若有足够图）")
    args = ap.parse_args()

    root = os.path.abspath(args.data)
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "val")
    if not os.path.isdir(train_root):
        print(f"[ERR] 找不到 {train_root}")
        return 1

    random.seed(args.seed)
    ratio = max(0.05, min(0.4, float(args.ratio)))
    total_moved = 0

    for cname in CLASSES:
        src_dir = os.path.join(train_root, cname)
        dst_dir = os.path.join(val_root, cname)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.isdir(src_dir):
            print(f"[SKIP] 无目录 {src_dir}")
            continue

        files = [
            f
            for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
            and os.path.splitext(f.lower())[1] in IMG_EXT
        ]
        n = len(files)
        if n == 0:
            print(f"[{cname}] train=0")
            continue

        k = max(args.min_val, int(round(n * ratio)))
        k = min(k, max(1, n - 1))  # 至少留 1 张在 train
        pick = random.sample(files, k)

        for fn in pick:
            s = os.path.join(src_dir, fn)
            d = os.path.join(dst_dir, fn)
            if os.path.exists(d):
                base, ext = os.path.splitext(fn)
                d = os.path.join(dst_dir, f"{base}_v{random.randint(1000,9999)}{ext}")
            shutil.move(s, d)
            total_moved += 1

        n_train = len(
            [
                f
                for f in os.listdir(src_dir)
                if os.path.splitext(f.lower())[1] in IMG_EXT
            ]
        )
        n_val = len(
            [
                f
                for f in os.listdir(dst_dir)
                if os.path.isfile(os.path.join(dst_dir, f))
                and os.path.splitext(f.lower())[1] in IMG_EXT
            ]
        )
        print(f"[{cname}] moved {len(pick)} -> val | train={n_train} val={n_val}")

    print(f"[DONE] total moved to val: {total_moved}")
    print(f"下一步: python tools_fer_train/train_fer.py --data {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)