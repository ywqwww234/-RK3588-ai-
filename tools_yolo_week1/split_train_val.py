#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 images/raw 或已标注的 train 池按比例划入 train/val（复制图片与 labels）。

用法（仅复制图片，标注需你随后用 LabelImg 标好放到 labels/）:
  python tools_yolo_week1/split_train_val.py --ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO = os.path.join(ROOT, "data", "yolo")
RAW = os.path.join(YOLO, "images", "raw")


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    )


def copy_pair(img_name, src_img_dir, dst_img_dir, src_lbl_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    shutil.copy2(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
    base = os.path.splitext(img_name)[0]
    lbl = base + ".txt"
    sp = os.path.join(src_lbl_dir, lbl)
    if os.path.isfile(sp):
        shutil.copy2(sp, os.path.join(dst_lbl_dir, lbl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=RAW, help="源图片目录，默认 images/raw")
    ap.add_argument("--ratio", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = list_images(args.source)
    if not names:
        print(f"[ERR] 未找到图片: {args.source}")
        return 1

    random.seed(args.seed)
    random.shuffle(names)
    n_val = max(1, int(len(names) * args.ratio))
    val_set = set(names[:n_val])

    img_train = os.path.join(YOLO, "images", "train")
    img_val = os.path.join(YOLO, "images", "val")
    lbl_train = os.path.join(YOLO, "labels", "train")
    lbl_val = os.path.join(YOLO, "labels", "val")
    src_lbl = os.path.join(YOLO, "labels", "raw")

    for name in names:
        if name in val_set:
            copy_pair(name, args.source, img_val, src_lbl, lbl_val)
        else:
            copy_pair(name, args.source, img_train, src_lbl, lbl_train)

    print(f"[DONE] total={len(names)} train={len(names)-n_val} val={n_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)