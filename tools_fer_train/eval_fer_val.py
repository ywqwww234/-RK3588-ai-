#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 val 集上测 ONNX（与 main.py 相同预处理），看哪几类容易错。"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools_fer_train.train_fer import CLASSES, preprocess_gray64


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "fer_labeled"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "emotion-ferplus-8.onnx"))
    args = ap.parse_args()

    val_root = os.path.join(args.data, "val")
    net = cv2.dnn.readNetFromONNX(args.onnx)
    ext = {".jpg", ".jpeg", ".png", ".bmp"}

    y_true, y_pred = [], []
    per_class = {c: {"ok": 0, "n": 0} for c in CLASSES}

    for ci, cname in enumerate(CLASSES):
        d = os.path.join(val_root, cname)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if os.path.splitext(fn.lower())[1] not in ext:
                continue
            path = os.path.join(d, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            blob = preprocess_gray64(img).reshape(1, 1, 64, 64).astype(np.float32)
            net.setInput(blob)
            out = net.forward()[0]
            pred = int(np.argmax(out))
            y_true.append(ci)
            y_pred.append(pred)
            per_class[cname]["n"] += 1
            if pred == ci:
                per_class[cname]["ok"] += 1

    if not y_true:
        print("[ERR] val 无图片，先 split 或采数据")
        return 1

    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    print(f"val 准确率 (与采集图一致预处理): {acc:.1%}  n={len(y_true)}")
    print("\n每类:")
    for c in CLASSES:
        st = per_class[c]
        if st["n"] == 0:
            print(f"  {c}: (无 val)")
            continue
        print(f"  {c}: {st['ok']}/{st['n']} = {st['ok']/st['n']:.0%}")

    # 混淆：真实类 -> 最常错成
    from collections import Counter

    print("\n常见混淆 (真实类 -> 预测最多):")
    for ci, cname in enumerate(CLASSES):
        idxs = [y_pred[i] for i, t in enumerate(y_true) if t == ci]
        if not idxs:
            continue
        cnt = Counter(idxs)
        top = cnt.most_common(2)
        parts = [f"{CLASSES[p]}({n})" for p, n in top]
        print(f"  {cname} -> {', '.join(parts)}")

    print("\n若 val 高但 main.py 仍不准：多半是运行时 YuNet 裁脸与采集时不一致，需加采或统一裁脸。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)