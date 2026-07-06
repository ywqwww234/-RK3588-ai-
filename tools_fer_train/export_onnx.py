#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 ONNX，替换 models/emotion-ferplus-8.onnx（需与 vision_yolov8 mean128 nchw 一致）。"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "fer", "best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "emotion-ferplus-8.onnx"))
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"[ERR] 找不到 {args.ckpt}，先 train_fer.py")
        return 1

    from tools_fer_train.train_fer import build_model, CLASSES

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 1, 64, 64)
    onnx_path = args.out
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    if args.backup and os.path.isfile(onnx_path):
        bak = onnx_path + ".bak"
        import shutil

        shutil.copy2(onnx_path, bak)
        print(f"[INFO] backup -> {bak}")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )

    import cv2

    net = cv2.dnn.readNetFromONNX(onnx_path)
    x = np.zeros((64, 64), np.float32)
    blob = ((cv2.resize(x, (64, 64)) - 128.0) / 128.0).reshape(1, 1, 64, 64).astype(np.float32)
    net.setInput(blob)
    out = net.forward()[0]
    print(f"[OK] ONNX output len={len(out)} classes={CLASSES}")
    print(f"[OK] 已写入 {onnx_path}，重启 main.py 生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)