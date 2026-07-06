#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL 端：YOLOv8 ONNX → RKNN 转换（不量化，FP16）。
用法:
  python3.9 tools_yolo_week1/convert_yolo_rknn.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    from rknn.api import RKNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "yolov8_classroom.onnx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "yolov8_classroom.rknn"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset_yolo.txt"))
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--quant", default=False, action="store_true",
                    help="enable INT8 quantization (default: False = FP16)")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[ERR] ONNX not found: {args.onnx}")
        print("  First run: python tools_yolo_week1/export_yolo_onnx.py")
        return 1

    do_quant = bool(args.quant)

    rknn = RKNN(verbose=True)

    # ---------- Step 1: Config ----------
    print("\n" + "=" * 60)
    print("[STEP 1] Config model")
    print(f"  platform     = {args.platform}")
    print(f"  do_quant     = {do_quant}")
    print(f"  mean_values  = [[0, 0, 0]]")
    print(f"  std_values   = [[255, 255, 255]]   # x/255 normalization")
    print("=" * 60)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
    )
    print("done.")

    # ---------- Step 2: Load ONNX ----------
    print("\n[STEP 2] Loading ONNX")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print(f"[FAIL] Load ONNX failed! ret={ret}")
        rknn.release()
        return ret
    print("done.")

    # ---------- Step 3: Build ----------
    print("\n[STEP 3] Building RKNN model")
    if do_quant:
        if not os.path.isfile(args.dataset):
            print(f"[WARN] dataset.txt missing, fallback to FP16")
            do_quant = False
        else:
            ret = rknn.build(do_quantization=True, dataset=args.dataset)
    if not do_quant:
        print("  FP16 mode (no quantization)")
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"[FAIL] Build failed! ret={ret}")
        rknn.release()
        return ret
    print("done.")

    # ---------- Step 4: Export ----------
    print("\n[STEP 4] Exporting RKNN")
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print(f"[FAIL] Export failed! ret={ret}")
        rknn.release()
        return ret
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"[OK] -> {args.out} ({size_mb:.1f} MB)")

    # ---------- Step 5: PC simulator quick sanity ----------
    print("\n[STEP 5] PC simulator sanity check")
    import numpy as np

    ret = rknn.init_runtime()
    if ret != 0:
        print(f"[WARN] Simulator init failed (ret={ret}), skipping")
    else:
        dummy = (np.random.rand(1, 640, 640, 3) * 255).astype(np.float32)
        outputs = rknn.inference(inputs=[dummy], data_format="nhwc")
        for o in outputs:
            print(f"  output shape: {o.shape}")
        expected = (1, 5, 8400)
        if outputs[0].shape == expected:
            print(f"  -> shape matches expected {expected}")
        else:
            print(f"  -> WARNING: expected {expected}, got {outputs[0].shape}")

    rknn.release()
    print("\n[DONE] YOLOv8 RKNN conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
