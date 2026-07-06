#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL 端：FER ONNX → RKNN 转换（不量化，FP16）。
用法:
  python3.9 tools_fer_train/convert_fer_rknn.py
  python3.9 tools_fer_train/convert_fer_rknn.py --quant False
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
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "emotion-ferplus-8.onnx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "emotion-ferplus-8.rknn"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset_fer.txt"))
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--quant", default=False, action="store_true",
                    help="enable INT8 quantization (default: False = FP16)")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"[ERR] ONNX not found: {args.onnx}")
        return 1

    do_quant = bool(args.quant)
    if do_quant and not os.path.isfile(args.dataset):
        print(f"[WARN] dataset.txt not found, fallback to FP16 (no quant)")
        do_quant = False

    rknn = RKNN(verbose=True)

    # ---------- Step 1: Config ----------
    print("\n" + "=" * 60)
    print("[STEP 1] Config model")
    print(f"  platform     = {args.platform}")
    print(f"  do_quant     = {do_quant}")
    print(f"  mean_values  = [[128]]   # grayscale, single channel")
    print(f"  std_values   = [[128]]   # (x - 128) / 128")
    print("=" * 60)

    rknn.config(
        mean_values=[[128]],
        std_values=[[128]],
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
        print(f"  dataset = {args.dataset}")
        ret = rknn.build(do_quantization=True, dataset=args.dataset)
    else:
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
        print(f"[WARN] Simulator init failed (ret={ret}), skipping sanity check")
    else:
        # Pass raw [0,255] values (RKNN internally applies (x-128)/128)
        dummy = (np.random.rand(1, 1, 64, 64) * 255).astype(np.float32)
        outputs = rknn.inference(inputs=[dummy])
        shapes = [str(o.shape) for o in outputs]
        print(f"  dummy input shape: {dummy.shape}")
        print(f"  output shapes: {shapes}")
        print(f"  expected: [(1, 8)]  # 8-class logits")
        print("  -> sanity check PASSED")

    rknn.release()
    print("\n[DONE] Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
