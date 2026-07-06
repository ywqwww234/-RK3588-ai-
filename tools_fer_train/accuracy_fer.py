#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL 端：FER RKNN 连板精度分析。
前提: adb connect <ELF2_IP>:5555 且 rknn_server 已在板端运行

用法:
  # 先连接开发板
  python3.9 tools_fer_train/accuracy_fer.py --device-id 192.168.1.100:5555

  # 如果没连板，用 PC 模拟器跑
  python3.9 tools_fer_train/accuracy_fer.py --sim
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    from rknn.api import RKNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default=os.path.join(ROOT, "models", "emotion-ferplus-8.rknn"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "emotion-ferplus-8.onnx"))
    ap.add_argument("--device-id", default=None, help="e.g. 192.168.1.100:5555")
    ap.add_argument("--sim", action="store_true", help="use PC simulator instead of board")
    ap.add_argument("--n-samples", type=int, default=50, help="number of images for accuracy check")
    args = ap.parse_args()

    if not os.path.isfile(args.rknn):
        print(f"[ERR] RKNN not found: {args.rknn}")
        print("  Run convert_fer_rknn.py first")
        return 1

    # ---------- Load test images ----------
    calib_dir = os.path.join(ROOT, "data", "fer_calib")
    if not os.path.isdir(calib_dir):
        print(f"[ERR] No calibration images at {calib_dir}")
        print("  Run prepare_fer_calib.py first")
        return 1

    test_files = sorted(os.listdir(calib_dir))[: args.n_samples]
    if not test_files:
        print("[ERR] No images found")
        return 1
    print(f"Using {len(test_files)} test images from {calib_dir}")

    # ---------- Init RKNN ----------
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(args.rknn)
    if ret != 0:
        print(f"[FAIL] Load RKNN failed: ret={ret}")
        rknn.release()
        return ret

    target = None if args.sim else "rk3588"
    device_id = None if args.sim else args.device_id

    print(f"\nInitializing runtime: target={target or 'simulator'}, device_id={device_id or 'none'}")
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print(f"[FAIL] Init runtime failed: ret={ret}")
        if not args.sim:
            print("  Troubleshooting:")
            print("    1. adb devices           # check connection")
            print("    2. adb shell ps|grep rknn # check rknn_server")
            print("    3. adb connect <IP>:5555  # reconnect")
        rknn.release()
        return ret
    print("Runtime initialized OK")

    # ---------- Compare RKNN vs ONNX ----------
    import onnxruntime as ort

    onnx_sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_in_name = onnx_sess.get_inputs()[0].name

    diffs = []
    rknn_times = []
    onnx_times = []

    print(f"\nComparing RKNN vs ONNX on {len(test_files)} images...")
    for fi, fn in enumerate(test_files):
        img_path = os.path.join(calib_dir, fn)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))

        # ONNX: manual normalization (x-128)/128
        blob_onnx = ((img.astype(np.float32) - 128.0) / 128.0).reshape(1, 1, 64, 64)
        t0 = time.perf_counter()
        out_onnx = onnx_sess.run([], {onnx_in_name: blob_onnx})[0][0]
        dt_onnx = (time.perf_counter() - t0) * 1000

        # RKNN: raw [0,255] input (RKNN does (x-128)/128 internally)
        blob_rknn = img.astype(np.float32).reshape(1, 1, 64, 64)
        t0 = time.perf_counter()
        out_rknn = rknn.inference(inputs=[blob_rknn])[0][0]
        dt_rknn = (time.perf_counter() - t0) * 1000

        # Compare
        diff = np.abs(out_onnx - out_rknn).max()
        cls_onnx = int(np.argmax(out_onnx))
        cls_rknn = int(np.argmax(out_rknn))
        match = "MATCH" if cls_onnx == cls_rknn else "MISMATCH"

        diffs.append(diff)
        rknn_times.append(dt_rknn)
        onnx_times.append(dt_onnx)

        if fi < 5 or match == "MISMATCH":
            print(f"  [{fi:3d}] {match:8s}  max_diff={diff:.5f}  "
                  f"onnx_cls={cls_onnx}  rknn_cls={cls_rknn}  "
                  f"rknn={dt_rknn:.1f}ms  onnx={dt_onnx:.1f}ms")

    # ---------- Summary ----------
    n_mismatch = sum(1 for d in diffs if d > 0.01)
    print(f"\n{'=' * 50}")
    print(f"SUMMARY (n={len(diffs)})")
    print(f"  Class mismatch: {n_mismatch} / {len(diffs)}")
    print(f"  Max logit diff: {np.max(diffs):.5f}")
    print(f"  Mean logit diff: {np.mean(diffs):.5f}")
    print(f"  RKNN latency: mean={np.mean(rknn_times):.1f}ms  p95={np.percentile(rknn_times, 95):.1f}ms")
    print(f"  ONNX latency: mean={np.mean(onnx_times):.1f}ms  p95={np.percentile(onnx_times, 95):.1f}ms")

    if n_mismatch == 0:
        print("\n  >>> FER RKNN vs ONNX: FULLY CONSISTENT <<<")
    elif n_mismatch / len(diffs) < 0.05:
        print(f"\n  >>> Acceptable: {n_mismatch} minor diffs (within FP16 tolerance) <<<")
    else:
        print(f"\n  >>> WARNING: {n_mismatch} mismatches, investigate <<<")

    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
