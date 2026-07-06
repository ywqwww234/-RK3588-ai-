#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 周：legacy vs YOLOv8 基线评测（单图目录 / 摄像头短视频）。

指标:
  - person 检出率（YOLO: 有 person 框；legacy: analyze_face 非 none 或有人脸）
  - 推理延迟 ms（均值 / P95）
  - 失败率

用法:
  python tools_yolo_week1/benchmark_yolo.py --images data/yolo/images/raw
  python tools_yolo_week1/benchmark_yolo.py --camera 0 --seconds 30
  python tools_yolo_week1/benchmark_yolo.py --images data/yolo/images/val --report reports/baseline_week1.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass
class BenchStats:
    name: str
    n_frames: int = 0
    n_detect: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def detect_rate(self) -> float:
        return self.n_detect / self.n_frames if self.n_frames else 0.0

    @property
    def latency_mean(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def latency_p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0


def load_yolo_backend():
    from vision_yolov8 import YoloV8VisionBackend
    b = YoloV8VisionBackend()
    if not b.ready:
        raise RuntimeError(f"YOLO not ready: {b.last_error}")
    return b


def load_legacy_backend():
    try:
        from Anti_depress import LocalFaceAnalyzer
        return LocalFaceAnalyzer()
    except Exception:
        pass
    # 回退：OpenCV Haar 人脸（近似 legacy 人脸通路）
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(cascade_path):
        raise RuntimeError("无法加载 LocalFaceAnalyzer 且 Haar 不可用")
    cascade = cv2.CascadeClassifier(cascade_path)

    class _HaarLegacy:
        def analyze_face(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
            if len(faces) == 0:
                return "none", 0.0, 0.0, 0.0
            return "neutral", 0.5, 0.15, 0.1

    return _HaarLegacy()


def yolo_detected(backend, frame) -> Tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        if not backend.ready:
            return False, (time.perf_counter() - t0) * 1000
        conf = float(__import__("config").YOLOV8_CONF)
        imgsz = int(__import__("config").YOLOV8_IMG_SIZE)
        res = backend.model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        dt = (time.perf_counter() - t0) * 1000
        if boxes is None or len(boxes) == 0:
            return False, dt
        cls_arr = boxes.cls.detach().cpu().numpy()
        names = getattr(r0, "names", {}) or {}
        for c in cls_arr:
            cid = int(c)
            cname = str(names.get(cid, cid)).lower()
            if cid == 0 or cname == "person":
                return True, dt
        return False, dt
    except Exception:
        return False, (time.perf_counter() - t0) * 1000


def legacy_detected(analyzer, frame) -> Tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        expr, prob, eye_f, post = analyzer.analyze_face(frame)
        dt = (time.perf_counter() - t0) * 1000
        ok = str(expr) != "none" and float(prob) >= 0.0
        return ok, dt
    except Exception:
        return False, (time.perf_counter() - t0) * 1000


def run_on_frames(frames: List[np.ndarray], yolo_b, legacy_b) -> Tuple[BenchStats, BenchStats]:
    ys, ls = BenchStats("yolov8"), BenchStats("legacy")
    for frame in frames:
        ok, dt = yolo_detected(yolo_b, frame)
        ys.n_frames += 1
        ys.latencies_ms.append(dt)
        if ok:
            ys.n_detect += 1

        ok, dt = legacy_detected(legacy_b, frame)
        ls.n_frames += 1
        ls.latencies_ms.append(dt)
        if ok:
            ls.n_detect += 1
    return ys, ls


def collect_from_images(folder: str, max_n: int) -> List[np.ndarray]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f.lower())[1] in exts:
            paths.append(os.path.join(folder, f))
    paths = paths[:max_n]
    frames = []
    for p in paths:
        im = cv2.imread(p)
        if im is not None:
            frames.append(im)
    return frames


def collect_from_camera(camera: int, seconds: float) -> List[np.ndarray]:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        return []
    frames = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        time.sleep(0.03)
    cap.release()
    return frames


def write_report(path: str, ys: BenchStats, ls: BenchStats, extra: str = ""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# YOLO Week1 Baseline",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        extra,
        "",
        "| Backend | Frames | Detect rate | Latency mean (ms) | P95 (ms) |",
        "|---------|--------|-------------|-------------------|----------|",
        f"| yolov8 | {ys.n_frames} | {ys.detect_rate:.1%} | {ys.latency_mean:.1f} | {ys.latency_p95:.1f} |",
        f"| legacy | {ls.n_frames} | {ls.detect_rate:.1%} | {ls.latency_mean:.1f} | {ls.latency_p95:.1f} |",
        "",
        "## 解读（第 1 周）",
        "- 若 YOLO 检出率高但延迟大：第 2 周再训 + 调 `YOLO_INFER_EVERY_N`。",
        "- 若 legacy 人脸高、YOLO person 低：检查是否全身未入画，或需加 `face` 类。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] {path}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, help="图片目录")
    src.add_argument("--camera", type=int, help="摄像头实时采样")
    ap.add_argument("--seconds", type=float, default=20.0, help="摄像头采样时长")
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--report", type=str, default=os.path.join(ROOT, "reports", "baseline_week1.md"))
    args = ap.parse_args()

    if args.images:
        if not os.path.isdir(args.images):
            print(f"[ERR] 目录不存在: {args.images}")
            return 1
        frames = collect_from_images(args.images, args.max_images)
        extra = f"\n- Source: images `{args.images}` ({len(frames)} frames)\n"
    else:
        frames = collect_from_camera(args.camera, args.seconds)
        extra = f"\n- Source: camera {args.camera} ({args.seconds}s, {len(frames)} frames)\n"

    if not frames:
        print("[ERR] 无有效帧，请先运行 capture_frames.py 或使用 --camera")
        return 1

    print(f"[INFO] 加载后端，共 {len(frames)} 帧...")
    try:
        yolo_b = load_yolo_backend()
    except Exception as e:
        print(f"[ERR] YOLO: {e}")
        return 1
    try:
        legacy_b = load_legacy_backend()
    except Exception as e:
        print(f"[ERR] legacy: {e}")
        return 1

    ys, ls = run_on_frames(frames, yolo_b, legacy_b)
    print(f"\n=== yolov8 ===\n  detect_rate={ys.detect_rate:.1%}  mean={ys.latency_mean:.1f}ms  p95={ys.latency_p95:.1f}ms")
    print(f"\n=== legacy ===\n  detect_rate={ls.detect_rate:.1%}  mean={ls.latency_mean:.1f}ms  p95={ls.latency_p95:.1f}ms")

    write_report(args.report, ys, ls, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)