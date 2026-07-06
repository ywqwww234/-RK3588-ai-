#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 周：从摄像头或视频按间隔抽帧，写入 data/yolo/images/raw/ 并记录 manifest。

用法:
  python tools_yolo_week1/capture_frames.py --camera 0 --tag front_normal
  python tools_yolo_week1/capture_frames.py --video path/to/classroom.mp4 --tag multi_person --interval 0.5
  python tools_yolo_week1/capture_frames.py --camera 0 --tag head_down --max 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "yolo", "images", "raw")
MANIFEST = os.path.join(ROOT, "data", "yolo", "manifest_capture.csv")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.isfile(MANIFEST):
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "tag", "source", "ts_iso", "width", "height"])


def append_manifest(row):
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_frame(frame, tag: str, source: str) -> str:
    h, w = frame.shape[:2]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
    name = f"{tag}_{ts}.jpg"
    path = os.path.join(RAW_DIR, name)
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    append_manifest([name, tag, source, datetime.now(timezone.utc).isoformat(), w, h])
    return path


def run_camera(camera_id: int, tag: str, interval_sec: float, max_frames: int):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERR] 无法打开摄像头 index={camera_id}")
        return 1
    ensure_dirs()
    source = f"camera:{camera_id}"
    count = 0
    last_save = 0.0
    print(f"[INFO] 摄像头 {camera_id} | tag={tag} | 间隔 {interval_sec}s | 最多 {max_frames} 张")
    print("[INFO] 按 q 退出，按 s 立即存一张")

    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        cv2.imshow("capture (q=quit, s=save)", frame)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if key == ord("q"):
            break
        if key == ord("s") or (now - last_save >= interval_sec):
            p = save_frame(frame, tag, source)
            count += 1
            last_save = now
            print(f"  saved [{count}/{max_frames}] {os.path.basename(p)}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] 共保存 {count} 张 -> {RAW_DIR}")
    return 0


def run_video(video_path: str, tag: str, interval_sec: float, max_frames: int):
    if not os.path.isfile(video_path):
        print(f"[ERR] 视频不存在: {video_path}")
        return 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {video_path}")
        return 1
    ensure_dirs()
    source = f"video:{os.path.basename(video_path)}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * interval_sec)))
    count = 0
    idx = 0
    print(f"[INFO] 视频 {video_path} | fps≈{fps:.1f} | 每 {step} 帧存一张 | tag={tag}")

    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = save_frame(frame, tag, source)
            count += 1
            print(f"  saved [{count}/{max_frames}] frame#{idx} {os.path.basename(p)}")
        idx += 1

    cap.release()
    print(f"[DONE] 共保存 {count} 张 -> {RAW_DIR}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="YOLO 第1周：教室场景抽帧")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, default=None, help="摄像头索引，通常 0")
    src.add_argument("--video", type=str, default=None, help="视频文件路径")
    ap.add_argument("--tag", type=str, required=True,
                    help="场景标签: front_normal, head_down, side_face, multi_person, backlight, low_light, walk")
    ap.add_argument("--interval", type=float, default=1.0, help="抽帧间隔（秒）；摄像头模式为最小间隔")
    ap.add_argument("--max", type=int, default=50, help="最多保存张数")
    args = ap.parse_args()

    if args.camera is not None:
        return run_camera(args.camera, args.tag, args.interval, args.max)
    return run_video(args.video, args.tag, args.interval, args.max)


if __name__ == "__main__":
    sys.exit(main() or 0)