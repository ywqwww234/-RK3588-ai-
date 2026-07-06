#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频/图片裁脸并放入 8 类文件夹，供 train_fer.py 使用。

用法 A — 已有按类分好的图（推荐）:
  把图放进 data/fer_labeled/train/{neutral,happy,surprise,sad,anger,disgust,fear,contempt}/

用法 B — 从 raw 图 + YuNet 自动裁脸（需你随后检查/移动错类）:
  python tools_fer_train/prepare_dataset.py --images data/yolo/images/raw --out data/fer_crops/raw_neutral

用法 C — 摄像头现场采集（按键标情绪）:
  python tools_fer_train/prepare_dataset.py --capture --out data/fer_labeled/train
  键: 0 neutral  1 happy  2 surprise  3 sad  4 anger  5 disgust  6 fear  7 contempt  空格保存  q 退出
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def ensure_class_dirs(base: str):
    for c in CLASSES:
        os.makedirs(os.path.join(base, c), exist_ok=True)


def crop_faces_yunet(frame, yunet):
    fb, _ = yunet.detect_best_face(frame, None)
    return fb


def run_capture(out_root: str, cam: int = 0):
    from vision_face_detect import YuNetFaceDetector

    ensure_class_dirs(out_root)
    det = YuNetFaceDetector(ROOT)
    if not det.ready:
        print(f"[ERR] YuNet: {det.last_error}")
        return 1
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("[ERR] 打不开摄像头")
        return 1
    label_idx = 0
    counts = {c: 0 for c in CLASSES}
    out_root = os.path.abspath(out_root)
    print("保存目录:", out_root)
    print("操作: 先鼠标点一下「FER capture」窗口 | 0-7 选类别 | S 或 空格 保存 | Q 退出")
    print("类别:", CLASSES)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        face = crop_faces_yunet(frame, det)
        disp = frame.copy()
        if face is not None:
            fh, fw = face.shape[:2]
            disp[10 : 10 + min(fh, 120), 10 : 10 + min(fw, 120)] = cv2.resize(
                face, (min(fw, 120), min(fh, 120))
            )
            cv2.putText(disp, "FACE OK - press S to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "NO FACE - move closer / face camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        txt = f"label={CLASSES[label_idx]}  saved={counts[CLASSES[label_idx]]}  total={sum(counts.values())}"
        cv2.putText(disp, txt, (10, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("FER capture", disp)
        # waitKey(20) 比 1 更容易在 Windows 上收到按键（需窗口有焦点）
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        if ord("0") <= k <= ord("7"):
            label_idx = k - ord("0")
            print(f"[label] -> {CLASSES[label_idx]}")
        save_key = k in (ord(" "), ord("s"), ord("S"))
        if save_key:
            if face is None:
                print("[WARN] 未检测到脸，没保存。正对镜头、光线亮一点再按 S")
            else:
                cname = CLASSES[label_idx]
                fn = f"{cname}_{int(time.time() * 1000)}.jpg"
                path = os.path.join(out_root, cname, fn)
                cv2.imwrite(path, face)
                counts[cname] += 1
                print(f"[OK] saved ({counts[cname]}) {path}")
    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] counts:", counts)
    return 0


def run_batch_crop(images_dir: str, out_class: str, out_root: str):
    from vision_face_detect import YuNetFaceDetector

    det = YuNetFaceDetector(ROOT)
    if not det.ready:
        print(f"[ERR] YuNet: {det.last_error}")
        return 1
    dst = os.path.join(out_root, out_class)
    os.makedirs(dst, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    for name in os.listdir(images_dir):
        if os.path.splitext(name.lower())[1] not in exts:
            continue
        path = os.path.join(images_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            continue
        face = crop_faces_yunet(frame, det)
        if face is None:
            continue
        cv2.imwrite(os.path.join(dst, f"{os.path.splitext(name)[0]}_face.jpg"), face)
        n += 1
    print(f"[DONE] cropped {n} -> {dst}")
    return 0


def count_dataset(root: str):
    for split in ("train", "val"):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            continue
        print(f"--- {split} ---")
        for c in CLASSES:
            d = os.path.join(base, c)
            if os.path.isdir(d):
                print(f"  {c}: {len(os.listdir(d))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="摄像头采集")
    ap.add_argument("--images", default="", help="批量裁脸源目录")
    ap.add_argument("--class-name", default="neutral", choices=CLASSES)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "fer_labeled", "train"))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--count", default="", help="统计数据集 data/fer_labeled")
    args = ap.parse_args()

    if args.count:
        count_dataset(args.count)
        return 0
    if args.capture:
        return run_capture(args.out, args.cam)
    if args.images:
        return run_batch_crop(args.images, args.class_name, os.path.dirname(args.out) or args.out)
    ensure_class_dirs(args.out)
    val = args.out.replace("train", "val")
    ensure_class_dirs(val)
    print(f"[INFO] 已创建目录。请把图片放入:\n  {args.out}/<class>/\n  {val}/<class>/")
    print("每类建议 train≥300 val≥50。然后运行 train_fer.py")
    count_dataset(os.path.dirname(args.out.rstrip("/\\").replace("\\train", "")) or os.path.join(ROOT, "data", "fer_labeled"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)