# -*- coding: utf-8 -*-
"""Optional YuNet face detector (OpenCV Zoo) for FER crop — better than person-box upper half."""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _ensure_yunet_model(base_dir: str) -> str:
    path = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")
    if os.path.isfile(path) and os.path.getsize(path) > 10000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[Vision] Downloading YuNet -> {path} ...")
    urllib.request.urlretrieve(_YUNET_URL, path)
    return path


class YuNetFaceDetector:
    def __init__(self, base_dir: str | None = None):
        self.detector = None
        self.ready = False
        self.last_error = ""
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        if not hasattr(cv2, "FaceDetectorYN"):
            self.last_error = "OpenCV FaceDetectorYN not available (need opencv-contrib >= 4.5.4)"
            return
        try:
            onnx = _ensure_yunet_model(base_dir)
            self.detector = cv2.FaceDetectorYN.create(
                onnx,
                "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.ready = True
        except Exception as e:
            self.last_error = str(e)
            self.detector = None

    def detect_best_face(self, frame_bgr, person_box=None):
        """
        Return (face_bgr, (x1,y1,x2,y2)) in full-frame coords, or (None, None).
        If person_box=(x1,y1,x2,y2) given, prefer face inside that region.
        """
        if not self.ready or self.detector is None or frame_bgr is None:
            return None, None
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None, None

        best = None
        best_score = -1.0
        px1 = py1 = px2 = py2 = None
        if person_box is not None and len(person_box) >= 4:
            px1, py1, px2, py2 = [int(v) for v in person_box[:4]]

        for row in faces:
            x, y, bw, bh = row[0], row[1], row[2], row[3]
            score = float(row[14]) if len(row) > 14 else 0.5
            cx, cy = x + bw * 0.5, y + bh * 0.5
            if px1 is not None:
                if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                    continue
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)

        if best is None and person_box is not None:
            for row in faces:
                score = float(row[14]) if len(row) > 14 else 0.5
                if score > best_score:
                    best_score = score
                    best = (row[0], row[1], row[2], row[3])

        if best is None:
            return None, None

        x, y, bw, bh = best
        pad = 0.15 * max(bw, bh)
        x1 = int(max(0, x - pad))
        y1 = int(max(0, y - pad))
        x2 = int(min(w, x + bw + pad))
        y2 = int(min(h, y + bh + pad))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        return crop, (x1, y1, x2, y2)