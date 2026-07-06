#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNNLite 推理封装 — ELF 2 (RK3588) 端运行。

FER 模型: MobileNetV3-Small, 1ch grayscale 64x64, mean128
YOLO 模型: YOLOv8, 3ch RGB 640x640, x/255

用法:
  from rknn_inference import RknnInference
  rknn = RknnInference(fer_path='models/emotion-ferplus-8.rknn',
                       yolo_path='models/yolov8_classroom.rknn')
  if rknn.ready:
      expr, prob = rknn.infer_fer(gray_crop_64)
      boxes, scores = rknn.infer_yolo(frame_bgr)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite

    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    RKNNLite = None

try:
    from tools_yolo_week1.yolo_postprocess import yolov8_post_process

    HAS_POSTPROC = True
except ImportError:
    HAS_POSTPROC = False


class RknnInference:
    """RKNN 推理管理器（FER + YOLOv8 双模型）。"""

    EMOTIONS = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]

    def __init__(
        self,
        fer_path: str = "models/emotion-ferplus-8.rknn",
        yolo_path: str = "models/yolov8_classroom.rknn",
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.45,
    ):
        self.fer_path = fer_path
        self.yolo_path = yolo_path
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        self.fer_rknn = None
        self.yolo_rknn = None
        self.ready_fer = False
        self.ready_yolo = False
        self.last_error = ""

        if not HAS_RKNN:
            self.last_error = "rknnlite.api not available (not on ARM board?)"
            return

        # ── Init FER ──
        self._init_fer()

        # ── Init YOLO ──
        self._init_yolo()

    @property
    def ready(self) -> bool:
        return self.ready_fer and self.ready_yolo

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  FER: MobileNetV3-Small, 1ch grayscale 64x64
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_fer(self):
        if not os.path.isfile(self.fer_path):
            self.last_error = f"FER RKNN not found: {self.fer_path}"
            return
        try:
            self.fer_rknn = RKNNLite()
            ret = self.fer_rknn.load_rknn(self.fer_path)
            if ret != 0:
                self.last_error = f"FER load_rknn failed: ret={ret}"
                return
            ret = self.fer_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                self.last_error = f"FER init_runtime failed: ret={ret}"
                self.fer_rknn.release()
                self.fer_rknn = None
                return
            self.ready_fer = True
            print("[RKNN] FER model ready (NPU_CORE_0)")
        except Exception as e:
            self.last_error = f"FER init exception: {e}"

    def infer_fer(self, crop_gray: np.ndarray) -> Tuple[str, float]:
        """
        FER 推理。
        Args:
            crop_gray: 灰度人脸裁切，任意尺寸
        Returns:
            (emotion_label, probability)
        """
        _, probs = self._fer_raw(crop_gray)
        if probs is None:
            return "neutral", 0.40
        idx = int(np.argmax(probs))
        return self.EMOTIONS[idx], float(probs[idx])

    def infer_fer_full(self, crop_gray: np.ndarray):
        """返回 (emotion_label, probability, probs_dict, neg_sum, sad_p, surprise_p, max_prob)。"""
        probs, raw = self._fer_raw(crop_gray)
        if probs is None or raw is None:
            return "neutral", 0.40, {}, 0.0, 0.0, 0.0, 0.0
        idx = int(np.argmax(probs))
        top3 = sorted(
            [(self.EMOTIONS[i], float(probs[i])) for i in range(len(self.EMOTIONS))],
            key=lambda x: x[1], reverse=True
        )[:3]
        neg_names = ("sad", "fear", "anger", "disgust")
        neg_sum = sum(float(probs[self.EMOTIONS.index(n)]) for n in neg_names if n in self.EMOTIONS)
        sad_p = float(probs[self.EMOTIONS.index("sad")]) if "sad" in self.EMOTIONS else 0.0
        sur_p = float(probs[self.EMOTIONS.index("surprise")]) if "surprise" in self.EMOTIONS else 0.0
        max_p = float(probs[idx])
        return self.EMOTIONS[idx], max_p, top3, neg_sum, sad_p, sur_p, max_p

    def _fer_raw(self, crop_gray: np.ndarray):
        if not self.ready_fer or crop_gray is None or crop_gray.size == 0:
            return None, None
        try:
            face = cv2.resize(crop_gray, (64, 64))
            blob = face.reshape(1, 64, 64, 1).astype(np.float32)
            outputs = self.fer_rknn.inference(inputs=[blob], data_format='nhwc')
            logits = outputs[0][0]
            logits = logits - logits.max()
            ex = np.exp(logits)
            probs = ex / (ex.sum() + 1e-8)
            return probs, logits
        except Exception as e:
            self.last_error = f"FER inference: {e}"
            return None, None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  YOLOv8: 3ch RGB 640x640, x/255
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_yolo(self):
        if not os.path.isfile(self.yolo_path):
            self.last_error = f"YOLO RKNN not found: {self.yolo_path}"
            return
        try:
            self.yolo_rknn = RKNNLite()
            ret = self.yolo_rknn.load_rknn(self.yolo_path)
            if ret != 0:
                self.last_error = f"YOLO load_rknn failed: ret={ret}"
                return
            # YOLO 用全核
            ret = self.yolo_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                self.last_error = f"YOLO init_runtime failed: ret={ret}"
                self.yolo_rknn.release()
                self.yolo_rknn = None
                return
            self.ready_yolo = True
            print("[RKNN] YOLO model ready (NPU_CORE_0_1_2)")
        except Exception as e:
            self.last_error = f"YOLO init exception: {e}"

    def infer_yolo(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        YOLOv8 人物检测推理。

        Args:
            frame_bgr: BGR 图像 (H, W, 3)

        Returns:
            (boxes, classes, scores)
              boxes: (N, 4) [x1,y1,x2,y2] in original image coords
              All None if no person detected
        """
        if not self.ready_yolo or frame_bgr is None:
            return None, None, None

        try:
            orig_h, orig_w = frame_bgr.shape[:2]

            # 预处理: BGR→RGB + letterbox 640×640
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            r = min(640 / orig_h, 640 / orig_w)
            new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            dw = (640 - new_w) // 2
            dh = (640 - new_h) // 2
            img_padded = cv2.copyMakeBorder(
                img_resized, dh, dh, dw, dw,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )

            # NHWC float32, raw [0,255] (RKNN 内部做 x/255)
            blob = np.expand_dims(img_padded, 0).astype(np.float32)  # (1, 640, 640, 3)

            t0 = time.perf_counter()
            outputs = self.yolo_rknn.inference(inputs=[blob], data_format=["nhwc"])
            dt = (time.perf_counter() - t0) * 1000

            if not HAS_POSTPROC:
                self.last_error = "yolo_postprocess not imported"
                return None, None, None

            boxes, classes, scores = yolov8_post_process(
                outputs[0], conf_thres=self.yolo_conf, iou_thres=self.yolo_iou
            )

            if boxes is None:
                return None, None, None

            # 坐标映射回原图
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= r
            boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

            return boxes, classes, scores

        except Exception as e:
            self.last_error = f"YOLO inference: {e}"
            return None, None, None

    def release(self):
        for rk in (self.fer_rknn, self.yolo_rknn):
            if rk is not None:
                try:
                    rk.release()
                except Exception:
                    pass
        self.fer_rknn = None
        self.yolo_rknn = None
        self.ready_fer = False
        self.ready_yolo = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  自测（ELF2 端运行）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import os

    ROOT = os.path.dirname(os.path.abspath(__file__))

    rknn = RknnInference(
        fer_path=os.path.join(ROOT, "models", "emotion-ferplus-8.rknn"),
        yolo_path=os.path.join(ROOT, "models", "yolov8_classroom.rknn"),
    )

    if not rknn.ready:
        print(f"[FAIL] {rknn.last_error}")
        exit(1)

    print("\n=== FER quick test ===")
    dummy_face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    for i in range(5):
        t0 = time.perf_counter()
        emo, prob = rknn.infer_fer(dummy_face)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  [{i}] {emo} ({prob:.3f})  {dt:.1f}ms")

    print("\n=== YOLO quick test ===")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for i in range(3):
        boxes, classes, scores = rknn.infer_yolo(dummy_frame)
        dt_str = rknn.last_error or ""
        n = 0 if boxes is None else len(boxes)
        print(f"  [{i}] {n} detections  (dummy frame, expect 0)")

    rknn.release()
    print("\n[DONE] RKNN inference test complete")
