# -*- coding: utf-8 -*-
"""Week3: MediaPipe Face Mesh on face ROI — EAR, down-head, forward-head.

Supports:
- mediapipe < 0.10.11: mp.solutions.face_mesh
- mediapipe >= 0.10.11: tasks FaceLandmarker (auto-download .task model)
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

_MP_IMPORT_ERR = ""
try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERR = str(e)

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_model(base_dir: str) -> str:
    path = os.path.join(base_dir, "models", "face_landmarker.task")
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[Vision] Downloading face_landmarker.task -> {path} ...")
    urllib.request.urlretrieve(_FACE_LANDMARKER_URL, path)
    return path


class _ClassicMeshWrapper:
    def __init__(self, face_mesh):
        self._fm = face_mesh

    def process_rgb(self, rgb):
        return self._fm.process(rgb)


class _TasksMeshWrapper:
    def __init__(self, landmarker):
        self._lm = landmarker

    def process_rgb(self, rgb):
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect(mp_image)


def _create_face_mesh_backend(base_dir: str):
    """Return (wrapper, api_name) or raise."""
    if mp is None:
        raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERR or 'unknown'}")

    if hasattr(mp, "solutions"):
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        return _ClassicMeshWrapper(fm), "solutions"

    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_face_landmarker_model(base_dir)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return _TasksMeshWrapper(mp_vision.FaceLandmarker.create_from_options(opts)), "tasks"


def _extract_landmark_list(result, api: str):
    if api == "solutions":
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark
    faces = getattr(result, "face_landmarks", None)
    if not faces:
        return None
    return faces[0]


class FaceMeshAnalyzer:
    def __init__(self):
        self.face_mesh = None
        self._api = ""
        self.ready = False
        self.last_error = ""

        self.blink_times = deque(maxlen=256)
        self.eye_closed_start_t = None
        self.closed_ratio_ema = 0.0
        self.ear_close_thresh = 0.25
        self.blink_min_duration = 0.06

        self.pitch_threshold_deg = 15.0
        self.forward_ratio_baseline = None
        self.forward_ratio_alpha = 0.02
        self.forward_ratio_margin = 0.04

        self.last_ear = None
        self.last_eye_fatigue_index = 0.0
        self.last_posture_risk = 0.0
        self.last_down_head_risk = 0.0
        self.last_forward_head_risk = 0.0
        self.last_smile_score = 0.0
        self.last_frown_score = 0.0
        self.last_mouth_open_score = 0.0
        self.last_eye_boxes = 0
        self._frame_idx = 0
        self.tracking_stride = 1

        if mp is None:
            self.last_error = f"mediapipe not installed ({_MP_IMPORT_ERR})".strip()
            return
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.face_mesh, self._api = _create_face_mesh_backend(base_dir)
            self.ready = True
        except Exception as e:
            self.last_error = str(e)
            self.face_mesh = None

    @staticmethod
    def _landmark_xy(landmarks, idx, w, h):
        p = landmarks[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    @staticmethod
    def _euclidean(a, b):
        return float(np.linalg.norm(a - b))

    def _compute_ear(self, landmarks, w, h):
        li = [33, 160, 158, 133, 153, 144]
        ri = [362, 385, 387, 263, 373, 380]
        l = [self._landmark_xy(landmarks, i, w, h) for i in li]
        r = [self._landmark_xy(landmarks, i, w, h) for i in ri]
        left_ear = (self._euclidean(l[1], l[5]) + self._euclidean(l[2], l[4])) / (
            2.0 * self._euclidean(l[0], l[3]) + 1e-6
        )
        right_ear = (self._euclidean(r[1], r[5]) + self._euclidean(r[2], r[4])) / (
            2.0 * self._euclidean(r[0], r[3]) + 1e-6
        )
        return 0.5 * (left_ear + right_ear)

    def _compute_posture_features(self, landmarks, w, h):
        nose = self._landmark_xy(landmarks, 1, w, h)
        left_ear = self._landmark_xy(landmarks, 234, w, h)
        right_ear = self._landmark_xy(landmarks, 454, w, h)
        ear_mid = 0.5 * (left_ear + right_ear)
        ear_dist = self._euclidean(left_ear, right_ear)
        dx = abs(float(nose[0] - ear_mid[0]))
        dy = float(nose[1] - ear_mid[1])
        down_pitch_deg = float(np.degrees(np.arctan2(dy, dx + 1e-6)))
        forward_ratio = float(dy / (ear_dist + 1e-6))
        return down_pitch_deg, forward_ratio

    def _update_eye_fatigue(self, ear, now_t):
        is_closed = ear < self.ear_close_thresh
        if is_closed and self.eye_closed_start_t is None:
            self.eye_closed_start_t = now_t
        elif (not is_closed) and self.eye_closed_start_t is not None:
            close_dur = now_t - self.eye_closed_start_t
            if close_dur >= self.blink_min_duration:
                self.blink_times.append(now_t)
            self.eye_closed_start_t = None
        while self.blink_times and (now_t - self.blink_times[0] > 60.0):
            self.blink_times.popleft()
        blink_per_min = float(len(self.blink_times))
        close_duration = (now_t - self.eye_closed_start_t) if self.eye_closed_start_t is not None else 0.0
        alpha = 0.12
        self.closed_ratio_ema = (1 - alpha) * self.closed_ratio_ema + alpha * (1.0 if is_closed else 0.0)
        low_blink_score = np.clip((10.0 - blink_per_min) / 10.0, 0.0, 1.0)
        long_close_score = np.clip(close_duration / 0.35, 0.0, 1.0)
        fatigue = 0.62 * self.closed_ratio_ema + 0.30 * long_close_score + 0.08 * low_blink_score
        self.last_eye_fatigue_index = float(np.clip(fatigue, 0.0, 1.0))

    def _mouth_geometry_scores(self, landmarks, w, h):
        """smile / frown / open 均在 0~1，辅助 FER+。"""
        try:
            lm = lambda i: self._landmark_xy(landmarks, i, w, h)
            left, right = lm(61), lm(291)
            upper, lower = lm(13), lm(14)
            mouth_w = self._euclidean(left, right) + 1e-6
            corner_y = 0.5 * (left[1] + right[1])
            center_y = 0.5 * (upper[1] + lower[1])
            lift = (center_y - corner_y) / mouth_w
            open_gap = self._euclidean(upper, lower) / mouth_w
            mouth_open = float(np.clip((open_gap - 0.12) * 3.5, 0.0, 1.0))
            # 张嘴时上下唇拉开，嘴角相对中心会误判为「上抬」→ 必须先算 open 再压 smile
            smile_raw = float(np.clip(lift * 4.0 + 0.15, 0.0, 1.0))
            smile = smile_raw * float(np.clip(1.0 - mouth_open * 1.15, 0.0, 1.0))
            frown = float(np.clip((-lift) * 4.0 + 0.12, 0.0, 1.0))
            return smile, frown, mouth_open
        except Exception:
            return 0.0, 0.0, 0.0

    def _update_posture_risk(self, down_pitch_deg, forward_ratio):
        down_excess = max(0.0, down_pitch_deg - self.pitch_threshold_deg)
        self.last_down_head_risk = float(np.clip(down_excess * 0.32, 0.0, 1.0))
        if self.forward_ratio_baseline is None:
            self.forward_ratio_baseline = float(forward_ratio)
        else:
            a = self.forward_ratio_alpha
            self.forward_ratio_baseline = (1.0 - a) * self.forward_ratio_baseline + a * float(forward_ratio)
        forward_excess = float(forward_ratio - self.forward_ratio_baseline - self.forward_ratio_margin)
        self.last_forward_head_risk = float(np.clip(forward_excess / 0.08, 0.0, 1.0))
        self.last_posture_risk = float(
            np.clip(0.70 * self.last_down_head_risk + 0.30 * self.last_forward_head_risk, 0.0, 1.0)
        )

    def process_bgr_roi(self, roi_bgr):
        if not self.ready or self.face_mesh is None or roi_bgr is None or roi_bgr.size == 0:
            return None
        self._frame_idx += 1
        if self._frame_idx % self.tracking_stride != 0:
            return {
                "ear": self.last_ear,
                "eye_fatigue": self.last_eye_fatigue_index,
                "posture_risk": self.last_posture_risk,
                "down_head_risk": self.last_down_head_risk,
                "forward_head_risk": self.last_forward_head_risk,
                "smile_score": getattr(self, "last_smile_score", 0.0),
                "frown_score": getattr(self, "last_frown_score", 0.0),
                "mouth_open_score": getattr(self, "last_mouth_open_score", 0.0),
                "ok": True,
                "cached": True,
            }
        h, w = roi_bgr.shape[:2]
        if w < 32 or h < 32:
            return None
        work = roi_bgr
        # 小脸放大，tasks API 在 <160px 时极易丢点
        min_side = min(w, h)
        if min_side < 160:
            scale = 160.0 / min_side
            nw = int(w * scale)
            nh = int(h * scale)
            work = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            h, w = nh, nw
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process_rgb(rgb)
        landmarks = _extract_landmark_list(result, self._api)
        if landmarks is None:
            return None
        ear = self._compute_ear(landmarks, w, h)
        self.last_ear = float(ear)
        self.last_eye_boxes = 2
        now_t = time.time()
        self._update_eye_fatigue(ear, now_t)
        down_pitch_deg, forward_ratio = self._compute_posture_features(landmarks, w, h)
        self._update_posture_risk(down_pitch_deg, forward_ratio)
        smile_score, frown_score, mouth_open = self._mouth_geometry_scores(landmarks, w, h)
        self.last_smile_score = smile_score
        self.last_frown_score = frown_score
        self.last_mouth_open_score = mouth_open
        return {
            "ear": self.last_ear,
            "eye_fatigue": self.last_eye_fatigue_index,
            "posture_risk": self.last_posture_risk,
            "down_head_risk": self.last_down_head_risk,
            "forward_head_risk": self.last_forward_head_risk,
            "smile_score": smile_score,
            "frown_score": frown_score,
            "mouth_open_score": mouth_open,
            "ok": True,
            "cached": False,
        }


def crop_face_roi_from_person(frame, x1, y1, x2, y2, face_cascade=None):
    fh, fw = frame.shape[:2]
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    mx = int(0.08 * pw)
    fy1 = y1
    fy2 = y1 + int(0.55 * ph)
    fx1 = max(0, x1 + mx)
    fx2 = min(fw - 1, x2 - mx)
    fy2 = min(fh - 1, max(fy2, fy1 + 32))
    roi = frame[fy1:fy2, fx1:fx2]
    if face_cascade is not None and not face_cascade.empty() and roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(28, 28))
        if len(faces) > 0:
            (fx, fy, fw2, fh2) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            pad = int(0.12 * max(fw2, fh2))
            x0 = max(0, fx - pad)
            y0 = max(0, fy - pad)
            x3 = min(roi.shape[1], fx + fw2 + pad)
            y3 = min(roi.shape[0], fy + fh2 + pad)
            face = roi[y0:y3, x0:x3]
            if face.size > 0:
                return face, (fx1 + x0, fy1 + y0, fx1 + x3, fy1 + y3)
    return roi, (fx1, fy1, fx2, fy2)