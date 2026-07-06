"""
本地视觉分析器。

基于 OpenCV、MediaPipe 与 FER 模型完成人脸检测、表情识别、眼疲劳和姿态风险估计。
"""

import os
import sys
import time
from collections import deque
import config

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

YOLO = None
ULTRA_IMPORT_ERR = None  # 方案A不使用

# 兼容部分环境下 mediapipe 缺少 solutions 子模块的情况
if mp is not None and not hasattr(mp, "solutions"):
    mp = None


class LocalFaceAnalyzer:
    """本地视觉推理入口，输出表情概率、眼疲劳和姿态风险。"""
    def __init__(self, model_path='models/emotion-ferplus-8.onnx'):
        # 1. 轻量级人脸检测（用于表情分类 ROI）
        # 兼容 PyInstaller：优先从 _MEIPASS 下查找 haarcascade 文件
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        cv_haar_dir = os.path.join(base_dir, 'cv2', 'data')

        face_xml = os.path.join(cv_haar_dir, 'haarcascade_frontalface_default.xml')
        eye_xml = os.path.join(cv_haar_dir, 'haarcascade_eye_tree_eyeglasses.xml')

        if not os.path.exists(face_xml):
            face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(eye_xml):
            eye_xml = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        # 回退眼睛检测（MediaPipe 不可用或丢跟踪时使用）
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

        if self.face_cascade.empty():
            print(f"[Vision Warn] 人脸级联加载失败: {face_xml}")
        if self.eye_cascade.empty():
            print(f"[Vision Warn] 眼睛级联加载失败: {eye_xml}")

        # 2. OpenCV DNN 表情模型
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.emotions = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']
            print('OpenCV DNN 视觉模型加载成功！')
        except Exception as e:
            print(f'本地视觉模型加载失败，请检查 models 文件夹: {e}')
            self.net = None

        # 2.5 OpenCV DNN 人脸检测（阶段B方案A）
        self.dnn_face_net = None
        dnn_pb = os.path.join(base_dir, 'models', 'opencv_face_detector_uint8.pb')
        dnn_pbtxt = os.path.join(base_dir, 'models', 'opencv_face_detector.pbtxt')
        alt_pb = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'opencv_face_detector_uint8.pb')
        alt_pbtxt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'opencv_face_detector.pbtxt')
        if not os.path.exists(dnn_pb):
            dnn_pb = alt_pb
        if not os.path.exists(dnn_pbtxt):
            dnn_pbtxt = alt_pbtxt

        try:
            if os.path.exists(dnn_pb) and os.path.exists(dnn_pbtxt):
                self.dnn_face_net = cv2.dnn.readNetFromTensorflow(dnn_pb, dnn_pbtxt)
                print(f'[Vision] DNN face detector loaded: {dnn_pb}')
            else:
                print('[Vision Warn] DNN face model files not found, fallback to Haar path.')
        except Exception as _exc:
            self.dnn_face_net = None
            print(f'[Vision Warn] DNN face load failed: {_exc}')

        # 3. MediaPipe FaceMesh（低开销参数）
        self.face_mesh = None
        if mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )

        # 4. 眼疲劳追踪状态
        self.blink_times = deque(maxlen=256)
        self.eye_closed_start_t = None
        self.closed_ratio_ema = 0.0
        # 提高敏感度：更容易判定为闭眼/眯眼
        self.ear_close_thresh = 0.25
        self.blink_min_duration = 0.06  # 秒，过滤噪声（更敏感）

        # 5. 姿态追踪状态（可解释：探头 + 低头）
        # 高敏模式：15°开始即高风险
        self.pitch_threshold_deg = 15.0
        self.forward_ratio_baseline = None
        self.forward_ratio_alpha = 0.02
        self.forward_ratio_margin = 0.04
        self.last_forward_head_risk = 0.0
        self.last_down_head_risk = 0.0

        # 快速修正: 死区 + EMA + 连续帧判定
        self.posture_deadzone = 0.08
        self.posture_ema_alpha = 0.25
        self.posture_high_gate = 0.20
        self.posture_high_need_frames = 8
        self.posture_low_release_frames = 5
        self._posture_high_streak = 0
        self._posture_low_streak = 0

        # 6. 降低 CPU：隔帧做 FaceMesh
        self.frame_idx = 0
        self.tracking_stride = 2
        self.last_eye_fatigue_index = 0.0
        self.last_posture_risk = 0.0

        # 7. 跟踪诊断信息（便于现场调参）
        self.last_ear = None
        self.last_eye_boxes = 0
        self.last_track_mode = "none"
        self.face_miss_streak = 0
        self.face_miss_limit = 10

        # 8. 表情稳定器（提升准确观感）
        self._expr_hist = deque(maxlen=12)
        self._expr_hold = 'none'
        self._expr_hold_prob = 0.0

        # 9. Haar 回退姿态估计缓存
        self._fallback_prev_face = None

        # 10. 质量门控状态（阶段A）
        self.last_conf_level = 'LOW'
        self.last_gate_on = True
        self._stable_expr = 'none'
        self._stable_expr_prob = 0.0

        self.conf_miss_good = int(getattr(config, 'VISION_CONF_MISS_GOOD', 2))
        self.conf_ear_good = float(getattr(config, 'VISION_CONF_EAR_GOOD', 0.20))
        self.conf_expr_hold_min = float(getattr(config, 'VISION_CONF_EXPR_HOLD_MIN', 0.45))
        self.gate_decay_posture = float(getattr(config, 'VISION_GATE_DECAY_POSTURE', 0.94))
        self.gate_decay_forward = float(getattr(config, 'VISION_GATE_DECAY_FORWARD', 0.95))
        self.gate_decay_down = float(getattr(config, 'VISION_GATE_DECAY_DOWN', 0.95))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-8)

    @staticmethod
    def _landmark_xy(landmarks, idx, w, h):
        p = landmarks[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    @staticmethod
    def _euclidean(a, b):
        return float(np.linalg.norm(a - b))

    def _compute_ear(self, landmarks, w, h):
        """
        EAR 公式（单眼）：
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        双眼 EAR 取均值。
        MediaPipe 索引（轻量可用）：
          左眼: [33, 160, 158, 133, 153, 144]
          右眼: [362, 385, 387, 263, 373, 380]
        """
        li = [33, 160, 158, 133, 153, 144]
        ri = [362, 385, 387, 263, 373, 380]

        l = [self._landmark_xy(landmarks, i, w, h) for i in li]
        r = [self._landmark_xy(landmarks, i, w, h) for i in ri]

        left_ear = (self._euclidean(l[1], l[5]) + self._euclidean(l[2], l[4])) / (2.0 * self._euclidean(l[0], l[3]) + 1e-6)
        right_ear = (self._euclidean(r[1], r[5]) + self._euclidean(r[2], r[4])) / (2.0 * self._euclidean(r[0], r[3]) + 1e-6)
        return 0.5 * (left_ear + right_ear)

    def _compute_posture_features(self, landmarks, w, h):
        """
        低算力姿态特征（仅鼻尖+双耳）：
        - down_pitch_deg: 低头角近似（鼻尖相对耳中点的垂向角）
        - forward_ratio: 探头比值（鼻尖到耳线中点的前向量 / 耳间距）

        定义：
          ear_mid = (L_ear + R_ear) / 2
          ear_dist = ||L_ear - R_ear||
          dx = |nose.x - ear_mid.x|
          dy = nose.y - ear_mid.y
          down_pitch_deg = atan2(dy, dx + eps) * 180/pi
          forward_ratio = dy / (ear_dist + eps)
        """
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

        # 60 秒眨眼频率
        while self.blink_times and (now_t - self.blink_times[0] > 60.0):
            self.blink_times.popleft()
        blink_per_min = float(len(self.blink_times))

        # 当前闭眼持续时长
        close_duration = (now_t - self.eye_closed_start_t) if self.eye_closed_start_t is not None else 0.0

        # EMA 近似 PERCLOS（提升收敛速度）
        alpha = 0.12
        self.closed_ratio_ema = (1 - alpha) * self.closed_ratio_ema + alpha * (1.0 if is_closed else 0.0)

        # eye_fatigue_index [0,1]：强调“连续闭眼”
        low_blink_score = np.clip((10.0 - blink_per_min) / 10.0, 0.0, 1.0)
        long_close_score = np.clip(close_duration / 0.35, 0.0, 1.0)
        fatigue = 0.62 * self.closed_ratio_ema + 0.30 * long_close_score + 0.08 * low_blink_score
        self.last_eye_fatigue_index = float(np.clip(fatigue, 0.0, 1.0))

    def _update_eye_fatigue_from_closed(self, is_closed, now_t):
        """无 EAR 时的回退闭眼更新逻辑。"""
        pseudo_ear = (self.ear_close_thresh - 0.02) if is_closed else (self.ear_close_thresh + 0.02)
        self._update_eye_fatigue(pseudo_ear, now_t)

    def _update_posture_risk(self, down_pitch_deg, forward_ratio):
        """
        posture_risk = 0.55 * down_head_risk + 0.45 * forward_head_risk

        - down_head_risk: 基于低头角 down_pitch_deg
        - forward_head_risk: 基于探头比值 forward_ratio（相对个体基线）
        """
        # 1) 低头风险（高敏映射：15° -> >=0.8）
        down_excess = max(0.0, down_pitch_deg - self.pitch_threshold_deg)
        # 仅增加约 2.5° 就到 0.8（0.8/2.5=0.32）
        self.last_down_head_risk = float(np.clip(down_excess * 0.32, 0.0, 1.0))

        # 2) 探头风险（自适应基线，慢速更新）
        if self.forward_ratio_baseline is None:
            self.forward_ratio_baseline = float(forward_ratio)
        else:
            a = self.forward_ratio_alpha
            self.forward_ratio_baseline = (1.0 - a) * self.forward_ratio_baseline + a * float(forward_ratio)

        forward_excess = float(forward_ratio - self.forward_ratio_baseline - self.forward_ratio_margin)
        # 高敏探头映射：轻微前伸也快速拉高
        self.last_forward_head_risk = float(np.clip(forward_excess / 0.08, 0.0, 1.0))

        # 3) 融合总姿态风险（偏向低头项）
        fused_raw = float(np.clip(0.70 * self.last_down_head_risk + 0.30 * self.last_forward_head_risk, 0.0, 1.0))

        # 4) 死区：轻微抖动直接视为0
        if fused_raw < self.posture_deadzone:
            fused_raw = 0.0

        # 5) EMA 平滑：压制单帧尖峰
        self.last_posture_risk = float(
            (1.0 - self.posture_ema_alpha) * float(self.last_posture_risk) +
            self.posture_ema_alpha * fused_raw
        )

        # 6) 连续帧门控：需要连续异常才“放行”，连续低值才“释放”
        if self.last_posture_risk >= self.posture_high_gate:
            self._posture_high_streak += 1
            self._posture_low_streak = 0
        else:
            self._posture_low_streak += 1
            self._posture_high_streak = max(0, self._posture_high_streak - 1)

        if self._posture_high_streak < self.posture_high_need_frames:
            # 未达到连续帧条件时压低输出
            self.last_posture_risk = float(self.last_posture_risk * 0.35)
        elif self._posture_low_streak >= self.posture_low_release_frames:
            # 连续低值时快速释放
            self.last_posture_risk = float(self.last_posture_risk * 0.75)

        self.last_posture_risk = float(np.clip(self.last_posture_risk, 0.0, 1.0))

    def _stabilize_expr(self, expr_type, expr_prob):
        p = float(expr_prob)
        e = str(expr_type)
        if e != 'none' and p >= 0.45:
            self._expr_hist.append((e, p))
        elif e == 'none' and self._expr_hold != 'none' and self._expr_hold_prob >= 0.55:
            # 短时丢失时维持上一状态，减少 none 抖动
            self._expr_hist.append((self._expr_hold, self._expr_hold_prob * 0.92))

        if not self._expr_hist:
            self._expr_hold, self._expr_hold_prob = 'none', 0.0
            return 'none', 0.0

        score = {}
        for name, prob in self._expr_hist:
            score[name] = score.get(name, 0.0) + float(prob)
        best = max(score.items(), key=lambda kv: kv[1])[0]
        avg_p = float(np.mean([pp for nn, pp in self._expr_hist if nn == best]))
        self._expr_hold, self._expr_hold_prob = best, avg_p
        return best, avg_p

    def analyze_face(self, frame):
        """
        返回: (expr_type, expr_prob, eye_fatigue_index, posture_risk)
        - expr_type: smile / negative / neutral / none
        - expr_prob: 对应概率
        - eye_fatigue_index: [0,1]
        - posture_risk: [0,1]
        """
        expr_type, expr_prob = 'none', 0.0

        # ---------- A) 表情（Haar + FERPlus；模型缺失时仅跳过表情） ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        # 阶段B方案A：优先 OpenCV DNN 人脸检测，失败再回退 Haar
        if self.dnn_face_net is not None:
            try:
                h0, w0 = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.dnn_face_net.setInput(blob)
                det = self.dnn_face_net.forward()
                for i in range(det.shape[2]):
                    conf = float(det[0, 0, i, 2])
                    if conf < 0.45:
                        continue
                    x1 = int(det[0, 0, i, 3] * w0)
                    y1 = int(det[0, 0, i, 4] * h0)
                    x2 = int(det[0, 0, i, 5] * w0)
                    y2 = int(det[0, 0, i, 6] * h0)
                    x = max(0, x1); y = max(0, y1)
                    w = max(1, min(w0 - x, x2 - x1)); h = max(1, min(h0 - y, y2 - y1))
                    faces.append((x, y, w, h))
                if len(faces) > 0:
                    self.last_track_mode = 'dnn_face'
            except Exception as _exc:
                if not hasattr(self, '_dnn_err_once'):
                    self._dnn_err_once = False
                if not self._dnn_err_once:
                    print(f'[Vision Warn] DNN face predict failed: {_exc}')
                    self._dnn_err_once = True

        if len(faces) == 0:
            if self.face_cascade is not None and (not self.face_cascade.empty()):
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            else:
                faces = []

        if self.net is not None and len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64)).astype(np.float32) / 255.0

            blob = cv2.dnn.blobFromImage(face_roi)
            self.net.setInput(blob)
            outputs = self.net.forward()[0]
            probs = self._softmax(outputs)

            max_idx = int(np.argmax(probs))
            predicted_emotion = self.emotions[max_idx]
            expr_prob = float(probs[max_idx])

            if predicted_emotion == 'happy':
                expr_type = 'smile'
            elif predicted_emotion in ['sad', 'fear', 'anger', 'disgust']:
                expr_type = 'negative'
            elif predicted_emotion in ['neutral', 'surprise', 'contempt']:
                expr_type = 'neutral'
            else:
                expr_type = 'none'

        # ---------- B) 眼疲劳 + 头姿态 ----------
        self.frame_idx += 1
        now_t = time.time()
        h, w = frame.shape[:2]

        tracked = False
        used_fallback = False
        self.last_eye_boxes = 0

        # B1) 主路径：MediaPipe
        if self.face_mesh is not None and (self.frame_idx % self.tracking_stride == 0):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if result.multi_face_landmarks:
                tracked = True
                self.face_miss_streak = 0
                landmarks = result.multi_face_landmarks[0].landmark

                ear = self._compute_ear(landmarks, w, h)
                self.last_ear = float(ear)
                self._update_eye_fatigue(ear, now_t)

                down_pitch_deg, forward_ratio = self._compute_posture_features(landmarks, w, h)
                self._update_posture_risk(down_pitch_deg, forward_ratio)
                self.last_track_mode = 'mediapipe'

        # B2) 回退路径：Haar Eye（解决 MP 失败时眼疲劳长期为0）
        if not tracked:
            self.face_miss_streak += 1
            used_fallback = True

            if self.eye_cascade is not None and (not self.eye_cascade.empty()):
                eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
            else:
                eyes = []
            eye_cnt = len(eyes)
            self.last_eye_boxes = int(eye_cnt)

            # 粗粒度判定：检测到 >=2 只眼 -> 睁眼；否则视作疑似闭眼
            is_closed = eye_cnt < 2
            self.last_ear = 0.18 if is_closed else 0.32
            self._update_eye_fatigue_from_closed(is_closed, now_t)

            # 回退姿态估计：使用 Haar 人脸框中心与尺寸变化估计低头/前倾
            if len(faces) > 0:
                (fx, fy, fw, fh) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                cx = fx + fw * 0.5
                cy = fy + fh * 0.5
                if self._fallback_prev_face is not None:
                    pcx, pcy, pfw, pfh = self._fallback_prev_face
                    dy = max(0.0, (cy - pcy) / max(1.0, frame.shape[0]))
                    scale = (fh / max(1.0, pfh)) - 1.0
                    raw_down = float(np.clip(dy * 8.0, 0.0, 1.0))
                    raw_fwd = float(np.clip(scale * 3.0, 0.0, 1.0))
                    self.last_down_head_risk = 0.75 * self.last_down_head_risk + 0.25 * raw_down
                    self.last_forward_head_risk = 0.75 * self.last_forward_head_risk + 0.25 * raw_fwd
                    self.last_posture_risk = float(np.clip(0.7 * self.last_down_head_risk + 0.3 * self.last_forward_head_risk, 0.0, 1.0))
                self._fallback_prev_face = (cx, cy, fw, fh)

            # 丢脸太久：姿态风险缓慢衰减，避免残值锁死
            if self.face_miss_streak >= self.face_miss_limit:
                self.last_posture_risk = max(0.0, self.last_posture_risk * 0.96)
                self.last_forward_head_risk = max(0.0, self.last_forward_head_risk * 0.96)
                self.last_down_head_risk = max(0.0, self.last_down_head_risk * 0.96)

            self.last_track_mode = 'haar_eye'

        eye_fatigue = float(self.last_eye_fatigue_index)
        posture_risk = float(self.last_posture_risk)

        # ---- 质量门控（阶段A）----
        conf_score = 0
        if self.last_track_mode == 'mediapipe':
            conf_score += 2
        if self.face_miss_streak <= self.conf_miss_good:
            conf_score += 1
        if self.last_eye_boxes >= 1:
            conf_score += 1
        if self.last_ear is not None and self.last_ear > self.conf_ear_good:
            conf_score += 1

        if conf_score >= 4:
            conf_level = 'HIGH'
        elif conf_score >= 2:
            conf_level = 'MID'
        else:
            conf_level = 'LOW'
        self.last_conf_level = conf_level

        gate_on = conf_level == 'LOW'
        self.last_gate_on = gate_on

        # 表情稳定输出：低置信时不让 none 抖动污染决策
        if expr_type != 'none' and float(expr_prob) >= self.conf_expr_hold_min:
            self._stable_expr = expr_type
            self._stable_expr_prob = float(expr_prob)
        if gate_on and self._stable_expr != 'none':
            expr_type = self._stable_expr
            expr_prob = max(0.40, self._stable_expr_prob * 0.90)

        # 姿态门控：低置信时不更新，缓慢释放
        if gate_on:
            self.last_posture_risk = max(0.0, float(self.last_posture_risk) * self.gate_decay_posture)
            self.last_forward_head_risk = max(0.0, float(self.last_forward_head_risk) * self.gate_decay_forward)
            self.last_down_head_risk = max(0.0, float(self.last_down_head_risk) * self.gate_decay_down)
            posture_risk = float(self.last_posture_risk)

        print(
            f">>> [Vision Debug] mode:{self.last_track_mode}, face_miss:{self.face_miss_streak}, "
            f"eyes:{self.last_eye_boxes}, ear:{(self.last_ear if self.last_ear is not None else -1):.3f}, "
            f"表情:{expr_type}, 概率:{float(expr_prob):.2f}, 眼疲劳:{eye_fatigue:.2f}, 姿态:{posture_risk:.2f}, "
            f"fallback:{1 if used_fallback else 0}, conf:{self.last_conf_level}, gate:{1 if self.last_gate_on else 0}"
        )
        return expr_type, float(expr_prob), eye_fatigue, posture_risk
