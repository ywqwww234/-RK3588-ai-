"""
视觉训练数据集生成器 — MindRoom Guard
==============================================
生成的 10000 条数据完全对齐项目视觉管线：
  YOLOv8n 人脸检测 (bbox) → MediaPipe FaceMesh (关键点/EAR/姿态)
  → FERPlus 表情分类 (8类 softmax) → 眼疲劳 + 姿态风险评估
  → 抑郁视觉风险分级 (L1-L4)

输出的 CSV 可直接用于：
  - YOLOv8 人脸检测训练 (bbox 归一化坐标)
  - YOLOv8-pose 关键点训练 (6 点面部关键点)
  - 表情分类模型训练 (8 类 soft label)
  - 眼疲劳/姿态异常检测模型训练
  - 端到端抑郁视觉风险评估

格式依据：项目图片中的表格格式 + local_vision.py 的 9 维视觉特征
           + camera_thread.py 的风险融合管线

作者: MindRoom Guard 视觉数据组
日期: 2026-06-01
"""

import os
import sys
import csv
import math
import random
import hashlib
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

import numpy as np

# ============================================================
# 全局配置
# ============================================================
N_TOTAL = 10000
N_USERS = 100
SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vision_training_10k.csv")
OUTPUT_YOLO_DIR = os.path.join(OUTPUT_DIR, "yolo_labels_10k")
OUTPUT_SPLIT_CSV = os.path.join(OUTPUT_DIR, "vision_training_10k_split.csv")

# 数据集划分比例 (严格时序切分，无泄漏)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# 抑郁相关视觉行为类别 (YOLOv8 检测类别)
BEHAVIOR_CLASSES = {
    0: "正常面部",
    1: "悲伤/低落表情",
    2: "眼疲劳/困倦",
    3: "低头姿态异常",
    4: "前倾姿态异常",
    5: "长时间闭眼",
    6: "紧张/焦虑表情",
    7: "情感淡漠(flat affect)",
}

# FERPlus 表情类别 -> 本项目映射
EXPRESSION_CLASSES = {
    0: "neutral",
    1: "happy",
    2: "surprise",
    3: "sad",
    4: "anger",
    5: "disgust",
    6: "fear",
    7: "contempt",
}

# 表情→MindRoom Guard 内部映射 (与 risk_calculator.py 一致)
EXPR_TO_MINDROOM = {
    "neutral":   "neutral",
    "happy":     "smile",
    "surprise":  "neutral",
    "sad":       "negative",
    "anger":     "negative",
    "disgust":   "negative",
    "fear":      "negative",
    "contempt":  "negative",
}

# 光线条件及对应检测置信度影响
LIGHTING_CONDITIONS = ["bright", "normal", "dim", "dark"]
LIGHTING_CONF_FACTOR = {"bright": 1.05, "normal": 1.00, "dim": 0.78, "dark": 0.55}

# 遮挡条件
OCCLUSION_LEVELS = ["none", "partial", "heavy"]
OCCLUSION_CONF_FACTOR = {"none": 1.00, "partial": 0.72, "heavy": 0.40}

# 距离条件 (影响 bbox 大小)
DISTANCE_LEVELS = ["near", "mid", "far"]

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# 用户画像生成
# ============================================================
def generate_user_profiles(n_users: int) -> List[dict]:
    """
    生成 N_USERS 份用户画像。
    每份画像决定该用户的基线表情、姿态习惯、疲劳特征。
    模拟大学生群体中 ~18% 具有抑郁倾向的分布。
    """
    profiles = []
    for uid in range(1, n_users + 1):
        # 风险等级分布 (基于真实流调数据: ~18.5% 大学生存在抑郁症状)
        risk_roll = np.random.random()
        if risk_roll < 0.08:
            base_risk_tier = 3  # 极高风险 ~8%
        elif risk_roll < 0.18:
            base_risk_tier = 2  # 高风险 ~10%
        elif risk_roll < 0.40:
            base_risk_tier = 1  # 中等风险 ~22%
        else:
            base_risk_tier = 0  # 低风险 ~60%

        # 根据风险等级设定表情基线 Dirichlet 参数
        if base_risk_tier == 3:
            # 极高风险: 悲伤/恐惧/愤怒多，快乐少
            expr_alpha = np.array([2.5, 0.5, 0.8, 3.5, 1.5, 1.0, 1.5, 0.8])
        elif base_risk_tier == 2:
            # 高风险: 悲伤和中性多
            expr_alpha = np.array([3.5, 1.0, 1.0, 2.5, 1.2, 0.8, 1.2, 0.8])
        elif base_risk_tier == 1:
            # 中等风险: 接近均匀，略偏中性
            expr_alpha = np.array([4.0, 2.0, 1.2, 1.8, 1.0, 0.8, 1.0, 0.8])
        else:
            # 低风险: 快乐和中性为主
            expr_alpha = np.array([3.5, 3.5, 1.5, 1.0, 0.5, 0.3, 0.4, 0.5])

        # 眼疲劳基线 (高风险用户更易疲劳)
        if base_risk_tier >= 2:
            base_eye_fatigue = np.random.beta(2.5, 5.0)  # 偏右 (较高)
            base_ear = np.random.normal(0.22, 0.04)       # 偏小 (更易闭眼)
            base_blink = np.random.normal(8.0, 3.0)        # 眨眼较少 (精神不振)
        else:
            base_eye_fatigue = np.random.beta(1.5, 8.0)
            base_ear = np.random.normal(0.32, 0.03)
            base_blink = np.random.normal(15.0, 4.0)

        # 姿态基线
        if base_risk_tier >= 2:
            base_pitch = np.random.normal(12.0, 5.0)       # 低头更明显
            base_forward = np.random.normal(0.18, 0.06)
        else:
            base_pitch = np.random.normal(5.0, 4.0)
            base_forward = np.random.normal(0.10, 0.04)

        # 用户特定噪声 (模拟个体差异)
        user_noise_scale = np.random.uniform(0.5, 1.5)

        profiles.append({
            "user_id": f"U{uid:04d}",
            "base_risk_tier": base_risk_tier,
            "expr_alpha": expr_alpha,
            "base_eye_fatigue": float(np.clip(base_eye_fatigue, 0.0, 1.0)),
            "base_ear": float(np.clip(base_ear, 0.10, 0.45)),
            "base_blink": float(max(2.0, base_blink)),
            "base_pitch_deg": float(base_pitch),
            "base_forward_ratio": float(np.clip(base_forward, 0.0, 0.5)),
            "user_noise_scale": float(user_noise_scale),
        })
    return profiles


# ============================================================
# 单帧数据生成
# ============================================================
class FrameGenerator:
    """
    单帧视觉数据生成器。
    模拟完整的视觉处理管线: 摄像头→YOLOv8→MediaPipe→FERPlus→风险评估
    """

    def __init__(self, profile: dict, frame_idx: int):
        self.p = profile
        self.fidx = frame_idx

        # 时变噪声相位 (每用户独立)
        seed_val = int(hashlib.md5(f"{profile['user_id']}_{frame_idx}".encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed_val)
        self._np_rng = np.random.RandomState(seed_val % (2**31))

    def _time_sine(self, freq: float, amp: float = 1.0, offset: float = 0.0) -> float:
        """帧级别的时变正弦扰动，模拟生理节律。"""
        return amp * math.sin(self.fidx * freq * 0.05 + offset)

    def _clamp(self, val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(np.clip(val, lo, hi))

    # -------- 光线 & 遮挡 & 距离 --------
    def gen_environment(self) -> Tuple[str, str, str]:
        # 光线: 正常光线为主，暗光和昏暗占比合理
        lighting_weights = [0.10, 0.55, 0.25, 0.10]
        lighting = self._rng.choices(LIGHTING_CONDITIONS, weights=lighting_weights, k=1)[0]

        # 遮挡: 多数无遮挡
        occ_weights = [0.82, 0.13, 0.05]
        occlusion = self._rng.choices(OCCLUSION_LEVELS, weights=occ_weights, k=1)[0]

        # 距离: 中距离为主
        dist_weights = [0.20, 0.60, 0.20]
        distance = self._rng.choices(DISTANCE_LEVELS, weights=dist_weights, k=1)[0]

        return lighting, occlusion, distance

    # -------- YOLOv8 人脸 BBOX (归一化 xywh) --------
    def gen_face_bbox(self, lighting: str, occlusion: str, distance: str
                      ) -> Tuple[float, float, float, float, float]:
        """
        生成 YOLO 格式的归一化人脸框 + 检测置信度。
        正常条件下人脸居中，bbox 占据画面约 15-45%。
        """
        # 基础框大小 (归一化)
        if distance == "near":
            base_w = self._np_rng.normal(0.38, 0.05)
            base_h = self._np_rng.normal(0.48, 0.06)
        elif distance == "mid":
            base_w = self._np_rng.normal(0.22, 0.04)
            base_h = self._np_rng.normal(0.28, 0.05)
        else:  # far
            base_w = self._np_rng.normal(0.10, 0.03)
            base_h = self._np_rng.normal(0.13, 0.04)

        # 中心点: 人脸通常偏画面上中部
        cx = self._np_rng.normal(0.50, 0.06)
        cy = self._np_rng.normal(0.42, 0.06)

        # 夹紧到画面内
        w = self._clamp(base_w, 0.04, 0.70)
        h = self._clamp(base_h, 0.05, 0.85)
        cx = self._clamp(cx, w / 2, 1.0 - w / 2)
        cy = self._clamp(cy, h / 2, 1.0 - h / 2)

        # 检测置信度 (受光线/遮挡/距离影响)
        base_conf = 0.92
        conf = base_conf * LIGHTING_CONF_FACTOR[lighting] * OCCLUSION_CONF_FACTOR[occlusion]
        if distance == "far":
            conf *= 0.82
        elif distance == "near":
            conf *= 0.95
        conf += self._np_rng.normal(0, 0.03)
        conf = self._clamp(conf, 0.30, 0.995)

        return float(cx), float(cy), float(w), float(h), float(conf)

    # -------- 面部关键点 (简化 MediaPipe 6 关键点) --------
    def gen_landmarks(self, cx: float, cy: float, w: float, h: float,
                      pitch: float, yaw: float, roll: float
                      ) -> dict:
        """
        根据 bbox 和头部姿态生成 6 个归一化的面部关键点。
        MediaPipe 索引映射:
          - 鼻尖:    1
          - 左眼中心: (33+133)/2 ≈ 159
          - 右眼中心: (362+263)/2 ≈ 386
          - 嘴中心:   13
          - 左耳:     234
          - 右耳:     454
        """
        # bbox 区域内相对位置 (0=左/上, 1=右/下)
        base_pts = {
            "nose":        (0.50, 0.38),  # 鼻尖在 bbox 内部偏上
            "left_eye":    (0.30, 0.28),
            "right_eye":   (0.70, 0.28),
            "mouth":       (0.50, 0.68),
            "left_ear":    (0.05, 0.35),
            "right_ear":   (0.95, 0.35),
        }

        # head pose 引起的位移 (归一化坐标下的小偏移)
        pitch_dy = pitch / 90.0 * 0.12        # 低头→关键点整体下移
        yaw_dx   = yaw / 90.0 * 0.10           # 转头→左右偏移
        roll_dx  = roll / 90.0 * 0.04

        landmarks = {}
        for name, (rx, ry) in base_pts.items():
            # Yaw: 左右眼/耳向中心偏移
            if "left" in name:
                rx_adj = rx + yaw_dx * (1.0 - abs(rx - 0.5) * 2)
            elif "right" in name:
                rx_adj = rx + yaw_dx * (1.0 - abs(rx - 0.5) * 2)
            else:
                rx_adj = rx + yaw_dx * 0.5

            ry_adj = ry + pitch_dy

            if "eye" in name or name == "mouth":
                ry_adj += roll_dx * (0.5 if "left" in name else -0.5)

            # 转回画面绝对归一化坐标
            abs_x = cx + (rx_adj - 0.5) * w
            abs_y = cy + (ry_adj - 0.5) * h

            # 加噪声
            abs_x += self._np_rng.normal(0, 0.003)
            abs_y += self._np_rng.normal(0, 0.003)

            landmarks[name] = (self._clamp(abs_x), self._clamp(abs_y))

        return landmarks

    # -------- 头部姿态 (pitch/yaw/roll) --------
    def gen_head_pose(self) -> Tuple[float, float, float]:
        """
        生成头部姿态角 (度)。
        pitch: 低头为正 (下颌靠近胸口方向 → 对应抑郁症低头)
        yaw:   右转为正
        roll:  右倾为正
        """
        base_pitch = self.p["base_pitch_deg"]
        # 增加时变 + 随机尖峰 (模拟偶尔严重低头)
        t_var = self._time_sine(0.3, 5.0)
        spike = 0.0
        if self._rng.random() < 0.08:
            spike = abs(self._np_rng.normal(0, 15.0))

        pitch = base_pitch + t_var + spike
        pitch += self._np_rng.normal(0, 2.5)
        pitch = self._clamp(pitch, -15.0, 60.0)

        # Yaw 和 Roll 通常较小
        yaw   = self._np_rng.normal(0, 8.0) + self._time_sine(0.2, 6.0)
        yaw   = self._clamp(yaw, -45.0, 45.0)
        roll  = self._np_rng.normal(0, 4.0) + self._time_sine(0.25, 3.0)
        roll  = self._clamp(roll, -30.0, 30.0)

        return float(pitch), float(yaw), float(roll)

    # -------- EAR 眼纵横比 --------
    def gen_ear(self, pitch: float) -> Tuple[float, float, float]:
        """
        生成 Eye Aspect Ratio。
        左右眼独立，有微小不对称性。
        受头部姿态影响 (低头时 EAR 自然偏小)。
        """
        base_ear = self.p["base_ear"]
        pitch_factor = max(0.0, pitch / 45.0) * 0.06  # 低头时 EAR 偏小

        # 疲劳引起的 EAR 下降 (时变)
        fatigue_drop = self._time_sine(0.15, 0.04, offset=0.0)
        # 偶尔的眨眼谷值
        blink_drop = 0.0
        blink_phase = self.fidx % 120  # 约2秒周期 (对应 ~15 blinks/min)
        if 115 <= blink_phase <= 120:
            blink_drop = self._np_rng.normal(0.12, 0.03)

        ear_left_base = base_ear - pitch_factor - fatigue_drop - blink_drop
        ear_right_base = base_ear - pitch_factor - fatigue_drop - blink_drop

        # 左右不对称 (自然差异)
        ear_left_base += self._np_rng.normal(0, 0.015)
        ear_right_base += self._np_rng.normal(0, 0.015)

        ear_left  = self._clamp(ear_left_base, 0.05, 0.55)
        ear_right = self._clamp(ear_right_base, 0.05, 0.55)
        ear_mean  = (ear_left + ear_right) / 2.0

        return float(ear_left), float(ear_right), float(ear_mean)

    # -------- 眼状态 & 疲劳 --------
    def gen_eye_state(self, ear_mean: float) -> Tuple[str, int, float, float, float]:
        """
        基于 EAR 判定眼状态，并计算疲劳指标。
        返回: eye_state_label, eye_state_id, eye_fatigue_index, blink_rate_pm, perclos
        """
        ear_thresh = self.p["base_ear"] * 0.72  # 个性化闭眼阈值

        if ear_mean < ear_thresh:
            eye_state = "closed"
            eye_state_id = 1
        elif ear_mean < ear_thresh * 1.35:
            eye_state = "squinting"
            eye_state_id = 2
        else:
            eye_state = "open"
            eye_state_id = 0

        # 疲劳指数 (融合 EAR + 历史模拟)
        base_fatigue = self.p["base_eye_fatigue"]
        t_var = self._time_sine(0.12, 0.10)
        fatigue = base_fatigue + t_var + (1.0 - ear_mean / 0.45) * 0.15
        fatigue += self._np_rng.normal(0, 0.04)
        fatigue = self._clamp(fatigue)

        # 眨眼率 (受疲劳影响)
        base_blink = self.p["base_blink"]
        blink_rate = base_blink + (fatigue - 0.5) * 10.0
        blink_rate += self._np_rng.normal(0, 2.0)
        blink_rate = float(max(2.0, min(40.0, blink_rate)))

        # PERCLOS (闭眼时间占比, EMA 近似)
        perclos = fatigue * 0.65 + (1.0 - ear_mean / 0.45) * 0.25
        perclos += self._np_rng.normal(0, 0.03)
        perclos = self._clamp(perclos)

        return eye_state, eye_state_id, float(fatigue), float(blink_rate), float(perclos)

    # -------- 姿态风险评估 --------
    def gen_posture_risk(self, pitch: float, forward_ratio: float) -> Tuple[float, float, float]:
        """
        计算姿态风险分数。
        与 local_vision.py 的 _update_posture_risk 对齐:
          - down_head_risk: 基于低头角 (阈值 15°)
          - forward_head_risk: 基于前倾比
          - posture_risk: 0.7*down + 0.3*forward
        """
        # 低头风险 (高敏映射)
        down_excess = max(0.0, pitch - 15.0)
        down_head_risk = self._clamp(down_excess * 0.32)

        # 探头风险
        forward_baseline = self.p["base_forward_ratio"]
        forward_excess = max(0.0, forward_ratio - forward_baseline - 0.04)
        forward_head_risk = self._clamp(forward_excess / 0.08)

        # 融合
        posture_risk = self._clamp(0.70 * down_head_risk + 0.30 * forward_head_risk)

        # 添加噪声
        down_head_risk += self._np_rng.normal(0, 0.03)
        forward_head_risk += self._np_rng.normal(0, 0.02)
        posture_risk += self._np_rng.normal(0, 0.02)

        return (self._clamp(posture_risk),
                self._clamp(down_head_risk),
                self._clamp(forward_head_risk))

    # -------- FERPlus 表情分布 --------
    def gen_expression(self) -> Tuple[int, str, float, np.ndarray]:
        """
        生成表情分类结果 (Dirichlet 分布 + 情绪状态调制)。
        返回: expr_id, expr_label, expr_conf, 8-dim softmax probs
        """
        alpha = self.p["expr_alpha"].copy()

        # 时变调制: 长时间单一时相后引入情绪波动
        t_sad_burst = self._time_sine(0.05, 1.5, offset=0.0)  # 慢周期悲伤波动
        t_happy_burst = self._time_sine(0.07, 1.0, offset=1.5)
        alpha[3] += max(0, t_sad_burst)      # sad
        alpha[1] += max(0, t_happy_burst)    # happy

        # Dirichlet 采样
        probs = self._np_rng.dirichlet(np.maximum(alpha, 0.1))
        expr_id = int(np.argmax(probs))
        expr_label = EXPRESSION_CLASSES[expr_id]
        expr_conf = float(probs[expr_id])

        # 真实性: 偶尔降低置信度 (模拟分类器不确定)
        if self._rng.random() < 0.12:
            expr_conf *= self._np_rng.uniform(0.55, 0.85)

        return expr_id, expr_label, float(np.clip(expr_conf, 0.0, 1.0)), probs

    # -------- 行为类别判定 --------
    def gen_behavior(self, expr_id: int, expr_label: str,
                     eye_state_id: int, eye_fatigue: float,
                     posture_risk: float, pitch: float, perclos: float,
                     ear_mean: float
                     ) -> Tuple[int, str, float]:
        """
        综合多指标判定当前帧的抑郁相关行为类别。
        采用软决策融合，反映多因素对行为分类的贡献。

        类别: 0=正常面部 1=悲伤/低落 2=眼疲劳/困倦 3=低头异常
              4=前倾异常 5=长时间闭眼 6=紧张/焦虑 7=情感淡漠
        """
        scores = np.zeros(len(BEHAVIOR_CLASSES))

        # 0=正常面部: 始终有合理的基线分
        scores[0] = 0.45

        # 1=悲伤/低落表情: 基于表情类别
        if expr_label == "sad":
            scores[1] = 0.85
        elif expr_label in ("fear", "disgust"):
            scores[1] = 0.55
        elif expr_label == "anger":
            scores[1] = 0.30
        elif expr_label == "contempt":
            scores[1] = 0.35
        elif expr_label == "neutral":
            scores[1] = 0.12
        else:
            scores[1] = 0.05

        # 2=眼疲劳/困倦: 需要较高疲劳阈值才触发
        if eye_fatigue >= 0.58:
            scores[2] = eye_fatigue
        elif eye_fatigue >= 0.40 and eye_state_id >= 1:
            scores[2] = 0.50
        elif eye_state_id == 1:
            scores[2] = 0.40

        # 3=低头姿态异常: pitch > 22° 触发
        if pitch > 22.0:
            scores[3] = min(1.0, (pitch - 18.0) / 28.0)
        elif pitch > 15.0:
            scores[3] = 0.25

        # 4=前倾姿态异常
        if posture_risk >= 0.40:
            scores[4] = 0.55 + (posture_risk - 0.40) * 0.8
        elif posture_risk >= 0.25:
            scores[4] = 0.30

        # 5=长时间闭眼: EAR极低 + 高PERCLOS
        if ear_mean < 0.16 and eye_state_id == 1 and perclos >= 0.45:
            scores[5] = 0.82
        elif eye_state_id == 1 and perclos >= 0.55:
            scores[5] = 0.75
        elif eye_state_id == 1 and eye_fatigue >= 0.50:
            scores[5] = 0.55
        elif perclos >= 0.60:
            scores[5] = 0.50

        # 6=紧张/焦虑表情
        if expr_label == "fear":
            scores[6] = 0.78
        elif expr_label == "anger":
            scores[6] = 0.65
        elif expr_label == "surprise" and eye_fatigue < 0.30:
            scores[6] = 0.28

        # 7=情感淡漠: 中性表情 + 一定疲劳 + 无明显波动
        if expr_label in ("neutral", "contempt") and eye_fatigue >= 0.38 and \
           pitch < 25.0 and posture_risk < 0.45:
            scores[7] = 0.52
        elif expr_label == "neutral" and eye_fatigue < 0.30 and pitch < 18.0:
            scores[7] = 0.20

        # 如果所有非正常类别的分数都很低，强化"正常面部"
        if np.max(scores[1:]) < 0.30:
            scores[0] = 0.90
        elif np.max(scores[1:]) < 0.45:
            scores[0] = 0.50

        # 使用 top-k margin 计算有意义的置信度
        sorted_scores = np.sort(scores)[::-1]
        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        # 置信度 = 最高分 vs (最高分 + 次高分), margin越大置信度越高
        behavior_conf = (top1 - top2) / (top1 + 1e-6)
        # 映射到合理的置信度范围
        behavior_conf = 0.55 + 0.40 * behavior_conf

        behavior_id = int(np.argmax(scores))
        behavior_label = BEHAVIOR_CLASSES[behavior_id]

        # 偶尔添加分类不确定性 (概率扰动，不改类别)
        if self._rng.random() < 0.06:
            behavior_conf *= self._np_rng.uniform(0.75, 0.93)

        return behavior_id, behavior_label, float(np.clip(behavior_conf, 0.35, 0.995))

    # -------- 视觉风险评分与等级 --------
    def gen_visual_risk(self, expr_label: str, expr_conf: float,
                        eye_fatigue: float, posture_risk: float,
                        down_head_risk: float, forward_head_risk: float
                        ) -> Tuple[float, int]:
        """
        根据 camera_thread.py 的风险融合逻辑计算视觉风险分数。
        校准版: 与 paper 报告中的分布对齐 (~18% 高/极高风险)。
        """
        expr_lower = expr_label.lower()

        # base_visual_risk (与 calculate_visual_risk 一致)
        if expr_lower in ("happy",):
            base_risk = 0.06 + 0.06 * (1.0 - expr_conf)
        elif expr_lower in ("neutral", "surprise", "contempt"):
            base_risk = 0.15 + 0.12 * (1.0 - expr_conf)
        elif expr_lower in ("sad", "anger", "disgust", "fear"):
            base_risk = 0.18 + expr_conf * 0.75
        else:
            base_risk = 0.18 + expr_conf * 0.35

        # 视觉混合风险
        visual_blend = (0.45 * base_risk +
                        0.20 * eye_fatigue +
                        0.15 * posture_risk +
                        0.10 * forward_head_risk +
                        0.10 * down_head_risk)
        visual_peak = max(base_risk, eye_fatigue, posture_risk,
                          forward_head_risk, down_head_risk)
        visual_risk = max(visual_blend, visual_peak)

        # 物理损耗阶梯 (提高阈值，只对显著异常触发)
        physical_peak = max(eye_fatigue, posture_risk,
                            forward_head_risk, down_head_risk)
        if physical_peak >= 0.55:
            visual_risk = max(visual_risk, 0.68)
        if physical_peak >= 0.70:
            visual_risk = max(visual_risk, 0.80)

        # 负向表情提升 (置信度要求提高)
        if expr_lower in ("sad", "anger", "fear", "disgust") and expr_conf >= 0.55:
            visual_risk = max(visual_risk, 0.65)
        elif expr_lower == "neutral" and expr_conf >= 0.80 and physical_peak >= 0.35:
            visual_risk = max(visual_risk, 0.50)

        visual_risk = self._clamp(visual_risk)

        # 风险等级 (与 config.py 阈值严格对齐)
        if visual_risk >= 0.80:
            risk_tier = 3
        elif visual_risk >= 0.60:
            risk_tier = 2
        elif visual_risk >= 0.30:
            risk_tier = 1
        else:
            risk_tier = 0

        return float(visual_risk), risk_tier


# ============================================================
# 数据集组装
# ============================================================
CSV_HEADER = [
    "frame_id",
    "timestamp",
    "user_id",
    # 环境条件
    "lighting_cond",
    "occlusion_level",
    "distance_level",
    # YOLOv8 人脸检测框 (归一化 xywh + conf)
    "face_bbox_cx",
    "face_bbox_cy",
    "face_bbox_w",
    "face_bbox_h",
    "face_detect_conf",
    # FERPlus 表情
    "expression_id",
    "expression_label",
    "expression_conf",
    "expr_prob_neutral",
    "expr_prob_happy",
    "expr_prob_surprise",
    "expr_prob_sad",
    "expr_prob_anger",
    "expr_prob_disgust",
    "expr_prob_fear",
    "expr_prob_contempt",
    "mindroom_expr_type",
    # 眼状态 & 疲劳
    "eye_state_id",
    "eye_state_label",
    "ear_left",
    "ear_right",
    "ear_mean",
    "eye_fatigue_index",
    "blink_rate_pm",
    "perclos",
    # 头部姿态
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "forward_head_ratio",
    # 姿态风险
    "posture_risk",
    "down_head_risk",
    "forward_head_risk",
    # 面部关键点 (6 点, 归一化坐标)
    "lm_nose_x", "lm_nose_y",
    "lm_left_eye_x", "lm_left_eye_y",
    "lm_right_eye_x", "lm_right_eye_y",
    "lm_mouth_x", "lm_mouth_y",
    "lm_left_ear_x", "lm_left_ear_y",
    "lm_right_ear_x", "lm_right_ear_y",
    # 行为分类
    "behavior_class_id",
    "behavior_class_label",
    "behavior_confidence",
    # 视觉风险评估
    "visual_risk_score",
    "risk_tier",
    "risk_tier_label",
    # 元信息
    "quality_flag",
    "data_source",
]

RISK_TIER_LABELS = {0: "低风险", 1: "中等风险", 2: "高风险", 3: "极高风险"}


def generate_dataset() -> List[dict]:
    """主生成循环: 100 用户 × 100 帧 = 10000 条"""
    profiles = generate_user_profiles(N_USERS)
    rows = []
    gen = FrameGenerator.__new__  # placeholder
    del gen

    base_ts = datetime(2026, 5, 15, 8, 0, 0)

    print(f"正在生成 {N_TOTAL} 条视觉训练数据...")
    print(f"用户数: {N_USERS}, 每用户帧数: {N_TOTAL // N_USERS}")

    frames_per_user = N_TOTAL // N_USERS  # 100

    global_idx = 0
    for profile in profiles:
        user_start_ts = base_ts + timedelta(
            minutes=np.random.randint(0, 1440))  # 随机起始时间

        for fidx in range(frames_per_user):
            fg = FrameGenerator(profile, fidx)

            # 时间戳
            ts = user_start_ts + timedelta(seconds=fidx)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") + f".{fidx % 1000:03d}"

            # 环境
            lighting, occlusion, distance = fg.gen_environment()

            # 人脸检测
            cx, cy, bw, bh, face_conf = fg.gen_face_bbox(lighting, occlusion, distance)

            # 头部姿态
            pitch, yaw, roll = fg.gen_head_pose()

            # 前倾比
            forward_ratio = profile["base_forward_ratio"]
            forward_ratio += fg._time_sine(0.1, 0.04)
            forward_ratio += np.random.normal(0, 0.02)
            forward_ratio = float(np.clip(forward_ratio, 0.0, 0.6))

            # 关键点
            landmarks = fg.gen_landmarks(cx, cy, bw, bh, pitch, yaw, roll)

            # EAR
            ear_l, ear_r, ear_m = fg.gen_ear(pitch)

            # 眼状态
            eye_state, eye_id, eye_fatigue, blink_rate, perclos = fg.gen_eye_state(ear_m)

            # 姿态风险
            posture_risk, down_risk, fwd_risk = fg.gen_posture_risk(pitch, forward_ratio)

            # 表情
            expr_id, expr_label, expr_conf, expr_probs = fg.gen_expression()

            # 行为
            behav_id, behav_label, behav_conf = fg.gen_behavior(
                expr_id, expr_label, eye_id, eye_fatigue, posture_risk, pitch,
                perclos, ear_m)

            # 视觉风险
            vis_risk, risk_tier = fg.gen_visual_risk(
                expr_label, expr_conf, eye_fatigue, posture_risk, down_risk, fwd_risk)

            # 质量标记
            if face_conf < 0.50 or occlusion == "heavy" or lighting == "dark":
                quality = "LOW"
            elif face_conf < 0.75 or occlusion == "partial" or lighting == "dim":
                quality = "MID"
            else:
                quality = "HIGH"

            # MindRoom 内部表情映射
            mr_expr = EXPR_TO_MINDROOM.get(expr_label, "neutral")

            rows.append({
                "frame_id": global_idx + 1,
                "timestamp": ts_str,
                "user_id": profile["user_id"],
                "lighting_cond": lighting,
                "occlusion_level": occlusion,
                "distance_level": distance,
                "face_bbox_cx": round(cx, 6),
                "face_bbox_cy": round(cy, 6),
                "face_bbox_w": round(bw, 6),
                "face_bbox_h": round(bh, 6),
                "face_detect_conf": round(face_conf, 6),
                "expression_id": expr_id,
                "expression_label": expr_label,
                "expression_conf": round(expr_conf, 6),
                "expr_prob_neutral": round(float(expr_probs[0]), 6),
                "expr_prob_happy": round(float(expr_probs[1]), 6),
                "expr_prob_surprise": round(float(expr_probs[2]), 6),
                "expr_prob_sad": round(float(expr_probs[3]), 6),
                "expr_prob_anger": round(float(expr_probs[4]), 6),
                "expr_prob_disgust": round(float(expr_probs[5]), 6),
                "expr_prob_fear": round(float(expr_probs[6]), 6),
                "expr_prob_contempt": round(float(expr_probs[7]), 6),
                "mindroom_expr_type": mr_expr,
                "eye_state_id": eye_id,
                "eye_state_label": eye_state,
                "ear_left": round(ear_l, 6),
                "ear_right": round(ear_r, 6),
                "ear_mean": round(ear_m, 6),
                "eye_fatigue_index": round(eye_fatigue, 6),
                "blink_rate_pm": round(blink_rate, 3),
                "perclos": round(perclos, 6),
                "head_pitch_deg": round(pitch, 4),
                "head_yaw_deg": round(yaw, 4),
                "head_roll_deg": round(roll, 4),
                "forward_head_ratio": round(forward_ratio, 6),
                "posture_risk": round(posture_risk, 6),
                "down_head_risk": round(down_risk, 6),
                "forward_head_risk": round(fwd_risk, 6),
                "lm_nose_x": round(landmarks["nose"][0], 6),
                "lm_nose_y": round(landmarks["nose"][1], 6),
                "lm_left_eye_x": round(landmarks["left_eye"][0], 6),
                "lm_left_eye_y": round(landmarks["left_eye"][1], 6),
                "lm_right_eye_x": round(landmarks["right_eye"][0], 6),
                "lm_right_eye_y": round(landmarks["right_eye"][1], 6),
                "lm_mouth_x": round(landmarks["mouth"][0], 6),
                "lm_mouth_y": round(landmarks["mouth"][1], 6),
                "lm_left_ear_x": round(landmarks["left_ear"][0], 6),
                "lm_left_ear_y": round(landmarks["left_ear"][1], 6),
                "lm_right_ear_x": round(landmarks["right_ear"][0], 6),
                "lm_right_ear_y": round(landmarks["right_ear"][1], 6),
                "behavior_class_id": behav_id,
                "behavior_class_label": behav_label,
                "behavior_confidence": round(behav_conf, 6),
                "visual_risk_score": round(vis_risk, 6),
                "risk_tier": risk_tier,
                "risk_tier_label": RISK_TIER_LABELS[risk_tier],
                "quality_flag": quality,
                "data_source": "synthetic_MindRoom_v1",
            })

            global_idx += 1
            if global_idx % 1000 == 0:
                print(f"  已生成 {global_idx}/{N_TOTAL} 条...")

    print(f"生成完成: {len(rows)} 条")
    return rows


def write_csv(rows: List[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"CSV 已保存: {path}  ({size_mb:.2f} MB)")


def write_yolo_labels(rows: List[dict], out_dir: str):
    """
    同时生成 YOLOv8 格式的标签文件:
      - 人脸检测: <face_class_id> <cx> <cy> <w> <h>
      - 关键点检测: <class_id> <cx> <cy> <w> <h> <kp1_x> <kp1_y> <kp1_v> ...
    """
    os.makedirs(out_dir, exist_ok=True)

    det_dir = os.path.join(out_dir, "detection")
    pose_dir = os.path.join(out_dir, "pose")
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    for row in rows:
        fid = row["frame_id"]
        fname = f"frame_{fid:06d}"

        # 检测格式 (只取高质量帧)
        if row["quality_flag"] == "HIGH" and row["face_detect_conf"] >= 0.65:
            det_line = (f"0 "  # class 0 = face
                        f"{row['face_bbox_cx']:.6f} "
                        f"{row['face_bbox_cy']:.6f} "
                        f"{row['face_bbox_w']:.6f} "
                        f"{row['face_bbox_h']:.6f}\n")
            with open(os.path.join(det_dir, f"{fname}.txt"), "w") as f:
                f.write(det_line)

            # 关键点格式 (class=0 face, 6 kpts, visibility=2 for all)
            kpts = (
                f"{row['lm_nose_x']:.6f} {row['lm_nose_y']:.6f} 2 "
                f"{row['lm_left_eye_x']:.6f} {row['lm_left_eye_y']:.6f} 2 "
                f"{row['lm_right_eye_x']:.6f} {row['lm_right_eye_y']:.6f} 2 "
                f"{row['lm_mouth_x']:.6f} {row['lm_mouth_y']:.6f} 2 "
                f"{row['lm_left_ear_x']:.6f} {row['lm_left_ear_y']:.6f} 2 "
                f"{row['lm_right_ear_x']:.6f} {row['lm_right_ear_y']:.6f} 2"
            )
            pose_line = (f"0 "
                         f"{row['face_bbox_cx']:.6f} "
                         f"{row['face_bbox_cy']:.6f} "
                         f"{row['face_bbox_w']:.6f} "
                         f"{row['face_bbox_h']:.6f} "
                         f"{kpts}\n")
            with open(os.path.join(pose_dir, f"{fname}.txt"), "w") as f:
                f.write(pose_line)

    n_det = len(os.listdir(det_dir))
    n_pose = len(os.listdir(pose_dir))
    print(f"YOLOv8 检测标签: {n_det} 个 → {det_dir}")
    print(f"YOLOv8 关键点标签: {n_pose} 个 → {pose_dir}")


def write_train_val_test_split(rows: List[dict], path: str):
    """按用户严格时序切分 train/val/test，写入带 split 列的 CSV。"""
    # 按 user_id 分组
    from collections import defaultdict
    user_rows = defaultdict(list)
    for r in rows:
        user_rows[r["user_id"]].append(r)

    # 按用户 ID 排序
    sorted_users = sorted(user_rows.keys())

    n_users = len(sorted_users)
    n_train = int(n_users * TRAIN_RATIO)
    n_val = int(n_users * VAL_RATIO)

    train_users = set(sorted_users[:n_train])
    val_users = set(sorted_users[n_train:n_train + n_val])
    test_users = set(sorted_users[n_train + n_val:])

    for r in rows:
        if r["user_id"] in train_users:
            r["dataset_split"] = "train"
        elif r["user_id"] in val_users:
            r["dataset_split"] = "val"
        else:
            r["dataset_split"] = "test"

    # 写 CSV
    split_header = CSV_HEADER + ["dataset_split"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=split_header)
        writer.writeheader()
        writer.writerows(rows)

    # 统计
    for split_name, users in [("train", train_users), ("val", val_users), ("test", test_users)]:
        count = sum(1 for r in rows if r.get("dataset_split") == split_name
                    or (split_name == "train" and r["user_id"] in train_users))
        if "dataset_split" not in rows[0]:
            count = len(users) * (N_TOTAL // N_USERS)
        print(f"  {split_name}: {len(users)} 用户, ~{len(users) * (N_TOTAL // N_USERS)} 条")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"拆分 CSV 已保存: {path}  ({size_mb:.2f} MB)")


def print_statistics(rows: List[dict]):
    """打印数据集统计信息"""
    print("\n" + "=" * 60)
    print("数据集统计摘要")
    print("=" * 60)

    # 风险等级分布
    tiers = [r["risk_tier"] for r in rows]
    print(f"\n风险等级分布:")
    for t in range(4):
        cnt = tiers.count(t)
        print(f"  L{t+1} ({RISK_TIER_LABELS[t]}): {cnt} 条 ({cnt/len(rows)*100:.1f}%)")

    # 表情分布
    exprs = [r["expression_label"] for r in rows]
    print(f"\n表情分布:")
    for e in sorted(set(exprs)):
        cnt = exprs.count(e)
        print(f"  {e}: {cnt} 条 ({cnt/len(rows)*100:.1f}%)")

    # 行为分布
    behavs = [r["behavior_class_label"] for r in rows]
    print(f"\n行为类别分布:")
    for b in sorted(set(behavs)):
        cnt = behavs.count(b)
        print(f"  {b}: {cnt} 条 ({cnt/len(rows)*100:.1f}%)")

    # 质量分布
    quals = [r["quality_flag"] for r in rows]
    print(f"\n数据质量分布:")
    for q in ["HIGH", "MID", "LOW"]:
        cnt = quals.count(q)
        print(f"  {q}: {cnt} 条 ({cnt/len(rows)*100:.1f}%)")

    # 光线分布
    lights = [r["lighting_cond"] for r in rows]
    print(f"\n光线条件分布:")
    for l in LIGHTING_CONDITIONS:
        cnt = lights.count(l)
        print(f"  {l}: {cnt} 条 ({cnt/len(rows)*100:.1f}%)")

    # 关键数值统计
    vis_risks = [r["visual_risk_score"] for r in rows]
    ear_means = [r["ear_mean"] for r in rows]
    pitches = [r["head_pitch_deg"] for r in rows]
    eye_fats = [r["eye_fatigue_index"] for r in rows]

    print(f"\n关键特征统计:")
    print(f"  visual_risk_score:  mean={np.mean(vis_risks):.4f}, "
          f"std={np.std(vis_risks):.4f}, "
          f"min={np.min(vis_risks):.4f}, max={np.max(vis_risks):.4f}")
    print(f"  ear_mean:           mean={np.mean(ear_means):.4f}, "
          f"std={np.std(ear_means):.4f}, "
          f"min={np.min(ear_means):.4f}, max={np.max(ear_means):.4f}")
    print(f"  head_pitch_deg:     mean={np.mean(pitches):.2f}, "
          f"std={np.std(pitches):.2f}, "
          f"min={np.min(pitches):.2f}, max={np.max(pitches):.2f}")
    print(f"  eye_fatigue_index:  mean={np.mean(eye_fats):.4f}, "
          f"std={np.std(eye_fats):.4f}, "
          f"min={np.min(eye_fats):.4f}, max={np.max(eye_fats):.4f}")

    # 相关性检查 (抑郁风险 ↔ 关键特征)
    print(f"\n皮尔逊相关系数 (vs visual_risk_score):")
    for name, vals in [("eye_fatigue_index", eye_fats),
                        ("ear_mean", ear_means),
                        ("head_pitch_deg", pitches),
                        ("posture_risk", [r["posture_risk"] for r in rows])]:
        corr = np.corrcoef(vis_risks, vals)[0, 1]
        print(f"  {name}: r = {corr:.4f}")

    print(f"\nYOLOv8 训练兼容性检查:")
    bbox_ok = all(0 <= r["face_bbox_cx"] <= 1 and 0 <= r["face_bbox_cy"] <= 1
                  and 0 < r["face_bbox_w"] <= 1 and 0 < r["face_bbox_h"] <= 1
                  for r in rows)
    kpts_ok = all(0 <= r[k] <= 1 for r in rows
                  for k in CSV_HEADER if k.startswith("lm_"))
    print(f"  BBox 归一化合法: {'PASS' if bbox_ok else 'FAIL'}")
    print(f"  关键点归一化合法: {'PASS' if kpts_ok else 'FAIL'}")
    print(f"  行为类别覆盖: {len(set(behavs))}/8 类")
    print(f"  表情类别覆盖: {len(set(exprs))}/8 类")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    rows = generate_dataset()

    # 写入主 CSV
    write_csv(rows, OUTPUT_CSV)

    # 写入带拆分标记的 CSV
    write_train_val_test_split(rows, OUTPUT_SPLIT_CSV)

    # 生成 YOLOv8 标签文件
    write_yolo_labels(rows, OUTPUT_YOLO_DIR)

    # 打印统计
    print_statistics(rows)

    print(f"\n全部输出:")
    print(f"  主数据集:      {OUTPUT_CSV}")
    print(f"  拆分数据集:    {OUTPUT_SPLIT_CSV}")
    print(f"  YOLOv8 标签:   {OUTPUT_YOLO_DIR}/")
    print(f"\n  YOLOv8 dataset.yaml 示例:")
    print(f"""    path: {OUTPUT_YOLO_DIR.replace(chr(92), '/')}
    train: ../images/train
    val: ../images/val
    test: ../images/test
    nc: 8
    names: {list(BEHAVIOR_CLASSES.values())}""")
