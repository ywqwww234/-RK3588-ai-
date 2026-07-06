"""
主动守护系统 — Active Guardian System (JITAI范式)
================================================
从"等待风险超阈值→被动报警"转型为"预测轨迹→预判→分级主动干预"

核心转变:
  被动模式:  risk > 0.6 → alert ("已经出问题了才通知")
  主动模式:  velocity > threshold → preempt ("预测即将出问题, 现在就行动")

设计原理 (基于JITAI — Just-in-Time Adaptive Intervention):
  1. 风险速度 (Risk Velocity): d(risk)/dt, 量化为 0-100 "风险加速度"
  2. 轨迹预测 (Trajectory Prediction): 线性外推 + NN不确定性修正
  3. 注意力模式分析: NN的60帧时间注意力揭示"积累型恶化"vs"孤立尖峰"
  4. 分级主动响应: 4级自主决策, 每级有具体可执行的干预行为

参考文献:
  - Nahum-Shani et al. (2018) JITAI framework, Health Psychology
  - MindGuard (2025) Anticipatory Monitoring for Mental Health
  - npj Mental Health Research (2025) Personalized algorithms for sensing mental health

用法:
    from active_guardian import ActiveGuardian
    guardian = ActiveGuardian()
    # 每秒推送NN推理结果
    decision = guardian.tick(nn_result)
    # decision 包含: 当前等级、预测风险、建议干预、是否主动触发
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# 决策输出数据类
# ============================================================
@dataclass
class GuardianDecision:
    """主动守护系统每次tick的决策输出。"""
    timestamp: float = field(default_factory=time.time)

    # 当前状态
    current_risk: float = 0.0
    current_tier: int = 0

    # 预测状态
    risk_velocity: float = 0.0          # 风险变化率 (单位: /min)
    risk_acceleration: float = 0.0       # 风险加速度 (二阶导数)
    predicted_risk_5m: float = 0.0       # 5分钟后预测风险
    predicted_risk_15m: float = 0.0      # 15分钟后预测风险
    prediction_confidence: float = 0.0   # 预测置信度 (0-1)

    # 注意力分析
    attn_concentration: float = 0.0      # 注意力集中度 (高=模型锁定关键帧)
    attn_trend: str = 'stable'           # stable/rising_buildup/shifting/isolated_spike
    is_buildup_pattern: bool = False     # 是否识别到积累型恶化模式

    # 模态驱动
    dominant_modality: str = 'hrv'       # 当前主要风险驱动模态 vision/hrv/eeg
    modal_drive_strength: float = 0.0    # 主导模态的驱动强度

    # 决策
    guardian_level: int = 0              # 0=静默 1=感知 2=守护 3=预警
    guardian_level_name: str = '静默监测'
    intervention_triggered: bool = False  # 是否触发主动干预
    intervention_type: str = ''          # sensing/suggestion/escalation
    intervention_message: str = ''       # 干预描述
    escalation_ready: bool = False       # 是否准备好升级报告
    preemptive: bool = False             # 是否为预判性触发(而非事后响应)


# ============================================================
# 主动守护引擎
# ============================================================
class ActiveGuardian:
    """JITAI主动守护引擎。

    核心循环 (每秒调用):
      1. 接收 NN推理结果 (risk_score, tier, tier_probs, attn, modal_w)
      2. 更新风险轨迹缓冲区 (60秒滚动)
      3. 计算风险速度 + 加速度
      4. 线性外推预测 5min/15min 风险值
      5. 分析NN注意力模式 (积累 vs 尖峰)
      6. 分级决策: 是否需要主动干预
    """

    def __init__(self,
                 vel_window: int = 30,          # 速度计算窗口(秒)
                 pred_horizon_short: int = 300,  # 短期预测 (5分钟)
                 pred_horizon_long: int = 900,   # 长期预测 (15分钟)
                 vel_thresh_vigilant: float = 0.008,   # 感知阈值 (/min)
                 vel_thresh_active: float = 0.025,     # 守护阈值
                 vel_thresh_urgent: float = 0.050,     # 预警阈值
                 ):
        self.vel_window = vel_window
        self.pred_horizon_short = pred_horizon_short
        self.pred_horizon_long = pred_horizon_long

        # 阈值 (可调, 基于临床零容忍原则)
        self.VEL_VIGILANT = vel_thresh_vigilant   # ~0.5/hr 上升
        self.VEL_ACTIVE  = vel_thresh_active      # ~1.5/hr 上升
        self.VEL_URGENT  = vel_thresh_urgent      # ~3.0/hr 上升

        # 缓冲区
        self._risk_history = deque(maxlen=120)     # 120秒 = 2分钟滚动
        self._tier_history = deque(maxlen=60)
        self._attn_history = deque(maxlen=60)
        self._modal_w_history = deque(maxlen=60)
        self._vel_history = deque(maxlen=60)

        # 状态
        self._last_guardian_level = 0
        self._level_duration = 0.0
        self._level_start_ts = time.time()
        self._intervention_cooldown = 0.0

    # ============== 主入口 ==============
    def tick(self, nn_result: dict) -> GuardianDecision:
        """每秒调用一次, 返回主动守护决策。

        nn_result 字段:
          risk_score: float   0-1风险值
          tier: int           0/1/2/3等级
          tier_probs: list[4] 四档概率
          attn: np[60]        时间注意力权重
          modal_w: tuple(3)   模态贡献 (vision, hrv, eeg)
        """
        risk = float(nn_result.get('risk_score', 0.0))
        tier = int(nn_result.get('tier', 0))
        attn = np.asarray(nn_result.get('attn', [1/60]*60), dtype=np.float64)
        modal_w = nn_result.get('modal_w', (0.35, 0.40, 0.25))
        tier_probs = nn_result.get('tier_probs', [0.7, 0.2, 0.08, 0.02])

        # 更新缓冲区
        self._risk_history.append(risk)
        self._tier_history.append(tier)
        self._attn_history.append(attn)
        self._modal_w_history.append(modal_w)

        d = GuardianDecision(
            timestamp=time.time(),
            current_risk=risk,
            current_tier=tier,
        )

        # ---- 1. 风险速度与加速度 ----
        if len(self._risk_history) >= max(4, self.vel_window // 2):
            risks = np.array(self._risk_history)
            # 线性回归计算速度 (单位: /min)
            n = min(len(risks), self.vel_window)
            recent = risks[-n:]
            t = np.arange(n, dtype=np.float64)
            # 简单线性回归: slope = Cov(t,risk)/Var(t)
            t_mean = t.mean()
            r_mean = recent.mean()
            slope = np.sum((t - t_mean) * (recent - r_mean)) / max(1e-12, np.sum((t - t_mean)**2))
            d.risk_velocity = float(slope * 60.0)  # 转/min

            # 加速度 (速度的变化率)
            self._vel_history.append(d.risk_velocity)
            if len(self._vel_history) >= 6:
                vels = np.array(self._vel_history)
                vn = min(len(vels), 20)
                vt = np.arange(vn, dtype=np.float64)
                vm = vt.mean(); vv = vels[-vn:].mean()
                acc = np.sum((vt - vm) * (vels[-vn:] - vv)) / max(1e-12, np.sum((vt - vm)**2))
                d.risk_acceleration = float(acc * 60.0)

        # ---- 2. 轨迹预测 ----
        if abs(d.risk_velocity) > 1e-8:
            d.predicted_risk_5m = float(np.clip(
                risk + d.risk_velocity * (self.pred_horizon_short / 60.0), 0.0, 1.0))
            d.predicted_risk_15m = float(np.clip(
                risk + d.risk_velocity * (self.pred_horizon_long / 60.0), 0.0, 1.0))
        else:
            d.predicted_risk_5m = risk
            d.predicted_risk_15m = risk

        # 预测置信度: 基于(a)轨迹稳定性 (b)NN概率分布熵
        if len(self._risk_history) >= 10:
            recent_var = float(np.var(list(self._risk_history)[-10:]))
            stability = 1.0 / (1.0 + 20.0 * recent_var)  # 方差越小越稳定→置信度越高
        else:
            stability = 0.3
        # NN概率熵 (低熵=高置信)
        probs = np.clip(tier_probs, 1e-6, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(4))  # 归一化到0-1
        nn_confidence = 1.0 - entropy
        d.prediction_confidence = float(0.5 * stability + 0.5 * nn_confidence)

        # ---- 3. 注意力模式分析 ----
        d.attn_concentration = float(np.max(attn) / (np.mean(attn) + 1e-6))
        # 分析最近30秒vs前30秒的注意力分布
        if len(attn) >= 40:
            front_half = attn[:30].sum()
            back_half = attn[30:].sum()
            ratio = back_half / max(1e-6, front_half)
            if ratio > 2.0:
                d.attn_trend = 'rising_buildup'
                d.is_buildup_pattern = True
            elif ratio > 1.3:
                d.attn_trend = 'shifting'
            elif d.attn_concentration > 3.0:
                d.attn_trend = 'isolated_spike'
            else:
                d.attn_trend = 'stable'

        # ---- 4. 模态驱动分析 ----
        if isinstance(modal_w, (tuple, list)) and len(modal_w) == 3:
            mv, mp, mb = float(modal_w[0]), float(modal_w[1]), float(modal_w[2])
            max_i = np.argmax([mv, mp, mb])
            d.dominant_modality = ['vision', 'hrv', 'eeg'][max_i]
            d.modal_drive_strength = float([mv, mp, mb][max_i])

        # ---- 5. 分级决策 ----
        d.guardian_level = self._decide_level(d)
        d.guardian_level_name = {
            0: '静默监测', 1: '主动感知', 2: '主动守护', 3: '主动预警'
        }[d.guardian_level]

        # 判断是否触发主动干预
        now = time.time()
        if d.guardian_level >= 1 and now - self._intervention_cooldown > 30.0:
            d.intervention_triggered = True
            d.intervention_type = {
                1: 'sensing',       # 主动感知: 发起微探询
                2: 'suggestion',    # 主动守护: 结构化建议
                3: 'escalation',    # 主动预警: 自动升级
            }[d.guardian_level]
            d.intervention_message = self._build_message(d)
            self._intervention_cooldown = now

        d.preemptive = (d.risk_velocity > self.VEL_VIGILANT and d.current_risk < 0.5)
        d.escalation_ready = (d.guardian_level >= 2)

        # 更新等级持续时间
        if d.guardian_level != self._last_guardian_level:
            self._level_start_ts = now
            self._last_guardian_level = d.guardian_level
        d._level_duration = now - self._level_start_ts

        return d

    # ============== 分级决策引擎 ==============
    def _decide_level(self, d: GuardianDecision) -> int:
        """基于多维度证据的自主分级决策。

        决策逻辑 (优先级从高到低):
          预警 (L3): risk超高 OR 速度极快 OR 高危+积累模式
          守护 (L2): 预测5min后超阈值 OR 速度较快且持续
          感知 (L1): 速度开始上升 OR 注意力显示积累模式 OR 模态驱动强度异常
          静默 (L0): 其余
        """
        vel = d.risk_velocity
        cur = d.current_risk
        pred5 = d.predicted_risk_5m
        is_buildup = d.is_buildup_pattern

        # === L3 预警: 多证据高危 ===
        if cur >= 0.80:
            return 3
        if vel >= self.VEL_URGENT:  # >0.05/min = 3/hr
            return 3
        if cur >= 0.55 and vel >= self.VEL_ACTIVE and is_buildup:
            return 3
        if pred5 >= 0.85:
            return 3

        # === L2 守护: 预测阈值交叉 OR 持续上升 ===
        if pred5 >= 0.60:  # 5分钟后将超中风险线
            return 2
        if vel >= self.VEL_ACTIVE and cur >= 0.35:
            return 2
        if vel >= self.VEL_ACTIVE * 0.7 and is_buildup:
            return 2
        if cur >= 0.55 and d.modal_drive_strength > 0.55:
            return 2

        # === L1 感知: 早期信号 ===
        if vel >= self.VEL_VIGILANT:  # 速度开始抬头
            return 1
        if is_buildup and cur >= 0.25:
            return 1
        if d.attn_concentration > 2.5 and cur >= 0.30:
            return 1
        if d.risk_acceleration > 0.0005:  # 加速度为正
            return 1

        # === L0 静默 ===
        return 0

    # ============== 消息构建 ==============
    def _build_message(self, d: GuardianDecision) -> str:
        vel = d.risk_velocity
        acc = d.risk_acceleration
        dom = {'vision': '面部表情', 'hrv': '心率变异性', 'eeg': '脑电专注度'}[d.dominant_modality]

        if d.guardian_level == 3:
            if d.current_risk >= 0.80:
                return f'[主动预警] 当前风险已达高危水平({d.current_risk:.2f})。系统已准备家校联动报告。'
            elif vel >= self.VEL_URGENT:
                return f'[主动预警] 风险急速上升({vel*60:.1f}/hr)。{dom}指标显著恶化, 建议立即关注。'
            else:
                return f'[主动预警] 多指标积累型恶化, {dom}为主要驱动。5分钟后预测风险={d.predicted_risk_5m:.2f}。'

        elif d.guardian_level == 2:
            if d.predicted_risk_5m >= 0.60:
                return f'[主动守护] 预测5分钟后风险将达{d.predicted_risk_5m:.2f}。建议现在进行呼吸训练。{dom}为当前主要信号源。'
            elif vel >= self.VEL_ACTIVE:
                return f'[主动守护] 生理指标持续恶化(速度={vel*60:.1f}/hr)。系统建议短休+正念练习。'
            else:
                return f'[主动守护] {dom}异常已持续{d._level_duration:.0f}秒。已准备守护人通知草稿。'

        elif d.guardian_level == 1:
            if d.is_buildup_pattern:
                return f'[主动感知] 注意到{dom}指标的积累性变化。你还好吗？需要休息一下吗？'
            elif vel >= self.VEL_VIGILANT:
                return f'[主动感知] 生理信号有轻微波动。建议短暂离开屏幕, 活动2分钟。'
            else:
                return f'[主动感知] 系统检测到细微变化, 正在关注中。'

        return ''

    # ============== 查询接口 ==============
    def status(self) -> dict:
        """获取当前守护状态摘要 (供UI顶栏使用)。"""
        if not self._risk_history:
            return {'level': 0, 'name': '初始化', 'ready': False}
        d = GuardianDecision(
            current_risk=self._risk_history[-1],
            current_tier=self._tier_history[-1] if self._tier_history else 0,
            risk_velocity=self._vel_history[-1] if self._vel_history else 0.0,
        )
        d.guardian_level = self._decide_level(d)
        d.guardian_level_name = {0:'静默监测',1:'主动感知',2:'主动守护',3:'主动预警'}[d.guardian_level]
        return {
            'level': d.guardian_level,
            'name': d.guardian_level_name,
            'risk': d.current_risk,
            'velocity': d.risk_velocity,
            'ready': d.guardian_level >= 1,
        }
