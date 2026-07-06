"""
主动感知系统 — Active Sensing System
=====================================
从"悄悄说(被动等待用户说话)"转型为"主动感知(系统发起微探询)"

核心转变:
  被动模式: 用户点击"悄悄说" → 系统被动接收
  主动模式: NN检测到不确定性/趋势变化 → 系统主动发起定向微探询

设计原理 (基于Ecological Momentary Assessment + JITAI):
  1. 不确定性驱动: NN四档概率分布熵高 → 系统不确定 → 主动探询澄清
  2. 趋势驱动: 风险速度上扬 → 在峰值前发起预防性微干预
  3. 模态驱动: 哪个模态是当前主要风险源 → 定向探询该维度
  4. 自适应时机: 根据用户响应模式学习最佳干预时机

主动感知类型:
  - 不确定性探询 (Uncertainty Probe): NN不确定时, 主动问一个定向问题
  - 预防性微干预 (Preventive Nudge): 速度上升时, 在达峰前发起小动作
  - 模态定向提问 (Modal-Targeted Query): 根据主导模态问针对性问题
  - 节律性签到 (Rhythmic Check-in): 长时间低风险时偶尔确认状态

用法:
    from active_sensing import ActiveSensing
    sensing = ActiveSensing()
    # 每秒推送NN结果 + 守护者决策
    prompt = sensing.tick(nn_result, guardian_decision)
    # prompt 可能为 None (无需干预) 或包含主动探询文本
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# 主动感知输出
# ============================================================
@dataclass
class SensingPrompt:
    """一次主动感知探询。"""
    timestamp: float = field(default_factory=time.time)
    triggered: bool = False
    prompt_type: str = ''         # uncertainty / preventive / modal_targeted / rhythmic
    message: str = ''
    suggested_action: str = ''    # 建议的微行动
    urgency: int = 0              # 0=温和 1=关注 2=紧迫
    auto_dismiss_sec: int = 0     # 自动消失时间(秒), 0=不自动消失
    target_modality: str = ''     # 针对哪个模态


# ============================================================
# 主动感知引擎
# ============================================================
class ActiveSensing:
    """主动感知引擎: 系统主动发起微探询和预防性干预。

    设计理念:
      - 不是等用户来"说", 而是系统主动"问"
      - 问的问题基于NN的证据缺口, 不是随机
      - 干预时机基于风险轨迹, 不是固定阈值
    """

    def __init__(self,
                 cooldown_uncertainty: float = 45.0,   # 不确定性探询冷却(秒)
                 cooldown_preventive: float = 60.0,     # 预防性干预冷却
                 cooldown_rhythmic: float = 900.0,      # 节律签到冷却(15分钟)
                 entropy_threshold: float = 0.55,       # 概率熵阈值 (归一化后)
                 response_window: int = 6,              # 响应学习窗口大小
                 ):
        self.cooldown_uncertainty = cooldown_uncertainty
        self.cooldown_preventive = cooldown_preventive
        self.cooldown_rhythmic = cooldown_rhythmic
        self.entropy_threshold = entropy_threshold

        # 冷却计时器
        self._last_uncertainty_prompt = 0.0
        self._last_preventive_prompt = 0.0
        self._last_rhythmic_prompt = 0.0
        self._last_any_prompt = 0.0
        self._min_inter_prompt = 25.0  # 两次主动探询最小间隔

        # 响应学习
        self._response_times = deque(maxlen=response_window)    # 用户响应延迟记录
        self._response_rate = 0.6  # 初始假设60%响应率
        self._optimal_timing_hint = None  # 学习到的最佳干预时机(秒)

        # 状态追踪
        self._prev_risk = 0.0
        self._prev_entropy = 0.0
        self._silence_duration = 0.0

        # 探询模板库 (按类型组织)
        self._templates = self._build_templates()

    # ============== 主入口 ==============
    def tick(self, nn_result: dict, guardian_decision=None) -> Optional[SensingPrompt]:
        """每秒调用, 返回主动探询或None。

        nn_result: NN推理完整输出 (risk_score, tier, tier_probs, attn, modal_w)
        guardian_decision: ActiveGuardian.tick() 的输出
        """
        now = time.time()

        # 全局冷却: 任何主动探询后至少间隔 _min_inter_prompt 秒
        if now - self._last_any_prompt < self._min_inter_prompt:
            return None

        risk = float(nn_result.get('risk_score', 0.0))
        tier_probs = nn_result.get('tier_probs', [0.7, 0.2, 0.08, 0.02])
        modal_w = nn_result.get('modal_w', (0.35, 0.40, 0.25))
        attn = np.asarray(nn_result.get('attn', [1/60]*60), dtype=np.float64)

        # 更新状态
        risk_delta = risk - self._prev_risk
        self._prev_risk = risk

        # 计算NN不确定性 (概率分布熵)
        probs = np.clip(tier_probs, 1e-6, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(4))
        self._prev_entropy = entropy

        # 获取守护者等级 (如果有)
        glevel = 0
        if guardian_decision is not None:
            glevel = guardian_decision.guardian_level

        # ---- 1. 不确定性驱动探询 ----
        if entropy > self.entropy_threshold and now - self._last_uncertainty_prompt > self.cooldown_uncertainty:
            prompt = self._make_uncertainty_probe(nn_result, entropy)
            if prompt:
                self._last_uncertainty_prompt = now
                self._last_any_prompt = now
                return prompt

        # ---- 2. 预防性微干预 ----
        if glevel >= 1 and now - self._last_preventive_prompt > self.cooldown_preventive:
            prompt = self._make_preventive_nudge(nn_result, guardian_decision)
            if prompt:
                self._last_preventive_prompt = now
                self._last_any_prompt = now
                return prompt

        # ---- 3. 模态定向提问 ----
        if glevel >= 2:
            prompt = self._make_modal_query(nn_result, guardian_decision)
            if prompt and now - self._last_any_prompt > self.cooldown_uncertainty:
                self._last_any_prompt = now
                return prompt

        # ---- 4. 节律性签到 ----
        if risk < 0.25 and glevel == 0 and now - self._last_rhythmic_prompt > self.cooldown_rhythmic:
            prompt = self._make_rhythmic_checkin(nn_result)
            if prompt:
                self._last_rhythmic_prompt = now
                self._last_any_prompt = now
                return prompt

        return None

    # ============== 探询生成 ==============
    def _make_uncertainty_probe(self, nn_result: dict, entropy: float) -> SensingPrompt:
        """NN不确定时, 主动问一个问题来澄清状态。

        例: NN输出 [L1:45%, L2:40%, L3:12%, L4:3%]
        → 系统不确定是正常还是中等风险 → 主动探询
        """
        tier_probs = nn_result.get('tier_probs', [0.5, 0.3, 0.15, 0.05])
        modal_w = nn_result.get('modal_w', (0.35, 0.40, 0.25))

        # 找出最不确定的两个等级
        sorted_idx = np.argsort(tier_probs)[::-1]
        top2_probs = [tier_probs[sorted_idx[0]], tier_probs[sorted_idx[1]]]
        gap = top2_probs[0] - top2_probs[1]  # 差距越小 = 越不确定

        if gap > 0.15:  # 差距足够大, 不需要探询
            return None

        # 找出主导模态
        dom = np.argmax(modal_w)
        dom_name = ['面部表情', '心率变异性', '专注度'][dom]

        tier_names = ['正常', '轻度关注', '中度风险', '高危']
        t1 = tier_names[sorted_idx[0]]
        t2 = tier_names[sorted_idx[1]]

        messages = [
            f'我注意到你的{dom_name}信号有些模糊。现在的状态更接近"{t1}"还是"{t2}"呢？',
            f'数据有些不确定——你感觉怎么样？{dom_name}指标在两种状态间摇摆。',
            f'系统不太确定当前的评估。方便确认一下你现在的心情吗？({dom_name}有轻微波动)',
        ]

        actions = [
            '深呼吸3次, 专注当下10秒',
            '站起来伸展一下',
            '喝一杯水',
        ]

        return SensingPrompt(
            triggered=True,
            prompt_type='uncertainty',
            message=random.choice(messages),
            suggested_action=random.choice(actions),
            urgency=1,
            auto_dismiss_sec=30,
            target_modality=['vision', 'hrv', 'eeg'][dom],
        )

    def _make_preventive_nudge(self, nn_result: dict, gd) -> Optional[SensingPrompt]:
        """风险速度上升时, 在达峰前发起预防性微干预。

        关键: 这是"主动"的核心体现 —— 不是在风险已经高了才行动,
        而是在风险开始上升时就出手阻断。
        """
        if gd is None:
            return None

        vel = gd.risk_velocity
        pred5 = gd.predicted_risk_5m
        cur = gd.current_risk

        # 仅在风险尚可控时(当前<0.5)且速度显著上升时触发预防
        if cur > 0.55 or vel < 0.005:
            return None

        # 干预强度与速度成正比
        intensity = min(1.0, vel / 0.03)  # 归一化到0-1

        if intensity > 0.6:
            messages = [
                f'⚠ 检测到生理信号正在上升。预测5分钟后可能达到{pred5:.2f}。现在花1分钟做呼吸调整, 可能避免进入高风险区。',
                f'系统注意到你的压力指标在爬升。趁现在还在可控范围, 试试4-7-8呼吸法: 吸气4秒, 屏息7秒, 呼气8秒。',
                f'我预判你的状态可能在未来几分钟内恶化。主动休息5分钟比被动应对危机更有效——要试试吗？',
            ]
            urgency = 2
            auto_dismiss = 0  # 不自动消失
        else:
            messages = [
                f'轻微的生理波动被检测到。也许可以短暂离开屏幕, 望向窗外2分钟。',
                f'身体可能在提醒你需要休息了。试试闭上眼睛, 听30秒自己的呼吸声。',
                f'一个温和的提醒: 数据有小幅上扬。站起来活动一下就好。',
            ]
            urgency = 1
            auto_dismiss = 25

        return SensingPrompt(
            triggered=True,
            prompt_type='preventive',
            message=random.choice(messages),
            suggested_action='4-7-8呼吸法: 吸4秒→屏7秒→呼8秒, 重复3轮',
            urgency=urgency,
            auto_dismiss_sec=auto_dismiss,
            target_modality=gd.dominant_modality,
        )

    def _make_modal_query(self, nn_result: dict, gd) -> Optional[SensingPrompt]:
        """风险较高时, 根据主导模态问针对性问题。"""
        if gd is None:
            return None
        dom = gd.dominant_modality
        strength = gd.modal_drive_strength

        queries = {
            'hrv': [
                f'心率变异性是最强的风险信号(强度{strength:.0%})。最近睡眠质量怎么样？',
                f'你的自主神经似乎处于应激状态。有没有什么特别担心的deadline或考试？',
            ],
            'vision': [
                f'面部表情分析显示疲劳迹象明显(强度{strength:.0%})。眼睛需要休息了。',
                f'检测到你的表情偏向负面情绪。需要跟朋友或家人聊聊吗？',
            ],
            'eeg': [
                f'脑电专注度指标异常(强度{strength:.0%})。可能是注意力过度消耗。建议暂停当前任务5分钟。',
                f'你似乎很难放松下来。试试渐进式肌肉放松:从脚趾开始, 逐步紧绷再放松每个肌群。',
            ],
        }

        msgs = queries.get(dom, queries['hrv'])
        return SensingPrompt(
            triggered=True,
            prompt_type='modal_targeted',
            message=random.choice(msgs),
            suggested_action='接受系统建议, 进行5分钟定向恢复',
            urgency=2,
            auto_dismiss_sec=0,
            target_modality=dom,
        )

    def _make_rhythmic_checkin(self, nn_result: dict) -> SensingPrompt:
        """长时间低风险时, 偶尔主动签到确认状态良好。

        这听起来像"被动", 但实际上是"主动"——系统主动发起,
        防止"假阴性"(系统以为一切正常, 但其实用户已经不好了)。
        """
        risk = nn_result.get('risk_score', 0.0)
        messages = [
            f'一切看起来都很平稳(风险{risk:.2f})。只是确认一下——你确实感觉还好吗？',
            f'系统检测到你的状态不错。继续保持！有需要的话我随时在这里。',
            f'好久没打招呼了。你的生理指标都很稳定, 只是想确认: 心情也OK吗？',
        ]
        return SensingPrompt(
            triggered=True,
            prompt_type='rhythmic',
            message=random.choice(messages),
            suggested_action='',
            urgency=0,
            auto_dismiss_sec=20,
            target_modality='',
        )

    # ============== 模板库 ==============
    def _build_templates(self) -> dict:
        return {
            'uncertainty': {
                'vision': [
                    '你的面部数据有些模糊。光线是否充足？',
                    '摄像头似乎捕捉到混合表情。你现在的心情是？',
                ],
                'hrv': [
                    '心率变异性数据在两个评估区间交界处。感觉紧张还是平静？',
                ],
                'eeg': [
                    '脑电信号信噪比偏低。TGAM电极接触是否良好？',
                ],
            },
            'preventive': {
                'rising': [],
                'buildup': [],
            },
        }

    # ============== 自适应学习 ==============
    def record_response(self, responded: bool, delay_sec: float = 0.0):
        """记录用户对主动探询的响应。

        系统学习:
          - 用户通常在什么时间响应
          - 哪种类型的探询响应率最高
          - 最佳干预间隔
        """
        self._response_times.append(delay_sec if responded else -1.0)
        responded_count = sum(1 for t in self._response_times if t >= 0)
        self._response_rate = responded_count / max(1, len(self._response_times))

        # 学习最佳时机: 取用户响应的中位延迟
        valid = [t for t in self._response_times if t > 0]
        if len(valid) >= 3:
            self._optimal_timing_hint = float(np.median(valid))

    # ============== 状态查询 ==============
    def status(self) -> dict:
        return {
            'response_rate': self._response_rate,
            'optimal_timing_hint': self._optimal_timing_hint,
            'last_any_prompt': self._last_any_prompt,
            'silence_duration': self._silence_duration,
        }
