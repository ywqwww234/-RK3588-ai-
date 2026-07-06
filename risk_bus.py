"""
RiskBus · 神经网络推理结果中央分发总线（D:/G/2）。

设计:
  - 单例：全工程共享同一个总线实例（RiskBus.instance()）
  - 上游：ui_main._nn_tick 每秒一次 push(nn_out)
  - 下游：三大模态卡 / 风险时序卡 / 神经网络洞察卡 / 工具箱 7 个对话框
  - 历史：维护最近 600s 滚动窗口（默认 deque maxlen=600）
  - 决策：tier_window(N) + decide_alert_tier(N) 给求助转介用（论文 3.3.5 节
          连续 N 次窗口投票，避免单次误报）

NNResult dict 字段:
  risk_score   float    0~1 综合风险（来自模型回归头 reg）
  tier         int      0/1/2/3 → L1/L2/L3/L4
  tier_probs   list[4]  四类概率（softmax 后）
  attn         np[60]   时间维注意力权重
  modal_w      tuple    (vision, hrv, eeg) 模态贡献
  latency_ms   float    本次推理耗时
  mode         str      'onnx' | 'fallback'
  ts           float    时间戳

不替代 risk_calculator.py：后者保留为 InferenceEngine 不可用时的备份。
"""
import time
from collections import deque

from PyQt5.QtCore import QObject, pyqtSignal


_HIST_LEN = 600  # 10 分钟 @ 1Hz


class RiskBus(QObject):
    """全工程唯一的 NN 推理结果分发总线。"""

    nn_result_changed = pyqtSignal(dict)       # 每次 push 触发，完整 NNResult
    tier_changed = pyqtSignal(int, list)       # tier 跨档变化：(new_tier, window)
    alert_triggered = pyqtSignal(dict)         # 投票判定升级到 ≥L3 时触发

    _instance = None

    def __init__(self):
        super().__init__()
        self._history = deque(maxlen=_HIST_LEN)
        self._last_tier = -1
        self._last_alert_tier = -1
        self._modal_factors = {
            'vision': '—',
            'hrv': '—',
            'eeg': '—',
        }

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = RiskBus()
        return cls._instance

    # ---------- 上游 ----------
    def push(self, nn_out):
        """接收 InferenceEngine.predict 输出，补全字段后入队并广播。"""
        if not isinstance(nn_out, dict):
            return
        nn_out = dict(nn_out)
        nn_out.setdefault('ts', time.time())
        if 'tier_probs' not in nn_out:
            nn_out['tier_probs'] = _synthetic_probs(nn_out.get('risk_score', 0.0))
        nn_out.setdefault('mode', 'fallback')
        nn_out.setdefault('modal_w', (0.35, 0.40, 0.25))
        nn_out.setdefault('attn', [1.0 / 60] * 60)
        nn_out.setdefault('latency_ms', 0.0)

        # 等级 = argmax(tier_probs)；如果引擎已经给了 tier，直接用
        if 'tier' not in nn_out:
            probs = nn_out['tier_probs']
            nn_out['tier'] = int(max(range(4), key=lambda i: probs[i]))

        self._history.append(nn_out)
        cur_tier = int(nn_out['tier'])

        self.nn_result_changed.emit(nn_out)

        if cur_tier != self._last_tier:
            self._last_tier = cur_tier
            self.tier_changed.emit(cur_tier, self.tier_window(3))

        alert_tier = self.decide_alert_tier(3)
        if alert_tier >= 2 and alert_tier != self._last_alert_tier:
            self._last_alert_tier = alert_tier
            self.alert_triggered.emit({
                'alert_tier': alert_tier,
                'tier_window': self.tier_window(3),
                'nn_result': nn_out,
            })
        elif alert_tier < 2:
            self._last_alert_tier = alert_tier

    # 让模块卡可以把"主因短句"反向写回，供报告/工具箱复用
    def set_modal_factor(self, modality, text):
        if modality in self._modal_factors:
            self._modal_factors[modality] = text

    def get_modal_factor(self, modality):
        return self._modal_factors.get(modality, '—')

    # ---------- 下游访问 ----------
    def latest(self):
        if not self._history:
            return None
        return self._history[-1]

    def history(self, n=_HIST_LEN):
        if n is None or n >= len(self._history):
            return list(self._history)
        return list(self._history)[-n:]

    def tier_window(self, n=3):
        if not self._history:
            return []
        return [int(h.get('tier', 0)) for h in list(self._history)[-n:]]

    def decide_alert_tier(self, n=3):
        """投票（论文 3.3.5）：count(L4)≥1→L4 / count(L3)≥2→L3 / count(L2)≥2→L2 / 否则 L1。"""
        window = self.tier_window(n)
        if not window:
            return 0
        c_l4 = sum(1 for t in window if t >= 3)
        c_l3 = sum(1 for t in window if t >= 2)
        c_l2 = sum(1 for t in window if t >= 1)
        if c_l4 >= 1:
            return 3
        if c_l3 >= 2:
            return 2
        if c_l2 >= 2:
            return 1
        return 0


_PROB_TEMPLATES = [
    (0.0,  [0.78, 0.17, 0.04, 0.01]),
    (0.30, [0.45, 0.40, 0.12, 0.03]),
    (0.50, [0.20, 0.50, 0.25, 0.05]),
    (0.65, [0.08, 0.32, 0.48, 0.12]),
    (0.80, [0.04, 0.18, 0.43, 0.35]),
    (0.95, [0.02, 0.08, 0.30, 0.60]),
]


def _synthetic_probs(risk):
    """从 risk_score 合成 4 类概率，fallback 模式也能有概率分布展示。"""
    r = max(0.0, min(1.0, float(risk or 0.0)))
    for i in range(len(_PROB_TEMPLATES) - 1):
        r0, p0 = _PROB_TEMPLATES[i]
        r1, p1 = _PROB_TEMPLATES[i + 1]
        if r0 <= r <= r1:
            t = (r - r0) / max(1e-6, r1 - r0)
            return [p0[j] * (1 - t) + p1[j] * t for j in range(4)]
    return list(_PROB_TEMPLATES[-1][1])
