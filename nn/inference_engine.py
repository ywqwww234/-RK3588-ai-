"""
推理引擎 v2：包装 ONNX Runtime，输出风险四档概率 + 趋势 + 注意力权重 + 模态贡献。

设计原则：
  - 与训练共用 scaler：默认加载 nn/models/scaler.json，保证训练/推理同分布
  - 模式可控：
      * strict=True   → 仅当 ONNX 可用才允许推理，否则抛错（生产模式）
      * strict=False  → ONNX 失败时降级到 numpy fallback，并在输出中显式标注 mode/warning
  - Provider 可配置：preferred_providers 显式列出（如 ['CUDAExecutionProvider','CPUExecutionProvider']）
  - 监控：last_error / last_load_error / metrics（推理次数、延迟、fallback 次数）
  - 模态贡献 modal_w 优先采用模型门控输出（v2 dl_model），fallback 才用先验
  - 解释口径统一：返回字典包含 attn (60) / modal_w (3) / tier_probs (4) / feat_imp (25)

UI 接入（向后兼容）：
    eng = InferenceEngine()                   # 宽松模式（默认）
    eng = InferenceEngine(strict=True)        # 生产模式：拒绝降级
    out = eng.predict(window_60x25)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import deque
from typing import List, Optional, Sequence

import numpy as np

ONNX_OK = True
ORT_IMPORT_ERR = None
try:
    import onnxruntime as ort
except Exception as e:
    ONNX_OK = False
    ORT_IMPORT_ERR = f"{type(e).__name__}: {e}"


VISION_DIM = 9
PHYSIO_DIM = 5
BRAIN_DIM = 11
INPUT_SEQ = 60
INPUT_DIM = VISION_DIM + PHYSIO_DIM + BRAIN_DIM  # 25


_DEFAULT_PROVIDERS_PRIORITY = [
    'CUDAExecutionProvider',
    'CoreMLExecutionProvider',     # macOS
    'DmlExecutionProvider',        # Windows DirectML / 部分 NPU
    'OpenVINOExecutionProvider',
    'CPUExecutionProvider',
]


def _load_scaler(path: str):
    """加载 scaler.json，失败返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return {
            'mean': np.asarray(d['mean'], dtype=np.float32),
            'std':  np.asarray(d['std'], dtype=np.float32),
        }
    except Exception:
        return None


class InferenceEngine:
    """ONNX Runtime 推理引擎（v2）。"""

    def __init__(self,
                 onnx_path: str = 'nn/models/baseline.onnx',
                 scaler_path: str = 'nn/models/scaler.json',
                 meta_path: str = 'nn/models/meta.json',
                 preferred_providers: Optional[Sequence[str]] = None,
                 strict: bool = False):
        self.path = onnx_path
        self.scaler_path = scaler_path
        self.meta_path = meta_path
        self.strict = bool(strict)
        self.sess = None
        self.in_name = 'x'
        self.out_names: List[str] = ['cls', 'reg', 'attn']  # 兼容 v1
        self.providers_used: List[str] = []
        self.last_error: Optional[str] = None
        self.last_load_error: Optional[str] = None
        self.meta: dict = {}

        # 推理监控指标
        self._n_calls = 0
        self._n_fallback = 0
        self._lat_window = deque(maxlen=120)  # 最近 120 次延迟（≈2 min @1Hz）
        self._fallback_warned_once = False     # 避免每帧重复刷屏警告

        self.scaler = _load_scaler(scaler_path)

        # 加载 meta（可选）
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = {}

        self._load_session(preferred_providers)

        if self.strict and self.sess is None:
            raise RuntimeError(
                f'[InferenceEngine strict] ONNX 不可用：{self.last_load_error}')

    # ---------------- 会话管理 ----------------

    def _load_session(self, preferred: Optional[Sequence[str]]):
        if not ONNX_OK:
            self.last_load_error = f'onnxruntime 导入失败: {ORT_IMPORT_ERR or "unknown"}'
            return
        if not os.path.isfile(self.path):
            self.last_load_error = f'ONNX 文件不存在: {self.path}'
            return
        available = set(ort.get_available_providers())
        if preferred:
            providers = [p for p in preferred if p in available]
        else:
            providers = [p for p in _DEFAULT_PROVIDERS_PRIORITY if p in available]
        if not providers:
            providers = ['CPUExecutionProvider']
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(self.path, sess_options=so,
                                             providers=providers)
            self.providers_used = list(self.sess.get_providers())
            self.in_name = self.sess.get_inputs()[0].name
            self.out_names = [o.name for o in self.sess.get_outputs()]
            self.last_load_error = None
        except Exception as e:
            self.sess = None
            self.last_load_error = f'{type(e).__name__}: {e}'

    def reload(self, onnx_path: Optional[str] = None,
               preferred_providers: Optional[Sequence[str]] = None):
        """热加载新模型（训练完成后可在线切换）。"""
        if onnx_path:
            self.path = onnx_path
        self.sess = None
        self.scaler = _load_scaler(self.scaler_path)
        self._load_session(preferred_providers)
        if self.strict and self.sess is None:
            raise RuntimeError(
                f'[InferenceEngine strict] reload 失败: {self.last_load_error}')
        return self.loaded

    # ---------------- 属性 / 监控 ----------------

    @property
    def loaded(self) -> bool:
        return self.sess is not None

    def stats(self) -> dict:
        lats = list(self._lat_window)
        return {
            'mode': 'onnx' if self.loaded else 'fallback',
            'providers': list(self.providers_used),
            'n_calls': self._n_calls,
            'n_fallback': self._n_fallback,
            'fallback_ratio': (self._n_fallback / self._n_calls) if self._n_calls else 0.0,
            'lat_p50_ms': float(np.median(lats)) if lats else 0.0,
            'lat_p95_ms': float(np.percentile(lats, 95)) if lats else 0.0,
            'last_error': self.last_error,
            'last_load_error': self.last_load_error,
            'has_scaler': self.scaler is not None,
            'meta': self.meta,
        }

    # ---------------- 预处理 ----------------

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """x: [1, T, D]。训练/推理用同一 scaler。"""
        if self.scaler is None:
            return x
        return ((x - self.scaler['mean']) / self.scaler['std']).astype(np.float32)

    # ---------------- 推理 ----------------

    def predict(self, window) -> dict:
        """window: [60, 25] -> 结果 dict。"""
        t0 = time.time()
        self._n_calls += 1
        try:
            x_raw = np.asarray(window, dtype=np.float32)[None, ...]
            if x_raw.shape != (1, INPUT_SEQ, INPUT_DIM):
                raise ValueError(
                    f'window shape 必须为 [{INPUT_SEQ}, {INPUT_DIM}]，'
                    f'实得 {x_raw.shape[1:]}')
        except Exception as e:
            self.last_error = f'input: {e}'
            if self.strict:
                raise
            return self._fallback_dict(window, t0, warn=str(e))

        x = self._scale(x_raw)

        if self.sess is not None:
            try:
                outs = self.sess.run(self.out_names, {self.in_name: x})
                out_map = dict(zip(self.out_names, outs))
                logits = out_map.get('cls')
                reg    = out_map.get('reg')
                attn   = out_map.get('attn')
                modal_w_model = out_map.get('modal_w')   # v2 模型才有

                # softmax 概率
                z = logits[0]
                z = z - z.max()
                ez = np.exp(z)
                tier_probs = (ez / ez.sum()).astype(np.float32)
                tier = int(np.argmax(tier_probs))
                risk = float(reg[0])
                attn = np.asarray(attn[0], dtype=np.float32)

                if modal_w_model is not None:
                    mw = np.asarray(modal_w_model[0], dtype=np.float32)
                    modal_w = (float(mw[0]), float(mw[1]), float(mw[2]))
                else:
                    modal_w = self._modal_from_prior(x_raw[0])

                lat = (time.time() - t0) * 1000.0
                self._lat_window.append(lat)
                self.last_error = None
                return {
                    'risk_score': risk,
                    'tier': tier,
                    'tier_probs': tier_probs.tolist(),
                    'attn': attn,
                    'modal_w': modal_w,
                    'feat_imp': self._feat_imp(x_raw[0], modal_w),
                    'latency_ms': lat,
                    'mode': 'onnx',
                    'providers': list(self.providers_used),
                    'scaled': self.scaler is not None,
                    'warning': None,
                }
            except Exception as e:
                self.last_error = f'onnx run: {type(e).__name__}: {e}'
                if self.strict:
                    raise

        # 没有 session → fallback
        return self._fallback_dict(window, t0,
                                    warn=self.last_error or self.last_load_error)

    # ---------------- Fallback ----------------

    def _fallback_dict(self, window, t0: float,
                       warn: Optional[str] = None) -> dict:
        self._n_fallback += 1
        msg = (
            'inference fallback：未使用训练模型，仅供演示。'
            '请检查 ONNX 文件 / onnxruntime / scaler。'
        )
        if warn:
            msg += f' [details] {warn}'
        if not self._fallback_warned_once:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            self._fallback_warned_once = True
        x_arr = np.asarray(window, dtype=np.float32)
        # 保持 fallback 输入语义稳定（不缩放，避免无 scaler 时偏差）
        risk, tier, attn = self._fallback(x_arr)
        modal_w = self._modal_from_prior(x_arr)
        tier_probs = self._synthetic_probs(risk)
        lat = (time.time() - t0) * 1000.0
        self._lat_window.append(lat)
        return {
            'risk_score': float(risk),
            'tier': int(tier),
            'tier_probs': tier_probs,
            'attn': attn,
            'modal_w': modal_w,
            'feat_imp': self._feat_imp(x_arr, modal_w),
            'latency_ms': lat,
            'mode': 'fallback',
            'providers': [],
            'scaled': False,
            'warning': msg,
        }

    def _fallback(self, win: np.ndarray):
        last = win[-1]
        v_score = float(np.tanh(last[:VISION_DIM].sum()))
        p_score = float(np.tanh(last[VISION_DIM:VISION_DIM + PHYSIO_DIM].sum()))
        b_score = float(np.tanh(last[VISION_DIM + PHYSIO_DIM:].sum()))
        risk = float(np.clip(0.35 * v_score + 0.40 * p_score + 0.25 * b_score, 0, 1))
        if risk < 0.3:
            tier = 0
        elif risk < 0.6:
            tier = 1
        elif risk < 0.8:
            tier = 2
        else:
            tier = 3
        attn = np.full(INPUT_SEQ, 1.0 / INPUT_SEQ, dtype=np.float32)
        return risk, tier, attn

    def _modal_from_prior(self, win: np.ndarray):
        """fallback 模态贡献：论文设计权重 0.35/0.40/0.25 + 当前帧幅度调节。"""
        last = np.abs(win[-1])
        v_mag = float(last[:VISION_DIM].sum())
        p_mag = float(last[VISION_DIM:VISION_DIM + PHYSIO_DIM].sum())
        b_mag = float(last[VISION_DIM + PHYSIO_DIM:].sum())
        v = 0.35 * (1 + 0.3 * np.tanh(v_mag - 1))
        p = 0.40 * (1 + 0.3 * np.tanh(p_mag - 1))
        b = 0.25 * (1 + 0.3 * np.tanh(b_mag - 1))
        s = v + p + b
        return (float(v / s), float(p / s), float(b / s))

    def _feat_imp(self, win: np.ndarray, modal_w) -> list:
        """简单的 25 维特征显著度（最后一帧 |值| × 模态权重）。"""
        last = np.abs(win[-1])
        v, p, b = modal_w
        out = np.zeros_like(last, dtype=np.float32)
        out[:VISION_DIM] = last[:VISION_DIM] * v
        out[VISION_DIM:VISION_DIM + PHYSIO_DIM] = last[VISION_DIM:VISION_DIM + PHYSIO_DIM] * p
        out[VISION_DIM + PHYSIO_DIM:] = last[VISION_DIM + PHYSIO_DIM:] * b
        s = out.sum()
        if s > 1e-6:
            out = out / s
        return out.tolist()

    @staticmethod
    def _synthetic_probs(risk: float):
        """从 risk_score 合成 4 类概率（与 risk_bus 一致）。"""
        templates = [
            (0.0,  [0.78, 0.17, 0.04, 0.01]),
            (0.30, [0.45, 0.40, 0.12, 0.03]),
            (0.50, [0.20, 0.50, 0.25, 0.05]),
            (0.65, [0.08, 0.32, 0.48, 0.12]),
            (0.80, [0.04, 0.18, 0.43, 0.35]),
            (0.95, [0.02, 0.08, 0.30, 0.60]),
        ]
        r = max(0.0, min(1.0, float(risk or 0.0)))
        for i in range(len(templates) - 1):
            r0, p0 = templates[i]
            r1, p1 = templates[i + 1]
            if r0 <= r <= r1:
                t = (r - r0) / max(1e-6, r1 - r0)
                return [p0[j] * (1 - t) + p1[j] * t for j in range(4)]
        return list(templates[-1][1])
