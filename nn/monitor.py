"""
推理监控与数据漂移检测（生产可观测性）。

提供：
  1. RollingStats     —— 用 Welford 在线算法记录每维特征的均值/方差
  2. DriftDetector    —— 把当前窗口与训练分布（scaler.json）比较，输出 z-score
  3. InferenceMonitor —— 包装 InferenceEngine，记录 fallback/异常/漂移并输出状态

使用：
    from nn.inference_engine import InferenceEngine
    from nn.monitor import InferenceMonitor
    eng = InferenceEngine()
    mon = InferenceMonitor(eng)
    out = mon.predict(window)
    health = mon.health()       # → 给 UI 顶栏 pill 用
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np


class RollingStats:
    """在线均值/方差（Welford）。"""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        # x: [D] 或 [N, D]
        x = np.atleast_2d(x).astype(np.float64)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-12))


class DriftDetector:
    """把当前数据与训练分布（scaler）比较，给出每维 |z| 与异常率。"""

    def __init__(self, scaler_path: str = 'nn/models/scaler.json',
                 z_thresh: float = 3.0):
        self.scaler = None
        self.z_thresh = z_thresh
        if os.path.isfile(scaler_path):
            try:
                with open(scaler_path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                self.scaler = {
                    'mean': np.asarray(d['mean'], dtype=np.float64),
                    'std':  np.asarray(d['std'],  dtype=np.float64),
                }
            except Exception:
                self.scaler = None

    def score(self, window: np.ndarray) -> dict:
        """window: [60, 25] -> {'avg_abs_z': float, 'pct_outlier': float, 'top_dim': int}"""
        if self.scaler is None:
            return {'avg_abs_z': 0.0, 'pct_outlier': 0.0, 'top_dim': -1,
                    'has_baseline': False}
        x = np.asarray(window, dtype=np.float64)
        z = (x - self.scaler['mean']) / np.maximum(self.scaler['std'], 1e-6)
        absz = np.abs(z)
        avg_z = float(absz.mean())
        pct = float((absz > self.z_thresh).mean())
        top = int(absz.mean(axis=0).argmax())
        return {'avg_abs_z': avg_z, 'pct_outlier': pct, 'top_dim': top,
                'has_baseline': True}


class InferenceMonitor:
    """推理监控装饰器：记录最近 N 次推理的延迟、fallback、漂移。"""

    def __init__(self, engine, window: int = 600,
                 scaler_path: str = 'nn/models/scaler.json'):
        self.engine = engine
        self.lat = deque(maxlen=window)
        self.tiers = deque(maxlen=window)
        self.fallback_flags = deque(maxlen=window)
        self.errors = deque(maxlen=20)
        self.drift = DriftDetector(scaler_path=scaler_path)
        self.t_start = time.time()

    def predict(self, window) -> dict:
        out = self.engine.predict(window)
        self.lat.append(float(out.get('latency_ms', 0.0)))
        self.tiers.append(int(out.get('tier', 0)))
        self.fallback_flags.append(out.get('mode') == 'fallback')
        if out.get('warning'):
            self.errors.append({'ts': time.time(), 'msg': out['warning']})
        # 把漂移信号写进结果，便于 UI 显示
        d = self.drift.score(window)
        out['drift'] = d
        return out

    def health(self) -> dict:
        n = len(self.lat)
        if n == 0:
            return {'state': 'idle', 'reason': '尚无推理记录',
                    'mode': 'onnx' if self.engine.loaded else 'fallback'}
        fb_ratio = sum(self.fallback_flags) / n
        p95 = float(np.percentile(self.lat, 95))
        # 状态判定
        if fb_ratio > 0.8:
            state = 'offline'
            reason = f'fallback 占比 {fb_ratio*100:.0f}%（疑似 ONNX/scaler 缺失）'
        elif fb_ratio > 0.1:
            state = 'warn'
            reason = f'fallback 占比 {fb_ratio*100:.0f}%（间歇降级）'
        elif p95 > 200:
            state = 'warn'
            reason = f'p95 延迟 {p95:.0f} ms（性能告警）'
        else:
            state = 'live'
            reason = '运行正常'
        # 等级分布
        tier_dist = np.bincount(list(self.tiers), minlength=4).tolist()
        return {
            'state': state,
            'reason': reason,
            'mode': 'onnx' if self.engine.loaded else 'fallback',
            'providers': list(getattr(self.engine, 'providers_used', [])),
            'n': n,
            'fallback_ratio': fb_ratio,
            'lat_p50_ms': float(np.median(self.lat)),
            'lat_p95_ms': p95,
            'tier_dist': tier_dist,
            'uptime_s': int(time.time() - self.t_start),
            'last_errors': list(self.errors)[-5:],
            'engine_stats': self.engine.stats() if hasattr(self.engine, 'stats') else {},
        }
