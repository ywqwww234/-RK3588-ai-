"""
PPG 信号处理（与 PPG_Monitor.py / 1.py 真机链对齐）
- 带通 filtfilt + 轻平滑
- 显示层幅度归一化（不影响峰值检测）
- RR → SDNN / RMSSD / LF-HF
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 100.0
TAIL_N = 400
BUF_LEN = 800
VALID_COUNT_MIN = 120


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2
) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, np.asarray(data, dtype=np.float64))


def smooth_signal(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    window = np.ones(window_size) / window_size
    return np.convolve(np.asarray(data, dtype=np.float64), window, mode="same")


def calculate_hrv_indices(rr_intervals: Sequence[float]) -> Tuple[float, float, float]:
    """RR 间期（秒）→ SDNN/RMSSD/LF-HF（ms 或比值），与 1.py 一致。"""
    rr = np.asarray(rr_intervals, dtype=np.float64)
    rr = rr[np.isfinite(rr)]
    rr = rr[(rr >= 60.0 / 130.0) & (rr <= 60.0 / 45.0)]
    if len(rr) >= 5:
        med = float(np.median(rr))
        rr = rr[np.abs(rr - med) <= 0.20 * med]
    if len(rr) < 5:
        return 0.0, 0.0, 0.0

    sdnn = float(np.std(rr, ddof=1) * 1000.0)
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000.0)
    lf_hf = float(np.clip(1.5 + (sdnn - rmssd) / 120.0, 0.2, 5.0))
    sdnn = float(np.clip(sdnn, 5.0, 200.0))
    rmssd = float(np.clip(rmssd, 5.0, 200.0))
    return sdnn, rmssd, lf_hf


def _clean_ppg_values(ppg_values: Sequence[Any]) -> np.ndarray:
    out: List[float] = []
    if ppg_values is None:
        return np.asarray(out, dtype=np.float64)
    for v in ppg_values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                out.append(fv)
        except Exception:
            pass
    return np.asarray(out, dtype=np.float64)


def filter_ppg_tail(
    ppg: np.ndarray,
    fs: float = FS,
    tail_n: int = TAIL_N,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对缓冲尾部做带通+平滑。返回 (filtered_full_len, filtered_tail)。
    filtered_full 前段为 0，仅 [-tail_n:] 有效。
    """
    ppg = np.asarray(ppg, dtype=np.float64)
    valid_count = int(np.count_nonzero(ppg))
    if valid_count <= VALID_COUNT_MIN:
        return np.zeros_like(ppg), np.array([], dtype=np.float64)

    tail_n = min(tail_n, valid_count)
    ppg_tail = ppg[-tail_n:]
    filtered_tail = butter_bandpass_filter(ppg_tail, 0.5, 6.0, fs, order=2)
    filtered_tail = smooth_signal(filtered_tail, window_size=5)

    filtered = np.zeros_like(ppg)
    filtered[-tail_n:] = filtered_tail
    return filtered, filtered_tail


def scale_ppg_for_display(
    filtered: np.ndarray,
    filtered_tail: np.ndarray,
    target_amp: float = 1800.0,
) -> np.ndarray:
    """仅显示：按近期 P95 幅度缩放，过平则清零。"""
    if filtered_tail is None or len(filtered_tail) < 10:
        return filtered
    recent = filtered_tail[-min(200, len(filtered_tail)) :]
    recent_amp = float(np.percentile(np.abs(recent), 95))
    if recent_amp > 1e-6:
        out = filtered * (target_amp / recent_amp)
    else:
        out = filtered.copy()
    tail = out[-80:] if len(out) >= 80 else out
    if len(tail) > 0 and (float(np.max(tail)) - float(np.min(tail))) < 20:
        return np.zeros_like(out)
    return out


def find_ppg_peaks(filtered: np.ndarray, fs: float = FS) -> np.ndarray:
    """与 1.py 一致；仅在有效滤波段寻峰，避免前段零填充拉低阈值。"""
    arr = np.asarray(filtered, dtype=np.float64)
    if len(arr) < 50:
        return np.array([], dtype=int)
    nz = np.flatnonzero(np.abs(arr) > 1e-9)
    if len(nz) < 50:
        return np.array([], dtype=int)
    i0 = int(nz[0])
    i1 = int(nz[-1]) + 1
    seg = arr[i0:i1]
    if len(seg) < 50:
        return np.array([], dtype=int)
    height = float(np.mean(seg) + np.std(seg) * 0.4)
    peaks, _ = find_peaks(
        seg,
        height=height,
        distance=int(fs * 0.25),
    )
    return peaks + i0


def extract_rr_intervals_from_ppg(
    ppg_values: Sequence[Any], fs: float = FS
) -> List[float]:
    """从 IR 缓冲提取本帧有效 RR（秒），供滚动缓冲累计（对齐 1.py ≥5 才出 SDNN）。"""
    vals = _clean_ppg_values(ppg_values)
    if len(vals) < VALID_COUNT_MIN:
        return []
    buf = np.zeros(BUF_LEN, dtype=np.float64)
    n = min(len(vals), BUF_LEN)
    buf[-n:] = vals[-n:]
    filtered, _ = filter_ppg_tail(buf, fs=fs)
    peaks = find_ppg_peaks(filtered, fs=fs)
    if len(peaks) <= 1:
        return []
    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr > 0.3) & (rr < 1.5)]
    return [float(x) for x in rr if np.isfinite(x)]


def metrics_from_rr_buffer(
    rr_buffer: Sequence[float],
) -> Optional[Dict[str, float]]:
    """与 1.py：滚动 RR ≥5 时计算 SDNN/RMSSD/LF-HF。"""
    rr = [float(x) for x in rr_buffer if x is not None and math.isfinite(float(x))]
    if len(rr) < 5:
        return None
    rr = rr[-8:]
    hr = float(np.clip(60.0 / np.mean(rr), 50.0, 100.0))
    sdnn, rmssd, lf_hf = calculate_hrv_indices(rr)
    if sdnn <= 0 or rmssd <= 0:
        return {"bpm": hr, "sdnn": 0.0, "rmssd": 0.0, "lfhf": 0.0}
    return {"bpm": hr, "sdnn": sdnn, "rmssd": rmssd, "lfhf": lf_hf}


def metrics_from_filtered(
    filtered: np.ndarray, fs: float = FS
) -> Optional[Dict[str, float]]:
    peaks = find_ppg_peaks(filtered, fs=fs)
    if len(peaks) <= 1:
        return None
    rr_intervals = np.diff(peaks) / fs
    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
    if len(rr_intervals) == 0:
        return None

    hr = float(np.clip(60.0 / np.mean(rr_intervals), 50.0, 100.0))
    rr_list = rr_intervals.tolist()
    sdnn, rmssd, lf_hf = calculate_hrv_indices(rr_list)
    if sdnn <= 0 or rmssd <= 0:
        return {"bpm": hr, "sdnn": 0.0, "rmssd": 0.0, "lfhf": 0.0}

    return {
        "bpm": hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "lfhf": lf_hf,
    }


def compute_ppg_metrics(
    ppg_values: Sequence[Any], fs: float = FS
) -> Optional[Dict[str, float]]:
    """学生端/远端：从 IR 序列估算 HR 与 HRV（与 1.py 峰值链一致）。"""
    vals = _clean_ppg_values(ppg_values)
    if len(vals) < VALID_COUNT_MIN:
        return None
    buf = np.zeros(BUF_LEN, dtype=np.float64)
    n = min(len(vals), BUF_LEN)
    buf[-n:] = vals[-n:]
    filtered, _ = filter_ppg_tail(buf, fs=fs)
    return metrics_from_filtered(filtered, fs=fs)


def compute_ppg_display(
    ppg_values: Sequence[Any],
    fs: float = FS,
    buf_len: int = BUF_LEN,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    返回 (display_curve, metrics)。
    display_curve 长度 buf_len，供 pyqtgraph 直接 setData。
    """
    vals = _clean_ppg_values(ppg_values)
    buf = np.zeros(buf_len, dtype=np.float64)
    if len(vals) == 0:
        return buf, None
    n = min(len(vals), buf_len)
    buf[-n:] = vals[-n:]

    filtered, tail = filter_ppg_tail(buf, fs=fs)
    display = scale_ppg_for_display(filtered, tail)
    metrics = metrics_from_filtered(filtered, fs=fs) if tail.size else None
    return display, metrics


def sim_ppg_sample(counter: int, fs: float = FS) -> float:
    """与 PPG_Monitor 模拟 IR 一致。"""
    t = counter / fs
    heart_rate = 1.0
    respiration = 0.2
    ir_val = (
        10000
        + 2000 * np.sin(2 * np.pi * heart_rate * t)
        + 500 * np.sin(2 * np.pi * respiration * t)
        + 100 * np.sin(2 * np.pi * 10 * t)
        + np.random.randn() * 50
    )
    return float(ir_val)