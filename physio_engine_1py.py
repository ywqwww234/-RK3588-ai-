# -*- coding: utf-8 -*-
"""1.py 心率/HRV 内核（无 UI、无 HTTP 转发），供 PhysioThread 使用。"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

import ppg_signal

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

FS = 100.0
BUF_LEN = 800
SAMPLES_PER_TICK = 5  # 50ms @ 100Hz


class SerialLineReader:
    def __init__(self, ser, max_queue=800):
        self.ser = ser
        self.queue = queue.Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.last_line_ts = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                if self.ser is None or not self.ser.is_open:
                    break
                raw = self.ser.readline()
                if not raw:
                    time.sleep(0.002)
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.last_line_ts = time.time()
                while True:
                    try:
                        self.queue.put_nowait(line)
                        break
                    except queue.Full:
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
            except Exception:
                time.sleep(0.02)

    def get_lines(self, max_lines=80):
        lines = []
        for _ in range(max_lines):
            try:
                lines.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def stop(self):
        self.stop_event.set()


class SimGenerator:
    def __init__(self):
        self.sim_counter = 0
        self.hr_val = 62.0
        self.hrv_val = 28.0

    def gen_ppg_full(self) -> float:
        self.sim_counter += 1
        t = self.sim_counter / FS
        return (
            10000.0
            + 2000 * np.sin(2 * np.pi * 1.0 * t)
            + 500 * np.sin(2 * np.pi * 0.2 * t)
            + 100 * np.sin(2 * np.pi * 10 * t)
            + np.random.randn() * 50
        )

    def gen_hr(self) -> float:
        self.hr_val += random.uniform(-0.3, 0.3)
        return float(np.clip(self.hr_val, 55, 88))

    def gen_hrv(self) -> float:
        self.hrv_val += random.uniform(-1, 1)
        return float(np.clip(self.hrv_val, 18, 55))


def auto_find_hr_port(exclude_substrings: Optional[List[str]] = None) -> Optional[str]:
    if serial is None:
        return None
    exclude_substrings = exclude_substrings or ["bluetooth", "hc-05"]
    try:
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if any(x in desc or x in hwid for x in exclude_substrings):
                continue
            if "usb serial" in desc or "esp32" in desc or "usb-serial" in desc:
                return p.device
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if "ch340" in desc and "bluetooth" not in desc:
                return p.device
    except Exception:
        pass
    return None


def parse_serial_line(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not line:
        return out
    if "[DATA]" in line:
        m = re.search(r"ir\s*=\s*(\d+)", line)
        if m:
            out["ir"] = int(m.group(1))
        m = re.search(r"\bfinger\s*=\s*(-?\d+)", line)
        if m:
            out["finger"] = int(m.group(1)) == 1
        m = re.search(r"\bbpm\s*=\s*(-?\d+)", line)
        if m:
            out["bpm"] = int(m.group(1))
    elif "," in line:
        m = re.search(r"(-?\d+)", line.split(",")[0])
        if m:
            out["ir"] = int(m.group(1))
    else:
        m = re.search(r"(-?\d+)", line)
        if m:
            out["ir"] = int(m.group(1))
    return out


class PhysioEngine1Py:
    """与 1.py 真机链一致的缓冲与指标计算。"""

    def __init__(self, fs: float = FS):
        self.fs = fs
        self.data_ppg = np.zeros(BUF_LEN, dtype=np.float64)
        self.data_hr: List[float] = []
        self.data_hrv: List[float] = []
        self.data_ecg: List[float] = []
        self.rr_intervals_buffer: List[float] = []
        self.hrv_buffer: List[float] = []
        self.ir_diff_hist: List[float] = []
        self.last_ir_val: Optional[float] = None
        self.serial_finger_present = True
        self.hand_present = True
        self.current_hr = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_lf_hf = 0.0
        self.hrv_target: Optional[float] = None
        self.hrv_display_ema: Optional[float] = None
        self.counter = 0
        self.sim = SimGenerator()
        self.mode = "offline"  # serial | sim | offline

    def ingest_ir(self, ir_val: int, finger: Optional[bool] = None) -> None:
        if ir_val is None or ir_val <= 0:
            return
        if finger is not None:
            self.serial_finger_present = bool(finger)
        ir_f = float(ir_val)
        if self.last_ir_val is not None:
            diff = abs(ir_f - self.last_ir_val)
            self.ir_diff_hist.append(diff)
            if len(self.ir_diff_hist) > 120:
                self.ir_diff_hist = self.ir_diff_hist[-120:]
            med_diff = float(np.median(self.ir_diff_hist)) if len(self.ir_diff_hist) > 10 else 300.0
            jump_th = max(1200.0, 8.0 * med_diff)
            if diff > jump_th:
                ir_f = self.last_ir_val
        self.data_ppg = np.roll(self.data_ppg, -1)
        self.data_ppg[-1] = ir_f
        self.last_ir_val = ir_f

    def hold_last_sample(self) -> None:
        if self.last_ir_val is not None:
            self.data_ppg = np.roll(self.data_ppg, -1)
            self.data_ppg[-1] = self.last_ir_val

    def tick_sim(self, n_samples: int = SAMPLES_PER_TICK) -> None:
        self.mode = "sim"
        self.hand_present = True
        for _ in range(n_samples):
            ir_val = self.sim.gen_ppg_full()
            self.data_ppg = np.roll(self.data_ppg, -1)
            self.data_ppg[-1] = float(ir_val)
            self.last_ir_val = float(ir_val)
        hr = self.sim.gen_hr()
        hrv = self.sim.gen_hrv()
        self.data_hr.append(hr)
        if len(self.data_hr) > 200:
            self.data_hr.pop(0)
        self.data_hrv.append(hrv)
        if len(self.data_hrv) > 200:
            self.data_hrv.pop(0)
        self.current_hr = hr
        self.current_rmssd = hrv
        self.current_sdnn = 40.0 + random.uniform(-5, 5)
        self.current_lf_hf = 1.2 + random.uniform(-0.2, 0.2)
        self._append_ecg_sample()

    def process_ppg_signal(self) -> None:
        self.hand_present = bool(self.serial_finger_present)
        if not self.hand_present:
            return
        valid_count = int(np.count_nonzero(self.data_ppg))
        if valid_count <= 120:
            return
        filtered_ppg, filtered_tail = ppg_signal.filter_ppg_tail(
            self.data_ppg, fs=self.fs, tail_n=400
        )
        peaks, _ = find_peaks(
            filtered_ppg,
            height=np.mean(filtered_ppg) + np.std(filtered_ppg) * 0.4,
            distance=int(self.fs * 0.25),
        )
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.fs
            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
            if len(rr_intervals) > 0:
                hr = 60.0 / np.mean(rr_intervals)
                hr = float(np.clip(hr, 50, 100))
                self.data_hr.append(hr)
                if len(self.data_hr) > 200:
                    self.data_hr.pop(0)
                self.current_hr = float(hr)
                self.rr_intervals_buffer.extend(rr_intervals.tolist())
                if len(self.rr_intervals_buffer) > 8:
                    self.rr_intervals_buffer = self.rr_intervals_buffer[-8:]
                hrv = float(np.std(self.rr_intervals_buffer) * 1000.0)
                hrv = float(np.clip(hrv, 20, 100))
                self.hrv_buffer.append(hrv)
                if len(self.hrv_buffer) > 8:
                    self.hrv_buffer = self.hrv_buffer[-8:]
                self.hrv_target = float(np.mean(self.hrv_buffer))
                if len(self.rr_intervals_buffer) >= 5:
                    sdnn, rmssd, lf_hf = ppg_signal.calculate_hrv_indices(self.rr_intervals_buffer)
                    if sdnn > 0 and rmssd > 0 and lf_hf > 0:
                        self.current_sdnn = float(sdnn)
                        self.current_rmssd = float(rmssd)
                        self.current_lf_hf = float(lf_hf)
        if self.hrv_target is not None:
            alpha = 0.18
            if self.hrv_display_ema is None:
                self.hrv_display_ema = float(self.hrv_target)
            else:
                self.hrv_display_ema = self.hrv_display_ema * (1 - alpha) + float(self.hrv_target) * alpha
            self.data_hrv.append(float(self.hrv_display_ema))
            if len(self.data_hrv) > 200:
                self.data_hrv.pop(0)
        self._append_ecg_sample()

    def _append_ecg_sample(self) -> None:
        self.counter += 1
        t = self.counter * 0.01
        ecg_value = 0.0
        phase = t % 1.0
        if 0.1 < phase < 0.2:
            ecg_value = 0.1 * np.sin((phase - 0.1) * 30 * np.pi)
        elif 0.2 < phase < 0.25:
            ecg_value = 0.8 * np.sin((phase - 0.2) * 100 * np.pi)
        elif 0.25 < phase < 0.3:
            ecg_value = -0.4 * np.sin((phase - 0.25) * 80 * np.pi)
        elif 0.3 < phase < 0.5:
            ecg_value = 0.3 * np.sin((phase - 0.4) * 40 * np.pi)
        ecg_value += np.random.randn() * 0.03
        self.data_ecg.append(float(ecg_value))
        if len(self.data_ecg) > 800:
            self.data_ecg.pop(0)

    def display_ppg_point(self) -> float:
        display, metrics = ppg_signal.compute_ppg_display(
            self.data_ppg, fs=self.fs, buf_len=len(self.data_ppg)
        )
        if display is not None and len(display) > 0:
            return float(display[-1])
        if self.last_ir_val is not None:
            return float(self.last_ir_val)
        return 0.0

    def build_features_dict(self) -> Dict[str, Any]:
        ppg_pt = self.display_ppg_point()
        rmssd = self.current_rmssd if self.current_rmssd > 0 else (
            float(self.hrv_target) if self.hrv_target else None
        )
        bpm = self.current_hr if self.current_hr > 0 else None
        rr_ratio = 0.85 if self.hand_present and bpm else 0.0
        if bpm and 45 <= bpm <= 120:
            rr_ratio = max(rr_ratio, 0.75)
        qua = "good" if rr_ratio > 0.75 and self.hand_present else (
            "fair" if self.hand_present else "poor"
        )
        return {
            "timestamp": time.time(),
            "fs": float(self.fs),
            "bpm": bpm,
            "hrv_rmssd": rmssd,
            "sdnn": self.current_sdnn if self.current_sdnn > 0 else None,
            "lfhf": self.current_lf_hf if self.current_lf_hf > 0 else None,
            "rr_valid_ratio": float(rr_ratio),
            "ppg_amp_p95": float(ppg_pt),
            "ppg_quality": qua,
            "hand_present": bool(self.hand_present),
            "source": self.mode,
            "ir_value": int(self.last_ir_val) if self.last_ir_val is not None else None,
        }

    def build_wave_packet(self) -> Dict[str, Any]:
        ppg_pt = self.display_ppg_point()
        hrv = self.current_rmssd if self.current_rmssd > 0 else self.hrv_display_ema
        ecg = self.data_ecg[-1] if self.data_ecg else 0.0
        return {
            "ppg": ppg_pt,
            "ecg": float(ecg),
            "hr": float(self.current_hr) if self.current_hr > 0 else None,
            "hrv": float(hrv) if hrv is not None and hrv > 0 else None,
            "ir_values": [float(x) for x in self.data_ppg[-80:] if x > 0],
        }