"""
生理信号采集线程。

优先读取 ESP32/串口上传的 PPG 数据，提取 BPM、HRV 等特征；无硬件时可按配置回退到模拟模式。
"""

import time
import re
import random
from collections import deque

import numpy as np
try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.signal import butter, filtfilt, find_peaks
import config


def _auto_find_port():
    if serial is None:
        return None
    try:
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if "usb serial" in desc or "ch340" in desc or "esp32" in desc:
                return p.device
    except Exception:
        pass
    return None


class PhysioThread(QThread):
    """心率/HRV 采集线程，向融合层持续发送生理特征。"""
    features_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, port=None, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.ser = None

        self.raw_ppg = deque(maxlen=1200)
        self.ts_buf = deque(maxlen=1200)
        self.rr_hist = deque(maxlen=20)

        self.last_emit_t = 0.0
        self.last_fs_t = time.time()
        self.sample_counter = 0
        self.auto_fs = 25.0

        # 本地联调伪数据状态
        self._sim_phase = 0.0
        self._sim_bpm = 72.0
        self._sim_hrv = 32.0

    def stop(self):
        self._running = False

    def _open_serial(self):
        use_port = self.port or _auto_find_port()
        if not use_port:
            self.status_signal.emit("> 生理线程：未找到ESP32串口，等待重试...")
            return False
        try:
            self.ser = serial.Serial(
                port=use_port,
                baudrate=self.baudrate,
                timeout=0.02,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            self.status_signal.emit(f"> 生理线程：串口已连接 {use_port}")
            return True
        except Exception as e:
            self.status_signal.emit(f"> 生理线程：串口连接失败 {use_port} ({e})")
            self.ser = None
            return False

    @staticmethod
    def _parse_ir(line: str):
        if "[DATA]" in line:
            m = re.search(r"ir\s*=\s*(\d+)", line)
            return int(m.group(1)) if m else None
        if "," in line:
            m = re.search(r"(-?\d+)", line.split(",")[0])
            return int(m.group(1)) if m else None
        m = re.search(r"(-?\d+)", line)
        return int(m.group(1)) if m else None

    @staticmethod
    def _bandpass(sig, fs, low=0.5, high=4.0):
        nyq = 0.5 * fs
        high = min(high, nyq * 0.95)
        low = max(0.2, min(low, high - 0.1))
        if len(sig) < 40 or high <= low:
            return sig
        b, a = butter(2, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, sig)

    def _compute_features(self):
        if len(self.raw_ppg) < 120:
            return None

        ppg = np.array(self.raw_ppg, dtype=np.float64)
        fs = max(8.0, min(120.0, float(self.auto_fs)))

        # 现实导向：保留真实形态，做温和平滑与带通
        ppg_centered = ppg - np.mean(ppg[-200:])
        ppg_f = self._bandpass(ppg_centered, fs=fs, low=0.5, high=4.0)
        ppg_f = np.convolve(ppg_f, np.ones(5) / 5.0, mode="same")

        peak_distance = max(2, int(fs * 0.30))
        peak_height = np.mean(ppg_f) + 0.35 * np.std(ppg_f)
        peaks, _ = find_peaks(ppg_f, height=peak_height, distance=peak_distance)

        bpm = None
        hrv_rmssd = None
        rr_valid_ratio = 0.0

        if len(peaks) > 2:
            rr = np.diff(peaks) / fs
            rr_valid = rr[(rr > 0.35) & (rr < 1.6)]
            rr_valid_ratio = float(len(rr_valid) / max(1, len(rr)))
            if len(rr_valid) > 0:
                bpm = float(np.clip(60.0 / np.mean(rr_valid), 45.0, 120.0))
                rr_ms = rr_valid * 1000.0
                if len(rr_ms) > 1:
                    hrv_rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

        amp_p95 = float(np.percentile(np.abs(ppg_f[-200:]), 95))
        if rr_valid_ratio > 0.75 and amp_p95 > 20:
            quality = "good"
        elif rr_valid_ratio > 0.4 and amp_p95 > 10:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "timestamp": time.time(),
            "fs": float(fs),
            "bpm": bpm,
            "hrv_rmssd": hrv_rmssd,
            "rr_valid_ratio": rr_valid_ratio,
            "ppg_amp_p95": amp_p95,
            "ppg_quality": quality,
        }

    def _sim_features(self):
        self._sim_phase += 0.12
        self._sim_bpm += random.uniform(-0.6, 0.6)
        self._sim_bpm = max(60.0, min(88.0, self._sim_bpm))

        self._sim_hrv += random.uniform(-1.2, 1.2)
        self._sim_hrv = max(20.0, min(55.0, self._sim_hrv))

        # 构造近似真实的平滑幅值与质量
        amp = 900.0 + 180.0 * np.sin(self._sim_phase) + random.uniform(-40.0, 40.0)
        rr_valid_ratio = max(0.65, min(0.98, 0.85 + random.uniform(-0.08, 0.08)))

        if rr_valid_ratio > 0.8:
            q = "good"
        elif rr_valid_ratio > 0.65:
            q = "fair"
        else:
            q = "poor"

        return {
            "timestamp": time.time(),
            "fs": 30.0,
            "bpm": float(self._sim_bpm),
            "hrv_rmssd": float(self._sim_hrv),
            "rr_valid_ratio": float(rr_valid_ratio),
            "ppg_amp_p95": float(max(100.0, amp)),
            "ppg_quality": q,
        }

    def run(self):
        while self._running:
            if not self.ser or not self.ser.is_open:
                if not self._open_serial():
                    if getattr(config, "PHYSIO_SIM_WHEN_NO_SERIAL", False):
                        feat = self._sim_features()
                        self.features_signal.emit(feat)
                        self.status_signal.emit("> 生理线程：无串口，已切换本地伪数据模式")
                        self.msleep(1000)
                        continue

                    self.msleep(1200)
                    continue

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        ir = self._parse_ir(line)
                        if ir is not None and ir > 0:
                            self.raw_ppg.append(ir)
                            self.ts_buf.append(time.time())
                            self.sample_counter += 1

                now = time.time()
                if now - self.last_fs_t >= 1.0:
                    if self.sample_counter > 0:
                        self.auto_fs = 0.8 * self.auto_fs + 0.2 * float(self.sample_counter)
                    self.sample_counter = 0
                    self.last_fs_t = now

                if now - self.last_emit_t >= 1.0:
                    feat = self._compute_features()
                    if feat is not None:
                        self.features_signal.emit(feat)
                    self.last_emit_t = now

                self.msleep(10)
            except Exception as e:
                self.status_signal.emit(f"> 生理线程：读取异常，重连中 ({e})")
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self.msleep(800)

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
