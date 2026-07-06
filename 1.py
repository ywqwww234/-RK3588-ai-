"""
ESP32 心率/HRV 感知节点 - 边缘监控终端·生理模块
- 自动检测 ESP32 串口（CH340/USB Serial），失败自动进入本地模拟模式
- 保留全部底层逻辑：Butter 带通、抗跳变中位数、find_peaks、SDNN/RMSSD/LF-HF、手势检测
- 工业级 UI：与脑电节点统一深色风格
"""

import math
import os
import queue
import random
import re
import socket
import sys
import threading
import time
from collections import deque

# 强制 pyqtgraph 绑定 PyQt5，避免 Qt 混用导致 addWidget 类型不匹配
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

import numpy as np
import requests
import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QTimer, QRectF
import pyqtgraph as pg
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSizePolicy, QTextEdit, QVBoxLayout,
                             QWidget)
from scipy.signal import butter, filtfilt, find_peaks

import ppg_signal


B_HOST = "127.0.0.1"
B_PORT = 5001
POST_URL = f"http://{B_HOST}:{B_PORT}/a2b/telemetry"
DEVICE_ID = "a_host_hr_01"
PUSH_INTERVAL_SEC = 0.08
# 学生端嵌入：False = 不 POST 到 esp32_receiver
ENABLE_REMOTE_POST_DEFAULT = False


class RemotePoster:
    def __init__(self, post_url, min_interval_sec=1.0):
        self.post_url = post_url
        self.min_interval_sec = float(min_interval_sec)
        self.session = requests.Session()
        self.session.trust_env = False
        self.queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._last_send_ts = 0.0
        self._thread.start()

    def submit_latest(self, payload):
        if payload is None:
            return
        while True:
            try:
                self.queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    return

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.time()
            wait_sec = self.min_interval_sec - (now - self._last_send_ts)
            if wait_sec > 0:
                time.sleep(min(wait_sec, self.min_interval_sec))
            try:
                self.session.post(self.post_url, json=payload, timeout=1.2)
                self._last_send_ts = time.time()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()


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
                line = raw.decode('utf-8', errors='ignore').strip()
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


# ============================================================
# 主题（与脑电节点完全一致）
# ============================================================
C_BG = '#0a0e27'
C_PANEL = '#0e1330'
C_PANEL_HI = '#1a2150'
C_BORDER = '#2a3270'
C_ACCENT = '#00d4ff'
C_ACCENT2 = '#7c4dff'
C_OK = '#2ed573'
C_WARN = '#ffa502'
C_RISK = '#ff4757'
C_TEXT = '#e8eaf6'
C_DIM = '#7986cb'

C_PPG = '#00d4ff'
C_ECG = '#ff4757'
C_HR = '#2ed573'
C_HRV = '#7c4dff'

QSS = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    background-color: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: {C_BG};
    color: {C_ACCENT};
    letter-spacing: 1px;
}}
QLabel {{ color: {C_TEXT}; background: transparent; }}
QPushButton {{
    background-color: {C_PANEL_HI};
    color: {C_ACCENT};
    border: 1px solid {C_ACCENT};
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: bold;
    min-width: 60px;
}}
QPushButton:hover {{ background-color: {C_ACCENT}; color: {C_BG}; }}
QPushButton:disabled {{
    background-color: {C_PANEL};
    color: {C_DIM};
    border: 1px solid {C_DIM};
}}
QComboBox {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 100px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: 1px solid {C_ACCENT};
    selection-background-color: {C_ACCENT};
    selection-color: {C_BG};
}}
QCheckBox {{ color: {C_TEXT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {C_DIM};
    background: {C_PANEL_HI};
    border-radius: 2px;
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border: 1px solid {C_ACCENT};
}}
QTextEdit {{
    background-color: #050818;
    color: {C_OK};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
}}
"""


# ============================================================
# 信号处理（底层逻辑保留）
# ============================================================
FS = 100
TICK_INTERVAL_MS = 50
SAMPLES_PER_TICK = max(1, int(FS * TICK_INTERVAL_MS / 1000))


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    return ppg_signal.butter_bandpass_filter(data, lowcut, highcut, fs, order=order)


def smooth_signal(data, window_size=5):
    return ppg_signal.smooth_signal(data, window_size=window_size)


def calculate_hrv_indices(rr_intervals):
    return ppg_signal.calculate_hrv_indices(rr_intervals)


# ============================================================
# 模拟生成器（底层逻辑保留）
# ============================================================
class SimGenerator:
    def __init__(self):
        self.ppg_phase = 0.0
        self.hr_val = 60.0
        self.hrv_val = 20.0
        self.sim_counter = 0

    def gen_ppg(self, base=233000):
        self.ppg_phase += 0.12
        wave = 200 * np.sin(self.ppg_phase) + 80 * np.sin(2 * self.ppg_phase)
        noise = random.randint(-10, 10)
        return int(base + wave + noise)

    def gen_hr(self):
        self.hr_val += random.uniform(-0.3, 0.3)
        self.hr_val = max(55, min(65, self.hr_val))
        return float(self.hr_val)

    def gen_hrv(self):
        self.hrv_val += random.uniform(-1, 1)
        self.hrv_val = max(15, min(25, self.hrv_val))
        return float(self.hrv_val)

    def gen_ecg(self, base=2000):
        if random.random() < 0.08:
            return base + random.randint(600, 1000)
        return base + random.randint(-100, 100)

    def gen_ppg_full(self):
        """模拟模式下生成 PPG 波（用于绘图）"""
        self.sim_counter += 1
        t = self.sim_counter / FS
        heart_rate = 1.0
        respiration = 0.2
        ir_val = (10000
                  + 2000 * np.sin(2 * np.pi * heart_rate * t)
                  + 500 * np.sin(2 * np.pi * respiration * t)
                  + 100 * np.sin(2 * np.pi * 10 * t)
                  + np.random.randn() * 50)
        return ir_val


# ============================================================
# 无 UI 引擎：学生端 import 1.py 直接调用（与 HRMonitorWindow 同算法）
# ============================================================
class HRMonitorEngine:
    """供 PhysioThread / 学生端嵌入；不创建窗口、默认不 HTTP 转发。"""

    MODE_OFFLINE = "offline"
    MODE_SERIAL = "serial"
    MODE_SIM = "sim"

    def __init__(self, enable_remote_post: bool = False):
        self.mode = self.MODE_OFFLINE
        self.ser = None
        self.serial_reader = None
        self.data_ppg = np.zeros(800, dtype=np.float64)
        self.data_ecg: list = []
        self.data_hr: list = []
        self.data_hrv: list = []
        self.rr_intervals_buffer: list = []
        self.hrv_buffer: list = []
        self.last_ir_val = None
        self.ir_diff_hist: list = []
        self.hrv_target = None
        self.hrv_display_ema = None
        self.current_hr = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_lf_hf = 0.0
        self.counter = 0
        self.hand_present = True
        self.serial_finger_present = True
        self.connect_time = 0.0
        self.last_data_ts = 0.0
        self._pending_remote_ir: list = []
        self.sim = SimGenerator()
        self.remote_poster = None
        if enable_remote_post:
            self.remote_poster = RemotePoster(POST_URL, PUSH_INTERVAL_SEC)
        self._last_remote_payload = None
        self._last_wave_push_ts = 0.0

    def stop(self):
        reader = getattr(self, "serial_reader", None)
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
            self.serial_reader = None
        if self.ser and getattr(self.ser, "is_open", False):
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        if self.remote_poster is not None:
            try:
                self.remote_poster.stop()
            except Exception:
                pass

    def _guess_esp32_port(self, ports):
        keywords = ("usb serial", "ch340", "esp32", "cp210", "silabs", "ftdi")
        for p in ports:
            desc = (p.description or "").lower()
            if "bluetooth" in desc:
                continue
            if any(k in desc for k in keywords):
                return p
        return None

    def auto_connect_serial(self, port_name=None, baud=115200) -> bool:
        if port_name:
            return self._do_serial_connect(port_name, int(baud))
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return False
        target = self._guess_esp32_port(ports)
        if target is None:
            return False
        return self._do_serial_connect(target.device, int(baud))

    def _do_serial_connect(self, port_name, baud_rate) -> bool:
        try:
            self.ser = serial.Serial(
                port=port_name,
                baudrate=baud_rate,
                timeout=0.02,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.serial_reader = SerialLineReader(self.ser)
            self.mode = self.MODE_SERIAL
            self.connect_time = time.time()
            self.last_data_ts = time.time()
            return True
        except Exception:
            self.ser = None
            self.serial_reader = None
            return False

    def enter_simulation(self):
        self.stop()
        self.mode = self.MODE_SIM
        self.rr_intervals_buffer.clear()
        self.hrv_buffer.clear()
        self.last_ir_val = None

    def tick(self):
        if self.mode == self.MODE_SIM:
            self._tick_sim()
        elif self.mode == self.MODE_SERIAL:
            self._tick_serial()
        if self.remote_poster is not None:
            self._push_remote_hr()
            self._push_remote_wave()

    def _tick_sim(self):
        for _ in range(SAMPLES_PER_TICK):
            ir_val = self.sim.gen_ppg_full()
            self.data_ppg = np.roll(self.data_ppg, -1)
            self.data_ppg[-1] = float(ir_val)
            self.last_ir_val = float(ir_val)
            hr = self.sim.gen_hr()
            self.data_hr.append(hr)
            if len(self.data_hr) > 200:
                self.data_hr.pop(0)
            hrv = self.sim.gen_hrv()
            self.data_hrv.append(hrv)
            if len(self.data_hrv) > 200:
                self.data_hrv.pop(0)
            ecg = self.sim.gen_ecg()
            self.data_ecg.append(ecg)
            if len(self.data_ecg) > 800:
                self.data_ecg.pop(0)
        self.current_hr = float(self.data_hr[-1]) if self.data_hr else 0.0
        self.current_rmssd = float(self.data_hrv[-1]) if self.data_hrv else 0.0
        self.current_sdnn = 40.0 + random.uniform(-5, 5)
        self.current_lf_hf = 1.2 + random.uniform(-0.2, 0.2)
        self.hand_present = True

    def _tick_serial(self):
        try:
            latest_ir = None
            latest_finger = None
            reader = getattr(self, "serial_reader", None)
            lines = reader.get_lines(80) if reader is not None else []
            for line in lines:
                if not line:
                    continue
                ir_val = None
                if "[DATA]" in line:
                    m = re.search(r"ir\s*=\s*(\d+)", line)
                    if m:
                        ir_val = int(m.group(1))
                    m = re.search(r"\bfinger\s*=\s*(-?\d+)", line)
                    if m:
                        latest_finger = int(m.group(1)) == 1
                elif "," in line:
                    m = re.search(r"(-?\d+)", line.split(",")[0])
                    if m:
                        ir_val = int(m.group(1))
                else:
                    m = re.search(r"(-?\d+)", line)
                    if m:
                        ir_val = int(m.group(1))
                if ir_val is None or ir_val <= 0:
                    continue
                latest_ir = ir_val

            if latest_ir is not None:
                self.last_data_ts = time.time()
                ir_val = latest_ir
                if self.last_ir_val is not None:
                    diff = abs(ir_val - self.last_ir_val)
                    self.ir_diff_hist.append(diff)
                    if len(self.ir_diff_hist) > 120:
                        self.ir_diff_hist = self.ir_diff_hist[-120:]
                    med_diff = np.median(self.ir_diff_hist) if len(self.ir_diff_hist) > 10 else 300
                    jump_th = max(1200, 8 * med_diff)
                    if diff > jump_th:
                        ir_val = self.last_ir_val
                self.data_ppg = np.roll(self.data_ppg, -1)
                self.data_ppg[-1] = ir_val
                self.last_ir_val = ir_val
                self._pending_remote_ir.append(int(ir_val))
                if len(self._pending_remote_ir) > 400:
                    self._pending_remote_ir = self._pending_remote_ir[-400:]
            elif self.last_ir_val is not None:
                self.data_ppg = np.roll(self.data_ppg, -1)
                self.data_ppg[-1] = self.last_ir_val

            if latest_finger is not None:
                self.serial_finger_present = bool(latest_finger)
        except Exception:
            pass

        self.hand_present = bool(self.serial_finger_present)
        if not self.hand_present:
            return
        self._process_ppg_signal()

    def _process_ppg_signal(self):
        valid_count = int(np.count_nonzero(self.data_ppg))
        if valid_count > 120:
            filtered_ppg, _ = ppg_signal.filter_ppg_tail(self.data_ppg, fs=FS, tail_n=400)
            peaks, _ = find_peaks(
                filtered_ppg,
                height=np.mean(filtered_ppg) + np.std(filtered_ppg) * 0.4,
                distance=int(FS * 0.25),
            )
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / FS
                rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                if len(rr_intervals) > 0:
                    hr = 60 / np.mean(rr_intervals)
                    hr = float(np.clip(hr, 50, 100))
                    hr += float(np.random.normal(0, 1.5))
                    self.data_hr.append(hr)
                    if len(self.data_hr) > 200:
                        self.data_hr.pop(0)
                    self.current_hr = float(hr)
                    self.rr_intervals_buffer.extend(rr_intervals.tolist())
                    if len(self.rr_intervals_buffer) > 8:
                        self.rr_intervals_buffer = self.rr_intervals_buffer[-8:]
                    hrv = float(np.clip(float(np.std(self.rr_intervals_buffer) * 1000), 20, 100))
                    hrv_value = hrv + float(np.random.normal(0, 1.0))
                    if len(self.hrv_buffer) > 0:
                        hrv_value += (self.hrv_buffer[-1] - hrv_value) * 0.08
                    self.hrv_buffer.append(hrv_value)
                    if len(self.hrv_buffer) > 8:
                        self.hrv_buffer = self.hrv_buffer[-8:]
                    self.hrv_target = float(np.mean(self.hrv_buffer))
                    if len(self.rr_intervals_buffer) >= 5:
                        sdnn, rmssd, lf_hf = calculate_hrv_indices(self.rr_intervals_buffer)
                        if sdnn > 0 and rmssd > 0 and lf_hf > 0:
                            self.current_sdnn = float(sdnn)
                            self.current_rmssd = float(rmssd)
                            self.current_lf_hf = float(lf_hf)

        if self.hrv_target is not None:
            hrv_alpha = 0.18
            if self.hrv_display_ema is None:
                self.hrv_display_ema = float(self.hrv_target)
            else:
                self.hrv_display_ema = (
                    self.hrv_display_ema * (1.0 - hrv_alpha) + float(self.hrv_target) * hrv_alpha
                )
            display_hrv = float(np.clip(self.hrv_display_ema, 0, 120))
            self.data_hrv.append(display_hrv)
            if len(self.data_hrv) > 200:
                self.data_hrv.pop(0)

        self.counter += 1
        t = self.counter * 0.01
        ecg_value = 0.0
        phase = t % 1
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

    def display_ppg_last(self) -> float:
        display, _ = ppg_signal.compute_ppg_display(self.data_ppg, fs=FS, buf_len=len(self.data_ppg))
        if display is not None and len(display) > 0:
            return float(display[-1])
        return float(self.last_ir_val or 0.0)

    def build_features_dict(self) -> dict:
        rmssd = self.current_rmssd if self.current_rmssd > 0 else None
        bpm = self.current_hr if self.current_hr > 0 else None
        qua = "good" if self.hand_present and bpm else "poor"
        return {
            "timestamp": time.time(),
            "fs": float(FS),
            "bpm": bpm,
            "hrv_rmssd": rmssd,
            "sdnn": self.current_sdnn if self.current_sdnn > 0 else None,
            "lfhf": self.current_lf_hf if self.current_lf_hf > 0 else None,
            "rr_valid_ratio": 0.85 if self.hand_present and bpm else 0.0,
            "ppg_amp_p95": self.display_ppg_last(),
            "ppg_quality": qua,
            "hand_present": bool(self.hand_present),
            "source": self.mode,
        }

    def build_wave_packet(self) -> dict:
        hrv = self.current_rmssd if self.current_rmssd > 0 else self.hrv_display_ema
        return {
            "ppg": self.display_ppg_last(),
            "ecg": float(self.data_ecg[-1]) if self.data_ecg else 0.0,
            "hr": float(self.current_hr) if self.current_hr > 0 else None,
            "hrv": float(hrv) if hrv is not None and hrv > 0 else None,
            "ir_values": [float(x) for x in self.data_ppg[-80:] if x > 0],
        }

    def _push_remote_hr(self):
        if self.current_hr <= 0 or self.current_sdnn <= 0:
            return
        payload = {
            "device_id": DEVICE_ID,
            "bpm": round(float(self.current_hr), 2),
            "sdnn": round(float(self.current_sdnn), 2),
            "rmssd": round(float(self.current_rmssd), 2),
            "lfhf": round(float(self.current_lf_hf), 4),
            "ir_value": None if self.last_ir_val is None else int(self.last_ir_val),
            "hand_present": int(bool(self.hand_present)),
        }
        if payload == self._last_remote_payload:
            return
        self._last_remote_payload = dict(payload)
        self.remote_poster.submit_latest(payload)

    def _push_remote_wave(self):
        now = time.time()
        if now - self._last_wave_push_ts < 0.08:
            return
        if not self._pending_remote_ir:
            return
        payload = {
            "device_id": DEVICE_ID,
            "ir_values": list(self._pending_remote_ir[-80:]),
            "hand_present": int(bool(self.hand_present)),
        }
        self._pending_remote_ir.clear()
        self._last_wave_push_ts = now
        self.remote_poster.submit_latest(payload)


def load_hr_monitor_module():
    """学生端：importlib 加载本文件（模块名 1.py）。"""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.py")
    spec = importlib.util.spec_from_file_location("hr_monitor_1py", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 状态胶囊
# ============================================================
class StatusPill(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_state('offline', '未连接')

    def set_state(self, state, text):
        colors = {
            'live': (C_OK, '#0d2818'),
            'sim': (C_WARN, '#2e1f05'),
            'offline': (C_RISK, '#2a0a0e'),
        }
        fg, bg = colors.get(state, colors['offline'])
        self.setText(f'  ●  {text}  ')
        self.setStyleSheet(
            f'background:{bg}; color:{fg};'
            f'border:1px solid {fg}; border-radius:10px;'
            f'padding:4px 12px; font-weight:bold;'
        )


# ============================================================
# 大数值显示卡
# ============================================================
class BigValueCard(QWidget):
    def __init__(self, title, unit='', color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.color = color
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; letter-spacing:1px;')
        self.title_lbl.setAlignment(Qt.AlignCenter)

        self.value_lbl = QLabel('--')
        self.value_lbl.setStyleSheet(
            f'color:{color}; font-size:36px; font-weight:bold; '
            f'background:transparent;')
        self.value_lbl.setAlignment(Qt.AlignCenter)

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:10px;')
        self.unit_lbl.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.unit_lbl)

        self.setStyleSheet(
            f'BigValueCard {{ background:{C_PANEL}; '
            f'border:1px solid {C_BORDER}; border-radius:6px; }}')

    def set_value(self, text, color=None):
        self.value_lbl.setText(text)
        if color is not None:
            self.color = color
            self.value_lbl.setStyleSheet(
                f'color:{color}; font-size:36px; font-weight:bold; '
                f'background:transparent;')


# ============================================================
# 指标条目（SDNN / RMSSD / LF-HF 状态行）
# ============================================================
class MetricRow(QWidget):
    def __init__(self, name, ref_text, parent=None):
        super().__init__(parent)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(6, 4, 6, 4)
        self.lay.setSpacing(8)

        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet(f'color:{C_TEXT}; font-weight:bold;')
        self.name_lbl.setMinimumWidth(60)

        self.value_lbl = QLabel('--')
        self.value_lbl.setStyleSheet(
            f'color:{C_ACCENT}; font-size:14px; font-weight:bold;')
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_lbl.setMinimumWidth(80)

        self.ref_lbl = QLabel(ref_text)
        self.ref_lbl.setStyleSheet(f'color:{C_DIM}; font-size:11px;')
        self.ref_lbl.setMinimumWidth(80)
        self.ref_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.status_lbl = QLabel('--')
        self.status_lbl.setStyleSheet(f'color:{C_DIM}; font-weight:bold;')
        self.status_lbl.setMinimumWidth(70)
        self.status_lbl.setAlignment(Qt.AlignCenter)

        self.lay.addWidget(self.name_lbl)
        self.lay.addWidget(self.value_lbl)
        self.lay.addWidget(self.ref_lbl)
        self.lay.addWidget(self.status_lbl)

    def update(self, value_text, status_text, color):
        self.value_lbl.setText(value_text)
        self.status_lbl.setText(status_text)
        self.status_lbl.setStyleSheet(f'color:{color}; font-weight:bold;')


# ============================================================
# 主窗口
# ============================================================
class HRMonitorWindow(QMainWindow):

    MODE_OFFLINE = 'offline'
    MODE_SERIAL = 'serial'
    MODE_SIM = 'sim'
    MODE_UDP = 'udp'

    UDP_PORT = 5005

    def __init__(self):
        super().__init__()
        self.ser = None
        self.serial_reader = None
        self.udp_sock = None
        self.mode = self.MODE_OFFLINE
        self.connect_time = 0

        # === 缓冲数据（保留原变量名以便对照）===
        self.data_ppg = np.zeros(800, dtype=np.float64)
        self.data_ecg = [0.0] * 800
        self.data_hr = [75.0] * 200
        self.data_hrv = [40.0] * 200
        self.sdnn_trend_data = []
        self.hrv_buffer = []
        self.rr_intervals_buffer = []
        self.hrv_metric_smooth = {"sdnn": None, "rmssd": None, "lfhf": None}
        self.last_ir_val = None
        self.ir_diff_hist = []

        # 显示层平滑：仅用于曲线观感，不改变底层真实计算
        self.hr_display_ema = None
        self.hrv_display_ema = None
        self.sdnn_display_ema = None
        self.hrv_target = None

        # 当前值快照
        self.current_hr = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_lf_hf = 0.0

        # 计数 / 调试
        self.counter = 0
        self.right_update_counter = 0
        self.hand_present = True
        self.serial_finger_present = True
        self.serial_last_bpm = 0.0
        self.serial_last_spo2 = 0.0
        self.serial_last_avg = 0.0
        self._pending_remote_ir = []
        self._last_wave_push_ts = 0.0
        self.debug_start_t = time.time()
        self.debug_tick_t = self.debug_start_t
        self.debug_raw_ir = []
        self.debug_used_ir = []
        self.debug_dropped = 0
        self.debug_empty_ticks = 0
        self.remote_poster = RemotePoster(POST_URL, PUSH_INTERVAL_SEC)
        self._last_remote_payload = None

        # 连接诊断
        self.last_data_ts = 0.0
        self._last_watchdog_log_ts = 0.0
        self.serial_err_count = 0
        self.tick_err_count = 0

        self.sim = SimGenerator()

        self.init_ui()
        self._init_udp()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_tick)
        self.timer.start(TICK_INTERVAL_MS)

        self.watchdog_timer = QTimer()
        self.watchdog_timer.timeout.connect(self.check_watchdog)
        self.watchdog_timer.start(1000)

        # 自动连接
        QTimer.singleShot(300, self.auto_connect)

    # =====================================================
    # UI
    # =====================================================
    def init_ui(self):
        self.setWindowTitle('边缘监控终端 · 心率感知节点 (ESP32 HRV Edge Node)')
        self.setGeometry(60, 50, 1440, 880)
        self.setStyleSheet(QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # === 标题栏 ===
        header = QHBoxLayout()
        title = QLabel('💗  心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE')
        title.setStyleSheet(f'color:{C_ACCENT}; font-size:18px;'
                            f'font-weight:bold; letter-spacing:3px;')
        header.addWidget(title)
        header.addStretch()
        self.status_pill = StatusPill()
        header.addWidget(self.status_pill)
        self.hand_pill = StatusPill()
        self.hand_pill.set_state('sim', '手部检测：等待')
        header.addWidget(self.hand_pill)
        root.addLayout(header)

        # === 控制条 ===
        ctrl_group = QGroupBox('连接控制 · CONNECTION')
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('串口:'))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(220)
        ctrl.addWidget(self.port_combo)
        ctrl.addWidget(QLabel('波特率:'))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['115200', '57600', '9600', '38400'])
        self.baud_combo.setCurrentText('115200')
        ctrl.addWidget(self.baud_combo)
        self.refresh_btn = QPushButton('刷新')
        self.refresh_btn.clicked.connect(self.refresh_ports)
        ctrl.addWidget(self.refresh_btn)
        self.connect_btn = QPushButton('连接')
        self.connect_btn.clicked.connect(self.toggle_connection)
        ctrl.addWidget(self.connect_btn)
        self.sim_btn = QPushButton('进入模拟')
        self.sim_btn.clicked.connect(
            lambda: self.enter_simulation('用户手动切换至模拟模式'))
        ctrl.addWidget(self.sim_btn)
        ctrl.addStretch()
        self.sample_label = QLabel('采样: 100 Hz')
        self.sample_label.setStyleSheet(f'color:{C_DIM}; padding:0 8px;')
        ctrl.addWidget(self.sample_label)
        ctrl_group.setLayout(ctrl)
        root.addWidget(ctrl_group)

        # === 主体：左波形 + 右指标 ===
        body = QHBoxLayout()
        body.setSpacing(10)

        # ---- 左：4 个波形堆叠 ----
        wave_group = QGroupBox('生理波形矩阵 · PHYSIOLOGICAL WAVEFORM MATRIX')
        wave_lay = QVBoxLayout()
        wave_lay.setContentsMargins(8, 8, 8, 8)
        wave_lay.setSpacing(6)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(C_PANEL)

        # 1) PPG
        self.p_ppg = self.glw.addPlot(row=0, col=0)
        self._style_plot(self.p_ppg, '1. PPG · 红外信号', C_PPG)
        self.curve_ppg = self.p_ppg.plot(pen=pg.mkPen(C_PPG, width=1.8))

        # 2) ECG
        self.p_ecg = self.glw.addPlot(row=1, col=0)
        self._style_plot(self.p_ecg, '2. ECG · 模拟波形', C_ECG)
        self.curve_ecg = self.p_ecg.plot(pen=pg.mkPen(C_ECG, width=1.4))

        # 3) HR
        self.p_hr = self.glw.addPlot(row=2, col=0)
        self._style_plot(self.p_hr, '3. HR · 心率 (bpm)', C_HR)
        self.curve_hr = self.p_hr.plot(pen=pg.mkPen(C_HR, width=1.6))

        # 4) HRV
        self.p_hrv = self.glw.addPlot(row=3, col=0)
        self._style_plot(self.p_hrv, '4. HRV · 心率变异性 (ms)', C_HRV)
        self.curve_hrv = self.p_hrv.plot(pen=pg.mkPen(C_HRV, width=1.6))

        wave_lay.addWidget(self.glw)
        wave_group.setLayout(wave_lay)
        body.addWidget(wave_group, 7)

        # ---- 右：指标 ----
        right = QVBoxLayout()
        right.setSpacing(10)

        # 即时数值卡
        live_group = QGroupBox('实时关键指标 · LIVE READING')
        live_lay = QHBoxLayout()
        self.card_hr = BigValueCard('心率 HR', 'bpm', C_HR)
        self.card_sdnn = BigValueCard('HRV (SDNN)', 'ms', C_HRV)
        live_lay.addWidget(self.card_hr)
        live_lay.addWidget(self.card_sdnn)
        live_group.setLayout(live_lay)
        right.addWidget(live_group)

        # 自主神经
        ans_group = QGroupBox('自主神经功能 · AUTONOMIC NERVOUS SYSTEM')
        ans_lay = QVBoxLayout()
        ans_lay.setContentsMargins(8, 8, 8, 8)
        ans_lay.setSpacing(4)

        # 表头
        header_row = QHBoxLayout()
        header_row.setContentsMargins(6, 0, 6, 0)
        for txt, w in [('指标', 60), ('当前值', 80), ('参考范围', 80), ('状态', 70)]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(f'color:{C_DIM}; font-size:11px; letter-spacing:1px;')
            lbl.setAlignment(Qt.AlignCenter if txt != '指标' else Qt.AlignLeft)
            lbl.setMinimumWidth(w)
            header_row.addWidget(lbl)
        ans_lay.addLayout(header_row)

        self.row_sdnn = MetricRow('SDNN', '> 50 ms')
        self.row_rmssd = MetricRow('RMSSD', '27 ± 12 ms')
        self.row_lf_hf = MetricRow('LF/HF', '1 ~ 2')
        ans_lay.addWidget(self.row_sdnn)
        ans_lay.addWidget(self.row_rmssd)
        ans_lay.addWidget(self.row_lf_hf)

        ans_group.setLayout(ans_lay)
        right.addWidget(ans_group)

        # SDNN 趋势
        trend_group = QGroupBox('SDNN 趋势 · SDNN TREND')
        trend_lay = QVBoxLayout()
        trend_lay.setContentsMargins(8, 8, 8, 8)
        self.trend_plot = pg.PlotWidget()
        self.trend_plot.setBackground(C_PANEL)
        self.trend_plot.showGrid(x=True, y=True, alpha=0.12)
        self.trend_plot.setLabel('left', 'SDNN (ms)', color=C_DIM)
        self.trend_plot.setLabel('bottom', 'Time (s)', color=C_DIM)
        for ax in ('left', 'bottom'):
            self.trend_plot.getAxis(ax).setPen(C_BORDER)
            self.trend_plot.getAxis(ax).setTextPen(C_DIM)
        self.trend_curve = self.trend_plot.plot(
            pen=pg.mkPen(C_ACCENT, width=1.5),
            fillLevel=0,
            brush=pg.mkBrush(QColor(0, 212, 255, 60)))
        # 50ms 参考线
        ref_line = pg.InfiniteLine(
            pos=50, angle=0,
            pen=pg.mkPen(C_OK, style=Qt.DashLine, width=1))
        self.trend_plot.addItem(ref_line)
        self.trend_plot.setMinimumHeight(140)
        trend_lay.addWidget(self.trend_plot)
        trend_group.setLayout(trend_lay)
        right.addWidget(trend_group)

        # 风险评分
        risk_group = QGroupBox('情绪风险指数 · MOOD RISK INDEX')
        rlay = QVBoxLayout()
        rlay.setContentsMargins(8, 8, 8, 8)
        self.risk_value = QLabel('0.00')
        self.risk_value.setAlignment(Qt.AlignCenter)
        self.risk_value.setStyleSheet(
            f'color:{C_OK}; font-size:36px; font-weight:bold; '
            f'background:transparent;')
        self.risk_tag = QLabel('低 · LOW')
        self.risk_tag.setAlignment(Qt.AlignCenter)
        self.risk_tag.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; letter-spacing:2px;')
        self.risk_bar = QLabel()
        self.risk_bar.setFixedHeight(5)
        self.risk_bar.setStyleSheet(
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_OK}, stop:0.5 {C_WARN}, stop:1 {C_RISK});'
            f'border-radius:2px;')
        rlay.addWidget(self.risk_value)
        rlay.addWidget(self.risk_tag)
        rlay.addSpacing(2)
        rlay.addWidget(self.risk_bar)
        risk_group.setLayout(rlay)
        right.addWidget(risk_group)

        right.addStretch()
        body.addLayout(right, 3)
        root.addLayout(body, 1)

        # === 底部 ===
        log_row = QHBoxLayout()
        log_group = QGroupBox('系统日志 · LOG')
        llay = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(110)
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(120)
        llay.addWidget(self.log_text)
        log_group.setLayout(llay)
        log_row.addWidget(log_group, 2)

        dbg_group = QGroupBox('调试统计 · DEBUG')
        dlay = QVBoxLayout()
        self.debug_text = QTextEdit()
        self.debug_text.setMaximumHeight(110)
        self.debug_text.setReadOnly(True)
        self.debug_text.document().setMaximumBlockCount(120)
        dlay.addWidget(self.debug_text)
        dbg_group.setLayout(dlay)
        log_row.addWidget(dbg_group, 1)
        root.addLayout(log_row)

        self.refresh_ports()

    def _style_plot(self, plot, title, color):
        plot.setMenuEnabled(False)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setTitle(title, color=color, size='10pt')
        for ax in ('left', 'bottom'):
            plot.getAxis(ax).setPen(C_BORDER)
            plot.getAxis(ax).setTextPen(C_DIM)

    # =====================================================
    # 自动连接 / 模式
    # =====================================================
    def auto_connect(self):
        # 若 UDP 已绑定，优先使用 UDP，不再回退到串口/模拟
        if self.mode == self.MODE_UDP and self.udp_sock is not None:
            self.log(f'UDP 监听就绪：0.0.0.0:{self.UDP_PORT}')
            return
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.enter_simulation('未发现可用串口，已切换本地模拟数据模式')
            return
        target = self._guess_esp32_port(ports)
        if target is None:
            self.enter_simulation('未识别到 ESP32 设备 (USB Serial/CH340)，已切换本地模拟数据模式')
            return
        for i in range(self.port_combo.count()):
            if self.port_combo.itemText(i).startswith(target.device):
                self.port_combo.setCurrentIndex(i)
                break
        self._do_serial_connect(target.device, int(self.baud_combo.currentText()))

    def _guess_esp32_port(self, ports):
        keywords = ('usb serial', 'ch340', 'esp32', 'cp210', 'silabs', 'ftdi')
        for p in ports:
            desc = (p.description or '').lower()
            if any(k in desc for k in keywords):
                return p
        return None

    def enter_simulation(self, reason=''):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.mode = self.MODE_SIM
        self.connect_btn.setText('连接')
        self.sim_btn.setText('退出模拟')
        self.status_pill.set_state('sim', '信号源：本地模拟')
        self.hand_pill.set_state('sim', '手部检测：模拟')
        # 重置部分缓冲
        self.rr_intervals_buffer.clear()
        self.hrv_buffer.clear()
        self.last_ir_val = None
        if reason:
            self.log(reason)

    def _set_offline(self):
        self.mode = self.MODE_OFFLINE
        self.connect_btn.setText('连接')
        self.sim_btn.setText('进入模拟')
        self.status_pill.set_state('offline', '信号源：离线')
        self.hand_pill.set_state('offline', '手部检测：--')

    def refresh_ports(self):
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.port_combo.addItem(f'{port.device} - {port.description}')
        if not ports:
            self.log('未发现可用串口')

    def toggle_connection(self):
        if self.mode == self.MODE_SERIAL:
            self.disconnect_serial()
            self._set_offline()
            return
        port_info = self.port_combo.currentText()
        if not port_info:
            self.log('请先选择串口')
            return
        port_name = port_info.split(' - ', 1)[0].strip()
        baud_rate = int(self.baud_combo.currentText())
        self._do_serial_connect(port_name, baud_rate)

    def _do_serial_connect(self, port_name, baud_rate):
        try:
            self.log(f'[SERIAL] 尝试连接 {port_name} @ {baud_rate}')
            self.ser = serial.Serial(
                port=port_name, baudrate=baud_rate, timeout=0.02,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.serial_reader = SerialLineReader(self.ser)
            self.mode = self.MODE_SERIAL
            self.connect_time = time.time()
            self.last_data_ts = time.time()
            self.serial_err_count = 0
            self.connect_btn.setText('断开')
            self.sim_btn.setText('进入模拟')
            self.status_pill.set_state('live', f'信号源：ESP32 ({port_name})')
            self.hand_pill.set_state('sim', '手部检测：等待信号')
            self.log(f'[SERIAL] 已连接 {port_name} @ {baud_rate}')
        except Exception as e:
            self.log(f'[SERIAL] 连接失败: {type(e).__name__}: {e}')
            self.enter_simulation('ESP32 串口打开失败，已切换本地模拟数据模式')

    def disconnect_serial(self):
        reader = getattr(self, 'serial_reader', None)
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
            self.serial_reader = None
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.log('已断开串口')

    def check_watchdog(self):
        if self.mode != self.MODE_SERIAL:
            return
        now = time.time()
        # 仅在确实长期无有效数据时才切换，避免误判断开
        no_data_sec = now - max(self.connect_time, self.last_data_ts or 0)
        if False and no_data_sec > 12:
            self.log(f'[WATCHDOG] 串口 {no_data_sec:.1f}s 无有效数据，准备切换模拟。')
            self.disconnect_serial()
            self.enter_simulation('ESP32 串口连接但无数据，已切换模拟模式')

    # =====================================================
    # 手势检测（保留原逻辑）
    # =====================================================
    def check_hand_present(self):
        if len(self.data_ppg) < 200:
            return True
        recent = self.data_ppg[-200:]
        if np.max(recent) - np.min(recent) < 50:
            return False
        return True

    # =====================================================
    # 主循环
    # =====================================================
    def update_tick(self):
        if self.mode == self.MODE_SIM:
            self._update_sim_mode()
        elif self.mode == self.MODE_SERIAL:
            self._update_serial_mode()
        elif self.mode == self.MODE_UDP:
            self._update_udp_mode()
        else:
            return

        # 计算风险评分
        self._update_risk()

    # ----------- UDP 模式（ESP32-C3 WiFi 广播）-----------
    def _init_udp(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind(('0.0.0.0', self.UDP_PORT))
            s.setblocking(False)
            self.udp_sock = s
            self.mode = self.MODE_UDP
            self.status_pill.set_state('live', f'信号源：UDP :{self.UDP_PORT}')
            self.hand_pill.set_state('sim', '手部检测：等待信号')
        except Exception as e:
            self.udp_sock = None
            # 静默失败，回退到原有 serial/sim 自动连接逻辑
            print(f'[UDP] bind failed: {e}')

    def _update_udp_mode(self):
        lines = []
        try:
            while True:
                data, _ = self.udp_sock.recvfrom(512)
                lines.append(data.decode('utf-8', errors='ignore').strip())
        except BlockingIOError:
            pass
        except Exception:
            pass

        latest_ir = None
        for line in lines:
            if not line:
                continue
            m = re.search(r'ir\s*=\s*(\d+)', line)
            if m:
                ir_val = int(m.group(1))
                if ir_val > 0:
                    self.debug_raw_ir.append(ir_val)
                    latest_ir = ir_val

        if latest_ir is not None:
            ir_val = latest_ir
            if self.last_ir_val is not None:
                diff = abs(ir_val - self.last_ir_val)
                self.ir_diff_hist.append(diff)
                if len(self.ir_diff_hist) > 120:
                    self.ir_diff_hist = self.ir_diff_hist[-120:]
                med_diff = (np.median(self.ir_diff_hist)
                            if len(self.ir_diff_hist) > 10 else 300)
                jump_th = max(1200, 8 * med_diff)
                if diff > jump_th:
                    ir_val = self.last_ir_val
                    self.debug_dropped += 1
            self.data_ppg = np.roll(self.data_ppg, -1)
            self.data_ppg[-1] = ir_val
            self.last_ir_val = ir_val
            self.debug_used_ir.append(ir_val)
        else:
            self.debug_empty_ticks += 1
            if self.last_ir_val is not None:
                self.data_ppg = np.roll(self.data_ppg, -1)
                self.data_ppg[-1] = self.last_ir_val

        now_t = time.time()
        if now_t - self.debug_tick_t >= 1.0:
            self._flush_debug_panel()
            self.debug_tick_t = now_t

        # Serial mode trusts the board's finger flag. The amplitude heuristic
        # can briefly misclassify real low-motion PPG and clear the plot.
        self.hand_present = bool(self.serial_finger_present)
        if not self.hand_present:
            self.hand_pill.set_state('offline', '手部检测：未佩戴')
            self._hold_waveforms()
            return
        else:
            self.hand_pill.set_state('live', '手部检测：佩戴中')

        # 复用 PPG → HR/HRV/ECG 处理：直接调用 serial 模式的后半段
        self._process_ppg_signal()

    # ----------- 模拟模式（还原原行为）-----------
    def _update_sim_mode(self):
        for _ in range(SAMPLES_PER_TICK):
            ir_val = self.sim.gen_ppg_full()
            self.data_ppg = np.roll(self.data_ppg, -1)
            self.data_ppg[-1] = int(ir_val)
            hr = self.sim.gen_hr()
            self.data_hr.append(hr)
            if len(self.data_hr) > 200:
                self.data_hr.pop(0)
            hrv = self.sim.gen_hrv()
            self.data_hrv.append(hrv)
            if len(self.data_hrv) > 200:
                self.data_hrv.pop(0)
            ecg = self.sim.gen_ecg()
            self.data_ecg.append(ecg)
            if len(self.data_ecg) > 800:
                self.data_ecg.pop(0)

        # 显示波形（与 PPG_Monitor 真机链一致）
        if np.count_nonzero(self.data_ppg) > 20:
            display, _ = ppg_signal.compute_ppg_display(self.data_ppg, fs=FS, buf_len=len(self.data_ppg))
            self.curve_ppg.setData(display)
        self.curve_hr.setData(self.data_hr)
        self.curve_hrv.setData(self.data_hrv)
        self.curve_ecg.setData(self.data_ecg)

        # HRV 指标（原始模拟逻辑）
        self.right_update_counter += 1
        t = self.right_update_counter / 100.0
        sdnn = 40 + 15 * np.sin(t) + np.random.normal(0, 2)
        rmssd = 15 + 10 * np.sin(t + 1) + np.random.normal(0, 1.5)
        lf_hf = 1.5 + 1.0 * np.sin(t + 2) + np.random.normal(0, 0.3)

        self.current_hr = float(self.data_hr[-1]) if self.data_hr else 0.0
        self.current_sdnn = float(sdnn)
        self.current_rmssd = float(rmssd)
        self.current_lf_hf = float(lf_hf)

        self._refresh_metrics_panel()

        self.sdnn_trend_data.append(sdnn)
        if len(self.sdnn_trend_data) > 1000:
            self.sdnn_trend_data = self.sdnn_trend_data[-1000:]
        x_vals = np.arange(len(self.sdnn_trend_data)) * 0.1
        self.trend_curve.setData(x_vals, self.sdnn_trend_data)

    # ----------- 真机模式 -----------
    def _update_serial_mode(self):
        # 第一步：读串口（保留原逻辑）
        try:
            latest_ir = None
            latest_finger = None
            latest_bpm = None
            latest_spo2 = None
            latest_avg = None
            reader = getattr(self, 'serial_reader', None)
            lines = reader.get_lines(80) if reader is not None else []
            for line in lines:
                if not line:
                    continue
                ir_val = None
                if "[DATA]" in line:
                    m = re.search(r"ir\s*=\s*(\d+)", line)
                    if m:
                        ir_val = int(m.group(1))
                    m = re.search(r"\bfinger\s*=\s*(-?\d+)", line)
                    if m:
                        latest_finger = int(m.group(1)) == 1
                    m = re.search(r"\bbpm\s*=\s*(-?\d+)", line)
                    if m:
                        latest_bpm = int(m.group(1))
                    m = re.search(r"\bspo2\s*=\s*(-?\d+)", line)
                    if m:
                        latest_spo2 = int(m.group(1))
                    m = re.search(r"\bavg\s*=\s*(-?\d+)", line)
                    if m:
                        latest_avg = int(m.group(1))
                elif "," in line:
                    m = re.search(r"(-?\d+)", line.split(',')[0])
                    if m:
                        ir_val = int(m.group(1))
                else:
                    m = re.search(r"(-?\d+)", line)
                    if m:
                        ir_val = int(m.group(1))
                if ir_val is None or ir_val <= 0:
                    continue
                self.debug_raw_ir.append(ir_val)
                latest_ir = ir_val

            if latest_ir is not None:
                self.last_data_ts = time.time()
                ir_val = latest_ir
                # 抗跳变中位数过滤（保留原逻辑）
                if self.last_ir_val is not None:
                    diff = abs(ir_val - self.last_ir_val)
                    self.ir_diff_hist.append(diff)
                    if len(self.ir_diff_hist) > 120:
                        self.ir_diff_hist = self.ir_diff_hist[-120:]
                    med_diff = (np.median(self.ir_diff_hist)
                                if len(self.ir_diff_hist) > 10 else 300)
                    jump_th = max(1200, 8 * med_diff)
                    if diff > jump_th:
                        ir_val = self.last_ir_val
                        self.debug_dropped += 1
                self.data_ppg = np.roll(self.data_ppg, -1)
                self.data_ppg[-1] = ir_val
                self.last_ir_val = ir_val
                self.debug_used_ir.append(ir_val)
                self._pending_remote_ir.append(int(ir_val))
                if len(self._pending_remote_ir) > 400:
                    self._pending_remote_ir = self._pending_remote_ir[-400:]
            else:
                self.debug_empty_ticks += 1
                if self.last_ir_val is not None:
                    self.data_ppg = np.roll(self.data_ppg, -1)
                    self.data_ppg[-1] = self.last_ir_val

            if latest_finger is not None:
                self.serial_finger_present = bool(latest_finger)
            if latest_bpm is not None and 30 <= latest_bpm <= 220:
                self.serial_last_bpm = float(latest_bpm)
            if latest_spo2 is not None and 0 <= latest_spo2 <= 100:
                self.serial_last_spo2 = float(latest_spo2)
            if latest_avg is not None and latest_avg > 0:
                self.serial_last_avg = float(latest_avg)
            self._push_remote_wave()

            # 每秒打调试日志（输出到 UI 调试面板而不是 stdout）
            now_t = time.time()
            if now_t - self.debug_tick_t >= 1.0:
                self._flush_debug_panel()
                self.debug_tick_t = now_t

        except Exception as e:
            self.log(f'串口错误: {e}')

        # Serial mode trusts the board finger flag. Do not clear plots because
        # of short low-amplitude segments.
        self.hand_present = bool(self.serial_finger_present)
        if not self.hand_present:
            self.hand_pill.set_state('offline', '手部检测：未佩戴')
            self._hold_waveforms()
            return
        else:
            self.hand_pill.set_state('live', '手部检测：佩戴中')

        self._process_ppg_signal()

    def _process_ppg_signal(self):
        """与 PPG_Monitor.py 真机链一致：带通、显示缩放、寻峰、SDNN/RMSSD。"""
        valid_count = int(np.count_nonzero(self.data_ppg))
        if valid_count > 120:
            filtered_ppg, filtered_tail = ppg_signal.filter_ppg_tail(
                self.data_ppg, fs=FS, tail_n=400
            )
            filtered_ppg_display = ppg_signal.scale_ppg_for_display(
                filtered_ppg, filtered_tail, target_amp=1800.0
            )
            self.curve_ppg.setData(filtered_ppg_display)

            # 与 PPG_Monitor.py 一致：整段 filtered_ppg 寻峰 + 滚动 RR 缓冲（≥5 才出 SDNN）
            peaks, _ = find_peaks(
                filtered_ppg,
                height=np.mean(filtered_ppg) + np.std(filtered_ppg) * 0.4,
                distance=int(FS * 0.25),
            )
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / FS
                rr_intervals = rr_intervals[
                    (rr_intervals > 0.3) & (rr_intervals < 1.5)
                ]
                if len(rr_intervals) > 0:
                    # —— 以下 HR/HRV 波形逻辑保持原 1.py，勿改 ——
                    hr = 60 / np.mean(rr_intervals)
                    hr = float(np.clip(hr, 50, 100))
                    hr += float(np.random.normal(0, 1.5))

                    self.data_hr.append(hr)
                    if len(self.data_hr) > 200:
                        self.data_hr.pop(0)
                    self.curve_hr.setData(self.data_hr)
                    self.current_hr = float(hr)

                    self.rr_intervals_buffer.extend(rr_intervals.tolist())
                    if len(self.rr_intervals_buffer) > 8:
                        self.rr_intervals_buffer = self.rr_intervals_buffer[-8:]

                    hrv = float(np.std(self.rr_intervals_buffer) * 1000)
                    hrv = float(np.clip(hrv, 20, 100))
                    hrv_value = hrv + float(np.random.normal(0, 1.0))
                    if len(self.hrv_buffer) > 0:
                        hrv_value += (self.hrv_buffer[-1] - hrv_value) * 0.08

                    self.hrv_buffer.append(hrv_value)
                    if len(self.hrv_buffer) > 8:
                        self.hrv_buffer = self.hrv_buffer[-8:]

                    self.hrv_target = float(np.mean(self.hrv_buffer))

                    # 仅右侧指标：滚动 RR ≥5 时更新 SDNN/RMSSD/LF-HF
                    if len(self.rr_intervals_buffer) >= 5:
                        sdnn, rmssd, lf_hf = calculate_hrv_indices(
                            self.rr_intervals_buffer
                        )
                        if sdnn > 0 and rmssd > 0 and lf_hf > 0:
                            self.current_sdnn = float(sdnn)
                            self.current_rmssd = float(rmssd)
                            self.current_lf_hf = float(lf_hf)

                    self._push_remote_hr()

        # B) 固定频率重采样显示：即使当前帧无新RR，也平滑逼近目标值
        # 仅增强左下 HRV 曲线的显示层动态，不改变右侧 SDNN/RMSSD/LF-HF 真实指标。
        if self.hrv_target is not None:
            hrv_alpha = 0.18
            if self.hrv_display_ema is None:
                self.hrv_display_ema = float(self.hrv_target)
            else:
                self.hrv_display_ema = (
                    self.hrv_display_ema * (1.0 - hrv_alpha)
                    + float(self.hrv_target) * hrv_alpha
                )

            display_hrv = float(self.hrv_display_ema)
            if len(self.data_hrv) >= 24:
                recent_hrv_span = float(np.max(self.data_hrv[-24:]) - np.min(self.data_hrv[-24:]))
            else:
                recent_hrv_span = 999.0
            if recent_hrv_span < 2.5:
                display_phase = self.counter * 0.11
                display_hrv += float(1.8 * np.sin(display_phase) + np.random.normal(0, 0.35))
            display_hrv = float(np.clip(display_hrv, 0, 120))

            self.data_hrv.append(display_hrv)
            if len(self.data_hrv) > 200:
                self.data_hrv.pop(0)
            self.curve_hrv.setData(self.data_hrv)

            # SDNN 趋势仍只跟随真实 current_sdnn，同步刷新但不加入显示层动态。
            if self.current_sdnn > 0:
                sdnn_alpha = 0.10
                if self.sdnn_display_ema is None:
                    self.sdnn_display_ema = float(self.current_sdnn)
                else:
                    self.sdnn_display_ema = (
                        self.sdnn_display_ema * (1.0 - sdnn_alpha)
                        + float(self.current_sdnn) * sdnn_alpha
                    )
                self.sdnn_trend_data.append(float(self.sdnn_display_ema))
                if len(self.sdnn_trend_data) > 1000:
                    self.sdnn_trend_data = self.sdnn_trend_data[-1000:]
                x_vals = np.arange(len(self.sdnn_trend_data)) * 0.1
                self.trend_curve.setData(x_vals, self.sdnn_trend_data)
            self._refresh_metrics_panel()

        # ECG 模拟（保留原逻辑）
        self.counter += 1
        t = self.counter * 0.01
        ecg_value = 0
        phase = t % 1
        if 0.1 < phase < 0.2:
            ecg_value = 0.1 * np.sin((phase - 0.1) * 30 * np.pi)
        elif 0.2 < phase < 0.25:
            ecg_value = 0.8 * np.sin((phase - 0.2) * 100 * np.pi)
        elif 0.25 < phase < 0.3:
            ecg_value = -0.4 * np.sin((phase - 0.25) * 80 * np.pi)
        elif 0.3 < phase < 0.5:
            ecg_value = 0.3 * np.sin((phase - 0.4) * 40 * np.pi)
        ecg_value += np.random.randn() * 0.03
        self.data_ecg.append(ecg_value)
        if len(self.data_ecg) > 800:
            self.data_ecg.pop(0)
        self.curve_ecg.setData(self.data_ecg)

    def _hold_waveforms(self):
        """Keep the last real curves visible during short no-finger intervals."""
        if np.count_nonzero(self.data_ppg) > 20:
            display, _ = ppg_signal.compute_ppg_display(
                self.data_ppg, fs=FS, buf_len=len(self.data_ppg)
            )
            self.curve_ppg.setData(display)
        self.curve_hr.setData(self.data_hr)
        self.curve_hrv.setData(self.data_hrv)
        self.curve_ecg.setData(self.data_ecg)

    def _render_offline_waveforms(self):
        """手离开时把波形清零"""
        self.data_ppg[:] = 0
        self.data_hr = [0.0] * 200
        self.data_hrv = [0.0] * 200
        self.data_ecg = [0.0] * 800
        self.rr_intervals_buffer.clear()
        self.hrv_buffer.clear()
        self.hrv_metric_smooth = {"sdnn": None, "rmssd": None, "lfhf": None}
        self.hrv_target = None
        self.hrv_display_ema = None
        self.sdnn_trend_data.clear()
        self.curve_ppg.setData(np.zeros(800))
        self.curve_hr.setData(self.data_hr)
        self.curve_hrv.setData(self.data_hrv)
        self.curve_ecg.setData(self.data_ecg)
        self.current_hr = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_lf_hf = 0.0
        self.card_hr.set_value('--', C_DIM)
        self.card_sdnn.set_value('--', C_DIM)
        self.row_sdnn.update('--', '无信号', C_DIM)
        self.row_rmssd.update('--', '无信号', C_DIM)
        self.row_lf_hf.update('--', '无信号', C_DIM)
        self.trend_curve.setData([], [])

    def _push_remote_hr(self):
        if self.current_hr <= 0 or self.current_sdnn <= 0:
            return
        payload = {
            "device_id": DEVICE_ID,
            "bpm": round(float(self.current_hr), 2),
            "sdnn": round(float(self.current_sdnn), 2),
            "rmssd": round(float(self.current_rmssd), 2),
            "lfhf": round(float(self.current_lf_hf), 4),
            "ir_value": None if self.last_ir_val is None else int(self.last_ir_val),
            "hand_present": int(bool(self.hand_present)),
        }
        if payload == self._last_remote_payload:
            return
        self._last_remote_payload = dict(payload)
        self.remote_poster.submit_latest(payload)

    def _push_remote_wave(self):
        now = time.time()
        if now - self._last_wave_push_ts < 0.08:
            return
        if not self._pending_remote_ir:
            return
        payload = {
            "device_id": DEVICE_ID,
            "bpm": round(float(self.current_hr), 2) if self.current_hr > 0 else None,
            "sdnn": round(float(self.current_sdnn), 2) if self.current_sdnn > 0 else None,
            "rmssd": round(float(self.current_rmssd), 2) if self.current_rmssd > 0 else None,
            "lfhf": round(float(self.current_lf_hf), 4) if self.current_lf_hf > 0 else None,
            "ir_value": int(self._pending_remote_ir[-1]),
            "ir_values": list(self._pending_remote_ir[-80:]),
            "hr_display": {
                "ppg": self._safe_float_list(self.curve_ppg.getData()[1], 800),
                "ecg": self._safe_float_list(self.data_ecg, 800),
                "hr": self._safe_float_list(self.data_hr, 200),
                "hrv": self._safe_float_list(self.data_hrv, 200),
                "sdnn_trend": self._safe_float_list(self.sdnn_trend_data, 1000),
            },
            "hand_present": int(bool(self.hand_present)),
        }
        self._pending_remote_ir.clear()
        self._last_wave_push_ts = now
        self.remote_poster.submit_latest(payload)

    @staticmethod
    def _safe_float_list(values, limit):
        out = []
        try:
            seq = list(values)[-limit:]
        except Exception:
            return out
        for v in seq:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
        return out

    def _refresh_metrics_panel(self):
        # 大数值卡
        if self.current_hr > 0:
            hr_color = C_HR if 60 <= self.current_hr <= 90 else C_WARN
            self.card_hr.set_value(f'{self.current_hr:.0f}', hr_color)
        if self.current_sdnn > 0:
            sd_color = C_OK if self.current_sdnn > 50 else C_WARN
            self.card_sdnn.set_value(f'{self.current_sdnn:.1f}', sd_color)
        else:
            self.card_sdnn.set_value('--', C_DIM)

        # 指标行
        if self.current_sdnn <= 0:
            self.row_sdnn.update('--', '等待 RR', C_DIM)
            self.row_rmssd.update('--', '等待 RR', C_DIM)
            self.row_lf_hf.update('--', '等待 RR', C_DIM)
        else:
            if self.current_sdnn > 50:
                self.row_sdnn.update(f'{self.current_sdnn:.1f} ms', '✓ 正常', C_OK)
            else:
                self.row_sdnn.update(f'{self.current_sdnn:.1f} ms', '⚠ 偏低', C_WARN)
            if 15 <= self.current_rmssd <= 39:
                self.row_rmssd.update(f'{self.current_rmssd:.1f} ms', '✓ 正常', C_OK)
            else:
                self.row_rmssd.update(f'{self.current_rmssd:.1f} ms', '⚠ 异常', C_WARN)
            if 1 <= self.current_lf_hf <= 2:
                self.row_lf_hf.update(f'{self.current_lf_hf:.2f}', '✓ 正常', C_OK)
            elif self.current_lf_hf > 2:
                self.row_lf_hf.update(f'{self.current_lf_hf:.2f}', '⚠ 偏高', C_RISK)
            else:
                self.row_lf_hf.update(f'{self.current_lf_hf:.2f}', '⚠ 偏低', C_WARN)

    def _update_risk(self):
        """生理风险评分：HRV 降低 + HR 异常 + LF/HF 偏高 → 情绪压力高"""
        if self.current_sdnn <= 0 or self.current_hr <= 0:
            return
        # 各分量归一化到 0-1
        sdnn_risk = max(0.0, min(1.0, (50 - self.current_sdnn) / 50))
        hr_risk = 0.0
        if self.current_hr < 60:
            hr_risk = (60 - self.current_hr) / 30
        elif self.current_hr > 90:
            hr_risk = (self.current_hr - 90) / 30
        hr_risk = max(0.0, min(1.0, hr_risk))
        lfhf_risk = max(0.0, min(1.0, (self.current_lf_hf - 2.0) / 2.0))

        risk = 0.5 * sdnn_risk + 0.25 * hr_risk + 0.25 * lfhf_risk
        risk = max(0.0, min(1.0, risk))

        self.risk_value.setText(f'{risk:.2f}')
        if risk > 0.66:
            c, tag = C_RISK, '高 · HIGH'
        elif risk > 0.33:
            c, tag = C_WARN, '中 · MEDIUM'
        else:
            c, tag = C_OK, '低 · LOW'
        self.risk_value.setStyleSheet(
            f'color:{c}; font-size:36px; font-weight:bold; '
            f'background:transparent;')
        self.risk_tag.setText(tag)
        self.risk_tag.setStyleSheet(
            f'color:{c}; font-size:11px; letter-spacing:2px;')

    def _flush_debug_panel(self):
        raw_count = len(self.debug_raw_ir)
        used_count = len(self.debug_used_ir)
        if raw_count > 0:
            raw_min = int(np.min(self.debug_raw_ir))
            raw_max = int(np.max(self.debug_raw_ir))
            raw_med = int(np.median(self.debug_raw_ir))
        else:
            raw_min = raw_max = raw_med = -1
        if used_count > 1:
            used_diff = np.diff(np.array(self.debug_used_ir, dtype=np.int64))
            diff_med = float(np.median(np.abs(used_diff)))
            diff_p95 = float(np.percentile(np.abs(used_diff), 95))
        else:
            diff_med = diff_p95 = 0.0
        msg = (f'raw_cnt={raw_count} used={used_count} '
               f'empty={self.debug_empty_ticks} drop={self.debug_dropped} '
               f'raw(min/med/max)={raw_min}/{raw_med}/{raw_max} '
               f'diff(med/p95)={diff_med:.1f}/{diff_p95:.1f}')
        ts = time.strftime('%H:%M:%S')
        self.debug_text.append(
            f'<span style="color:{C_DIM}">[{ts}]</span> '
            f'<span style="color:{C_OK}">{msg}</span>')
        self.debug_raw_ir.clear()
        self.debug_used_ir.clear()
        self.debug_dropped = 0
        self.debug_empty_ticks = 0

    def log(self, message):
        ts = time.strftime('%H:%M:%S')
        self.log_text.append(
            f'<span style="color:{C_DIM}">[{ts}]</span> {message}')

    def closeEvent(self, event):
        try:
            self.remote_poster.stop()
        except Exception:
            pass
        self.disconnect_serial()
        event.accept()


# ============================================================
# 入口
# ============================================================
def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = HRMonitorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
