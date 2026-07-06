"""
脑机接口 (BCI) 线程 - TGAM 蓝牙串口直驱
本地优先；失败自动降级到模拟数据；仍保留远端 RemoteEEGThread 作为兜底。
协议解析复用 D:\\G\\11.py 中已经在真机上验证过的 TGAMParser。
"""

import math
import random
import struct
import time
from collections import deque
from typing import Optional

import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal


SYNC = 0xAA
EXCODE = 0x55

CODE_POOR_SIGNAL = 0x02
CODE_HEART_RATE = 0x03
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_RAW_WAVE = 0x80
CODE_EEG_POWER = 0x81
CODE_ASIC_EEG_POWER = 0x83
CODE_RRINTERVAL = 0x86


class TGAMParser:
    """解析 TGAM 数据包，并维护最新脑电状态。"""
    """完整 TGAM 协议解析器（从 11.py 移植，已在真机验证）。"""

    def __init__(self):
        self.buffer = bytearray()
        self.raw_wave_history = deque(maxlen=15360)  # 30s @ 512Hz
        self.attention = 0
        self.meditation = 0
        self.poor_signal = 200
        self.heart_rate = 0
        self.rr_interval = 0
        self.eeg_power = {}
        self.last_update = time.time()
        self.packet_count = 0
        self.error_count = 0

    def parse_byte(self, byte):
        self.buffer.append(byte)
        while True:
            if len(self.buffer) < 2:
                break
            if not (self.buffer[0] == SYNC and self.buffer[1] == SYNC):
                del self.buffer[0]
                continue
            if len(self.buffer) < 3:
                break
            payload_length = self.buffer[2]
            if payload_length > 169:
                del self.buffer[0]
                self.error_count += 1
                continue
            packet_total_len = 3 + payload_length + 1
            if len(self.buffer) < packet_total_len:
                break
            payload = self.buffer[3:3 + payload_length]
            checksum_recv = self.buffer[3 + payload_length]
            checksum_calc = (~sum(payload) & 0xFF)
            if checksum_recv == checksum_calc:
                self._parse_payload(payload)
                self.last_update = time.time()
                self.packet_count += 1
            else:
                self.error_count += 1
            del self.buffer[:packet_total_len]
        if len(self.buffer) > 2048:
            self.buffer = bytearray()

    def _parse_payload(self, payload):
        i = 0
        while i < len(payload):
            while i < len(payload) and payload[i] == EXCODE:
                i += 1
            if i >= len(payload):
                break
            code = payload[i]
            i += 1
            if code >= 0x80:
                if i >= len(payload):
                    break
                value_length = payload[i]
                i += 1
            else:
                value_length = 1
            if i + value_length > len(payload):
                break
            value = payload[i:i + value_length]
            i += value_length
            self._parse_data(code, value)

    def _parse_data(self, code, value):
        try:
            if code == CODE_POOR_SIGNAL:
                self.poor_signal = value[0]
            elif code == CODE_HEART_RATE:
                self.heart_rate = value[0]
            elif code == CODE_ATTENTION:
                self.attention = value[0]
            elif code == CODE_MEDITATION:
                self.meditation = value[0]
            elif code == CODE_RAW_WAVE and len(value) == 2:
                raw_value = struct.unpack('>h', bytes(value))[0]
                self.raw_wave_history.append(raw_value)
            elif code == CODE_ASIC_EEG_POWER and len(value) == 24:
                self.eeg_power = {
                    'delta':      struct.unpack('>I', b'\x00' + value[0:3])[0],
                    'theta':      struct.unpack('>I', b'\x00' + value[3:6])[0],
                    'low_alpha':  struct.unpack('>I', b'\x00' + value[6:9])[0],
                    'high_alpha': struct.unpack('>I', b'\x00' + value[9:12])[0],
                    'low_beta':   struct.unpack('>I', b'\x00' + value[12:15])[0],
                    'high_beta':  struct.unpack('>I', b'\x00' + value[15:18])[0],
                    'low_gamma':  struct.unpack('>I', b'\x00' + value[18:21])[0],
                    'mid_gamma':  struct.unpack('>I', b'\x00' + value[21:24])[0],
                }
            elif code == CODE_RRINTERVAL and len(value) == 2:
                self.rr_interval = struct.unpack('>H', bytes(value))[0]
        except Exception:
            pass


class BCIThread(QThread):
    """脑电采集主线程，优先本地串口，失败时自动降级到模拟模式。"""
    """脑机接口主线程：本地串口优先，失败自动模拟。"""

    eeg_signal = pyqtSignal(dict)          # {attention, meditation, poor_signal, eeg_power, raw_value}
    raw_wave_signal = pyqtSignal(list)     # 最近一段 RAW 波（int 列表）
    status_signal = pyqtSignal(str)        # 状态日志
    source_signal = pyqtSignal(str)        # 'serial' / 'sim' / 'offline'

    PORT_KEYWORDS = ('bluetooth', 'hc-05', 'hc05', 'tgam', 'ch340',
                     'cp210', 'usb-serial', 'usb serial', 'silabs', 'ftdi')

    def __init__(self, port: Optional[str] = None, baud: int = 57600,
                 prefer_serial: bool = True, push_interval_ms: int = 800):
        super().__init__()
        self.port_hint = port
        self.baud = int(baud)
        self.prefer_serial = bool(prefer_serial)
        self.push_interval_ms = max(100, int(push_interval_ms))
        self._running = True
        self._ser = None
        self.parser = TGAMParser()
        self.mode = 'offline'

        self._sim_phase = random.random() * 100.0
        self._sim_att = 50.0
        self._sim_med = 50.0
        self._sim_last_band_t = 0.0
        self._sim_noise_burst_t = 0.0

    def stop(self):
        self._running = False

    # -------- 串口 --------
    def _guess_port(self):
        if self.port_hint:
            return self.port_hint
        for p in serial.tools.list_ports.comports():
            desc = (p.description or '').lower()
            if any(k in desc for k in self.PORT_KEYWORDS):
                return p.device
        return None

    def _open_serial(self):
        port = self._guess_port()
        if not port:
            return False
        try:
            self._ser = serial.Serial(
                port=port, baudrate=self.baud, timeout=0,
                bytesize=8, parity='N', stopbits=1,
                xonxoff=False, rtscts=False, dsrdtr=False, exclusive=False,
            )
            try:
                self._ser.setDTR(False)
                self._ser.setRTS(False)
            except Exception:
                pass
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self.status_signal.emit(f"> [BCI] TGAM 串口已连接: {port}@{self.baud}")
            self.source_signal.emit('serial')
            self.mode = 'serial'
            return True
        except Exception as e:
            self.status_signal.emit(f"> [BCI] 串口打开失败: {e}")
            self._ser = None
            return False

    def _close_serial(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    # -------- 模拟数据（符合 TGAM 真实节拍）--------
    def _sim_tick(self):
        t = time.time()
        self._sim_phase += 0.02
        for k in range(10):
            sub_t = t + k * 0.002
            sample = (
                7000 * math.sin(2 * math.pi * 2 * sub_t) +
                5000 * math.sin(2 * math.pi * 6 * sub_t) +
                6000 * math.sin(2 * math.pi * 10 * sub_t) +
                3500 * math.sin(2 * math.pi * 18 * sub_t) +
                random.randint(-1500, 1500)
            )
            self.parser.raw_wave_history.append(int(sample))

        self._sim_att = max(25.0, min(95.0,
            self._sim_att + random.uniform(-2.5, 2.5) +
            4.0 * math.sin(self._sim_phase * 0.5)))
        self._sim_med = max(25.0, min(95.0,
            self._sim_med + random.uniform(-2.5, 2.5) +
            4.0 * math.cos(self._sim_phase * 0.4)))
        self.parser.attention = int(self._sim_att)
        self.parser.meditation = int(self._sim_med)

        if t - self._sim_noise_burst_t > 12 and random.random() < 0.005:
            self._sim_noise_burst_t = t
        if t - self._sim_noise_burst_t < 2.0:
            self.parser.poor_signal = random.choice([25, 26, 51])
        else:
            self.parser.poor_signal = 0

        if t - self._sim_last_band_t > 1.0:
            self._sim_last_band_t = t
            base = self._sim_phase
            self.parser.eeg_power = {
                'delta':      int(1800000 + 600000 * abs(math.sin(base * 0.13))),
                'theta':      int(400000  + 200000 * abs(math.sin(base * 0.21))),
                'low_alpha':  int(80000   + 50000  * abs(math.sin(base * 0.33))),
                'high_alpha': int(60000   + 40000  * abs(math.sin(base * 0.41))),
                'low_beta':   int(40000   + 25000  * abs(math.sin(base * 0.53))),
                'high_beta':  int(35000   + 20000  * abs(math.sin(base * 0.61))),
                'low_gamma':  int(20000   + 12000  * abs(math.sin(base * 0.73))),
                'mid_gamma':  int(15000   + 8000   * abs(math.sin(base * 0.81))),
            }
            self.parser.last_update = t
            self.parser.packet_count += 1

    # -------- 输出 --------
    def _emit_snapshot(self):
        pkt = {
            "attention": int(self.parser.attention),
            "meditation": int(self.parser.meditation),
            "poor_signal": int(self.parser.poor_signal),
            "eeg_power": dict(self.parser.eeg_power) if self.parser.eeg_power else None,
            "raw_value": self.parser.raw_wave_history[-1] if self.parser.raw_wave_history else None,
            "source": self.mode,
        }
        self.eeg_signal.emit(pkt)
        if self.parser.raw_wave_history:
            self.raw_wave_signal.emit(list(self.parser.raw_wave_history))

    def run(self):
        connected = False
        if self.prefer_serial:
            connected = self._open_serial()
        if not connected:
            self.mode = 'sim'
            self.source_signal.emit('sim')
            self.status_signal.emit("> [BCI] 未识别 TGAM 设备，本地模拟模式启动。")

        last_push = 0.0
        last_serial_data_t = time.time()

        while self._running:
            now = time.time()

            if self.mode == 'serial' and self._ser is not None:
                try:
                    waiting = self._ser.in_waiting
                    if waiting > 0:
                        data = self._ser.read(waiting)
                        for byte in data:
                            self.parser.parse_byte(byte)
                        if self.parser.packet_count > 0:
                            last_serial_data_t = now
                    # 看门狗：5 秒无包则降级
                    if now - last_serial_data_t > 5.0:
                        self.status_signal.emit("> [BCI] TGAM 5 秒无数据，降级为模拟模式。")
                        self._close_serial()
                        self.mode = 'sim'
                        self.source_signal.emit('sim')
                except Exception as e:
                    self.status_signal.emit(f"> [BCI] 串口异常，降级为模拟模式: {e}")
                    self._close_serial()
                    self.mode = 'sim'
                    self.source_signal.emit('sim')
            else:
                self._sim_tick()

            if (now - last_push) * 1000.0 >= self.push_interval_ms:
                self._emit_snapshot()
                last_push = now

            self.msleep(20)

        self._close_serial()


if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    th = BCIThread(prefer_serial=True)
    th.status_signal.connect(lambda s: print(s))
    th.eeg_signal.connect(lambda d: print("[EEG]", d))
    th.start()
    sys.exit(app.exec_())
