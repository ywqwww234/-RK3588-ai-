import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import serial
from PyQt5.QtCore import QThread, pyqtSignal


@dataclass
class EEGState:
    poor_signal: Optional[int] = None
    attention: Optional[int] = None
    meditation: Optional[int] = None
    blink_strength: Optional[int] = None
    raw_value: Optional[int] = None
    eeg_power: Optional[List[int]] = None
    last_update_ts: float = field(default_factory=time.time)


class ThinkGearParser:
    HEADER = 0xAA
    MAX_PAYLOAD_LEN = 169

    @staticmethod
    def _checksum(payload: bytes) -> int:
        return 0xFF - (sum(payload) & 0xFF)

    def parse_payload(self, payload: bytes) -> Dict:
        i = 0
        out: Dict = {}

        while i < len(payload):
            code = payload[i]
            i += 1

            if code in (0x02, 0x04, 0x05, 0x16):
                if i < len(payload):
                    val = payload[i]
                    i += 1
                    if code == 0x02:
                        out["poor_signal"] = val
                    elif code == 0x04:
                        out["attention"] = val
                    elif code == 0x05:
                        out["meditation"] = val
                    elif code == 0x16:
                        out["blink_strength"] = val
                continue

            if code == 0x80:
                if i >= len(payload):
                    break
                ln = payload[i]
                i += 1
                if ln == 2 and i + 1 < len(payload):
                    raw = int.from_bytes(payload[i:i + 2], byteorder="big", signed=True)
                    out["raw_value"] = raw
                    i += 2
                else:
                    i += ln
                continue

            if code == 0x83:
                if i >= len(payload):
                    break
                ln = payload[i]
                i += 1
                if ln == 24 and i + 23 < len(payload):
                    bands = []
                    for k in range(8):
                        b = i + 3 * k
                        v = (payload[b] << 16) | (payload[b + 1] << 8) | payload[b + 2]
                        bands.append(v)
                    out["eeg_power"] = bands
                    i += 24
                else:
                    i += ln
                continue

            # 未识别字段：保守跳过1字节，避免陷入死循环
            if i < len(payload):
                i += 1

        return out


class EEGSerialReader:
    def __init__(self, port: str, baudrate: int = 57600, timeout: float = 0.2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.parser = ThinkGearParser()
        self.state = EEGState()


class EEGDataThread(QThread):
    eeg_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, source: str = "mock", port: str = "COM6", baudrate: int = 57600, interval_ms: int = 1000):
        super().__init__()
        self.source = source
        self.port = port
        self.baudrate = baudrate
        self.interval_ms = max(100, int(interval_ms))
        self._running = True
        self.reader: Optional[EEGSerialReader] = None

    def stop(self):
        self._running = False

    def _mock_packet(self):
        attention = random.randint(35, 80)
        meditation = random.randint(30, 75)
        poor_signal = random.choice([0, 0, 0, 20, 40])
        eeg_power = [
            random.randint(10000, 60000),
            random.randint(10000, 60000),
            random.randint(5000, 45000),
            random.randint(5000, 45000),
            random.randint(3000, 35000),
            random.randint(3000, 35000),
            random.randint(1000, 18000),
            random.randint(1000, 18000),
        ]
        return {
            "poor_signal": poor_signal,
            "attention": attention,
            "meditation": meditation,
            "raw_value": random.randint(-2048, 2048),
            "eeg_power": eeg_power,
        }

    def run(self):
        if self.source == "serial":
            try:
                self.reader = EEGSerialReader(self.port, self.baudrate)
                self.reader.open()
                self.status_signal.emit(f"> EEG串口已连接: {self.port}@{self.baudrate}")
            except Exception as e:
                self.status_signal.emit(f"> EEG串口连接失败，回退Mock: {e}")
                self.source = "mock"

        while self._running:
            try:
                if self.source == "serial" and self.reader is not None:
                    pkt = self.reader.read_packet()
                    if pkt:
                        self.eeg_signal.emit(pkt)
                else:
                    self.eeg_signal.emit(self._mock_packet())
                    self.msleep(self.interval_ms)
            except Exception as e:
                self.status_signal.emit(f"> EEG读取异常，回退Mock: {e}")
                self.source = "mock"

        if self.reader is not None:
            self.reader.close()

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _read_exact(self, n: int) -> Optional[bytes]:
        if self.ser is None:
            return None
        data = self.ser.read(n)
        if len(data) != n:
            return None
        return data

    def read_packet(self) -> Optional[Dict]:
        if self.ser is None:
            return None

        while True:
            b = self._read_exact(1)
            if not b:
                return None
            if b[0] != ThinkGearParser.HEADER:
                continue

            b2 = self._read_exact(1)
            if not b2:
                return None
            if b2[0] != ThinkGearParser.HEADER:
                continue

            plen_b = self._read_exact(1)
            if not plen_b:
                return None
            plen = plen_b[0]
            if plen > ThinkGearParser.MAX_PAYLOAD_LEN:
                continue

            payload = self._read_exact(plen)
            if payload is None:
                return None

            chk_b = self._read_exact(1)
            if not chk_b:
                return None

            calc = ThinkGearParser._checksum(payload)
            if chk_b[0] != calc:
                continue

            parsed = self.parser.parse_payload(payload)
            if parsed:
                self._update_state(parsed)
            return parsed

    def _update_state(self, parsed: Dict):
        for key, val in parsed.items():
            if hasattr(self.state, key):
                setattr(self.state, key, val)
        self.state.last_update_ts = time.time()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ThinkGear EEG 串口解析测试")
    parser.add_argument("--port", required=True, help="串口号，如 COM6")
    parser.add_argument("--baud", type=int, default=57600, help="波特率，默认57600")
    args = parser.parse_args()

    reader = EEGSerialReader(port=args.port, baudrate=args.baud)
    print(f"[EEG] opening {args.port} @ {args.baud}")
    reader.open()

    try:
        while True:
            pkt = reader.read_packet()
            if not pkt:
                continue
            fields = []
            for k in ["poor_signal", "attention", "meditation", "blink_strength", "raw_value", "eeg_power"]:
                if k in pkt:
                    fields.append(f"{k}={pkt[k]}")
            if fields:
                print("[EEG] " + " | ".join(fields))
    except KeyboardInterrupt:
        print("\n[EEG] stopped")
    finally:
        reader.close()
