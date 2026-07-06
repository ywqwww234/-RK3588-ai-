import json
import re
import time
from collections import deque
from datetime import datetime

import requests

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

# A 电脑 -> B 电脑（请按实际情况修改）
B_HOST = "10.36.28.119"
B_PORT = 5001
POST_URL = f"http://{B_HOST}:{B_PORT}/a2b/telemetry"

# 发送频率（秒）
INTERVAL_SEC = 1.0

# 串口参数
BAUDRATE = 115200
SERIAL_TIMEOUT_SEC = 0.05

# 设备标识
DEVICE_ID = "a_host_01"


def auto_find_port():
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


class PhysioEstimator:
    """从串口IR数据估计 BPM，输出给 B 端。"""

    def __init__(self):
        self.ir_buf = deque(maxlen=300)
        self.ts_buf = deque(maxlen=300)
        self.last_bpm = None

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

    def ingest_line(self, line: str):
        ir = self._parse_ir(line)
        if ir is None or ir <= 0:
            return
        now = time.time()
        self.ir_buf.append(float(ir))
        self.ts_buf.append(now)

    def estimate_bpm(self):
        if len(self.ir_buf) < 80 or len(self.ts_buf) < 80:
            return self.last_bpm

        vals = list(self.ir_buf)
        tss = list(self.ts_buf)
        duration = tss[-1] - tss[0]
        if duration <= 1.0:
            return self.last_bpm

        mean_v = sum(vals) / len(vals)
        centered = [v - mean_v for v in vals]
        abs_dev = [abs(v) for v in centered]
        amp = sum(abs_dev) / max(1, len(abs_dev))
        th = amp * 0.9

        peaks = []
        for i in range(1, len(centered) - 1):
            if centered[i] > centered[i - 1] and centered[i] > centered[i + 1] and centered[i] > th:
                peaks.append(i)

        if len(peaks) < 3:
            return self.last_bpm

        rr = []
        for i in range(1, len(peaks)):
            dt = tss[peaks[i]] - tss[peaks[i - 1]]
            if 0.35 < dt < 1.6:
                rr.append(dt)

        if not rr:
            return self.last_bpm

        bpm = 60.0 / (sum(rr) / len(rr))
        bpm = max(45.0, min(120.0, bpm))
        self.last_bpm = round(bpm, 1)
        return self.last_bpm


def build_payload_from_real_data(bpm):
    return {
        "device_id": DEVICE_ID,
        "bpm": bpm,
        # 你接入真实脑电后，把这三项替换为真实值
        "attention": None,
        "meditation": None,
        "poor_signal": None,
    }


def main():
    print(f"[A端] 开始推送到: {POST_URL}")
    print("[A端] 按 Ctrl+C 停止")

    if serial is None:
        print("[A端] 未安装 pyserial，无法读取真实串口。请先: pip install pyserial")
        return

    port = auto_find_port()
    if not port:
        print("[A端] 未找到 ESP32 串口（USB Serial/CH340/ESP32）。")
        return

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=SERIAL_TIMEOUT_SEC)
        print(f"[A端] 串口已连接: {port}@{BAUDRATE}")
    except Exception as e:
        print(f"[A端] 串口打开失败: {e}")
        return

    estimator = PhysioEstimator()
    session = requests.Session()
    session.trust_env = False

    last_ok = False
    last_send_ts = 0.0

    try:
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        estimator.ingest_line(line)

                now_ts = time.time()
                if now_ts - last_send_ts >= INTERVAL_SEC:
                    bpm = estimator.estimate_bpm()
                    if bpm is not None:
                        payload = build_payload_from_real_data(bpm)
                        resp = session.post(POST_URL, json=payload, timeout=2)

                        now = datetime.now().strftime("%H:%M:%S")
                        if resp.status_code == 200:
                            result = {}
                            try:
                                result = resp.json()
                            except Exception:
                                pass

                            print(
                                f"[{now}] OK -> bpm={payload.get('bpm')} "
                                f"att={payload.get('attention')} med={payload.get('meditation')} "
                                f"resp={json.dumps(result, ensure_ascii=False)}"
                            )
                            last_ok = True
                        else:
                            print(f"[{now}] HTTP {resp.status_code}: {resp.text}")
                            last_ok = False
                    else:
                        now = datetime.now().strftime("%H:%M:%S")
                        print(f"[{now}] 串口数据不足，等待稳定后再发送...")

                    last_send_ts = now_ts

                time.sleep(0.01)

            except KeyboardInterrupt:
                print("\n[A端] 手动停止。")
                break
            except Exception as e:
                now = datetime.now().strftime("%H:%M:%S")
                if last_ok:
                    print(f"[{now}] 连接中断，重试中... err={e}")
                else:
                    print(f"[{now}] 发送失败: {e}")
                last_ok = False
                time.sleep(0.2)
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
