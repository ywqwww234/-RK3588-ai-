import socket
import re
import time
from collections import deque

UDP_IP = "192.168.66.137"
UDP_PORT = 5005
BUFFER_SIZE = 4096

# 兼容格式示例：
# "ir=12345,bpm=78,spo2=97"
# "[DATA] ir=12345 bpm=78"
PAT_IR = re.compile(r"\bir\s*=\s*([0-9]+)\b", re.IGNORECASE)
PAT_BPM = re.compile(r"\bbpm\s*=\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
PAT_SPO2 = re.compile(r"\bspo2\s*=\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

def jump_filter(new_val, last_val, max_delta=50000):
    """简单跳变滤波，防止异常突刺。"""
    if last_val is None:
        return new_val
    if abs(new_val - last_val) > max_delta:
        return last_val
    return new_val

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    print(f"[B端] UDP监听中: {UDP_IP}:{UDP_PORT}")

    last_ir = None
    ir_window = deque(maxlen=20)
    last_recv_ts = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            last_recv_ts = time.time()

            msg = data.decode("utf-8", errors="ignore").strip()

            m_ir = PAT_IR.search(msg)
            m_bpm = PAT_BPM.search(msg)
            m_spo2 = PAT_SPO2.search(msg)

            ir = int(m_ir.group(1)) if m_ir else None
            bpm = float(m_bpm.group(1)) if m_bpm else None
            spo2 = float(m_spo2.group(1)) if m_spo2 else None

            if ir is not None:
                ir = jump_filter(ir, last_ir, max_delta=50000)
                last_ir = ir
                ir_window.append(ir)

            print(
                f"[RECV] from={addr[0]}:{addr[1]} "
                f"ir={ir} bpm={bpm} spo2={spo2} raw='{msg}'"
            )

            # 这里可以接你学生端/家长端的更新函数
            # update_ui(ir=ir, bpm=bpm, spo2=spo2)

        except socket.timeout:
            # 超时仅用于提示链路状态
            if time.time() - last_recv_ts > 3:
                print("[B端] 3秒未收到数据，等待中...")
                last_recv_ts = time.time()
        except KeyboardInterrupt:
            print("\n[B端] 手动停止。")
            break
        except Exception as e:
            print(f"[B端] 异常: {e}")

if __name__ == "__main__":
    main()
