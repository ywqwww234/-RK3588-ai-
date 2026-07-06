#!/usr/bin/env python3
"""PC 端测试：摄像头采帧 → 发 ELF2 推理 → 打印结果。纯标准库，零依赖。"""
import socket, struct, json, time, cv2, sys

ELF2_IP = "192.168.137.100"
PORT = 9999

def send_frame(sock, jpg_bytes: bytes) -> dict:
    sock.sendall(struct.pack('>I', len(jpg_bytes)) + jpg_bytes)
    header = sock.recv(4)
    nbytes = struct.unpack('>I', header)[0]
    return json.loads(sock.recv(nbytes).decode())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERR: Cannot open camera"); sys.exit(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect((ELF2_IP, PORT))
print(f"Connected to ELF2 {ELF2_IP}:{PORT}")

for i in range(5):
    ret, frame = cap.read()
    if not ret: continue
    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    t0 = time.perf_counter()
    result = send_frame(sock, jpg.tobytes())
    dt = (time.perf_counter()-t0)*1000
    n = len(result.get('persons', []))
    print(f"\n  [{i}] total={dt:.0f}ms yolo={result.get('yolo_ms','?')}ms fer={result.get('fer_ms','?')}ms")
    print(f"       detections={n}")
    for p in result.get('persons', []):
        print(f"       box={p['box']} conf={p['conf']:.2f} emo={p.get('emotion','?')} ({p.get('emo_conf','?')})")

sock.close()
cap.release()
print("\n[DONE]")
