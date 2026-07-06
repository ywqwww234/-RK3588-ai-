#!/usr/bin/env python3
"""PC摄像头 → ELF2 JPEG推流（临时，等USB摄像头到后废弃）。

用法:
  python pc_camera_stream.py [ELF2_IP] [PORT]
  默认: ELF2_IP=192.168.137.100, PORT=9998
"""
import socket, struct, time, sys
import cv2

ELF2_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.137.100"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9998

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERR: Cannot open camera 0"); sys.exit(1)

print(f"Connecting to ELF2 {ELF2_IP}:{PORT} ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
for _ in range(30):
    try:
        sock.connect((ELF2_IP, PORT))
        break
    except ConnectionRefusedError:
        print(".", end="", flush=True)
        time.sleep(1)
else:
    print("\nERR: Could not connect to ELF2")
    cap.release(); sys.exit(1)
print(" connected.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = jpg.tobytes()
        try:
            sock.sendall(struct.pack('>I', len(data)) + data)
        except (ConnectionError, BrokenPipeError, OSError):
            print("Connection lost, exiting."); break
        time.sleep(0.04)
finally:
    cap.release(); sock.close()
