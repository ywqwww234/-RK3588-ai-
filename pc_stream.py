#!/usr/bin/env python3
"""PC 摄像头 → HTTP JPEG 推流，供 ELF2 拉取推理"""
import cv2, time
from flask import Flask, Response

app = Flask(__name__)
frame_jpg = b'\x00'

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERR: Cannot open camera 0"); return
    while True:
        ret, frame = cap.read()
        if not ret: continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        global frame_jpg
        frame_jpg = buf.tobytes()
        time.sleep(0.05)

@app.route('/frame')
def frame():
    return Response(frame_jpg, mimetype='image/jpeg')

if __name__ == '__main__':
    import threading
    threading.Thread(target=gen_frames, daemon=True).start()
    print("PC stream ready: http://<你的IP>:8080/frame")
    app.run(host='0.0.0.0', port=8080, debug=False)
