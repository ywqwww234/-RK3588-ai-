#!/usr/bin/env python3
"""测试 RKNN 远程后端接入主程序"""
import config
config.VISION_BACKEND = 'rknn_remote'
from vision_yolov8 import YoloV8VisionBackend
import cv2

be = YoloV8VisionBackend()
print(f'ready={be.ready} mode={be.last_track_mode}')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    r = be.analyze_face(frame)
    print(f'expr={r[0]} prob={r[1]:.2f} eye={r[2]:.2f} posture={r[3]:.2f}')
else:
    print('camera fail')
