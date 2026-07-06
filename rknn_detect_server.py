#!/usr/bin/env python3
"""ELF2 推理服务器 — 接收 JPEG 帧，返回 YOLO+FER 检测结果。纯标准库，零依赖。"""
import socket, json, struct, time, os, sys, io, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from tools_yolo_week1.yolo_postprocess import yolov8_post_process

ROOT = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ["neutral","happy","surprise","sad","anger","disgust","fear","contempt"]
PORT = 9999
CONF_YOLO = 0.35
IOU_YOLO = 0.45

class InferEngine:
    def __init__(self):
        self.yolo = RKNNLite()
        self.yolo.load_rknn(f'{ROOT}/models/yolov8_classroom.rknn')
        self.yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.fer = RKNNLite()
        self.fer.load_rknn(f'{ROOT}/models/emotion-ferplus-8.rknn')
        self.fer.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print(f'[SRV] Models loaded. Port={PORT}')

    def process(self, jpg_bytes: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {'error': 'decode failed'}
        h, w = frame.shape[:2]

        # ── YOLO ──
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = min(640/h, 640/w)
        nw, nh = int(w*r), int(h*r)
        img_rs = cv2.resize(img_rgb, (nw, nh))
        dw = (640-nw)//2; dh = (640-nh)//2
        img_pad = cv2.copyMakeBorder(img_rs, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114,114,114))
        blob = np.expand_dims(img_pad, 0).astype(np.float32)
        t0 = time.perf_counter()
        out = self.yolo.inference(inputs=[blob], data_format=['nhwc'])
        dt_yolo = (time.perf_counter()-t0)*1000
        boxes, classes, scores = yolov8_post_process(out[0], conf_thres=CONF_YOLO, iou_thres=IOU_YOLO)

        result = {'yolo_ms': round(dt_yolo, 1), 'persons': []}
        if boxes is None:
            return result

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1r = int(max(0, x1-dw)/r); y1r = int(max(0, y1-dh)/r)
            x2r = int(max(0, x2-dw)/r); y2r = int(max(0, y2-dh)/r)
            x1r, y1r = max(0,x1r), max(0,y1r)
            x2r, y2r = min(w,x2r), min(h,y2r)
            score = float(scores[i])
            person = {'box': [x1r, y1r, x2r, y2r], 'conf': score}

            # ── FER on upper half ──
            face_h = max(10, (y2r-y1r)//2)
            face_roi = frame[y1r:y1r+face_h, x1r:x2r]
            if face_roi.size > 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                fer_blob = cv2.resize(gray, (64,64)).reshape(1,64,64,1).astype(np.float32)
                logits = self.fer.inference(inputs=[fer_blob], data_format='nhwc')[0][0]
                logits = logits - logits.max()
                probs = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
                idx = int(np.argmax(probs))
                person['emotion'] = EMOTIONS[idx]
                person['emo_conf'] = round(float(probs[idx]), 3)
            result['persons'].append(person)

        result['fer_ms'] = round((time.perf_counter()-t0)*1000 - dt_yolo, 1)
        result['total_ms'] = round((time.perf_counter()-t0)*1000, 1)
        return result

def recvn(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionError()
        buf += chunk
    return buf

engine = InferEngine()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', PORT))
sock.listen(1)
print(f'[SRV] Listening on :{PORT}')

while True:
    try:
        conn, addr = sock.accept()
        print(f'[SRV] Connect from {addr[0]}')
        while True:
            header = recvn(conn, 4)
            nbytes = struct.unpack('>I', header)[0]
            if nbytes == 0: break
            jpg = recvn(conn, nbytes)
            try:
                result = engine.process(jpg)
            except Exception as e:
                traceback.print_exc()
                result = {'error': str(e)}
            resp = json.dumps(result, ensure_ascii=False).encode()
            conn.sendall(struct.pack('>I', len(resp)) + resp)
    except (ConnectionError, BrokenPipeError, TimeoutError):
        pass
    except KeyboardInterrupt:
        break
    except Exception as e:
        traceback.print_exc()
    finally:
        try: conn.close()
        except: pass

sock.close()
print('[SRV] Stopped')
