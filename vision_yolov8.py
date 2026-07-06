"""
YOLOv8 / RKNN 视觉后端适配层。

统一封装 PC 调试、RK3588 本地 NPU 和远端 RKNN 服务三类视觉推理后端。
"""

import os
import time
from collections import deque
import cv2
import numpy as np
import socket, struct, json

# ── RKNN local（ELF2 本地 NPU 直驱）──
_RKNN_LOCAL_AVAIL = False
try:
    from rknn_inference import RknnInference
    _RKNN_LOCAL_AVAIL = True
except ImportError:
    pass

ULTRA_IMPORT_ERR = None
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ULTRA_IMPORT_ERR = str(e)

import config

try:
    from vision_face_mesh import FaceMeshAnalyzer, crop_face_roi_from_person
except Exception:
    FaceMeshAnalyzer = None
    crop_face_roi_from_person = None

try:
    from vision_face_detect import YuNetFaceDetector
except Exception:
    YuNetFaceDetector = None


class YoloV8VisionBackend:
    """统一封装 YOLOv8、RKNN local、RKNN remote 三类视觉推理后端。"""
    """Day1-2: 仅做检测接入 + 兼容字段输出，不改业务下游接口。"""

    def __init__(self):
        self.model = None
        self.ready = False
        self.last_track_mode = "yolo_none"
        self.face_miss_streak = 0
        self.last_eye_boxes = 0
        self.last_ear = None
        self.last_forward_head_risk = 0.0
        self.last_down_head_risk = 0.0
        self.last_conf_level = "LOW"

        self._eye_fatigue = 0.0
        self._posture_risk = 0.0
        self._prev_person_box = None

        # Day4.5 稳定器
        self._frame_idx = 0
        self._infer_every_n = int(getattr(config, 'YOLO_INFER_EVERY_N', 2))
        self._cached_output = ("none", 0.0, 0.0, 0.0)
        self._expr_hist = deque(maxlen=int(getattr(config, 'VISION_EXPR_VOTE_WINDOW', 10)))
        self._posture_high_streak = 0
        self._posture_low_streak = 0
        self._posture_gate = float(getattr(config, 'VISION_POSTURE_HIGH_GATE', 0.22))
        self._posture_high_need = int(getattr(config, 'VISION_POSTURE_HIGH_NEED', 6))
        self._posture_low_release = int(getattr(config, 'VISION_POSTURE_LOW_RELEASE', 4))
        self._mesh_hold = 0  # mesh 成功后保持 MODE +mesh 的剩余帧数
        self.last_smile_score = 0.0
        self.last_fer_top3 = []

        # 情绪分类头（复用项目现有 FER ONNX）
        self.emotion_net = None
        self.emotions = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']

        self.last_error = ""
        self.model_path = ""

        # 通用属性默认值（远程/本地都要）
        self._yunet = None
        self._face_mesh = None
        self._face_cascade = None

        # ── RKNN remote backend ──
        self._use_remote = (str(getattr(config, "VISION_BACKEND", "yolov8")).lower() == "rknn_remote")
        self._remote_sock = None
        self._remote_host = str(getattr(config, "RKNN_REMOTE_HOST", "192.168.137.100"))
        self._remote_port = int(getattr(config, "RKNN_REMOTE_PORT", 9999))
        self._remote_timeout = float(getattr(config, "RKNN_REMOTE_TIMEOUT", 5.0))
        if self._use_remote:
            try:
                self._remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._remote_sock.settimeout(self._remote_timeout)
                self._remote_sock.connect((self._remote_host, self._remote_port))
                self.ready = True
                self.last_track_mode = "rknn_remote"
                print(f"[YOLO] RKNN remote backend: {self._remote_host}:{self._remote_port}")
                return
            except Exception as e:
                self.last_error = f"RKNN remote connect failed: {e}"
                print(f"[YOLO] {self.last_error}")
                self._remote_sock = None
                return

        # ── RKNN local backend（ELF2 本地 NPU，无 socket）──
        self._use_local_rknn = (str(getattr(config, "VISION_BACKEND", "yolov8")).lower() == "rknn_local")
        self._local_rknn = None
        if self._use_local_rknn:
            if not _RKNN_LOCAL_AVAIL:
                self.last_error = "rknn_local requested but rknn_inference not available"
                print(f"[YOLO] {self.last_error}")
                return
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self._local_rknn = RknnInference(
                    fer_path=os.path.join(base_dir, "models", "emotion-ferplus-8.rknn"),
                    yolo_path=os.path.join(base_dir, "models", "yolov8_classroom.rknn"),
                    yolo_conf=float(getattr(config, "YOLOV8_CONF", 0.35)),
                    yolo_iou=0.45,
                )
                if self._local_rknn.ready:
                    self.ready = True
                    self.last_track_mode = "rknn_local"
                    print("[YOLO] RKNN local backend ready (YOLO NPU + FER NPU)")
                else:
                    self.last_error = f"RKNN local init: {self._local_rknn.last_error}"
                    print(f"[YOLO] {self.last_error}")
            except Exception as e:
                self.last_error = f"RKNN local init exception: {e}"
                print(f"[YOLO] {self.last_error}")
            return

        if YOLO is None:
            self.last_error = f"ultralytics import failed: {ULTRA_IMPORT_ERR or 'unknown'}"
            return
        try:
            raw_path = str(getattr(config, "YOLOV8_MODEL_PATH", "models/yolov8n.pt"))
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
            self.model_path = model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found: {model_path}")

            self.model = YOLO(model_path)

            # 尝试加载 FER 情绪模型（可选）
            fer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'emotion-ferplus-8.onnx')
            if os.path.exists(fer_path):
                try:
                    self.emotion_net = cv2.dnn.readNetFromONNX(fer_path)
                    print(f"[YOLO] emotion head ready. model={fer_path}")
                except Exception as ee:
                    print(f"[YOLO] emotion head load failed: {ee}")
                    self.emotion_net = None

            # Week3: MediaPipe on face ROI + Haar for tighter FER crop
            self._face_mesh = None
            self._face_cascade = None
            if getattr(config, "VISION_YOLO_USE_MEDIAPIPE", True) and FaceMeshAnalyzer is not None:
                self._face_mesh = FaceMeshAnalyzer()
                if self._face_mesh.ready:
                    self._face_mesh.tracking_stride = int(
                        getattr(config, "VISION_MEDIAPIPE_STRIDE", 1)
                    )
                    api = getattr(self._face_mesh, "_api", "?")
                    print(f"[YOLO] MediaPipe face mesh ready (Week3, api={api})")
                else:
                    print(f"[YOLO] MediaPipe unavailable: {self._face_mesh.last_error}")
                    self._face_mesh = None
            if getattr(config, "VISION_YOLO_USE_FACE_ROI", True):
                try:
                    import sys
                    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
                    face_xml = os.path.join(base_dir, "cv2", "data", "haarcascade_frontalface_default.xml")
                    if not os.path.exists(face_xml):
                        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    self._face_cascade = cv2.CascadeClassifier(face_xml)
                    if self._face_cascade.empty():
                        self._face_cascade = None
                except Exception:
                    self._face_cascade = None

            self._yunet = None
            mode = str(getattr(config, "VISION_FACE_DETECTOR", "yunet")).lower()
            if mode == "yunet" and YuNetFaceDetector is not None:
                self._yunet = YuNetFaceDetector(
                    os.path.dirname(os.path.abspath(__file__))
                )
                if self._yunet.ready:
                    print("[YOLO] YuNet face detector ready (FER crop)")
                else:
                    print(f"[YOLO] YuNet skipped: {self._yunet.last_error}")
                    self._yunet = None

            self.ready = True
            self.last_track_mode = "yolo_ready"
            print(f"[YOLO] backend ready. model={model_path}")
        except Exception as e:
            self.model = None
            self.ready = False
            self.last_error = str(e)
            self.last_track_mode = "yolo_init_err"
            print(f"[YOLO] init failed: {e}")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-8)

    def _vote_expr(self, expr_type: str, expr_prob: float):
        et = str(expr_type)
        ep = float(expr_prob)
        if et != 'none':
            self._expr_hist.append((et, ep))

        if not self._expr_hist:
            return et, ep

        score = {}
        for n, p in self._expr_hist:
            score[n] = score.get(n, 0.0) + float(p)
        best = max(score.items(), key=lambda kv: kv[1])[0]
        best_avg = float(np.mean([p for n, p in self._expr_hist if n == best]))
        return best, best_avg

    def _log_expr_terminal(self, expr_type: str, expr_prob: float):
        if not getattr(config, "VISION_EXPR_LOG_TERMINAL", True):
            return
        et = str(expr_type)
        ep = float(expr_prob)
        now = time.time()
        interval = float(getattr(config, "VISION_EXPR_LOG_INTERVAL_SEC", 0.35))
        changed = et != getattr(self, "_last_expr_logged", None)
        if not changed and (now - getattr(self, "_last_expr_log_ts", 0.0)) < interval:
            return
        self._last_expr_logged = et
        self._last_expr_log_ts = now
        top3 = getattr(self, "last_fer_top3", None) or []
        top3_s = ", ".join(f"{n}:{p:.2f}" for n, p in top3[:3]) if top3 else "-"
        mode = str(getattr(self, "last_track_mode", ""))
        line = f"[EXPR] {et}  prob={ep:.2f}  mode={mode}  fer_top3=[{top3_s}]"
        print(line, flush=True)

    def _apply_fer_non_neutral_override(self, expr_type: str, expr_prob: float):
        """FER 输出 neutral 但 happy/负类概率不低时，改为 smile/negative。"""
        if not getattr(config, "VISION_FER_OVERRIDE_NEUTRAL", True):
            return expr_type, expr_prob
        if str(expr_type) != "neutral":
            return expr_type, expr_prob
        happy_p = 0.0
        if getattr(self, "last_fer_top3", None):
            for n, p in self.last_fer_top3:
                if str(n).lower() == "happy":
                    happy_p = float(p)
                    break
        neg_sum = float(getattr(self, "last_fer_neg_sum", 0.0))
        h_min = float(getattr(config, "VISION_FER_HAPPY_OVERRIDE_MIN", 0.10))
        n_min = float(getattr(config, "VISION_FER_NEG_OVERRIDE_MIN", 0.18))
        if happy_p >= h_min and happy_p >= neg_sum * 0.85:
            return "smile", max(float(expr_prob), min(0.88, 0.42 + happy_p * 0.55))
        if neg_sum >= n_min:
            sad_p = float(getattr(self, "last_fer_sad_p", 0.0))
            return "negative", max(float(expr_prob), min(0.86, 0.38 + neg_sum * 0.5 + sad_p * 0.2))
        return expr_type, expr_prob

    def _fer_blob_from_gray64(self, crop_gray):
        face_roi = cv2.resize(crop_gray, (64, 64)).astype(np.float32)
        prep = str(getattr(config, "VISION_FER_PREPROCESS", "mean128")).lower()
        if prep == "mean128":
            face_roi = (face_roi - 128.0) / 128.0
        else:
            face_roi = face_roi / 255.0
        layout = str(getattr(config, "VISION_FER_INPUT_LAYOUT", "nchw")).lower()
        if layout == "nchw":
            return face_roi.reshape(1, 1, 64, 64)
        img = np.clip(face_roi * 128.0 + 128.0, 0, 255).astype(np.float32) / 255.0
        return cv2.dnn.blobFromImage(img)

    def _map_fer_class(self, emo: str, p: float):
        emo = str(emo).lower()
        if emo == "happy":
            return "smile", p
        if emo in ("sad", "fear", "anger", "disgust"):
            return "negative", p
        if emo == "surprise":
            return "surprise", p
        return "neutral", p

    def _infer_emotion_from_crop(self, crop_gray):
        if self.emotion_net is None or crop_gray is None or crop_gray.size == 0:
            return "neutral", 0.40
        try:
            blob = self._fer_blob_from_gray64(crop_gray)
            self.emotion_net.setInput(blob)
            out = self.emotion_net.forward()[0]
            probs = self._softmax(out)
            order = np.argsort(probs)[::-1]
            i = int(order[0])
            margin = float(getattr(config, "VISION_FER_NEUTRAL_MARGIN", 0.12))
            if self.emotions[i] == "neutral" and len(order) > 1:
                j = int(order[1])
                if float(probs[i] - probs[j]) < margin:
                    i = j
            emo = self.emotions[i]
            p = float(probs[i])
            self.last_fer_top3 = [
                (self.emotions[int(k)], float(probs[int(k)])) for k in order[:3]
            ]
            neg_names = ("sad", "fear", "anger", "disgust")
            self.last_fer_neg_sum = float(
                sum(float(probs[self.emotions.index(n)]) for n in neg_names if n in self.emotions)
            )
            self.last_fer_surprise_p = float(probs[self.emotions.index("surprise")]) if "surprise" in self.emotions else 0.0
            self.last_fer_sad_p = float(probs[self.emotions.index("sad")]) if "sad" in self.emotions else 0.0
            self.last_fer_max_prob = float(probs[i])
            if getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                print(f"[FER] top3={self.last_fer_top3} -> {self._map_fer_class(emo, p)[0]}")
            return self._map_fer_class(emo, p)
        except Exception as ex:
            if getattr(config, "VISION_FER_DEBUG", False):
                print(f"[FER] infer failed: {ex}")
            return "neutral", 0.40

    def _apply_mesh_output(self, mesh_out):
        if not mesh_out or not mesh_out.get("ok"):
            return False
        self._eye_fatigue = float(
            np.clip(
                0.82 * self._eye_fatigue + 0.18 * float(mesh_out["eye_fatigue"]),
                0.0,
                1.0,
            )
        )
        self._posture_risk = float(
            np.clip(
                0.78 * self._posture_risk + 0.22 * float(mesh_out["posture_risk"]),
                0.0,
                1.0,
            )
        )
        self.last_down_head_risk = float(mesh_out["down_head_risk"])
        self.last_forward_head_risk = float(mesh_out["forward_head_risk"])
        if mesh_out.get("ear") is not None:
            self.last_ear = float(mesh_out["ear"])
        if mesh_out.get("smile_score") is not None:
            self.last_smile_score = float(mesh_out["smile_score"])
        if mesh_out.get("frown_score") is not None:
            self.last_frown_score = float(mesh_out["frown_score"])
        if mesh_out.get("mouth_open_score") is not None:
            self.last_mouth_open_score = float(mesh_out["mouth_open_score"])
        self.last_eye_boxes = 2
        self._mesh_hold = int(getattr(config, "VISION_MESH_HOLD_FRAMES", 15))
        return True

    def _get_face_bgr_for_fer(self, frame, x1, y1, x2, y2):
        """FER + Mesh 用人脸 BGR：YuNet > Haar > person 上半身。"""
        mode = str(getattr(config, "VISION_FACE_DETECTOR", "yunet")).lower()
        if mode == "yunet" and self._yunet is not None and self._yunet.ready:
            if getattr(config, "VISION_FER_USE_FULLFRAME_FACE", True):
                fb, _ = self._yunet.detect_best_face(frame, None)
                if fb is not None and fb.size > 0:
                    return fb
            fb, _ = self._yunet.detect_best_face(frame, (x1, y1, x2, y2))
            if fb is not None and fb.size > 0:
                return fb
        if mode != "person_crop" and getattr(config, "VISION_YOLO_USE_FACE_ROI", True):
            if crop_face_roi_from_person is not None:
                fb, _ = crop_face_roi_from_person(
                    frame, x1, y1, x2, y2, self._face_cascade
                )
                if fb is not None and fb.size > 0:
                    return fb
        return frame[y1:y2, x1:x2]

    def _try_mediapipe(self, frame, face_bgr, x1, y1, x2, y2):
        if not getattr(config, "VISION_YOLO_USE_MEDIAPIPE", True) or self._face_mesh is None:
            return False
        mesh_out = None
        if face_bgr is not None and face_bgr.size > 0:
            mesh_out = self._face_mesh.process_bgr_roi(face_bgr)
        if (not mesh_out or not mesh_out.get("ok")) and getattr(
            config, "VISION_MESH_TRY_FULL_FRAME", True
        ):
            mesh_out = self._face_mesh.process_bgr_roi(frame)
        if (not mesh_out or not mesh_out.get("ok")) and crop_face_roi_from_person is not None:
            face_wide, _ = crop_face_roi_from_person(frame, x1, y1, x2, y2, None)
            if face_wide is not None and face_wide.size > 0:
                mesh_out = self._face_mesh.process_bgr_roi(face_wide)
        return self._apply_mesh_output(mesh_out)

    def _fer_assist_enabled(self):
        any_on = (
            getattr(config, "VISION_FER_SMILE_ASSIST", False)
            or getattr(config, "VISION_FER_FROWN_ASSIST", False)
            or getattr(config, "VISION_FER_SAD_DOMINANT", False)
            or getattr(config, "VISION_FER_SURPRISE_ASSIST", False)
        )
        if not any_on:
            return False
        top3 = getattr(self, "last_fer_top3", None) or []
        if getattr(config, "VISION_FER_ASSIST_WHEN_NEUTRAL_FER_TOP", True) and top3:
            if str(top3[0][0]).lower() == "neutral":
                return True
        if not getattr(config, "VISION_FER_ASSIST_WHEN_UNCERTAIN", False):
            return True
        th = float(getattr(config, "VISION_FER_UNCERTAIN_THRESH", 0.52))
        return float(getattr(self, "last_fer_max_prob", 1.0)) < th

    def _sync_track_mode_after_mesh(self, mesh_ok_this_frame: bool):
        hold_max = int(getattr(config, "VISION_MESH_HOLD_FRAMES", 15))
        if mesh_ok_this_frame:
            self._mesh_hold = hold_max
        elif self._mesh_hold > 0:
            self._mesh_hold -= 1
        if mesh_ok_this_frame or self._mesh_hold > 0:
            self.last_track_mode = "yolo_det+mesh"
        else:
            self.last_track_mode = "yolo_det"

    def _remote_detect(self, frame) -> dict:
        """发送帧到 ELF2，断线自动重连。"""
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = jpg.tobytes()
        for attempt in range(2):
            try:
                self._remote_sock.sendall(struct.pack('>I', len(data)) + data)
                header = self._remote_sock.recv(4)
                nbytes = struct.unpack('>I', header)[0]
                return json.loads(self._remote_sock.recv(nbytes).decode())
            except (ConnectionError, BrokenPipeError, TimeoutError, OSError) as e:
                if attempt == 0:
                    print(f"[YOLO] remote disconnected, reconnecting... ({e})")
                    try: self._remote_sock.close()
                    except: pass
                    try:
                        self._remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self._remote_sock.settimeout(self._remote_timeout)
                        self._remote_sock.connect((self._remote_host, self._remote_port))
                        print(f"[YOLO] reconnected to {self._remote_host}:{self._remote_port}")
                    except Exception as e2:
                        print(f"[YOLO] reconnect failed: {e2}")
                        self._remote_sock = None
                        raise
        return {}

    def _local_rknn_detect(self, frame) -> dict:
        """ELF2 本地 RKNNLite 推理，返回与 _remote_detect 同格式 dict。"""
        t0 = time.perf_counter()
        boxes, classes, scores = self._local_rknn.infer_yolo(frame)
        dt_yolo = (time.perf_counter() - t0) * 1000
        result = {"yolo_ms": round(dt_yolo, 1), "fer_ms": 0.0, "persons": []}
        if boxes is None:
            return result
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            person = {"box": [x1, y1, x2, y2], "conf": round(float(scores[i]), 3)}
            face_h = max(10, (y2 - y1) // 2)
            face_roi = frame[y1:y1 + face_h, x1:x2]
            if face_roi.size > 0:
                emo, emo_p = self._local_rknn.infer_fer(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                person["emotion"] = emo
                person["emo_conf"] = round(emo_p, 3)
            result["persons"].append(person)
        result["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["fer_ms"] = round(result["total_ms"] - dt_yolo, 1)
        return result

    def analyze_face(self, frame):
        """兼容 LocalFaceAnalyzer 返回:
        (expr_type, expr_prob, eye_fatigue_index, posture_risk)
        """
        self._frame_idx += 1

        # Day4.5: 推理限频 + 结果复用
        if self._infer_every_n > 1 and (self._frame_idx % self._infer_every_n) != 0:
            expr_type, expr_prob, _, _ = self._cached_output
            # 非推理帧做平滑释放，保持实时感
            self._eye_fatigue = float(np.clip(0.96 * self._eye_fatigue + 0.04 * 0.10, 0.0, 1.0))
            self._posture_risk = float(np.clip(0.97 * self._posture_risk, 0.0, 1.0))
            return expr_type, float(expr_prob), float(self._eye_fatigue), float(self._posture_risk)

        if not self.ready:
            self.face_miss_streak += 1
            self.last_track_mode = "yolo_unavailable"
            if self.last_error:
                print(f"[YOLO] unavailable: {self.last_error}")
            self.last_conf_level = "LOW"
            self._eye_fatigue = max(0.0, self._eye_fatigue * 0.98)
            self._posture_risk = max(0.0, self._posture_risk * 0.98)
            return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)

        try:
            # ── 统一分发：remote / local RKNN / local YOLO ──
            is_rknn = self._use_remote or self._use_local_rknn
            result = None

            if self._use_remote:
                if self._remote_sock is None:
                    self.face_miss_streak += 1
                    self.last_track_mode = "rknn_disconnected"
                    self.last_conf_level = "LOW"
                    self._eye_fatigue = min(1.0, self._eye_fatigue + 0.02)
                    self._posture_risk = max(0.0, self._posture_risk * 0.97)
                    return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)
                result = self._remote_detect(frame)
            elif self._use_local_rknn:
                result = self._local_rknn_detect(frame)

            if result is not None:
                # ── 共享 RKNN 后处理（remote + local）──
                persons = result.get('persons', [])
                person_count = len(persons)
                if person_count == 0:
                    self.face_miss_streak += 1
                    self.last_track_mode = "yolo_no_det"
                    self.last_conf_level = "LOW"
                    self._eye_fatigue = min(1.0, self._eye_fatigue + 0.01)
                    self._posture_risk = max(0.0, self._posture_risk * 0.97)
                    return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)

                self.face_miss_streak = 0
                max_person_conf = max(p['conf'] for p in persons)
                p0 = persons[0]
                x1, y1, x2, y2 = p0['box']
                x1 = int(max(0, x1)); y1 = int(max(0, y1))
                x2 = int(min(frame.shape[1] - 1, x2)); y2 = int(min(frame.shape[0] - 1, y2))
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                ratio = h / w
                frame_h = float(frame.shape[0])
                top_norm = y1 / max(1.0, frame_h)
                mesh_mode = False
                face_bgr = self._get_face_bgr_for_fer(frame, x1, y1, x2, y2)
                crop_gray = (
                    cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                    if face_bgr is not None and face_bgr.size > 0
                    else None
                )
                # rknn_local: 用 YuNet 精脸裁切重新跑 FER，不用 _local_rknn_detect 里的粗略裁切
                if self._use_local_rknn and self._local_rknn is not None and crop_gray is not None:
                    emo_local, emo_p_local, top3, neg_sum, sad_p, sur_p, max_p = \
                        self._local_rknn.infer_fer_full(crop_gray)
                    self.last_fer_top3 = top3
                    self.last_fer_neg_sum = neg_sum
                    self.last_fer_sad_p = sad_p
                    self.last_fer_surprise_p = sur_p
                    self.last_fer_max_prob = max_p
                    expr_type = self._map_fer_class(emo_local, emo_p_local)[0]
                    emo_prob = emo_p_local
                else:
                    expr_type = self._map_fer_class(p0.get('emotion', 'neutral'), p0.get('emo_conf', 0.4))[0]
                    emo_prob = p0.get('emo_conf', 0.4)
                expr_prob = max(0.35, min(0.98, 0.55 * float(emo_prob) + 0.45 * max_person_conf))
                self.last_track_mode = f"rknn_t{result.get('yolo_ms','?')}ms"

            else:
                # ── 本地 YOLO .pt 推理路径 ──
                conf = float(getattr(config, "YOLOV8_CONF", 0.35))
                imgsz = int(getattr(config, "YOLOV8_IMG_SIZE", 640))
                res = self.model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
                r0 = res[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    self.face_miss_streak += 1
                    self.last_track_mode = "yolo_no_det"
                    self.last_conf_level = "LOW"
                    self._eye_fatigue = min(1.0, self._eye_fatigue + 0.01)
                    self._posture_risk = max(0.0, self._posture_risk * 0.97)
                    return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)

                self.face_miss_streak = 0

                cls_arr = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.array([])
                conf_arr = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.array([])
                xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.zeros((0, 4))

                names = getattr(r0, "names", {}) or {}
                person_indices = []
                for i, c in enumerate(cls_arr):
                    cid = int(c)
                    cname = str(names.get(cid, cid)).lower()
                    if cid == 0 or cname == "person":
                        person_indices.append(i)

                person_count = len(person_indices)
                max_person_conf = float(max([conf_arr[i] for i in person_indices], default=0.0))

                if person_count == 0:
                    self.last_conf_level = "LOW"
                    self._eye_fatigue = min(1.0, self._eye_fatigue + 0.02)
                    self._posture_risk = max(0.0, self._posture_risk * 0.97)
                    return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)

                idx = person_indices[int(np.argmax([conf_arr[i] for i in person_indices]))]
                x1, y1, x2, y2 = xyxy[idx]
                x1 = int(max(0, x1)); y1 = int(max(0, y1))
                x2 = int(min(frame.shape[1] - 1, x2)); y2 = int(min(frame.shape[0] - 1, y2))
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                ratio = h / w
                frame_h = float(frame.shape[0])
                top_norm = y1 / max(1.0, frame_h)

                mesh_mode = False
                face_bgr = self._get_face_bgr_for_fer(frame, x1, y1, x2, y2)
                crop_gray = (
                    cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                    if face_bgr is not None and face_bgr.size > 0
                    else None
                )
                expr_type, emo_prob = self._infer_emotion_from_crop(crop_gray)
                expr_prob = max(0.35, min(0.98, 0.55 * float(emo_prob) + 0.45 * max_person_conf))

            mesh_mode = self._try_mediapipe(frame, face_bgr, x1, y1, x2, y2)

            if not mesh_mode:
                # --- Day4 回退：person 框启发式 ---
                raw_post = 0.0
                if ratio < 1.45:
                    raw_post += 0.35
                if ratio < 1.15:
                    raw_post += 0.35
                down_inst = 0.0
                if ratio < 1.35:
                    down_inst += 0.45
                if ratio < 1.20:
                    down_inst += 0.35
                if top_norm > 0.22:
                    down_inst += min(0.45, (top_norm - 0.18) * 2.2)
                forward_inst = 0.0
                if ratio > 1.75:
                    forward_inst += 0.40
                if ratio > 2.05:
                    forward_inst += 0.35
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                if self._prev_person_box is not None:
                    pcx, pcy, pw, ph = self._prev_person_box
                    drift_y = max(0.0, (cy - pcy) / max(1.0, frame_h))
                    scale_jump = max(0.0, (h / max(1.0, ph)) - 1.0)
                    scale_shrink = max(0.0, 1.0 - (h / max(1.0, ph)))
                    raw_post += min(0.30, drift_y * 2.6)
                    raw_post += min(0.25, scale_jump * 1.8)
                    down_inst = max(down_inst, min(1.0, drift_y * 4.0))
                    forward_inst = max(forward_inst, min(1.0, scale_jump * 3.0))
                    if scale_shrink > 0.08:
                        forward_inst = max(forward_inst, min(1.0, scale_shrink * 2.5))
                down_inst = float(np.clip(down_inst, 0.0, 1.0))
                forward_inst = float(np.clip(forward_inst, 0.0, 1.0))
                self.last_down_head_risk = float(
                    np.clip(0.75 * self.last_down_head_risk + 0.25 * down_inst, 0.0, 1.0)
                )
                self.last_forward_head_risk = float(
                    np.clip(0.75 * self.last_forward_head_risk + 0.25 * forward_inst, 0.0, 1.0)
                )
                self._prev_person_box = (cx, cy, w, h)
                raw_post = float(np.clip(raw_post, 0.0, 1.0))
                self._posture_risk = float(
                    np.clip(0.78 * self._posture_risk + 0.22 * raw_post, 0.0, 1.0)
                )
                fatigue_target = 0.10 if person_count == 1 else 0.18
                if max_person_conf < 0.5:
                    fatigue_target += 0.08
                self._eye_fatigue = float(
                    np.clip(0.88 * self._eye_fatigue + 0.12 * fatigue_target, 0.0, 1.0)
                )
            else:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                self._prev_person_box = (cx, cy, w, h)

            self._sync_track_mode_after_mesh(mesh_mode)

            # 表情 assist 顺序：负向/sad → 惊讶(大张嘴) → 微笑(闭嘴咧嘴)，避免张嘴误判 smile
            mo = float(getattr(self, "last_mouth_open_score", 0.0))
            ss = float(getattr(self, "last_smile_score", 0.0))
            fr = float(getattr(self, "last_frown_score", 0.0))
            neg_sum = float(getattr(self, "last_fer_neg_sum", 0.0))
            sad_p = float(getattr(self, "last_fer_sad_p", 0.0))
            sp = float(getattr(self, "last_fer_surprise_p", 0.0))
            mouth_block_smile = mo >= float(
                getattr(config, "VISION_FER_SMILE_BLOCK_MOUTH_OPEN", 0.42)
            )

            if mesh_mode and self._fer_assist_enabled() and getattr(config, "VISION_FER_SAD_DOMINANT", True):
                sad_min = float(getattr(config, "VISION_FER_SAD_DOMINANT_MIN", 0.18))
                top0 = self.last_fer_top3[0][1] if getattr(self, "last_fer_top3", None) else 0.0
                if sad_p >= sad_min and (sad_p >= top0 * 0.85 or neg_sum >= 0.20):
                    expr_type = "negative"
                    expr_prob = max(
                        float(expr_prob),
                        min(0.82, 0.40 + sad_p * 0.55 + neg_sum * 0.2),
                    )
                    if getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                        print(f"[FER] sad_dominant sad={sad_p:.2f} neg={neg_sum:.2f} -> negative")

            if (
                mesh_mode
                and self._fer_assist_enabled()
                and getattr(config, "VISION_FER_FROWN_ASSIST", True)
                and expr_type not in ("negative",)
            ):
                fr_th = float(getattr(config, "VISION_FER_FROWN_THRESH", 0.32))
                neg_min = float(getattr(config, "VISION_FER_NEG_FER_MIN", 0.14))
                if fr >= fr_th and neg_sum >= neg_min and ss < 0.30:
                    expr_type = "negative"
                    expr_prob = max(
                        float(expr_prob),
                        min(0.88, 0.38 + fr * 0.35 + neg_sum * 0.45 + sad_p * 0.25),
                    )
                    if getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                        print(
                            f"[FER] frown_assist fr={fr:.2f} neg_sum={neg_sum:.2f} sad={sad_p:.2f} -> negative"
                        )

            if (
                mesh_mode
                and self._fer_assist_enabled()
                and getattr(config, "VISION_FER_SURPRISE_ASSIST", True)
                and expr_type not in ("negative", "smile")
            ):
                mo_th = float(getattr(config, "VISION_FER_MOUTH_OPEN_THRESH", 0.38))
                if mo >= mo_th and ss < 0.40:
                    expr_type = "surprise"
                    expr_prob = max(
                        float(expr_prob),
                        min(0.85, 0.32 + mo * 0.40 + sp * 0.35),
                    )
                    if getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                        print(f"[FER] surprise_assist open={mo:.2f} fer_sp={sp:.2f} -> surprise")

            if (
                mesh_mode
                and self._fer_assist_enabled()
                and getattr(config, "VISION_FER_SMILE_ASSIST", True)
                and not mouth_block_smile
                and expr_type not in ("negative", "surprise")
            ):
                th = float(getattr(config, "VISION_FER_SMILE_BLEND_THRESH", 0.35))
                if ss >= th and expr_type in ("neutral",):
                    expr_type = "smile"
                    expr_prob = max(float(expr_prob), min(0.92, 0.42 + ss * 0.48))
                    if getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                        print(f"[FER] smile_assist score={ss:.2f} open={mo:.2f} -> smile")
                elif mouth_block_smile and getattr(config, "VISION_FER_DEBUG", False) and (self._frame_idx % 30 == 0):
                    print(f"[FER] smile_blocked mouth_open={mo:.2f} raw_smile={ss:.2f}")

            # 情绪投票稳定
            expr_type, expr_prob = self._vote_expr(expr_type, expr_prob)
            expr_type, expr_prob = self._apply_fer_non_neutral_override(expr_type, expr_prob)
            self._log_expr_terminal(expr_type, expr_prob)

            # 姿态门控+释放
            if self._posture_risk >= self._posture_gate:
                self._posture_high_streak += 1
                self._posture_low_streak = 0
            else:
                self._posture_low_streak += 1
                self._posture_high_streak = max(0, self._posture_high_streak - 1)

            if self._posture_high_streak < self._posture_high_need:
                self._posture_risk = float(self._posture_risk * 0.45)
            elif self._posture_low_streak >= self._posture_low_release:
                self._posture_risk = float(self._posture_risk * 0.75)

            self.last_eye_boxes = 2 if max_person_conf >= 0.55 else 1
            self.last_ear = 0.29 if self._eye_fatigue < 0.25 else (0.23 if self._eye_fatigue < 0.45 else 0.18)
            self.last_conf_level = "HIGH" if max_person_conf >= 0.70 else ("MID" if max_person_conf >= 0.50 else "LOW")

            self._cached_output = (expr_type, float(expr_prob), float(self._eye_fatigue), float(self._posture_risk))
            return self._cached_output

        except Exception as e:
            self.face_miss_streak += 1
            self.last_track_mode = "yolo_err"
            self.last_conf_level = "LOW"
            self.last_error = str(e)
            print(f"[YOLO] predict failed: {e}")
            self._eye_fatigue = max(0.0, self._eye_fatigue * 0.98)
            self._posture_risk = max(0.0, self._posture_risk * 0.98)
            return "none", 0.0, float(self._eye_fatigue), float(self._posture_risk)
