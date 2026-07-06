"""
摄像头与多模态融合主线程。

负责采集视频流、调用视觉分析、汇总外部传感器特征，并向 UI 推送实时风险结果。
"""

from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import socket
import time
from local_vision import LocalFaceAnalyzer
from vision_yolov8 import YoloV8VisionBackend
# 导入融合算法和模拟传感器
from risk_calculator import calculate_visual_risk, calculate_hrv_risk, calculate_eeg_risk, calculate_total_risk, get_risk_level
from mock_sensors import MockHardwareSensors
import config

class CameraThread(QThread):
    """学生端实时监测主线程。"""
    change_pixmap_signal = pyqtSignal(object)          # 实时画面
    # 实时风险信号 (总分, 等级, 表情类型, 概率, 眼疲劳指数, 姿态风险, 探头风险, 低头风险)
    realtime_risk_signal = pyqtSignal(float, str, str, float, float, float, float, float)
    
    # 多模态详细数据信号
    # (视觉风险, HRV风险, EEG风险, 眼疲劳, 姿态风险, RMSSD, 平均心率, 专注度, 放松度)
    multimodal_data_signal = pyqtSignal(float, float, float, float, float, float, float, int, int) 

    record_risk_signal = pyqtSignal(float, str)        # 定时记录的风险（用于持久化）
    alert_signal = pyqtSignal()                        # 预警信号
    battery_signal = pyqtSignal(float)                 # 社交电量

    def __init__(self, recorder):
        super().__init__()
        vision_backend = str(getattr(config, 'VISION_BACKEND', 'legacy')).lower()
        if vision_backend in ('yolov8', 'rknn_remote', 'rknn_local'):
            self.api_client = YoloV8VisionBackend()
        else:
            self.api_client = LocalFaceAnalyzer()
        self._camera_source = str(getattr(config, 'CAMERA_SOURCE', 'local')).lower()
        self._stream_port = int(getattr(config, 'CAMERA_STREAM_PORT', 9998))
        self.recorder = recorder
        self.sensor = MockHardwareSensors()            # 初始化虚拟硬件
        self.external_eeg_data = None
        
        # 不在初始化时锁死模拟状态，后续根据视觉结果动态联动
        self.running = True
        self.real_time_interval = config.REAL_TIME_INTERVAL
        self.record_interval = config.RECORD_INTERVAL
        self.last_real_time = 0
        self.last_record_time = 0
        self.recent_risks = []
        self.social_battery = 1.0
        self._battery_high_streak = 0
        self._battery_low_streak = 0

        # 记录开关（由 UI 的“启动全节点加密监测”按钮控制）
        self.recording_enabled = False
        self.force_record_once = False

        # 风险平滑与趋势控制：避免“突然冲顶后难以下降”的锯齿
        self._risk_ema = 0.0
        self._risk_ema_inited = False

        # 视觉低置信门控（连续低置信时动态降权）
        self._vision_low_conf_streak = 0
        self._vision_low_conf_apply_after = int(getattr(config, 'VISION_LOW_CONF_APPLY_AFTER', 20))
        self._vision_low_conf_weight_scale = float(getattr(config, 'VISION_LOW_CONF_WEIGHT_SCALE', 0.4))

        # 画面调试叠加状态（用于“一眼确认闭眼是否生效”）
        self._dbg_expr = "none"
        self._dbg_prob = 0.0
        self._dbg_eye = 0.0
        self._dbg_post = 0.0

    def _open_stream_receiver(self):
        """临时：接收 PC 推送的 JPEG 帧。返回类 VideoCapture 对象。USB摄像头到后废弃。"""
        import struct
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self._stream_port))
        server.listen(1)
        print(f"[CAM] Waiting for PC stream on :{self._stream_port} ...")
        conn, addr = server.accept()
        print(f"[CAM] PC stream connected from {addr[0]}")
        server.close()

        class _StreamCap:
            def __init__(self, sock):
                self._sock = sock
            def read(self):
                try:
                    header = self._sock.recv(4)
                    if len(header) < 4:
                        return False, None
                    nbytes = struct.unpack('>I', header)[0]
                    buf = b''
                    while len(buf) < nbytes:
                        chunk = self._sock.recv(nbytes - len(buf))
                        if not chunk:
                            return False, None
                        buf += chunk
                    frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                    return frame is not None, frame
                except Exception:
                    return False, None
            def release(self):
                try: self._sock.close()
                except: pass

        return _StreamCap(conn)

    def run(self):
        if self._camera_source == 'stream':
            cap = self._open_stream_receiver()
        else:
            cap = cv2.VideoCapture(21, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                print("[CAM] No camera found, retrying every 3s...")
                cap.release()
                while self.running and self._camera_source == 'local':
                    time.sleep(3)
                    cap = cv2.VideoCapture(21, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    if cap.isOpened():
                        print("[CAM] Camera connected!")
                        break

        while self.running:
            ret, frame = cap.read()
            if ret:
                current_time = time.time()

                if current_time - self.last_real_time >= self.real_time_interval:
                    self.last_real_time = current_time
                    self._analyze_frame(frame, is_record=False)

                if current_time - self.last_record_time >= self.record_interval:
                    self.last_record_time = current_time
                    self._analyze_frame(frame, is_record=True)

                # 每帧叠加调试信息，确保现场可见
                self._draw_debug_overlay(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.change_pixmap_signal.emit(rgb_image)

            else:
                time.sleep(0.01)
        cap.release()

    def _draw_debug_overlay(self, frame):
        """在原始 BGR 帧上叠加关键调试信息。"""
        try:
            dbg_on = bool(getattr(config, 'CAMERA_DEBUG_OVERLAY', True))
            if not dbg_on:
                return

            mode = str(getattr(self.api_client, 'last_track_mode', 'none'))
            miss = int(getattr(self.api_client, 'face_miss_streak', 0))
            eyes = int(getattr(self.api_client, 'last_eye_boxes', 0))
            ear = getattr(self.api_client, 'last_ear', None)
            ear_text = f"{float(ear):.3f}" if ear is not None else "NA"

            lines = [
                f"MODE:{mode}  MISS:{miss}  EYES:{eyes}  EAR:{ear_text}",
                f"EXPR:{self._dbg_expr}  PROB:{self._dbg_prob:.2f}",
                f"EYE_FATIGUE:{self._dbg_eye:.2f}  POSTURE:{self._dbg_post:.2f}",
            ]

            x0, y0 = 12, 28
            line_h = 24
            box_w = 520
            box_h = 84 + 18  # extra space for title bar

            # 半透明背景 + 标题分隔线，与检测框视觉区分
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 - 8 + box_w, y0 - 20 + box_h), (18, 18, 22), -1)
            frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            # 标题栏
            title_h = 20
            cv2.rectangle(frame, (x0 - 8, y0 - 20), (x0 - 8 + box_w, y0 - 20 + title_h), (40, 40, 50), -1)
            cv2.putText(frame, "SYS INFO", (x0 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 170), 1, cv2.LINE_AA)

            # 外框 — 用暗灰而非亮色，不混淆为检测框
            cv2.rectangle(frame, (x0 - 8, y0 - 20), (x0 - 8 + box_w, y0 - 20 + box_h), (70, 70, 80), 1)

            text_y0 = y0 + 4
            for i, txt in enumerate(lines):
                y = text_y0 + i * line_h
                cv2.putText(frame, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 210), 1, cv2.LINE_AA)
        except Exception:
            pass

    def _analyze_frame(self, frame, is_record=False):
        """三模态数据采集与融合分析"""

        # 1. 视觉模态（兼容旧版2返回值与新版4返回值）
        analyze_ret = self.api_client.analyze_face(frame)

        expr_type = None
        expr_prob = 0.0
        eye_fatigue_index = 0.0
        posture_risk = 0.0

        if isinstance(analyze_ret, (tuple, list)):
            if len(analyze_ret) >= 1:
                expr_type = analyze_ret[0]
            if len(analyze_ret) >= 2:
                expr_prob = analyze_ret[1]
            if len(analyze_ret) >= 3:
                eye_fatigue_index = analyze_ret[2]
            if len(analyze_ret) >= 4:
                posture_risk = analyze_ret[3]

        # 防NoneType：统一安全化与归一化
        safe_expr = expr_type if expr_type is not None else "未检测到人脸"
        safe_prob = float(expr_prob) if expr_prob is not None else 0.0
        safe_prob = max(0.0, min(1.0, safe_prob))

        eye_fatigue = float(eye_fatigue_index) if eye_fatigue_index is not None else 0.0
        neck_posture = float(posture_risk) if posture_risk is not None else 0.0
        eye_fatigue = max(0.0, min(1.0, eye_fatigue))
        neck_posture = max(0.0, min(1.0, neck_posture))

        # 更新调试叠加字段
        self._dbg_expr = str(safe_expr)
        self._dbg_prob = float(safe_prob)
        self._dbg_eye = float(eye_fatigue)
        self._dbg_post = float(neck_posture)

        # 视觉风险池融合（避免被“平均效应”稀释）
        base_visual_risk = calculate_visual_risk(safe_expr, safe_prob)

        # 读取视觉置信状态（由 LocalFaceAnalyzer 提供）
        conf_level = str(getattr(self.api_client, 'last_conf_level', 'LOW')).upper()
        if conf_level == 'LOW':
            self._vision_low_conf_streak += 1
        else:
            self._vision_low_conf_streak = 0

        # 额外姿态特征（来自视觉模块缓存）
        forward_head_risk = float(getattr(self.api_client, 'last_forward_head_risk', 0.0))
        down_head_risk = float(getattr(self.api_client, 'last_down_head_risk', 0.0))
        forward_head_risk = max(0.0, min(1.0, forward_head_risk))
        down_head_risk = max(0.0, min(1.0, down_head_risk))

        # 混合基线 + 最大值独裁，确保任何单项恶化都会被放大反映
        visual_blend = 0.45 * base_visual_risk + 0.20 * eye_fatigue + 0.15 * neck_posture + 0.10 * forward_head_risk + 0.10 * down_head_risk
        visual_peak = max(base_visual_risk, eye_fatigue, neck_posture, forward_head_risk, down_head_risk)
        visual_risk = max(visual_blend, visual_peak)

        # 分类器输出不稳定时，启用“高敏兜底”（结合物理损耗）
        expr_lower = str(safe_expr).lower()
        physical_peak = max(eye_fatigue, neck_posture, forward_head_risk, down_head_risk)
        if expr_lower in ("none", "unknown", "未检测到人脸") and safe_prob >= 0.55 and physical_peak >= 0.18:
            visual_risk = max(visual_risk, 0.72)

        # 物理损耗触发阶梯升高（提高灵敏度）
        if physical_peak >= 0.35:
            visual_risk = max(visual_risk, 0.72)
        if physical_peak >= 0.50:
            visual_risk = max(visual_risk, 0.82)

        # 负向表情或可疑中性（置信度高）下，进一步提高下限
        if expr_lower in ("negative", "sad", "angry", "fear", "disgust") and safe_prob >= 0.45:
            visual_risk = max(visual_risk, 0.78)
        elif expr_lower == "neutral" and safe_prob >= 0.80 and physical_peak >= 0.22:
            visual_risk = max(visual_risk, 0.60)

        visual_risk = max(0.0, min(1.0, visual_risk))

        # 连续低置信帧时降低视觉分量权重，减少误报污染
        if self._vision_low_conf_streak >= self._vision_low_conf_apply_after:
            visual_risk *= self._vision_low_conf_weight_scale

        # 视觉->虚拟硬件动态联动：无真实硬件时，让HRV/EEG随表情实时变化
        expr_lower = str(safe_expr).lower()
        if expr_lower in ("happy", "smile", "neutral", "surprise"):
            self.sensor.set_simulation_state("normal")
        elif expr_lower in ("sad", "negative"):
            self.sensor.set_simulation_state("depressed")
        elif expr_lower in ("angry", "anger", "fear", "disgust", "contempt"):
            self.sensor.set_simulation_state("stressed")
        else:
            # 未检测到人脸或未知类别：回到中性，避免长期卡在高风险状态
            self.sensor.set_simulation_state("normal")

        # 2. 生理模态 HRV + 脑电模态 EEG
        hrv_data = self.sensor.get_hrv_data()
        eeg_data = self.external_eeg_data if isinstance(self.external_eeg_data, dict) else self.sensor.get_eeg_data()
        
        hrv_risk = calculate_hrv_risk(**hrv_data)
        eeg_risk = calculate_eeg_risk(
            attention=int(eeg_data.get('attention', 50)),
            meditation=int(eeg_data.get('meditation', 50))
        )

        # 3. 核心大融合计算（保留高风险敏感，同时保证恢复期可下降）
        average_risk = calculate_total_risk(visual_risk, hrv_risk, eeg_risk)

        # 峰值分量：捕捉突发恶化
        peak_risk = max(
            float(visual_risk),
            float(hrv_risk),
            float(eeg_risk),
            float(eye_fatigue),
            float(neck_posture),
            float(forward_head_risk),
            float(down_head_risk),
        )

        # 候选风险：平均+峰值混合，避免“max独裁”导致卡高
        candidate_risk = 0.70 * average_risk + 0.30 * peak_risk

        # 明确高危门控：仅当核心模态达到高阈值才强制抬升
        if max(float(visual_risk), float(hrv_risk), float(eeg_risk)) >= 0.80:
            candidate_risk = max(candidate_risk, peak_risk)

        # 非对称平滑：上升快、下降慢，但可持续回落
        rise_alpha = float(getattr(config, 'RISK_EMA_ALPHA_RISE', 0.55))
        fall_alpha = float(getattr(config, 'RISK_EMA_ALPHA_FALL', 0.25))
        if not self._risk_ema_inited:
            self._risk_ema = candidate_risk
            self._risk_ema_inited = True
        else:
            alpha = rise_alpha if candidate_risk >= self._risk_ema else fall_alpha
            self._risk_ema = (1.0 - alpha) * self._risk_ema + alpha * candidate_risk

        total_risk = max(0.0, min(1.0, float(self._risk_ema)))
        level = get_risk_level(total_risk)

        # 4. 社交电量算法：基于平滑后风险，放电/回充更稳定
        drain_th = float(getattr(config, 'BATTERY_DRAIN_THRESHOLD', 0.45))
        drain_step = float(getattr(config, 'BATTERY_DRAIN_STEP', 0.02))
        recover_th = float(getattr(config, 'BATTERY_RECOVER_THRESHOLD', 0.3))
        recover_step = float(getattr(config, 'BATTERY_RECOVER_STEP', 0.005))

        drain_n = int(getattr(config, 'BATTERY_DRAIN_CONSECUTIVE', 3))
        recover_n = int(getattr(config, 'BATTERY_RECOVER_CONSECUTIVE', 4))

        if total_risk > drain_th:
            self._battery_high_streak += 1
            self._battery_low_streak = 0
            if self._battery_high_streak >= max(1, drain_n):
                self.social_battery = max(0.0, self.social_battery - drain_step)
                self._battery_high_streak = 0
        elif total_risk < recover_th:
            self._battery_low_streak += 1
            self._battery_high_streak = 0
            if self._battery_low_streak >= max(1, recover_n):
                self.social_battery = min(1.0, self.social_battery + recover_step)
                self._battery_low_streak = 0
        else:
            self._battery_high_streak = 0
            self._battery_low_streak = 0

        self.battery_signal.emit(float(self.social_battery))

        if is_record and (self.recording_enabled or self.force_record_once):
            self.recorder.add_record(total_risk)
            self.recent_risks.append(total_risk)
            if len(self.recent_risks) > config.ALERT_CONSECUTIVE_HIGH:
                self.recent_risks.pop(0)
            if len(self.recent_risks) == config.ALERT_CONSECUTIVE_HIGH:
                if all(r >= config.RISK_THRESHOLD_MEDIUM for r in self.recent_risks):
                    self.alert_signal.emit()
            self.record_risk_signal.emit(total_risk, level)
            self.force_record_once = False
        else:
            # 实时更新逻辑：发送数据（使用修复后的 safe_prob）
            forward_head_risk = float(getattr(self.api_client, 'last_forward_head_risk', 0.0))
            down_head_risk = float(getattr(self.api_client, 'last_down_head_risk', 0.0))
            forward_head_risk = max(0.0, min(1.0, forward_head_risk))
            down_head_risk = max(0.0, min(1.0, down_head_risk))

            self.realtime_risk_signal.emit(
                total_risk,
                level,
                safe_expr,
                safe_prob,
                eye_fatigue,
                neck_posture,
                forward_head_risk,
                down_head_risk,
            )
            
            self.multimodal_data_signal.emit(
                visual_risk, hrv_risk, eeg_risk,
                eye_fatigue, neck_posture,
                float(hrv_data['rmssd']), float(hrv_data['hr']),
                int(eeg_data['attention']), int(eeg_data['meditation'])
            )

    def set_external_eeg_data(self, eeg_data: dict):
        if isinstance(eeg_data, dict):
            self.external_eeg_data = eeg_data

    def stop(self):
        self.running = False
        self.wait()
