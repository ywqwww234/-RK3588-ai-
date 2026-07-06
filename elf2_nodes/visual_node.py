"""
视觉感知节点 - 边缘监控终端·视觉分析模块 (Vision Edge Node)
- 独立运行的工业级 QMainWindow，可单独 `python visual_node.py` 启动
- 与心率节点(1.py) / 脑电节点(11.py) 同级独立节点
- 大画幅实时摄像头 + 8类表情概率 + 眼疲劳/姿态仪表 + 视觉风险评分
- 统一 NASA 深海军蓝主题 (#04111f + #00e5ff)
- Embeddable: 支持被 embedded_node.EmbeddedNodeOverlay 动态加载嵌入主系统
"""

import os, sys, time, random, math
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QImage, QPixmap, QLinearGradient
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSizePolicy, QTextEdit, QVBoxLayout,
                             QWidget, QFrame, QProgressBar)


# ============================================================
# 主题 - 与心率/脑电节点统一
# ============================================================
C_BG        = '#04111f'
C_BG_DEEP   = '#020912'
C_PANEL     = '#0a1f3d'
C_PANEL_HI  = '#13315c'
C_BORDER    = '#1c4076'
C_BORDER_HI = '#2a5ca8'
C_ACCENT    = '#00e5ff'
C_ACCENT2   = '#7c4dff'
C_VISION    = '#ffb74d'
C_OK        = '#26d07c'
C_WARN      = '#ffc857'
C_RISK      = '#ff3860'
C_TEXT      = '#eef1ff'
C_DIM       = '#a4b1d8'

EMOTION_NAMES = ['中性','高兴','惊讶','悲伤','愤怒','厌恶','恐惧','蔑视']
EMOTION_COLORS = ['#8899aa','#26d07c','#ffb74d','#448aff','#ff3860','#7c4dff','#ff8a3d','#e040fb']

QSS = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    background-color: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 10px; background-color: {C_BG}; color: {C_ACCENT}; letter-spacing: 1px;
}}
QLabel {{ color: {C_TEXT}; background: transparent; }}
QPushButton {{
    background-color: {C_PANEL_HI}; color: {C_ACCENT};
    border: 1px solid {C_ACCENT}; border-radius: 4px;
    padding: 5px 14px; font-weight: bold; min-width: 60px;
}}
QPushButton:hover {{ background-color: {C_ACCENT}; color: {C_BG}; }}
QPushButton:disabled {{ background-color: {C_PANEL}; color: {C_DIM}; border: 1px solid {C_DIM}; }}
QComboBox {{
    background-color: {C_PANEL_HI}; color: {C_TEXT};
    border: 1px solid {C_BORDER}; border-radius: 4px; padding: 4px 8px; min-width: 100px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QProgressBar {{
    background: {C_BG_DEEP}; border: 1px solid {C_BORDER}; border-radius: 3px;
    text-align: center; color: {C_TEXT}; font-size: 9px;
}}
QTextEdit {{
    background-color: #020912; color: {C_OK};
    border: 1px solid {C_BORDER}; border-radius: 4px;
    font-family: Consolas, "Courier New", monospace; font-size: 11px;
}}
"""


# ============================================================
# 圆环仪表 (眼疲劳 / 姿态风险)
# ============================================================
class CircularGauge(QWidget):
    def __init__(self, title, color=C_ACCENT, unit='%', parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.unit = unit
        self.value = 0
        self.setMinimumSize(120, 120)

    def set_value(self, v):
        self.value = max(0, min(100, int(v * 100)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 16
        rect = QRectF((self.width()-side)/2, (self.height()-side)/2, side, side)
        # 底环
        p.setPen(QPen(QColor(C_BORDER), 8, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 0, 360*16)
        # 进度环 (红色到绿色渐变)
        progress = self.value / 100.0
        if progress < 0.33: cfill = C_OK
        elif progress < 0.66: cfill = C_WARN
        else: cfill = C_RISK
        p.setPen(QPen(QColor(cfill), 8, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90*16, int(-self.value * 3.6 * 16))
        # 中心值
        p.setPen(QColor(C_TEXT))
        p.setFont(QFont('Consolas', max(14, int(side/5)), QFont.Bold))
        p.drawText(rect, Qt.AlignCenter, f'{self.value}{self.unit}')
        # 标题
        p.setPen(QColor(C_DIM)); p.setFont(QFont('Microsoft YaHei', 9))
        tr = QRectF(rect.x(), rect.y()+rect.height()*0.7, rect.width(), 20)
        p.drawText(tr, Qt.AlignCenter, self.title)


# ============================================================
# 状态胶囊
# ============================================================
class StatusPill(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_state('offline', '未连接')
    def set_state(self, state, text):
        colors = {'live': (C_OK, '#0d2818'), 'sim': (C_WARN, '#2e1f05'), 'offline': (C_RISK, '#2a0a0e')}
        fg, bg = colors.get(state, colors['offline'])
        self.setText(f'  ●  {text}  ')
        self.setStyleSheet(f'background:{bg}; color:{fg}; border:1px solid {fg}; '
                          f'border-radius:10px; padding:4px 12px; font-weight:bold;')


# ============================================================
# 表情概率条组
# ============================================================
class EmotionProbBars(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs = [0.0] * 8
        self.setMinimumHeight(180)
    def set_probs(self, probs):
        if probs is None or len(probs) < 8: return
        self._probs = list(probs[:8])
        self.update()
    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        n = len(self._probs)
        bar_h = max(8, (H - 4) / n - 4)
        max_w = W - 90
        for i in range(n):
            y = 4 + i * (bar_h + 4)
            prob = self._probs[i]
            bw = int(max_w * prob)
            # 底条
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(C_BG_DEEP)))
            p.drawRoundedRect(65, int(y), int(max_w), int(bar_h), 2, 2)
            # 概率条
            if bw > 0:
                color = QColor(EMOTION_COLORS[i])
                grad = QLinearGradient(65, y, 65 + bw, y)
                grad.setColorAt(0.0, QColor(color).lighter(120))
                grad.setColorAt(1.0, color)
                p.setBrush(QBrush(grad))
                p.drawRoundedRect(65, int(y), bw, int(bar_h), 2, 2)
            # 标签
            p.setPen(QColor(EMOTION_COLORS[i]))
            p.setFont(QFont('Microsoft YaHei', 9))
            p.drawText(QRectF(0, y, 62, bar_h), Qt.AlignRight | Qt.AlignVCenter, EMOTION_NAMES[i])
            # 百分比
            p.setPen(QColor(C_DIM))
            p.setFont(QFont('Consolas', 8))
            p.drawText(QRectF(65 + max_w + 4, y, 24, bar_h), Qt.AlignLeft | Qt.AlignVCenter,
                      f'{int(prob*100)}%')


# ============================================================
# 主窗口: VisionMonitorWindow
# ============================================================
class VisionMonitorWindow(QMainWindow):

    MODE_OFFLINE = 'offline'
    MODE_CAMERA = 'camera'
    MODE_SIM = 'sim'

    def __init__(self):
        super().__init__()
        self.mode = self.MODE_OFFLINE
        self.cap = None
        self.analyzer = None

        # 当前值快照
        self.current_expr = 'neutral'
        self.current_prob = 0.0
        self.current_eye_fatigue = 0.0
        self.current_posture_risk = 0.0
        self.current_visual_risk = 0.0
        self.expression_probs = [0.0] * 8

        # 模拟状态
        self._sim_phase = 0.0
        self._sim_expr_idx = 0
        self._sim_timer = 0.0

        # 缓冲
        self._frame_count = 0
        self._last_analysis_t = 0.0
        self._dbg_lines = []

        self.init_ui()

        # 定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(50)  # 20fps display

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._capture_tick)
        self.capture_timer.start(33)  # 30fps capture

        self.watchdog_timer = QTimer()
        self.watchdog_timer.timeout.connect(self._watchdog)
        self.watchdog_timer.start(2000)

        QTimer.singleShot(500, self._auto_start)

    # ================ UI ================
    def init_ui(self):
        self.setWindowTitle('边缘监控终端 · 视觉感知节点 (Vision Edge Node)')
        self.setGeometry(40, 40, 1440, 900)
        self.setStyleSheet(QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10); root.setSpacing(8)

        # === 顶部标题栏 ===
        head = QHBoxLayout()
        title = QLabel('👁  视 觉 感 知 节 点  ·  VISION  EDGE  NODE')
        title.setStyleSheet(f'color:{C_VISION}; font-size:18px; font-weight:bold; letter-spacing:3px;')
        head.addWidget(title); head.addStretch()
        self.pill_status = StatusPill(); head.addWidget(self.pill_status)
        self.pill_quality = StatusPill(); self.pill_quality.set_state('offline', '分析: --')
        head.addWidget(self.pill_quality)
        root.addLayout(head)

        # === 控制条 ===
        ctrl_box = QGroupBox('采集控制 · ACQUISITION')
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('摄像头:'))
        self.cam_combo = QComboBox(); self.cam_combo.setMinimumWidth(120)
        self.cam_combo.addItems(['0 (默认)', '1', '2'])
        ctrl.addWidget(self.cam_combo)
        self.btn_refresh = QPushButton('刷新'); self.btn_refresh.clicked.connect(self._refresh_cams)
        ctrl.addWidget(self.btn_refresh)
        self.btn_connect = QPushButton('启动采集'); self.btn_connect.clicked.connect(self._toggle_camera)
        ctrl.addWidget(self.btn_connect)
        self.btn_sim = QPushButton('进入模拟'); self.btn_sim.clicked.connect(self._enter_simulation)
        ctrl.addWidget(self.btn_sim)
        self.chk_overlay = QCheckBox('调试叠加'); self.chk_overlay.setChecked(True)
        self.chk_overlay.setStyleSheet(f'color:{C_TEXT};')
        ctrl.addWidget(self.chk_overlay)
        ctrl.addStretch()
        self.lbl_fps = QLabel('FPS: --'); self.lbl_fps.setStyleSheet(f'color:{C_DIM};')
        ctrl.addWidget(self.lbl_fps)
        ctrl_box.setLayout(ctrl)
        root.addWidget(ctrl_box)

        # === 主体: 左摄像头大画幅 + 右分析面板 ===
        body = QHBoxLayout(); body.setSpacing(10)

        # ---- 左: 摄像头 (占 60%) ----
        left_panel = QVBoxLayout(); left_panel.setSpacing(6)

        cam_group = QGroupBox('实时画面 · LIVE FEED')
        cam_lay = QVBoxLayout()
        cam_lay.setContentsMargins(4, 4, 4, 4)
        # 大画幅视频标签
        self.video_frame = QLabel('等待摄像头启动…')
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setScaledContents(True)
        self.video_frame.setStyleSheet(
            f'background:{C_BG_DEEP}; color:{C_VISION}; font-size:16px; '
            f'border:2px solid {C_BORDER}; border-radius:6px;')
        cam_lay.addWidget(self.video_frame, 1)
        cam_group.setLayout(cam_lay)
        left_panel.addWidget(cam_group, 3)

        # 表情概率条
        expr_group = QGroupBox('表情分类 · EXPRESSION (FER+)')
        expr_lay = QVBoxLayout(); expr_lay.setContentsMargins(8, 8, 8, 4)
        self.emotion_bars = EmotionProbBars()
        expr_lay.addWidget(self.emotion_bars)
        expr_group.setLayout(expr_lay)
        left_panel.addWidget(expr_group, 1)

        body.addLayout(left_panel, 6)

        # ---- 右: 分析面板 (占 40%) ----
        right_col = QVBoxLayout(); right_col.setSpacing(8)

        # 眼疲劳仪表
        eye_group = QGroupBox('眼疲劳监测 · EYE FATIGUE')
        eye_lay = QVBoxLayout()
        eye_row = QHBoxLayout()
        self.gauge_eye = CircularGauge('眼疲劳指数', C_WARN, '%')
        self.gauge_blink = CircularGauge('眨眼频率', C_ACCENT, '/min')
        eye_row.addWidget(self.gauge_eye); eye_row.addWidget(self.gauge_blink)
        eye_lay.addLayout(eye_row)

        # EAR 实时值
        self.lbl_ear = QLabel('EAR: --')
        self.lbl_ear.setStyleSheet(f'color:{C_DIM}; font-size:11px; letter-spacing:1px;')
        self.lbl_ear.setAlignment(Qt.AlignCenter)
        eye_lay.addWidget(self.lbl_ear)
        eye_group.setLayout(eye_lay)
        right_col.addWidget(eye_group)

        # 姿态风险仪表
        post_group = QGroupBox('姿态分析 · POSTURE')
        post_lay = QVBoxLayout()
        post_row = QHBoxLayout()
        self.gauge_posture = CircularGauge('姿态风险', C_RISK, '%')
        self.gauge_down = CircularGauge('低头角', C_WARN, '°')
        post_row.addWidget(self.gauge_posture); post_row.addWidget(self.gauge_down)
        post_lay.addLayout(post_row)

        # 探头/低头分解
        self.lbl_forward = QLabel('探头风险: --')
        self.lbl_forward.setStyleSheet(f'color:{C_DIM}; font-size:11px; letter-spacing:1px;')
        self.lbl_forward.setAlignment(Qt.AlignCenter)
        self.lbl_down = QLabel('低头风险: --')
        self.lbl_down.setStyleSheet(f'color:{C_DIM}; font-size:11px; letter-spacing:1px;')
        self.lbl_down.setAlignment(Qt.AlignCenter)
        post_lay.addWidget(self.lbl_forward); post_lay.addWidget(self.lbl_down)
        post_group.setLayout(post_lay)
        right_col.addWidget(post_group)

        # 视觉风险评分
        risk_group = QGroupBox('视觉风险评分 · VISUAL RISK INDEX')
        rlay = QVBoxLayout()
        self.risk_value = QLabel('0.00')
        self.risk_value.setAlignment(Qt.AlignCenter)
        self.risk_value.setStyleSheet(f'color:{C_OK}; font-size:48px; font-weight:bold; background:transparent;')
        self.risk_label = QLabel('低 · LOW')
        self.risk_label.setAlignment(Qt.AlignCenter)
        self.risk_label.setStyleSheet(f'color:{C_DIM}; font-size:13px; letter-spacing:3px;')
        risk_bar = QLabel(); risk_bar.setFixedHeight(6)
        risk_bar.setStyleSheet(
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_OK}, stop:0.5 {C_WARN}, stop:1 {C_RISK}); border-radius:3px;')
        rlay.addWidget(self.risk_value); rlay.addWidget(self.risk_label)
        rlay.addSpacing(4); rlay.addWidget(risk_bar)

        # 风险分解
        sub_row = QHBoxLayout()
        self.lbl_expr_risk = QLabel('表情: --'); self.lbl_expr_risk.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        self.lbl_eye_risk = QLabel('眼疲劳: --'); self.lbl_eye_risk.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        self.lbl_post_risk = QLabel('姿态: --'); self.lbl_post_risk.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        sub_row.addWidget(self.lbl_expr_risk); sub_row.addWidget(self.lbl_eye_risk)
        sub_row.addWidget(self.lbl_post_risk)
        rlay.addLayout(sub_row)
        risk_group.setLayout(rlay)
        right_col.addWidget(risk_group)

        right_col.addStretch()
        body.addLayout(right_col, 4)
        root.addLayout(body, 1)

        # === 底部日志 ===
        log_group = QGroupBox('系统日志 · LOG')
        log_lay = QVBoxLayout()
        self.log_text = QTextEdit(); self.log_text.setMaximumHeight(80); self.log_text.setReadOnly(True)
        log_lay.addWidget(self.log_text)
        log_group.setLayout(log_lay)
        root.addWidget(log_group)

    # ================ 采集 ================
    def _auto_start(self):
        """自动尝试打开默认摄像头，失败则进入模拟。"""
        try:
            self._open_camera(0)
            self.log('已自动启动默认摄像头')
        except Exception:
            self._enter_simulation('摄像头不可用，已切换本地模拟数据模式')

    def _refresh_cams(self):
        self.cam_combo.clear()
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cam_combo.addItem(f'{i} (可用)')
                cap.release()
            else:
                self.cam_combo.addItem(f'{i}')
        self.log('摄像头列表已刷新')

    def _toggle_camera(self):
        if self.mode == self.MODE_CAMERA:
            self._release_camera()
            self.mode = self.MODE_OFFLINE
            self.btn_connect.setText('启动采集')
            self.pill_status.set_state('offline', '采集: 离线')
            return
        idx = self.cam_combo.currentIndex()
        try:
            self._open_camera(idx)
        except Exception as e:
            self.log(f'打开摄像头失败: {e}')
            self._enter_simulation('摄像头打开失败，已切换模拟模式')

    def _open_camera(self, idx):
        if self.cap is not None:
            self._release_camera()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f'摄像头 {idx} 无法打开')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.mode = self.MODE_CAMERA
        self.btn_connect.setText('停止采集')
        self.pill_status.set_state('live', '采集: 实时')
        self.log(f'已启动摄像头 #{idx}')

        # 尝试加载视觉分析器
        if self.analyzer is None:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '0', 'Anti_depression'))
                from local_vision import LocalFaceAnalyzer
                self.analyzer = LocalFaceAnalyzer()
                self.pill_quality.set_state('live', '分析: FER+ / MediaPipe')
                self.log('视觉分析器已加载')
            except Exception as e:
                self.pill_quality.set_state('sim', f'分析: 降级 ({e})')
                self.log(f'视觉分析器加载失败，使用模拟分析: {e}')

    def _release_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.cap = None

    def _enter_simulation(self, reason=''):
        self._release_camera()
        self.mode = self.MODE_SIM
        self.btn_connect.setText('启动采集')
        self.btn_sim.setText('退出模拟')
        self.pill_status.set_state('sim', '信号源: 本地模拟')
        self.pill_quality.set_state('sim', '分析: 模拟')
        self._sim_phase = random.random() * 100
        if reason: self.log(reason)

    def _capture_tick(self):
        """每帧采集: 摄像头或模拟。"""
        if self.mode == self.MODE_CAMERA and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self._frame_count += 1
                # 每2秒分析一次
                now = time.time()
                if now - self._last_analysis_t >= 2.0:
                    self._last_analysis_t = now
                    self._analyze_frame(frame)
                # 显示(带叠加)
                if self.chk_overlay.isChecked():
                    frame = self._draw_overlay(frame)
                self._show_frame(frame)
        elif self.mode == self.MODE_SIM:
            self._generate_simulated()

    def _analyze_frame(self, frame):
        """调用视觉分析器(或模拟分析)。"""
        if self.analyzer is not None:
            try:
                ret = self.analyzer.analyze_face(frame)
                if isinstance(ret, (tuple, list)):
                    self.current_expr = str(ret[0]) if len(ret) > 0 and ret[0] is not None else 'none'
                    self.current_prob = float(ret[1]) if len(ret) > 1 and ret[1] is not None else 0.0
                    self.current_eye_fatigue = float(ret[2]) if len(ret) > 2 and ret[2] is not None else 0.0
                    self.current_posture_risk = float(ret[3]) if len(ret) > 3 and ret[3] is not None else 0.0
                # 生成模拟表情概率(基于检测到的表情)
                self.expression_probs = self._make_expr_probs(self.current_expr, self.current_prob)
                # 计算视觉风险
                self.current_visual_risk = self._calc_visual_risk()
            except Exception as e:
                self._do_sim_analysis()
        else:
            self._do_sim_analysis()

    def _do_sim_analysis(self):
        """模拟分析(无真实分析器时)。"""
        t = time.time()
        phase = self._sim_phase + t * 0.3
        # 模拟表情
        if t - self._sim_timer > 3.0:
            self._sim_expr_idx = (self._sim_expr_idx + random.randint(0, 2)) % 8
            self._sim_timer = t
        self.current_expr = EMOTION_NAMES[self._sim_expr_idx].lower() if self._sim_expr_idx < 8 else 'neutral'
        self.current_prob = 0.45 + random.uniform(0, 0.45)
        self.current_eye_fatigue = abs(math.sin(phase * 0.7)) * 0.4 + random.uniform(0, 0.15)
        self.current_posture_risk = abs(math.cos(phase * 0.5)) * 0.35 + random.uniform(0, 0.1)
        self.expression_probs = self._make_expr_probs(self.current_expr, self.current_prob)
        self.current_visual_risk = self._calc_visual_risk()

    def _make_expr_probs(self, expr, prob):
        probs = [0.0] * 8
        try:
            idx = [e.lower() for e in EMOTION_NAMES].index(str(expr).lower())
        except ValueError:
            idx = 0
        probs[idx] = prob
        remaining = 1.0 - prob
        for i in range(8):
            if i != idx:
                probs[i] = remaining * random.uniform(0.05, 0.3)
        s = sum(probs) or 1.0
        return [p/s for p in probs]

    def _calc_visual_risk(self):
        """视觉风险计算(与 risk_calculator 逻辑一致)。"""
        expr_lower = str(self.current_expr).lower()
        prob = self.current_prob
        if expr_lower in ('happy', 'happily'):
            expr_risk = 0.08 + 0.08 * (1 - prob)
        elif expr_lower in ('neutral', 'surprise', 'contempt'):
            expr_risk = 0.20 + 0.15 * (1 - prob)
        else:
            expr_risk = 0.20 + prob * 0.85
        eye = self.current_eye_fatigue
        post = self.current_posture_risk
        blend = 0.45 * expr_risk + 0.20 * eye + 0.15 * post + 0.10 * max(eye, post)
        peak = max(expr_risk, eye, post)
        risk = max(blend, peak)
        if eye > 0.35: risk = max(risk, 0.72)
        if eye > 0.50: risk = max(risk, 0.82)
        if expr_lower in ('sad', 'angry', 'anger', 'fear', 'disgust') and prob > 0.45:
            risk = max(risk, 0.78)
        return max(0.0, min(1.0, risk))

    def _generate_simulated(self):
        """生成模拟摄像头画面。"""
        self._frame_count += 1
        t = time.time()
        w, h = 800, 600
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # 渐变背景
        for y in range(h):
            v = int(15 + 12 * y / h)
            img[y, :] = [v, v+3, v+8]
        # 模拟人脸框
        cx, cy = w//2 + int(20*math.sin(t*0.5)), h//2 + int(15*math.cos(t*0.3))
        cv2.ellipse(img, (cx, cy), (80, 110), 0, 0, 360, (40, 60, 100), 2)
        cv2.circle(img, (cx-30, cy-30), 8, (40, 60, 100), 2)
        cv2.circle(img, (cx+30, cy-30), 8, (40, 60, 100), 2)
        # 模拟嘴
        mouth_y = cy + 40
        if 'happy' in str(self.current_expr).lower():
            cv2.ellipse(img, (cx, mouth_y), (25, 15), 0, 10, 170, (40, 100, 60), 2)
        elif str(self.current_expr).lower() in ('sad', 'negative'):
            cv2.ellipse(img, (cx, mouth_y+10), (25, 12), 0, 190, 350, (60, 40, 100), 2)
        else:
            cv2.line(img, (cx-25, mouth_y), (cx+25, mouth_y), (40, 60, 100), 2)
        # 叠加调试信息
        self._frame_count += 1
        now = time.time()
        if now - self._last_analysis_t >= 2.0:
            self._last_analysis_t = now
            self._do_sim_analysis()
        if self.chk_overlay.isChecked():
            img = self._draw_overlay(img)
        self._show_frame(img)

    def _draw_overlay(self, frame):
        """叠加调试信息到画面上。"""
        h, w = frame.shape[:2]
        # 半透明遮罩背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 6), (560, 108), (2, 9, 18), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.rectangle(frame, (8, 6), (560, 108), (0, 229, 255), 1)

        lines = [
            f"MODE:{self.mode.upper()}  |  EXPR:{self.current_expr}({self.current_prob:.2f})",
            f"EYE_FATIGUE:{self.current_eye_fatigue:.2f}  |  POSTURE:{self.current_posture_risk:.2f}",
            f"VISUAL_RISK:{self.current_visual_risk:.3f}  |  FPS:{self._frame_count/max(1,time.time()-self._last_analysis_t+0.01):.0f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (16, 32 + i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1, cv2.LINE_AA)
        return frame

    def _show_frame(self, frame):
        """显示帧到video_label。"""
        h, w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, w*3, QImage.Format_BGR888)
        self.video_frame.setPixmap(QPixmap.fromImage(qimg))

    # ================ 显示刷新 ================
    def _update_display(self):
        # 眼疲劳仪表
        self.gauge_eye.set_value(self.current_eye_fatigue)
        blink_rate = int(12 + random.uniform(-3, 3))  # 模拟眨眼
        self.gauge_blink.set_value(blink_rate / 30.0)  # 归一化

        # EAR标签(模拟)
        if self.analyzer is not None and hasattr(self.analyzer, 'last_ear') and self.analyzer.last_ear is not None:
            ear = self.analyzer.last_ear
            ear_color = C_OK if ear > 0.22 else (C_WARN if ear > 0.18 else C_RISK)
            self.lbl_ear.setText(f'EAR: {ear:.3f}')
            self.lbl_ear.setStyleSheet(f'color:{ear_color}; font-size:11px; letter-spacing:1px;')
        else:
            ear = 0.32 - self.current_eye_fatigue * 0.25
            self.lbl_ear.setText(f'EAR: {ear:.3f} (sim)')

        # 姿态仪表
        self.gauge_posture.set_value(self.current_posture_risk)
        down_deg = 18.0 * self.current_posture_risk
        self.gauge_down.set_value(down_deg / 45.0)
        self.lbl_forward.setText(f'探头风险: {self.current_posture_risk*0.45:.2f}')
        self.lbl_down.setText(f'低头风险: {self.current_posture_risk*0.55:.2f}')

        # 表情概率条
        self.emotion_bars.set_probs(self.expression_probs)

        # 视觉风险评分
        risk = self.current_visual_risk
        self.risk_value.setText(f'{risk:.2f}')
        if risk > 0.66: c, tag = C_RISK, '高 · HIGH'
        elif risk > 0.33: c, tag = C_WARN, '中 · MEDIUM'
        else: c, tag = C_OK, '低 · LOW'
        self.risk_value.setStyleSheet(f'color:{c}; font-size:48px; font-weight:bold; background:transparent;')
        self.risk_label.setText(tag)
        self.risk_label.setStyleSheet(f'color:{c}; font-size:13px; letter-spacing:3px;')

        # 风险分解
        self.lbl_expr_risk.setText(f'表情: {0.45*risk:.3f}')
        self.lbl_eye_risk.setText(f'眼疲劳: {0.20*self.current_eye_fatigue:.3f}')
        self.lbl_post_risk.setText(f'姿态: {0.15*self.current_posture_risk:.3f}')

        # FPS
        self.lbl_fps.setText(f'FPS: {self._frame_count // max(1, int(time.time()-self._last_analysis_t+0.01))}')

    def _watchdog(self):
        """摄像头健康检查。"""
        if self.mode == self.MODE_CAMERA and (self.cap is None or not self.cap.isOpened()):
            self.log('摄像头连接丢失，已切换模拟')
            self._enter_simulation('摄像头连接丢失')

    # ================ 日志 ================
    def log(self, message):
        ts = time.strftime('%H:%M:%S')
        self.log_text.append(f'<span style="color:{C_DIM}">[{ts}]</span> {message}')

    # ================ 关闭 ================
    def shutdown(self):
        self.update_timer.stop()
        self.capture_timer.stop()
        self.watchdog_timer.stop()
        self._release_camera()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


# ============================================================
# 入口
# ============================================================
def main():
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication([]); app.setStyle('Fusion')
    window = VisionMonitorWindow(); window.show()
    app.exec_()

if __name__ == '__main__':
    main()
