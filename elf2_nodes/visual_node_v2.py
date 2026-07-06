"""
视觉感知节点 v2 — 预测性可视化 (Predictive Vision Edge Node)
============================================================
从"展示7种表情概率条"转型为"预测性风险可视化"

核心转变:
  v1: 静态展示 → 7情绪概率条 + 眼疲劳/姿态仪表 + 当前风险值
  v2: 预测性展示 → 风险轨迹预测曲线 + 速度仪表 + NN注意力热力图 + 干预就绪指示

设计原理:
  1. 风险轨迹预测: 当前值 + 5min/15min外推 + 置信区间
  2. 风险速度仪表: 量化为 -100~+100 的"风险加速度计"
  3. 时间注意力热力图: 60秒窗口中哪些时刻是关键
  4. 干预就绪度: 四级色标 (绿=静默/黄=感知/橙=守护/红=预警)
  5. 主动感知日志: 替代被动"悄悄说"的主动探询记录
"""

import os, sys, time, math, random
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QBrush,
                         QImage, QPixmap, QLinearGradient, QRadialGradient)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
                             QFrame, QGridLayout, QScrollArea)


# ============================================================
# 主题
# ============================================================
C_BG        = '#04111f'
C_BG_DEEP   = '#020912'
C_PANEL     = '#0a1f3d'
C_PANEL_HI  = '#13315c'
C_BORDER    = '#1c4076'
C_ACCENT    = '#00e5ff'
C_VISION    = '#ffb74d'
C_HEART     = '#ff5277'
C_BRAIN     = '#b388ff'
C_OK        = '#26d07c'
C_WARN      = '#ffc857'
C_RISK_HIGH = '#ff8a3d'
C_RISK_CRIT = '#ff3860'
C_TEXT      = '#eef1ff'
C_DIM       = '#a4b1d8'

QSS = f"""
QMainWindow, QWidget {{ background:{C_BG}; color:{C_TEXT};
    font-family:"Microsoft YaHei","PingFang SC",sans-serif; font-size:11px; }}
QGroupBox {{ background:{C_PANEL}; border:1px solid {C_BORDER};
    border-radius:6px; margin-top:14px; padding-top:8px; font-weight:bold; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left;
    padding:2px 10px; background:{C_BG}; color:{C_ACCENT}; letter-spacing:1px; }}
QLabel {{ color:{C_TEXT}; background:transparent; }}
QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};
    border:1px solid {C_ACCENT}; border-radius:4px; padding:5px 14px; font-weight:bold; }}
QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG}; }}
QTextEdit {{ background:{C_BG_DEEP}; color:{C_OK}; border:1px solid {C_BORDER};
    border-radius:4px; font-family:Consolas; font-size:10px; }}
"""


# ============================================================
# 风险速度仪表 (Risk Velocity Gauge)
# ============================================================
class RiskVelocityGauge(QWidget):
    """半圆形速度计: -100(快速下降) 到 +100(快速上升)。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.velocity = 0.0   # -100 to +100
        self.accel = 0.0
        self.setMinimumSize(180, 120)

    def set_values(self, vel, acc=0.0):
        self.velocity = max(-100, min(100, vel))
        self.accel = max(-50, min(50, acc))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W/2, H*0.78
        r = min(W, H) * 0.72

        # 半圆弧背景
        rect = QRectF(cx-r, cy-r, 2*r, 2*r)
        p.setPen(QPen(QColor(C_BORDER), 10, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 0, 180*16)

        # 渐变弧: 绿(左)→黄(中)→红(右)
        grad = QLinearGradient(cx-r, cy, cx+r, cy)
        grad.setColorAt(0.0, QColor(C_OK))
        grad.setColorAt(0.35, QColor(C_OK))
        grad.setColorAt(0.55, QColor(C_WARN))
        grad.setColorAt(0.75, QColor(C_RISK_HIGH))
        grad.setColorAt(1.0, QColor(C_RISK_CRIT))
        p.setPen(QPen(QBrush(grad), 10, Qt.SolidLine, Qt.RoundCap))
        # 映射 velocity [-100,100] → angle [0,180]
        angle = int((self.velocity + 100) / 200.0 * 180)
        p.drawArc(rect, 0, angle * 16)

        # 指针
        rad = math.radians(90 - angle)
        nx, ny = cx + (r-16)*math.cos(rad), cy - (r-16)*math.sin(rad)
        p.setPen(QPen(QColor(C_ACCENT), 3))
        p.drawLine(int(cx), int(cy-r*0.15), int(nx), int(ny))

        # 中心标签
        p.setPen(QColor(C_TEXT))
        p.setFont(QFont('Consolas', 11, QFont.Bold))
        label = '→ 平稳' if abs(self.velocity) < 10 else (
            f'↗ 上升 {self.velocity:.0f}' if self.velocity > 0 else f'↘ 下降 {abs(self.velocity):.0f}')
        p.drawText(QRectF(cx-70, cy-35, 140, 22), Qt.AlignCenter, label)

        # 底部标识
        p.setPen(QColor(C_DIM)); p.setFont(QFont('Microsoft YaHei', 8))
        p.drawText(QRectF(cx-r, cy-5, 2*r, 16), Qt.AlignCenter, '风险速度 · VELOCITY')


# ============================================================
# 干预就绪度指示器 (Intervention Readiness)
# ============================================================
class InterventionReadiness(QWidget):
    """四级同心圆: 绿/黄/橙/红逐级亮起。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0  # 0=静默 1=感知 2=守护 3=预警
        self.setMinimumSize(100, 100)

    def set_level(self, level):
        self.level = max(0, min(3, level))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width()/2, self.height()/2
        colors = [C_OK, C_WARN, C_RISK_HIGH, C_RISK_CRIT]
        labels = ['静默', '感知', '守护', '预警']

        for i in range(4):
            r = 44 - i * 10
            alpha = 255 if i <= self.level else 50
            c = QColor(colors[i]); c.setAlpha(alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c))
            p.drawEllipse(int(cx-r), int(cy-r), int(2*r), int(2*r))
        # 中心文字
        p.setPen(QColor(colors[self.level]))
        p.setFont(QFont('Microsoft YaHei', 11, QFont.Bold))
        p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignCenter,
                   labels[self.level])

        # 底部标签
        p.setPen(QColor(C_DIM)); p.setFont(QFont('Microsoft YaHei', 7))
        p.drawText(QRectF(0, self.height()-16, self.width(), 14), Qt.AlignCenter,
                   '干预就绪度')


# ============================================================
# 时间注意力热力图 (Temporal Attention Heatmap)
# ============================================================
class AttentionHeatmap(QWidget):
    """60帧×1列的垂直热力图，展示NN注意力在各秒的分布。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._attn = np.ones(60) / 60
        self.setMinimumWidth(28)
        self.setMinimumHeight(240)

    def set_attention(self, attn):
        if attn is None: return
        self._attn = np.asarray(attn[-60:], dtype=np.float64) if len(attn) > 60 else np.asarray(attn, dtype=np.float64)
        self._attn = self._attn / (self._attn.sum() + 1e-6)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width() - 4, self.height() - 8
        n = len(self._attn)
        cell_h = H / max(1, n)

        max_a = self._attn.max() if len(self._attn) > 0 else 1.0
        for i in range(n):
            y = int(4 + i * cell_h)
            intensity = float(self._attn[i]) / max(1e-6, max_a)
            # 蓝→黄→红渐变
            r = int(0 + intensity * 255)
            g = int(50 + intensity * 150)
            b = int(180 - intensity * 140)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(r, g, b, 200)))
            p.drawRect(2, y, int(W), max(2, int(cell_h + 1)))

        # 边框
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(2, 4, int(W), int(H))
        # 标签
        p.setPen(QColor(C_DIM)); p.setFont(QFont('Microsoft YaHei', 7))
        p.drawText(QRectF(0, H+2, W+4, 14), Qt.AlignCenter, '注意')


# ============================================================
# 风险轨迹预测曲线
# ============================================================
class RiskTrajectoryPlot(QWidget):
    """pyqtgraph 曲线: 历史风险 + 预测外推 + 置信区间。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(C_PANEL)
        self.plot.showGrid(x=True, y=True, alpha=0.1)
        self.plot.setLabel('left', 'Risk', color=C_DIM)
        self.plot.setLabel('bottom', 'Time (min)', color=C_DIM)
        self.plot.setYRange(0, 1, padding=0)
        for ax in ('left', 'bottom'):
            self.plot.getAxis(ax).setPen(C_BORDER)
            self.plot.getAxis(ax).setTextPen(C_DIM)
        # 参考线
        for y, c, ls, label in [(0.3, C_OK, Qt.DashLine, 'L1'), (0.6, C_WARN, Qt.DashLine, 'L2'),
                                  (0.8, C_RISK_CRIT, Qt.DashLine, 'L3')]:
            li = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(c, style=ls, width=0.8))
            self.plot.addItem(li)
        # 曲线
        self.curve_hist = self.plot.plot(pen=pg.mkPen(C_ACCENT, width=2), name='历史')
        self.curve_pred = self.plot.plot(pen=pg.mkPen(C_WARN, width=2, style=Qt.DashLine), name='预测')
        self.fill_pred = None
        lay.addWidget(self.plot)
        self._hist = deque(maxlen=120)
        self._current_risk = 0.0
        self._pred_5m = 0.0
        self._pred_15m = 0.0
        self._conf = 0.0

    def update_data(self, risk, pred5, pred15, confidence):
        self._hist.append(risk)
        self._current_risk = risk
        self._pred_5m = pred5
        self._pred_15m = pred15
        self._conf = confidence
        self._redraw()

    def _redraw(self):
        n = len(self._hist)
        if n < 2: return
        # 历史: x轴(分钟), 当前=0
        hist_x = [-(n - 1 - i)/60.0 for i in range(n)]
        hist_y = list(self._hist)
        self.curve_hist.setData(hist_x, hist_y)

        # 预测: 当前→5分钟后→15分钟后
        pred_x = [0, 5, 15]
        pred_y = [self._current_risk, self._pred_5m, self._pred_15m]
        self.curve_pred.setData(pred_x, pred_y)

        self.plot.setXRange(-2, 18, padding=0)


# ============================================================
# 主窗口: PredictiveVisionWindow
# ============================================================
class PredictiveVisionWindow(QMainWindow):

    MODE_OFFLINE = 'offline'
    MODE_CAMERA = 'camera'
    MODE_SIM = 'sim'

    def __init__(self):
        super().__init__()
        self.mode = self.MODE_OFFLINE
        self.cap = None
        self.analyzer = None
        self._frame_count = 0
        self._last_analysis_t = 0.0

        # 风险状态
        self.current_risk = 0.0
        self.current_tier = 0
        self.risk_velocity = 0.0
        self.risk_acceleration = 0.0
        self.predicted_risk_5m = 0.0
        self.predicted_risk_15m = 0.0
        self.prediction_confidence = 0.0
        self.guardian_level = 0
        self._attn = np.ones(60) / 60

        # 主动感知日志
        self._sensing_log = deque(maxlen=20)

        # 模拟
        self._sim_t = 0.0
        self._sim_risk = 0.15
        self._sim_vel = 0.0
        self._sim_phase = random.random() * 100

        self.init_ui()

        self.update_timer = QTimer(); self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(50)
        self.capture_timer = QTimer(); self.capture_timer.timeout.connect(self._capture_tick)
        self.capture_timer.start(33)

        QTimer.singleShot(500, self._auto_start)

    # ================ UI ================
    def init_ui(self):
        self.setWindowTitle('边缘监控终端 · 预测性视觉感知节点 (Predictive Vision v2)')
        self.setGeometry(40, 40, 1500, 920)
        self.setStyleSheet(QSS)

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10); root.setSpacing(8)

        # === 顶栏 ===
        head = QHBoxLayout()
        title = QLabel('👁  预 测 性 视 觉 感 知  ·  PREDICTIVE  VISION  EDGE')
        title.setStyleSheet(f'color:{C_VISION}; font-size:17px; font-weight:bold; letter-spacing:2px;')
        head.addWidget(title); head.addStretch()
        # 就绪度指示
        self.readiness = InterventionReadiness();head.addWidget(self.readiness)
        head.addSpacing(12)
        head.addWidget(self._pill('', C_DIM, ''))
        root.addLayout(head)

        # === 主体: 左5(摄像头+轨迹) / 右5(仪表+注意力+日志) ===
        body = QHBoxLayout(); body.setSpacing(10)

        # ---- 左列 ----
        left = QVBoxLayout(); left.setSpacing(8)

        # 摄像头
        cam_g = QGroupBox('实时画面 · LIVE FEED')
        cam_l = QVBoxLayout(); cam_l.setContentsMargins(4,4,4,4)
        self.video_frame = QLabel('摄像头启动中…')
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(560, 380)
        self.video_frame.setScaledContents(True)
        self.video_frame.setStyleSheet(
            f'background:{C_BG_DEEP}; color:{C_VISION}; font-size:14px; '
            f'border:2px solid {C_BORDER}; border-radius:6px;')
        cam_l.addWidget(self.video_frame, 1)
        cam_g.setLayout(cam_l)
        left.addWidget(cam_g, 3)

        # 风险轨迹预测曲线
        traj_g = QGroupBox('风险轨迹预测 · RISK TRAJECTORY (历史+预测外推)')
        traj_l = QVBoxLayout(); traj_l.setContentsMargins(6,6,6,4)
        self.trajectory_plot = RiskTrajectoryPlot()
        traj_l.addWidget(self.trajectory_plot)
        traj_g.setLayout(traj_l)
        left.addWidget(traj_g, 2)

        body.addLayout(left, 6)

        # ---- 右列 ----
        right = QVBoxLayout(); right.setSpacing(8)

        # 风险速度仪表 + 就绪度
        vel_g = QGroupBox('风险动态 · RISK DYNAMICS')
        vel_l = QVBoxLayout()
        self.vel_gauge = RiskVelocityGauge()
        vel_l.addWidget(self.vel_gauge)
        # 数值
        nums = QHBoxLayout()
        self.lbl_vel = QLabel('速度: -- '); self.lbl_vel.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        self.lbl_acc = QLabel('加速度: --'); self.lbl_acc.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        nums.addWidget(self.lbl_vel); nums.addStretch(); nums.addWidget(self.lbl_acc)
        vel_l.addLayout(nums)
        vel_g.setLayout(vel_l)
        right.addWidget(vel_g)

        # 预测数值
        pred_g = QGroupBox('预测外推 · PREDICTION')
        pl = QGridLayout(); pl.setContentsMargins(8,6,8,6); pl.setSpacing(6)
        self.lbl_cur = QLabel('0.00'); self.lbl_cur.setStyleSheet(
            f'color:{C_ACCENT}; font-size:32px; font-weight:bold; font-family:Consolas;')
        self.lbl_cur.setAlignment(Qt.AlignCenter)
        self.lbl_p5 = QLabel('5m: --'); self.lbl_p5.setStyleSheet(f'color:{C_WARN}; font-size:13px;')
        self.lbl_p5.setAlignment(Qt.AlignCenter)
        self.lbl_p15 = QLabel('15m: --'); self.lbl_p15.setStyleSheet(f'color:{C_RISK_HIGH}; font-size:13px;')
        self.lbl_p15.setAlignment(Qt.AlignCenter)
        self.lbl_conf = QLabel('置信: --'); self.lbl_conf.setStyleSheet(f'color:{C_DIM}; font-size:10px;')
        self.lbl_conf.setAlignment(Qt.AlignCenter)

        pl.addWidget(QLabel('当前风险'), 0, 0)
        pl.addWidget(self.lbl_cur, 1, 0)
        pl.addWidget(QLabel('5分钟预测'), 0, 1)
        pl.addWidget(self.lbl_p5, 1, 1)
        pl.addWidget(QLabel('15分钟预测'), 0, 2)
        pl.addWidget(self.lbl_p15, 1, 2)
        pl.addWidget(self.lbl_conf, 2, 0, 1, 3)
        pred_g.setLayout(pl)
        right.addWidget(pred_g)

        # 时间注意力热力图
        att_g = QGroupBox('NN注意力分布 · ATTENTION (60s)')
        att_l = QHBoxLayout(); att_l.setContentsMargins(4,4,4,4)
        self.attn_heatmap = AttentionHeatmap()
        att_l.addWidget(self.attn_heatmap)
        att_l.addStretch()
        att_g.setLayout(att_l)
        right.addWidget(att_g)

        # 主动感知日志
        log_g = QGroupBox('主动感知记录 · ACTIVE SENSING LOG')
        log_l = QVBoxLayout()
        self.sensing_text = QTextEdit(); self.sensing_text.setMaximumHeight(120); self.sensing_text.setReadOnly(True)
        log_l.addWidget(self.sensing_text)
        log_g.setLayout(log_l)
        right.addWidget(log_g)

        right.addStretch()
        body.addLayout(right, 4)
        root.addLayout(body, 1)

        # === 底栏: 控制 ===
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        ctrl.addWidget(QLabel('摄像头:'))
        self.cam_combo = QComboBox(); self.cam_combo.addItems(['0','1','2']); ctrl.addWidget(self.cam_combo)
        self.btn_toggle = QPushButton('启动'); self.btn_toggle.clicked.connect(self._toggle); ctrl.addWidget(self.btn_toggle)
        self.btn_sim = QPushButton('模拟'); self.btn_sim.clicked.connect(self._toggle_sim); ctrl.addWidget(self.btn_sim)
        self.chk_overlay = QCheckBox('叠加'); self.chk_overlay.setChecked(True)
        self.chk_overlay.setStyleSheet(f'color:{C_TEXT};'); ctrl.addWidget(self.chk_overlay)
        ctrl.addStretch()
        ctrl.addWidget(QLabel('FPS:')); self.lbl_fps = QLabel('--')
        self.lbl_fps.setStyleSheet(f'color:{C_DIM};'); ctrl.addWidget(self.lbl_fps)
        root.addLayout(ctrl)

    def _pill(self, text, color, title):
        l = QLabel(f'  {text}  ' if text else '')
        l.setStyleSheet(f'color:{color}; background:transparent; font-weight:bold;')
        return l

    # ================ 采集 ================
    def _auto_start(self):
        try: self._open_camera(0)
        except Exception: self._enter_sim('摄像头不可用, 已切换模拟')

    def _open_camera(self, idx):
        if self.cap: self._release_cam()
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mode = self.MODE_CAMERA
        self.btn_toggle.setText('停止')

    def _release_cam(self):
        if self.cap:
            try: self.cap.release()
            except: pass
        self.cap = None

    def _toggle(self):
        if self.mode == self.MODE_CAMERA:
            self._release_cam(); self.mode = self.MODE_OFFLINE; self.btn_toggle.setText('启动')
        else:
            self._open_camera(self.cam_combo.currentIndex())

    def _toggle_sim(self):
        if self.mode == self.MODE_SIM:
            self._enter_camera()
        else:
            self._enter_sim('切换到模拟')

    def _enter_sim(self, reason=''):
        self._release_cam()
        self.mode = self.MODE_SIM
        self.btn_toggle.setText('启动'); self.btn_sim.setText('退出模拟')
        if reason: self._log(reason)

    def _enter_camera(self):
        self.btn_sim.setText('模拟')
        self._auto_start()

    # ================ 采集tick ================
    def _capture_tick(self):
        if self.mode == self.MODE_CAMERA and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._frame_count += 1
                now = time.time()
                if now - self._last_analysis_t >= 2.0:
                    self._last_analysis_t = now
                    self._do_analysis(frame)
                if self.chk_overlay.isChecked():
                    frame = self._draw_overlay(frame)
                self._show_frame(frame)
            return
        if self.mode == self.MODE_SIM:
            self._sim_tick()

    def _do_analysis(self, frame):
        """模拟分析: 生成随时间变化的伪风险值。"""
        self._sim_t += 2.0
        t = self._sim_t
        # 模拟风险波动: 带趋势的随机游走 + 周期性分量
        trend = 0.0003 * t  # 缓慢上升趋势
        cycle = 0.08 * math.sin(t * 0.02) + 0.05 * math.cos(t * 0.05)
        noise = random.gauss(0, 0.03)
        self._sim_risk += (trend + cycle + noise - self._sim_risk) * 0.15
        self._sim_risk += random.gauss(0, 0.01)
        self._sim_risk = max(0.05, min(0.95, self._sim_risk))

        # 速度(导数)
        self._sim_vel = (0.02 * math.cos(t * 0.02) - 0.01 * math.sin(t * 0.05) + random.gauss(0, 0.005)) * 60

        self.current_risk = self._sim_risk
        self.risk_velocity = self._sim_vel
        self.risk_acceleration = self._sim_vel * random.uniform(-0.2, 0.2)
        self.predicted_risk_5m = np.clip(self.current_risk + self.risk_velocity * 5.0, 0, 1)
        self.predicted_risk_15m = np.clip(self.current_risk + self.risk_velocity * 15.0, 0, 1)
        self.prediction_confidence = 0.55 + 0.35 * random.random()

        # 模拟注意力: 随机集中
        t_mod = t % 60
        self._attn = np.ones(60) * 0.01
        peak = int(t_mod)
        for i in range(max(0, peak-3), min(60, peak+4)):
            self._attn[i] = 0.03 + 0.06 * (1 - abs(i - peak) / 4.0)
        self._attn = self._attn / self._attn.sum()

        # 就绪度
        if self.current_risk > 0.75: self.guardian_level = 3
        elif self.current_risk > 0.55 or self.risk_velocity > 0.02: self.guardian_level = 2
        elif self.risk_velocity > 0.008: self.guardian_level = 1
        else: self.guardian_level = 0

    def _sim_tick(self):
        self._frame_count += 1
        w, h = 800, 600
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            v = int(15 + 12 * y / h)
            img[y, :] = [v, v+3, v+8]
        now = time.time()
        if now - self._last_analysis_t >= 2.0:
            self._last_analysis_t = now
            self._do_analysis(None)
        if self.chk_overlay.isChecked():
            img = self._draw_overlay(img)
        self._show_frame(img)

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 6), (620, 130), (2, 9, 18), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.rectangle(frame, (8, 6), (620, 130), (0, 229, 255), 1)
        lines = [
            f"MODE:{self.mode.upper()}  |  RISK:{self.current_risk:.3f}  |  VEL:{self.risk_velocity*60:.1f}/hr",
            f"PRED 5m:{self.predicted_risk_5m:.3f}  |  15m:{self.predicted_risk_15m:.3f}",
            f"CONF:{self.prediction_confidence:.2f}  |  GUARDIAN LV:{self.guardian_level}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (16, 32+i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1, cv2.LINE_AA)
        return frame

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, w*3, QImage.Format_BGR888)
        self.video_frame.setPixmap(QPixmap.fromImage(qimg))

    # ================ 显示刷新 ================
    def _update_display(self):
        # 速度仪表
        vel_display = self.risk_velocity * 2000  # 缩放到 -100 ~ +100 范围
        acc_display = self.risk_acceleration * 1000
        self.vel_gauge.set_values(vel_display, acc_display)
        self.lbl_vel.setText(f'速度: {self.risk_velocity*60:.2f}/hr')
        self.lbl_acc.setText(f'加速度: {self.risk_acceleration:+.4f}/min²')

        # 预测数值
        self.lbl_cur.setText(f'{self.current_risk:.3f}')
        c_cur = C_OK if self.current_risk < 0.3 else (C_WARN if self.current_risk < 0.6 else C_RISK_CRIT)
        self.lbl_cur.setStyleSheet(f'color:{c_cur}; font-size:32px; font-weight:bold; font-family:Consolas;')
        self.lbl_p5.setText(f'5m: {self.predicted_risk_5m:.3f}')
        self.lbl_p15.setText(f'15m: {self.predicted_risk_15m:.3f}')
        self.lbl_conf.setText(f'置信度: {self.prediction_confidence:.0%}')

        # 轨迹曲线
        self.trajectory_plot.update_data(self.current_risk, self.predicted_risk_5m,
                                         self.predicted_risk_15m, self.prediction_confidence)

        # 注意力热力图
        self.attn_heatmap.set_attention(self._attn)

        # 就绪度
        self.readiness.set_level(self.guardian_level)

        # FPS
        if self._frame_count > 0 and self._last_analysis_t > 0:
            elapsed = max(1, time.time() - self._last_analysis_t + 0.01)
            self.lbl_fps.setText(f'FPS: {self._frame_count/elapsed:.0f}')

    # ================ 日志 ================
    def _log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self._sensing_log.append(f'[{ts}] {msg}')
        if hasattr(self, 'sensing_text'):
            text = '\n'.join(list(self._sensing_log)[-8:])
            self.sensing_text.setText(text)

    def add_sensing_prompt(self, msg):
        """从主动感知系统接收探询消息。"""
        self._log(f'🔔 {msg}')

    def shutdown(self):
        self.update_timer.stop(); self.capture_timer.stop(); self._release_cam()

    def closeEvent(self, ev):
        self.shutdown(); ev.accept()


# ============================================================
# 入口
# ============================================================
def main():
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication([]); app.setStyle('Fusion')
    win = PredictiveVisionWindow(); win.show()
    app.exec_()

if __name__ == '__main__':
    main()
