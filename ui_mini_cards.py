"""
MiniCard 组件 - 心率、脑电、视觉卡片的微缩放大版

设计理念：
- 卡片是完整功能的缩略版（放大后的微缩）
- 内部布局可自由调整
- 统一的赛博朋克风格
- 支持实时数据更新

卡片类型：
1. HeartRateMiniCard - 心率卡片
2. BrainMiniCard - 脑电卡片
3. VisionMiniCard - 视觉卡片
"""

import time
from collections import deque
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import (QFont, QColor, QPainter, QPen, QBrush,
                         QLinearGradient, QRadialGradient, QPainterPath)
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QGraphicsDropShadowEffect

from theme import (C_BG, C_BG_DEEP, C_PANEL, C_PANEL_HI, C_BORDER, C_BORDER_HI,
                   C_ACCENT, C_ACCENT2, C_DIM, C_TEXT, C_TEXT_DIM,
                   C_OK, C_WARN, C_RISK, C_VISION, C_HEART, C_BRAIN)


class MiniCard(QFrame):
    """卡片基类 - 统一样式和阴影效果。"""

    def __init__(self, title, icon, accent_color, parent=None):
        super().__init__(parent)
        self.title = title
        self.icon = icon
        self.accent_color = accent_color

        self.setObjectName('MiniCard')
        self.setStyleSheet(
            f'#MiniCard {{ background:{C_PANEL}; '
            f'border:1px solid {accent_color}; '
            f'border-radius:12px; }}')

        # 发光阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        # 从十六进制颜色字符串创建QColor
        color = QColor(accent_color)
        color.setAlpha(60)
        shadow.setColor(color)
        self.setGraphicsEffect(shadow)

        self.setMinimumSize(280, 200)
        self.setMaximumSize(400, 300)

        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 14, 16, 14)
        self.main_layout.setSpacing(8)

        # 标题栏
        self._build_header()

    def _build_header(self):
        """构建标题栏。"""
        header = QHBoxLayout()
        header.setSpacing(8)

        # 图标
        icon_lbl = QLabel(self.icon)
        icon_lbl.setStyleSheet(
            f'color:{self.accent_color}; font-size:20px; background:transparent;')
        header.addWidget(icon_lbl)

        # 标题
        title_lbl = QLabel(self.title)
        title_lbl.setStyleSheet(
            f'color:{self.accent_color}; font-size:13px; font-weight:bold; '
            f'letter-spacing:1.6px; background:transparent;')
        header.addWidget(title_lbl)

        header.addStretch()

        # 状态指示器
        self.status_dot = QLabel('●')
        self.status_dot.setStyleSheet(
            f'color:{C_OK}; font-size:10px; background:transparent;')
        header.addWidget(self.status_dot)

        self.main_layout.addLayout(header)

    def set_status(self, status='ok'):
        """设置状态指示器颜色。"""
        color_map = {
            'ok': C_OK,
            'warn': C_WARN,
            'error': C_RISK,
            'offline': C_DIM,
        }
        color = color_map.get(status, C_OK)
        self.status_dot.setStyleSheet(
            f'color:{color}; font-size:10px; background:transparent;')


class MiniWaveWidget(QWidget):
    """迷你波形组件 - 用于卡片内的实时波形显示。"""

    def __init__(self, color=C_ACCENT, max_points=60, parent=None):
        super().__init__(parent)
        self.color = color
        self.max_points = max_points
        self.buffer = deque(maxlen=max_points)
        self.setMinimumHeight(50)
        self.setMaximumHeight(80)

        # 初始化为0
        for _ in range(max_points):
            self.buffer.append(0.5)

    def add_data(self, value):
        """添加新数据点。"""
        self.buffer.append(float(value))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.buffer) < 2:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # 背景网格
        grid_color = QColor(self.color)
        grid_color.setAlpha(20)
        p.setPen(QPen(grid_color, 1))
        for i in range(1, 4):
            y = h * i / 4
            p.drawLine(0, int(y), int(w), int(y))

        # 数据归一化
        data = list(self.buffer)
        d_min = min(data)
        d_max = max(data)
        d_range = d_max - d_min if d_max > d_min else 1.0

        # 绘制波形
        path = QPainterPath()
        step = w / (len(data) - 1)

        for i, val in enumerate(data):
            x = i * step
            y = h - ((val - d_min) / d_range * h * 0.8 + h * 0.1)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # 渐变填充
        fill_path = QPainterPath(path)
        fill_path.lineTo(w, h)
        fill_path.lineTo(0, h)
        fill_path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        color_start = QColor(self.color)
        color_start.setAlpha(100)
        color_end = QColor(self.color)
        color_end.setAlpha(0)
        grad.setColorAt(0.0, color_start)
        grad.setColorAt(1.0, color_end)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(fill_path)

        # 主线
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(self.color), 2.0))
        p.drawPath(path)


class HeartRateMiniCard(MiniCard):
    """心率卡片 - 显示心率、HRV和迷你波形。"""

    def __init__(self, parent=None):
        super().__init__('心率 · HR', '❤️', C_HEART, parent)
        self._build_content()

    def _build_content(self):
        """构建卡片内容。"""
        # 大数字显示
        self.value_lbl = QLabel('--')
        self.value_lbl.setStyleSheet(
            f'color:{C_HEART}; font-size:48px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        self.value_lbl.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.value_lbl)

        # 单位
        unit_lbl = QLabel('bpm')
        unit_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:12px; letter-spacing:2px; '
            f'background:transparent;')
        unit_lbl.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(unit_lbl)

        # 迷你波形
        self.wave = MiniWaveWidget(color=C_HEART, max_points=40)
        self.main_layout.addWidget(self.wave)

        # 次要指标
        metrics = QHBoxLayout()
        metrics.setSpacing(16)

        # HRV
        hrv_box = QVBoxLayout()
        hrv_box.setSpacing(2)
        hrv_label = QLabel('HRV')
        hrv_label.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1px; '
            f'background:transparent;')
        hrv_label.setAlignment(Qt.AlignCenter)
        self.hrv_value = QLabel('--')
        self.hrv_value.setStyleSheet(
            f'color:{C_TEXT}; font-size:13px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        self.hrv_value.setAlignment(Qt.AlignCenter)
        hrv_box.addWidget(hrv_label)
        hrv_box.addWidget(self.hrv_value)
        metrics.addLayout(hrv_box)

        # 质量
        quality_box = QVBoxLayout()
        quality_box.setSpacing(2)
        quality_label = QLabel('质量')
        quality_label.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1px; '
            f'background:transparent;')
        quality_label.setAlignment(Qt.AlignCenter)
        self.quality_value = QLabel('--')
        self.quality_value.setStyleSheet(
            f'color:{C_TEXT}; font-size:13px; font-weight:bold; '
            f'background:transparent;')
        self.quality_value.setAlignment(Qt.AlignCenter)
        quality_box.addWidget(quality_label)
        quality_box.addWidget(self.quality_value)
        metrics.addLayout(quality_box)

        self.main_layout.addLayout(metrics)
        self.main_layout.addStretch()

    def update_data(self, bpm=None, hrv=None, quality=None, wave_data=None):
        """更新卡片数据。"""
        if bpm is not None:
            self.value_lbl.setText(f'{int(bpm)}')
            # 根据心率设置颜色
            if 60 <= bpm <= 90:
                color = C_HEART
                self.set_status('ok')
            elif 50 <= bpm < 60 or 90 < bpm <= 100:
                color = C_WARN
                self.set_status('warn')
            else:
                color = C_RISK
                self.set_status('error')
            self.value_lbl.setStyleSheet(
                f'color:{color}; font-size:48px; font-weight:bold; '
                f'background:transparent; font-family:Consolas;')

        if hrv is not None:
            self.hrv_value.setText(f'{int(hrv)}ms')

        if quality is not None:
            self.quality_value.setText(quality)

        if wave_data is not None:
            self.wave.add_data(wave_data)


class BrainMiniCard(MiniCard):
    """脑电卡片 - 显示专注度、放松度和信号质量。"""

    def __init__(self, parent=None):
        super().__init__('脑电 · EEG', '🧠', C_BRAIN, parent)
        self._build_content()

    def _build_content(self):
        """构建卡片内容。"""
        # 专注度进度条
        attn_box = QVBoxLayout()
        attn_box.setSpacing(4)

        attn_header = QHBoxLayout()
        attn_label = QLabel('专注度')
        attn_label.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; font-weight:bold; '
            f'background:transparent;')
        self.attn_value = QLabel('--')
        self.attn_value.setStyleSheet(
            f'color:{C_BRAIN}; font-size:16px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        attn_header.addWidget(attn_label)
        attn_header.addStretch()
        attn_header.addWidget(self.attn_value)
        attn_box.addLayout(attn_header)

        self.attn_bar = QFrame()
        self.attn_bar.setFixedHeight(12)
        self.attn_bar.setStyleSheet(
            f'background:{C_BG_DEEP}; border:1px solid {C_BORDER}; border-radius:6px;')
        attn_box.addWidget(self.attn_bar)

        self.main_layout.addLayout(attn_box)

        # 放松度进度条
        med_box = QVBoxLayout()
        med_box.setSpacing(4)

        med_header = QHBoxLayout()
        med_label = QLabel('放松度')
        med_label.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; font-weight:bold; '
            f'background:transparent;')
        self.med_value = QLabel('--')
        self.med_value.setStyleSheet(
            f'color:{C_ACCENT2}; font-size:16px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        med_header.addWidget(med_label)
        med_header.addStretch()
        med_header.addWidget(self.med_value)
        med_box.addLayout(med_header)

        self.med_bar = QFrame()
        self.med_bar.setFixedHeight(12)
        self.med_bar.setStyleSheet(
            f'background:{C_BG_DEEP}; border:1px solid {C_BORDER}; border-radius:6px;')
        med_box.addWidget(self.med_bar)

        self.main_layout.addLayout(med_box)

        self.main_layout.addSpacing(8)

        # 信号质量
        quality_box = QHBoxLayout()
        quality_label = QLabel('信号质量')
        quality_label.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1px; '
            f'background:transparent;')
        self.quality_dots = QLabel('●●●●●')
        self.quality_dots.setStyleSheet(
            f'color:{C_OK}; font-size:12px; background:transparent;')
        quality_box.addWidget(quality_label)
        quality_box.addStretch()
        quality_box.addWidget(self.quality_dots)

        self.main_layout.addLayout(quality_box)
        self.main_layout.addStretch()

    def update_data(self, attention=None, meditation=None, poor_signal=None):
        """更新卡片数据。"""
        if attention is not None:
            self.attn_value.setText(f'{int(attention)}')
            # 更新进度条（通过动态样式）
            self._update_bar(self.attn_bar, attention, C_BRAIN)

            # 根据专注度设置状态
            if attention >= 60:
                self.set_status('ok')
            elif attention >= 40:
                self.set_status('warn')
            else:
                self.set_status('error')

        if meditation is not None:
            self.med_value.setText(f'{int(meditation)}')
            self._update_bar(self.med_bar, meditation, C_ACCENT2)

        if poor_signal is not None:
            # 信号质量：poor_signal 越低越好
            if poor_signal < 20:
                dots = '●●●●●'
                color = C_OK
            elif poor_signal < 50:
                dots = '●●●●○'
                color = C_WARN
            elif poor_signal < 100:
                dots = '●●●○○'
                color = C_WARN
            else:
                dots = '●○○○○'
                color = C_RISK
            self.quality_dots.setText(dots)
            self.quality_dots.setStyleSheet(
                f'color:{color}; font-size:12px; background:transparent;')

    def _update_bar(self, bar, value, color):
        """更新进度条样式。"""
        percent = max(0, min(100, int(value)))
        bar.setStyleSheet(
            f'background: qlineargradient(x1:0, y1:0, x2:1, y2:0, '
            f'stop:0 {color}, stop:{percent/100} {color}, '
            f'stop:{percent/100} {C_BG_DEEP}, stop:1 {C_BG_DEEP}); '
            f'border:1px solid {C_BORDER}; border-radius:6px;')


class VisionMiniCard(MiniCard):
    """视觉卡片 - 显示表情、眼疲劳和姿态风险。"""

    def __init__(self, parent=None):
        super().__init__('视觉 · VISION', '👁', C_VISION, parent)
        self._build_content()

    def _build_content(self):
        """构建卡片内容。"""
        # 表情显示
        expr_box = QVBoxLayout()
        expr_box.setSpacing(4)

        expr_label = QLabel('当前表情')
        expr_label.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1px; '
            f'background:transparent;')
        expr_label.setAlignment(Qt.AlignCenter)

        self.expr_value = QLabel('--')
        self.expr_value.setStyleSheet(
            f'color:{C_VISION}; font-size:24px; font-weight:bold; '
            f'background:transparent;')
        self.expr_value.setAlignment(Qt.AlignCenter)

        self.expr_prob = QLabel('--')
        self.expr_prob.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:11px; '
            f'background:transparent; font-family:Consolas;')
        self.expr_prob.setAlignment(Qt.AlignCenter)

        expr_box.addWidget(expr_label)
        expr_box.addWidget(self.expr_value)
        expr_box.addWidget(self.expr_prob)

        self.main_layout.addLayout(expr_box)
        self.main_layout.addSpacing(8)

        # 眼疲劳和姿态风险
        metrics = QHBoxLayout()
        metrics.setSpacing(16)

        # 眼疲劳
        eye_box = QVBoxLayout()
        eye_box.setSpacing(2)
        eye_label = QLabel('眼疲劳')
        eye_label.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1px; '
            f'background:transparent;')
        eye_label.setAlignment(Qt.AlignCenter)
        self.eye_value = QLabel('--')
        self.eye_value.setStyleSheet(
            f'color:{C_TEXT}; font-size:18px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        self.eye_value.setAlignment(Qt.AlignCenter)
        eye_box.addWidget(eye_label)
        eye_box.addWidget(self.eye_value)
        metrics.addLayout(eye_box)

        # 姿态风险
        posture_box = QVBoxLayout()
        posture_box.setSpacing(2)
        posture_label = QLabel('姿态')
        posture_label.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1px; '
            f'background:transparent;')
        posture_label.setAlignment(Qt.AlignCenter)
        self.posture_value = QLabel('--')
        self.posture_value.setStyleSheet(
            f'color:{C_TEXT}; font-size:18px; font-weight:bold; '
            f'background:transparent; font-family:Consolas;')
        self.posture_value.setAlignment(Qt.AlignCenter)
        posture_box.addWidget(posture_label)
        posture_box.addWidget(self.posture_value)
        metrics.addLayout(posture_box)

        self.main_layout.addLayout(metrics)
        self.main_layout.addStretch()

    def update_data(self, expr=None, expr_prob=None, eye_fatigue=None, posture_risk=None):
        """更新卡片数据。"""
        if expr is not None:
            self.expr_value.setText(expr)

            # 根据表情设置状态
            if expr in ['开心', '平静', 'happy', 'neutral']:
                self.set_status('ok')
            elif expr in ['悲伤', 'sad']:
                self.set_status('warn')
            elif expr == '未检测到人脸':
                self.set_status('offline')
            else:
                self.set_status('ok')

        if expr_prob is not None:
            self.expr_prob.setText(f'置信度 {expr_prob:.2f}')

        if eye_fatigue is not None:
            value = int(eye_fatigue * 100)
            self.eye_value.setText(f'{value}%')
            # 根据眼疲劳设置颜色
            if value < 50:
                color = C_OK
            elif value < 70:
                color = C_WARN
            else:
                color = C_RISK
            self.eye_value.setStyleSheet(
                f'color:{color}; font-size:18px; font-weight:bold; '
                f'background:transparent; font-family:Consolas;')

        if posture_risk is not None:
            value = int(posture_risk * 100)
            self.posture_value.setText(f'{value}%')
            # 根据姿态风险设置颜色
            if value < 50:
                color = C_OK
            elif value < 70:
                color = C_WARN
            else:
                color = C_RISK
            self.posture_value.setStyleSheet(
                f'color:{color}; font-size:18px; font-weight:bold; '
                f'background:transparent; font-family:Consolas;')
