"""
重构组件库 (D:/G/3)：
  - RadarWidget       五维雷达图（赛博朋克）
  - ConcentricRing    同心圆环（家长端苹果医疗风）
  - BigStat           生理指标大字报（数+单位+标签 + 状态点）
  - CyberDialog       无边框半透明赛博朋克 QDialog 基类
  - LightDialog       苹果医疗风纯白圆角 QDialog 基类
  - DockButton        底部 Dock 工具按钮（描边发光）
"""

import math
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (QColor, QPainter, QBrush, QPen, QFont,
                         QLinearGradient, QRadialGradient, QPainterPath,
                         QPolygonF)
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QDialog, QPushButton, QFrame, QGraphicsDropShadowEffect)

from theme import (C_BG, C_BG_DEEP, C_PANEL, C_BORDER, C_ACCENT, C_DIM,
                   C_TEXT, C_OK, C_WARN, C_RISK, C_VISION, C_HEART, C_BRAIN,
                   C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT)


# ============== 雷达图 ==============
class RadarWidget(QWidget):
    """五维雷达图。values dict: {'expr':0~1, 'hr':0~1, 'hrv':0~1, 'eye':0~1, 'eeg':0~1}"""

    AXES = [('表情', C_VISION), ('心率', C_HEART), ('HRV', '#ff9bb6'),
            ('眼疲劳', '#ffd54f'), ('脑电', C_BRAIN)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vals = {'expr': 0.2, 'hr': 0.4, 'hrv': 0.5, 'eye': 0.3, 'eeg': 0.45}
        self.setMinimumSize(220, 220)

    def set_values(self, expr=None, hr=None, hrv=None, eye=None, eeg=None):
        if expr is not None: self._vals['expr'] = max(0, min(1, expr))
        if hr   is not None: self._vals['hr']   = max(0, min(1, hr))
        if hrv  is not None: self._vals['hrv']  = max(0, min(1, hrv))
        if eye  is not None: self._vals['eye']  = max(0, min(1, eye))
        if eeg  is not None: self._vals['eeg']  = max(0, min(1, eeg))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W / 2, H / 2
        R = min(W, H) / 2 - 36
        n = 5
        # 五边背景网格（4 圈）
        for k in range(4):
            r = R * (k + 1) / 4
            path = QPainterPath()
            for i in range(n):
                ang = -math.pi / 2 + 2 * math.pi * i / n
                x = cx + r * math.cos(ang)
                y = cy + r * math.sin(ang)
                if i == 0: path.moveTo(x, y)
                else: path.lineTo(x, y)
            path.closeSubpath()
            p.setPen(QPen(QColor(C_BORDER), 1, Qt.SolidLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        # 轴线 + 标签
        p.setPen(QPen(QColor(C_BORDER), 1))
        for i, (name, color) in enumerate(self.AXES):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            x = cx + R * math.cos(ang)
            y = cy + R * math.sin(ang)
            p.drawLine(QPointF(cx, cy), QPointF(x, y))
            # 文字
            p.setPen(QColor(color))
            f = QFont('微软雅黑', 9, QFont.Bold)
            p.setFont(f)
            tx = cx + (R + 18) * math.cos(ang) - 18
            ty = cy + (R + 18) * math.sin(ang) + 4
            p.drawText(int(tx), int(ty), 36, 14, Qt.AlignCenter, name)
        # 数据多边形
        keys = ['expr', 'hr', 'hrv', 'eye', 'eeg']
        poly = QPolygonF()
        for i, k in enumerate(keys):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            r = R * self._vals[k]
            poly.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        # 渐变填充
        g = QRadialGradient(cx, cy, R)
        g.setColorAt(0, QColor(0, 229, 255, 110))
        g.setColorAt(1, QColor(124, 77, 255, 60))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(C_ACCENT), 2))
        p.drawPolygon(poly)
        # 数据点
        for i, k in enumerate(keys):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            r = R * self._vals[k]
            x = cx + r * math.cos(ang); y = cy + r * math.sin(ang)
            p.setBrush(QBrush(QColor(self.AXES[i][1])))
            p.setPen(QPen(QColor(C_BG_DEEP), 1))
            p.drawEllipse(QPointF(x, y), 4, 4)


# ============== 同心圆环 ==============
class ConcentricRing(QWidget):
    """家长端：巨型同心圆环（外环风险 / 内环 HRV）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._risk = 0.0
        self._hrv_ratio = 0.5
        self._tag = '低 · LOW'
        self._color = C_RISK_LOW
        self.setMinimumSize(360, 360)

    def set_data(self, risk_score, hrv_ratio=0.5, tag='', color=None):
        self._risk = max(0, min(1, risk_score))
        self._hrv_ratio = max(0, min(1, hrv_ratio))
        self._tag = tag or self._tag
        if color: self._color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W / 2, H / 2
        R_out = min(W, H) / 2 - 14
        R_in = R_out - 22

        # 外环底色
        p.setPen(QPen(QColor(C_BORDER), 14, Qt.SolidLine, Qt.FlatCap))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(cx - R_out, cy - R_out, R_out * 2, R_out * 2),
                  90 * 16, -360 * 16)
        # 外环值（顺时针起 12 点）
        p.setPen(QPen(QColor(self._color), 14, Qt.SolidLine, Qt.RoundCap))
        span = -int(self._risk * 360 * 16)
        p.drawArc(QRectF(cx - R_out, cy - R_out, R_out * 2, R_out * 2),
                  90 * 16, span)
        # 内环底色
        p.setPen(QPen(QColor(C_BORDER), 8))
        p.drawArc(QRectF(cx - R_in, cy - R_in, R_in * 2, R_in * 2),
                  90 * 16, -360 * 16)
        # 内环 HRV
        p.setPen(QPen(QColor(C_HEART), 8, Qt.SolidLine, Qt.RoundCap))
        span2 = -int(self._hrv_ratio * 360 * 16)
        p.drawArc(QRectF(cx - R_in, cy - R_in, R_in * 2, R_in * 2),
                  90 * 16, span2)
        # 中央数字
        p.setPen(QColor(self._color))
        f = QFont('Consolas', 56, QFont.Bold)
        p.setFont(f)
        p.drawText(QRectF(0, cy - 60, W, 80),
                   Qt.AlignCenter, f'{self._risk:.2f}')
        # 标签
        p.setPen(QColor(C_DIM))
        f2 = QFont('微软雅黑', 11)
        f2.setBold(True)
        p.setFont(f2)
        p.drawText(QRectF(0, cy - 100, W, 24),
                   Qt.AlignCenter, '当前综合风险')
        p.setPen(QColor(self._color))
        f3 = QFont('微软雅黑', 12, QFont.Bold)
        p.setFont(f3)
        p.drawText(QRectF(0, cy + 12, W, 24),
                   Qt.AlignCenter, self._tag)


# ============== 生理指标大字报 ==============
class BigStat(QFrame):
    """大字报：标签 + 主数(40px) + 单位 + 趋势小箭头/状态点。"""

    def __init__(self, label, unit='', color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.color = color
        self.setObjectName('BigStat')
        self.setStyleSheet(
            f'#BigStat {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {color}; border-radius:8px; }}')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.lab_label = QLabel(label)
        self.lab_label.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.6px; background:transparent;')
        self.lab_dot = QLabel('●')
        self.lab_dot.setStyleSheet(f'color:{color}; font-size:10px; background:transparent;')
        self.lab_dot.setAlignment(Qt.AlignRight)
        head.addWidget(self.lab_label); head.addStretch(); head.addWidget(self.lab_dot)
        lay.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.lab_value = QLabel('--')
        self.lab_value.setStyleSheet(
            f'color:{color}; font-size:38px; font-weight:bold;'
            f'font-family:Consolas; background:transparent;')
        self.lab_value.setMinimumWidth(80)
        self.lab_unit = QLabel(unit)
        self.lab_unit.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; background:transparent;')
        self.lab_unit.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        row.addWidget(self.lab_value)
        row.addWidget(self.lab_unit)
        row.addStretch()
        lay.addLayout(row)

    def set_value(self, text, color=None):
        self.lab_value.setText(str(text))
        if color is not None:
            self.color = color
            self.lab_value.setStyleSheet(
                f'color:{color}; font-size:38px; font-weight:bold;'
                f'font-family:Consolas; background:transparent;')
            self.lab_dot.setStyleSheet(
                f'color:{color}; font-size:10px; background:transparent;')


# ============== 赛博朋克 Dialog ==============
class CyberDialog(QDialog):
    """无边框 + 半透明 + 霓虹边 + 拖动。子类放 self.body 里。"""

    def __init__(self, title='', parent=None, w=820, h=560):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(w, h)
        self._drag = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)

        self._frame = QFrame()
        self._frame.setObjectName('CD')
        self._frame.setStyleSheet(
            f'#CD {{ background:rgba(4,17,31,0.94);'
            f'border:1px solid {C_ACCENT}; border-radius:14px; }}')
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(40); eff.setOffset(0, 0)
        eff.setColor(QColor(0, 229, 255, 160))
        self._frame.setGraphicsEffect(eff)
        outer.addWidget(self._frame)

        flay = QVBoxLayout(self._frame)
        flay.setContentsMargins(20, 14, 20, 20)
        flay.setSpacing(12)

        head = QHBoxLayout()
        dot = QLabel('●')
        dot.setStyleSheet(f'color:{C_ACCENT}; font-size:14px; background:transparent;')
        t = QLabel(title)
        t.setStyleSheet(
            f'color:{C_ACCENT}; font-size:16px; font-weight:bold;'
            f'letter-spacing:3px; background:transparent;')
        close = QPushButton('✕')
        close.setFixedSize(28, 28)
        close.setCursor(Qt.PointingHandCursor)
        close.setStyleSheet(
            f'QPushButton {{ color:{C_RISK}; background:transparent;'
            f'border:1px solid {C_RISK}; border-radius:14px; font-weight:bold; }}'
            f'QPushButton:hover {{ background:{C_RISK}; color:{C_BG_DEEP}; }}')
        close.clicked.connect(self.close)
        head.addWidget(dot); head.addSpacing(6); head.addWidget(t); head.addStretch(); head.addWidget(close)
        flay.addLayout(head)

        self.body = QWidget()
        self.body.setStyleSheet('background:transparent;')
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        flay.addWidget(self.body, 1)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = ev.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPos() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None


# ============== 苹果医疗风 Dialog ==============
class LightDialog(QDialog):
    """白底 + 大圆角 + 弥散阴影。"""

    def __init__(self, title='', parent=None, w=820, h=560):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(w, h)
        self._drag = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        self._frame = QFrame()
        self._frame.setObjectName('LD')
        self._frame.setStyleSheet(
            '#LD { background: white; border-radius: 16px; }')
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(60); eff.setOffset(0, 12)
        eff.setColor(QColor(0, 0, 0, 70))
        self._frame.setGraphicsEffect(eff)
        outer.addWidget(self._frame)

        flay = QVBoxLayout(self._frame)
        flay.setContentsMargins(28, 22, 28, 22)
        flay.setSpacing(14)

        head = QHBoxLayout()
        t = QLabel(title)
        t.setStyleSheet(
            'color:#1d1d1f; font-size:18px; font-weight:600;'
            'background:transparent; letter-spacing:0.6px;')
        close = QPushButton('×')
        close.setFixedSize(30, 30)
        close.setCursor(Qt.PointingHandCursor)
        close.setStyleSheet(
            'QPushButton { color:#86868b; background:#f2f2f7;'
            'border:none; border-radius:15px; font-size:18px; font-weight:bold; }'
            'QPushButton:hover { background:#ff3b30; color:white; }')
        close.clicked.connect(self.close)
        head.addWidget(t); head.addStretch(); head.addWidget(close)
        flay.addLayout(head)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet('background:#e5e5ea; border:none;')
        flay.addWidget(sep)

        self.body = QWidget()
        self.body.setStyleSheet('background:transparent; color:#1d1d1f;')
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 6, 0, 0)
        flay.addWidget(self.body, 1)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = ev.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPos() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None


# ============== 底部 Dock 按钮 ==============
class DockButton(QPushButton):
    def __init__(self, icon_text, label, color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.color = color
        self.setText(f'{icon_text}  {label}')
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setStyleSheet(
            f'QPushButton {{ background:rgba(10,31,61,0.6);'
            f'color:{color}; border:1px solid {color};'
            f'border-radius:19px; padding:0 22px; font-weight:bold;'
            f'letter-spacing:2px; font-size:12px; }}'
            f'QPushButton:hover {{ background:{color}; color:{C_BG_DEEP}; }}')
