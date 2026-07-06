"""
脑电感知节点 UI - 嵌入 MainWindow 的 page_bci。
MindViewer 风格：双轴时序图（8 频段堆叠 + RAW 波）+ Att/Med 圆环 + PoorSignal 诊断。
"""

from collections import deque
from itertools import product

from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QTextEdit, QVBoxLayout, QWidget)

import pyqtgraph as pg


# 统一从 theme 模块取色（避免与 ui_main 不一致）
from theme import (C_BG, C_PANEL, C_PANEL_HI, C_BORDER, C_ACCENT,
                   C_DIM, C_TEXT, C_OK, C_WARN, C_RISK,
                   C_RAW, C_ATT, C_MED, C_NOISE, BAND_LIST)


def decode_poor_signal(ps):
    if ps == 0:
        return ('信号良好', C_OK)
    if ps == 200:
        return ('设备脱落 (>4s)', C_RISK)
    for a, b, c, d in product([0, 1], repeat=4):
        total = 25 * a + 26 * b + 27 * c + 29 * d
        if total == ps and total > 0:
            names = []
            if a: names.append('信号过平')
            if b: names.append('肌电干扰')
            if c: names.append('环境噪声')
            if d: names.append('脱头')
            color = C_RISK if d else C_WARN
            return (' + '.join(names), color)
    return (f'未知旗标 ({ps})', C_WARN)


class CircularGauge(QWidget):
    def __init__(self, title, color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.value = 0
        self.setMinimumSize(118, 118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v):
        self.value = max(0, min(100, int(v)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height())
        # 为中心数字和标签留足内径空间
        ring_margin = max(12, int(side * 0.12))
        ring_w = max(8, int(side * 0.12))
        rect = QRectF(
            (self.width() - side) / 2 + ring_margin,
            (self.height() - side) / 2 + ring_margin,
            side - 2 * ring_margin,
            side - 2 * ring_margin,
        )

        # 底环
        p.setPen(QPen(QColor(C_BORDER), ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 0, 360 * 16)

        # 值环 + 轻微外发光
        glow = QColor(self.color)
        glow.setAlpha(95)
        p.setPen(QPen(glow, ring_w + 3, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90 * 16, int(-self.value * 3.6 * 16))
        p.setPen(QPen(QColor(self.color), ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90 * 16, int(-self.value * 3.6 * 16))

        # 中心文本层级置顶 + 自适应字号
        value_font = max(20, min(40, int(side * 0.26)))
        title_font = max(10, min(14, int(side * 0.10)))

        p.setPen(QColor('#001122'))
        p.setFont(QFont('Microsoft YaHei', value_font, QFont.Bold))
        p.drawText(rect.adjusted(1, 1, 1, 1), Qt.AlignCenter, f'{self.value}')
        p.setPen(QColor('#EAF2FF'))
        p.drawText(rect, Qt.AlignCenter, f'{self.value}')

        p.setPen(QColor('#001122'))
        p.setFont(QFont('Microsoft YaHei', title_font, QFont.Bold))
        tr = QRectF(rect.x(), rect.y() + rect.height() * 0.70, rect.width(), max(18, int(side * 0.15)))
        p.drawText(tr.adjusted(1, 1, 1, 1), Qt.AlignCenter, self.title)
        p.setPen(QColor('#9BB6E8'))
        p.drawText(tr, Qt.AlignCenter, self.title)


class StatusPill(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_state('offline', '未连接')

    def set_state(self, state, text):
        colors = {
            'live': (C_OK, '#0d2818'),
            'sim': (C_WARN, '#2e1f05'),
            'offline': (C_RISK, '#2a0a0e'),
        }
        fg, bg = colors.get(state, colors['offline'])
        self.setText(f'  ●  {text}  ')
        self.setStyleSheet(
            f'background:{bg}; color:{fg};'
            f'border:1px solid {fg}; border-radius:10px;'
            f'padding:4px 12px; font-weight:bold;'
        )


class BCIPanel(QWidget):
    """脑电感知节点面板（接收 BCIThread 信号即可刷新）。"""

    WINDOW_SEC = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.attention = 0
        self.meditation = 0
        self.poor_signal = 200
        self.eeg_power = {}
        self.raw_wave = []

        self.band_history = {k: deque(maxlen=self.WINDOW_SEC) for k, _, _ in BAND_LIST}
        self.att_history = deque(maxlen=self.WINDOW_SEC)
        self.med_history = deque(maxlen=self.WINDOW_SEC)
        self.noise_history = deque(maxlen=self.WINDOW_SEC)

        self._init_ui()

        self.history_timer = QTimer(self)
        self.history_timer.timeout.connect(self._history_tick)
        self.history_timer.start(1000)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_plots)
        self.refresh_timer.start(120)

    def _init_ui(self):
        """按 11.py 独立版的纵向次序排版：图表占主体，下方一行三块（圆环 / 风险 / 诊断），日志最小。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 8, 6)
        root.setSpacing(6)

        # 标题 + 两个状态胶囊
        title = QLabel('脑电感知节点 · TGAM EDGE NODE')
        title.setStyleSheet(f'color:{C_ACCENT}; font-size:14px; font-weight:bold; letter-spacing:1px;')
        header_row = QHBoxLayout()
        header_row.addWidget(title)
        header_row.addStretch()
        self.pill_source = StatusPill()
        self.pill_quality = StatusPill()
        self.pill_quality.set_state('offline', '信号质量：--')
        header_row.addWidget(self.pill_source)
        header_row.addWidget(self.pill_quality)
        root.addLayout(header_row)

        # 1) 频段功率 + RAW 双轴时序  ← stretch=3
        top_group = QGroupBox('频段功率与原始脑电波 · POWER + RAW')
        top_group.setStyleSheet(self._gbox_qss())
        top_lay = QVBoxLayout(top_group)
        top_lay.setContentsMargins(6, 10, 6, 4)
        self.plot_top = self._build_top_plot()
        self.plot_top.setMinimumHeight(180)
        top_lay.addWidget(self.plot_top, 1)
        top_lay.addLayout(self._legend_top())
        root.addWidget(top_group, 3)

        # 2) Att / Med / Noise 时序  ← stretch=2
        bot_group = QGroupBox('精神状态时序 · ATTENTION / MEDITATION / NOISE')
        bot_group.setStyleSheet(self._gbox_qss())
        bot_lay = QVBoxLayout(bot_group)
        bot_lay.setContentsMargins(6, 10, 6, 4)
        self.plot_bot = self._build_bottom_plot()
        self.plot_bot.setMinimumHeight(120)
        bot_lay.addWidget(self.plot_bot, 1)
        bot_lay.addLayout(self._legend_bottom())
        root.addWidget(bot_group, 2)

        # 3) 一行三块：精神状态圆环 | 情绪风险评分 | 信号质量诊断
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        gauge_group = QGroupBox('精神状态 · MENTAL STATE')
        gauge_group.setStyleSheet(self._gbox_qss())
        glay = QHBoxLayout(gauge_group)
        glay.setContentsMargins(6, 12, 6, 6)
        glay.setSpacing(4)
        self.gauge_att = CircularGauge('专注度', C_ATT)
        self.gauge_med = CircularGauge('放松度', C_MED)
        self.gauge_att.setMinimumSize(110, 110)
        self.gauge_med.setMinimumSize(110, 110)
        glay.addWidget(self.gauge_att)
        glay.addWidget(self.gauge_med)
        info_row.addWidget(gauge_group, 4)

        risk_group = QGroupBox('情绪风险评分 · RISK INDEX')
        risk_group.setStyleSheet(self._gbox_qss())
        rlay = QVBoxLayout(risk_group)
        rlay.setContentsMargins(10, 16, 10, 8)
        rlay.setSpacing(2)
        self.risk_value = QLabel('0.00')
        self.risk_value.setAlignment(Qt.AlignCenter)
        self.risk_value.setStyleSheet(
            f'color:{C_OK}; font-size:36px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self.risk_tag = QLabel('低 · LOW')
        self.risk_tag.setAlignment(Qt.AlignCenter)
        self.risk_tag.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; letter-spacing:2px; background:transparent;')
        self.risk_bar = QLabel()
        self.risk_bar.setFixedHeight(5)
        self.risk_bar.setStyleSheet(
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_OK}, stop:0.5 {C_WARN}, stop:1 {C_RISK});'
            f'border-radius:2px;')
        rlay.addStretch()
        rlay.addWidget(self.risk_value)
        rlay.addWidget(self.risk_tag)
        rlay.addSpacing(4)
        rlay.addWidget(self.risk_bar)
        rlay.addStretch()
        info_row.addWidget(risk_group, 3)

        sq_group = QGroupBox('信号质量诊断 · SIGNAL DIAGNOSIS')
        sq_group.setStyleSheet(self._gbox_qss())
        sqlay = QGridLayout(sq_group)
        sqlay.setContentsMargins(10, 14, 10, 8)
        sqlay.setVerticalSpacing(6)
        ps_label = QLabel('PoorSignal:')
        ps_label.setStyleSheet(f'color:{C_DIM}; font-size:11px; background:transparent;')
        sqlay.addWidget(ps_label, 0, 0)
        self.ps_value = QLabel('--')
        self.ps_value.setStyleSheet(
            f'color:{C_ACCENT}; font-size:18px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        sqlay.addWidget(self.ps_value, 0, 1)
        diag_label = QLabel('诊断:')
        diag_label.setStyleSheet(f'color:{C_DIM}; font-size:11px; background:transparent;')
        sqlay.addWidget(diag_label, 1, 0)
        self.ps_diag = QLabel('--')
        self.ps_diag.setStyleSheet(
            f'color:{C_DIM}; font-size:12px; font-weight:bold; background:transparent;')
        self.ps_diag.setWordWrap(True)
        sqlay.addWidget(self.ps_diag, 1, 1)
        sqlay.setColumnStretch(1, 1)
        info_row.addWidget(sq_group, 4)

        root.addLayout(info_row, 0)

        # 4) 节点日志（max 70px，最小占用）
        log_group = QGroupBox('节点日志 · LOG')
        log_group.setStyleSheet(self._gbox_qss())
        ll = QVBoxLayout(log_group)
        ll.setContentsMargins(6, 12, 6, 4)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            f'QTextEdit {{ background:#02050a; color:{C_OK};'
            f' border:1px solid {C_BORDER}; border-radius:8px;'
            f' font-family:Consolas; font-size:10px; padding:4px; }}')
        self.log_text.setMaximumHeight(58)
        ll.addWidget(self.log_text)
        root.addWidget(log_group, 0)

    def _gbox_qss(self):
        return (
            f'QGroupBox {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f' border-radius:12px; margin-top:14px; padding-top:6px;'
            f' color:{C_ACCENT}; font-weight:bold; }}'
            f'QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left;'
            f' padding:2px 10px; background:{C_BG}; color:{C_ACCENT}; letter-spacing:1px; }}'
            f'QLabel {{ color:{C_TEXT}; }}'
        )

    def _legend_top(self):
        row = QHBoxLayout()
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(8)
        for _, label, color in BAND_LIST:
            lbl = QLabel(f'<span style="color:{color}">■</span> '
                         f'<span style="color:{C_DIM}">{label}</span>')
            row.addWidget(lbl)
        row.addStretch()
        row.addWidget(QLabel(f'<span style="color:{C_RAW}">━</span> '
                             f'<span style="color:{C_DIM}">RAW Wave</span>'))
        return row

    def _legend_bottom(self):
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 0)
        for color, text in [(C_NOISE, 'Noise (信号噪声)'),
                            (C_ATT, 'Attention (专注度)'),
                            (C_MED, 'Meditation (放松度)')]:
            row.addWidget(QLabel(f'<span style="color:{color}">■</span> '
                                 f'<span style="color:{C_DIM}">{text}</span>'))
        row.addStretch()
        return row

    def _build_top_plot(self):
        plot = pg.PlotWidget()
        plot.setBackground(C_PANEL)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel('left', 'Power', color=C_DIM)
        plot.setLabel('bottom', 'Time (s)', color=C_DIM)
        plot.showAxis('right')
        plot.setLabel('right', 'RAW Wave', color=C_RAW)
        for ax in ('left', 'bottom', 'right'):
            plot.getAxis(ax).setPen(C_BORDER)
            plot.getAxis(ax).setTextPen(C_DIM if ax != 'right' else C_RAW)
        plot.setXRange(-self.WINDOW_SEC, 0, padding=0)

        self.band_curves = {}
        for key, label, color in BAND_LIST:
            qc = QColor(color)
            fill = QColor(color)
            fill.setAlphaF(0.45)
            curve = plot.plot(
                pen=pg.mkPen(qc, width=1.2),
                fillLevel=0, brush=QBrush(fill), name=label,
            )
            self.band_curves[key] = curve

        self.raw_vb = pg.ViewBox()
        plot.scene().addItem(self.raw_vb)
        plot.getAxis('right').linkToView(self.raw_vb)
        self.raw_vb.setXLink(plot.getViewBox())
        self.raw_vb.setYRange(-1800, 1800, padding=0)

        self.curve_raw = pg.PlotDataItem(pen=pg.mkPen(C_RAW, width=1))
        self.raw_vb.addItem(self.curve_raw)
        self.curve_raw.setDownsampling(auto=True, method='peak')
        self.curve_raw.setClipToView(True)

        def _resize_raw_vb():
            self.raw_vb.setGeometry(plot.getViewBox().sceneBoundingRect())
            self.raw_vb.linkedViewChanged(plot.getViewBox(), self.raw_vb.XAxis)

        plot.getViewBox().sigResized.connect(_resize_raw_vb)
        return plot

    def _build_bottom_plot(self):
        plot = pg.PlotWidget()
        plot.setBackground(C_PANEL)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel('left', 'Level (0-100)', color=C_DIM)
        plot.setLabel('bottom', 'Time (s)', color=C_DIM)
        for ax in ('left', 'bottom'):
            plot.getAxis(ax).setPen(C_BORDER)
            plot.getAxis(ax).setTextPen(C_DIM)
        plot.setYRange(0, 100, padding=0)
        plot.setXRange(-self.WINDOW_SEC, 0, padding=0)

        def filled(color, alpha):
            c = QColor(color)
            c.setAlphaF(alpha)
            return plot.plot(
                pen=pg.mkPen(QColor(color), width=1.5),
                fillLevel=0, brush=QBrush(c))

        self.curve_noise = filled(C_NOISE, 0.45)
        self.curve_med = filled(C_MED, 0.45)
        self.curve_att = filled(C_ATT, 0.45)
        return plot

    # ===== 数据入口 =====
    def update_eeg_packet(self, pkt: dict):
        if not isinstance(pkt, dict):
            return
        self.attention = int(pkt.get('attention', self.attention))
        self.meditation = int(pkt.get('meditation', self.meditation))
        self.poor_signal = int(pkt.get('poor_signal', self.poor_signal))
        eeg_p = pkt.get('eeg_power')
        if isinstance(eeg_p, dict) and eeg_p:
            self.eeg_power = eeg_p

        source = pkt.get('source', 'sim')
        if source == 'serial':
            self.pill_source.set_state('live', '信号源：TGAM 本地串口')
        elif source == 'sim':
            self.pill_source.set_state('sim', '信号源：本地模拟')
        else:
            self.pill_source.set_state('offline', '信号源：离线')

        ps = self.poor_signal
        diag, color = decode_poor_signal(ps)
        self.ps_value.setText(str(ps))
        self.ps_value.setStyleSheet(f'color:{color}; font-size:18px; font-weight:bold;')
        self.ps_diag.setText(diag)
        self.ps_diag.setStyleSheet(f'color:{color}; font-weight:bold;')
        if ps == 0:
            self.pill_quality.set_state('live', f'信号质量：优 ({ps})')
        elif ps < 50:
            self.pill_quality.set_state('live', f'信号质量：良 ({ps})')
        elif ps < 150:
            self.pill_quality.set_state('sim', f'信号质量：弱 ({ps})')
        else:
            self.pill_quality.set_state('offline', f'信号质量：无 ({ps})')

    def update_raw_wave(self, raw_list):
        if raw_list:
            self.raw_wave = list(raw_list)

    def append_log(self, text):
        self.log_text.append(text)

    def _history_tick(self):
        eeg = self.eeg_power
        for key, _, _ in BAND_LIST:
            self.band_history[key].append(eeg.get(key, 0))
        self.att_history.append(self.attention)
        self.med_history.append(self.meditation)
        self.noise_history.append(min(100, self.poor_signal / 2))

    def _refresh_plots(self):
        N = len(self.att_history)
        if N > 0:
            x = list(range(-(N - 1), 1))
            for key, _, _ in BAND_LIST:
                ys = list(self.band_history[key])
                if ys:
                    self.band_curves[key].setData(x, ys)
            self.curve_att.setData(x, list(self.att_history))
            self.curve_med.setData(x, list(self.med_history))
            self.curve_noise.setData(x, list(self.noise_history))

        raw = self.raw_wave
        if raw:
            M = len(raw)
            raw_x = [(i - (M - 1)) / 500.0 for i in range(M)]
            self.curve_raw.setData(raw_x, raw)

        self.gauge_att.set_value(self.attention)
        self.gauge_med.set_value(self.meditation)

        if self.eeg_power:
            p = self.eeg_power
            alpha = p.get('low_alpha', 0) + p.get('high_alpha', 0)
            beta = p.get('low_beta', 0) + p.get('high_beta', 0)
            total = sum(p.values()) + 1
            alpha_ratio = alpha / total
            beta_ratio = beta / total
            risk = max(0.0, min(1.0,
                0.45 * beta_ratio / 0.10 +
                0.30 * max(0, (0.15 - alpha_ratio) / 0.15) +
                0.25 * (1 - self.attention / 100)
            ))
            self.risk_value.setText(f'{risk:.2f}')
            if risk > 0.66:
                c, tag = C_RISK, '高 · HIGH'
            elif risk > 0.33:
                c, tag = C_WARN, '中 · MEDIUM'
            else:
                c, tag = C_OK, '低 · LOW'
            self.risk_value.setStyleSheet(
                f'color:{c}; font-size:42px; font-weight:bold; background:transparent;')
            self.risk_tag.setText(tag)
            self.risk_tag.setStyleSheet(
                f'color:{c}; font-size:12px; letter-spacing:2px;')
