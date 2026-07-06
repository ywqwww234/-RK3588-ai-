"""
TGAM 脑波监测 - 边缘监控终端·脑电感知节点
- MindViewer 风格双轴时序图（频段堆叠 + RAW 原始波 + Att/Med/Noise）
- 自动检测串口；未识别 TGAM 设备 → 本地模拟数据模式（与其他模块行为一致）
- 协议解析层完整保留
- 仅脑电相关：心率/RR 属于 ESP32 手环模块，不在本节点
"""

import math
import random
import struct
import time
from collections import deque
from itertools import product

import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSizePolicy, QTextEdit, QVBoxLayout,
                             QWidget)
import pyqtgraph as pg


# ============================================================
# TGAM 协议常量
# ============================================================
SYNC = 0xAA
EXCODE = 0x55

CODE_POOR_SIGNAL = 0x02
CODE_HEART_RATE = 0x03
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_8BIT_RAW = 0x06
CODE_RAW_MARKER = 0x07
CODE_RAW_WAVE = 0x80
CODE_EEG_POWER = 0x81
CODE_ASIC_EEG_POWER = 0x83
CODE_RRINTERVAL = 0x86


# ============================================================
# 频段配色 - MindViewer 调试软件风格
# ============================================================
BAND_LIST = [
    ('delta',      'Delta',      '#9aa3ad'),
    ('theta',      'Theta',      '#6c8ebf'),
    ('low_alpha',  'Low Alpha',  '#b89020'),
    ('high_alpha', 'High Alpha', '#e6c200'),
    ('low_beta',   'Low Beta',   '#3b7a3b'),
    ('high_beta',  'High Beta',  '#5fb85f'),
    ('low_gamma',  'Low Gamma',  '#e89030'),
    ('mid_gamma',  'Mid Gamma',  '#d04030'),
]

# ============================================================
# 主题
# ============================================================
C_BG = '#0a0e27'
C_PANEL = '#0e1330'
C_PANEL_HI = '#1a2150'
C_BORDER = '#2a3270'
C_ACCENT = '#00d4ff'
C_ACCENT2 = '#7c4dff'
C_OK = '#2ed573'
C_WARN = '#ffa502'
C_RISK = '#ff4757'
C_TEXT = '#e8eaf6'
C_DIM = '#7986cb'
C_RAW = '#ff7a00'
C_ATT = '#2ed573'
C_MED = '#448aff'
C_NOISE = '#ff4757'

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
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: {C_BG};
    color: {C_ACCENT};
    letter-spacing: 1px;
}}
QLabel {{ color: {C_TEXT}; background: transparent; }}
QPushButton {{
    background-color: {C_PANEL_HI};
    color: {C_ACCENT};
    border: 1px solid {C_ACCENT};
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: bold;
    min-width: 60px;
}}
QPushButton:hover {{ background-color: {C_ACCENT}; color: {C_BG}; }}
QPushButton:disabled {{
    background-color: {C_PANEL};
    color: {C_DIM};
    border: 1px solid {C_DIM};
}}
QComboBox {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 100px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: 1px solid {C_ACCENT};
    selection-background-color: {C_ACCENT};
    selection-color: {C_BG};
}}
QCheckBox {{ color: {C_TEXT}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {C_DIM};
    background: {C_PANEL_HI};
    border-radius: 2px;
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border: 1px solid {C_ACCENT};
}}
QTextEdit {{
    background-color: #050818;
    color: {C_OK};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
}}
"""


# ============================================================
# PoorSignal 解码（依据官方 NS0304 文档）
# 0=优 / 200=脱头 / 25,26,27,29 为旗标位值的和
# ============================================================
def decode_poor_signal(ps):
    """返回 (描述, 状态颜色)"""
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


# ============================================================
# TGAM 数据解析器（底层逻辑保留，仅扩大原始波缓冲以支持30秒可视化）
# ============================================================
class TGAMParser:
    """TGAM 数据解析器 - 完整协议规范"""

    def __init__(self):
        self.buffer = bytearray()
        self.raw_wave_history = deque(maxlen=15360)  # 30s @ 512Hz
        self.attention = 0
        self.meditation = 0
        self.poor_signal = 200
        self.heart_rate = 0
        self.rr_interval = 0
        self.eeg_power = {}
        self.last_update = time.time()
        self.packet_count = 0
        self.error_count = 0

    def parse_byte(self, byte):
        self.buffer.append(byte)
        while True:
            if len(self.buffer) < 2:
                break
            if not (self.buffer[0] == SYNC and self.buffer[1] == SYNC):
                del self.buffer[0]
                continue
            if len(self.buffer) < 3:
                break
            payload_length = self.buffer[2]
            if payload_length > 169:
                del self.buffer[0]
                self.error_count += 1
                continue
            packet_total_len = 3 + payload_length + 1
            if len(self.buffer) < packet_total_len:
                break
            payload = self.buffer[3:3 + payload_length]
            received_checksum = self.buffer[3 + payload_length]
            calculated_checksum = self.calculate_checksum(payload)
            if received_checksum == calculated_checksum:
                self.parse_payload(payload)
                self.last_update = time.time()
                self.packet_count += 1
            else:
                self.error_count += 1
            del self.buffer[:packet_total_len]
        if len(self.buffer) > 2048:
            self.buffer = bytearray()

    def calculate_checksum(self, payload):
        return (~sum(payload) & 0xFF)

    def parse_payload(self, payload):
        index = 0
        while index < len(payload):
            excode_count = 0
            while index < len(payload) and payload[index] == EXCODE:
                excode_count += 1
                index += 1
            if index >= len(payload):
                break
            code = payload[index]
            index += 1
            if code >= 0x80:
                if index >= len(payload):
                    break
                value_length = payload[index]
                index += 1
            else:
                value_length = 1
            if index + value_length > len(payload):
                break
            value = payload[index:index + value_length]
            index += value_length
            self.parse_data(code, value)

    def parse_data(self, code, value):
        try:
            if code == CODE_POOR_SIGNAL:
                self.poor_signal = value[0]
            elif code == CODE_HEART_RATE:
                self.heart_rate = value[0]
            elif code == CODE_ATTENTION:
                self.attention = value[0]
            elif code == CODE_MEDITATION:
                self.meditation = value[0]
            elif code == CODE_RAW_WAVE:
                if len(value) == 2:
                    raw_value = struct.unpack('>h', bytes(value))[0]
                    self.raw_wave_history.append(raw_value)
            elif code == CODE_EEG_POWER:
                if len(value) == 24:
                    self.eeg_power = {
                        'delta':      struct.unpack('>I', b'\x00' + value[0:3])[0],
                        'theta':      struct.unpack('>I', b'\x00' + value[3:6])[0],
                        'low_alpha':  struct.unpack('>I', b'\x00' + value[6:9])[0],
                        'high_alpha': struct.unpack('>I', b'\x00' + value[9:12])[0],
                        'low_beta':   struct.unpack('>I', b'\x00' + value[12:15])[0],
                        'high_beta':  struct.unpack('>I', b'\x00' + value[15:18])[0],
                        'low_gamma':  struct.unpack('>I', b'\x00' + value[18:21])[0],
                        'mid_gamma':  struct.unpack('>I', b'\x00' + value[21:24])[0],
                    }
            elif code == CODE_RRINTERVAL:
                if len(value) == 2:
                    self.rr_interval = struct.unpack('>H', bytes(value))[0]
        except Exception as e:
            print(f"解析数据错误: {e}")

    def is_timeout(self, timeout=2.0):
        return (time.time() - self.last_update) > timeout


# ============================================================
# 圆环仪表（专注度/放松度）
# ============================================================
class CircularGauge(QWidget):
    def __init__(self, title, color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.value = 0
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v):
        self.value = max(0, min(100, int(v)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height())
        m = 10
        rect = QRectF(
            (self.width() - side) / 2 + m,
            (self.height() - side) / 2 + m,
            side - 2 * m, side - 2 * m,
        )
        # 底环
        p.setPen(QPen(QColor(C_BORDER), 9, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 0, 360 * 16)
        # 进度环
        p.setPen(QPen(QColor(self.color), 9, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90 * 16, int(-self.value * 3.6 * 16))
        # 中央
        p.setPen(QColor(C_TEXT))
        p.setFont(QFont('Microsoft YaHei', max(16, int(side / 8)), QFont.Bold))
        p.drawText(rect, Qt.AlignCenter, f'{self.value}')
        # 标题
        p.setPen(QColor(C_DIM))
        p.setFont(QFont('Microsoft YaHei', 9))
        tr = QRectF(rect.x(), rect.y() + rect.height() * 0.72, rect.width(), 20)
        p.drawText(tr, Qt.AlignCenter, self.title)


# ============================================================
# 状态胶囊
# ============================================================
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


# ============================================================
# 频段图例（横向色块标签）
# ============================================================
class BandLegend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)
        for key, label, color in BAND_LIST:
            w = QLabel(f'<span style="color:{color}">■</span> '
                       f'<span style="color:{C_DIM}">{label}</span>')
            layout.addWidget(w)
        # RAW 标记
        raw_lbl = QLabel(f'<span style="color:{C_RAW}">━</span> '
                         f'<span style="color:{C_DIM}">RAW Wave</span>')
        layout.addStretch()
        layout.addWidget(raw_lbl)


# ============================================================
# 主窗口
# ============================================================
class EEGMonitorWindow(QMainWindow):

    MODE_OFFLINE = 'offline'
    MODE_SERIAL = 'serial'
    MODE_SIM = 'sim'

    WINDOW_SEC = 30  # 显示窗口（秒）

    def __init__(self):
        super().__init__()
        self.parser = TGAMParser()
        self.serial_port = None
        self.mode = self.MODE_OFFLINE
        self.connect_time = 0
        self.raw_data_buffer = ''
        self.show_raw_data = False

        # 历史缓冲 - 30秒窗口，1Hz采样
        self.band_history = {k: deque(maxlen=self.WINDOW_SEC)
                             for k, _, _ in BAND_LIST}
        self.attention_history = deque(maxlen=self.WINDOW_SEC)
        self.meditation_history = deque(maxlen=self.WINDOW_SEC)
        self.noise_history = deque(maxlen=self.WINDOW_SEC)

        # 模拟基线
        self._sim_phase = random.random() * 100
        self._sim_attention = 50.0
        self._sim_meditation = 50.0
        self._sim_last_band_t = 0
        self._sim_noise_burst_t = 0

        self.init_ui()

        # 显示刷新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)

        # 数据读取
        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self.read_data_tick)
        self.read_timer.start(20)

        # 1Hz 历史采样
        self.history_timer = QTimer()
        self.history_timer.timeout.connect(self.history_tick)
        self.history_timer.start(1000)

        # 看门狗
        self.watchdog_timer = QTimer()
        self.watchdog_timer.timeout.connect(self.check_signal_watchdog)
        self.watchdog_timer.start(1000)

        # 自动连接
        QTimer.singleShot(300, self.auto_connect)

    # =====================================================
    # UI
    # =====================================================
    def init_ui(self):
        self.setWindowTitle('边缘监控终端 · 脑电感知节点 (TGAM Edge Node)')
        self.setGeometry(60, 50, 1400, 860)
        self.setStyleSheet(QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # === 顶部标题栏 ===
        header = QHBoxLayout()
        title = QLabel('🧠  脑 电 感 知 节 点  ·  TGAM  EDGE  NODE')
        title.setStyleSheet(f'color:{C_ACCENT}; font-size:18px;'
                            f'font-weight:bold; letter-spacing:3px;')
        header.addWidget(title)
        header.addStretch()
        self.status_pill = StatusPill()
        header.addWidget(self.status_pill)
        self.quality_pill = StatusPill()
        self.quality_pill.set_state('offline', '信号质量：--')
        header.addWidget(self.quality_pill)
        root.addLayout(header)

        # === 控制条 ===
        ctrl_group = QGroupBox('连接控制 · CONNECTION')
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('串口:'))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(220)
        ctrl.addWidget(self.port_combo)
        ctrl.addWidget(QLabel('波特率:'))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['57600', '115200', '9600', '38400'])
        self.baud_combo.setCurrentText('57600')
        ctrl.addWidget(self.baud_combo)
        self.refresh_btn = QPushButton('刷新')
        self.refresh_btn.clicked.connect(self.refresh_ports)
        ctrl.addWidget(self.refresh_btn)
        self.connect_btn = QPushButton('连接')
        self.connect_btn.clicked.connect(self.toggle_connection)
        ctrl.addWidget(self.connect_btn)
        self.sim_btn = QPushButton('进入模拟')
        self.sim_btn.clicked.connect(
            lambda: self.enter_simulation('用户手动切换至模拟模式'))
        ctrl.addWidget(self.sim_btn)
        self.raw_checkbox = QCheckBox('原始字节流')
        self.raw_checkbox.stateChanged.connect(self.toggle_raw_data)
        ctrl.addWidget(self.raw_checkbox)
        ctrl.addStretch()
        self.packet_label = QLabel('数据包: 0')
        self.error_label = QLabel('错误: 0')
        for lbl in (self.packet_label, self.error_label):
            lbl.setStyleSheet(f'color:{C_DIM}; padding:0 8px;')
        ctrl.addWidget(self.packet_label)
        ctrl.addWidget(self.error_label)
        ctrl_group.setLayout(ctrl)
        root.addWidget(ctrl_group)

        # === 主体：左时序图 + 右仪表盘 ===
        body = QHBoxLayout()
        body.setSpacing(10)

        # ---- 左：双层时序图 ----
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        # 上图：频段堆叠 + RAW
        top_group = QGroupBox('频段功率与原始脑电波 · POWER + RAW')
        top_lay = QVBoxLayout()
        top_lay.setContentsMargins(8, 8, 8, 4)

        self.plot_top = self._build_top_plot()
        top_lay.addWidget(self.plot_top, 1)
        top_lay.addWidget(BandLegend())
        top_group.setLayout(top_lay)
        left_col.addWidget(top_group, 2)

        # 下图：Att/Med/Noise
        bot_group = QGroupBox('精神状态时序 · ATTENTION / MEDITATION / NOISE')
        bot_lay = QVBoxLayout()
        bot_lay.setContentsMargins(8, 8, 8, 4)
        self.plot_bot = self._build_bottom_plot()
        bot_lay.addWidget(self.plot_bot, 1)

        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(8, 0, 8, 0)
        for color, text in [(C_NOISE, 'Noise (信号噪声)'),
                            (C_ATT, 'Attention (专注度)'),
                            (C_MED, 'Meditation (放松度)')]:
            legend_row.addWidget(
                QLabel(f'<span style="color:{color}">■</span> '
                       f'<span style="color:{C_DIM}">{text}</span>'))
        legend_row.addStretch()
        bot_lay.addLayout(legend_row)
        bot_group.setLayout(bot_lay)
        left_col.addWidget(bot_group, 1)

        body.addLayout(left_col, 7)

        # ---- 右：仪表 + 风险 + 信号质量 ----
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        gauge_group = QGroupBox('精神状态 · MENTAL STATE')
        glay = QHBoxLayout()
        self.gauge_att = CircularGauge('专注度 Attention', C_ATT)
        self.gauge_med = CircularGauge('放松度 Meditation', C_MED)
        glay.addWidget(self.gauge_att)
        glay.addWidget(self.gauge_med)
        gauge_group.setLayout(glay)
        right_col.addWidget(gauge_group)

        # 风险评分
        risk_group = QGroupBox('情绪风险评分 · RISK INDEX')
        rlay = QVBoxLayout()
        self.risk_value = QLabel('0.00')
        self.risk_value.setAlignment(Qt.AlignCenter)
        self.risk_value.setStyleSheet(
            f'color:{C_OK}; font-size:42px; font-weight:bold; '
            f'background:transparent;')
        self.risk_label_text = QLabel('低 · LOW')
        self.risk_label_text.setAlignment(Qt.AlignCenter)
        self.risk_label_text.setStyleSheet(
            f'color:{C_DIM}; font-size:12px; letter-spacing:2px;')
        self.risk_bar = QLabel()
        self.risk_bar.setFixedHeight(6)
        self.risk_bar.setStyleSheet(
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_OK}, stop:0.5 {C_WARN}, stop:1 {C_RISK});'
            f'border-radius:3px;')
        rlay.addWidget(self.risk_value)
        rlay.addWidget(self.risk_label_text)
        rlay.addSpacing(4)
        rlay.addWidget(self.risk_bar)
        risk_group.setLayout(rlay)
        right_col.addWidget(risk_group)

        # 信号质量解读
        sq_group = QGroupBox('信号质量诊断 · SIGNAL DIAGNOSIS')
        sqlay = QGridLayout()
        sqlay.setContentsMargins(8, 6, 8, 6)
        sqlay.addWidget(QLabel('PoorSignal:'), 0, 0)
        self.ps_value = QLabel('--')
        self.ps_value.setStyleSheet(
            f'color:{C_ACCENT}; font-size:18px; font-weight:bold;')
        sqlay.addWidget(self.ps_value, 0, 1)
        sqlay.addWidget(QLabel('诊断:'), 1, 0)
        self.ps_diag = QLabel('--')
        self.ps_diag.setStyleSheet(f'color:{C_DIM}; font-weight:bold;')
        self.ps_diag.setWordWrap(True)
        sqlay.addWidget(self.ps_diag, 1, 1)
        sqlay.setColumnStretch(1, 1)
        sq_group.setLayout(sqlay)
        right_col.addWidget(sq_group)

        right_col.addStretch()
        body.addLayout(right_col, 3)
        root.addLayout(body, 1)

        # === 底部日志 ===
        log_row = QHBoxLayout()
        log_group = QGroupBox('系统日志 · LOG')
        llay = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(110)
        self.log_text.setReadOnly(True)
        llay.addWidget(self.log_text)
        log_group.setLayout(llay)
        log_row.addWidget(log_group, 2)

        raw_group = QGroupBox('原始字节流 · RAW BYTES')
        rlay2 = QVBoxLayout()
        self.raw_text = QTextEdit()
        self.raw_text.setMaximumHeight(110)
        self.raw_text.setReadOnly(True)
        rlay2.addWidget(self.raw_text)
        raw_group.setLayout(rlay2)
        log_row.addWidget(raw_group, 1)
        root.addLayout(log_row)

        self.refresh_ports()

    def _build_top_plot(self):
        """上图：8 频段堆叠面积 + RAW Wave 双轴"""
        plot = pg.PlotWidget()
        plot.setBackground(C_PANEL)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel('left', 'Power', color=C_DIM)
        plot.setLabel('bottom', 'Time (s)', color=C_DIM)
        plot.showAxis('right')
        plot.setLabel('right', 'RAW Wave (raw)', color=C_RAW)
        for ax in ('left', 'bottom', 'right'):
            plot.getAxis(ax).setPen(C_BORDER)
            plot.getAxis(ax).setTextPen(C_DIM if ax != 'right' else C_RAW)
        plot.setXRange(-self.WINDOW_SEC, 0, padding=0)

        # 8 频段填色曲线
        self.band_curves = {}
        for key, label, color in BAND_LIST:
            qc = QColor(color)
            fill = QColor(color)
            fill.setAlphaF(0.45)
            curve = plot.plot(
                pen=pg.mkPen(qc, width=1.2),
                fillLevel=0, brush=QBrush(fill),
                name=label,
            )
            self.band_curves[key] = curve

        # 右轴 ViewBox（RAW）
        self.raw_vb = pg.ViewBox()
        plot.scene().addItem(self.raw_vb)
        plot.getAxis('right').linkToView(self.raw_vb)
        self.raw_vb.setXLink(plot.getViewBox())
        self.raw_vb.setYRange(-1800, 1800, padding=0)

        self.curve_raw = pg.PlotDataItem(
            pen=pg.mkPen(C_RAW, width=1))
        self.raw_vb.addItem(self.curve_raw)
        self.curve_raw.setDownsampling(auto=True, method='peak')
        self.curve_raw.setClipToView(True)

        def _resize_raw_vb():
            self.raw_vb.setGeometry(plot.getViewBox().sceneBoundingRect())
            self.raw_vb.linkedViewChanged(
                plot.getViewBox(), self.raw_vb.XAxis)

        plot.getViewBox().sigResized.connect(_resize_raw_vb)
        return plot

    def _build_bottom_plot(self):
        """下图：Attention / Meditation / Noise"""
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

    # =====================================================
    # 自动连接 / 模式
    # =====================================================
    def auto_connect(self):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.enter_simulation('未发现可用串口，已切换本地模拟数据模式')
            return
        target = self._guess_tgam_port(ports)
        if target is None:
            self.enter_simulation('未识别到 TGAM 设备，已切换本地模拟数据模式')
            return
        for i in range(self.port_combo.count()):
            if self.port_combo.itemText(i).startswith(target.device):
                self.port_combo.setCurrentIndex(i)
                break
        self._do_serial_connect(target.device, int(self.baud_combo.currentText()))

    def _guess_tgam_port(self, ports):
        keywords = ('bluetooth', 'hc-05', 'tgam', 'ch340', 'cp210',
                    'usb-serial', 'silabs', 'ftdi')
        for p in ports:
            desc = (p.description or '').lower()
            if any(k in desc for k in keywords):
                return p
        return None

    def enter_simulation(self, reason=''):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None
        self.mode = self.MODE_SIM
        self.connect_btn.setText('连接')
        self.sim_btn.setText('退出模拟')
        self.status_pill.set_state('sim', '信号源：本地模拟')
        self.quality_pill.set_state('sim', '信号质量：模拟')
        self.parser = TGAMParser()
        self.parser.poor_signal = 0
        self._clear_history()
        if reason:
            self.log(reason)

    def _set_offline(self):
        self.mode = self.MODE_OFFLINE
        self.connect_btn.setText('连接')
        self.sim_btn.setText('进入模拟')
        self.status_pill.set_state('offline', '信号源：离线')
        self.quality_pill.set_state('offline', '信号质量：--')

    def _clear_history(self):
        for h in self.band_history.values():
            h.clear()
        self.attention_history.clear()
        self.meditation_history.clear()
        self.noise_history.clear()

    # =====================================================
    # 串口
    # =====================================================
    def refresh_ports(self):
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.port_combo.addItem(f'{port.device} - {port.description}')
        if not ports:
            self.log('未发现可用串口')

    def toggle_connection(self):
        if self.mode == self.MODE_SERIAL:
            self.disconnect_serial()
            self._set_offline()
            return
        port_info = self.port_combo.currentText()
        if not port_info:
            self.log('请先选择串口')
            return
        port_name = port_info.split(' - ', 1)[0].strip()
        baud_rate = int(self.baud_combo.currentText())
        self._do_serial_connect(port_name, baud_rate)

    def _do_serial_connect(self, port_name, baud_rate):
        self.raw_data_buffer = ''
        self.raw_text.clear()
        try:
            self.serial_port = serial.Serial(
                port=port_name, baudrate=baud_rate, timeout=0,
                bytesize=8, parity='N', stopbits=1,
                xonxoff=False, rtscts=False, dsrdtr=False, exclusive=False,
            )
            try:
                self.serial_port.setDTR(False)
                self.serial_port.setRTS(False)
            except Exception:
                pass
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            self.parser = TGAMParser()
            self._clear_history()
            self.mode = self.MODE_SERIAL
            self.connect_time = time.time()
            self.connect_btn.setText('断开')
            self.sim_btn.setText('进入模拟')
            self.status_pill.set_state('live', f'信号源：TGAM ({port_name})')
            self.quality_pill.set_state('sim', '信号质量：等待数据…')
            self.log(f'已连接 {port_name} @ {baud_rate}')
        except Exception as e:
            self.log(f'连接失败: {e}')
            self.enter_simulation('串口打开失败，已切换本地模拟数据模式')

    def disconnect_serial(self):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.log('已断开串口')

    # =====================================================
    # 数据读取
    # =====================================================
    def read_data_tick(self):
        if self.mode == self.MODE_SIM:
            self.generate_simulated_data()
            return
        if self.mode != self.MODE_SERIAL:
            return
        if not (self.serial_port and self.serial_port.is_open):
            return
        try:
            waiting = self.serial_port.in_waiting
            if waiting <= 0:
                return
            data = self.serial_port.read(waiting)
            if not data:
                return
            if self.show_raw_data:
                hex_str = ' '.join(f'{b:02X}' for b in data)
                self.raw_data_buffer = (self.raw_data_buffer + hex_str + ' ')[-1200:]
                self.raw_text.setText(self.raw_data_buffer)
            for byte in data:
                self.parser.parse_byte(byte)
        except Exception as e:
            self.log(f'读取错误: {e}')

    def check_signal_watchdog(self):
        if self.mode != self.MODE_SERIAL:
            return
        if (time.time() - self.connect_time) > 5 and self.parser.packet_count == 0:
            self.disconnect_serial()
            self.enter_simulation('TGAM 串口连接成功但无数据，已切换模拟模式')
            return
        if self.parser.packet_count > 0 and self.parser.is_timeout(3.0):
            self.disconnect_serial()
            self.enter_simulation('TGAM 信号中断 3 秒，已切换模拟模式')

    # =====================================================
    # 模拟数据 - 符合 TGAM 真实输出节拍
    # =====================================================
    def generate_simulated_data(self):
        t = time.time()
        self._sim_phase += 0.02

        # RAW: 每 tick 生成 10 个样本 ≈ 500Hz（接近真实 512Hz）
        for k in range(10):
            sub_t = t + k * 0.002
            sample = (
                7000 * math.sin(2 * math.pi * 2  * sub_t) +    # delta 2Hz
                5000 * math.sin(2 * math.pi * 6  * sub_t) +    # theta 6Hz
                6000 * math.sin(2 * math.pi * 10 * sub_t) +    # alpha 10Hz
                3500 * math.sin(2 * math.pi * 18 * sub_t) +    # beta 18Hz
                random.randint(-1500, 1500)
            )
            self.parser.raw_wave_history.append(int(sample))

        # eSense（专注/放松）：缓慢漫游
        self._sim_attention = max(25, min(95,
            self._sim_attention + random.uniform(-2.5, 2.5) +
            4 * math.sin(self._sim_phase * 0.5)))
        self._sim_meditation = max(25, min(95,
            self._sim_meditation + random.uniform(-2.5, 2.5) +
            4 * math.cos(self._sim_phase * 0.4)))
        self.parser.attention = int(self._sim_attention)
        self.parser.meditation = int(self._sim_meditation)

        # PoorSignal：默认 0，偶尔模拟干扰
        if t - self._sim_noise_burst_t > 12 and random.random() < 0.005:
            self._sim_noise_burst_t = t
        if t - self._sim_noise_burst_t < 2.0:
            self.parser.poor_signal = random.choice([25, 26, 51])
        else:
            self.parser.poor_signal = 0

        # 频段：每 1 秒刷新一次（符合 TGAM 大包节拍）
        if t - self._sim_last_band_t > 1.0:
            self._sim_last_band_t = t
            base = self._sim_phase
            self.parser.eeg_power = {
                'delta':      int(1800000 + 600000 * abs(math.sin(base * 0.13))),
                'theta':      int(400000  + 200000 * abs(math.sin(base * 0.21))),
                'low_alpha':  int(80000   + 50000  * abs(math.sin(base * 0.33))),
                'high_alpha': int(60000   + 40000  * abs(math.sin(base * 0.41))),
                'low_beta':   int(40000   + 25000  * abs(math.sin(base * 0.53))),
                'high_beta':  int(35000   + 20000  * abs(math.sin(base * 0.61))),
                'low_gamma':  int(20000   + 12000  * abs(math.sin(base * 0.73))),
                'mid_gamma':  int(15000   + 8000   * abs(math.sin(base * 0.81))),
            }
            self.parser.last_update = t
            self.parser.packet_count += 1

    # =====================================================
    # 1Hz 历史采样
    # =====================================================
    def history_tick(self):
        eeg = self.parser.eeg_power
        for key, _, _ in BAND_LIST:
            self.band_history[key].append(eeg.get(key, 0))
        self.attention_history.append(self.parser.attention)
        self.meditation_history.append(self.parser.meditation)
        # Noise: PoorSignal 0-200 -> 0-100
        self.noise_history.append(min(100, self.parser.poor_signal / 2))

    # =====================================================
    # 显示刷新
    # =====================================================
    def update_display(self):
        # === 上图：频段 + RAW ===
        N = len(self.attention_history)
        # X 轴：相对时间（秒），最右边是 0（当前）
        if N > 0:
            band_x = list(range(-(N - 1), 1))
        else:
            band_x = []

        for key, _, _ in BAND_LIST:
            ys = list(self.band_history[key])
            if ys:
                self.band_curves[key].setData(band_x, ys)

        # RAW Wave
        raw = list(self.parser.raw_wave_history)
        if raw:
            M = len(raw)
            # 整体压在 [-WINDOW, 0] 范围内（500Hz × 30s = 15000 个点）
            raw_x = [(i - (M - 1)) / 500.0 for i in range(M)]
            self.curve_raw.setData(raw_x, raw)

        # === 下图：Att / Med / Noise ===
        if N > 0:
            self.curve_att.setData(band_x, list(self.attention_history))
            self.curve_med.setData(band_x, list(self.meditation_history))
            self.curve_noise.setData(band_x, list(self.noise_history))

        # === 仪表 ===
        att = self.parser.attention
        med = self.parser.meditation
        self.gauge_att.set_value(att)
        self.gauge_med.set_value(med)

        # === 风险评分 ===
        if self.parser.eeg_power:
            p = self.parser.eeg_power
            alpha = p.get('low_alpha', 0) + p.get('high_alpha', 0)
            beta = p.get('low_beta', 0) + p.get('high_beta', 0)
            total = sum(p.values()) + 1
            alpha_ratio = alpha / total
            beta_ratio = beta / total
            risk = max(0.0, min(1.0,
                0.45 * beta_ratio / 0.10 +
                0.30 * max(0, (0.15 - alpha_ratio) / 0.15) +
                0.25 * (1 - att / 100)
            ))
            self.risk_value.setText(f'{risk:.2f}')
            if risk > 0.66:
                c = C_RISK
                tag = '高 · HIGH'
            elif risk > 0.33:
                c = C_WARN
                tag = '中 · MEDIUM'
            else:
                c = C_OK
                tag = '低 · LOW'
            self.risk_value.setStyleSheet(
                f'color:{c}; font-size:42px; font-weight:bold; '
                f'background:transparent;')
            self.risk_label_text.setText(tag)
            self.risk_label_text.setStyleSheet(
                f'color:{c}; font-size:12px; letter-spacing:2px;')

        # === 包/错计数 ===
        self.packet_label.setText(f'数据包: {self.parser.packet_count}')
        self.error_label.setText(f'错误: {self.parser.error_count}')

        # === 信号质量解读 ===
        ps = self.parser.poor_signal
        diag, color = decode_poor_signal(ps)
        self.ps_value.setText(str(ps))
        self.ps_value.setStyleSheet(
            f'color:{color}; font-size:18px; font-weight:bold;')
        self.ps_diag.setText(diag)
        self.ps_diag.setStyleSheet(f'color:{color}; font-weight:bold;')

        # 顶部信号质量胶囊
        if self.mode == self.MODE_SERIAL:
            if ps == 0:
                self.quality_pill.set_state('live', f'信号质量：优 ({ps})')
            elif ps < 50:
                self.quality_pill.set_state('live', f'信号质量：良 ({ps})')
            elif ps < 150:
                self.quality_pill.set_state('sim', f'信号质量：弱 ({ps})')
            else:
                self.quality_pill.set_state('offline', f'信号质量：无 ({ps})')

    def toggle_raw_data(self, state):
        self.show_raw_data = (state == Qt.Checked)

    def log(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(
            f'<span style="color:{C_DIM}">[{timestamp}]</span> {message}')

    def closeEvent(self, event):
        self.disconnect_serial()
        event.accept()


class BrainMonitorWidget(QWidget):
    """可嵌入学生端的脑电监测组件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._host = EEGMonitorWindow()
        core = self._host.takeCentralWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(core)

    def shutdown(self):
        try:
            self._host.update_timer.stop()
            self._host.read_timer.stop()
            self._host.history_timer.stop()
            self._host.watchdog_timer.stop()
            self._host.disconnect_serial()
        except Exception:
            pass


# ============================================================
# 入口
# ============================================================
def main():
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    app = QApplication([])
    app.setStyle('Fusion')
    window = EEGMonitorWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
