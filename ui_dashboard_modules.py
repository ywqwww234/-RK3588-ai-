"""
三模态子区构造器（D:/G/3）。
分别构造视觉 / 心率 / 脑电 三个 ModuleCard，并把所需子控件挂到 self（MainWindow）上：
  视觉:  self.video_label / self.expr_label / self.eye_bar / self.posture_bar / ...
  心率:  self.card_hr_big / self.card_sdnn_big / self.row_sdnn / row_rmssd / row_lfhf / row_pnn50 / row_hf
         self.p_ppg / curve_wave_ppg / p_ecg / curve_wave_ecg / p_hr / p_hrv
         self.trend_plot / trend_curve
  脑电:  self.bci_panel (BCIPanel)
"""

from collections import deque
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QGridLayout, QSizePolicy, QFrame, QProgressBar,
                              QScrollArea)
import pyqtgraph as pg
import theme as _theme
from theme import (C_BG, C_PANEL, C_BORDER, C_ACCENT, C_DIM, C_TEXT,
                   C_VISION, C_HEART, C_BRAIN, C_OK, C_WARN, C_RISK,
                   C_PPG, C_ECG, C_HR, C_HRV, C_PANEL_HI, C_BG_DEEP,
                   SP_XS, SP_SM, SP_MD, SP_LG,
                   FS_TINY, FS_XS, FS_SM, FS_MD,
                   R_SM, R_MD, BORDER_W)
from ui_widgets import (ModuleCard, BigValueCard, MetricRow, StatusPill,
                         ModalContribGauge)


def _bar_label(name, color):
    """单条横向进度条 (名称 + ProgressBar + 数值)。"""
    wrap = QWidget()
    wrap.setStyleSheet('background:transparent;')
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(8)
    lbl = QLabel(name)
    lbl.setStyleSheet(f'color:{C_DIM}; font-size:11px; background:transparent;')
    lbl.setMinimumWidth(60)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(8)
    bar.setStyleSheet(
        f'QProgressBar {{ background:{C_BG_DEEP}; border:1px solid {C_BORDER}; border-radius:3px; }}'
        f'QProgressBar::chunk {{ background:{color}; border-radius:2px; }}')
    val = QLabel('0%')
    val.setStyleSheet(f'color:{color}; font-size:11px; font-weight:bold; background:transparent;')
    val.setMinimumWidth(40)
    val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(lbl)
    lay.addWidget(bar, 1)
    lay.addWidget(val)
    wrap._bar = bar
    wrap._val = val
    return wrap


def build_vision_card(self):
    card = ModuleCard('视觉感知 · VISION', 'YOLOv8n · FER+ · MediaPipe', C_VISION)
    card.status_pill.set_state('sim', '模型: 就绪')

    # === 摄像头主显示 - 弹性大画幅 (min 480×320, 优先扩展) ===
    self.video_label = QLabel('正在初始化视觉识别引擎…')
    self.video_label.setMinimumSize(440, 300)
    self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.video_label.setAlignment(Qt.AlignCenter)
    self.video_label.setScaledContents(True)
    self.video_label.setStyleSheet(
        f'background:{C_BG_DEEP}; color:{C_VISION}; font-size:{FS_MD}px; '
        f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_MD}px;')
    card.add_content(self.video_label)

    # === 表情与视觉风险状态行 ===
    self.expr_label = QLabel('当前状态: 等待人脸…')
    self.expr_label.setStyleSheet(
        f'color:{C_TEXT}; font-size:12px; letter-spacing:0.6px; background:transparent;')
    card.add_content(self.expr_label)

    # === 视觉子指标条 (4项, 工业级配色) ===
    self.bar_eye = _bar_label('眼疲劳', C_VISION)
    self.bar_posture = _bar_label('姿态风险', C_VISION)
    self.bar_forward = _bar_label('探头', C_WARN)
    self.bar_down = _bar_label('低头', C_RISK)
    card.add_content(self.bar_eye)
    card.add_content(self.bar_posture)
    card.add_content(self.bar_forward)
    card.add_content(self.bar_down)

    # === 视觉风险评分(紧凑式) ===
    risk_row = QHBoxLayout(); risk_row.setSpacing(6)
    self.lbl_vis_risk = QLabel('0.00')
    self.lbl_vis_risk.setStyleSheet(
        f'color:{C_OK}; font-size:20px; font-weight:bold; '
        f'background:{C_BG_DEEP}; border:1px solid {C_BORDER}; '
        f'border-radius:5px; padding:4px 10px; font-family:Consolas;')
    self.lbl_vis_risk.setAlignment(Qt.AlignCenter)
    self.lbl_vis_tag = QLabel('LOW')
    self.lbl_vis_tag.setStyleSheet(
        f'color:{C_OK}; font-size:10px; letter-spacing:2px; background:transparent;')
    self.lbl_vis_tag.setAlignment(Qt.AlignCenter)
    risk_row.addWidget(self.lbl_vis_risk)
    risk_row.addWidget(self.lbl_vis_tag)
    risk_row.addStretch()
    card.add_content(risk_row)

    # === NN 模态贡献仪表 ===
    self.vision_contrib_gauge = ModalContribGauge('vision', C_VISION)
    card.add_content(self.vision_contrib_gauge)

    self.vision_card = card
    return card


def build_heart_card(self):
    card = ModuleCard('心率与生理 · CARDIAC', 'PPG · ECG · HR · HRV', C_HEART)
    card.status_pill.set_state('sim', '信号: 模拟')

    pg.setConfigOptions(antialias=True)

    # 四联波形图
    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(C_PANEL)
    glw.setMinimumHeight(320)  # 提高让四联波形看清形状（对齐 1.py 独立版）
    glw.setMinimumWidth(0)

    self.p_ppg = glw.addPlot(row=0, col=0)
    _theme.style_pg_plot(self.p_ppg, 'PPG · 红外信号', C_PPG)
    self.p_ppg.disableAutoRange()
    self.p_ppg.setXRange(0, 800, padding=0)
    self.p_ppg.setYRange(8500, 11500, padding=0)
    self.curve_wave_ppg_glow = self.p_ppg.plot(
        pen=pg.mkPen(QColor(255, 82, 119, 95), width=6.0),
    )
    self.curve_wave_ppg = self.p_ppg.plot(
        pen=pg.mkPen(C_PPG, width=3.3),
        fillLevel=9000,
        brush=pg.mkBrush(QColor(255, 82, 119, 35))
    )
    self.ppg_peak_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush('#FFD166'), pen=pg.mkPen('#FFE8A3', width=1.2))
    self.p_ppg.addItem(self.ppg_peak_scatter)

    self.p_ecg = glw.addPlot(row=1, col=0)
    _theme.style_pg_plot(self.p_ecg, 'ECG · 模拟波形', C_ECG)
    self.p_ecg.disableAutoRange()
    self.p_ecg.setXRange(0, 800, padding=0)
    self.p_ecg.setYRange(-1.8, 1.8, padding=0)
    self.curve_wave_ecg_glow = self.p_ecg.plot(
        pen=pg.mkPen(QColor(0, 229, 255, 90), width=5.2),
    )
    self.curve_wave_ecg = self.p_ecg.plot(
        pen=pg.mkPen(C_ECG, width=3.0),
        fillLevel=-1.2,
        brush=pg.mkBrush(QColor(0, 229, 255, 26))
    )
    self.ecg_peak_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush('#7CFBFF'), pen=pg.mkPen('#C6FFFF', width=1.0))
    self.p_ecg.addItem(self.ecg_peak_scatter)

    self.p_hr = glw.addPlot(row=2, col=0)
    _theme.style_pg_plot(self.p_hr, 'HR · 心率 (bpm)', C_HEART)
    self.p_hr.disableAutoRange()
    self.p_hr.setXRange(0, 200, padding=0)
    self.p_hr.setYRange(55, 110, padding=0)   # 扩展到覆盖紧张+静息（对齐 1.py）
    self.curve_wave_hr = self.p_hr.plot(pen=pg.mkPen(C_HEART, width=2.4))

    self.p_hrv = glw.addPlot(row=3, col=0)
    _theme.style_pg_plot(self.p_hrv, 'HRV · 心率变异性 (ms)', C_HRV)
    self.p_hrv.disableAutoRange()
    self.p_hrv.setXRange(0, 200, padding=0)
    self.p_hrv.setYRange(10, 80, padding=0)    # 覆盖抑郁倾向高低区间
    self.curve_wave_hrv = self.p_hrv.plot(pen=pg.mkPen(C_HRV, width=2.4))

    self._buf_ppg = deque([0.0] * 800, maxlen=800)
    self._buf_ecg = deque([0.0] * 800, maxlen=800)
    self._buf_hr  = deque([0.0] * 200, maxlen=200)
    self._buf_hrv = deque([0.0] * 200, maxlen=200)
    self._buf_sdnn_trend = deque(maxlen=600)

    card.add_content(glw)

    # 大数（HR / RMSSD）
    big_row = QHBoxLayout()
    big_row.setSpacing(8)
    self.card_hr_big = BigValueCard('心率 HR', 'bpm', C_HEART)
    self.card_sdnn_big = BigValueCard('HRV (RMSSD)', 'ms', C_HRV)
    big_row.addWidget(self.card_hr_big)
    big_row.addWidget(self.card_sdnn_big)
    card.add_content(big_row)

    # 自主神经表（RMSSD/SDNN/LF-HF/pNN50/HF/HR）
    head = QHBoxLayout()
    head.setContentsMargins(8, 0, 8, 0)
    for txt, w in [('指标', 60), ('当前值', 72), ('参考范围', 70), ('状态', 64)]:
        lbl = QLabel(txt)
        lbl.setStyleSheet(f'color:{C_DIM}; font-size:10px; letter-spacing:1px; background:transparent;')
        lbl.setAlignment(Qt.AlignLeft if txt == '指标' else Qt.AlignCenter)
        lbl.setMinimumWidth(w)
        head.addWidget(lbl)
    card.add_content(head)

    self.row_rmssd = MetricRow('RMSSD', '≥50 ms')
    self.row_sdnn  = MetricRow('SDNN',  '≥100 ms')
    self.row_lfhf  = MetricRow('LF/HF', '1.0–2.5')
    self.row_pnn50 = MetricRow('pNN50', '>15%')
    self.row_hf    = MetricRow('HF',    '≥600 ms²')
    self.row_hrmean = MetricRow('HR均', '60–75 bpm')
    for r in (self.row_rmssd, self.row_sdnn, self.row_lfhf,
              self.row_pnn50, self.row_hf, self.row_hrmean):
        card.add_content(r)

    # HRV 趋势小图
    self.trend_plot = pg.PlotWidget()
    self.trend_plot.setBackground(C_PANEL)
    self.trend_plot.setMinimumHeight(80)
    self.trend_plot.setMaximumHeight(110)
    _theme.style_pg_plot(self.trend_plot.getPlotItem(),
                          'HRV 趋势 · HRV TREND', C_HEART)
    self.trend_plot.getPlotItem().disableAutoRange()
    self.trend_plot.setXRange(0, 600, padding=0)
    self.trend_plot.setYRange(0, 100, padding=0)
    self.trend_curve = self.trend_plot.plot(
        pen=pg.mkPen(C_HEART, width=1.4),
        fillLevel=0,
        brush=pg.mkBrush(QColor(255, 82, 119, 50)))
    ref_line = pg.InfiniteLine(pos=50, angle=0,
        pen=pg.mkPen(C_OK, style=Qt.DashLine, width=1))
    self.trend_plot.addItem(ref_line)
    card.add_content(self.trend_plot)

    # 情绪风险指数 · MOOD RISK INDEX —— 由 RiskBus 综合分驱动（NN 输出）
    from PyQt5.QtCore import Qt as _Qt
    from theme import C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT, risk_tier
    mood_card = QFrame()
    mood_card.setStyleSheet(
        f'QFrame {{ background:{C_BG_DEEP}; border:1px solid {C_BORDER};'
        f' border-radius:8px; }}')
    mlay = QVBoxLayout(mood_card)
    mlay.setContentsMargins(10, 8, 10, 8); mlay.setSpacing(2)
    mhead = QLabel('情绪风险指数 · MOOD RISK INDEX')
    mhead.setStyleSheet(
        f'color:{C_DIM}; font-size:10px; letter-spacing:1.4px; background:transparent;')
    self.mood_risk_value = QLabel('0.00')
    self.mood_risk_value.setAlignment(_Qt.AlignCenter)
    self.mood_risk_value.setStyleSheet(
        f'color:{C_RISK_LOW}; font-size:28px; font-weight:bold;'
        f'background:transparent; font-family:Consolas;')
    self.mood_risk_tag = QLabel('低 · LOW')
    self.mood_risk_tag.setAlignment(_Qt.AlignCenter)
    self.mood_risk_tag.setStyleSheet(
        f'color:{C_RISK_LOW}; font-size:10px; letter-spacing:2px; background:transparent;')
    mood_bar = QLabel()
    mood_bar.setFixedHeight(4)
    mood_bar.setStyleSheet(
        f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
        f'stop:0 {C_RISK_LOW}, stop:0.33 {C_RISK_MED}, '
        f'stop:0.66 {C_RISK_HIGH}, stop:1 {C_RISK_CRIT});'
        f'border-radius:2px;')
    mlay.addWidget(mhead)
    mlay.addWidget(self.mood_risk_value)
    mlay.addWidget(self.mood_risk_tag)
    mlay.addSpacing(2)
    mlay.addWidget(mood_bar)
    card.add_content(mood_card)

    def _mood_on_nn(nn_out, _mv=self.mood_risk_value, _mt=self.mood_risk_tag):
        try:
            r = float(nn_out.get('risk_score', 0.0))
            tag, c = risk_tier(r)
            _mv.setText(f'{r:.2f}')
            _mv.setStyleSheet(
                f'color:{c}; font-size:28px; font-weight:bold;'
                f'background:transparent; font-family:Consolas;')
            _mt.setText(tag)
            _mt.setStyleSheet(
                f'color:{c}; font-size:10px; letter-spacing:2px; background:transparent;')
        except Exception:
            pass
    try:
        from risk_bus import RiskBus
        RiskBus.instance().nn_result_changed.connect(_mood_on_nn)
    except Exception:
        pass

    # NN 模态贡献仪表（HRV 模态实时贡献，由总线 modal_w[1] 更新）
    self.heart_contrib_gauge = ModalContribGauge('hrv', C_HEART)
    card.add_content(self.heart_contrib_gauge)

    self.heart_card = card
    return card


def build_brain_card(self, bci_panel_widget):
    """脑电卡 - 保持原始完整视图，保证整块内容可见。"""
    card = ModuleCard('脑电感知 · NEURAL', 'TGAM · Att/Med · 8 Bands', C_BRAIN)
    card.status_pill.set_state('sim', '设备: 模拟')

    bci_panel_widget.setStyleSheet('background:transparent;')
    bci_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    sa = QScrollArea()
    sa.setWidget(bci_panel_widget)
    sa.setWidgetResizable(True)
    # 🔧 消抖关键 7: 强制隐藏横向滚动条,避免内容微超界时反复出没
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    sa.setStyleSheet(
        f'QScrollArea {{ background:{C_PANEL}; border:none; }}')
    sa.setMinimumWidth(0)
    sa.setMinimumHeight(480)
    card.add_content(sa)

    # NN 模态贡献仪表（脑电模态实时贡献，由总线 modal_w[2] 更新）
    self.brain_contrib_gauge = ModalContribGauge('eeg', C_BRAIN)
    card.add_content(self.brain_contrib_gauge)

    self.brain_card = card
    return card
