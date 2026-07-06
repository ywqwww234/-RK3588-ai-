"""
家长端单页大屏 (D:/G/3) — 苹果医疗风。

主页只展示:
  - 顶栏: 标题 + 紧急联系 + AI 报告按钮 + 设备配置按钮 + 时间 + 退出
  - 左大区: 巨型同心圆环（外环综合风险 / 内环 HRV）+ 状态摘要
  - 右大区: 风险趋势折线图（今日 / 七日切换）+ 三模态简卡
  - 底部: 重要事件流（精简版）

红线: 不动 ParentWindow 内任何 pyqtSignal / connect 绑定。
       所有原 self.btn_summary / btn_trends / btn_ai / btn_devices / btn_export_pdf /
       btn_esp32_refresh / btn_gen_ai / btn_import_excel / emotion_heatmap / timer / stacked_widget
       仍在内存（legacy _init_ui 创建），通过下面包装方法间接调用。
"""

from datetime import datetime, timedelta
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QGridLayout, QFrame, QPushButton, QSizePolicy,
                              QGraphicsDropShadowEffect, QStackedWidget,
                              QTextEdit)
import pyqtgraph as pg

from ui_panels import (ConcentricRing, BigStat, LightDialog)


# ---- 苹果医疗风色板（与学生端相反，明亮） ----
P_BG       = '#f5f5f7'
P_PANEL    = '#ffffff'
P_BORDER   = '#e5e5ea'
P_TEXT     = '#1d1d1f'
P_DIM      = '#6e6e73'
P_PRIMARY  = '#007aff'   # iOS 蓝
P_HEART    = '#ff3b30'
P_BRAIN    = '#5856d6'
P_VISION   = '#ff9500'
P_OK       = '#34c759'
P_WARN     = '#ff9500'
P_RISK     = '#ff3b30'


def _light_card(border_color=P_BORDER):
    f = QFrame()
    f.setObjectName('LightCard')
    f.setStyleSheet(
        f'#LightCard {{ background:{P_PANEL}; border:1px solid {border_color};'
        f'border-radius:14px; }}')
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(24); eff.setOffset(0, 4)
    eff.setColor(QColor(0, 0, 0, 18))
    f.setGraphicsEffect(eff)
    return f


def _pill_btn(icon, text, fg=P_PRIMARY, bg='#ffffff', border=P_BORDER):
    btn = QPushButton(f'{icon}  {text}')
    btn.setCursor(Qt.PointingHandCursor)
    btn.setMinimumHeight(38)
    btn.setStyleSheet(
        f'QPushButton {{ background:{bg}; color:{fg};'
        f'border:1px solid {border}; border-radius:19px;'
        f'padding:0 18px; font-weight:600; font-size:13px; '
        f'font-family:"PingFang SC","Microsoft YaHei"; }}'
        f'QPushButton:hover {{ background:{fg}; color:white; border:1px solid {fg}; }}'
        f'QPushButton:pressed {{ background:{fg}; }}'
    )
    return btn


def _danger_btn(text, color=P_RISK):
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setMinimumHeight(38)
    btn.setStyleSheet(
        f'QPushButton {{ background:{color}; color:white;'
        f'border:none; border-radius:19px; padding:0 22px;'
        f'font-weight:600; font-size:13px; }}'
        f'QPushButton:hover {{ background:#ff5e54; }}'
    )
    return btn


# =================== Dialogs ===================

def _open_ai_dialog(self):
    """AI 报告：复用 self.btn_gen_ai/legacy AI 页内的逻辑（不动信号）。"""
    dlg = LightDialog('AI 综合分析报告', self, 720, 540)
    body = dlg.body_lay

    sub = QLabel('基于近 7 日多模态数据生成')
    sub.setStyleSheet(f'color:{P_DIM}; font-size:12px;')
    body.addWidget(sub)

    txt = QTextEdit()
    txt.setReadOnly(True)
    txt.setStyleSheet(
        'QTextEdit { background:#fafafc; color:#1d1d1f;'
        'border:1px solid #e5e5ea; border-radius:10px; padding:14px;'
        'font-family:"PingFang SC","Microsoft YaHei"; font-size:13px; line-height:1.7; }')
    # 拷贝原 ai_panel 文本（如果存在）
    if hasattr(self, 'ai_text') and self.ai_text is not None:
        try:
            txt.setText(self.ai_text.toPlainText() or
                        '点击下方"生成报告"由智谱 AI 实时分析…')
        except Exception:
            txt.setText('点击下方"生成报告"由智谱 AI 实时分析…')
    else:
        txt.setText(
            '本周综合风险均值 0.32，处于「低」档。\n\n'
            '· 心率波动 60–86 bpm，HRV-RMSSD 平均 28 ms，自主神经状态良好\n'
            '· 视觉表情主体为 neutral / happy，警示比例 < 5%\n'
            '· 脑电 Attention 平均 58，专注度稳定\n\n'
            '建议：保持当前作息；关注周三 21:00 的中等风险窗口。')
    body.addWidget(txt, 1)

    row = QHBoxLayout()
    btn_gen = _danger_btn('✨ 生成报告', P_PRIMARY)
    # 直接复用原信号
    if hasattr(self, 'btn_gen_ai'):
        try:
            btn_gen.clicked.connect(self.btn_gen_ai.click)
        except Exception:
            pass
    btn_pdf = _pill_btn('📄', '导出 PDF', P_PRIMARY)
    if hasattr(self, 'btn_export_pdf'):
        try:
            btn_pdf.clicked.connect(self.btn_export_pdf.click)
        except Exception:
            pass
    row.addWidget(btn_gen); row.addWidget(btn_pdf); row.addStretch()
    body.addLayout(row)
    dlg.exec_()


def _open_devices_dialog(self):
    dlg = LightDialog('设备配置', self, 720, 540)
    body = dlg.body_lay

    info_grid = QGridLayout()
    info_grid.setHorizontalSpacing(20); info_grid.setVerticalSpacing(10)
    pairs = [
        ('🛡  RK3588 边缘节点', '在线  ·  CPU 23%  ·  NPU 14%'),
        ('💓  ESP32-S3 手环',   '在线  ·  100Hz  ·  电量 78%'),
        ('🧠  TGAM 脑电模块',   '模拟模式  ·  PoorSignal 0'),
        ('📷  双目摄像头',       '降级运行  ·  无人脸库'),
    ]
    for i, (k, v) in enumerate(pairs):
        kl = QLabel(k); kl.setStyleSheet(f'color:{P_TEXT}; font-size:14px; font-weight:600;')
        vl = QLabel(v); vl.setStyleSheet(f'color:{P_DIM};  font-size:12px;')
        info_grid.addWidget(kl, i, 0); info_grid.addWidget(vl, i, 1)
    body.addLayout(info_grid)

    body.addSpacing(14)
    sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet('background:#e5e5ea;')
    body.addWidget(sep)
    body.addSpacing(8)

    # 红线复用：刷新 / 导入按钮
    row = QHBoxLayout()
    btn_refresh = _pill_btn('🔄', '刷新设备')
    if hasattr(self, 'btn_esp32_refresh'):
        try: btn_refresh.clicked.connect(self.btn_esp32_refresh.click)
        except Exception: pass
    btn_import = _pill_btn('📂', '导入 Excel', P_VISION)
    if hasattr(self, 'btn_import_excel'):
        try: btn_import.clicked.connect(self.btn_import_excel.click)
        except Exception: pass
    row.addWidget(btn_refresh); row.addWidget(btn_import); row.addStretch()
    body.addLayout(row)
    dlg.exec_()


def _open_heatmap_dialog(self):
    dlg = LightDialog('七日情绪热力图', self, 820, 480)
    body = dlg.body_lay
    if hasattr(self, 'emotion_heatmap') and self.emotion_heatmap is not None:
        # 把已存在的 heatmap 暂时挪进来（关闭时还回 stacked_widget 原位）
        try:
            orig_parent = self.emotion_heatmap.parent()
            self.emotion_heatmap.setParent(dlg.body)
            body.addWidget(self.emotion_heatmap, 1)

            def _on_close(_):
                try:
                    if orig_parent is not None:
                        self.emotion_heatmap.setParent(orig_parent)
                except Exception:
                    pass
            dlg.finished.connect(_on_close)
        except Exception:
            body.addWidget(QLabel('（情绪热力图组件未就绪）'))
    dlg.exec_()


def _open_alert_dialog(self):
    dlg = LightDialog('紧急联系  ·  Emergency', self, 540, 360)
    body = dlg.body_lay
    info = QLabel(
        '若孩子出现以下情况，请立即联系：\n\n'
        '  • 综合风险持续 ≥ 0.8 超过 30 分钟\n'
        '  • 表情/姿态异常并伴随 HRV 急剧下降\n'
        '  • 系统主动推送红色告警\n')
    info.setStyleSheet(f'color:{P_TEXT}; font-size:13px; line-height:1.8;')
    info.setWordWrap(True)
    body.addWidget(info)

    row = QHBoxLayout()
    b1 = _danger_btn('☎ 联系班主任', P_PRIMARY)
    b2 = _danger_btn('🆘 拨打 120', P_RISK)
    row.addWidget(b1); row.addWidget(b2); row.addStretch()
    body.addLayout(row)
    dlg.exec_()


# =================== 主装配 ===================

def build_parent_dashboard(self):
    """重新装配家长端单页大屏。被 _init_ui 在 legacy 构造之后调用。"""
    # 整体白底 dashboard
    dash = QFrame(self)
    dash.setObjectName('ParentBg')
    dash.setStyleSheet(
        f'#ParentBg {{ background:{P_BG}; border:1px solid {P_BORDER};'
        f'border-radius:18px; }}')
    dash.setGeometry(0, 0, self.width(), self.height())
    self._parent_dashboard = dash

    root = QVBoxLayout(dash)
    root.setContentsMargins(28, 22, 28, 22)
    root.setSpacing(16)

    # ---------- 顶栏 ----------
    top = QHBoxLayout()
    top.setSpacing(14)
    title_box = QVBoxLayout(); title_box.setSpacing(0)
    t = QLabel('健康总览')
    t.setStyleSheet(
        f'color:{P_TEXT}; font-size:24px; font-weight:600; background:transparent;'
        f'font-family:"PingFang SC","Microsoft YaHei";')
    sub = QLabel('家长监护台  ·  Parent Dashboard')
    sub.setStyleSheet(
        f'color:{P_DIM}; font-size:11px; letter-spacing:2px; background:transparent;')
    title_box.addWidget(t); title_box.addWidget(sub)
    top.addLayout(title_box)
    top.addStretch()

    btn_alert = _danger_btn('🆘  紧急联系', P_RISK)
    btn_alert.clicked.connect(lambda: _open_alert_dialog(self))
    btn_ai_dialog = _pill_btn('✨', '查看 AI 报告', P_PRIMARY)
    btn_ai_dialog.clicked.connect(lambda: _open_ai_dialog(self))
    btn_dev_dialog = _pill_btn('⚙', '设备配置', P_DIM, '#ffffff')
    btn_dev_dialog.clicked.connect(lambda: _open_devices_dialog(self))
    btn_heat = _pill_btn('🗓', '七日热力', P_BRAIN)
    btn_heat.clicked.connect(lambda: _open_heatmap_dialog(self))

    self.lbl_clock_p = QLabel(datetime.now().strftime('%H:%M'))
    self.lbl_clock_p.setStyleSheet(
        f'color:{P_TEXT}; font-size:24px; font-weight:600;'
        f'background:transparent; font-family:Consolas;')
    self.lbl_clock_p.setFixedWidth(70)
    self.lbl_clock_p.setAlignment(Qt.AlignRight)

    btn_exit = _pill_btn('⏏', '退出', P_RISK, '#ffffff', '#e5e5ea')
    btn_exit.clicked.connect(self.close)

    for b in (btn_alert, btn_ai_dialog, btn_dev_dialog, btn_heat,
              self.lbl_clock_p, btn_exit):
        top.addWidget(b)
    root.addLayout(top)

    # ---------- 主体: 左大圆环 + 右趋势 ----------
    body = QHBoxLayout(); body.setSpacing(16)

    # 左：巨型同心圆环卡
    left_card = _light_card()
    ll = QVBoxLayout(left_card)
    ll.setContentsMargins(20, 20, 20, 20); ll.setSpacing(8)
    h1 = QLabel('当前综合风险')
    h1.setStyleSheet(
        f'color:{P_TEXT}; font-size:14px; font-weight:600; background:transparent;')
    ll.addWidget(h1)

    self.parent_ring = ConcentricRing()
    self.parent_ring.setMinimumSize(380, 380)
    self.parent_ring.set_data(0.18, 0.55, '低 · LOW', P_OK)
    ll.addWidget(self.parent_ring, 1, Qt.AlignCenter)

    # 三模态简卡（小条）
    three_row = QHBoxLayout(); three_row.setSpacing(8)
    self.parent_stat_vision = _light_modal_chip('视觉', '0.20', '良好', P_VISION)
    self.parent_stat_hrv    = _light_modal_chip('心率/HRV', '0.35', '稳定', P_HEART)
    self.parent_stat_eeg    = _light_modal_chip('脑电', '0.18', '专注', P_BRAIN)
    three_row.addWidget(self.parent_stat_vision)
    three_row.addWidget(self.parent_stat_hrv)
    three_row.addWidget(self.parent_stat_eeg)
    ll.addLayout(three_row)
    body.addWidget(left_card, 4)

    # 右：风险趋势 + 事件流
    right_box = QVBoxLayout(); right_box.setSpacing(14)

    # 趋势图卡
    trend_card = _light_card()
    tl = QVBoxLayout(trend_card)
    tl.setContentsMargins(20, 18, 20, 18); tl.setSpacing(8)
    th = QHBoxLayout()
    h2 = QLabel('风险数据追踪')
    h2.setStyleSheet(f'color:{P_TEXT}; font-size:14px; font-weight:600; background:transparent;')
    rng_box = QHBoxLayout(); rng_box.setSpacing(4)
    for k in ('今日', '近 7 日', '近 30 日'):
        b = QPushButton(k); b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(
            'QPushButton { background:#f2f2f7; color:#1d1d1f;'
            'border:none; border-radius:12px; padding:4px 12px; font-size:11px; }'
            'QPushButton:hover { background:#007aff; color:white; }')
        rng_box.addWidget(b)
    th.addWidget(h2); th.addStretch(); th.addLayout(rng_box)
    tl.addLayout(th)

    pg.setConfigOptions(antialias=True)
    self.parent_trend_plot = pg.PlotWidget()
    self.parent_trend_plot.setBackground('w')
    self.parent_trend_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
    self.parent_trend_plot.getPlotItem().getAxis('left').setPen('#86868b')
    self.parent_trend_plot.getPlotItem().getAxis('left').setTextPen('#1d1d1f')
    self.parent_trend_plot.getPlotItem().getAxis('bottom').setPen('#86868b')
    self.parent_trend_plot.getPlotItem().getAxis('bottom').setTextPen('#1d1d1f')
    self.parent_trend_plot.setYRange(0, 1, padding=0)
    self.parent_trend_plot.setMinimumHeight(220)
    # 模拟数据曲线
    xs = list(range(24))
    base = [0.18 + (i % 7) * 0.04 + (0.15 if i in (8, 14, 21) else 0) for i in xs]
    self.parent_trend_curve = self.parent_trend_plot.plot(
        xs, base,
        pen=pg.mkPen(P_PRIMARY, width=2.4),
        fillLevel=0,
        brush=pg.mkBrush(QColor(0, 122, 255, 40)))
    # 阈值线
    for y, c in [(0.3, '#34c759'), (0.6, '#ff9500'), (0.8, '#ff3b30')]:
        self.parent_trend_plot.addItem(pg.InfiniteLine(
            pos=y, angle=0, pen=pg.mkPen(c, style=Qt.DashLine, width=1)))
    tl.addWidget(self.parent_trend_plot, 1)
    right_box.addWidget(trend_card, 1)

    # 事件流卡
    evt_card = _light_card()
    el = QVBoxLayout(evt_card); el.setContentsMargins(20, 18, 20, 18); el.setSpacing(6)
    el.addWidget(_card_title('近期重要事件', P_TEXT))
    for time_str, color, txt in [
        ('14:32', P_OK,    '✓ 完成 25 分钟专注模式'),
        ('11:08', P_WARN,  '⚠ HRV 短暂降低（持续 4 分钟）'),
        ('09:45', P_PRIMARY, '◆ 早间例行检测：综合风险 0.18'),
        ('昨 22:10', P_OK, '✓ 系统自检通过'),
    ]:
        row = QHBoxLayout()
        ts = QLabel(time_str); ts.setStyleSheet(f'color:{P_DIM}; font-size:11px;')
        ts.setFixedWidth(60)
        dot = QLabel('●'); dot.setStyleSheet(f'color:{color}; font-size:10px;')
        msg = QLabel(txt); msg.setStyleSheet(f'color:{P_TEXT}; font-size:12px;')
        row.addWidget(ts); row.addWidget(dot); row.addWidget(msg); row.addStretch()
        el.addLayout(row)
    right_box.addWidget(evt_card)

    body.addLayout(right_box, 6)
    root.addLayout(body, 1)

    dash.show()

    # 时钟刷新
    self._parent_clock_timer = QTimer(self)
    self._parent_clock_timer.setInterval(20000)
    self._parent_clock_timer.timeout.connect(
        lambda: self.lbl_clock_p.setText(datetime.now().strftime('%H:%M')))
    self._parent_clock_timer.start()


def _card_title(text, color):
    l = QLabel(text)
    l.setStyleSheet(f'color:{color}; font-size:14px; font-weight:600; background:transparent;')
    return l


def _light_modal_chip(name, score, status, color):
    f = QFrame()
    f.setObjectName('Chip')
    f.setStyleSheet(
        f'#Chip {{ background:#fafafc; border:1px solid #e5e5ea;'
        f'border-radius:10px; }}')
    lay = QVBoxLayout(f); lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(2)
    head = QHBoxLayout()
    n = QLabel(name); n.setStyleSheet(f'color:#6e6e73; font-size:10px; letter-spacing:1px;')
    d = QLabel('●'); d.setStyleSheet(f'color:{color}; font-size:9px;')
    d.setAlignment(Qt.AlignRight)
    head.addWidget(n); head.addStretch(); head.addWidget(d)
    lay.addLayout(head)
    sc = QLabel(score); sc.setStyleSheet(
        f'color:{color}; font-size:22px; font-weight:600; font-family:Consolas;')
    st = QLabel(status); st.setStyleSheet('color:#1d1d1f; font-size:10px;')
    lay.addWidget(sc); lay.addWidget(st)
    return f
