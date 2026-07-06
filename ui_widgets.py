"""
工业级控件库 (D:/G/3 大屏版)

- StatusPill         状态胶囊
- BigValueCard       大数字卡（标题/数值/单位）
- MetricRow          指标行（名/值/参考范围/状态）
- RiskBarLabel       风险渐变条（绿→黄→橙→红，四档）
- ModuleCard         三模态主区容器（带左侧色条 + 标题 + 模态色边框）
- RiskHeroCard       顶部风险中央显示（标签 + 大数 + 渐变条 + 四档色态）
- ContribStackBar    模态贡献堆叠条（视觉/HRV/EEG）
"""

from collections import deque

from PyQt5.QtCore import Qt, QSize, QTimer, QRectF, QPointF, QEvent, QEvent
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QLinearGradient, QFont, QRadialGradient
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy

from theme import (C_ACCENT, C_BORDER, C_DIM, C_OK, C_PANEL, C_RISK,
                   C_TEXT, C_WARN, C_VISION, C_HEART, C_BRAIN,
                   C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT,
                   C_BG, C_BG_DEEP, C_PANEL_HI, risk_tier,
                   SP_XS, SP_SM, SP_MD, SP_LG, SP_XL,
                   FS_TINY, FS_XS, FS_SM, FS_MD, FS_LG, FS_XL, FS_XXL, FS_HERO,
                   R_SM, R_MD, R_LG, BORDER_W)


class StatusPill(QLabel):
    PALETTE = {
        'live':    (C_OK,   '#0d2818'),
        'sim':     (C_WARN, '#2e1f05'),
        'offline': (C_RISK, '#2a0a0e'),
        'warn':    (C_WARN, '#2e1f05'),
        'info':    (C_ACCENT, '#06243a'),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_state('offline', '未连接')

    def set_state(self, state, text):
        fg, bg = self.PALETTE.get(state, self.PALETTE['offline'])
        self.setText(f'  ●  {text}  ')
        self.setStyleSheet(
            f'background:{bg}; color:{fg};'
            f'border:1.4px solid {fg}; border-radius:13px;'
            f'padding:{max(3, SP_XS)}px {max(8, SP_SM)}px; font-weight:bold; font-size:{FS_SM}px;'
            f'letter-spacing:1.2px;'
        )


class BigValueCard(QWidget):
    def __init__(self, title, unit='', color=C_ACCENT, parent=None):
        super().__init__(parent)
        self.color = color
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SP_MD, SP_MD, SP_MD, SP_MD)
        lay.setSpacing(2)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.4px; background:transparent;')
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.value_lbl = QLabel('--')
        self.value_lbl.setStyleSheet(
            f'color:{color}; font-size:{FS_XXL}px; font-weight:bold; background:transparent;')
        self.value_lbl.setAlignment(Qt.AlignCenter)
        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        self.unit_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.title_lbl)
        lay.addWidget(self.value_lbl)
        lay.addWidget(self.unit_lbl)
        self.setStyleSheet(
            f'BigValueCard {{ background:{C_PANEL}; '
            f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_MD}px; }}')

    def set_value(self, text, color=None):
        self.value_lbl.setText(str(text))
        if color is not None:
            self.color = color
            self.value_lbl.setStyleSheet(
                f'color:{color}; font-size:{FS_XXL}px; font-weight:bold; background:transparent;')


class MetricRow(QWidget):
    def __init__(self, name, ref_text='', parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(8)
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet(
            f'color:{C_TEXT}; font-weight:bold; background:transparent; font-size:12px;')
        self.name_lbl.setMinimumWidth(60)
        self.value_lbl = QLabel('--')
        self.value_lbl.setStyleSheet(
            f'color:{C_ACCENT}; font-size:13px; font-weight:bold; background:transparent;')
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_lbl.setMinimumWidth(72)
        self.ref_lbl = QLabel(ref_text)
        self.ref_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; background:transparent;')
        self.ref_lbl.setMinimumWidth(70)
        self.ref_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_lbl = QLabel('--')
        self.status_lbl.setStyleSheet(
            f'color:{C_DIM}; font-weight:bold; background:transparent;')
        self.status_lbl.setMinimumWidth(64)
        self.status_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.value_lbl)
        lay.addWidget(self.ref_lbl)
        lay.addWidget(self.status_lbl)

    def update_value(self, value_text, status_text='--', color=None):
        c = color or C_DIM
        self.value_lbl.setText(str(value_text))
        self.status_lbl.setText(status_text)
        self.status_lbl.setStyleSheet(
            f'color:{c}; font-weight:bold; background:transparent;')


class RiskBarLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(5)
        self.setStyleSheet(
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_RISK_LOW}, stop:0.33 {C_RISK_MED}, '
            f'stop:0.66 {C_RISK_HIGH}, stop:1 {C_RISK_CRIT});'
            f'border-radius:2px;')


# ============== 大屏专用控件 ==============

class ModuleCard(QFrame):
    """三模态主区容器：左侧 4px 模态色条 + 标题 + 发光效果。"""

    def __init__(self, title, en_title, brand_color, parent=None):
        super().__init__(parent)
        self.brand = brand_color
        self.setObjectName('ModuleCard')
        # 🎨 工业级:左侧 3px 模态色高亮条 + 微妙发光
        self.setStyleSheet(
            f'#ModuleCard {{ background:{C_PANEL}; '
            f'border:{BORDER_W}px solid {C_BORDER}; border-left:3px solid {brand_color}; '
            f'border-radius:{R_MD}px; }}')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_MD)
        outer.setSpacing(SP_SM)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SP_SM)
        dot = QLabel('●')
        dot.setStyleSheet(f'color:{brand_color}; font-size:{FS_MD}px; background:transparent;')
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f'color:{brand_color}; font-weight:bold; font-size:{FS_MD}px;'
            f'letter-spacing:2px; background:transparent;')
        en_lbl = QLabel(en_title)
        en_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:2px; background:transparent;')
        en_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_pill = StatusPill()
        self.status_pill.set_state('sim', 'SIM')

        head.addWidget(dot)
        head.addWidget(title_lbl)
        head.addStretch()
        head.addWidget(en_lbl)
        head.addSpacing(SP_SM)
        head.addWidget(self.status_pill)
        outer.addLayout(head)

        self._content = QWidget()
        self._content.setStyleSheet('background:transparent;')
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(SP_SM)
        outer.addWidget(self._content, 1)

    def add_content(self, widget_or_layout):
        if isinstance(widget_or_layout, QWidget):
            self.content_layout.addWidget(widget_or_layout)
        else:
            self.content_layout.addLayout(widget_or_layout)


class RiskHeroCard(QFrame):
    """顶栏中央: 当前综合风险大显示（四档色态）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('RiskHero')
        self.setStyleSheet(
            f'#RiskHero {{ background:{C_BG_DEEP}; '
            f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_MD}px; }}')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SP_XL, SP_SM, SP_XL, SP_SM)
        lay.setSpacing(SP_XL)

        left = QVBoxLayout()
        left.setSpacing(0)
        self.head = QLabel('综合风险评分 · TOTAL RISK')
        self.head.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:2.5px; background:transparent;')
        self.tag = QLabel('低 · LOW')
        self.tag.setStyleSheet(
            f'color:{C_RISK_LOW}; font-size:{FS_MD}px; font-weight:bold;'
            f'letter-spacing:3px; background:transparent;')
        left.addWidget(self.head)
        left.addWidget(self.tag)
        lay.addLayout(left)

        self.value = QLabel('0.00')
        self.value.setStyleSheet(
            f'color:{C_RISK_LOW}; font-size:{FS_HERO}px; font-weight:bold;'
            f'background:transparent; font-family:"Consolas","Courier New";')
        self.value.setAlignment(Qt.AlignCenter)
        self.value.setFixedWidth(180)
        lay.addWidget(self.value)

        right = QVBoxLayout()
        right.setSpacing(SP_XS)
        self.bar = RiskBarLabel()
        self.scale = QLabel('0.0          0.3          0.6          0.8          1.0')
        self.scale.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_TINY}px; letter-spacing:1px; background:transparent;')
        self.scale.setAlignment(Qt.AlignCenter)
        right.addStretch()
        right.addWidget(self.bar)
        right.addWidget(self.scale)
        right.addStretch()
        right_wrap = QWidget()
        right_wrap.setStyleSheet('background:transparent;')
        right_wrap.setLayout(right)
        right_wrap.setMinimumWidth(180)
        lay.addWidget(right_wrap, 1)

    def set_score(self, score):
        tag, color = risk_tier(score)
        self.value.setText(f'{score:.2f}')
        self.value.setStyleSheet(
            f'color:{color}; font-size:{FS_HERO}px; font-weight:bold;'
            f'background:transparent; font-family:"Consolas","Courier New";')
        self.tag.setText(tag)
        self.tag.setStyleSheet(
            f'color:{color}; font-size:{FS_MD}px; font-weight:bold;'
            f'letter-spacing:3px; background:transparent;')


class ModalContribGauge(QFrame):
    """单模态 NN 贡献仪表：横向 0~100% 进度条 + 主因短句 + NN 实时数值。

    用法:
        from risk_bus import RiskBus
        g = ModalContribGauge('vision', '视觉', C_VISION)   # modality in (vision/hrv/eeg)
        layout.addWidget(g)
        # 总线信号会自动订阅；如果总线还未启动，控件保持静态默认值

    设计:
        - 自动订阅 RiskBus.nn_result_changed
        - 根据当前 modal_w 中本模态贡献占比着色：>0.4 时变 brand 高亮，否则灰
        - 右侧短句来自 RiskBus.get_modal_factor(modality)
    """

    _MODAL_IDX = {'vision': 0, 'hrv': 1, 'eeg': 2}
    _MODAL_ZH = {'vision': '视觉', 'hrv': 'HRV', 'eeg': '脑电'}

    def __init__(self, modality, color, parent=None):
        super().__init__(parent)
        self.modality = modality
        self.color = color
        self._w = 0.33
        self._factor = '等待 NN 推理…'
        self.setObjectName('ContribGauge')
        self.setStyleSheet(
            f'#ContribGauge {{ background:{C_BG_DEEP}; border:1px solid {C_BORDER};'
            f' border-radius:6px; }}')
        self.setMinimumHeight(46)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        self._title = QLabel(f'{self._MODAL_ZH.get(modality, modality)} 贡献')
        self._title.setStyleSheet(
            f'color:{color}; font-size:10px; font-weight:bold;'
            f'letter-spacing:1.4px; background:transparent;')
        self._sub = QLabel('NN MODAL')
        self._sub.setStyleSheet(
            f'color:{C_DIM}; font-size:8px; letter-spacing:1.3px; background:transparent;')
        title_box.addWidget(self._title)
        title_box.addWidget(self._sub)
        wrap = QWidget(); wrap.setStyleSheet('background:transparent;')
        wrap.setLayout(title_box); wrap.setFixedWidth(56)
        lay.addWidget(wrap)

        self._pct = QLabel('33%')
        self._pct.setStyleSheet(
            f'color:{color}; font-size:18px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self._pct.setFixedWidth(56)
        self._pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._pct)

        # 自绘渐变条（不用 QProgressBar 以便控制色态）
        self._bar = _GaugeBar(color)
        self._bar.setMinimumWidth(60)
        self._bar.setMaximumWidth(120)
        lay.addWidget(self._bar, 0)

        # 主因短句：固定宽度 + 文字截断，避免每秒文本变化触发整列 reflow
        self._factor_lbl = QLabel(self._factor)
        self._factor_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; background:transparent;'
            f'font-style:italic;')
        self._factor_lbl.setFixedWidth(170)
        self._factor_lbl.setWordWrap(False)
        self._factor_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        lay.addWidget(self._factor_lbl, 0)

        # 整个 gauge 横向不扩展，避免影响所在列宽度
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # 订阅总线（按需，total bus 已是单例）
        try:
            from risk_bus import RiskBus
            RiskBus.instance().nn_result_changed.connect(self._on_nn)
        except Exception:
            pass

    def _on_nn(self, nn_out):
        idx = self._MODAL_IDX.get(self.modality, 0)
        try:
            mw = nn_out.get('modal_w') or (0.33, 0.33, 0.34)
            self._w = float(mw[idx])
        except Exception:
            return
        self._pct.setText(f'{self._w * 100:.0f}%')
        self._bar.set_progress(self._w, dominant=self._w >= 0.40)
        try:
            from risk_bus import RiskBus
            txt = RiskBus.instance().get_modal_factor(self.modality)
            if txt and txt != '—':
                # 用 elidedText 截断，避免文本变化撑大整个 gauge → 三大卡抖动
                fm = self._factor_lbl.fontMetrics()
                elided = fm.elidedText(txt, Qt.ElideRight, self._factor_lbl.width() - 4)
                self._factor_lbl.setText(elided)
        except Exception:
            pass


class _GaugeBar(QWidget):
    """ModalContribGauge 内部用的迷你渐变条。"""

    def __init__(self, color):
        super().__init__()
        self._color = color
        self._p = 0.33
        self._dominant = False
        self.setMinimumHeight(10)

    def set_progress(self, p, dominant=False):
        self._p = max(0.0, min(1.0, float(p)))
        self._dominant = bool(dominant)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width(); H = self.height()
        bar_h = max(6, H - 4)
        y = (H - bar_h) // 2
        # 底
        p.setBrush(QBrush(QColor(C_PANEL)))
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawRoundedRect(0, y, W - 1, bar_h, 3, 3)
        # 前
        seg = int((W - 2) * self._p)
        if seg > 0:
            grad = QLinearGradient(0, 0, W, 0)
            c = QColor(self._color)
            if not self._dominant:
                c.setAlphaF(0.55)
            grad.setColorAt(0.0, c)
            c2 = QColor(self._color)
            grad.setColorAt(1.0, c2)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(1, y + 1, seg, bar_h - 2, 3, 3)


class ContribStackBar(QWidget):
    """三模态贡献堆叠条（横向 100%）：视觉/HRV/EEG。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._w_v = 0.35
        self._w_h = 0.40
        self._w_e = 0.25
        self.setMinimumHeight(30)

    def set_weights(self, w_vision, w_hrv, w_eeg):
        s = max(1e-6, w_vision + w_hrv + w_eeg)
        self._w_v = w_vision / s
        self._w_h = w_hrv / s
        self._w_e = w_eeg / s
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width(); H = self.height()
        bar_h = 16
        y = (H - bar_h) // 2
        x = 0
        for w, c, name in [(self._w_v, C_VISION, '视觉'),
                           (self._w_h, C_HEART,  'HRV'),
                           (self._w_e, C_BRAIN,  '脑电')]:
            seg = int(W * w)
            p.setBrush(QBrush(QColor(c)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(x, y, max(1, seg), bar_h, 3, 3)
            if seg > 50:
                p.setPen(QPen(QColor(C_BG_DEEP)))
                f = QFont('Microsoft YaHei', 8)
                f.setBold(True)
                p.setFont(f)
                p.drawText(x + 6, y + bar_h - 4, f'{name} {w*100:.0f}%')
            x += seg


class ZoomOverlay(QFrame):
    """统一单实例全屏放大覆盖层：双击模块进入单模块全屏，再双击/返回恢复。"""

    _active = None

    @staticmethod
    def install_zoom(card_widget, host_window, title=''):
        def show_overlay():
            if ZoomOverlay._active is not None:
                try:
                    ZoomOverlay._active._restore_and_close()
                except Exception:
                    pass
            ov = ZoomOverlay(host_window, card_widget, title)
            ZoomOverlay._active = ov
            try:
                from PyQt5.QtWidgets import QApplication
                screen = QApplication.screenAt(card_widget.mapToGlobal(card_widget.rect().center()))
                if screen is not None:
                    ov.setGeometry(screen.geometry())
                else:
                    ov.setGeometry(ov.host_window.frameGeometry())
            except Exception:
                ov.setGeometry(ov.host_window.frameGeometry())
            ov.showFullScreen()
            try:
                g = ov.geometry()
                print(f"[ZoomOverlay] title={title or 'ZOOM'} fs={ov.isFullScreen()} vis={ov.isVisible()} "
                      f"geo=({g.x()},{g.y()},{g.width()}x{g.height()}) "
                      f"flags={int(ov.windowFlags())}")
            except Exception:
                pass

        def _h(ev):
            show_overlay()
            try:
                ev.accept()
            except Exception:
                pass

        card_widget.mouseDoubleClickEvent = _h
        card_widget.setCursor(Qt.PointingHandCursor)
        card_widget.setToolTip('双击放大')

    def __init__(self, host_window, src_widget, title=''):
        # 关键：覆盖层挂到最外层窗口，而不是某个子容器/scroll 内容区
        # 否则在某些布局下会只出现在底部区域，看起来像“没全屏铺满”。
        top_win = host_window.window() if hasattr(host_window, 'window') else host_window
        super().__init__(None)
        self.host_window = top_win
        self.host = top_win
        self.src = src_widget
        self.setObjectName('ZoomOv')
        self.setStyleSheet(f'#ZoomOv {{ background:rgba(4,17,31,0.94); }}')
        # 真正独立弹窗：必须是 Window 类型；Qt.Tool 在 Windows 上常表现为工具窗，
        # showFullScreen 也可能只在局部区域显示（正是“底部放大一点”的现象）。
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.host_window.installEventFilter(self)
        self.raise_()
        self.setFocus()

        self._orig_parent = src_widget.parent()
        self._orig_index = -1
        self._orig_visible = []
        self._host_hidden_children = []
        try:
            if self._orig_parent and self._orig_parent.layout() is not None:
                self._orig_index = self._orig_parent.layout().indexOf(src_widget)
        except Exception:
            self._orig_index = -1

        self._hide_main_children()
        self._hide_siblings()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 20, 28, 24)
        outer.setSpacing(10)

        head = QHBoxLayout()
        t = QLabel(f'●  {title or "ZOOM"}')
        t.setStyleSheet(
            f'color:{C_ACCENT}; font-size:{FS_MD}px; font-weight:bold;'
            f'letter-spacing:2px; background:transparent;')

        from PyQt5.QtWidgets import QPushButton
        self.btn_back = QPushButton('⬅ 返回')
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
            f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_SM}px;'
            f'padding:{SP_XS}px {SP_MD}px; font-size:{FS_SM}px; font-weight:bold; }}'
            f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG}; }}')
        self.btn_back.clicked.connect(self._restore_and_close)

        hint = QLabel('双击空白 / Esc 返回')
        hint.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        head.addWidget(t)
        head.addStretch()
        head.addWidget(hint)
        head.addSpacing(SP_MD)
        head.addWidget(self.btn_back)
        outer.addLayout(head)

        big = QFrame()
        big.setObjectName('ZoomBig')
        big.setStyleSheet(
            f'#ZoomBig {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_MD}px; }}')
        big_lay = QVBoxLayout(big)
        big_lay.setContentsMargins(10, 10, 10, 10)

        try:
            src_widget.setParent(big)
            src_widget.show()
            big_lay.addWidget(src_widget)
        except Exception:
            pass

        outer.addWidget(big, 1)
        self._big = big

    def _hide_main_children(self):
        try:
            for ch in self.host_window.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
                if ch is self:
                    continue
                if ch.isVisible():
                    self._host_hidden_children.append(ch)
                    ch.hide()
        except Exception:
            pass

    def _hide_siblings(self):
        try:
            if self._orig_parent is None:
                return
            for ch in self._orig_parent.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
                if ch is self.src:
                    continue
                if ch.isVisible():
                    self._orig_visible.append(ch)
                    ch.hide()
        except Exception:
            pass

    def _restore_siblings(self):
        for ch in self._orig_visible:
            try:
                ch.show()
            except Exception:
                pass
        self._orig_visible = []

    def _restore_main_children(self):
        for ch in self._host_hidden_children:
            try:
                ch.show()
            except Exception:
                pass
        self._host_hidden_children = []

    def mouseDoubleClickEvent(self, _e):
        self._restore_and_close()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self._restore_and_close()

    def eventFilter(self, obj, ev):
        # 窗口尺寸变化时实时铺满，避免停留在底部/局部区域
        try:
            if obj is self.host_window and ev.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
                if self.isVisible():
                    try:
                        from PyQt5.QtWidgets import QApplication
                        screen = QApplication.screenAt(self.host_window.mapToGlobal(self.host_window.rect().center()))
                        if screen is not None:
                            self.setGeometry(screen.geometry())
                        else:
                            self.setGeometry(self.host_window.frameGeometry())
                    except Exception:
                        self.setGeometry(self.host_window.frameGeometry())
                    self.showFullScreen()
                    self.raise_()
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    def _restore_and_close(self):
        try:
            try:
                self.host_window.removeEventFilter(self)
            except Exception:
                pass
            if self._orig_parent and self._orig_parent.layout() is not None:
                lay = self._orig_parent.layout()
                self.src.setParent(self._orig_parent)
                self.src.show()
                if self._orig_index >= 0:
                    lay.insertWidget(self._orig_index, self.src)
                else:
                    lay.addWidget(self.src)
            self._restore_siblings()
            self._restore_main_children()
        except Exception:
            pass
        ZoomOverlay._active = None
        self.deleteLater()


# ============================================================
# Flash 融合曲线组件群（D 阶段）
# ============================================================

_TIER_COLORS = [C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT]
_TIER_LABELS_EN = ['L1', 'L2', 'L3', 'L4']
_TIER_LABELS_ZH = ['低', '中', '高', '极高']


class TierProbBars(QWidget):
    """4 档概率温度计：4 条竖直渐变条 + L1/L2/L3/L4 标签 + 百分比。

    用法:
        tpb = TierProbBars(); tpb.set_probs([0.12, 0.41, 0.38, 0.09])
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs = [0.25, 0.25, 0.25, 0.25]
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)

    def set_probs(self, probs):
        if probs is None or len(probs) != 4:
            return
        s = sum(probs) or 1.0
        self._probs = [max(0.0, float(p) / s) for p in probs]
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width(); H = self.height()
        col_w = W / 4
        bar_w = col_w * 0.55
        margin_h = 24
        H_bar = H - margin_h - 16

        cur_max = max(self._probs) if self._probs else 0
        for i in range(4):
            cx = col_w * i + col_w * 0.5
            x = cx - bar_w * 0.5
            prob = self._probs[i]
            seg_h = int(H_bar * prob)
            base_y = H - margin_h
            # 底框
            p.setPen(QPen(QColor(C_BORDER), 1))
            p.setBrush(QBrush(QColor(C_BG_DEEP)))
            p.drawRoundedRect(int(x), int(base_y - H_bar), int(bar_w), int(H_bar), 3, 3)
            # 渐变填充
            if seg_h > 0:
                color = QColor(_TIER_COLORS[i])
                grad = QLinearGradient(x, base_y - seg_h, x, base_y)
                bright = QColor(color); bright.setAlphaF(0.95)
                dark = QColor(color); dark.setAlphaF(0.55)
                grad.setColorAt(0.0, bright)
                grad.setColorAt(1.0, dark)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(grad))
                p.drawRoundedRect(int(x + 1), int(base_y - seg_h + 1),
                                  int(bar_w - 2), int(seg_h - 1), 3, 3)
            # 主导项高亮边框
            if prob == cur_max and cur_max > 0:
                p.setPen(QPen(QColor(_TIER_COLORS[i]), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(int(x - 1), int(base_y - H_bar - 1),
                                  int(bar_w + 2), int(H_bar + 2), 4, 4)
            # 顶部百分比
            p.setPen(QColor(_TIER_COLORS[i]))
            f = QFont('Consolas', 9); f.setBold(True); p.setFont(f)
            p.drawText(QRectF(col_w * i, base_y - H_bar - 14, col_w, 14),
                       Qt.AlignCenter, f'{prob * 100:.0f}%')
            # 底部标签
            p.setPen(QColor(C_DIM if prob != cur_max else _TIER_COLORS[i]))
            f2 = QFont('Microsoft YaHei', 9); f2.setBold(True); p.setFont(f2)
            p.drawText(QRectF(col_w * i, base_y + 2, col_w, 18),
                       Qt.AlignCenter, f'{_TIER_LABELS_EN[i]} · {_TIER_LABELS_ZH[i]}')


class AttnHeatStrip(QWidget):
    """时间维注意力热力条：60 格横向，颜色亮度 ∝ attn[i]。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._attn = [1.0 / 60] * 60
        self._max = 1.0 / 60
        self.setMinimumHeight(28)
        self.setMaximumHeight(36)

    def set_attn(self, attn):
        if attn is None or len(attn) == 0:
            return
        a = list(attn[-60:])
        if len(a) < 60:
            a = [a[0]] * (60 - len(a)) + a
        self._attn = a
        self._max = max(a) or 1.0
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width(); H = self.height()
        cell_w = W / 60
        bar_h = max(8, H - 14)
        y = (H - bar_h) // 2 - 2
        for i in range(60):
            v = self._attn[i] / self._max if self._max > 0 else 0.0
            c = QColor(C_ACCENT)
            c.setAlphaF(0.15 + 0.85 * v)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(c))
            x = cell_w * i
            p.drawRect(QRectF(x, y, max(1.0, cell_w - 0.6), bar_h))
        # 边框
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, y, W - 1, bar_h)
        # 两端时间标
        p.setPen(QColor(C_DIM))
        f = QFont('Microsoft YaHei', 8); p.setFont(f)
        p.drawText(QRectF(2, y + bar_h + 1, 80, 12),
                   Qt.AlignLeft | Qt.AlignVCenter, '-60s')
        p.drawText(QRectF(W - 60, y + bar_h + 1, 58, 12),
                   Qt.AlignRight | Qt.AlignVCenter, '现在')


class FlashFusionPanel(QFrame):
    """融合 Flash 卡：风险流光曲线 + 4 档概率温度计 + 时间注意力热力条 + 右上角大字 tier。

    设计:
      - 顶部一行：左 = 当前 tier+confidence 大字，右 = TierProbBars
      - 中部：综合风险流光时序曲线（带 glow 末端点 + tier 着色 fill）
      - 底部：AttnHeatStrip + 推理元信息

    数据接入:
      自动订阅 RiskBus.nn_result_changed，无需外部调用
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('FlashFusionPanel')
        self.setStyleSheet(
            f'#FlashFusionPanel {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f' border-radius:10px; }}')

        self._risk_buf = deque([0.0] * 60, maxlen=600)
        self._tier_buf = deque([0] * 60, maxlen=600)
        self._last_tier = 0
        self._last_risk = 0.0
        self._glow_phase = 0.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(6)

        # ---- 顶部头部 ----
        head = QHBoxLayout(); head.setSpacing(10)
        dot = QLabel('●')
        dot.setStyleSheet(f'color:{C_ACCENT}; font-size:11px; background:transparent;')
        head.addWidget(dot)
        zh = QLabel('融合 FLASH 曲线')
        zh.setStyleSheet(
            f'color:{C_ACCENT}; font-weight:bold; font-size:13px;'
            f'letter-spacing:1.6px; background:transparent;')
        head.addWidget(zh)
        en = QLabel('NN-DRIVEN MULTIMODAL RISK')
        en.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1.4px; background:transparent;')
        head.addStretch(); head.addWidget(en)
        outer.addLayout(head)

        # ---- 中部：左大字 + 右概率温度计 ----
        mid = QHBoxLayout(); mid.setSpacing(14)

        tier_box = QVBoxLayout(); tier_box.setSpacing(0)
        self._tier_label = QLabel('L1')
        self._tier_label.setStyleSheet(
            f'color:{C_RISK_LOW}; font-size:42px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self._tier_label.setAlignment(Qt.AlignCenter)
        self._tier_zh = QLabel('低风险 · LOW')
        self._tier_zh.setStyleSheet(
            f'color:{C_RISK_LOW}; font-size:11px; letter-spacing:2px;'
            f'background:transparent;')
        self._tier_zh.setAlignment(Qt.AlignCenter)
        self._conf_label = QLabel('置信度 25%')
        self._conf_label.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px;'
            f'background:transparent;')
        self._conf_label.setAlignment(Qt.AlignCenter)
        tier_box.addStretch()
        tier_box.addWidget(self._tier_label)
        tier_box.addWidget(self._tier_zh)
        tier_box.addWidget(self._conf_label)
        tier_box.addStretch()
        tier_wrap = QWidget(); tier_wrap.setStyleSheet('background:transparent;')
        tier_wrap.setLayout(tier_box); tier_wrap.setFixedWidth(140)
        mid.addWidget(tier_wrap)

        self._prob_bars = TierProbBars()
        mid.addWidget(self._prob_bars, 1)
        outer.addLayout(mid)

        # ---- 流光曲线 ----
        try:
            import pyqtgraph as _pg
            self._plot = _pg.PlotWidget()
            self._plot.setBackground(C_PANEL)
            self._plot.getPlotItem().setMenuEnabled(False)
            self._plot.showGrid(x=True, y=True, alpha=0.10)
            self._plot.getPlotItem().getAxis('left').setPen(C_BORDER)
            self._plot.getPlotItem().getAxis('left').setTextPen(C_DIM)
            self._plot.getPlotItem().getAxis('bottom').setPen(C_BORDER)
            self._plot.getPlotItem().getAxis('bottom').setTextPen(C_DIM)
            self._plot.getPlotItem().disableAutoRange()
            self._plot.setXRange(0, 600, padding=0)
            self._plot.setYRange(0, 1.0, padding=0)
            self._plot.setMinimumHeight(120)
            self._curve = self._plot.plot(
                pen=_pg.mkPen(C_RISK_LOW, width=2.4),
                fillLevel=0,
                brush=_pg.mkBrush(QColor(38, 208, 124, 60)))
            self._head_point = _pg.ScatterPlotItem(
                size=14, pen=_pg.mkPen(C_RISK_LOW, width=2),
                brush=_pg.mkBrush(QColor(38, 208, 124, 200)))
            self._plot.addItem(self._head_point)
            # 4 档参考线
            for y, c in [(0.3, C_RISK_LOW), (0.6, C_RISK_MED), (0.8, C_RISK_HIGH)]:
                self._plot.addItem(_pg.InfiniteLine(
                    pos=y, angle=0, pen=_pg.mkPen(c, style=Qt.DashLine, width=1)))
            outer.addWidget(self._plot, 1)
            self._pg_ok = True
        except Exception:
            self._pg_ok = False

        # ---- 底部 attn 热力条 ----
        attn_label = QLabel('时间注意力 · TIME ATTENTION  (NN 看到的关键时段)')
        attn_label.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1.2px;'
            f'background:transparent;')
        outer.addWidget(attn_label)
        self._attn_strip = AttnHeatStrip()
        outer.addWidget(self._attn_strip)

        # 推理元信息行
        meta = QHBoxLayout(); meta.setSpacing(8)
        self._meta_mode = QLabel('Mode: ONNX')
        self._meta_lat = QLabel('Latency: -- ms')
        self._meta_score = QLabel('Risk: 0.00')
        for w in (self._meta_mode, self._meta_lat, self._meta_score):
            w.setStyleSheet(
                f'color:{C_DIM}; font-size:10px; font-family:Consolas;'
                f'background:transparent;')
        meta.addWidget(self._meta_mode); meta.addStretch()
        meta.addWidget(self._meta_lat); meta.addStretch()
        meta.addWidget(self._meta_score)
        outer.addLayout(meta)

        # 流光呼吸动画：让 head_point 周期性变 alpha
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim)
        self._anim_timer.start()

        # 订阅总线
        try:
            from risk_bus import RiskBus
            RiskBus.instance().nn_result_changed.connect(self._on_nn)
        except Exception:
            pass

    def _on_anim(self):
        import math
        self._glow_phase += 0.18
        if not self._pg_ok:
            return
        try:
            import pyqtgraph as _pg
            pulse = 0.5 + 0.5 * math.sin(self._glow_phase)
            color = QColor(_TIER_COLORS[self._last_tier])
            color.setAlphaF(0.35 + 0.55 * pulse)
            size = 12 + 8 * pulse
            self._head_point.setBrush(_pg.mkBrush(color))
            self._head_point.setSize(size)
        except Exception:
            pass

    def _on_nn(self, nn_out):
        try:
            risk = float(nn_out.get('risk_score', 0.0))
            tier = int(nn_out.get('tier', 0))
            probs = nn_out.get('tier_probs') or [0.25] * 4
            attn = nn_out.get('attn') or []
            lat = float(nn_out.get('latency_ms', 0.0))
            mode = nn_out.get('mode', 'fallback')
        except Exception:
            return
        self._last_tier = max(0, min(3, tier))
        self._last_risk = risk
        self._risk_buf.append(risk)
        self._tier_buf.append(self._last_tier)
        color = _TIER_COLORS[self._last_tier]

        # tier 大字 + 置信度
        self._tier_label.setText(_TIER_LABELS_EN[self._last_tier])
        self._tier_label.setStyleSheet(
            f'color:{color}; font-size:42px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        zh_map = ['低风险 · LOW', '中风险 · MEDIUM', '高风险 · HIGH', '极高 · CRITICAL']
        self._tier_zh.setText(zh_map[self._last_tier])
        self._tier_zh.setStyleSheet(
            f'color:{color}; font-size:11px; letter-spacing:2px;'
            f'background:transparent;')
        self._conf_label.setText(f'置信度 {probs[self._last_tier] * 100:.0f}%')

        self._prob_bars.set_probs(probs)
        if attn is not None:
            try:
                self._attn_strip.set_attn(list(attn))
            except Exception:
                pass

        # 曲线
        if self._pg_ok:
            import pyqtgraph as _pg
            n = len(self._risk_buf)
            xs = list(range(n))
            ys = list(self._risk_buf)
            self._curve.setData(xs, ys)
            # 着色：pen 颜色和 fill brush 跟 tier
            self._curve.setPen(_pg.mkPen(color, width=2.4))
            fill_c = QColor(color); fill_c.setAlphaF(0.30)
            self._curve.setBrush(_pg.mkBrush(fill_c))
            self._curve.setFillLevel(0)
            # head 点
            if n > 0:
                self._head_point.setData([xs[-1]], [ys[-1]])
            # 窗口
            self._plot.setXRange(max(0, n - 600), max(60, n), padding=0)

        # 元信息
        self._meta_mode.setText(f'Mode: {"ONNX" if mode == "onnx" else "Fallback"}')
        self._meta_mode.setStyleSheet(
            f'color:{C_OK if mode == "onnx" else C_WARN};'
            f'font-size:10px; font-family:Consolas; background:transparent;')
        self._meta_lat.setText(f'Latency: {lat:.1f} ms')
        self._meta_score.setText(f'Risk: {risk:.2f}')
