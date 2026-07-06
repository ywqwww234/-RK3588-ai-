"""
学生端单页大屏 (D:/G/3) — 工业级 1080p 单页 (1:1:1 + 5:4:3 版)。

布局:
  顶栏:   LOGO + 三模态状态胶囊 + 风险 Hero(标签/54px大数/渐变条) + 时间/退出
  中栏:   视觉(33%) / 心率(33%) / 脑电(33%)，每张卡左侧 3px 模态色条 + 融合权重 0.35/0.40/0.25
  底栏:   风险时序(0.3/0.6/0.8) / 神经网络洞察(贡献+延迟+量化+模型+准确率) / AI 干预日志   [5:4:3]

红线: 不动信号槽，所有数据接口与原 _route_mock_waves / _on_physio_features /
      _update_dashboard_data / _update_risk / _nn_tick 保持兼容。
"""

from collections import deque
from datetime import datetime
import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QGridLayout, QFrame, QPushButton, QTextEdit,
                              QScrollArea, QComboBox)
import pyqtgraph as pg

# VNC 检测
_is_vnc = (os.environ.get('MINDROOM_VNC', '') == '1' or
           os.environ.get('DISPLAY', '').startswith(':99'))

import theme as _theme
from theme import (C_BG, C_BG_DEEP, C_PANEL, C_PANEL_HI, C_BORDER,
                   C_ACCENT, C_DIM, C_TEXT, C_OK, C_WARN, C_RISK,
                   C_VISION, C_HEART, C_BRAIN,
                   C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT, GLOBAL_QSS,
                   SP_XS, SP_SM, SP_MD, SP_LG, SP_XL, SP_XXL,
                   FS_TINY, FS_XS, FS_SM, FS_MD, FS_LG, FS_XL, FS_HERO,
                   R_SM, R_MD, R_LG, BORDER_W)
from ui_widgets import (StatusPill, RiskHeroCard, ContribStackBar, ZoomOverlay,
                        FlashFusionPanel)
from ui_dashboard_modules import (build_vision_card, build_heart_card,
                                   build_brain_card)


def _section_head(title_zh, title_en, color):
    head = QHBoxLayout()
    head.setSpacing(SP_SM)
    dot = QLabel('●'); dot.setStyleSheet(f'color:{color}; font-size:{FS_XS}px; background:transparent;')
    z = QLabel(title_zh); z.setStyleSheet(
        f'color:{color}; font-weight:bold; font-size:{FS_MD}px;'
        f'letter-spacing:1.6px; background:transparent;')
    e = QLabel(title_en); e.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_TINY}px; letter-spacing:1.5px; background:transparent;')
    e.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    head.addWidget(dot); head.addWidget(z); head.addStretch(); head.addWidget(e)
    return head


def _wrap_card(layout):
    f = QFrame(); f.setObjectName('SubCard')
    f.setStyleSheet(
        f'#SubCard {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_BORDER};'
        f'border-radius:{R_MD}px; }}')
    f.setLayout(layout)
    return f


# ===================== 主装配 =====================

def build_dashboard(self, bci_panel_widget):
    self.setStyleSheet((self.styleSheet() or '') + GLOBAL_QSS)

    from PyQt5.QtWidgets import QMainWindow as _QMW
    if isinstance(self, _QMW):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 🔧 消抖关键 1: 强制隐藏横向滚动条,避免内容微超界时反复出没
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded if _is_vnc else Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(f'QScrollArea {{ background:{C_BG}; border:none; }}')
        host = QWidget(); host.setStyleSheet(f'background:{C_BG};')
        scroll.setWidget(host); self.setCentralWidget(scroll)
    else:
        host = self

    self.bg = QFrame(); self.bg.setObjectName('MainBg')
    self.bg.setStyleSheet(
        f'#MainBg {{ background:{C_BG}; border-radius:{R_LG}px;'
        f'border:{BORDER_W}px solid {C_BORDER}; }}')
    root_outer = QVBoxLayout(host)
    root_outer.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
    root_outer.addWidget(self.bg)

    root = QVBoxLayout(self.bg)
    root.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_MD)
    root.setSpacing(SP_MD)

    # ============== 顶栏 ==============
    top = QFrame(); top.setObjectName('TopBar')
    top.setStyleSheet(
        f'#TopBar {{ background:{C_BG_DEEP}; border:{BORDER_W}px solid {C_BORDER};'
        f'border-radius:{R_MD}px; }}')
    top.setMinimumHeight(96)
    top_lay = QHBoxLayout(top)
    top_lay.setContentsMargins(SP_XL, SP_MD, SP_XL, SP_MD); top_lay.setSpacing(SP_XL)

    logo_box = QVBoxLayout(); logo_box.setSpacing(0)
    logo = QLabel('🛡  MINDROOM  GUARD')
    logo.setStyleSheet(
        f'color:{C_ACCENT}; font-size:{FS_MD}px; font-weight:bold;'
        f'letter-spacing:2px; background:transparent;')
    sub = QLabel('基于边缘计算的抑郁风险预警系统  ·  RK3588  ·  Edge-AI')
    sub.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_TINY}px; letter-spacing:1.2px; background:transparent;')
    logo_box.addWidget(logo); logo_box.addWidget(sub)
    top_lay.addLayout(logo_box)

    from PyQt5.QtWidgets import QSizePolicy
    pill_box = QVBoxLayout(); pill_box.setSpacing(SP_SM); pill_box.setContentsMargins(0, 0, 0, 0)
    self.pill_top_vision = StatusPill(); self.pill_top_vision.set_state('sim', '视觉  SIM')
    self.pill_top_heart  = StatusPill(); self.pill_top_heart.set_state('sim',  '心率  SIM')
    self.pill_top_brain  = StatusPill(); self.pill_top_brain.set_state('sim',  '脑电  SIM')
    for _pill in (self.pill_top_vision, self.pill_top_heart, self.pill_top_brain):
        _pill.setFixedHeight(34)
        _pill.setMinimumWidth(130)
        _pill.setAlignment(Qt.AlignCenter)
        _pill.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    row1 = QHBoxLayout(); row1.setSpacing(SP_MD); row1.setContentsMargins(0, 0, 0, 0)
    row1.addWidget(self.pill_top_vision, 1)
    row1.addWidget(self.pill_top_heart,  1)
    row1.addWidget(self.pill_top_brain,  1)
    pill_box.addLayout(row1)
    self.pill_top_nn = StatusPill(); self.pill_top_nn.set_state('info', '神经网络  待加载')
    self.pill_top_nn.setFixedHeight(34)
    # 顶部红色状态文案较长，放宽宽度并允许动态伸缩，避免裁切
    self.pill_top_nn.setMinimumWidth(300 if _is_vnc else 500)
    self.pill_top_nn.setAlignment(Qt.AlignCenter)
    self.pill_top_nn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    pill_box.addWidget(self.pill_top_nn)
    pill_wrap = QWidget(); pill_wrap.setStyleSheet('background:transparent;')
    pill_wrap.setLayout(pill_box)
    pill_wrap.setMinimumWidth(0 if _is_vnc else 760)
    pill_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    top_lay.addWidget(pill_wrap, 2)

    # 🔧 消抖关键 3: risk_hero 固定宽度,避免数字"0.82"→"低·LOW"切换时挤压两侧
    self.risk_hero = RiskHeroCard()
    self.risk_hero.setMinimumWidth(300 if _is_vnc else 420)
    self.risk_hero.setMaximumWidth(520)
    top_lay.addWidget(self.risk_hero, 1)

    cl_box = QVBoxLayout(); cl_box.setSpacing(2)
    self.lbl_clock = QLabel('00:00:00')
    self.lbl_clock.setStyleSheet(
        f'color:{C_ACCENT}; font-size:{FS_XL}px; font-weight:bold;'
        f'letter-spacing:3px; font-family:Consolas; background:transparent;')
    self.lbl_clock.setAlignment(Qt.AlignRight); self.lbl_clock.setFixedWidth(140)
    self.lbl_date = QLabel('----/--/--')
    self.lbl_date.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.6px; background:transparent;')
    self.lbl_date.setAlignment(Qt.AlignRight); self.lbl_date.setFixedWidth(220)
    btn_exit = QPushButton('⏏  退出')
    btn_exit.setCursor(Qt.PointingHandCursor)
    btn_exit.clicked.connect(self.close)
    btn_exit.setStyleSheet(
        f'QPushButton {{ background:{C_PANEL_HI}; color:{C_RISK};'
        f'border:{BORDER_W}px solid {C_RISK}; border-radius:{R_SM}px; padding:{SP_XS}px {SP_LG}px;'
        f'font-weight:bold; }}'
        f'QPushButton:hover {{ background:{C_RISK}; color:{C_BG_DEEP}; }}')

    self.btn_toolbox = QPushButton('⚙  工具箱')
    self.btn_toolbox.setCursor(Qt.PointingHandCursor)
    self.btn_toolbox.setStyleSheet(
        f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
        f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_SM}px; padding:{SP_XS}px {SP_LG}px;'
        f'font-weight:bold; letter-spacing:1px; }}'
        f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG_DEEP}; }}')

    def _open_toolbox_clicked(_=False, win=self):
        try:
            from ui_toolbox import ToolboxDialog
            dlg = ToolboxDialog(win)
            dlg.exec_()
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f'[Toolbox] 启动失败:\n{tb}')
            try:
                if hasattr(win, 'ai_console'):
                    win.ai_console.append(f'> [Toolbox 错误] {exc}')
                    for line in tb.splitlines()[-5:]:
                        win.ai_console.append(f'>   {line}')
            except Exception:
                pass
    self.btn_toolbox.clicked.connect(_open_toolbox_clicked)

    btn_row = QHBoxLayout(); btn_row.setSpacing(SP_SM)
    btn_row.addStretch(); btn_row.addWidget(self.btn_toolbox); btn_row.addWidget(btn_exit)

    cl_box.addWidget(self.lbl_clock); cl_box.addWidget(self.lbl_date)
    cl_box.addLayout(btn_row)
    top_lay.addLayout(cl_box)

    self._clock_timer = QTimer(self)
    self._clock_timer.setInterval(1000)
    self._clock_timer.timeout.connect(_tick_clock(self))
    self._clock_timer.start(); _tick_clock(self)()

    root.addWidget(top)

    # ============== 中栏 1:1:1 (固定宽度,不受内容影响) ==============
    mid = QHBoxLayout(); mid.setSpacing(SP_MD)
    # 🔧 消抖关键 4: 三大卡固定宽度比例,避免内部文本/图表变化时互相挤压
    self.vision_card = build_vision_card(self)
    self.heart_card = build_heart_card(self)
    self.brain_card = build_brain_card(self, bci_panel_widget)
    # 用 addWidget(w, stretch) 而非 addWidget(w, 1) 确保等宽
    mid.addWidget(self.vision_card, 1)
    mid.addWidget(self.heart_card, 1)
    mid.addWidget(self.brain_card, 1)
    root.addLayout(mid, 1)

    # ===== 三模态节点放大: 嵌入式工业级全屏（走 11.py/1.py UI，数据直连主系统） =====
    _installed_nodes = 0
    try:
        from embedded_node import EmbeddedNodeOverlay as _EOV2
        import os as _os
        _node_root = _os.path.dirname(_os.path.abspath(__file__))
        _hr_py = _os.path.join(_node_root, '1.py')
        _brain_py = _os.path.join(_node_root, '11.py')
        if _os.path.exists(_hr_py):
            _EOV2.install(
                card_widget=self.heart_card, host_window=self,
                module_path=_hr_py,
                window_class='HRMonitorWindow',
                title='心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE',
            )
            _installed_nodes += 1
        if _os.path.exists(_brain_py):
            _EOV2.install(
                card_widget=self.brain_card, host_window=self,
                module_path=_brain_py,
                window_class='EEGMonitorWindow',
                title='脑 电 感 知 节 点  ·  TGAM  EDGE  NODE')
            _installed_nodes += 1
    except Exception as _exc2:
        print(f'[Dashboard] embedded_node 加载失败: {_exc2}')

    # 最终回退: ZoomOverlay 普通放大
    if _installed_nodes < 3:
        try:
            if _installed_nodes < 1:
                ZoomOverlay.install_zoom(self.vision_card, self, '视觉感知 · VISION')
            if _installed_nodes < 2:
                ZoomOverlay.install_zoom(self.brain_card, self, '脑电感知 · NEURAL')
        except Exception:
            pass

    # ============== 底栏 5:4:3 ==============
    bot = QHBoxLayout(); bot.setSpacing(SP_MD)

    # ---- 综合风险 Flash 融合曲线（NN 驱动：曲线+4 档概率+attn 热力条+tier 大字）----
    self.flash_panel = FlashFusionPanel()
    self.risk_trend_card = self.flash_panel  # 保留旧名字供别处引用
    # 旧代码（_nn_tick 等）写 self.risk_trend_curve 的，给个内部 alias，双写不影响 FlashFusionPanel（它自己订阅总线）
    self.risk_trend_plot = self.flash_panel._plot if getattr(self.flash_panel, '_pg_ok', False) else None
    self.risk_trend_curve = self.flash_panel._curve if getattr(self.flash_panel, '_pg_ok', False) else None
    self._buf_risk_trend = deque(maxlen=600)
    bot.addWidget(self.flash_panel, 5)

    # ---- 神经网络洞察 ----
    nn_lay = QVBoxLayout(); nn_lay.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_MD); nn_lay.setSpacing(SP_SM)
    nn_lay.addLayout(_section_head('神经网络洞察', 'NEURAL INSIGHT  ·  Multimodal Fusion', C_BRAIN))

    cl = QLabel('模态贡献 · MODALITY CONTRIBUTION')
    cl.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.5px; background:transparent;')
    nn_lay.addWidget(cl)
    self.contrib_bar = ContribStackBar()
    self.contrib_bar.set_weights(0.35, 0.40, 0.25)
    nn_lay.addWidget(self.contrib_bar)

    legend = QHBoxLayout(); legend.setSpacing(SP_LG)
    for color, name in [(C_VISION, '视觉 0.35'),
                        (C_HEART, 'HRV 0.40'),
                        (C_BRAIN, '脑电 0.25')]:
        l = QLabel(f'■ {name}')
        l.setStyleSheet(f'color:{color}; font-size:{FS_XS}px; font-weight:bold; background:transparent;')
        legend.addWidget(l)
    legend.addStretch()
    nn_lay.addLayout(legend)
    nn_lay.addSpacing(SP_XS)

    grid = QGridLayout(); grid.setHorizontalSpacing(SP_LG); grid.setVerticalSpacing(2)
    self.nn_lbl_model = QLabel('CNN-BiGRU-Att')
    self.nn_lbl_model.setStyleSheet(
        f'color:{C_ACCENT}; font-size:{FS_MD}px; font-weight:bold;'
        f'background:transparent; font-family:Consolas;')
    self.nn_lbl_model.setFixedWidth(140)
    self.nn_lbl_lat = QLabel('-- ms')
    self.nn_lbl_lat.setStyleSheet(
        f'color:{C_OK}; font-size:{FS_LG}px; font-weight:bold;'
        f'background:transparent; font-family:Consolas;')
    self.nn_lbl_lat.setFixedWidth(80)
    self.nn_lbl_acc = QLabel('89.7%')
    self.nn_lbl_acc.setStyleSheet(
        f'color:{C_BRAIN}; font-size:{FS_LG}px; font-weight:bold;'
        f'background:transparent; font-family:Consolas;')
    self.nn_lbl_acc.setFixedWidth(80)
    self.nn_lbl_quant = QLabel('INT8 / RKNN')
    self.nn_lbl_quant.setStyleSheet(
        f'color:{C_VISION}; font-size:{FS_MD}px; font-weight:bold;'
        f'background:transparent; font-family:Consolas;')
    self.nn_lbl_quant.setFixedWidth(120)

    self.nn_lbl_model_h = QLabel('模型 MODEL')
    self.nn_lbl_lat_h   = QLabel('延迟 LATENCY')
    self.nn_lbl_acc_h   = QLabel('准确率 ACC')
    self.nn_lbl_quant_h = QLabel('量化 QUANT')
    for h in (self.nn_lbl_model_h, self.nn_lbl_lat_h, self.nn_lbl_acc_h, self.nn_lbl_quant_h):
        h.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.4px; background:transparent;')
    grid.addWidget(self.nn_lbl_model_h, 0, 0); grid.addWidget(self.nn_lbl_model, 1, 0)
    grid.addWidget(self.nn_lbl_lat_h,   0, 1); grid.addWidget(self.nn_lbl_lat,   1, 1)
    grid.addWidget(self.nn_lbl_acc_h,   0, 2); grid.addWidget(self.nn_lbl_acc,   1, 2)
    grid.addWidget(self.nn_lbl_quant_h, 0, 3); grid.addWidget(self.nn_lbl_quant, 1, 3)
    nn_lay.addLayout(grid)

    # 决策溯源：实时拼成 "NN 主因：HRV 41% · RMSSD=28 偏低"（订阅总线 modal_w + factor）
    nn_lay.addSpacing(SP_SM)
    src_head = QLabel('决策溯源 · DECISION TRACE')
    src_head.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.5px; background:transparent;')
    nn_lay.addWidget(src_head)
    # 🔧 消抖关键 5: 固定宽度+单行截断,避免文本变化触发 nn_card 高度变化
    self.nn_lbl_reason = QLabel('等待 NN 推理…')
    self.nn_lbl_reason.setStyleSheet(
        f'color:{C_OK}; font-size:{FS_XS}px; font-weight:bold;'
        f'background:transparent;')
    self.nn_lbl_reason.setWordWrap(False)
    self.nn_lbl_reason.setFixedHeight(18)
    nn_lay.addWidget(self.nn_lbl_reason)

    # 决策路径：固定的网络结构展示 + 末端 tier
    path_head = QLabel('决策路径 · ARCHITECTURE')
    path_head.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.5px;'
        f'background:transparent; padding-top:{SP_XS}px;')
    nn_lay.addWidget(path_head)
    self.nn_lbl_path = QLabel('[60×25] → CNN → BiGRU → SoftAttn → [—,—,—,—] → argmax')
    self.nn_lbl_path.setStyleSheet(
        f'color:{C_ACCENT}; font-size:{FS_XS}px; font-family:Consolas;'
        f'background:transparent;')
    self.nn_lbl_path.setWordWrap(False)
    self.nn_lbl_path.setFixedHeight(18)
    nn_lay.addWidget(self.nn_lbl_path)

    def _nn_update(nn_out, _lr=self.nn_lbl_reason, _lp=self.nn_lbl_path):
        try:
            from risk_bus import RiskBus as _RB
            mw = nn_out.get('modal_w') or (0.33, 0.33, 0.34)
            names = ('视觉', 'HRV', '脑电')
            facts = ('vision', 'hrv', 'eeg')
            i = int(max(range(3), key=lambda k: mw[k]))
            factor_text = _RB.instance().get_modal_factor(facts[i])
            _lr.setText(
                f'NN 主因：{names[i]} 贡献 {mw[i] * 100:.0f}%  ·  {factor_text}')
            probs = nn_out.get('tier_probs') or [0, 0, 0, 0]
            tier = int(nn_out.get('tier', 0))
            tier_label = ['L1', 'L2', 'L3', 'L4'][tier]
            probs_txt = ', '.join(f'{p:.2f}' for p in probs)
            _lp.setText(
                f'[60×25] → CNN → BiGRU → SoftAttn → [{probs_txt}] → argmax → {tier_label}')
        except Exception:
            pass
    try:
        from risk_bus import RiskBus
        RiskBus.instance().nn_result_changed.connect(_nn_update)
    except Exception:
        pass

    nn_lay.addStretch(1)
    self.nn_card = _wrap_card(nn_lay)
    bot.addWidget(self.nn_card, 4)

    # ---- AI 干预日志 ----
    ai_lay = QVBoxLayout(); ai_lay.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_MD); ai_lay.setSpacing(SP_SM)

    _ai_head_row = QHBoxLayout(); _ai_head_row.setSpacing(SP_SM)
    _ai_head_col = QVBoxLayout(); _ai_head_col.setSpacing(2)
    _dot_ai = QLabel('●'); _dot_ai.setStyleSheet(f'color:{C_OK}; font-size:{FS_XS}px; background:transparent;')
    _zh_ai = QLabel('AI 干预终端')
    _zh_ai.setStyleSheet(
        f'color:{C_OK}; font-weight:bold; font-size:{FS_MD}px;'
        f'letter-spacing:1.6px; background:transparent;')

    self.btn_whisper = QPushButton('💬 悄悄说')
    self.btn_whisper.setCursor(Qt.PointingHandCursor)
    self.btn_whisper.setToolTip('今天怎么样？想说点什么都可以，仅 AI 用于了解你的状态，不会原文告诉任何人。')
    self.btn_whisper.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_DIM};'
        f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_MD}px;'
        f'padding:2px {SP_MD}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ color:{C_ACCENT}; border:{BORDER_W}px solid {C_ACCENT}; }}')

    self.btn_ai_ack = QPushButton('✅ 我已处理')
    self.btn_ai_ack.setCursor(Qt.PointingHandCursor)
    self.btn_ai_ack.setToolTip('确认已查看建议，恢复正常监测状态。')
    self.btn_ai_ack.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_OK};'
        f'border:{BORDER_W}px solid {C_OK}; border-radius:{R_MD}px;'
        f'padding:2px {SP_MD}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_OK}; color:{C_BG_DEEP}; }}')

    self.btn_replay_manual = QPushButton('⏺ 手动回放')
    self.btn_replay_manual.setCursor(Qt.PointingHandCursor)
    self.btn_replay_manual.setToolTip('手动触发一次事件回放采集（演示模式）。')
    self.btn_replay_manual.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_ACCENT};'
        f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_MD}px;'
        f'padding:2px {SP_MD}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG_DEEP}; }}')

    self.btn_ai_deep = QPushButton('深度分析')
    self.btn_ai_deep.setCursor(Qt.PointingHandCursor)
    self.btn_ai_deep.setToolTip('使用 glm-4 做双阶段归因+建议（较慢，适合人工查看）。')
    self.btn_ai_deep.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_WARN};'
        f'border:{BORDER_W}px solid {C_WARN}; border-radius:{R_MD}px;'
        f'padding:2px {SP_MD}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_WARN}; color:{C_BG_DEEP}; }}')

    self.combo_llm_target = QComboBox()
    self.combo_llm_target.addItems([
        '学生+老师',
        '仅学生',
        '家长',
        '仅老师',
    ])
    self.combo_llm_target.setToolTip('AI 建议的表述对象（写入结构化上下文 output_target）。')
    self.combo_llm_target.setFixedHeight(26)
    self.combo_llm_target.setStyleSheet(
        f'QComboBox {{ background:{C_BG_DEEP}; color:{C_DIM}; border:1px solid {C_BORDER};'
        f'border-radius:{R_MD}px; padding:2px 6px; font-size:{FS_XS}px; min-width:88px; }}'
        f'QComboBox:hover {{ color:{C_ACCENT}; border:1px solid {C_ACCENT}; }}')

    self.pill_replay = StatusPill()
    self.pill_replay.set_state('sim', '回放 待机')
    self.pill_replay.setFixedHeight(30)
    self.pill_replay.setMinimumWidth(120)

    self.btn_replay = QPushButton('⏪ 回放')
    self.btn_replay.setCursor(Qt.PointingHandCursor)
    self.btn_replay.setToolTip('打开最近一次风险事件回放(JSON)。')
    self.btn_replay.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_ACCENT};'
        f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_MD}px;'
        f'padding:2px {SP_SM}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG_DEEP}; }}')

    self.btn_record_toggle = QPushButton('🔴 开始采集')
    self.btn_record_toggle.setCursor(Qt.PointingHandCursor)
    self.btn_record_toggle.setToolTip('开启/关闭风险持久化记录。')
    self.btn_record_toggle.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_WARN};'
        f'border:{BORDER_W}px solid {C_WARN}; border-radius:{R_MD}px;'
        f'padding:2px {SP_SM}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_WARN}; color:{C_BG_DEEP}; }}')

    self.btn_export_plain = QPushButton('📤 明文CSV')
    self.btn_export_plain.setCursor(Qt.PointingHandCursor)
    self.btn_export_plain.setToolTip('导出可用Excel打开的明文风险CSV。')
    self.btn_export_plain.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_OK};'
        f'border:{BORDER_W}px solid {C_OK}; border-radius:{R_MD}px;'
        f'padding:2px {SP_SM}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_OK}; color:{C_BG_DEEP}; }}')

    self.btn_export_train = QPushButton('📦 训练集')
    self.btn_export_train.setCursor(Qt.PointingHandCursor)
    self.btn_export_train.setToolTip('导出可训练数据: train/val/test。')
    self.btn_export_train.setStyleSheet(
        f'QPushButton {{ background:transparent; color:{C_VISION};'
        f'border:{BORDER_W}px solid {C_VISION}; border-radius:{R_MD}px;'
        f'padding:2px {SP_SM}px; font-size:{FS_XS}px; font-weight:bold;'
        f'letter-spacing:1px; min-width:0; }}'
        f'QPushButton:hover {{ background:{C_VISION}; color:{C_BG_DEEP}; }}')

    def _open_whisper_clicked(_, win=self):
        try:
            from ui_toolbox import QuietWhisperDialog
            dlg = QuietWhisperDialog(win)
            dlg.exec_()
        except Exception as exc:
            print(f'[Whisper] 启动失败: {exc}')
    self.btn_whisper.clicked.connect(_open_whisper_clicked)

    _en_ai = QLabel('AI INTERVENTION\nLOG')
    _en_ai.setStyleSheet(
        f'color:{C_DIM}; font-size:{FS_TINY}px; letter-spacing:1.2px; background:transparent;')
    _en_ai.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    _ai_head_top = QHBoxLayout(); _ai_head_top.setSpacing(SP_SM)
    _ai_head_top.addWidget(_dot_ai)
    _ai_head_top.addWidget(_zh_ai)
    _ai_head_top.addStretch()

    _ai_head_row1 = QHBoxLayout(); _ai_head_row1.setSpacing(4)
    _ai_head_row1.addWidget(self.btn_whisper)
    _ai_head_row1.addWidget(self.btn_ai_ack)
    _ai_head_row1.addWidget(self.btn_replay_manual)
    _ai_head_row1.addWidget(self.pill_replay)
    _ai_head_row1.addStretch()

    _ai_head_row1b = QHBoxLayout(); _ai_head_row1b.setSpacing(4)
    _ai_head_row1b.addWidget(self.btn_ai_deep)
    _ai_head_row1b.addWidget(self.combo_llm_target)
    _ai_head_row1b.addStretch()

    _ai_head_row2 = QHBoxLayout(); _ai_head_row2.setSpacing(4)
    _ai_head_row2.addWidget(self.btn_replay)
    _ai_head_row2.addWidget(self.btn_record_toggle)
    _ai_head_row2.addWidget(self.btn_export_plain)
    _ai_head_row2.addWidget(self.btn_export_train)
    _ai_head_row2.addStretch()

    _en_ai.setVisible(False)
    _ai_head_col.addLayout(_ai_head_top)
    _ai_head_col.addLayout(_ai_head_row1)
    _ai_head_col.addLayout(_ai_head_row1b)
    _ai_head_col.addLayout(_ai_head_row2)
    ai_lay.addLayout(_ai_head_col)
    self.ai_console = QTextEdit()
    self.ai_console.setReadOnly(True)
    # 🔧 消抖关键 6: 强制换行+禁止横向滚动条,避免长文本撑宽 ai_card → 三大卡抖动
    self.ai_console.setLineWrapMode(QTextEdit.WidgetWidth)
    self.ai_console.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    from PyQt5.QtWidgets import QSizePolicy as _SP
    self.ai_console.setSizePolicy(_SP.Ignored, _SP.Expanding)
    self.ai_console.setStyleSheet(
        f'QTextEdit {{ background:{C_BG_DEEP}; color:{C_OK};'
        f'border:{BORDER_W}px solid {C_BORDER}; border-radius:{R_SM}px;'
        f'font-family:Consolas; font-size:{FS_XS}px; padding:{SP_SM}px; }}')
    self.ai_console.setText('> 系统就绪，等待数据流…\n')
    ai_lay.addWidget(self.ai_console, 1)
    self.ai_card = _wrap_card(ai_lay)
    bot.addWidget(self.ai_card, 3)

    root.addLayout(bot, 0)

    try:
        ZoomOverlay.install_zoom(self.risk_trend_card, self, '综合风险时序 · RISK TIMELINE')
        ZoomOverlay.install_zoom(self.nn_card,         self, '神经网络洞察 · NEURAL INSIGHT')
        ZoomOverlay.install_zoom(self.ai_card,         self, 'AI 干预终端 · AI INTERVENTION')
    except Exception:
        pass


def _tick_clock(self):
    def _():
        now = datetime.now()
        self.lbl_clock.setText(now.strftime('%H:%M:%S'))
        self.lbl_date.setText(now.strftime('%Y / %m / %d  %A'))
    return _
