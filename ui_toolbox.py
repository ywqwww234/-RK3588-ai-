"""
功能工具箱 · TOOLBOX

定位：MindRoom Guard 主面板顶部"⚙ 工具箱"按钮的弹出层，承载所有
辅助筛查 / 守护响应 / 标定 / 回放等独立功能。每个功能都是独立的
QDialog，带"← 返回工具箱"导航，不破坏主面板布局。

红线（贯穿全部对话框）：
  - 系统是「AI 辅助筛查」，不是临床诊断、不替代专业人员
  - 设计前提：抑郁人员可能不愿说 / 自己不知道
    → 不依赖本人按按钮求救
    → AI 监测到异常 → 主动通知预设守护人 → 守护人温和介入
  - 紧急专业渠道（12320 / 120）只在 L4 紧急预警里由系统附带显示，
    不作为学生面板的"主入口"

当前实现：
  - ToolboxDialog            工具箱菜单（7 入口，1 已开通）
  - GuardianResponseDialog   守护人响应（已开通）
  - QuietWhisperDialog       学生侧"悄悄说"低门槛入口
"""

import os
import json
import time

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter
from PyQt5.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout,
                              QHBoxLayout, QGridLayout, QFrame, QScrollArea,
                              QApplication, QGraphicsDropShadowEffect, QMessageBox,
                              QTextEdit, QCheckBox, QSizePolicy)

from theme import (C_BG, C_BG_DEEP, C_PANEL, C_PANEL_HI, C_BORDER, C_BORDER_HI,
                   C_ACCENT, C_ACCENT2, C_DIM, C_TEXT, C_TEXT_DIM,
                   C_OK, C_WARN, C_RISK, C_VISION, C_HEART, C_BRAIN,
                   C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT)


# ===================== 通用：全屏遮罩对话框基类 =====================

class _OverlayDialog(QDialog):
    """半透明遮罩 + 居中卡片的全屏对话框。子类只需要填 card_lay。

    点击遮罩区域（card 外）关闭；首次弹出 300ms 内防抖，避免上层按钮的
    残余鼠标事件秒关对话框。
    """

    def __init__(self, parent=None, card_w=1100, card_h=720):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(True)
        self._card_w = card_w
        self._card_h = card_h
        self._opened_ts = time.time()
        # 弹出后短延时再标记为"可点击关闭"
        QTimer.singleShot(300, self._enable_click_close)
        self._click_close_armed = False

        if parent is not None:
            self.resize(parent.size())
        else:
            screen = QApplication.primaryScreen().size()
            self.resize(screen)

        self._root = QHBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(self)
        self.card.setObjectName('OverlayCard')
        self.card.setFixedSize(self._card_w, self._card_h)
        self.card.setStyleSheet(
            f'#OverlayCard {{ background:{C_BG_DEEP};'
            f'border:1px solid {C_ACCENT}; border-radius:14px; }}')
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 14)
        shadow.setColor(QColor(0, 229, 255, 80))
        self.card.setGraphicsEffect(shadow)

        self._root.addStretch()
        self._root.addWidget(self.card, alignment=Qt.AlignCenter)
        self._root.addStretch()

        self.card_lay = QVBoxLayout(self.card)
        self.card_lay.setContentsMargins(0, 0, 0, 0)
        self.card_lay.setSpacing(0)

    def _enable_click_close(self):
        self._click_close_armed = True

    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(2, 6, 12, 210))

    def mousePressEvent(self, evt):
        # 防抖：弹出后 300ms 内不响应背景点击（避免上层按钮的残余事件秒关）
        if not self._click_close_armed:
            return
        if not self.card.geometry().contains(evt.pos()):
            self.accept()
        else:
            super().mousePressEvent(evt)


def _make_title_bar(title_zh, title_en, on_back, back_text='✕ 关闭', accent=C_ACCENT):
    bar = QFrame()
    bar.setObjectName('TitleBar')
    bar.setStyleSheet(
        f'#TitleBar {{ background:{C_BG}; border-top-left-radius:14px;'
        f'border-top-right-radius:14px; border-bottom:1px solid {C_BORDER}; }}')
    bar.setFixedHeight(60)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(22, 0, 14, 0)
    lay.setSpacing(14)

    title_box = QVBoxLayout()
    title_box.setSpacing(0)
    zh = QLabel(title_zh)
    zh.setStyleSheet(
        f'color:{accent}; font-size:17px; font-weight:bold;'
        f'letter-spacing:2px; background:transparent;')
    en = QLabel(title_en)
    en.setStyleSheet(
        f'color:{C_DIM}; font-size:10px; letter-spacing:1.6px; background:transparent;')
    title_box.addWidget(zh)
    title_box.addWidget(en)
    lay.addLayout(title_box)
    lay.addStretch()

    btn_back = QPushButton(back_text)
    btn_back.setCursor(Qt.PointingHandCursor)
    btn_back.setStyleSheet(
        f'QPushButton {{ background:{C_PANEL_HI}; color:{C_RISK};'
        f'border:1px solid {C_RISK}; border-radius:6px; padding:6px 18px;'
        f'font-weight:bold; letter-spacing:1px; }}'
        f'QPushButton:hover {{ background:{C_RISK}; color:{C_BG_DEEP}; }}')
    btn_back.clicked.connect(on_back)
    lay.addWidget(btn_back)
    return bar


def _make_disclaimer_bar(text=None):
    """红线声明栏。"""
    if text is None:
        text = ('⚠  本系统为 AI 辅助筛查工具，不替代专业人员诊断。'
                '所有预警均为参考，最终判断与处置请由家长 / 心理老师 / 医疗机构完成。')
    bar = QFrame()
    bar.setObjectName('DisclaimerBar')
    bar.setStyleSheet(
        f'#DisclaimerBar {{ background:{C_BG}; border-bottom-left-radius:14px;'
        f'border-bottom-right-radius:14px; border-top:1px solid {C_BORDER}; }}')
    bar.setFixedHeight(46)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(22, 0, 22, 0)
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f'color:{C_WARN}; font-size:11px; letter-spacing:1px; background:transparent;')
    lbl.setWordWrap(True)
    lay.addWidget(lbl)
    return bar


def _section_header(title_zh, title_en, color):
    h = QHBoxLayout()
    h.setSpacing(8)
    dot = QLabel('●')
    dot.setStyleSheet(f'color:{color}; font-size:11px; background:transparent;')
    zh = QLabel(title_zh)
    zh.setStyleSheet(
        f'color:{color}; font-weight:bold; font-size:13px;'
        f'letter-spacing:2px; background:transparent;')
    en = QLabel(title_en)
    en.setStyleSheet(
        f'color:{C_DIM}; font-size:9px; letter-spacing:1.4px; background:transparent;')
    h.addWidget(dot)
    h.addWidget(zh)
    h.addSpacing(8)
    h.addWidget(en)
    h.addStretch()
    return h


# ===================== 工具箱菜单 =====================

class _ToolCard(QFrame):
    """工具箱按钮卡片：图标 + 中文 + 英文 + 状态标 + 描述。"""

    def __init__(self, icon, zh, en, desc, accent=C_ACCENT,
                 status='ready', on_click=None, parent=None):
        super().__init__(parent)
        self.on_click = on_click
        self.status = status
        self.accent = accent
        self.setObjectName('ToolCard')
        self.setCursor(Qt.PointingHandCursor if status == 'ready' else Qt.ForbiddenCursor)
        self.setFixedSize(310, 160)

        border = accent if status == 'ready' else C_BORDER
        bg = C_PANEL if status == 'ready' else C_BG_DEEP
        self.setStyleSheet(
            f'#ToolCard {{ background:{bg}; border:1px solid {border};'
            f'border-radius:10px; }}'
            f'#ToolCard:hover {{ border:1px solid {accent}; }}')

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(8)
        ic = QLabel(icon)
        ic.setStyleSheet(
            f'color:{accent}; font-size:22px; background:transparent;')
        head.addWidget(ic)
        ttl_box = QVBoxLayout()
        ttl_box.setSpacing(0)
        zh_lbl = QLabel(zh)
        zh_lbl.setStyleSheet(
            f'color:{C_TEXT if status=="ready" else C_DIM};'
            f'font-size:14px; font-weight:bold; letter-spacing:1.4px; background:transparent;')
        en_lbl = QLabel(en)
        en_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:9px; letter-spacing:1.6px; background:transparent;')
        ttl_box.addWidget(zh_lbl)
        ttl_box.addWidget(en_lbl)
        head.addLayout(ttl_box)
        head.addStretch()

        status_text = '已开通' if status == 'ready' else '敬请期待'
        status_fg = C_OK if status == 'ready' else C_DIM
        status_bg = '#0d2818' if status == 'ready' else C_BG
        status_lbl = QLabel(f'  ●  {status_text}  ')
        status_lbl.setStyleSheet(
            f'background:{status_bg}; color:{status_fg};'
            f'border:1px solid {status_fg}; border-radius:9px;'
            f'padding:2px 8px; font-size:10px; font-weight:bold;')
        head.addWidget(status_lbl)
        lay.addLayout(head)

        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:11px; line-height:1.5;'
            f'background:transparent;')
        desc_lbl.setWordWrap(True)
        lay.addWidget(desc_lbl, 1)

    def mousePressEvent(self, evt):
        if self.status == 'ready' and self.on_click is not None and evt.button() == Qt.LeftButton:
            self.on_click()
        super().mousePressEvent(evt)


class ToolboxDialog(_OverlayDialog):
    """7 功能入口的工业风菜单。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=700)
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('⚙  系统工具箱  ·  TOOLBOX',
                            'AUXILIARY FUNCTIONS  ·  MINDROOM EDGE-AI',
                            self.accept))

        body = QFrame()
        body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 22, 28, 18)
        body_lay.setSpacing(16)

        hint = QLabel('选择需要打开的辅助功能。每个功能均为独立面板，关闭后回到主监测界面。')
        hint.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:12px; letter-spacing:1px;'
            f'background:transparent;')
        body_lay.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(16)

        cards = [
            ('🛡', '守护人响应', 'GUARDIAN RESPONSE',
             'AI 主动预警机制：检测到持续异常 → 自动通知家长/班主任/心理老师。学生无需主动求助，系统自动守护。',
             C_OK, 'ready', self._open_guardian),

            ('🔍', '实时风险溯源', 'RISK TRACE',
             'NN 决策可解释性：当前风险由哪些模态贡献、置信度多少、主因是什么。第一性原理溯源，透明可信。',
             C_VISION, 'ready', self._open_report),

            ('⚡', '实时干预建议', 'LIVE INTERVENTION',
             'AI 主动生成：检测到风险 → 自动推送呼吸训练/放松音乐/休息提醒。无需学生点击，系统主动关怀。',
             C_HEART, 'ready', self._open_intervention),

            ('📊', '多模态信号质量', 'SIGNAL QUALITY',
             '实时监控三路信号：视觉（人脸置信度、光照）、生理（PPG质量、RR有效率）、脑电（信号强度、电极接触）。',
             C_BRAIN, 'ready', self._open_signal_quality),

            ('📡', '设备连接管理', 'DEVICE HUB',
             '查看 TGAM 脑电、ESP32 手环、摄像头连接状态与固件版本，支持一键重连与校准。',
             C_ACCENT, 'ready', self._open_device_hub),

            ('🎯', '个体基线标定', 'BASELINE CALIBRATION',
             '90 秒静息采集，建立个体化 HR / HRV / 专注度基线，提升后续 NN 推理精度（纯工程优化）。',
             C_ACCENT2, 'ready', self._open_baseline),

            ('⏪', '多模态回放', 'TIMELINE REPLAY',
             '回放近 10 分钟视觉/生理/脑电时序，定位关键时刻并支持标记导出，用于复盘与优化。',
             C_RISK_MED, 'ready', self._open_replay),
        ]

        for idx, (ic, zh, en, desc, color, st, cb) in enumerate(cards):
            r, c = divmod(idx, 3)
            card = _ToolCard(ic, zh, en, desc, accent=color,
                             status=st, on_click=cb, parent=body)
            grid.addWidget(card, r, c)

        body_lay.addLayout(grid)
        body_lay.addStretch()
        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(
            '⚠  本系统为 AI 辅助筛查工具，不替代专业人员诊断。功能用于辅助监测与转介，'
            '最终判断与处置请由专业人员完成。'))

    def _open_guardian(self):
        dlg = GuardianResponseDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_report(self):
        from ui_toolbox_extras import AIAssistedReportDialog
        dlg = AIAssistedReportDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_intervention(self):
        from ui_toolbox_extras import LiveInterventionDialog
        dlg = LiveInterventionDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_signal_quality(self):
        from ui_toolbox_extras import SignalQualityDialog
        dlg = SignalQualityDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_device_hub(self):
        from ui_toolbox_extras import DeviceHubDialog
        dlg = DeviceHubDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_baseline(self):
        from ui_toolbox_extras import BaselineCalibrationDialog
        dlg = BaselineCalibrationDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()

    def _open_replay(self):
        from ui_toolbox_extras import TimelineReplayDialog
        dlg = TimelineReplayDialog(self.parent())
        self.hide(); dlg.exec_(); self.show()


# ===================== 守护人响应面板 =====================

# 演示用预设守护人（实际项目可在「设备/联系人设置」里编辑）
DEFAULT_GUARDIANS = [
    {
        'role': '家长 · PARENT',
        'name': '张妈妈',
        'phone': '138-0000-0001',
        'channels': ['短信', '电话'],
        'priority': 'P1',
        'color': C_OK,
        'icon': '👩',
    },
    {
        'role': '班主任 · TEACHER',
        'name': '李老师',
        'phone': '138-0000-0002',
        'channels': ['短信', 'App 推送'],
        'priority': 'P2',
        'color': C_ACCENT,
        'icon': '🧑‍🏫',
    },
    {
        'role': '心理老师 · COUNSELOR',
        'name': '王老师',
        'phone': '138-0000-0003',
        'channels': ['App 推送'],
        'priority': 'P3',
        'color': C_ACCENT2,
        'icon': '🧑‍⚕️',
    },
]

# 预警阶梯（演示规则）
ALERT_TIERS = [
    {
        'tier': 'L1',
        'name': '观察',
        'cn': '本地记录',
        'trigger': '风险 ≥ 0.6 持续 30 分钟',
        'action': '系统本地记录一次事件，不通知任何人',
        'targets': '—',
        'color': C_OK,
    },
    {
        'tier': 'L2',
        'name': '提醒',
        'cn': '通知家长',
        'trigger': '风险 ≥ 0.6 持续 2 小时',
        'action': '向家长发短信，附简短关心话术，不暴露原始数据',
        'targets': '家长 (P1)',
        'color': C_WARN,
    },
    {
        'tier': 'L3',
        'name': '升级',
        'cn': '家长 + 班主任',
        'trigger': '风险 ≥ 0.6 持续 4 小时 或 单次 ≥ 0.8',
        'action': '同时通知家长与班主任，附介入剧本链接',
        'targets': '家长 (P1) + 班主任 (P2)',
        'color': C_RISK_HIGH,
    },
    {
        'tier': 'L4',
        'name': '紧急',
        'cn': '全员 + 120',
        'trigger': '风险 ≥ 0.9 或检测到自伤模式',
        'action': '通知全部守护人，附医疗急救号码 120 与心理援助 12320',
        'targets': '家长 + 班主任 + 心理老师 + 紧急通道',
        'color': C_RISK_CRIT,
    },
]

# 介入剧本（不要 vs 建议）
INTERVENTION_PLAYBOOK = [
    ('直接问"你是不是抑郁了"', '"我注意到你最近有点累，能跟我聊聊吗"'),
    ('急着给建议、讲道理或类比"我以前也这样"', '先安静倾听，承认对方的感受'),
    ('命令式："你必须去看医生"', '"我陪你一起去找心理老师好不好"'),
    ('在他人面前讨论 / 转述给亲戚朋友', '选择私密、不被打扰的时间和空间'),
    ('"想开点 / 想多了 / 别钻牛角尖"', '"不管怎样，我都在你这边"'),
    ('强迫立刻回答或表态', '允许对方"先不说"，下次再聊'),
]

# 演示用响应历史（评审能看到完整闭环）
DEMO_HISTORY = [
    ('2026-05-15 22:18', 'L2', '家长', '已查看 / 已沟通', 0.63),
    ('2026-05-15 14:20', 'L1', '—', '系统记录', 0.61),
    ('2026-05-10 09:42', 'L3', '家长 + 班主任', '已联合介入', 0.82),
    ('2026-05-04 19:55', 'L2', '家长', '已查看', 0.65),
]


def _chip(text, fg, bg):
    lbl = QLabel(f'  {text}  ')
    lbl.setStyleSheet(
        f'background:{bg}; color:{fg};'
        f'border:1px solid {fg}; border-radius:9px;'
        f'padding:2px 6px; font-size:10px; font-weight:bold; letter-spacing:0.6px;')
    return lbl


class _GuardianCard(QFrame):
    """守护人列表项。"""

    def __init__(self, info, parent=None):
        super().__init__(parent)
        accent = info['color']
        self.setObjectName('GuardianCard')
        self.setStyleSheet(
            f'#GuardianCard {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {accent}; border-radius:8px; }}')
        self.setFixedHeight(86)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(14)

        ic = QLabel(info['icon'])
        ic.setStyleSheet(f'font-size:30px; background:transparent;')
        ic.setFixedWidth(46)
        ic.setAlignment(Qt.AlignCenter)
        lay.addWidget(ic)

        info_box = QVBoxLayout()
        info_box.setSpacing(2)
        role_lbl = QLabel(info['role'])
        role_lbl.setStyleSheet(
            f'color:{accent}; font-size:10px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        name_lbl = QLabel(f'{info["name"]}  ·  {info["phone"]}')
        name_lbl.setStyleSheet(
            f'color:{C_TEXT}; font-size:14px; font-weight:bold;'
            f'background:transparent;')
        info_box.addWidget(role_lbl)
        info_box.addWidget(name_lbl)
        lay.addLayout(info_box, 1)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        for ch in info['channels']:
            chips.addWidget(_chip(ch, C_ACCENT, '#06243a'))
        chips.addWidget(_chip(info['priority'], accent, C_BG))
        lay.addLayout(chips)


class _AlertTierRow(QFrame):
    """预警阶梯单行。"""

    def __init__(self, tier_info, parent=None):
        super().__init__(parent)
        accent = tier_info['color']
        self.setObjectName('TierRow')
        self.setStyleSheet(
            f'#TierRow {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {accent}; border-radius:8px; }}')
        self.setFixedHeight(78)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(14)

        tier_box = QVBoxLayout(); tier_box.setSpacing(0)
        t = QLabel(tier_info['tier'])
        t.setStyleSheet(
            f'color:{accent}; font-size:22px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        n = QLabel(tier_info['name'])
        n.setStyleSheet(
            f'color:{accent}; font-size:11px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        tier_box.addWidget(t); tier_box.addWidget(n)
        tier_wrap = QWidget(); tier_wrap.setLayout(tier_box)
        tier_wrap.setFixedWidth(72)
        lay.addWidget(tier_wrap)

        mid = QVBoxLayout(); mid.setSpacing(2)
        trg = QLabel(f'触发：{tier_info["trigger"]}')
        trg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; font-weight:bold;'
            f'background:transparent;')
        act = QLabel(f'动作：{tier_info["action"]}')
        act.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:11px; background:transparent;')
        act.setWordWrap(True)
        mid.addWidget(trg); mid.addWidget(act)
        lay.addLayout(mid, 1)

        tgt = QLabel(tier_info['targets'])
        tgt.setStyleSheet(
            f'color:{accent}; font-size:11px; font-weight:bold;'
            f'background:transparent;')
        tgt.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tgt.setFixedWidth(220)
        lay.addWidget(tgt)


class GuardianResponseDialog(_OverlayDialog):
    """守护人响应面板 — 预警系统的人侧闭环。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1180, card_h=820)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('🛡  守护人响应  ·  GUARDIAN RESPONSE',
                            'AI-INITIATED EARLY INTERVENTION  ·  NO STUDENT TRIGGER REQUIRED',
                            self.accept, back_text='←  返回工具箱', accent=C_OK))

        # 顶部强调横条
        notice = QFrame()
        notice.setStyleSheet(
            f'background:#0d2818; border-left:4px solid {C_OK};')
        notice.setFixedHeight(64)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        icon = QLabel('🛡')
        icon.setStyleSheet(
            f'color:{C_OK}; font-size:26px; background:transparent;')
        n_lay.addWidget(icon)
        msg = QLabel(
            '本系统的核心机制：AI 持续监测多模态信号 → 检测到持续异常 → '
            '主动通知预设守护人，由人来温和介入。\n学生本人不需要触发任何按钮，'
            '也无需"主动求救"。')
        msg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        msg.setWordWrap(True)
        n_lay.addWidget(msg, 1)
        self.card_lay.addWidget(notice)

        # 主体可滚动
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea {{ background:{C_BG_DEEP}; border:none; }}')
        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 18, 28, 18)
        body_lay.setSpacing(14)

        # === Section A：守护人名单 ===
        body_lay.addLayout(_section_header(
            '守护人名单', 'GUARDIAN ROSTER  ·  who responds', C_OK))
        for g in DEFAULT_GUARDIANS:
            body_lay.addWidget(_GuardianCard(g))
        add_hint = QLabel(
            '＋ 添加 / 编辑守护人  （演示模式：使用上述预设。正式版可在「设备管理」中绑定真实联系人）')
        add_hint.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; font-style:italic;'
            f'padding:4px 6px; background:transparent;')
        body_lay.addWidget(add_hint)

        body_lay.addSpacing(6)

        # === Section B：预警阶梯 ===
        body_lay.addLayout(_section_header(
            '预警阶梯规则', 'ALERT ESCALATION LADDER  ·  when & how', C_ACCENT))
        for t in ALERT_TIERS:
            body_lay.addWidget(_AlertTierRow(t))

        body_lay.addSpacing(6)

        # === Section C：介入剧本 ===
        body_lay.addLayout(_section_header(
            '守护人介入剧本', 'INTERVENTION PLAYBOOK  ·  guide for guardians', C_VISION))
        pb_box = QFrame()
        pb_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        pb_lay = QGridLayout(pb_box)
        pb_lay.setContentsMargins(18, 14, 18, 14)
        pb_lay.setHorizontalSpacing(16)
        pb_lay.setVerticalSpacing(8)

        h1 = QLabel('✗  不建议')
        h1.setStyleSheet(
            f'color:{C_RISK}; font-size:12px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        h2 = QLabel('✓  建议这样')
        h2.setStyleSheet(
            f'color:{C_OK}; font-size:12px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        pb_lay.addWidget(h1, 0, 0)
        pb_lay.addWidget(h2, 0, 1)

        for i, (bad, good) in enumerate(INTERVENTION_PLAYBOOK):
            b = QLabel(f'• {bad}')
            b.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:11px; line-height:1.6;'
                f'background:transparent;')
            b.setWordWrap(True)
            g = QLabel(f'• {good}')
            g.setStyleSheet(
                f'color:{C_TEXT}; font-size:11px; line-height:1.6;'
                f'background:transparent;')
            g.setWordWrap(True)
            pb_lay.addWidget(b, i + 1, 0)
            pb_lay.addWidget(g, i + 1, 1)
        body_lay.addWidget(pb_box)

        body_lay.addSpacing(6)

        # === Section D：学生悄悄说说明 ===
        body_lay.addLayout(_section_header(
            '学生侧 · 悄悄说通道', 'STUDENT QUIET CHANNEL  ·  low-barrier input', C_BRAIN))
        qc_box = QFrame()
        qc_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        qc_lay = QVBoxLayout(qc_box)
        qc_lay.setContentsMargins(18, 14, 18, 14)
        qc_lay.setSpacing(6)
        for line in [
            '· 主面板「AI 干预终端」标题旁已开放「💬 悄悄说」入口，学生可文字或心情勾选。',
            '· 学生写的原文仅 AI 用于趋势分析，守护人收到的是「最近状态偏低」的提示，不会看到原文。',
            '· 即使学生从不写任何内容，AI 也会通过多模态生理 / 行为信号自主预警，不依赖主动输入。',
        ]:
            lbl = QLabel(line)
            lbl.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:11px; line-height:1.6;'
                f'background:transparent;')
            lbl.setWordWrap(True)
            qc_lay.addWidget(lbl)
        body_lay.addWidget(qc_box)

        body_lay.addSpacing(6)

        # === Section E：响应历史 + 模拟按钮 ===
        body_lay.addLayout(_section_header(
            '预警响应历史', 'RESPONSE TIMELINE  ·  closed-loop tracking', C_RISK_MED))
        hist_box = QFrame()
        hist_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        hist_lay = QGridLayout(hist_box)
        hist_lay.setContentsMargins(18, 12, 18, 12)
        hist_lay.setHorizontalSpacing(14)
        hist_lay.setVerticalSpacing(6)

        cols = ['时间', '等级', '通知对象', '响应状态', '当时风险']
        for c, name in enumerate(cols):
            h = QLabel(name)
            h.setStyleSheet(
                f'color:{C_DIM}; font-size:10px; font-weight:bold;'
                f'letter-spacing:1.4px; background:transparent;')
            hist_lay.addWidget(h, 0, c)
        for r, (ts, tier, who, resp, risk) in enumerate(DEMO_HISTORY, start=1):
            tier_color = {'L1': C_OK, 'L2': C_WARN,
                          'L3': C_RISK_HIGH, 'L4': C_RISK_CRIT}.get(tier, C_DIM)
            for c, val in enumerate([ts, tier, who, resp, f'{risk:.2f}']):
                cell = QLabel(str(val))
                if c == 1:
                    cell.setStyleSheet(
                        f'color:{tier_color}; font-size:12px; font-weight:bold;'
                        f'font-family:Consolas; background:transparent;')
                elif c == 4:
                    cell.setStyleSheet(
                        f'color:{tier_color}; font-size:12px; font-weight:bold;'
                        f'font-family:Consolas; background:transparent;')
                else:
                    cell.setStyleSheet(
                        f'color:{C_TEXT}; font-size:11px; background:transparent;')
                hist_lay.addWidget(cell, r, c)
        body_lay.addWidget(hist_box)

        # 模拟当前预警按钮 + 日志区
        sim_row = QHBoxLayout()
        sim_row.setSpacing(10)
        self._sim_btn = QPushButton('▶  模拟发送当前预警')
        self._sim_btn.setCursor(Qt.PointingHandCursor)
        self._sim_btn.setFixedHeight(40)
        self._sim_btn.setStyleSheet(
            f'QPushButton {{ background:{C_OK}; color:{C_BG_DEEP};'
            f'border:1px solid {C_OK}; border-radius:8px; padding:6px 22px;'
            f'font-weight:bold; font-size:13px; letter-spacing:2px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{C_OK}; }}')
        self._sim_btn.clicked.connect(self._simulate_alert)
        sim_row.addWidget(self._sim_btn)

        hint = QLabel('  演示模式：模拟根据当前实时风险触发一次预警链路（实际版接入短信网关 / App 推送）。')
        hint.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; font-style:italic; background:transparent;')
        sim_row.addWidget(hint, 1)
        body_lay.addLayout(sim_row)

        self._sim_log = QTextEdit()
        self._sim_log.setReadOnly(True)
        self._sim_log.setFixedHeight(140)
        self._sim_log.setStyleSheet(
            f'QTextEdit {{ background:{C_BG_DEEP}; color:{C_OK};'
            f'border:1px solid {C_BORDER}; border-radius:6px;'
            f'font-family:Consolas; font-size:11px; padding:8px; }}')
        self._sim_log.setText('> 等待触发预警链路演示...')
        body_lay.addWidget(self._sim_log)

        body_lay.addStretch()
        scroll.setWidget(body)
        self.card_lay.addWidget(scroll, 1)

        self.card_lay.addWidget(_make_disclaimer_bar())

    # ----------- 演示链路（改为 NN 投票判定，论文 3.3.5 节）-----------

    def _read_nn_state(self):
        """从 RiskBus 取 (risk_score, alert_tier_idx, tier_probs, modal_w, modal_factor_text)。"""
        try:
            from risk_bus import RiskBus
            bus = RiskBus.instance()
            latest = bus.latest()
            if latest is None:
                return 0.0, 0, [0.25, 0.25, 0.25, 0.25], (0.35, 0.40, 0.25), '—', []
            alert_tier_idx = bus.decide_alert_tier(3)
            risk = float(latest.get('risk_score', 0.0))
            probs = latest.get('tier_probs', [0.25] * 4)
            mw = latest.get('modal_w', (0.35, 0.40, 0.25))
            names = ('vision', 'hrv', 'eeg')
            names_zh = ('视觉', 'HRV', '脑电')
            top = max(range(3), key=lambda i: mw[i])
            factor = f'{names_zh[top]} 贡献 {mw[top] * 100:.0f}%  ·  {bus.get_modal_factor(names[top])}'
            window = bus.tier_window(3)
            return risk, alert_tier_idx, probs, mw, factor, window
        except Exception:
            return 0.0, 0, [0.25, 0.25, 0.25, 0.25], (0.35, 0.40, 0.25), '—', []

    def _simulate_alert(self):
        self._sim_btn.setEnabled(False)
        self._sim_log.clear()
        risk, alert_idx, probs, mw, factor_text, window = self._read_nn_state()
        tier = ALERT_TIERS[alert_idx]

        # 把 tier_window 渲染成 "[L1, L2, L3]" 这样的可读串
        labels = ['L1', 'L2', 'L3', 'L4']
        window_str = '[' + ', '.join(labels[t] if 0 <= t < 4 else '?' for t in window) + ']' if window else '[尚无足够样本]'
        conf = probs[alert_idx] * 100 if 0 <= alert_idx < 4 else 0.0

        steps = [
            (0,   f'> [T+0.0s] NN 推理：综合风险 {risk:.2f}  ·  4 档概率 [{" ".join(f"{p:.2f}" for p in probs)}]'),
            (300, f'> [T+0.3s] 连续 3 次 tier 窗口 {window_str}  →  投票判定 = {tier["tier"]} · {tier["name"]}'),
            (700, f'> [T+0.7s] NN 主因模态：{factor_text}'),
            (1000, f'> [T+1.0s] 触发条件：{tier["trigger"]}（NN 置信度 {conf:.0f}%）'),
        ]

        targets = []
        if tier['tier'] in ('L2', 'L3', 'L4'):
            targets.append(DEFAULT_GUARDIANS[0])
        if tier['tier'] in ('L3', 'L4'):
            targets.append(DEFAULT_GUARDIANS[1])
        if tier['tier'] == 'L4':
            targets.append(DEFAULT_GUARDIANS[2])

        delay = 1400
        for g in targets:
            channels = ' + '.join(g['channels'])
            steps.append((delay, f'> [T+{delay/1000:.1f}s] 正在通过 {channels} 通知 {g["role"].split("·")[0].strip()} {g["name"]} ({g["phone"]})...'))
            delay += 700
            preview = '您好，AI 守护系统注意到孩子近期状态需要关注，' \
                      '建议今晚找个轻松的时机聊聊。详情请打开家长端 App。'
            steps.append((delay, f'> [T+{delay/1000:.1f}s] ✓ 已送达 {g["name"]}  ·  内容预览："{preview[:32]}..."'))
            delay += 600

        if tier['tier'] == 'L4':
            steps.append((delay, f'> [T+{delay/1000:.1f}s] ⚠ 已附带紧急医疗号码 120 与心理援助 12320'))
            delay += 600
        elif tier['tier'] == 'L1':
            steps.append((delay, f'> [T+{delay/1000:.1f}s] L1 等级仅本地记录，未通知任何人'))
            delay += 600

        steps.append((delay, f'> [T+{delay/1000:.1f}s] 预警已记录至响应历史，等待守护人回执'))
        steps.append((delay + 800, '> [完成] 守护人响应链路演示结束'))

        self._scheduled = []
        for at, text in steps:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda txt=text: self._sim_log.append(txt))
            t.start(at)
            self._scheduled.append(t)

        # 主面板 ai_console 同步广播一行
        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                win.ai_console.append(
                    f'> [守护人响应] 已模拟触发 {tier["tier"]} · {tier["name"]}，'
                    f'通知 {len(targets)} 位守护人')
        except Exception:
            pass

        end_timer = QTimer(self)
        end_timer.setSingleShot(True)
        end_timer.timeout.connect(lambda: self._sim_btn.setEnabled(True))
        end_timer.start(delay + 1200)
        self._scheduled.append(end_timer)


# ===================== 学生侧：悄悄说 =====================

WHISPER_LOG_PATH = os.path.join(os.path.dirname(__file__), 'whisper_log.jsonl')

MOOD_OPTIONS = [
    ('😊', '还好'),
    ('😐', '平淡'),
    ('😔', '低落'),
    ('😢', '想哭'),
    ('😴', '累'),
    ('😡', '烦'),
    ('😰', '紧张'),
    ('😟', '孤单'),
]


class _MoodChip(QPushButton):
    def __init__(self, emoji, label, parent=None):
        super().__init__(parent)
        self.emoji = emoji
        self.label = label
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setText(f'{emoji}  {label}')
        self.setFixedHeight(40)
        self.setMinimumWidth(96)
        self.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL}; color:{C_TEXT_DIM};'
            f'border:1px solid {C_BORDER}; border-radius:20px;'
            f'padding:6px 14px; font-size:13px; font-weight:bold;'
            f'letter-spacing:1px; }}'
            f'QPushButton:hover {{ border:1px solid {C_ACCENT};'
            f'color:{C_TEXT}; }}'
            f'QPushButton:checked {{ background:{C_ACCENT};'
            f'color:{C_BG_DEEP}; border:1px solid {C_ACCENT}; }}')


class QuietWhisperDialog(_OverlayDialog):
    """学生侧低门槛输入入口。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=720, card_h=640)
        self._parent_window = parent
        self._chips = []
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('💬  悄悄说  ·  QUIET CHANNEL',
                            'A SAFE PLACE TO PUT YOUR MOOD',
                            self.accept, back_text='✕ 关闭', accent=C_BRAIN))

        body = QFrame(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(32, 22, 32, 22)
        body_lay.setSpacing(14)

        greet = QLabel('今天怎么样？')
        greet.setStyleSheet(
            f'color:{C_TEXT}; font-size:22px; font-weight:bold;'
            f'letter-spacing:2px; background:transparent;')
        body_lay.addWidget(greet)

        intro = QLabel(
            '想说点什么都可以，也可以什么都不说。\n'
            '你写的原文只被 AI 用来了解你最近的状态趋势，'
            '不会被家长 / 老师原文看到。')
        intro.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.7;'
            f'background:transparent;')
        intro.setWordWrap(True)
        body_lay.addWidget(intro)

        # 心情勾选（多选）
        mood_lbl = QLabel('心情  ·  可以多选，也可以不选')
        mood_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        body_lay.addWidget(mood_lbl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, (emoji, label) in enumerate(MOOD_OPTIONS):
            chip = _MoodChip(emoji, label)
            self._chips.append(chip)
            grid.addWidget(chip, i // 4, i % 4)
        body_lay.addLayout(grid)

        # 文字框
        wr_lbl = QLabel('想写点什么  ·  完全可选')
        wr_lbl.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')
        body_lay.addWidget(wr_lbl)

        self._editor = QTextEdit()
        self._editor.setPlaceholderText('比如：今天有点累，懒得说话…')
        self._editor.setFixedHeight(120)
        self._editor.setStyleSheet(
            f'QTextEdit {{ background:{C_PANEL}; color:{C_TEXT};'
            f'border:1px solid {C_BORDER}; border-radius:8px;'
            f'padding:10px; font-family:"Microsoft YaHei","PingFang SC",sans-serif;'
            f'font-size:13px; line-height:1.6; }}'
            f'QTextEdit:focus {{ border:1px solid {C_ACCENT}; }}')
        body_lay.addWidget(self._editor)

        # 匿名 + 按钮
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self._chk_anon = QCheckBox('匿名提交  ·  不绑定我的账号')
        self._chk_anon.setStyleSheet(
            f'QCheckBox {{ color:{C_TEXT_DIM}; font-size:11px; spacing:6px; }}'
            f'QCheckBox::indicator {{ width:14px; height:14px; }}')
        bottom.addWidget(self._chk_anon)
        bottom.addStretch()

        btn_cancel = QPushButton('✕  取消')
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setStyleSheet(
            f'QPushButton {{ background:transparent; color:{C_DIM};'
            f'border:1px solid {C_BORDER}; border-radius:6px;'
            f'padding:6px 18px; font-weight:bold; }}'
            f'QPushButton:hover {{ color:{C_RISK}; border:1px solid {C_RISK}; }}')
        btn_cancel.clicked.connect(self.accept)
        bottom.addWidget(btn_cancel)

        btn_save = QPushButton('✓  悄悄保存')
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setStyleSheet(
            f'QPushButton {{ background:{C_BRAIN}; color:{C_BG_DEEP};'
            f'border:1px solid {C_BRAIN}; border-radius:6px;'
            f'padding:6px 22px; font-weight:bold; letter-spacing:1px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{C_BRAIN}; }}')
        btn_save.clicked.connect(self._save)
        bottom.addWidget(btn_save)
        body_lay.addLayout(bottom)

        body_lay.addStretch()
        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(
            '🕊  这是你和系统之间的悄悄话。如果 AI 发现你最近状态持续不好，'
            '会通知你预设的守护人来关心你。'))

    def _save(self):
        moods = [(c.emoji, c.label) for c in self._chips if c.isChecked()]
        text = self._editor.toPlainText().strip()
        anon = self._chk_anon.isChecked()

        if not moods and not text:
            QMessageBox.information(
                self, '什么都没写哦',
                '看起来你今天没什么想说的，这很正常。\n你可以随时再回来悄悄说一句。')
            return

        record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'anonymous': anon,
            'moods': moods,
            'text': text,
        }
        try:
            with open(WHISPER_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as exc:
            QMessageBox.warning(
                self, '保存失败', f'无法写入日志：{exc}')
            return

        # 主面板 ai_console 提示已记录（不显示原文，只显示已收到）
        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                mood_str = ' '.join(e for e, _ in moods) if moods else '（未选心情）'
                win.ai_console.append(
                    f'> [悄悄说] 已收到一条心声  ·  {mood_str}  ·  AI 会纳入趋势分析')
        except Exception:
            pass

        QMessageBox.information(
            self, '已悄悄收到',
            '已悄悄保存。\n\n如果你想，可以随时再回来写。\n'
            '系统会默默陪着你。')
        self.accept()


# ===================== 便捷入口 =====================

def open_toolbox(parent=None):
    dlg = ToolboxDialog(parent)
    return dlg.exec_()


def open_guardian_response(parent=None):
    dlg = GuardianResponseDialog(parent)
    return dlg.exec_()


def open_quiet_whisper(parent=None):
    dlg = QuietWhisperDialog(parent)
    return dlg.exec_()


# 向后兼容
HelpChannelDialog = GuardianResponseDialog
open_help_channel = open_guardian_response
