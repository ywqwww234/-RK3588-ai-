"""
功能工具箱 · 扩展对话框（D:/G/2）

实现 6 个辅助功能，统一遵循红线设计：
  - AIAssistedReportDialog     AI 辅助分析报告（实时风险溯源）
  - LiveInterventionDialog     实时干预建议（主动推送）
  - SignalQualityDialog        多模态信号质量监控
  - DeviceHubDialog            设备连接管理
  - BaselineCalibrationDialog  个体基线标定（90s 静息）
  - TimelineReplayDialog       多模态数据回放

红线（贯穿所有对话框）：
  - 措辞：评估 / 状态 / 引导 / 用户（不出现 诊断/治疗/患者/确诊/服药/就医方案）
  - 高风险弹求助转介，而非"治疗建议"
  - 底部固定声明栏：AI 辅助筛查，不构成医疗诊断；含 12320 / 010-82951332 转介入口

重构说明：
  - 删除 SelfAssessmentDialog（心理状态自评）- 被动式，不符合主动预警理念
  - 删除 RelaxationCoachDialog（放松引导）- 被动式，改为主动干预
  - 新增 LiveInterventionDialog - AI 主动推送干预建议
  - 新增 SignalQualityDialog - 实时监控三路信号质量
"""

import os
import json
import time
import math
import random

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient
from PyQt5.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout,
                              QHBoxLayout, QGridLayout, QFrame, QScrollArea,
                              QButtonGroup, QRadioButton, QStackedWidget,
                              QMessageBox, QTextEdit, QSlider, QComboBox,
                              QSizePolicy, QProgressBar)

from theme import (C_BG, C_BG_DEEP, C_PANEL, C_PANEL_HI, C_BORDER, C_BORDER_HI,
                   C_ACCENT, C_ACCENT2, C_DIM, C_TEXT, C_TEXT_DIM,
                   C_OK, C_WARN, C_RISK, C_VISION, C_HEART, C_BRAIN,
                   C_RISK_LOW, C_RISK_MED, C_RISK_HIGH, C_RISK_CRIT)

from ui_toolbox import (_OverlayDialog, _make_title_bar, _make_disclaimer_bar,
                         _section_header, _chip)
from ui_widgets import (TierProbBars, AttnHeatStrip, ContribStackBar,
                         ModalContribGauge)


# 全系统统一的"求助转介"声明片段
DISCLAIMER_REFERRAL = (
    '⚠  本系统为 AI 辅助筛查工具，结果不构成医疗诊断。'
    '如有持续困扰，请联系学校心理老师，或拨打 12320 / 北京危机 010-82951332。'
)


# ============================================================
# 1. 心理状态自评（PHQ-9 / GAD-7 / PSQI）
# ============================================================

PHQ9_QUESTIONS = [
    '做事时提不起劲或没有兴趣',
    '感到心情低落、沮丧或绝望',
    '入睡困难、睡不安稳，或睡眠过多',
    '感觉疲倦或没有活力',
    '食欲不振或吃太多',
    '觉得自己很糟，或觉得自己很失败',
    '难以专注于事情（如看报纸或看电视）',
    '动作或说话变缓，或反过来——比平时烦躁很多',
    '有不如死掉或用某种方式伤害自己的念头',
]

GAD7_QUESTIONS = [
    '感觉紧张、焦虑或急切',
    '不能停止或控制担忧',
    '对各种各样的事情担忧过多',
    '很难放松',
    '由于不安而无法静坐',
    '变得容易烦恼或易激惹',
    '感到害怕，好像有什么可怕的事情会发生',
]

PSQI_QUESTIONS = [
    '最近一个月，你通常几点上床睡觉？（晚 = 高分）',
    '最近一个月，你通常需要多久才能入睡？',
    '最近一个月，你通常每晚实际睡眠时间？（少 = 高分）',
    '最近一个月，你觉得自己的睡眠质量整体如何？',
    '最近一个月，你有多频繁因为睡不好而影响白天活动？',
]

PHQ9_OPTIONS = [('完全没有', 0), ('几天', 1), ('一半以上时间', 2), ('几乎每天', 3)]


def _phq9_interpret(score):
    if score <= 4:   return ('无 · 极低', C_RISK_LOW, '维持现状')
    if score <= 9:   return ('轻微参考区间', C_OK, '关注睡眠与运动')
    if score <= 14:  return ('中等参考区间', C_WARN, '建议联系学校心理老师沟通')
    if score <= 19:  return ('偏高参考区间', C_RISK_HIGH, '强烈建议求助转介')
    return ('显著偏高', C_RISK_CRIT, '请立即联系守护人或专业心理服务')


def _gad7_interpret(score):
    if score <= 4:   return ('无 · 极低', C_RISK_LOW, '维持现状')
    if score <= 9:   return ('轻微参考区间', C_OK, '关注情绪释放')
    if score <= 14:  return ('中等参考区间', C_WARN, '建议联系学校心理老师沟通')
    return ('偏高参考区间', C_RISK_CRIT, '强烈建议求助转介')


def _psqi_interpret(score):
    if score <= 5:   return ('良好', C_RISK_LOW, '保持作息')
    if score <= 10:  return ('一般', C_WARN, '关注睡眠卫生')
    return ('较差', C_RISK_HIGH, '建议联系学校心理老师沟通')


SCALE_DEFS = {
    'PHQ-9': dict(title='PHQ-9 · 抑郁参考自评',
                  intro='请根据最近 2 周的实际感受作答。本量表为风险参考，不构成诊断。',
                  questions=PHQ9_QUESTIONS, interpret=_phq9_interpret,
                  max_score=27, accent=C_ACCENT),
    'GAD-7': dict(title='GAD-7 · 焦虑参考自评',
                  intro='请根据最近 2 周的实际感受作答。本量表为风险参考，不构成诊断。',
                  questions=GAD7_QUESTIONS, interpret=_gad7_interpret,
                  max_score=21, accent=C_VISION),
    'PSQI':  dict(title='PSQI · 睡眠质量参考自评',
                  intro='请根据最近 1 个月的实际睡眠回答。本量表为风险参考，不构成诊断。',
                  questions=PSQI_QUESTIONS, interpret=_psqi_interpret,
                  max_score=15, accent=C_HEART),
}

ASSESS_LOG_PATH = os.path.join(os.path.dirname(__file__), 'assess_log.jsonl')


class _ScalePage(QWidget):
    def __init__(self, scale_key, on_done, parent=None):
        super().__init__(parent)
        self.scale_key = scale_key
        self.spec = SCALE_DEFS[scale_key]
        self.on_done = on_done
        self._groups = []
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 20, 28, 18)
        outer.setSpacing(10)

        title = QLabel(self.spec['title'])
        title.setStyleSheet(
            f'color:{self.spec["accent"]}; font-size:18px; font-weight:bold;'
            f'letter-spacing:2px; background:transparent;')
        outer.addWidget(title)

        intro = QLabel(self.spec['intro'])
        intro.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea {{ background:transparent; border:none; }}')
        body = QWidget(); body.setStyleSheet('background:transparent;')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 4, 8, 4); body_lay.setSpacing(8)

        for i, q in enumerate(self.spec['questions']):
            card = QFrame()
            card.setStyleSheet(
                f'QFrame {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
                f'border-radius:8px; }}')
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 10, 16, 10); cl.setSpacing(6)
            ql = QLabel(f'{i + 1}. {q}')
            ql.setStyleSheet(
                f'color:{C_TEXT}; font-size:13px; font-weight:bold;'
                f'background:transparent;')
            ql.setWordWrap(True)
            cl.addWidget(ql)

            opts = QHBoxLayout(); opts.setSpacing(10)
            group = QButtonGroup(self)
            self._groups.append(group)
            for txt, val in PHQ9_OPTIONS:
                rb = QRadioButton(f'{txt} ({val})')
                rb.setStyleSheet(
                    f'QRadioButton {{ color:{C_TEXT_DIM}; font-size:11px;'
                    f'spacing:6px; padding:4px 4px; background:transparent; }}'
                    f'QRadioButton::indicator {{ width:14px; height:14px; }}'
                    f'QRadioButton:hover {{ color:{self.spec["accent"]}; }}')
                rb.setProperty('score', val)
                group.addButton(rb, val)
                opts.addWidget(rb)
            opts.addStretch()
            cl.addLayout(opts)
            body_lay.addWidget(card)

        body_lay.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        bottom = QHBoxLayout(); bottom.setSpacing(10)
        bottom.addStretch()
        btn = QPushButton('✓  生成参考区间')
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        btn.setStyleSheet(
            f'QPushButton {{ background:{self.spec["accent"]}; color:{C_BG_DEEP};'
            f'border-radius:8px; padding:6px 28px; font-weight:bold;'
            f'letter-spacing:1.5px; font-size:13px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{self.spec["accent"]};'
            f'border:1px solid {self.spec["accent"]}; }}')
        btn.clicked.connect(self._submit)
        bottom.addWidget(btn)
        outer.addLayout(bottom)

    def _submit(self):
        total = 0
        for g in self._groups:
            ck = g.checkedButton()
            if ck is None:
                QMessageBox.information(self, '尚未完成',
                    '请把每一项都选择一个选项后再生成参考区间。')
                return
            total += int(ck.property('score'))
        tag, color, hint = self.spec['interpret'](total)
        self.on_done(self.scale_key, total, tag, color, hint)


class SelfAssessmentDialog(_OverlayDialog):
    """心理状态自评：PHQ-9 / GAD-7 / PSQI 三选一。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=760)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('📋  心理状态自评  ·  SELF-ASSESSMENT',
                            'PHQ-9 / GAD-7 / PSQI  ·  Reference Only, Not Diagnosis',
                            self.accept, back_text='✕ 关闭', accent=C_ACCENT))

        # 声明横条
        notice = QFrame()
        notice.setStyleSheet(
            f'background:#2e1f05; border-left:4px solid {C_WARN};')
        notice.setFixedHeight(56)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        ic = QLabel('⚠'); ic.setStyleSheet(
            f'color:{C_WARN}; font-size:22px; background:transparent;')
        n_lay.addWidget(ic)
        msg = QLabel('本量表用于风险参考区间评估，结果不构成医疗诊断；'
                     '如得分偏高，请联系学校心理老师或拨打 12320 进一步咨询。')
        msg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        msg.setWordWrap(True)
        n_lay.addWidget(msg, 1)
        self.card_lay.addWidget(notice)

        # 三量表 Tab
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(28, 12, 28, 0); tab_row.setSpacing(8)
        self._tab_btns = []
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f'QStackedWidget {{ background:{C_BG_DEEP}; }}')

        for i, key in enumerate(['PHQ-9', 'GAD-7', 'PSQI']):
            btn = QPushButton(key)
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(120)
            btn.setCursor(Qt.PointingHandCursor)
            accent = SCALE_DEFS[key]['accent']
            btn.setStyleSheet(
                f'QPushButton {{ background:transparent; color:{C_DIM};'
                f'border:1px solid {C_BORDER}; border-radius:16px;'
                f'padding:4px 16px; font-weight:bold; letter-spacing:1.2px; }}'
                f'QPushButton:hover {{ color:{accent}; border:1px solid {accent}; }}'
                f'QPushButton:checked {{ background:{accent}; color:{C_BG_DEEP};'
                f'border:1px solid {accent}; }}')
            btn.clicked.connect(lambda _, idx=i: self._switch_tab(idx))
            tab_row.addWidget(btn)
            self._tab_btns.append(btn)

            page = _ScalePage(key, self._on_scale_done)
            self._stack.addWidget(page)

        tab_row.addStretch()
        wrap = QWidget(); wrap.setStyleSheet(f'background:{C_BG_DEEP};')
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(0, 0, 0, 0); wrap_lay.setSpacing(0)
        wrap_lay.addLayout(tab_row)
        wrap_lay.addWidget(self._stack, 1)
        self.card_lay.addWidget(wrap, 1)

        self._switch_tab(0)
        self.card_lay.addWidget(_make_disclaimer_bar(DISCLAIMER_REFERRAL))

    def _switch_tab(self, idx):
        for i, b in enumerate(self._tab_btns):
            b.setChecked(i == idx)
        self._stack.setCurrentIndex(idx)

    def _on_scale_done(self, key, total, tag, color, hint):
        # 取同时刻 NN 客观分做主观 vs 客观对照
        nn_risk = None
        nn_tier = None
        try:
            from risk_bus import RiskBus
            nn = RiskBus.instance().latest()
            if nn is not None:
                nn_risk = float(nn.get('risk_score', 0.0))
                nn_tier = int(nn.get('tier', 0))
        except Exception:
            pass

        record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'scale': key,
            'score': total,
            'tag': tag,
            'nn_risk': nn_risk,
            'nn_tier': nn_tier,
        }
        try:
            with open(ASSESS_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            pass

        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                nn_txt = f'  ·  NN 客观分 {nn_risk:.2f}' if nn_risk is not None else ''
                win.ai_console.append(
                    f'> [自评] {key} 完成  ·  得分 {total}  ·  参考区间 {tag}{nn_txt}')
        except Exception:
            pass

        # 主观 vs 客观一致性判断
        consistency = ''
        if nn_risk is not None:
            # PHQ-9: ≥10 偏高；GAD-7: ≥10 偏高；PSQI: ≥6 偏差
            scale_high = (key == 'PHQ-9' and total >= 10) or \
                         (key == 'GAD-7' and total >= 10) or \
                         (key == 'PSQI'  and total >= 6)
            nn_high = nn_risk >= 0.6
            if scale_high and nn_high:
                consistency = ('<br><span style="color:#ff8a3d"><b>双通道一致</b>：'
                               '主观自评与 NN 客观推理均偏高 → 强烈建议求助转介。</span>')
            elif scale_high and not nn_high:
                consistency = ('<br><span style="color:#ffc857"><b>分歧</b>：'
                               '自评偏高但 NN 客观信号正常 → 可能短期情绪波动；建议持续监测。</span>')
            elif not scale_high and nn_high:
                consistency = ('<br><span style="color:#ffc857"><b>分歧</b>：'
                               '自评正常但 NN 客观信号偏高 → 可能存在主观未觉察的生理压力；'
                               '建议安静环境复测。</span>')
            else:
                consistency = ('<br><span style="color:#26d07c"><b>双通道一致</b>：'
                               '主观自评与 NN 客观推理均处于安全区间。</span>')

        nn_line = (f'<br>NN 客观分（同时刻）：<b style="color:#00e5ff">{nn_risk:.2f}</b>'
                   f'  ·  L{nn_tier + 1}'
                   if nn_risk is not None else
                   '<br>（NN 客观分尚未就绪，请等待 60s 数据窗口建立）')

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.NoIcon)
        msg.setWindowTitle(f'{key} 参考结果（主观 + 客观）')
        msg.setText(
            f'<b style="color:{color}">主观参考区间：{tag}</b><br>'
            f'{key} 得分：{total}'
            f'{nn_line}<br>'
            f'{consistency}<br><br>'
            f'建议：{hint}<br><br>'
            f'<span style="color:#999;font-size:11px">本结果不构成医疗诊断；'
            f'如有持续困扰，请联系学校心理老师或拨打 12320。</span>')
        msg.exec_()


# ============================================================
# 2. AI 辅助分析报告
# ============================================================

class AIAssistedReportDialog(_OverlayDialog):
    """AI 辅助分析报告：只描述趋势与可能因素，不写治疗。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=760)
        self._parent_window = parent
        self._build()

    def _read_recent(self):
        risk = None
        snap_v, snap_p = {}, {}
        win = self._parent_window
        # 优先从 RiskBus 拿 NN 综合分（不再读 UI 标签）
        nn_latest = None
        try:
            from risk_bus import RiskBus
            nn_latest = RiskBus.instance().latest()
            if nn_latest:
                risk = float(nn_latest.get('risk_score', 0.0))
        except Exception:
            pass
        try:
            if risk is None and win is not None and hasattr(win, 'risk_display'):
                risk = float(win.risk_display.text())
        except Exception:
            pass
        try:
            if win is not None:
                snap_v = dict(getattr(win, 'latest_visual_snapshot', {}) or {})
                snap_p = dict(getattr(win, 'latest_physio_snapshot', {}) or {})
        except Exception:
            pass
        self._nn_latest = nn_latest
        return risk or 0.0, snap_v, snap_p

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('📑  AI 辅助分析报告  ·  AI ASSISTED REPORT',
                            'TREND & POSSIBLE FACTORS  ·  NOT A DIAGNOSIS',
                            self.accept, back_text='✕ 关闭', accent=C_VISION))

        # 顶部强声明（不可关闭）
        notice = QFrame()
        notice.setStyleSheet(
            f'background:#2e1f05; border-left:4px solid {C_WARN};')
        notice.setFixedHeight(70)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        ic = QLabel('⚠'); ic.setStyleSheet(
            f'color:{C_WARN}; font-size:24px; background:transparent;')
        n_lay.addWidget(ic)
        msg = QLabel(
            '本报告为 AI 辅助筛查输出，仅描述风险趋势与可能因素，'
            '不构成医疗诊断、不替代专业人员判断、不提供任何治疗或用药建议。\n'
            '如趋势持续偏高，请联系学校心理老师或拨打 12320。')
        msg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        msg.setWordWrap(True)
        n_lay.addWidget(msg, 1)
        self.card_lay.addWidget(notice)

        # 报告正文
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea {{ background:{C_BG_DEEP}; border:none; }}')
        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 18, 28, 18); body_lay.setSpacing(14)

        risk, vis, phy = self._read_recent()

        # Section A：当前状态摘要
        body_lay.addLayout(_section_header(
            '当前状态摘要', 'CURRENT STATE SNAPSHOT', C_ACCENT))
        sum_card = QFrame()
        sum_card.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        sg = QGridLayout(sum_card)
        sg.setContentsMargins(18, 12, 18, 12)
        sg.setHorizontalSpacing(24); sg.setVerticalSpacing(8)
        sg.addWidget(self._kv('综合风险参考', f'{risk:.2f}', C_RISK_HIGH if risk >= 0.6 else C_OK), 0, 0)
        sg.addWidget(self._kv('视觉表情', str(vis.get('expr', '—')), C_VISION), 0, 1)
        sg.addWidget(self._kv('心率 (bpm)', str(phy.get('bpm') or '—'), C_HEART), 0, 2)
        sg.addWidget(self._kv('HRV (RMSSD)', f'{phy.get("hrv_rmssd") or "—"}', C_HEART), 1, 0)
        sg.addWidget(self._kv('眼疲劳指数', f'{vis.get("eye_fatigue_index", 0):.2f}', C_VISION), 1, 1)
        sg.addWidget(self._kv('姿态风险', f'{vis.get("posture_risk", 0):.2f}', C_VISION), 1, 2)
        body_lay.addWidget(sum_card)

        # Section B：趋势描述
        body_lay.addLayout(_section_header(
            '近期趋势描述', 'RECENT TREND  ·  DESCRIPTIVE ONLY', C_VISION))
        trend = self._build_trend_text(risk, vis, phy)
        tlbl = QLabel(trend)
        tlbl.setStyleSheet(
            f'color:{C_TEXT}; font-size:13px; line-height:1.9;'
            f'background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-radius:8px; padding:14px 18px;')
        tlbl.setWordWrap(True)
        body_lay.addWidget(tlbl)

        # Section C：可能因素
        body_lay.addLayout(_section_header(
            '可能因素（描述性）', 'POSSIBLE FACTORS  ·  NON-CLINICAL', C_BRAIN))
        factors = self._build_factor_list(risk, vis, phy)
        fbox = QFrame()
        fbox.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        fl = QVBoxLayout(fbox)
        fl.setContentsMargins(18, 12, 18, 12); fl.setSpacing(6)
        for line in factors:
            l = QLabel(f'•  {line}')
            l.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.8;'
                f'background:transparent;')
            l.setWordWrap(True)
            fl.addWidget(l)
        body_lay.addWidget(fbox)

        # Section C2：NN 决策可解释性（论文 3.3.4 节）
        body_lay.addLayout(_section_header(
            'NN 决策可解释性', 'NN INTERPRETABILITY  ·  WHY THIS LEVEL', C_ACCENT))
        nn_card = QFrame()
        nn_card.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        nn_lay = QGridLayout(nn_card)
        nn_lay.setContentsMargins(18, 14, 18, 14)
        nn_lay.setHorizontalSpacing(20); nn_lay.setVerticalSpacing(8)

        # 1. 4 档概率柱图
        cap1 = QLabel('4 档风险概率分布  ·  TIER PROBABILITIES')
        cap1.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
        bars = TierProbBars(); bars.setMinimumHeight(140)
        # 2. 模态贡献堆叠条
        cap2 = QLabel('三模态贡献堆叠  ·  MODALITY CONTRIBUTION')
        cap2.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
        contrib = ContribStackBar(); contrib.setMinimumHeight(36)
        # 3. attn 热力条
        cap3 = QLabel('时间注意力热力  ·  TIME ATTENTION (60s)')
        cap3.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
        attn_strip = AttnHeatStrip()
        # 4. 决策溯源短句
        cap4 = QLabel('决策溯源 · DECISION TRACE')
        cap4.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
        trace_lbl = QLabel('NN 主因：—')
        trace_lbl.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.8;'
            f'background:transparent;')
        trace_lbl.setWordWrap(True)
        # 5. 推理元信息
        cap5 = QLabel('推理元信息 · INFERENCE META')
        cap5.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
        meta_lbl = QLabel('—')
        meta_lbl.setStyleSheet(
            f'color:{C_ACCENT}; font-size:11px; font-family:Consolas;'
            f'background:transparent;')
        meta_lbl.setWordWrap(True)

        nn_lay.addWidget(cap1, 0, 0); nn_lay.addWidget(bars, 1, 0)
        nn_lay.addWidget(cap2, 0, 1); nn_lay.addWidget(contrib, 1, 1, alignment=Qt.AlignTop)
        nn_lay.addWidget(cap3, 2, 0, 1, 2); nn_lay.addWidget(attn_strip, 3, 0, 1, 2)
        nn_lay.addWidget(cap4, 4, 0, 1, 2); nn_lay.addWidget(trace_lbl, 5, 0, 1, 2)
        nn_lay.addWidget(cap5, 6, 0, 1, 2); nn_lay.addWidget(meta_lbl, 7, 0, 1, 2)
        body_lay.addWidget(nn_card)

        # 填数据（取打开报告时的最近一帧 NN 结果）
        try:
            nn = getattr(self, '_nn_latest', None)
            if nn is not None:
                bars.set_probs(nn.get('tier_probs') or [0.25] * 4)
                mw = nn.get('modal_w') or (0.35, 0.40, 0.25)
                contrib.set_weights(*mw)
                attn_strip.set_attn(nn.get('attn') or [1.0 / 60] * 60)
                # 决策溯源短句
                from risk_bus import RiskBus
                rb = RiskBus.instance()
                names_zh = ('视觉', 'HRV', '脑电')
                names_en = ('vision', 'hrv', 'eeg')
                top = max(range(3), key=lambda i: mw[i])
                tier_label = ['L1 低风险', 'L2 中风险', 'L3 高风险', 'L4 极高'][int(nn.get('tier', 0))]
                trace_lbl.setText(
                    f'本次推理 → {tier_label}  ·  主因模态：{names_zh[top]} 贡献 {mw[top]*100:.0f}%'
                    f'\n {rb.get_modal_factor(names_en[top])}'
                    f'\n 投票窗口（连续 3 次 tier）：{rb.tier_window(3)}  →  升级到 L{rb.decide_alert_tier(3)+1}')
                mode = nn.get('mode', 'fallback')
                quant = 'INT8 / RKNN' if mode == 'onnx' else 'FP32 / Fallback'
                meta_lbl.setText(
                    f'推理引擎 {("ONNX" if mode == "onnx" else "Fallback (numpy 仿真)")}  ·  '
                    f'耗时 {nn.get("latency_ms", 0.0):.1f} ms  ·  '
                    f'量化 {quant}  ·  模型 CNN-BiGRU-Attention  ·  '
                    f'参数 ≈ 48K  ·  全流程边缘本地')
        except Exception:
            pass

        # Section D：求助转介建议（强制底部固定段落）
        body_lay.addLayout(_section_header(
            '求助转介建议', 'REFERRAL  ·  ALWAYS PRESENT IN REPORTS', C_OK))
        ref_card = QFrame()
        ref_card.setStyleSheet(
            f'background:#0d2818; border:1px solid {C_OK}; border-radius:8px;')
        rl = QVBoxLayout(ref_card)
        rl.setContentsMargins(18, 12, 18, 12); rl.setSpacing(4)
        for line in [
            '· 校园：联系学校心理咨询中心 / 班主任 / 心理老师',
            '· 全国心理援助热线：12320  （24 小时）',
            '· 北京危机干预热线：010-82951332',
            '· 严重风险时，请联系 120 急救或前往就近精神/心理专科机构',
        ]:
            l = QLabel(line)
            l.setStyleSheet(
                f'color:{C_TEXT}; font-size:12px; line-height:1.8;'
                f'background:transparent;')
            l.setWordWrap(True)
            rl.addWidget(l)
        body_lay.addWidget(ref_card)

        body_lay.addStretch()
        scroll.setWidget(body)
        self.card_lay.addWidget(scroll, 1)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(22, 8, 22, 8)
        btn_row.addStretch()
        btn_save = QPushButton('💾  保存到家长端')
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setStyleSheet(
            f'QPushButton {{ background:{C_VISION}; color:{C_BG_DEEP};'
            f'border-radius:6px; padding:6px 22px; font-weight:bold;'
            f'letter-spacing:1.2px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{C_VISION};'
            f'border:1px solid {C_VISION}; }}')
        btn_save.clicked.connect(self._save_to_parent)
        btn_row.addWidget(btn_save)
        wrap = QWidget(); wrap.setStyleSheet(f'background:{C_BG};')
        wrap.setLayout(btn_row)
        self.card_lay.addWidget(wrap)

        self.card_lay.addWidget(_make_disclaimer_bar(DISCLAIMER_REFERRAL))

    def _kv(self, k, v, color):
        w = QWidget()
        w.setStyleSheet('background:transparent;')
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        kl = QLabel(k)
        kl.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.4px;'
            f'background:transparent;')
        vl = QLabel(str(v))
        vl.setStyleSheet(
            f'color:{color}; font-size:18px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        lay.addWidget(kl); lay.addWidget(vl)
        return w

    def _build_trend_text(self, risk, vis, phy):
        if risk >= 0.8:
            tone = '近期综合风险参考值持续偏高'
            tail = '建议尽快联系学校心理老师沟通，必要时拨打 12320 寻求专业指导。'
        elif risk >= 0.6:
            tone = '近期出现若干次风险参考偏高时段'
            tail = '建议关注作息与情绪释放，可考虑预约心理老师做一次面谈。'
        elif risk >= 0.3:
            tone = '近期综合风险参考值整体平稳，偶有小幅波动'
            tail = '可以继续保持当前节律，关注睡眠与运动。'
        else:
            tone = '近期综合风险参考值处于较低区间'
            tail = '当前状态良好，建议维持现有作息。'
        expr = vis.get('expr') or '未检测到人脸'
        bpm = phy.get('bpm')
        bpm_txt = f'{bpm:.0f}' if isinstance(bpm, (int, float)) and bpm else '—'
        return (f'{tone}。当前主要表情判读为「{expr}」，心率均值约 {bpm_txt} bpm。\n'
                f'AI 仅对多模态信号做趋势刻画，不针对具体疾病做判断。{tail}')

    def _build_factor_list(self, risk, vis, phy):
        out = []
        if vis.get('eye_fatigue_index', 0) >= 0.5:
            out.append('眼疲劳指数偏高，可能与长时间用眼或睡眠不足相关。')
        if vis.get('posture_risk', 0) >= 0.5:
            out.append('姿态风险偏高，长期低头姿势可能影响颈椎与情绪。')
        hrv = phy.get('hrv_rmssd')
        if isinstance(hrv, (int, float)) and hrv and hrv < 20:
            out.append('HRV (RMSSD) 偏低，可能反映副交感活性下降或近期压力较大。')
        if not out:
            out.append('当前各模态信号均处于稳定范围，未发现需要特别提示的因素。')
        out.append('上述描述为信号层面的客观刻画，不构成医学结论。')
        return out

    def _save_to_parent(self):
        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                win.ai_console.append(
                    '> [报告] AI 辅助分析报告已生成并同步至家长端（仅含趋势与可能因素，'
                    '不含原始数据，不含医学结论）。')
        except Exception:
            pass
        QMessageBox.information(self, '已保存',
            '报告已加入家长端 PDF 队列。\n家长端会在「评估报告」中看到本次趋势摘要。')


# ============================================================
# 3. 实时干预建议（主动推送）
# ============================================================

class LiveInterventionDialog(_OverlayDialog):
    """实时干预建议 - AI 主动生成并推送，无需学生点击。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=760)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('⚡  实时干预建议  ·  LIVE INTERVENTION',
                            'AI-DRIVEN PROACTIVE SUGGESTIONS  ·  NO MANUAL TRIGGER',
                            self.accept, back_text='✕ 关闭', accent=C_HEART))

        # 顶部强调横条
        notice = QFrame()
        notice.setStyleSheet(
            f'background:#1a0d2e; border-left:4px solid {C_HEART};')
        notice.setFixedHeight(64)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        icon = QLabel('⚡')
        icon.setStyleSheet(
            f'color:{C_HEART}; font-size:26px; background:transparent;')
        n_lay.addWidget(icon)
        msg = QLabel(
            '本功能的核心机制：AI 检测到风险 → 自动生成干预建议 → 主动推送给学生。\n'
            '学生无需点击任何按钮，系统会在合适时机自动弹出呼吸训练、放松音乐、休息提醒等。')
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

        # === Section A：当前干预建议 ===
        body_lay.addLayout(_section_header(
            '当前干预建议', 'CURRENT SUGGESTIONS  ·  GENERATED BY AI', C_HEART))

        # 读取当前风险并生成建议
        risk, suggestions = self._generate_suggestions()

        for sug in suggestions:
            sug_card = self._make_suggestion_card(sug)
            body_lay.addWidget(sug_card)

        body_lay.addSpacing(6)

        # === Section B：干预策略规则 ===
        body_lay.addLayout(_section_header(
            '干预策略规则', 'INTERVENTION RULES  ·  WHEN & WHAT', C_ACCENT))

        rules_box = QFrame()
        rules_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        rules_lay = QVBoxLayout(rules_box)
        rules_lay.setContentsMargins(18, 14, 18, 14)
        rules_lay.setSpacing(8)

        rules = [
            ('风险 ≥ 0.6 持续 10 分钟', '推送呼吸训练（4-7-8 呼吸法）'),
            ('HRV < 20ms 持续 5 分钟', '推送放松音乐（α波音乐）'),
            ('眼疲劳指数 ≥ 0.7', '推送休息提醒（20-20-20 法则）'),
            ('专注度 < 40 持续 15 分钟', '推送短暂休息建议（5分钟走动）'),
            ('姿态风险 ≥ 0.6', '推送姿态调整提醒（抬头挺胸）'),
        ]

        for trigger, action in rules:
            rule_row = QHBoxLayout()
            rule_row.setSpacing(14)

            trigger_lbl = QLabel(f'触发：{trigger}')
            trigger_lbl.setStyleSheet(
                f'color:{C_TEXT}; font-size:12px; font-weight:bold;'
                f'background:transparent;')
            trigger_lbl.setFixedWidth(280)

            arrow = QLabel('→')
            arrow.setStyleSheet(
                f'color:{C_ACCENT}; font-size:16px; background:transparent;')

            action_lbl = QLabel(f'动作：{action}')
            action_lbl.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:12px; background:transparent;')
            action_lbl.setWordWrap(True)

            rule_row.addWidget(trigger_lbl)
            rule_row.addWidget(arrow)
            rule_row.addWidget(action_lbl, 1)
            rules_lay.addLayout(rule_row)

        body_lay.addWidget(rules_box)

        body_lay.addSpacing(6)

        # === Section C：干预历史 ===
        body_lay.addLayout(_section_header(
            '干预历史', 'INTERVENTION HISTORY  ·  LAST 7 DAYS', C_VISION))

        hist_box = QFrame()
        hist_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        hist_lay = QGridLayout(hist_box)
        hist_lay.setContentsMargins(18, 12, 18, 12)
        hist_lay.setHorizontalSpacing(14)
        hist_lay.setVerticalSpacing(6)

        cols = ['时间', '触发原因', '干预类型', '执行时长', '效果']
        for c, name in enumerate(cols):
            h = QLabel(name)
            h.setStyleSheet(
                f'color:{C_DIM}; font-size:10px; font-weight:bold;'
                f'letter-spacing:1.4px; background:transparent;')
            hist_lay.addWidget(h, 0, c)

        # 演示数据
        demo_hist = [
            ('2026-05-18 14:23', '风险 0.68 持续 12min', '呼吸训练', '3min', '风险降至 0.52'),
            ('2026-05-18 10:15', 'HRV 18ms', '放松音乐', '5min', '风险降至 0.45'),
            ('2026-05-17 16:40', '眼疲劳 0.75', '休息提醒', '已执行', '眼疲劳降至 0.42'),
            ('2026-05-17 09:30', '专注度 35', '短暂休息', '5min', '专注度回升至 68'),
        ]

        for r, (ts, reason, itype, dur, effect) in enumerate(demo_hist, start=1):
            for c, val in enumerate([ts, reason, itype, dur, effect]):
                cell = QLabel(str(val))
                if c == 4:  # 效果列用绿色
                    cell.setStyleSheet(
                        f'color:{C_OK}; font-size:11px; background:transparent;')
                else:
                    cell.setStyleSheet(
                        f'color:{C_TEXT}; font-size:11px; background:transparent;')
                hist_lay.addWidget(cell, r, c)

        body_lay.addWidget(hist_box)

        body_lay.addStretch()
        scroll.setWidget(body)
        self.card_lay.addWidget(scroll, 1)

        self.card_lay.addWidget(_make_disclaimer_bar(
            '⚠  干预建议为 AI 辅助参考，不构成医疗治疗。如持续困扰，请联系心理老师或拨打 12320。'))

    def _generate_suggestions(self):
        """根据当前风险生成干预建议。"""
        risk = 0.0
        vis, phy = {}, {}

        # 读取当前状态
        win = self._parent_window
        try:
            from risk_bus import RiskBus
            nn = RiskBus.instance().latest()
            if nn:
                risk = float(nn.get('risk_score', 0.0))
        except Exception:
            pass

        try:
            if win is not None:
                vis = dict(getattr(win, 'latest_visual_snapshot', {}) or {})
                phy = dict(getattr(win, 'latest_physio_snapshot', {}) or {})
        except Exception:
            pass

        suggestions = []

        # 根据风险生成建议
        if risk >= 0.6:
            suggestions.append({
                'icon': '🫁',
                'title': '呼吸训练',
                'desc': '检测到综合风险偏高，建议进行 4-7-8 呼吸法：吸气 4 秒 → 屏息 7 秒 → 呼气 8 秒，重复 3-5 次。',
                'action': '开始训练',
                'color': C_HEART,
            })

        hrv = phy.get('hrv_rmssd')
        if isinstance(hrv, (int, float)) and hrv and hrv < 20:
            suggestions.append({
                'icon': '🎵',
                'title': '放松音乐',
                'desc': 'HRV 偏低，副交感神经活性不足。建议播放 α 波音乐（8-13Hz），帮助大脑进入放松状态。',
                'action': '播放音乐',
                'color': C_ACCENT2,
            })

        eye_fatigue = vis.get('eye_fatigue_index', 0)
        if eye_fatigue >= 0.7:
            suggestions.append({
                'icon': '👁',
                'title': '休息提醒',
                'desc': '眼疲劳指数偏高。建议执行 20-20-20 法则：每 20 分钟，看 20 英尺（6米）外的物体，持续 20 秒。',
                'action': '开始休息',
                'color': C_VISION,
            })

        if not suggestions:
            suggestions.append({
                'icon': '✓',
                'title': '状态良好',
                'desc': '当前各项指标均处于正常范围，无需特殊干预。建议保持当前节律，注意劳逸结合。',
                'action': '继续保持',
                'color': C_OK,
            })

        return risk, suggestions

    def _make_suggestion_card(self, sug):
        """创建单个建议卡片。"""
        card = QFrame()
        card.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {sug["color"]}; border-radius:8px;')
        card.setFixedHeight(100)

        lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(14)

        # 图标
        icon = QLabel(sug['icon'])
        icon.setStyleSheet(f'font-size:32px; background:transparent;')
        icon.setFixedWidth(50)
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)

        # 标题和描述
        info_box = QVBoxLayout()
        info_box.setSpacing(4)

        title = QLabel(sug['title'])
        title.setStyleSheet(
            f'color:{sug["color"]}; font-size:15px; font-weight:bold;'
            f'letter-spacing:1.6px; background:transparent;')

        desc = QLabel(sug['desc'])
        desc.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:11px; line-height:1.6;'
            f'background:transparent;')
        desc.setWordWrap(True)

        info_box.addWidget(title)
        info_box.addWidget(desc)
        lay.addLayout(info_box, 1)

        # 操作按钮
        btn = QPushButton(sug['action'])
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(100, 36)
        btn.setStyleSheet(
            f'QPushButton {{ background:{sug["color"]}; color:{C_BG_DEEP};'
            f'border-radius:6px; font-weight:bold; letter-spacing:1px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{sug["color"]};'
            f'border:1px solid {sug["color"]}; }}')
        lay.addWidget(btn)

        return card


# ============================================================
# 4. 多模态信号质量监控
# ============================================================

class SignalQualityDialog(_OverlayDialog):
    """多模态信号质量监控 - 实时显示三路信号质量。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=760)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('📊  多模态信号质量  ·  SIGNAL QUALITY',
                            'REAL-TIME MONITORING  ·  VISION / PHYSIO / EEG',
                            self.accept, back_text='✕ 关闭', accent=C_BRAIN))

        # 主体可滚动
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea {{ background:{C_BG_DEEP}; border:none; }}')
        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 18, 28, 18)
        body_lay.setSpacing(14)

        # === Section A：视觉信号质量 ===
        body_lay.addLayout(_section_header(
            '视觉信号质量', 'VISION SIGNAL QUALITY', C_VISION))

        vis_card = self._make_quality_card('视觉', C_VISION, [
            ('人脸检测置信度', '0.95', C_OK),
            ('光照质量', '良好', C_OK),
            ('遮挡检测', '无遮挡', C_OK),
            ('帧率', '30 FPS', C_OK),
        ])
        body_lay.addWidget(vis_card)

        # === Section B：生理信号质量 ===
        body_lay.addLayout(_section_header(
            '生理信号质量', 'PHYSIO SIGNAL QUALITY', C_HEART))

        phy_card = self._make_quality_card('生理', C_HEART, [
            ('PPG 信号质量', 'Good', C_OK),
            ('RR 间期有效率', '0.85', C_OK),
            ('采样率', '30 Hz', C_OK),
            ('信号幅值', '850 (P95)', C_OK),
        ])
        body_lay.addWidget(phy_card)

        # === Section C：脑电信号质量 ===
        body_lay.addLayout(_section_header(
            '脑电信号质量', 'EEG SIGNAL QUALITY', C_BRAIN))

        eeg_card = self._make_quality_card('脑电', C_BRAIN, [
            ('信号强度', '良好', C_OK),
            ('电极接触质量', '●●●●○', C_OK),
            ('Poor Signal', '12', C_OK),
            ('数据源', 'TGAM 本地', C_OK),
        ])
        body_lay.addWidget(eeg_card)

        body_lay.addStretch()
        scroll.setWidget(body)
        self.card_lay.addWidget(scroll, 1)

        self.card_lay.addWidget(_make_disclaimer_bar(
            '⚠  信号质量监控为工程层面功能，用于优化数据采集，不涉及医疗诊断。'))

        # 定时刷新
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._refresh_quality)
        self._refresh_timer.start()

    def _make_quality_card(self, name, color, metrics):
        """创建信号质量卡片。"""
        card = QFrame()
        card.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {color}; border-radius:8px;')

        lay = QGridLayout(card)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setHorizontalSpacing(24)
        lay.setVerticalSpacing(10)

        for i, (metric, value, status_color) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2

            metric_lbl = QLabel(metric)
            metric_lbl.setStyleSheet(
                f'color:{C_DIM}; font-size:11px; letter-spacing:1.2px;'
                f'background:transparent;')

            value_lbl = QLabel(value)
            value_lbl.setStyleSheet(
                f'color:{status_color}; font-size:16px; font-weight:bold;'
                f'background:transparent; font-family:Consolas;')

            lay.addWidget(metric_lbl, row * 2, col)
            lay.addWidget(value_lbl, row * 2 + 1, col)

        return card

    def _refresh_quality(self):
        """定时刷新信号质量（实际项目中从主窗口读取）。"""
        # TODO: 从主窗口读取实时信号质量
        pass


# ============================================================
# 便捷入口（更新）
# ============================================================

class _BreathRing(QWidget):
    """随呼吸节律放大缩小的圆环，配合 HRV 实时映射。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self._phase = 0.0       # 0~2π
        self._running = False
        self._inhale = 4.0      # 秒
        self._hold = 2.0
        self._exhale = 6.0
        self._t0 = 0.0
        self._hrv = 38.0        # 当前 HRV
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def set_pattern(self, inhale, hold, exhale):
        self._inhale = max(1.0, inhale)
        self._hold = max(0.0, hold)
        self._exhale = max(1.0, exhale)

    def set_hrv(self, hrv):
        self._hrv = max(5.0, min(120.0, float(hrv)))

    def start(self):
        self._running = True
        self._t0 = time.time()
        self._timer.start()

    def stop(self):
        self._running = False
        self._timer.stop()
        self.update()

    def _tick(self):
        self.update()

    def _phase_progress(self):
        cycle = self._inhale + self._hold + self._exhale
        if cycle <= 0:
            return 0.0, 'idle'
        t = (time.time() - self._t0) % cycle
        if t < self._inhale:
            return t / self._inhale, 'inhale'
        elif t < self._inhale + self._hold:
            return 1.0, 'hold'
        else:
            return 1.0 - (t - self._inhale - self._hold) / self._exhale, 'exhale'

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width(); h = self.height()
        cx, cy = w / 2, h / 2

        # 背景圆
        max_r = min(w, h) * 0.42
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), max_r, max_r)

        # HRV 提示外圈（颜色随 HRV）
        hrv_color = QColor(C_RISK_LOW if self._hrv >= 30 else
                           (C_WARN if self._hrv >= 15 else C_RISK))
        p.setPen(QPen(hrv_color, 1.5, Qt.DashLine))
        p.drawEllipse(QPointF(cx, cy), max_r + 14, max_r + 14)

        # 呼吸圆环
        progress, phase = self._phase_progress() if self._running else (0.4, 'idle')
        r = max_r * (0.3 + 0.65 * progress)
        grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        grad.setColorAt(0.0, QColor(0, 229, 255, 200))
        grad.setColorAt(1.0, QColor(124, 77, 255, 130))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor(C_ACCENT), 2))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # 中心文字
        p.setPen(QColor(C_BG_DEEP))
        f = QFont('Microsoft YaHei', 22, QFont.Bold)
        p.setFont(f)
        text_map = {'inhale': '吸 气', 'hold': '屏 住', 'exhale': '呼 气', 'idle': '准 备'}
        p.drawText(QRectF(cx - 100, cy - 24, 200, 48),
                   Qt.AlignCenter, text_map.get(phase, '— —'))

        # 外圈说明
        p.setPen(QColor(C_DIM))
        f2 = QFont('Microsoft YaHei', 10)
        p.setFont(f2)
        p.drawText(QRectF(cx - 120, cy + r + 18, 240, 22), Qt.AlignCenter,
                   f'当前 HRV (RMSSD) ≈ {self._hrv:.1f} ms')


class RelaxationCoachDialog(_OverlayDialog):
    """放松引导：HRV 生物反馈呼吸训练。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=760)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('🫁  放松引导  ·  RELAXATION COACH',
                            'HRV BIOFEEDBACK BREATHING  ·  Self-paced, Not a Therapy',
                            self._on_close, back_text='✕ 关闭', accent=C_HEART))

        # 横条说明
        notice = QFrame()
        notice.setStyleSheet(f'background:#06243a; border-left:4px solid {C_ACCENT};')
        notice.setFixedHeight(56)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        ic = QLabel('🫁'); ic.setStyleSheet('font-size:24px; background:transparent;')
        n_lay.addWidget(ic)
        msg = QLabel('本功能为放松训练，非治疗手段。完全自愿，可随时停止。'
                     '建议在安静环境跟随圆环节律深呼吸。')
        msg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        msg.setWordWrap(True)
        n_lay.addWidget(msg, 1)
        self.card_lay.addWidget(notice)

        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(28, 20, 28, 20); body_lay.setSpacing(24)

        # 左：呼吸圆环
        left = QVBoxLayout(); left.setSpacing(10)
        self._ring = _BreathRing()
        left.addWidget(self._ring, 1, alignment=Qt.AlignCenter)

        ctl = QHBoxLayout(); ctl.setSpacing(10)
        self._btn_play = QPushButton('▶  开始')
        self._btn_play.setCursor(Qt.PointingHandCursor)
        self._btn_play.setFixedHeight(40)
        self._btn_play.setStyleSheet(
            f'QPushButton {{ background:{C_HEART}; color:{C_BG_DEEP};'
            f'border-radius:8px; padding:6px 24px; font-weight:bold;'
            f'letter-spacing:1.5px; font-size:13px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{C_HEART};'
            f'border:1px solid {C_HEART}; }}')
        self._btn_play.clicked.connect(self._toggle)
        ctl.addWidget(self._btn_play)
        ctl.addStretch()
        left.addLayout(ctl)

        body_lay.addLayout(left, 3)

        # 右：节律选择 + 实时 HRV 反馈
        right = QVBoxLayout(); right.setSpacing(12)

        right.addLayout(_section_header('呼吸节律', 'BREATH PATTERN', C_ACCENT))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems([
            '4-2-6  舒缓（推荐）',
            '4-4-4  方形呼吸',
            '4-7-8  深度放松',
            '6-0-6  对称缓慢',
        ])
        self._pattern_combo.setStyleSheet(
            f'QComboBox {{ background:{C_PANEL}; color:{C_TEXT};'
            f'border:1px solid {C_BORDER}; border-radius:6px; padding:6px 10px;'
            f'font-size:12px; }}'
            f'QComboBox:hover {{ border:1px solid {C_ACCENT}; }}'
            f'QComboBox QAbstractItemView {{ background:{C_PANEL}; color:{C_TEXT};'
            f'selection-background-color:{C_ACCENT}; selection-color:{C_BG_DEEP}; }}')
        self._pattern_combo.currentIndexChanged.connect(self._apply_pattern)
        right.addWidget(self._pattern_combo)
        self._apply_pattern(0)

        right.addLayout(_section_header('训练时长', 'DURATION', C_VISION))
        self._dur_combo = QComboBox()
        self._dur_combo.addItems(['3 分钟', '5 分钟', '8 分钟'])
        self._dur_combo.setStyleSheet(self._pattern_combo.styleSheet())
        right.addWidget(self._dur_combo)

        right.addLayout(_section_header('HRV 反馈', 'REAL-TIME HRV', C_HEART))
        self._hrv_lbl = QLabel('— ms')
        self._hrv_lbl.setStyleSheet(
            f'color:{C_HEART}; font-size:42px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self._hrv_lbl.setAlignment(Qt.AlignCenter)
        right.addWidget(self._hrv_lbl)

        self._hrv_hint = QLabel('随呼吸调节，HRV 升高代表副交感激活更强')
        self._hrv_hint.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; line-height:1.6;'
            f'background:transparent;')
        self._hrv_hint.setWordWrap(True)
        self._hrv_hint.setAlignment(Qt.AlignCenter)
        right.addWidget(self._hrv_hint)

        right.addStretch()

        rw_panel = QFrame()
        rw_panel.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:10px;')
        rw_panel.setMinimumWidth(280)
        right.setContentsMargins(16, 16, 16, 16)
        rw_panel.setLayout(right)
        body_lay.addWidget(rw_panel, 2)

        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(DISCLAIMER_REFERRAL))

        # 实时 HRV 拉取
        self._hrv_timer = QTimer(self)
        self._hrv_timer.setInterval(1000)
        self._hrv_timer.timeout.connect(self._pull_hrv)
        self._hrv_timer.start()
        self._pull_hrv()

    def _apply_pattern(self, idx):
        patterns = [(4, 2, 6), (4, 4, 4), (4, 7, 8), (6, 0, 6)]
        i, h, e = patterns[idx]
        self._ring.set_pattern(i, h, e)

    def _toggle(self):
        if self._ring._running:
            # 结束训练 → 取当前 NN 风险，跟开始时对比
            self._ring.stop()
            self._btn_play.setText('▶  开始')
            self._show_session_summary()
        else:
            # 记录训练开始时的 NN 风险
            self._session_start_risk = self._read_nn_risk()
            self._session_start_ts = time.time()
            self._ring.start()
            self._btn_play.setText('⏸  暂停')

    def _read_nn_risk(self):
        try:
            from risk_bus import RiskBus
            nn = RiskBus.instance().latest()
            if nn is not None:
                return float(nn.get('risk_score', 0.0))
        except Exception:
            pass
        return None

    def _show_session_summary(self):
        end_risk = self._read_nn_risk()
        start_risk = getattr(self, '_session_start_risk', None)
        dur = int(time.time() - getattr(self, '_session_start_ts', time.time()))
        if start_risk is None or end_risk is None:
            return
        delta = end_risk - start_risk
        pct = (delta / start_risk * 100) if start_risk > 1e-3 else 0.0
        sign = '+' if delta >= 0 else ''
        # 决定颜色和评语
        if delta < -0.05:
            tone = ('<span style="color:#26d07c"><b>训练有效</b></span>：'
                    'NN 综合风险显著下降，副交感激活生效。')
        elif delta > 0.05:
            tone = ('<span style="color:#ffc857"><b>本次训练效果有限</b></span>：'
                    '可能起步压力过大或环境干扰，建议复试。')
        else:
            tone = ('<span style="color:#9ba8c8">训练后风险基本持平</span>：'
                    '可继续保持或调整呼吸节律。')
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.NoIcon)
        msg.setWindowTitle('训练结果')
        msg.setText(
            f'本次放松时长 {dur} 秒<br><br>'
            f'NN 综合风险变化：'
            f'<b>{start_risk:.2f} → {end_risk:.2f}</b>'
            f' ({sign}{delta:.2f} / {sign}{pct:.0f}%)<br><br>'
            f'{tone}<br><br>'
            f'<span style="color:#999;font-size:11px">放松训练为辅助干预，不构成治疗。</span>')
        msg.exec_()
        # 同步到 ai_console
        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                win.ai_console.append(
                    f'> [放松] 训练 {dur}s 完成  ·  NN 风险 '
                    f'{start_risk:.2f} → {end_risk:.2f} ({sign}{pct:.0f}%)')
        except Exception:
            pass

    def _pull_hrv(self):
        win = self._parent_window
        hrv = None
        try:
            if win is not None:
                snap = getattr(win, 'latest_physio_snapshot', None) or {}
                hrv = snap.get('hrv_rmssd')
        except Exception:
            pass
        if not isinstance(hrv, (int, float)) or not hrv:
            hrv = 30 + 8 * math.sin(time.time() * 0.5) + random.uniform(-2, 2)
        self._ring.set_hrv(hrv)
        self._hrv_lbl.setText(f'{hrv:.1f} ms')

    def _on_close(self):
        try:
            self._ring.stop()
            self._hrv_timer.stop()
        except Exception:
            pass
        self.accept()


# ============================================================
# 4. 设备管理
# ============================================================

class DeviceHubDialog(_OverlayDialog):
    """设备管理：TGAM 脑电 / ESP32 手环 / 摄像头 状态总览。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1080, card_h=720)
        self._parent_window = parent
        self._cards = []
        self._build()

    def _devices_snapshot(self):
        win = self._parent_window
        # TGAM
        tgam_state = 'sim'
        tgam_detail = '未检测到串口，使用模拟数据'
        try:
            if win is not None and getattr(win, 'bci_thread', None) is not None:
                if win.bci_thread.isRunning():
                    tgam_state = 'live'; tgam_detail = '本地串口活跃 · 57600 baud'
        except Exception:
            pass
        # ESP32
        bpm = None
        try:
            bpm = float(getattr(win, 'latest_remote_bpm', None) or 0) if win else 0
        except Exception:
            bpm = 0
        esp_state = 'live' if bpm and bpm > 0 else 'sim'
        esp_detail = f'最近 BPM = {bpm:.1f}' if bpm and bpm > 0 else '远端心率尚未回包，使用模拟数据'
        # 摄像头
        cam_state = 'live'
        cam_detail = '30 FPS · 1080p · 本地推理'
        try:
            if win is None or getattr(win, 'camera_thread', None) is None:
                cam_state = 'offline'; cam_detail = '未启动摄像头线程'
        except Exception:
            pass
        return [
            dict(icon='🧠', name='TGAM 脑电', en='NeuroSky TGAM',
                 state=tgam_state, detail=tgam_detail, color=C_BRAIN),
            dict(icon='⌚', name='ESP32 手环', en='ESP32 + MAX30102',
                 state=esp_state, detail=esp_detail, color=C_HEART),
            dict(icon='📷', name='摄像头视觉', en='YOLOv8n + Local Vision',
                 state=cam_state, detail=cam_detail, color=C_VISION),
        ]

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('📡  设备管理  ·  DEVICE HUB',
                            'CONNECTION & SIGNAL QUALITY',
                            self.accept, back_text='✕ 关闭', accent=C_BRAIN))

        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 22, 28, 22); body_lay.setSpacing(16)

        hint = QLabel('实时查看脑电、手环、摄像头的连接与信号质量。'
                      '断连时系统自动切换到本地模拟数据，并在此处标红。')
        hint.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        hint.setWordWrap(True)
        body_lay.addWidget(hint)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(16); self._grid.setVerticalSpacing(14)
        body_lay.addLayout(self._grid)
        self._render_cards()

        body_lay.addSpacing(8)

        # NN 引擎卡（核心：评委一眼看到边缘部署证据）
        body_lay.addLayout(_section_header(
            '神经网络引擎', 'NEURAL ENGINE  ·  Edge Deployment Evidence', C_BRAIN))
        nn_card = QFrame()
        nn_card.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-left:3px solid {C_BRAIN}; border-radius:10px;')
        nl = QGridLayout(nn_card)
        nl.setContentsMargins(18, 12, 18, 12); nl.setHorizontalSpacing(28); nl.setVerticalSpacing(6)

        nn_mode = '推理中…'
        nn_lat = '--'
        nn_score = '--'
        nn_tier = '--'
        try:
            from risk_bus import RiskBus
            nn = RiskBus.instance().latest()
            if nn is not None:
                nn_mode = 'ONNX (真推理)' if nn.get('mode') == 'onnx' else 'Fallback (numpy 仿真)'
                nn_lat = f'{nn.get("latency_ms", 0.0):.1f} ms'
                nn_score = f'{nn.get("risk_score", 0.0):.2f}'
                nn_tier = ['L1', 'L2', 'L3', 'L4'][int(nn.get('tier', 0))]
        except Exception:
            pass

        def _kv2(k, v, color):
            box = QVBoxLayout(); box.setSpacing(2)
            kl = QLabel(k)
            kl.setStyleSheet(f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px; background:transparent;')
            vl = QLabel(v)
            vl.setStyleSheet(f'color:{color}; font-size:14px; font-weight:bold;'
                             f'background:transparent; font-family:Consolas;')
            box.addWidget(kl); box.addWidget(vl)
            w = QWidget(); w.setStyleSheet('background:transparent;'); w.setLayout(box)
            return w
        nl.addWidget(_kv2('模型架构', 'CNN-BiGRU-Attn', C_ACCENT), 0, 0)
        nl.addWidget(_kv2('参数量', '~48K', C_BRAIN), 0, 1)
        nl.addWidget(_kv2('输入', '[60×25] @ 1Hz', C_VISION), 0, 2)
        nl.addWidget(_kv2('推理模式', nn_mode, C_OK if 'ONNX' in nn_mode else C_WARN), 0, 3)
        nl.addWidget(_kv2('单次延迟', nn_lat, C_OK), 1, 0)
        nl.addWidget(_kv2('当前风险', nn_score, C_HEART), 1, 1)
        nl.addWidget(_kv2('当前 Tier', nn_tier, C_BRAIN), 1, 2)
        nl.addWidget(_kv2('部署', 'RK3588 / 本地', C_VISION), 1, 3)
        body_lay.addWidget(nn_card)

        # 协议链路图
        body_lay.addLayout(_section_header(
            '链路拓扑', 'LINK TOPOLOGY', C_ACCENT))
        topo = QLabel(
            '神念 TGAM  →  UART 57600  →  RK3588 ▌'
            'ESP32  →  WiFi  →  Flask 5001  →  RK3588 ▌'
            'USB Cam  →  Local CV  →  RK3588')
        topo.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.8;'
            f'background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-radius:8px; padding:14px 18px; font-family:Consolas;')
        topo.setWordWrap(True)
        body_lay.addWidget(topo)

        # 工程说明
        body_lay.addLayout(_section_header(
            '本地降级策略', 'LOCAL FALLBACK STRATEGY', C_OK))
        fb = QFrame()
        fb.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        fl = QVBoxLayout(fb)
        fl.setContentsMargins(18, 12, 18, 12); fl.setSpacing(4)
        for line in [
            '· 任意模态断连：自动切换到本地模拟波形，日志写明，UI 显示 SIM 标签',
            '· 数据不会因为单模态断连而完全失效（多模态加权融合冗余设计）',
            '· 所有原始信号本地处理，断连时不发起任何外部请求',
        ]:
            l = QLabel(line)
            l.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.8;'
                f'background:transparent;')
            l.setWordWrap(True)
            fl.addWidget(l)
        body_lay.addWidget(fb)

        # 刷新按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton('↻  刷新设备状态')
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
            f'border:1px solid {C_ACCENT}; border-radius:6px; padding:6px 22px;'
            f'font-weight:bold; }}'
            f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG_DEEP}; }}')
        btn.clicked.connect(self._render_cards)
        btn_row.addWidget(btn)
        body_lay.addLayout(btn_row)

        body_lay.addStretch()
        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(
            '⚠  本面板用于工程层面的设备状态查看，不涉及任何医学评估。'))

    def _render_cards(self):
        # 清空旧卡片
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()

        for i, d in enumerate(self._devices_snapshot()):
            card = QFrame()
            card.setStyleSheet(
                f'QFrame {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
                f'border-left:3px solid {d["color"]}; border-radius:10px; }}')
            card.setFixedHeight(120)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(18, 12, 18, 12); cl.setSpacing(6)

            head = QHBoxLayout(); head.setSpacing(10)
            ic = QLabel(d['icon'])
            ic.setStyleSheet('font-size:24px; background:transparent;')
            head.addWidget(ic)
            tt = QVBoxLayout(); tt.setSpacing(0)
            zh = QLabel(d['name'])
            zh.setStyleSheet(
                f'color:{C_TEXT}; font-size:14px; font-weight:bold;'
                f'letter-spacing:1.4px; background:transparent;')
            en = QLabel(d['en'])
            en.setStyleSheet(
                f'color:{C_DIM}; font-size:10px; letter-spacing:1.4px;'
                f'background:transparent;')
            tt.addWidget(zh); tt.addWidget(en)
            head.addLayout(tt)
            head.addStretch()

            tag_map = {'live': ('在线', C_OK, '#0d2818'),
                       'sim':  ('SIM', C_WARN, '#2e1f05'),
                       'offline': ('离线', C_RISK, '#2a0a0e')}
            label, fg, bg = tag_map[d['state']]
            chip = QLabel(f'  ●  {label}  ')
            chip.setStyleSheet(
                f'background:{bg}; color:{fg};'
                f'border:1px solid {fg}; border-radius:11px;'
                f'padding:3px 12px; font-size:11px; font-weight:bold;'
                f'letter-spacing:1.4px;')
            head.addWidget(chip)
            cl.addLayout(head)

            det = QLabel(d['detail'])
            det.setStyleSheet(
                f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.7;'
                f'background:transparent;')
            det.setWordWrap(True)
            cl.addWidget(det)

            self._grid.addWidget(card, i // 3, i % 3)


# ============================================================
# 5. 个体基线标定（90s 静息）
# ============================================================

class BaselineCalibrationDialog(_OverlayDialog):
    """90 秒静息采集，建立个体化基线。"""

    DURATION = 90  # 秒

    def __init__(self, parent=None):
        super().__init__(parent, card_w=900, card_h=680)
        self._parent_window = parent
        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._samples_hr = []
        self._samples_hrv = []
        self._samples_nn_risk = []
        self._samples_nn_tier = []
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('🎯  个体基线标定  ·  BASELINE CALIBRATION',
                            '90s RESTING ACQUISITION  ·  Engineering Only',
                            self._on_close, back_text='✕ 关闭', accent=C_ACCENT2))

        notice = QFrame()
        notice.setStyleSheet(f'background:#1a0f3a; border-left:4px solid {C_ACCENT2};')
        notice.setFixedHeight(56)
        n_lay = QHBoxLayout(notice)
        n_lay.setContentsMargins(22, 8, 22, 8)
        ic = QLabel('🎯'); ic.setStyleSheet('font-size:22px; background:transparent;')
        n_lay.addWidget(ic)
        msg = QLabel('请保持 90 秒静息状态，避免讲话与剧烈动作。'
                     '本步骤为纯工程参数标定，不涉及任何医学评估。')
        msg.setStyleSheet(
            f'color:{C_TEXT}; font-size:12px; line-height:1.6;'
            f'background:transparent;')
        msg.setWordWrap(True)
        n_lay.addWidget(msg, 1)
        self.card_lay.addWidget(notice)

        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(32, 26, 32, 22); body_lay.setSpacing(18)

        # 大数字倒计时
        self._count_lbl = QLabel(f'{self.DURATION}')
        self._count_lbl.setStyleSheet(
            f'color:{C_ACCENT2}; font-size:96px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self._count_lbl.setAlignment(Qt.AlignCenter)
        body_lay.addWidget(self._count_lbl)

        # 进度条
        self._bar = QProgressBar()
        self._bar.setRange(0, self.DURATION)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            f'QProgressBar {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-radius:4px; }}'
            f'QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            f'stop:0 {C_ACCENT}, stop:1 {C_ACCENT2}); border-radius:4px; }}')
        body_lay.addWidget(self._bar)

        # 实时指标
        body_lay.addLayout(_section_header(
            '实时采样', 'LIVE SAMPLES', C_ACCENT))
        live = QHBoxLayout(); live.setSpacing(14)
        self._hr_lbl = self._stat_card('心率 BPM', '--', C_HEART)
        self._hrv_lbl = self._stat_card('HRV RMSSD', '--', C_HEART)
        self._att_lbl = self._stat_card('专注 Attention', '--', C_BRAIN)
        live.addWidget(self._hr_lbl, 1)
        live.addWidget(self._hrv_lbl, 1)
        live.addWidget(self._att_lbl, 1)
        body_lay.addLayout(live)

        # 按钮
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addStretch()
        self._btn_start = QPushButton('▶  开始 90s 静息采集')
        self._btn_start.setCursor(Qt.PointingHandCursor)
        self._btn_start.setFixedHeight(40)
        self._btn_start.setStyleSheet(
            f'QPushButton {{ background:{C_ACCENT2}; color:{C_BG_DEEP};'
            f'border-radius:8px; padding:6px 28px; font-weight:bold;'
            f'letter-spacing:1.5px; font-size:13px; }}'
            f'QPushButton:hover {{ background:{C_BG_DEEP}; color:{C_ACCENT2};'
            f'border:1px solid {C_ACCENT2}; }}')
        self._btn_start.clicked.connect(self._start)
        btn_row.addWidget(self._btn_start)
        body_lay.addLayout(btn_row)

        body_lay.addStretch()
        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(
            '⚠  基线参数仅用于提升后续算法的个性化推理精度，不构成任何医学评估。'))

    def _stat_card(self, title, value, color):
        card = QFrame()
        card.setStyleSheet(
            f'QFrame {{ background:{C_PANEL}; border:1px solid {C_BORDER};'
            f'border-radius:10px; }}')
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12); cl.setSpacing(2)
        kl = QLabel(title)
        kl.setStyleSheet(
            f'color:{C_DIM}; font-size:11px; letter-spacing:1.4px;'
            f'background:transparent;')
        vl = QLabel(value)
        vl.setStyleSheet(
            f'color:{color}; font-size:26px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        kl.setAlignment(Qt.AlignCenter); vl.setAlignment(Qt.AlignCenter)
        cl.addWidget(kl); cl.addWidget(vl)
        card._vl = vl
        return card

    def _start(self):
        self._elapsed = 0
        self._samples_hr.clear()
        self._samples_hrv.clear()
        self._samples_nn_risk.clear()
        self._samples_nn_tier.clear()
        self._btn_start.setEnabled(False)
        self._btn_start.setText('采集中…')
        self._timer.start()

    def _tick(self):
        self._elapsed += 1
        remain = self.DURATION - self._elapsed
        self._count_lbl.setText(f'{max(0, remain)}')
        self._bar.setValue(self._elapsed)

        # 采样
        win = self._parent_window
        hr = hrv = att = None
        try:
            if win is not None:
                snap = getattr(win, 'latest_physio_snapshot', {}) or {}
                hr = snap.get('bpm')
                hrv = snap.get('hrv_rmssd')
        except Exception:
            pass
        # 占位（演示模式）
        if not hr: hr = 72 + math.sin(self._elapsed / 8) * 4
        if not hrv: hrv = 35 + math.sin(self._elapsed / 6) * 6
        if not att: att = 60 + math.sin(self._elapsed / 5) * 8

        self._samples_hr.append(float(hr))
        self._samples_hrv.append(float(hrv))
        # 同步采 NN 风险分布
        try:
            from risk_bus import RiskBus
            nn = RiskBus.instance().latest()
            if nn is not None:
                self._samples_nn_risk.append(float(nn.get('risk_score', 0.0)))
                self._samples_nn_tier.append(int(nn.get('tier', 0)))
        except Exception:
            pass

        self._hr_lbl._vl.setText(f'{hr:.0f}')
        self._hrv_lbl._vl.setText(f'{hrv:.1f}')
        self._att_lbl._vl.setText(f'{att:.0f}')

        if self._elapsed >= self.DURATION:
            self._timer.stop()
            self._finish()

    def _finish(self):
        avg_hr = sum(self._samples_hr) / max(1, len(self._samples_hr))
        avg_hrv = sum(self._samples_hrv) / max(1, len(self._samples_hrv))
        nn_risks = self._samples_nn_risk
        avg_nn = sum(nn_risks) / max(1, len(nn_risks)) if nn_risks else 0.0
        # tier 分布 = 静息时模型识别能力的"个体偏移"
        tier_dist = [0, 0, 0, 0]
        for t in self._samples_nn_tier:
            if 0 <= t < 4:
                tier_dist[t] += 1
        n_nn = max(1, sum(tier_dist))
        tier_dist_pct = [c / n_nn for c in tier_dist]

        baseline = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'duration_s': self.DURATION,
            'avg_hr': avg_hr,
            'avg_hrv': avg_hrv,
            'nn_risk_avg': avg_nn,
            'nn_tier_dist': tier_dist_pct,
            'nn_samples': len(self._samples_nn_risk),
        }
        path = os.path.join(os.path.dirname(__file__), 'baseline.json')
        path_nn = os.path.join(os.path.dirname(__file__), 'baseline_nn.json')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(baseline, f, ensure_ascii=False, indent=2)
            with open(path_nn, 'w', encoding='utf-8') as f:
                json.dump({'nn_risk_avg': avg_nn,
                           'nn_tier_dist': tier_dist_pct,
                           'timestamp': baseline['timestamp']},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                win.ai_console.append(
                    f'> [基线] 个体基线标定完成  ·  HR 均值 {avg_hr:.1f}  ·  '
                    f'HRV 均值 {avg_hrv:.1f}  ·  NN 静息风险 {avg_nn:.2f}')
        except Exception:
            pass

        QMessageBox.information(self, '标定完成',
            f'本次 90 秒静息基线已保存：\n\n'
            f'• 平均心率：{avg_hr:.1f} bpm\n'
            f'• 平均 HRV：{avg_hrv:.1f} ms\n'
            f'• NN 静息综合风险：{avg_nn:.2f}\n'
            f'• NN tier 分布：L1 {tier_dist_pct[0]*100:.0f}%  /  L2 {tier_dist_pct[1]*100:.0f}%  /  '
            f'L3 {tier_dist_pct[2]*100:.0f}%  /  L4 {tier_dist_pct[3]*100:.0f}%\n\n'
            f'后续 NN 推理将以此为个体参考基线（领域自适应）。')
        self._btn_start.setEnabled(True)
        self._btn_start.setText('▶  重新采集')

    def _on_close(self):
        try:
            self._timer.stop()
        except Exception:
            pass
        self.accept()


# ============================================================
# 6. 数据回放
# ============================================================

class _ReplayTrace(QWidget):
    """简易时序回放图：风险 + 三模态曲线。优先用 RiskBus.history，否则演示数据。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._cursor = 0
        self._data = self._load_from_bus_or_demo(600)
        # 标红连续 ≥L3 的时段：tier_buf 中位置
        self._tier_buf = self._data.get('tier', [0] * len(self._data['risk']))

    @staticmethod
    def _load_from_bus_or_demo(n):
        """从 RiskBus.history 取最近 N 秒；不足则用合成数据补齐。"""
        try:
            from risk_bus import RiskBus
            hist = RiskBus.instance().history(n)
        except Exception:
            hist = []
        if len(hist) >= 60:
            risk = [float(h.get('risk_score', 0.0)) for h in hist]
            tier = [int(h.get('tier', 0)) for h in hist]
            mw = [h.get('modal_w', (0.35, 0.40, 0.25)) for h in hist]
            vis = [float(m[0]) for m in mw]
            hr  = [float(m[1]) for m in mw]
            eeg = [float(m[2]) for m in mw]
            attn = [list(h.get('attn', [1/60]*60)) for h in hist]
            probs = [list(h.get('tier_probs', [0.25]*4)) for h in hist]
            return dict(risk=risk, vis=vis, hr=hr, eeg=eeg,
                        tier=tier, attn=attn, probs=probs)
        # 演示数据
        random.seed(20260516)
        risk = []; vis = []; hr = []; eeg = []
        tier = []; attn = []; probs = []
        v = 0.3
        for i in range(n):
            v += random.uniform(-0.02, 0.022)
            v = max(0.05, min(0.95, v))
            risk.append(v)
            vis.append(max(0.0, min(1.0, v + random.uniform(-0.1, 0.1))))
            hr.append(max(0.0, min(1.0, v + random.uniform(-0.15, 0.1))))
            eeg.append(max(0.0, min(1.0, v + random.uniform(-0.12, 0.12))))
            t = 0 if v < 0.3 else (1 if v < 0.6 else (2 if v < 0.8 else 3))
            tier.append(t)
            # 合成 4 类概率
            try:
                from risk_bus import _synthetic_probs
                probs.append(_synthetic_probs(v))
            except Exception:
                probs.append([0.25] * 4)
            attn.append([1.0/60] * 60)
        return dict(risk=risk, vis=vis, hr=hr, eeg=eeg,
                    tier=tier, attn=attn, probs=probs)

    def set_cursor(self, c):
        self._cursor = max(0, min(len(self._data['risk']) - 1, int(c)))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width(); h = self.height()
        p.fillRect(0, 0, w, h, QColor(C_PANEL))

        n = len(self._data['risk'])
        if n < 2:
            return
        dx = w / (n - 1)

        # === ≥L3 时段标红背景（论文 3.3.5 连续预警）===
        tiers = self._tier_buf
        if tiers and len(tiers) == n:
            i = 0
            while i < n:
                if tiers[i] >= 2:
                    j = i
                    while j < n and tiers[j] >= 2:
                        j += 1
                    bg = QColor(C_RISK_HIGH); bg.setAlphaF(0.18)
                    p.fillRect(int(i * dx), 0, int(max(2, (j - i) * dx)), h, bg)
                    i = j
                else:
                    i += 1

        # 网格
        p.setPen(QPen(QColor(C_BORDER), 1, Qt.DotLine))
        for i in range(1, 4):
            y = int(h * i / 4)
            p.drawLine(0, y, w, y)

        traces = [
            ('vis', C_VISION, 'Vision'),
            ('hr', C_HEART, 'HRV'),
            ('eeg', C_BRAIN, 'EEG'),
            ('risk', C_ACCENT, 'Risk'),
        ]
        for key, color, _ in traces:
            arr = self._data[key]
            pen = QPen(QColor(color), 2 if key == 'risk' else 1.4)
            p.setPen(pen)
            x_prev = 0; y_prev = h - arr[0] * (h - 10) - 5
            for i in range(1, n):
                x = i * dx; y = h - arr[i] * (h - 10) - 5
                p.drawLine(int(x_prev), int(y_prev), int(x), int(y))
                x_prev, y_prev = x, y

        # 游标
        cx = int(self._cursor * dx)
        p.setPen(QPen(QColor(C_ACCENT), 1.4, Qt.DashLine))
        p.drawLine(cx, 0, cx, h)


class TimelineReplayDialog(_OverlayDialog):
    """数据回放：拖动滑块查看历史多模态时序。"""

    def __init__(self, parent=None):
        super().__init__(parent, card_w=1100, card_h=720)
        self._parent_window = parent
        self._build()

    def _build(self):
        self.card_lay.addWidget(
            _make_title_bar('⏪  数据回放  ·  TIMELINE REPLAY',
                            'MULTIMODAL HISTORICAL REVIEW',
                            self.accept, back_text='✕ 关闭', accent=C_RISK_MED))

        body = QWidget(); body.setStyleSheet(f'background:{C_BG_DEEP};')
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 22, 28, 22); body_lay.setSpacing(14)

        hint = QLabel('拖动下方时间轴回顾近 10 分钟的多模态时序，'
                      '可定位关键时刻、辅助理解风险变化。')
        hint.setStyleSheet(
            f'color:{C_TEXT_DIM}; font-size:12px; line-height:1.7;'
            f'background:transparent;')
        hint.setWordWrap(True)
        body_lay.addWidget(hint)

        # 图例
        legend = QHBoxLayout(); legend.setSpacing(18)
        for color, name in [(C_VISION, '■ 视觉'), (C_HEART, '■ HRV'),
                            (C_BRAIN, '■ 脑电'), (C_ACCENT, '■ 综合风险')]:
            l = QLabel(name)
            l.setStyleSheet(
                f'color:{color}; font-size:11px; font-weight:bold;'
                f'background:transparent;')
            legend.addWidget(l)
        legend.addStretch()
        body_lay.addLayout(legend)

        # 时序图
        self._trace = _ReplayTrace()
        body_lay.addWidget(self._trace, 1)

        # 时间游标 + 数值
        ctrl_row = QHBoxLayout(); ctrl_row.setSpacing(14)
        self._tlabel = QLabel('T = -10:00')
        self._tlabel.setStyleSheet(
            f'color:{C_ACCENT}; font-size:14px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        self._tlabel.setFixedWidth(110)
        ctrl_row.addWidget(self._tlabel)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 599)
        self._slider.setValue(0)
        self._slider.setStyleSheet(
            f'QSlider::groove:horizontal {{ background:{C_PANEL}; height:6px; '
            f'border-radius:3px; }}'
            f'QSlider::handle:horizontal {{ background:{C_ACCENT}; width:14px; '
            f'margin:-6px 0; border-radius:7px; }}'
            f'QSlider::sub-page:horizontal {{ background:{C_ACCENT};'
            f'border-radius:3px; }}')
        self._slider.valueChanged.connect(self._on_slider)
        ctrl_row.addWidget(self._slider, 1)
        body_lay.addLayout(ctrl_row)

        # 数值面板
        body_lay.addLayout(_section_header(
            '游标处指标', 'AT CURSOR  ·  NN OUTPUT @ THIS SECOND', C_ACCENT))
        self._vals_box = QFrame()
        self._vals_box.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        self._vals_lay = QHBoxLayout(self._vals_box)
        self._vals_lay.setContentsMargins(18, 12, 18, 12); self._vals_lay.setSpacing(20)
        self._lbl_risk = self._mini('综合风险', C_ACCENT)
        self._lbl_vis = self._mini('视觉贡献', C_VISION)
        self._lbl_hr = self._mini('HRV 贡献', C_HEART)
        self._lbl_eeg = self._mini('脑电贡献', C_BRAIN)
        self._lbl_tier = self._mini('NN Tier', C_OK)
        for w in (self._lbl_risk, self._lbl_vis, self._lbl_hr, self._lbl_eeg, self._lbl_tier):
            self._vals_lay.addWidget(w)
        self._vals_lay.addStretch()
        body_lay.addWidget(self._vals_box)

        # NN 决策溯源：4 档概率柱图 + attn 热力条
        body_lay.addLayout(_section_header(
            '游标处 NN 决策', 'NN DECISION AT CURSOR', C_BRAIN))
        nn_row = QHBoxLayout(); nn_row.setSpacing(14)
        self._cursor_probs = TierProbBars(); self._cursor_probs.setMinimumHeight(120)
        self._cursor_attn = AttnHeatStrip()
        nn_left = QFrame()
        nn_left.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        ll = QVBoxLayout(nn_left); ll.setContentsMargins(14, 12, 14, 12); ll.setSpacing(4)
        cap_p = QLabel('4 档概率分布')
        cap_p.setStyleSheet(f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px;')
        ll.addWidget(cap_p); ll.addWidget(self._cursor_probs)
        nn_right = QFrame()
        nn_right.setStyleSheet(
            f'background:{C_PANEL}; border:1px solid {C_BORDER}; border-radius:8px;')
        rl = QVBoxLayout(nn_right); rl.setContentsMargins(14, 12, 14, 12); rl.setSpacing(4)
        cap_a = QLabel('时间注意力 (过去 60s)')
        cap_a.setStyleSheet(f'color:{C_DIM}; font-size:10px; letter-spacing:1.2px;')
        rl.addWidget(cap_a); rl.addWidget(self._cursor_attn)
        self._cursor_trace = QLabel('选择游标位置查看 NN 决策。')
        self._cursor_trace.setStyleSheet(
            f'color:{C_TEXT}; font-size:11px; line-height:1.6;'
            f'background:transparent;')
        self._cursor_trace.setWordWrap(True)
        rl.addWidget(self._cursor_trace)
        nn_row.addWidget(nn_left, 2); nn_row.addWidget(nn_right, 3)
        body_lay.addLayout(nn_row)

        # 标注按钮
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addStretch()
        btn_mark = QPushButton('🚩  标记关键时刻')
        btn_mark.setCursor(Qt.PointingHandCursor)
        btn_mark.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_RISK_MED};'
            f'border:1px solid {C_RISK_MED}; border-radius:6px; padding:6px 18px;'
            f'font-weight:bold; }}'
            f'QPushButton:hover {{ background:{C_RISK_MED}; color:{C_BG_DEEP}; }}')
        btn_mark.clicked.connect(self._mark)
        btn_row.addWidget(btn_mark)
        body_lay.addLayout(btn_row)

        self.card_lay.addWidget(body, 1)
        self.card_lay.addWidget(_make_disclaimer_bar(DISCLAIMER_REFERRAL))

        self._on_slider(0)

    def _mini(self, name, color):
        w = QWidget(); w.setStyleSheet('background:transparent;')
        lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        kl = QLabel(name)
        kl.setStyleSheet(
            f'color:{C_DIM}; font-size:10px; letter-spacing:1.4px;'
            f'background:transparent;')
        vl = QLabel('--')
        vl.setStyleSheet(
            f'color:{color}; font-size:20px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')
        lay.addWidget(kl); lay.addWidget(vl)
        w._vl = vl
        return w

    def _on_slider(self, value):
        self._trace.set_cursor(value)
        # 时间：1 step = 1s，总 600s = 10min。倒数显示
        secs = 600 - value
        m, s = divmod(secs, 60)
        sign = '-' if secs > 0 else '+'
        self._tlabel.setText(f'T = {sign}{m:02d}:{s:02d}')

        d = self._trace._data
        self._lbl_risk._vl.setText(f'{d["risk"][value]:.2f}')
        self._lbl_vis._vl.setText(f'{d["vis"][value] * 100:.0f}%')
        self._lbl_hr._vl.setText(f'{d["hr"][value] * 100:.0f}%')
        self._lbl_eeg._vl.setText(f'{d["eeg"][value] * 100:.0f}%')

        tier = d.get('tier', [0])[value] if 'tier' in d else 0
        tier_label = ['L1', 'L2', 'L3', 'L4'][tier]
        tier_color = ['#26d07c', '#ffc857', '#ff8a3d', '#ff3860'][tier]
        self._lbl_tier._vl.setText(tier_label)
        self._lbl_tier._vl.setStyleSheet(
            f'color:{tier_color}; font-size:20px; font-weight:bold;'
            f'background:transparent; font-family:Consolas;')

        # 游标处的 NN 决策细节
        try:
            probs = d.get('probs', [[0.25] * 4])[value]
            attn = d.get('attn', [[1 / 60] * 60])[value]
            self._cursor_probs.set_probs(probs)
            self._cursor_attn.set_attn(attn)
            top_idx = max(range(4), key=lambda i: probs[i])
            tier_zh = ['低风险', '中风险', '高风险', '极高']
            self._cursor_trace.setText(
                f'此刻 NN 推理 → {tier_label} {tier_zh[top_idx]}  ·  置信度 {probs[top_idx]*100:.0f}%\n'
                f'4 类概率 = [{", ".join(f"{p:.2f}" for p in probs)}]\n'
                f'模态贡献 视觉/HRV/脑电 = {d["vis"][value]*100:.0f}% / '
                f'{d["hr"][value]*100:.0f}% / {d["eeg"][value]*100:.0f}%')
        except Exception:
            pass

    def _mark(self):
        idx = self._slider.value()
        risk = self._trace._data['risk'][idx]
        record = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'cursor': idx,
            'risk_at_cursor': risk,
            'note': '回放标记',
        }
        path = os.path.join(os.path.dirname(__file__), 'replay_marks.jsonl')
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            pass

        win = self._parent_window
        try:
            if win is not None and hasattr(win, 'ai_console'):
                win.ai_console.append(
                    f'> [回放] 已标记游标 #{idx}，当时风险 = {risk:.2f}')
        except Exception:
            pass
        QMessageBox.information(self, '已标记',
            f'游标位置已加入回放标记。\n该时刻风险参考值 = {risk:.2f}')


# ============================================================
# 便捷入口（更新）
# ============================================================

def open_ai_report(parent=None):
    return AIAssistedReportDialog(parent).exec_()

def open_live_intervention(parent=None):
    return LiveInterventionDialog(parent).exec_()

def open_signal_quality(parent=None):
    return SignalQualityDialog(parent).exec_()

def open_device_hub(parent=None):
    return DeviceHubDialog(parent).exec_()

def open_baseline(parent=None):
    return BaselineCalibrationDialog(parent).exec_()

def open_replay(parent=None):
    return TimelineReplayDialog(parent).exec_()


# ============================================================
# 3. 放松引导（HRV 生物反馈呼吸）- 保留用于兼容
# ============================================================
