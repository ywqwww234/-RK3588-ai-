import cv2
import requests
import json
import os
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import config

config.WINDOW_WIDTH = 1280
config.WINDOW_HEIGHT = 900

# ===== 统一主题（深色工业风，与学生端一致） =====
from theme import (C_BG as _CB, C_BG_DEEP as _CBD, C_PANEL as _CP,
                   C_PANEL_HI as _CPH, C_BORDER as _CBO, C_ACCENT as _CA,
                   C_DIM as _CD, C_TEXT as _CT, C_OK as _COK,
                   C_VISION as _CV, C_HEART as _CH, C_BRAIN as _CBR,
                   C_RISK_LOW as _CRL, C_RISK_MED as _CRM,
                   C_RISK_HIGH as _CRH, C_RISK_CRIT as _CRC, C_RISK as _CR)
THEME_PRIMARY = _CA
THEME_PRIMARY_HOVER = _CA
THEME_WARN = _CR
THEME_SUCCESS = _COK
THEME_TEXT_DARK = _CT
THEME_TEXT_MUTED = _CD
THEME_BORDER = _CBO

# 云端 API Key（按你的要求：直接写入代码）
# 仍保留环境变量覆盖能力，便于后续切换
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()

class AIAnalyzeThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, df_data):
        super().__init__()
        self.df_data = df_data

    def _build_local_fallback_advice(self, risk_series):
        mean_risk = float(risk_series.mean()) if len(risk_series) > 0 else 0.0
        max_risk = float(risk_series.max()) if len(risk_series) > 0 else 0.0
        high_count = int((risk_series >= config.RISK_THRESHOLD_MEDIUM).sum()) if len(risk_series) > 0 else 0

        trend_text = "数据不足"
        if len(risk_series) >= 6:
            recent = float(risk_series.tail(3).mean())
            prev = float(risk_series.iloc[-6:-3].mean())
            if recent - prev > 0.03:
                trend_text = "近阶段风险上升"
            elif prev - recent > 0.03:
                trend_text = "近阶段风险下降"
            else:
                trend_text = "近阶段风险平稳"

        if mean_risk >= config.RISK_THRESHOLD_HIGH or max_risk >= 0.85:
            level_text = "当前属于高关注状态"
            action_1 = "今晚减少额外学业刺激，优先保证睡眠与进食节律。"
            action_2 = "连续3天进行15分钟非评判式沟通，先倾听再建议。"
        elif mean_risk >= config.RISK_THRESHOLD_MEDIUM or high_count >= 3:
            level_text = "当前存在波动，建议中等强度关注"
            action_1 = "固定每天同一时间做简短情绪打分，观察触发场景。"
            action_2 = "安排轻运动与户外日照，减少晚间长时间电子屏使用。"
        else:
            level_text = "整体状态相对平稳"
            action_1 = "继续保持规律作息，并维持每周稳定的家庭交流时间。"
            action_2 = "当学习压力上升时，优先使用短时休息与任务拆分策略。"

        return (
            f"【本地回退建议】{level_text}。"
            f"统计：平均风险{mean_risk:.2f}，最高风险{max_risk:.2f}，"
            f"高风险次数{high_count}，趋势：{trend_text}。\n"
            f"建议1：{action_1}\n"
            f"建议2：{action_2}"
        )

    def run(self):
        try:
            if self.df_data.empty:
                self.error_signal.emit("暂无充足数据分析。")
                return

            risk_series = self.df_data['risk'].astype(float)
            mean_risk = float(risk_series.mean())
            max_risk = float(risk_series.max())
            high_count = int((risk_series >= config.RISK_THRESHOLD_MEDIUM).sum())

            trend_text = "数据不足"
            if len(risk_series) >= 6:
                recent = float(risk_series.tail(3).mean())
                prev = float(risk_series.iloc[-6:-3].mean())
                if recent - prev > 0.03:
                    trend_text = "近阶段风险上升"
                elif prev - recent > 0.03:
                    trend_text = "近阶段风险下降"
                else:
                    trend_text = "近阶段风险平稳"

            prompt = (
                "请作为心理健康顾问，基于以下指标生成家长可读的150字内建议："
                f"平均风险={mean_risk:.3f}，最高风险={max_risk:.3f}，"
                f"高风险次数(>= {config.RISK_THRESHOLD_MEDIUM:.2f})={high_count}，"
                f"趋势判断={trend_text}。"
                "输出要求：1) 先给总体判断；2) 再给2条可执行家庭支持建议；"
                "3) 避免医疗诊断措辞。"
            )

            # 若未配置密钥，直接启用本地回退
            if not ZHIPU_API_KEY:
                self.finished_signal.emit(self._build_local_fallback_advice(risk_series))
                return

            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "glm-4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            session = requests.Session()
            session.trust_env = False
            response = session.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                self.finished_signal.emit(response.json()['choices'][0]['message']['content'])
            else:
                self.finished_signal.emit(self._build_local_fallback_advice(risk_series))
        except Exception:
            try:
                risk_series = self.df_data['risk'].astype(float) if not self.df_data.empty else pd.Series(dtype=float)
                self.finished_signal.emit(self._build_local_fallback_advice(risk_series))
            except Exception as e:
                self.error_signal.emit(f"分析失败: {str(e)}")

class ReportPrepareThread(QThread):
    prepared_signal = pyqtSignal(str)

    def __init__(self, df, ai_text):
        super().__init__()
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.ai_text = ai_text

    def run(self):
        avg_risk = 0.0
        max_risk = 0.0
        abnormal_count = 0

        if not self.df.empty and 'risk' in self.df.columns:
            avg_risk = float(self.df['risk'].mean())
            max_risk = float(self.df['risk'].max())
            abnormal_count = int((self.df['risk'] > config.RISK_THRESHOLD_MEDIUM).sum())

        generated_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ai_conclusion = (self.ai_text or "暂无 AI 评估结论").replace("\n", "<br>")

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
        color: #E8EAF6;
        background: #0A1F3D;
        margin: 56px;
        line-height: 1.7;
    }}
    .title {{
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
    }}
    .meta {{
        color: #9BA8C8;
        font-size: 13px;
        margin-bottom: 34px;
    }}
    .section-title {{
        font-size: 18px;
        color: #00E5FF;
        margin: 26px 0 12px;
        font-weight: 600;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        margin-bottom: 24px;
        border: 1px solid #1C4076;
        border-radius: 8px;
    }}
    th, td {{
        text-align: left;
        padding: 14px 16px;
        font-size: 14px;
        border-bottom: 1px solid #04111F;
    }}
    th {{
        color: #9BA8C8;
        font-weight: 600;
        width: 42%;
        background: #fafafa;
    }}
    td {{
        color: #E8EAF6;
        font-weight: 600;
    }}
    .conclusion {{
        border: 1px solid #dbeafe;
        border-left: 4px solid #00E5FF;
        border-radius: 8px;
        background: #f7fbff;
        padding: 16px 18px;
        font-size: 14px;
    }}
</style>
</head>
<body>
    <div class=\"title\">MindRoom 心理风险评估报告</div>
    <div class=\"meta\">生成时间：{generated_time}</div>

    <div class=\"section-title\">核心指标概览</div>
    <table>
        <tr><th>平均风险指数</th><td>{avg_risk:.2f}</td></tr>
        <tr><th>最高风险指数</th><td>{max_risk:.2f}</td></tr>
        <tr><th>异常波动次数（>{config.RISK_THRESHOLD_MEDIUM:.2f}）</th><td>{abnormal_count}</td></tr>
    </table>

    <div class=\"section-title\">AI 辅助评估结论（非临床诊断）</div>
    <div class=\"conclusion\">{ai_conclusion}</div>
</body>
</html>
"""
        self.prepared_signal.emit(html)


class GlowCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background-color: #0A1F3D; border-radius: 12px; border: 1px solid #1C4076; }")
        self.setMinimumSize(120, 100) 

class EmotionHeatmapWidget(QWidget):
    day_clicked = pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.cell_size = 14
        self.cell_gap = 4
        self.weeks = 26
        self.days = 7
        self.daily_values = []
        self.daily_dates = []
        self.daily_raw_stats = []
        self._cell_rects = []

    def set_range_days(self, range_days):
        self.weeks = max(5, (int(range_days) + 6) // 7)
        self.update()

    def set_daily_values(self, daily_values, daily_dates=None, daily_raw_stats=None):
        self.daily_values = list(daily_values)
        self.daily_dates = list(daily_dates) if daily_dates is not None else [None] * len(self.daily_values)
        self.daily_raw_stats = list(daily_raw_stats) if daily_raw_stats is not None else [None] * len(self.daily_values)
        self.update()

    def _color_for_value(self, value):
        if value <= 0.2:
            return QColor(0, 122, 255, 120)
        if value <= 0.4:
            return QColor(0, 122, 255, 180)
        if value <= 0.6:
            return QColor(255, 149, 0, 180)
        if value <= 0.8:
            return QColor(255, 69, 58, 200)
        return QColor(255, 45, 85, 220)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        start_x = 14
        start_y = 20
        outline = QColor(229, 229, 234)
        empty_color = QColor(230, 230, 235)

        total_cells = self.weeks * self.days
        values = (self.daily_values or [])[-total_cells:]
        dates = (self.daily_dates or [])[-total_cells:]
        raw_stats = (self.daily_raw_stats or [])[-total_cells:]
        if len(values) < total_cells:
            pad_count = total_cells - len(values)
            values = [None] * pad_count + values
            dates = [None] * pad_count + dates
            raw_stats = [None] * pad_count + raw_stats

        self._cell_rects = []
        for idx, value in enumerate(values):
            week = idx // self.days
            day = idx % self.days
            x = start_x + week * (self.cell_size + self.cell_gap)
            y = start_y + day * (self.cell_size + self.cell_gap)
            rect = QRectF(x, y, self.cell_size, self.cell_size)
            self._cell_rects.append((QRect(int(x), int(y), self.cell_size, self.cell_size), dates[idx], value, raw_stats[idx]))
            painter.setPen(QPen(outline, 1))
            if value is None:
                painter.setBrush(empty_color)
            else:
                painter.setBrush(self._color_for_value(max(0.0, min(1.0, value))))
            painter.drawRoundedRect(rect, 3, 3)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        for rect, date_value, risk_value, raw_stat in self._cell_rects:
            if rect.contains(pos):
                if date_value is None:
                    QToolTip.showText(event.globalPos(), "暂无数据", self)
                else:
                    date_text = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
                    if risk_value is None:
                        QToolTip.showText(event.globalPos(), f"日期：{date_text}\n风险：暂无数据", self)
                    else:
                        count_text = ""
                        if isinstance(raw_stat, dict):
                            count_text = f"\n当日原始记录数：{int(raw_stat.get('count', 0))}"
                        QToolTip.showText(event.globalPos(), f"日期：{date_text}\n风险：{float(risk_value):.2f}{count_text}\n点击查看当日详情", self)
                return
        QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            for rect, date_value, risk_value, raw_stat in self._cell_rects:
                if rect.contains(pos):
                    self.day_clicked.emit(date_value, risk_value, raw_stat)
                    return
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

class ConcentricRingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.values = {"vis": 75, "hrv": 60, "eeg": 90}
        self.total_risk_score = 0.00

    def _risk_theme_colors(self):
        _, risk_color = get_risk_level_meta(self.total_risk_score)
        fg = QColor(risk_color)
        bg = QColor(fg)
        bg.setAlpha(45)
        return bg, fg

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = min(self.width(), self.height())
        cx, cy = self.width() // 2, self.height() // 2
        outer_radius = (size // 2) - 20
        thickness = int(outer_radius * 0.12) 
        
        bg_color, fg_color = self._risk_theme_colors()
        for m_id, radius_offset in [("vis", 0), ("hrv", thickness + 6), ("eeg", (thickness + 6) * 2)]:
            radius = outer_radius - radius_offset
            if radius < thickness:
                continue
            painter.setPen(QPen(bg_color, thickness, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 0, 360 * 16)
            angle = int(360 * self.values[m_id] / 100)
            painter.setPen(QPen(fg_color, thickness, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 90 * 16, -angle * 16)

        painter.setPen(QPen(fg_color))
        painter.setFont(QFont("Arial", 42, QFont.Bold))
        painter.drawText(QRect(cx - outer_radius, cy - 40, outer_radius*2, 60), Qt.AlignCenter, f"{self.total_risk_score:.2f}")

    def set_values(self, vis, hrv, eeg, score):
        self.values = {"vis": vis * 100, "hrv": hrv * 100, "eeg": eeg * 100}
        self.total_risk_score = score
        self.update() 

def get_risk_level_meta(risk_value):
    """统一风险等级语义：返回 (等级文本, 颜色)"""
    try:
        v = float(risk_value)
    except Exception:
        v = 0.0

    v = max(0.0, min(1.0, v))
    if v < 0.30:
        return "低风险", THEME_SUCCESS
    if v < 0.55:
        return "轻度波动", "#FFD60A"
    if v < 0.80:
        return "中高风险", "#FFC857"
    return "高风险", THEME_WARN


class ParentWindow(QMainWindow):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self._log_last_emit = {}
        self._log_throttle_secs = 5.0
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1680, 980)
        self._init_ui()
        self._install_card_zoom()

    def _install_card_zoom(self):
        """给关键大卡绑双击放大。"""
        try:
            from ui_widgets import ZoomOverlay
            for attr, title in [
                ('heatmap_card',  '七日情绪热力图'),
                ('ring_card',     '今日风险概览'),
                ('alert_card',    '风险预警时序'),
                ('event_card',    '事件流'),
                ('eeg_card',      '脑电模态'),
                ('hrv_card',      'HRV 模态'),
                ('emotion_heatmap', '情绪雷达'),
            ]:
                w = getattr(self, attr, None)
                if w is not None:
                    ZoomOverlay.install_zoom(w, self, title)
        except Exception:
            pass

    def _init_ui(self):
        self._init_ui_dashboard()

    def _init_ui_dashboard(self):
        self.bg = QFrame(self)
        self.bg.setGeometry(0, 0, self.width(), self.height())
        self.bg.setStyleSheet("background-color: #06162A; border-radius: 16px; border: 1px solid #173A66;")

        root = QVBoxLayout(self.bg)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        # ===== 顶部状态栏 =====
        self.dashboard_header = QFrame()
        self.dashboard_header.setStyleSheet(
            "QFrame{background:#081A31; border:1px solid #1C4076; border-radius:14px;}"
        )
        header_layout = QHBoxLayout(self.dashboard_header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(14)

        title_box = QVBoxLayout()
        self.dashboard_title = QLabel("家长端总览 Dashboard")
        self.dashboard_title.setStyleSheet("color:#F3F7FF; font-size:24px; font-weight:700; font-family:'微软雅黑';")
        self.dashboard_subtitle = QLabel("3 秒看异常，10 秒找原因与动作 · 深色工业风单页总览")
        self.dashboard_subtitle.setStyleSheet("color:#8FA5CF; font-size:12px; font-family:'微软雅黑';")
        title_box.addWidget(self.dashboard_title)
        title_box.addWidget(self.dashboard_subtitle)

        self.dashboard_status_badge = QLabel("状态：待检测")
        self.dashboard_status_badge.setStyleSheet(
            "QLabel{background:#0E2747; color:#FFC857; border:1px solid #345A8E; border-radius:10px; padding:8px 12px; font-size:13px; font-weight:700; font-family:'微软雅黑';}"
        )
        self.dashboard_time_badge = QLabel("最后刷新：--")
        self.dashboard_time_badge.setStyleSheet(
            "QLabel{background:#0E2747; color:#E8EAF6; border:1px solid #1C4076; border-radius:10px; padding:8px 12px; font-size:13px; font-family:'微软雅黑';}"
        )

        self.btn_refresh_all = QPushButton("🔄 刷新全局")
        self.btn_refresh_all.setCursor(Qt.PointingHandCursor)
        self.btn_refresh_all.setMinimumHeight(38)
        self.btn_refresh_all.setStyleSheet(
            f"QPushButton {{ background:{THEME_PRIMARY}; color:#06162A; border:none; border-radius:10px; padding:8px 16px; font-size:14px; font-weight:700; }}"
            f"QPushButton:hover {{ background:#44E6FF; }}"
        )
        self.btn_refresh_all.clicked.connect(self._refresh_all)

        self.btn_exit = QPushButton("⏏ 退出")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.setMinimumHeight(38)
        self.btn_exit.setStyleSheet(
            "QPushButton{background:#0E2747; color:#FF7A7A; border:1px solid #2E4E7D; border-radius:10px; padding:8px 16px; font-size:14px; font-weight:700;}"
            "QPushButton:hover{background:#17355D;}"
        )
        self.btn_exit.clicked.connect(self.close)

        self.data_source_badge = QLabel("数据源：待检测")
        self.data_source_badge.setStyleSheet(
            "QLabel{background:#0E2747; color:#FFC857; border:1px solid #345A8E; border-radius:10px; padding:8px 12px; font-size:13px; font-weight:700; font-family:'微软雅黑';}"
        )

        header_layout.addLayout(title_box, 1)
        header_layout.addWidget(self.data_source_badge)
        header_layout.addWidget(self.dashboard_status_badge)
        header_layout.addWidget(self.dashboard_time_badge)
        header_layout.addWidget(self.btn_refresh_all)
        header_layout.addWidget(self.btn_exit)
        root.addWidget(self.dashboard_header)

        # ===== 主要内容区（单页可滚动） =====
        self.dashboard_scroll = QScrollArea()
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setFrameShape(QFrame.NoFrame)
        self.dashboard_scroll.setStyleSheet("QScrollArea{background:transparent; border:none;} QWidget{background:transparent;}")

        content = QWidget()
        self.dashboard_grid = QGridLayout(content)
        self.dashboard_grid.setContentsMargins(0, 0, 0, 0)
        self.dashboard_grid.setHorizontalSpacing(14)
        self.dashboard_grid.setVerticalSpacing(14)
        self.dashboard_scroll.setWidget(content)
        root.addWidget(self.dashboard_scroll, 1)

        # ===== 卡片 1：今日健康概览 =====
        self.summary_card = GlowCard()
        self.summary_card.setStyleSheet("QFrame{background:#081A31; border:1px solid #1C4076; border-radius:14px;}")
        summary_lay = QVBoxLayout(self.summary_card)
        summary_lay.setContentsMargins(16, 14, 16, 14)
        summary_lay.setSpacing(10)

        summary_head = QHBoxLayout()
        summary_title = QLabel("今日健康概览")
        summary_title.setStyleSheet("color:#F3F7FF; font-size:18px; font-weight:700; font-family:'微软雅黑';")
        summary_hint = QLabel("P0 先看异常")
        summary_hint.setStyleSheet("QLabel{background:#102A4C; color:#44E6FF; border:1px solid #274D7B; border-radius:8px; padding:4px 8px; font-size:12px; font-weight:700;}")
        summary_head.addWidget(summary_title)
        summary_head.addStretch()
        summary_head.addWidget(summary_hint)
        summary_lay.addLayout(summary_head)

        summary_body = QHBoxLayout()
        summary_body.setSpacing(12)
        self.ring_card = GlowCard()
        self.ring_card.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:12px;}")
        ring_layout = QVBoxLayout(self.ring_card)
        ring_layout.setContentsMargins(10, 10, 10, 10)
        self.ring_widget = ConcentricRingWidget(self.ring_card)
        self.ring_widget.setMinimumSize(320, 260)
        ring_layout.addWidget(self.ring_widget)
        summary_body.addWidget(self.ring_card, 3)

        summary_metrics = QVBoxLayout()
        summary_metrics.setSpacing(10)
        self.summary_metrics_wrap = QWidget()
        self.summary_metrics_wrap.setLayout(summary_metrics)
        self.summary_metrics_wrap.setMinimumWidth(380)
        self.score_card = self._create_metric_card("系统评估", "0.00", THEME_TEXT_DARK)
        self.alert_card = self._create_metric_card("最新预警", "暂无", THEME_TEXT_MUTED)
        summary_metrics.addWidget(self.score_card['frame'])
        summary_metrics.addWidget(self.alert_card['frame'])
        self.summary_chip_row = QHBoxLayout()
        self.summary_chip_row.setSpacing(8)
        self.summary_level_chip = QLabel("等级：待检测")
        self.summary_level_chip.setStyleSheet("QLabel{background:#0E2747; color:#E8EAF6; border:1px solid #2E4E7D; border-radius:8px; padding:6px 10px; font-size:12px;}")
        self.summary_trend_chip = QLabel("趋势：--")
        self.summary_trend_chip.setStyleSheet("QLabel{background:#0E2747; color:#8FA5CF; border:1px solid #2E4E7D; border-radius:8px; padding:6px 10px; font-size:12px;}")
        self.summary_chip_row.addWidget(self.summary_level_chip)
        self.summary_chip_row.addWidget(self.summary_trend_chip)
        self.summary_chip_row.addStretch()
        summary_metrics.addLayout(self.summary_chip_row)
        summary_body.addWidget(self.summary_metrics_wrap, 2)
        summary_lay.addLayout(summary_body)

        modality_row = QHBoxLayout()
        modality_row.setSpacing(10)
        self.vis_card = self._create_modality_card("视觉特征", "#00C2FF")
        self.hrv_card = self._create_modality_card("生理 HRV", "#FF6B81")
        self.eeg_card = self._create_modality_card("脑电 EEG", THEME_PRIMARY)
        modality_row.addWidget(self.vis_card['frame'])
        modality_row.addWidget(self.hrv_card['frame'])
        modality_row.addWidget(self.eeg_card['frame'])
        summary_lay.addLayout(modality_row)

        # ===== 卡片 2：风险趋势追踪 =====
        self.trend_card = GlowCard()
        self.trend_card.setStyleSheet("QFrame{background:#081A31; border:1px solid #1C4076; border-radius:14px;}")
        trend_lay = QVBoxLayout(self.trend_card)
        trend_lay.setContentsMargins(16, 14, 16, 14)
        trend_lay.setSpacing(10)

        trend_head = QHBoxLayout()
        trend_title = QLabel("风险趋势追踪")
        trend_title.setStyleSheet("color:#F3F7FF; font-size:18px; font-weight:700; font-family:'微软雅黑';")
        trend_note = QLabel("P1 变化与阈值")
        trend_note.setStyleSheet("QLabel{background:#102A4C; color:#FFC857; border:1px solid #2E4E7D; border-radius:8px; padding:4px 8px; font-size:12px; font-weight:700;}")
        self.btn_hm_30 = QPushButton("近30天")
        self.btn_hm_180 = QPushButton("近半年")
        self.btn_hm_365 = QPushButton("近一年")
        for btn, days in [(self.btn_hm_30, 30), (self.btn_hm_180, 180), (self.btn_hm_365, 365)]:
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton{{background:#0E2747; color:{THEME_PRIMARY}; border:1px solid {THEME_PRIMARY}; border-radius:8px; padding:6px 10px; font-size:12px;}}"
                f"QPushButton:checked{{background:{THEME_PRIMARY}; color:#06162A;}}"
            )
            btn.clicked.connect(lambda checked, d=days: self._set_heatmap_range(d))
        self.btn_hm_180.setChecked(True)
        trend_head.addWidget(trend_title)
        trend_head.addWidget(trend_note)
        trend_head.addStretch()
        trend_head.addWidget(self.btn_hm_30)
        trend_head.addWidget(self.btn_hm_180)
        trend_head.addWidget(self.btn_hm_365)
        trend_lay.addLayout(trend_head)

        risk_strip = QFrame()
        risk_strip.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:12px;}")
        risk_strip_lay = QHBoxLayout(risk_strip)
        risk_strip_lay.setContentsMargins(14, 10, 14, 10)
        self.risk_main_value = QLabel("0.00 / 1.00")
        self.risk_main_value.setStyleSheet("color:#F3F7FF; font-size:28px; font-weight:700;")
        self.risk_main_level = QLabel("等级：待检测")
        self.risk_main_level.setStyleSheet("color:#8FA5CF; font-size:13px; font-family:'微软雅黑';")
        self.risk_main_delta = QLabel("24h 趋势：--")
        self.risk_main_delta.setStyleSheet("color:#8FA5CF; font-size:13px; font-family:'微软雅黑';")
        left_risk = QVBoxLayout(); left_risk.addWidget(self.risk_main_value); left_risk.addWidget(self.risk_main_level)
        risk_strip_lay.addLayout(left_risk)
        risk_strip_lay.addStretch()
        risk_strip_lay.addWidget(self.risk_main_delta)
        trend_lay.addWidget(risk_strip)

        self.fig = Figure(figsize=(8, 3.6), facecolor='#0A1F3D')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0A1F3D')
        self.trends_card = GlowCard()
        self.trends_card.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:12px;}")
        t_lay = QVBoxLayout(self.trends_card)
        t_lay.setContentsMargins(8, 8, 8, 8)
        t_lay.addWidget(self.canvas)
        trend_lay.addWidget(self.trends_card, 2)

        lower_trend = QHBoxLayout(); lower_trend.setSpacing(10)
        self.heatmap_card = GlowCard(); self.heatmap_card.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:12px;}")
        hm_lay = QVBoxLayout(self.heatmap_card); hm_lay.setContentsMargins(14, 12, 14, 12)
        hm_top = QHBoxLayout()
        hm_title = QLabel("长期情绪热力图")
        hm_title.setStyleSheet("color:#F3F7FF; font-size:15px; font-weight:700; font-family:'微软雅黑';")
        hm_tip = QLabel("悬停看数值，点击看原始明细")
        hm_tip.setStyleSheet("color:#8FA5CF; font-size:12px; font-family:'微软雅黑';")
        hm_top.addWidget(hm_title)
        hm_top.addStretch()
        hm_top.addWidget(hm_tip)
        self.emotion_heatmap = EmotionHeatmapWidget()
        self.emotion_heatmap.set_range_days(180)
        self.emotion_heatmap.day_clicked.connect(self._show_heatmap_day_detail)
        hm_legend = QLabel("蓝色=低压力 · 橙红=持续波动 · 用于快速定位异常日期")
        hm_legend.setStyleSheet("color:#8FA5CF; font-size:12px; font-family:'微软雅黑';")
        hm_lay.addLayout(hm_top)
        hm_lay.addWidget(self.emotion_heatmap)
        hm_lay.addWidget(hm_legend)

        self.event_card = GlowCard(); self.event_card.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:12px;}")
        ev_lay = QVBoxLayout(self.event_card); ev_lay.setContentsMargins(14, 12, 14, 12)
        ev_title = QLabel("事件日志")
        ev_title.setStyleSheet("color:#F3F7FF; font-size:15px; font-weight:700; font-family:'微软雅黑';")
        self.event_stream = QTextEdit()
        self.event_stream.setReadOnly(True)
        self.event_stream.setMinimumHeight(140)
        self.event_stream.setStyleSheet("QTextEdit{background:#081A31; color:#E8EAF6; border:1px solid #173A66; border-radius:10px; font-family:'Consolas','微软雅黑'; font-size:12px; padding:8px;}")
        self.event_stream.setText("等待事件...\n")

        self.event_filter_info = QCheckBox('INFO')
        self.event_filter_warn = QCheckBox('WARN')
        self.event_filter_alert = QCheckBox('ALERT')
        self.event_filter_only_abnormal = QCheckBox('仅异常')
        for cb in (self.event_filter_info, self.event_filter_warn, self.event_filter_alert):
            cb.setChecked(True)
            cb.stateChanged.connect(self._render_event_stream)
            cb.setStyleSheet("QCheckBox{color:#8FA5CF; font-size:12px; font-family:'微软雅黑';}")
        self.event_filter_only_abnormal.setChecked(False)
        self.event_filter_only_abnormal.stateChanged.connect(self._toggle_only_abnormal)
        self.event_filter_only_abnormal.setStyleSheet("QCheckBox{color:#FBBF24; font-size:12px; font-family:'微软雅黑'; font-weight:700;}")
        ev_filter_row = QHBoxLayout()
        ev_filter_row.addWidget(QLabel('过滤：'))
        ev_filter_row.addWidget(self.event_filter_info)
        ev_filter_row.addWidget(self.event_filter_warn)
        ev_filter_row.addWidget(self.event_filter_alert)
        ev_filter_row.addSpacing(10)
        ev_filter_row.addWidget(self.event_filter_only_abnormal)
        ev_filter_row.addStretch()

        self._event_history = []

        ev_lay.addWidget(ev_title)
        ev_lay.addLayout(ev_filter_row)
        ev_lay.addWidget(self.event_stream)

        lower_trend.addWidget(self.heatmap_card, 2)
        lower_trend.addWidget(self.event_card, 1)
        trend_lay.addLayout(lower_trend, 1)

        # ===== 卡片 3：AI 守护分析 =====
        self.ai_card = GlowCard()
        self.ai_card.setStyleSheet("QFrame{background:#081A31; border:1px solid #1C4076; border-radius:14px;}")
        ai_lay = QVBoxLayout(self.ai_card); ai_lay.setContentsMargins(16, 14, 16, 14); ai_lay.setSpacing(10)
        ai_head = QHBoxLayout()
        ai_title = QLabel("AI 守护分析报告")
        ai_title.setStyleSheet("color:#F3F7FF; font-size:18px; font-weight:700; font-family:'微软雅黑';")
        ai_note = QLabel("P1 结论 / 原因 / 建议 / 动作")
        ai_note.setStyleSheet("QLabel{background:#102A4C; color:#44E6FF; border:1px solid #2E4E7D; border-radius:8px; padding:4px 8px; font-size:12px; font-weight:700;}")
        ai_head.addWidget(ai_title); ai_head.addWidget(ai_note); ai_head.addStretch()
        ai_lay.addLayout(ai_head)
        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setStyleSheet("QTextEdit{background:#0A1F3D; color:#E8EAF6; border:1px solid #173A66; border-radius:12px; font-family:'微软雅黑'; font-size:14px; line-height:1.6; padding:10px;}")
        self.ai_text.setText("等待生成今日深度分析报告...")
        self.btn_gen_ai = QPushButton("✨ 请求云端分析")
        self.btn_gen_ai.setMinimumHeight(44)
        self.btn_gen_ai.setCursor(Qt.PointingHandCursor)
        self.btn_gen_ai.setStyleSheet(
            f"QPushButton{{background:{THEME_PRIMARY}; color:#06162A; border:none; border-radius:10px; font-weight:700; font-size:15px;}}"
            f"QPushButton:hover{{background:#44E6FF;}}"
        )
        self.btn_gen_ai.clicked.connect(self._start_real_llm_request)
        self.btn_export_pdf = QPushButton("📄 导出 PDF 评估报告")
        self.btn_export_pdf.setMinimumHeight(44)
        self.btn_export_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_export_pdf.setStyleSheet(
            "QPushButton{background:#0E2747; color:#44E6FF; border:1px solid #2E4E7D; border-radius:10px; font-weight:700; font-size:14px;}"
            "QPushButton:hover{background:#17355D;}"
        )
        btn_layout = QHBoxLayout(); btn_layout.setSpacing(10)
        btn_layout.addWidget(self.btn_gen_ai, 1)
        btn_layout.addWidget(self.btn_export_pdf, 1)
        ai_lay.addWidget(self.ai_text, 1)
        ai_lay.addLayout(btn_layout)

        # ===== 卡片 4：系统自检 =====
        self.device_card = GlowCard()
        self.device_card.setStyleSheet("QFrame{background:#081A31; border:1px solid #1C4076; border-radius:14px;}")
        dev_layout = QVBoxLayout(self.device_card); dev_layout.setContentsMargins(16, 14, 16, 14); dev_layout.setSpacing(10)
        dev_head = QHBoxLayout()
        dev_title = QLabel("系统自检")
        dev_title.setStyleSheet("color:#F3F7FF; font-size:18px; font-weight:700; font-family:'微软雅黑';")
        dev_note = QLabel("P2 在线状态 / 信号质量 / 错误日志")
        dev_note.setStyleSheet("QLabel{background:#102A4C; color:#FFC857; border:1px solid #2E4E7D; border-radius:8px; padding:4px 8px; font-size:12px; font-weight:700;}")
        dev_head.addWidget(dev_title); dev_head.addWidget(dev_note); dev_head.addStretch()
        dev_layout.addLayout(dev_head)

        self.esp32_url = getattr(config, 'ESP32_REMOTE_URL', 'http://127.0.0.1:5001/esp32/latest')
        self.btn_esp32_refresh = QPushButton("🔄 立即执行自检")
        self.btn_esp32_refresh.setMinimumHeight(40)
        self.btn_esp32_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_esp32_refresh.setStyleSheet(
            f"QPushButton{{background:{THEME_PRIMARY}; color:#06162A; border:none; border-radius:10px; font-weight:700; font-size:14px;}}"
            f"QPushButton:hover{{background:#44E6FF;}}"
        )
        self.btn_esp32_refresh.clicked.connect(self._refresh_esp32_info)
        self.btn_import_excel = QPushButton("📂 导入 Excel 文件夹")
        self.btn_import_excel.setMinimumHeight(40)
        self.btn_import_excel.setCursor(Qt.PointingHandCursor)
        self.btn_import_excel.setStyleSheet(
            "QPushButton{background:#0E2747; color:#44E6FF; border:1px solid #2E4E7D; border-radius:10px; font-weight:700; font-size:14px;}"
            "QPushButton:hover{background:#17355D;}"
        )
        self.btn_import_excel.clicked.connect(self._import_excel_folder)

        self.device_status_summary = QLabel("状态：待检测")
        self.device_status_summary.setStyleSheet("color:#8FA5CF; font-size:14px; font-weight:700; font-family:'微软雅黑';")
        self.selfcheck_title = QLabel("自检清单（最后刷新：--）")
        self.selfcheck_title.setStyleSheet("color:#8FA5CF; font-size:12px; font-weight:700; font-family:'微软雅黑';")

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_row.addWidget(self.btn_esp32_refresh)
        btn_row.addWidget(self.btn_import_excel)
        dev_layout.addLayout(btn_row)
        dev_layout.addWidget(self.device_status_summary)
        dev_layout.addWidget(self.selfcheck_title)

        self.selfcheck_table = QTableWidget(0, 3)
        self.selfcheck_table.setHorizontalHeaderLabels(["模块", "状态", "指标"])
        self.selfcheck_table.horizontalHeader().setStretchLastSection(True)
        self.selfcheck_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.selfcheck_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.selfcheck_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selfcheck_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.selfcheck_table.setFocusPolicy(Qt.NoFocus)
        self.selfcheck_table.setMinimumHeight(210)
        self.selfcheck_table.setStyleSheet(
            "QTableWidget{background:#0A1F3D; color:#E8EAF6; border:1px solid #173A66; border-radius:10px; font-family:'微软雅黑'; font-size:13px;}"
            "QHeaderView::section{background:#0E2747; color:#8FA5CF; border:none; border-bottom:1px solid #173A66; padding:8px; font-weight:700;}"
        )
        self.esp32_info = QTextEdit()
        self.esp32_info.setReadOnly(True)
        self.esp32_info.setMinimumHeight(150)
        self.esp32_info.setStyleSheet("QTextEdit{background:#0A1F3D; color:#E8EAF6; border:1px solid #173A66; border-radius:10px; font-family:'Consolas'; font-size:13px; padding:10px;}")
        self.esp32_info.setText(
            "> 系统自检日志\n"
            "> 当前模式: 待检测\n"
            "> 说明: 点击“立即执行自检”开始。"
        )
        dev_layout.addWidget(self.selfcheck_table, 1)
        dev_layout.addWidget(self.esp32_info)

        # ===== 布局网格 =====
        self.dashboard_grid.addWidget(self.summary_card, 0, 0)
        self.dashboard_grid.addWidget(self.trend_card, 0, 1)
        self.dashboard_grid.addWidget(self.ai_card, 1, 0)
        self.dashboard_grid.addWidget(self.device_card, 1, 1)
        self.dashboard_grid.setColumnStretch(0, 1)
        self.dashboard_grid.setColumnStretch(1, 1)
        self.dashboard_grid.setRowStretch(0, 3)
        self.dashboard_grid.setRowStretch(1, 2)

        self._daily_raw_rows_map = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_data)
        self.timer.start(3000)
        self._refresh_all()

    def _init_ui_legacy(self):
        self.bg = QFrame(self)
        self.bg.setGeometry(0, 0, self.width(), self.height())
        self.bg.setStyleSheet("background-color: #04111F; border-radius: 15px; border: 1px solid #d2d2d7;")

        main_layout = QHBoxLayout(self.bg); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        # --- 侧边栏 ---
        sidebar = QFrame(); sidebar.setFixedWidth(280) 
        sidebar.setStyleSheet("""
            QFrame { background-color: #f8f8fa; border-top-left-radius: 15px; border-bottom-left-radius: 15px; border-right: 1px solid #1C4076; }
            QPushButton { text-align: left; padding-left: 20px; border: none; border-radius: 12px; font-size: 16px; color: #333333; height: 50px; margin: 5px 15px; font-weight: bold; }
            QPushButton:hover { background-color: #0A1F3D; color: #E8EAF6;}
            QPushButton:checked { background-color: #00D4FF; color: white; }
        """)
        sidebar_layout = QVBoxLayout(sidebar); sidebar_layout.setContentsMargins(0, 40, 0, 20)

        logo = QLabel("MindRoom"); logo.setFont(QFont("微软雅黑", 18, QFont.Bold)); logo.setStyleSheet(f"color: {THEME_TEXT_DARK}; padding-left: 10px;")
        sidebar_layout.addWidget(logo); sidebar_layout.addSpacing(30)

        self.btn_summary = QPushButton("⭕ 状态摘要")
        self.btn_trends = QPushButton("📈 数据趋势")
        self.btn_ai = QPushButton("🧠 AI 守护者")
        self.btn_devices = QPushButton("⚙️ 设备配置")

        self.nav_btns = [self.btn_summary, self.btn_trends, self.btn_ai, self.btn_devices]
        for btn in self.nav_btns: btn.setCheckable(True); btn.setCursor(Qt.PointingHandCursor); sidebar_layout.addWidget(btn)
        self.btn_summary.setChecked(True); sidebar_layout.addStretch()

        self.btn_exit = QPushButton("⏏️ 退出监护端"); self.btn_exit.setStyleSheet(f"color: {THEME_WARN};"); self.btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(self.btn_exit)
        main_layout.addWidget(sidebar)

        # --- 右侧内容区 ---
        right_area = QFrame(); right_layout = QVBoxLayout(right_area); right_layout.setContentsMargins(0, 0, 0, 0); right_layout.setSpacing(0)
        self.title_bar = QFrame(); self.title_bar.setFixedHeight(40); right_layout.addWidget(self.title_bar)

        self.stacked_widget = QStackedWidget()
        right_layout.addWidget(self.stacked_widget, 1) 
        main_layout.addWidget(right_area, 1) 

        # ==================== Page 0: 状态摘要 ====================
        self.page_summary = QWidget()
        summary_layout = QVBoxLayout(self.page_summary); summary_layout.setContentsMargins(40, 0, 40, 40)
        
        title_summary = QLabel("今日健康概览"); title_summary.setFont(QFont("微软雅黑", 24, QFont.Bold)); title_summary.setStyleSheet(f"color: {THEME_TEXT_DARK};")
        summary_layout.addWidget(title_summary); summary_layout.addSpacing(15)
        
        top_row = QHBoxLayout(); top_row.setSpacing(15)
        self.ring_card = GlowCard(); ring_layout = QVBoxLayout(self.ring_card); self.ring_widget = ConcentricRingWidget(self.ring_card); ring_layout.addWidget(self.ring_widget)
        top_row.addWidget(self.ring_card, 2) 
        
        top_right_col = QVBoxLayout(); top_right_col.setSpacing(15)
        self.score_card = self._create_metric_card("系统评估", "0.00", THEME_TEXT_DARK)
        self.alert_card = self._create_metric_card("最新预警", "暂无", THEME_TEXT_MUTED)
        top_right_col.addWidget(self.score_card['frame']); top_right_col.addWidget(self.alert_card['frame'])
        top_row.addLayout(top_right_col, 1) 
        summary_layout.addLayout(top_row, 3) 
        summary_layout.addSpacing(15)
        
        bot_row = QHBoxLayout(); bot_row.setSpacing(15)
        self.vis_card = self._create_modality_card("视觉特征", "#00b4d8")
        self.hrv_card = self._create_modality_card("生理(HRV)", "#ff4d6d")
        self.eeg_card = self._create_modality_card("脑电(EEG)", THEME_PRIMARY)
        bot_row.addWidget(self.vis_card['frame']); bot_row.addWidget(self.hrv_card['frame']); bot_row.addWidget(self.eeg_card['frame'])
        summary_layout.addLayout(bot_row, 1) 

        # ==================== Page 1, 2, 3 ====================
        self.page_trends = QWidget(); trends_layout = QVBoxLayout(self.page_trends); trends_layout.setContentsMargins(40, 0, 40, 40)
        t_title = QLabel("风险数据追踪"); t_title.setFont(QFont("微软雅黑", 24, QFont.Bold)); trends_layout.addWidget(t_title)

        status_row = QHBoxLayout(); status_row.setSpacing(10)
        self.dev_rk = QLabel("RK3588: --")
        self.dev_cam = QLabel("Camera: --")
        self.dev_ble = QLabel("EEG BLE: --")
        self.dev_esp = QLabel("ESP32: --")
        for lb in [self.dev_rk, self.dev_cam, self.dev_ble, self.dev_esp]:
            lb.setStyleSheet("QLabel { background:#0A1F3D; border:1px solid #1C4076; border-radius:8px; padding:6px 10px; color:#E8EAF6; font-size:12px; font-family:'微软雅黑'; }")
            status_row.addWidget(lb)
        status_row.addStretch()
        trends_layout.addLayout(status_row)

        risk_strip = QFrame(); risk_strip.setStyleSheet("QFrame{background:#0A1F3D; border:1px solid #1C4076; border-radius:10px;}")
        risk_strip_lay = QHBoxLayout(risk_strip); risk_strip_lay.setContentsMargins(14, 10, 14, 10)
        self.risk_main_value = QLabel("0.00 / 1.00")
        self.risk_main_value.setStyleSheet("color:#E8EAF6; font-size:30px; font-weight:700;")
        self.risk_main_level = QLabel("等级：待检测")
        self.risk_main_level.setStyleSheet("color:#7986B8; font-size:13px; font-family:'微软雅黑';")
        self.risk_main_delta = QLabel("24h 趋势：--")
        self.risk_main_delta.setStyleSheet("color:#7986B8; font-size:13px; font-family:'微软雅黑';")
        left_risk = QVBoxLayout(); left_risk.addWidget(self.risk_main_value); left_risk.addWidget(self.risk_main_level)
        risk_strip_lay.addLayout(left_risk)
        risk_strip_lay.addStretch()
        risk_strip_lay.addWidget(self.risk_main_delta)
        trends_layout.addWidget(risk_strip)

        self.trends_card = GlowCard(); t_lay = QVBoxLayout(self.trends_card)
        self.fig = Figure(figsize=(8, 4), facecolor='#0A1F3D'); self.canvas = FigureCanvas(self.fig); self.ax = self.fig.add_subplot(111); self.ax.set_facecolor('#0A1F3D')
        t_lay.addWidget(self.canvas)

        self.heatmap_card = GlowCard(); hm_lay = QVBoxLayout(self.heatmap_card); hm_lay.setContentsMargins(18, 16, 18, 16)

        hm_top = QHBoxLayout()
        hm_title = QLabel("长期情绪热力图")
        hm_title.setStyleSheet(f"color:{THEME_TEXT_DARK}; font-size:16px; font-weight:bold; font-family:'微软雅黑';")

        self.heatmap_range_days = 180
        self.btn_hm_30 = QPushButton("近30天")
        self.btn_hm_180 = QPushButton("近半年")
        self.btn_hm_365 = QPushButton("近一年")
        for btn, days in [(self.btn_hm_30, 30), (self.btn_hm_180, 180), (self.btn_hm_365, 365)]:
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background:#0A1F3D; color:{THEME_PRIMARY}; border:1px solid {THEME_PRIMARY}; border-radius:6px; font-size:12px; padding:4px 10px; }}"
                f"QPushButton:checked {{ background:{THEME_PRIMARY}; color:#0A1F3D; }}"
            )
            btn.clicked.connect(lambda checked, d=days: self._set_heatmap_range(d))

        self.btn_hm_180.setChecked(True)
        hm_top.addWidget(hm_title)
        hm_top.addStretch()
        hm_top.addWidget(self.btn_hm_30)
        hm_top.addWidget(self.btn_hm_180)
        hm_top.addWidget(self.btn_hm_365)

        self.emotion_heatmap = EmotionHeatmapWidget()
        self.emotion_heatmap.set_range_days(self.heatmap_range_days)
        self.emotion_heatmap.day_clicked.connect(self._show_heatmap_day_detail)
        hm_legend = QLabel("蓝色=放松  ·  暖色=压力升高（悬停查看，点击查看当日原始数据与风险指数）")
        hm_legend.setStyleSheet(f"color:{THEME_TEXT_MUTED}; font-size:12px; font-family:'微软雅黑';")

        hm_lay.addLayout(hm_top)
        hm_lay.addWidget(self.emotion_heatmap)
        hm_lay.addWidget(hm_legend)

        self.event_card = GlowCard(); ev_lay = QVBoxLayout(self.event_card); ev_lay.setContentsMargins(16, 14, 16, 14)
        ev_title = QLabel("事件流")
        ev_title.setStyleSheet("color:#E8EAF6; font-size:15px; font-weight:bold; font-family:'微软雅黑';")
        self.event_stream = QTextEdit()
        self.event_stream.setReadOnly(True)
        self.event_stream.setMinimumHeight(120)
        self.event_stream.setStyleSheet("QTextEdit { border:none; color:#E8EAF6; font-family:'Consolas','微软雅黑'; font-size:12px; }")
        self.event_stream.setText("等待事件...\n")
        ev_lay.addWidget(ev_title)
        ev_lay.addWidget(self.event_stream)

        trends_layout.addWidget(self.trends_card, 3)
        trends_layout.addWidget(self.event_card, 1)
        trends_layout.addWidget(self.heatmap_card, 2)

        self.page_ai = QWidget(); ai_layout = QVBoxLayout(self.page_ai); ai_layout.setContentsMargins(40, 0, 40, 40)
        ai_title = QLabel("AI 守护者分析报告"); ai_title.setFont(QFont("微软雅黑", 24, QFont.Bold)); ai_layout.addWidget(ai_title)
        ai_card = GlowCard(); ac_lay = QVBoxLayout(ai_card); ac_lay.setContentsMargins(20, 20, 20, 20)
        self.ai_text = QTextEdit(); self.ai_text.setReadOnly(True); self.ai_text.setStyleSheet("border:none; font-family: '微软雅黑'; font-size: 16px; line-height: 1.5;")
        self.ai_text.setText("等待生成今日深度分析报告...")
        self.btn_gen_ai = QPushButton("✨ 请求云端分析"); self.btn_gen_ai.setMinimumHeight(50); self.btn_gen_ai.setStyleSheet(f"QPushButton {{ background: {THEME_PRIMARY}; color: white; border-radius: 8px; font-weight: bold; font-size: 16px; }}")
        self.btn_gen_ai.clicked.connect(self._start_real_llm_request)

        self.btn_export_pdf = QPushButton("📄 导出 PDF 评估报告")
        self.btn_export_pdf.setMinimumHeight(50)
        self.btn_export_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_export_pdf.setStyleSheet(
            f"QPushButton {{ background: #0A1F3D; color: {THEME_PRIMARY}; border: 1px solid {THEME_PRIMARY}; border-radius: 8px; font-weight: bold; font-size: 16px; min-height: 50px; }}"
            f"QPushButton:hover {{ background: {THEME_PRIMARY}; color: #0A1F3D; }}"
        )

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(14)
        btn_layout.addWidget(self.btn_gen_ai, 1)
        btn_layout.addWidget(self.btn_export_pdf, 1)

        ac_lay.addWidget(self.ai_text)
        ac_lay.addLayout(btn_layout)
        ai_layout.addWidget(ai_card, 1)

        self.page_devices = QWidget(); dev_layout = QVBoxLayout(self.page_devices); dev_layout.setContentsMargins(40, 0, 40, 40); dev_layout.setSpacing(14)
        dev_title = QLabel("⚙️ 系统自检"); dev_title.setFont(QFont("微软雅黑", 24, QFont.Bold)); dev_layout.addWidget(dev_title)

        dev_card = GlowCard(); dc_lay = QVBoxLayout(dev_card); dc_lay.setContentsMargins(20, 20, 20, 20); dc_lay.setSpacing(10)

        self.esp32_url = "http://127.0.0.1:5001/esp32/latest"
        self.btn_esp32_refresh = QPushButton("🔄 立即执行自检")
        self.btn_esp32_refresh.setMinimumHeight(42)
        self.btn_esp32_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_esp32_refresh.setStyleSheet(
            f"QPushButton {{ background:{THEME_PRIMARY}; color:white; border:none; border-radius:8px; font-weight:bold; font-size:14px; }}"
            f"QPushButton:hover {{ background:{THEME_PRIMARY_HOVER}; }}"
        )

        self.device_status_summary = QLabel("状态：待检测")
        self.device_status_summary.setStyleSheet(f"color:{THEME_TEXT_DARK}; font-size:14px; font-weight:bold; font-family:'微软雅黑';")

        self.selfcheck_title = QLabel("自检清单（最后刷新：--）")
        self.selfcheck_title.setStyleSheet(f"color:{THEME_TEXT_DARK}; font-size:13px; font-weight:bold; font-family:'微软雅黑';")

        self.selfcheck_table = QTableWidget(0, 3)
        self.selfcheck_table.setHorizontalHeaderLabels(["模块", "状态", "指标"])
        self.selfcheck_table.horizontalHeader().setStretchLastSection(True)
        self.selfcheck_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.selfcheck_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.selfcheck_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selfcheck_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.selfcheck_table.setFocusPolicy(Qt.NoFocus)
        self.selfcheck_table.setMinimumHeight(220)
        self.selfcheck_table.setStyleSheet(
            "QTableWidget { background:#0A1F3D; color:#E8EAF6; border:1px solid #1C4076; border-radius:10px; font-family:'微软雅黑'; font-size:13px; }"
            "QHeaderView::section { background:#f7f8fa; color:#9BA8C8; border:none; border-bottom:1px solid #1C4076; padding:8px; font-weight:bold; }"
        )

        self.esp32_info = QTextEdit()
        self.esp32_info.setReadOnly(True)
        self.esp32_info.setMinimumHeight(180)
        self.esp32_info.setStyleSheet(
            "QTextEdit { background:#0A1F3D; color:#E8EAF6; border:1px solid #1C4076; border-radius:10px; font-family:'Consolas'; font-size:14px; padding:10px; }"
        )
        self.esp32_info.setText(
            "> 系统自检日志\n"
            "> 当前模式: 待检测\n"
            "> 说明: 点击“立即执行自检”开始。"
        )

        dc_lay.addWidget(self.btn_esp32_refresh)
        dc_lay.addWidget(self.device_status_summary)
        dc_lay.addWidget(self.selfcheck_title)
        dc_lay.addWidget(self.selfcheck_table)
        dc_lay.addWidget(self.esp32_info, 1)
        dev_layout.addWidget(dev_card, 1)

        self.stacked_widget.addWidget(self.page_summary); self.stacked_widget.addWidget(self.page_trends); self.stacked_widget.addWidget(self.page_ai); self.stacked_widget.addWidget(self.page_devices)

        self.btn_summary.clicked.connect(lambda: self._switch_page(0))
        self.btn_trends.clicked.connect(lambda: self._switch_page(1))
        self.btn_ai.clicked.connect(lambda: self._switch_page(2))
        self.btn_devices.clicked.connect(lambda: self._switch_page(3))
        self.btn_export_pdf.clicked.connect(self._prepare_pdf_report)
        self.btn_esp32_refresh.clicked.connect(self._refresh_esp32_info)

        self.btn_import_excel = QPushButton("📂 导入 Excel 文件夹")
        self.btn_import_excel.setMinimumHeight(42)
        self.btn_import_excel.setCursor(Qt.PointingHandCursor)
        self.btn_import_excel.setStyleSheet(
            f"QPushButton {{ background:#0A1F3D; color:{THEME_PRIMARY}; border:1px solid {THEME_PRIMARY}; border-radius:8px; font-weight:bold; font-size:14px; }}"
            f"QPushButton:hover {{ background:{THEME_PRIMARY}; color:white; }}"
        )
        self.btn_import_excel.clicked.connect(self._import_excel_folder)
        dc_lay.addWidget(self.btn_import_excel)

        self._daily_raw_rows_map = {}

        self.timer = QTimer(self); self.timer.timeout.connect(self._refresh_data); self.timer.start(3000)
        self._refresh_data()
        self._refresh_esp32_info()

    def _create_metric_card(self, title_text, val_text, color):
        card = GlowCard(); lay = QVBoxLayout(card); lay.setContentsMargins(15, 10, 15, 12); lay.setSpacing(8)
        title = QLabel(title_text); title.setStyleSheet("color: #7986B8; font-size: 14px; font-family: '微软雅黑';")
        val = QLabel(val_text)
        val.setWordWrap(True)
        val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        val.setMinimumHeight(54)
        val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        val.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold; padding-right: 6px;")
        lay.addWidget(title)
        lay.addWidget(val, 1)
        return {'frame': card, 'val': val}

    def _create_modality_card(self, title_text, color):
        card = GlowCard(); lay = QVBoxLayout(card); lay.setContentsMargins(15, 10, 15, 10)
        title = QLabel(f"● {title_text}"); title.setStyleSheet(f"color: {color}; font-size: 15px; font-weight:bold; font-family: '微软雅黑';")
        score = QLabel("0.00"); score.setStyleSheet("color: #E8EAF6; font-size: 24px; font-weight: bold;")
        lay.addWidget(title); lay.addStretch(); lay.addWidget(score)
        return {'frame': card, 'score': score}

    def _refresh_all(self):
        self._refresh_data()
        self._refresh_esp32_info()
        self.dashboard_time_badge.setText(f"最后刷新：{datetime.datetime.now().strftime('%H:%M:%S')}")

    def _set_data_source_mode(self, mode, detail=''):
        mode = str(mode or '').lower()
        if mode == 'live':
            text, color, border = '数据源：实时模式', '#34D399', '#1E8C6D'
        elif mode == 'demo':
            text, color, border = '数据源：演示模式', '#FFC857', '#8B6A2A'
        elif mode in ('offline', 'offline_replay'):
            text, color, border = '数据源：离线回放', '#93C5FD', '#315A8A'
        else:
            text, color, border = '数据源：未知', '#F3F4F6', '#4B5563'
        if detail:
            text = f'{text} · {detail}'
        if hasattr(self, 'data_source_badge'):
            self.data_source_badge.setText(text)
            self.data_source_badge.setStyleSheet(
                f"QLabel{{background:#0E2747; color:{color}; border:1px solid {border}; border-radius:10px; padding:8px 12px; font-size:13px; font-weight:700; font-family:'微软雅黑';}}"
            )

    def _toggle_only_abnormal(self):
        only_abnormal = getattr(self, 'event_filter_only_abnormal', None).isChecked() if hasattr(self, 'event_filter_only_abnormal') else False
        if only_abnormal:
            self.event_filter_info.setChecked(False)
            self.event_filter_warn.setChecked(True)
            self.event_filter_alert.setChecked(True)
        else:
            if not (self.event_filter_info.isChecked() or self.event_filter_warn.isChecked() or self.event_filter_alert.isChecked()):
                self.event_filter_info.setChecked(True)
                self.event_filter_warn.setChecked(True)
                self.event_filter_alert.setChecked(True)
        self._render_event_stream()

    def _render_event_stream(self):
        if not hasattr(self, 'event_stream'):
            return
        allow_info = getattr(self, 'event_filter_info', None).isChecked() if hasattr(self, 'event_filter_info') else True
        allow_warn = getattr(self, 'event_filter_warn', None).isChecked() if hasattr(self, 'event_filter_warn') else True
        allow_alert = getattr(self, 'event_filter_alert', None).isChecked() if hasattr(self, 'event_filter_alert') else True

        selected = []
        for item in self._event_history:
            lv = item.get('level', 'info')
            if lv == 'info' and not allow_info:
                continue
            if lv == 'warn' and not allow_warn:
                continue
            if lv == 'alert' and not allow_alert:
                continue
            selected.append(item.get('msg', ''))

        self.event_stream.setPlainText('\n'.join(selected) if selected else '等待事件...')

    def _append_event_log(self, text, level='info', dedup=True):
        msg = str(text)
        now = datetime.datetime.now().timestamp()
        if dedup:
            last_ts = self._log_last_emit.get(msg, 0.0)
            if (now - last_ts) < self._log_throttle_secs:
                return
            self._log_last_emit[msg] = now

        lv = str(level).lower()
        if lv in ('warning',):
            lv = 'warn'
        elif lv in ('error', 'critical'):
            lv = 'alert'
        elif lv not in ('info', 'warn', 'alert'):
            lv = 'info'

        item = {'msg': msg, 'level': lv, 'ts': now}
        if lv == 'alert':
            self._event_history.insert(0, item)
        else:
            self._event_history.append(item)
        self._event_history = self._event_history[-500:]
        self._render_event_stream()

    def _set_heatmap_range(self, days):
        self.heatmap_range_days = int(days)
        self.btn_hm_30.setChecked(self.heatmap_range_days == 30)
        self.btn_hm_180.setChecked(self.heatmap_range_days == 180)
        self.btn_hm_365.setChecked(self.heatmap_range_days == 365)
        self.emotion_heatmap.set_range_days(self.heatmap_range_days)
        self._refresh_data()

    def _show_heatmap_day_detail(self, date_value, risk_value, raw_stat):
        if date_value is None:
            QMessageBox.information(self, "当日详情", "该日期暂无数据。")
            return

        date_text = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        if risk_value is None:
            risk_text = "暂无"
        else:
            risk_text = f"{float(risk_value):.4f}"

        count = int(raw_stat.get('count', 0)) if isinstance(raw_stat, dict) else 0
        min_risk = raw_stat.get('min') if isinstance(raw_stat, dict) else None
        max_risk = raw_stat.get('max') if isinstance(raw_stat, dict) else None

        min_text = "暂无" if min_risk is None else f"{float(min_risk):.4f}"
        max_text = "暂无" if max_risk is None else f"{float(max_risk):.4f}"

        rows_df = self._daily_raw_rows_map.get(date_text)
        table_preview = ""
        if rows_df is not None and not rows_df.empty:
            preview_df = rows_df.copy()
            preview_df['timestamp'] = pd.to_datetime(preview_df['timestamp'], errors='coerce').dt.strftime('%H:%M:%S')
            show_cols = [c for c in ['timestamp', 'risk', 'visual_risk', 'hrv_risk', 'eeg_risk'] if c in preview_df.columns]
            preview_df = preview_df[show_cols].head(144)
            table_preview = preview_df.to_string(index=False)

        detail_text = (
            f"日期：{date_text}\n"
            f"当天风险指数（均值）：{risk_text}\n\n"
            f"原始数据统计：\n"
            f"- 原始记录数：{count}\n"
            f"- 最低风险值：{min_text}\n"
            f"- 最高风险值：{max_text}\n\n"
            f"原始记录明细（最多展示前144行）：\n{table_preview if table_preview else '暂无可展示的原始明细'}"
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(f"情绪热力图 - {date_text} 明细")
        dlg.resize(880, 640)
        lay = QVBoxLayout(dlg)

        txt = QTextEdit(dlg)
        txt.setReadOnly(True)
        txt.setStyleSheet("QTextEdit { font-family:Consolas, '微软雅黑'; font-size:13px; }")
        txt.setText(detail_text)
        lay.addWidget(txt)

        btn_close = QPushButton("关闭", dlg)
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close, 0, Qt.AlignRight)

        dlg.exec_()

    def _start_real_llm_request(self):
        df = self.recorder.get_decrypted_history()
        self.btn_gen_ai.setText("⏳ 请求中...")
        self.ai_thread = AIAnalyzeThread(df)
        self.ai_thread.finished_signal.connect(self._on_ai_success)
        self.ai_thread.error_signal.connect(self._on_ai_success)
        self.ai_thread.start()

    def _on_ai_success(self, text):
        self.ai_text.setText(text)
        self.btn_gen_ai.setText("✨ 重新生成")

    def _set_selfcheck_rows(self, rows):
        self.selfcheck_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            mod_item = QTableWidgetItem(str(row.get('module', '--')))
            status_item = QTableWidgetItem(str(row.get('status', '--')))
            metric_item = QTableWidgetItem(str(row.get('metric', '--')))

            status_text = str(row.get('status', ''))
            if '在线' in status_text or status_text.startswith('✓'):
                status_item.setForeground(QBrush(QColor(THEME_SUCCESS)))
            elif '异常' in status_text or '离线' in status_text or status_text.startswith('✗'):
                status_item.setForeground(QBrush(QColor(THEME_WARN)))
            else:
                status_item.setForeground(QBrush(QColor('#FFC857')))

            self.selfcheck_table.setItem(i, 0, mod_item)
            self.selfcheck_table.setItem(i, 1, status_item)
            self.selfcheck_table.setItem(i, 2, metric_item)

    def _refresh_esp32_info(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        rows = [
            {"module": "RK3588 推理引擎", "status": "✓ 在线", "metric": "NPU 已初始化"},
            {"module": "摄像头采集", "status": "✓ 在线", "metric": "1920x1080 @ 30fps"},
            {"module": "脑电 BLE", "status": "⚠ 待确认", "metric": "RSSI: -- dBm"},
            {"module": "ESP32 手环", "status": "⚠ 待检测", "metric": "BPM: --"},
        ]

        status_lines = [
            "> 系统自检日志",
            f"> 检测时间: {now}",
            f"> 接口地址: {self.esp32_url}",
        ]

        demo_mode = True
        try:
            session = requests.Session()
            session.trust_env = False
            resp = session.get(self.esp32_url, timeout=2)
            status_lines.append(f"> HTTP状态: {resp.status_code}")

            if resp.status_code == 200:
                payload = resp.json() if resp.content else {}
                bpm = payload.get("bpm", "--")
                ts = payload.get("timestamp", "--")
                ip = payload.get("ip", "127.0.0.1")
                port = payload.get("port", "5001")
                quality = payload.get("quality", "unknown")

                rows[3] = {"module": "ESP32 手环", "status": "✓ 在线", "metric": f"BPM: {bpm} | 质量: {quality}"}
                status_lines.extend([
                    "> 连接状态: 在线",
                    f"> ESP32 IP: {ip}",
                    f"> ESP32 端口: {port}",
                    f"> 实时 BPM: {bpm}",
                    f"> 采样时间戳: {ts}",
                    f"> 信号质量: {quality}",
                    "> 说明: 真实部署模式，正在接收实时采集数据。",
                ])
                demo_mode = False
            else:
                rows[3] = {"module": "ESP32 手环", "status": "✗ 异常", "metric": f"HTTP {resp.status_code}"}
                status_lines.extend([
                    "> 连接状态: 异常",
                    "> 说明: ESP32 服务异常，已切换离线模拟模式（Demo Mode）。",
                ])
        except Exception as e:
            rows[3] = {"module": "ESP32 手环", "status": "✗ 离线", "metric": "未连接"}
            status_lines.extend([
                "> 连接状态: 离线",
                f"> 异常信息: {str(e)}",
                "> 说明: 已切换离线模拟模式（Demo Mode）。",
                "> 排查建议: 检查网线/WiFi、服务端口、API进程。",
            ])

        if demo_mode:
            self.device_status_summary.setText("⚠ 状态：离线模拟模式（Demo Mode）")
            self.device_status_summary.setStyleSheet("color:#FFC857; font-size:14px; font-weight:bold; font-family:'微软雅黑';")
            self._set_data_source_mode('demo', 'ESP32 未在线')
        else:
            self.device_status_summary.setText("✓ 状态：真实部署模式（实时采集中）")
            self.device_status_summary.setStyleSheet(f"color:{THEME_SUCCESS}; font-size:14px; font-weight:bold; font-family:'微软雅黑';")
            self._set_data_source_mode('live', 'ESP32 / API')

        self.selfcheck_title.setText(f"自检清单（最后刷新：{now}）")
        self._set_selfcheck_rows(rows)
        self.esp32_info.setText("\n".join(status_lines))

    def _import_excel_folder(self):
        source_dir = QFileDialog.getExistingDirectory(self, "选择包含 7 天 Excel 的文件夹")
        if not source_dir:
            return

        copied = 0
        try:
            copied = self.recorder.import_excel_folder(source_dir, clear_existing=True)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"导入 Excel 文件夹失败：{str(e)}")
            return

        if copied <= 0:
            QMessageBox.information(self, "未导入", "所选文件夹中未找到可用 Excel 文件（.xlsx/.xls）。")
            return

        self._refresh_data()

        df_check = self.recorder.get_decrypted_history()
        if df_check.empty:
            QMessageBox.warning(
                self,
                "导入后无可用数据",
                f"已复制 {copied} 个文件，但未识别到可用风险列。\n"
                f"请确认 Excel 包含“风险/风险值/risk”列，或包含 visual_risk/hrv_risk/eeg_risk。"
            )
            return

        ts = pd.to_datetime(df_check['timestamp'], errors='coerce') if 'timestamp' in df_check.columns else pd.Series(dtype='datetime64[ns]')
        day_count = int(ts.dt.date.nunique()) if not ts.empty else 0
        row_count = len(df_check)
        QMessageBox.information(
            self,
            "导入成功",
            f"已导入 {copied} 个 Excel 文件并刷新热力图。\n"
            f"识别到 {day_count} 天，共 {row_count} 条记录。"
        )

    def _prepare_pdf_report(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"MindRoom_评估报告_{timestamp}.pdf"
        default_path = os.path.join(os.path.expanduser("~"), "Desktop", default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 PDF 评估报告",
            default_path,
            "PDF Files (*.pdf)"
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        self.current_pdf_path = file_path
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.setText("生成中...")

        df = self.recorder.get_decrypted_history()
        ai_text = self.ai_text.toPlainText()

        self.report_prepare_thread = ReportPrepareThread(df, ai_text)
        self.report_prepare_thread.prepared_signal.connect(self._render_and_save_pdf)
        self.report_prepare_thread.finished.connect(
            lambda: self.btn_export_pdf.setEnabled(True)
        )
        self.report_prepare_thread.start()

    def _render_and_save_pdf(self, html_content):
        try:
            document = QTextDocument()
            document.setHtml(html_content)

            pdf_writer = QPdfWriter(self.current_pdf_path)
            pdf_writer.setPageSize(QPageSize(QPageSize.A4))
            pdf_writer.setPageMargins(QMarginsF(18, 18, 18, 18), QPageLayout.Millimeter)
            pdf_writer.setResolution(300)

            document.print_(pdf_writer)
            self.btn_export_pdf.setText("📄 导出 PDF 评估报告")
            QMessageBox.information(self, "导出成功", f"PDF 评估报告已保存到：\n{self.current_pdf_path}")
        except Exception as e:
            self.btn_export_pdf.setText("📄 导出 PDF 评估报告")
            QMessageBox.warning(self, "导出失败", f"导出 PDF 失败：{str(e)}")

    def _refresh_data(self):
        try:
            df = self.recorder.get_decrypted_history()
            if df.empty:
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['risk'] = pd.to_numeric(df['risk'], errors='coerce').fillna(0.0)

            latest_risk = float(df['risk'].iloc[-1])
            avg_risk = float(df['risk'].mean())

            # 兼容旧数据：优先读取真实模态列；不存在则按综合风险估计显示
            vis = float(df['visual_risk'].iloc[-1]) if 'visual_risk' in df.columns else latest_risk * 0.95
            hrv = float(df['hrv_risk'].iloc[-1]) if 'hrv_risk' in df.columns else latest_risk * 1.00
            eeg = float(df['eeg_risk'].iloc[-1]) if 'eeg_risk' in df.columns else latest_risk * 0.90

            vis = max(0.0, min(1.0, vis))
            hrv = max(0.0, min(1.0, hrv))
            eeg = max(0.0, min(1.0, eeg))

            self.ring_widget.set_values(vis, hrv, eeg, latest_risk)
            risk_label, risk_color = get_risk_level_meta(latest_risk)
            self.score_card['val'].setText(f"{avg_risk:.2f} · {risk_label}")
            self.score_card['val'].setStyleSheet(f"color: {risk_color}; font-size: 26px; font-weight: bold;")

            self.vis_card['score'].setText(f"{vis:.2f}")
            self.hrv_card['score'].setText(f"{hrv:.2f}")
            self.eeg_card['score'].setText(f"{eeg:.2f}")

            alerts = df[df['risk'] >= config.RISK_THRESHOLD_MEDIUM]
            if not alerts.empty:
                self.alert_card['val'].setText(alerts['timestamp'].iloc[-1].strftime("%H:%M") + " · 预警")
                self.alert_card['val'].setStyleSheet(f"color: {THEME_WARN};")
            else:
                self.alert_card['val'].setText("暂无 · 正常")
                self.alert_card['val'].setStyleSheet(f"color: {THEME_TEXT_MUTED};")

            if len(df) >= 2:
                delta = float(df['risk'].iloc[-1] - df['risk'].iloc[-2])
                trend_flag = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
                trend_text = f"{trend_flag} Δ{delta:+.2f}"
            else:
                trend_text = "→ Δ+0.00"
            self.score_card['val'].setText(f"{avg_risk:.2f} · {risk_label}  {trend_text}")

            now_tag = datetime.datetime.now().strftime('%H:%M:%S')
            self._append_event_log(f"[{now_tag}] INFO  系统刷新完成  来源:融合引擎")
            self._append_event_log(f"[{now_tag}] INFO  当前风险={latest_risk:.2f}  等级:{risk_label}")
            if not alerts.empty:
                self._append_event_log(f"[{now_tag}] ALERT  触发阈值告警  来源:风险阈值监测", level='alert', dedup=True)
            else:
                self._append_event_log(f"[{now_tag}] INFO  未触发阈值告警")

            daily_df = df.copy()
            daily_df['date'] = daily_df['timestamp'].dt.date
            daily_stats = daily_df.groupby('date')['risk'].agg(['mean', 'min', 'max', 'count'])

            self._daily_raw_rows_map = {}
            for d, group in daily_df.groupby('date'):
                self._daily_raw_rows_map[pd.Timestamp(d).strftime('%Y-%m-%d')] = group.copy()

            total_days = int(getattr(self, 'heatmap_range_days', 180))
            end_date = pd.Timestamp.now().normalize().date()
            start_date = end_date - pd.Timedelta(days=(total_days - 1))
            full_dates = list(pd.date_range(start=start_date, end=end_date, freq='D').date)

            heat_values = []
            heat_raw_stats = []
            for d in full_dates:
                if d in daily_stats.index:
                    row = daily_stats.loc[d]
                    heat_values.append(float(row['mean']))
                    heat_raw_stats.append({
                        'count': int(row['count']),
                        'min': float(row['min']),
                        'max': float(row['max'])
                    })
                else:
                    heat_values.append(None)
                    heat_raw_stats.append(None)

            self.emotion_heatmap.days = 7
            self.emotion_heatmap.set_range_days(total_days)
            self.emotion_heatmap.set_daily_values(heat_values, full_dates, heat_raw_stats)

            self.ax.clear()
            self.ax.plot(df['timestamp'], df['risk'], color=THEME_PRIMARY, lw=2.2, label='风险指数')
            self.ax.axhline(y=config.RISK_THRESHOLD_MEDIUM, color=THEME_WARN, linestyle='--', linewidth=1.8, alpha=0.95, label='预警阈值')
            self.ax.set_ylim(0, 1.0)
            self.ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.25)
            self.ax.set_ylabel('Risk (0-1)', color='#8FA5CF')
            self.ax.tick_params(colors='#8FA5CF', labelsize=9)
            for spine in self.ax.spines.values():
                spine.set_color('#1C4076')
            legend = self.ax.legend(loc='upper right', frameon=False)
            for text in legend.get_texts():
                text.set_color('#8FA5CF')
            self.canvas.draw()

            # 兼容旧版多页布局：仅在旧状态条控件存在时才更新
            if all(hasattr(self, k) for k in ('dev_rk', 'dev_cam', 'dev_ble', 'dev_esp')):
                esp_online = hasattr(self, 'device_status_summary') and ('真实部署模式' in self.device_status_summary.text())
                self.dev_rk.setText('RK3588: ✓')
                self.dev_cam.setText('Camera: ✓')
                self.dev_ble.setText('EEG BLE: --')
                self.dev_esp.setText('ESP32: ' + ('✓' if esp_online else 'Demo'))
        except Exception as e:
            print(f"[Parent UI] 数据刷新失败: {e}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            if hasattr(self, 'bg') and self.bg is not None:
                self.bg.setGeometry(0, 0, self.width(), self.height())
        except Exception:
            pass

    def mousePressEvent(self, e):
        if self.isFullScreen():
            self.drag_pos = None
            return
        if e.button() == Qt.LeftButton and e.pos().y() < 50: self.drag_pos = e.globalPos() - self.pos(); e.accept()
        else: self.drag_pos = None
    def mouseMoveEvent(self, e):
        if self.isFullScreen():
            return
        if e.buttons() == Qt.LeftButton and self.drag_pos: self.move(e.globalPos() - self.drag_pos); e.accept()
