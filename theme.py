"""
统一主题 - NASA 任务控制台风（D:/G/3 工业化大屏版）。
约束（来自论文文档）:
  - 风险四档: 低<0.3 / 中 0.3-0.6 / 高 0.6-0.8 / 极高 >=0.8
  - 三模态权重: 视觉 0.35 / HRV 0.40 / EEG 0.25
"""

# ============== 主色板 - NASA 深海军蓝 ==============
C_BG        = '#04111f'
C_BG_DEEP   = '#020912'
C_PANEL     = '#0a1f3d'
C_PANEL_HI  = '#13315c'
C_BORDER    = '#1c4076'
C_BORDER_HI = '#2a5ca8'

C_ACCENT    = '#00e5ff'
C_ACCENT2   = '#7c4dff'

C_VISION    = '#ffb74d'
C_HEART     = '#ff5277'
C_BRAIN     = '#b388ff'

C_RISK_LOW   = '#26d07c'
C_RISK_MED   = '#ffc857'
C_RISK_HIGH  = '#ff8a3d'
C_RISK_CRIT  = '#ff3860'

C_OK        = C_RISK_LOW
C_WARN      = C_RISK_MED
C_RISK      = C_RISK_CRIT

C_TEXT      = '#eef1ff'
C_TEXT_DIM  = '#c2cbe8'
C_DIM       = '#a4b1d8'   # WCAG AA on C_BG (#04111f): 6.4:1, 提升自原 #7986b8 (4.0:1)

C_PPG       = C_HEART
C_ECG       = '#ff7eb3'
C_HR        = C_HEART
C_HRV       = '#ff9bb6'
C_RAW       = '#ffb74d'
C_ATT       = '#26d07c'
C_MED       = '#448aff'
C_NOISE     = C_RISK_CRIT

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

# ============== 工业级设计 tokens（UI-UX-Pro-Max 规范）==============
# 间距栅格：4 的倍数，避免随手写出 5/7/13 这类不可预期值
SP_XS = 4
SP_SM = 8
SP_MD = 12
SP_LG = 16
SP_XL = 24
SP_XXL = 32

# 字号层级：收敛到 7 档，避免上下文里字号膨胀
FS_TINY = 9
FS_XS   = 10
FS_SM   = 12
FS_MD   = 14
FS_LG   = 18
FS_XL   = 24
FS_XXL  = 36
FS_HERO = 54

# 圆角与边框
R_SM = 6
R_MD = 10
R_LG = 14
BORDER_W = 1


def risk_tier(score):
    if score < 0.3:
        return ('低 · LOW', C_RISK_LOW)
    if score < 0.6:
        return ('中 · MEDIUM', C_RISK_MED)
    if score < 0.8:
        return ('高 · HIGH', C_RISK_HIGH)
    return ('极高 · CRITICAL', C_RISK_CRIT)


def risk_tier_glyph(score):
    """返回 (文本, 颜色, 字形)。字形提供形状冗余，色盲友好。"""
    if score < 0.3:
        return ('低 · LOW', C_RISK_LOW, '●')
    if score < 0.6:
        return ('中 · MEDIUM', C_RISK_MED, '▲')
    if score < 0.8:
        return ('高 · HIGH', C_RISK_HIGH, '◆')
    return ('极高 · CRITICAL', C_RISK_CRIT, '■')


GLOBAL_QSS = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    font-size: {FS_SM}px;
}}
QGroupBox {{
    background-color: {C_PANEL};
    border: {BORDER_W}px solid {C_BORDER};
    border-radius: {R_MD}px;
    margin-top: {SP_LG}px;
    padding-top: {SP_MD}px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: {SP_XS}px {SP_MD}px;
    background-color: {C_BG};
    color: {C_ACCENT};
    letter-spacing: 1.4px;
    font-size: {FS_SM}px;
}}
QLabel {{ color: {C_TEXT}; background: transparent; }}
QPushButton {{
    background-color: {C_PANEL_HI};
    color: {C_ACCENT};
    border: {BORDER_W}px solid {C_ACCENT};
    border-radius: {R_SM}px;
    padding: {SP_SM}px {SP_LG}px;
    font-weight: bold;
    min-width: 64px;
}}
QPushButton:hover {{ background-color: {C_ACCENT}; color: {C_BG}; }}
QPushButton:pressed {{ background-color: {C_BORDER_HI}; }}
QPushButton:disabled {{
    background-color: {C_PANEL};
    color: {C_DIM};
    border: {BORDER_W}px solid {C_DIM};
}}
QComboBox, QTimeEdit, QSpinBox, QLineEdit {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: {BORDER_W}px solid {C_BORDER};
    border-radius: {R_SM}px;
    padding: {SP_XS}px {SP_MD}px;
    selection-background-color: {C_ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {C_PANEL_HI};
    color: {C_TEXT};
    border: {BORDER_W}px solid {C_ACCENT};
    selection-background-color: {C_ACCENT};
    selection-color: {C_BG};
}}
QCheckBox {{ color: {C_TEXT}; spacing: {SP_SM}px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: {BORDER_W}px solid {C_DIM};
    background: {C_PANEL_HI};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border: {BORDER_W}px solid {C_ACCENT};
}}
QTextEdit, QPlainTextEdit {{
    background-color: #020912;
    color: {C_OK};
    border: {BORDER_W}px solid {C_BORDER};
    border-radius: {R_SM}px;
    font-family: Consolas, "Courier New", monospace;
    font-size: {FS_XS}px;
    padding: {SP_SM}px;
}}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C_BORDER_HI}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C_BORDER_HI}; border-radius: 4px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
"""

SIDEBAR_QSS = ''


def style_pg_plot(plot, title=None, title_color=None):
    plot.setMenuEnabled(False)
    plot.showGrid(x=True, y=True, alpha=0.10)
    if title is not None:
        plot.setTitle(title, color=title_color or C_ACCENT, size='10pt')
    for ax in ('left', 'bottom'):
        a = plot.getAxis(ax)
        a.setPen(C_BORDER)
        a.setTextPen(C_DIM)
