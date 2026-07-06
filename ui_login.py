"""登录页 — 明亮玻璃质感版（D:/G/3）。

设计要点：
  - 背景：深蓝→品蓝径向渐变 + 顶部高光带 + 噪点描边
  - 卡片：白色 6% 透明度 + 内描边 1px 青光
  - 输入框：内嵌图标（👤/🔒）+ 胶囊形 + 焦点态边框光晕动画
  - 登录按钮：紫→青→紫渐变 + 多层投影（外发光 + 底部阴影）
  - 角色快捷：右上角"我是学生 / 我是家长" 一键填充
"""

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt5.QtGui import (QColor, QFont, QPainter, QBrush, QPen,
                         QLinearGradient, QRadialGradient, QPainterPath, QRegion)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QGraphicsDropShadowEffect,
                             QFrame, QWidget, QSizePolicy, QSpacerItem)


# ============== 霓虹按钮 ==============
class GlowButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(56)
        self._eff = QGraphicsDropShadowEffect()
        self._eff.setBlurRadius(28); self._eff.setOffset(0, 4)
        self._eff.setColor(QColor(0, 229, 255, 220))
        self.setGraphicsEffect(self._eff)
        self.setStyleSheet(
            'QPushButton {'
            ' background: qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            ' stop:0 #7c4dff, stop:0.5 #00e5ff, stop:1 #b388ff);'
            ' border:none; border-radius:28px; color:#04111f;'
            ' font-size:15px; font-weight:bold; letter-spacing:8px;'
            ' font-family:"Microsoft YaHei","Arial"; }'
            'QPushButton:hover {'
            ' background: qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            ' stop:0 #b388ff, stop:0.5 #00e5ff, stop:1 #7c4dff); }'
            'QPushButton:pressed {'
            ' background: qlineargradient(x1:0,y1:0,x2:1,y2:0,'
            ' stop:0 #6a3aff, stop:0.5 #00b8d4, stop:1 #9575ff); }'
        )

    def enterEvent(self, ev):
        self._eff.setBlurRadius(48)
        self._eff.setColor(QColor(124, 77, 255, 240))
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._eff.setBlurRadius(28)
        self._eff.setColor(QColor(0, 229, 255, 220))
        super().leaveEvent(ev)


# ============== 玻璃背景 ==============
class _GlassBg(QFrame):
    """毛玻璃背景：深蓝径向渐变 + 顶部高光 + 边框光。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('GlassBg')

    def resizeEvent(self, ev):
        # 将子控件裁切在圆角卡片内部，避免底部/边角外溢
        r = self.rect()
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()), float(r.width()), float(r.height()), 22.0, 22.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        super().resizeEvent(ev)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        # 底色径向（中心更亮一点的品蓝）
        rg = QRadialGradient(W * 0.5, H * 0.35, max(W, H) * 0.7)
        rg.setColorAt(0,   QColor(20, 50, 100))
        rg.setColorAt(0.5, QColor(8,  28, 60))
        rg.setColorAt(1,   QColor(2,  9,  18))
        p.setBrush(QBrush(rg)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, W, H, 22, 22)

        # 左上紫光晕
        rg1 = QRadialGradient(W * 0.18, H * 0.18, 320)
        rg1.setColorAt(0, QColor(124, 77, 255, 110))
        rg1.setColorAt(1, QColor(124, 77, 255, 0))
        p.setBrush(QBrush(rg1))
        p.drawRect(0, 0, W, H)

        # 右下青光晕
        rg2 = QRadialGradient(W * 0.85, H * 0.9, 360)
        rg2.setColorAt(0, QColor(0, 229, 255, 110))
        rg2.setColorAt(1, QColor(0, 229, 255, 0))
        p.setBrush(QBrush(rg2))
        p.drawRect(0, 0, W, H)

        # 顶部高光带（玻璃感）
        hg = QLinearGradient(0, 0, 0, H * 0.28)
        hg.setColorAt(0, QColor(255, 255, 255, 28))
        hg.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(hg))
        p.drawRoundedRect(0, 0, W, int(H * 0.28), 22, 22)

        # 内描边 1px 青光 + 外描边 1px 暗
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 229, 255, 160), 1))
        p.drawRoundedRect(1, 1, W - 3, H - 3, 21, 21)
        p.setPen(QPen(QColor(0, 229, 255, 50), 1))
        p.drawRoundedRect(0, 0, W - 1, H - 1, 22, 22)


# ============== 内嵌图标的输入框 ==============
class IconLineEdit(QFrame):
    """胶囊容器 + 左侧图标 + QLineEdit + 焦点态发光。"""

    def __init__(self, icon, placeholder='', is_password=False, parent=None):
        super().__init__(parent)
        self.setObjectName('IconLE')
        self.setMinimumHeight(50)
        self.setStyleSheet(
            '#IconLE { background:rgba(255,255,255,0.06);'
            ' border:1px solid rgba(124,77,255,0.45);'
            ' border-radius:25px; }')
        # 焦点态发光效果
        self._eff = QGraphicsDropShadowEffect()
        self._eff.setBlurRadius(0); self._eff.setOffset(0, 0)
        self._eff.setColor(QColor(0, 229, 255, 0))
        self.setGraphicsEffect(self._eff)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)

        self.icon_lbl = QLabel(icon)
        self.icon_lbl.setStyleSheet(
            'color:#9ba8c8; font-size:16px; background:transparent;')
        self.icon_lbl.setFixedWidth(20)
        self.icon_lbl.setAlignment(Qt.AlignCenter)

        self.line = QLineEdit()
        self.line.setPlaceholderText(placeholder)
        if is_password:
            self.line.setEchoMode(QLineEdit.Password)
        self.line.setStyleSheet(
            'QLineEdit { background:transparent; border:none;'
            ' color:#e8eaf6; font-size:14px; padding:0;'
            ' selection-background-color:#00e5ff;'
            ' font-family:"Microsoft YaHei","Arial"; }'
            'QLineEdit::placeholder { color:#7986b8; }')
        # 监听焦点变化
        self.line.installEventFilter(self)

        lay.addWidget(self.icon_lbl)
        lay.addWidget(self.line, 1)

    # 转发常用 API，外部直接当 QLineEdit 用
    def text(self):           return self.line.text()
    def setText(self, t):     self.line.setText(t)
    def setEchoMode(self, m): self.line.setEchoMode(m)
    def setFocus(self):       self.line.setFocus()
    @property
    def returnPressed(self):  return self.line.returnPressed

    def eventFilter(self, obj, ev):
        if obj is self.line:
            if ev.type() == ev.FocusIn:
                self._set_focus_glow(True)
            elif ev.type() == ev.FocusOut:
                self._set_focus_glow(False)
        return super().eventFilter(obj, ev)

    def _set_focus_glow(self, on):
        if on:
            self.setStyleSheet(
                '#IconLE { background:rgba(0,229,255,0.08);'
                ' border:1px solid #00e5ff; border-radius:25px; }')
            self.icon_lbl.setStyleSheet(
                'color:#00e5ff; font-size:16px; background:transparent;')
            self._eff.setBlurRadius(24)
            self._eff.setColor(QColor(0, 229, 255, 200))
        else:
            self.setStyleSheet(
                '#IconLE { background:rgba(255,255,255,0.06);'
                ' border:1px solid rgba(124,77,255,0.45);'
                ' border-radius:25px; }')
            self.icon_lbl.setStyleSheet(
                'color:#9ba8c8; font-size:16px; background:transparent;')
            self._eff.setBlurRadius(0)
            self._eff.setColor(QColor(0, 229, 255, 0))


# ============== 角色快捷胶囊 ==============
class RoleChip(QPushButton):
    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(28)
        self.setStyleSheet(
            f'QPushButton {{ background:transparent; color:{color};'
            f' border:1px solid {color}; border-radius:14px;'
            f' padding:0 14px; font-size:11px; font-weight:bold;'
            f' letter-spacing:2px; }}'
            f'QPushButton:hover {{ background:{color}; color:#04111f; }}')


# ============== 登录窗 ==============
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.role = None
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(760, 620)
        self.resize(820, 680)
        self._drag = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        bg = _GlassBg()
        outer.addWidget(bg)

        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(60); eff.setOffset(0, 16)
        eff.setColor(QColor(0, 229, 255, 110))
        bg.setGraphicsEffect(eff)

        lay = QVBoxLayout(bg)
        lay.setContentsMargins(64, 28, 64, 28)
        lay.setSpacing(0)

        # 将登录主体垂直居中，避免底部露出下一层内容
        lay.addStretch(1)

        # ---- 顶栏 ----
        topbar = QHBoxLayout()
        topbar.setContentsMargins(0, 0, 0, 0)
        crumb = QLabel('●  MINDROOM  ·  EDGE  GUARDIAN')
        crumb.setStyleSheet(
            'color:#9ba8c8; font-size:11px; letter-spacing:3px;'
            'background:transparent; font-weight:bold;')
        close = QPushButton('×')
        close.setCursor(Qt.PointingHandCursor)
        close.setFixedSize(30, 30)
        close.setStyleSheet(
            'QPushButton { color:#9ba8c8; background:transparent;'
            'border:1px solid rgba(255,255,255,0.15); border-radius:15px;'
            'font-size:18px; font-weight:bold; }'
            'QPushButton:hover { background:#ff3860; color:white;'
            'border:1px solid #ff3860; }')
        close.clicked.connect(self.reject)
        topbar.addWidget(crumb); topbar.addStretch(); topbar.addWidget(close)
        lay.addLayout(topbar)

        lay.addSpacing(20)

        # ---- 标题 + 副标题 ----
        title = QLabel('MindRoom Guard')
        title_font = QFont('Arial', 36, QFont.Bold)
        title_font.setLetterSpacing(QFont.PercentageSpacing, 105)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color:#e8eaf6; background:transparent;')
        eff_t = QGraphicsDropShadowEffect()
        eff_t.setBlurRadius(36); eff_t.setOffset(0, 0)
        eff_t.setColor(QColor(0, 229, 255, 200))
        title.setGraphicsEffect(eff_t)
        lay.addWidget(title)

        sub = QLabel('基于边缘计算的抑郁风险预警系统')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            'color:#9ba8c8; font-size:11px; letter-spacing:6px;'
            'background:transparent; font-family:微软雅黑;')
        lay.addWidget(sub)

        lay.addSpacing(8)

        # ---- 角色快捷 ----
        chip_row = QHBoxLayout()
        chip_row.setSpacing(10)
        chip_row.addStretch()
        chip_stu = RoleChip('👨‍🎓 我是学生', '#00e5ff')
        chip_par = RoleChip('👪 我是家长',   '#b388ff')
        chip_stu.clicked.connect(lambda: self._fill('stu'))
        chip_par.clicked.connect(lambda: self._fill('par'))
        chip_row.addWidget(chip_stu); chip_row.addWidget(chip_par); chip_row.addStretch()
        lay.addLayout(chip_row)

        lay.addSpacing(16)

        # ---- 中段表单区（固定节奏，避免被拉扯）----
        form_wrap = QWidget()
        form_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form_lay = QVBoxLayout(form_wrap)
        form_lay.setContentsMargins(0, 0, 0, 0)
        form_lay.setSpacing(14)

        self.user_input = IconLineEdit('👤', '账号  ·  USERNAME  ( stu / par )')
        self.pwd_input  = IconLineEdit('🔒', '密码  ·  PASSWORD  ( 123 )', is_password=True)
        self.user_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pwd_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_login = GlowButton('S E C U R E   L O G I N')
        self.btn_login.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_login.clicked.connect(self._check)

        self.err_lbl = QLabel(' ')
        self.err_lbl.setAlignment(Qt.AlignCenter)
        self.err_lbl.setMinimumHeight(18)
        self.err_lbl.setMaximumHeight(26)
        self.err_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.err_lbl.setStyleSheet(
            'color:#ff3860; font-size:11px; background:transparent;'
            'letter-spacing:1.5px;')

        form_lay.addWidget(self.user_input)
        form_lay.addWidget(self.pwd_input)
        form_lay.addSpacing(6)
        form_lay.addWidget(self.btn_login)
        form_lay.addWidget(self.err_lbl)

        lay.addWidget(form_wrap)
        lay.addSpacing(12)

        # ---- 底部 ----
        foot = QLabel('© 2025  MindRoom Edge AI  ·  RK3588  ·  Powered by NPU')
        foot.setAlignment(Qt.AlignCenter)
        foot.setWordWrap(True)
        foot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        foot.setStyleSheet(
            'color:#7986b8; font-size:9px; letter-spacing:2px;'
            'background:transparent;')
        lay.addWidget(foot)

        lay.addStretch(1)

        # 回车提交
        self.user_input.returnPressed.connect(lambda: self.pwd_input.setFocus())
        self.pwd_input.returnPressed.connect(self._check)

        # 默认聚焦账号
        QTimer.singleShot(50, self.user_input.setFocus)

    def _fill(self, who):
        self.user_input.setText(who)
        self.pwd_input.setText('123')
        self.pwd_input.setFocus()

    def _check(self):
        u = self.user_input.text().strip()
        p = self.pwd_input.text().strip()
        if u == 'stu' and p == '123':
            self.role = 'student'; self.accept()
        elif u == 'par' and p == '123':
            self.role = 'parent';  self.accept()
        else:
            self.err_lbl.setText('  账号或密码错误  ·  ACCESS DENIED  ')

    # 仅顶栏可拖拽（避免按钮、输入框被吞事件）
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and ev.y() < 70:
            self._drag = ev.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPos() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None
