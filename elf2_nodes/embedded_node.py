"""
增强嵌入式节点放大覆盖层 — 工业级全屏覆盖 (UI-UX-Pro-Max v3)
支持三模态: 心率(1.py) / 脑电(11.py) / 视觉(visual_node.py)

核心改进：
1. 多路径搜索: 自动在 D:/G/ 和 ../0/Anti_depression/ 等多处定位独立节点 .py
2. 视觉节点: 调用 visual_node.VisionMonitorWindow 的全屏嵌入
3. 统一的工业级覆盖层(顶部标题+返回按钮+Esc/双击返回)
4. 自动停止所有子窗口QTimer、关闭串口/摄像头
5. 模块缓存复用、每次新建实例保证状态干净
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                              QShortcut, QVBoxLayout, QWidget)

# 主题常量—保持与D:/G/3 theme.py一致
C_BG        = '#04111f'
C_PANEL     = '#0a1f3d'
C_PANEL_HI  = '#13315c'
C_BORDER    = '#1c4076'
C_ACCENT    = '#00e5ff'
C_DIM       = '#a4b1d8'

BORDER_W = 1
SP_XS, SP_SM, SP_MD, SP_LG, SP_XL = 4, 8, 12, 20, 28
FS_XS, FS_SM, FS_MD, FS_LG = 10, 11, 13, 17
R_SM, R_MD = 4, 8


# ============================================================
# 多路径模块加载器
# ============================================================
_MODULE_CACHE: dict[str, object] = {}

# 节点文件候选搜索路径（按优先级）
_SEARCH_ROOTS = [
    lambda: str(Path(__file__).resolve().parent.parent),   # D:/G/
    lambda: os.path.join(str(Path(__file__).resolve().parent.parent), '0', 'Anti_depression'),  # D:/G/0/Anti_depression
    lambda: os.getcwd(),
    lambda: 'D:/G',
    lambda: 'D:/G/0/Anti_depression',
]

_NODE_DEFS = {
    'heart': {
        'filename': '1.py',
        'class_name': 'HRMonitorWindow',
        'title': '心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE',
        'icon': '❤',
        'module_path_override': None,
    },
    'brain': {
        'filename': '11.py',
        'class_name': 'EEGMonitorWindow',
        'title': '脑 电 感 知 节 点  ·  TGAM  EDGE  NODE',
        'icon': '🧠',
        'module_path_override': None,
    },
    'vision': {
        'filename': 'visual_node_v2.py',
        'class_name': 'PredictiveVisionWindow',
        'title': '预 测 性 视 觉 感 知  ·  PREDICTIVE  VISION  EDGE  v2',
        'icon': '👁',
        'module_path_override': 'D:/G/M/visual_node_v2.py',
    },
    'vision_v1': {
        'filename': 'visual_node.py',
        'class_name': 'VisionMonitorWindow',
        'title': '视 觉 感 知 节 点  ·  FER+  VISION  EDGE  NODE',
        'icon': '👁',
        'module_path_override': 'D:/G/M/visual_node.py',
    },
}


def _find_node_file(filename: str) -> str | None:
    """在多个搜索根目录下查找节点文件。"""
    for root_fn in _SEARCH_ROOTS:
        try:
            root = root_fn()
            full = os.path.join(root, filename)
            if os.path.isfile(full):
                return str(Path(full).resolve())
        except Exception:
            continue
    return None


def _load_external_module(module_path: str, class_name: str):
    """从绝对路径加载 .py 文件, 返回 (module, window_class)。"""
    abs_path = str(Path(module_path).resolve())
    cache_key = f'{abs_path}:{class_name}'
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f'节点源文件不存在: {abs_path}')

    safe_name = '_embedded_' + os.path.basename(abs_path).replace('.', '_').replace(' ', '_')
    spec = importlib.util.spec_from_file_location(safe_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法构造 spec: {abs_path}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[safe_name] = module
    spec.loader.exec_module(module)

    win_cls = getattr(module, class_name, None)
    if win_cls is None:
        raise AttributeError(f'模块 {abs_path} 中未找到类 {class_name}')

    _MODULE_CACHE[cache_key] = (module, win_cls)
    return module, win_cls


# ============================================================
# 节点描述符
# ============================================================
class NodeDescriptor:
    """描述一个可嵌入的独立节点。"""
    def __init__(self, node_type: str, module_path: str | None = None):
        info = _NODE_DEFS[node_type]
        self.node_type = node_type
        self.filename = info['filename']
        self.class_name = info['class_name']
        self.title = info['title']
        self.icon = info['icon']

        # 确定模块路径
        if module_path:
            self.module_path = module_path
        elif info.get('module_path_override'):
            self.module_path = info['module_path_override']
        else:
            found = _find_node_file(self.filename)
            if found is None:
                raise FileNotFoundError(
                    f'未找到 {node_type} 节点文件 "{self.filename}"。'
                    f'已搜索的根目录包括 D:/G/, D:/G/0/Anti_depression/ 等。'
                    f'请确保 {self.filename} 存在于上述位置。')
            self.module_path = found

    def __repr__(self):
        return f'NodeDescriptor({self.node_type}, {self.module_path})'


# ============================================================
# EmbeddedNodeOverlay (增强版)
# ============================================================
class EmbeddedNodeOverlay(QFrame):
    """全屏覆盖层, 内嵌外部独立运行的 QMainWindow。

    用法:
        EmbeddedNodeOverlay.install(
            card_widget=self.heart_card,
            host_window=self,
            node_type='heart',    # 'heart' | 'brain' | 'vision'
        )
        之后双击 card_widget 即可全屏放大。
    """

    # ------------------ 安装入口 ------------------
    @staticmethod
    def install(card_widget: QWidget, host_window: QWidget,
                node_type: str, title: str = '') -> None:
        """绑定双击事件到 card_widget, 双击时创建覆盖层嵌入节点。"""
        def _show():
            try:
                ov = EmbeddedNodeOverlay(host_window, node_type, title)
                ov.show()
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f'[EmbeddedNodeOverlay] 启动失败 ({node_type}):\n{tb}')
                console = getattr(host_window, 'ai_console', None)
                if console is not None:
                    console.append(f'> [节点放大错误] {exc}')
                    for line in tb.splitlines()[-4:]:
                        console.append(f'>   {line}')

        orig = getattr(card_widget, 'mouseDoubleClickEvent', None)

        def _h(ev):
            _show()
            if orig:
                try: orig(ev)
                except Exception: pass

        card_widget.mouseDoubleClickEvent = _h
        card_widget.setCursor(Qt.PointingHandCursor)
        card_widget.setToolTip(f'双击放大查看完整{node_type}节点界面 (Esc 返回)')

    # ------------------ 构造 ------------------
    def __init__(self, host_window: QWidget, node_type: str, title: str = ''):
        top_win = host_window.window() if hasattr(host_window, 'window') else host_window
        super().__init__(top_win)
        self.host = top_win

        # 解析节点描述符
        self._descriptor = NodeDescriptor(node_type)
        if title:
            self._descriptor.title = title

        self._embedded_win = None
        self._embedded_central = None

        self.setObjectName('EmbedOv')
        self.setStyleSheet(f'#EmbedOv {{ background:{C_BG}; }}')
        self.setAutoFillBackground(True)
        self.setGeometry(self.host.rect())
        self.raise_(); self.setFocus()

        self.host.installEventFilter(self)

        # ---- UI ----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_LG); outer.setSpacing(SP_SM)

        head = QHBoxLayout(); head.setSpacing(SP_MD)

        dot = QLabel(self._descriptor.icon)
        dot.setStyleSheet(f'color:{C_ACCENT}; font-size:{FS_MD}px; background:transparent;')
        t = QLabel(self._descriptor.title)
        t.setStyleSheet(
            f'color:{C_ACCENT}; font-size:{FS_LG}px; font-weight:bold;'
            f'letter-spacing:3px; background:transparent;')

        sub = QLabel('NODE FULLSCREEN  ·  独立节点完整视图  ·  LIVE/SIM AUTODETECT')
        sub.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.6px; background:transparent;')
        hint = QLabel('Esc / 双击空白 返回')
        hint.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.4px; background:transparent;')

        self.btn_back = QPushButton('⬅  返回系统界面')
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setShortcut('Esc')
        self.btn_back.setToolTip('返回到 MindRoom 主控大屏 (Esc)')
        self.btn_back.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
            f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_SM}px;'
            f'padding:{SP_SM}px {SP_XL}px; font-size:{FS_MD}px;'
            f'font-weight:bold; letter-spacing:2px; min-width:140px; }}'
            f'QPushButton:hover {{ background:{C_ACCENT}; color:#020912; }}'
            f'QPushButton:pressed {{ background:{C_BORDER}; color:{C_ACCENT}; }}')
        self.btn_back.clicked.connect(self._close)

        head.addWidget(dot); head.addWidget(t); head.addSpacing(SP_LG)
        head.addWidget(sub); head.addStretch(1); head.addWidget(hint)
        head.addSpacing(SP_LG); head.addWidget(self.btn_back)
        outer.addLayout(head)

        # 嵌入容器
        big = QFrame()
        big.setObjectName('EmbedBig')
        big.setStyleSheet(
            f'#EmbedBig {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_ACCENT};'
            f'border-radius:{R_MD}px; }}')
        big_lay = QVBoxLayout(big)
        big_lay.setContentsMargins(0, 0, 0, 0); big_lay.setSpacing(0)
        outer.addWidget(big, 1)
        self._big = big

        # 加载并嵌入外部窗口
        self._embed_external(big_lay)

        # Esc
        self._sc_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._sc_esc.setContext(Qt.WindowShortcut)
        self._sc_esc.activated.connect(self._close)

    # ------------------ 内部: 创建独立窗口 (不嵌入控件树) ------------------
    def _embed_external(self, container_layout):
        module, win_cls = _load_external_module(
            self._descriptor.module_path, self._descriptor.class_name)

        win = win_cls()
        self._embedded_win = win

        # ===== 嵌入节点永远不碰硬件 =====
        self._force_simulation_mode(win)

        # ===== 作为独立顶层窗口显示, 不嵌入 overlay 控件树 =====
        # QMainWindow 不是设计来当子控件的。强行 setParent 嵌入会与其内部
        # QMainWindowLayout 冲突, 导致析构时野指针/死锁。
        # 方案: win 保持为独立顶层窗口, overlay 只提供"返回"按钮UI。
        win.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        win.setWindowModality(Qt.ApplicationModal)
        try:
            win.setGeometry(self.host.geometry())
        except Exception:
            pass
        win.show()

    def _force_simulation_mode(self, win):
        """切模拟模式, 覆盖 auto_connect 防止 singleShot(300) 抢硬件。"""
        if hasattr(win, 'enter_simulation'):
            try: win.enter_simulation('嵌入模式 · 硬件由主系统管理')
            except: pass
        if hasattr(win, 'auto_connect'):
            self._saved_auto_connect = win.auto_connect
            try: win.auto_connect = lambda: None
            except: pass

    # ------------------ 关闭 ------------------
    def _close(self):
        """安全返回: 先关 overlay, 再独立关闭底层窗口。

        overlay(self)和win是两个独立的顶层窗口, 互不依赖 parent-child 关系。
        销毁顺序: overlay先消失 → win独立close → win独立deleteLater。
        """
        win = self._embedded_win

        # 1) 立即隐藏 overlay
        try: self.hide()
        except: pass

        # 2) 移除事件过滤器 (对host)
        try: self.host.removeEventFilter(self)
        except: pass

        # 3) overlay 自毁 (与 win 无关, win 不受影响)
        try: self.deleteLater()
        except: pass

        # 4) 现在处理 win: 停timer → 关glw → 关硬件 → close
        if win is not None:
            # 停属性级 timer
            for attr in ('timer', 'watchdog_timer', 'update_timer',
                         'read_timer', 'history_timer', 'capture_timer'):
                t = getattr(win, attr, None)
                if isinstance(t, QTimer):
                    try: t.stop()
                    except: pass
                    try: t.deleteLater()
                    except: pass
            # 停子控件 timer
            for t in list(win.findChildren(QTimer)):
                try: t.stop()
                except: pass
                try: t.deleteLater()
                except: pass

            # 关 pyqtgraph
            if hasattr(win, 'glw'):
                try: win.glw.close()
                except: pass

            # 关硬件
            for name in ('ser', 'serial_port', 'udp_sock'):
                dev = getattr(win, name, None)
                if dev is None: continue
                try:
                    if hasattr(dev, 'is_open') and dev.is_open:
                        dev.close()
                except: pass

            # 恢复 auto_connect
            saved = getattr(self, '_saved_auto_connect', None)
            if saved is not None:
                try: win.auto_connect = saved
                except: pass
            if hasattr(win, 'shutdown'):
                try: win.shutdown()
                except: pass

            # 关闭并延迟销毁独立窗口
            try: win.hide()
            except: pass
            try: win.close()
            except: pass
            try: win.deleteLater()
            except: pass

    # ------------------ 事件 ------------------
    def mouseDoubleClickEvent(self, _e):
        self._close()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self._close()
            return
        super().keyPressEvent(ev)

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            try:
                self.setGeometry(self.host.rect())
                self.raise_()
            except Exception:
                pass
        return super().eventFilter(obj, ev)
