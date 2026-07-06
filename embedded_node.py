"""
嵌入式节点放大覆盖层 (UI-UX-Pro-Max / Superpowers).

把 D:/G/1.py (HRMonitorWindow) 与 D:/G/11.py (EEGMonitorWindow) 这两个独立运行的
QMainWindow 直接嵌入主系统(MindRoom Guard)之上, 让用户双击 心率卡 / 脑电卡 后, 看到的
就是它们独立运行时的完整工业级界面, 而不是简单地把仪表卡放大.

设计要点
--------
* 不修改 1.py / 11.py 的任何一行代码 (它们要保持随时单独 `python 1.py` 可跑).
* 通过 `importlib` 直接加载源文件 (1.py / 11.py 不是合法的模块名, 只能这么加载).
* 取出 `QMainWindow.centralWidget()` 重新挂到本覆盖层, 这样原代码里所有 `self.xxx` 控件
  / QTimer / 信号槽全部继续有效.
* 顶部右上角放一个统一风格的 `⬅ 返回` 按钮; 同时支持 `Esc` / 双击空白区返回.
* 关闭时主动 `stop()` 所有 QTimer, `close()` 串口, 避免子页面的后台轮询继续跑.
* 单例缓存: 同一外部 .py 模块只导入一次; 但每次打开都生成全新的窗口实例, 保证状态干净.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QEvent, Qt, QTimer
import time
import traceback
from pathlib import Path
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                              QShortcut, QVBoxLayout, QWidget)

from theme import (BORDER_W, C_ACCENT, C_BG, C_BG_DEEP, C_BORDER, C_DIM,
                   C_PANEL, C_PANEL_HI, FS_LG, FS_MD, FS_XS, GLOBAL_QSS,
                   R_MD, R_SM, SP_LG, SP_MD, SP_SM, SP_XL, SP_XS)


# ---------------------------------------------------------------------------
# 模块加载 (绝对路径 -> Python 模块对象)
# ---------------------------------------------------------------------------
_MODULE_CACHE: dict[str, object] = {}
_ACTIVE_OVERLAY = None
_OPENING = False


def _load_external_module(path: str):
    """从绝对路径加载 .py 文件为 Python 模块对象 (带缓存)."""
    abs_path = str(Path(path).resolve())
    if abs_path in _MODULE_CACHE:
        return _MODULE_CACHE[abs_path]
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f'外部节点源文件不存在: {abs_path}')

    safe_name = '_embedded_' + os.path.basename(abs_path).replace('.', '_')
    spec = importlib.util.spec_from_file_location(safe_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法构造 spec: {abs_path}')
    module = importlib.util.module_from_spec(spec)
    # 注册到 sys.modules 以避免内部 dataclass / pickle 等回查找时报 KeyError
    sys.modules[safe_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _MODULE_CACHE[abs_path] = module
    return module


# ---------------------------------------------------------------------------
# 主类: EmbeddedNodeOverlay
# ---------------------------------------------------------------------------
class EmbeddedNodeOverlay(QFrame):
    """全屏覆盖层, 内嵌外部独立运行的 QMainWindow.

    使用方式::

        EmbeddedNodeOverlay.install(
            card_widget=self.heart_card,
            host_window=self,
            module_path='D:/Anti_depression/1.py',
            window_class='HRMonitorWindow',
            title='心 率 感 知 节 点  ·  ESP32  HRV  EDGE  NODE',
        )

    之后 `card_widget` 双击即可放大.
    """

    # ------------------ 安装入口 ------------------
    @staticmethod
    def install(card_widget: QWidget, host_window: QWidget, module_path: str,
                window_class: str, title: str = '') -> None:
        def _show():
            global _ACTIVE_OVERLAY, _OPENING
            if _OPENING:
                return
            if _ACTIVE_OVERLAY is not None:
                return

            _OPENING = True
            try:
                ov = EmbeddedNodeOverlay(host_window, module_path,
                                         window_class, title)
                _ACTIVE_OVERLAY = ov
                ov.show()
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f'[EmbeddedNodeOverlay] 启动失败:\n{tb}')
                console = getattr(host_window, 'ai_console', None)
                if console is not None:
                    console.append(f'> [节点放大错误] {exc}')
                    for line in tb.splitlines()[-4:]:
                        console.append(f'>   {line}')
            finally:
                _OPENING = False

        # 给卡片绑双击（不再调用原始双击处理，避免重复触发引发重入）
        def _h(ev):
            try:
                ev.accept()
            except Exception:
                pass
            _show()

        card_widget.mouseDoubleClickEvent = _h
        card_widget.setCursor(Qt.PointingHandCursor)
        card_widget.setToolTip('双击放大查看完整节点界面 (Esc 返回)')

    # ------------------ 构造 ------------------
    def __init__(self, host_window: QWidget, module_path: str,
                 window_class_name: str, title: str = ''):
        # 根因修复：host_window 在当前工程里通常是 dashboard 子 QWidget（常驻 QScrollArea 内），
        # 覆盖层如果挂在它上面就只会出现在内容区（常见表现就是“卡在底部”）。
        # 必须挂到最外层 window 才能真正铺满可视区域。
        top_win = host_window.window() if hasattr(host_window, 'window') else host_window
        super().__init__(top_win)
        self.host = top_win
        self._module_path = module_path
        self._window_class_name = window_class_name
        self._title = title
        self._embedded_win = None
        self._embedded_central = None

        self.setObjectName('EmbedOv')
        self.setStyleSheet(f'#EmbedOv {{ background:{C_BG}; }}')
        self.setAutoFillBackground(True)
        self.setGeometry(self.host.rect())
        self.raise_()
        self.setFocus()

        # 跟随最外层窗口尺寸
        self.host.installEventFilter(self)

        # ---------- 顶部 (标题 + 返回) ----------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_LG)
        outer.setSpacing(SP_SM)

        head = QHBoxLayout()
        head.setSpacing(SP_MD)

        dot = QLabel('●')
        dot.setStyleSheet(
            f'color:{C_ACCENT}; font-size:{FS_MD}px; background:transparent;')
        t = QLabel(title or 'NODE ZOOM')
        t.setStyleSheet(
            f'color:{C_ACCENT}; font-size:{FS_LG}px; font-weight:bold;'
            f'letter-spacing:3px; background:transparent;')

        sub = QLabel('NODE FULLSCREEN  ·  独立节点完整视图')
        sub.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.6px;'
            f'background:transparent;')

        hint = QLabel('Esc / 双击空白 返回')
        hint.setStyleSheet(
            f'color:{C_DIM}; font-size:{FS_XS}px; letter-spacing:1.4px;'
            f'background:transparent;')

        # 右上角返回按钮 (UI-UX-Pro-Max 一致风格)
        self.btn_back = QPushButton('⬅  返回系统界面')
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setShortcut('Esc')
        self.btn_back.setToolTip('返回到 MindRoom 主控大屏 (Esc)')
        self.btn_back.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
            f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_SM}px;'
            f'padding:{SP_SM}px {SP_XL}px; font-size:{FS_MD}px;'
            f'font-weight:bold; letter-spacing:2px; min-width:140px; }}'
            f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG}; }}'
            f'QPushButton:pressed {{ background:{C_BORDER}; color:{C_ACCENT}; }}')
        self.btn_back.clicked.connect(self._close)

        head.addWidget(dot)
        head.addWidget(t)
        head.addSpacing(SP_LG)
        head.addWidget(sub)
        head.addStretch(1)
        head.addWidget(hint)
        head.addSpacing(SP_LG)
        head.addWidget(self.btn_back)
        outer.addLayout(head)

        # ---------- 嵌入容器 ----------
        big = QFrame()
        big.setObjectName('EmbedBig')
        big.setStyleSheet(
            f'#EmbedBig {{ background:{C_PANEL};'
            f'border:{BORDER_W}px solid {C_ACCENT};'
            f'border-radius:{R_MD}px; }}')
        big_lay = QVBoxLayout(big)
        big_lay.setContentsMargins(0, 0, 0, 0)
        big_lay.setSpacing(0)
        outer.addWidget(big, 1)
        self._big = big

        # 加载并嵌入外部窗口
        self._embed_external(big_lay)

        # Esc 快捷键 (即使焦点在子控件里也生效)
        self._sc_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._sc_esc.setContext(Qt.WindowShortcut)
        self._sc_esc.activated.connect(self._close)

    # ------------------ 内部: 加载并 reparent 外部窗口 ------------------
    def _embed_external(self, container_layout):
        print(f"[EmbeddedNodeOverlay] load module: {self._module_path}")
        module = _load_external_module(self._module_path)
        print(f"[EmbeddedNodeOverlay] module loaded from: {getattr(module, '__file__', 'UNKNOWN')}")
        win_cls = getattr(module, self._window_class_name, None)
        if win_cls is None:
            raise AttributeError(
                f'模块 {self._module_path} 中未找到类 {self._window_class_name}')

        # 实例化但不 show() —— 拿走它的 centralWidget 即可.
        print(f"[EmbeddedNodeOverlay] instantiate class: {self._window_class_name}")
        win = win_cls()
        print(f"[EmbeddedNodeOverlay] instance type: {type(win)}")
        # 让外部窗口本身也被完整保留 (不要 show), 否则会出现一个孤立顶层窗口闪一下.
        win.setAttribute(Qt.WA_DontShowOnScreen, True)
        # 挂在自己上以延长生命周期
        self._embedded_win = win

        # 叠加主题 QSS, 让控件颜色与系统主体一致 (1.py / 11.py 自带 QSS, 这里只是兜底).
        try:
            win.setStyleSheet((win.styleSheet() or '') + GLOBAL_QSS)
        except Exception:
            pass

        central = win.centralWidget()
        print(f"[EmbeddedNodeOverlay] centralWidget is None? {central is None}")
        if central is None:
            # 兜底: 如果没有 centralWidget, 直接把 QMainWindow 当作 widget 嵌入
            win.setParent(self._big)
            win.setWindowFlags(Qt.Widget)
            container_layout.addWidget(win)
            self._embedded_central = win
        else:
            central.setParent(self._big)
            container_layout.addWidget(central)
            self._embedded_central = central

    # ------------------ 关闭 (安全销毁序列, 修复卡死) ------------------
    def _close(self):
        """极速非阻塞关闭：优先保证UI不卡死。"""
        t0 = time.perf_counter()
        print('[EmbeddedNodeOverlay] _close enter')
        if getattr(self, '_closing', False):
            print('[EmbeddedNodeOverlay] _close ignored: already closing')
            return
        self._closing = True

        win = self._embedded_win
        try:
            if win is not None:
                setattr(win, '_shutting_down', True)
        except Exception:
            pass

        try:
            self.hide()
            print(f"[EmbeddedNodeOverlay] stage1 hide ok, dt={(time.perf_counter()-t0)*1000:.1f}ms")
        except Exception:
            print('[EmbeddedNodeOverlay] stage1 hide err')
            traceback.print_exc()

        from PyQt5.QtCore import QTimer as _QTimer
        _QTimer.singleShot(0, self._dispose_embedded_win)
        _QTimer.singleShot(0, self._final_cleanup)
        print(f"[EmbeddedNodeOverlay] _close exit, dt={(time.perf_counter()-t0)*1000:.1f}ms")

    def _dispose_embedded_win(self):
        t0 = time.perf_counter()
        win = self._embedded_win

        try:
            self.host.removeEventFilter(self)
            print(f"[EmbeddedNodeOverlay] dispose remove event filter ok, dt={(time.perf_counter()-t0)*1000:.1f}ms")
        except Exception:
            print('[EmbeddedNodeOverlay] dispose remove event filter err')
            traceback.print_exc()

        try:
            if self._embedded_central is not None:
                self._embedded_central.setParent(None)
            print(f"[EmbeddedNodeOverlay] dispose central detach ok, dt={(time.perf_counter()-t0)*1000:.1f}ms")
        except Exception:
            print('[EmbeddedNodeOverlay] dispose central detach err')
            traceback.print_exc()
        self._embedded_central = None

        if win is not None:
            try:
                for name in ('timer', 'watchdog_timer'):
                    t = getattr(win, name, None)
                    if t is not None:
                        try:
                            t.stop()
                            t.blockSignals(True)
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                win.hide()
            except Exception:
                pass
            try:
                win.setParent(None)
            except Exception:
                pass
            try:
                win.close()
            except Exception:
                pass
            try:
                win.deleteLater()
            except Exception:
                pass
            self._embedded_win = None
        print(f"[EmbeddedNodeOverlay] dispose win scheduled, dt={(time.perf_counter()-t0)*1000:.1f}ms")

    def _final_cleanup(self):
        global _ACTIVE_OVERLAY
        _ACTIVE_OVERLAY = None
        try: self.hide()
        except: pass
        try: self.setParent(None)
        except: pass
        try: self.deleteLater()
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
        # 跟随最外层窗口几何变化，始终铺满
        if obj is self.host and ev.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            try:
                self.setGeometry(self.host.rect())
                self.raise_()
            except Exception:
                pass
        return super().eventFilter(obj, ev)
