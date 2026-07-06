from __future__ import annotations

import random
import time

import numpy as np

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

import ppg_signal

from theme import (
    BORDER_W,
    C_ACCENT,
    C_BG,
    C_BORDER,
    C_DIM,
    C_ECG,
    C_HEART,
    C_HR,
    C_HRV,
    C_OK,
    C_PANEL,
    C_PANEL_HI,
    C_PPG,
    C_RISK,
    C_WARN,
    FS_LG,
    FS_MD,
    FS_XS,
    R_MD,
    R_SM,
    SP_LG,
    SP_MD,
    SP_SM,
    SP_XL,
    SP_XS,
)


_ACTIVE_HEART_OVERLAY = None


class WavePanel(QWidget):
    def __init__(self, title, color, fixed_range=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.fixed_range = fixed_range
        self.data = []
        self.setMinimumHeight(120)
        self.setStyleSheet(f'background:{C_PANEL};')

    def set_data(self, values):
        self.data = list(values or [])
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect().adjusted(14, 18, -14, -12)
        if rect.width() <= 2 or rect.height() <= 2:
            return

        p.setPen(QPen(QColor(C_BORDER), 1))
        for i in range(1, 4):
            y = rect.top() + rect.height() * i / 4
            p.drawLine(rect.left(), int(y), rect.right(), int(y))

        p.setPen(QPen(self.color, 1))
        p.setFont(QFont('Consolas', 10, QFont.Bold))
        p.drawText(QRectF(rect.left(), 0, rect.width(), 18), Qt.AlignCenter, self.title)

        vals = []
        for v in self.data:
            try:
                vals.append(float(v))
            except Exception:
                pass
        if len(vals) < 2:
            return

        if self.fixed_range is not None:
            v_min, v_max = self.fixed_range
        else:
            v_min, v_max = min(vals), max(vals)
            span = max(1.0, v_max - v_min)
            v_min -= span * 0.12
            v_max += span * 0.12
        if v_max <= v_min:
            v_max = v_min + 1.0

        max_points = min(len(vals), 800)
        vals = vals[-max_points:]
        dx = rect.width() / max(1, len(vals) - 1)
        scale = rect.height() / (v_max - v_min)

        path = QPainterPath()
        for i, v in enumerate(vals):
            x = rect.left() + i * dx
            y = rect.bottom() - (v - v_min) * scale
            y = max(rect.top(), min(rect.bottom(), y))
            if i == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))

        pen = QPen(self.color, 2)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawPath(path)


class HeartDataZoomOverlay(QFrame):
    """Fullscreen cardiac view backed by existing dashboard buffers only."""

    REFRESH_MS = 350

    @staticmethod
    def install(card_widget: QWidget, host_window: QWidget) -> None:
        def _show():
            global _ACTIVE_HEART_OVERLAY
            if _ACTIVE_HEART_OVERLAY is None:
                _ACTIVE_HEART_OVERLAY = HeartDataZoomOverlay(host_window)
            _ACTIVE_HEART_OVERLAY.source = host_window
            _ACTIVE_HEART_OVERLAY.reopen()

        def _h(ev):
            try:
                ev.accept()
            except Exception:
                pass
            _show()

        card_widget.mouseDoubleClickEvent = _h
        card_widget.setCursor(Qt.PointingHandCursor)
        card_widget.setToolTip('双击放大心率模块')

    def __init__(self, host_window: QWidget):
        top_win = host_window.window() if hasattr(host_window, 'window') else host_window
        super().__init__(top_win)
        self.host = top_win
        self.source = host_window
        self._closed = True
        self._rr_intervals_buffer = []
        self._hrv_buffer_1py = []
        self._hrv_display_ema_1py = None
        self._sdnn_display_ema_1py = None
        self._sdnn_trend_1py = []
        self._last_peaks_sig = None
        self._refresh_counter = 0

        self.setObjectName('HeartDataZoomOverlay')
        self.setStyleSheet(f'#HeartDataZoomOverlay {{ background:{C_BG}; }}')
        self.setGeometry(self.host.rect())
        self.host.installEventFilter(self)

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)

        self._esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._esc.setContext(Qt.WindowShortcut)
        self._esc.activated.connect(self.close_overlay)
        self.hide()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(SP_LG, SP_MD, SP_LG, SP_LG)
        root.setSpacing(SP_SM)

        head = QHBoxLayout()
        title = QLabel('心率感知节点 · ESP32 HRV EDGE NODE')
        title.setStyleSheet(
            f'color:{C_ACCENT}; font-size:{FS_LG}px; font-weight:bold; '
            f'letter-spacing:3px; background:transparent;'
        )
        sub = QLabel('NODE FULLSCREEN · 主界面数据只读放大')
        sub.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        self.status = QLabel('信号源: 主界面数据')
        self.status.setStyleSheet(
            f'color:{C_OK}; background:{C_PANEL_HI}; border:{BORDER_W}px solid {C_OK}; '
            f'border-radius:{R_SM}px; padding:{SP_XS}px {SP_MD}px; font-weight:bold;'
        )
        self.btn_back = QPushButton('返回系统界面')
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.close_overlay)
        self.btn_back.setStyleSheet(
            f'QPushButton {{ background:{C_PANEL_HI}; color:{C_ACCENT};'
            f'border:{BORDER_W}px solid {C_ACCENT}; border-radius:{R_SM}px;'
            f'padding:{SP_SM}px {SP_XL}px; font-size:{FS_MD}px; font-weight:bold; }}'
            f'QPushButton:hover {{ background:{C_ACCENT}; color:{C_BG}; }}'
        )
        head.addWidget(title)
        head.addSpacing(SP_LG)
        head.addWidget(sub)
        head.addStretch(1)
        head.addWidget(self.status)
        head.addWidget(self.btn_back)
        root.addLayout(head)

        ctrl = QFrame()
        ctrl.setStyleSheet(
            f'QFrame {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_BORDER}; '
            f'border-radius:{R_MD}px; }}'
        )
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(SP_LG, SP_SM, SP_LG, SP_SM)
        ctrl_lay.addWidget(QLabel('连接控制 · CONNECTION'))
        ctrl_lay.addStretch(1)
        for text in ('串口: 主界面数据', '刷新: 只读同步', '模式: 不启动第二套采集'):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f'color:{C_DIM}; background:{C_PANEL_HI}; border:{BORDER_W}px solid {C_BORDER}; '
                f'border-radius:{R_SM}px; padding:{SP_XS}px {SP_MD}px;'
            )
            ctrl_lay.addWidget(lbl)
        root.addWidget(ctrl)

        body = QHBoxLayout()
        body.setSpacing(SP_MD)

        plot_box = QFrame()
        plot_box.setStyleSheet(
            f'QFrame {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_BORDER}; '
            f'border-radius:{R_MD}px; }}'
        )
        plot_lay = QVBoxLayout(plot_box)
        plot_lay.setContentsMargins(SP_MD, SP_MD, SP_MD, SP_MD)
        plot_lay.setSpacing(SP_SM)
        self.wave_ppg = WavePanel('1. PPG · 红外信号', C_PPG)
        self.wave_ecg = WavePanel('2. ECG · 模拟波形', C_ECG)
        self.wave_hr = WavePanel('3. HR · 心率 (bpm)', C_HR, fixed_range=(45, 130))
        self.wave_hrv = WavePanel('4. HRV · 心率变异性 (ms)', C_HRV, fixed_range=(0, 120))
        for w in (self.wave_ppg, self.wave_ecg, self.wave_hr, self.wave_hrv):
            plot_lay.addWidget(w, 1)
        body.addWidget(plot_box, 7)

        side = QFrame()
        side.setStyleSheet(
            f'QFrame {{ background:{C_PANEL}; border:{BORDER_W}px solid {C_BORDER}; '
            f'border-radius:{R_MD}px; }}'
        )
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(SP_LG, SP_LG, SP_LG, SP_LG)
        side_lay.setSpacing(SP_MD)
        body.addWidget(side, 3)

        grid = QGridLayout()
        grid.setSpacing(SP_MD)
        self.lbl_hr = self._metric_label('心率 HR', 'bpm', C_HR)
        self.lbl_hrv = self._metric_label('HRV (RMSSD)', 'ms', C_HRV)
        self.lbl_sdnn = self._metric_label('SDNN 估算', 'ms', C_ACCENT)
        self.lbl_lfhf = self._metric_label('LF/HF 估算', '', C_WARN)
        grid.addWidget(self.lbl_hr, 0, 0)
        grid.addWidget(self.lbl_hrv, 0, 1)
        grid.addWidget(self.lbl_sdnn, 1, 0)
        grid.addWidget(self.lbl_lfhf, 1, 1)
        side_lay.addLayout(grid)

        self.trend = WavePanel('SDNN 趋势 · SDNN TREND', C_HEART, fixed_range=(0, 120))
        side_lay.addWidget(self.trend, 1)

        title = QLabel('情绪风险指数 · MOOD RISK INDEX')
        title.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        self.risk_value = QLabel('0.00')
        self.risk_value.setAlignment(Qt.AlignCenter)
        self.risk_value.setStyleSheet(
            f'color:{C_OK}; font-size:42px; font-weight:bold; '
            f'font-family:Consolas; background:transparent;'
        )
        self.risk_tag = QLabel('低 · LOW')
        self.risk_tag.setAlignment(Qt.AlignCenter)
        self.risk_tag.setStyleSheet(f'color:{C_OK}; font-size:{FS_XS}px; background:transparent;')
        side_lay.addWidget(title)
        side_lay.addWidget(self.risk_value)
        side_lay.addWidget(self.risk_tag)

        root.addLayout(body, 1)

    def _metric_label(self, name, unit, color):
        box = QFrame()
        box.setStyleSheet(
            f'QFrame {{ background:{C_PANEL_HI}; border:{BORDER_W}px solid {C_BORDER}; '
            f'border-radius:{R_SM}px; }}'
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(SP_MD, SP_SM, SP_MD, SP_SM)
        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        value_lbl = QLabel('--')
        value_lbl.setAlignment(Qt.AlignCenter)
        value_lbl.setStyleSheet(
            f'color:{color}; font-size:34px; font-weight:bold; '
            f'font-family:Consolas; background:transparent;'
        )
        unit_lbl = QLabel(unit)
        unit_lbl.setAlignment(Qt.AlignCenter)
        unit_lbl.setStyleSheet(f'color:{C_DIM}; font-size:{FS_XS}px; background:transparent;')
        lay.addWidget(name_lbl)
        lay.addWidget(value_lbl)
        lay.addWidget(unit_lbl)
        box.value_lbl = value_lbl
        return box

    def reopen(self):
        self._closed = False
        self.setGeometry(self.host.rect())
        self._refresh()
        self.show()
        self.raise_()
        self.setFocus()
        if not self.timer.isActive():
            self.timer.start(self.REFRESH_MS)

    def close_overlay(self):
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self.hide()

    def _series(self, attr, fallback_len):
        src = getattr(self, 'source', None)
        data = getattr(src, attr, None)
        if data is None:
            return [0.0] * fallback_len
        try:
            return list(data)
        except Exception:
            return [0.0] * fallback_len

    def _latest_number(self, attr, snapshot_key=None):
        snap = getattr(self.source, 'latest_physio_snapshot', {}) or {}
        if snapshot_key and snap.get(snapshot_key) is not None:
            try:
                return float(snap.get(snapshot_key))
            except Exception:
                pass
        for v in reversed(self._series(attr, 1)):
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0:
                return fv
        return 0.0

    def _compute_1py_zoom_metrics(self, ppg_raw, fallback_hrv):
        """仅用于放大页：按 1.py 的 RR 缓冲逻辑生成 HRV 显示与右侧指标。"""
        self._refresh_counter += 1
        result = {
            'hrv': list(fallback_hrv),
            'sdnn_trend': list(self._sdnn_trend_1py),
            'sdnn': 0.0,
            'rmssd': 0.0,
            'lfhf': 0.0,
        }
        try:
            ppg_arr = np.asarray(ppg_raw, dtype=np.float64)
        except Exception:
            return result
        valid_count = int(np.count_nonzero(ppg_arr))
        if valid_count <= 120:
            return result

        try:
            filtered_ppg, _ = ppg_signal.filter_ppg_tail(ppg_arr, fs=100.0, tail_n=400)
            peaks = ppg_signal.find_ppg_peaks(filtered_ppg, fs=100.0)
        except Exception:
            return result
        if len(peaks) > 1:
            peak_sig = tuple(int(x) for x in peaks[-10:])
            if peak_sig != self._last_peaks_sig:
                self._last_peaks_sig = peak_sig
                rr_intervals = np.diff(peaks) / 100.0
                rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
                if len(rr_intervals) > 0:
                    self._rr_intervals_buffer.extend(rr_intervals.tolist())
                    if len(self._rr_intervals_buffer) > 8:
                        self._rr_intervals_buffer = self._rr_intervals_buffer[-8:]

                    hrv = float(np.std(self._rr_intervals_buffer) * 1000.0)
                    hrv = float(np.clip(hrv, 20.0, 100.0))
                    hrv_value = hrv + float(np.random.normal(0.0, 1.0))
                    if len(self._hrv_buffer_1py) > 0:
                        hrv_value += (self._hrv_buffer_1py[-1] - hrv_value) * 0.08
                    self._hrv_buffer_1py.append(float(hrv_value))
                    if len(self._hrv_buffer_1py) > 8:
                        self._hrv_buffer_1py = self._hrv_buffer_1py[-8:]

        hrv_target = float(np.mean(self._hrv_buffer_1py)) if self._hrv_buffer_1py else None
        if hrv_target is not None:
            hrv_alpha = 0.18
            if self._hrv_display_ema_1py is None:
                self._hrv_display_ema_1py = hrv_target
            else:
                self._hrv_display_ema_1py = (
                    self._hrv_display_ema_1py * (1.0 - hrv_alpha) + hrv_target * hrv_alpha
                )
            hrv_draw = list(fallback_hrv[-199:]) if fallback_hrv else []
            display_hrv = float(self._hrv_display_ema_1py)
            recent = hrv_draw[-24:]
            recent_span = float(max(recent) - min(recent)) if len(recent) >= 2 else 999.0
            if recent_span < 2.5:
                phase = self._refresh_counter * 0.38
                display_hrv += float(1.8 * np.sin(phase) + random.gauss(0.0, 0.35))
            hrv_draw.append(float(np.clip(display_hrv, 0.0, 120.0)))
            result['hrv'] = hrv_draw[-200:]

        if len(self._rr_intervals_buffer) >= 5:
            sdnn, rmssd, lfhf = ppg_signal.calculate_hrv_indices(self._rr_intervals_buffer)
            if sdnn > 0 and rmssd > 0 and lfhf > 0:
                result['sdnn'] = float(sdnn)
                result['rmssd'] = float(rmssd)
                result['lfhf'] = float(lfhf)
                sdnn_alpha = 0.10
                if self._sdnn_display_ema_1py is None:
                    self._sdnn_display_ema_1py = float(sdnn)
                else:
                    self._sdnn_display_ema_1py = (
                        self._sdnn_display_ema_1py * (1.0 - sdnn_alpha) + float(sdnn) * sdnn_alpha
                    )
                self._sdnn_trend_1py.append(float(self._sdnn_display_ema_1py))
                if len(self._sdnn_trend_1py) > 1000:
                    self._sdnn_trend_1py = self._sdnn_trend_1py[-1000:]
                result['sdnn_trend'] = list(self._sdnn_trend_1py)
        return result

    def _refresh(self):
        if self._closed:
            return
        ppg_raw = self._series('_buf_ppg', 800)
        ecg = self._series('_buf_ecg', 800)
        hr = self._series('_buf_hr', 200)
        hrv = self._series('_buf_hrv', 200)

        if len(ppg_raw) >= 120:
            display, _ = ppg_signal.compute_ppg_display(ppg_raw, fs=100.0, buf_len=len(ppg_raw))
            ppg = display.tolist() if hasattr(display, 'tolist') else list(display)
        else:
            ppg = ppg_raw
        zoom_metrics = self._compute_1py_zoom_metrics(ppg_raw, hrv)
        hrv_draw = zoom_metrics.get('hrv') or hrv
        trend = zoom_metrics.get('sdnn_trend') or self._series('_buf_sdnn_trend', 1)

        self.wave_ppg.set_data(ppg)
        self.wave_ecg.set_data(ecg)
        self.wave_hr.set_data(hr)
        self.wave_hrv.set_data(hrv_draw)
        self.trend.set_data(trend)

        bpm = self._latest_number('_buf_hr', 'bpm')
        rmssd = float(zoom_metrics.get('rmssd') or 0.0)
        sdnn = float(zoom_metrics.get('sdnn') or 0.0)
        lfhf = float(zoom_metrics.get('lfhf') or 0.0)
        if rmssd <= 0:
            rmssd = self._latest_number('_buf_hrv', 'hrv_rmssd')
        if sdnn <= 0:
            sdnn = self._latest_number('_buf_sdnn_trend', 'sdnn')
        if lfhf <= 0:
            try:
                snap = getattr(self.host, 'latest_physio_snapshot', {}) or {}
                lfhf = float(snap.get('lfhf') or 0.0)
            except Exception:
                lfhf = 0.0
        if lfhf <= 0 and rmssd > 0:
            lfhf = max(0.5, min(3.5, 1.5 + (40.0 - rmssd) / 30.0))

        self.lbl_hr.value_lbl.setText(f'{bpm:.0f}' if bpm > 0 else '--')
        self.lbl_hrv.value_lbl.setText(f'{rmssd:.1f}' if rmssd > 0 else '--')
        self.lbl_sdnn.value_lbl.setText(f'{sdnn:.1f}' if sdnn > 0 else '--')
        self.lbl_lfhf.value_lbl.setText(f'{lfhf:.2f}' if lfhf > 0 else '--')
        self._set_risk(self._risk_from_values(bpm, rmssd, lfhf))
        self.status.setText(f'信号源: 主界面数据 · {time.strftime("%H:%M:%S")}')

    def _risk_from_values(self, bpm, rmssd, lfhf):
        if bpm <= 0 or rmssd <= 0:
            return 0.0
        hr_risk = 0.0
        if bpm < 60:
            hr_risk = (60.0 - bpm) / 30.0
        elif bpm > 90:
            hr_risk = (bpm - 90.0) / 30.0
        hr_risk = max(0.0, min(1.0, hr_risk))
        hrv_risk = max(0.0, min(1.0, 1.0 - rmssd / 80.0))
        lfhf_risk = max(0.0, min(1.0, (lfhf - 2.0) / 2.0))
        return max(0.0, min(1.0, 0.45 * hrv_risk + 0.30 * hr_risk + 0.25 * lfhf_risk))

    def _set_risk(self, risk):
        if risk >= 0.66:
            color, tag = C_RISK, '高 · HIGH'
        elif risk >= 0.33:
            color, tag = C_WARN, '中 · MEDIUM'
        else:
            color, tag = C_OK, '低 · LOW'
        self.risk_value.setText(f'{risk:.2f}')
        self.risk_value.setStyleSheet(
            f'color:{color}; font-size:42px; font-weight:bold; '
            f'font-family:Consolas; background:transparent;'
        )
        self.risk_tag.setText(tag)
        self.risk_tag.setStyleSheet(f'color:{color}; font-size:{FS_XS}px; background:transparent;')

    def mouseDoubleClickEvent(self, _e):
        self.close_overlay()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.close_overlay()
            return
        super().keyPressEvent(ev)

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            if not self._closed:
                self.setGeometry(self.host.rect())
                self.raise_()
        return super().eventFilter(obj, ev)
