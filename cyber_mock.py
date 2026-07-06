import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QRadialGradient, QFont, QPainterPath, QBrush
from PyQt5.QtWidgets import QWidget


class SocialBatteryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.72
        self.setMinimumSize(220, 220)

    def set_level(self, value: float):
        try:
            v = float(value)
        except Exception:
            return
        self._level = max(0.0, min(1.0, v))
        self.update()

    def _level_color(self):
        if self._level > 0.6:
            return QColor("#00f3ff")
        if self._level >= 0.3:
            return QColor("#ffb703")
        return QColor("#ff007f")

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        side = min(w, h)
        cx, cy = w * 0.5, h * 0.5

        ring_r = side * 0.34
        ring_w = max(10.0, side * 0.075)
        color = self._level_color()

        # 外部泛光
        halo = QRadialGradient(cx, cy, ring_r * 1.8)
        halo.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 90))
        halo.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        p.setPen(Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QPointF(cx, cy), ring_r * 1.8, ring_r * 1.8)

        # 基础环
        base_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2.0, ring_r * 2.0)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(31, 41, 55, 220), ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(base_rect, 0, 360 * 16)

        # 能量环（270° 扫描）
        start_deg = 225.0
        span_deg = 270.0 * self._level
        grad = QLinearGradient(base_rect.left(), base_rect.top(), base_rect.right(), base_rect.bottom())
        grad.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 230))
        grad.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 160))
        p.setPen(QPen(grad, ring_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(base_rect, int(start_deg * 16), int(-span_deg * 16))

        # 内芯板
        core_rect = QRectF(cx - ring_r * 0.63, cy - ring_r * 0.63, ring_r * 1.26, ring_r * 1.26)
        core_bg = QLinearGradient(core_rect.left(), core_rect.top(), core_rect.right(), core_rect.bottom())
        core_bg.setColorAt(0.0, QColor(8, 16, 28, 240))
        core_bg.setColorAt(1.0, QColor(5, 10, 18, 240))
        p.setPen(QPen(QColor(75, 85, 99, 120), 1.0))
        p.setBrush(core_bg)
        p.drawRoundedRect(core_rect, 18, 18)

        # 文本
        percent = int(round(self._level * 100.0))
        p.setPen(color)
        p.setFont(QFont("Arial", max(18, int(side * 0.10)), QFont.Bold))
        p.drawText(core_rect, Qt.AlignCenter, f"{percent}%")

        sub_rect = QRectF(core_rect.left(), core_rect.bottom() + 10, core_rect.width(), 20)
        p.setPen(QColor(148, 163, 184))
        p.setFont(QFont("微软雅黑", 9, QFont.Bold))
        p.drawText(sub_rect, Qt.AlignCenter, "社交电量槽")


class PredictionCurveWidget(QWidget):
    def __init__(self, parent=None, max_points=240):
        super().__init__(parent)
        self._max_points = max(60, int(max_points))
        self.history_data = []
        self.future_data = []
        self.setMinimumHeight(150)

    def set_data(self, history_data, future_data):
        self.history_data = [float(v) for v in (history_data or [])][-self._max_points:]
        self.future_data = [float(v) for v in (future_data or [])][-self._max_points:]
        self.update()

    def _map_points(self, values, x0, x1, vmin, vmax, top, bottom):
        if not values:
            return []
        rng = vmax - vmin
        if rng <= 1e-12:
            y = (top + bottom) * 0.5
            if len(values) == 1:
                return [QPointF((x0 + x1) * 0.5, y)]
            dx = (x1 - x0) / max(1, len(values) - 1)
            return [QPointF(x0 + i * dx, y) for i in range(len(values))]

        if len(values) == 1:
            t = (values[0] - vmin) / rng
            y = bottom - t * (bottom - top)
            return [QPointF((x0 + x1) * 0.5, y)]

        dx = (x1 - x0) / max(1, len(values) - 1)
        pts = []
        for i, v in enumerate(values):
            t = (v - vmin) / rng
            y = bottom - t * (bottom - top)
            pts.append(QPointF(x0 + i * dx, y))
        return pts

    def _build_smooth_path(self, pts):
        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(pts[0])
        if len(pts) == 1:
            return path
        x0, y0 = pts[0].x(), pts[0].y()
        for i in range(1, len(pts)):
            x1, y1 = pts[i].x(), pts[i].y()
            xm = (x0 + x1) * 0.5
            ym = (y0 + y1) * 0.5
            path.quadTo(QPointF(x0, y0), QPointF(xm, ym))
            x0, y0 = x1, y1
        path.lineTo(QPointF(x0, y0))
        return path

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w, h = float(self.width()), float(self.height())
        if w <= 4 or h <= 4:
            return

        left, right = 10.0, w - 10.0
        top, bottom = 10.0, h - 12.0

        p.setPen(QPen(QColor(0, 243, 255, 22), 1))
        for i in range(1, 4):
            y = top + (bottom - top) * i / 4.0
            p.drawLine(int(left), int(y), int(right), int(y))

        all_vals = self.history_data + self.future_data
        if len(all_vals) < 2:
            return

        vmin, vmax = min(all_vals), max(all_vals)
        split_x = left + (right - left) * 0.68

        hist_pts = self._map_points(self.history_data, left, split_x, vmin, vmax, top, bottom)
        fut_pts = self._map_points(self.future_data, split_x, right, vmin, vmax, top, bottom)

        if hist_pts:
            hist_path = self._build_smooth_path(hist_pts)
            p.setPen(QPen(QColor("#00f3ff"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(hist_path)

        if fut_pts:
            anchor = hist_pts[-1] if hist_pts else fut_pts[0]
            fut_chain = [anchor] + fut_pts
            fut_path = self._build_smooth_path(fut_chain)

            glow = QLinearGradient(split_x, top, right, bottom)
            glow.setColorAt(0.0, QColor(255, 149, 0, 200))
            glow.setColorAt(1.0, QColor(255, 149, 0, 30))
            p.setPen(QPen(QBrush(glow), 2.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(fut_path)

            fade_path = QPainterPath(fut_path)
            fade_path.lineTo(QPointF(right, bottom))
            fade_path.lineTo(QPointF(split_x, bottom))
            fade_path.closeSubpath()
            fill_grad = QLinearGradient(split_x, top, right, bottom)
            fill_grad.setColorAt(0.0, QColor(255, 149, 0, 46))
            fill_grad.setColorAt(1.0, QColor(255, 149, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(fill_grad))
            p.drawPath(fade_path)


class MockWaveformThread(QThread):
    # 一次性发射四路波形: ppg, ecg, hr, hrv
    data_signal = pyqtSignal(float, float, float, float)
    # 仪表盘五维风险输入: vis, hrv_risk, eeg_risk, eye_fatigue, posture_risk
    dashboard_signal = pyqtSignal(float, float, float, float, float)

    def __init__(self, fs=50.0, parent=None):
        super().__init__(parent)
        self.fs = float(fs)
        self._running = True

        self._rng = np.random.default_rng()
        self._t = 0.0
        self._dt = 1.0 / self.fs

        self._hr = 75.0

    def stop(self):
        self._running = False

    def run(self):
        # 并发引擎：每个周期同时生成四路数据
        while self._running:
            self._t += self._dt

            # 1) PPG: 基线10000 + 心跳主波 + 呼吸低频漂移 + 少量噪声
            heart_hz = 1.2  # ~72 BPM
            resp_hz = 0.25
            ppg = (
                10000.0
                + 650.0 * np.sin(2.0 * np.pi * heart_hz * self._t)
                + 180.0 * np.sin(2.0 * np.pi * resp_hz * self._t)
                + 35.0 * self._rng.normal()
            )

            # 2) ECG: 基线0 + 尖锐QRS窄脉冲 + 微弱P/T波 + 轻噪声
            # 以心率驱动周期
            period = 60.0 / max(45.0, min(120.0, self._hr))
            phase = (self._t % period) / period

            # QRS 使用高斯窄脉冲
            qrs = 1.4 * np.exp(-((phase - 0.30) ** 2) / (2.0 * (0.010 ** 2)))
            # Q 波与 S 波（负向窄波）
            q_wave = -0.25 * np.exp(-((phase - 0.285) ** 2) / (2.0 * (0.006 ** 2)))
            s_wave = -0.35 * np.exp(-((phase - 0.325) ** 2) / (2.0 * (0.008 ** 2)))
            # P / T 波
            p_wave = 0.12 * np.exp(-((phase - 0.18) ** 2) / (2.0 * (0.025 ** 2)))
            t_wave = 0.30 * np.exp(-((phase - 0.58) ** 2) / (2.0 * (0.05 ** 2)))
            ecg = qrs + q_wave + s_wave + p_wave + t_wave + 0.02 * self._rng.normal()

            # 3) HR: 均值75 + 缓慢随机游走
            self._hr += self._rng.normal(0.0, 0.06)
            self._hr = float(np.clip(self._hr, 68.0, 84.0))
            hr = self._hr

            # 4) HRV: 均值40 + 低频波动 + 少量噪声
            hrv = (
                40.0
                + 6.0 * np.sin(2.0 * np.pi * 0.07 * self._t)
                + 1.2 * np.sin(2.0 * np.pi * 0.02 * self._t + 0.7)
                + 0.8 * self._rng.normal()
            )
            hrv = float(np.clip(hrv, 20.0, 65.0))

            # 新增测试维度：眼疲劳、姿态风险（小幅波动）
            eye_fatigue = 0.38 + 0.12 * np.sin(2.0 * np.pi * 0.035 * self._t + 0.4) + 0.02 * self._rng.normal()
            posture_risk = 0.32 + 0.10 * np.sin(2.0 * np.pi * 0.028 * self._t + 1.1) + 0.02 * self._rng.normal()
            eye_fatigue = float(np.clip(eye_fatigue, 0.0, 1.0))
            posture_risk = float(np.clip(posture_risk, 0.0, 1.0))

            # mock 仪表盘其他维度（用于无硬件联调）
            vis_risk = float(np.clip(0.40 + 0.20 * np.sin(2.0 * np.pi * 0.020 * self._t), 0.0, 1.0))
            hrv_risk = float(np.clip(abs(hr - 75.0) / 30.0, 0.0, 1.0))
            eeg_risk = float(np.clip(0.35 + 0.15 * np.sin(2.0 * np.pi * 0.017 * self._t + 0.8), 0.0, 1.0))

            self.data_signal.emit(float(ppg), float(ecg), float(hr), float(hrv))
            self.dashboard_signal.emit(vis_risk, hrv_risk, eeg_risk, eye_fatigue, posture_risk)
            self.msleep(int(max(5, 1000.0 / self.fs)))
