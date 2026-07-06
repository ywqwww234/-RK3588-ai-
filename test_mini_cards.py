"""
MiniCard 测试程序 - 展示新设计的卡片效果

运行此文件可以看到三种卡片的实时效果：
- 心率卡片（HeartRateMiniCard）
- 脑电卡片（BrainMiniCard）
- 视觉卡片（VisionMiniCard）
"""

import sys
import random
import math
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from ui_mini_cards import HeartRateMiniCard, BrainMiniCard, VisionMiniCard
from theme import C_BG_DEEP


class MiniCardDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MiniCard 组件演示 - 心率/脑电/视觉卡片')
        self.setStyleSheet(f'background:{C_BG_DEEP};')
        self.resize(1200, 400)

        # 主容器
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 创建三个卡片
        self.heart_card = HeartRateMiniCard()
        self.brain_card = BrainMiniCard()
        self.vision_card = VisionMiniCard()

        layout.addWidget(self.heart_card)
        layout.addWidget(self.brain_card)
        layout.addWidget(self.vision_card)

        # 模拟数据生成
        self._phase = 0.0
        self._bpm = 72.0
        self._hrv = 35.0
        self._attention = 65
        self._meditation = 55
        self._poor_signal = 15

        # 定时更新
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100ms更新一次
        self.timer.timeout.connect(self._update_data)
        self.timer.start()

    def _update_data(self):
        """模拟实时数据更新。"""
        self._phase += 0.1

        # 模拟心率数据
        self._bpm += random.uniform(-1.5, 1.5)
        self._bpm = max(55, min(95, self._bpm))

        self._hrv += random.uniform(-2, 2)
        self._hrv = max(20, min(60, self._hrv))

        # 生成心率波形数据（模拟PPG信号）
        wave_value = 0.5 + 0.3 * math.sin(self._phase * 2) + random.uniform(-0.05, 0.05)

        # 更新心率卡片
        quality = 'Good' if self._hrv > 30 else 'Fair' if self._hrv > 20 else 'Poor'
        self.heart_card.update_data(
            bpm=self._bpm,
            hrv=self._hrv,
            quality=quality,
            wave_data=wave_value
        )

        # 模拟脑电数据
        self._attention += random.randint(-3, 3)
        self._attention = max(0, min(100, self._attention))

        self._meditation += random.randint(-2, 2)
        self._meditation = max(0, min(100, self._meditation))

        self._poor_signal += random.randint(-5, 5)
        self._poor_signal = max(0, min(100, self._poor_signal))

        # 更新脑电卡片
        self.brain_card.update_data(
            attention=self._attention,
            meditation=self._meditation,
            poor_signal=self._poor_signal
        )

        # 模拟视觉数据
        expressions = ['开心', '平静', '悲伤', '专注']
        expr = random.choice(expressions)
        expr_prob = random.uniform(0.75, 0.98)
        eye_fatigue = random.uniform(0.2, 0.8)
        posture_risk = random.uniform(0.1, 0.7)

        # 更新视觉卡片
        self.vision_card.update_data(
            expr=expr,
            expr_prob=expr_prob,
            eye_fatigue=eye_fatigue,
            posture_risk=posture_risk
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MiniCardDemo()
    demo.show()
    sys.exit(app.exec_())
