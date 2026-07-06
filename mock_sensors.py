import random
import time
import math

class MockHardwareSensors:
    def __init__(self):
        # 初始状态设为健康
        self.state = "depressed" 
        # 记录启动时间，用于生成平滑的正弦波形
        self.start_time = time.time()

    def set_simulation_state(self, state):
        """支持设置三种状态: normal(正常), stressed(轻度压力), depressed(抑郁倾向)"""
        self.state = state

    def get_hrv_data(self):
        """模拟读取 ESP32 手环数据 (高保真平滑波形版)"""
        elapsed = time.time() - self.start_time
        
        # 基础波动 (模拟呼吸和自然生理律动)
        wave1 = math.sin(elapsed * 0.5)
        wave2 = math.cos(elapsed * 0.2)
        
        if self.state == "normal":
            return {
                "rmssd": 60 + wave1 * 8,       # 52~68 (健康)
                "sdnn": 110 + wave2 * 10,      # 100~120
                "hf": 700 + wave1 * 50,        # 650~750
                "lf_hf": 1.5 + wave2 * 0.3,    # 1.2~1.8
                "hr": 65 + wave1 * 5           # 60~70
            }
        elif self.state == "depressed":
            # 模拟高风险抑郁状态 (完美触发你的高危阈值)
            return {
                "rmssd": 25 + wave1 * 5,       # 20~30 (极低，高危)
                "sdnn": 55 + wave2 * 10,       # 45~65
                "hf": 200 + wave1 * 50,        # 150~250
                "lf_hf": 4.8 + wave2 * 0.5,    # 4.3~5.3
                "hr": 92 + wave1 * 6           # 86~98 (心率偏高)
            }
        else: # stressed
            return {
                "rmssd": 42 + wave1 * 5,       # 37~47 (中度)
                "sdnn": 85 + wave2 * 10,       # 75~95
                "hf": 450 + wave1 * 50,        # 400~500
                "lf_hf": 3.3 + wave2 * 0.4,    # 2.9~3.7
                "hr": 80 + wave1 * 4           # 76~84
            }

    def get_eeg_data(self):
        """模拟读取 TGAM 脑电模块数据 (平滑波形版)"""
        elapsed = time.time() - self.start_time
        wave = math.sin(elapsed * 0.3)
        
        if self.state == "normal":
            # 专注和放松都在良好区间
            attention = 70 + wave * 10
            meditation = 60 - wave * 10
            return {"attention": int(attention), "meditation": int(meditation)}
        elif self.state == "depressed":
            # 专注度极低，放松度走极端
            attention = 25 + wave * 10
            # 模拟偶尔走神(极低)和偶尔极度放空(极高)的两极化
            meditation = 20 if math.cos(elapsed * 0.1) > 0 else 90 
            return {"attention": int(attention), "meditation": int(meditation)}
        else:
            return {"attention": int(50 + wave * 8), "meditation": int(35 + wave * 4)}