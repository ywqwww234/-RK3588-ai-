import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from scipy.signal import butter, lfilter, find_peaks

# -------------------------- 1. 自动适配串口号 --------------------------
def get_esp32_port():
    try:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "USB Serial" in port.description or "CH340" in port.description or "ESP32" in port.description:
                return port.device
    except:
        pass
    return None

ser_port = get_esp32_port()
ser = None
if ser_port:
    try:
        ser = serial.Serial(
            port=ser_port,
            baudrate=115200,
            timeout=0.002,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        print(f"成功连接ESP32，串口号：{ser_port}")
    except:
        ser = None
        print(f"连接ESP32失败，串口号：{ser_port}")
else:
    print("未找到ESP32设备！请检查接线、驱动和Arduino串口监视器是否关闭")
    print("将使用模拟数据进行演示")

# -------------------------- 2. 滤波参数 --------------------------
fs = 400
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smooth_signal(data, window_size=5):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# -------------------------- 3. 界面 --------------------------
pg.setConfigOptions(antialias=True)
app = pg.mkQApp()

win = pg.GraphicsLayoutWidget(title="REAL-TIME PPG/ECG/HR/HRV MONITOR")
win.resize(1000, 800)
win.setBackground('w')

# PPG
p1 = win.addPlot(title="1. PPG (红外信号)")
p1.setMenuEnabled(False)
p1.showGrid(x=True, y=True, alpha=0.3)
curve_ppg = p1.plot(pen=pg.mkPen('b', width=2))
data_ppg = [0] * 800

# ECG
win.nextRow()
p2 = win.addPlot(title="2. ECG (模拟)")
p2.setMenuEnabled(False)
p2.showGrid(x=True, y=True, alpha=0.3)
curve_ecg = p2.plot(pen=pg.mkPen('r', width=1.5))
data_ecg = [0] * 800

# HR
win.nextRow()
p3 = win.addPlot(title="3. HR (心率)")
p3.setMenuEnabled(False)
p3.showGrid(x=True, y=True, alpha=0.3)
curve_hr = p3.plot(pen=pg.mkPen('g', width=1.5))
data_hr = [75] * 200

# HRV
win.nextRow()
p4 = win.addPlot(title="4. HRV (心率变异性)")
p4.setMenuEnabled(False)
p4.showGrid(x=True, y=True, alpha=0.3)
curve_hrv = p4.plot(pen=pg.mkPen('m', width=1.5))
data_hrv = [40] * 200

win.show()

# -------------------------- 4. 核心更新 --------------------------
his_peak_time = 0
hrv_buffer = []
rr_intervals_buffer = []
ecg_last_peak = 0
counter = 0
sim_counter = 0

def update():
    global data_ppg, data_hr, data_hrv, data_ecg
    global hrv_buffer, rr_intervals_buffer, counter, sim_counter

    # 处理串口数据
    if ser and ser.is_open:
        try:
            if ser.in_waiting > 0:
                # 一次性读完，不卡
                raw = ser.read(ser.in_waiting)
                lines = raw.decode('utf-8', errors='ignore').split('\n')
                # 批量处理数据，减少操作次数
                new_data = []
                for line in lines:
                    line = line.strip()
                    try:
                        if "[DATA]" in line:
                            # 提取数据部分
                            data_part = line.split("[DATA] ")[1]
                            # 分割各个数据项
                            data_items = data_part.split(",")
                            ir_val = 0
                            for item in data_items:
                                if "ir=" in item:
                                    ir_val = int(item.split("=")[1])
                            # 使用IR值作为PPG信号
                            new_data.append(ir_val)
                        # 兼容旧格式
                        elif "," in line:
                            parts = line.split(',')
                            if len(parts) == 2:
                                try:
                                    ir_val = int(parts[0])
                                    new_data.append(ir_val)
                                except:
                                    pass
                    except:
                        continue
                # 批量添加数据
                data_ppg.extend(new_data)
                # 保持缓冲区大小
                if len(data_ppg) > 800:
                    data_ppg = data_ppg[-800:]
        except:
            pass
    else:
        # 生成模拟数据 - 更接近真实PPG波形
        for _ in range(2):  # 每次循环生成2个数据点，提高流畅度
            sim_counter += 1
            t = sim_counter * 0.003  # 适当的时间步长
            
            # 生成更接近真实PPG的波形
            # 基础心率波形 (约1Hz，60BPM)
            heart_rate = 1.0
            # 呼吸调制 (约0.2Hz)
            respiration = 0.2
            # 生成PPG波形，包含基础波、呼吸调制和高频噪声
            ir_val = 10000 + \
                     2000 * np.sin(2 * np.pi * heart_rate * t) + \
                     500 * np.sin(2 * np.pi * respiration * t) + \
                     100 * np.sin(2 * np.pi * 10 * t) + \
                     np.random.randn() * 50
            
            # 直接添加到数据缓冲区，不经过字符串解析
            data_ppg.append(int(ir_val))
            if len(data_ppg) > 800:
                data_ppg.pop(0)

    if len(data_ppg) >= 20:  # 进一步减少等待时间，从30改为20
        # 调整滤波参数，使PPG周期更短，增加高频截止频率
        # 使用较低的滤波器阶数，减少计算时间
        filtered_ppg = butter_bandpass_filter(data_ppg, 0.5, 20, fs, order=2)  # 适当的高频截止频率
        # 应用平滑处理，使PPG波形更圆滑连续
        filtered_ppg = smooth_signal(filtered_ppg, window_size=3)  # 适当的窗口大小，平衡平滑度和响应速度

        # 松手归零 - 立即反应，手松直接变直线
        if np.max(np.abs(filtered_ppg[-20:])) < 20:  # 进一步减少检测窗口，降低阈值，提高敏感度
            # 四个曲线都归0成直线
            filtered_ppg = np.zeros_like(filtered_ppg)
            data_hr = [0]*200
            data_hrv = [0]*200
            data_ecg = [0]*800
            curve_ppg.setData(filtered_ppg)
            curve_hr.setData(data_hr)
            curve_hrv.setData(data_hrv)
            curve_ecg.setData(data_ecg)
            # 重置缓冲区
            hrv_buffer = []
            rr_intervals_buffer = []
            return

        curve_ppg.setData(filtered_ppg)

        # 峰值检测 - 更敏感，使PPG周期更短
        peaks, _ = find_peaks(
            filtered_ppg,
            height=np.mean(filtered_ppg) + np.std(filtered_ppg)*0.4,
            distance=int(fs * 0.25)
        )

        # HR 计算 - 减少等待时间
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
            if len(rr_intervals) > 0:
                hr = 60 / np.mean(rr_intervals)
                hr = np.clip(hr, 50, 100)
                # 加更大的波动
                hr += np.random.normal(0, 1.5)
                data_hr.append(hr)
                if len(data_hr) > 200:
                    data_hr.pop(0)
                curve_hr.setData(data_hr)

                # HRV 计算 - 与HR一样波动，避免周期性
                rr_intervals_buffer.extend(rr_intervals)
                if len(rr_intervals_buffer) > 8:  # 增加缓冲区大小，使曲线更平滑
                    rr_intervals_buffer = rr_intervals_buffer[-8:]
                hrv = np.std(rr_intervals_buffer) * 1000
                hrv = np.clip(hrv, 20, 100)
                
                # 生成与HR类似的波动，避免周期性
                hrv_value = hrv
                
                # 添加随机波动，使曲线更自然
                hrv_value += np.random.normal(0, 3.0)
                
                # 添加一些缓慢的趋势变化，避免周期性
                if len(hrv_buffer) > 0:
                    hrv_value += (hrv_buffer[-1] - hrv_value) * 0.1
                
                hrv_buffer.append(hrv_value)
                if len(hrv_buffer) > 6:  # 增加缓冲区大小，使曲线更平滑
                    hrv_buffer = hrv_buffer[-6:]
                
                # 使用移动平均，使曲线与HR类似
                data_hrv.append(np.mean(hrv_buffer))
                if len(data_hrv) > 200:
                    data_hrv.pop(0)
                curve_hrv.setData(data_hrv)

        # ---------------- ECG 恢复到之前的状态 ----------------
        counter += 1
        t = counter * 0.01  # 恢复时间步长
        
        # 模拟ECG数据
        ecg_value = 0
        phase = t % 1
        if 0.1 < phase < 0.2:
            ecg_value = 0.1 * np.sin((phase - 0.1) * 30 * np.pi)
        elif 0.2 < phase < 0.25:
            ecg_value = 0.8 * np.sin((phase - 0.2) * 100 * np.pi)
        elif 0.25 < phase < 0.3:
            ecg_value = -0.4 * np.sin((phase - 0.25) * 80 * np.pi)
        elif 0.3 < phase < 0.5:
            ecg_value = 0.3 * np.sin((phase - 0.4) * 40 * np.pi)
        
        # 添加少量噪声
        ecg_value += np.random.randn() * 0.03

        # 每次循环添加1个ECG数据点，恢复到之前的速率
        data_ecg.append(ecg_value)
        if len(data_ecg) > 800:
            data_ecg.pop(0)
        curve_ecg.setData(data_ecg)

# 更流畅的刷新
timer = QTimer()
timer.timeout.connect(update)
timer.start(0)  # 尽可能快地刷新

app.exec_()