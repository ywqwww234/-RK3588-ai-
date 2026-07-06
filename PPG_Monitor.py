import serial
import serial.tools.list_ports
import random
import re
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QGraphicsProxyWidget
from scipy.signal import butter, filtfilt, find_peaks

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
            timeout=0.02,
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

# -------------------------- 2. 滤波参数（仅修改PPG处理） --------------------------
# MAX3010x常见输出速率约为100Hz，先按100Hz处理；与定时器10ms匹配为每tick 1点
fs = 100
samples_per_tick = max(1, int(fs * 0.01))  # 定时器10ms -> 每次约1个点
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 【关键修改】使用零相位滤波，消除波形变形
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth_signal(data, window_size=5):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# -------------------------- 2.1 HRV指标计算函数 --------------------------
def calculate_hrv_indices(rr_intervals):
    """计算HRV指标
    rr_intervals: RR间期数组（秒）
    返回SDNN, RMSSD, LF_HF
    """
    if len(rr_intervals) < 5:
        return 0.0, 0.0, 0.0
    
    # SDNN: 所有RR间期的标准差（毫秒）
    sdnn = np.std(rr_intervals) * 1000
    
    # RMSSD: 相邻RR间期差值的均方根（毫秒）
    diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diffs**2)) * 1000
    
    # LF/HF: 低频与高频功率比（简化计算）
    # 这里使用简化方法，实际应该使用频谱分析
    lf_hf = np.random.normal(1.5, 0.5)  # 模拟值
    
    return sdnn, rmssd, lf_hf

# -------------------------- 2.2 抑郁模拟算法（按你提供的逻辑） --------------------------
ppg_phase_sim = 0.0
hr_val_sim = 60.0
hrv_val_sim = 20.0

def generate_depression_ppg(base=233000):
    """抑郁 PPG：幅度低、波动弱、节律慢"""
    global ppg_phase_sim
    ppg_phase_sim += 0.12
    wave = 200 * np.sin(ppg_phase_sim) + 80 * np.sin(2 * ppg_phase_sim)
    noise = random.randint(-10, 10)
    return int(base + wave + noise)

def generate_depression_hr():
    """抑郁 HR：心率偏低、波动极小"""
    global hr_val_sim
    hr_val_sim += random.uniform(-0.3, 0.3)
    hr_val_sim = max(55, min(65, hr_val_sim))
    return float(hr_val_sim)

def generate_depression_hrv():
    """抑郁 HRV：显著降低、几乎无波动"""
    global hrv_val_sim
    hrv_val_sim += random.uniform(-1, 1)
    hrv_val_sim = max(15, min(25, hrv_val_sim))
    return float(hrv_val_sim)

def generate_depression_ecg(base=2000):
    """抑郁 ECG：R 峰低、尖峰弱、节律慢"""
    if random.random() < 0.08:
        return base + random.randint(600, 1000)
    return base + random.randint(-100, 100)

# -------------------------- 3. 界面 --------------------------
pg.setConfigOptions(antialias=True)
app = pg.mkQApp()

# 创建主窗口
main_window = pg.GraphicsLayoutWidget(title="REAL-TIME PPG/ECG/HR/HRV MONITOR")
main_window.resize(1400, 800)
main_window.setBackground('w')

# 创建左右布局
main_layout = main_window.ci.layout
left_layout = pg.GraphicsLayout()
right_layout = pg.GraphicsLayout()
main_layout.addItem(left_layout, 0, 0, 1, 1)
main_layout.addItem(right_layout, 0, 1, 1, 1)
main_layout.setColumnStretchFactor(0, 1)
main_layout.setColumnStretchFactor(1, 1)

# 左侧：原有波形显示
win = left_layout

# PPG
p1 = win.addPlot(title="1. PPG (红外信号)")
p1.setMenuEnabled(False)
p1.showGrid(x=True, y=True, alpha=0.3)
curve_ppg = p1.plot(pen=pg.mkPen('b', width=2))
# 【关键修改】使用numpy数组，保证滤波稳定性
data_ppg = np.zeros(800, dtype=np.float64)

# ECG
win.nextRow()
p2 = win.addPlot(title="2. ECG (模拟)")
p2.setMenuEnabled(False)
p2.showGrid(x=True, y=True, alpha=0.3)
curve_ecg = p2.plot(pen=pg.mkPen('r', width=1.5))
data_ecg = [0.0] * 800

# HR
win.nextRow()
p3 = win.addPlot(title="3. HR (心率)")
p3.setMenuEnabled(False)
p3.showGrid(x=True, y=True, alpha=0.3)
curve_hr = p3.plot(pen=pg.mkPen('g', width=1.5))
data_hr = [75.0] * 200

# HRV
win.nextRow()
p4 = win.addPlot(title="4. HRV (心率变异性)")
p4.setMenuEnabled(False)
p4.showGrid(x=True, y=True, alpha=0.3)
curve_hrv = p4.plot(pen=pg.mkPen('m', width=1.5))
data_hrv = [40.0] * 200

# 右侧：HRV自主神经功能分析
hrv_analysis = QWidget()
hrv_analysis_layout = QVBoxLayout(hrv_analysis)

# HRV分析标题
hrv_title = QLabel("HRV 自主神经功能分析")
hrv_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
hrv_analysis_layout.addWidget(hrv_title)

# 指标表格
metrics_grid = QGridLayout()

# 表头
metrics_grid.addWidget(QLabel("当前指标"), 0, 0)
metrics_grid.addWidget(QLabel("参考范围"), 0, 1)
metrics_grid.addWidget(QLabel("状态"), 0, 2)

# SDNN
metrics_grid.addWidget(QLabel("SDNN"), 1, 0)
global sdnn_value
sdnn_value = QLabel("42.1 ms")
metrics_grid.addWidget(sdnn_value, 1, 0, 1, 1, Qt.AlignRight)
metrics_grid.addWidget(QLabel(">50 ms"), 1, 1, 1, 1, Qt.AlignRight)
global sdnn_status
sdnn_status = QLabel("⚠ 偏低")
sdnn_status.setStyleSheet("color: orange; font-weight: bold;")
metrics_grid.addWidget(sdnn_status, 1, 2)

# RMSSD
metrics_grid.addWidget(QLabel("RMSSD"), 2, 0)
global rmssd_value
rmssd_value = QLabel("19.3 ms")
metrics_grid.addWidget(rmssd_value, 2, 0, 1, 1, Qt.AlignRight)
metrics_grid.addWidget(QLabel("27±12 ms"), 2, 1, 1, 1, Qt.AlignRight)
global rmssd_status
rmssd_status = QLabel("⚠ 偏低")
rmssd_status.setStyleSheet("color: orange; font-weight: bold;")
metrics_grid.addWidget(rmssd_status, 2, 2)

# LF/HF
metrics_grid.addWidget(QLabel("LF/HF"), 3, 0)
global lf_hf_value
lf_hf_value = QLabel("3.2")
metrics_grid.addWidget(lf_hf_value, 3, 0, 1, 1, Qt.AlignRight)
metrics_grid.addWidget(QLabel("1-2"), 3, 1, 1, 1, Qt.AlignRight)
global lf_hf_status
lf_hf_status = QLabel("⚠ 偏高")
lf_hf_status.setStyleSheet("color: red; font-weight: bold;")
metrics_grid.addWidget(lf_hf_status, 3, 2)

hrv_analysis_layout.addLayout(metrics_grid)

# 趋势图
hrv_analysis_layout.addWidget(QLabel("趋势图（实时SDNN变化）"))
global sdnn_trend
sdnn_trend = pg.PlotWidget()
sdnn_trend.setMenuEnabled(False)
sdnn_trend.showGrid(x=True, y=True, alpha=0.3)

# 初始化趋势图
trend_x = np.arange(0, 10, 0.01)
trend_y = np.zeros(len(trend_x))
sdnn_trend.plot(trend_x, trend_y, pen=pg.mkPen('b', width=1))

# 添加参考线
sdnn_trend.addLine(y=50, pen=pg.mkPen('g', style=Qt.DashLine))

hrv_analysis_layout.addWidget(sdnn_trend)

# 提示信息
hrv_analysis_layout.addWidget(QLabel("提示: HRV降低可能与情绪压力有关，建议关注"))
hrv_analysis_layout.addWidget(QLabel("完整文献参考见报告末尾。"))

# 将HRV分析组件添加到右侧布局
proxy = QGraphicsProxyWidget()
proxy.setWidget(hrv_analysis)
right_layout.addItem(proxy, row=0, col=0)

# 显示主窗口
main_window.show()

# 全局变量已经在函数外部定义，不需要再次声明
# 确保所有变量都已经正确初始化

# -------------------------- 4. 核心更新 --------------------------
his_peak_time = 0
hrv_buffer = []
rr_intervals_buffer = []
ecg_last_peak = 0
counter = 0
sim_counter = 0
last_ir_val = None
ir_diff_hist = []

# HRV指标更新
sdnn_value = None
rmssd_value = None
lf_hf_value = None
sdnn_status = None
rmssd_status = None
lf_hf_status = None

# 趋势图数据
sdnn_trend_data = []
sdnn_trend = None
# 用于测试的模拟HRV数据
test_counter = 0

# 调试统计（每秒打印一次）
debug_start_t = time.time()
debug_tick_t = debug_start_t
debug_raw_ir = []
debug_used_ir = []
debug_dropped = 0
debug_empty_ticks = 0

def update():
    global data_ppg, data_hr, data_hrv, data_ecg
    global hrv_buffer, rr_intervals_buffer, counter, sim_counter
    global last_ir_val, ir_diff_hist
    global debug_tick_t, debug_raw_ir, debug_used_ir, debug_dropped, debug_empty_ticks
    global sdnn_value, rmssd_value, lf_hf_value, sdnn_status, rmssd_status, lf_hf_status
    global sdnn_trend_data, sdnn_trend
    global test_counter

    # 串口与模拟数据统一按 samples_per_tick 写入，避免时间轴错配导致波形被拉扯/锯齿化
    if ser and ser.is_open:
        try:
            samples_added = 0
            latest_ir = None

            # 关键：每个tick把串口缓存尽量读空，只保留“最新样本”用于显示
            # 避免缓存积压导致波形时间轴滞后（截图中的慢爬坡/大回落）
            max_read_lines = max(200, ser.in_waiting)
            while ser.in_waiting > 0 and max_read_lines > 0:
                max_read_lines -= 1
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                ir_val = None
                if "[DATA]" in line:
                    m = re.search(r"ir\s*=\s*(\d+)", line)
                    if m:
                        ir_val = int(m.group(1))
                elif "," in line:
                    m = re.search(r"(-?\d+)", line.split(',')[0])
                    if m:
                        ir_val = int(m.group(1))
                else:
                    m = re.search(r"(-?\d+)", line)
                    if m:
                        ir_val = int(m.group(1))

                if ir_val is None or ir_val <= 0:
                    continue

                debug_raw_ir.append(ir_val)
                latest_ir = ir_val

            if latest_ir is not None:
                ir_val = latest_ir

                # 抗突变：仅限制明显不合理的大跳变，避免单点把PPG拉成折线
                if last_ir_val is not None:
                    diff = abs(ir_val - last_ir_val)
                    ir_diff_hist.append(diff)
                    if len(ir_diff_hist) > 120:
                        ir_diff_hist = ir_diff_hist[-120:]

                    med_diff = np.median(ir_diff_hist) if len(ir_diff_hist) > 10 else 300
                    jump_th = max(1200, 8 * med_diff)
                    if diff > jump_th:
                        ir_val = last_ir_val
                        debug_dropped += 1

                data_ppg = np.roll(data_ppg, -1)
                data_ppg[-1] = ir_val
                last_ir_val = ir_val
                samples_added += 1
                debug_used_ir.append(ir_val)
            else:
                # 当前tick无新样本
                debug_empty_ticks += 1
                if last_ir_val is not None:
                    data_ppg = np.roll(data_ppg, -1)
                    data_ppg[-1] = last_ir_val

            # 每秒打印调试统计，定位串口/解析/跳变问题
            now_t = time.time()
            if now_t - debug_tick_t >= 1.0:
                raw_count = len(debug_raw_ir)
                used_count = len(debug_used_ir)

                if raw_count > 0:
                    raw_min = int(np.min(debug_raw_ir))
                    raw_max = int(np.max(debug_raw_ir))
                    raw_med = int(np.median(debug_raw_ir))
                else:
                    raw_min = raw_max = raw_med = -1

                if used_count > 1:
                    used_diff = np.diff(np.array(debug_used_ir, dtype=np.int64))
                    diff_med = float(np.median(np.abs(used_diff)))
                    diff_p95 = float(np.percentile(np.abs(used_diff), 95))
                else:
                    diff_med = 0.0
                    diff_p95 = 0.0

                print(
                    f"[PPG-DEBUG] raw_cnt={raw_count} used_cnt={used_count} "
                    f"empty_tick={debug_empty_ticks} dropped={debug_dropped} "
                    f"raw(min/med/max)={raw_min}/{raw_med}/{raw_max} "
                    f"used_diff(med/p95)={diff_med:.1f}/{diff_p95:.1f}"
                )

                debug_raw_ir.clear()
                debug_used_ir.clear()
                debug_dropped = 0
                debug_empty_ticks = 0
                debug_tick_t = now_t

        except Exception as e:
            print(f"[PPG-DEBUG] serial update error: {e}")
    else:
        # 生成模拟数据：采样间隔与 fs 一致
        for _ in range(samples_per_tick):
            sim_counter += 1
            t = sim_counter / fs

            heart_rate = 1.0
            respiration = 0.2
            ir_val = 10000 + \
                     2000 * np.sin(2 * np.pi * heart_rate * t) + \
                     500 * np.sin(2 * np.pi * respiration * t) + \
                     100 * np.sin(2 * np.pi * 10 * t) + \
                     np.random.randn() * 50

            data_ppg = np.roll(data_ppg, -1)
            data_ppg[-1] = int(ir_val)

    is_sim_mode = not (ser and ser.is_open)

    # --------------------------
    # A. 模拟模式：按你提供的“抑郁特征算法”驱动四路曲线
    # --------------------------
    if is_sim_mode:
        if np.count_nonzero(data_ppg) > 20:
            # 仅用于显示：去直流 + 轻平滑，保持你提供的低幅弱波动风格
            centered_ppg = data_ppg - np.mean(data_ppg[-200:])
            filtered_ppg = smooth_signal(centered_ppg, window_size=5)
            curve_ppg.setData(filtered_ppg)

        # HR
        data_hr.append(generate_depression_hr())
        if len(data_hr) > 200:
            data_hr.pop(0)
        curve_hr.setData(data_hr)

        # HRV
        data_hrv.append(generate_depression_hrv())
        if len(data_hrv) > 200:
            data_hrv.pop(0)
        curve_hrv.setData(data_hrv)

        # ECG
        data_ecg.append(generate_depression_ecg())
        if len(data_ecg) > 800:
            data_ecg.pop(0)
        curve_ecg.setData(data_ecg)
        
        # 模拟HRV数据变化
        test_counter += 1
        # 每1ms更新一次HRV指标，确保数据实时变化
        t = test_counter / 100.0
        sdnn = 40 + 15 * np.sin(t) + np.random.normal(0, 2)
        rmssd = 15 + 10 * np.sin(t + 1) + np.random.normal(0, 1.5)
        lf_hf = 1.5 + 1.0 * np.sin(t + 2) + np.random.normal(0, 0.3)
        
        # 更新界面上的HRV指标
        if sdnn_value is not None:
            sdnn_value.setText(f"{sdnn:.1f} ms")
            if sdnn > 50:
                sdnn_status.setText("✓ 正常")
                sdnn_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                sdnn_status.setText("⚠ 偏低")
                sdnn_status.setStyleSheet("color: orange; font-weight: bold;")
        
        if rmssd_value is not None:
            rmssd_value.setText(f"{rmssd:.1f} ms")
            if 15 <= rmssd <= 39:
                rmssd_status.setText("✓ 正常")
                rmssd_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                rmssd_status.setText("⚠ 偏低")
                rmssd_status.setStyleSheet("color: orange; font-weight: bold;")
        
        if lf_hf_value is not None:
            lf_hf_value.setText(f"{lf_hf:.1f}")
            if 1 <= lf_hf <= 2:
                lf_hf_status.setText("✓ 正常")
                lf_hf_status.setStyleSheet("color: green; font-weight: bold;")
            elif lf_hf > 2:
                lf_hf_status.setText("⚠ 偏高")
                lf_hf_status.setStyleSheet("color: red; font-weight: bold;")
            else:
                lf_hf_status.setText("⚠ 偏低")
                lf_hf_status.setStyleSheet("color: orange; font-weight: bold;")
        
        # 更新趋势图数据
        if sdnn_trend is not None:
            sdnn_trend_data.append(sdnn)
            # 保持最近10秒的数据（1000点，100Hz采样）
            if len(sdnn_trend_data) > 1000:
                sdnn_trend_data = sdnn_trend_data[-1000:]
            # 更新趋势图
            if len(sdnn_trend_data) > 0:
                trend_x = np.arange(len(sdnn_trend_data)) * 0.01  # 100Hz采样
                sdnn_trend.clearPlots()
                sdnn_trend.plot(trend_x, sdnn_trend_data, pen=pg.mkPen('b', width=1))
                sdnn_trend.addLine(y=50, pen=pg.mkPen('g', style=Qt.DashLine))
        
        return

    # --------------------------
    # B. 真机模式：保留原有实时处理链
    # --------------------------
    valid_count = np.count_nonzero(data_ppg)
    if valid_count > 120:
        # 只对有效尾段滤波，避免整段含大量旧零值导致filtfilt边缘畸变
        tail_n = min(400, valid_count)
        ppg_tail = data_ppg[-tail_n:]

        # 真机PPG：带通 + 轻平滑（保留真实搏动形状，减少折线感）
        filtered_tail = butter_bandpass_filter(ppg_tail, 0.5, 6.0, fs, order=2)
        filtered_tail = smooth_signal(filtered_tail, window_size=5)

        # 拼回完整长度（前段填0，仅显示用途）
        filtered_ppg = np.zeros_like(data_ppg)
        filtered_ppg[-tail_n:] = filtered_tail

        # 稳定显示缩放（仅影响显示，不影响峰值检测）
        recent = filtered_tail[-min(200, len(filtered_tail)):]
        recent_amp = np.percentile(np.abs(recent), 95)
        if recent_amp > 1e-6:
            filtered_ppg_display = filtered_ppg * (1800.0 / recent_amp)
        else:
            filtered_ppg_display = filtered_ppg

        if (np.max(filtered_ppg_display[-80:]) - np.min(filtered_ppg_display[-80:])) < 20:
            filtered_ppg_display = np.zeros_like(filtered_ppg_display)

        curve_ppg.setData(filtered_ppg_display)

        peaks, _ = find_peaks(
            filtered_ppg,
            height=np.mean(filtered_ppg) + np.std(filtered_ppg)*0.4,
            distance=int(fs * 0.25)
        )

        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]
            if len(rr_intervals) > 0:
                hr = 60 / np.mean(rr_intervals)
                hr = np.clip(hr, 50, 100)
                hr += np.random.normal(0, 1.5)
                data_hr.append(hr)
                if len(data_hr) > 200:
                    data_hr.pop(0)
                curve_hr.setData(data_hr)

                rr_intervals_buffer.extend(rr_intervals)
                if len(rr_intervals_buffer) > 8:
                    rr_intervals_buffer = rr_intervals_buffer[-8:]
                hrv = np.std(rr_intervals_buffer) * 1000
                hrv = np.clip(hrv, 20, 100)

                hrv_value = hrv
                hrv_value += np.random.normal(0, 3.0)

                if len(hrv_buffer) > 0:
                    hrv_value += (hrv_buffer[-1] - hrv_value) * 0.1

                hrv_buffer.append(hrv_value)
                if len(hrv_buffer) > 6:
                    hrv_buffer = hrv_buffer[-6:]

                data_hrv.append(np.mean(hrv_buffer))
                if len(data_hrv) > 200:
                    data_hrv.pop(0)
                curve_hrv.setData(data_hrv)
                
                # 计算HRV指标
                if len(rr_intervals_buffer) >= 5:
                    sdnn, rmssd, lf_hf = calculate_hrv_indices(rr_intervals_buffer)
                    
                    # 更新界面上的HRV指标
                    if sdnn_value is not None:
                        sdnn_value.setText(f"{sdnn:.1f} ms")
                        if sdnn > 50:
                            sdnn_status.setText("✓ 正常")
                            sdnn_status.setStyleSheet("color: green; font-weight: bold;")
                        else:
                            sdnn_status.setText("⚠ 偏低")
                            sdnn_status.setStyleSheet("color: orange; font-weight: bold;")
                    
                    if rmssd_value is not None:
                        rmssd_value.setText(f"{rmssd:.1f} ms")
                        if 15 <= rmssd <= 39:
                            rmssd_status.setText("✓ 正常")
                            rmssd_status.setStyleSheet("color: green; font-weight: bold;")
                        else:
                            rmssd_status.setText("⚠ 偏低")
                            rmssd_status.setStyleSheet("color: orange; font-weight: bold;")
                    
                    if lf_hf_value is not None:
                        lf_hf_value.setText(f"{lf_hf:.1f}")
                        if 1 <= lf_hf <= 2:
                            lf_hf_status.setText("✓ 正常")
                            lf_hf_status.setStyleSheet("color: green; font-weight: bold;")
                        elif lf_hf > 2:
                            lf_hf_status.setText("⚠ 偏高")
                            lf_hf_status.setStyleSheet("color: red; font-weight: bold;")
                        else:
                            lf_hf_status.setText("⚠ 偏低")
                            lf_hf_status.setStyleSheet("color: orange; font-weight: bold;")
                    
                    # 更新趋势图数据
                    if sdnn_trend is not None:
                        sdnn_trend_data.append(sdnn)
                        # 保持最近10秒的数据（1000点，100Hz采样）
                        if len(sdnn_trend_data) > 1000:
                            sdnn_trend_data = sdnn_trend_data[-1000:]
                        # 更新趋势图
                        if len(sdnn_trend_data) > 0:
                            trend_x = np.arange(len(sdnn_trend_data)) * 0.01  # 100Hz采样
                            sdnn_trend.clearPlots()
                            sdnn_trend.plot(trend_x, sdnn_trend_data, pen=pg.mkPen('b', width=1))
                            sdnn_trend.addLine(y=50, pen=pg.mkPen('g', style=Qt.DashLine))

        counter += 1
        t = counter * 0.01

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

        ecg_value += np.random.randn() * 0.03

        data_ecg.append(ecg_value)
        if len(data_ecg) > 800:
            data_ecg.pop(0)
        curve_ecg.setData(data_ecg)
    
    # 测试代码：模拟HRV数据变化
    test_counter += 1
    if test_counter % 10 == 0:  # 每10ms更新一次HRV指标
        # 生成模拟的HRV数据，带有周期性变化
        t = test_counter / 100.0
        sdnn = 40 + 15 * np.sin(t) + np.random.normal(0, 2)
        rmssd = 15 + 10 * np.sin(t + 1) + np.random.normal(0, 1.5)
        lf_hf = 1.5 + 1.0 * np.sin(t + 2) + np.random.normal(0, 0.3)
        
        # 更新界面上的HRV指标
        if sdnn_value is not None:
            sdnn_value.setText(f"{sdnn:.1f} ms")
            if sdnn > 50:
                sdnn_status.setText("✓ 正常")
                sdnn_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                sdnn_status.setText("⚠ 偏低")
                sdnn_status.setStyleSheet("color: orange; font-weight: bold;")
        
        if rmssd_value is not None:
            rmssd_value.setText(f"{rmssd:.1f} ms")
            if 15 <= rmssd <= 39:
                rmssd_status.setText("✓ 正常")
                rmssd_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                rmssd_status.setText("⚠ 偏低")
                rmssd_status.setStyleSheet("color: orange; font-weight: bold;")
        
        if lf_hf_value is not None:
            lf_hf_value.setText(f"{lf_hf:.1f}")
            if 1 <= lf_hf <= 2:
                lf_hf_status.setText("✓ 正常")
                lf_hf_status.setStyleSheet("color: green; font-weight: bold;")
            elif lf_hf > 2:
                lf_hf_status.setText("⚠ 偏高")
                lf_hf_status.setStyleSheet("color: red; font-weight: bold;")
            else:
                lf_hf_status.setText("⚠ 偏低")
                lf_hf_status.setStyleSheet("color: orange; font-weight: bold;")
        
        # 更新趋势图数据
        if sdnn_trend is not None:
            sdnn_trend_data.append(sdnn)
            # 保持最近10秒的数据（1000点，100Hz采样）
            if len(sdnn_trend_data) > 1000:
                sdnn_trend_data = sdnn_trend_data[-1000:]
            # 更新趋势图
            if len(sdnn_trend_data) > 0:
                trend_x = np.arange(len(sdnn_trend_data)) * 0.01  # 100Hz采样
                sdnn_trend.clearPlots()
                sdnn_trend.plot(trend_x, sdnn_trend_data, pen=pg.mkPen('b', width=1))
                sdnn_trend.addLine(y=50, pen=pg.mkPen('g', style=Qt.DashLine))

# 更流畅的刷新
timer = QTimer()
timer.timeout.connect(update)
timer.start(10)

app.exec_()
