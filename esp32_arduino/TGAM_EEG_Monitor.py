6) 总结判定：当前阶段
判定：“可推理 + 可基础训练（原型级）”，整体属于 “仅原型/PoC 接近可用”
依据
有完整模型定义、训练入口、ONNX导出、推理入口、数据对齐器 —— 说明“跑得通”
但关键工程能力不足：
可训练可信度不足（随机标签、回归头未训练、无checkpoint）
可推理真实性有退化路径掩盖风险（fallback）
缺少规范化验证和监控
所以不是“不可用”，但也不能认定为“生产可训练可推理系统”。

7) 改进清单（按优先级）
P0（必须修）
禁止随机标签训练
load_windows 在 labels.csv 缺失时应直接报错并停止训练
补齐多任务损失
train_dl 增加 reg_loss（如 MSE/Huber），联合优化 cls + λ*reg
推理模式强约束
predict 输出中明确告警；可配置“禁止fallback”模式，避免线上误用
P1（建议尽快）
加入 checkpoint + early stopping + best model 选择
加入训练日志（loss/acc/f1/auc）和验证曲线
增加标准化流程
训练保存 scaler，推理复用同一 scaler
补测试集与时序验证策略
至少 train/val/test 三段，避免时间泄漏
P2（优化项）
模型正则化
Conv/GRU后加 dropout 或 layernorm
解释性口径统一
区分“时间注意力 attn”与“模态贡献 modal_w”的来源
推理性能优化
provider可配置（CPU/NPU），并记录加载失败原因"""
TGAM脑波接收与显示程序
支持蓝牙接收器和蓝牙模块（串口）两种连接方式
包含数据校验，防止传输损坏
"""

import serial
import serial.tools.list_ports
import struct
import time
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QTextEdit, QGroupBox, QGridLayout, QCheckBox)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg


# TGAM协议常量
SYNC = 0xAA
EXCODE = 0x55

# 数据码定义
CODE_POOR_SIGNAL = 0x02
CODE_HEART_RATE = 0x03
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_8BIT_RAW = 0x06
CODE_RAW_MARKER = 0x07
CODE_RAW_WAVE = 0x80
CODE_EEG_POWER = 0x81
CODE_ASIC_EEG_POWER = 0x83
CODE_RRINTERVAL = 0x86


class TGAMParser:
    """TGAM数据解析器"""

    def __init__(self):
        self.buffer = bytearray()
        self.raw_wave_history = deque(maxlen=512)
        self.attention = 0
        self.meditation = 0
        self.poor_signal = 0
        self.heart_rate = 0
        self.rr_interval = 0
        self.eeg_power = {}
        self.last_update = time.time()
        self.packet_count = 0
        self.error_count = 0

    def parse_byte(self, byte):
        """解析单个字节"""
        self.buffer.append(byte)

        # 查找同步头 0xAA 0xAA
        if len(self.buffer) >= 2:
            if self.buffer[0] == SYNC and self.buffer[1] == SYNC:
                # 找到同步头，读取数据长度
                if len(self.buffer) >= 3:
                    payload_length = self.buffer[2]
                    # 检查是否收到完整数据包
                    if len(self.buffer) >= 3 + payload_length + 2:
                        # 提取payload和校验和
                        payload = self.buffer[3:3 + payload_length]
                        received_checksum = self.buffer[3 + payload_length]
                        calculated_checksum = self.calculate_checksum(payload)

                        # 校验和验证
                        if received_checksum == calculated_checksum:
                            self.parse_payload(payload)
                            self.last_update = time.time()
                            self.packet_count += 1
                        else:
                            self.error_count += 1

                        # 移除已处理的数据
                        self.buffer = self.buffer[3 + payload_length + 2:]

        # 防止缓冲区溢出
        if len(self.buffer) > 512:
            self.buffer = bytearray()

    def calculate_checksum(self, payload):
        """计算校验和（对payload所有字节求和后取低8位，再取反）"""
        return (~sum(payload) & 0xFF)

    def parse_payload(self, payload):
        """解析数据载荷"""
        index = 0
        while index < len(payload):
            # 处理EXCODE扩展码
            excode_count = 0
            while index < len(payload) and payload[index] == EXCODE:
                excode_count += 1
                index += 1

            if index >= len(payload):
                break

            code = payload[index]
            index += 1

            # 获取数据长度
            if code >= 0x80:
                if index >= len(payload):
                    break
                value_length = payload[index]
                index += 1
            else:
                value_length = 1

            # 提取数据
            if index + value_length > len(payload):
                break

            value = payload[index:index + value_length]
            index += value_length

            # 解析数据
            self.parse_data(code, value)

    def parse_data(self, code, value):
        """解析单个数据字段"""
        try:
            if code == CODE_POOR_SIGNAL:
                self.poor_signal = value[0]

            elif code == CODE_HEART_RATE:
                self.heart_rate = value[0]

            elif code == CODE_ATTENTION:
                self.attention = value[0]

            elif code == CODE_MEDITATION:
                self.meditation = value[0]

            elif code == CODE_RAW_WAVE:
                if len(value) == 2:
                    raw_value = struct.unpack('>h', bytes(value))[0]
                    self.raw_wave_history.append(raw_value)

            elif code == CODE_EEG_POWER:
                if len(value) == 24:
                    self.eeg_power = {
                        'delta': struct.unpack('>I', b'\x00' + value[0:3])[0],
                        'theta': struct.unpack('>I', b'\x00' + value[3:6])[0],
                        'low_alpha': struct.unpack('>I', b'\x00' + value[6:9])[0],
                        'high_alpha': struct.unpack('>I', b'\x00' + value[9:12])[0],
                        'low_beta': struct.unpack('>I', b'\x00' + value[12:15])[0],
                        'high_beta': struct.unpack('>I', b'\x00' + value[15:18])[0],
                        'low_gamma': struct.unpack('>I', b'\x00' + value[18:21])[0],
                        'mid_gamma': struct.unpack('>I', b'\x00' + value[21:24])[0],
                    }

            elif code == CODE_RRINTERVAL:
                if len(value) == 2:
                    self.rr_interval = struct.unpack('>H', bytes(value))[0]

        except Exception as e:
            print(f"解析数据错误: {e}")

    def is_timeout(self, timeout=2.0):
        """检查是否超时（无新数据）"""
        return (time.time() - self.last_update) > timeout


class EEGMonitorWindow(QMainWindow):
    """脑波监测主窗口"""

    def __init__(self):
        super().__init__()
        self.parser = TGAMParser()
        self.serial_port = None
        self.is_connected = False
        self.data_history = deque(maxlen=1000)
        self.eeg_wave_data = deque(maxlen=512)
        self.raw_data_buffer = ''
        self.show_raw_data = False

        # 定时器用于更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 50ms更新一次

        # 串口读取定时器（更高优先级）
        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self.read_serial_data)
        self.read_timer.start(10)  # 10ms读取一次

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("TGAM脑波监测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 连接控制区
        control_group = QGroupBox("连接控制")
        control_layout = QHBoxLayout()

        # 连接类型选择
        self.conn_type_combo = QComboBox()
        self.conn_type_combo.addItems(["串口 (蓝牙模块)", "模拟模式"])
        control_layout.addWidget(QLabel("连接方式:"))
        control_layout.addWidget(self.conn_type_combo)

        # 串口选择
        self.port_combo = QComboBox()
        
        # 波特率选择
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["57600", "115200", "9600", "38400", "19200", "28800"])
        self.baud_combo.setCurrentText("57600")  # TGAM标准波特率
        
        # 日志组件
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        
        self.refresh_ports()
        control_layout.addWidget(QLabel("串口:"))
        control_layout.addWidget(self.port_combo)
        control_layout.addWidget(QLabel("波特率:"))
        control_layout.addWidget(self.baud_combo)

        # 刷新串口按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.refresh_btn)

        # 连接/断开按钮
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_btn)

        # 显示原始数据选项
        self.raw_data_checkbox = QCheckBox("显示原始数据")
        self.raw_data_checkbox.stateChanged.connect(self.toggle_raw_data)
        control_layout.addWidget(self.raw_data_checkbox)

        control_layout.addStretch()
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 状态显示区
        status_group = QGroupBox("实时数据")
        status_layout = QGridLayout()

        self.attention_label = QLabel("专注度: --")
        self.meditation_label = QLabel("放松度: --")
        self.heart_rate_label = QLabel("心率: --")
        self.poor_signal_label = QLabel("信号质量: --")
        self.rr_interval_label = QLabel("RR间期: --")
        self.packet_label = QLabel("数据包: 0")
        self.error_label = QLabel("错误: 0")

        labels = [self.attention_label, self.meditation_label,
                  self.heart_rate_label, self.poor_signal_label,
                  self.rr_interval_label, self.packet_label, self.error_label]
        for i, label in enumerate(labels):
            status_layout.addWidget(label, 0, i)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # 脑波波形显示区
        wave_group = QGroupBox("脑电波形")
        wave_layout = QVBoxLayout()

        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground('#1a1a2e')
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wave_plot.setLabel('left', '振幅')
        self.wave_plot.setLabel('bottom', '样本')
        self.wave_plot.setYRange(-32768, 32767)
        self.curve_eeg = self.wave_plot.plot(pen=pg.mkPen('#00ff00', width=1.5))

        wave_layout.addWidget(self.wave_plot)
        wave_group.setLayout(wave_layout)
        main_layout.addWidget(wave_group)

        # EEG频段能量显示
        power_group = QGroupBox("频段能量")
        power_layout = QGridLayout()

        self.delta_label = QLabel("Delta: --")
        self.theta_label = QLabel("Theta: --")
        self.alpha_label = QLabel("Alpha: --")
        self.beta_label = QLabel("Beta: --")
        self.gamma_label = QLabel("Gamma: --")

        power_layout.addWidget(self.delta_label, 0, 0)
        power_layout.addWidget(self.theta_label, 0, 1)
        power_layout.addWidget(self.alpha_label, 0, 2)
        power_layout.addWidget(self.beta_label, 0, 3)
        power_layout.addWidget(self.gamma_label, 0, 4)

        power_group.setLayout(power_layout)
        main_layout.addWidget(power_group)

        # 原始数据显示区
        self.raw_text = QTextEdit()
        self.raw_text.setMaximumHeight(100)
        self.raw_text.setReadOnly(True)
        self.raw_text.setVisible(False)
        
        # 日志区
        main_layout.addWidget(QLabel("数据日志:"))
        main_layout.addWidget(self.log_text)
        main_layout.addWidget(QLabel("原始数据:"))
        main_layout.addWidget(self.raw_text)

    def toggle_raw_data(self, state):
        """切换原始数据显示"""
        self.show_raw_data = (state == Qt.Checked)
        self.raw_text.setVisible(self.show_raw_data)

    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")

        if not ports:
            self.log("未发现可用串口")

    def toggle_connection(self):
        """切换连接状态"""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """建立连接"""
        conn_type = self.conn_type_combo.currentText()

        if conn_type == "模拟模式":
            self.log("进入模拟模式")
            self.is_connected = True
            self.connect_btn.setText("断开")
            return

        # 串口连接
        port_info = self.port_combo.currentText()
        if not port_info:
            self.log("请选择串口")
            return

        port_name = port_info.split(" - ")[0]
        baud_rate = int(self.baud_combo.currentText())

        try:
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=baud_rate,
                timeout=0.01,
                bytesize=8,
                parity='N',
                stopbits=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            self.is_connected = True
            self.connect_btn.setText("断开")
            self.log(f"已连接到 {port_name}, 波特率: {baud_rate}")
            self.parser = TGAMParser()  # 重置解析器
        except Exception as e:
            self.log(f"连接失败: {e}")

    def disconnect(self):
        """断开连接"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        self.is_connected = False
        self.connect_btn.setText("连接")
        self.log("已断开连接")

    def read_serial_data(self):
        """读取串口数据（独立定时器，更高频率）"""
        if not self.is_connected:
            return

        conn_type = self.conn_type_combo.currentText()

        if conn_type == "模拟模式":
            self.generate_simulated_data()
            return

        # 串口读取
        if self.serial_port and self.serial_port.is_open:
            try:
                # 读取所有可用数据
                data = self.serial_port.read(self.serial_port.in_waiting or 1)
                if data:
                    # 显示原始数据
                    if self.show_raw_data:
                        hex_str = ' '.join(f'{b:02X}' for b in data)
                        self.raw_data_buffer += hex_str + ' '
                        if len(self.raw_data_buffer) > 500:
                            self.raw_data_buffer = self.raw_data_buffer[-500:]
                        self.raw_text.setText(self.raw_data_buffer)
                    
                    # 解析数据
                    for byte in data:
                        self.parser.parse_byte(byte)
                        
            except Exception as e:
                self.log(f"读取错误: {e}")

    def generate_simulated_data(self):
        """生成模拟数据（用于测试）"""
        t = time.time()

        # 模拟脑波信号
        wave = 15000 * (
            0.5 * (t % 1) +
            0.3 * (t % 0.3) +
            0.2 * (t % 0.1)
        )
        self.eeg_wave_data.append(int(wave))

        # 模拟专注度和放松度
        self.parser.attention = int(50 + 30 * abs((t % 2) - 1))
        self.parser.meditation = int(50 + 30 * abs((t % 3) - 1.5))
        self.parser.heart_rate = int(70 + 10 * abs((t % 1) - 0.5))
        self.parser.poor_signal = 0
        self.parser.rr_interval = int(850 + 100 * abs((t % 0.8) - 0.4))

        # 模拟频段能量
        self.parser.eeg_power = {
            'delta': int(100000 + 50000 * abs((t % 5))),
            'theta': int(80000 + 40000 * abs((t % 4))),
            'low_alpha': int(60000 + 30000 * abs((t % 3))),
            'high_alpha': int(50000 + 25000 * abs((t % 2.5))),
            'low_beta': int(40000 + 20000 * abs((t % 2))),
            'high_beta': int(30000 + 15000 * abs((t % 1.5))),
            'low_gamma': int(20000 + 10000 * abs((t % 1))),
            'mid_gamma': int(15000 + 7500 * abs((t % 0.8))),
        }

    def update_display(self):
        """更新显示"""
        # 更新波形
        if len(self.parser.raw_wave_history) > 0:
            self.curve_eeg.setData(list(self.parser.raw_wave_history))

        # 更新标签
        self.attention_label.setText(f"专注度: {self.parser.attention}")
        self.meditation_label.setText(f"放松度: {self.parser.meditation}")
        self.heart_rate_label.setText(f"心率: {self.parser.heart_rate}")
        self.poor_signal_label.setText(f"信号质量: {self.parser.poor_signal}")
        self.rr_interval_label.setText(f"RR间期: {self.parser.rr_interval}")
        self.packet_label.setText(f"数据包: {self.parser.packet_count}")
        self.error_label.setText(f"错误: {self.parser.error_count}")

        # 更新频段能量
        if self.parser.eeg_power:
            p = self.parser.eeg_power
            alpha = p.get('low_alpha', 0) + p.get('high_alpha', 0)
            beta = p.get('low_beta', 0) + p.get('high_beta', 0)
            gamma = p.get('low_gamma', 0) + p.get('mid_gamma', 0)

            self.delta_label.setText(f"Delta: {p.get('delta', 0):,}")
            self.theta_label.setText(f"Theta: {p.get('theta', 0):,}")
            self.alpha_label.setText(f"Alpha: {alpha:,}")
            self.beta_label.setText(f"Beta: {beta:,}")
            self.gamma_label.setText(f"Gamma: {gamma:,}")

    def log(self, message):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """关闭事件"""
        self.disconnect()
        event.accept()


def main():
    app = QApplication([])
    window = EEGMonitorWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()