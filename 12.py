"""
TGAM EEG monitor window.

Runs on computer B and can read:
1. Local bridge HTTP API (default)
2. Local serial TGAM device
3. Mock mode

This window only shows EEG-related data:
- attention
- meditation
- signal quality
- raw wave
- EEG band power

Heart-rate-related data belongs to the separate physio window.
"""

import json
import struct
import time
from collections import deque

import pyqtgraph as pg
import requests
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


SYNC = 0xAA
EXCODE = 0x55

CODE_POOR_SIGNAL = 0x02
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_RAW_WAVE = 0x80
CODE_EEG_POWER = 0x83

DEFAULT_BRIDGE_URL = "http://127.0.0.1:5001/eeg/latest"
DEFAULT_WAVE_Y_MIN = -1500
DEFAULT_WAVE_Y_MAX = 1500


class TGAMParser:
    def __init__(self):
        self.buffer = bytearray()
        self.raw_wave_history = deque(maxlen=160)
        self.attention = 0
        self.meditation = 0
        self.poor_signal = 200
        self.eeg_power = {}
        self.last_update = time.time()
        self.packet_count = 0
        self.error_count = 0

    def parse_byte(self, byte_value: int):
        self.buffer.append(byte_value)

        while True:
            if len(self.buffer) < 2:
                break

            if not (self.buffer[0] == SYNC and self.buffer[1] == SYNC):
                del self.buffer[0]
                continue

            if len(self.buffer) < 3:
                break

            payload_length = self.buffer[2]
            if payload_length > 169:
                del self.buffer[0]
                self.error_count += 1
                continue

            packet_total_len = 3 + payload_length + 1
            if len(self.buffer) < packet_total_len:
                break

            payload = self.buffer[3:3 + payload_length]
            received_checksum = self.buffer[3 + payload_length]
            calculated_checksum = self.calculate_checksum(payload)

            if received_checksum == calculated_checksum:
                self.parse_payload(payload)
                self.last_update = time.time()
                self.packet_count += 1
            else:
                self.error_count += 1

            del self.buffer[:packet_total_len]

        if len(self.buffer) > 2048:
            self.buffer = bytearray()

    @staticmethod
    def calculate_checksum(payload):
        return (~sum(payload)) & 0xFF

    def parse_payload(self, payload):
        index = 0
        while index < len(payload):
            while index < len(payload) and payload[index] == EXCODE:
                index += 1

            if index >= len(payload):
                break

            code = payload[index]
            index += 1

            if code >= 0x80:
                if index >= len(payload):
                    break
                value_length = payload[index]
                index += 1
            else:
                value_length = 1

            if index + value_length > len(payload):
                break

            value = payload[index:index + value_length]
            index += value_length
            self.parse_data(code, value)

    def parse_data(self, code, value):
        try:
            if code == CODE_POOR_SIGNAL and value:
                self.poor_signal = int(value[0])
            elif code == CODE_ATTENTION and value:
                self.attention = int(value[0])
            elif code == CODE_MEDITATION and value:
                self.meditation = int(value[0])
            elif code == CODE_RAW_WAVE and len(value) == 2:
                raw_value = struct.unpack(">h", bytes(value))[0]
                self.raw_wave_history.append(raw_value)
            elif code == CODE_EEG_POWER and len(value) == 24:
                self.eeg_power = {
                    "delta": struct.unpack(">I", b"\x00" + value[0:3])[0],
                    "theta": struct.unpack(">I", b"\x00" + value[3:6])[0],
                    "low_alpha": struct.unpack(">I", b"\x00" + value[6:9])[0],
                    "high_alpha": struct.unpack(">I", b"\x00" + value[9:12])[0],
                    "low_beta": struct.unpack(">I", b"\x00" + value[12:15])[0],
                    "high_beta": struct.unpack(">I", b"\x00" + value[15:18])[0],
                    "low_gamma": struct.unpack(">I", b"\x00" + value[18:21])[0],
                    "mid_gamma": struct.unpack(">I", b"\x00" + value[21:24])[0],
                }
        except Exception:
            self.error_count += 1


class EEGMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.parser = TGAMParser()
        self.serial_port = None
        self.is_connected = False
        self.show_raw_data = False
        self.raw_data_buffer = ""

        self.bridge_url = DEFAULT_BRIDGE_URL
        self.http_session = requests.Session()
        self.last_http_fetch = 0.0
        self.http_fetch_interval = 0.02

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)

        self.read_timer = QTimer()
        self.read_timer.timeout.connect(self.read_data)
        self.read_timer.start(20)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TGAM EEG Monitor")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_group = QGroupBox("Connection")
        control_layout = QHBoxLayout()

        self.conn_type_combo = QComboBox()
        self.conn_type_combo.addItems(["Bridge HTTP", "Serial", "Mock"])
        control_layout.addWidget(QLabel("Mode:"))
        control_layout.addWidget(self.conn_type_combo)

        self.port_combo = QComboBox()
        self.refresh_ports()
        control_layout.addWidget(QLabel("Port:"))
        control_layout.addWidget(self.port_combo)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["57600", "115200", "9600", "38400", "19200", "28800"])
        self.baud_combo.setCurrentText("57600")
        control_layout.addWidget(QLabel("Baud:"))
        control_layout.addWidget(self.baud_combo)

        self.bridge_url_edit = QLineEdit(self.bridge_url)
        self.bridge_url_edit.setMinimumWidth(260)
        control_layout.addWidget(QLabel("URL:"))
        control_layout.addWidget(self.bridge_url_edit)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        control_layout.addWidget(self.refresh_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_btn)

        self.raw_data_checkbox = QCheckBox("Show raw/debug data")
        self.raw_data_checkbox.stateChanged.connect(self.toggle_raw_data)
        control_layout.addWidget(self.raw_data_checkbox)

        control_layout.addStretch()
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        status_group = QGroupBox("EEG status")
        status_layout = QGridLayout()

        self.attention_label = QLabel("Attention: --")
        self.meditation_label = QLabel("Meditation: --")
        self.poor_signal_label = QLabel("Signal quality: --")
        self.packet_label = QLabel("Packets: 0")
        self.error_label = QLabel("Errors: 0")

        labels = [
            self.attention_label,
            self.meditation_label,
            self.poor_signal_label,
            self.packet_label,
            self.error_label,
        ]
        for i, label in enumerate(labels):
            status_layout.addWidget(label, 0, i)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        wave_group = QGroupBox("Raw EEG wave")
        wave_layout = QVBoxLayout()
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground("#1a1a2e")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Sample")
        self.wave_plot.setYRange(DEFAULT_WAVE_Y_MIN, DEFAULT_WAVE_Y_MAX)
        self.curve_eeg = self.wave_plot.plot(pen=pg.mkPen("#00ff00", width=1.5))
        wave_layout.addWidget(self.wave_plot)
        wave_group.setLayout(wave_layout)
        main_layout.addWidget(wave_group)

        power_group = QGroupBox("EEG band power")
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

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

        self.raw_text = QTextEdit()
        self.raw_text.setMaximumHeight(140)
        self.raw_text.setReadOnly(True)
        self.raw_text.setVisible(False)
        main_layout.addWidget(QLabel("Raw/debug:"))
        main_layout.addWidget(self.raw_text)

    def toggle_raw_data(self, state):
        self.show_raw_data = state == Qt.Checked
        self.raw_text.setVisible(self.show_raw_data)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
        if not ports:
            self.log("No serial ports found.")

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        mode = self.conn_type_combo.currentText()

        self.parser = TGAMParser()
        self.raw_data_buffer = ""
        self.raw_text.clear()

        if mode == "Mock":
            self.is_connected = True
            self.connect_btn.setText("Disconnect")
            self.log("Entered mock mode.")
            return

        if mode == "Bridge HTTP":
            self.bridge_url = self.bridge_url_edit.text().strip() or DEFAULT_BRIDGE_URL
            self.is_connected = True
            self.connect_btn.setText("Disconnect")
            self.log(f"Connected to local bridge: {self.bridge_url}")
            return

        port_info = self.port_combo.currentText()
        if not port_info:
            self.log("Please select a serial port.")
            return

        port_name = port_info.split(" - ", 1)[0].strip()
        baud_rate = int(self.baud_combo.currentText())

        try:
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=baud_rate,
                timeout=0,
                bytesize=8,
                parity="N",
                stopbits=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                exclusive=False,
            )
            try:
                self.serial_port.setDTR(False)
                self.serial_port.setRTS(False)
            except Exception:
                pass

            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()

            self.is_connected = True
            self.connect_btn.setText("Disconnect")
            self.log(f"Connected to serial {port_name} @ {baud_rate}")
        except Exception as exc:
            self.log(f"Serial connect failed: {exc}")

    def disconnect(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.is_connected = False
        self.connect_btn.setText("Connect")
        self.log("Disconnected.")

    def read_data(self):
        if not self.is_connected:
            return

        mode = self.conn_type_combo.currentText()
        if mode == "Mock":
            self.generate_simulated_data()
            return
        if mode == "Bridge HTTP":
            self.read_bridge_data()
            return
        self.read_serial_data()

    def read_bridge_data(self):
        now = time.time()
        if now - self.last_http_fetch < self.http_fetch_interval:
            return
        self.last_http_fetch = now

        try:
            response = self.http_session.get(self.bridge_url, timeout=1.5)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            self.apply_bridge_payload(payload)
        except Exception as exc:
            self.log(f"Bridge read failed: {exc}")

    def apply_bridge_payload(self, payload):
        if not isinstance(payload, dict):
            return

        attention = payload.get("attention")
        meditation = payload.get("meditation")
        poor_signal = payload.get("poor_signal")
        raw_value = payload.get("raw_value")
        eeg_power = payload.get("eeg_power")

        if attention is not None:
            self.parser.attention = int(attention)
        if meditation is not None:
            self.parser.meditation = int(meditation)
        if poor_signal is not None:
            self.parser.poor_signal = int(poor_signal)
        if raw_value is not None:
            self.parser.raw_wave_history.append(int(raw_value))
        if isinstance(eeg_power, list) and len(eeg_power) >= 8:
            self.parser.eeg_power = {
                "delta": int(eeg_power[0]),
                "theta": int(eeg_power[1]),
                "low_alpha": int(eeg_power[2]),
                "high_alpha": int(eeg_power[3]),
                "low_beta": int(eeg_power[4]),
                "high_beta": int(eeg_power[5]),
                "low_gamma": int(eeg_power[6]),
                "mid_gamma": int(eeg_power[7]),
            }

        self.parser.packet_count += 1
        self.parser.last_update = time.time()

        if self.show_raw_data:
            self.raw_text.setText(json.dumps(payload, ensure_ascii=False, indent=2))

    def read_serial_data(self):
        if self.serial_port and self.serial_port.is_open:
            try:
                waiting = self.serial_port.in_waiting
                if waiting <= 0:
                    return

                data = self.serial_port.read(waiting)
                if data:
                    if self.show_raw_data:
                        hex_str = " ".join(f"{b:02X}" for b in data)
                        self.raw_data_buffer += hex_str + " "
                        if len(self.raw_data_buffer) > 1200:
                            self.raw_data_buffer = self.raw_data_buffer[-1200:]
                        self.raw_text.setText(self.raw_data_buffer)

                    for byte_value in data:
                        self.parser.parse_byte(byte_value)

            except Exception as exc:
                self.log(f"Serial read error: {exc}")

    def generate_simulated_data(self):
        t = time.time()
        self.parser.raw_wave_history.append(int(300 * (0.5 - (t % 1))))
        self.parser.attention = int(50 + 30 * abs((t % 2) - 1))
        self.parser.meditation = int(50 + 30 * abs((t % 3) - 1.5))
        self.parser.poor_signal = 0
        self.parser.eeg_power = {
            "delta": int(100000 + 50000 * abs((t % 5))),
            "theta": int(80000 + 40000 * abs((t % 4))),
            "low_alpha": int(60000 + 30000 * abs((t % 3))),
            "high_alpha": int(50000 + 25000 * abs((t % 2.5))),
            "low_beta": int(40000 + 20000 * abs((t % 2))),
            "high_beta": int(30000 + 15000 * abs((t % 1.5))),
            "low_gamma": int(20000 + 10000 * abs((t % 1))),
            "mid_gamma": int(15000 + 7500 * abs((t % 0.8))),
        }
        self.parser.packet_count += 1

    def update_display(self):
        if len(self.parser.raw_wave_history) > 0:
            wave = list(self.parser.raw_wave_history)
            self.curve_eeg.setData(wave)
            self._update_wave_range(wave)

        self.attention_label.setText(f"Attention: {self.parser.attention}")
        self.meditation_label.setText(f"Meditation: {self.parser.meditation}")
        self.poor_signal_label.setText(f"Signal quality: {self.parser.poor_signal}")
        self.packet_label.setText(f"Packets: {self.parser.packet_count}")
        self.error_label.setText(f"Errors: {self.parser.error_count}")

        if self.parser.eeg_power:
            p = self.parser.eeg_power
            alpha = p.get("low_alpha", 0) + p.get("high_alpha", 0)
            beta = p.get("low_beta", 0) + p.get("high_beta", 0)
            gamma = p.get("low_gamma", 0) + p.get("mid_gamma", 0)
            self.delta_label.setText(f"Delta: {p.get('delta', 0):,}")
            self.theta_label.setText(f"Theta: {p.get('theta', 0):,}")
            self.alpha_label.setText(f"Alpha: {alpha:,}")
            self.beta_label.setText(f"Beta: {beta:,}")
            self.gamma_label.setText(f"Gamma: {gamma:,}")

    def _update_wave_range(self, wave):
        if not wave:
            return

        w_min = min(wave)
        w_max = max(wave)
        span = w_max - w_min

        min_span = 200
        if span < min_span:
            center = (w_max + w_min) / 2.0
            half = min_span / 2.0
            y_min = center - half
            y_max = center + half
        else:
            pad = max(50, span * 0.15)
            y_min = w_min - pad
            y_max = w_max + pad

        y_min = min(y_min, DEFAULT_WAVE_Y_MIN)
        y_max = max(y_max, DEFAULT_WAVE_Y_MAX)
        self.wave_plot.setYRange(y_min, y_max, padding=0)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        self.disconnect()
        event.accept()


def main():
    app = QApplication([])
    window = EEGMonitorWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
