"""
学生端主界面。

负责组织实时监测、风险展示、脑电/生理波形和 AI 干预建议等交互逻辑。
"""

import cv2
import requests
import json
import time
import math
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import config
from physio_thread import PhysioThread
from cyber_mock import MockWaveformThread, SocialBatteryWidget, PredictionCurveWidget
from bci_thread import BCIThread
from bci_panel import BCIPanel
import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import ppg_signal
from collections import deque
import theme as _theme
from theme import (C_BG, C_PANEL, C_PANEL_HI, C_BORDER, C_ACCENT, C_ACCENT2,
                   C_OK, C_WARN, C_RISK, C_DIM, C_TEXT, C_PPG, C_ECG, C_HR, C_HRV)
from ui_widgets import StatusPill, BigValueCard, MetricRow, RiskBarLabel
from risk_bus import RiskBus

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 

if not hasattr(config, 'WINDOW_WIDTH') or config.WINDOW_WIDTH == 0:
    config.WINDOW_WIDTH = 1680
    config.WINDOW_HEIGHT = 980

# ==========================================
# 🛑 在这里填入你的 智谱 API KEY
# ==========================================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()

# 固定 JSON 字段说明（写入 Stage A/B prompt，减少模型编造）
LLM_CONTEXT_SCHEMA_HINT = """
schema_version: mindroom_llm_ctx_v1
risk: current(0-1), level(low|medium|high), trend_10m(up|down|flat), trend_24h(object|insufficient_data)
modalities.visual/hrv/eeg: score(0-1), weight(和≈1), features(仅使用其中已有字段)
anomalies: 最近告警列表; recent_state_memory: 10分钟摘要
data_source_mode: live|demo|offline_replay; output_target: student|teacher|parent|student_teacher
禁止引用 JSON 中未出现的数值或事件。
"""


class _NoOp:
    """单页大屏改造后，旧多页控件用 NoOp 占位以吸收所有调用。"""
    def __getattr__(self, name):
        return _NoOp()
    def __call__(self, *a, **kw):
        return self
    def __bool__(self):
        return False
    def setText(self, *a, **kw): pass
    def setStyleSheet(self, *a, **kw): pass
    def setChecked(self, *a, **kw): pass
    def isChecked(self): return False
    def setEnabled(self, *a, **kw): pass
    def setVisible(self, *a, **kw): pass
    def show(self): pass
    def hide(self): pass
    def setValue(self, *a, **kw): pass
    def setRange(self, *a, **kw): pass
    def append(self, *a, **kw): pass
    def clear(self): pass
    def add_data(self, *a, **kw): pass
    def text(self): return ''
    def time(self):
        from PyQt5.QtCore import QTime
        return QTime(0, 0)

class AIInterventionThread(QThread):
    """异步请求大模型生成干预建议，避免阻塞主界面。"""
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, current_risk, context_payload=None, mode='realtime'):
        super().__init__()
        self.current_risk = float(current_risk)
        self.context_payload = context_payload or {}
        self.mode = str(mode or 'realtime').lower()

    def _pick_model_and_params(self):
        if self.mode in ('deep', 'analysis', 'deep_analysis'):
            ft = str(getattr(config, 'ZHIPU_FT_MODEL_DEEP', '') or '').strip()
            model = ft if ft else 'glm-4'
            return {
                "model": model,
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 1024,
            }
        ft = str(getattr(config, 'ZHIPU_FT_MODEL_REALTIME', '') or '').strip()
        model = ft if ft else 'glm-4-flash'
        return {
            "model": model,
            "temperature": 0.45,
            "top_p": 0.9,
            "max_tokens": 768,
        }

    @staticmethod
    def _quality_ok(text):
        txt = str(text or '').strip()
        must_keys = ["立即动作", "短期动作", "升级条件"]
        return len(txt) >= 40 and all(k in txt for k in must_keys)

    def _rewrite_via_api(self, draft_text):
        if not getattr(config, 'ZHIPU_LLM_REWRITE_ON_LOW_QUALITY', True):
            return None
        try:
            cfg = self._pick_model_and_params()
            rewrite_model = 'glm-4-flash'
            ft = str(getattr(config, 'ZHIPU_FT_MODEL_REALTIME', '') or '').strip()
            if ft and self.mode not in ('deep', 'analysis', 'deep_analysis'):
                rewrite_model = ft
            prompt = (
                "你是校园心理守护助手。将下列草稿整理为固定结构，不要新增未提及事实。\n"
                "必须逐项输出：\n"
                "1) 给学生：一句40字内温和建议\n"
                "2) 给老师：一句20字内观察提示\n"
                "3) 立即动作（今天）\n"
                "4) 短期动作（3天）\n"
                "5) 升级条件（何时通知家长/老师）\n"
                "禁止临床诊断与药物建议。\n"
                f"草稿:\n{draft_text}"
            )
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": rewrite_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.35,
                "top_p": 0.85,
                "max_tokens": 512,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code != 200:
                return None
            root = resp.json()
            return (((root.get('choices') or [{}])[0].get('message') or {}).get('content') or '').strip()
        except Exception:
            return None

    def _persist_llm_artifacts(self, final_text, attribution=''):
        try:
            root = Path(__file__).resolve().parent
            data_dir = root / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            if getattr(config, 'ZHIPU_LLM_SAVE_LAST_CONTEXT', True):
                snap = {
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'mode': self.mode,
                    'context': self.context_payload,
                    'stage_a_attribution': attribution,
                    'response': final_text,
                }
                with open(data_dir / 'llm_last_context.json', 'w', encoding='utf-8') as f:
                    json.dump(snap, f, ensure_ascii=False, indent=2)
            if getattr(config, 'ZHIPU_LLM_ONLINE_SAMPLES', True):
                user_msg = json.dumps(self.context_payload, ensure_ascii=False)
                row = {
                    "messages": [
                        {"role": "system", "content": "你是校园心理守护助手。基于结构化 JSON 输出五段式建议。"},
                        {"role": "user", "content": f"输入JSON:\n{user_msg}"},
                        {"role": "assistant", "content": final_text},
                    ],
                    "meta": {
                        "ts": time.time(),
                        "mode": self.mode,
                        "risk_level": (self.context_payload.get('risk') or {}).get('level'),
                        "output_target": self.context_payload.get('output_target'),
                    },
                }
                with open(data_dir / 'zhipu_online_samples.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _fallback_text(self):
        risk_level = self.context_payload.get('risk', {}).get('level', 'unknown')
        if risk_level == 'high':
            return (
                "【立即动作（今天）】暂停高压任务，先做3轮缓慢呼吸并补水；\n"
                "【短期动作（3天）】固定作息与运动，减少夜间高刺激内容；\n"
                "【升级条件】若连续高风险≥15分钟或再次出现明显异常，请联系家长与心理老师共同关注。"
            )
        return (
            "【立即动作（今天）】短暂休息5-10分钟，做一次呼吸放松；\n"
            "【短期动作（3天）】保持规律作息，减少连续久坐；\n"
            "【升级条件】若风险持续升高或影响学习状态，请通知家长或老师。"
        )

    def _rewrite_if_low_quality(self, text):
        txt = str(text or '').strip()
        if self._quality_ok(txt):
            return txt
        rewritten = self._rewrite_via_api(txt)
        if rewritten and self._quality_ok(rewritten):
            return rewritten
        return self._fallback_text()

    def run(self):
        attribution = ''
        try:
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
            cfg = self._pick_model_and_params()
            context_json = json.dumps(self.context_payload, ensure_ascii=False, indent=2)
            schema = LLM_CONTEXT_SCHEMA_HINT.strip()

            stage_a_prompt = (
                "你是校园心理守护助手。先只做风险归因，不给建议。\n"
                "要求：\n"
                "1) 禁用临床词汇：诊断/病情/治疗/患者/确诊；\n"
                "2) 仅基于输入 JSON，输出3条以内归因；\n"
                "3) 指明主导模态(visual/hrv/eeg)与证据；\n"
                "4) 输出格式固定为中文要点。\n"
                f"{schema}\n输入JSON:\n{context_json}"
            )

            payload_a = {
                "model": cfg["model"],
                "messages": [{"role": "user", "content": stage_a_prompt}],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": cfg["max_tokens"],
            }
            resp_a = requests.post(url, headers=headers, json=payload_a, timeout=12)
            if resp_a.status_code != 200:
                fb = self._fallback_text()
                self._persist_llm_artifacts(fb, attribution='')
                self.finished_signal.emit(fb)
                return
            root_a = resp_a.json()
            attribution = (((root_a.get('choices') or [{}])[0].get('message') or {}).get('content') or '').strip()
            if not attribution:
                attribution = "归因信息不足，建议按规则护栏输出。"

            stage_b_prompt = (
                "你是校园心理守护助手。基于归因生成行动建议。\n"
                "硬性输出结构（逐项输出，缺一不可）：\n"
                "1) 给学生：一句40字内温和建议\n"
                "2) 给老师：一句20字内观察提示\n"
                "3) 立即动作（今天）\n"
                "4) 短期动作（3天）\n"
                "5) 升级条件（何时通知家长/老师）\n"
                "高风险时，第3-5项必须具体可执行。\n"
                "禁止给药物与治疗建议，禁止临床诊断用语。\n"
                f"{schema}\n输入JSON:\n{context_json}\n\n阶段A归因:\n{attribution}"
            )
            payload_b = {
                "model": cfg["model"],
                "messages": [{"role": "user", "content": stage_b_prompt}],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": cfg["max_tokens"],
            }
            resp_b = requests.post(url, headers=headers, json=payload_b, timeout=15)
            if resp_b.status_code != 200:
                fb = self._fallback_text()
                self._persist_llm_artifacts(fb, attribution=attribution)
                self.finished_signal.emit(fb)
                return

            root_b = resp_b.json()
            answer = (((root_b.get('choices') or [{}])[0].get('message') or {}).get('content') or '').strip()
            final_text = self._rewrite_if_low_quality(answer)
            self._persist_llm_artifacts(final_text, attribution=attribution)
            self.finished_signal.emit(final_text)
        except Exception:
            fb = self._fallback_text()
            self._persist_llm_artifacts(fb, attribution=attribution)
            self.finished_signal.emit(fb)

class TTSThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            resp = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/audio/speech",
                json={
                    "model": "glm-tts",
                    "input": self.text,
                    "voice": "female",
                    "response_format": "wav",
                },
                headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"},
                timeout=15,
            )
            if resp.status_code == 200:
                import struct, wave
                # 保存原始 mono 24000Hz
                fd1, raw = tempfile.mkstemp(suffix="_raw.wav")
                with os.fdopen(fd1, "wb") as f:
                    f.write(resp.content)
                # 转立体声（NAU8822 只支持 2ch）
                with wave.open(raw, "rb") as wi:
                    fr = wi.readframes(wi.getnframes())
                    sw, sr = wi.getsampwidth(), wi.getframerate()
                samples = struct.unpack(f"<{len(fr)//2}h", fr)
                stereo = struct.pack(f"<{len(samples)*2}h", *[v for s in samples for v in (s, s)])
                fd2, out = tempfile.mkstemp(suffix="_2ch.wav")
                with wave.open(os.fdopen(fd2, "wb"), "w") as wo:
                    wo.setnchannels(2); wo.setsampwidth(sw); wo.setframerate(sr)
                    wo.writeframes(stereo)
                subprocess.run(["aplay", "-D", "hw:1,0", "-q", out], timeout=30)
                os.unlink(raw); os.unlink(out)
        except Exception:
            pass

class RemoteBPMThread(QThread):
    bpm_signal = pyqtSignal(float)
    telemetry_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, url="http://127.0.0.1:5001/esp32/latest", interval_ms=1000):
        super().__init__()
        self.url = url
        self.wave_url = url.rsplit('/', 1)[0] + '/wave'
        self.interval_ms = interval_ms
        self._running = True
        self._session = None
        self._last_ok = False

    def stop(self):
        self._running = False

    def run(self):
        self._session = requests.Session()
        self._session.trust_env = False
        self.status_signal.emit(f"> 远端心率拉取启动: {self.url}")

        while self._running:
            try:
                resp = self._session.get(self.url, timeout=2)
                if resp.status_code == 200:
                    payload = resp.json()
                    try:
                        wave_resp = self._session.get(self.wave_url, timeout=1)
                        if wave_resp.status_code == 200:
                            wave_payload = wave_resp.json()
                            payload["ir_values"] = wave_payload.get("ir_values")
                            payload["hr_display"] = wave_payload.get("display")
                    except Exception:
                        pass
                    bpm = payload.get("bpm")
                    if bpm is not None:
                        bpm = float(bpm)
                        if bpm > 0:
                            self.bpm_signal.emit(bpm)
                            self.telemetry_signal.emit(payload)
                            if not self._last_ok:
                                self.status_signal.emit("> 远端心率连接已建立。")
                            self._last_ok = True
                else:
                    if self._last_ok:
                        self.status_signal.emit(f"> 远端心率服务异常: HTTP {resp.status_code}")
                    self._last_ok = False
            except Exception:
                if self._last_ok:
                    self.status_signal.emit("> 远端心率连接中断，等待重连...")
                self._last_ok = False

            self.msleep(self.interval_ms)


class RemoteEEGThread(QThread):
    eeg_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self, url=None, interval_ms=None):
        super().__init__()
        self.url = url or getattr(config, "EEG_REMOTE_URL", "http://127.0.0.1:5001/eeg/latest")
        self.wave_url = self.url.rsplit('/', 1)[0] + '/wave'
        self.interval_ms = interval_ms if interval_ms is not None else getattr(config, "EEG_REMOTE_INTERVAL_MS", 800)
        self._running = True
        self._session = None
        self._last_ok = False

    def stop(self):
        self._running = False

    def run(self):
        self._session = requests.Session()
        self._session.trust_env = False
        self.status_signal.emit(f"> 远端EEG拉取启动: {self.url}")

        while self._running:
            try:
                resp = self._session.get(self.url, timeout=2)
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    try:
                        wave_resp = self._session.get(self.wave_url, timeout=1)
                        if wave_resp.status_code == 200:
                            wave_payload = wave_resp.json()
                            payload["raw_values"] = wave_payload.get("raw_values")
                    except Exception:
                        pass
                    att = payload.get("attention")
                    med = payload.get("meditation")
                    if att is not None and med is not None:
                        eeg_pkt = {
                            "attention": int(att),
                            "meditation": int(med),
                            "poor_signal": int(payload.get("poor_signal", 0)),
                            "eeg_power": payload.get("eeg_power"),
                            "raw_value": payload.get("raw_value"),
                            "raw_values": payload.get("raw_values"),
                            "source": "remote",
                        }
                        self.eeg_signal.emit(eeg_pkt)
                        if not self._last_ok:
                            self.status_signal.emit("> 远端EEG连接已建立。")
                        self._last_ok = True
                else:
                    if self._last_ok:
                        self.status_signal.emit(f"> 远端EEG服务异常: HTTP {resp.status_code}")
                    self._last_ok = False
            except Exception:
                if self._last_ok:
                    self.status_signal.emit("> 远端EEG连接中断，等待重连...")
                self._last_ok = False

            self.msleep(self.interval_ms)


class FlowFirewallThread(QThread):
    """Windows 最小可用心流防火墙：进程检测 + 可选终止。"""
    log_signal = pyqtSignal(str)

    def __init__(self, block_processes=None, interval_ms=2500, enforce_block=False):
        super().__init__()
        self.block_processes = [p.lower() for p in (block_processes or ["WeChat.exe", "QQ.exe", "DingTalk.exe", "Feishu.exe"])]
        self.interval_ms = int(interval_ms)
        self.enforce_block = bool(enforce_block)
        self._running = True
        self._seen = set()

    def set_enforce_block(self, value):
        self.enforce_block = bool(value)

    def set_block_processes(self, process_names):
        names = []
        for p in (process_names or []):
            sp = str(p).strip().lower()
            if sp:
                names.append(sp)
        self.block_processes = sorted(set(names))
        self._seen = set()

    def stop(self):
        self._running = False

    def _list_processes(self):
        try:
            out = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"],
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return []

        names = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # CSV 第1列是镜像名
            if line.startswith('"'):
                parts = [p.strip('"') for p in line.split('","')]
                if parts:
                    names.append(parts[0].lower())
        return names

    def _kill_process(self, name):
        try:
            subprocess.check_call(
                ["taskkill", "/F", "/IM", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def run(self):
        self.log_signal.emit("> [FLOW-FW] 实时检测引擎已启动（Windows）。")
        while self._running:
            proc_names = self._list_processes()
            running_blocked = sorted({p for p in proc_names if p in self.block_processes})

            if running_blocked:
                self._seen.discard("idle")
                for name in running_blocked:
                    key = f"detect:{name}"
                    if key not in self._seen:
                        self.log_signal.emit(f"> [FLOW-FW] 检测到干扰应用运行: {name}")
                        self._seen.add(key)

                    if self.enforce_block:
                        ok = self._kill_process(name)
                        self.log_signal.emit(
                            f"> [FLOW-FW] {'已终止' if ok else '终止失败'}: {name}"
                        )
                    else:
                        self.log_signal.emit(f"> [FLOW-FW] 建议关闭应用: {name}（当前为提醒模式）")
            else:
                # 避免刷屏：仅首次报告空闲
                if "idle" not in self._seen:
                    self.log_signal.emit("> [FLOW-FW] 未检测到干扰应用，专注环境正常。")
                    self._seen.add("idle")

            self.msleep(max(800, self.interval_ms))


class CyberRadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vis = 0.0
        self.hrv = 0.0
        self.eeg = 0.0
        self.eye = 0.0
        self.post = 0.0
        self.setMinimumHeight(190)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.34

        # 五维等分顶点（72°）: Vis, Phys, EEG, Eye, Post
        axis_names = ["Vis", "Phys", "EEG", "Eye", "Post"]
        values = [self.vis, self.hrv, self.eeg, self.eye, self.post]

        base_pts = []
        data_pts = []
        for i in range(5):
            ang = -math.pi / 2 + i * (2 * math.pi / 5)
            ux = math.cos(ang)
            uy = math.sin(ang)

            base_pts.append(QPointF(cx + ux * radius, cy + uy * radius))

            v = max(0.0, min(1.0, float(values[i])))
            data_pts.append(QPointF(cx + ux * radius * v, cy + uy * radius * v))

        # 外框五边形
        p.setPen(QPen(QColor(0, 243, 255, 80), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF(base_pts))

        # 轴线
        p.setPen(QPen(QColor("#2a3270"), 1))
        for pt in base_pts:
            p.drawLine(QPointF(cx, cy), pt)

        # 数据面（荧光青渐变填充）
        radar_poly = QPolygonF(data_pts)
        poly_bounds = radar_poly.boundingRect()
        grad = QLinearGradient(poly_bounds.left(), poly_bounds.top(), poly_bounds.right(), poly_bounds.bottom())
        grad.setColorAt(0.0, QColor(0, 243, 255, 130))
        grad.setColorAt(1.0, QColor(0, 243, 255, 55))

        p.setPen(QPen(QColor("#00d4ff"), 2.2))
        p.setBrush(QBrush(grad))
        p.drawPolygon(radar_poly)

        # 节点高亮
        p.setBrush(QColor("#00d4ff"))
        p.setPen(Qt.NoPen)
        for pt in data_pts:
            p.drawEllipse(pt, 3.2, 3.2)

        # 轴标签
        p.setPen(QColor(0, 243, 255, 170))
        p.setFont(QFont("Arial", 8, QFont.Bold))
        for i, pt in enumerate(base_pts):
            tx = cx + (pt.x() - cx) * 0.88
            ty = cy + (pt.y() - cy) * 0.88
            p.drawText(QRectF(tx - 18, ty - 8, 36, 16), Qt.AlignCenter, axis_names[i])


class CyberWaveWidget(QWidget):
    """高性能赛博风 PPG 波形组件（仅使用 PyQt5 原生绘制）。"""

    def __init__(self, parent=None, max_points=240):
        super().__init__(parent)
        self.setMinimumHeight(150)

        self._max_points = max(60, int(max_points))
        self._buffer = [0.0] * self._max_points
        self._head = 0
        self._count = 0

        self._v_min = 0.0
        self._v_max = 1.0

        self._x_cache = []
        self._y_cache = []
        self._ordered_cache = []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = max(1, self.width())
        step = w / max(1, self._max_points - 1)
        self._x_cache = [i * step for i in range(self._max_points)]
        self._y_cache = [0.0] * self._max_points
        self._ordered_cache = [0.0] * self._max_points

    def set_value_range(self, v_min, v_max):
        v_min = float(v_min)
        v_max = float(v_max)
        if v_max <= v_min:
            return
        self._v_min = v_min
        self._v_max = v_max

    def add_data(self, value):
        """推入一个新采样点，波形从右向左滚动。"""
        try:
            v = float(value)
        except Exception:
            return

        self._buffer[self._head] = v
        self._head = (self._head + 1) % self._max_points
        if self._count < self._max_points:
            self._count += 1

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._count <= 1:
            return

        w = float(self.width())
        h = float(self.height())
        if w <= 2.0 or h <= 2.0:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        # 背景网格（轻量）
        p.setPen(QPen(QColor(0, 243, 255, 22), 1))
        grid_rows = 4
        for i in range(1, grid_rows):
            y = h * i / grid_rows
            p.drawLine(0, int(y), int(w), int(y))

        # 按时间顺序展开环形缓冲区：最老 -> 最新
        start = self._head if self._count == self._max_points else 0
        for i in range(self._count):
            idx = (start + i) % self._max_points
            self._ordered_cache[i] = self._buffer[idx]

        # 动态自适应缩放：每帧使用当前窗口数据范围映射到可视高度
        d_min = self._ordered_cache[0]
        d_max = self._ordered_cache[0]
        for i in range(1, self._count):
            v = self._ordered_cache[i]
            if v < d_min:
                d_min = v
            if v > d_max:
                d_max = v

        top_pad = 10.0
        bottom_pad = 12.0
        usable_h = max(1.0, h - top_pad - bottom_pad)
        d_rng = d_max - d_min

        if d_rng <= 1e-12:
            mid_y = top_pad + usable_h * 0.5
            for i in range(self._count):
                self._y_cache[i] = mid_y
        else:
            inv_rng = 1.0 / d_rng
            for i in range(self._count):
                t = (self._ordered_cache[i] - d_min) * inv_rng
                self._y_cache[i] = top_pad + (1.0 - t) * usable_h

        if self._count < self._max_points:
            dx = w / max(1, self._count - 1)
            x_get = lambda i: i * dx
        else:
            x_get = lambda i: self._x_cache[i]

        line_path = QPainterPath()
        x0 = x_get(0)
        y0 = self._y_cache[0]
        line_path.moveTo(x0, y0)

        for i in range(1, self._count):
            x1 = x_get(i)
            y1 = self._y_cache[i]
            xm = (x0 + x1) * 0.5
            ym = (y0 + y1) * 0.5
            line_path.quadTo(x0, y0, xm, ym)
            x0, y0 = x1, y1
        line_path.lineTo(x0, y0)

        # 幽灵阴影（向下渐隐）
        fill_path = QPainterPath(line_path)
        fill_path.lineTo(x0, h)
        fill_path.lineTo(x_get(0), h)
        fill_path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(0, 243, 255, 92))
        grad.setColorAt(0.45, QColor(0, 243, 255, 42))
        grad.setColorAt(1.0, QColor(0, 243, 255, 0))

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(fill_path)

        # 主波形（荧光青）
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#00d4ff"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(line_path)


class RiskTraceDialog(QDialog):
    """风险分数第一性原理溯源面板（纯 QPainter 绘制）。"""

    def __init__(self, total_score=0.75, items=None, parent=None):
        super().__init__(parent)
        self.total_score = float(total_score)
        self.items = items or [
            ("视觉疲劳", 0.20),
            ("HRV 降低", 0.40),
            ("专注度涣散", 0.15),
        ]
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(False)
        self.setFixedSize(600, 350)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(38)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 243, 255, 65))
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        rect = QRectF(12.0, 12.0, w - 24.0, h - 24.0)

        # 毛玻璃暗黑底
        bg = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        bg.setColorAt(0.0, QColor(9, 14, 28, 225))
        bg.setColorAt(1.0, QColor(5, 8, 16, 235))
        p.setPen(QPen(QColor(0, 243, 255, 120), 1.2))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rect, 16, 16)

        # 标题
        p.setPen(QColor("#00d4ff"))
        p.setFont(QFont("微软雅黑", 11, QFont.Bold))
        p.drawText(QRectF(rect.left() + 18, rect.top() + 10, rect.width() - 36, 24), Qt.AlignLeft | Qt.AlignVCenter, "第一性原理溯源面板 · Risk Trace")

        # 左侧总分节点
        cx = rect.left() + 132
        cy = rect.center().y() + 12
        r = 46

        halo = QRadialGradient(cx, cy, r * 1.9)
        halo.setColorAt(0.0, QColor(0, 243, 255, 80))
        halo.setColorAt(1.0, QColor(0, 243, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(cx, cy), r * 1.9, r * 1.9)

        node_grad = QRadialGradient(cx - 8, cy - 8, r * 1.2)
        node_grad.setColorAt(0.0, QColor(18, 40, 60, 245))
        node_grad.setColorAt(1.0, QColor(6, 12, 22, 250))
        p.setBrush(QBrush(node_grad))
        p.setPen(QPen(QColor("#00d4ff"), 2.0))
        p.drawEllipse(QPointF(cx, cy), r, r)

        p.setPen(QColor("#00d4ff"))
        p.setFont(QFont("Arial", 20, QFont.Bold))
        p.drawText(QRectF(cx - 32, cy - 18, 64, 30), Qt.AlignCenter, f"{self.total_score:.2f}")
        p.setFont(QFont("微软雅黑", 9))
        p.drawText(QRectF(cx - 46, cy + 14, 92, 22), Qt.AlignCenter, "综合风险")

        # 右侧因果分支（赛博神经元连线）
        right_x = rect.left() + 332
        start_y = rect.top() + 86
        step = 74

        for i, (name, weight) in enumerate(self.items):
            ny = start_y + i * step
            nx = right_x

            path = QPainterPath()
            path.moveTo(cx + r, cy)
            c1 = QPointF(cx + 105, cy + (ny - cy) * 0.22)
            c2 = QPointF(nx - 70, ny - (ny - cy) * 0.18)
            path.cubicTo(c1, c2, QPointF(nx - 26, ny))

            p.setPen(QPen(QColor(0, 243, 255, 150), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

            # 分支节点
            p.setPen(QPen(QColor("#00d4ff"), 1.5))
            p.setBrush(QColor(0, 243, 255, 40))
            p.drawEllipse(QPointF(nx - 26, ny), 7, 7)

            # 分支文本框
            box = QRectF(nx - 14, ny - 22, 184, 44)
            p.setPen(QPen(QColor(0, 243, 255, 90), 1.0))
            p.setBrush(QColor(8, 18, 32, 195))
            p.drawRoundedRect(box, 8, 8)

            p.setPen(QColor("#a5f3fc"))
            p.setFont(QFont("微软雅黑", 9, QFont.Bold))
            p.drawText(QRectF(box.left() + 10, box.top() + 5, box.width() - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, name)

            p.setPen(QColor("#22d3ee"))
            p.setFont(QFont("Consolas", 9, QFont.Bold))
            p.drawText(QRectF(box.left() + 10, box.top() + 21, box.width() - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, f"+{weight:.2f}")

        # 右上角关闭提示
        p.setPen(QColor("#6b7280"))
        p.setFont(QFont("微软雅黑", 8))
        p.drawText(QRectF(rect.right() - 120, rect.top() + 10, 100, 20), Qt.AlignRight | Qt.AlignVCenter, "点击面板外关闭")

    def mousePressEvent(self, event):
        self.close()
        event.accept()


class CyberCard(QFrame):
    def __init__(self, parent=None, is_transparent=False):
        super().__init__(parent)
        bg_color = "transparent" if is_transparent else "#0e1330"
        self.setStyleSheet(f"QFrame {{ background-color: {bg_color}; border: 1px solid #2a3270; border-radius: 12px; }}")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15); shadow.setColor(QColor(0, 243, 255, 30)); shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

class MainWindow(QMainWindow):
    def __init__(self, camera_thread, recorder):
        super().__init__()
        self.camera_thread = camera_thread
        self.recorder = recorder
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        _vnc = os.environ.get('DISPLAY', '').startswith(':99') or os.environ.get('MINDROOM_VNC', '') == '1'
        if not _vnc:
            self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.setMinimumSize(1024, 680)  # 放低下限，VNC 下 main.py 会自适应
        self.high_risk_counter = 0
        self.last_ai_time = 0
        self.is_ai_generating = False
        self.flow_firewall_active = False
        self.flow_firewall_enforce = False
        self.ai_cooldown_secs = 60
        self.ai_cooldown_until = 0.0
        self.flow_firewall_thread = None
        self.flow_schedule_enabled = True
        self.flow_schedule_start = QTime(8, 0)
        self.flow_schedule_end = QTime(22, 30)
        self.flow_block_processes = ["WeChat.exe", "QQ.exe", "DingTalk.exe", "Feishu.exe"]
        self.latest_remote_bpm = None
        self.latest_visual_snapshot = {"expr": "未检测到人脸", "expr_prob": 0.0, "visual_risk": 0.0}
        self.latest_physio_snapshot = {
            "bpm": None,
            "hrv_rmssd": None,
            "rr_valid_ratio": 0.0,
            "ppg_amp_p95": 0.0,
            "ppg_quality": "unknown",
            "fs": 0.0,
        }
        self._pred_history = []
        self._log_last_emit = {}
        self._log_throttle_secs = 5.0
        self._risk_recent_states = deque(maxlen=600)
        self._recent_anomaly_events = deque(maxlen=30)
        self._llm_mode = 'realtime'
        self._llm_target = 'student_teacher'
        self._managed_threads = {}
        self._event_replay_init()
        self._alert_policy_init()
        self._baseline_init()
        self._model_health_init()
        self._intervention_eval_init()
        self._ui_tick_fast_ms = int(getattr(config, 'UI_TICK_FAST_MS', 25))
        self._ui_tick_medium_ms = int(getattr(config, 'UI_TICK_MEDIUM_MS', 66))
        self._ui_tick_slow_ms = int(getattr(config, 'UI_TICK_SLOW_MS', 1000))
        # 显示层平滑（不改底层真实值）
        self._hr_display_ema = None
        self._hrv_display_ema = None
        self._hr_target = None
        self._hrv_target = None
        self._sdnn_target = None
        self._sdnn_display_ema = None
        self._last_sdnn_push_ts = 0.0
        # 与 1.py rr_intervals_buffer 一致：滚动 RR ≥5 才出 SDNN/RMSSD
        self._remote_rr_buffer = deque(maxlen=8)
        self._init_ui()
        self._connect_signals()
        self._start_remote_bpm_monitor()
        self._start_physio_monitor()
        self._start_eeg_monitor()
        self._start_mock_wave_engine()
        self._start_nn_engine()
        self._set_data_source_mode('offline_replay')

    def _event_replay_init(self):
        self._replay_pre_sec = int(getattr(config, 'REPLAY_PRE_SEC', 30))
        self._replay_post_sec = int(getattr(config, 'REPLAY_POST_SEC', 30))
        self._replay_trigger_threshold = float(getattr(config, 'REPLAY_TRIGGER_THRESHOLD', getattr(config, 'RISK_THRESHOLD_HIGH', 0.8)))
        self._replay_cooldown_sec = int(getattr(config, 'REPLAY_COOLDOWN_SEC', 120))
        self._replay_last_trigger_ts = 0.0
        self._replay_capture_active = False
        self._replay_capture_end_ts = 0.0
        self._replay_session = []
        self._replay_ring = deque(maxlen=max(300, self._replay_pre_sec * 10))
        out_dir = Path(getattr(config, 'REPLAY_OUTPUT_DIR', str(Path(__file__).resolve().parent / 'event_replays')))
        out_dir.mkdir(parents=True, exist_ok=True)
        self._replay_out_dir = out_dir

    def _alert_policy_init(self):
        self._alert_cfg = {
            'L2': {
                'threshold': float(getattr(config, 'ALERT_L2_THRESHOLD', getattr(config, 'RISK_THRESHOLD_MEDIUM', 0.6))),
                'duration': int(getattr(config, 'ALERT_L2_DURATION_SEC', 10)),
                'cooldown': int(getattr(config, 'ALERT_L2_COOLDOWN_SEC', 60)),
            },
            'L3': {
                'threshold': float(getattr(config, 'ALERT_L3_THRESHOLD', getattr(config, 'RISK_THRESHOLD_HIGH', 0.8))),
                'duration': int(getattr(config, 'ALERT_L3_DURATION_SEC', 15)),
                'cooldown': int(getattr(config, 'ALERT_L3_COOLDOWN_SEC', 180)),
            },
            'L4': {
                'threshold': float(getattr(config, 'ALERT_L4_THRESHOLD', 0.9)),
                'duration': int(getattr(config, 'ALERT_L4_DURATION_SEC', 20)),
                'cooldown': int(getattr(config, 'ALERT_L4_COOLDOWN_SEC', 300)),
            },
        }
        self._alert_active_since = {'L2': None, 'L3': None, 'L4': None}
        self._alert_last_emit = {'L2': 0.0, 'L3': 0.0, 'L4': 0.0}

    def _baseline_init(self):
        self._baseline_warmup_sec = int(getattr(config, 'BASELINE_WARMUP_SEC', 600))
        self._baseline_start_ts = time.time()
        self._baseline_samples = []
        self._baseline_stats = None

    def _baseline_update(self, risk):
        now = time.time()
        if (now - self._baseline_start_ts) <= self._baseline_warmup_sec:
            self._baseline_samples.append(float(risk))
            if len(self._baseline_samples) > 5000:
                self._baseline_samples = self._baseline_samples[-5000:]
            if (self._baseline_stats is None) and (now - self._baseline_start_ts) >= self._baseline_warmup_sec and len(self._baseline_samples) >= 30:
                vals = sorted(self._baseline_samples)
                n = len(vals)
                mean = sum(vals) / n
                p90 = vals[int(0.9 * (n - 1))]
                self._baseline_stats = {'mean': mean, 'p90': p90}
                self._append_terminal_log(f"> [BASELINE] 个体基线已建立: mean={mean:.2f}, p90={p90:.2f}", dedup=False)
        return self._baseline_normalized_risk(risk)

    def _baseline_normalized_risk(self, risk):
        if not self._baseline_stats:
            return float(risk)
        mean = self._baseline_stats['mean']
        p90 = max(mean + 1e-6, self._baseline_stats['p90'])
        norm = (float(risk) - mean) / (p90 - mean)
        return max(0.0, min(1.0, norm))

    def _alert_policy_tick(self, risk):
        now = time.time()
        for level in ('L2', 'L3', 'L4'):
            cfg = self._alert_cfg[level]
            if risk >= cfg['threshold']:
                if self._alert_active_since[level] is None:
                    self._alert_active_since[level] = now
                hold = now - self._alert_active_since[level]
                cool = now - self._alert_last_emit[level]
                if hold >= cfg['duration'] and cool >= cfg['cooldown']:
                    self._alert_last_emit[level] = now
                    self._emit_alert(level, risk, hold)
            else:
                self._alert_active_since[level] = None

    def _emit_alert(self, level, risk, hold_sec):
        msg = {
            'L2': f'> [ALERT-L2] 风险持续 {hold_sec:.0f}s (risk={risk:.2f})：建议学生短休+呼吸训练。',
            'L3': f'> [ALERT-L3] 风险持续 {hold_sec:.0f}s (risk={risk:.2f})：建议通知辅导员关注。',
            'L4': f'> [ALERT-L4] 风险持续 {hold_sec:.0f}s (risk={risk:.2f})：建议人工复核并启动家校联动。',
        }[level]
        self._recent_anomaly_events.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'level': level,
            'risk': round(float(risk), 3),
            'duration_sec': int(hold_sec),
            'is_continuous': True,
        })
        self._append_terminal_log(msg, level='warn', dedup=False)
        self._intervention_eval_mark(level, float(risk))
        try:
            if hasattr(self, 'ai_console'):
                self.ai_console.append(msg)
        except Exception:
            pass

    def _intervention_eval_mark(self, level, risk_at_alert):
        evt = {
            'ts': time.time(),
            'level': level,
            'risk_at_alert': float(risk_at_alert),
            'done': set(),
        }
        self._intervention_events.append(evt)
        if len(self._intervention_events) > 100:
            self._intervention_events = self._intervention_events[-100:]
        self._append_terminal_log(f"> [EVAL] 记录干预事件: {level} @ risk={risk_at_alert:.2f}", dedup=False)

    def _intervention_eval_tick(self, current_risk, ts_now):
        if not self._intervention_events:
            return
        for evt in self._intervention_events:
            dt = ts_now - evt['ts']
            for sec in self._intervention_eval_targets:
                if sec in evt['done']:
                    continue
                if dt >= sec:
                    delta = float(current_risk) - float(evt['risk_at_alert'])
                    trend = '下降' if delta < -0.05 else ('上升' if delta > 0.05 else '持平')
                    self._append_terminal_log(
                        f"> [EVAL] {evt['level']} 后 {int(sec/60)}min: 当前risk={current_risk:.2f}, 变化={delta:+.2f} ({trend})",
                        dedup=False)
                    evt['done'].add(sec)

    def _emit_model_health(self):
        mon = getattr(self, 'nn_monitor', None)
        if mon is None or not mon.lat:
            return
        h = mon.health()
        fb = float(h.get('fallback_ratio', 0.0))
        p95 = float(h.get('lat_p95_ms', 0.0))
        state = h.get('state', 'idle')
        msg = f"> [HEALTH] state={state}, fallback={fb*100:.1f}%, p95={p95:.1f}ms"
        self._append_terminal_log(msg, dedup=False)
        try:
            if hasattr(self, 'ai_console'):
                self.ai_console.append(msg)
        except Exception:
            pass

    def _model_health_init(self):
        self._health_last_emit_ts = 0.0
        self._health_emit_interval_sec = int(getattr(config, 'HEALTH_EMIT_INTERVAL_SEC', 30))
        self._health_fallback_warn_threshold = float(getattr(config, 'HEALTH_FALLBACK_WARN_RATIO', 0.1))
        self._health_latency_warn_threshold = float(getattr(config, 'HEALTH_LAT_P95_WARN_MS', 200.0))

    def _intervention_eval_init(self):
        self._intervention_eval_targets = [
            int(getattr(config, 'INTERVENTION_EVAL_5M_SEC', 300)),
            int(getattr(config, 'INTERVENTION_EVAL_10M_SEC', 600)),
            int(getattr(config, 'INTERVENTION_EVAL_20M_SEC', 1200)),
        ]
        self._intervention_events = []

    def _event_replay_push_sample(self, risk):
        snap = {
            'ts': time.time(),
            'risk': float(risk),
            'visual': dict(self.latest_visual_snapshot or {}),
            'physio': dict(self.latest_physio_snapshot or {}),
        }
        self._replay_ring.append(snap)
        self._intervention_eval_tick(float(risk), snap['ts'])
        if self._replay_capture_active:
            self._replay_session.append(snap)
            if snap['ts'] >= self._replay_capture_end_ts:
                self._event_replay_finalize()

    def _event_replay_maybe_trigger(self, risk):
        now = time.time()
        if risk < self._replay_trigger_threshold:
            return
        if self._replay_capture_active:
            return
        if now - self._replay_last_trigger_ts < self._replay_cooldown_sec:
            return
        self._replay_last_trigger_ts = now
        self._replay_capture_active = True
        self._replay_capture_end_ts = now + self._replay_post_sec
        self._replay_session = [x for x in self._replay_ring if x['ts'] >= (now - self._replay_pre_sec)]
        self._set_replay_pill('warn', '回放 采集中')
        self._append_terminal_log(
            f"> [REPLAY] 触发事件回放采集: risk={risk:.2f}, 窗口=-{self._replay_pre_sec}s~+{self._replay_post_sec}s", dedup=False)

    def _event_replay_finalize(self):
        if not self._replay_capture_active:
            return
        self._replay_capture_active = False
        if not self._replay_session:
            self._set_replay_pill('sim', '回放 待机')
            return
        t0 = self._replay_session[0]['ts']
        event_id = time.strftime('%Y%m%d_%H%M%S', time.localtime(t0))
        out_path = self._replay_out_dir / f'replay_{event_id}.json'
        payload = {
            'event_id': event_id,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'window_sec': {'pre': self._replay_pre_sec, 'post': self._replay_post_sec},
            'threshold': self._replay_trigger_threshold,
            'samples': self._replay_session,
        }
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._append_terminal_log(f"> [REPLAY] 已保存事件回放: {out_path}", dedup=False)
            self._set_replay_pill('live', '回放 已保存')
            if hasattr(self, 'ai_console'):
                self.ai_console.append(f"> [REPLAY] 已保存: {out_path.name}")
        except Exception as exc:
            self._set_replay_pill('offline', '回放 保存失败')
            self._append_terminal_log(f"> [REPLAY] 保存失败: {exc}", level='warn', dedup=False)
        finally:
            self._replay_session = []

    def _set_replay_pill(self, state, text):
        try:
            if hasattr(self, 'pill_replay') and self.pill_replay:
                self.pill_replay.set_state(state, text)
        except Exception:
            pass

    def _manual_trigger_replay(self):
        if self._replay_capture_active:
            self._append_terminal_log('> [REPLAY] 正在采集中，请稍后。', dedup=False)
            return
        now = time.time()
        self._replay_last_trigger_ts = now
        self._replay_capture_active = True
        self._replay_capture_end_ts = now + self._replay_post_sec
        self._replay_session = [x for x in self._replay_ring if x['ts'] >= (now - self._replay_pre_sec)]
        self._set_replay_pill('warn', '回放 手动采集')
        self._append_terminal_log(
            f'> [REPLAY] 手动触发采集: 窗口=-{self._replay_pre_sec}s~+{self._replay_post_sec}s', dedup=False)

    def _start_nn_engine(self):
        """1Hz 神经网络推理：FeatureAligner + InferenceEngine + InferenceMonitor。"""
        try:
            from nn.feature_aligner import FeatureAligner
            from nn.inference_engine import InferenceEngine
            try:
                from nn.monitor import InferenceMonitor
            except Exception:
                InferenceMonitor = None
            self.aligner = FeatureAligner(out_csv='nn/aligned_features.csv')
            self.aligner.start()
            # 默认宽松模式（允许 fallback），生产环境可改成 strict=True
            self.inference = InferenceEngine(strict=False)
            self.nn_monitor = InferenceMonitor(self.inference) if InferenceMonitor else None
            mode = 'ONNX' if self.inference.loaded else 'Fallback'
            self.pill_top_nn.set_state('live' if self.inference.loaded else 'sim',
                                        f'神经网络  {mode}')
            self.nn_lbl_quant.setText('INT8 / RKNN' if self.inference.loaded else 'FP32 / FB')
            # 把 provider 信息也展示出来（如果 UI 有对应 label）
            if hasattr(self, 'nn_lbl_provider'):
                provs = list(self.inference.providers_used) or ['—']
                self.nn_lbl_provider.setText(' / '.join(p.replace('ExecutionProvider', '') for p in provs))
        except Exception as e:
            self.pill_top_nn.set_state('offline', f'神经网络  禁用')
            self.aligner = None
            self.inference = None
            self.nn_monitor = None
            return
        # 1Hz tick 定时器
        from PyQt5.QtCore import QTimer
        self._nn_timer = QTimer(self)
        self._nn_timer.setInterval(1000)
        self._nn_timer.timeout.connect(self._nn_tick)
        self._nn_timer.start()

    def _nn_tick(self):
        if not getattr(self, 'aligner', None):
            return
        try:
            v = self.latest_visual_snapshot
            self.aligner.push_vision(
                expr_probs=[v.get('expr_prob', 0.0)] + [0.0] * 6,
                eye_fatigue=v.get('eye_fatigue', 0.0),
                posture_risk=v.get('posture_risk', 0.0))
            p = self.latest_physio_snapshot
            self.aligner.push_physio(
                bpm=p.get('bpm') or 0,
                hrv_rmssd=p.get('hrv_rmssd') or 0,
                sdnn=p.get('sdnn') or 0,
                lf_hf=1.5,
                rr_valid=p.get('rr_valid_ratio') or 0)
            self.aligner.tick()
            # 优先走 monitor（带漂移检测），不可用时退回直接推理
            if getattr(self, 'nn_monitor', None) is not None:
                out = self.nn_monitor.predict(self.aligner.get_window())
            else:
                out = self.inference.predict(self.aligner.get_window())
            self.nn_lbl_lat.setText(f'{out["latency_ms"]:.1f} ms')
            mw = out['modal_w']
            self.contrib_bar.set_weights(
                max(0.01, float(mw[0])),
                max(0.01, float(mw[1])),
                max(0.01, float(mw[2])),
            )
            # 监控状态：当 fallback 持续偏高 / 出现漂移时，把顶栏 pill 切到 warn
            mon = getattr(self, 'nn_monitor', None)
            if mon is not None and mon.lat:
                h = mon.health()
                pill_state = {'live': 'live', 'warn': 'warn',
                              'offline': 'offline', 'idle': 'sim'}.get(h['state'], 'sim')
                self.pill_top_nn.set_state(
                    pill_state, f'神经网络  {h["mode"].upper()}  {h["reason"]}')
                now = time.time()
                if now - self._health_last_emit_ts >= self._health_emit_interval_sec:
                    self._health_last_emit_ts = now
                    self._emit_model_health()
            # 风险时序：固定 600 帧滚动窗口（10min @ 1Hz）
            self._buf_risk_trend.append(out['risk_score'])
            n = len(self._buf_risk_trend)
            xs = list(range(n))
            self.risk_trend_curve.setData(xs, list(self._buf_risk_trend))
            self.risk_trend_plot.setXRange(max(0, n - 600), max(60, n), padding=0)
            if self.inference.loaded:
                self.risk_hero.set_score(out['risk_score'])

            # 把"主因短句"写回总线供工具箱/报告复用
            bus = RiskBus.instance()
            v_snap = self.latest_visual_snapshot or {}
            p_snap = self.latest_physio_snapshot or {}
            bus.set_modal_factor('vision',
                f'表情 {v_snap.get("expr", "—")} · 眼疲劳 {v_snap.get("eye_fatigue", 0):.2f}')
            rmssd = p_snap.get('hrv_rmssd') or 0
            bus.set_modal_factor('hrv',
                f'RMSSD {rmssd:.1f} ms' + ('（参考 ≥50）' if rmssd and rmssd < 50 else ''))
            att = getattr(self, 'bci_panel', None)
            if att is not None and hasattr(att, 'attention'):
                bus.set_modal_factor('eeg',
                    f'专注度 {att.attention} · 放松 {att.meditation}')
            # 把完整 NN 结果推上总线（自带 tier_probs / 投票决策 / 历史）
            bus.push(out)
        except Exception:
            pass

    def _init_ui(self):
        from ui_dashboard import build_dashboard
        self._bci_panel_widget = BCIPanel()
        self.bci_panel = self._bci_panel_widget
        self.page_bci = self._bci_panel_widget  # 兼容老 _start_eeg_monitor
        build_dashboard(self, self._bci_panel_widget)
        # “我已处理”按钮（在 ui_dashboard.py 顶栏 AI 卡中）
        try:
            if hasattr(self, 'btn_ai_ack') and self.btn_ai_ack:
                self.btn_ai_ack.clicked.connect(self._ack_ai_intervention)
            if hasattr(self, 'btn_ai_deep') and self.btn_ai_deep:
                self.btn_ai_deep.clicked.connect(
                    lambda: self._trigger_ai_intervention_manual(deep=True))
            if hasattr(self, 'combo_llm_target') and self.combo_llm_target:
                self.combo_llm_target.currentIndexChanged.connect(self._on_llm_target_changed)
                self._on_llm_target_changed(self.combo_llm_target.currentIndex())
            if hasattr(self, 'btn_replay') and self.btn_replay:
                self.btn_replay.clicked.connect(self._open_latest_replay)
            if hasattr(self, 'btn_replay_manual') and self.btn_replay_manual:
                self.btn_replay_manual.clicked.connect(self._manual_trigger_replay)
            if hasattr(self, 'btn_export_plain') and self.btn_export_plain:
                self.btn_export_plain.clicked.connect(self._export_plain_risk_csv)
            if hasattr(self, 'btn_export_train') and self.btn_export_train:
                self.btn_export_train.clicked.connect(self._export_train_dataset)
            if hasattr(self, 'btn_record_toggle') and self.btn_record_toggle:
                self.btn_record_toggle.clicked.connect(self._toggle_recording)
                self._sync_record_button_state()
        except Exception:
            pass
        self._install_stub_attrs()

    def _install_stub_attrs(self):
        """新单页大屏不再有的旧控件占位（避免老逻辑 AttributeError）。"""
        from PyQt5.QtWidgets import QTextEdit, QLabel, QPushButton, QCheckBox, QComboBox, QTimeEdit, QLineEdit, QWidget
        # 终端 / 焦点 / 防火墙日志统一指向 ai_console（单页只剩它）
        for n in ('terminal_text', 'firewall_log', 'zen_log', 'physio_summary'):
            if not hasattr(self, n):
                setattr(self, n, self.ai_console)
        # 兼容旧波形别名
        for old, new in [('wave_ppg','curve_wave_ppg'), ('wave_ecg','curve_wave_ecg'),
                         ('wave_hr','curve_wave_hr'),  ('wave_hrv','curve_wave_hrv'),
                         ('ppg_wave',None)]:
            if not hasattr(self, old):
                setattr(self, old, getattr(self, new, None) if new else _NoOp())
        # 旧导航/按钮占位
        for n in ('btn_nav_live','btn_nav_chart','btn_nav_waves','btn_nav_flow',
                  'btn_nav_zen','btn_nav_sys','btn_nav_bci','btn_rec',
                  'btn_apply_fw_list','btn_zen_focus','btn_flow_shield','btn_shadow_mode',
                  'chk_flow_enforce','chk_flow_schedule','chk_fw_enforce','chk_fw_schedule',
                  'time_flow_start','time_flow_end','time_fw_start','time_fw_end',
                  'edit_fw_processes','combo_severity','stacked_widget'):
            if not hasattr(self, n):
                setattr(self, n, _NoOp())
        for n in ('expr_label','risk_display','risk_tag_label','radar_widget',
                  'pred_curve','social_battery','fig','canvas','ax',
                  'card_status_eye','card_status_posture','card_status_vis'):
            if not hasattr(self, n):
                setattr(self, n, _NoOp())
        # metric_values 字典占位
        self.metric_values = {k: _NoOp() for k in ('vis','hrv','eeg','eye','post')}
        # risk_display alias 指向 risk_hero.value（保持 setText/setStyleSheet 工作）
        self.risk_display = self.risk_hero.value
        # 兼容老的 _switch_page 调用（变成 no-op）
        self.nav_btns = []

    def _switch_page(self, index):
        # 单页大屏：保留接口但不切换页
        return


    def _on_risk_display_click(self, event):
        if event.button() != Qt.LeftButton:
            return

        try:
            total = float(self.risk_display.text())
        except Exception:
            total = 0.0

        vis_v = float(self.metric_values.get("vis", QLabel("0")).text()) if "vis" in self.metric_values else 0.0
        hrv_v = float(self.metric_values.get("hrv", QLabel("0")).text()) if "hrv" in self.metric_values else 0.0
        eeg_v = float(self.metric_values.get("eeg", QLabel("0")).text()) if "eeg" in self.metric_values else 0.0

        items = [
            ("视觉疲劳", vis_v),
            ("HRV 降低", hrv_v),
            ("专注度涣散", eeg_v),
        ]

        dlg = RiskTraceDialog(total_score=total, items=items, parent=self)
        center = self.mapToGlobal(self.rect().center())
        dlg.move(int(center.x() - dlg.width() / 2), int(center.y() - dlg.height() / 2))
        dlg.show()

        self._risk_trace_dialog = dlg

    def _update_image(self, img):
        if self.stacked_widget.currentIndex() in (4, 6): return
        h, w, c = img.shape
        qi = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
        # 直接 setPixmap,不用 scaled — video_label 已设 setScaledContents(True)
        self.video_label.setPixmap(QPixmap.fromImage(qi))

    def _update_risk(
        self,
        risk,
        level,
        expr,
        prob,
        eye_fatigue_index=0.0,
        posture_risk=0.0,
        forward_head_risk=0.0,
        down_head_risk=0.0,
    ):
        self.risk_display.setText(f"{risk:.2f}")
        safe_eye = max(0.0, min(1.0, float(eye_fatigue_index) if eye_fatigue_index is not None else 0.0))
        safe_posture = max(0.0, min(1.0, float(posture_risk) if posture_risk is not None else 0.0))
        safe_forward = max(0.0, min(1.0, float(forward_head_risk) if forward_head_risk is not None else 0.0))
        safe_down = max(0.0, min(1.0, float(down_head_risk) if down_head_risk is not None else 0.0))
        self.latest_visual_snapshot = {
            "expr": expr if expr is not None else "未检测到人脸",
            "expr_prob": float(prob) if prob is not None else 0.0,
            "visual_risk": float(self.radar_widget.vis) if hasattr(self, 'radar_widget') else 0.0,
            "eye_fatigue_index": safe_eye,
            "posture_risk": safe_posture,
            "forward_head_risk": safe_forward,
            "down_head_risk": safe_down,
        }
        if expr == "未检测到人脸":
            self.expr_label.setText("当前状态: 未检测到人脸")
        else:
            self.expr_label.setText(f"当前状态: {expr} ({prob:.2f})")

        try:
            self.bar_eye._bar.setValue(int(safe_eye * 100))
            self.bar_eye._val.setText(f'{int(safe_eye * 100)}%')
            self.bar_posture._bar.setValue(int(safe_posture * 100))
            self.bar_posture._val.setText(f'{int(safe_posture * 100)}%')
            self.bar_forward._bar.setValue(int(safe_forward * 100))
            self.bar_forward._val.setText(f'{int(safe_forward * 100)}%')
            self.bar_down._bar.setValue(int(safe_down * 100))
            self.bar_down._val.setText(f'{int(safe_down * 100)}%')
        except Exception:
            pass

        # 风险三档态：低 / 中 / 高
        if risk < config.RISK_THRESHOLD_MEDIUM:
            color, tag = C_OK, "低 · LOW"
        elif risk < config.RISK_THRESHOLD_HIGH:
            color, tag = C_WARN, "中 · MEDIUM"
        else:
            color, tag = C_RISK, "高 · HIGH"
        self.risk_display.setStyleSheet(f"color: {color}; border:none; background:transparent;")
        if hasattr(self, 'risk_tag_label'):
            self.risk_tag_label.setText(tag)
            self.risk_tag_label.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold; letter-spacing:3px; border:none;")

        now_ts = time.time()
        self._risk_recent_states.append({
            'ts': now_ts,
            'risk': float(risk),
            'level': self._risk_level_text(risk),
        })
        self._event_replay_push_sample(risk)
        self._event_replay_maybe_trigger(risk)

        baseline_risk = self._baseline_update(risk)
        self._alert_policy_tick(baseline_risk)

        if baseline_risk >= config.RISK_THRESHOLD_MEDIUM:
            self.high_risk_counter += 1
        else:
            self.high_risk_counter = 0

        if self.high_risk_counter >= config.ALERT_CONSECUTIVE_HIGH and not self.is_ai_generating:
            now = time.time()
            if now < self.ai_cooldown_until:
                remain = int(self.ai_cooldown_until - now)
                self._append_terminal_log(f"> AI 触发冷却中：剩余 {remain}s", dedup=True)
                return
            if now - self.last_ai_time > 30:
                self._trigger_ai_intervention(risk)

    def _risk_level_text(self, risk_value):
        rv = float(risk_value)
        if rv < float(getattr(config, 'RISK_THRESHOLD_MEDIUM', 0.6)):
            return 'low'
        if rv < float(getattr(config, 'RISK_THRESHOLD_HIGH', 0.8)):
            return 'medium'
        return 'high'

    def _build_trend_24h(self, now_ts=None):
        """24h 趋势：优先解密历史，其次 NN 10min 缓冲，再次当前 _risk_recent_states。"""
        now_ts = float(now_ts or time.time())
        horizon = 86400.0
        risks = []
        source = 'insufficient_data'

        try:
            if hasattr(self, 'recorder') and self.recorder is not None:
                df = self.recorder.get_decrypted_history()
                if df is not None and not df.empty and 'risk' in df.columns and 'timestamp' in df.columns:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df['risk'] = pd.to_numeric(df['risk'], errors='coerce')
                    df = df.dropna(subset=['timestamp', 'risk'])
                    cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=horizon)
                    df = df[df['timestamp'] >= cutoff]
                    if len(df) >= 3:
                        risks = [float(x) for x in df['risk'].tolist()]
                        source = 'recorder_24h'
        except Exception:
            pass

        if len(risks) < 3:
            buf = getattr(self, '_buf_risk_trend', None)
            if buf is not None and len(buf) >= 3:
                risks = [float(x) for x in list(buf)]
                source = 'nn_buffer_10m'

        if len(risks) < 3:
            recent = [x for x in self._risk_recent_states
                      if (now_ts - float(x.get('ts', now_ts))) <= horizon]
            if len(recent) >= 3:
                risks = [float(x.get('risk', 0.0)) for x in recent]
                source = 'risk_states_24h'

        if len(risks) < 3:
            return {
                'status': 'insufficient_data',
                'sample_count': len(risks),
                'trend': 'unknown',
                'mean_risk': None,
                'delta': None,
                'source': source,
            }

        mean_risk = sum(risks) / len(risks)
        delta = risks[-1] - risks[0]
        trend = 'up' if delta > 0.08 else ('down' if delta < -0.08 else 'flat')
        return {
            'status': 'ok',
            'sample_count': len(risks),
            'trend': trend,
            'mean_risk': round(mean_risk, 3),
            'delta': round(delta, 3),
            'source': source,
        }

    def _build_recent_summary(self, now_ts=None):
        now_ts = float(now_ts or time.time())
        horizon = 600.0
        recent = [x for x in self._risk_recent_states if (now_ts - float(x.get('ts', now_ts))) <= horizon]
        if not recent:
            return {
                'window_sec': int(horizon),
                'sample_count': 0,
                'mean_risk': round(float(self.risk_display.text()) if hasattr(self, 'risk_display') else 0.0, 3),
                'trend_10m': 'flat',
                'top_levels': [],
            }
        risks = [float(x.get('risk', 0.0)) for x in recent]
        mean_risk = sum(risks) / max(1, len(risks))
        delta = risks[-1] - risks[0]
        trend = 'up' if delta > 0.08 else ('down' if delta < -0.08 else 'flat')
        levels = {}
        for x in recent:
            lv = x.get('level', 'unknown')
            levels[lv] = levels.get(lv, 0) + 1
        top_levels = sorted(levels.items(), key=lambda kv: kv[1], reverse=True)
        return {
            'window_sec': int(horizon),
            'sample_count': len(recent),
            'mean_risk': round(mean_risk, 3),
            'trend_10m': trend,
            'top_levels': top_levels[:3],
        }

    def _get_modal_weights_for_llm(self):
        """与神经网络 modal_w 对齐：(vision, hrv, eeg)。"""
        default = (0.35, 0.40, 0.25)
        try:
            nn_out = RiskBus.instance().latest()
            if nn_out and nn_out.get('modal_w'):
                mw = nn_out['modal_w']
                if len(mw) >= 3:
                    return (round(float(mw[0]), 3), round(float(mw[1]), 3), round(float(mw[2]), 3))
        except Exception:
            pass
        try:
            lm = getattr(self, '_latest_modal', None)
            if lm and len(lm) >= 3:
                s = sum(max(0.0, float(x)) for x in lm[:3]) or 1.0
                return (
                    round(float(lm[0]) / s, 3),
                    round(float(lm[1]) / s, 3),
                    round(float(lm[2]) / s, 3),
                )
        except Exception:
            pass
        return default

    def _collect_eeg_features_for_llm(self):
        panel = getattr(self, 'bci_panel', None) or getattr(self, 'page_bci', None)
        feats = {
            'attention': None,
            'meditation': None,
            'poor_signal': None,
            'source': 'unknown',
            'factor_summary': RiskBus.instance().get_modal_factor('eeg'),
        }
        if panel is not None:
            feats['attention'] = int(getattr(panel, 'attention', 0))
            feats['meditation'] = int(getattr(panel, 'meditation', 0))
            feats['poor_signal'] = int(getattr(panel, 'poor_signal', 200))
            if time.time() < float(getattr(self, '_remote_eeg_live_until', 0.0)):
                feats['source'] = 'remote'
            else:
                src = str(getattr(panel, 'pill_source', None) and panel.pill_source.text() or '')
                if '模拟' in src:
                    feats['source'] = 'sim'
                elif '串口' in src or 'TGAM' in src:
                    feats['source'] = 'serial'
                else:
                    feats['source'] = 'local'
        att = feats.get('attention')
        med = feats.get('meditation')
        if att is not None and med is not None:
            feats['eeg_risk_proxy'] = round(
                max(0.0, min(1.0, (100 - int(att)) / 100.0 * 0.6 + (100 - int(med)) / 100.0 * 0.4)),
                3,
            )
        else:
            feats['eeg_risk_proxy'] = 0.0
        return feats

    def _build_llm_context_payload(self, current_risk):
        risk = float(current_risk)
        level = self._risk_level_text(risk)
        vis = dict(self.latest_visual_snapshot or {})
        phy = dict(self.latest_physio_snapshot or {})
        now_ts = time.time()
        recent_summary = self._build_recent_summary(now_ts)
        trend_24h = self._build_trend_24h(now_ts)
        events = [e for e in self._recent_anomaly_events][-5:]

        w_vis, w_hrv, w_eeg = self._get_modal_weights_for_llm()
        eeg_feats = self._collect_eeg_features_for_llm()

        visual_risk = float(vis.get('visual_risk', 0.0) or 0.0)
        if visual_risk <= 0 and hasattr(self, 'metric_values') and 'vis' in self.metric_values:
            try:
                visual_risk = float((self.metric_values.get('vis').text() or '0').strip() or 0)
            except Exception:
                pass

        hrv_rmssd = phy.get('hrv_rmssd')
        hrv_risk = max(0.0, min(1.0, 1.0 - (float(hrv_rmssd) / 80.0))) if hrv_rmssd is not None else 0.0
        if hasattr(self, 'metric_values') and 'hrv' in self.metric_values:
            try:
                hrv_risk = float((self.metric_values.get('hrv').text() or '').strip() or hrv_risk)
            except Exception:
                pass

        eeg_risk = float(eeg_feats.get('eeg_risk_proxy', 0.0))
        if eeg_risk <= 0 and hasattr(self, 'metric_values') and 'eeg' in self.metric_values:
            try:
                eeg_text = (self.metric_values.get('eeg').text() or '').strip()
                eeg_risk = float(eeg_text) if eeg_text else 0.0
            except Exception:
                pass

        nn_latest = None
        try:
            nn_latest = RiskBus.instance().latest()
        except Exception:
            pass

        vis_features = {k: vis.get(k) for k in (
            'expr', 'expr_prob', 'eye_fatigue_index', 'posture_risk',
            'forward_head_risk', 'down_head_risk', 'visual_risk')}
        vis_features['factor_summary'] = RiskBus.instance().get_modal_factor('vision')

        phy_features = {k: phy.get(k) for k in (
            'bpm', 'hrv_rmssd', 'sdnn', 'lfhf', 'rr_valid_ratio', 'ppg_quality', 'fs')}
        phy_features['factor_summary'] = RiskBus.instance().get_modal_factor('hrv')

        return {
            'schema_version': 'mindroom_llm_ctx_v1',
            'risk': {
                'current': round(risk, 3),
                'level': level,
                'trend_24h': trend_24h,
                'trend_10m': recent_summary.get('trend_10m', 'flat'),
                'nn_risk_score': round(float(nn_latest.get('risk_score', risk)), 3) if nn_latest else None,
                'nn_tier': int(nn_latest.get('tier', 0)) if nn_latest else None,
            },
            'modalities': {
                'visual': {
                    'score': round(visual_risk, 3),
                    'weight': w_vis,
                    'features': vis_features,
                },
                'hrv': {
                    'score': round(hrv_risk, 3),
                    'weight': w_hrv,
                    'features': phy_features,
                },
                'eeg': {
                    'score': round(eeg_risk, 3),
                    'weight': w_eeg,
                    'features': eeg_feats,
                },
            },
            'anomalies': events,
            'recent_state_memory': recent_summary,
            'data_source_mode': getattr(self, '_data_source_mode', 'offline_replay'),
            'output_target': getattr(self, '_llm_target', 'student_teacher'),
            'llm_mode': getattr(self, '_llm_mode', 'realtime'),
            'constraints': {
                'non_clinical_only': True,
                'no_medication': True,
            },
        }

    def _read_current_risk_for_ai(self):
        try:
            nn_out = RiskBus.instance().latest()
            if nn_out and nn_out.get('risk_score') is not None:
                return float(nn_out['risk_score'])
        except Exception:
            pass
        try:
            return float(self.risk_display.text())
        except Exception:
            return 0.5

    def _on_llm_target_changed(self, index):
        mapping = {
            0: 'student_teacher',
            1: 'student',
            2: 'parent',
            3: 'teacher',
        }
        self._llm_target = mapping.get(int(index), 'student_teacher')

    def _trigger_ai_intervention_manual(self, deep=False):
        if self.is_ai_generating:
            self._append_terminal_log('> [AI] 正在生成中，请稍候。', dedup=False)
            return
        now = time.time()
        if now < self.ai_cooldown_until:
            remain = int(self.ai_cooldown_until - now)
            self._append_terminal_log(f'> [AI] 冷却中，剩余 {remain}s（可点「我已处理」解除）', dedup=False)
            return
        prev_mode = self._llm_mode
        if deep:
            self._llm_mode = 'deep_analysis'
        try:
            risk = self._read_current_risk_for_ai()
            self._trigger_ai_intervention(risk)
        finally:
            if deep:
                self._llm_mode = prev_mode

    def _trigger_ai_intervention(self, current_risk):
        self.is_ai_generating = True
        self.last_ai_time = time.time()
        self.ai_console.setStyleSheet("QTextEdit { background: #1a0505; color: #ff007f; border: 1px solid #ff007f; border-radius: 8px; font-family: '微软雅黑'; font-size: 14px; padding: 10px; }")
        self.ai_console.setText(
            f"> ⚠️ 警告：检测到持续偏高的风险指数 ({current_risk:.2f})\n"
            f"> 正在生成 AI 辅助参考建议（不构成诊断）...\n"
            f"> 🛡 守护人响应链路已就绪：⚙ 工具箱 → 🛡 守护人响应  "
            f"可查看 / 模拟通知家长、班主任、心理老师。")

        context_payload = self._build_llm_context_payload(current_risk)
        mode_label = '深度·glm-4' if str(self._llm_mode).lower() in ('deep', 'analysis', 'deep_analysis') else '实时·glm-4-flash'
        self.ai_console.append(
            f"> 推理模式: {mode_label}  ·  输出对象: {context_payload.get('output_target', 'student_teacher')}")
        self.ai_thread = AIInterventionThread(current_risk, context_payload=context_payload, mode=self._llm_mode)
        self.ai_thread.finished_signal.connect(self._on_ai_intervention_success)
        self.ai_thread.error_signal.connect(self._on_ai_intervention_success)
        self.ai_thread.start()

    def _on_ai_intervention_success(self, text):
        self.ai_console.append(f"\n> AI 辅助建议: {text}")
        self.ai_console.append(
            "> ⚠ 上述内容为 AI 辅助参考，不替代专业判断。"
            "守护人响应链路：⚙ 工具箱 → 🛡 守护人响应。")

        self.tts_thread = TTSThread(text)
        self.tts_thread.start()

        self.is_ai_generating = False
        self.ai_cooldown_until = time.time() + float(self.ai_cooldown_secs)
        self.ai_console.append(f"> AI 触发冷却已开启：{self.ai_cooldown_secs}s（可点“我已处理”提前恢复）")
        QTimer.singleShot(10000, self._reset_ai_console)

    def _reset_ai_console(self):
        if not self.is_ai_generating:
            self.ai_console.setStyleSheet("QTextEdit { background: #0a0e27; color: #00d4ff; border: 1px solid #2a3270; border-radius: 8px; font-family: '微软雅黑'; font-size: 14px; padding: 10px; }")
            self.ai_console.setText("> 系统运行正常。AI 正在后台静默监测特征数据...")

    def _update_dashboard_data(self, vis_r, hrv_r, eeg_r, eye_fatigue, posture_risk, *args):
        hrv_from_remote = self._bpm_to_hrv_risk(self.latest_remote_bpm) if self.latest_remote_bpm is not None else hrv_r

        eye_r = max(0.0, min(1.0, float(eye_fatigue)))
        post_r = max(0.0, min(1.0, float(posture_risk)))

        # 新 RadarWidget 用 set_values；保留 .vis/.hrv/.eeg/.eye/.post 兼容
        try:
            self.radar_widget.set_values(
                expr=max(0.0, min(1.0, vis_r)),
                hr=max(0.0, min(1.0, hrv_from_remote)),
                hrv=max(0.0, min(1.0, hrv_from_remote)),
                eye=eye_r,
                eeg=max(0.0, min(1.0, eeg_r)))
        except Exception:
            pass
        self.radar_widget.vis = max(0.0, min(1.0, vis_r))
        self.radar_widget.hrv = max(0.0, min(1.0, hrv_from_remote))
        self.radar_widget.eeg = max(0.0, min(1.0, eeg_r))
        self.radar_widget.eye = eye_r
        self.radar_widget.post = post_r
        self.radar_widget.update()
        # BigStat 眼疲劳 / 姿态风险（中列大数 2x2）
        try:
            self.stat_eye_big.set_value(f'{int(eye_r*100)}', C_VISION if eye_r < 0.6 else C_WARN)
            self.stat_post_big.set_value(f'{int(post_r*100)}', C_OK if post_r < 0.5 else C_WARN)
        except Exception:
            pass

        self.metric_values["vis"].setText(f"{vis_r:.2f}")
        self.metric_values["hrv"].setText(f"{hrv_from_remote:.2f}")
        self.metric_values["eeg"].setText(f"{eeg_r:.2f}")
        self.metric_values["eye"].setText(f"{eye_r:.2f}")
        self.metric_values["post"].setText(f"{post_r:.2f}")
        # 视觉风险标签 (dashboard_modules_v2 新增)
        try:
            self.lbl_vis_risk.setText(f'{vis_r:.2f}')
            if vis_r > 0.66: vc, vt = C_RISK, 'HIGH'
            elif vis_r > 0.33: vc, vt = C_WARN, 'MED'
            else: vc, vt = C_OK, 'LOW'
            self.lbl_vis_risk.setStyleSheet(
                f'color:{vc}; font-size:20px; font-weight:bold; '
                f'background:{C_BG_DEEP}; border:1px solid {C_BORDER}; '
                f'border-radius:5px; padding:4px 10px; font-family:Consolas;')
            self.lbl_vis_tag.setText(vt)
            self.lbl_vis_tag.setStyleSheet(
                f'color:{vc}; font-size:10px; letter-spacing:2px; background:transparent;')
        except Exception:
            pass
        try:
            self.bar_eye._bar.setValue(int(eye_r*100)); self.bar_eye._val.setText(f'{int(eye_r*100)}%')
            self.bar_posture._bar.setValue(int(post_r*100)); self.bar_posture._val.setText(f'{int(post_r*100)}%')
        except Exception:
            pass
        # 缓存最新指标，统一在 1Hz NN tick 里画时序与堆叠条（避免 30 FPS 抖动）
        self._latest_modal = (vis_r, hrv_from_remote, eeg_r)

    def _connect_signals(self):
        self.camera_thread.change_pixmap_signal.connect(self._update_image)
        self.camera_thread.realtime_risk_signal.connect(self._update_risk)
        self.camera_thread.multimodal_data_signal.connect(self._update_dashboard_data)
        self.camera_thread.record_risk_signal.connect(self._plot_history)
        self.camera_thread.battery_signal.connect(self.social_battery.set_level)
        self.btn_rec.clicked.connect(self._toggle_rec)
        self.btn_flow_shield.clicked.connect(self._toggle_flow_firewall)
        self.chk_fw_enforce.toggled.connect(self._on_flow_enforce_changed)
        self.chk_fw_schedule.toggled.connect(self._on_flow_schedule_changed)
        self.time_fw_start.timeChanged.connect(self._on_flow_schedule_time_changed)
        self.time_fw_end.timeChanged.connect(self._on_flow_schedule_time_changed)
        self.btn_apply_fw_list.clicked.connect(self._apply_flow_process_list)
        self.btn_zen_focus.clicked.connect(self._toggle_zen_focus)
        self.btn_shadow_mode.clicked.connect(self._trigger_shadow_mode)

    def _is_in_flow_schedule_window(self):
        now_t = QTime.currentTime()
        start_t = self.flow_schedule_start
        end_t = self.flow_schedule_end

        if start_t <= end_t:
            return start_t <= now_t <= end_t
        return (now_t >= start_t) or (now_t <= end_t)

    def _effective_flow_enforce(self):
        manual = bool(self.flow_firewall_enforce)
        if not self.flow_schedule_enabled:
            return manual
        return manual and self._is_in_flow_schedule_window()

    def _on_flow_enforce_changed(self, checked):
        self.flow_firewall_enforce = bool(checked)
        effective = self._effective_flow_enforce()
        if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
            self.flow_firewall_thread.set_enforce_block(effective)
        mode_text = "强拦截" if effective else "提醒"
        self.firewall_log.append(f"> [FLOW-FW] 模式切换: {mode_text}模式")

    def _on_flow_schedule_changed(self, checked):
        self.flow_schedule_enabled = bool(checked)
        effective = self._effective_flow_enforce()
        if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
            self.flow_firewall_thread.set_enforce_block(effective)
        self.firewall_log.append(
            f"> [FLOW-FW] 时间窗策略: {'启用' if self.flow_schedule_enabled else '关闭'}"
        )

    def _on_flow_schedule_time_changed(self):
        self.flow_schedule_start = self.time_fw_start.time()
        self.flow_schedule_end = self.time_fw_end.time()
        effective = self._effective_flow_enforce()
        if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
            self.flow_firewall_thread.set_enforce_block(effective)
        self.firewall_log.append(
            f"> [FLOW-FW] 时间窗更新: {self.flow_schedule_start.toString('HH:mm')} - {self.flow_schedule_end.toString('HH:mm')}"
        )

    def _apply_flow_process_list(self):
        text = self.edit_fw_processes.text().strip()
        items = [x.strip() for x in text.replace('，', ',').split(',') if x.strip()]
        if not items:
            self.firewall_log.append("> [FLOW-FW] 黑名单为空，保持原配置不变。")
            self.edit_fw_processes.setText(", ".join(self.flow_block_processes))
            return

        normalized = []
        for p in items:
            if not p.lower().endswith('.exe'):
                p = f"{p}.exe"
            normalized.append(p)
        self.flow_block_processes = sorted(set(normalized), key=lambda s: s.lower())
        self.edit_fw_processes.setText(", ".join(self.flow_block_processes))

        if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
            self.flow_firewall_thread.set_block_processes(self.flow_block_processes)
        self.firewall_log.append(f"> [FLOW-FW] 黑名单已更新: {', '.join(self.flow_block_processes)}")

    def _toggle_flow_firewall(self):
        self.flow_firewall_active = not self.flow_firewall_active

        if self.flow_firewall_active:
            self.btn_flow_shield.setText("✅ 绝对心流防御屏障运行中")
            self.btn_flow_shield.setStyleSheet("""
                QPushButton {
                    background-color: #083344;
                    color: #67e8f9;
                    border: 2px solid #22d3ee;
                    border-radius: 12px;
                    font-size: 24px;
                    font-weight: bold;
                    letter-spacing: 2px;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #155e75;
                    color: #ecfeff;
                    border: 2px solid #67e8f9;
                }
            """)

            # 每次激活都创建新线程，避免停用后 _running 状态残留
            if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
                self.flow_firewall_thread.stop()
                self.flow_firewall_thread.wait(800)

            self.flow_firewall_thread = FlowFirewallThread(
                block_processes=self.flow_block_processes,
                interval_ms=2500,
                enforce_block=self._effective_flow_enforce(),
            )
            self.flow_firewall_thread.log_signal.connect(self.firewall_log.append)
            self.flow_firewall_thread.start()

            logs = [
                "> [FLOW-FW] 防火墙已激活：开始监测干扰进程。",
                f"> [FLOW-FW] 当前模式：{'强拦截' if self._effective_flow_enforce() else '提醒'}。",
                f"> [FLOW-FW] 时间窗：{'启用' if self.flow_schedule_enabled else '关闭'} ({self.flow_schedule_start.toString('HH:mm')}-{self.flow_schedule_end.toString('HH:mm')})",
                "> [FLOW-FW] 审计日志已开启。",
            ]
        else:
            self.btn_flow_shield.setText("🛡️ 激活绝对心流防御屏障")
            self.btn_flow_shield.setStyleSheet("""
                QPushButton {
                    background-color: #0a1324;
                    color: #00d4ff;
                    border: 2px solid #00d4ff;
                    border-radius: 12px;
                    font-size: 24px;
                    font-weight: bold;
                    letter-spacing: 2px;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #00d4ff;
                    color: #0a0e27;
                    border: 2px solid #7df9ff;
                    padding-top: 6px;
                    padding-bottom: 10px;
                }
                QPushButton:pressed {
                    background-color: #08243a;
                    color: #9ee8ff;
                    border: 2px solid #22d3ee;
                    padding-top: 12px;
                    padding-bottom: 4px;
                }
            """)

            if self.flow_firewall_thread is not None and self.flow_firewall_thread.isRunning():
                self.flow_firewall_thread.stop()
                self.flow_firewall_thread.wait(1200)

            logs = [
                "> [FLOW-FW] 收到停用指令，开始回收防御规则...",
                "> [FLOW-FW] 进程监测已停止。",
                "> [FLOW-FW] 绝对心流防御屏障已解除。",
            ]

        for line in logs:
            self.firewall_log.append(line)

    def _toggle_zen_focus(self):
        if not hasattr(self, '_zen_focus_running'):
            self._zen_focus_running = False

        self._zen_focus_running = not self._zen_focus_running
        if self._zen_focus_running:
            self.btn_zen_focus.setText("⏸️ 暂停专注计时")
            self.zen_log.append("> [ZEN] 已开始 25 分钟专注计时。")
            self.zen_log.append("> [ZEN] 提示：保持单任务，不处理即时消息。")
        else:
            self.btn_zen_focus.setText("🎯 开始 25 分钟专注计时")
            self.zen_log.append("> [ZEN] 专注计时已暂停。")

    def _trigger_shadow_mode(self):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            risk_value = float(self.risk_display.text())
        except Exception:
            risk_value = 0.0

        case_payload = {
            "timestamp": ts,
            "risk": risk_value,
            "visual": dict(self.latest_visual_snapshot),
            "physio": dict(self.latest_physio_snapshot),
        }

        out_path = os.path.join(os.path.dirname(__file__), "shadow_cases.json")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(case_payload, ensure_ascii=False) + "\n")

        self.ai_console.append("> 影子模式：当前多模态状态已打包隔离，用于未来模型微调...")

    def _register_thread(self, name, thread_obj, stop_timeout_ms=1200):
        if thread_obj is None:
            return
        self._managed_threads[name] = {
            'thread': thread_obj,
            'timeout_ms': int(stop_timeout_ms),
        }

    def _stop_all_threads(self):
        for item in self._managed_threads.values():
            th = item.get('thread')
            if th is None:
                continue
            try:
                if hasattr(th, 'isRunning') and th.isRunning() and hasattr(th, 'stop'):
                    th.stop()
            except Exception:
                pass
        for item in self._managed_threads.values():
            th = item.get('thread')
            if th is None:
                continue
            try:
                if hasattr(th, 'isRunning') and th.isRunning() and hasattr(th, 'wait'):
                    th.wait(int(item.get('timeout_ms', 1200)))
            except Exception:
                pass

    def _start_remote_bpm_monitor(self):
        self._local_physio_live_until = 0.0
        if bool(getattr(config, "PHYSIO_LOCAL_SERIAL", True)):
            self.remote_bpm_thread = None
            self._append_terminal_log(
                "> [HR] 本机直连模式：心率走 PhysioThread（1.py），不轮询 esp32_receiver",
                dedup=False,
            )
            return
        url = getattr(config, 'ESP32_REMOTE_URL', 'http://127.0.0.1:5001/esp32/latest')
        interval = int(getattr(config, 'ESP32_REMOTE_INTERVAL_MS', 1000))
        self.remote_bpm_thread = RemoteBPMThread(url=url, interval_ms=interval)
        self.remote_bpm_thread.bpm_signal.connect(self._on_remote_bpm)
        self.remote_bpm_thread.telemetry_signal.connect(self._on_remote_physio)
        self.remote_bpm_thread.status_signal.connect(self._append_terminal_log)
        self.remote_bpm_thread.start()
        self._register_thread('remote_bpm', self.remote_bpm_thread, stop_timeout_ms=1000)

    def _set_data_source_mode(self, mode, detail=''):
        mode = str(mode or '').lower()
        self._data_source_mode = mode
        if mode == 'live':
            txt, state = '数据源：实时模式', 'live'
        elif mode == 'demo':
            txt, state = '数据源：演示模式', 'warn'
        elif mode in ('offline', 'offline_replay'):
            txt, state = '数据源：离线回放', 'sim'
        else:
            txt, state = '数据源：未知', 'info'

        if detail:
            txt = f'{txt} · {detail}'

        if hasattr(self, 'pill_top_heart'):
            self.pill_top_heart.set_state(state, txt)
        if hasattr(self, 'pill_top_vision'):
            vis_state = 'live' if mode in ('live',) else 'sim'
            self.pill_top_vision.set_state(vis_state, '视觉  实时·Camera' if vis_state == 'live' else '视觉  本地·Camera')
        if hasattr(self, 'pill_top_brain'):
            brain_state = 'live' if mode in ('live',) else 'sim'
            brain_detail = '脑电  实时·TGAM/BLE' if brain_state == 'live' else '脑电  本地·Mock'
            self.pill_top_brain.set_state(brain_state, brain_detail)

    def _start_physio_monitor(self):
        if not bool(getattr(config, "PHYSIO_EMBED_THREAD", True)):
            self.physio_thread = None
            if bool(getattr(config, "PHYSIO_SHOW_1PY_WINDOW", False)):
                self._append_terminal_log(
                    "> [HR] 心率由 1.py 独立窗口采集（已关闭嵌入式 PhysioThread，避免抢串口）",
                    dedup=False,
                )
            return
        self.physio_thread = PhysioThread()
        self.physio_thread.features_signal.connect(self._on_physio_features)
        self.physio_thread.wave_signal.connect(self._on_physio_local_wave)
        self.physio_thread.status_signal.connect(self._append_terminal_log)
        self.physio_thread.source_signal.connect(self._on_physio_source)
        self.physio_thread.start()
        self._register_thread('physio', self.physio_thread, stop_timeout_ms=1000)

    def _launch_1py_hr_window(self):
        """弹出 1.py 完整 HRMonitorWindow（与 python 1.py 相同界面）。"""
        try:
            import importlib.util
            import os

            import pyqtgraph as pg

            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.py")
            spec = importlib.util.spec_from_file_location("hr_monitor_1py", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pg.setConfigOptions(antialias=True)
            win = mod.HRMonitorWindow()
            win.setWindowTitle("心率感知节点 · 1.py（学生端）")
            win.show()
            self._hr_1py_window = win
            self._append_terminal_log("> [HR] 已打开 1.py 完整窗口", dedup=False)
        except Exception as e:
            self._append_terminal_log(f"> [HR] 打开 1.py 窗口失败: {e}", dedup=False)

    def _start_eeg_monitor(self):
        """脑电：同机直连 TGAM 串口；双击脑电卡 = 11.py 全屏（与心率 1.py 一致）。不启 esp32_receiver。"""
        self._remote_eeg_live_until = 0.0
        self.eeg_thread = None
        self.bci_thread = None

        def _wire_bci_thread(th):
            th.eeg_signal.connect(self._on_eeg_packet)
            th.eeg_signal.connect(self.page_bci.update_eeg_packet)
            th.raw_wave_signal.connect(self.page_bci.update_raw_wave)
            th.status_signal.connect(self._append_terminal_log)
            th.status_signal.connect(self.page_bci.append_log)
            th.source_signal.connect(self._on_bci_source_changed)

        force_remote = bool(getattr(config, "BCI_FORCE_REMOTE", False))
        fallback_remote = bool(getattr(config, "BCI_FALLBACK_TO_REMOTE", False))
        embed_bci = bool(getattr(config, "BCI_EMBED_THREAD", True))

        if force_remote or fallback_remote:
            self.eeg_thread = RemoteEEGThread()
            self.eeg_thread.eeg_signal.connect(self._on_eeg_packet)
            self.eeg_thread.eeg_signal.connect(self.page_bci.update_eeg_packet)
            self.eeg_thread.status_signal.connect(self._append_terminal_log)
            self.eeg_thread.start()
            self._register_thread("remote_eeg", self.eeg_thread, stop_timeout_ms=1000)
            self._append_terminal_log("> [BCI] 远端 EEG 轮询已启用（HTTP）", dedup=False)
        else:
            self._append_terminal_log(
                "> [BCI] 本机直连模式：不轮询 esp32_receiver；双击脑电卡进入 11.py 读板子",
                dedup=False,
            )

        if embed_bci and getattr(config, "BCI_ENABLE", True) and not force_remote:
            self.bci_thread = BCIThread(
                port=getattr(config, "BCI_PORT", None),
                baud=getattr(config, "BCI_BAUD", 57600),
                prefer_serial=getattr(config, "BCI_PREFER_LOCAL_SERIAL", True),
                push_interval_ms=getattr(config, "BCI_PUSH_INTERVAL_MS", 800),
            )
            _wire_bci_thread(self.bci_thread)
            self.bci_thread.start()
            self._register_thread("bci", self.bci_thread, stop_timeout_ms=1500)
        elif not embed_bci:
            self._append_terminal_log(
                "> [BCI] 主界面未启 BCIThread（BCI_EMBED_THREAD=False），避免与 11.py 抢蓝牙串口",
                dedup=False,
            )
            if hasattr(self, "brain_card"):
                self.brain_card.status_pill.set_state("info", "设备: 双击进入11.py")
            if hasattr(self, "pill_top_brain"):
                self.pill_top_brain.set_state("info", "脑电  双击·11.py")

    def _on_bci_source_changed(self, source):
        self._append_terminal_log(f"> [BCI] 当前数据源切换为: {source}")
        if str(source) == 'sim' and time.time() >= float(getattr(self, '_remote_eeg_live_until', 0.0)):
            if hasattr(self, 'brain_card'):
                self.brain_card.status_pill.set_state('sim', '设备: 本地模拟')
            if hasattr(self, 'pill_top_brain'):
                self.pill_top_brain.set_state('sim', '脑电  本地·Mock')

    def _on_eeg_packet(self, pkt):
        if not isinstance(pkt, dict):
            return

        source = str(pkt.get("source") or "")
        sender = self.sender()
        remote_thread = getattr(self, "eeg_thread", None)
        bci_thread = getattr(self, "bci_thread", None)
        from_remote = remote_thread is not None and sender is remote_thread

        if from_remote:
            self._remote_eeg_live_until = time.time() + 3.0
        elif source == "sim" and time.time() < float(getattr(self, "_remote_eeg_live_until", 0.0)):
            return

        # 将 attention/meditation 映射为你现有算法可消费的 eeg_data
        att = int(pkt.get("attention", 50))
        poor = pkt.get("poor_signal")
        poor_limit = int(getattr(config, "EEG_POOR_SIGNAL_MAX", 99))
        if poor is not None and int(poor) > poor_limit:
            if not from_remote:
                self._append_terminal_log(f"> EEG信号较差，已忽略该帧: poor_signal={poor}")
            return
        med = int(pkt.get("meditation", 50))
        att = max(0, min(100, att))
        med = max(0, min(100, med))

        eeg_data = {
            "attention": att,
            "meditation": med,
            # 简单映射：专注↑风险↓，放松↑风险↓
            "alpha": max(0.0, med / 100.0),
            "beta": max(0.0, att / 100.0),
            "theta": max(0.0, 1.0 - att / 100.0),
        }
        if hasattr(self.camera_thread, "set_external_eeg_data"):
            self.camera_thread.set_external_eeg_data(eeg_data)

        if hasattr(self, 'brain_card'):
            if from_remote:
                self.brain_card.status_pill.set_state('live', '设备: 接收服务')
            elif source == 'sim':
                self.brain_card.status_pill.set_state('sim', '设备: 本地模拟')
            else:
                self.brain_card.status_pill.set_state('live', '设备: TGAM 本地')
        if from_remote and hasattr(self, 'pill_top_brain'):
            self.pill_top_brain.set_state('live', '脑电  远端·Wi-Fi')
        elif source == 'sim' and hasattr(self, 'pill_top_brain'):
            if time.time() >= float(getattr(self, '_remote_eeg_live_until', 0.0)):
                self.pill_top_brain.set_state('sim', '脑电  本地·Mock')
        if hasattr(self, 'page_bci') and hasattr(self.page_bci, 'update_raw_wave'):
            raw_values = pkt.get("raw_values")
            if isinstance(raw_values, list) and raw_values:
                try:
                    self.page_bci.update_raw_wave([int(v) for v in raw_values[-15000:] if v is not None])
                except Exception:
                    pass
            else:
                raw = pkt.get("raw_value")
                if raw is not None:
                    try:
                        cur = list(getattr(self.page_bci, 'raw_wave', []))[-999:]
                        cur.append(int(raw))
                        self.page_bci.update_raw_wave(cur)
                    except Exception:
                        pass

        poor = pkt.get("poor_signal")
        if hasattr(self, 'brain_core_att'):
            self.brain_core_att.setText(f'Attention: {att}')
        if hasattr(self, 'brain_core_med'):
            self.brain_core_med.setText(f'Meditation: {med}')
        if hasattr(self, 'brain_core_ps'):
            self.brain_core_ps.setText(f'PoorSignal: {poor if poor is not None else "--"}')
        if hasattr(self, 'brain_core_risk'):
            eeg_risk = max(0.0, min(1.0, (100 - att) / 100.0 * 0.6 + (100 - med) / 100.0 * 0.4))
            self.brain_core_risk.setText(f'EEG Risk: {eeg_risk:.2f}')

        if poor is not None:
            self._append_terminal_log(f"> EEG更新: attention={att}, meditation={med}, poor_signal={poor}")

        # ── 喂给 NN 特征对齐器 ──
        if getattr(self, 'aligner', None) is not None:
            ep = pkt.get('eeg_power') or {}
            bands8 = [
                float(ep.get('delta', 0)), float(ep.get('theta', 0)),
                float(ep.get('low_alpha', 0)), float(ep.get('high_alpha', 0)),
                float(ep.get('low_beta', 0)), float(ep.get('high_beta', 0)),
                float(ep.get('low_gamma', 0)), float(ep.get('mid_gamma', 0)),
            ]
            self.aligner.push_brain(
                bands8,
                float(att), float(med),
                1.0 if (poor is not None and int(poor) > 50) else 0.0,
            )

        # ── 喂给嵌入的 11.py 放大窗口 ──
        try:
            from embedded_node import _ACTIVE_OVERLAY as _eov
            ov = _eov
            if ov is not None:
                win = getattr(ov, '_embedded_win', None)
                if win is not None and hasattr(win, 'push_external_packet'):
                    win.push_external_packet(pkt)
        except Exception:
            pass

    def _start_mock_wave_engine(self):
        self.mock_wave_thread = MockWaveformThread(fs=50.0)

        # 统一路由：同一帧数据同时喂给“主界面迷你PPG + 波形矩阵四路”
        self.mock_wave_thread.data_signal.connect(self._route_mock_waves)
        # 无硬件联调：mock 直接驱动五维仪表盘（含 eye/post）
        self.mock_wave_thread.dashboard_signal.connect(self._update_dashboard_data)
        self.mock_wave_thread.start()
        self._register_thread('mock_wave', self.mock_wave_thread, stop_timeout_ms=1000)

    def _route_mock_waves(self, ppg, ecg, hr, hrv):
        if getattr(self, '_remote_physio_active_until', 0.0) > time.time():
            return
        if getattr(self, '_local_physio_live_until', 0.0) > time.time():
            return
        # 1) 主界面迷你PPG
        if hasattr(self, 'ppg_wave') and self.ppg_wave is not None:
            self.ppg_wave.add_data(ppg)

        # 2) 波形矩阵页四路（pyqtgraph 工业级版）
        self._push_wave_sample(ppg=ppg, ecg=ecg, hr=hr, hrv=hrv)

    def _refresh_ppg_curves_from_buffer(self):
        """按 PPG_Monitor 链刷新 PPG 曲线与 Y 轴。"""
        if not hasattr(self, '_buf_ppg'):
            return
        raw = list(self._buf_ppg)
        if len(raw) < 20:
            return
        display, metrics = ppg_signal.compute_ppg_display(raw, fs=100.0, buf_len=len(raw))
        disp = display.tolist() if hasattr(display, 'tolist') else list(display)
        self.curve_wave_ppg.setData(disp)
        if hasattr(self, 'curve_wave_ppg_glow'):
            self.curve_wave_ppg_glow.setData(disp)
        if len(disp) > 20:
            p_min, p_max = float(np.min(disp)), float(np.max(disp))
            p_span = max(1.0, p_max - p_min)
            self.p_ppg.setYRange(p_min - p_span * 0.12, p_max + p_span * 0.12, padding=0)
            if len(disp) % 10 == 0 and hasattr(self, 'ppg_peak_scatter'):
                peak_idx = [
                    i for i in range(2, len(disp) - 2)
                    if disp[i] > disp[i - 1] and disp[i] > disp[i + 1]
                    and disp[i] > (p_min + p_span * 0.75)
                ]
                peak_idx = peak_idx[-12:]
                self.ppg_peak_scatter.setData(peak_idx, [disp[i] for i in peak_idx])
        if metrics:
            if metrics.get('bpm'):
                self._hr_target = float(metrics['bpm'])
            rmssd = float(metrics.get('rmssd') or 0)
            sdnn = float(metrics.get('sdnn') or 0)
            if rmssd > 0:
                self._hrv_target = rmssd
            if sdnn > 0:
                self._sdnn_target = sdnn
            snap = dict(getattr(self, 'latest_physio_snapshot', None) or {})
            if rmssd > 0:
                snap['hrv_rmssd'] = rmssd
            if sdnn > 0:
                snap['sdnn'] = sdnn
            if metrics.get('lfhf'):
                snap['lfhf'] = float(metrics['lfhf'])
            self.latest_physio_snapshot = snap

    def _push_wave_sample(self, ppg=None, ecg=None, hr=None, hrv=None):
        """把单点样本推入波形矩阵页的四路 deque，并刷新 pyqtgraph 曲线。

        优化：添加限流，避免过于频繁的波形重绘导致UI卡顿。
        """
        if not hasattr(self, '_buf_ppg'):
            return

        # 限流：波形绘制最多30Hz（每33ms一次）
        now = time.time()
        if not hasattr(self, '_last_wave_draw'):
            self._last_wave_draw = 0

        # 始终更新数据缓冲区
        if ppg is not None:
            self._buf_ppg.append(float(ppg))
        if ecg is not None:
            self._buf_ecg.append(float(ecg))
        if hr is not None:
            self._buf_hr.append(float(hr))
        if hrv is not None:
            self._buf_hrv.append(float(hrv))

        # 但只在限流允许时才重绘UI
        if now - self._last_wave_draw < (self._ui_tick_fast_ms / 1000.0):
            return

        self._last_wave_draw = now

        # PPG：与 PPG_Monitor / 1.py 相同带通+显示缩放（不用原始 IR 直接画）
        if ppg is not None:
            self._refresh_ppg_curves_from_buffer()
        if ecg is not None:
            ecg_data = list(self._buf_ecg)
            self.curve_wave_ecg.setData(ecg_data)
            if hasattr(self, 'curve_wave_ecg_glow'):
                self.curve_wave_ecg_glow.setData(ecg_data)
            if len(ecg_data) > 20:
                e_min, e_max = min(ecg_data), max(ecg_data)
                e_span = max(0.1, e_max - e_min)
                self.p_ecg.setYRange(e_min - e_span * 0.18, e_max + e_span * 0.18, padding=0)
                # R 峰增强（简版）- 降低计算频率
                if len(ecg_data) % 10 == 0:
                    r_idx = [i for i in range(2, len(ecg_data)-2) if ecg_data[i] > ecg_data[i-1] and ecg_data[i] > ecg_data[i+1] and ecg_data[i] > (e_min + e_span * 0.78)]
                    r_idx = r_idx[-16:]
                    if hasattr(self, 'ecg_peak_scatter'):
                        self.ecg_peak_scatter.setData(r_idx, [ecg_data[i] for i in r_idx])
        if hr is not None and hr > 0:
            self._hr_target = float(hr)
        if hrv is not None and hrv > 0:
            self._hrv_target = float(hrv)

        # 渲染与到包解耦：无新包时也按固定节奏补帧推进，避免“停-跳”
        if len(self._buf_hr) > 0 and self._hr_target is not None:
            if self._hr_display_ema is None:
                self._hr_display_ema = float(self._hr_target)
            else:
                self._hr_display_ema = self._hr_display_ema * 0.80 + float(self._hr_target) * 0.20
            self._buf_hr.append(float(self._hr_display_ema))
            hr_draw = list(self._buf_hr)
            self.curve_wave_hr.setData(hr_draw)
            hr_show = float(self._hr_display_ema)
            hr_color = C_HR if 60 <= hr_show <= 90 else C_WARN
            self.card_hr_big.set_value(f'{hr_show:.0f}', hr_color)

        if len(self._buf_hrv) > 0 and self._hrv_target is not None:
            if self._hrv_display_ema is None:
                self._hrv_display_ema = float(self._hrv_target)
            else:
                self._hrv_display_ema = self._hrv_display_ema * 0.84 + float(self._hrv_target) * 0.16
            self._buf_hrv.append(float(self._hrv_display_ema))
            hrv_draw = list(self._buf_hrv)
            self.curve_wave_hrv.setData(hrv_draw)
            hrv_show = float(self._hrv_display_ema)
            hv_color = C_OK if hrv_show >= 30 else (C_WARN if hrv_show >= 15 else C_RISK)
            # 大数卡标题为 HRV(RMSSD)，与 row_rmssd 一致
            self.card_sdnn_big.set_value(f'{hrv_show:.1f}', hv_color)
            if hasattr(self, 'row_rmssd'):
                self.row_rmssd.update_value(
                    f'{hrv_show:.1f} ms',
                    '✓ 正常' if 15 <= hrv_show <= 39 else '⚠ 异常',
                    C_OK if 15 <= hrv_show <= 39 else C_WARN)

        # SDNN：与 HRV 曲线同一刷新节拍、同一 EMA 系数，避免“HRV 动、SDNN 不动”
        sdnn_tgt = getattr(self, '_sdnn_target', None)
        if sdnn_tgt is None or sdnn_tgt <= 0:
            try:
                snap_sd = (getattr(self, 'latest_physio_snapshot', {}) or {}).get('sdnn')
                if snap_sd is not None and float(snap_sd) > 0:
                    sdnn_tgt = float(snap_sd)
                    self._sdnn_target = sdnn_tgt
            except Exception:
                sdnn_tgt = None
        if sdnn_tgt is not None and sdnn_tgt > 0:
            if self._sdnn_display_ema is None:
                self._sdnn_display_ema = float(sdnn_tgt)
            else:
                self._sdnn_display_ema = (
                    self._sdnn_display_ema * 0.84 + float(sdnn_tgt) * 0.16
                )
            sdnn_show = float(self._sdnn_display_ema)
            if hasattr(self, 'row_sdnn'):
                self.row_sdnn.update_value(
                    f'{sdnn_show:.1f} ms',
                    '✓ 正常' if sdnn_show > 50 else '⚠ 偏低',
                    C_OK if sdnn_show > 50 else C_WARN)
            now_ts = time.time()
            if now_ts - float(getattr(self, '_last_sdnn_push_ts', 0.0)) >= 0.10:
                self._buf_sdnn_trend.append(sdnn_show)
                xs = [i * 0.1 for i in range(len(self._buf_sdnn_trend))]
                self.trend_curve.setData(xs, list(self._buf_sdnn_trend))
                self._last_sdnn_push_ts = now_ts

    def _append_wave_values(self, attr, values, limit):
        if not hasattr(self, attr) or not isinstance(values, list) or not values:
            return []
        cleaned = []
        for v in values[-limit:]:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    cleaned.append(fv)
            except Exception:
                pass
        if not cleaned:
            return []
        buf = getattr(self, attr)
        for fv in cleaned[-limit:]:
            buf.append(fv)
        return list(buf)

    def _apply_remote_hr_display(self, display):
        if not isinstance(display, dict):
            return False
        applied = False

        appended = self._append_wave_values('_buf_ppg', display.get('ppg'), 800)
        if appended:
            self._refresh_ppg_curves_from_buffer()
            applied = True

        ecg = self._append_wave_values('_buf_ecg', display.get('ecg'), 80)
        if ecg:
            self.curve_wave_ecg.setData(ecg)
            if hasattr(self, 'curve_wave_ecg_glow'):
                self.curve_wave_ecg_glow.setData(ecg)
            applied = True

        hr = self._append_wave_values('_buf_hr', display.get('hr'), 20)
        if hr:
            latest_hr = hr[-1]
            if latest_hr > 0:
                self._hr_target = float(latest_hr)
            applied = True

        hrv = self._append_wave_values('_buf_hrv', display.get('hrv'), 20)
        if hrv:
            latest_hrv = hrv[-1]
            if latest_hrv > 0:
                self._hrv_target = float(latest_hrv)
            applied = True

        # 对齐 1.py：不直接消费远端 sdnn_trend 快照，避免台阶/压平。
        # 学生端 trend 由本地显示层 hrv_show 固定时基推进。

        return applied

    def _update_remote_metric_rows(self, bpm, rmssd, sdnn, lfhf):
        if bpm is not None and bpm > 0 and hasattr(self, 'card_hr_big'):
            color = C_HR if 60 <= bpm <= 90 else C_WARN
            self.card_hr_big.set_value(f'{bpm:.0f}', color)

        sdnn_show = None
        if sdnn is not None and sdnn > 0:
            sdnn_show = float(sdnn)

        if sdnn_show is not None and hasattr(self, 'card_sdnn_big'):
            color = C_OK if sdnn_show > 50 else C_WARN
            self.card_sdnn_big.set_value(f'{sdnn_show:.1f}', color)

        if rmssd is not None and rmssd > 0 and hasattr(self, 'row_rmssd'):
            self.row_rmssd.update_value(
                f'{rmssd:.1f} ms',
                '✓ 正常' if 15 <= rmssd <= 39 else '⚠ 异常',
                C_OK if 15 <= rmssd <= 39 else C_WARN)

        if sdnn_show is not None and hasattr(self, 'row_sdnn'):
            self.row_sdnn.update_value(
                f'{sdnn_show:.1f} ms',
                '✓ 正常' if sdnn_show > 50 else '⚠ 偏低',
                C_OK if sdnn_show > 50 else C_WARN)

        if lfhf is not None and lfhf > 0 and hasattr(self, 'row_lfhf'):
            if 1 <= lfhf <= 2:
                self.row_lfhf.update_value(f'{lfhf:.2f}', '✓ 正常', C_OK)
            elif lfhf > 2:
                self.row_lfhf.update_value(f'{lfhf:.2f}', '⚠ 偏高', C_RISK)
            else:
                self.row_lfhf.update_value(f'{lfhf:.2f}', '⚠ 偏低', C_WARN)

    @staticmethod
    def _calc_hrv_from_ppg_like_1py(ppg_values, fs=100.0):
        """与 PPG_Monitor.py / 1.py 相同的带通、寻峰与 SDNN 计算。"""
        return ppg_signal.compute_ppg_metrics(ppg_values, fs=fs)

    def _append_remote_ir_to_buf(self, ir_values):
        """把 A 端每包 ir_values 写入 800 点滚动缓冲（与 1.py data_ppg 对齐）。"""
        if not isinstance(ir_values, list) or not ir_values:
            return
        if not hasattr(self, '_buf_ppg'):
            return
        for v in ir_values[-80:]:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    self._buf_ppg.append(fv)
            except Exception:
                pass

    def _merge_remote_ppg_metrics(self, payload, fs=100.0):
        """
        优先 A 端已算好的 sdnn/rmssd；否则用本地 800 点 IR 缓冲 + 滚动 RR（≥5）。
        不用「仅 80 点快照」重算，避免 peaks=1、rmssd=0 盖掉真值。
        """
        def _f(k, alt=None):
            for key in (k, alt) if alt else (k,):
                if key is None:
                    continue
                v = payload.get(key)
                if v is None:
                    continue
                try:
                    return float(v)
                except Exception:
                    pass
            return None

        bpm = _f('bpm')
        rmssd = _f('rmssd', 'hrv_rmssd')
        sdnn = _f('sdnn')
        lfhf = _f('lfhf')
        if rmssd is None:
            rmssd = _f('hrv')

        buf = list(getattr(self, '_buf_ppg', []) or [])
        buf_len = len(buf)

        for rr in ppg_signal.extract_rr_intervals_from_ppg(buf, fs=fs):
            self._remote_rr_buffer.append(rr)

        local = None
        if len(self._remote_rr_buffer) >= 5:
            local = ppg_signal.metrics_from_rr_buffer(list(self._remote_rr_buffer))
        if local is None and buf_len >= ppg_signal.VALID_COUNT_MIN:
            local = ppg_signal.compute_ppg_metrics(buf, fs=fs)

        if local:
            lb, lr, ls, ll = (
                local.get('bpm'),
                local.get('rmssd'),
                local.get('sdnn'),
                local.get('lfhf'),
            )
            if bpm is None or bpm <= 0:
                bpm = lb
            if (rmssd is None or rmssd <= 0) and lr and lr > 0:
                rmssd = lr
            if (sdnn is None or sdnn <= 0) and ls and ls > 0:
                sdnn = ls
            if (lfhf is None or lfhf <= 0) and ll and ll > 0:
                lfhf = ll

        return bpm, rmssd, sdnn, lfhf, buf_len

    def _debug_remote_ppg_metrics(self, ppg_values, display_ppg_len=0, metrics=None, fs=100.0, rr_buf_len=0):
        """远端 PPG 诊断：只输出学生端重算链路信息，不改变指标。"""
        now = time.time()
        if now - float(getattr(self, '_last_remote_ppg_debug_ts', 0.0)) < 2.0:
            return
        self._last_remote_ppg_debug_ts = now
        vals = []
        try:
            src = ppg_values
            if src is None or (isinstance(src, list) and len(src) < ppg_signal.VALID_COUNT_MIN):
                src = list(getattr(self, '_buf_ppg', []) or [])
            vals = [float(v) for v in (src or []) if v is not None and math.isfinite(float(v))]
        except Exception:
            vals = []
        peaks_count = 0
        rr_raw_count = 0
        rr_ok_count = 0
        rr_raw_mean = None
        rr_ok_mean = None
        try:
            if len(vals) >= ppg_signal.VALID_COUNT_MIN:
                _, filtered = ppg_signal.filter_ppg_tail(np.asarray(vals, dtype=np.float64), fs=fs)
                peaks = ppg_signal.find_ppg_peaks(filtered, fs=fs)
                peaks_count = int(len(peaks))
                if len(peaks) > 1:
                    rr_raw = np.diff(peaks) / float(fs)
                    rr_raw_count = int(len(rr_raw))
                    if len(rr_raw) > 0:
                        rr_raw_mean = float(np.mean(rr_raw))
                    rr_ok = rr_raw[(rr_raw > 0.3) & (rr_raw < 1.5)]
                    rr_ok_count = int(len(rr_ok))
                    if len(rr_ok) > 0:
                        rr_ok_mean = float(np.mean(rr_ok))
        except Exception as e:
            self._append_terminal_log(f"> [REMOTE_PPG] debug failed: {type(e).__name__}: {e}", dedup=True)
            return

        metric_text = "None"
        if metrics:
            metric_text = (
                f"bpm={float(metrics.get('bpm') or 0):.1f} "
                f"rmssd={float(metrics.get('rmssd') or 0):.1f} "
                f"sdnn={float(metrics.get('sdnn') or 0):.1f}"
            )
        raw_mean_text = f"{rr_raw_mean:.3f}s" if rr_raw_mean is not None else "--"
        ok_mean_text = f"{rr_ok_mean:.3f}s" if rr_ok_mean is not None else "--"
        line = (
            f"> [REMOTE_PPG] ir_len={len(vals)} buf={int(rr_buf_len or len(vals))} "
            f"display_ppg={int(display_ppg_len)} rr_buf={len(getattr(self, '_remote_rr_buffer', []))} "
            f"fs={float(fs):.0f} peaks={peaks_count} rr_raw={rr_raw_count} rr_ok={rr_ok_count} "
            f"rr_mean_raw={raw_mean_text} rr_mean_ok={ok_mean_text} metrics={metric_text}"
        )
        self._append_terminal_log(line, dedup=False)
        try:
            print(line, flush=True, file=sys.stdout)
        except Exception:
            pass

    def _on_physio_source(self, source):
        src = str(source or "")
        if src == "serial":
            self._local_physio_live_until = time.time() + 5.0
            self._set_data_source_mode("live", "ESP32 本地串口")
            if hasattr(self, "heart_card"):
                self.heart_card.status_pill.set_state("live", "信号: 本地串口")
            if hasattr(self, "pill_top_heart"):
                self.pill_top_heart.set_state("live", "心率  本地·1.py")
        elif src == "sim":
            if time.time() >= float(getattr(self, "_local_physio_live_until", 0.0)):
                if hasattr(self, "heart_card"):
                    self.heart_card.status_pill.set_state("sim", "信号: 本地模拟")
                if hasattr(self, "pill_top_heart"):
                    self.pill_top_heart.set_state("sim", "心率  本地·Mock")

    def _on_physio_local_wave(self, wave):
        if not isinstance(wave, dict):
            return
        self._local_physio_live_until = time.time() + 5.0
        ir_vals = wave.get("ir_values")
        if isinstance(ir_vals, list):
            self._append_remote_ir_to_buf(ir_vals)
        ppg = wave.get("ppg")
        hr = wave.get("hr")
        hrv = wave.get("hrv")
        ecg = wave.get("ecg")
        self._push_wave_sample(
            ppg=float(ppg) if ppg is not None else None,
            ecg=float(ecg) if ecg is not None else None,
            hr=float(hr) if hr is not None else None,
            hrv=float(hrv) if hrv is not None else None,
        )

    def _on_physio_features(self, feat):
        # 限流：避免过于频繁的UI更新导致卡顿
        now = time.time()
        if not hasattr(self, '_last_physio_update'):
            self._last_physio_update = 0

        # 按配置节奏限制指标刷新频率（默认10Hz）
        if now - self._last_physio_update < (self._ui_tick_medium_ms / 1000.0):
            # 只更新快照，不触发UI重绘
            self.latest_physio_snapshot = {
                "bpm": feat.get("bpm"),
                "hrv_rmssd": feat.get("hrv_rmssd"),
                "sdnn": feat.get("sdnn"),
                "lfhf": feat.get("lfhf"),
                "rr_valid_ratio": float(feat.get("rr_valid_ratio", 0.0)),
                "ppg_amp_p95": float(feat.get("ppg_amp_p95", 0.0)),
                "ppg_quality": feat.get("ppg_quality", "unknown"),
                "fs": float(feat.get("fs", 0.0)),
            }
            return

        self._last_physio_update = now
        self._local_physio_live_until = time.time() + 5.0

        self.latest_physio_snapshot = {
            "bpm": feat.get("bpm"),
            "hrv_rmssd": feat.get("hrv_rmssd"),
            "sdnn": feat.get("sdnn"),
            "lfhf": feat.get("lfhf"),
            "rr_valid_ratio": float(feat.get("rr_valid_ratio", 0.0)),
            "ppg_amp_p95": float(feat.get("ppg_amp_p95", 0.0)),
            "ppg_quality": feat.get("ppg_quality", "unknown"),
            "fs": float(feat.get("fs", 0.0)),
        }

        bpm = self.latest_physio_snapshot["bpm"]
        hrv = self.latest_physio_snapshot["hrv_rmssd"]
        rrv = self.latest_physio_snapshot["rr_valid_ratio"]
        qua = self.latest_physio_snapshot["ppg_quality"]
        fs = self.latest_physio_snapshot["fs"]

        bpm_text = f"{float(bpm):.1f}" if bpm is not None else "--"
        hrv_text = f"{float(hrv):.1f}" if hrv is not None else "--"
        self.physio_summary.setText(
            f"> 生理摘要(实时): fs={fs:.1f}Hz\n"
            f"> BPM: {bpm_text}  HRV(RMSSD): {hrv_text}\n"
            f"> RR有效率: {rrv:.2f}  质量: {qua}"
        )

        ppg_amp = float(feat.get("ppg_amp_p95", 0.0))
        if hasattr(self, 'ppg_wave'):
            self.ppg_wave.add_data(ppg_amp)

        sdnn = feat.get("sdnn")
        if hrv is not None and float(hrv) > 0:
            self._hrv_target = float(hrv)
        if sdnn is not None and float(sdnn) > 0:
            self._sdnn_target = float(sdnn)
        self._update_remote_metric_rows(
            float(bpm) if bpm is not None else None,
            float(hrv) if hrv is not None else None,
            float(sdnn) if sdnn is not None else None,
            float(feat.get("lfhf")) if feat.get("lfhf") else None,
        )

        local_live = time.time() < float(getattr(self, "_local_physio_live_until", 0.0) or 0.0)
        remote_active = time.time() < float(getattr(self, '_remote_physio_active_until', 0.0) or 0.0)
        if local_live or not remote_active:
            self._push_wave_sample(
                ppg=ppg_amp,
                hr=float(bpm) if bpm is not None else None,
                hrv=float(hrv) if hrv is not None else None,
            )

        # 信号源/质量胶囊
        if hasattr(self, 'waves_signal_pill'):
            if fs and fs > 1.0:
                self.waves_signal_pill.set_state('live', f'信号源：ESP32  ({fs:.0f}Hz)')
            else:
                self.waves_signal_pill.set_state('sim', '信号源：本地模拟')
        if hasattr(self, 'waves_quality_pill') and qua:
            q_state = {'good': 'live', 'fair': 'warn', 'poor': 'offline'}.get(str(qua).lower(), 'info')
            self.waves_quality_pill.set_state(q_state, f'质量：{qua}  RR有效率 {rrv:.0%}')

        if bpm is not None:
            self.latest_remote_bpm = float(bpm)
            self._append_terminal_log(
                f"> 生理线程更新: BPM={float(bpm):.1f}, HRV={hrv_text}, RR有效率={rrv:.2f}, 质量={qua}"
            )

        # 心力消耗预测曲线（历史100点 + 未来50点，轻量线性外推）
        hr_risk = self._bpm_to_hrv_risk(self.latest_remote_bpm if self.latest_remote_bpm is not None else bpm)
        hrv_risk = 0.0
        if hrv is not None:
            hrv_risk = max(0.0, min(1.0, 1.0 - float(hrv) / 80.0))
        drain = max(0.0, min(1.0, 0.65 * hr_risk + 0.35 * hrv_risk))

        self._pred_history.append(drain)
        if len(self._pred_history) > 100:
            self._pred_history = self._pred_history[-100:]

        future = []
        n = len(self._pred_history)
        if n >= 2:
            x = list(range(n))
            y = self._pred_history
            mx = sum(x) / n
            my = sum(y) / n
            den = sum((xi - mx) ** 2 for xi in x)
            if den > 1e-9:
                slope = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / den
            else:
                slope = 0.0

            # 预测起点使用最近5点均值，降低单点噪声
            base = sum(y[-min(5, n):]) / min(5, n)
            for i in range(1, 51):
                pred = base + slope * i
                future.append(max(0.0, min(1.0, pred)))

        if hasattr(self, 'pred_curve'):
            self.pred_curve.set_data(self._pred_history, future)

    def _on_remote_bpm(self, bpm):
        self.latest_remote_bpm = bpm
        self._set_data_source_mode('live', 'ESP32 / API')
        self._append_terminal_log(f"> 远端 ESP32 心率输入: {bpm:.1f} BPM")

    def _on_remote_physio(self, payload):
        if not isinstance(payload, dict):
            return

        def _float_or_none(v):
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        ir_value = _float_or_none(payload.get("ir_value"))
        hand_present = payload.get("hand_present")

        ir_values = payload.get("ir_values")
        self._append_remote_ir_to_buf(ir_values)

        display = payload.get("hr_display")
        display_ppg_len = 0
        if isinstance(display, dict):
            display_ppg = display.get("ppg")
            if isinstance(display_ppg, list):
                display_ppg_len = len(display_ppg)

        bpm, rmssd, sdnn, lfhf, buf_len = self._merge_remote_ppg_metrics(payload, fs=100.0)
        dbg_metrics = None
        if bpm is not None and bpm > 0:
            dbg_metrics = {
                "bpm": bpm,
                "rmssd": float(rmssd or 0),
                "sdnn": float(sdnn or 0),
                "lfhf": float(lfhf or 0),
            }
        self._debug_remote_ppg_metrics(
            list(getattr(self, '_buf_ppg', []) or []),
            display_ppg_len,
            dbg_metrics,
            fs=100.0,
            rr_buf_len=buf_len,
        )

        if bpm is None or bpm <= 0:
            return

        self.latest_remote_bpm = float(bpm)
        self._remote_physio_active_until = time.time() + 3.0
        self.latest_physio_snapshot = {
            "bpm": float(bpm),
            "hrv_rmssd": float(rmssd) if rmssd is not None else None,
            "rr_valid_ratio": 1.0 if hand_present in (None, 1, True, "1", "true", "True") else 0.0,
            "ppg_amp_p95": float(ir_value) if ir_value is not None else 0.0,
            "ppg_quality": "remote",
            "fs": 1.0,
            "sdnn": sdnn,
            "lfhf": lfhf,
        }

        self._set_data_source_mode('live', 'A端远程ESP32')
        if hasattr(self, 'heart_card'):
            self.heart_card.status_pill.set_state('live', '信号: 接收服务')
        display_applied = self._apply_remote_hr_display(payload.get("hr_display"))
        self._update_remote_metric_rows(
            float(bpm),
            float(rmssd) if rmssd is not None else None,
            float(sdnn) if sdnn is not None else None,
            float(lfhf) if lfhf is not None else None,
        )
        if rmssd is not None and rmssd > 0:
            self._hrv_target = float(rmssd)
        if sdnn is not None and sdnn > 0:
            self._sdnn_target = float(sdnn)
        if display_applied:
            self._push_wave_sample(
                ppg=None,
                hr=float(bpm),
                hrv=float(rmssd) if rmssd is not None and rmssd > 0 else None,
            )
            return
        try:
            self._refresh_ppg_curves_from_buffer()
        except Exception:
            pass
        self._push_wave_sample(
            ppg=None if isinstance(ir_values, list) and ir_values else (float(ir_value) if ir_value is not None else None),
            hr=float(bpm),
            hrv=float(rmssd) if rmssd is not None and rmssd > 0 else None,
        )

    def _bpm_to_hrv_risk(self, bpm):
        if bpm is None:
            return 0.0
        return max(0.0, min(1.0, abs(float(bpm) - 75.0) / 30.0))

    def _append_terminal_log(self, text, level='info', dedup=True):
        msg = str(text)
        now = time.time()
        if dedup:
            last_ts = self._log_last_emit.get(msg, 0.0)
            if (now - last_ts) < self._log_throttle_secs:
                return
            self._log_last_emit[msg] = now

        if str(level).lower() in ('alert', 'warn', 'warning', 'error', 'critical'):
            try:
                self.terminal_text.moveCursor(QTextCursor.Start)
                self.terminal_text.insertPlainText(msg + "\n")
                self.terminal_text.moveCursor(QTextCursor.End)
                return
            except Exception:
                pass

        self.terminal_text.append(msg)

    def _ack_ai_intervention(self):
        self.ai_cooldown_until = 0.0
        self.high_risk_counter = 0
        self.is_ai_generating = False
        self._append_terminal_log('> 已确认处理：AI 冷却解除，恢复正常监测。', level='info', dedup=False)
        self._reset_ai_console()

    def _open_latest_replay(self):
        try:
            files = sorted(self._replay_out_dir.glob('replay_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                self._append_terminal_log('> [REPLAY] 暂无可回放事件。', dedup=False)
                return
            target = files[0]
            os.startfile(str(target))
            self._append_terminal_log(f'> [REPLAY] 已打开: {target.name}', dedup=False)
        except Exception as exc:
            self._append_terminal_log(f'> [REPLAY] 打开失败: {exc}', level='warn', dedup=False)

    def _sync_record_button_state(self):
        try:
            on = bool(getattr(self.camera_thread, 'recording_enabled', False))
            if not hasattr(self, 'btn_record_toggle') or not self.btn_record_toggle:
                return
            if on:
                self.btn_record_toggle.setText('⏹ 停止采集')
                self.btn_record_toggle.setStyleSheet("QPushButton { background:transparent; color:#ff6b6b; border:1px solid #ff6b6b; border-radius:8px; padding:2px 8px; font-size:12px; font-weight:bold; } QPushButton:hover { background:#ff6b6b; color:#04111f; }")
            else:
                self.btn_record_toggle.setText('🔴 开始采集')
                self.btn_record_toggle.setStyleSheet("QPushButton { background:transparent; color:#f59e0b; border:1px solid #f59e0b; border-radius:8px; padding:2px 8px; font-size:12px; font-weight:bold; } QPushButton:hover { background:#f59e0b; color:#04111f; }")
        except Exception:
            pass

    def _toggle_recording(self):
        try:
            cur = bool(getattr(self.camera_thread, 'recording_enabled', False))
            self.camera_thread.recording_enabled = not cur
            if self.camera_thread.recording_enabled:
                self.camera_thread.force_record_once = True
                self._append_terminal_log('> [REC] 已开启持久化采集（10秒/条）。', dedup=False)
            else:
                self._append_terminal_log('> [REC] 已停止持久化采集。', dedup=False)
            self._sync_record_button_state()
        except Exception as exc:
            self._append_terminal_log(f'> [REC] 切换失败: {exc}', level='warn', dedup=False)

    def _export_plain_risk_csv(self):
        try:
            if not hasattr(self, 'recorder') or self.recorder is None:
                self._append_terminal_log('> [EXPORT] 导出失败：Recorder 不可用。', level='warn', dedup=False)
                return
            df = self.recorder.get_decrypted_history()
            if df is None or df.empty:
                self._append_terminal_log('> [EXPORT] 暂无可导出数据，请先开始采集。', dedup=False)
                return
            out_dir = Path(getattr(self.recorder, 'data_dir', Path(__file__).resolve().parent / 'data'))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / 'risk_log_plain.csv'
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            self._append_terminal_log(f'> [EXPORT] 已导出明文CSV: {out_path}', dedup=False)
            try:
                os.startfile(str(out_path))
            except Exception:
                pass
        except Exception as exc:
            self._append_terminal_log(f'> [EXPORT] 导出失败: {exc}', level='warn', dedup=False)

    def _export_train_dataset(self):
        try:
            root = Path(__file__).resolve().parent
            src = root / 'nn' / 'aligned_features.csv'
            if not src.exists():
                self._append_terminal_log('> [EXPORT] 未找到 nn/aligned_features.csv，请先运行一段时间再导出。', level='warn', dedup=False)
                return
            df = pd.read_csv(src)
            if df.empty:
                self._append_terminal_log('> [EXPORT] aligned_features.csv 为空。', level='warn', dedup=False)
                return

            # 自动识别 day/label 列
            day_col = None
            for c in ['day', 'Day', '日期', 'date']:
                if c in df.columns:
                    day_col = c
                    break
            label_col = None
            for c in ['label', 'risk_level', 'tier', 'Label']:
                if c in df.columns:
                    label_col = c
                    break

            if day_col is None:
                # 无 day 时按顺序伪造 7 天切分
                n = len(df)
                df['day'] = (pd.Series(range(n)) * 7 // max(1, n)).clip(upper=6) + 1
                day_col = 'day'

            if label_col is None:
                # 无标签时用 risk_score 弱标注
                rc = None
                for c in ['risk', 'risk_score', 'total_risk']:
                    if c in df.columns:
                        rc = c
                        break
                if rc is None:
                    # 兜底：从 recorder 解密历史构造最小训练集（仅 risk 特征）
                    hist = self.recorder.get_decrypted_history() if hasattr(self, 'recorder') else pd.DataFrame()
                    if hist is None or hist.empty or 'risk' not in hist.columns:
                        self._append_terminal_log('> [EXPORT] 缺少 label 与 risk 列，且无历史风险可兜底。', level='warn', dedup=False)
                        return
                    hist = hist.copy()
                    hist['timestamp'] = pd.to_datetime(hist['timestamp'], errors='coerce')
                    hist = hist.dropna(subset=['timestamp', 'risk'])
                    hist['day'] = hist['timestamp'].dt.dayofyear.rank(method='dense').astype(int)
                    hist['risk'] = pd.to_numeric(hist['risk'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
                    hist['label'] = pd.cut(hist['risk'], bins=[-1, 0.3, 0.6, 0.8, 2], labels=[0, 1, 2, 3]).astype(int)
                    # 补齐基础特征列，避免训练脚本缺列
                    for c in ['sad', 'neutral', 'happy', 'rmssd', 'sdnn', 'lfhf', 'attention', 'meditation']:
                        if c not in hist.columns:
                            hist[c] = 0.0
                    df = hist[['day', 'sad', 'neutral', 'happy', 'rmssd', 'sdnn', 'lfhf', 'attention', 'meditation', 'risk', 'label']]
                    day_col = 'day'
                    label_col = 'label'
                else:
                    s = pd.to_numeric(df[rc], errors='coerce').fillna(0.0)
                    df['label'] = pd.cut(s, bins=[-1, 0.3, 0.6, 0.8, 2], labels=[0, 1, 2, 3]).astype(int)
                    label_col = 'label'

            out_dir = root / 'data'
            out_dir.mkdir(parents=True, exist_ok=True)
            train = df[df[day_col].astype(int).isin([1, 2, 3, 4, 5])].copy()
            val = df[df[day_col].astype(int) == 6].copy()
            test = df[df[day_col].astype(int) == 7].copy()
            train.to_csv(out_dir / 'train.csv', index=False, encoding='utf-8-sig')
            val.to_csv(out_dir / 'val.csv', index=False, encoding='utf-8-sig')
            test.to_csv(out_dir / 'test.csv', index=False, encoding='utf-8-sig')

            self._append_terminal_log(f'> [EXPORT] 训练集导出完成: train={len(train)} val={len(val)} test={len(test)}', dedup=False)
            try:
                os.startfile(str(out_dir))
            except Exception:
                pass
        except Exception as exc:
            self._append_terminal_log(f'> [EXPORT] 训练集导出失败: {exc}', level='warn', dedup=False)

    def _toggle_rec(self):
        if "启动" in self.btn_rec.text():
            self.btn_rec.setText("🛑 终止运行并封存数据")
            self.btn_rec.setStyleSheet("QPushButton{background:#ff007f; color:white; border-radius:8px; font-weight:bold; font-size:24px; letter-spacing: 2px;}")
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                self.camera_thread.recording_enabled = True
                self.camera_thread.force_record_once = True
            self._append_terminal_log("> 加密监测已启动：开始写入风险记录。")
        else:
            self.btn_rec.setText("⚡ 启动全节点加密监测")
            self.btn_rec.setStyleSheet("QPushButton{background:#111827; color:#00d4ff; border:2px solid #00d4ff; border-radius:8px; font-weight:bold; font-size:24px; letter-spacing: 2px;}")
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                self.camera_thread.recording_enabled = False
            self._append_terminal_log("> 加密监测已停止：暂停写入风险记录。")

    def _plot_history(self, *args):
        if self.stacked_widget.currentIndex() != 1: return
        self.ax.clear()
        try:
            df = self.recorder.get_decrypted_history()
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.ax.plot(df['timestamp'], df['risk'], color='#00d4ff', lw=2)
                self.ax.fill_between(df['timestamp'], df['risk'], color='#00d4ff', alpha=0.15)
                self.ax.axhline(y=config.RISK_THRESHOLD_MEDIUM, color='#ff007f', linestyle='--', alpha=0.7)
                self.ax.set_ylim(0, 1.0)
                self.ax.grid(True, linestyle=':', alpha=0.3, color='#00d4ff')
                self.ax.tick_params(colors='#6b7280')
                for spine in self.ax.spines.values(): spine.set_color('#2a3270')
            self.canvas.draw()
        except: pass

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            if hasattr(self, 'bg') and self.bg is not None:
                self.bg.setGeometry(0, 0, self.width(), self.height())
        except Exception:
            pass

    def mousePressEvent(self, e):
        if self.isFullScreen():
            self.drag_pos = None
            return
        if e.button() == Qt.LeftButton and e.pos().y() < 50: self.drag_pos = e.globalPos() - self.pos(); e.accept()
        else: self.drag_pos = None

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self.drag_pos: self.move(e.globalPos() - self.drag_pos); e.accept()

    def closeEvent(self, e):
        try:
            self._stop_all_threads()
        except Exception:
            pass
        super().closeEvent(e)
