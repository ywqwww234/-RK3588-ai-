"""
三模态特征对齐器（1 Hz 节拍）。

输入：
  - 视觉  (1 Hz):  expression(7类 softmax) + eye_fatigue + posture_risk        (9 维)
  - 心率  (~0.2 Hz): bpm + hrv_rmssd + sdnn + lf_hf + rr_valid_ratio          (5 维)
  - 脑电  (1 Hz):  8 频段 log-power + Att + Med + PoorSignal_mask              (11 维)

输出：
  - 25 维 1 Hz 对齐向量 + 滚动写 CSV
  - 60 帧滑窗张量 [60,25] 供模型推理

用法（在 MainWindow.__init__ 末尾接入）：
    from nn.feature_aligner import FeatureAligner
    self.aligner = FeatureAligner(out_csv='nn/aligned_features.csv')
    self.aligner.start()
    # 然后在每路数据回调里 push:
    self.aligner.push_vision(expr_probs, eye, posture)
    self.aligner.push_physio(bpm, hrv, sdnn, lf_hf, rr_valid)
    self.aligner.push_brain(bands, att, med, poor_mask)
"""

import csv
import os
import time
from collections import deque
from threading import Lock

import numpy as np


VISION_DIM = 9
PHYSIO_DIM = 5
BRAIN_DIM = 11
TOTAL_DIM = VISION_DIM + PHYSIO_DIM + BRAIN_DIM   # 25
WINDOW_SEC = 60


HEADER = (
    [f'v_{i}' for i in range(VISION_DIM)] +
    ['p_bpm', 'p_rmssd', 'p_sdnn', 'p_lfhf', 'p_rrv'] +
    ['b_delta','b_theta','b_lalpha','b_halpha','b_lbeta','b_hbeta','b_lgamma','b_mgamma',
     'b_att','b_med','b_poor'] +
    ['ts']
)


class FeatureAligner:
    """1Hz 三模态对齐器（线程安全 push）。"""

    def __init__(self, out_csv='nn/aligned_features.csv', label_callable=None):
        self.out_csv = out_csv
        self._lock = Lock()
        self._latest_vision = np.zeros(VISION_DIM, dtype=np.float32)
        self._latest_physio = np.zeros(PHYSIO_DIM, dtype=np.float32)
        self._latest_brain = np.zeros(BRAIN_DIM, dtype=np.float32)
        self._window = deque(maxlen=WINDOW_SEC)
        self._csv_fp = None
        self._csv_w = None
        self._label_fn = label_callable  # 可选，给训练数据打标签
        self._running = False

    # -------- push 接口（线程安全） --------
    def push_vision(self, expr_probs, eye_fatigue, posture_risk):
        with self._lock:
            arr = np.zeros(VISION_DIM, dtype=np.float32)
            if expr_probs is not None and len(expr_probs) >= 7:
                arr[:7] = np.asarray(expr_probs[:7], dtype=np.float32)
            arr[7] = float(eye_fatigue or 0.0)
            arr[8] = float(posture_risk or 0.0)
            self._latest_vision = arr

    def push_physio(self, bpm, hrv_rmssd, sdnn, lf_hf, rr_valid):
        with self._lock:
            self._latest_physio = np.asarray(
                [bpm or 0, hrv_rmssd or 0, sdnn or 0, lf_hf or 0, rr_valid or 0],
                dtype=np.float32)

    def push_brain(self, bands8, att, med, poor_signal_mask):
        with self._lock:
            arr = np.zeros(BRAIN_DIM, dtype=np.float32)
            if bands8 is not None and len(bands8) >= 8:
                arr[:8] = np.log1p(np.asarray(bands8[:8], dtype=np.float32))
            arr[8] = float(att or 0)
            arr[9] = float(med or 0)
            arr[10] = float(poor_signal_mask or 0)
            self._latest_brain = arr

    # -------- 1Hz 节拍 --------
    def tick(self):
        """每秒调用一次：合并最新三路 → 25 维 → 推 window + 写 CSV。"""
        with self._lock:
            row = np.concatenate([self._latest_vision, self._latest_physio, self._latest_brain])
        self._window.append(row)
        if self._csv_w is not None:
            try:
                self._csv_w.writerow(list(row) + [time.time()])
                self._csv_fp.flush()
            except Exception:
                pass
        return row

    def get_window(self):
        """返回当前 60 帧滑窗 (right-padded with last row)。"""
        if len(self._window) == 0:
            return np.zeros((WINDOW_SEC, TOTAL_DIM), dtype=np.float32)
        if len(self._window) < WINDOW_SEC:
            pad = np.tile(self._window[0], (WINDOW_SEC - len(self._window), 1))
            return np.vstack([pad, np.array(self._window)])
        return np.array(self._window)

    def start(self):
        if self._running:
            return
        os.makedirs(os.path.dirname(self.out_csv) or '.', exist_ok=True)
        is_new = not os.path.isfile(self.out_csv)
        self._csv_fp = open(self.out_csv, 'a', newline='', encoding='utf-8')
        self._csv_w = csv.writer(self._csv_fp)
        if is_new:
            self._csv_w.writerow(HEADER)
            self._csv_fp.flush()
        self._running = True

    def close(self):
        if self._csv_fp is not None:
            try:
                self._csv_fp.close()
            except Exception:
                pass
        self._csv_fp = None
        self._csv_w = None
        self._running = False
