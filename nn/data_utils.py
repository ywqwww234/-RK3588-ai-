"""
数据工具：标准化、时序切分、标签校验。

设计原则（避免时间泄漏）：
  - 时序切分：先按行号顺序切 train/val/test（默认 70/15/15）
  - Scaler 只在 train 集上 fit，val/test 复用同一 scaler
  - 60 帧滑窗在切分边界严格不跨段
  - 训练保存 scaler.json，推理加载同一 scaler

标签策略（禁止随机标签）：
  - 必须存在 labels.csv，且行数 >= aligned_features.csv
  - label 列取值 ∈ {0,1,2,3}（四档风险），可选第二列 risk_score ∈ [0,1] 用于回归头
  - 缺失/越界都报错
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

WIN = 60
FEAT_DIM = 25
N_CLASSES = 4


class DataError(RuntimeError):
    pass


@dataclass
class Scaler:
    """简易 z-score scaler（与 sklearn StandardScaler 等价，便于 JSON 持久化）。"""
    mean: list
    std: list

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        # x: [N, D]
        m = x.mean(axis=0)
        s = x.std(axis=0)
        s = np.where(s < 1e-6, 1.0, s)
        return cls(mean=m.tolist(), std=s.tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        m = np.asarray(self.mean, dtype=np.float32)
        s = np.asarray(self.std, dtype=np.float32)
        return ((x - m) / s).astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str) -> "Scaler":
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))


def _validate_labels(lab_df: pd.DataFrame, n_rows: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """校验 labels.csv。返回 (cls_labels, reg_targets|None)。"""
    if 'label' not in lab_df.columns:
        raise DataError("labels.csv 必须包含 'label' 列（取值 0/1/2/3）")
    if len(lab_df) < n_rows:
        raise DataError(
            f"labels.csv 行数 {len(lab_df)} 少于特征行数 {n_rows}，存在标签缺失")
    lab = lab_df['label'].values[:n_rows]
    if np.isnan(lab.astype(float)).any():
        raise DataError("labels.csv 'label' 列存在 NaN")
    lab = lab.astype(np.int64)
    if (lab < 0).any() or (lab >= N_CLASSES).any():
        bad = np.unique(lab[(lab < 0) | (lab >= N_CLASSES)])
        raise DataError(f"label 越界：{bad.tolist()} 不在 [0,{N_CLASSES - 1}]")
    reg = None
    if 'risk_score' in lab_df.columns:
        reg = lab_df['risk_score'].values[:n_rows].astype(np.float32)
        reg = np.clip(reg, 0.0, 1.0)
    return lab, reg


def load_raw(csv_path: str, label_csv: str, *, allow_missing_label: bool = False
             ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """读取原始特征 + 标签（严格校验，禁止随机标签）。

    Returns:
        feats:  [N, 25] float32
        lab:    [N]     int64
        reg:    [N]     float32 | None
    """
    if not os.path.isfile(csv_path):
        raise DataError(f"特征文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    feats = df.iloc[:, :FEAT_DIM].values.astype(np.float32)
    if feats.shape[1] != FEAT_DIM:
        raise DataError(f"特征维度异常: 期望 {FEAT_DIM}，实得 {feats.shape[1]}")

    if not os.path.isfile(label_csv):
        if allow_missing_label:
            return feats, np.zeros(len(feats), dtype=np.int64), None
        raise DataError(
            f"标签文件不存在: {label_csv}\n"
            "已禁止随机标签训练。请准备 labels.csv (列: label[, risk_score])。\n"
            "提示：可用 risk_calculator 的输出做半监督标注。"
        )
    lab_df = pd.read_csv(label_csv)
    lab, reg = _validate_labels(lab_df, len(feats))
    return feats, lab, reg


def windowize(feats: np.ndarray, lab: np.ndarray, reg: Optional[np.ndarray],
              win: int = WIN
              ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """切 60 帧滑窗。每个窗口的标签 = 窗口末帧的标签。"""
    if len(feats) <= win:
        raise DataError(f"样本不足以构成窗口（需要 > {win}，实得 {len(feats)}）")
    xs, ys, rs = [], [], []
    for i in range(win, len(feats)):
        xs.append(feats[i - win:i])
        ys.append(lab[i])
        if reg is not None:
            rs.append(reg[i])
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    r = np.asarray(rs, dtype=np.float32) if reg is not None else None
    return X, y, r


def time_split(n: int, ratios=(0.7, 0.15, 0.15)) -> Tuple[slice, slice, slice]:
    """按时间顺序切 train/val/test。避免时间泄漏。"""
    assert abs(sum(ratios) - 1.0) < 1e-6
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def prepare_datasets(csv_path: str, label_csv: str, scaler_out: str,
                     win: int = WIN, ratios=(0.7, 0.15, 0.15),
                     max_rows: int = 0):
    """完整数据准备流水线。

    1. 加载并严格校验
    2. 时序切分 train/val/test
    3. 在 train 上 fit Scaler，全集 transform
    4. 切 60 帧滑窗（窗口不跨段）
    5. 持久化 scaler.json

    max_rows>0 时只取前 N 行（省内存，适合 16GB 本）。
    """
    feats, lab, reg = load_raw(csv_path, label_csv)
    if max_rows and max_rows > 0 and len(feats) > max_rows:
        feats, lab = feats[:max_rows], lab[:max_rows]
        if reg is not None:
            reg = reg[:max_rows]
        print(f'[data] 子采样 max_rows={max_rows:,}（时序前段）')
    n = len(feats)
    s_tr, s_va, s_te = time_split(n, ratios)

    scaler = Scaler.fit(feats[s_tr])
    feats_n = scaler.transform(feats)
    scaler.save(scaler_out)

    def slice_pack(sl):
        f = feats_n[sl]
        l = lab[sl]
        r = reg[sl] if reg is not None else None
        return windowize(f, l, r, win=win)

    X_tr, y_tr, r_tr = slice_pack(s_tr)
    X_va, y_va, r_va = slice_pack(s_va)
    X_te, y_te, r_te = slice_pack(s_te)
    return {
        'train': (X_tr, y_tr, r_tr),
        'val':   (X_va, y_va, r_va),
        'test':  (X_te, y_te, r_te),
        'scaler': scaler,
        'n_total': n,
    }


def class_weights(y: np.ndarray, n_class: int = N_CLASSES) -> np.ndarray:
    """逆频率类权重（处理样本不平衡）。"""
    cnt = np.bincount(y, minlength=n_class).astype(np.float32)
    cnt = np.where(cnt < 1, 1.0, cnt)
    w = 1.0 / cnt
    w = w * n_class / w.sum()
    return w.astype(np.float32)


def derive_reg_from_label(lab: np.ndarray) -> np.ndarray:
    """没有 risk_score 标注时，从离散 label 派生回归目标（每档中点）。"""
    table = np.array([0.15, 0.45, 0.70, 0.90], dtype=np.float32)
    return table[lab]
