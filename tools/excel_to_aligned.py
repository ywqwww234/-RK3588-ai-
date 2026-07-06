"""
步骤 B-1：从 data/single_run/all_merged.csv 生成 nn/aligned_features.csv（25 维 + ts）

与 build_dataset.py 同源、同序，保证与 nn/labels.csv 行数一致。

用法:
  python tools/excel_to_aligned.py
  python tools/excel_to_aligned.py --max-rows 100000
  python tools/excel_to_aligned.py --verify-only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_PATH = os.path.join(ROOT, "data", "single_run", "all_merged.csv")
ALIGNED_PATH = os.path.join(ROOT, "nn", "aligned_features.csv")
LABELS_PATH = os.path.join(ROOT, "nn", "labels.csv")

# 与 nn/feature_aligner.py HEADER 一致（不含 ts 的 25 维特征列）
FEAT_COLS = (
    [f"v_{i}" for i in range(9)]
    + ["p_bpm", "p_rmssd", "p_sdnn", "p_lfhf", "p_rrv"]
    + [
        "b_delta",
        "b_theta",
        "b_lalpha",
        "b_halpha",
        "b_lbeta",
        "b_hbeta",
        "b_lgamma",
        "b_mgamma",
        "b_att",
        "b_med",
        "b_poor",
    ]
)
OUT_COLS = FEAT_COLS + ["ts"]

EXPR_LABELS = ("happy", "sad", "angry", "fear", "disgust", "surprise", "neutral")


def _num(s: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def row_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """将 all_merged 列映射为 25 维 + ts。"""
    n = len(df)
    vis_sad = _num(df["vis_sad"]) if "vis_sad" in df.columns else np.zeros(n, np.float32)
    vis_neu = _num(df["vis_neu"]) if "vis_neu" in df.columns else np.zeros(n, np.float32)
    vis_happy = _num(df["vis_happy"]) if "vis_happy" in df.columns else np.zeros(n, np.float32)

    # 7 类表情伪分布（与线上一致：happy/sad/neutral 为主，其余均分残差）
    v = np.zeros((n, 9), dtype=np.float32)
    v[:, 0] = np.clip(vis_happy, 0, 1)
    v[:, 1] = np.clip(vis_sad, 0, 1)
    v[:, 6] = np.clip(vis_neu, 0, 1)
    other = np.clip(1.0 - v[:, 0] - v[:, 1] - v[:, 6], 0, 1) / 5.0
    for j in range(2, 6):
        v[:, j] = other
    row_sum = v[:, :7].sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-6] = 1.0
    v[:, :7] /= row_sum
    v[:, 7] = np.clip(vis_sad * 0.6 + (1 - vis_happy) * 0.2, 0, 1)  # eye_fatigue 代理
    v[:, 8] = np.clip(vis_sad * 0.4 + vis_neu * 0.35, 0, 1)  # posture_risk 代理

    rmssd = _num(df["rmssd"]) if "rmssd" in df.columns else np.full(n, 40.0, np.float32)
    sdnn = _num(df["sdnn"]) if "sdnn" in df.columns else np.full(n, 50.0, np.float32)
    lf_hf = _num(df["lf_hf"]) if "lf_hf" in df.columns else np.full(n, 1.5, np.float32)
    att = _num(df["attention"], 50.0) if "attention" in df.columns else np.full(n, 50.0, np.float32)
    med = _num(df["meditation"], 50.0) if "meditation" in df.columns else np.full(n, 50.0, np.float32)

    if "p_bpm" in df.columns:
        bpm = _num(df["p_bpm"], 75.0)
    else:
        bpm = np.clip(110 - rmssd * 0.15, 55, 110).astype(np.float32)

    p = np.stack(
        [
            bpm,
            rmssd,
            sdnn,
            lf_hf,
            np.ones(n, dtype=np.float32),
        ],
        axis=1,
    )

    # 脑电频段：无原始功率时用 att/med 生成弱占位（训练可学权重）
    scale = (100.0 - att) * 1000 + 1
    b = np.zeros((n, 11), dtype=np.float32)
    b[:, 0] = scale * 0.35
    b[:, 1] = scale * 0.20
    b[:, 2] = scale * 0.10
    b[:, 3] = scale * 0.08
    b[:, 4] = scale * 0.07
    b[:, 5] = scale * 0.06
    b[:, 6] = scale * 0.05
    b[:, 7] = scale * 0.04
    b[:, 8] = att
    b[:, 9] = med
    b[:, 10] = 0.0

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts_unix = (ts.view("int64") // 10**9).astype(np.float64)
        ts_unix = np.where(np.isnan(ts_unix), np.arange(n, dtype=np.float64), ts_unix)
    else:
        t0 = time.time()
        ts_unix = t0 + np.arange(n, dtype=np.float64)

    out = np.hstack([v, p, b])
    data = {c: out[:, i] for i, c in enumerate(FEAT_COLS)}
    data["ts"] = ts_unix
    return pd.DataFrame(data)


def verify_lengths() -> bool:
    if not os.path.isfile(ALIGNED_PATH) or not os.path.isfile(LABELS_PATH):
        print("[VERIFY] 缺少 aligned_features.csv 或 labels.csv")
        return False
    na = sum(1 for _ in open(ALIGNED_PATH, "rb")) - 1
    nl = sum(1 for _ in open(LABELS_PATH, "rb")) - 1
    ok = na == nl
    print(f"[VERIFY] aligned 数据行={na:,}  labels 行={nl:,}  -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="仅转换前 N 行（0=全部）")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    if args.verify_only:
        sys.exit(0 if verify_lengths() else 1)

    if not os.path.isfile(MERGED_PATH):
        print(f"[ERROR] 请先运行步骤 A: {MERGED_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(ALIGNED_PATH), exist_ok=True)
    print(f"[INFO] 读取 {MERGED_PATH}")

    if args.max_rows and args.max_rows > 0:
        df = pd.read_csv(MERGED_PATH, nrows=args.max_rows)
        feat = row_to_features(df)
        feat.to_csv(ALIGNED_PATH, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {ALIGNED_PATH} rows={len(feat)} (max-rows={args.max_rows})")
    else:
        first = True
        total = 0
        for chunk in pd.read_csv(MERGED_PATH, chunksize=args.chunksize):
            feat = row_to_features(chunk)
            feat.to_csv(
                ALIGNED_PATH,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig",
            )
            total += len(feat)
            first = False
            print(f"[INFO] 已写入 {total:,} 行…")
        print(f"[SAVE] {ALIGNED_PATH} rows={total:,}")

    if not verify_lengths():
        print(
            "[WARN] 行数不一致：若用了 --max-rows，请对 labels 同样截断，"
            "或重新 python tools/build_dataset.py 后全量转换。"
        )
        sys.exit(1)
    print("[DONE] 步骤 B-1/2 完成，可运行: python -m nn.train_baseline")


if __name__ == "__main__":
    main()