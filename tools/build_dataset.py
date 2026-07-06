"""
步骤 A：从 data/single_run 下 Excel 合并 → all_merged / train|val|test / nn/labels.csv

用法:
  python tools/build_dataset.py
  python tools/build_dataset.py --force-excel   # 忽略已有 nn/aligned_features.csv，强制读 Excel
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "single_run")
OUT_DIR = os.path.join(ROOT, "data", "single_run")
NN_DIR = os.path.join(ROOT, "nn")
ALIGNED_PATH = os.path.join(NN_DIR, "aligned_features.csv")

T1, T2, T3 = 0.30, 0.55, 0.80


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "时间": "timestamp",
        "timestamp": "timestamp",
        "time": "timestamp",
        "风险": "risk_score",
        "风险值": "risk_score",
        "综合风险": "risk_score",
        "risk": "risk_score",
        "risk_score": "risk_score",
        "score": "risk_score",
        "visual_risk": "visual_risk",
        "视觉风险": "visual_risk",
        "hrv_risk": "hrv_risk",
        "生理风险": "hrv_risk",
        "eeg_risk": "eeg_risk",
        "脑电风险": "eeg_risk",
        "悲伤视觉值": "vis_sad",
        "中性视觉值": "vis_neu",
        "愉悦视觉值": "vis_happy",
        "rmssd": "rmssd",
        "RMSSD": "rmssd",
        "sdnn": "sdnn",
        "SDNN": "sdnn",
        "lf/hf": "lf_hf",
        "LF/HF": "lf_hf",
        "注意力值": "attention",
        "冥想值": "meditation",
        "label": "label",
        "风险等级": "risk_tier",
        "风险级别": "risk_tier",
        "等级": "risk_tier",
    }

    normalized_cols = []
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "")
        mapped = rename_map.get(key, rename_map.get(str(c).strip(), str(c).strip()))
        normalized_cols.append(mapped)
    df.columns = normalized_cols
    return df


def to_label(r):
    if r < T1:
        return 0
    if r < T2:
        return 1
    if r < T3:
        return 2
    return 3


def parse_risk_tier_column(series: pd.Series) -> pd.Series:
    """Excel「风险等级」→ 0..3；无法解析为 NaN。

    注意：表内若是 1~4 档，必须先减 1。若先当 0~3 用，会把 1/2/3 误标成低/中/高且没有 0 档。
    """
    s = series.astype(str).str.strip()
    num = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    # 1-4 档 → 0-3（优先于 0-3 判断）
    m14 = num.notna() & (num >= 1) & (num <= 4)
    out.loc[m14] = num.loc[m14] - 1
    # 已是 0-3（且不是上一步已填的）
    m03 = num.notna() & (num >= 0) & (num <= 3) & out.isna()
    out.loc[m03] = num.loc[m03]
    text_map = {
        "低": 0, "低风险": 0, "l1": 0, "1级": 0,
        "中": 1, "中等": 1, "中等风险": 1, "l2": 1, "2级": 1,
        "高": 2, "高风险": 2, "l3": 2, "3级": 2,
        "极高": 3, "极高风险": 3, "严重": 3, "l4": 3, "4级": 3,
    }
    for k, v in text_map.items():
        hit = s.str.contains(k, case=False, na=False) & out.isna()
        out = out.fillna(pd.Series(v, index=out.index).where(hit))
    return pd.to_numeric(out, errors="coerce")


def infer_risk_score(df: pd.DataFrame) -> pd.Series:
    def _num_col(name: str, default: float, clip_min=None, clip_max=None):
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
        else:
            s = pd.Series([default] * len(df), index=df.index, dtype=float)
        s = s.fillna(default)
        if clip_min is not None or clip_max is not None:
            s = s.clip(clip_min, clip_max)
        return s

    if "risk_score" in df.columns:
        return pd.to_numeric(df["risk_score"], errors="coerce")

    if any(c in df.columns for c in ["visual_risk", "hrv_risk", "eeg_risk"]):
        vis = _num_col("visual_risk", 0.0, 0, 1)
        hrv = _num_col("hrv_risk", 0.0, 0, 1)
        eeg = _num_col("eeg_risk", 0.0, 0, 1)
        return (0.35 * vis + 0.40 * hrv + 0.25 * eeg).clip(0, 1)

    vis_sad = _num_col("vis_sad", 0.0)
    vis_neu = _num_col("vis_neu", 0.0)
    vis_happy = _num_col("vis_happy", 0.0)
    vis_risk = (vis_sad - vis_happy + 0.5 * vis_neu).clip(0, 1)

    rmssd = _num_col("rmssd", 0.0)
    sdnn = _num_col("sdnn", 0.0)
    lf_hf = _num_col("lf_hf", 1.5)
    rmssd_r = (1 - (rmssd / 80.0)).clip(0, 1)
    sdnn_r = (1 - (sdnn / 120.0)).clip(0, 1)
    lfhf_r = ((lf_hf - 1.5).abs() / 2.0).clip(0, 1)
    hrv_risk = (0.45 * rmssd_r + 0.35 * sdnn_r + 0.20 * lfhf_r).clip(0, 1)

    att = _num_col("attention", 50.0, 0, 100)
    med = _num_col("meditation", 50.0, 0, 100)
    eeg_risk = ((100 - att) / 100.0 * 0.6 + (100 - med) / 100.0 * 0.4).clip(0, 1)

    return (0.35 * vis_risk + 0.40 * hrv_risk + 0.25 * eeg_risk).clip(0, 1)


def build_labels_from_aligned():
    if not os.path.exists(ALIGNED_PATH):
        return False

    feat_df = pd.read_csv(ALIGNED_PATH)
    feat_df = normalize_columns(feat_df)

    risk = infer_risk_score(feat_df)
    risk = risk.fillna(risk.median() if risk.notna().any() else 0.5).clip(0, 1)
    label = risk.apply(to_label).astype(int)

    os.makedirs(NN_DIR, exist_ok=True)
    labels_df = pd.DataFrame({"label": label, "risk_score": risk})
    labels_path = os.path.join(NN_DIR, "labels.csv")
    labels_df.to_csv(labels_path, index=False, encoding="utf-8-sig")

    if len(labels_df) != len(feat_df):
        raise RuntimeError(f"labels行数{len(labels_df)} != aligned_features行数{len(feat_df)}")

    print(f"[SAFE] 检测到 aligned_features.csv，labels 已按同源逐行生成")
    print(f"[SAVE] {labels_path} rows={len(labels_df)}")
    return True


def merge_from_excel():
    xlsx_files = sorted(
        glob.glob(os.path.join(SRC_DIR, "*.xlsx"))
        + glob.glob(os.path.join(SRC_DIR, "*.xls"))
    )
    csv_files = glob.glob(os.path.join(SRC_DIR, "*.csv"))
    frames = []

    for f in xlsx_files:
        try:
            df = pd.read_excel(f)
            df = normalize_columns(df)
            df["__source__"] = os.path.basename(f)
            frames.append(df)
            print(f"[OK] excel: {os.path.basename(f)} rows={len(df)}")
        except Exception as e:
            print(f"[SKIP] excel: {os.path.basename(f)} err={e}")

    skip_names = {"train.csv", "val.csv", "test.csv", "all_merged.csv"}
    for f in csv_files:
        bn = os.path.basename(f).lower()
        if bn in skip_names:
            continue
        try:
            df = pd.read_csv(f)
            df = normalize_columns(df)
            df["__source__"] = os.path.basename(f)
            frames.append(df)
            print(f"[OK] csv: {os.path.basename(f)} rows={len(df)}")
        except Exception as e:
            print(f"[SKIP] csv: {os.path.basename(f)} err={e}")

    if not frames:
        raise RuntimeError(f"没有读取到可用数据文件，请检查目录: {SRC_DIR}")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] 合并后总行数: {len(all_df)}")

    tier_parsed = None
    if "risk_tier" in all_df.columns:
        tier_parsed = parse_risk_tier_column(all_df["risk_tier"])
        n_ok = tier_parsed.notna().sum()
        print(f"[INFO] 风险等级列解析成功: {n_ok}/{len(all_df)}")

    all_df["risk_score"] = infer_risk_score(all_df)
    all_df["risk_score"] = pd.to_numeric(all_df["risk_score"], errors="coerce")
    all_df = all_df.dropna(subset=["risk_score"]).copy()
    all_df["risk_score"] = all_df["risk_score"].clip(0, 1)

    if tier_parsed is not None and tier_parsed.notna().sum() > len(all_df) * 0.5:
        all_df["label"] = tier_parsed.reindex(all_df.index).fillna(
            all_df["risk_score"].apply(to_label)
        ).astype(int)
        print("[INFO] label 优先使用 Excel「风险等级」，缺失行用 risk_score 分档")
    elif "label" not in all_df.columns:
        all_df["label"] = all_df["risk_score"].apply(to_label).astype(int)
    else:
        all_df["label"] = pd.to_numeric(all_df["label"], errors="coerce")
        all_df = all_df.dropna(subset=["label"]).copy()
        all_df["label"] = all_df["label"].astype(int)

    before = len(all_df)
    if "timestamp" in all_df.columns:
        ts = pd.to_datetime(all_df["timestamp"], errors="coerce")
        n_ts = int(ts.notna().sum())
        n_unique_ts = int(ts.nunique())
        # 仅当时间戳有效且足够分散时才按时间去重
        if n_ts > len(all_df) * 0.05 and n_unique_ts > max(1000, len(all_df) * 0.001):
            all_df = all_df.copy()
            all_df["_ts_parsed"] = ts
            all_df = all_df.sort_values("_ts_parsed").drop_duplicates(
                subset=["_ts_parsed", "__source__"] if "__source__" in all_df.columns else ["_ts_parsed"],
                keep="last",
            )
            all_df = all_df.drop(columns=["_ts_parsed"], errors="ignore")
            print(f"[INFO] 按时间戳去重 {before} -> {len(all_df)}")
        else:
            print(
                f"[WARN] 时间列有效行={n_ts} 唯一值={n_unique_ts}，跳过 timestamp 去重"
                "（避免 129 万行被压成 3 行）"
            )
    feat_cols = [
        c
        for c in (
            "vis_sad", "vis_neu", "vis_happy", "rmssd", "sdnn", "lf_hf",
            "attention", "meditation", "risk_score", "label",
        )
        if c in all_df.columns
    ]
    if feat_cols:
        all_df = all_df.drop_duplicates(subset=feat_cols, keep="first")
        print(f"[INFO] 按特征列精确去重 {before} -> {len(all_df)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_path = os.path.join(OUT_DIR, "all_merged.csv")
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {all_path} rows={len(all_df)}")

    strat = all_df["label"] if all_df["label"].value_counts().min() >= 2 else None
    train_df, tmp_df = train_test_split(
        all_df, test_size=0.30, random_state=42, stratify=strat
    )
    strat2 = tmp_df["label"] if tmp_df["label"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=42, stratify=strat2
    )

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"[SAVE] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    os.makedirs(NN_DIR, exist_ok=True)
    labels_df = all_df[["label", "risk_score"]].copy().reset_index(drop=True)
    labels_path = os.path.join(NN_DIR, "labels.csv")
    labels_df.to_csv(labels_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {labels_path} rows={len(labels_df)}")
    print(
        "[WARN] 尚未生成 nn/aligned_features.csv（25 维）；"
        "深度学习训练前需运行 excel_to_aligned 或采集对齐特征。"
    )


def main():
    ap = argparse.ArgumentParser(description="步骤 A: single_run Excel → 合并与 labels")
    ap.add_argument(
        "--force-excel",
        action="store_true",
        help="即使存在 nn/aligned_features.csv 也从 Excel 合并（会覆盖 nn/labels.csv）",
    )
    args = ap.parse_args()

    print(f"[INFO] SRC_DIR={SRC_DIR}")
    if not args.force_excel and build_labels_from_aligned():
        print("[DONE] 仅根据已有 aligned_features 更新了 labels.csv")
        return

    if args.force_excel:
        print("[INFO] --force-excel：从 Excel 合并，忽略 aligned 快捷路径")
    merge_from_excel()
    print("[DONE] 步骤 A 完成")


if __name__ == "__main__":
    main()