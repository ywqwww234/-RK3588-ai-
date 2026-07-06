"""
半监督标签工具：用 risk_calculator 的规则化输出对 aligned_features.csv 做软标注。

设计动机：
  - P0 改进清单要求"禁止随机标签"，但项目早期采集到的数据没有人工标注
  - 项目已经有完整的规则化风险评估（risk_calculator）作为"专家先验"
  - 用规则输出生成 weak/silver label，配合训练时的类权重 + 后期人工复核，比
    随机标签强 N 个数量级，且与论文体系自洽

产出：
    nn/labels.csv 含三列：
        label       int    {0/1/2/3} 四档
        risk_score  float  [0,1]
        source      str    'rule_weak' / 'human' / 'mixed'

用法：
    # 直接从 aligned_features.csv 全表生成弱标签（覆盖写）
    python -m nn.label_tool weak

    # 把已有人工标注（labels_human.csv）合并进来（人工优先级最高）
    python -m nn.label_tool merge --human nn/labels_human.csv

注意：
  - 弱标签不是真值，仅用于 PoC 流水线打通与早期热启动
  - 论文/答辩口径必须诚实说明：分类头训练数据来源为"规则化弱监督"
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

# 路径与模型/aligner 保持一致
FEAT_CSV  = 'nn/aligned_features.csv'
LABEL_CSV = 'nn/labels.csv'

# 与 feature_aligner.HEADER 一致的列顺序
V_COLS = [f'v_{i}' for i in range(9)]
P_COLS = ['p_bpm', 'p_rmssd', 'p_sdnn', 'p_lfhf', 'p_rrv']
B_COLS = ['b_delta','b_theta','b_lalpha','b_halpha','b_lbeta','b_hbeta','b_lgamma','b_mgamma',
          'b_att','b_med','b_poor']


def _rule_visual_risk(row) -> float:
    """v_0..v_6 = 7 类表情概率；v_7 eye_fatigue；v_8 posture_risk。"""
    probs = np.array([row[c] for c in V_COLS[:7]], dtype=np.float32)
    eye = float(row['v_7'] or 0)
    posture = float(row['v_8'] or 0)
    # 表情类型粗分（与 risk_calculator 等价的判别）：
    #   index 0 happy / 1 sad / 2 angry / 3 fear / 4 disgust / 5 surprise / 6 neutral
    # 这里采用通用映射：取 argmax，并用 risk_calculator 的负向类逻辑近似。
    if probs.sum() < 1e-6:
        expr_risk = 0.2
    else:
        i = int(np.argmax(probs))
        p = float(probs[i])
        if i == 0:                              # happy
            expr_risk = 0.08 + 0.08 * (1 - p)
        elif i == 6:                            # neutral
            expr_risk = 0.20 + 0.15 * (1 - p)
        else:                                   # 负向
            expr_risk = 0.20 + 0.85 * p
    # 视觉总分 = max(表情, 0.8*posture, 0.6*eye_fatigue)
    return float(max(expr_risk, 0.8 * posture, 0.6 * eye))


def _rule_hrv_risk(row) -> float:
    rmssd = float(row['p_rmssd'] or 0)
    sdnn = float(row['p_sdnn'] or 0)
    lf_hf = float(row['p_lfhf'] or 1.5)
    hr = float(row['p_bpm'] or 70)
    score = 0
    if rmssd <= 34: score += 2
    elif rmssd <= 49: score += 1
    if sdnn <= 69: score += 2
    elif sdnn <= 99: score += 1
    # HF 在 aligner 里没单独存，简化掉（满分 8）
    if lf_hf >= 4.1: score += 2
    elif lf_hf >= 2.6: score += 1
    if hr >= 86: score += 2
    elif hr >= 76: score += 1
    return float(min(1.0, score / 8.0))


def _rule_eeg_risk(row) -> float:
    att = float(row['b_att'] or 50)
    med = float(row['b_med'] or 50)
    if att <= 39: a = 0.8
    elif att <= 59: a = 0.5
    else: a = 0.2
    if med <= 29 or med > 85: m = 0.8
    elif med <= 39 or med >= 71: m = 0.5
    else: m = 0.2
    return float((a + m) / 2.0)


def _to_tier(risk: float) -> int:
    if risk < 0.30: return 0
    if risk < 0.60: return 1
    if risk < 0.80: return 2
    return 3


def make_weak_labels(feat_csv: str = FEAT_CSV, out_csv: str = LABEL_CSV,
                     veto_threshold: float = 0.6) -> pd.DataFrame:
    df = pd.read_csv(feat_csv)
    visual = df.apply(_rule_visual_risk, axis=1).values
    hrv    = df.apply(_rule_hrv_risk, axis=1).values
    eeg    = df.apply(_rule_eeg_risk, axis=1).values
    avg = 0.35 * visual + 0.40 * hrv + 0.25 * eeg
    # 视觉一票否决（与 risk_calculator.calculate_total_risk 等价）
    veto = visual >= veto_threshold
    risk = np.where(veto, np.maximum(avg, visual), avg)
    risk = np.clip(risk, 0.0, 1.0)
    tiers = np.array([_to_tier(r) for r in risk], dtype=np.int64)
    out = pd.DataFrame({
        'label': tiers,
        'risk_score': risk.astype(np.float32),
        'source': 'rule_weak',
    })
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f'[ok] weak labels -> {out_csv}  rows={len(out)}  '
          f'dist={np.bincount(tiers, minlength=4).tolist()}')
    return out


def merge_human(weak_csv: str = LABEL_CSV,
                human_csv: Optional[str] = None,
                out_csv: str = LABEL_CSV) -> pd.DataFrame:
    """合并人工标注（按行号匹配；human 文件需含 'label' 列与可选 'risk_score'）。"""
    if human_csv is None or not os.path.isfile(human_csv):
        print(f'[skip] 未提供 human 标注: {human_csv}')
        return pd.read_csv(weak_csv)
    wk = pd.read_csv(weak_csv)
    hm = pd.read_csv(human_csv)
    n = min(len(wk), len(hm))
    merged = wk.copy()
    if 'label' in hm.columns:
        mask = hm['label'].iloc[:n].notna()
        merged.loc[:n - 1, 'label'] = np.where(mask, hm['label'].iloc[:n].values,
                                                merged['label'].iloc[:n].values)
        merged.loc[mask, 'source'] = 'human'
    if 'risk_score' in hm.columns and 'risk_score' in merged.columns:
        mask = hm['risk_score'].iloc[:n].notna()
        merged.loc[:n - 1, 'risk_score'] = np.where(mask, hm['risk_score'].iloc[:n].values,
                                                     merged['risk_score'].iloc[:n].values)
    # 既有人工又有弱标签 → mixed（仅做标注信息记录）
    merged['source'] = merged['source'].fillna('rule_weak')
    merged.to_csv(out_csv, index=False)
    print(f'[ok] merged -> {out_csv}  human_rows={int((merged.source=="human").sum())}')
    return merged


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('weak', help='生成规则化弱标签')
    p1.add_argument('--feat', default=FEAT_CSV)
    p1.add_argument('--out',  default=LABEL_CSV)
    p2 = sub.add_parser('merge', help='合并人工标注（人工优先）')
    p2.add_argument('--weak', default=LABEL_CSV)
    p2.add_argument('--human', required=True)
    p2.add_argument('--out',  default=LABEL_CSV)
    args = ap.parse_args()
    if args.cmd == 'weak':
        make_weak_labels(args.feat, args.out)
    else:
        merge_human(args.weak, args.human, args.out)


if __name__ == '__main__':
    main()
