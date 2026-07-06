#!/usr/bin/env python3
"""
智谱 GLM 微调数据生成器 —— 从 MindRoom Guard 项目运行中产生训练数据。

设计理念：
  你的现有代码已经在你需要微调的所有环节中工作——risk_calculator 产生风险值、
  FeatureAligner 产生 25 维特征、InferenceEngine 产生推理结果。这个脚本把它们的
  输出装进智谱 GLM 微调所需的 JSONL 格式（每行一个 messages 对象）。

用法：
  # 从 aligned_features.csv 批量生成 5 条示例（用于测试格式）
  python data_toolkit/glm_data_generator.py --mode demo --out glm_demo.jsonl

  # 从运行中的 aligned_features.csv + labels.csv 生成全部训练数据
  python data_toolkit/glm_data_generator.py --mode full \
      --feat nn/aligned_features.csv \
      --labels nn/labels.csv \
      --out glm_train.jsonl \
      --samples-per-tier 50  # 每档风险选 50 条

  # 交互式人工标注模式（逐条审核弱标签，修正后再生成）
  python data_toolkit/glm_data_generator.py --mode annotate --out glm_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ---- 路径设置 ----
FEAT_CSV = 'nn/aligned_features.csv'
LABEL_CSV = 'nn/labels.csv'

VISION_DIM = 9
PHYSIO_DIM = 5
BRAIN_DIM = 11
N_CLASSES = 4

RISK_TIERS = ['低风险', '中等风险', '高风险', '极高风险']
AUDIENCES = ['学生', '老师', '家长']
DATA_SOURCES = ['实时', '演示', '离线']

# 与 feature_aligner.HEADER 一致
V_COLS = [f'v_{i}' for i in range(9)]
P_COLS = ['p_bpm', 'p_rmssd', 'p_sdnn', 'p_lfhf', 'p_rrv']
B_COLS = ['b_delta','b_theta','b_lalpha','b_halpha','b_lbeta','b_hbeta','b_lgamma','b_mgamma',
          'b_att','b_med','b_poor']


class GLMDataGenerator:
    """从 MindRoom 三模态特征生成智谱微调 JSONL 数据。"""

    def __init__(self, feat_csv: str = FEAT_CSV, label_csv: str = LABEL_CSV):
        self.feat_path = feat_csv
        self.label_path = label_csv
        self.feats: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        """加载特征和标签文件。"""
        if not os.path.isfile(self.feat_path):
            print(f'[ERROR] 特征文件不存在: {self.feat_path}')
            return False
        self.feats = pd.read_csv(self.feat_path)
        print(f'[OK] 加载特征: {len(self.feats)} 行 × {self.feats.shape[1]} 列')

        if os.path.isfile(self.label_path):
            self.labels = pd.read_csv(self.label_path)
            print(f'[OK] 加载标签: {len(self.labels)} 行')
        else:
            print(f'[WARN] 标签文件不存在: {self.label_path}（将使用模拟标签）')
            n = len(self.feats)
            self.labels = pd.DataFrame({
                'label': np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.2, 0.2]),
                'risk_score': np.clip(np.random.beta(2, 5, n), 0, 1),
                'source': 'simulated',
            })
        return True

    def _build_visual_detail(self, row) -> str:
        """从 9 维视觉特征构建文字描述。"""
        probs = row[V_COLS[:7]].values
        if probs.sum() < 1e-6:
            return "视觉数据缺失"
        top_i = int(np.argmax(probs))
        top_p = float(probs[top_i])
        labels = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
        top_expr = labels[top_i] if top_i < len(labels) else 'unknown'
        eye = float(row.get('v_7', 0) or 0)
        posture = float(row.get('v_8', 0) or 0)
        parts = [f"表情以 {top_expr} 为主（概率 {top_p:.2f}）"]
        if eye > 0.5:
            parts.append(f"眼疲劳指数偏高 {eye:.2f}")
        if posture > 0.5:
            parts.append(f"姿态异常（风险 {posture:.2f}）")
        return "，".join(parts)

    def _build_hrv_detail(self, row) -> str:
        bpm = float(row.get('p_bpm', 0) or 0)
        rmssd = float(row.get('p_rmssd', 0) or 0)
        sdnn = float(row.get('p_sdnn', 0) or 0)
        lfhf = float(row.get('p_lfhf', 0) or 0)
        return f"RMSSD={rmssd:.0f}ms, SDNN={sdnn:.0f}ms, LF/HF={lfhf:.1f}, HR={bpm:.0f}bpm"

    def _build_eeg_detail(self, row) -> str:
        att = float(row.get('b_att', 50) or 50)
        med = float(row.get('b_med', 50) or 50)
        detail = f"专注度={att:.0f}, 放松度={med:.0f}"
        if att < 40:
            detail += "（专注度显著偏低）"
        if med < 30 or med > 85:
            detail += "（放松度两极化）"
        return detail

    def _estimate_visual_risk(self, row) -> float:
        """简易视觉风险估算（与 risk_calculator 等价）。"""
        probs = row[V_COLS[:7]].values.astype(float)
        if probs.sum() < 1e-6:
            return 0.2
        i = int(np.argmax(probs))
        p = float(probs[i])
        if i == 0:      return 0.08 + 0.08 * (1 - p)
        elif i == 6:    return 0.20 + 0.15 * (1 - p)
        else:           return 0.20 + 0.85 * p

    def _estimate_hrv_risk(self, row) -> float:
        rmssd = float(row.get('p_rmssd', 0) or 0)
        sdnn = float(row.get('p_sdnn', 0) or 0)
        lfhf = float(row.get('p_lfhf', 0) or 0)
        hr = float(row.get('p_bpm', 0) or 0)
        score = 0
        if rmssd <= 34: score += 2
        elif rmssd <= 49: score += 1
        if sdnn <= 69: score += 2
        elif sdnn <= 99: score += 1
        if lfhf >= 4.1: score += 2
        elif lfhf >= 2.6: score += 1
        if hr >= 86: score += 2
        elif hr >= 76: score += 1
        return min(1.0, score / 8.0)

    def _estimate_eeg_risk(self, row) -> float:
        att = float(row.get('b_att', 50) or 50)
        med = float(row.get('b_med', 50) or 50)
        a = 0.8 if att <= 39 else (0.5 if att <= 59 else 0.2)
        if med <= 29 or med > 85: m = 0.8
        elif med <= 39 or med >= 71: m = 0.5
        else: m = 0.2
        return (a + m) / 2.0

    def _trend_description(self, risk_window: list) -> tuple:
        """简易 24h 趋势描述。"""
        if len(risk_window) < 3:
            return '平稳', {'slope': 0.0, 'volatility': 0.0}
        arr = np.array(risk_window)
        slope = float(np.polyfit(range(len(arr)), arr, 1)[0])
        vol = float(np.std(arr))
        if slope > 0.1: trend = '急剧恶化'
        elif slope > 0.03: trend = '持续上升'
        elif slope > 0.01: trend = '缓慢上升'
        elif slope < -0.1: trend = '快速好转'
        elif slope < -0.03: trend = '缓慢下降'
        elif vol > 0.15: trend = '波动'
        else: trend = '平稳'
        return trend, {'slope': round(slope, 4), 'volatility': round(vol, 4)}

    def build_one_sample(self, idx: int, risk_score: float,
                         tier: int, audience: str, source: str) -> dict:
        """构建单条训练样本（user 侧的 structured context）。"""
        row = self.feats.iloc[idx]

        v_risk = self._estimate_visual_risk(row)
        p_risk = self._estimate_hrv_risk(row)
        e_risk = self._estimate_eeg_risk(row)

        # 最近状态摘要（模拟相邻行）
        recent = []
        start = max(0, idx - 4)
        for j in range(start, idx):
            if j < len(self.labels):
                rj = float(self.labels.iloc[j].get('risk_score', 0.3))
                recent.append({
                    'time': f'{14:02d}:{20 + (j - start) * 2:02d}',
                    'risk': round(rj, 2),
                })
        if not recent:
            recent = [{'time': '14:20', 'risk': round(risk_score, 2)}]

        # 模拟最近异常
        anomalies = []
        if tier >= 2:  # 高风险或以上才加异常
            v_anom = (v_risk > 0.6)
            p_anom = (p_risk > 0.6)
            e_anom = (e_risk > 0.6)
            if v_anom and p_anom and e_anom:
                anomalies = [
                    {'time': '14:05', 'duration_min': 5, 'type': '三模态同时高危', 'consecutive': False},
                    {'time': '14:12', 'duration_min': 8, 'type': '多维同步报警', 'consecutive': True},
                ]
            elif p_anom:
                anomalies = [
                    {'time': '13:45', 'duration_min': 15, 'type': 'HRV 持续偏高', 'consecutive': True},
                ]
            elif v_anom:
                anomalies = [
                    {'time': '14:08', 'duration_min': 12, 'type': '连续低头+负面表情', 'consecutive': True},
                ]
            elif e_anom:
                anomalies = [
                    {'time': '14:09', 'duration_min': 2, 'type': '专注度骤降', 'consecutive': False},
                ]

        trend, trend_detail = self._trend_description(
            [risk_score - 0.1, risk_score - 0.05, risk_score])

        context = {
            'current_risk': round(risk_score, 2),
            'risk_level': RISK_TIERS[tier],
            'trend_24h': trend,
            'modalities': {
                'visual': {'risk': round(v_risk, 2), 'weight': 0.35,
                           'detail': self._build_visual_detail(row)},
                'hrv':    {'risk': round(p_risk, 2), 'weight': 0.40,
                           'detail': self._build_hrv_detail(row)},
                'eeg':    {'risk': round(e_risk, 2), 'weight': 0.25,
                           'detail': self._build_eeg_detail(row)},
            },
            'recent_anomalies': anomalies,
            'data_source': source,
            'output_audience': audience,
            'recent_state_summary': recent,
        }

        user_content = json.dumps({'context': context}, ensure_ascii=False)

        # system prompt（来自升级计划的"固定 JSON 模板"设计）
        system_prompt = (
            "你是 MindRoom Guard 校园心理健康助手。你收到多模态生理数据，"
            "需要先做风险归因分析，再给出建议动作。输出给学生的回复应该语气温暖、共情，语言简练。"
            "输出给家长/老师时应该结构化、具可操作性。"
        )

        # assistant 内容（弱标注版本——你的 label_tool 规则逻辑）
        # 注意：真实训练时，这些 assistant 回复需要人工审核和润色
        assistant = self._generate_weak_assistant(tier, v_risk, p_risk, e_risk, audience)

        return {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': json.dumps(assistant, ensure_ascii=False)},
            ]
        }

    def _generate_weak_assistant(self, tier: int, v_risk: float,
                                  p_risk: float, e_risk: float,
                                  audience: str) -> dict:
        """基于规则映射生成弱标注 assistant 回复（需人工审核）。"""
        tier_names = ['低风险', '中等风险', '高风险', '极高风险']

        # 归因
        dominant = None
        risks = {'visual': v_risk, 'hrv': p_risk, 'eeg': e_risk}
        if tier >= 1:
            dominant = max(risks, key=risks.get)

        if tier == 0:
            attr = "三模态指标均在正常范围，综合评估无异常信号。"
            conf = 0.92
        elif tier == 1:
            attr = f"主导因素是 {dominant} 模态轻微偏差，整体不达警戒线。可能原因：疲劳累积或注意力波动。"
            conf = 0.82
        elif tier == 2:
            attr = f"三模态中 {dominant} 模态偏差明显，需关注趋势变化。"
            conf = 0.85
        else:
            attr = f"三模态同步报警，{dominant} 模态最严重。学生可能处于急性应激状态。"
            conf = 0.90

        assistant = {
            'stage_a': {
                'attribution': attr,
                'dominant_modality': dominant,
                'confidence': conf,
            },
            'stage_b': {
                'advice': self._audience_advice(tier, audience),
                'immediate_action': None,
                'short_term_action': None,
                'escalation_condition': None,
            }
        }

        # 高风险时强制添加干预字段（对应升级计划"规则护栏"）
        if tier >= 2:
            assistant['stage_b']['immediate_action'] = {
                'description': '安排老师或心理辅导员到学生身边轻声询问，确认安全状态。',
                'deadline': '5分钟',
            }
            assistant['stage_b']['short_term_action'] = {
                'description': '安排心理老师在未来 3 天内做一对一访谈，并告知家长。',
                'deadline': '3天',
            }
            assistant['stage_b']['escalation_condition'] = (
                '如出现自伤言语或下一监测周期风险仍 > 0.75，启动校园危机干预流程。'
            )

        return assistant

    def _audience_advice(self, tier: int, audience: str) -> str:
        """按目标受众生成不同语气的建议。"""
        if tier == 0:
            if audience == '学生':
                return "你当前状态很好，保持现在的节奏。课间起来活动一下。"
            elif audience == '老师':
                return "该生当前状态良好，无需特别关注。"
            else:
                return "您的孩子今天在校状态良好，各项指标均正常。"
        elif tier == 1:
            if audience == '学生':
                return "我注意到你注意力有些波动，试着做几次深呼吸放松一下。"
            elif audience == '老师':
                return "该生注意力轻微波动，建议课间关注是否需要帮助。"
            else:
                return "今天孩子在校稍微有些疲劳的信号，建议今晚早点休息。"
        elif tier == 2:
            if audience == '学生':
                return "你好像有些压力在累积。如果需要聊聊，心理老师今天在办公室。"
            elif audience == '老师':
                return "该生风险指标偏高，建议课后做一次简短沟通，了解是否有困扰。"
            else:
                return "今天孩子的生理和心理压力指标偏高，建议今晚和孩子聊聊学校生活。"
        else:
            if audience == '学生':
                return "你现在需要一点帮助，这完全没关系。会有老师来陪你聊聊。"
            elif audience == '老师':
                return "该生处于高风险状态，请立即安排老师前往查看并通知家长。"
            else:
                return "您的孩子在校情绪状态非常糟糕，学校已启动紧急关怀程序，请保持电话畅通。"

    def generate(self, output: str, samples_per_tier: int = 30,
                 audiences: list = None, sources: list = None):
        """批量生成 JSONL 训练数据。"""
        if audiences is None:
            audiences = AUDIENCES
        if sources is None:
            sources = DATA_SOURCES

        n = len(self.feats)
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        samples = []
        for tier in range(N_CLASSES):
            # 从 labels 里找该 tier 的行号
            if 'label' in self.labels.columns:
                tier_indices = self.labels.index[
                    self.labels['label'] == tier
                ].tolist()
            else:
                # fallback: 按 risk_score 分档
                risk_col = self.labels['risk_score']
                if tier == 0:
                    mask = risk_col < 0.30
                elif tier == 1:
                    mask = (risk_col >= 0.30) & (risk_col < 0.60)
                elif tier == 2:
                    mask = (risk_col >= 0.60) & (risk_col < 0.80)
                else:
                    mask = risk_col >= 0.80
                tier_indices = self.labels.index[mask].tolist()

            # 随机选取 samples_per_tier 条
            if len(tier_indices) > samples_per_tier:
                tier_indices = random.sample(tier_indices, samples_per_tier)

            for idx in tier_indices:
                if idx >= n:
                    continue
                risk_score = float(
                    self.labels.iloc[idx].get('risk_score', 0.3)
                )
                audience = random.choice(audiences)
                source = random.choice(sources)
                sample = self.build_one_sample(idx, risk_score, tier, audience, source)
                samples.append(sample)

            print(f'  [tier {tier} - {RISK_TIERS[tier]}] 选取 {len(tier_indices)} 条'
                  f' → 最终 {min(len(tier_indices), samples_per_tier)} 条')

        # 写入 JSONL
        with open(output, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

        print(f'\n[OK] 共生成 {len(samples)} 条训练样本 → {output}')
        print(f'     等级分布: {np.bincount([s["messages"][1]["content"].count(RISK_TIERS[i]) for i, s in zip([0,1,2,3], [samples]*4)]).tolist() if False else "手动统计"}')


def mode_demo(output: str):
    """演示模式：生成 5 条高质量手工样本（即 glm_finetune_examples.jsonl 内容）。"""
    # 直接复制 examples 文件
    examples_path = os.path.join(
        os.path.dirname(__file__), 'glm_finetune_examples.jsonl'
    )
    if os.path.isfile(examples_path):
        import shutil
        shutil.copy(examples_path, output)
        print(f'[OK] 已复制 {examples_path} → {output}')
    else:
        print('[WARN] 示例文件不存在，生成 0 条。请先确保 glm_finetune_examples.jsonl 存在。')


def mode_annotate(output: str):
    """交互式人工标注模式（逐条审核并修正）。"""
    print('=' * 60)
    print('交互式标注模式')
    print('逐条输入用户上下文，然后手动写 assistant 回复')
    print('输入 q 退出，s 跳过当前条')
    print('=' * 60)

    samples = []
    count = 0
    while True:
        print(f'\n--- 样本 #{count + 1} ---')
        tier = input('风险等级 (0=低 1=中 2=高 3=极高): ').strip()
        if tier.lower() == 'q':
            break
        if tier.lower() == 's' or tier not in ['0', '1', '2', '3']:
            continue
        tier = int(tier)
        risk = float(input('风险分数 (0-1): ').strip() or '0.3')
        audience = input('目标受众 (学生/老师/家长): ').strip() or '学生'
        source = input('数据源 (实时/演示/离线): ').strip() or '实时'

        user_json = input('粘贴 user context JSON（留空用简单模板）: ').strip()
        if not user_json:
            user_json = json.dumps({
                'context': {
                    'current_risk': risk,
                    'risk_level': RISK_TIERS[tier],
                    'trend_24h': '平稳',
                    'modalities': {
                        'visual': {'risk': 0.2, 'weight': 0.35, 'detail': ''},
                        'hrv': {'risk': 0.2, 'weight': 0.40, 'detail': ''},
                        'eeg': {'risk': 0.2, 'weight': 0.25, 'detail': ''},
                    },
                    'recent_anomalies': [],
                    'data_source': source,
                    'output_audience': audience,
                    'recent_state_summary': [],
                }
            }, ensure_ascii=False)

        attr = input('阶段A 归因: ').strip()
        advice = input('阶段B 建议: ').strip()
        if not attr or not advice:
            print('[SKIP] 归因和建议不能为空')
            continue

        assistant = {
            'stage_a': {
                'attribution': attr,
                'dominant_modality': None,
                'confidence': 0.85,
            },
            'stage_b': {
                'advice': advice,
                'immediate_action': None,
                'short_term_action': None,
                'escalation_condition': None,
            }
        }

        sample = {
            'messages': [
                {'role': 'system', 'content': '你是 MindRoom Guard 校园心理健康助手。'},
                {'role': 'user', 'content': user_json},
                {'role': 'assistant', 'content': json.dumps(assistant, ensure_ascii=False)},
            ]
        }
        samples.append(sample)
        count += 1
        print(f'[OK] 已接受 #{count}')

    if samples:
        with open(output, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f'\n[OK] 共标注 {len(samples)} 条 → {output}')
    else:
        print('\n[INFO] 未生成任何样本')


def main():
    ap = argparse.ArgumentParser(
        description='智谱 GLM 微调数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python data_toolkit/glm_data_generator.py --mode demo --out glm_demo.jsonl
  python data_toolkit/glm_data_generator.py --mode full --feat nn/aligned_features.csv --out glm_train.jsonl
  python data_toolkit/glm_data_generator.py --mode annotate --out glm_manual.jsonl
        ''',
    )
    ap.add_argument('--mode', choices=['demo', 'full', 'annotate'], required=True,
                    help='demo=复制示例文件 | full=批量生成 | annotate=交互式标注')
    ap.add_argument('--feat', default=FEAT_CSV, help='aligned_features.csv 路径')
    ap.add_argument('--labels', default=LABEL_CSV, help='labels.csv 路径')
    ap.add_argument('--out', default='glm_train.jsonl', help='输出 JSONL 路径')
    ap.add_argument('--samples-per-tier', type=int, default=30,
                    help='每档风险抽取的样本数（full 模式）')
    args = ap.parse_args()

    if args.mode == 'demo':
        mode_demo(args.out)
    elif args.mode == 'annotate':
        mode_annotate(args.out)
    elif args.mode == 'full':
        gen = GLMDataGenerator(args.feat, args.labels)
        if not gen.load():
            sys.exit(1)
        gen.generate(args.out, samples_per_tier=args.samples_per_tier)


if __name__ == '__main__':
    main()
