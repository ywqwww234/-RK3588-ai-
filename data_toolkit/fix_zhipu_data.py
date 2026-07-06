#!/usr/bin/env python3
"""
智谱数据格式修复脚本 —— 修复你上传到智谱开放平台时遇到的 4 个结构性问题。

用法：
    python data_toolkit/fix_zhipu_data.py --dry-run   # 先扫描问题
    python data_toolkit/fix_zhipu_data.py              # 执行修复（备份原文件到 .bak）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from typing import List, Dict, Any

# ---- 路径 ----
ZHIPU_DIR = 'data/zhipu'
FILES = [
    'zhipu_sft_train.jsonl',
    'zhipu_sft_val.jsonl',
    'zhipu_sft_test.jsonl',
    'zhipu_dpo_pairs.jsonl',
    'zhipu_eval_test.jsonl',
    'zhipu_fallback_gold.jsonl',
]

# ---- 统一的 System Prompt ----
UNIFIED_SYSTEM_PROMPT = (
    "你是 MindRoom Guard 校园心理健康助手，运行在 RK3588 边缘计算平台。"
    "你接收多模态生理数据（视觉表情/HRV心率变异性/EEG脑电）的结构化 JSON 上下文，"
    "需要先完成阶段A·风险归因分析（识别主导模态和影响因素），"
    "再完成阶段B·干预动作规划（依据风险等级和受众生成分级建议）。"
    "给学生的回复语气温暖共情，给家长/老师的回复结构化具可操作性。"
    "高风险时必须输出立即动作（今天）、短期动作（3天）、升级条件（何时通知家长/老师）。"
)


def load_lines(path: str) -> List[Dict]:
    """加载 JSONL 文件，跳过空行，返回 (obj, line_number) 列表。"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'  [ERROR] {path} 第 {i+1} 行 JSON 解析失败: {e}')
    return records


def safe_backup(path: str) -> str:
    """备份原文件 → .bak 后缀。"""
    bak = path + '.bak'
    if os.path.isfile(bak):
        # 如果 .bak 已存在，加时间戳
        import time
        bak = path + f'.bak.{int(time.time())}'
    shutil.copy2(path, bak)
    print(f'  [BACKUP] {path} → {bak}')
    return bak


def fix_assistant_content(assistant_str: str) -> str:
    """
    统一 assistant 的 content 格式：
    - 如果是单阶段 dict → 包装成 [dict] （统一为两阶段列表）
    - 如果已经是两阶段 list → 保持
    - 如果是普通文本 → 包装成 fallback 格式
    """
    try:
        inner = json.loads(assistant_str)
    except (json.JSONDecodeError, TypeError):
        # 非 JSON 文本 → 包装
        return json.dumps([{
            "stage": "mixed",
            "content": assistant_str,
            "note": "auto-fixed: 原始非JSON内容已包装"
        }], ensure_ascii=False)

    if isinstance(inner, list):
        # 已经是列表格式 → 保持
        return assistant_str

    if isinstance(inner, dict):
        stage = inner.get('stage', 'unknown')
        if stage in ('attribution', 'actions', 'fallback'):
            # 单阶段 dict → 包装成列表
            return json.dumps([inner], ensure_ascii=False)
        else:
            return json.dumps([{"stage": stage, **inner}], ensure_ascii=False)

    return assistant_str


def fix_sft_file(path: str, dry_run: bool = False) -> dict:
    """修复 SFT 文件：统一 system prompt + 统一 assistant 格式。"""
    records = load_lines(path)
    stats = {
        'total': len(records),
        'sys_fix': 0,
        'asst_fix': 0,
        'mixed_task': 0,
        'issues': [],
    }

    for obj in records:
        msgs = obj.get('messages', [])
        if not msgs:
            continue

        # Fix 1: 统一 system prompt
        if msgs[0].get('role') == 'system':
            old_sys = msgs[0]['content']
            if old_sys != UNIFIED_SYSTEM_PROMPT:
                msgs[0]['content'] = UNIFIED_SYSTEM_PROMPT
                stats['sys_fix'] += 1

        # Fix 2: 检查是否混合了单任务/多任务的记录
        has_attribution = False
        has_actions = False
        for m in msgs:
            if m.get('role') == 'system':
                if '阶段A' in m['content'] and '阶段B' not in m['content']:
                    has_attribution = True
                if '阶段B' in m['content']:
                    has_actions = True

        if has_attribution and not has_actions:
            stats['mixed_task'] += 1

        # Fix 3: 统一 assistant content 格式
        for m in msgs:
            if m.get('role') == 'assistant':
                old = m['content']
                new = fix_assistant_content(old)
                if old != new:
                    m['content'] = new
                    stats['asst_fix'] += 1

    if not dry_run:
        bak = safe_backup(path)
        with open(path, 'w', encoding='utf-8') as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return stats


def fix_dpo_file(path: str, dry_run: bool = False) -> dict:
    """修复 DPO 文件：chosen/rejected 从纯字符串改成 message 对象。"""
    records = load_lines(path)
    stats = {'total': len(records), 'chosen_fix': 0, 'rejected_fix': 0}

    for obj in records:
        # Fix chosen
        chosen = obj.get('chosen')
        if isinstance(chosen, str):
            # 智谱 DPO 格式要求: chosen = {"role": "assistant", "content": "..."}
            obj['chosen'] = {
                "role": "assistant",
                "content": fix_assistant_content(chosen)
            }
            stats['chosen_fix'] += 1

        # Fix rejected
        rejected = obj.get('rejected')
        if isinstance(rejected, str):
            obj['rejected'] = {
                "role": "assistant",
                "content": fix_assistant_content(rejected)
            }
            stats['rejected_fix'] += 1

        # Fix system prompt
        msgs = obj.get('messages', [])
        if msgs and msgs[0].get('role') == 'system':
            if msgs[0]['content'] != UNIFIED_SYSTEM_PROMPT:
                msgs[0]['content'] = UNIFIED_SYSTEM_PROMPT

    if not dry_run:
        bak = safe_backup(path)
        with open(path, 'w', encoding='utf-8') as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return stats


def fix_eval_file(path: str, dry_run: bool = False) -> dict:
    """修复 eval_test：统一 system prompt + 确保 assistant 格式与训练集一致。"""
    records = load_lines(path)
    stats = {'total': len(records), 'sys_fix': 0, 'asst_fix': 0}

    for obj in records:
        msgs = obj.get('messages', [])
        if msgs and msgs[0].get('role') == 'system':
            if msgs[0]['content'] != UNIFIED_SYSTEM_PROMPT:
                msgs[0]['content'] = UNIFIED_SYSTEM_PROMPT
                stats['sys_fix'] += 1

        for m in msgs:
            if m.get('role') == 'assistant':
                old = m['content']
                new = fix_assistant_content(old)
                if old != new:
                    m['content'] = new
                    stats['asst_fix'] += 1

    if not dry_run:
        bak = safe_backup(path)
        with open(path, 'w', encoding='utf-8') as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return stats


def fix_fallback_file(path: str, dry_run: bool = False) -> dict:
    """修复 fallback_gold：保持离线 system prompt 不变（它本身就是独立的降级模式），
    但统一 assistant 格式。"""
    records = load_lines(path)
    stats = {'total': len(records), 'asst_fix': 0}

    for obj in records:
        msgs = obj.get('messages', [])
        for m in msgs:
            if m.get('role') == 'assistant':
                old = m['content']
                new = fix_assistant_content(old)
                if old != new:
                    m['content'] = new
                    stats['asst_fix'] += 1

    if not dry_run:
        bak = safe_backup(path)
        with open(path, 'w', encoding='utf-8') as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return stats


def validate_fixed(path: str) -> dict:
    """修复后校验。"""
    records = load_lines(path)
    stats = {
        'valid_json': len(records),
        'has_messages': 0,
        'has_system': 0,
        'assistant_json_list': 0,
        'assistant_dict': 0,
        'assistant_other': 0,
        'dpo_chosen_ok': 0,
        'dpo_chosen_bad': 0,
    }

    for obj in records:
        msgs = obj.get('messages')
        if msgs:
            stats['has_messages'] += 1
            if msgs[0].get('role') == 'system':
                stats['has_system'] += 1

            for m in msgs:
                if m.get('role') == 'assistant':
                    content = m.get('content', '')
                    try:
                        inner = json.loads(content)
                        if isinstance(inner, list):
                            stats['assistant_json_list'] += 1
                        elif isinstance(inner, dict):
                            stats['assistant_dict'] += 1
                    except:
                        stats['assistant_other'] += 1

        # DPO checks
        chosen = obj.get('chosen')
        if chosen is not None:
            if isinstance(chosen, dict) and chosen.get('role') == 'assistant':
                stats['dpo_chosen_ok'] += 1
            else:
                stats['dpo_chosen_bad'] += 1

    return stats


def scan_all(base_dir: str) -> dict:
    """扫描所有文件，报告问题并生成修复统计。"""
    results = {}
    print('=' * 70)
    print('智谱训练数据问题扫描报告')
    print('=' * 70)

    for fname in FILES:
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            print(f'\n[SKIP] {fname} - 文件不存在')
            continue

        records = load_lines(fpath)
        print(f'\n--- {fname} ({len(records)} 条) ---')

        issues = {
            'dpo_chosen_is_str': 0,
            'dpo_rejected_is_str': 0,
            'asst_is_dict': 0,
            'asst_is_list': 0,
            'asst_is_text': 0,
            'sys_variants': Counter(),
            'task_split': Counter(),  # 'attribution_only' / 'actions_only' / 'full' / 'fallback'
        }

        for obj in records:
            # DPO 专项
            chosen = obj.get('chosen')
            rejected = obj.get('rejected')
            if isinstance(chosen, str):
                issues['dpo_chosen_is_str'] += 1
            if isinstance(rejected, str):
                issues['dpo_rejected_is_str'] += 1

            # system prompt 变体统计
            msgs = obj.get('messages', [])
            for m in msgs:
                if m.get('role') == 'system':
                    preview = m['content'][:60]
                    issues['sys_variants'][preview] += 1

                    # 任务类型
                    c = m['content']
                    has_a = '阶段A' in c or '归因' in c
                    has_b = '阶段B' in c or '干预' in c or '动作' in c
                    is_fallback = '离线' in c or '降级' in c or 'fallback' in c or '不可用' in c
                    if is_fallback:
                        issues['task_split']['fallback'] += 1
                    elif has_a and has_b:
                        issues['task_split']['full'] += 1
                    elif has_a:
                        issues['task_split']['attribution_only'] += 1
                    elif has_b:
                        issues['task_split']['actions_only'] += 1
                    else:
                        issues['task_split']['unclear'] += 1
                    break  # 只看第一条 system

                # assistant 格式统计
                if m.get('role') == 'assistant':
                    content = m.get('content', '')
                    try:
                        inner = json.loads(content)
                        if isinstance(inner, dict):
                            issues['asst_is_dict'] += 1
                        elif isinstance(inner, list):
                            issues['asst_is_list'] += 1
                    except:
                        issues['asst_is_text'] += 1

        # 打印
        if issues['dpo_chosen_is_str']:
            print(f'  ⚠ DPO chosen 是字符串: {issues["dpo_chosen_is_str"]} 条（需改成 message 对象）')
        if issues['dpo_rejected_is_str']:
            print(f'  ⚠ DPO rejected 是字符串: {issues["dpo_rejected_is_str"]} 条（需改成 message 对象）')

        print(f'  Assistant 格式: dict={issues["asst_is_dict"]} list={issues["asst_is_list"]} text={issues["asst_is_text"]}')
        if issues['asst_is_dict'] > 0:
            print(f'    → {issues["asst_is_dict"]} 条需包装成 list（统一两阶段格式）')

        print(f'  System Prompt 变体: {len(issues["sys_variants"])} 种')
        for sp, cnt in issues['sys_variants'].most_common(3):
            print(f'    [{sp}...] ×{cnt}')

        print(f'  任务类型分布: {dict(issues["task_split"])}')
        if len(issues['task_split']) > 1:
            print(f'    → 建议统一为 "full" 模式（两阶段）或保持 fallback 独立')

        results[fname] = issues

    return results


def main():
    ap = argparse.ArgumentParser(description='修复智谱微调数据格式')
    ap.add_argument('--dry-run', action='store_true',
                    help='仅扫描问题不修改文件')
    ap.add_argument('--dir', default=ZHIPU_DIR,
                    help=f'数据目录（默认: {ZHIPU_DIR}）')
    ap.add_argument('--validate-only', action='store_true',
                    help='仅验证已经修复过的文件')
    args = ap.parse_args()

    base_dir = args.dir

    if args.validate_only:
        print('=' * 70)
        print('修复后验证')
        print('=' * 70)
        for fname in FILES:
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            stats = validate_fixed(fpath)
            print(f'\n{fname}:')
            for k, v in stats.items():
                if v:
                    print(f'  {k}: {v}')
        return

    if args.dry_run:
        scan_all(base_dir)
        return

    # --- 执行修复 ---
    print('=' * 70)
    print('智谱数据格式修复')
    print('=' * 70)
    print()

    for fname in FILES:
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue

        print(f'修复 {fname}...')

        if 'dpo' in fname:
            stats = fix_dpo_file(fpath, dry_run=False)
            print(f'  → chosen 修复: {stats["chosen_fix"]} 条')
            print(f'  → rejected 修复: {stats["rejected_fix"]} 条')
        elif 'eval' in fname:
            stats = fix_eval_file(fpath, dry_run=False)
            print(f'  → system prompt 统一: {stats["sys_fix"]} 条')
            print(f'  → assistant 格式统一: {stats["asst_fix"]} 条')
        elif 'fallback' in fname:
            stats = fix_fallback_file(fpath, dry_run=False)
            print(f'  → assistant 格式统一: {stats["asst_fix"]} 条')
        else:
            stats = fix_sft_file(fpath, dry_run=False)
            print(f'  → system prompt 统一: {stats["sys_fix"]} 条')
            print(f'  → assistant 格式统一: {stats["asst_fix"]} 条')
            if stats['mixed_task']:
                print(f'  ⚠ System prompt 中混合了单阶段/双阶段任务: {stats["mixed_task"]} 条')

    print(f'\n修复完成！原文件已备份为 .bak')

    # 修复后验证
    print()
    print('=' * 70)
    print('修复后验证')
    print('=' * 70)
    for fname in FILES:
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stats = validate_fixed(fpath)
        issues = []
        if stats['assistant_dict']:
            issues.append(f'assistant dict 格式: {stats["assistant_dict"]} 条(应为0)')
        if stats['dpo_chosen_bad']:
            issues.append(f'DPO chosen 异常: {stats["dpo_chosen_bad"]} 条(应为0)')
        if stats['has_messages'] and not stats['has_system']:
            issues.append('无 system prompt')

        status = '✓' if not issues else '⚠ ' + '; '.join(issues)
        print(f'  {fname}: {status}')


if __name__ == '__main__':
    main()
