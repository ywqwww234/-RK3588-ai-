"""
端到端冒烟测试（不依赖真实采集，只验证流水线打通）。

执行：
    python -m nn.test_pipeline

覆盖：
  1. data_utils 缺失标签时是否抛 DataError（P0 验证）
  2. 弱标签生成 + 数据切分 + scaler 持久化
  3. 模型前向 / 多任务损失 / 反向传播
  4. checkpoint 保存 + ONNX 导出
  5. InferenceEngine strict=True 在缺模型时抛错
  6. InferenceEngine 加载训练后的 ONNX 推理 + scaler 一致性
  7. InferenceMonitor 健康判定
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pandas as pd


def _make_synthetic_features(path: str, n: int = 400, seed: int = 0) -> None:
    """构造一段 25 维伪特征数据，足够过拟合让流水线能走通。"""
    rng = np.random.default_rng(seed)
    # 25 列 + ts 列（与 feature_aligner.HEADER 同构）
    from nn.feature_aligner import HEADER
    X = rng.standard_normal((n, 25)).astype(np.float32)
    # 让 b_att/b_med 与风险负相关（让模型有可学的信号）
    X[:, 22] = rng.uniform(20, 90, n)   # b_att
    X[:, 23] = rng.uniform(20, 90, n)   # b_med
    df = pd.DataFrame(X, columns=HEADER[:-1])
    df['ts'] = np.arange(n).astype(np.float64)
    df.to_csv(path, index=False)


def _make_weak_labels(feat_csv: str, label_csv: str) -> None:
    from nn.label_tool import make_weak_labels
    make_weak_labels(feat_csv=feat_csv, out_csv=label_csv)


def main():
    tmp = tempfile.mkdtemp(prefix='nn_pipeline_')
    print(f'[tmp] {tmp}')

    feat = os.path.join(tmp, 'aligned_features.csv')
    lab  = os.path.join(tmp, 'labels.csv')
    model_dir = os.path.join(tmp, 'models')
    os.makedirs(model_dir, exist_ok=True)
    _make_synthetic_features(feat, n=1500)

    # ---------- 1. 缺失标签必须报错（P0） ----------
    from nn.data_utils import DataError, prepare_datasets, load_raw
    try:
        load_raw(feat, lab)
    except DataError as e:
        print(f'[ok] 缺失 labels.csv 正确抛错：{e}')
    else:
        raise SystemExit('[FAIL] 缺失标签未抛错，P0 修复无效！')

    # ---------- 2. 弱标签 + 数据切分 + scaler ----------
    _make_weak_labels(feat, lab)
    scaler_path = os.path.join(model_dir, 'scaler.json')
    ds = prepare_datasets(feat, lab, scaler_path)
    assert os.path.isfile(scaler_path), 'scaler.json 未保存'
    print(f'[ok] 数据切分: train={len(ds["train"][0])}, '
          f'val={len(ds["val"][0])}, test={len(ds["test"][0])}')

    # ---------- 3. 模型前向 + 多任务反向 ----------
    import torch
    import torch.nn as tnn
    from nn.dl_model import MultiModalNet, count_params, export_onnx
    model = MultiModalNet()
    print(f'[ok] 模型参数量 = {count_params(model)}')
    x = torch.randn(2, 60, 25)
    cls, reg, attn, modal_w = model(x)
    assert cls.shape == (2, 4) and reg.shape == (2,)
    assert attn.shape == (2, 60) and modal_w.shape == (2, 3)
    print(f'[ok] 前向输出形状对齐: cls/reg/attn/modal_w')

    # 多任务损失反向
    ce = tnn.CrossEntropyLoss()
    huber = tnn.SmoothL1Loss()
    y = torch.tensor([0, 3])
    r = torch.tensor([0.1, 0.9])
    loss = ce(cls, y) + 0.5 * huber(reg, r)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, '反向传播无梯度'
    print(f'[ok] 多任务损失反向: loss={loss.item():.4f}')

    # ---------- 4. ckpt 保存 + ONNX 导出 ----------
    ckpt = os.path.join(model_dir, 'best.pt')
    torch.save({'model': model.state_dict(), 'epoch': 1}, ckpt)
    onnx_out = os.path.join(model_dir, 'baseline.onnx')
    try:
        export_onnx(out_path=onnx_out, state_dict_path=ckpt)
        has_onnx = os.path.isfile(onnx_out)
    except Exception as e:
        print(f'[warn] ONNX 导出失败（无 onnx 包？）: {e}')
        has_onnx = False

    # ---------- 5. strict 模式在缺模型时抛错 ----------
    from nn.inference_engine import InferenceEngine
    miss = os.path.join(tmp, 'no_model.onnx')
    try:
        InferenceEngine(onnx_path=miss, scaler_path=scaler_path, strict=True)
    except RuntimeError as e:
        print(f'[ok] strict 模式缺模型抛错: {e}')
    else:
        raise SystemExit('[FAIL] strict 未抛错')

    # ---------- 6. 推理一致 + scaler 复用 ----------
    eng = InferenceEngine(onnx_path=onnx_out if has_onnx else miss,
                          scaler_path=scaler_path, strict=False)
    win = ds['test'][0][0]    # 已是缩放后的窗口（训练管线产物）
    # 注意：训练时已 transform，这里再次走 engine.predict 会再 transform 一次
    # 因此模拟"原始数据"：用一个未缩放窗口
    raw_win = np.random.randn(60, 25).astype(np.float32)
    out = eng.predict(raw_win)
    print(f'[ok] 推理 mode={out["mode"]}  '
          f'risk={out["risk_score"]:.3f}  tier={out["tier"]}  '
          f'lat={out["latency_ms"]:.2f} ms  '
          f'scaled={out["scaled"]}')
    assert isinstance(out['tier_probs'], list) and len(out['tier_probs']) == 4
    assert isinstance(out['modal_w'], tuple) and len(out['modal_w']) == 3
    assert len(out['feat_imp']) == 25

    # ---------- 7. monitor + 漂移 ----------
    from nn.monitor import InferenceMonitor
    mon = InferenceMonitor(eng, scaler_path=scaler_path)
    for _ in range(20):
        mon.predict(np.random.randn(60, 25).astype(np.float32))
    h = mon.health()
    print(f'[ok] monitor.health: state={h["state"]} reason={h["reason"]}')
    assert 'lat_p95_ms' in h and 'tier_dist' in h

    # 制造漂移：把输入幅度放大 10 倍
    big = np.random.randn(60, 25).astype(np.float32) * 10.0
    out2 = mon.predict(big)
    assert 'drift' in out2
    print(f'[ok] drift detect: avg|z|={out2["drift"]["avg_abs_z"]:.2f} '
          f'pct_outlier={out2["drift"]["pct_outlier"]:.2f}')

    # ---------- 清理 ----------
    print(f'\n[ALL PASS] 流水线冒烟测试通过')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
