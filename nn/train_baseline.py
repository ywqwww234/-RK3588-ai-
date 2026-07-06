"""
基线训练脚本 v2（生产可用骨架）。

变更对比 v1（按改进清单实施）：
  P0 ✓ 禁止随机标签：缺失 labels.csv 直接报错
  P0 ✓ 多任务损失：cls (CE，类权重) + λ * reg (Huber)
  P0 ✓ 推理一致：训练保存 scaler.json，与推理共用
  P1 ✓ checkpoint + early stopping + best model
  P1 ✓ 训练日志：每 epoch 写 train/val 的 loss/acc/f1/auc 到 train_log.csv
  P1 ✓ train/val/test 三段时序切分（避免时间泄漏）
  P2 ✓ 类不平衡：CE 类权重
  扩展 ✓ 支持 --resume 续训；--export-only 仅从 ckpt 导出 ONNX

用法:
    python -m nn.train_baseline --epochs 30 --batch 64 --lambda-reg 0.5
    python -m nn.train_baseline --model rf
    python -m nn.train_baseline --export-only --resume nn/models/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from .data_utils import (
    DataError, prepare_datasets, class_weights, derive_reg_from_label,
    WIN, FEAT_DIM, N_CLASSES,
)


# 与 inference_engine 共用的产出物路径
CSV_PATH      = 'nn/aligned_features.csv'
LABEL_CSV     = 'nn/labels.csv'
MODEL_DIR     = 'nn/models'
ONNX_PATH     = os.path.join(MODEL_DIR, 'baseline.onnx')
SCALER_PATH   = os.path.join(MODEL_DIR, 'scaler.json')
BEST_CKPT     = os.path.join(MODEL_DIR, 'best.pt')
LAST_CKPT     = os.path.join(MODEL_DIR, 'last.pt')
TRAIN_LOG     = os.path.join(MODEL_DIR, 'train_log.csv')
META_PATH     = os.path.join(MODEL_DIR, 'meta.json')


# ---------------- 指标 ----------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = N_CLASSES) -> float:
    f1s = []
    for c in range(n_class):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def macro_auc_ovr(y_true: np.ndarray, prob: np.ndarray, n_class: int = N_CLASSES) -> float:
    """One-vs-Rest 宏 AUC（无 sklearn 依赖的近似实现）。"""
    aucs = []
    for c in range(n_class):
        y = (y_true == c).astype(np.int32)
        s = prob[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        # rank-based AUC
        order = np.argsort(-s)
        y_sorted = y[order]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        # 累加正样本前的负样本数
        cum_neg = 0
        wins = 0
        for v in y_sorted:
            if v == 1:
                wins += cum_neg
            else:
                cum_neg += 1
        aucs.append(wins / max(1, n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float('nan')


# ---------------- RF 基线（不变，仅修复随机标签） ----------------

def train_rf(feat_csv: str = CSV_PATH, label_csv: str = LABEL_CSV):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    ds = prepare_datasets(feat_csv, label_csv, SCALER_PATH)
    X_tr, y_tr, _ = ds['train']
    X_va, y_va, _ = ds['val']
    X_te, y_te, _ = ds['test']
    Xtr = X_tr.reshape(len(X_tr), -1)
    Xva = X_va.reshape(len(X_va), -1)
    Xte = X_te.reshape(len(X_te), -1)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                                 class_weight='balanced')
    clf.fit(Xtr, y_tr)
    print('--- val ---')
    print(classification_report(y_va, clf.predict(Xva), zero_division=0))
    print('--- test ---')
    print(classification_report(y_te, clf.predict(Xte), zero_division=0))


# ---------------- DL 训练 ----------------

def _make_loaders(ds, batch: int):
    def to_dl(split, shuffle):
        X, y, r = ds[split]
        if r is None:
            r = derive_reg_from_label(y)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y).long()
        rt = torch.from_numpy(r).float()
        return DataLoader(TensorDataset(Xt, yt, rt),
                          batch_size=batch, shuffle=shuffle)
    return to_dl('train', True), to_dl('val', False), to_dl('test', False)


def _eval(model, dl, ce, huber, lam: float, device):
    model.eval()
    losses, ys, ps, probs = [], [], [], []
    regs_t, regs_p = [], []
    with torch.no_grad():
        for xb, yb, rb in dl:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            cls, reg, *_ = model(xb)
            l_cls = ce(cls, yb)
            l_reg = huber(reg, rb)
            loss = l_cls + lam * l_reg
            losses.append(loss.item())
            prob = torch.softmax(cls, dim=1).cpu().numpy()
            probs.append(prob)
            ps.append(prob.argmax(axis=1))
            ys.append(yb.cpu().numpy())
            regs_t.append(rb.cpu().numpy())
            regs_p.append(reg.cpu().numpy())
    if not ys:
        return {'loss': float('nan'), 'acc': float('nan'),
                'f1': float('nan'), 'auc': float('nan'), 'mae': float('nan')}
    y = np.concatenate(ys); p = np.concatenate(ps); pr = np.concatenate(probs)
    rt = np.concatenate(regs_t); rp = np.concatenate(regs_p)
    return {
        'loss': float(np.mean(losses)),
        'acc':  float((p == y).mean()),
        'f1':   macro_f1(y, p),
        'auc':  macro_auc_ovr(y, pr),
        'mae':  float(np.mean(np.abs(rt - rp))),
    }


def train_dl(epochs: int = 30, batch: int = 64, lr: float = 1e-3,
             lam_reg: float = 0.5, patience: int = 6, device: str = 'cpu',
             resume: Optional[str] = None, export_only: bool = False,
             feat_csv: str = CSV_PATH, label_csv: str = LABEL_CSV):
    if not TORCH_OK:
        raise RuntimeError('PyTorch 不可用')

    from .dl_model import MultiModalNet, export_onnx

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # 仅导出（从 checkpoint 还原后导 ONNX，确保推理用训练好的权重）
    if export_only:
        ckpt = resume or BEST_CKPT
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f'未找到 checkpoint: {ckpt}')
        export_onnx(out_path=ONNX_PATH, state_dict_path=ckpt)
        print(f'[ok] export-only 完成 -> {ONNX_PATH}')
        return

    ds = prepare_datasets(feat_csv, label_csv, SCALER_PATH)
    tr_dl, va_dl, te_dl = _make_loaders(ds, batch)

    # 类权重（处理不平衡）
    cw = class_weights(ds['train'][1])
    print(f'[info] class_weights = {cw.tolist()}')

    model = MultiModalNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).to(device))
    huber = nn.SmoothL1Loss()

    start_ep = 0
    best_metric = -math.inf
    bad_epochs = 0

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'opt' in ckpt:
            opt.load_state_dict(ckpt['opt'])
        start_ep = int(ckpt.get('epoch', 0))
        best_metric = float(ckpt.get('best_metric', -math.inf))
        print(f'[resume] from {resume} @ epoch {start_ep}, best={best_metric:.4f}')

    # 训练日志
    new_log = not os.path.isfile(TRAIN_LOG)
    flog = open(TRAIN_LOG, 'a', newline='', encoding='utf-8')
    wlog = csv.writer(flog)
    if new_log:
        wlog.writerow(['epoch', 'lr',
                       'tr_loss', 'tr_acc', 'tr_f1', 'tr_auc', 'tr_mae',
                       'va_loss', 'va_acc', 'va_f1', 'va_auc', 'va_mae',
                       'time_s'])

    for ep in range(start_ep, epochs):
        t0 = time.time()
        model.train()
        tr_losses = []
        tr_ys, tr_ps, tr_probs = [], [], []
        tr_rt, tr_rp = [], []
        for xb, yb, rb in tr_dl:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            cls, reg, *_ = model(xb)
            l_cls = ce(cls, yb)
            l_reg = huber(reg, rb)
            loss = l_cls + lam_reg * l_reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
            with torch.no_grad():
                pr = torch.softmax(cls, dim=1).cpu().numpy()
                tr_probs.append(pr)
                tr_ps.append(pr.argmax(axis=1))
                tr_ys.append(yb.cpu().numpy())
                tr_rt.append(rb.cpu().numpy())
                tr_rp.append(reg.detach().cpu().numpy())
        sched.step()

        y = np.concatenate(tr_ys); p = np.concatenate(tr_ps); pr = np.concatenate(tr_probs)
        rt = np.concatenate(tr_rt); rp = np.concatenate(tr_rp)
        tr = {
            'loss': float(np.mean(tr_losses)),
            'acc':  float((p == y).mean()),
            'f1':   macro_f1(y, p),
            'auc':  macro_auc_ovr(y, pr),
            'mae':  float(np.mean(np.abs(rt - rp))),
        }
        va = _eval(model, va_dl, ce, huber, lam_reg, device)

        # early stopping 监控量：val_f1 + 0.5*(1-val_mae)（兼顾分类与回归）
        cur_metric = va['f1'] + 0.5 * (1.0 - va['mae'])

        dt = time.time() - t0
        print(f'ep {ep+1:3d}/{epochs}  '
              f'tr loss={tr["loss"]:.4f} acc={tr["acc"]:.3f} f1={tr["f1"]:.3f} '
              f'| va loss={va["loss"]:.4f} acc={va["acc"]:.3f} '
              f'f1={va["f1"]:.3f} auc={va["auc"]:.3f} mae={va["mae"]:.3f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  '
              f'{dt:.1f}s')
        wlog.writerow([ep + 1, opt.param_groups[0]['lr'],
                       tr['loss'], tr['acc'], tr['f1'], tr['auc'], tr['mae'],
                       va['loss'], va['acc'], va['f1'], va['auc'], va['mae'],
                       round(dt, 2)])
        flog.flush()

        # checkpoint
        ckpt = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'epoch': ep + 1,
            'best_metric': max(best_metric, cur_metric),
            'val_metrics': va,
        }
        torch.save(ckpt, LAST_CKPT)
        if cur_metric > best_metric:
            best_metric = cur_metric
            bad_epochs = 0
            torch.save(ckpt, BEST_CKPT)
            print(f'  [best] metric={best_metric:.4f} -> {BEST_CKPT}')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'[early-stop] no improvement for {patience} epochs')
                break

    flog.close()

    # 测试集评估
    if os.path.isfile(BEST_CKPT):
        ckpt = torch.load(BEST_CKPT, map_location=device)
        model.load_state_dict(ckpt['model'])
    te = _eval(model, te_dl, ce, huber, lam_reg, device)
    print('--- test ---')
    print(json.dumps(te, indent=2))

    # 元信息（推理引擎可以读取核对版本）
    meta = {
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epochs_run': ep + 1,
        'best_val_metric': best_metric,
        'test_metrics': te,
        'feat_dim': FEAT_DIM,
        'win': WIN,
        'n_classes': N_CLASSES,
        'lambda_reg': lam_reg,
        'class_weights': cw.tolist(),
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 用 best.pt 导出 ONNX
    from .dl_model import export_onnx
    export_onnx(out_path=ONNX_PATH, state_dict_path=BEST_CKPT)


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dl', choices=['rf', 'dl'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-reg', type=float, default=0.5,
                    dest='lam_reg', help='回归损失权重 λ')
    ap.add_argument('--patience', type=int, default=6,
                    help='early stopping patience')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--resume', default=None)
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--feat', default=CSV_PATH, help='aligned_features.csv 路径')
    ap.add_argument('--label', default=LABEL_CSV, help='labels.csv 路径')
    args = ap.parse_args()
    try:
        if args.model == 'rf':
            train_rf(feat_csv=args.feat, label_csv=args.label)
        else:
            train_dl(epochs=args.epochs, batch=args.batch, lr=args.lr,
                     lam_reg=args.lam_reg, patience=args.patience,
                     device=args.device, resume=args.resume,
                     export_only=args.export_only,
                     feat_csv=args.feat, label_csv=args.label)
    except DataError as e:
        print(f'[FATAL] 数据校验失败：{e}')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
