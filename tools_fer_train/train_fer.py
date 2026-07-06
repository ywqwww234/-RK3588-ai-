#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8 类 FER 微调，导出与线上一致的 mean128 + 64x64 灰度输入。

  pip install torch torchvision
  python tools_fer_train/train_fer.py --data data/fer_labeled --epochs 60 --batch 32

数据目录:
  data/fer_labeled/train/{neutral,happy,...}/
  data/fer_labeled/val/{...}/

输出:
  runs/fer/best.pt
  runs/fer/best.onnx  (可用 export_onnx.py 再导一遍)
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLASSES = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def preprocess_gray64(img_gray):
    import cv2

    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(img_gray, (64, 64)).astype(np.float32)
    x = (x - 128.0) / 128.0
    return x[np.newaxis, ...]  # 1,64,64


class FerFolderDataset:
    def __init__(self, root: str, augment: bool = True):
        import cv2

        self.cv2 = cv2
        self.augment = augment
        self.samples = []
        for ci, cname in enumerate(CLASSES):
            d = os.path.join(root, cname)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if os.path.splitext(fn.lower())[1] in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((os.path.join(d, fn), ci))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        path, ci = self.samples[idx]
        img = self.cv2.imread(path, self.cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if self.augment:
            if random.random() < 0.5:
                img = self.cv2.flip(img, 1)
            if random.random() < 0.3:
                img = np.clip(img.astype(np.float32) * random.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        x = preprocess_gray64(img)
        return torch.from_numpy(x).float(), ci


def build_model():
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    try:
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        backbone = mobilenet_v3_small(weights=None)
    old = backbone.features[0][0]
    backbone.features[0][0] = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        backbone.features[0][0].weight[:] = old.weight.mean(dim=1, keepdim=True)
    in_f = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_f, len(CLASSES))
    return backbone


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "fer_labeled"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "fer"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_root = os.path.join(args.data, "train")
    val_root = os.path.join(args.data, "val")
    if not os.path.isdir(train_root):
        print(f"[ERR] 缺少 {train_root}，先运行 prepare_dataset.py")
        return 1

    train_ds = FerFolderDataset(train_root, augment=True)
    if len(train_ds) < 80:
        print(f"[WARN] 训练样本仅 {len(train_ds)}，建议每类≥300。继续训练可能过拟合。")

    val_ds = FerFolderDataset(val_root, augment=False) if os.path.isdir(val_root) else None
    if val_ds is None or len(val_ds) == 0:
        # 从 train 划 15% 做 val
        n = len(train_ds.samples)
        n_val = max(1, int(n * 0.15))
        val_samples = train_ds.samples[:n_val]
        train_ds.samples = train_ds.samples[n_val:]
        val_ds = FerFolderDataset(train_root, augment=False)
        val_ds.samples = val_samples
        print(f"[INFO] 自动划分 val={len(val_ds)} train={len(train_ds)}")

    labels = [s[1] for s in train_ds.samples]
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64) + 1e-6
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    os.makedirs(args.out, exist_ok=True)
    model = build_model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()
        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                vc += (logits.argmax(1) == y).sum().item()
                vt += x.size(0)
        va = vc / max(1, vt)
        print(f"epoch {ep}/{args.epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  val_acc={va:.3f}")
        if va >= best_acc:
            best_acc = va
            path = os.path.join(args.out, "best.pt")
            torch.save({"model": model.state_dict(), "classes": CLASSES, "val_acc": va}, path)
            print(f"  -> saved {path}")

    print(f"[DONE] best val_acc={best_acc:.3f}")
    print("下一步: python tools_fer_train/export_onnx.py --ckpt runs/fer/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)