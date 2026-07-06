"""
深度学习模型：1D-CNN + BiGRU + 时间注意力 + 模态门控（多模态融合，参数 < 200K）。

输入: [B, 60, 25]   (60 帧 1Hz 滑窗，25 维三模态特征)
头 A: 风险四档分类 (4 logits → softmax)
头 B: 风险趋势回归 (1 标量, 0~1)

副输出（统一解释口径）:
    attn_t  : [B, 60]  时间注意力（softmax over 时间维）
    modal_w : [B, 3]   模态门控权重（softmax over vision/physio/brain）
    feat_imp: [B, 25]  最后一帧特征级显著度（用于 UI 25 列细粒度热力）

设计要点:
  * GRU 不用 LSTM：参数 ~25% 少，RKNN 量化更稳
  * 不用 Transformer：60 帧序列短，self-attn 收益小，NPU 不友好
  * 三模态独立 Conv 编码后再门控融合，区分"时间 attn"与"模态贡献 modal_w"
  * dropout + layernorm 提供正则化与训练稳定性
  * 全部线性 / 卷积 / GRU 都被 RKNN-toolkit 原生支持

导出:
    python -m nn.dl_model export   # 写出 nn/models/baseline.onnx
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_SEQ = 60
INPUT_DIM = 25
N_CLASSES = 4    # 四档风险

# 三模态切片（与 feature_aligner.HEADER 对齐）
VISION_DIM = 9
PHYSIO_DIM = 5
BRAIN_DIM = 11


class _ModalEncoder(nn.Module):
    """单模态时序编码器：1D-Conv x2 + LayerNorm + Dropout。"""

    def __init__(self, in_dim, hid=16, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hid, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hid, hid, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = h.transpose(1, 2)            # [B, T, hid]
        h = self.ln(h)
        h = self.drop(h)
        return h


class MultiModalNet(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, seq=INPUT_SEQ, hidden=64,
                 n_class=N_CLASSES, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.seq = seq

        # 三路独立编码（解耦"模态贡献"语义）
        self.enc_v = _ModalEncoder(VISION_DIM, hid=16, dropout=dropout)
        self.enc_p = _ModalEncoder(PHYSIO_DIM, hid=16, dropout=dropout)
        self.enc_b = _ModalEncoder(BRAIN_DIM, hid=16, dropout=dropout)

        # 模态门控：把每路全局表征压成一个权重（softmax over 3）
        self.gate_v = nn.Linear(16, 1)
        self.gate_p = nn.Linear(16, 1)
        self.gate_b = nn.Linear(16, 1)

        # 拼接后过 BiGRU 做长时建模
        self.gru = nn.GRU(48, hidden, batch_first=True,
                          bidirectional=True, dropout=0.0)
        self.gru_drop = nn.Dropout(dropout)

        # 时间注意力（softmax over T）
        self.attn_fc = nn.Linear(hidden * 2, 1)

        # 双头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, return_feat_imp: bool = False):
        # x: [B, T=60, D=25]
        xv = x[:, :, :VISION_DIM]
        xp = x[:, :, VISION_DIM:VISION_DIM + PHYSIO_DIM]
        xb = x[:, :, VISION_DIM + PHYSIO_DIM:]

        hv = self.enc_v(xv)             # [B, T, 16]
        hp = self.enc_p(xp)
        hb = self.enc_b(xb)

        # 模态门控：用每路时间均值 → 单标量 → softmax(3)
        gv = self.gate_v(hv.mean(dim=1))   # [B, 1]
        gp = self.gate_p(hp.mean(dim=1))
        gb = self.gate_b(hb.mean(dim=1))
        gates = torch.cat([gv, gp, gb], dim=1)        # [B, 3]
        modal_w = F.softmax(gates, dim=1)             # [B, 3]

        # 把模态权重广播到各自时间帧后拼接（让模型学得到模态门控）
        hv = hv * modal_w[:, 0:1].unsqueeze(1)
        hp = hp * modal_w[:, 1:2].unsqueeze(1)
        hb = hb * modal_w[:, 2:3].unsqueeze(1)
        h = torch.cat([hv, hp, hb], dim=2)            # [B, T, 48]

        out, _ = self.gru(h)                          # [B, T, 128]
        out = self.gru_drop(out)

        # 时间注意力
        a = self.attn_fc(out).squeeze(-1)             # [B, T]
        a = F.softmax(a, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), out).squeeze(1)  # [B, 128]

        cls = self.cls_head(ctx)                      # [B, 4]
        reg = torch.sigmoid(self.reg_head(ctx)).squeeze(-1)  # [B]

        if return_feat_imp:
            # 训练/调试用：最后一帧的 25 维特征显著度（梯度近似的轻量替代）
            # 取每模态的 |hidden| 作为本模态强度，反向均匀分配到原始特征
            mag_v = hv[:, -1, :].abs().mean(dim=1, keepdim=True).expand(-1, VISION_DIM)
            mag_p = hp[:, -1, :].abs().mean(dim=1, keepdim=True).expand(-1, PHYSIO_DIM)
            mag_b = hb[:, -1, :].abs().mean(dim=1, keepdim=True).expand(-1, BRAIN_DIM)
            feat_imp = torch.cat([mag_v, mag_p, mag_b], dim=1)
            feat_imp = feat_imp / (feat_imp.sum(dim=1, keepdim=True) + 1e-6)
            return cls, reg, a, modal_w, feat_imp
        return cls, reg, a, modal_w


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def export_onnx(out_path='nn/models/baseline.onnx', state_dict_path=None):
    """导出 ONNX。可选传入训练好的 state_dict 路径。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = MultiModalNet().eval()
    if state_dict_path and os.path.isfile(state_dict_path):
        try:
            sd = torch.load(state_dict_path, map_location='cpu')
            if isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            model.load_state_dict(sd, strict=False)
            print(f'[ok] loaded weights from {state_dict_path}')
        except Exception as e:
            print(f'[warn] failed to load weights: {e}')
    dummy = torch.randn(1, INPUT_SEQ, INPUT_DIM)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['x'],
        output_names=['cls', 'reg', 'attn', 'modal_w'],
        opset_version=13,
        dynamic_axes={'x': {0: 'B'}, 'cls': {0: 'B'},
                      'reg': {0: 'B'}, 'attn': {0: 'B'},
                      'modal_w': {0: 'B'}},
        dynamo=False,
    )
    print(f'[ok] exported -> {out_path}  ({count_params(model)} params)')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        sd = sys.argv[2] if len(sys.argv) > 2 else None
        export_onnx(state_dict_path=sd)
    else:
        m = MultiModalNet()
        print('params:', count_params(m))
        cls, reg, a, mw = m(torch.randn(2, INPUT_SEQ, INPUT_DIM))
        print('cls', cls.shape, 'reg', reg.shape,
              'attn', a.shape, 'modal_w', mw.shape)
