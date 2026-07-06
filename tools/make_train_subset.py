"""从全量 nn 数据随机抽 N 行，供 16GB 内存本训练。"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NN = os.path.join(ROOT, "nn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fa = os.path.join(NN, "aligned_features.csv")
    lb = os.path.join(NN, "labels.csv")
    f = pd.read_csv(fa)
    l = pd.read_csv(lb)
    if len(f) != len(l):
        raise SystemError(f"行数不一致: feat={len(f)} label={len(l)}，先运行 tools/sync_nn_pair.py")

    n = min(args.rows, len(f))
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(f), size=n, replace=False))

    out_f = os.path.join(NN, "aligned_features_sub.csv")
    out_l = os.path.join(NN, "labels_sub.csv")
    f.iloc[idx].to_csv(out_f, index=False, encoding="utf-8-sig")
    l.iloc[idx].to_csv(out_l, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_f}  {out_l}  rows={n}")
    print(l.iloc[idx]["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()