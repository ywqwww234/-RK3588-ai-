"""对齐 nn/aligned_features.csv 与 nn/labels.csv 行数（取较短长度截断较长文件）。"""
import os
import shutil
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGNED = os.path.join(ROOT, "nn", "aligned_features.csv")
LABELS = os.path.join(ROOT, "nn", "labels.csv")


def main():
    fa = pd.read_csv(ALIGNED)
    lb = pd.read_csv(LABELS)
    na, nl = len(fa), len(lb)
    print(f"[INFO] aligned={na:,}  labels={nl:,}")
    if na == nl:
        print("[OK] 已对齐，无需修改")
        return
    n = min(na, nl)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if na > nl:
        shutil.copy2(ALIGNED, ALIGNED + f".bak_{ts}")
        fa.iloc[:n].to_csv(ALIGNED, index=False, encoding="utf-8-sig")
        print(f"[FIX] 已截断 aligned_features -> {n:,} 行，备份 *.bak_{ts}")
    if nl > na:
        shutil.copy2(LABELS, LABELS + f".bak_{ts}")
        lb.iloc[:n].to_csv(LABELS, index=False, encoding="utf-8-sig")
        print(f"[FIX] 已截断 labels -> {n:,} 行，备份 *.bak_{ts}")
    fa2 = pd.read_csv(ALIGNED)
    lb2 = pd.read_csv(LABELS)
    print(f"[VERIFY] aligned={len(fa2):,} labels={len(lb2):,}")


if __name__ == "__main__":
    main()