#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 PascalVOC 的 .xml 转为 YOLO .txt（class 0 = person）。
默认：lables/raw -> labels/raw（与 split_train_val 一致）

用法:
  python tools_yolo_week1/voc_xml_to_yolo.py
  python tools_yolo_week1/voc_xml_to_yolo.py --src data/yolo/lables/raw --dst data/yolo/labels/raw
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO = os.path.join(ROOT, "data", "yolo")

# VOC 里常见类名 -> YOLO class_id（你 data.yaml 只有 person=0）
NAME_TO_ID = {
    "person": 0,
    "people": 0,
}


def voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h):
    bw = xmax - xmin
    bh = ymax - ymin
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0
    return (
        xc / w,
        yc / h,
        bw / w,
        bh / h,
    )


def convert_one(xml_path: str, out_txt: str) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return 0
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        return 0

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        cid = NAME_TO_ID.get(name)
        if cid is None:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        xc, yc, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    if not lines:
        return 0
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(YOLO, "lables", "raw"))
    ap.add_argument("--dst", default=os.path.join(YOLO, "labels", "raw"))
    ap.add_argument("--copy-txt", action="store_true", default=True,
                    help="同时复制已有 .txt 到 dst（默认开启）")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        print(f"[ERR] 源目录不存在: {src}")
        return 1
    os.makedirs(dst, exist_ok=True)

    n_xml, n_txt_copy, n_conv = 0, 0, 0
    for fn in os.listdir(src):
        base, ext = os.path.splitext(fn)
        ext = ext.lower()
        sp = os.path.join(src, fn)
        if ext == ".xml":
            out = os.path.join(dst, base + ".txt")
            n = convert_one(sp, out)
            if n:
                n_xml += 1
        elif ext == ".txt" and args.copy_txt:
            # 已有 YOLO txt：直接复制（不覆盖更大文件可手改）
            import shutil
            dp = os.path.join(dst, fn)
            if os.path.getsize(sp) > 0:
                shutil.copy2(sp, dp)
                n_txt_copy += 1

    print(f"[DONE] xml->yolo: {n_xml}  files | copied txt: {n_txt_copy}")
    print(f"       output dir: {dst}")
    non_empty = sum(
        1 for f in os.listdir(dst)
        if f.endswith(".txt") and os.path.getsize(os.path.join(dst, f)) > 0
    )
    print(f"       non-empty txt in dst: {non_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)