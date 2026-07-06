#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 自定义后处理：适配 [1,5,8400] 单输出格式（非拆分检测头）。
单类 (person) 版本，为 RKNN 部署优化。

与 ELF2 官方示例 func_yolov8_optimize 的逻辑等价，但输入格式不同。
官方示例: 9 个输出分支 (3 scales x 3 branches: reg + cls + score_sum)
本模块:   1 个输出 [1,5,8400] (bbox[4] + cls[1], 3 scales concatenated)

用法:
  from yolo_postprocess import yolov8_post_process, draw_boxes
  boxes, classes, scores = yolov8_post_process(output_array, conf_thres=0.25, iou_thres=0.45)
"""

from __future__ import annotations

import cv2
import numpy as np

# ── 常量 ──────────────────────────────────────────────
IMG_SIZE = 640
CLASSES = ["person"]


def yolov8_post_process(
    output: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> tuple:
    """
    YOLOv8 后处理 — 适配 ultralytics 8.x 导出格式。

    ONNX/RKNN 输出已经过 DFL decode + dist2bbox，输出为:
      [x1, x2, y1, y2] 或 [x1, y1, x2, y2] (640×640 像素坐标)
      + class_score

    本函数使用 min/max 鲁棒处理各种通道顺序。

    Args:
        output: (1, 5, 8400) 或 (5, 8400)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值

    Returns:
        (boxes, classes, scores) — 均为 640×640 像素空间
    """
    if output.ndim == 3:
        output = output[0]  # (5, 8400)

    bbox_raw = output[:4].T  # (8400, 4)
    cls_scores = output[4]   # (8400,)

    # ONNX/RKNN 输出格式: [cx, cy, w, h] 像素坐标 (640×640 空间)
    # 来源: ultralytics export → dist2bbox(xywh=True) → concat
    cx = bbox_raw[:, 0]
    cy = bbox_raw[:, 1]
    bw = bbox_raw[:, 2]
    bh = bbox_raw[:, 3]
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    # 置信度过滤
    mask = cls_scores > conf_thres
    if not mask.any():
        return None, None, None

    boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=-1)
    scores = cls_scores[mask]
    classes = np.zeros(mask.sum(), dtype=np.int32)

    # 按分数排序 + 限制
    order = np.argsort(scores)[::-1][:max_det]
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]

    # NMS
    keep = _nms(boxes, scores, iou_thres)
    if len(keep) == 0:
        return None, None, None

    return boxes[keep], classes[keep], scores[keep]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """单类 NMS。"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def letterbox(
    im: np.ndarray,
    new_shape: tuple = (640, 640),
    color: tuple = (114, 114, 114),
) -> tuple:
    """等比例缩放 + padding。返回 (img_padded, ratio, (dw, dh))"""
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = new_shape[1] - new_w
    dh = new_shape[0] - new_h
    dw //= 2
    dh //= 2

    if (new_w, new_h) != (w, h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def scale_boxes(boxes: np.ndarray, orig_shape: tuple, ratio: tuple, padding: tuple) -> np.ndarray:
    """将 640x640 空间的框映射回原图坐标。"""
    gain = min(ratio[0], ratio[1])
    pad_w, pad_h = padding
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= gain
    # Clip to image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_shape[0])
    return boxes


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    class_names: list = None,
    ratio: tuple = None,
    padding: tuple = None,
) -> np.ndarray:
    """在图像上画出检测框。"""
    if class_names is None:
        class_names = CLASSES

    img_out = img.copy()
    if boxes is None:
        return img_out

    if ratio is not None and padding is not None:
        boxes = scale_boxes(boxes.copy(), img.shape, ratio, padding)

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        cname = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"{cname} {score:.2f}"
        cv2.putText(img_out, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_out


# ── 自测 ──────────────────────────────────────────────
if __name__ == "__main__":
    import os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Test with ONNX
    import onnxruntime as ort

    onnx_path = os.path.join(ROOT, "models", "yolov8_classroom.onnx")
    if not os.path.exists(onnx_path):
        print(f"[ERR] ONNX not found: {onnx_path}")
        exit(1)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # Find a test image
    img_dir = os.path.join(ROOT, "data", "yolo", "images", "val")
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    if not imgs:
        print("[ERR] No test images")
        exit(1)

    for fn in imgs[:3]:
        img_path = os.path.join(img_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pad, ratio, padding = letterbox(img_rgb, (640, 640))
        img_norm = img_pad.astype(np.float32) / 255.0
        img_nchw = img_norm.transpose(2, 0, 1).reshape(1, 3, 640, 640)

        # Inference
        out = sess.run([], {in_name: img_nchw})[0]
        boxes, classes, scores = yolov8_post_process(out, conf_thres=0.35)

        # Draw
        result = draw_boxes(img, boxes, scores, classes, ratio=ratio, padding=padding)
        out_path = os.path.join(ROOT, "data", "yolo", f"_test_post_{fn}")
        cv2.imwrite(out_path, result)
        n = 0 if boxes is None else len(boxes)
        print(f"  {fn}: {n} person(s) detected -> {out_path}")
