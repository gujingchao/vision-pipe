"""NMS, confidence filter, and coordinate remapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from vision_pipe.preprocess import LetterboxMeta


def clip_xyxy(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    x1 = float(max(0.0, min(float(width), x1)))
    y1 = float(max(0.0, min(float(height), y1)))
    x2 = float(max(0.0, min(float(width), x2)))
    y2 = float(max(0.0, min(float(height), y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta,
) -> np.ndarray:
    """Map letterboxed model-space xyxy boxes back to original image coordinates."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4)
    out = boxes_xyxy.astype(np.float64).copy()
    out[:, [0, 2]] -= meta.pad_w
    out[:, [1, 3]] -= meta.pad_h
    out /= max(meta.ratio, 1e-9)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.orig_width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.orig_height)
    return out


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(
    detections: list[dict[str, Any]],
    iou_thresh: float = 0.45,
) -> list[dict[str, Any]]:
    """Greedy class-agnostic NMS on list-of-dict detections."""
    if not detections:
        return []
    order = sorted(range(len(detections)), key=lambda i: float(detections[i]["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(detections[i])
        box_i = np.asarray(detections[i]["xyxy"], dtype=np.float64)
        for j in order:
            if j == i or j in suppressed:
                continue
            box_j = np.asarray(detections[j]["xyxy"], dtype=np.float64)
            if box_iou_xyxy(box_i, box_j) >= iou_thresh:
                suppressed.add(j)
    return kept


def filter_confidence(
    detections: list[dict[str, Any]],
    conf: float,
) -> list[dict[str, Any]]:
    return [d for d in detections if float(d.get("score", 0.0)) >= conf]


def finalize_detections(
    detections: list[dict[str, Any]],
    *,
    conf: float,
    iou_thresh: float = 0.45,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, Any]]:
    """Confidence filter → NMS → optional clip to image bounds."""
    dets = filter_confidence(detections, conf)
    dets = nms(dets, iou_thresh=iou_thresh)
    if width is not None and height is not None:
        for d in dets:
            d["xyxy"] = clip_xyxy(list(d["xyxy"]), width, height)
            d["score"] = float(d["score"])
            d["label"] = str(d.get("label", "object"))
    return dets
