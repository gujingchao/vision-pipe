"""OpenCV letterbox / resize / normalize helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    """Metadata needed to map boxes from model space back to original image."""

    orig_width: int
    orig_height: int
    model_size: int
    ratio: float
    pad_w: float
    pad_h: float


def load_bgr(path: str) -> np.ndarray:
    """Load an image as BGR uint8. Supports common formats including PPM."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return img


def letterbox(
    image_bgr: np.ndarray,
    size: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    """
    Resize with aspect ratio preserved and pad to a square ``size x size``.

    Returns the padded BGR image and mapping metadata.
    """
    h, w = image_bgr.shape[:2]
    ratio = min(size / max(h, 1), size / max(w, 1))
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_w = (size - new_w) / 2.0
    pad_h = (size - new_h) / 2.0
    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    meta = LetterboxMeta(
        orig_width=w,
        orig_height=h,
        model_size=size,
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
    )
    return canvas, meta


def to_nchw_float(
    image_bgr: np.ndarray,
    *,
    swap_rb: bool = True,
    scale: float = 1.0 / 255.0,
) -> np.ndarray:
    """Convert HxWxC uint8 BGR to float32 NCHW tensor (batch=1)."""
    img = image_bgr.astype(np.float32) * scale
    if swap_rb:
        img = img[:, :, ::-1]  # BGR -> RGB
    chw = np.transpose(img, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def prepare_onnx_input(
    image_bgr: np.ndarray,
    size: int = 640,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Letterbox + normalize for YOLO-style ONNX models (1x3xSxS float32 RGB)."""
    boxed, meta = letterbox(image_bgr, size=size)
    tensor = to_nchw_float(boxed, swap_rb=True, scale=1.0 / 255.0)
    return tensor, meta
