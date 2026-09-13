"""ONNX Runtime YOLO-style adapter (best-effort output decoding)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vision_pipe.adapters.base import DetectorAdapter
from vision_pipe.postprocess import scale_boxes_to_original
from vision_pipe.preprocess import prepare_onnx_input

# Expected ONNX I/O (document for model authors):
#   Input:  float32 NCHW, shape (1, 3, S, S), RGB, values in [0, 1]
#           (S typically 640; letterboxed).
#   Output (any of):
#     - (1, N, 5+C):  cx, cy, w, h, obj?, class_scores...  (YOLOv5-ish)
#     - (1, 5+C, N):  same layout transposed (YOLOv8-ish)
#     - (1, N, 4+C):  cx, cy, w, h, class_scores...         (no obj)
# Boxes are in letterbox pixel space; we remap to original image coords.


def onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_yolo_output(
    raw: np.ndarray,
    conf: float,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Decode YOLO-ish tensor → (boxes_xyxy[N,4], scores[N], labels[N]).

    Boxes stay in model/letterbox pixel coordinates.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported ONNX output shape: {getattr(raw, 'shape', None)}")

    # Prefer channels-last candidates: (N, attrs) over (attrs, N)
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
        # likely (attrs, N) e.g. (84, 8400)
        arr = arr.T

    n_attrs = arr.shape[1]
    if n_attrs < 5:
        raise ValueError(f"YOLO output attrs too small: {n_attrs}")

    boxes_cxcywh = arr[:, :4]
    rest = arr[:, 4:]

    if rest.shape[1] == 1:
        # single objectness / score column
        scores = rest[:, 0]
        class_ids = np.zeros(len(scores), dtype=np.int64)
    elif rest.shape[1] >= 2:
        # Heuristic: if first column looks like objectness and rest are classes
        # (YOLOv5): obj * class. Else treat all as class scores (YOLOv8).
        if rest.shape[1] > 2:
            obj = rest[:, 0]
            cls = rest[:, 1:]
            # If obj is mostly in [0,1] and cls max also [0,1], use v5 formula
            if float(np.nanmean(obj)) <= 1.5 and float(np.nanmean(np.max(cls, axis=1))) <= 1.5:
                class_ids = np.argmax(cls, axis=1)
                scores = obj * cls[np.arange(len(cls)), class_ids]
            else:
                class_ids = np.argmax(rest, axis=1)
                scores = rest[np.arange(len(rest)), class_ids]
        else:
            class_ids = np.argmax(rest, axis=1)
            scores = rest[np.arange(len(rest)), class_ids]
    else:
        scores = np.ones(arr.shape[0], dtype=np.float32)
        class_ids = np.zeros(arr.shape[0], dtype=np.int64)

    # If scores look like logits, squash
    if float(np.nanmax(scores)) > 1.5 or float(np.nanmin(scores)) < 0.0:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))

    mask = scores >= conf
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes_cxcywh.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )

    cx, cy, bw, bh = boxes_cxcywh.T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(np.float32)

    labels: list[str] = []
    for cid in class_ids:
        if class_names and 0 <= int(cid) < len(class_names):
            labels.append(class_names[int(cid)])
        else:
            labels.append(f"class_{int(cid)}")
    return xyxy, scores.astype(np.float32), labels


class OnnxYoloAdapter(DetectorAdapter):
    """Run a YOLO-exported ONNX model via onnxruntime."""

    name = "onnx_yolo"

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        class_names: list[str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if not onnx_available():
            raise ImportError("onnxruntime is not installed")
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.input_size = input_size
        self.class_names = class_names
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        # Infer size from model if square
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
            self.input_size = int(shape[2])

    def detect(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[dict[str, Any]]:
        tensor, meta = prepare_onnx_input(image_bgr, size=self.input_size)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0]
        boxes, scores, labels = decode_yolo_output(raw, conf=conf, class_names=self.class_names)
        boxes = scale_boxes_to_original(boxes, meta)
        dets: list[dict[str, Any]] = []
        for i in range(len(scores)):
            dets.append(
                {
                    "xyxy": [float(x) for x in boxes[i].tolist()],
                    "score": float(scores[i]),
                    "label": labels[i],
                }
            )
        return dets

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]
