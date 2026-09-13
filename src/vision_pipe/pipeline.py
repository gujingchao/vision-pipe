"""Public inference API used by CLI and FastAPI service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_pipe.adapters.base import DetectorAdapter
from vision_pipe.adapters.opencv_blob import OpenCVBlobAdapter
from vision_pipe.adapters.onnx_yolo import OnnxYoloAdapter, onnx_available
from vision_pipe.metrics import Timer
from vision_pipe.postprocess import finalize_detections
from vision_pipe.preprocess import load_bgr


@dataclass
class InferResult:
    """Structured detection result (CLI accepts ``.detections`` or a bare list)."""

    detections: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    backend: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_adapter(model: str | None) -> DetectorAdapter:
    """Prefer ONNX when path exists and onnxruntime is importable; else blob."""
    if model:
        path = Path(model)
        if path.is_file() and path.suffix.lower() == ".onnx" and onnx_available():
            return OnnxYoloAdapter(path)
    return OpenCVBlobAdapter()


def infer_image(
    path: str,
    model: str | None = None,
    conf: float = 0.25,
    *,
    iou_thresh: float = 0.45,
    adapter: DetectorAdapter | None = None,
) -> InferResult:
    """
    Run detection on an image path.

    Returns :class:`InferResult` with ``detections`` as
    ``[{xyxy:[x1,y1,x2,y2], score:float, label:str}, ...]`` plus stage timings.
    """
    timer = Timer()
    with timer.measure("total"):
        with timer.measure("preprocess"):
            image_bgr = load_bgr(path)
            height, width = image_bgr.shape[:2]

        det = adapter or _select_adapter(model)
        with timer.measure("infer"):
            raw = det.detect(image_bgr, conf=conf)

        with timer.measure("postprocess"):
            detections = finalize_detections(
                raw,
                conf=conf,
                iou_thresh=iou_thresh,
                width=width,
                height=height,
            )

    timings = timer.as_dict()
    timings["backend"] = getattr(det, "name", "unknown")
    return InferResult(
        detections=detections,
        timings_ms=timings,
        width=width,
        height=height,
        backend=str(getattr(det, "name", "unknown")),
    )


def infer_ndarray(
    image_bgr: np.ndarray,
    model: str | None = None,
    conf: float = 0.25,
    *,
    iou_thresh: float = 0.45,
    adapter: DetectorAdapter | None = None,
) -> InferResult:
    """Same as :func:`infer_image` but for an in-memory BGR ndarray (API / WS)."""
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        raise ValueError("image_bgr must be a numpy ndarray")
    timer = Timer()
    with timer.measure("total"):
        with timer.measure("preprocess"):
            height, width = image_bgr.shape[:2]

        det = adapter or _select_adapter(model)
        with timer.measure("infer"):
            raw = det.detect(image_bgr, conf=conf)

        with timer.measure("postprocess"):
            detections = finalize_detections(
                raw,
                conf=conf,
                iou_thresh=iou_thresh,
                width=width,
                height=height,
            )

    timings = timer.as_dict()
    timings["backend"] = getattr(det, "name", "unknown")
    return InferResult(
        detections=detections,
        timings_ms=timings,
        width=int(width),
        height=int(height),
        backend=str(getattr(det, "name", "unknown")),
    )


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes + labels onto a copy of ``image_bgr``; return BGR image."""
    out = image_bgr.copy()
    for det in detections:
        xyxy = det.get("xyxy") or det.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        label = str(det.get("label", "object"))
        score = float(det.get("score", 0.0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {score:.2f}"
        cv2.putText(
            out,
            text,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


# Alias used in the layout docstring.
draw_overlay = draw_detections
