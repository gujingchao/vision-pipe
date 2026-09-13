"""Inference backend: real vision_pipe package if present, else dummy."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass
class Detection:
    xyxy: list[float]
    score: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferResult:
    image: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "detections": [d.to_dict() for d in self.detections],
            "timings_ms": self.timings_ms,
        }


def _image_size(path: Path) -> tuple[int, int]:
    try:
        cv2 = import_module("cv2")
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    # PPM P6 header, or dummy square
    data = path.read_bytes()
    if data.startswith(b"P6"):
        header, _, rest = data.partition(b"\n")
        while rest.startswith(b"#"):
            _, _, rest = rest.partition(b"\n")
        dims, _, _ = rest.partition(b"\n")
        w_s, h_s = dims.split()
        return int(w_s), int(h_s)
    return 64, 64


def dummy_infer(path: Path, conf: float = 0.25) -> InferResult:
    started = time.perf_counter()
    width, height = _image_size(path)
    margin = min(width, height) * 0.125
    det = Detection(
        xyxy=[margin, margin, width - margin, height - margin],
        score=0.9,
        label="object",
    )
    detections = [det] if det.score >= conf else []
    total = (time.perf_counter() - started) * 1000
    return InferResult(
        image=path.name,
        width=width,
        height=height,
        detections=detections,
        timings_ms={"total": round(total, 3), "backend": "dummy"},
    )


def _real_infer(path: Path, model: str | None, conf: float) -> InferResult | None:
    try:
        pipeline = import_module("vision_pipe.pipeline")
    except ImportError:
        return None
    infer_image = getattr(pipeline, "infer_image", None)
    if infer_image is None:
        return None
    started = time.perf_counter()
    raw = infer_image(str(path), model=model, conf=conf)
    total = (time.perf_counter() - started) * 1000
    width, height = _image_size(path)
    dets: list[Detection] = []
    items = raw if isinstance(raw, list) else getattr(raw, "detections", [])
    for item in items:
        if isinstance(item, dict):
            dets.append(
                Detection(
                    xyxy=list(item.get("xyxy") or item.get("bbox") or [0, 0, 0, 0]),
                    score=float(item.get("score", 0)),
                    label=str(item.get("label", "object")),
                )
            )
    return InferResult(
        image=path.name,
        width=width,
        height=height,
        detections=dets,
        timings_ms={"total": round(total, 3), "backend": "vision_pipe"},
    )


def infer(path: Path, model: str | None = None, conf: float = 0.25) -> InferResult:
    real = _real_infer(path, model, conf)
    if real is not None:
        return real
    return dummy_infer(path, conf=conf)
