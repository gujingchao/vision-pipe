"""Unit tests for preprocess, NMS, and fallback infer_image (no network)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_pipe.pipeline import draw_detections, infer_image, infer_ndarray
from vision_pipe.postprocess import box_iou_xyxy, finalize_detections, nms
from vision_pipe.preprocess import letterbox, prepare_onnx_input, to_nchw_float


def _write_ppm(path: Path, w: int = 64, h: int = 48, rgb: tuple[int, int, int] = (220, 40, 40)) -> Path:
    header = f"P6\n{w} {h}\n255\n".encode()
    path.write_bytes(header + bytes(rgb) * (w * h))
    return path


def _write_png_with_blob(path: Path, w: int = 80, h: int = 60) -> Path:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    # Red saturated rectangle (BGR)
    img[15:45, 20:55] = (40, 40, 220)
    cv2.imwrite(str(path), img)
    return path


def test_letterbox_and_normalize():
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    boxed, meta = letterbox(img, size=64)
    assert boxed.shape == (64, 64, 3)
    assert meta.orig_width == 80 and meta.orig_height == 40
    assert meta.ratio == pytest.approx(64 / 80)
    tensor = to_nchw_float(boxed)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    onnx_in, meta2 = prepare_onnx_input(img, size=32)
    assert onnx_in.shape == (1, 3, 32, 32)
    assert meta2.model_size == 32


def test_nms_and_conf_filter():
    dets = [
        {"xyxy": [0, 0, 10, 10], "score": 0.9, "label": "a"},
        {"xyxy": [1, 1, 11, 11], "score": 0.8, "label": "a"},  # overlaps
        {"xyxy": [50, 50, 60, 60], "score": 0.7, "label": "b"},
        {"xyxy": [0, 0, 5, 5], "score": 0.1, "label": "c"},  # low conf
    ]
    assert box_iou_xyxy(np.array(dets[0]["xyxy"]), np.array(dets[1]["xyxy"])) > 0.5
    kept = finalize_detections(dets, conf=0.25, iou_thresh=0.45, width=100, height=100)
    labels = {d["label"] for d in kept}
    assert "c" not in labels
    assert len(kept) == 2  # one of the overlapping pair + the far box
    assert nms(dets[:2], iou_thresh=0.45)[0]["score"] == 0.9


def test_infer_image_ppm_fallback(tmp_path: Path):
    path = _write_ppm(tmp_path / "solid.ppm")
    result = infer_image(str(path), model=None, conf=0.25)
    assert result.backend == "opencv_blob"
    assert result.width == 64 and result.height == 48
    assert result.detections
    d = result.detections[0]
    assert "xyxy" in d and "score" in d and "label" in d
    assert len(d["xyxy"]) == 4
    assert d["score"] >= 0.25
    for key in ("preprocess", "infer", "postprocess", "total"):
        assert key in result.timings_ms


def test_infer_image_png_blob(tmp_path: Path):
    path = _write_png_with_blob(tmp_path / "blob.png")
    result = infer_image(str(path), conf=0.25)
    assert result.detections
    # At least one detection should cover the red blob region roughly
    assert any(d["xyxy"][2] > d["xyxy"][0] for d in result.detections)


def test_draw_detections(tmp_path: Path):
    path = _write_png_with_blob(tmp_path / "draw.png")
    img = cv2.imread(str(path))
    dets = [{"xyxy": [10, 10, 40, 40], "score": 0.88, "label": "blob"}]
    out = draw_detections(img, dets)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_infer_ndarray_and_cli_contract(tmp_path: Path):
    path = _write_ppm(tmp_path / "c.ppm", w=32, h=24)
    img = cv2.imread(str(path))
    result = infer_ndarray(img, conf=0.25)
    # CLI accepts object with .detections as list of dicts
    items = result.detections
    assert isinstance(items, list)
    assert all(isinstance(x, dict) for x in items)
