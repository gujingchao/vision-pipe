"""Detector adapters (ONNX YOLO, classical OpenCV blob fallback)."""

from vision_pipe.adapters.base import DetectorAdapter
from vision_pipe.adapters.opencv_blob import OpenCVBlobAdapter
from vision_pipe.adapters.onnx_yolo import OnnxYoloAdapter, onnx_available

__all__ = [
    "DetectorAdapter",
    "OnnxYoloAdapter",
    "OpenCVBlobAdapter",
    "onnx_available",
]
