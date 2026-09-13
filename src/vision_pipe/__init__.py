"""vision_pipe core library: preprocess → detect → NMS → overlay."""

from vision_pipe.pipeline import InferResult, draw_detections, draw_overlay, infer_image

__version__ = "0.1.0"
__all__ = [
    "InferResult",
    "draw_detections",
    "draw_overlay",
    "infer_image",
    "__version__",
]
