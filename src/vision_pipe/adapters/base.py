"""Adapter interface for detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DetectorAdapter(ABC):
    """Run detection on a BGR image; return raw list-of-dict detections."""

    name: str = "base"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[dict[str, Any]]:
        """Return detections as ``{xyxy, score, label}`` in original image coords."""

    def close(self) -> None:
        """Release resources if any."""
