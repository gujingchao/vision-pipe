"""Image listing and overlay / JSON export."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from vision_pipe_cli.engine import InferResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"input not found: {source}")
    files = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images in {source}")
    return files


def write_json(result: InferResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
    dest.write_text(payload, encoding="utf-8")


def write_overlay(image_path: Path, result: InferResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cv2 = import_module("cv2")
    except ImportError:
        dest.write_bytes(image_path.read_bytes())
        return
    img = cv2.imread(str(image_path))
    if img is None:
        dest.write_bytes(image_path.read_bytes())
        return
    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{det.label} {det.score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(dest), img)
