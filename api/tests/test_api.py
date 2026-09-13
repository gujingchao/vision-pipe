"""API TestClient tests — no network downloads."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _png_bytes(w: int = 48, h: int = 36) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    img[8:28, 10:35] = (30, 30, 220)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "vision-pipe"


def test_infer_multipart(client: TestClient):
    data = _png_bytes()
    r = client.post(
        "/infer",
        files={"file": ("frame.png", io.BytesIO(data), "image/png")},
        params={"conf": 0.25},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 48
    assert body["height"] == 36
    assert isinstance(body["detections"], list)
    assert body["detections"]
    assert "timings_ms" in body
    assert "total" in body["timings_ms"]
    det = body["detections"][0]
    assert "xyxy" in det and "score" in det and "label" in det


def test_infer_bad_bytes(client: TestClient):
    r = client.post(
        "/infer",
        files={"file": ("bad.bin", io.BytesIO(b"not-an-image"), "application/octet-stream")},
    )
    assert r.status_code == 400
