"""FastAPI + WebSocket inference service (default port 8090)."""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vision_pipe.metrics import EmaFps
from vision_pipe.pipeline import InferResult, infer_ndarray

DEFAULT_CONF = 0.25
DEFAULT_MODEL: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.conf = DEFAULT_CONF
    app.state.model = DEFAULT_MODEL
    yield


app = FastAPI(title="vision-pipe API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _result_payload(result: InferResult, fps: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "width": result.width,
        "height": result.height,
        "detections": result.detections,
        "timings_ms": result.timings_ms,
    }
    if fps is not None:
        payload["fps"] = round(float(fps), 3)
    return payload


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vision-pipe", "port": 8090}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    conf: float = DEFAULT_CONF,
    model: str | None = None,
) -> JSONResponse:
    data = await file.read()
    try:
        image = _decode_image_bytes(data)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    use_model = model if model not in (None, "", "null") else getattr(app.state, "model", None)
    use_conf = conf if conf is not None else getattr(app.state, "conf", DEFAULT_CONF)
    result = infer_ndarray(image, model=use_model, conf=float(use_conf))
    return JSONResponse(_result_payload(result))


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """
    WebSocket protocol (for wayly frontend):

    Client → server:
      - binary frame: JPEG/PNG bytes
      - JSON text: ``{"type":"frame","data":"<base64>"}``
      - JSON text: ``{"type":"config","conf":0.25,"model":null}``

    Server → client:
      - JSON text: ``{"type":"result","width":W,"height":H,
         "detections":[...],"timings_ms":{...},"fps":<ema>}``
      - JSON text: ``{"type":"error","message":"..."}``
      - JSON text: ``{"type":"config_ack","conf":...,"model":...}``
    """
    await websocket.accept()
    conf = float(getattr(app.state, "conf", DEFAULT_CONF))
    model = getattr(app.state, "model", DEFAULT_MODEL)
    ema = EmaFps(alpha=0.2)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            image: np.ndarray | None = None
            if message.get("bytes") is not None:
                try:
                    image = _decode_image_bytes(message["bytes"])
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
            elif message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "invalid JSON"})
                    continue
                msg_type = payload.get("type")
                if msg_type == "config":
                    if "conf" in payload and payload["conf"] is not None:
                        conf = float(payload["conf"])
                    if "model" in payload:
                        model = payload["model"]
                        if model in ("", "null"):
                            model = None
                    await websocket.send_json(
                        {"type": "config_ack", "conf": conf, "model": model}
                    )
                    continue
                if msg_type == "frame":
                    try:
                        raw = base64.b64decode(payload.get("data") or "")
                        image = _decode_image_bytes(raw)
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json(
                            {"type": "error", "message": f"frame decode failed: {exc}"}
                        )
                        continue
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"unknown type: {msg_type}"}
                    )
                    continue
            else:
                continue

            result = infer_ndarray(image, model=model, conf=conf)
            fps = ema.tick()
            body = _result_payload(result, fps=fps)
            body["type"] = "result"
            await websocket.send_json(body)
    except WebSocketDisconnect:
        return
