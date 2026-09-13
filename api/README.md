# vision-pipe API

FastAPI service wrapping the `vision_pipe` core library. Default port **8090**.

## Install & run

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core lib (src/vision_pipe)
pip install -r api/requirements.txt
# or: pip install -e "./api"

uvicorn app.main:app --app-dir api --host 0.0.0.0 --port 8090
```

Health check:

```bash
curl http://127.0.0.1:8090/health
```

## HTTP

### `GET /health`

```json
{"status": "ok", "service": "vision-pipe", "port": 8090}
```

### `POST /infer`

Multipart form field `file` (image). Optional query params: `conf` (float, default 0.25), `model` (ONNX path or omit for OpenCV blob fallback).

```bash
curl -s -F file=@examples/images/sample.ppm \
  "http://127.0.0.1:8090/infer?conf=0.25" | jq .
```

Response (same detection schema as CLI JSON):

```json
{
  "width": 64,
  "height": 64,
  "detections": [{"xyxy": [8, 8, 56, 56], "score": 0.9, "label": "object"}],
  "timings_ms": {
    "preprocess": 0.4,
    "infer": 1.2,
    "postprocess": 0.1,
    "total": 1.7,
    "backend": "opencv_blob"
  }
}
```

CORS is open (`*`).

## WebSocket `/ws/stream` (for wayly)

Connect to `ws://127.0.0.1:8090/ws/stream`.

### Client → server

1. **Binary frame** — raw JPEG or PNG bytes.
2. **JSON frame** (base64):
   ```json
   {"type": "frame", "data": "<base64-encoded JPEG/PNG>"}
   ```
3. **Config** (optional, anytime):
   ```json
   {"type": "config", "conf": 0.25, "model": null}
   ```
   Server replies:
   ```json
   {"type": "config_ack", "conf": 0.25, "model": null}
   ```

### Server → client

On each successful frame:

```json
{
  "type": "result",
  "width": 640,
  "height": 480,
  "detections": [{"xyxy": [x1, y1, x2, y2], "score": 0.91, "label": "blob"}],
  "timings_ms": {"preprocess": 0.5, "infer": 2.0, "postprocess": 0.2, "total": 2.7, "backend": "opencv_blob"},
  "fps": 28.4
}
```

`fps` is an exponential moving average (α=0.2) across frames on that socket.

Errors:

```json
{"type": "error", "message": "..."}
```

## Tests

```bash
# from repo root, with core + api deps installed
pytest tests api/tests -q
```
