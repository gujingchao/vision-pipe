# vision-pipe

边缘友好的机器视觉推理流水线（开源中）。

| 目录 | 说明 |
| --- | --- |
| `src/vision_pipe/` | 核心库：OpenCV 预处理 → ONNX / blob 检测 → NMS / 画框（selyla） |
| `api/` | FastAPI + WebSocket 推理服务，默认端口 **8090**（selyla） |
| `models/` | 放置 YOLO ONNX 权重（说明见 `models/README.md`） |
| `web/` | 看流 / 画框 / FPS 面板（wayly） |
| `cli/` | 批量推理 CLI、JSON/画框导出（herry） |
| `bench/` | 延迟基准（p50 / p95 / FPS） |
| `docker/` | 可复现运行环境 |

计划仓库：`https://github.com/gujingchao/vision-pipe`。

## Core install

```bash
# from repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "from vision_pipe.pipeline import infer_image; print(infer_image)"
```

`pip install -e .` 把 `src/vision_pipe` 装成可导入包。CLI 会自动发现
`vision_pipe.pipeline.infer_image` 并走真推理；未安装时仍用 dummy。

行为摘要：

1. OpenCV 读图 + letterbox / normalize
2. 若 `model` 指向存在的 `.onnx` 且已装 `onnxruntime` → ONNX YOLO 适配器
3. 否则 → `opencv_blob` 经典轮廓检测（**无需权重**，CI/demo 可用）
4. 置信度过滤 + NMS
5. `draw_detections` / `draw_overlay` 画框
6. 分段耗时：`preprocess` / `infer` / `postprocess` / `total`（ms）

## API

```bash
pip install -r api/requirements.txt
uvicorn app.main:app --app-dir api --host 0.0.0.0 --port 8090
```

- `GET /health`
- `POST /infer` — multipart 图片 → JSON detections + timings
- `WS /ws/stream` — 二进制或 base64 帧流；详见 [`api/README.md`](api/README.md)

## CLI

```bash
cd cli
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: also install core from repo root so CLI uses real backend
pip install -e ".."
vision-pipe infer --input ../examples/images --output ../out --save-json --save-overlay
vision-pipe bench --input ../examples/images --repeat 20 --warmup 3
```

核心引擎未接入时，CLI 走确定性 dummy 检测器，方便测试和 CI。接上 `vision_pipe.pipeline.infer_image` 后自动切到真推理。

检测结果 JSON：

```json
{
  "image": "sample.jpg",
  "width": 64,
  "height": 64,
  "detections": [{"xyxy": [8, 8, 56, 56], "score": 0.9, "label": "object"}],
  "timings_ms": {"total": 1.2}
}
```

## Docker

```bash
docker build -f docker/Dockerfile -t vision-pipe .
docker run --rm -v $PWD/examples/images:/data -v $PWD/out:/out vision-pipe infer --input /data --output /out --save-json
```

## 开发 / 测试

```bash
# core + api
pip install -e ".[dev]"
pytest src/vision_pipe/tests api/tests -q

# CLI only
cd cli && ruff check . && pytest
```

CI：`.github/workflows/ci.yml`（lint + CLI 测试 + bench 冒烟）。

## License

MIT
