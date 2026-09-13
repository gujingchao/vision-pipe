# Models

Drop YOLO-exported ONNX weights here (or pass any path via `--model` / API `model=`).

## Expected ONNX contract

| | |
| --- | --- |
| **Input** | `float32` NCHW `(1, 3, S, S)`, RGB, values in `[0, 1]` (letterboxed; `S` usually 640) |
| **Output** | YOLO-ish tensor, any of: |
| | `(1, N, 5+C)` — `cx, cy, w, h, obj?, class_scores...` (YOLOv5-style) |
| | `(1, 5+C, N)` — same layout transposed (YOLOv8-style) |
| | `(1, N, 4+C)` — `cx, cy, w, h, class_scores...` (no separate obj) |

Boxes are decoded in letterbox pixel space, then mapped back to the original image.

## Example

```bash
# place weights
cp ~/Downloads/yolov8n.onnx /path/to/vision-pipe/models/

# CLI (after pip install -e .)
vision-pipe infer --input examples/images --output out \
  --model models/yolov8n.onnx --conf 0.25 --save-json

# API
curl -F file=@frame.jpg -F conf=0.25 \
  "http://127.0.0.1:8090/infer?model=models/yolov8n.onnx"
```

If the `.onnx` path is missing or `onnxruntime` is not installed, the core library
falls back to the classical OpenCV blob detector (no weights required).

Large `.onnx` files are gitignored — do not commit them.
