# Latency bench

```bash
vision-pipe bench --input ../examples/images --repeat 50 --warmup 5 --output ./bench.json
```

输出 `mean_ms` / `p50_ms` / `p95_ms` / `fps`。核心引擎接入后，同一命令会走 ONNX Runtime。
