# vision-pipe web（wayly）

Vue 3 + TypeScript + Vite 看流面板：WebSocket 推帧、画框叠加、FPS / 延迟指标、配置下发，以及可选的 `POST /infer` 单次推理。

## 前置

先启动 API（默认 `8090`），见仓库根目录与 [`api/README.md`](../api/README.md)：

```bash
# repo root
source .venv/bin/activate   # 或按 api README 安装
uvicorn app.main:app --app-dir api --host 0.0.0.0 --port 8090
```

## 安装与运行

```bash
cd web
npm install
npm run dev
```

浏览器打开 Vite 提示的地址（默认 `http://127.0.0.1:5173`）。

生产构建：

```bash
npm run build
npm run preview
```

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `VITE_API_HTTP` | `http://127.0.0.1:8090` | API HTTP 根地址；WebSocket 自动推导为 `ws(s)://<host>/ws/stream` |

复制 `.env.example` 为 `.env` 后修改即可。

## 功能

1. **连接** — 可改 API host/port，显示连接状态与推导出的 WS URL  
2. **实时流 (WS)** — 摄像头 / 上传图片或视频 / 图片 URL；JPEG 二进制帧优先（`canvas.toBlob`）  
3. **画框** — 叠加 `xyxy` 检测框 + label + score  
4. **指标** — 服务端 `fps`，以及 `timings_ms`（preprocess / infer / postprocess / total）条形图  
5. **配置** — `{type:"config", conf, model}`，接收 `config_ack`  
6. **单次推理** — `POST /infer` multipart（无摄像头的 headless / 演示场景）

协议细节见 [`api/README.md`](../api/README.md) 的 WebSocket `/ws/stream` 小节。

## 无摄像头说明

Headless 或浏览器拒绝 `getUserMedia` 时，「摄像头」按钮会禁用；请使用 **上传图片/视频** 或 **图片 URL**，或切到 **单次推理 (HTTP)** 标签。
