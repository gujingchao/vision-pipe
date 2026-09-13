<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import MetricsPanel from './components/MetricsPanel.vue'
import VideoStage from './components/VideoStage.vue'
import { useCapture } from './composables/useCapture'
import { useWebSocket } from './composables/useWebSocket'
import { getApiHttp, getInferUrl, getWsStreamUrl } from './env'
import type { Detection, TimingsMs } from './types'
import { drawOverlay } from './utils/draw'

const apiHttp = ref(getApiHttp())
const wsUrl = computed(() => getWsStreamUrl(apiHttp.value))

const {
  status,
  lastError,
  lastResult,
  configAck,
  connect,
  disconnect,
  sendConfig,
  sendFrameBlob,
} = useWebSocket(wsUrl)

const capture = useCapture()

const conf = ref(0.25)
const model = ref('')
const streaming = ref(false)
const targetFps = ref(10)
const imageUrl = ref('')
const oneshotBusy = ref(false)
const oneshotError = ref<string | null>(null)
const tab = ref<'stream' | 'oneshot'>('stream')

const hiddenVideo = ref<HTMLVideoElement | null>(null)
const captureCanvas = ref<HTMLCanvasElement | null>(null)
const stageRef = ref<InstanceType<typeof VideoStage> | null>(null)

const sourceCanvasRef = ref<HTMLCanvasElement | null>(null)
const overlayDets = ref<Detection[]>([])
const resultW = ref<number | null>(null)
const resultH = ref<number | null>(null)
const displayFps = ref<number | null>(null)
const displayTimings = ref<TimingsMs | null>(null)

let streamTimer: ReturnType<typeof setInterval> | null = null
let inFlight = false
let imageLoop = false

const statusLabel = computed(() => {
  switch (status.value) {
    case 'connected':
      return '已连接'
    case 'connecting':
      return '连接中…'
    case 'error':
      return '错误'
    default:
      return '未连接'
  }
})

const statusClass = computed(() => status.value)

watch(lastResult, (r) => {
  if (!r) return
  overlayDets.value = r.detections ?? []
  resultW.value = r.width
  resultH.value = r.height
  displayFps.value = r.fps ?? null
  displayTimings.value = r.timings_ms ?? null
  // redraw overlay on new result
  nextTick(() => stageRef.value?.redraw())
})

function applyConfig() {
  const m = model.value.trim() || null
  sendConfig(conf.value, m)
}

async function ensureConnected(): Promise<boolean> {
  if (status.value !== 'connected') {
    connect()
  }
  for (let i = 0; i < 40; i++) {
    const s: string = status.value
    if (s === 'connected') {
      applyConfig()
      return true
    }
    if (s === 'error') return false
    await new Promise((r) => setTimeout(r, 50))
  }
  const final: string = status.value
  if (final === 'connected') {
    applyConfig()
    return true
  }
  return false
}

async function tickFrame() {
  if (inFlight || !streaming.value) return
  const canvas = captureCanvas.value
  if (!canvas) return
  if (!capture.drawSourceToCanvas(canvas)) return
  sourceCanvasRef.value = canvas
  inFlight = true
  try {
    const blob = await capture.canvasToJpegBlob(canvas, 0.75)
    if (blob && streaming.value) {
      await sendFrameBlob(blob, true)
    }
  } finally {
    inFlight = false
  }
  // keep drawing live preview even while waiting for server
  stageRef.value?.redraw()
}

function startStreamLoop() {
  stopStreamLoop()
  streaming.value = true
  const ms = Math.max(50, Math.round(1000 / Math.max(1, targetFps.value)))
  streamTimer = setInterval(() => {
    void tickFrame()
  }, ms)
  void tickFrame()
}

function stopStreamLoop() {
  streaming.value = false
  if (streamTimer) {
    clearInterval(streamTimer)
    streamTimer = null
  }
  inFlight = false
}

async function onConnectToggle() {
  if (status.value === 'connected' || status.value === 'connecting') {
    stopStreamLoop()
    disconnect()
  } else {
    connect()
    // send config once connected
    const ok = await ensureConnected()
    if (ok) applyConfig()
  }
}

async function onStartWebcam() {
  const video = hiddenVideo.value
  if (!video) return
  const ok = await capture.startWebcam(video)
  if (!ok) return
  const conn = await ensureConnected()
  if (!conn) return
  applyConfig()
  startStreamLoop()
}

async function onFileChange(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return

  const isVideo = file.type.startsWith('video/')
  const video = hiddenVideo.value
  if (isVideo && video) {
    const ok = await capture.loadVideoFile(file, video, true)
    if (!ok) return
    const conn = await ensureConnected()
    if (!conn) return
    applyConfig()
    startStreamLoop()
    return
  }

  // image: single-shot or loop via stream
  const img = await capture.loadImageFile(file)
  if (!img) return
  const canvas = captureCanvas.value
  if (!canvas) return
  capture.drawSourceToCanvas(canvas)
  sourceCanvasRef.value = canvas
  overlayDets.value = []
  stageRef.value?.redraw()

  if (tab.value === 'oneshot') {
    await runOneshotBlob(await capture.canvasToJpegBlob(canvas, 0.9))
    return
  }

  const conn = await ensureConnected()
  if (!conn) return
  applyConfig()
  // for still image, send once then optionally loop
  imageLoop = true
  startStreamLoop()
}

async function onLoadUrl() {
  const url = imageUrl.value.trim()
  if (!url) return
  const img = await capture.loadImageUrl(url)
  if (!img) return
  const canvas = captureCanvas.value
  if (!canvas) return
  capture.drawSourceToCanvas(canvas)
  sourceCanvasRef.value = canvas
  overlayDets.value = []
  stageRef.value?.redraw()

  if (tab.value === 'oneshot') {
    await runOneshotBlob(await capture.canvasToJpegBlob(canvas, 0.9))
    return
  }

  const conn = await ensureConnected()
  if (!conn) return
  applyConfig()
  imageLoop = true
  startStreamLoop()
}

async function runOneshotBlob(blob: Blob | null) {
  oneshotError.value = null
  if (!blob) {
    oneshotError.value = '无法编码图片'
    return
  }
  oneshotBusy.value = true
  try {
    const form = new FormData()
    form.append('file', blob, 'frame.jpg')
    const m = model.value.trim() || null
    const url = getInferUrl(apiHttp.value, conf.value, m)
    const res = await fetch(url, { method: 'POST', body: form })
    const data = await res.json()
    if (!res.ok || data.error) {
      oneshotError.value = data.error || `HTTP ${res.status}`
      return
    }
    overlayDets.value = data.detections ?? []
    resultW.value = data.width ?? null
    resultH.value = data.height ?? null
    displayFps.value = null
    displayTimings.value = data.timings_ms ?? null
    // draw overlay locally
    const canvas = captureCanvas.value
    const display = stageRef.value?.displayCanvas
    if (canvas && display) {
      drawOverlay(display, canvas, overlayDets.value, resultW.value ?? undefined, resultH.value ?? undefined)
    } else {
      stageRef.value?.redraw()
    }
  } catch (e) {
    oneshotError.value = e instanceof Error ? e.message : String(e)
  } finally {
    oneshotBusy.value = false
  }
}

async function onOneshotFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  tab.value = 'oneshot'
  stopStreamLoop()
  const img = await capture.loadImageFile(file)
  if (!img) return
  const canvas = captureCanvas.value
  if (!canvas) return
  capture.drawSourceToCanvas(canvas)
  sourceCanvasRef.value = canvas
  await runOneshotBlob(file.type.startsWith('image/') ? file : await capture.canvasToJpegBlob(canvas, 0.9))
}

function onStop() {
  stopStreamLoop()
  imageLoop = false
  capture.stop()
  sourceCanvasRef.value = null
  overlayDets.value = []
}

function onStopStreamOnly() {
  stopStreamLoop()
  imageLoop = false
}

onUnmounted(() => {
  stopStreamLoop()
})
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="brand">
        <span class="logo">◎</span>
        <div>
          <h1>vision-pipe</h1>
          <p class="sub">看流 / 画框 / FPS 面板</p>
        </div>
      </div>
      <div class="conn" :class="statusClass">
        <span class="dot" />
        <span>{{ statusLabel }}</span>
      </div>
    </header>

    <div class="layout">
      <main class="main">
        <VideoStage
          ref="stageRef"
          :source-canvas="sourceCanvasRef"
          :detections="overlayDets"
          :result-width="resultW"
          :result-height="resultH"
        />
        <video ref="hiddenVideo" class="hidden" playsinline muted />
        <canvas ref="captureCanvas" class="hidden" />
      </main>

      <aside class="side">
        <section class="panel">
          <h2>连接</h2>
          <label class="field">
            <span>API HTTP</span>
            <input v-model="apiHttp" type="text" spellcheck="false" placeholder="http://127.0.0.1:8090" />
          </label>
          <p class="mono hint">WS: {{ wsUrl }}</p>
          <div class="btn-row">
            <button
              class="btn primary"
              type="button"
              @click="onConnectToggle"
            >
              {{ status === 'connected' || status === 'connecting' ? '断开' : '连接' }}
            </button>
            <button class="btn" type="button" :disabled="status !== 'connected'" @click="applyConfig">
              下发配置
            </button>
          </div>
          <p v-if="configAck" class="hint ok">
            config_ack: conf={{ configAck.conf }} model={{ configAck.model ?? 'null' }}
          </p>
          <p v-if="lastError" class="hint err">{{ lastError }}</p>
        </section>

        <section class="panel">
          <h2>推理配置</h2>
          <label class="field">
            <span>置信度 conf {{ conf.toFixed(2) }}</span>
            <input v-model.number="conf" type="range" min="0.05" max="0.95" step="0.05" />
          </label>
          <label class="field">
            <span>模型路径 (可选，空=默认)</span>
            <input v-model="model" type="text" spellcheck="false" placeholder="null / path.onnx" />
          </label>
          <label class="field">
            <span>发送帧率目标 ~{{ targetFps }} FPS</span>
            <input v-model.number="targetFps" type="range" min="1" max="30" step="1" />
          </label>
        </section>

        <section class="panel">
          <div class="tabs">
            <button
              type="button"
              class="tab"
              :class="{ active: tab === 'stream' }"
              @click="tab = 'stream'"
            >
              实时流 (WS)
            </button>
            <button
              type="button"
              class="tab"
              :class="{ active: tab === 'oneshot' }"
              @click="tab = 'oneshot'"
            >
              单次推理 (HTTP)
            </button>
          </div>

          <template v-if="tab === 'stream'">
            <div class="btn-row wrap">
              <button
                class="btn primary"
                type="button"
                :disabled="!capture.hasWebcam"
                @click="onStartWebcam"
              >
                摄像头
              </button>
              <label class="btn file-btn">
                上传图片/视频
                <input type="file" accept="image/*,video/*" hidden @change="onFileChange" />
              </label>
              <button
                v-if="streaming"
                class="btn danger"
                type="button"
                @click="onStopStreamOnly"
              >
                停止推流
              </button>
              <button class="btn" type="button" @click="onStop">清除</button>
            </div>
            <p v-if="!capture.hasWebcam" class="hint">无摄像头：请用文件上传或 URL</p>
            <label class="field">
              <span>图片 URL</span>
              <div class="inline">
                <input v-model="imageUrl" type="url" placeholder="https://… 或 /sample.jpg" />
                <button class="btn" type="button" @click="onLoadUrl">加载</button>
              </div>
            </label>
            <p v-if="capture.error" class="hint err">{{ capture.error }}</p>
            <p class="hint">推流状态：{{ streaming ? '运行中' : '停止' }}{{ imageLoop ? '（静图循环）' : '' }}</p>
          </template>

          <template v-else>
            <label class="btn file-btn block">
              选择图片 → POST /infer
              <input type="file" accept="image/*" hidden @change="onOneshotFile" />
            </label>
            <label class="field">
              <span>或图片 URL</span>
              <div class="inline">
                <input v-model="imageUrl" type="url" placeholder="https://…" />
                <button class="btn" type="button" :disabled="oneshotBusy" @click="onLoadUrl">
                  推理
                </button>
              </div>
            </label>
            <p v-if="oneshotBusy" class="hint">推理中…</p>
            <p v-if="oneshotError" class="hint err">{{ oneshotError }}</p>
          </template>
        </section>

        <section class="panel">
          <h2>指标</h2>
          <MetricsPanel
            :fps="displayFps"
            :timings="displayTimings"
            :det-count="overlayDets.length"
          />
        </section>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px 16px 40px;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  gap: 12px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.logo {
  font-size: 28px;
  color: var(--accent);
}
h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.sub {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}
.conn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--panel-2);
  border: 1px solid var(--border);
}
.conn .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #64748b;
}
.conn.connected .dot {
  background: #34d399;
  box-shadow: 0 0 8px #34d399aa;
}
.conn.connecting .dot {
  background: #fbbf24;
}
.conn.error .dot {
  background: #f87171;
}
.layout {
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
.main {
  min-width: 0;
}
.hidden {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
  pointer-events: none;
  overflow: hidden;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
}
.panel h2 {
  margin: 0 0 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
  font-size: 12px;
  color: var(--muted);
}
.field input[type='text'],
.field input[type='url'] {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  padding: 8px 10px;
  font-size: 13px;
  font-family: inherit;
}
.field input[type='range'] {
  width: 100%;
  accent-color: var(--accent);
}
.inline {
  display: flex;
  gap: 8px;
}
.inline input {
  flex: 1;
  min-width: 0;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  padding: 8px 10px;
  font-size: 13px;
}
.btn-row {
  display: flex;
  gap: 8px;
  flex-wrap: nowrap;
}
.btn-row.wrap {
  flex-wrap: wrap;
}
.btn {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--panel-2);
  color: var(--text);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  cursor: pointer;
  font-family: inherit;
}
.btn:hover:not(:disabled) {
  border-color: #3b4560;
  background: #1a2030;
}
.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.btn.primary {
  background: linear-gradient(135deg, #0891b2, #6366f1);
  border-color: transparent;
  color: #fff;
  font-weight: 600;
}
.btn.danger {
  background: #7f1d1d;
  border-color: #991b1b;
  color: #fecaca;
}
.file-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.file-btn.block {
  display: flex;
  width: 100%;
  margin-bottom: 10px;
}
.tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 12px;
  background: var(--panel-2);
  padding: 4px;
  border-radius: 10px;
}
.tab {
  flex: 1;
  border: none;
  background: transparent;
  color: var(--muted);
  padding: 8px;
  border-radius: 8px;
  font-size: 12px;
  cursor: pointer;
  font-family: inherit;
}
.tab.active {
  background: #1e293b;
  color: var(--text);
  font-weight: 600;
}
.hint {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--muted);
}
.hint.err {
  color: #f87171;
}
.hint.ok {
  color: #34d399;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  word-break: break-all;
}
</style>
