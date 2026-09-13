<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import type { Detection } from '../types'
import { drawOverlay } from '../utils/draw'

const props = defineProps<{
  sourceCanvas: HTMLCanvasElement | null
  detections: Detection[]
  resultWidth: number | null
  resultHeight: number | null
}>()

const displayCanvas = ref<HTMLCanvasElement | null>(null)

function redraw() {
  const canvas = displayCanvas.value
  const src = props.sourceCanvas
  if (!canvas || !src || !src.width) return
  drawOverlay(
    canvas,
    src,
    props.detections,
    props.resultWidth ?? undefined,
    props.resultHeight ?? undefined,
  )
}

watch(
  () => [props.sourceCanvas, props.detections, props.resultWidth, props.resultHeight],
  () => redraw(),
  { deep: true },
)

onMounted(() => redraw())

defineExpose({ redraw, displayCanvas })
</script>

<template>
  <div class="stage">
    <canvas ref="displayCanvas" class="view" />
    <div v-if="!sourceCanvas || !sourceCanvas.width" class="placeholder">
      <p>选择摄像头、上传图片/视频，或输入图片 URL</p>
      <p class="hint">帧会通过 WebSocket 发送到视觉推理服务</p>
    </div>
  </div>
</template>

<style scoped>
.stage {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #0a0c12;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
}
.view {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  display: block;
  object-fit: contain;
}
.placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  text-align: center;
  padding: 24px;
  pointer-events: none;
}
.placeholder p {
  margin: 0;
  font-size: 14px;
}
.hint {
  margin-top: 8px !important;
  font-size: 12px !important;
  opacity: 0.7;
}
</style>
