<script setup lang="ts">
import { computed } from 'vue'
import type { TimingsMs } from '../types'

const props = defineProps<{
  fps: number | null
  timings: TimingsMs | null
  detCount: number
}>()

const bars = computed(() => {
  const t = props.timings
  if (!t) return []
  const keys = ['preprocess', 'infer', 'postprocess', 'total'] as const
  const total = Number(t.total) || 1
  return keys.map((k) => {
    const v = Number(t[k] ?? 0)
    return {
      key: k,
      label:
        k === 'preprocess'
          ? '预处理'
          : k === 'infer'
            ? '推理'
            : k === 'postprocess'
              ? '后处理'
              : '合计',
      ms: v,
      pct: Math.min(100, (v / Math.max(total, 0.001)) * 100),
    }
  })
})

const backend = computed(() => (props.timings?.backend ? String(props.timings.backend) : '—'))
</script>

<template>
  <div class="metrics">
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">FPS</div>
        <div class="metric-value accent">{{ fps != null ? fps.toFixed(1) : '—' }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">检测数</div>
        <div class="metric-value">{{ detCount }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">后端</div>
        <div class="metric-value small">{{ backend }}</div>
      </div>
    </div>

    <div class="latency">
      <div class="latency-title">延迟 (ms)</div>
      <div v-for="b in bars" :key="b.key" class="bar-row">
        <span class="bar-label">{{ b.label }}</span>
        <div class="bar-track">
          <div
            class="bar-fill"
            :class="{ total: b.key === 'total' }"
            :style="{ width: `${b.pct}%` }"
          />
        </div>
        <span class="bar-ms">{{ b.ms.toFixed(2) }}</span>
      </div>
      <div v-if="!timings" class="empty">等待首帧结果…</div>
    </div>
  </div>
</template>

<style scoped>
.metrics {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.metric-card {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
}
.metric-label {
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.metric-value {
  font-size: 22px;
  font-weight: 700;
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
}
.metric-value.accent {
  color: var(--accent);
}
.metric-value.small {
  font-size: 13px;
  font-weight: 600;
  word-break: break-all;
}
.latency {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
}
.latency-title {
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 8px;
  font-weight: 600;
}
.bar-row {
  display: grid;
  grid-template-columns: 56px 1fr 52px;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.bar-label {
  font-size: 12px;
  color: var(--text);
}
.bar-track {
  height: 8px;
  background: #1a1f2e;
  border-radius: 999px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #22d3ee, #818cf8);
  border-radius: 999px;
  min-width: 2px;
  transition: width 0.2s ease;
}
.bar-fill.total {
  background: linear-gradient(90deg, #34d399, #22d3ee);
}
.bar-ms {
  font-size: 12px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
}
.empty {
  font-size: 12px;
  color: var(--muted);
  padding: 4px 0;
}
</style>
