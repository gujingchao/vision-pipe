import type { Detection } from '../types'

const COLORS = [
  '#22d3ee',
  '#a78bfa',
  '#34d399',
  '#fbbf24',
  '#f472b6',
  '#60a5fa',
  '#fb7185',
]

function colorForLabel(label: string): string {
  let h = 0
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0
  return COLORS[h % COLORS.length]!
}

/**
 * Draw source image + detection boxes onto display canvas.
 * Scales xyxy from result resolution to canvas display size if needed.
 */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement,
  detections: Detection[],
  resultW?: number,
  resultH?: number,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  let sw = 0
  let sh = 0
  if (source instanceof HTMLVideoElement) {
    sw = source.videoWidth
    sh = source.videoHeight
  } else if (source instanceof HTMLImageElement) {
    sw = source.naturalWidth
    sh = source.naturalHeight
  } else {
    sw = source.width
    sh = source.height
  }
  if (!sw || !sh) return

  if (canvas.width !== sw || canvas.height !== sh) {
    canvas.width = sw
    canvas.height = sh
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height)

  const rw = resultW && resultW > 0 ? resultW : sw
  const rh = resultH && resultH > 0 ? resultH : sh
  const sx = canvas.width / rw
  const sy = canvas.height / rh

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.xyxy
    const left = x1 * sx
    const top = y1 * sy
    const w = (x2 - x1) * sx
    const h = (y2 - y1) * sy
    const color = colorForLabel(det.label || 'obj')

    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 320))
    ctx.strokeRect(left, top, w, h)

    const label = `${det.label || 'obj'} ${(det.score * 100).toFixed(0)}%`
    ctx.font = `bold ${Math.max(12, Math.round(canvas.width / 40))}px ui-sans-serif, system-ui, sans-serif`
    const pad = 4
    const tw = ctx.measureText(label).width
    const th = Math.max(14, Math.round(canvas.width / 36))
    const ly = top > th + 4 ? top - 2 : top + th + 2

    ctx.fillStyle = 'rgba(0,0,0,0.65)'
    ctx.fillRect(left, ly - th, tw + pad * 2, th + 2)
    ctx.fillStyle = color
    ctx.fillText(label, left + pad, ly - 4)
  }
}
