import { ref, onUnmounted } from 'vue'

export type CaptureKind = 'none' | 'webcam' | 'video-file' | 'image'

/**
 * Capture helpers: webcam stream, video file loop, or static image.
 * Draws into a hidden canvas and yields JPEG blobs for streaming.
 */
export function useCapture() {
  const kind = ref<CaptureKind>('none')
  const error = ref<string | null>(null)
  const hasWebcam = ref(
    typeof navigator !== 'undefined' &&
      !!navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function',
  )

  const videoEl = ref<HTMLVideoElement | null>(null)
  const imageEl = ref<HTMLImageElement | null>(null)
  let mediaStream: MediaStream | null = null
  let objectUrl: string | null = null
  let rafId = 0
  let intervalId = 0

  function revokeUrl() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl)
      objectUrl = null
    }
  }

  async function startWebcam(video: HTMLVideoElement) {
    stop()
    error.value = null
    if (!hasWebcam.value) {
      error.value = '当前环境不支持摄像头 (getUserMedia)'
      return false
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      video.srcObject = mediaStream
      video.muted = true
      video.playsInline = true
      await video.play()
      videoEl.value = video
      kind.value = 'webcam'
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : '无法打开摄像头'
      kind.value = 'none'
      return false
    }
  }

  async function loadVideoFile(file: File, video: HTMLVideoElement, loop = true) {
    stop()
    error.value = null
    revokeUrl()
    objectUrl = URL.createObjectURL(file)
    video.srcObject = null
    video.src = objectUrl
    video.loop = loop
    video.muted = true
    video.playsInline = true
    try {
      await video.play()
      videoEl.value = video
      kind.value = 'video-file'
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : '无法播放视频'
      kind.value = 'none'
      return false
    }
  }

  async function loadImageFile(file: File): Promise<HTMLImageElement | null> {
    stop()
    error.value = null
    revokeUrl()
    objectUrl = URL.createObjectURL(file)
    const img = new Image()
    img.src = objectUrl
    try {
      await img.decode()
      imageEl.value = img
      kind.value = 'image'
      return img
    } catch (e) {
      error.value = e instanceof Error ? e.message : '无法加载图片'
      kind.value = 'none'
      return null
    }
  }

  async function loadImageUrl(url: string): Promise<HTMLImageElement | null> {
    stop()
    error.value = null
    revokeUrl()
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = url
    try {
      await img.decode()
      imageEl.value = img
      kind.value = 'image'
      return img
    } catch (e) {
      error.value = e instanceof Error ? e.message : '无法加载图片 URL'
      kind.value = 'none'
      return null
    }
  }

  function drawSourceToCanvas(canvas: HTMLCanvasElement): boolean {
    const ctx = canvas.getContext('2d')
    if (!ctx) return false

    if (kind.value === 'webcam' || kind.value === 'video-file') {
      const v = videoEl.value
      if (!v || v.readyState < 2) return false
      if (canvas.width !== v.videoWidth || canvas.height !== v.videoHeight) {
        canvas.width = v.videoWidth || 640
        canvas.height = v.videoHeight || 480
      }
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
      return true
    }
    if (kind.value === 'image' && imageEl.value) {
      const img = imageEl.value
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      return true
    }
    return false
  }

  function canvasToJpegBlob(canvas: HTMLCanvasElement, quality = 0.75): Promise<Blob | null> {
    return new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/jpeg', quality)
    })
  }

  function stop() {
    if (rafId) {
      cancelAnimationFrame(rafId)
      rafId = 0
    }
    if (intervalId) {
      clearInterval(intervalId)
      intervalId = 0
    }
    if (mediaStream) {
      for (const t of mediaStream.getTracks()) t.stop()
      mediaStream = null
    }
    if (videoEl.value) {
      videoEl.value.pause()
      videoEl.value.srcObject = null
      videoEl.value.removeAttribute('src')
    }
    videoEl.value = null
    imageEl.value = null
    revokeUrl()
    kind.value = 'none'
  }

  onUnmounted(() => stop())

  return {
    kind,
    error,
    hasWebcam,
    videoEl,
    imageEl,
    startWebcam,
    loadVideoFile,
    loadImageFile,
    loadImageUrl,
    drawSourceToCanvas,
    canvasToJpegBlob,
    stop,
  }
}
