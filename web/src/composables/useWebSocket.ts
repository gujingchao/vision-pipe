import { ref, shallowRef, onUnmounted, type Ref } from 'vue'
import type {
  ConnStatus,
  ConfigPayload,
  ServerMessage,
  ResultMessage,
} from '../types'

export function useWebSocket(url: Ref<string>) {
  const status = ref<ConnStatus>('disconnected')
  const lastError = ref<string | null>(null)
  const lastResult = shallowRef<ResultMessage | null>(null)
  const configAck = ref<{ conf: number; model: string | null } | null>(null)

  let ws: WebSocket | null = null
  let intentionalClose = false

  function connect() {
    disconnect()
    intentionalClose = false
    status.value = 'connecting'
    lastError.value = null

    try {
      ws = new WebSocket(url.value)
    } catch (e) {
      status.value = 'error'
      lastError.value = e instanceof Error ? e.message : String(e)
      return
    }

    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      status.value = 'connected'
    }

    ws.onclose = () => {
      ws = null
      if (!intentionalClose && status.value !== 'error') {
        status.value = 'disconnected'
      } else if (intentionalClose) {
        status.value = 'disconnected'
      }
    }

    ws.onerror = () => {
      lastError.value = 'WebSocket 连接失败'
      status.value = 'error'
    }

    ws.onmessage = (ev) => {
      try {
        const text = typeof ev.data === 'string' ? ev.data : new TextDecoder().decode(ev.data)
        const msg = JSON.parse(text) as ServerMessage
        if (msg.type === 'result') {
          lastResult.value = msg
        } else if (msg.type === 'error') {
          lastError.value = msg.message
        } else if (msg.type === 'config_ack') {
          configAck.value = { conf: msg.conf, model: msg.model }
        }
      } catch (e) {
        lastError.value = e instanceof Error ? e.message : '消息解析失败'
      }
    }
  }

  function disconnect() {
    intentionalClose = true
    if (ws) {
      try {
        ws.close()
      } catch {
        /* ignore */
      }
      ws = null
    }
    status.value = 'disconnected'
  }

  function sendBinary(data: Blob | ArrayBuffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    ws.send(data)
    return true
  }

  function sendJson(payload: object) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(payload))
    return true
  }

  function sendConfig(conf: number, model: string | null) {
    const payload: ConfigPayload = { type: 'config', conf, model }
    return sendJson(payload)
  }

  /** Prefer binary JPEG; fall back to base64 JSON frame. */
  async function sendFrameBlob(blob: Blob, preferBinary = true) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    if (preferBinary) {
      return sendBinary(blob)
    }
    const buf = await blob.arrayBuffer()
    const bytes = new Uint8Array(buf)
    let binary = ''
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!)
    const b64 = btoa(binary)
    return sendJson({ type: 'frame', data: b64 })
  }

  onUnmounted(() => disconnect())

  return {
    status,
    lastError,
    lastResult,
    configAck,
    connect,
    disconnect,
    sendBinary,
    sendJson,
    sendConfig,
    sendFrameBlob,
  }
}
