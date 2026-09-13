/** Protocol types matching api/README.md WebSocket + HTTP /infer. */

export interface Detection {
  xyxy: [number, number, number, number]
  score: number
  label: string
}

export interface TimingsMs {
  preprocess?: number
  infer?: number
  postprocess?: number
  total?: number
  backend?: string
  [key: string]: number | string | undefined
}

export interface ResultMessage {
  type: 'result'
  width: number
  height: number
  detections: Detection[]
  timings_ms: TimingsMs
  fps?: number
}

export interface ErrorMessage {
  type: 'error'
  message: string
}

export interface ConfigAckMessage {
  type: 'config_ack'
  conf: number
  model: string | null
}

export type ServerMessage = ResultMessage | ErrorMessage | ConfigAckMessage

export interface ConfigPayload {
  type: 'config'
  conf: number
  model: string | null
}

export interface FramePayload {
  type: 'frame'
  data: string
}

export type ConnStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export type SourceMode = 'webcam' | 'file' | 'url' | 'oneshot'
