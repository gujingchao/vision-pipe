/** HTTP base + derived WebSocket URL for the vision-pipe API. */

const DEFAULT_HTTP = 'http://127.0.0.1:8090'

export function getApiHttp(): string {
  const raw = (import.meta.env.VITE_API_HTTP as string | undefined)?.trim()
  return (raw && raw.length > 0 ? raw : DEFAULT_HTTP).replace(/\/$/, '')
}

export function getWsStreamUrl(httpBase?: string): string {
  const base = httpBase ?? getApiHttp()
  try {
    const u = new URL(base)
    const wsProto = u.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${wsProto}//${u.host}/ws/stream`
  } catch {
    return 'ws://127.0.0.1:8090/ws/stream'
  }
}

export function getInferUrl(httpBase?: string, conf?: number, model?: string | null): string {
  const base = httpBase ?? getApiHttp()
  const url = new URL('/infer', base.endsWith('/') ? base : `${base}/`)
  if (conf != null) url.searchParams.set('conf', String(conf))
  if (model) url.searchParams.set('model', model)
  return url.toString()
}
