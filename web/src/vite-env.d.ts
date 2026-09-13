/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_HTTP?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
