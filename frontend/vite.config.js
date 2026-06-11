import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const PROXY_TIMEOUT_MS = 10 * 60 * 1000

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/chat': {
        target: 'http://localhost:3333',
        changeOrigin: true,
        timeout: PROXY_TIMEOUT_MS,
        proxyTimeout: PROXY_TIMEOUT_MS,
      },
      '/query': {
        target: 'http://localhost:3333',
        changeOrigin: true,
      },
      '/metadata': {
        target: 'http://localhost:3333',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:3333',
        changeOrigin: true,
      },
    },
  },
})
