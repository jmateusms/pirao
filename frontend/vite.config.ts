import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Where the dev server finds the API; override when 8000 is taken.
const api = process.env.PIRAO_API ?? process.env.RELIMCMC_API ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  // The API is same-origin in production (one container, one origin); in dev
  // and preview it is proxied so the frontend never needs to know where the
  // backend lives.
  server: {
    port: 5173,
    proxy: { '/api': { target: api, changeOrigin: true } },
  },
  preview: {
    port: 5173,
    proxy: { '/api': { target: api, changeOrigin: true } },
  },
  build: { outDir: 'dist', sourcemap: true },
})
