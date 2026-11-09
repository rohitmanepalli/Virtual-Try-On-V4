import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Keep the older packaged Electron build in frontend/dist intact.
// Vite uses a separate output folder so it does not try to delete the existing executable.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist-app',
    emptyOutDir: true,
  },
})
