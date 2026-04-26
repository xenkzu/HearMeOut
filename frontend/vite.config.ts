import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../frontend/dist',
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/separate': 'http://localhost:8000',
      '/analyze': 'http://localhost:8000',
      '/job': 'http://localhost:8000',
      '/system-info': 'http://localhost:8000',
      '/setup': 'http://localhost:8000',
    }
  }
});
