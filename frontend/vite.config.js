import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config for the Beehive Health Monitor frontend.
// The dev server proxies /api to your FastAPI backend so the browser
// never has to deal with CORS during local development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  }
});
