import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

import { resolve } from "path";

// Vite config for the Beehive Health Monitor frontend.
// The dev server proxies /api to your FastAPI backend so the browser
// never has to deal with CORS during local development.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        app: resolve(__dirname, "app.html")
      }
    }
  },
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
