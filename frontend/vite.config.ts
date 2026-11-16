import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/**
 * Vite configuration (§18.1, §17.2).
 *
 * The dashboard talks to the FastAPI backend on a different port during development. The
 * proxy keeps REST, MJPEG, and the live WebSocket (`ws: true`) on the page's own origin, so
 * the frontend always uses same-origin paths and needs no CORS policy on the server.
 *
 * The backend address is configurable so a reviewer can use another local port:
 * `FALL_API_TARGET=http://127.0.0.1:9000 bun run dev`.
 */
const API_TARGET = process.env.FALL_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // One entry covers REST, MJPEG, the snapshot, and the WebSocket: every backend route
      // lives under `/api`, and `/health` is proxied so the dashboard can show readiness
      // without a second origin.
      "/api": {
        target: API_TARGET,
        changeOrigin: false,
        ws: true,
      },
      "/health": {
        target: API_TARGET,
        changeOrigin: false,
      },
    },
  },
});
