import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

// Proxy every backend route the UI hits so the dev server never sees CORS.
// The list mirrors the FastAPI routers in checkpoints/checkpoint-3-safety-rag/app/routes.
const PROXY_PATHS = ["/agent", "/safety", "/rag", "/tools", "/tasks", "/health"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      PROXY_PATHS.map((p) => [p, { target: BACKEND, changeOrigin: true }])
    ),
  },
});
