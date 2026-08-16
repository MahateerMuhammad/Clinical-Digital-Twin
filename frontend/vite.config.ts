import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The backend runs on 8010, not 8000 — 8000 belongs to a different app on this
// machine. Proxying rather than calling the origin directly keeps the browser
// on one origin, so CORS is a deployment concern and never a dev-time one.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8010", changeOrigin: true } },
  },
});
