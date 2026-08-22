import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-only proxy: `hflow ui` binds the JSON API on 127.0.0.1:4356 by default.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:4356",
    },
  },
});
