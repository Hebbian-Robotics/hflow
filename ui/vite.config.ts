import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-only proxy: `hflow ui` binds the JSON API on 127.0.0.1:4356 by default.
// `preview` gets the same one so the built bundle can be exercised against a
// running API — the only way to see the pre-paint theme script do its job,
// since dev injects the stylesheet from JS.
const apiProxy = { "/api": "http://127.0.0.1:4356" };

export default defineConfig({
  plugins: [react()],
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
});
