import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-only proxy: `hflow serve` binds the JSON API on 127.0.0.1:4356 by
// default. `preview` gets the same one so the built bundle can be exercised
// against a running API.
const apiProxy = { "/api": "http://127.0.0.1:4356" };

export default defineConfig({
  plugins: [react()],
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
});
