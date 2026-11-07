/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The production build goes into the Python package, so the wheel ships it and `weightlab start` serves it
// from the same origin as the API. Node is needed only to build, never at runtime.
const outDir = fileURLToPath(new URL("../weightlab/ui/static", import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir,
    emptyOutDir: true,
    assetsInlineLimit: 0, // every asset is a file under /assets/, nothing becomes a data: URI
    modulePreload: { polyfill: false },
    sourcemap: false,
    target: "es2022",
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
