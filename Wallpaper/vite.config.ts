import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const slash = (p: string) => p.replace(/\\/g, "/");
/** The battle itself lives in Site 2: the wallpaper is built from it, so the two are always the same battle. */
const site = slash(path.resolve(here, "../Site 2/src"));

/**
 * The wallpaper: the pixel scene on its own, full screen, built to one page that opens straight
 * off the disk (scripts/pack.mjs inlines the code into it and adds what Lively Wallpaper and
 * Wallpaper Engine need).
 */
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: [
      // (what the site does that a wallpaper mustn't: freeze the scene for "reduced motion", which
      // Windows turns on with its animations setting; and spend the site's full budget)
      { find: /^@\/lib\/smoothScroll$/, replacement: slash(path.join(here, "src/shims/smoothScroll.ts")) },
      { find: /^@\/lib\/device$/, replacement: slash(path.join(here, "src/shims/device.ts")) },
      { find: /^@\//, replacement: `${site}/` },
    ],
    // (one React for the wallpaper and the site's code it pulls in)
    dedupe: ["react", "react-dom"],
  },
  server: { fs: { allow: [slash(path.resolve(here, ".."))] } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2020",
    cssCodeSplit: false,
    chunkSizeWarningLimit: 2000,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
