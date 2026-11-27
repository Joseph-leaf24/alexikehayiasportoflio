import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const slash = (p: string) => p.replace(/\\/g, "/");
// The battle is built straight from Site 2's source.
const site = slash(path.resolve(here, "../../Site 2/src"));

/** Builds the wallpaper page; scripts/pack.mjs inlines it and scripts/build-exe.mjs embeds it in the exe. */
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: [
      // Replace the site's reduced-motion check and device budget with wallpaper versions.
      { find: /^@\/lib\/smoothScroll$/, replacement: slash(path.join(here, "src/shims/smoothScroll.ts")) },
      { find: /^@\/lib\/device$/, replacement: slash(path.join(here, "src/shims/device.ts")) },
      { find: /^@\//, replacement: `${site}/` },
    ],
    // One copy of React for the wallpaper and the Site 2 code it imports.
    dedupe: ["react", "react-dom"],
  },
  server: { fs: { allow: [slash(path.resolve(here, "../.."))] } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2020",
    cssCodeSplit: false,
    chunkSizeWarningLimit: 2000,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
