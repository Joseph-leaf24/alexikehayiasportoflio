import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * Production-only Content-Security-Policy: the page may run only its own scripts and reach only
 * Google Fonts and Power BI embeds. Delivered as a <meta> tag so it also applies under
 * `vite preview` or any static file server. Dev is left out because React refresh needs inline scripts.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data: blob:",
  "frame-src https://app.powerbi.com",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join("; ");

/** Injects the CSP and a referrer policy right after <meta charset> in the built index.html. */
function securityHeaders(): Plugin {
  return {
    name: "security-meta",
    apply: "build",
    transformIndexHtml(html) {
      const marker = '<meta charset="UTF-8" />';
      // Fail the build rather than ship a page without its CSP
      if (!html.includes(marker)) throw new Error(`security-meta: ${marker} not found in index.html`);
      return html.replace(
        marker,
        `${marker}
    <meta http-equiv="Content-Security-Policy" content="${CSP}" />
    <meta name="referrer" content="strict-origin-when-cross-origin" />`
      );
    },
  };
}

// Site 2 runs locally only, through `npm run dev` or `npm run build && npm run preview`, so it is
// served from the root path. Port 8081 keeps it clear of the main site's dev server on 8080.
export default defineConfig(() => ({
  base: "/",
  server: {
    host: "::",
    port: 8081,
  },
  preview: {
    port: 4174,
  },
  plugins: [react(), securityHeaders()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Never ship source maps — they would expose the original, readable source
    sourcemap: false,
    rollupOptions: {
      output: {
        // The game engine, the motion libraries and React each get a cacheable chunk; the engine
        // alone was most of a single 600 kB chunk
        manualChunks: {
          pixel: [path.resolve(__dirname, "./src/components/pixel/PixelScene.tsx")],
          motion: ["gsap", "lenis"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
}));
