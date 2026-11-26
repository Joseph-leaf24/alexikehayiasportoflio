import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * Production build only: a strict Content-Security-Policy so the page can only run its own
 * scripts and talk to the few services it needs (Google Fonts, Power BI embeds). Delivered
 * as a <meta> tag so it also applies under `vite preview` or any static file server.
 * (Not used in dev, where Vite / React refresh rely on inline scripts.)
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

// Site 2 runs locally only (`npm run dev`, or `npm run build && npm run preview`), so it is
// served from the root path. Port 8081 keeps it clear of the main site's dev server (8080).
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
        // Keep the motion libraries and React in their own cacheable chunks
        manualChunks: {
          // (the battle engine on its own, as the live site keeps three.js: it was most of one 600 kB chunk)
          pixel: [path.resolve(__dirname, "./src/components/pixel/PixelScene.tsx")],
          motion: ["gsap", "lenis"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
}));
