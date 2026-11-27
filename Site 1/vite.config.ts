import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * Production only: strips comments and whitespace from the inline GLSL in src/components/gl,
 * so the shipped shaders are compact and hard to lift.
 */
function minifyInlineGlsl(): Plugin {
  return {
    name: "minify-inline-glsl",
    apply: "build",
    // Run before the SWC/TypeScript transform, which strips the /* glsl */ markers
    enforce: "pre",
    transform(code, id) {
      if (!id.includes("/components/gl/")) return null;
      const out = code.replace(/\/\* glsl \*\/ `([\s\S]*?)`/g, (_m, body: string) => {
        const min = body
          .replace(/\/\*[\s\S]*?\*\//g, "")
          .replace(/\/\/[^\n]*/g, "")
          .replace(/\s+/g, " ")
          .replace(/\s*([{}()[\];,=+\-*/<>!?:&|])\s*/g, "$1")
          .trim();
        return "`" + min + "`";
      });
      return out === code ? null : { code: out, map: null };
    },
  };
}

/**
 * Production-only Content-Security-Policy: the page runs only its own scripts and talks only to
 * Google Fonts and Power BI embeds. GitHub Pages can't set response headers, so it ships as a
 * <meta> tag. Dev goes without it because React refresh relies on inline scripts.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:",
  "frame-src https://app.powerbi.com",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
  "upgrade-insecure-requests",
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

export default defineConfig(() => ({
  base: "/alexikehayiasportoflio/",
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), minifyInlineGlsl(), securityHeaders()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Never ship source maps — they would expose the original, readable source
    sourcemap: false,
    // three.js alone is ~530 kB (130 kB gzipped) and is cached in its own chunk
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // Keep the heavy WebGL + motion libraries in their own cacheable chunks
        manualChunks: {
          three: ["three"],
          motion: ["gsap", "lenis"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
}));
