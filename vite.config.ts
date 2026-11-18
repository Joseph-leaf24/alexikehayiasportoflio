import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

/**
 * Production only: strips comments and whitespace from the inline GLSL in
 * src/components/gl so the shipped shaders are compact and much harder to
 * read or lift. (JS is already minified/mangled by Vite.)
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
 * Production only: a strict Content-Security-Policy so the page can only run its
 * own scripts and talk to the few services it needs (Google Fonts, Power BI embeds).
 * GitHub Pages can't set response headers, so it's delivered as a <meta> tag.
 * (Not used in dev, where Vite/React refresh rely on inline scripts.)
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com data:",
  "img-src 'self' data: blob:",
  "frame-src https://app.powerbi.com",
  "connect-src 'self'",
  "media-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
  "upgrade-insecure-requests",
].join("; ");

function securityHeaders(): Plugin {
  return {
    name: "security-meta",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace(
        "<meta charset=\"UTF-8\" />",
        `<meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="${CSP}" />
    <meta name="referrer" content="strict-origin-when-cross-origin" />`
      );
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  base: "/alexikehayiasportoflio/",
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), minifyInlineGlsl(), securityHeaders(), mode === "development" && componentTagger()].filter(Boolean),
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
