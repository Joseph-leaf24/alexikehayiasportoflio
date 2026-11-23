import type { Config } from "tailwindcss";

// The site is styled with hand-written CSS in src/index.css. Tailwind supplies the
// base reset (preflight) and a handful of utilities used in components (sizes, sr-only…).
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;
