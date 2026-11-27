import type { Config } from "tailwindcss";

// The site is styled with hand-written CSS in src/index.css. Tailwind supplies only the
// preflight reset and a few utilities used in components, such as sizes and sr-only.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;
