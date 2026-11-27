import type { Config } from "tailwindcss";

// The site is styled with hand-written CSS in src/index.css. Tailwind supplies the base reset
// and a handful of utility classes used in components, such as sizes and sr-only.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config;
