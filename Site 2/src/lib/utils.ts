import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/** Join class names (clsx) and let later Tailwind utilities override conflicting ones (tailwind-merge). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Two-digit index ("07"), as used for numbering across the site. */
export const pad = (n: number) => String(n).padStart(2, "0");

/** Columns × rows of roughly square blocks that cover a w×h viewport, with `cols` across the longer side (the page wipe). */
export function wipeGrid(cols: number, w: number, h: number) {
  // (a zero-size window, e.g. a hidden frame, still gets one block)
  const cell = Math.max(1, w, h) / cols;
  return { cols: Math.max(1, Math.ceil(w / cell)), rows: Math.max(1, Math.ceil(h / cell)) };
}

/** The part of a iw×ih picture that `fit` shows in a w×h box (as a drawImage source rectangle). */
export function fitRect(iw: number, ih: number, w: number, h: number, fit: "cover" | "contain") {
  if (fit === "contain") return { sx: 0, sy: 0, sw: iw, sh: ih };
  const s = Math.max(w / iw, h / ih);
  const sw = w / s;
  const sh = h / s;
  return { sx: (iw - sw) / 2, sy: (ih - sh) / 2, sw, sh };
}
