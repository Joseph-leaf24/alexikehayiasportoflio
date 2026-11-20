import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/** Join class names (clsx) and let later Tailwind utilities override conflicting ones (tailwind-merge). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Two-digit index ("07"), as used for numbering across the site. */
export const pad = (n: number) => String(n).padStart(2, "0");
