import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Joins class names with clsx; tailwind-merge lets later Tailwind utilities override conflicting ones. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Two-digit index ("07"), as used for numbering across the site. */
export const pad = (n: number) => String(n).padStart(2, "0");
