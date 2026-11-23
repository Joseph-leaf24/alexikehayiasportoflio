import { sceneState, type LayoutKey } from "@/lib/store";

/**
 * Which two formations we are between, and how far (0..1), for the current scroll. Each
 * transition runs while the next section's top moves from 95% to 25% of the viewport
 * height; a page override (project / 404) wins outright.
 */
export function resolveBlend(scrollY: number, vh: number): { a: LayoutKey; b: LayoutKey; t: number } {
  if (sceneState.override) return { a: sceneState.override, b: sceneState.override, t: 0 };
  const stops = sceneState.stops;
  if (!stops.length) return { a: "village", b: "village", t: 0 };
  let a = stops[0].layout;
  let b = a;
  let t = 0;
  for (let i = 1; i < stops.length; i++) {
    const start = stops[i].y - vh * 0.95;
    const end = stops[i].y - vh * 0.25;
    if (scrollY >= end) {
      a = b = stops[i].layout;
      t = 0;
    } else if (scrollY > start) {
      a = stops[i - 1].layout;
      b = stops[i].layout;
      t = (scrollY - start) / (end - start);
      break;
    } else break;
  }
  return { a, b, t };
}
