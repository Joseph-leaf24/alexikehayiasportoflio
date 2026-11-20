/**
 * Rendering budget. Every device gets the full quality: the whole cube world, bloom and
 * grain at the screen's own resolution (up to MAX_DPR), and every page effect. Nothing is
 * scaled down automatically, at load or while running. `?quality=low|mid|high` in the URL
 * picks a lighter budget by hand, for testing.
 */
export type Tier = "low" | "mid" | "high";

/** Highest device-pixel ratio the 3D canvas renders at: beyond 2× extra pixels don't show under the grain. */
export const MAX_DPR = 2;

/** The budget for one tier, read by the WebGL scene, the page transition and the CSS (html[data-tier]). */
export interface Quality {
  tier: Tier;
  /** MSAA samples on the scene's render target */
  msaa: number;
  bloomStrength: number;
  /** Voxel terrain extents (x min/max, z min) — smaller = fewer cubes */
  terrain: { xMin: number; xMax: number; zMin: number };
  /** Edge length of the near-sea water blocks (bigger = fewer blocks) */
  waterCell: number;
  /** Drifting snow particles */
  particles: number;
  /** Ice-dust particles for the cursor trail and click bursts (0 = off) */
  dust: number;
  /** Ice floes bobbing at the foot of the cliff */
  floes: number;
  /** Edge length of the moon's cubes (bigger = fewer cubes) */
  moonCube: number;
  /** Ice-sheet slices in the page transition */
  shards: number;
}

/**
 * The budget for a tier. Touch screens skip MSAA (their pixel density already hides jagged
 * edges) and the cursor dust (there's no cursor); everything else is the same.
 */
export function qualityFor(tier: Tier, coarse: boolean): Quality {
  switch (tier) {
    case "high":
      return {
        tier,
        // 2× MSAA: side by side with 4× the edges look the same, at ~8% less GPU load
        msaa: coarse ? 0 : 2,
        bloomStrength: 0.6,
        terrain: { xMin: -28, xMax: 28, zMin: -30 },
        waterCell: 0.6,
        particles: 1800,
        dust: coarse ? 0 : 480,
        floes: 70,
        moonCube: 0.9,
        shards: 7,
      };
    case "mid":
      return {
        tier,
        msaa: coarse ? 0 : 2,
        bloomStrength: 0.45,
        terrain: { xMin: -24, xMax: 26, zMin: -26 },
        waterCell: 0.8,
        particles: 1000,
        dust: coarse ? 0 : 240,
        floes: 45,
        moonCube: 1.0,
        shards: 5,
      };
    default:
      return {
        tier: "low",
        msaa: 0,
        bloomStrength: 0.4,
        terrain: { xMin: -18, xMax: 22, zMin: -20 },
        waterCell: 1.2,
        particles: 300,
        dust: 0,
        floes: 18,
        moonCube: 1.25,
        shards: 3,
      };
  }
}

const TIERS: Tier[] = ["low", "mid", "high"];
const isTier = (v: unknown): v is Tier => typeof v === "string" && (TIERS as string[]).includes(v);

let cached: Quality | null = null;

/** The budget for this page load (the full one unless ?quality= says otherwise), exposed as html[data-tier]. */
export function getQuality(): Quality {
  if (cached) return cached;
  const forced = new URLSearchParams(window.location.search).get("quality");
  cached = qualityFor(isTier(forced) ? forced : "high", window.matchMedia("(pointer: coarse)").matches);
  document.documentElement.dataset.tier = cached.tier;
  return cached;
}

/** Test helper: forget the cached decision. */
export function resetQualityCache() {
  cached = null;
}
