/**
 * Rendering budget. Every device gets the full scene: every enemy, cloud and particle, and
 * every page effect. Nothing is scaled down automatically, at load or while running.
 * `?quality=low|mid|high` in the URL picks a lighter budget by hand, for testing.
 */
export type Tier = "low" | "mid" | "high";

/** The budget for one tier, read by the pixel scene, the page transition and the CSS (html[data-tier]). */
export interface Quality {
  tier: Tier;
  /** Most enemies alive at once */
  enemies: number;
  /** Most particles alive at once (debris, sparks, smoke, flames) */
  particles: number;
  /** Clouds drifting across the sky */
  clouds: number;
  /** Columns of blocks in the page-wipe transition */
  wipeCols: number;
}

/** The budget for a tier. (Touch screens get the same: the scene doesn't depend on a cursor.) */
export function qualityFor(tier: Tier): Quality {
  switch (tier) {
    case "high":
      return { tier, enemies: 200, particles: 3400, clouds: 12, wipeCols: 12 };
    case "mid":
      return { tier, enemies: 130, particles: 2000, clouds: 9, wipeCols: 10 };
    default:
      return { tier: "low", enemies: 64, particles: 800, clouds: 6, wipeCols: 8 };
  }
}

const TIERS: Tier[] = ["low", "mid", "high"];
const isTier = (v: unknown): v is Tier => typeof v === "string" && (TIERS as string[]).includes(v);

let cached: Quality | null = null;

/** The budget for this page load (the full one unless ?quality= says otherwise), exposed as html[data-tier]. */
export function getQuality(): Quality {
  if (cached) return cached;
  const forced = new URLSearchParams(window.location.search).get("quality");
  cached = qualityFor(isTier(forced) ? forced : "high");
  document.documentElement.dataset.tier = cached.tier;
  return cached;
}

/** Test helper: forget the cached decision. */
export function resetQualityCache() {
  cached = null;
}
