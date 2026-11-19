/**
 * Device tiering: picks a rendering budget so low-end phones stay smooth and
 * high-end PCs get the full effect. Detection is a pure function of the
 * signals below so it can be unit-tested; a runtime governor in the scene can
 * still step quality down if frames are slow.
 */
export type Tier = "low" | "mid" | "high";

/** What the browser reveals about the hardware (see readSignals). */
export interface DeviceSignals {
  /** navigator.hardwareConcurrency */
  cores?: number;
  /** navigator.deviceMemory in GB (Chromium only; capped at 8 by the browser) */
  memory?: number;
  dpr: number;
  /** Viewport width in CSS px */
  width: number;
  /** Primary pointer is coarse (touch) */
  coarse: boolean;
  /** Data Saver is on */
  saveData: boolean;
  webgl2: boolean;
  /** Unmasked WebGL renderer string, when the browser exposes it */
  gpu?: string;
}

/** The rendering budget for one tier, read by the WebGL scene and the page transition. */
export interface Quality {
  tier: Tier;
  /** Max device-pixel-ratio the WebGL canvas renders at */
  dprCap: number;
  /** Lowest DPR the runtime governor may drop to */
  dprFloor: number;
  /** MSAA samples on the scene's render target */
  msaa: number;
  bloom: boolean;
  bloomStrength: number;
  /** Grain / aberration / glitch composite pass */
  postFx: boolean;
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
  /** Frame cap in fps (0 = uncapped / display refresh) */
  fpsCap: number;
  /** Ice-sheet slices in the page transition */
  shards: number;
}

/** Software renderers and old mobile / Intel GPUs: always the low tier. */
const SLOW_GPU = /(swiftshader|llvmpipe|software|mali-4\d\d|mali-t\d|powervr|sgx|adreno \(tm\) [1-5]\d\d|intel\(r\) hd graphics [2-5]\d{2,3}|gma)/i;
/** Dedicated desktop / laptop GPUs and recent high-end mobile GPUs: a score bonus. */
const FAST_GPU =/(rtx|gtx 1[06-9]|gtx [2-9]\d{2,3}|radeon rx|radeon pro|apple m\d|apple gpu|adreno \(tm\) 7\d\d|mali-g7[1-9]|mali-g[89]\d|immortalis|arc a\d)/i;
/**
 * Integrated laptop/desktop graphics (Intel Iris / UHD / HD, Intel Arc iGPU, AMD Radeon
 * "Graphics" APUs). On Windows the browser usually runs on these even when the laptop
 * also has a powerful dedicated GPU, so they are never treated as high-end.
 */
const INTEGRATED_GPU = /(intel.*(iris|uhd graphics|hd graphics|arc\(tm\) graphics)|radeon\(tm\) graphics|radeon graphics|radeon \d{3}m|radeon vega \d+ graphics|vega \d+ graphics)/i;

/** True for integrated GPUs (see INTEGRATED_GPU). Exported for tests. */
export const isIntegratedGpu = (gpu?: string) => !!gpu && INTEGRATED_GPU.test(gpu) && !FAST_GPU.test(gpu);

/** Score the hardware and map it to a tier (integrated GPUs are capped at mid). */
export function scoreDevice(d: DeviceSignals): Tier {
  if (!d.webgl2 || d.saveData) return "low";
  if (d.gpu && SLOW_GPU.test(d.gpu)) return "low";
  let score = 0;
  const cores = d.cores ?? 4;
  const memory = d.memory ?? 4;
  if (cores >= 8) score += 2;
  else if (cores >= 4) score += 1;
  if (memory >= 8) score += 2;
  else if (memory >= 4) score += 1;
  else score -= 1;
  if (d.gpu && FAST_GPU.test(d.gpu)) score += 2;
  if (d.coarse) score -= 1;
  if (d.width < 768) score -= 1;
  if (score >= 4) return isIntegratedGpu(d.gpu) ? "mid" : "high";
  if (score >= 2) return "mid";
  return "low";
}

/** The budget for a tier. On mid/high, touch devices get a lighter mix (no MSAA or dust, fewer particles). */
export function qualityFor(tier: Tier, coarse: boolean): Quality {
  switch (tier) {
    case "high":
      return {
        tier,
        // Above ~1.5 the extra pixels are invisible under bloom + grain but cost ~1.8x the fill
        dprCap: 1.5,
        dprFloor: 1,
        msaa: coarse ? 0 : 4,
        bloom: true,
        bloomStrength: coarse ? 0.5 : 0.6,
        postFx: true,
        terrain: { xMin: -28, xMax: 28, zMin: -30 },
        waterCell: 0.6,
        particles: coarse ? 1000 : 1800,
        dust: coarse ? 0 : 480,
        floes: 70,
        moonCube: 0.9,
        fpsCap: 0,
        shards: 7,
      };
    case "mid":
      return {
        tier,
        dprCap: coarse ? 1.25 : 1.35,
        dprFloor: 0.85,
        msaa: coarse ? 0 : 2,
        bloom: true,
        bloomStrength: 0.45,
        postFx: true,
        terrain: { xMin: -24, xMax: 26, zMin: -26 },
        waterCell: 0.8,
        particles: coarse ? 600 : 1000,
        dust: coarse ? 0 : 240,
        floes: 45,
        moonCube: 1.0,
        fpsCap: 0,
        shards: 5,
      };
    default:
      return {
        tier: "low",
        dprCap: 1,
        dprFloor: 0.65,
        msaa: 0,
        bloom: false,
        bloomStrength: 0,
        postFx: false,
        terrain: { xMin: -18, xMax: 22, zMin: -20 },
        waterCell: 1.2,
        particles: 300,
        dust: 0,
        floes: 18,
        moonCube: 1.25,
        fpsCap: 30,
        shards: 3,
      };
  }
}

const TIERS: Tier[] = ["low", "mid", "high"];
const isTier = (v: unknown): v is Tier => typeof v === "string" && (TIERS as string[]).includes(v);
const STORAGE_KEY = "ak:tier";

/** Read the GPU name via WEBGL_debug_renderer_info (falls back to undefined). */
function probeGpu(): { webgl2: boolean; gpu?: string } {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") as WebGL2RenderingContext | null;
    if (!gl) return { webgl2: false };
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : undefined;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { webgl2: true, gpu };
  } catch {
    return { webgl2: false };
  }
}

/** Collect the live browser signals for scoreDevice. */
export function readSignals(): DeviceSignals {
  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } };
  return {
    cores: nav.hardwareConcurrency,
    memory: nav.deviceMemory,
    dpr: window.devicePixelRatio || 1,
    width: window.innerWidth,
    coarse: window.matchMedia("(pointer: coarse)").matches,
    saveData: !!nav.connection?.saveData,
    ...probeGpu(),
  };
}

let cached: Quality | null = null;

/**
 * Resolve the quality budget once per page load and expose the tier as html[data-tier].
 * Priority: ?quality=low|mid|high  >  a downgrade remembered this session  >  detection.
 */
export function getQuality(signals: DeviceSignals = readSignals()): Quality {
  if (cached) return cached;
  let tier: Tier = scoreDevice(signals);
  try {
    const remembered = sessionStorage.getItem(STORAGE_KEY);
    if (isTier(remembered) && TIERS.indexOf(remembered) < TIERS.indexOf(tier)) tier = remembered;
  } catch {
    /* storage blocked */
  }
  const forced = new URLSearchParams(window.location.search).get("quality");
  if (isTier(forced)) tier = forced;
  cached = qualityFor(tier, signals.coarse);
  document.documentElement.dataset.tier = cached.tier;
  return cached;
}

/**
 * Called by the runtime governor when a device can't keep up: remember the next tier down
 * for this session, mark html[data-perf="degraded"] so CSS drops heavier effects, and
 * return the new tier.
 */
export function rememberDowngrade(tier: Tier) {
  const lower = TIERS[Math.max(0, TIERS.indexOf(tier) - 1)];
  try {
    sessionStorage.setItem(STORAGE_KEY, lower);
  } catch {
    /* storage blocked */
  }
  document.documentElement.dataset.perf = "degraded";
  return lower;
}

/** Test helper: forget the cached decision. */
export function resetQualityCache() {
  cached = null;
}
