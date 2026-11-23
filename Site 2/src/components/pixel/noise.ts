/**
 * Seeded value noise in one and two dimensions, and fractal sums of it: the terrain rolls
 * on for ever without repeating, and a new seed gives a whole new world (the way a voxel
 * game's world generator works). Pure maths, no state: the same seed and position always
 * give the same value, so a level can be generated a chunk at a time.
 */

/** Deterministic 0..1 hash of a seed and up to two lattice coordinates. */
export function hash(seed: number, x: number, y = 0) {
  let h = Math.imul(seed | 0, 0x27d4eb2d) ^ Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** Quintic ease: smooth slopes, no creases at the lattice points. */
const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);

/**
 * 1D value noise in 0..1, one lattice point per unit. With `period` it wraps (whole numbers
 * only), so a strip drawn with it tiles seamlessly.
 */
export function noise1(seed: number, x: number, period = 0) {
  const i = Math.floor(x);
  const f = x - i;
  const a = period ? ((i % period) + period) % period : i;
  const b = period ? (a + 1) % period : i + 1;
  const ha = hash(seed, a);
  return ha + (hash(seed, b) - ha) * fade(f);
}

/** 2D value noise in 0..1. */
export function noise2(seed: number, x: number, y: number) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const u = fade(x - ix);
  const v = fade(y - iy);
  const a = hash(seed, ix, iy);
  const b = hash(seed, ix + 1, iy);
  const c = hash(seed, ix, iy + 1);
  const d = hash(seed, ix + 1, iy + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/**
 * Fractal 1D noise in -1..1: `octaves` layers, each twice the frequency and `gain` times the
 * height of the one before. `period` (in units of the first octave) makes it tile.
 */
export function fbm1(seed: number, x: number, octaves = 4, gain = 0.5, period = 0) {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let f = 1;
  for (let o = 0; o < octaves; o++) {
    sum += (noise1(seed + o * 101, x * f, period * f) * 2 - 1) * amp;
    norm += amp;
    amp *= gain;
    f *= 2;
  }
  return sum / norm;
}

/**
 * Ridged fractal noise in 0..1: sharp crests and soft valleys, like a mountain range seen
 * side on. `period` as for fbm1.
 */
export function ridged1(seed: number, x: number, octaves = 4, gain = 0.5, period = 0) {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let f = 1;
  for (let o = 0; o < octaves; o++) {
    const n = 1 - Math.abs(noise1(seed + o * 131, x * f, period * f) * 2 - 1);
    sum += n * n * amp;
    norm += amp;
    amp *= gain;
    f *= 2;
  }
  return sum / norm;
}

export const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
