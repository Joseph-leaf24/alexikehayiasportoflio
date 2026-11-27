/**
 * A tiny local PRNG and position hash, so the terrain stays deterministic without depending on the core's streams.
 * The game can still drive it from a core stream by seeding it with `rng.get("terrain").u32()`.
 */

/** mulberry32: one 32-bit word of state, a few integer ops per draw. */
export class TerrainRng {
  private s: number;

  constructor(seed = 1) {
    this.s = seed | 0;
  }

  reseed(seed: number): void {
    this.s = seed | 0;
  }

  /** A uniform unsigned 32-bit integer. */
  u32(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    return this.u32() / 4294967296;
  }

  /** Uniform integer in [0, n). */
  int(n: number): number {
    return (this.next() * n) | 0;
  }

  /** The state, for snapshots and determinism checks. */
  get state(): number {
    return this.s;
  }
}

/** A well-mixed 32-bit hash of an integer position and a seed (murmur3 finaliser). */
export function hashCell(col: number, row: number, seed: number): number {
  let x = Math.imul(col | 0, 0x27d4eb2d) ^ Math.imul(row | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b1);
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  return (x ^ (x >>> 16)) >>> 0;
}
