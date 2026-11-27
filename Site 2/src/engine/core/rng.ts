/**
 * Seeded randomness. `Rng` is sfc32 (128-bit state, passes PractRand, a handful of integer ops per number), and
 * `RngStreams` hands out one independent generator per name, so cosmetic draws ("fx") never shift the sequence a
 * gameplay stream ("ai", "spawn") sees. Same seed, same calls → same numbers on every run.
 */

/** murmur3's 32-bit finaliser: a cheap, well-mixed integer hash. */
export function hash32(x: number): number {
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  return (x ^ (x >>> 16)) >>> 0;
}

/** FNV-1a over UTF-16 code units, then finalised: stable ids from names. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return hash32(h);
}

/** Combine two integers into one hash (order matters). */
export const hashCombine = (a: number, b: number): number => hash32((a ^ Math.imul(b | 0, 0x9e3779b1)) | 0);

/** Deterministic value noise in [0, 1) for integer coordinates, with no generator state. */
export const hash2 = (x: number, y: number, seed = 0): number =>
  hash32((Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b1)) | 0) /
  4294967296;

export class Rng {
  private a = 0;
  private b = 0;
  private c = 0;
  private d = 0;

  constructor(seed = 0, salt = 0) {
    this.reseed(seed, salt);
  }

  /** Restart from a seed; `salt` derives an independent sequence from the same seed. */
  reseed(seed: number, salt = 0): this {
    // splitmix32 spreads the two words over the 128-bit state, then a warm-up decorrelates nearby seeds
    let s = hashCombine(seed | 0, salt | 0);
    const mix = () => {
      s = (s + 0x9e3779b9) | 0;
      return hash32(s) | 0;
    };
    this.a = mix();
    this.b = mix();
    this.c = mix();
    this.d = mix();
    for (let i = 0; i < 12; i++) this.u32();
    return this;
  }

  /** A uniform unsigned 32-bit integer. */
  u32(): number {
    const a = this.a,
      b = this.b,
      c = this.c,
      d = this.d;
    const t = (((a + b) | 0) + d) | 0;
    this.d = (d + 1) | 0;
    this.a = b ^ (b >>> 9);
    this.b = (c + (c << 3)) | 0;
    this.c = (((c << 21) | (c >>> 11)) + t) | 0;
    return t >>> 0;
  }

  /** A float in [0, 1). */
  next(): number {
    const a = this.a,
      b = this.b,
      c = this.c,
      d = this.d;
    const t = (((a + b) | 0) + d) | 0;
    this.d = (d + 1) | 0;
    this.a = b ^ (b >>> 9);
    this.b = (c + (c << 3)) | 0;
    this.c = (((c << 21) | (c >>> 11)) + t) | 0;
    return (t >>> 0) / 4294967296;
  }

  /** A float in [lo, hi). */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** An integer in [0, n) with one argument, or in [lo, hi) with two. */
  int(lo: number, hi?: number): number {
    if (hi === undefined) return Math.floor(this.next() * lo);
    return lo + Math.floor(this.next() * (hi - lo));
  }

  chance(p: number): boolean {
    return this.next() < p;
  }

  /** -1 or 1. */
  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }

  /** An angle in [0, 2π). */
  angle(): number {
    return this.next() * Math.PI * 2;
  }

  /** Normally distributed (Box–Muller; two draws per call, no cached spare, so the call count stays simple). */
  gauss(mean = 0, sd = 1): number {
    const u = 1 - this.next();
    const v = this.next();
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
  }

  pick<T>(items: ArrayLike<T>): T {
    return items[Math.floor(this.next() * items.length)];
  }

  /** An index chosen in proportion to its weight (negative weights count as 0); -1 when every weight is 0. */
  weighted(weights: ArrayLike<number>): number {
    let total = 0;
    for (let i = 0; i < weights.length; i++) if (weights[i] > 0) total += weights[i];
    if (!(total > 0)) return -1;
    let r = this.next() * total;
    let last = -1;
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      if (!(w > 0)) continue;
      last = i;
      r -= w;
      if (r < 0) return i;
    }
    // Rounding can leave r a hair above 0 after the last positive weight
    return last;
  }

  /** An item chosen in proportion to `weightOf(item)`; undefined when every weight is 0. */
  pickWeighted<T>(items: readonly T[], weightOf: (item: T) => number): T | undefined {
    let total = 0;
    for (let i = 0; i < items.length; i++) {
      const w = weightOf(items[i]);
      if (w > 0) total += w;
    }
    if (!(total > 0)) return undefined;
    let r = this.next() * total;
    let last: T | undefined;
    for (let i = 0; i < items.length; i++) {
      const w = weightOf(items[i]);
      if (!(w > 0)) continue;
      last = items[i];
      r -= w;
      if (r < 0) return items[i];
    }
    return last;
  }

  /** Fisher–Yates in place; returns the same array. */
  shuffle<A extends { length: number; [i: number]: unknown }>(arr: A): A {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
    return arr;
  }

  /** Write the 4-word state into `out` at `offset` (for snapshots and replays). */
  save(out: Int32Array | Uint32Array | number[], offset = 0): void {
    out[offset] = this.a;
    out[offset + 1] = this.b;
    out[offset + 2] = this.c;
    out[offset + 3] = this.d;
  }

  load(src: ArrayLike<number>, offset = 0): this {
    this.a = src[offset] | 0;
    this.b = src[offset + 1] | 0;
    this.c = src[offset + 2] | 0;
    this.d = src[offset + 3] | 0;
    return this;
  }

  clone(): Rng {
    const r = new Rng();
    r.a = this.a;
    r.b = this.b;
    r.c = this.c;
    r.d = this.d;
    return r;
  }

  /** A hash of the current state, for determinism checks. */
  hash(): number {
    return hashCombine(hashCombine(this.a, this.b), hashCombine(this.c, this.d));
  }
}

/** Named, independent generators derived from one seed. */
export class RngStreams {
  private readonly map = new Map<string, Rng>();

  constructor(readonly seed: number) {}

  /** The generator for `name`, created on first use; the same name always gives the same sequence. */
  get(name: string): Rng {
    let r = this.map.get(name);
    if (!r) {
      r = new Rng(this.seed, hashString(name));
      this.map.set(name, r);
    }
    return r;
  }

  has(name: string): boolean {
    return this.map.has(name);
  }

  /** Stream names in a stable order. */
  names(): string[] {
    return [...this.map.keys()].sort();
  }

  /** Rewind every stream to its starting state. */
  reset(): void {
    for (const [name, r] of this.map) r.reseed(this.seed, hashString(name));
  }

  /** Combined hash of every stream's state, independent of creation order. */
  hash(): number {
    let h = 0;
    for (const name of this.names()) h = hashCombine(h, hashCombine(hashString(name), this.get(name).hash()));
    return h;
  }
}

export const streams = (seed: number): RngStreams => new RngStreams(seed);
