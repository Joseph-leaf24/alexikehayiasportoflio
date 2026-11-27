/**
 * Scalar, angle and 2D vector helpers. Everything here is allocation-free: vector helpers write into an `out`
 * argument the caller owns, so hot loops can reuse a few scratch vectors.
 */

export const TAU = Math.PI * 2;
export const DEG = Math.PI / 180;

export const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);
export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
/** Where `v` sits between `a` and `b` (0 at a, 1 at b); 0 when the range is empty. */
export const invLerp = (a: number, b: number, v: number): number => (a === b ? 0 : (v - a) / (b - a));
export const remap = (v: number, a0: number, a1: number, b0: number, b1: number): number =>
  b0 + (b1 - b0) * invLerp(a0, a1, v);
export const smoothstep = (e0: number, e1: number, v: number): number => {
  const t = clamp01(invLerp(e0, e1, v));
  return t * t * (3 - 2 * t);
};
export const sign = (v: number): number => (v > 0 ? 1 : v < 0 ? -1 : 0);
/** Modulo that is never negative, for ring buffers and wrapping coordinates. */
export const mod = (a: number, n: number): number => ((a % n) + n) % n;

/**
 * Exponential approach of `a` towards `b` at `lambda` per second. Unlike `lerp(a, b, k)` per frame, the result
 * after one second is the same whatever the frame rate.
 */
export const damp = (a: number, b: number, lambda: number, dt: number): number =>
  a + (b - a) * (1 - Math.exp(-lambda * dt));

/** Move `v` towards `target` by at most `step` without overshooting. */
export const approach = (v: number, target: number, step: number): number =>
  v < target ? (v + step > target ? target : v + step) : v - step < target ? target : v - step;

/* Angles (radians) */

/** Wrap into [-π, π). */
export const wrapAngle = (a: number): number => a - TAU * Math.floor((a + Math.PI) / TAU);
/** Signed shortest turn from `from` to `to`, in [-π, π). */
export const angleDiff = (from: number, to: number): number => wrapAngle(to - from);
export const lerpAngle = (a: number, b: number, t: number): number => a + angleDiff(a, b) * t;
export const dampAngle = (a: number, b: number, lambda: number, dt: number): number =>
  a + angleDiff(a, b) * (1 - Math.exp(-lambda * dt));
/** Turn `a` towards `b` by at most `step`. */
export const approachAngle = (a: number, b: number, step: number): number => {
  const d = angleDiff(a, b);
  return d > step ? a + step : d < -step ? a - step : b;
};
export const angleTo = (x0: number, y0: number, x1: number, y1: number): number => Math.atan2(y1 - y0, x1 - x0);

/* Distances (Math.hypot is several times slower than the plain square root) */

export const len = (x: number, y: number): number => Math.sqrt(x * x + y * y);
export const dist2 = (x0: number, y0: number, x1: number, y1: number): number => {
  const dx = x1 - x0,
    dy = y1 - y0;
  return dx * dx + dy * dy;
};
export const dist = (x0: number, y0: number, x1: number, y1: number): number => Math.sqrt(dist2(x0, y0, x1, y1));

/** Squared distance from point (px, py) to the segment (x0, y0)-(x1, y1). */
export function segmentDist2(px: number, py: number, x0: number, y0: number, x1: number, y1: number): number {
  const dx = x1 - x0,
    dy = y1 - y0;
  const l2 = dx * dx + dy * dy;
  let t = l2 > 0 ? ((px - x0) * dx + (py - y0) * dy) / l2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const ex = x0 + dx * t - px,
    ey = y0 + dy * t - py;
  return ex * ex + ey * ey;
}

/* 2D vectors. Allocate scratch vectors once (at boot) with `vec2()`; the helpers return `out`. */

export interface Vec2 {
  x: number;
  y: number;
}

export const vec2 = (x = 0, y = 0): Vec2 => ({ x, y });

export function v2set(out: Vec2, x: number, y: number): Vec2 {
  out.x = x;
  out.y = y;
  return out;
}
export function v2copy(out: Vec2, a: Vec2): Vec2 {
  out.x = a.x;
  out.y = a.y;
  return out;
}
export function v2add(out: Vec2, a: Vec2, b: Vec2): Vec2 {
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}
export function v2sub(out: Vec2, a: Vec2, b: Vec2): Vec2 {
  out.x = a.x - b.x;
  out.y = a.y - b.y;
  return out;
}
export function v2scale(out: Vec2, a: Vec2, s: number): Vec2 {
  out.x = a.x * s;
  out.y = a.y * s;
  return out;
}
/** out = a + b × s */
export function v2addScaled(out: Vec2, a: Vec2, b: Vec2, s: number): Vec2 {
  out.x = a.x + b.x * s;
  out.y = a.y + b.y * s;
  return out;
}
export function v2lerp(out: Vec2, a: Vec2, b: Vec2, t: number): Vec2 {
  out.x = a.x + (b.x - a.x) * t;
  out.y = a.y + (b.y - a.y) * t;
  return out;
}
export const v2dot = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
/** The z of the 3D cross product: positive when b is counter-clockwise from a (y up). */
export const v2cross = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;
export const v2len = (a: Vec2): number => Math.sqrt(a.x * a.x + a.y * a.y);
export const v2dist = (a: Vec2, b: Vec2): number => dist(a.x, a.y, b.x, b.y);

/** Normalise `a` into `out` and return the original length; a zero vector stays zero. */
export function v2normalize(out: Vec2, a: Vec2): number {
  const l = Math.sqrt(a.x * a.x + a.y * a.y);
  if (l > 0) {
    out.x = a.x / l;
    out.y = a.y / l;
  } else {
    out.x = 0;
    out.y = 0;
  }
  return l;
}
export function v2rotate(out: Vec2, a: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle),
    s = Math.sin(angle);
  const x = a.x * c - a.y * s;
  out.y = a.x * s + a.y * c;
  out.x = x;
  return out;
}
export function v2fromAngle(out: Vec2, angle: number, length = 1): Vec2 {
  out.x = Math.cos(angle) * length;
  out.y = Math.sin(angle) * length;
  return out;
}
/** Scale `a` down into `out` so its length is at most `max`. */
export function v2limit(out: Vec2, a: Vec2, max: number): Vec2 {
  const l2 = a.x * a.x + a.y * a.y;
  if (l2 > max * max) {
    const k = max / Math.sqrt(l2);
    out.x = a.x * k;
    out.y = a.y * k;
  } else {
    out.x = a.x;
    out.y = a.y;
  }
  return out;
}

/** Smallest power of two ≥ n, for n up to 2^30. */
export const nextPow2 = (n: number): number => (n <= 1 ? 1 : 2 ** (32 - Math.clz32(Math.ceil(n) - 1)));
