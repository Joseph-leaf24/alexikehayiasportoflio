/**
 * A uniform-grid spatial hash over body boxes, rebuilt every tick with a counting sort into one packed array per
 * field (bucket order), so a query touches a few contiguous runs instead of every unit. Cells wrap modulo the
 * table, which keeps an endless world in a fixed table; exact box tests drop the aliased cells' entries. Bodies
 * bigger than a cell sit in a short side list every query scans, so one boss doesn't widen every query.
 *
 * A body's box is [x - hw, x + hw] × [y - h, y] (pos is the feet). Distances are from the query point to the
 * nearest point of the box, 0 inside it. Ties are broken by the smaller id, so results don't depend on the
 * traversal order. Queries write into caller-owned arrays and return the count; nothing allocates once warm.
 */

import { BODY_UNINDEXED } from "./components";
import type { Query } from "./ecs";
import { ENTITY_DOOMED, INDEX_BITS, INDEX_MASK } from "./ecs";
import { nextPow2 } from "./math";
import type { World } from "./world";

/** Every team. */
export const TEAM_ALL = -1;
export const teamBit = (team: number): number => 1 << (team & 31);
/** Every team except `team` (and, unless `withNeutral`, except neutral team 0). */
export const enemiesOf = (team: number, withNeutral = false): number =>
  ~teamBit(team) & (withNeutral ? -1 : ~teamBit(0));

/** Return false to skip a candidate. */
export type SpatialFilter = (id: number, index: number) => boolean;

export interface SpatialOptions {
  /** Cell size in px (default 64). */
  cell?: number;
  /** Table width and height in cells, rounded up to powers of two (default 64 × 32: 4096 × 2048 px). */
  cols?: number;
  rows?: number;
}

const grow64 = (a: Float64Array, n: number): Float64Array => {
  const b = new Float64Array(n);
  b.set(a);
  return b;
};
const grow32 = (a: Int32Array, n: number): Int32Array => {
  const b = new Int32Array(n);
  b.set(a);
  return b;
};

/** A packed list of boxes: insertion order while staging, bucket order once built. */
class Boxes {
  n = 0;
  id = new Int32Array(64);
  x0 = new Float64Array(64);
  y0 = new Float64Array(64);
  x1 = new Float64Array(64);
  y1 = new Float64Array(64);
  team = new Int32Array(64);
  flags = new Int32Array(64);

  reserve(n: number): void {
    if (n <= this.id.length) return;
    const c = nextPow2(n);
    this.id = grow32(this.id, c);
    this.x0 = grow64(this.x0, c);
    this.y0 = grow64(this.y0, c);
    this.x1 = grow64(this.x1, c);
    this.y1 = grow64(this.y1, c);
    this.team = grow32(this.team, c);
    this.flags = grow32(this.flags, c);
  }

  push(id: number, x0: number, y0: number, x1: number, y1: number, teamBits: number, flags: number): number {
    const k = this.n++;
    if (k >= this.id.length) this.reserve(k + 1);
    this.id[k] = id;
    this.x0[k] = x0;
    this.y0[k] = y0;
    this.x1[k] = x1;
    this.y1[k] = y1;
    this.team[k] = teamBits;
    this.flags[k] = flags;
    return k;
  }
}

export class SpatialHash {
  readonly cell: number;
  readonly cols: number;
  readonly rows: number;
  /** Distance to the result of the last `nearest` (Infinity when none). */
  lastDist = Infinity;
  /** Entry point along the segment (0..1) of the last `segmentFirst` hit. */
  lastT = 1;
  /** Entry point along the segment (0..1) per result of the last `segment` call. */
  hitT = new Float64Array(64);

  private readonly inv: number;
  private readonly maskX: number;
  private readonly maskY: number;
  /** CSR offsets: bucket b holds sorted entries [start[b], start[b + 1]). */
  private readonly start: Int32Array;
  private readonly cursor: Int32Array;
  /** Scratch for `rowRanges`: up to two [lo, hi) entry ranges. */
  private readonly ranges = new Int32Array(4);
  private staged = new Boxes();
  private bucketOf = new Int32Array(64);
  private grid = new Boxes();
  private big = new Boxes();
  /** Largest half extents among the gridded (small) boxes: how far a box can reach out of its centre's cell. */
  private ex = 0;
  private ey = 0;
  /** Read at query time (not cached) because the ECS replaces its arrays when it grows. */
  private ecs: { readonly state: Uint8Array } | null = null;
  private bodyQuery: Query | null = null;

  constructor(opts: SpatialOptions = {}) {
    this.cell = opts.cell ?? 64;
    this.inv = 1 / this.cell;
    this.cols = nextPow2(opts.cols ?? 64);
    this.rows = nextPow2(opts.rows ?? 32);
    this.maskX = this.cols - 1;
    this.maskY = this.rows - 1;
    this.start = new Int32Array(this.cols * this.rows + 1);
    this.cursor = new Int32Array(this.cols * this.rows + 1);
  }

  /** Boxes in the index (gridded and big). */
  get size(): number {
    return this.grid.n + this.big.n;
  }

  /* Building */

  /** Start a rebuild. `ecs` (optional) lets queries skip entities doomed after the build. */
  begin(ecs?: { readonly state: Uint8Array } | null): void {
    this.staged.n = 0;
    this.big.n = 0;
    this.ex = 0;
    this.ey = 0;
    this.ecs = ecs ?? null;
  }

  /** Add one box (x0 ≤ x1, y0 ≤ y1). */
  insert(id: number, x0: number, y0: number, x1: number, y1: number, team: number, flags = 0): void {
    // Extents measured from the rounded centre, so the query margins are exact
    const cx = (x0 + x1) * 0.5,
      cy = (y0 + y1) * 0.5;
    const hx = Math.max(cx - x0, x1 - cx),
      hy = Math.max(cy - y0, y1 - cy);
    const tb = teamBit(team);
    if (hx > this.cell || hy > this.cell) {
      this.big.push(id, x0, y0, x1, y1, tb, flags);
      return;
    }
    const k = this.staged.push(id, x0, y0, x1, y1, tb, flags);
    if (k >= this.bucketOf.length) this.bucketOf = grow32(this.bucketOf, nextPow2(k + 1));
    this.bucketOf[k] = this.bucket(Math.floor(cx * this.inv), Math.floor(cy * this.inv));
    if (hx > this.ex) this.ex = hx;
    if (hy > this.ey) this.ey = hy;
  }

  /** Finish a rebuild: counting-sort the staged boxes into bucket order. */
  end(): void {
    const n = this.staged.n,
      start = this.start,
      buckets = this.cols * this.rows;
    start.fill(0);
    const bucketOf = this.bucketOf;
    for (let k = 0; k < n; k++) start[bucketOf[k] + 1]++;
    for (let b = 1; b <= buckets; b++) start[b] += start[b - 1];
    const cursor = this.cursor;
    cursor.set(start);
    const s = this.staged,
      g = this.grid;
    g.reserve(n);
    for (let k = 0; k < n; k++) {
      const p = cursor[bucketOf[k]]++;
      g.id[p] = s.id[k];
      g.x0[p] = s.x0[k];
      g.y0[p] = s.y0[k];
      g.x1[p] = s.x1[k];
      g.y1[p] = s.y1[k];
      g.team[p] = s.team[k];
      g.flags[p] = s.flags[k];
    }
    g.n = n;
  }

  /**
   * Rebuild from every live entity with `pos` and `body` (team from `team`, else neutral), skipping bodies
   * flagged `BODY_UNINDEXED` and entities already doomed.
   */
  rebuild(world: World): void {
    const { pos, body, team } = world.core;
    this.bodyQuery ??= world.query(pos, body);
    const q = this.bodyQuery;
    const n = q.collect(true),
      idx = q.idx;
    const px = pos.x,
      py = pos.y,
      hw = body.hw,
      bh = body.h,
      bf = body.flags,
      tid = team.id,
      tsparse = team.$.sparse,
      gen = world.gen;
    this.begin(world);
    const s = this.staged;
    s.reserve(n);
    if (this.bucketOf.length < s.id.length) this.bucketOf = grow32(this.bucketOf, s.id.length);
    const bucketOf = this.bucketOf,
      cell = this.cell,
      inv = this.inv;
    let ex = 0,
      ey = 0;
    // insert(), inlined: a call per body would box its four coordinates
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      const flags = bf[i];
      if (flags & BODY_UNINDEXED) continue;
      const x = px[i],
        y = py[i];
      const x0 = x - hw[i],
        x1 = x + hw[i],
        y0 = y - bh[i];
      const id = (gen[i] << INDEX_BITS) | i,
        tb = teamBit(tsparse[i] >= 0 ? tid[i] : 0);
      const cx = (x0 + x1) * 0.5,
        cy = (y0 + y) * 0.5;
      const hx = cx - x0 > x1 - cx ? cx - x0 : x1 - cx,
        hy = cy - y0 > y - cy ? cy - y0 : y - cy;
      if (hx > cell || hy > cell) {
        this.big.push(id, x0, y0, x1, y, tb, flags);
        continue;
      }
      const j = s.n++;
      s.id[j] = id;
      s.x0[j] = x0;
      s.y0[j] = y0;
      s.x1[j] = x1;
      s.y1[j] = y;
      s.team[j] = tb;
      s.flags[j] = flags;
      bucketOf[j] = this.bucket(Math.floor(cx * inv), Math.floor(cy * inv));
      if (hx > ex) ex = hx;
      if (hy > ey) ey = hy;
    }
    this.ex = ex;
    this.ey = ey;
    this.end();
  }

  private bucket(cx: number, cy: number): number {
    return (cy & this.maskY) * this.cols + (cx & this.maskX);
  }

  /**
   * The entries of cells cx0..cx1 (at most one lap) in row cy. A row's buckets are adjacent in the sorted arrays,
   * so that is one contiguous range, or two when the span wraps round the table; they go into `ranges` as
   * [lo, hi) pairs and the pair count is returned. Empty cells cost nothing this way.
   */
  private rowRanges(cy: number, cx0: number, cx1: number): number {
    const rb = (cy & this.maskY) * this.cols,
      a = cx0 & this.maskX,
      b = cx1 & this.maskX,
      st = this.start,
      R = this.ranges;
    R[0] = st[rb + a];
    if (a <= b) {
      R[1] = st[rb + b + 1];
      return 1;
    }
    R[1] = st[rb + this.cols];
    R[2] = st[rb];
    R[3] = st[rb + b + 1];
    return 2;
  }

  /** Doomed since the build, or refused by the filter (the team test comes first, inline). */
  private rejects(id: number, filter: SpatialFilter | null | undefined): boolean {
    const ecs = this.ecs;
    if (ecs !== null && (ecs.state[id & INDEX_MASK] & ENTITY_DOOMED) !== 0) return true;
    return filter != null && !filter(id, id & INDEX_MASK);
  }

  /* Queries */

  /**
   * The closest body within `r` of (x, y) whose team is in `teamMask`, or 0. Probes the 3×3 cells around the
   * point, then, unless that settles it, scans the rectangle the best hit so far (or `r`) can reach: two passes at
   * most, each a few contiguous row ranges. `lastDist` gets its distance.
   */
  nearest(x: number, y: number, r: number, teamMask: number, filter?: SpatialFilter | null, exclude = 0): number {
    let best = r * r,
      bestId = 0;
    const g = this.grid,
      cell = this.cell,
      R = this.ranges;
    const reach = this.ex > this.ey ? this.ex : this.ey,
      inv = this.inv;
    const ccx = Math.floor(x * inv),
      ccy = Math.floor(y * inv);
    // Pass 0 probes the 3×3 cells around the point; pass 1 scans the rectangle the best hit (or r) needs
    let pass = r + reach <= cell ? 1 : 0;
    for (;;) {
      let cx0 = ccx - 1,
        cx1 = ccx + 1,
        cy0 = ccy - 1,
        cy1 = ccy + 1;
      if (pass === 1) {
        const reachOut = (bestId !== 0 ? Math.sqrt(best) : r) + reach;
        cx0 = Math.floor((x - reachOut) * inv);
        cx1 = Math.floor((x + reachOut) * inv);
        cy0 = Math.floor((y - reachOut) * inv);
        cy1 = Math.floor((y + reachOut) * inv);
      }
      if (cx1 - cx0 + 1 >= this.cols || cy1 - cy0 + 1 >= this.rows) {
        // The square would wrap round the table: a straight scan is exact and still cheap
        for (let k = 0; k < g.n; k++) {
          if ((g.team[k] & teamMask) === 0) continue;
          const id = g.id[k];
          if (id === exclude) continue;
          const bx0 = g.x0[k],
            bx1 = g.x1[k],
            by0 = g.y0[k],
            by1 = g.y1[k];
          const ddx = x < bx0 ? bx0 - x : x > bx1 ? x - bx1 : 0,
            ddy = y < by0 ? by0 - y : y > by1 ? y - by1 : 0;
          const d = ddx * ddx + ddy * ddy;
          if (d > best || (d === best && bestId !== 0 && id > bestId)) continue;
          if (this.rejects(id, filter)) continue;
          best = d;
          bestId = id;
        }
        break;
      }
      for (let cy = cy0; cy <= cy1; cy++) {
        const np = this.rowRanges(cy, cx0, cx1) * 2;
        for (let p = 0; p < np; p += 2) {
          for (let k = R[p], e = R[p + 1]; k < e; k++) {
            if ((g.team[k] & teamMask) === 0) continue;
            const id = g.id[k];
            if (id === exclude) continue;
            const bx0 = g.x0[k],
              bx1 = g.x1[k],
              by0 = g.y0[k],
              by1 = g.y1[k];
            const ddx = x < bx0 ? bx0 - x : x > bx1 ? x - bx1 : 0,
              ddy = y < by0 ? by0 - y : y > by1 ? y - by1 : 0;
            const d = ddx * ddx + ddy * ddy;
            if (d > best || (d === best && bestId !== 0 && id > bestId)) continue;
            if (this.rejects(id, filter)) continue;
            best = d;
            bestId = id;
          }
        }
      }
      if (pass === 1) break;
      // Every box centred outside the probe is more than `inner` away: done if the best hit is closer
      const inner = Math.min(x - cx0 * cell, (cx1 + 1) * cell - x, y - cy0 * cell, (cy1 + 1) * cell - y) - reach;
      if (inner > r || (bestId !== 0 && inner > 0 && best < inner * inner)) break;
      pass = 1;
    }
    const big = this.big;
    for (let k = 0; k < big.n; k++) {
      if ((big.team[k] & teamMask) === 0) continue;
      const id = big.id[k];
      if (id === exclude) continue;
      const bx0 = big.x0[k],
        bx1 = big.x1[k],
        by0 = big.y0[k],
        by1 = big.y1[k];
      const ddx = x < bx0 ? bx0 - x : x > bx1 ? x - bx1 : 0,
        ddy = y < by0 ? by0 - y : y > by1 ? y - by1 : 0;
      const d = ddx * ddx + ddy * ddy;
      if (d > best || (d === best && bestId !== 0 && id > bestId)) continue;
      if (this.rejects(id, filter)) continue;
      best = d;
      bestId = id;
    }
    this.lastDist = bestId !== 0 ? Math.sqrt(best) : Infinity;
    return bestId;
  }

  /** Every body within `r` of (x, y) whose team is in `teamMask`, written into `out`; returns the count. */
  inRadius(x: number, y: number, r: number, teamMask: number, out: number[], filter?: SpatialFilter | null): number {
    const r2 = r * r;
    let c = 0;
    const g = this.grid,
      R = this.ranges;
    const cx0 = Math.floor((x - r - this.ex) * this.inv),
      cx1 = Math.floor((x + r + this.ex) * this.inv);
    const cy0 = Math.floor((y - r - this.ey) * this.inv),
      cy1 = Math.floor((y + r + this.ey) * this.inv);
    // A range as wide as the table would visit buckets twice: clamp it to one lap
    const xa = cx1 - cx0 + 1 >= this.cols ? 0 : cx0,
      xb = cx1 - cx0 + 1 >= this.cols ? this.cols - 1 : cx1;
    const ya = cy1 - cy0 + 1 >= this.rows ? 0 : cy0,
      yb = cy1 - cy0 + 1 >= this.rows ? this.rows - 1 : cy1;
    for (let cy = ya; cy <= yb; cy++) {
      const np = this.rowRanges(cy, xa, xb) * 2;
      for (let p = 0; p < np; p += 2) {
        for (let k = R[p], e = R[p + 1]; k < e; k++) {
          if ((g.team[k] & teamMask) === 0) continue;
          const bx0 = g.x0[k],
            bx1 = g.x1[k],
            by0 = g.y0[k],
            by1 = g.y1[k];
          const ddx = x < bx0 ? bx0 - x : x > bx1 ? x - bx1 : 0,
            ddy = y < by0 ? by0 - y : y > by1 ? y - by1 : 0;
          const d = ddx * ddx + ddy * ddy;
          if (d > r2) continue;
          const id = g.id[k];
          if (this.rejects(id, filter)) continue;
          out[c++] = id;
        }
      }
    }
    const big = this.big;
    for (let k = 0; k < big.n; k++) {
      if ((big.team[k] & teamMask) === 0) continue;
      const bx0 = big.x0[k],
        bx1 = big.x1[k],
        by0 = big.y0[k],
        by1 = big.y1[k];
      const ddx = x < bx0 ? bx0 - x : x > bx1 ? x - bx1 : 0,
        ddy = y < by0 ? by0 - y : y > by1 ? y - by1 : 0;
      const d = ddx * ddx + ddy * ddy;
      if (d > r2) continue;
      const id = big.id[k];
      if (this.rejects(id, filter)) continue;
      out[c++] = id;
    }
    return c;
  }

  /** Every body overlapping the rectangle [x0, x1] × [y0, y1]; returns the count. */
  rect(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    teamMask: number,
    out: number[],
    filter?: SpatialFilter | null
  ): number {
    let c = 0;
    const g = this.grid;
    const cx0 = Math.floor((x0 - this.ex) * this.inv),
      cx1 = Math.floor((x1 + this.ex) * this.inv);
    const cy0 = Math.floor((y0 - this.ey) * this.inv),
      cy1 = Math.floor((y1 + this.ey) * this.inv);
    const xa = cx1 - cx0 + 1 >= this.cols ? 0 : cx0,
      xb = cx1 - cx0 + 1 >= this.cols ? this.cols - 1 : cx1;
    const ya = cy1 - cy0 + 1 >= this.rows ? 0 : cy0,
      yb = cy1 - cy0 + 1 >= this.rows ? this.rows - 1 : cy1;
    const R = this.ranges;
    for (let cy = ya; cy <= yb; cy++) {
      const np = this.rowRanges(cy, xa, xb) * 2;
      for (let p = 0; p < np; p += 2) {
        for (let k = R[p], e = R[p + 1]; k < e; k++) {
          if ((g.team[k] & teamMask) === 0) continue;
          if (g.x0[k] > x1 || g.x1[k] < x0 || g.y0[k] > y1 || g.y1[k] < y0) continue;
          const id = g.id[k];
          if (this.rejects(id, filter)) continue;
          out[c++] = id;
        }
      }
    }
    const big = this.big;
    for (let k = 0; k < big.n; k++) {
      if ((big.team[k] & teamMask) === 0) continue;
      if (big.x0[k] > x1 || big.x1[k] < x0 || big.y0[k] > y1 || big.y1[k] < y0) continue;
      const id = big.id[k];
      if (this.rejects(id, filter)) continue;
      out[c++] = id;
    }
    return c;
  }

  /**
   * Every body a shot `width` px thick from (x0, y0) to (x1, y1) touches, nearest first, written into `out`;
   * `hitT[k]` holds where along the segment (0..1) result k is entered. Returns the count.
   */
  segment(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    width: number,
    teamMask: number,
    out: number[],
    filter?: SpatialFilter | null
  ): number {
    this.segCount = 0;
    this.segOut = out;
    this.sweep(x0, y0, x1, y1, width, teamMask, filter, false);
    this.segOut = null;
    return this.segCount;
  }

  /** The first body a shot touches, or 0; `lastT` gets where along the segment (0..1) it is entered. */
  segmentFirst(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    width: number,
    teamMask: number,
    filter?: SpatialFilter | null
  ): number {
    this.firstId = 0;
    this.firstT = Infinity;
    this.sweep(x0, y0, x1, y1, width, teamMask, filter, true);
    this.lastT = this.firstId !== 0 ? this.firstT : 1;
    return this.firstId;
  }

  private segOut: number[] | null = null;
  private segCount = 0;
  private firstId = 0;
  private firstT = Infinity;
  /** The segment being swept: x0, y0, dx, dy, pad. A typed array, so the per-box test passes no doubles. */
  private readonly seg = new Float64Array(5);

  /** Visit the cells along a thick segment, row by row, testing each box once. */
  private sweep(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    width: number,
    teamMask: number,
    filter: SpatialFilter | null | undefined,
    firstOnly: boolean
  ): void {
    const dx = x1 - x0,
      dy = y1 - y0,
      pad = width * 0.5;
    const seg = this.seg;
    seg[0] = x0;
    seg[1] = y0;
    seg[2] = dx;
    seg[3] = dy;
    seg[4] = pad;
    const g = this.grid;
    const padX = pad + this.ex,
      padY = pad + this.ey;
    const ymin = Math.min(y0, y1),
      ymax = Math.max(y0, y1);
    const cy0 = Math.floor((ymin - padY) * this.inv),
      cy1 = Math.floor((ymax + padY) * this.inv);
    const xmin = Math.min(x0, x1),
      xmax = Math.max(x0, x1);
    const spanX = Math.floor((xmax + padX) * this.inv) - Math.floor((xmin - padX) * this.inv) + 1;
    if (cy1 - cy0 + 1 >= this.rows || spanX >= this.cols) {
      for (let k = 0; k < g.n; k++) this.segTest(g, k, teamMask, filter, firstOnly);
    } else {
      for (let cy = cy0; cy <= cy1; cy++) {
        // The part of the segment whose y can reach boxes centred in this row
        let ta = 0,
          tb = 1;
        if (dy !== 0) {
          const ya = cy * this.cell - padY,
            yb = (cy + 1) * this.cell + padY;
          let t0 = (ya - y0) / dy,
            t1 = (yb - y0) / dy;
          if (t0 > t1) {
            const t = t0;
            t0 = t1;
            t1 = t;
          }
          ta = t0 > 0 ? t0 : 0;
          tb = t1 < 1 ? t1 : 1;
          if (ta > tb) continue;
        }
        const xa = x0 + dx * ta,
          xb = x0 + dx * tb;
        const cxa = Math.floor((Math.min(xa, xb) - padX) * this.inv),
          cxb = Math.floor((Math.max(xa, xb) + padX) * this.inv);
        const np = this.rowRanges(cy, cxa, cxb) * 2,
          R = this.ranges;
        for (let p = 0; p < np; p += 2)
          for (let k = R[p], e = R[p + 1]; k < e; k++) this.segTest(g, k, teamMask, filter, firstOnly);
      }
    }
    const big = this.big;
    for (let k = 0; k < big.n; k++) this.segTest(big, k, teamMask, filter, firstOnly);
  }

  /** Test box k against the segment in `seg` (the slab test of `segBox`, inlined). */
  private segTest(
    b: Boxes,
    k: number,
    teamMask: number,
    filter: SpatialFilter | null | undefined,
    firstOnly: boolean
  ): void {
    if ((b.team[k] & teamMask) === 0) return;
    const seg = this.seg;
    const px = seg[0],
      py = seg[1],
      dx = seg[2],
      dy = seg[3],
      pad = seg[4];
    let t = 0,
      t1 = 1;
    if (dx !== 0) {
      let a = (b.x0[k] - pad - px) / dx,
        c = (b.x1[k] + pad - px) / dx;
      if (a > c) {
        const tmp = a;
        a = c;
        c = tmp;
      }
      if (a > t) t = a;
      if (c < t1) t1 = c;
      if (t > t1) return;
    } else if (px < b.x0[k] - pad || px > b.x1[k] + pad) return;
    if (dy !== 0) {
      let a = (b.y0[k] - pad - py) / dy,
        c = (b.y1[k] + pad - py) / dy;
      if (a > c) {
        const tmp = a;
        a = c;
        c = tmp;
      }
      if (a > t) t = a;
      if (c < t1) t1 = c;
      if (t > t1) return;
    } else if (py < b.y0[k] - pad || py > b.y1[k] + pad) return;
    const id = b.id[k];
    if (firstOnly) {
      if (t > this.firstT || (t === this.firstT && id > this.firstId)) return;
      if (this.rejects(id, filter)) return;
      this.firstT = t;
      this.firstId = id;
      return;
    }
    if (this.rejects(id, filter)) return;
    // Insertion into the nearest-first result list
    const out = this.segOut as number[];
    let j = this.segCount++;
    if (j >= this.hitT.length) {
      const h = new Float64Array(this.hitT.length * 2);
      h.set(this.hitT);
      this.hitT = h;
    }
    const hitT = this.hitT;
    while (j > 0 && (hitT[j - 1] > t || (hitT[j - 1] === t && out[j - 1] > id))) {
      hitT[j] = hitT[j - 1];
      out[j] = out[j - 1];
      j--;
    }
    hitT[j] = t;
    out[j] = id;
  }
}

/** Where (0..1) the segment p + t·d enters the box, or -1 when it misses. */
export function segBox(
  px: number,
  py: number,
  dx: number,
  dy: number,
  bx0: number,
  by0: number,
  bx1: number,
  by1: number
): number {
  let t0 = 0,
    t1 = 1;
  if (dx !== 0) {
    let a = (bx0 - px) / dx,
      b = (bx1 - px) / dx;
    if (a > b) {
      const t = a;
      a = b;
      b = t;
    }
    if (a > t0) t0 = a;
    if (b < t1) t1 = b;
    if (t0 > t1) return -1;
  } else if (px < bx0 || px > bx1) return -1;
  if (dy !== 0) {
    let a = (by0 - py) / dy,
      b = (by1 - py) / dy;
    if (a > b) {
      const t = a;
      a = b;
      b = t;
    }
    if (a > t0) t0 = a;
    if (b < t1) t1 = b;
    if (t0 > t1) return -1;
  } else if (py < by0 || py > by1) return -1;
  return t0;
}
