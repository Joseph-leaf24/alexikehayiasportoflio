import { describe, expect, it } from "vitest";
import {
  BODY_UNINDEXED,
  tick,
  Rng,
  SpatialHash,
  TEAM_ALL,
  World,
  enemiesOf,
  indexOf,
  place,
  segBox,
  teamBit,
  type SpatialFilter,
} from "@/engine/core";

interface Box {
  id: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  team: number;
}

const dist2 = (b: Box, x: number, y: number) => {
  const dx = x < b.x0 ? b.x0 - x : x > b.x1 ? x - b.x1 : 0;
  const dy = y < b.y0 ? b.y0 - y : y > b.y1 ? y - b.y1 : 0;
  return dx * dx + dy * dy;
};
const inMask = (b: Box, mask: number) => (teamBit(b.team) & mask) !== 0;
const ok = (b: Box, mask: number, filter?: SpatialFilter) =>
  inMask(b, mask) && (!filter || filter(b.id, indexOf(b.id)));

function bruteNearest(boxes: Box[], x: number, y: number, r: number, mask: number, filter?: SpatialFilter, ex = 0) {
  let best = r * r,
    id = 0;
  for (const b of boxes) {
    if (b.id === ex || !ok(b, mask, filter)) continue;
    const d = dist2(b, x, y);
    if (d < best || (d === best && (id === 0 || b.id < id))) {
      best = d;
      id = b.id;
    }
  }
  return id;
}
const bruteRadius = (boxes: Box[], x: number, y: number, r: number, mask: number, filter?: SpatialFilter) =>
  boxes
    .filter((b) => ok(b, mask, filter) && dist2(b, x, y) <= r * r)
    .map((b) => b.id)
    .sort((a, b) => a - b);
const bruteRect = (boxes: Box[], x0: number, y0: number, x1: number, y1: number, mask: number) =>
  boxes
    .filter((b) => inMask(b, mask) && !(b.x0 > x1 || b.x1 < x0 || b.y0 > y1 || b.y1 < y0))
    .map((b) => b.id)
    .sort((a, b) => a - b);
function bruteSegment(boxes: Box[], x0: number, y0: number, x1: number, y1: number, w: number, mask: number) {
  const p = w / 2;
  const hits: [number, number][] = [];
  for (const b of boxes) {
    if (!inMask(b, mask)) continue;
    const t = segBox(x0, y0, x1 - x0, y1 - y0, b.x0 - p, b.y0 - p, b.x1 + p, b.y1 + p);
    if (t >= 0) hits.push([t, b.id]);
  }
  return hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

/** A random scene: mostly unit-sized boxes, a few giants, spread past the table so cells alias. */
function scene(seed: number, n: number, spread: { x0: number; x1: number; y0: number; y1: number }, big = 6) {
  const r = new Rng(seed);
  const boxes: Box[] = [];
  for (let k = 0; k < n; k++) {
    const x = r.range(spread.x0, spread.x1),
      y = r.range(spread.y0, spread.y1);
    const giant = k < big;
    const hw = giant ? r.range(70, 160) : r.range(0, 24);
    const h = giant ? r.range(130, 300) : r.range(0, 48);
    boxes.push({ id: (1 << 20) * (1 + (k % 7)) + k, x0: x - hw, y0: y - h, x1: x + hw, y1: y, team: r.int(0, 6) });
  }
  // Exact duplicates exercise the tie-break
  for (let k = 0; k < 8; k++) boxes.push({ ...boxes[big + k], id: boxes[big + k].id + 500_000 });
  const grid = new SpatialHash();
  grid.begin();
  for (const b of boxes) grid.insert(b.id, b.x0, b.y0, b.x1, b.y1, b.team);
  grid.end();
  return { boxes, grid, r };
}

const SPREADS = {
  ring: { x0: -300, x1: 4400, y0: -200, y1: 2000 },
  wide: { x0: -9000, x1: 20000, y0: -5000, y1: 7000 },
  far: { x0: 1_000_000, x1: 1_006_000, y0: 0, y1: 1900 },
  dense: { x0: 100, x1: 260, y0: 100, y1: 200 },
};

describe("spatial hash against brute force", () => {
  for (const [name, spread] of Object.entries(SPREADS)) {
    it(`nearest, inRadius, rect, segment and segmentFirst agree (${name})`, () => {
      const { boxes, grid, r } = scene(name.length * 7919, 420, spread);
      expect(grid.size).toBe(boxes.length);
      const out: number[] = [];
      let found = 0,
        listed = 0,
        shot = 0;
      for (let q = 0; q < 400; q++) {
        const x = r.range(spread.x0 - 200, spread.x1 + 200),
          y = r.range(spread.y0 - 200, spread.y1 + 200);
        const rad = q % 10 === 0 ? r.range(800, 3000) : r.range(0, 400);
        const mask = q % 4 === 0 ? TEAM_ALL : r.int(1, 64);
        const ex = q % 5 === 0 ? boxes[r.int(0, boxes.length)].id : 0;
        const filter: SpatialFilter | undefined = q % 3 === 0 ? (id) => id % 3 !== 0 : undefined;

        const near = grid.nearest(x, y, rad, mask, filter, ex);
        expect(near).toBe(bruteNearest(boxes, x, y, rad, mask, filter, ex));
        if (near) found++;

        const n = grid.inRadius(x, y, rad, mask, out, filter);
        const got = out.slice(0, n).sort((a, b) => a - b);
        expect(got).toEqual(bruteRadius(boxes, x, y, rad, mask, filter));
        expect(new Set(got).size).toBe(n);
        listed += n;

        const w = r.range(10, 600),
          hgt = r.range(10, 600);
        const nr = grid.rect(x, y, x + w, y + hgt, mask, out);
        expect(out.slice(0, nr).sort((a, b) => a - b)).toEqual(bruteRect(boxes, x, y, x + w, y + hgt, mask));

        // Shots: short and long, all directions, some degenerate
        const len = q % 7 === 0 ? 0 : q % 9 === 0 ? r.range(2000, 7000) : r.range(20, 900);
        const ang = q % 11 === 0 ? 0 : q % 13 === 0 ? Math.PI / 2 : r.angle();
        const x1 = x + Math.cos(ang) * len,
          y1 = y + Math.sin(ang) * len,
          width = r.chance(0.3) ? 0 : r.range(0, 30);
        const ns = grid.segment(x, y, x1, y1, width, mask, out);
        const want = bruteSegment(boxes, x, y, x1, y1, width, mask);
        expect(out.slice(0, ns)).toEqual(want.map((h) => h[1]));
        shot += ns;
        for (let k = 0; k < ns; k++) expect(grid.hitT[k]).toBe(want[k][0]);
        const first = grid.segmentFirst(x, y, x1, y1, width, mask);
        expect(first).toBe(want.length ? want[0][1] : 0);
        if (want.length) expect(grid.lastT).toBe(want[0][0]);
      }
      // The scenes are busy enough that the comparisons mean something
      expect(found).toBeGreaterThan(40);
      expect(listed).toBeGreaterThan(200);
      expect(shot).toBeGreaterThan(5);
    });
  }

  it("nearest with an unlimited radius finds the closest anywhere, as brute force does", () => {
    const { boxes, grid, r } = scene(99, 300, SPREADS.wide);
    for (let q = 0; q < 50; q++) {
      const x = r.range(-20000, 30000),
        y = r.range(-9000, 9000),
        mask = r.int(1, 64);
      expect(grid.nearest(x, y, Infinity, mask)).toBe(bruteNearest(boxes, x, y, Infinity, mask));
    }
  });

  it("reports the distance of the nearest hit and Infinity for none", () => {
    const g = new SpatialHash();
    g.begin();
    g.insert(5, 100, 100, 110, 120, 1);
    g.end();
    expect(g.nearest(90, 110, 50, TEAM_ALL)).toBe(5);
    expect(g.lastDist).toBe(10);
    expect(g.nearest(105, 110, 50, TEAM_ALL)).toBe(5);
    expect(g.lastDist).toBe(0);
    expect(g.nearest(0, 0, 50, TEAM_ALL)).toBe(0);
    expect(g.lastDist).toBe(Infinity);
    expect(g.nearest(90, 110, 50, teamBit(2))).toBe(0);
    // Radius is inclusive, as in inRadius
    expect(g.nearest(90, 110, 10, TEAM_ALL)).toBe(5);
  });

  it("orders segment hits nearest first and reports entry points", () => {
    const g = new SpatialHash();
    g.begin();
    g.insert(3, 300, -10, 310, 10, 1);
    g.insert(1, 100, -10, 110, 10, 1);
    g.insert(2, 200, -10, 210, 10, 2);
    g.end();
    const out: number[] = [];
    expect(g.segment(0, 0, 400, 0, 0, TEAM_ALL, out)).toBe(3);
    expect(out.slice(0, 3)).toEqual([1, 2, 3]);
    expect(Array.from(g.hitT.subarray(0, 3))).toEqual([0.25, 0.5, 0.75]);
    expect(g.segment(400, 0, 0, 0, 0, teamBit(1), out)).toBe(2);
    expect(out.slice(0, 2)).toEqual([3, 1]);
    expect(g.segmentFirst(0, 0, 400, 0, 0, enemiesOf(1))).toBe(2);
    // A thick shot passing just above still clips the boxes
    expect(g.segment(0, -15, 400, -15, 12, TEAM_ALL, out)).toBe(3);
    expect(g.segment(0, -15, 400, -15, 8, TEAM_ALL, out)).toBe(0);
  });

  it("segBox handles parallel, inside and missing segments", () => {
    expect(segBox(0, 0, 10, 0, 2, -1, 4, 1)).toBeCloseTo(0.2);
    expect(segBox(3, 0, 10, 0, 2, -1, 4, 1)).toBe(0);
    expect(segBox(0, 5, 10, 0, 2, -1, 4, 1)).toBe(-1);
    expect(segBox(0, 0, 1, 0, 2, -1, 4, 1)).toBe(-1);
    expect(segBox(3, -5, 0, 10, 2, -1, 4, 1)).toBeCloseTo(0.4);
    expect(segBox(3, 0, 0, 0, 2, -1, 4, 1)).toBe(0);
  });

  it("team masks", () => {
    expect(teamBit(0)).toBe(1);
    expect(teamBit(3)).toBe(8);
    expect(enemiesOf(1) & teamBit(1)).toBe(0);
    expect(enemiesOf(1) & teamBit(0)).toBe(0);
    expect(enemiesOf(1, true) & teamBit(0)).toBe(1);
    expect(enemiesOf(1) & teamBit(2)).not.toBe(0);
  });
});

describe("spatial hash from the world", () => {
  it("indexes pos + body (+ team), skips unindexed and doomed bodies, and matches brute force", () => {
    const w = new World({ seed: 9, coreSystems: false });
    const { pos, body, team } = w.core;
    const r = new Rng(4);
    const boxes: Box[] = [];
    const ids: number[] = [];
    for (let k = 0; k < 400; k++) {
      const e = w.spawn(body);
      place(w, e, r.range(0, 4000), r.range(0, 1800));
      const i = indexOf(e);
      body.hw[i] = r.range(2, 12);
      body.h[i] = r.range(4, 30);
      const t = k % 4;
      if (t) {
        w.add(e, team);
        team.id[i] = t;
      }
      if (k % 50 === 0) body.flags[i] |= BODY_UNINDEXED;
      ids.push(e);
      if (k % 50 !== 0)
        boxes.push({
          id: e,
          x0: pos.x[i] - body.hw[i],
          y0: pos.y[i] - body.h[i],
          x1: pos.x[i] + body.hw[i],
          y1: pos.y[i],
          team: t,
        });
    }
    // Position without a body: not indexed
    place(w, w.create(), 100, 100);
    w.spatial.rebuild(w);
    expect(w.spatial.size).toBe(boxes.length);
    const out: number[] = [];
    for (let q = 0; q < 200; q++) {
      const x = r.range(0, 4000),
        y = r.range(0, 1800),
        rad = r.range(10, 500),
        mask = r.int(1, 32);
      expect(w.spatial.nearest(x, y, rad, mask)).toBe(bruteNearest(boxes, x, y, rad, mask));
      const n = w.spatial.inRadius(x, y, rad, mask, out);
      expect(out.slice(0, n).sort((a, b) => a - b)).toEqual(bruteRadius(boxes, x, y, rad, mask));
    }
    // Doomed after the build: queries stop returning it straight away
    const victim = boxes[10];
    const cx = (victim.x0 + victim.x1) / 2,
      cy = victim.y1;
    w.destroy(victim.id);
    const n = w.spatial.inRadius(cx, cy, 1, TEAM_ALL, out);
    expect(out.slice(0, n)).not.toContain(victim.id);
    expect(w.spatial.nearest(cx, cy, 0.5, TEAM_ALL, null, 0)).not.toBe(victim.id);
    w.flush();
    w.spatial.rebuild(w);
    expect(w.spatial.size).toBe(boxes.length - 1);
  });

  it("is rebuilt every tick by the core system after physics", () => {
    const w = new World({ seed: 1 });
    const { body } = w.core;
    const e = w.spawn(body);
    place(w, e, 50, 50);
    body.hw[indexOf(e)] = 4;
    body.h[indexOf(e)] = 8;
    let seenInAi = -1;
    w.addSystem({
      name: "probe",
      phase: "ai",
      run: (world) => (seenInAi = world.spatial.nearest(50, 46, 10, TEAM_ALL)),
    });
    // Tick 1's ai runs before the first rebuild; from tick 2 the body is there
    tick(w);
    expect(seenInAi).toBe(0);
    tick(w);
    expect(seenInAi).toBe(e);
  });
});
