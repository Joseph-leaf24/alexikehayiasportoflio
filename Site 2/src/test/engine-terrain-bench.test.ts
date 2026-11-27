/**
 * Terrain benchmark: 50 blasts a second on a full 512 × 240 world (ground, dunes, lakes, lava, trees, wooden houses
 * and a city with a skyscraper), plus a skyscraper collapse and a saturated avalanche for the worst-case tick.
 * Always runs a short version that checks the budgets; `BENCH=1 npx vitest run src/test/engine-terrain-bench` runs
 * the full 60 simulated seconds and prints ms per tick.
 */
import { describe, expect, it } from "vitest";
import { Terrain } from "@/engine/terrain/grid";
import { RING, ROWS, TILE } from "@/engine/terrain/constants";
import {
  BEDROCK,
  BRICK,
  CONCRETE,
  DIRT,
  GLASS,
  GRASS,
  LAVA,
  LEAF,
  SAND,
  STONE,
  TRUNK,
  WATER,
  WOOD,
} from "@/engine/terrain/materials";
import { TerrainRng } from "@/engine/terrain/random";
import { building, fill, GROUND, settle } from "./engine-terrain-fixtures";

const FULL = typeof process !== "undefined" && !!process.env.BENCH;

/** The benchmark world, every one of its 512 columns stored. */
function battlefield(seed = 1): Terrain {
  const t = new Terrain({
    seed,
    overshoot: 0,
    generate: (w, from, to) => {
      for (let c = from; c < to; c++) {
        const g = GROUND + Math.round(Math.sin(c / 23) * 4 + Math.sin(c / 7) * 1.5);
        for (let r = g; r < ROWS; r++)
          w.set(c, r, r === ROWS - 1 ? BEDROCK : r === g ? GRASS : r < g + 5 ? DIRT : STONE, { bg: DIRT });
      }
    },
  });
  t.ensure(RING);
  // Dunes, lakes and a lava pit dug into the ground
  fill(t, 20, GROUND - 6, 50, 6, SAND);
  fill(t, 80, GROUND - 1, 40, 12, 0);
  fill(t, 80, GROUND + 1, 40, 10, WATER);
  fill(t, 300, GROUND, 25, 8, 0);
  fill(t, 300, GROUND + 2, 25, 6, LAVA);
  fill(t, 440, GROUND - 2, 30, 10, 0);
  fill(t, 440, GROUND, 30, 8, WATER);
  // Trees and wooden houses
  for (let c = 130; c < 180; c += 9) {
    fill(t, c, GROUND - 12, 1, 12, TRUNK);
    fill(t, c - 2, GROUND - 16, 5, 4, LEAF);
  }
  for (let c = 185; c < 230; c += 15) building(t, c, 10, 2, 5, WOOD, GROUND - 4);
  // The city: a row of buildings and one skyscraper
  for (let c = 335; c < 430; c += 18) building(t, c, 14, 3 + ((c / 18) % 5), 5, c % 2 ? BRICK : CONCRETE, GROUND - 4);
  building(t, 240, 30, 20, 6, CONCRETE, GROUND - 4, 2);
  fill(t, 242, GROUND - 100, 26, 1, GLASS, { structural: true });
  settle(t, 400);
  return t;
}

interface Timing {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

function summarise(ms: number[]): Timing {
  const s = [...ms].sort((a, b) => a - b);
  const at = (q: number) => s[Math.min(s.length - 1, Math.floor(q * s.length))];
  return {
    mean: ms.reduce((a, b) => a + b, 0) / ms.length,
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: s[s.length - 1],
  };
}

const fmt = (t: Timing) =>
  `mean ${t.mean.toFixed(3)} · p50 ${t.p50.toFixed(3)} · p95 ${t.p95.toFixed(3)} · p99 ${t.p99.toFixed(3)} · max ${t.max.toFixed(3)} ms`;

describe("terrain benchmark", () => {
  it("50 blasts per second on a 512 × 240 world", () => {
    const t = battlefield();
    const rng = new TerrainRng(42);
    const seconds = FULL ? 60 : 8;
    const ticks = seconds * 30;
    const warm = 30;
    const tickMs: number[] = [];
    const carveMs: number[] = [];
    const stepMs: number[] = [];
    let blasts = 0;
    let peakMoves = 0;
    let peakWork = 0;
    let peakChunks = 0;
    let awake = 0;
    let falling = 0;
    let events = 0;
    const ev: number[] = [];
    for (let k = 0; k < ticks; k++) {
      const due = Math.floor((50 * (k + 1)) / 30) - Math.floor((50 * k) / 30);
      const t0 = performance.now();
      for (let b = 0; b < due; b++) {
        const col = 4 + rng.int(RING - 8);
        const r = 12 + rng.next() * 36;
        const y = t.columnTop(col) * TILE + (rng.next() - 0.3) * 48;
        t.carve(col * TILE + 4, y, r, r / 7);
        if (rng.next() < 0.1) t.ignite(col * TILE, y, r);
        blasts++;
      }
      const t1 = performance.now();
      t.step();
      const t2 = performance.now();
      events += t.drainEvents(ev);
      if (k < warm) continue;
      carveMs.push(t1 - t0);
      stepMs.push(t2 - t1);
      tickMs.push(t2 - t0);
      peakMoves = Math.max(peakMoves, t.stats.moves);
      peakWork = Math.max(peakWork, t.stats.work);
      peakChunks = Math.max(peakChunks, t.stats.chunks);
      awake += t.stats.awake;
      falling = Math.max(falling, t.stats.falling);
    }
    const n = tickMs.length;
    console.log(
      [
        `terrain bench: ${blasts} blasts over ${seconds} s (${ticks} ticks), 512 × 240 cells`,
        `  tick (blasts + step): ${fmt(summarise(tickMs))}`,
        `  carve per tick:       ${fmt(summarise(carveMs))}`,
        `  step per tick:        ${fmt(summarise(stepMs))}`,
        `  peaks: ${peakMoves} moves, ${peakWork} structure work, ${peakChunks} chunks simulated, ${falling} chunks falling;` +
          ` mean ${(awake / n).toFixed(1)} chunks awake of 480; ${events} events`,
      ].join("\n")
    );
    expect(peakMoves).toBeLessThanOrEqual(t.physics.cfg.maxMoves);
    expect(peakWork).toBeLessThanOrEqual(t.structure.cfg.budget);
    expect(peakChunks).toBeLessThanOrEqual(t.physics.cfg.maxActiveChunks);
  }, 120_000);

  it("a skyscraper collapse never spikes a tick", () => {
    const t = new Terrain({
      overshoot: 0,
      generate: (w, from, to) => {
        for (let c = from; c < to; c++) for (let r = 232; r < ROWS; r++) w.set(c, r, r === ROWS - 1 ? BEDROCK : STONE);
      },
    });
    t.ensure(RING);
    // Solid 60 × 200: 12,000 structural cells, plus a hollow 40-storey tower beside it
    fill(t, 100, 32, 60, 200, CONCRETE, { structural: true });
    const top = building(t, 200, 40, 40, 5, CONCRETE, 232, 2);
    for (let c = 100; c < 160; c++) for (let r = 229; r < 232; r++) t.damage(c, r, 255);
    for (const c of [200, 201, 238, 239]) for (let r = 227; r < 232; r++) t.damage(c, r, 255);
    const ms: number[] = [];
    let peakWork = 0;
    let peakMoves = 0;
    let peakFalling = 0;
    let ticks = 0;
    for (; ticks < 6000 && t.busy; ticks++) {
      const t0 = performance.now();
      t.step();
      ms.push(performance.now() - t0);
      peakWork = Math.max(peakWork, t.stats.work);
      peakMoves = Math.max(peakMoves, t.stats.moves);
      peakFalling = Math.max(peakFalling, t.stats.falling);
    }
    console.log(
      `skyscraper collapse (12,000 solid + ${top < 232 ? "40-storey hollow" : ""} tower): settled in ${ticks} ticks;` +
        ` ${fmt(summarise(ms.slice(1)))}; peaks ${peakWork} work, ${peakMoves} moves, ${peakFalling} chunks falling`
    );
    expect(ticks).toBeLessThan(6000);
    expect(peakWork).toBeLessThanOrEqual(t.structure.cfg.budget);
    expect(peakMoves).toBeLessThanOrEqual(t.physics.cfg.maxMoves);
    expect(peakFalling).toBeGreaterThan(2);
  }, 120_000);

  it("a saturated avalanche: the most a tick can cost", () => {
    const t = new Terrain({
      overshoot: 0,
      generate: (w, from, to) => {
        for (let c = from; c < to; c++) w.set(c, ROWS - 1, BEDROCK);
      },
    });
    t.ensure(RING);
    // A world-wide slab of sand and water over air: every chunk wants to move at once
    fill(t, 0, 20, RING, 50, SAND);
    fill(t, 0, 70, RING, 30, WATER);
    const ms: number[] = [];
    for (let k = 0; k < (FULL ? 600 : 150); k++) {
      const t0 = performance.now();
      t.step();
      ms.push(performance.now() - t0);
      expect(t.stats.moves).toBeLessThanOrEqual(3000);
    }
    console.log(`saturated avalanche (25,600 sand + 15,360 water cells falling): ${fmt(summarise(ms.slice(10)))}`);
  }, 120_000);
});
