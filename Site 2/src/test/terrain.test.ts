import { describe, expect, it } from "vitest";
import { BASE_ROW, cell, createLevel, CRYSTAL, extend, kindAt, LADDER, LAVA, RING, ROWS, SNOW, solidAt, TILE, type Level } from "@/components/pixel/level";
import { ROTATION, THEMES, type Theme } from "@/components/pixel/themes";
import { Game, type View } from "@/components/pixel/game";

/** Step a game for `seconds`, the camera following the squad's feet up and down. `each` runs every step. */
function play(g: Game, seconds: number, each?: (view: View) => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    const up = g.players.filter((p) => !p.dead && !p.chute);
    view.y = (up.length ? up.reduce((s, p) => s + p.y, 0) / up.length : BASE_ROW * TILE) - view.h * 0.68;
    g.step(1 / 30, view);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.(view);
  }
}

/** A quiet country (nothing coming in on its own) made of one kind of set piece and some flat ground. */
const only = (feature: "massif" | "abyss"): Theme => ({
  ...THEMES.summit,
  features: { [feature]: 3, flat: 1 },
  enemies: 0,
  barrels: 0,
  artillery: 0,
  helis: 0,
  paras: 0,
  jets: 0,
  drones: 0,
  relief: 0,
});

const grounds = (L: Level) => Array.from({ length: L.next - L.start }, (_, k) => ({ c: L.start + k, g: L.ground[(L.start + k) % RING], crust: L.crust[(L.start + k) % RING] }));

describe("great mountains", () => {
  it("rise ninety blocks and more over the plain, climbable step by step, snow on top and an outpost on the summit", () => {
    const L = createLevel(only("massif"), 3, { boss: null, length: 4000 });
    extend(L, RING - 20);
    const cols = grounds(L);
    const top = Math.min(...cols.map((k) => k.g));
    expect(BASE_ROW - top).toBeGreaterThanOrEqual(90);
    expect(top).toBeGreaterThan(8);
    // Every step up is a cliff the squad can climb (and the tallest have a ladder up them)
    let ladders = 0;
    for (let k = 1; k < cols.length; k++) {
      expect(cols[k - 1].g - cols[k].g, `column ${cols[k].c}`).toBeLessThanOrEqual(14);
      if (cols[k - 1].g - cols[k].g >= 8 && kindAt(L, cols[k].c - 1, cols[k].g + 2) === LADDER) ladders++;
    }
    expect(ladders).toBeGreaterThan(0);
    const summit = cols.find((k) => k.g === top)!;
    expect(kindAt(L, summit.c, top)).toBe(SNOW);
    expect(L.props.some((p) => p.kind === "radar" && p.y === top * TILE)).toBe(true);
    // ... and it comes back down to the plain
    expect(cols.some((k) => k.c > summit.c && k.g === BASE_ROW)).toBe(true);
  });
});

describe("the depths", () => {
  it("drop down a sinkhole into a cavern far under the ground, under a roof of rock, and come back up a chimney", () => {
    const L = createLevel(only("abyss"), 3, { boss: null, length: 4000 });
    extend(L, RING - 20);
    const cols = grounds(L);
    const cavern = cols.filter((k) => k.g - k.crust >= 50);
    expect(cavern.length).toBeGreaterThan(60);
    expect(Math.max(...cavern.map((k) => k.g - k.crust))).toBeGreaterThanOrEqual(58);
    let roofed = 0;
    for (const k of cavern) {
      expect(k.g).toBeLessThanOrEqual(ROWS - 4);
      // Headroom over the floor
      for (let r = k.g - 5; r < k.g - 1; r++) if (solidAt(L, k.c * TILE + 4, r * TILE + 4) && kindAt(L, k.c, r) !== CRYSTAL) expect(r, `column ${k.c}`).toBeLessThan(k.g - 3);
      // Under the land: its surface still there over the cavern (except at the sinkhole and the chimney)
      if (solidAt(L, k.c * TILE + 4, k.crust * TILE + 4)) roofed++;
    }
    expect(roofed).toBeGreaterThan(cavern.length * 0.7);
    // Things to see down there: lava, crystals, lamps
    let lava = 0;
    let crystal = 0;
    for (const k of cavern) for (let r = k.crust; r < ROWS; r++) {
      const b = L.fg[cell(k.c, r)];
      if (b === LAVA) lava++;
      if (b === CRYSTAL) crystal++;
    }
    expect(lava).toBeGreaterThan(0);
    expect(crystal).toBeGreaterThan(0);
    expect(L.props.filter((p) => p.kind === "lamp").length).toBeGreaterThan(3);
    // ... and out again, back to the surface
    const last = cavern[cavern.length - 1].c;
    expect(cols.some((k) => k.c > last && k.g === k.crust)).toBe(true);
  });

  it("are part of the rotation: two new countries built round them, and the others' share of them", () => {
    expect(ROTATION).toContain("summit");
    expect(ROTATION).toContain("depths");
    expect(THEMES.summit.features.massif).toBeGreaterThan(0);
    expect(THEMES.depths.features.abyss).toBeGreaterThan(0);
    expect(Object.values(THEMES).filter((t) => (t.features.massif ?? 0) > 0).length).toBeGreaterThanOrEqual(6);
    expect(Object.values(THEMES).filter((t) => (t.features.abyss ?? 0) > 0).length).toBeGreaterThanOrEqual(6);
  });
});

describe("the squad", () => {
  it("climbs a great mountain all the way to the top, and down the far side", () => {
    const g = new Game(only("massif"), { seed: 3, gunship: false, level: { boss: null, length: 4000 } });
    let high = Infinity;
    let summitAt = -1;
    let downAgain = false;
    play(g, 90, () => {
      const up = g.players.filter((p) => !p.dead && !p.chute);
      for (const p of up)
        if (p.y < high) {
          high = p.y;
          summitAt = p.x;
        }
      // (and on: past the summit, most of the squad back down on the plain)
      if (BASE_ROW * TILE - high > 80 * TILE && up.filter((p) => p.x > summitAt + 100 && Math.abs(p.y - BASE_ROW * TILE) < 6 * TILE).length > 4) downAgain = true;
    });
    expect(BASE_ROW * TILE - high).toBeGreaterThan(80 * TILE);
    expect(downAgain).toBe(true);
  });

  it("goes down into the depths, fights through the cavern and climbs back out, never stuck in the rock", () => {
    const g = new Game(only("abyss"), { seed: 3, gunship: false, level: { boss: null, length: 4000 } });
    let deepest = -Infinity;
    let deepAt = -1;
    let buried = 0;
    let inRock = 0;
    let outAgain = false;
    play(g, 140, (view) => {
      if ((g as unknown as { buried(v: View): boolean }).buried(view)) buried++;
      const up = g.players.filter((p) => !p.dead && !p.chute);
      for (const p of up) {
        if (p.y > deepest) {
          deepest = p.y;
          deepAt = p.x;
        }
        if (g.solid(p.x, p.y - 8)) inRock++;
      }
      // (and out again: most of the squad back up on the surface, past the cavern)
      if (deepest - BASE_ROW * TILE > 50 * TILE && up.filter((p) => p.x > deepAt + 60 && p.y <= BASE_ROW * TILE + 2 * TILE).length > 4) outAgain = true;
    });
    expect(deepest - BASE_ROW * TILE).toBeGreaterThan(50 * TILE);
    expect(buried).toBeGreaterThan(30 * 10);
    expect(inRock).toBeLessThan(30 * 8);
    expect(outAgain).toBe(true);
  });

  it("drops whoever comes back in down beside them in the cavern, and they come down to the floor", () => {
    const g = new Game(only("abyss"), { seed: 3, gunship: false, level: { boss: null, length: 4000 } });
    const deep = () => g.players.find((p) => !p.dead && !p.chute && p.ground && p.y - g.level.crust[Math.floor(p.x / TILE) % RING] * TILE > 50 * TILE);
    let t = 0;
    while (!deep() && t++ < 90 * 30) play(g, 1 / 30);
    const mate = deep()!;
    expect(mate).toBeDefined();
    const p = g.players.find((q) => q !== mate)!;
    (g as unknown as { release(p: unknown, x: number, y: number): void }).release(p, mate.x + 20, mate.y - 400);
    // Just over them, in the cavern, not up in the rock of its roof
    expect(p.y).toBeGreaterThan(mate.y - 48);
    expect(p.y).toBeLessThan(mate.y + 8);
    expect(g.solid(p.x, p.y - 8)).toBe(false);
    // ... and the parachute comes down, rather than hanging off a wall
    let landed = false;
    play(g, 5, () => {
      if (!p.chute) landed = true;
    });
    expect(landed).toBe(true);
  });
});
