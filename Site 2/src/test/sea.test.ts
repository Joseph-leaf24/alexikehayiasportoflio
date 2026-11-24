import { describe, expect, it } from "vitest";
import { BASE_ROW, kindAt, ROWS, TILE, WATER, waterAt, type Level } from "@/components/pixel/level";
import { THEMES, type Theme } from "@/components/pixel/themes";
import { Game, ROCKET, SHELL, type Enemy, type EnemyKind, type Jet, type PlaneType, type Shot, type View } from "@/components/pixel/game";

/** Step a game for `seconds`, the camera following the squad. `each` runs every step. */
function play(g: Game, seconds: number, each?: () => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    view.y = BASE_ROW * TILE - view.h * 0.68;
    g.step(1 / 30, view);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.();
  }
}

/** A quiet stretch (nothing coming in on its own) that runs out to open sea. */
const quiet: Theme = { ...THEMES.jungle, features: { flat: 1 }, enemies: 0, barrels: 0, artillery: 0, helis: 0, paras: 0, jets: 0, drones: 0, decor: "none" };
const sea: Theme = { ...quiet, features: { ocean: 1 } };

const place = (g: Game, kind: EnemyKind, x: number, y: number): Enemy => {
  const e = (g as unknown as { makeEnemy(k: EnemyKind, x: number, y: number): Enemy }).makeEnemy(kind, x, y);
  g.enemies.push(e);
  return e;
};
const tough = (e: Enemy) => Object.assign(e, { hp: 1e6, maxHp: 1e6 });
const jet = (kind: PlaneType, ally: boolean, x: number, y: number, vx: number, bombs = 3): Jet => ({ x, y, vx, bombs, dropT: 0, kind, ally, fireCd: 0, hp: 4, down: false, vy: 0, dead: false, alt: y, phase: 0 });

/** The first column of open water, and its surface row. */
function seaStart(L: Level) {
  for (let c = L.start; c < L.next; c++) for (let r = 0; r < ROWS; r++) if (kindAt(L, c, r) === WATER) return { first: c, surface: r };
  return { first: -1, surface: -1 };
}

/** Every shot fired from now on (each once), via `each`. */
function watchShots(g: Game) {
  const seen = new Set<Shot>();
  const fired: Shot[] = [];
  return {
    fired,
    each: () => {
      for (const s of g.shots)
        if (!seen.has(s)) {
          seen.add(s);
          fired.push({ ...s });
        }
    },
  };
}

describe("submarines", () => {
  it("roam the sea at all sorts of depths, never leaving the water", () => {
    const g = new Game(sea, { seed: 3, level: { boss: null, length: 4000 } });
    play(g, 1 / 30);
    const { first, surface } = seaStart(g.level);
    const sub = tough(place(g, "submarine", (first + 30) * TILE, surface * TILE));
    const x0 = sub.x;
    let shallow = Infinity;
    let deep = -Infinity;
    let wandered = 0;
    let dry = 0;
    play(g, 45, () => {
      const d = sub.y - surface * TILE;
      shallow = Math.min(shallow, d);
      deep = Math.max(deep, d);
      wandered = Math.max(wandered, Math.abs(sub.x - x0));
      if (!waterAt(g.level, sub.x, sub.y)) dry++;
    });
    expect(dry).toBe(0);
    expect(deep - shallow).toBeGreaterThan(30);
    expect(deep).toBeGreaterThan(40);
    expect(wandered).toBeGreaterThan(60);
    // Down there it can't be shot at
    expect(sub.spin === 0 || g.aimPoint(sub) === null).toBe(true);
  });

  it("ours come into the sea and send missiles up out of it that shoot down the enemy's bombers", () => {
    const g = new Game(sea, { seed: 3, level: { boss: null, length: 4000 } });
    play(g, 3);
    expect(g.navy.length).toBeGreaterThan(0);
    for (const s of g.navy) expect(waterAt(g.level, s.x, s.y)).toBe(true);
    const bomber = jet("bomber", false, g.advance + 180, (BASE_ROW - 18) * TILE, -30, 0);
    bomber.hp = 6;
    g.jets.push(bomber);
    let rose = false;
    let splash = false;
    play(g, 10, () => {
      if (g.sams.some((s) => s.side === "ally" && !s.wet && s.out > 0)) rose = true;
      if (g.popups.some((p) => p.text === "SPLASH ONE")) splash = true;
    });
    expect(rose).toBe(true);
    expect(splash).toBe(true);
    expect(bomber.down || bomber.dead).toBe(true);
  });

  it("the enemy's send missiles up at our aircraft", () => {
    const g = new Game(sea, { seed: 3, level: { boss: null, length: 4000 } });
    play(g, 1 / 30);
    const { first, surface } = seaStart(g.level);
    tough(place(g, "submarine", (first + 20) * TILE, surface * TILE + 40));
    const ours = jet("attacker", true, g.advance + 120, (BASE_ROW - 18) * TILE, 30, 0);
    ours.hp = 6;
    g.jets.push(ours);
    let launched = false;
    play(g, 12, () => {
      if (g.sams.some((s) => s.side === "enemy")) launched = true;
    });
    expect(launched).toBe(true);
    expect(ours.down || ours.dead).toBe(true);
  });
});

describe("bombing runs", () => {
  it("heavy bombers carpet-bomb from high up", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 900 } });
    const w = watchShots(g);
    g.jets.push(jet("heavy", false, g.advance + 160, (BASE_ROW - 28) * TILE, -120, 20));
    play(g, 3, w.each);
    expect(w.fired.filter((s) => s.kind === SHELL && s.pid === -3).length).toBeGreaterThanOrEqual(15);
  });

  it("stealth bombers let go of each bomb right over a target", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 900 } });
    const w = watchShots(g);
    const j = jet("stealth", false, g.advance + 160, (BASE_ROW - 26) * TILE, -250, 3);
    g.jets.push(j);
    // How far each bomb was from the nearest of the squad as it left the plane
    const misses: number[] = [];
    play(g, 2.5, () => {
      const before = w.fired.length;
      w.each();
      for (const s of w.fired.slice(before)) if (s.kind === SHELL) misses.push(Math.min(...g.players.filter((p) => !p.dead).map((p) => Math.abs(p.x - s.x))));
    });
    expect(misses.length).toBeGreaterThan(0);
    expect(misses.length).toBeLessThanOrEqual(3);
    for (const m of misses) expect(m).toBeLessThan(12);
  });

  it("dive bombers swoop down on the target, bomb it, and climb back out", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 900 } });
    const w = watchShots(g);
    const alt = (BASE_ROW - 22) * TILE;
    const j = jet("dive", false, g.advance + 200, alt, -210, 1);
    g.jets.push(j);
    let lowest = alt;
    play(g, 2.5, () => {
      w.each();
      lowest = Math.max(lowest, j.y);
    });
    expect(lowest - alt).toBeGreaterThan(40);
    expect(w.fired.filter((s) => s.kind === SHELL && s.pid === -3).length).toBe(1);
    expect(j.phase).toBeGreaterThanOrEqual(2);
    expect(j.y).toBeLessThan(lowest);
  });

  it("rocket jets fire salvos of rockets at the ground", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 900 } });
    const w = watchShots(g);
    g.jets.push(jet("rocketjet", false, g.advance + 160, (BASE_ROW - 20) * TILE, -150, 3));
    play(g, 2.5, w.each);
    const rockets = w.fired.filter((s) => s.kind === ROCKET && s.pid === -1);
    expect(rockets.length).toBeGreaterThanOrEqual(8);
    for (const r of rockets) expect(r.vy).toBeGreaterThan(0);
  });

  it("the new aircraft come over on the regular bombing runs, on both sides", () => {
    // (over a couple of battles: which planes come over on a run is down to chance)
    const kinds = new Set<string>();
    for (const seed of [4, 5]) {
      const g = new Game(THEMES.jungle, { seed, level: { boss: null, length: 4000 } });
      play(g, 90, () => {
        for (const j of g.jets) kinds.add(`${j.kind}:${j.ally ? "ours" : "theirs"}`);
      });
    }
    const fresh = ["heavy", "stealth", "dive", "rocketjet"].filter((k) => kinds.has(`${k}:ours`) || kinds.has(`${k}:theirs`));
    expect(fresh.length).toBeGreaterThanOrEqual(3);
  }, 30_000);
});
