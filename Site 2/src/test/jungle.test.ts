import { describe, expect, it } from "vitest";
import { BAMBOO, BASE_ROW, cell, createLevel, extend, kindAt, LAVA, LEAF, MOSS, RING, ROWS, solidAt, TILE, VINE, WATER } from "@/components/pixel/level";
import { lightOf, ROTATION, THEMES, WEATHER, type Theme } from "@/components/pixel/themes";
import { BULLET, FLAMESHOT, Game, GRENADE, PLAYER_HP, type Enemy, type EnemyKind, type View } from "@/components/pixel/game";

/** Step a game for `seconds`, with the camera following the squad like PixelScene's does. `each` runs every step. */
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

/** A flat, empty stretch with nothing coming in, so one enemy can be watched on its own. */
const quiet: Theme = { ...THEMES.jungle, features: { flat: 1 }, enemies: 0, barrels: 0, artillery: 0, helis: 0, paras: 0, jets: 0, drones: 0, decor: "none" };

/** Put an enemy down (the generator normally does this). */
const place = (g: Game, kind: EnemyKind, x: number, y = BASE_ROW * TILE): Enemy => {
  const e = (g as unknown as { makeEnemy(k: EnemyKind, x: number, y: number): Enemy }).makeEnemy(kind, x, y);
  g.enemies.push(e);
  return e;
};
const tough = (e: Enemy) => Object.assign(e, { hp: 1e6, maxHp: 1e6 });

/** How many blocks of each kind a level has stored. */
function census(theme: Theme, seed: number) {
  const L = createLevel(theme, seed, { boss: null, length: 2000 });
  extend(L, RING - 20);
  const n = new Map<number, number>();
  for (let c = L.start; c < L.next; c++) for (let r = 0; r < ROWS; r++) n.set(L.fg[cell(c, r)], (n.get(L.fg[cell(c, r)]) ?? 0) + 1);
  return { L, count: (k: number) => n.get(k) ?? 0 };
}

describe("jungle levels", () => {
  it("are seven of the seventeen levels in the rotation, each with its own weather and light", () => {
    expect(ROTATION.length).toBeGreaterThanOrEqual(17);
    for (const k of ["deepjungle", "temple", "monsoon", "nightraid"]) expect(ROTATION).toContain(k);
    expect(ROTATION.filter((k) => ["jungle", "deepjungle", "temple", "monsoon"].includes(THEMES[k].backdrop) || k === "swamp").length).toBeGreaterThanOrEqual(7);
    for (const t of Object.values(THEMES)) {
      expect(WEATHER[t.ambient]).toBeTruthy();
      expect(lightOf(t)).toBeTruthy();
    }
    expect(lightOf(THEMES.nightraid)).toBe("Night");
    expect(WEATHER[THEMES.monsoon.ambient]).toBe("Monsoon storm");
  });

  it("grow rainforest, bamboo, rivers and waterfalls; temples and ruins of mossy stone", () => {
    const deep = census(THEMES.deepjungle, 4);
    for (const k of [LEAF, VINE, BAMBOO, WATER]) expect(deep.count(k), `block ${k}`).toBeGreaterThan(0);
    expect(deep.L.props.some((p) => p.kind === "waterfall")).toBe(true);
    expect(deep.L.props.some((p) => p.kind === "tent")).toBe(true);
    const temple = census(THEMES.temple, 4);
    expect(temple.count(MOSS)).toBeGreaterThan(100);
    expect(temple.L.props.filter((p) => p.kind === "brazier").length).toBeGreaterThanOrEqual(2);
    expect(census(THEMES.monsoon, 4).count(WATER)).toBeGreaterThan(10);
  });

  it("keeps rivers and pools waist deep, and every pit jumpable", () => {
    for (const key of ["deepjungle", "temple", "monsoon", "nightraid"]) {
      const { L } = census(THEMES[key], 9);
      let run = 0;
      let lip = L.ground[L.start % RING];
      for (let c = L.start; c < L.next; c++) {
        for (let r = 0; r < ROWS - 1; r++) if (kindAt(L, c, r) === WATER) expect(solidAt(L, c * TILE + 4, (r + 1) * TILE + 4), `${key} ${c},${r}`).toBe(true);
        if (c === L.start) continue;
        const g = L.ground[c % RING];
        const pit = (g >= lip + 3 && !solidAt(L, c * TILE + 4, lip * TILE + 4)) || kindAt(L, c, g - 1) === LAVA;
        run = pit ? run + 1 : 0;
        if (!pit) lip = g;
        expect(run, `${key} column ${c}`).toBeLessThanOrEqual(3);
      }
    }
  });
});

describe("new enemies", () => {
  it("attack dogs run the squad down and bite", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 600 } });
    const front = g.players.reduce((a, p) => (p.x > a.x ? p : a));
    const dog = place(g, "dog", front.x + 120);
    play(g, 0.3);
    expect(dog.vx).toBeLessThan(0);
    expect(dog.x).toBeLessThan(front.x + 120);
    const p = g.players[3];
    const biter = place(g, "dog", p.x, p.y);
    tough(biter);
    play(g, 1 / 30);
    expect(p.hp).toBe(PLAYER_HP - 1);
  });

  it("riot shields turn bullets from the front, but not from behind, and not a blast", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 600 } });
    const front = g.players.reduce((a, p) => (p.x > a.x ? p : a));
    const e = place(g, "shield", front.x + 60);
    play(g, 1 / 30);
    expect(e.facing).toBe(-1);
    const bullet = (x: number, vx: number) => g.shots.push({ kind: BULLET, x, y: e.y - 8, vx, vy: 0, life: 1, dmg: 1, pid: 0, pierce: 0 });
    bullet(e.x - 10, 400);
    play(g, 2 / 30);
    expect(e.hp).toBe(4);
    bullet(e.x + 10, -400);
    play(g, 2 / 30);
    expect(e.hp).toBeLessThan(4);
    g.explode(e.x, e.y - 4, 18, 0, false);
    expect(e.alive).toBe(false);
  });

  it("flamethrowers hose the squad up close; grenadiers lob grenades over cover", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 600 } });
    const front = g.players.reduce((a, p) => (p.x > a.x ? p : a));
    tough(place(g, "flamer", front.x + 40));
    tough(place(g, "grenadier", front.x + 150));
    const seen = { flame: false, grenade: false };
    play(g, 2.5, () => {
      if (g.shots.some((s) => s.kind === FLAMESHOT && s.pid === -1)) seen.flame = true;
      if (g.shots.some((s) => s.kind === GRENADE && s.pid === -1)) seen.grenade = true;
    });
    expect(seen).toEqual({ flame: true, grenade: true });
  });

  it("troop trucks drive up, park near the squad and unload five troops", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 600 } });
    const truck = tough(place(g, "truck", g.advance + 300));
    truck.altCd = 0;
    const troops = new Set<Enemy>();
    play(g, 4, () => {
      for (const e of g.enemies) if (e !== truck && e.kind !== "truck") troops.add(e);
    });
    expect(truck.x).toBeLessThan(g.advance + 300);
    expect(truck.burst).toBe(5);
    expect(troops.size).toBe(5);
  });

  it("drone swarms sweep in and hover over the squad", () => {
    const g = new Game({ ...quiet, drones: 120 }, { seed: 1, level: { boss: null, length: 900 } });
    let flying = 0;
    play(g, 14, () => {
      flying = Math.max(flying, g.enemies.filter((e) => e.kind === "drone" && e.alive && e.y < BASE_ROW * TILE - 40).length);
    });
    expect(flying).toBeGreaterThanOrEqual(2);
  });

  it("patrol boats keep to their stretch of river, at the water line", () => {
    const g = new Game(quiet, { seed: 1, level: { boss: null, length: 600 } });
    const x = g.advance + 140;
    const y = (BASE_ROW + 1) * TILE;
    const boat = tough(place(g, "boat", x, y));
    let far = 0;
    play(g, 6, () => (far = Math.max(far, Math.abs(boat.x - x))));
    expect(far).toBeGreaterThan(5);
    expect(far).toBeLessThanOrEqual(11);
    expect(boat.y).toBe(y);
  });

  it("monsoon storms bring lightning, a few seconds apart", () => {
    const g = new Game(THEMES.monsoon, { seed: 2, level: { boss: null, length: 900 } });
    let strikes = 0;
    let last = -1;
    play(g, 30, () => {
      if (g.bolts.length && g.bolts[g.bolts.length - 1].t === 0 && last !== g.t) {
        strikes++;
        last = g.t;
      }
    });
    expect(strikes).toBeGreaterThanOrEqual(2);
    expect(strikes).toBeLessThanOrEqual(6);
  });
});

describe("the end of a level", () => {
  it("fades the whole screen to white, shows Level complete with the numbers, then fades into the next level", () => {
    const g = new Game(THEMES.fortress, { seed: 3, level: { boss: null, length: 50 } });
    expect(g.progress).toBe(0);
    let peak = 0;
    let card: Game["complete"] = null;
    let objective = "";
    let atStart: { white: number; t: number } | null = null;
    let progressed = 0;
    play(g, 70, () => {
      if (g.levelNo === 1) {
        peak = Math.max(peak, g.white);
        progressed = Math.max(progressed, g.progress);
        if (g.complete && !card) {
          card = { ...g.complete };
          objective = g.objective;
        }
      } else if (!atStart) atStart = { white: g.white, t: g.banner?.t ?? 0 };
    });
    expect(progressed).toBe(1);
    expect(peak).toBe(1);
    expect(card).toMatchObject({ no: 1, name: "Fortress" });
    expect(card!.kills).toBeGreaterThan(0);
    expect(card!.time).toBeGreaterThan(5);
    expect(objective).toBe("Level complete");
    expect(g.cleared).toBeGreaterThanOrEqual(1);
    expect(g.log[g.log.length - 1]).toMatchObject({ no: 1, name: "Fortress" });
    // The next level starts white, its banner waiting until the white has faded
    expect(atStart).toEqual({ white: 1, t: expect.any(Number) });
    expect(atStart!.t).toBeLessThan(0);
  });

  it("clears the white-out and shows the new level's banner once it has faded", () => {
    const g = new Game(THEMES.jungle, { seed: 1 });
    g.nextLevel(THEMES.temple, { boss: "mech" });
    expect(g.white).toBe(1);
    expect(g.complete).toBeNull();
    expect(g.banner?.t).toBeLessThan(0);
    play(g, 1.5);
    expect(g.white).toBe(0);
    expect(g.banner).toMatchObject({ text: "Level 2", sub: "Lost temple", info: "Boss ahead: Walker" });
    expect(g.banner!.t).toBeGreaterThanOrEqual(0);
    expect(g.bossState).toBe("ahead");
  });
});
