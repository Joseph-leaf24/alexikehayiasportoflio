import { describe, expect, it } from "vitest";
import {
  AIR,
  BASE_ROW,
  BEDROCK,
  blastCells,
  cell,
  createLevel,
  damageCell,
  extend,
  kindAt,
  LADDER,
  LAVA,
  LEVEL_H,
  WATER,
  lineOfSight,
  PEAK_ROW,
  RING,
  ROWS,
  solidAt,
  terrainAt,
  TILE,
  TILE_HP,
  VALLEY_ROW,
} from "@/components/pixel/level";
import { nextTheme, ROTATION, THEMES } from "@/components/pixel/themes";
import { Game, PLAYER_HP, SQUAD_SIZE, type View } from "@/components/pixel/game";
import { rng } from "@/components/pixel/rng";
import { SQUAD } from "@/data/squad";
import { BUILDINGS } from "@/components/pixel/buildings";
import { ASSETS } from "@/components/pixel/assets";
import { domains } from "@/data/domains";

/** Step a game for `seconds`, with the camera following the squad like PixelScene's does. `each` runs every step. */
function play(g: Game, seconds: number, frozen = false, each?: () => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    // (up the mountains and down the valleys with the squad's feet)
    const up = g.players.filter((p) => !p.dead && !p.chute);
    const feet = up.length ? up.reduce((s, p) => s + p.y, 0) / up.length : g.level.h * TILE;
    view.y = Math.max(-80, Math.min(LEVEL_H - view.h, feet - view.h * 0.68));
    g.step(1 / 30, view, frozen);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.();
  }
  return view;
}

describe("level", () => {
  it("is the same level for the same seed", () => {
    const a = createLevel(THEMES.fortress, 9);
    const b = createLevel(THEMES.fortress, 9);
    expect(extend(a, 200)).toEqual(extend(b, 200));
    expect(Array.from(a.fg)).toEqual(Array.from(b.fg));
  });

  it("always has ground, with unbreakable bedrock at the bottom", () => {
    for (const theme of Object.values(THEMES)) {
      const L = createLevel(theme, 3);
      extend(L, 300);
      for (let c = L.start; c < L.next; c++) {
        expect(kindAt(L, c, ROWS - 1)).toBe(BEDROCK);
        expect(kindAt(L, c, ROWS - 2)).toBe(BEDROCK);
        // (the great mountains go far over the ordinary hills, but always leave room over their summits)
        expect(L.ground[c % RING]).toBeGreaterThan(8);
        expect(L.ground[c % RING]).toBeLessThanOrEqual(ROWS - 3);
      }
    }
  });

  it("raises hills and mountain ranges from seeded noise: a different world every time", () => {
    // (the ordinary hills: without the great mountains and the descents, which have tests of their own)
    const hills = { ...THEMES.snow, features: Object.fromEntries(Object.entries(THEMES.snow.features).filter(([k]) => k !== "massif" && k !== "abyss")) };
    const heights = (seed: number) => {
      const L = createLevel(hills, seed, { boss: null, length: 4000 });
      // (short of the ring's length, so a long feature at the end can't wrap round over the start)
      extend(L, 250);
      return Array.from({ length: 250 }, (_, c) => L.ground[c % RING]);
    };
    const a = heights(1);
    const b = heights(2);
    // A flat landing zone, then real relief
    expect(a.slice(0, 40).every((h) => h === BASE_ROW)).toBe(true);
    expect(Math.max(...a.slice(40)) - Math.min(...a.slice(40))).toBeGreaterThan(14);
    expect(Math.min(...a)).toBeGreaterThanOrEqual(PEAK_ROW);
    // (valleys stay shallow; only lakes and the sea go deeper, never into the bedrock)
    expect(Math.max(...a)).toBeLessThanOrEqual(ROWS - 3);
    // (taken over several worlds, as one short stretch may well be mostly a lake or the sea)
    const many = [1, 2, 3, 4, 5, 6].flatMap(heights);
    expect(many.filter((h) => h > VALLEY_ROW + 9).length).toBeLessThan(many.length / 4);
    expect(a).not.toEqual(b);
    // The target height is a pure function of the seed: same seed, same hills
    const L1 = createLevel(THEMES.snow, 5);
    const L2 = createLevel(THEMES.snow, 5);
    for (let c = 0; c < 2000; c += 37) expect(terrainAt(L1, c)).toBe(terrainAt(L2, c));
    // A flat theme stays flat
    const flat = createLevel({ ...THEMES.jungle, relief: 0 }, 1);
    for (let c = 0; c < 2000; c += 37) expect(terrainAt(flat, c)).toBe(BASE_ROW);
  });

  it("hollows out caves deep under the ground", () => {
    const L = createLevel(THEMES.jungle, 6, { boss: null, length: 4000 });
    extend(L, 480);
    let caves = 0;
    // (only the columns still stored: a long set piece at the end may have run on past the ring's length)
    for (let c = Math.max(40, L.start + 1); c < 480; c++) {
      const g = L.ground[c % RING];
      for (let r = g + 7; r < ROWS - 4; r++) if (kindAt(L, c, r) === AIR && L.bg[cell(c, r)]) caves++;
      // Never into the bedrock
      expect(kindAt(L, c, ROWS - 1)).toBe(BEDROCK);
    }
    expect(caves).toBeGreaterThan(20);
  });

  it("keeps every pit and lava pool narrow enough to jump", () => {
    for (const key of ["jungle", "village", "inferno", "canyon", "swamp"]) {
      // (without the great descents: mountains, the depths and mineshafts are climbed down and up, not jumped)
      const t = THEMES[key];
      const L = createLevel({ ...t, features: Object.fromEntries(Object.entries(t.features).filter(([k]) => k !== "massif" && k !== "abyss" && k !== "mineshaft")) }, 5);
      extend(L, 400);
      let run = 0;
      let lip = L.ground[L.start % RING];
      for (let c = L.start + 1; c < L.next; c++) {
        const g = L.ground[c % RING];
        // A pit: ground well below the edge, with no bridge across at the edge's height (or lava in it)
        // (open water isn't a pit: it's swum)
        const wet = kindAt(L, c, lip + 1) === WATER || kindAt(L, c, lip) === WATER;
        const pit = (g >= lip + 3 && !wet && !solidAt(L, c * TILE + 4, lip * TILE + 4)) || kindAt(L, c, g - 1) === LAVA || kindAt(L, c, g - 2) === LAVA;
        run = pit ? run + 1 : 0;
        if (!pit) lip = g;
        expect(run, `${key} column ${c}`).toBeLessThanOrEqual(3);
      }
    }
  });

  it("keeps only the newest columns in its ring buffer", () => {
    const L = createLevel(THEMES.jungle, 1, { length: 4000 });
    extend(L, 3000);
    expect(L.next - L.start).toBeLessThanOrEqual(RING);
    expect(kindAt(L, L.start - 1, BASE_ROW)).toBe(AIR);
    expect(kindAt(L, L.next, BASE_ROW)).toBe(AIR);
  });

  it("opens on an empty landing zone and ends at a stronghold with targets for the ICBMs", () => {
    const L = createLevel(THEMES.desert, 4, { length: 120, boss: null, stronghold: true });
    const spawns = extend(L, 260);
    expect(spawns.filter((s) => s.x < 40 * TILE)).toEqual([]);
    expect(L.plan.fortPlaced).toBe(true);
    expect(L.plan.targets).toHaveLength(4);
    expect([...L.plan.targets].sort((a, b) => a - b)).toEqual(L.plan.targets);
    const fort = spawns.filter((s) => s.x >= L.plan.stronghold * TILE);
    expect(fort.some((s) => s.kind === "turret")).toBe(true);
    expect(fort.some((s) => s.kind === "mg")).toBe(true);
    expect(fort.some((s) => s.kind === "fuel" || s.kind === "tanker")).toBe(true);
  });

  it("clears an arena for the boss half way", () => {
    const L = createLevel(THEMES.snow, 2, { length: 160, boss: "mech" });
    extend(L, 200);
    expect(L.bossQueue).toHaveLength(1);
    const col = L.bossQueue[0].col;
    // Flat ground around the boss's mark
    for (let c = col - 8; c < col + 8; c++) expect(L.ground[c % RING]).toBe(L.ground[col % RING]);
  });

  it("places more enemies the more intense it gets", () => {
    // (over several levels: one level's mix of features is too much luck of the draw)
    const count = (k: number) =>
      [7, 8, 9, 10, 11, 12].reduce((n, seed) => {
        const L = createLevel(THEMES.city, seed, { length: 2000, boss: null });
        L.intensity = k;
        return n + extend(L, 400).filter((s) => ["mook", "heavy", "rocket", "sniper", "crawler", "mortar", "minigunner", "shield", "officer", "laser", "exosuit", "jetpacker"].includes(s.kind)).length;
      }, 0);
    expect(count(2)).toBeGreaterThan(count(1) * 1.4);
  });

  it("breaks blocks hit by hit, and never bedrock", () => {
    const L = createLevel(THEMES.jungle, 1);
    extend(L, 40);
    const c = 5;
    const r = L.ground[c];
    const kind = kindAt(L, c, r);
    for (let k = 1; k < TILE_HP[kind]; k++) expect(damageCell(L, c, r, 1)).toBe(0);
    expect(damageCell(L, c, r, 1)).toBe(kind);
    expect(kindAt(L, c, r)).toBe(AIR);
    expect(damageCell(L, c, ROWS - 1, 99)).toBe(0);
    const blast = blastCells(L, 10 * TILE, (ROWS - 1) * TILE, 30);
    expect(blast.every(([bc, br]) => L.fg[cell(bc, br)] !== BEDROCK)).toBe(true);
  });

  it("sees through air but not through the ground", () => {
    const L = createLevel(THEMES.jungle, 1);
    extend(L, 40);
    const y = (BASE_ROW - 10) * TILE;
    expect(lineOfSight(L, 8, y, 200, y)).toBe(true);
    expect(lineOfSight(L, 8, y, 200, (BASE_ROW + 5) * TILE)).toBe(false);
  });
});

describe("themes", () => {
  it("rotates through a dozen-plus distinct levels, each with its own boss line-up", () => {
    expect(ROTATION.length).toBeGreaterThanOrEqual(12);
    expect(new Set(ROTATION.map((k) => THEMES[k].name)).size).toBe(ROTATION.length);
    for (const k of ROTATION) expect(THEMES[k].bosses.length).toBeGreaterThan(0);
    expect(ROTATION).not.toContain("cave");
    expect(Object.values(THEMES).filter((t) => t.ceiling).map((t) => t.key)).toEqual(["cave"]);
  });

  it("never picks the same level twice in a row", () => {
    const r = rng(3);
    let current = "jungle";
    const seen = new Set<string>();
    for (let i = 0; i < 200; i++) {
      const next = nextTheme(r, current).key;
      expect(next).not.toBe(current);
      seen.add(next);
      current = next;
    }
    expect(seen.size).toBe(ROTATION.length);
  });
});

describe("game", () => {
  it("fields eight players, P1–P4 for the domains in their order and colours", () => {
    expect(SQUAD).toHaveLength(SQUAD_SIZE);
    expect(SQUAD.slice(0, 4).map((s) => s.domain)).toEqual(domains.map((d) => d.id));
    expect(new Set(SQUAD.map((s) => s.color)).size).toBe(SQUAD_SIZE);
    expect(new Set(SQUAD.map((s) => s.weapon)).size).toBe(SQUAD_SIZE);
    const g = new Game(THEMES.jungle, { seed: 1 });
    expect(g.players.map((p) => p.id)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(g.players.every((p) => p.hp === PLAYER_HP && !p.dead)).toBe(true);
    expect(g.banner?.text).toBe("Level 1");
    expect(g.banner?.sub).toBe("Jungle landing");
  });

  it("pushes the squad through the level, fighting as it goes", () => {
    const g = new Game(THEMES.jungle, { seed: 4, level: { boss: null, length: 600 } });
    play(g, 40);
    expect(g.advance).toBeGreaterThan(700);
    expect(g.kills).toBeGreaterThan(10);
    expect(g.booms).toBeGreaterThan(10);
    expect(g.players.reduce((s, p) => s + p.score, 0)).toBeGreaterThan(0);
    const up = g.players.filter((p) => !p.dead && !p.chute);
    expect(up.reduce((s, p) => s + Math.abs(p.x - g.advance), 0) / Math.max(1, up.length)).toBeLessThan(80);
  });

  it("stops for the boss at the arena and moves on once it's down", () => {
    const g = new Game(THEMES.desert, { seed: 2, level: { boss: "tank", length: 160 } });
    expect(g.bossKind).toBe("tank");
    expect(g.bossState).toBe("ahead");
    expect(g.banner?.info).toBe("Boss ahead: Siege tank");
    expect(g.objective).toMatch(/Siege tank ahead/);
    let bossSeen = false;
    let fought = false;
    let outcome = "";
    play(g, 90, false, () => {
      if (g.levelNo !== 1) return;
      if (g.enemies.some((e) => e.kind === "tank" && e.alive)) bossSeen = true;
      if (g.phase === "boss") {
        fought = true;
        expect(g.objective).toBe("Destroy the Siege tank");
      } else if (fought && !outcome) outcome = g.bossState;
    });
    expect(bossSeen).toBe(true);
    expect(fought).toBe(true);
    expect(outcome).toBe("destroyed");
  });

  it("calls ICBMs down on the stronghold, blows up the scene and moves on to a new level", () => {
    const g = new Game(THEMES.fortress, { seed: 3, level: { boss: null, length: 50, stronghold: true } });
    const first = g.theme.key;
    const seen = { nuke: false, clear: false, mushroom: false, missile: false, reset: false };
    play(g, 90, false, () => {
      if (g.phase === "nuke") seen.nuke = true;
      if (g.phase === "clear") seen.clear = true;
      if (g.mushrooms.length) seen.mushroom = true;
      if (g.missiles.some((m) => !m.up)) seen.missile = true;
      if (g.reset) seen.reset = true;
    });
    expect(seen).toEqual({ nuke: true, clear: true, mushroom: true, missile: true, reset: true });
    expect(g.levelNo).toBeGreaterThanOrEqual(2);
    expect(g.theme.key).not.toBe(first);
  });

  it("ends every level a different, random way, and the last warheads land on the squad", () => {
    const g = new Game(THEMES.fortress, { seed: 11, level: { boss: null, length: 50, stronghold: true } });
    const endings: string[] = [];
    let flung = 0;
    let cleared = 0;
    const titles = new Set<string>();
    play(g, 400, false, () => {
      if (g.phase === "nuke") flung = Math.max(flung, g.players.filter((p) => p.dead && p.vy < -100).length);
      if (g.complete?.title) titles.add(g.complete.title);
      if (g.cleared !== cleared) {
        cleared = g.cleared;
        endings.push(g.log[0].finale!);
      }
      // (keep the levels short)
      if (g.reset && !g.level.forceFort) g.nextLevel(undefined, { boss: null, length: 50, stronghold: true });
    });
    expect(endings.length).toBeGreaterThanOrEqual(5);
    for (let i = 1; i < endings.length; i++) expect(endings[i]).not.toBe(endings[i - 1]);
    expect(new Set(endings).size).toBeGreaterThanOrEqual(3);
    expect(flung).toBeGreaterThan(0);
    expect(titles.size).toBeGreaterThanOrEqual(2);
  }, 60000);

  it("dresses levels with props, big things in the background and ladders up the towers", () => {
    const counts = { props: 0, back: 0, ladders: 0 };
    for (const key of ["jungle", "fortress", "city", "desert", "deepjungle"]) {
      const L = createLevel(THEMES[key], 3, { boss: null, length: 2000 });
      extend(L, RING - 20);
      counts.props += L.props.filter((p) => p.kind === "sprite" && !p.back).length;
      counts.back += L.props.filter((p) => p.kind === "sprite" && p.back).length;
      for (let c = L.start; c < L.next; c++) for (let r = 0; r < ROWS; r++) if (kindAt(L, c, r) === LADDER) counts.ladders++;
    }
    expect(counts.props).toBeGreaterThan(40);
    expect(counts.back).toBeGreaterThan(4);
    expect(counts.ladders).toBeGreaterThan(10);
  });

  it("builds over a hundred different kinds of building, furnished from a catalogue that has everything they need", () => {
    expect(BUILDINGS.length).toBeGreaterThanOrEqual(100);
    expect(new Set(BUILDINGS.map((b) => b.key)).size).toBe(BUILDINGS.length);
    // No two alike: each has its own mix of material, roof, windows and contents
    expect(new Set(BUILDINGS.map((b) => `${b.wall}/${b.roof}/${b.windows}/${b.items.join()}`)).size).toBe(BUILDINGS.length);
    for (const b of BUILDINGS)
      for (const id of [...b.items, ...(b.walls ?? []), ...(b.upper ?? []), ...(b.top ?? []), ...(b.outside ?? [])]) expect(ASSETS[id], `${b.key}: ${id}`).toBeTruthy();
    expect(Object.keys(ASSETS).length).toBeGreaterThanOrEqual(240);
    const L = createLevel(THEMES.city, 2, { boss: null, length: 3000 });
    const spawns = extend(L, RING - 20);
    // Shops and offices with their signs, across a few city stretches (the old-style buildings have none)
    const signs = [L, ...[5, 6].map((seed) => {
      const M = createLevel(THEMES.city, seed, { boss: null, length: 3000 });
      extend(M, RING - 20);
      return M;
    })].reduce((n, M) => n + M.props.filter((p) => p.kind === "sign").length, 0);
    expect(signs).toBeGreaterThan(3);
    // (a harbour turns up somewhere along a city's miles of streets and metro lines)
    const harbour = [spawns, ...[3, 4].map((seed) => extend(createLevel(THEMES.city, seed, { boss: null, length: 6000 }), 2500))];
    expect(harbour.some((list) => list.some((s) => s.kind === "battleship"))).toBe(true);
  });

  it("sends a lit propane tank rocketing up before it goes off, and blasts smash props", () => {
    const g = new Game(THEMES.cave, { seed: 1 });
    const view = play(g, 0.1);
    const y = g.level.ground[4] * TILE;
    g.barrels.push({ x: 36, y, vx: 0, vy: 0, ground: true, facing: 1, type: "propane", hp: 1, fuse: -1, owner: -3, alive: true });
    const tank = g.barrels[g.barrels.length - 1];
    g.explode(20, y - 4, 10, -2, false);
    let top = y;
    for (let i = 0; i < 40 && tank.alive; i++) {
      g.step(1 / 30, view);
      top = Math.min(top, tank.y);
    }
    expect(top).toBeLessThan(y - 40);
    expect(tank.alive).toBe(false);
    g.level.props.push({ kind: "sprite", id: "crates", x: 60, y, flip: false, back: false });
    g.explode(60, y - 6, 16, -2, false);
    expect(g.level.props.some((p) => p.kind === "sprite" && p.x === 60)).toBe(false);
  });

  it("starts a new level with the squad parachuting in, keeping their scores", () => {
    const g = new Game(THEMES.jungle, { seed: 1 });
    play(g, 5);
    const scores = g.players.map((p) => p.score);
    g.nextLevel(THEMES.cave);
    expect(g.levelNo).toBe(2);
    expect(g.theme.name).toBe("Underground");
    expect(g.reset).toBe(true);
    expect(g.players.every((p) => p.chute && !p.dead)).toBe(true);
    expect(g.players.map((p) => p.score)).toEqual(scores);
    expect(g.enemies).toEqual([]);
  });

  it("drops a downed player back in by parachute", () => {
    const g = new Game(THEMES.cave, { seed: 1 });
    play(g, 1);
    const p = g.players[1];
    g.hurtPlayer(p, 99);
    expect(p.dead).toBe(true);
    expect(g.popups.some((u) => u.text === "P2 DOWN")).toBe(true);
    // (down for a couple of seconds, then the next transport over drops them back in)
    play(g, 4.5);
    expect(p.dead).toBe(false);
    expect(p.hp).toBe(PLAYER_HP);
  });

  it("chains barrels, breaks the ground, and doesn't hurt the squad with its own or the visitor's blasts", () => {
    const g = new Game(THEMES.cave, { seed: 1 });
    const view = play(g, 0.1);
    const y = g.level.ground[2] * TILE;
    g.barrels.push(...[20, 30].map((x) => ({ x, y, vx: 0, vy: 0, ground: true, facing: 1 as const, type: "barrel" as const, hp: 1, fuse: -1, owner: -3, alive: true })));
    const p = g.players[0];
    const hp = p.hp;
    g.strike(20, y - 2);
    g.explode(p.x, p.y - 4, 18, 0, false);
    expect(p.hp).toBe(hp);
    for (let i = 0; i < 30; i++) g.step(1 / 30, view);
    expect(g.barrels.filter((b) => b.x <= 30 && b.alive)).toHaveLength(0);
    expect(g.booms).toBeGreaterThanOrEqual(4);
    expect(kindAt(g.level, 2, g.level.ground[2])).toBe(AIR);
    // Artillery does hurt
    p.inv = 0;
    g.explode(p.x, p.y - 4, 18, -3, true);
    expect(p.hp).toBeLessThan(PLAYER_HP);
  });

  it("hatches alien eggs into crawlers when the squad comes close", () => {
    const g = new Game(THEMES.hive, { seed: 1 });
    const view = play(g, 0.1);
    const p = g.players[0];
    g.barrels.push({ x: p.x + 20, y: p.y, vx: 0, vy: 0, ground: true, facing: 1, type: "egg", hp: 3, fuse: -1, owner: -3, alive: true });
    g.step(1 / 30, view);
    expect(g.enemies.filter((e) => e.kind === "crawler").length).toBeGreaterThanOrEqual(2);
  });

  it("stands still when frozen (reduced motion), letting only what's in flight finish", () => {
    const g = new Game(THEMES.refinery, { seed: 3 });
    play(g, 3);
    const xs = g.players.map((p) => p.x);
    const advance = g.advance;
    play(g, 3, true);
    expect(g.players.map((p) => p.x)).toEqual(xs);
    expect(g.advance).toBe(advance);
  });
});
