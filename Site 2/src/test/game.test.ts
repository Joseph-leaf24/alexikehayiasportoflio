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
  LAVA,
  lineOfSight,
  RING,
  ROWS,
  solidAt,
  TILE,
  TILE_HP,
} from "@/components/pixel/level";
import { nextTheme, ROTATION, THEMES } from "@/components/pixel/themes";
import { Game, PLAYER_HP, SQUAD_SIZE, type View } from "@/components/pixel/game";
import { rng } from "@/components/pixel/rng";
import { SQUAD } from "@/data/squad";
import { domains } from "@/data/domains";

/** Step a game for `seconds`, with the camera following the squad like PixelScene's does. `each` runs every step. */
function play(g: Game, seconds: number, frozen = false, each?: () => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    view.y = BASE_ROW * TILE - view.h * 0.68;
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
        expect(L.ground[c % RING]).toBeGreaterThan(30);
        expect(L.ground[c % RING]).toBeLessThanOrEqual(ROWS - 3);
      }
    }
  });

  it("keeps every pit and lava pool narrow enough to jump", () => {
    for (const key of ["jungle", "village", "inferno", "canyon", "swamp"]) {
      const L = createLevel(THEMES[key], 5);
      extend(L, 400);
      let run = 0;
      let lip = L.ground[L.start % RING];
      for (let c = L.start + 1; c < L.next; c++) {
        const g = L.ground[c % RING];
        // A pit: ground well below the edge, with no bridge across at the edge's height (or lava in it)
        const pit = (g >= lip + 3 && !solidAt(L, c * TILE + 4, lip * TILE + 4)) || kindAt(L, c, g - 1) === LAVA || kindAt(L, c, g - 2) === LAVA;
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
    const L = createLevel(THEMES.desert, 4, { length: 120, boss: null });
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
    expect(L.plan.bossPlaced).toBe(true);
    const col = L.plan.boss!.col;
    // Flat ground around the boss's mark
    for (let c = col - 8; c < col + 8; c++) expect(L.ground[c % RING]).toBe(L.ground[col % RING]);
  });

  it("places more enemies the more intense it gets", () => {
    const count = (k: number) => {
      const L = createLevel(THEMES.city, 7, { length: 2000, boss: null });
      L.intensity = k;
      return extend(L, 600).filter((s) => ["mook", "heavy", "rocket", "sniper", "crawler", "mortar"].includes(s.kind)).length;
    };
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
    play(g, 60, false, () => {
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
    const g = new Game(THEMES.fortress, { seed: 3, level: { boss: null, length: 50 } });
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
    play(g, 3);
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
