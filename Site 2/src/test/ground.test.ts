import { describe, expect, it } from "vitest";
import {
  AIR,
  AMETHYST,
  BASE_ROW,
  cell,
  createLevel,
  CORAL,
  DRYGRASS,
  extend,
  GLACIER,
  GRANITE,
  kindAt,
  KINDS,
  LADDER,
  MAGMA,
  OBSIDIAN,
  PERMAFROST,
  RAIL,
  RING,
  ROWS,
  SAND,
  SCREE,
  SHALE,
  SOLID,
  TILE,
  TILE_HP,
  type Level,
} from "@/components/pixel/level";
import { THEMES, type Theme } from "@/components/pixel/themes";
import { Game, PAL_CHEM, PAL_FIRE, type View } from "@/components/pixel/game";

/** Step a game for `seconds`, the camera following the squad's feet up and down. */
function play(g: Game, seconds: number, each?: () => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    const up = g.players.filter((p) => !p.dead && !p.chute && !p.aboard);
    view.y =
      (up.length ? up.reduce((s, p) => s + p.y, 0) / up.length : g.ship ? g.ship.y + 58 : BASE_ROW * TILE) -
      view.h * 0.68;
    g.step(1 / 30, view);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.();
  }
}

/** A quiet country of mineshafts and some flat ground, with nothing coming in on its own. */
const mines: Theme = {
  ...THEMES.summit,
  features: { mineshaft: 3, flat: 1 },
  enemies: 0,
  barrels: 0,
  artillery: 0,
  helis: 0,
  paras: 0,
  jets: 0,
  drones: 0,
  relief: 0,
};

/** How many of each kind of block a stretch of level holds. */
function census(L: Level) {
  const n = new Array<number>(KINDS).fill(0);
  for (let c = L.start; c < L.next; c++) for (let r = 0; r < ROWS; r++) n[L.fg[cell(c, r)]]++;
  return n;
}

describe("the ground", () => {
  it("is made of many more kinds of earth and rock, each country its own", () => {
    const seen = (key: string, seed = 2) => {
      const L = createLevel(THEMES[key], seed, { boss: null, length: 4000 });
      extend(L, 400);
      return census(L);
    };
    const temperate = seen("autumn");
    expect(temperate[SHALE]).toBeGreaterThan(10);
    const volcanic = seen("inferno");
    expect(volcanic[OBSIDIAN] + volcanic[MAGMA]).toBeGreaterThan(10);
    const cold = seen("tundra");
    expect(cold[PERMAFROST]).toBeGreaterThan(10);
    const savanna = seen("savanna");
    expect(savanna[DRYGRASS]).toBeGreaterThan(5);
    // Deep down, everywhere: geodes
    expect(temperate[AMETHYST] + cold[AMETHYST] + savanna[AMETHYST]).toBeGreaterThan(0);
    // Reefs on the seabed
    const sea = createLevel({ ...THEMES.archipelago, features: { ocean: 1 } }, 3, { boss: null, length: 4000 });
    extend(sea, 300);
    expect(census(sea)[CORAL]).toBeGreaterThan(5);
  });

  it("gives every new kind a texture's worth of toughness, and makes granite breakable at last", () => {
    for (let k = 1; k < KINDS; k++) expect(TILE_HP[k], `kind ${k}`).toBeGreaterThan(0);
    expect(TILE_HP[GRANITE]).toBeLessThan(255);
    expect(TILE_HP[GRANITE]).toBeGreaterThan(6);
    // Up the great mountains: scree on the slopes, glacier ice under the summit's snow
    const L = createLevel({ ...mines, features: { massif: 3, flat: 1 } }, 3, { boss: null, length: 4000 });
    extend(L, RING - 20);
    const n = census(L);
    expect(n[SCREE]).toBeGreaterThan(0);
    expect(n[GLACIER]).toBeGreaterThan(0);
  });
});

describe("mineshafts", () => {
  it("go down a shaft twenty-odd blocks to a timbered gallery with rails and the mine's kit, and back up a second shaft", () => {
    const L = createLevel(mines, 2, { boss: null, length: 4000 });
    extend(L, 400);
    const deep = [];
    for (let c = L.start; c < L.next; c++) if (L.ground[c % RING] - L.crust[c % RING] >= 18) deep.push(c);
    expect(deep.length).toBeGreaterThan(40);
    let rails = 0;
    let ladders = 0;
    for (const c of deep) {
      if (kindAt(L, c, L.ground[c % RING] - 1) === RAIL) rails++;
      for (let r = L.crust[c % RING]; r < L.ground[c % RING]; r++) if (kindAt(L, c, r) === LADDER) ladders++;
    }
    expect(rails).toBeGreaterThan(30);
    expect(ladders).toBeGreaterThan(30);
    const kit = new Set(L.props.filter((p) => p.kind === "sprite").map((p) => (p as { id: string }).id));
    for (const id of ["headframe", "winch", "orebin"]) expect(kit.has(id), id).toBe(true);
    expect(
      ["cartcoal", "carttipped", "drillrig", "timbers", "dynamite", "railbuffer"].filter((id) => kit.has(id)).length
    ).toBeGreaterThan(2);
  });

  it("have carts rolling up and down the rails, and the squad goes down the shaft and on through", () => {
    const g = new Game(mines, { seed: 2, gunship: false, level: { boss: null, length: 4000 } });
    let rolling = 0;
    let lowest = 0;
    const where = new Map<object, number>();
    let moved = 0;
    play(g, 60, () => {
      rolling = Math.max(rolling, g.carts.filter((k) => k.vx !== 0).length);
      for (const k of g.carts) {
        const was = where.get(k);
        if (was !== undefined && Math.abs(was - k.x) > 40) moved++;
        if (was === undefined) where.set(k, k.x);
      }
      for (const p of g.players) if (!p.dead && !p.chute) lowest = Math.max(lowest, p.y / TILE - BASE_ROW);
    });
    expect(rolling).toBeGreaterThan(1);
    expect(moved).toBeGreaterThan(0);
    expect(lowest).toBeGreaterThan(18);
    // ... and on past the mines: the advance didn't stall in them
    expect(g.advance).toBeGreaterThan(1800);
  });
});

describe("blasts", () => {
  /** A game on flat ground in a quiet country, its first stretch built. */
  const quiet = (theme: Theme) => {
    const g = new Game(
      {
        ...theme,
        features: { flat: 1 },
        enemies: 0,
        barrels: 0,
        artillery: 0,
        helis: 0,
        paras: 0,
        jets: 0,
        drones: 0,
        relief: 0,
      },
      { seed: 1, gunship: false, level: { boss: null, length: 4000 } }
    );
    play(g, 1 / 30);
    return g;
  };

  it("dig deeper than they did, and loose sand slides down into the crater", () => {
    const g = quiet(THEMES.archipelago);
    const L = g.level;
    // Out ahead of the squad: falling ground stops short of anyone's head
    const c = 75;
    const top = L.ground[c % RING];
    for (let x = c - 4; x <= c + 4; x++)
      for (let r = top; r < top + 7; r++) {
        const i = cell(x, r);
        L.fg[i] = SAND;
        L.hp[i] = TILE_HP[SAND];
      }
    g.explode(c * TILE + 4, (top + 4) * TILE, 22, -3, false);
    // A hole straight away, under the surface
    let holes = 0;
    for (let r = top + 1; r < top + 7; r++) for (let x = c - 2; x <= c + 2; x++) if (kindAt(L, x, r) === AIR) holes++;
    expect(holes).toBeGreaterThan(4);
    const before = holes;
    // ... and the sand round it pours in: the air in the crater moves up, out of the ground
    play(g, 2);
    let after = 0;
    for (let r = top + 3; r < top + 7; r++) for (let x = c - 2; x <= c + 2; x++) if (kindAt(L, x, r) === AIR) after++;
    expect(after).toBeLessThan(before);
  });

  it("bring down a piece of rock cut off from the rest, block by block", () => {
    const g = quiet(THEMES.autumn);
    const L = g.level;
    const c = 75;
    const top = L.ground[c % RING];
    // A slab of rock hanging in the air over the ground, held up by one column of blocks
    for (let x = c - 3; x <= c + 3; x++) {
      const i = cell(x, top - 6);
      L.fg[i] = SHALE;
      L.hp[i] = TILE_HP[SHALE];
    }
    for (let r = top - 5; r < top; r++) {
      const i = cell(c, r);
      L.fg[i] = SHALE;
      L.hp[i] = TILE_HP[SHALE];
    }
    // Shoot the pillar out from under it
    g.explode(c * TILE + 4, (top - 3) * TILE, 14, -2, false);
    play(g, 2);
    // Nothing left up there: it's come down to the ground
    for (let x = c - 3; x <= c + 3; x++) expect(SOLID[kindAt(L, x, top - 6)], `column ${x}`).toBe(0);
    let fallen = 0;
    for (let x = c - 3; x <= c + 3; x++) if (kindAt(L, x, top - 1) === SHALE) fallen++;
    expect(fallen).toBeGreaterThan(3);
  });

  it("go up in colours: each one's palette, and glints hanging in the air after it", () => {
    const g = quiet(THEMES.jungle);
    g.explode(300, BASE_ROW * TILE - 20, 20, -2, false, PAL_CHEM);
    expect(g.blasts.at(-1)?.pal).toBe(PAL_CHEM);
    expect(g.parts.some((p) => p.k === 18)).toBe(true);
    g.explode(400, BASE_ROW * TILE - 20, 20, -2, false);
    expect(g.blasts.at(-1)?.pal).toBe(PAL_FIRE);
    // ... and little secondary bursts round the bigger ones, a moment after
    const balls = g.blasts.length;
    play(g, 0.6);
    expect(g.booms).toBeGreaterThanOrEqual(2);
    expect(balls).toBeGreaterThan(0);
  });
});
