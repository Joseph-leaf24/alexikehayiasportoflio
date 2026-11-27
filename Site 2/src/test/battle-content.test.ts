import { describe, expect, it, vi } from "vitest";
import { Atlas, createFrame, Registry, runHeadless, TextList, TICK } from "@/engine";
import { machine, TEAM_REGIME, TEAM_SQUAD, type ArtDef, type ArtSource, type Feature, type LevelEvent } from "@/battle";
import { ArtBook } from "@/battle/art";
import { GROUND_Y, ix, stagedGame } from "./battle-fixtures";

/** A canvas stand-in: the atlas only reads its size (jsdom has no 2D canvas to paint with). */
const fakeCanvas = (w: number, h: number) => ({ width: w, height: h }) as unknown as ArtSource;

/** Everything a content feature adds, in one place and nowhere else: art, an effect, a shot, a gun, a unit. */
const turretMod: Feature = {
  name: "turret-mod",
  install(g) {
    const c = g.content;
    c.art.register("mod.turret", { paint: () => [{ canvas: fakeCanvas(12, 8), ox: 6, oy: 7 }] });
    c.effects.register("mod.puff", { count: 2, speed: [5, 10], life: [0.2, 0.3], colors: [0xffffffff] });
    c.projectiles.register("mod.ball", { speed: 220, life: 2, damage: 3, trail: "mod.puff", trailRate: 20 });
    c.weapons.register("mod.cannon", { name: "Cannon", range: 300, cadence: 0.5, projectile: "mod.ball" });
    c.units.register("mod.turret", {
      name: "Turret",
      hw: 6,
      h: 8,
      hp: 20,
      team: TEAM_REGIME,
      art: "mod.turret",
      weapon: "mod.cannon",
      move: { speed: 0, jump: 0, climb: 0 },
      ai: {
        every: 3,
        think: machine({
          initial: "scan",
          states: {
            scan: {
              run(game, _id, i) {
                const t = game.world.spatial.nearest(game.core.pos.x[i], game.core.pos.y[i] - 4, 300, 1 << TEAM_SQUAD);
                game.c.intent.target[i] = t;
                if (t) return "fire";
              },
            },
            fire: {
              enter: (game, _id, i) => void (game.c.intent.fire[i] = 1),
              run: (game, _id, i) => (game.fighting(game.c.intent.target[i]) ? undefined : "scan"),
            },
          },
        }),
      },
    });
    c.units.register("mod.dummy", { name: "Dummy", hw: 3, h: 15, hp: 100, team: TEAM_SQUAD, art: "mod.turret" });
  },
};

describe("battle content and extension", () => {
  it("a feature adds art, an effect, a projectile, a weapon and a unit with an AI, and they work together", () => {
    const g = stagedGame({ extra: [turretMod] });
    const k = g.content.units.index("mod.turret");
    expect(g.links.unitWeapon[k]).toBe(g.content.weapons.index("mod.cannon"));
    expect(g.links.weaponProjectile[g.links.unitWeapon[k]]).toBe(g.content.projectiles.index("mod.ball"));
    const turret = g.spawn("mod.turret", 1100, GROUND_Y);
    const dummy = g.spawn("mod.dummy", 1300, GROUND_Y);
    let trail = 0;
    runHeadless(g.world, 3, { onTick: () => void (trail = Math.max(trail, g.particles.n)) });
    expect(g.c.ai.state[ix(turret)]).toBe(2);
    expect(g.core.health.hp[ix(dummy)]).toBeLessThan(100);
    expect(trail).toBeGreaterThan(0);
    expect(() => g.content.units.register("late", g.content.units.get(k))).toThrow(/frozen/);
  });

  it("a reference to content that doesn't exist fails at boot, naming it", () => {
    const broken: Feature = {
      name: "broken",
      install: (g) =>
        void g.content.units.register("bad.unit", {
          name: "Bad",
          hw: 3,
          h: 15,
          hp: 1,
          team: 2,
          art: "player.0.0",
          weapon: "no.such.gun",
        }),
    };
    expect(() => stagedGame({ extra: [broken] })).toThrow(/units "bad.unit" refers to unknown weapons "no.such.gun"/);
  });

  it("budgets hold: enemies, projectiles and particles never exceed their caps", () => {
    const g = stagedGame({ budgets: { maxEnemies: 3, maxProjectiles: 5, maxParticles: 40 } });
    const ids = [0, 1, 2, 3].map((k) => g.spawn("regime.trooper", 1100 + k * 20, GROUND_Y));
    expect(ids.filter((id) => id !== 0)).toHaveLength(3);
    expect(g.spawn("regime.trooper", 1200, GROUND_Y, { force: true })).not.toBe(0);
    const shots = Array.from({ length: 8 }, () => g.shoot("regime.bullet", 1000, GROUND_Y - 50, 100, 0));
    expect(shots.filter((id) => id !== 0)).toHaveLength(5);
    for (let k = 0; k < 5; k++) g.explode(1000 + k * 40, GROUND_Y, { r: 40 });
    expect(g.particles.n).toBe(40);
  });

  it("levelStart reaches features at boot, and the banner event goes out when the banner shows", () => {
    const seen: LevelEvent[] = [];
    const banners: string[] = [];
    const spy: Feature = {
      name: "spy",
      install: (g) => {
        g.world.events.on("levelStart", (e) => seen.push({ ...e }));
        g.world.events.on("banner", (e) => banners.push(e.text));
      },
    };
    const g = stagedGame({ extra: [spy], feature: { flow: { name: "Test range" } } });
    expect(seen).toEqual([{ no: 1, name: "Test range", time: 0 }]);
    runHeadless(g.world, TICK);
    expect(banners).toEqual(["Level 1"]);
  });

  it("delayed calls run in order of their time, in simulation time", () => {
    const g = stagedGame({});
    const order: string[] = [];
    g.at(0.5, () => order.push("b"));
    g.at(0.1, () => order.push("a"));
    g.at(0.5, () => order.push("c"));
    runHeadless(g.world, 0.3);
    expect(order).toEqual(["a"]);
    runHeadless(g.world, 0.4);
    expect(order).toEqual(["a", "b", "c"]);
  });

  it("debug turns system timing on; unattended runs carry on past a throwing system", () => {
    expect(stagedGame({}).world.stats.timing).toBe(false);
    expect(stagedGame({ debug: true }).world.stats.timing).toBe(true);
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const bad: Feature = {
      name: "bad",
      install: (g) =>
        void g.world.addSystem({
          name: "bad.system",
          phase: "ai",
          run: () => {
            throw new Error("boom");
          },
        }),
    };
    const g = stagedGame({ extra: [bad], unattended: true });
    runHeadless(g.world, 1);
    expect(g.world.time.tick).toBe(30);
    expect(g.world.stats.system("bad.system")?.errors).toBe(30);
    expect(err).toHaveBeenCalled();
  });
});

describe("battle render extraction (headless)", () => {
  it("fills a frame from the simulation: camera interpolated, sky by the clock, blasts' particles and light, the banner", () => {
    const g = stagedGame({});
    runHeadless(g.world, 0.5);
    const f = createFrame();
    g.camera.prevX = 1000;
    g.camera.x = 1100;
    g.buildFrame(f, 0.5);
    expect(f.camera.x).toBe(1050);
    expect(f.terrain).toBe(g.terrain);
    expect((f.text as TextList).count).toBe(2);
    const daySky = f.sky.top;
    g.explode(1200, GROUND_Y, { r: 30 });
    g.sky.clock = 0.5;
    g.buildFrame(f, 0);
    expect(f.particles.count).toBeGreaterThan(10);
    expect(f.lights.count).toBeGreaterThan(0);
    expect(f.sky.top).not.toBe(daySky);
    expect(f.ambient).not.toBe(0xffffff);
    expect(f.effects.shockwaves.some((w) => w.strength === 0)).toBe(true);
  });

  it("the art book paints a set the first time it is drawn and looks frames up by pose and facing", () => {
    const reg = new Registry<ArtDef>("art");
    let painted = 0;
    reg.register("two", {
      facings: 2,
      paint: () => {
        painted++;
        return [0, 1, 2, 3].map((k) => ({ canvas: fakeCanvas(4 + k, 4), ox: 0, oy: 3 }));
      },
    });
    reg.register("one", { paint: () => [{ canvas: fakeCanvas(3, 3), ox: 1, oy: 1 }] });
    reg.register("broken", {
      paint: () => {
        throw new Error("no canvas");
      },
    });
    const book = new ArtBook(reg);
    expect(book.frame(1, 0, 1)).toBe(-1);
    const atlas = new Atlas();
    book.attach(atlas);
    expect(painted).toBe(0);
    const r0 = book.frame(1, 0, 1);
    const l1 = book.frame(1, 1, -1);
    expect(painted).toBe(1);
    expect(atlas.frame(r0).w).toBe(4);
    expect(atlas.frame(l1).w).toBe(7);
    expect(book.frame(1, 9, 1)).toBe(book.frame(1, 1, 1));
    expect(book.poses(1)).toBe(2);
    expect(book.mirrors(2)).toBe(true);
    expect(book.mirrors(1)).toBe(false);
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(book.frame(3, 0, 1)).toBe(-1);
    expect(book.frame(3, 0, 1)).toBe(-1);
    expect(err).toHaveBeenCalledTimes(1);
    book.frame(1, 0, 1);
    expect(painted).toBe(1);
  });
});
