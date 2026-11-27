import { describe, expect, it } from "vitest";
import { materials as M, runHeadless, TICK } from "@/engine";
import {
  MODE_KNOCK,
  TAG_ARMOURED,
  TEAM_REGIME,
  TEAM_SQUAD,
  UNIT_DEAD,
  type BattleGame,
  type Feature,
  type KillEvent,
} from "@/battle";
import { hitscan } from "@/battle/physics";
import { at, fill, GROUND, GROUND_Y, ix, stagedGame } from "./battle-fixtures";

/** Targets that stand still, a slug far faster than a cell per tick, and a piercing one. */
const range: Feature = {
  name: "range",
  install(g) {
    const u = g.content.units;
    u.register("test.dummy", {
      name: "Dummy",
      hw: 3,
      h: 15,
      hp: 10,
      team: TEAM_REGIME,
      art: "troop.jungle.mook",
      score: 50,
    });
    u.register("test.friend", { name: "Friend", hw: 3, h: 15, hp: 10, team: TEAM_SQUAD, art: "player.0.0" });
    u.register("test.tank", {
      name: "Tank",
      hw: 10,
      h: 12,
      hp: 60,
      team: TEAM_REGIME,
      art: "troop.jungle.heavy",
      tags: TAG_ARMOURED,
    });
    const p = g.content.projectiles;
    p.register("test.slug", { speed: 3000, life: 1, damage: 4 });
    p.register("test.lance", { speed: 3000, life: 1, damage: 4, pierce: 1 });
  },
};

const setup = () => stagedGame({ extra: [range] });
const hp = (g: BattleGame, id: number) => g.core.health.hp[ix(id)];
const dead = (g: BattleGame, id: number) => (g.c.unit.flags[ix(id)] & UNIT_DEAD) !== 0;
/** Let the spatial hash pick up freshly spawned units before anything shoots at them. */
const settle = (g: BattleGame) => runHeadless(g.world, TICK * 2);

describe("battle projectiles", () => {
  it("a shot many cells a tick stops at a one-cell wall instead of tunnelling through it", () => {
    const g = setup();
    fill(g.terrain, 160, GROUND - 6, 1, 6, M.STONE);
    const id = g.shoot("test.slug", 1100, GROUND_Y - 12, 3000, 0, { team: TEAM_SQUAD });
    let furthest = 0;
    runHeadless(g.world, 0.5, {
      onTick: () => void (g.world.alive(id) && (furthest = Math.max(furthest, g.core.pos.x[ix(id)]))),
    });
    expect(g.world.alive(id)).toBe(false);
    expect(furthest).toBeLessThan(160 * 8);
  });

  it("a shot many cells a tick hits the body in its path rather than skipping past it", () => {
    const g = setup();
    const target = g.spawn("test.dummy", 1320, GROUND_Y);
    settle(g);
    g.shoot("test.slug", 1100, GROUND_Y - 8, 3000, 0, { team: TEAM_SQUAD });
    runHeadless(g.world, 0.3);
    expect(hp(g, target)).toBe(6);
  });

  it("a wall in front of a body takes the shot", () => {
    const g = setup();
    fill(g.terrain, 160, GROUND - 6, 1, 6, M.STONE);
    const target = g.spawn("test.dummy", 1320, GROUND_Y);
    settle(g);
    g.shoot("test.slug", 1100, GROUND_Y - 8, 3000, 0, { team: TEAM_SQUAD });
    runHeadless(g.world, 0.3);
    expect(hp(g, target)).toBe(10);
  });

  it("a piercing shot hits bodies nearest first, as many as it can pierce", () => {
    const g = setup();
    const a = g.spawn("test.dummy", 1200, GROUND_Y);
    const b = g.spawn("test.dummy", 1240, GROUND_Y);
    const c = g.spawn("test.dummy", 1280, GROUND_Y);
    settle(g);
    g.shoot("test.lance", 1100, GROUND_Y - 8, 3000, 0, { team: TEAM_SQUAD });
    runHeadless(g.world, 0.3);
    expect([hp(g, a), hp(g, b), hp(g, c)]).toEqual([6, 6, 10]);
  });

  it("team masks: a squad shot passes through squad bodies to the enemy behind", () => {
    const g = setup();
    const friend = g.spawn("test.friend", 1200, GROUND_Y);
    const foe = g.spawn("test.dummy", 1260, GROUND_Y);
    settle(g);
    g.shoot("test.slug", 1100, GROUND_Y - 8, 3000, 0, { team: TEAM_SQUAD });
    runHeadless(g.world, 0.3);
    expect(hp(g, friend)).toBe(10);
    expect(hp(g, foe)).toBe(6);
  });

  it("a hitscan beam hits bodies nearest first through its pierce, and stops at the first wall", () => {
    const g = setup();
    fill(g.terrain, 165, GROUND - 6, 1, 6, M.STONE);
    const a = g.spawn("test.dummy", 1200, GROUND_Y);
    const b = g.spawn("test.dummy", 1240, GROUND_Y);
    const c = g.spawn("test.dummy", 1280, GROUND_Y);
    const behind = g.spawn("test.dummy", 1360, GROUND_Y);
    settle(g);
    const r = hitscan(g, 0, TEAM_SQUAD, 1100, GROUND_Y - 8, 1500, GROUND_Y - 8, { damage: 3, pierce: 5 });
    expect(r.hits).toBe(3);
    expect(r.wall).toBe(true);
    expect(r.x).toBeCloseTo(165 * 8, 3);
    expect([hp(g, a), hp(g, b), hp(g, c), hp(g, behind)]).toEqual([7, 7, 7, 10]);
    const short = hitscan(g, 0, TEAM_SQUAD, 1100, GROUND_Y - 8, 1500, GROUND_Y - 8, { damage: 3 });
    expect(short.hits).toBe(1);
    expect(short.x).toBeLessThan(1200);
    expect(hp(g, a)).toBe(4);
  });

  it("bullets chip the cell they strike", () => {
    const g = setup();
    fill(g.terrain, 160, GROUND - 6, 1, 6, M.DIRT);
    const row = GROUND - 2;
    // Dirt takes two hits; the third shot flies through the hole (the cells above crumble down into it)
    const cells = () => {
      let n = 0;
      for (let c = 157; c <= 163; c++)
        for (let r = GROUND - 8; r < GROUND; r++) n += g.terrain.get(c, r) === M.DIRT ? 1 : 0;
      return n;
    };
    for (let k = 0; k < 3; k++)
      expect(g.shoot("regime.bullet", 1200, row * 8 + 4, 400, 0, { team: TEAM_REGIME })).not.toBe(0);
    runHeadless(g.world, 1);
    expect(cells()).toBe(5);
  });

  it("a grenade bounces off the ground and goes off on its fuse", () => {
    const g = setup();
    const id = g.shoot("regime.grenade", 1200, GROUND_Y - 30, 60, 100, { team: TEAM_REGIME });
    let bounced = false;
    runHeadless(g.world, 0.6, { onTick: () => void (bounced ||= g.c.projectile.bounces[ix(id)] > 0) });
    expect(bounced).toBe(true);
    expect(g.stats.booms).toBe(0);
    runHeadless(g.world, 1);
    expect(g.stats.booms).toBe(1);
    expect(g.terrain.solidAt(g.core.pos.x[ix(id)] || 1230, GROUND_Y + 2)).toBe(false);
  });
});

describe("battle explosions and the damage pipeline", () => {
  it("an explosion carves a crater, kills light infantry, hurts armour, and only pushes its own side", () => {
    const g = setup();
    const trooper = g.spawn("regime.trooper", 1212, GROUND_Y);
    const tank = g.spawn("test.tank", 1190, GROUND_Y);
    const friend = g.spawn("test.friend", 1222, GROUND_Y);
    settle(g);
    const blasts: number[] = [];
    g.world.events.on("blast", (e) => blasts.push(e.cells));
    const particles = g.particles.n;
    g.explode(1204, GROUND_Y, { r: 30 }, 0, TEAM_SQUAD);
    expect(g.terrain.solidAt(1204, GROUND_Y + 4)).toBe(false);
    expect(dead(g, trooper)).toBe(true);
    expect(hp(g, tank)).toBeLessThan(60);
    expect(dead(g, tank)).toBe(false);
    expect(hp(g, friend)).toBe(10);
    expect(g.c.motor.mode[ix(friend)]).toBe(MODE_KNOCK);
    expect(g.particles.n).toBeGreaterThan(particles + 20);
    expect(g.flashes.n).toBeGreaterThan(0);
    expect(g.screen.rings).toBe(1);
    expect(g.camera.shake).toBeGreaterThan(0);
    g.world.flush();
    expect(blasts.length).toBe(1);
    expect(blasts[0]).toBeGreaterThan(5);
  });

  it("a kill credits the player who made it, with a pooled unitKilled event", () => {
    const g = stagedGame({ extra: [range], withSquad: true });
    runHeadless(g.world, TICK);
    const shooter = g.world.query(g.c.player).first();
    const pid = g.c.player.id[shooter];
    const target = g.spawn("test.dummy", g.core.pos.x[shooter] + 60, GROUND_Y - 300, { force: true });
    const kills: KillEvent[] = [];
    g.world.events.on("unitKilled", (e) => kills.push({ ...e }));
    g.hurt(target, 99, g.world.idOf(shooter));
    g.world.flush();
    expect(dead(g, target)).toBe(true);
    expect(kills).toHaveLength(1);
    expect(kills[0].player).toBe(pid);
    expect(kills[0].score).toBe(50);
    expect(g.c.player.score[shooter]).toBe(50);
    expect(g.stats.kills).toBe(1);
  });

  it("a corpse tumbles, fades and is removed; the enemy count drops when it dies", () => {
    const g = setup();
    const t = g.spawn("regime.trooper", 1200, GROUND_Y);
    expect(g.stats.enemies).toBe(1);
    g.kill(t, 0, 80, -150);
    expect(g.stats.enemies).toBe(0);
    runHeadless(g.world, 0.5);
    expect(g.world.alive(t)).toBe(true);
    expect(at(g, t).x).toBeGreaterThan(1200);
    runHeadless(g.world, 1.2);
    expect(g.world.alive(t)).toBe(false);
  });

  it("lava kills whatever walks into it", () => {
    const g = setup();
    fill(g.terrain, 150, GROUND, 6, 3, M.LAVA);
    const t = g.spawn("test.dummy", 152 * 8, GROUND_Y + 16);
    runHeadless(g.world, 0.3);
    expect(dead(g, t)).toBe(true);
  });
});
