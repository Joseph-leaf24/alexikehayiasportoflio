import { describe, expect, it } from "vitest";
import { runHeadless } from "@/engine/headless";
import { BODY_UNINDEXED, TICK, World, enemiesOf, hashWorld, indexOf, place, stepFixed, tick } from "@/engine/core";

/**
 * A small battle that leans on everything determinism depends on: seeded streams, spatial queries, events,
 * spawning, deferred destruction with index reuse, budgeted systems and lifetimes.
 */
function skirmish(seed: number) {
  const w = new World({ seed, capacity: 64 });
  const { pos, vel, body, team, health, life } = w.core;
  const target = w.defineComponent("target", { id: "u32", cd: "f32" });
  const out: number[] = [];
  let kills = 0;
  w.events.on("destroyed", () => kills++);

  w.addSystem({
    name: "spawn",
    phase: "spawn",
    run(world) {
      const r = world.rng.get("spawn");
      while (health.$.size < 120) {
        const e = world.spawn(vel, body, team, health, target);
        const t = 1 + r.int(4);
        place(world, e, t * 900 + r.range(-200, 200), r.range(300, 900));
        const i = indexOf(e);
        team.id[i] = t;
        body.hw[i] = 4;
        body.h[i] = 12;
        health.hp[i] = health.max[i] = r.range(20, 40);
        target.cd[i] = r.range(0, 1);
      }
    },
  });
  w.addSystem({
    name: "ai",
    phase: "ai",
    every: 3,
    run(world) {
      const q = world.query(target, team);
      const n = q.collect(true),
        idx = q.idx;
      const r = world.rng.get("ai");
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const foe = world.spatial.nearest(pos.x[i], pos.y[i], 700, enemiesOf(team.id[i]), null, world.idOf(i));
        target.id[i] = foe;
        const fi = indexOf(foe);
        const dx = foe ? pos.x[fi] - pos.x[i] : r.range(-1, 1);
        const dy = foe ? pos.y[fi] - pos.y[i] : r.range(-1, 1);
        const l = Math.sqrt(dx * dx + dy * dy) || 1;
        vel.vx[i] = (dx / l) * 40 + r.range(-10, 10);
        vel.vy[i] = (dy / l) * 40 + r.range(-10, 10);
      }
    },
  });
  w.addSystem({
    name: "weapons",
    phase: "weapons",
    run(world, dt) {
      const q = world.query(target);
      const n = q.collect(true),
        idx = q.idx;
      const r = world.rng.get("weapons");
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        target.cd[i] -= dt;
        const foe = target.id[i];
        if (target.cd[i] > 0 || !world.alive(foe)) continue;
        target.cd[i] = r.range(0.4, 0.9);
        const fi = indexOf(foe);
        const hit = world.spatial.segmentFirst(
          pos.x[i],
          pos.y[i] - 6,
          pos.x[fi],
          pos.y[fi] - 6,
          2,
          enemiesOf(team.id[i])
        );
        if (hit) health.hp[indexOf(hit)] -= r.range(4, 12);
        // A tracer that only lives a few ticks
        const fx = world.spawn(life, body);
        place(world, fx, pos.x[i], pos.y[i]);
        body.flags[indexOf(fx)] = BODY_UNINDEXED;
        life.t[indexOf(fx)] = 0.1;
      }
    },
  });
  w.addSystem({
    name: "motion",
    phase: "motion",
    run(_world, dt) {
      const q = w.query(pos, vel);
      const n = q.collect(),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        pos.x[i] += vel.vx[i] * dt;
        pos.y[i] += vel.vy[i] * dt;
      }
    },
  });
  w.addSystem({
    name: "death",
    phase: "death",
    run(world) {
      const q = world.query(health);
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (health.hp[i] > 0) continue;
        // Splash: everyone near the body takes a little
        const c = world.spatial.inRadius(pos.x[i], pos.y[i], 30, -1, out);
        for (let j = 0; j < c; j++) if (world.has(out[j], health)) health.hp[indexOf(out[j])] -= 1;
        world.destroy(world.idOf(i));
      }
    },
  });
  return { w, kills: () => kills };
}

describe("determinism", () => {
  it("the same seed gives the same world, tick for tick", () => {
    const a = skirmish(2024),
      b = skirmish(2024);
    for (let k = 0; k < 600; k++) {
      tick(a.w);
      tick(b.w);
      if (k % 50 === 0) expect(hashWorld(a.w)).toBe(hashWorld(b.w));
    }
    expect(hashWorld(a.w)).toBe(hashWorld(b.w));
    expect(a.kills()).toBe(b.kills());
    // It's a real fight: units died, slots were recycled, the world grew past its first capacity
    expect(a.kills()).toBeGreaterThan(100);
    expect(a.w.capacity).toBeGreaterThan(64);
  });

  it("a different seed gives a different world", () => {
    const a = skirmish(1),
      b = skirmish(2);
    runHeadless(a.w, 5);
    runHeadless(b.w, 5);
    expect(hashWorld(a.w)).not.toBe(hashWorld(b.w));
  });

  it("frame pacing doesn't matter: ragged frames under stepFixed match headless ticks", () => {
    const paced = skirmish(77),
      head = skirmish(77);
    const jitter = paced.w.rng.get("test-frames");
    let ticks = 0;
    while (ticks < 300) ticks += stepFixed(paced.w, Math.min(jitter.range(0.002, 0.09), (300 - ticks) * TICK));
    // Cosmetic draws from another stream (as a renderer would) must not disturb the simulation either
    for (let k = 0; k < 500; k++) paced.w.rng.get("fx").next();
    runHeadless(head.w, 300 * TICK);
    expect(paced.w.time.tick).toBe(300);
    expect(head.w.time.tick).toBe(300);
    expect(hashWorld(paced.w, { rng: false })).toBe(hashWorld(head.w, { rng: false }));
    for (const name of ["sim", "spawn", "ai", "weapons"])
      expect(paced.w.rng.get(name).hash()).toBe(head.w.rng.get(name).hash());
  });

  it("hashWorld notices any change to a column, membership or the rng", () => {
    const { w } = skirmish(5);
    runHeadless(w, 1);
    const h0 = hashWorld(w);
    expect(hashWorld(w)).toBe(h0);
    const i = w.core.health.$.dense[0];
    w.core.health.hp[i] += 0.5;
    const h1 = hashWorld(w);
    expect(h1).not.toBe(h0);
    w.core.health.hp[i] -= 0.5;
    expect(hashWorld(w)).toBe(h0);
    w.add(w.idOf(i), w.core.sprite);
    expect(hashWorld(w)).not.toBe(h0);
    w.remove(w.idOf(i), w.core.sprite);
    expect(hashWorld(w)).toBe(h0);
    w.random.next();
    expect(hashWorld(w)).not.toBe(h0);
  });
});
