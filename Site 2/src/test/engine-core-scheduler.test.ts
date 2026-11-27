import { describe, expect, it } from "vitest";
import {
  MAX_TICKS_PER_STEP,
  PHASES,
  TICK,
  World,
  indexOf,
  place,
  stepFixed,
  tick,
  type Phase,
  type System,
} from "@/engine/core";

declare module "@/engine/core/events" {
  interface WorldEvents {
    "test:ping": string;
  }
}

const bare = () => new World({ seed: 1, coreSystems: false });
const sys = (name: string, phase: Phase, run: System["run"], extra: Partial<System> = {}): System => ({
  name,
  phase,
  run,
  ...extra,
});

describe("scheduler", () => {
  it("runs systems in phase order, then by order key, then by insertion", () => {
    const w = bare();
    const log: string[] = [];
    const reversed = [...PHASES].reverse();
    for (const p of reversed) w.addSystem(sys(`s.${p}`, p, () => log.push(p)));
    w.addSystem(sys("ai.late", "ai", () => log.push("ai.late"), { order: 5 }));
    w.addSystem(sys("ai.early", "ai", () => log.push("ai.early"), { order: -5 }));
    w.addSystem(sys("ai.second", "ai", () => log.push("ai.second")));
    tick(w);
    const expected: string[] = [];
    for (const p of PHASES) {
      if (p === "ai") expected.push("ai.early", "ai", "ai.second", "ai.late");
      else expected.push(p);
    }
    expect(log).toEqual(expected);
    expect(w.systems.order[0]).toBe("s.input");
  });

  it("rejects duplicates and unknown phases; systems can be disabled, re-enabled and removed", () => {
    const w = bare();
    let runs = 0;
    w.addSystem(sys("a", "ai", () => runs++));
    expect(() => w.addSystem(sys("a", "ai", () => {}))).toThrow(/already/);
    expect(() => w.addSystem(sys("b", "nope" as Phase, () => {}))).toThrow(/unknown phase/);
    tick(w);
    expect(w.systems.setEnabled("a", false)).toBe(true);
    expect(w.systems.isEnabled("a")).toBe(false);
    tick(w);
    w.systems.setEnabled("a", true);
    tick(w);
    expect(runs).toBe(2);
    expect(w.systems.remove("a")).toBe(true);
    expect(w.systems.remove("a")).toBe(false);
    expect(w.systems.setEnabled("a", true)).toBe(false);
    tick(w);
    expect(runs).toBe(2);
  });

  it("runs budgeted systems every n ticks on their offset, with dt scaled to match", () => {
    const w = bare();
    const ticks: number[] = [];
    const dts: number[] = [];
    w.addSystem(
      sys(
        "think",
        "ai",
        (world, dt) => {
          ticks.push(world.time.tick);
          dts.push(dt);
        },
        { every: 3, offset: 1 }
      )
    );
    for (let k = 0; k < 10; k++) tick(w);
    expect(ticks).toEqual([1, 4, 7]);
    expect(dts[0]).toBeCloseTo(3 * TICK);
  });

  it("copies pos into prev before each tick", () => {
    const w = bare();
    const { pos } = w.core;
    const e = w.create();
    place(w, e, 10, 20);
    w.addSystem(sys("move", "motion", () => (pos.x[indexOf(e)] += 5)));
    tick(w);
    expect(w.core.prev.x[indexOf(e)]).toBe(10);
    expect(pos.x[indexOf(e)]).toBe(15);
    tick(w);
    expect(w.core.prev.x[indexOf(e)]).toBe(15);
    expect(pos.x[indexOf(e)]).toBe(20);
  });

  it("flushes events between phases and destroys at the end of the tick", () => {
    const w = bare();
    const log: string[] = [];
    let victim = 0;
    w.events.on("test:ping", (msg) => log.push(`got ${msg}`));
    w.events.on("destroyed", (id) => log.push(id === victim ? "destroyed victim" : "destroyed ?"));
    w.addSystem(
      sys("spawn", "spawn", (world) => {
        log.push("spawn");
        world.events.emit("test:ping", "spawned");
        if (!victim) victim = world.create();
      })
    );
    w.addSystem(
      sys("ai", "ai", (world) => {
        log.push("ai");
        world.destroy(victim);
      })
    );
    w.addSystem(sys("late", "late", (world) => log.push(`late alive=${world.alive(victim)}`)));
    tick(w);
    expect(log).toEqual(["spawn", "got spawned", "ai", "late alive=true", "destroyed victim"]);
    expect(w.alive(victim)).toBe(false);
    expect(w.stats.events).toBe(2);
  });

  it("keeps time in whole ticks and guards against re-entrant ticks", () => {
    const w = bare();
    w.addSystem(sys("again", "late", (world) => (world.time.tick === 2 ? tick(world) : undefined)));
    tick(w);
    tick(w);
    expect(w.time.tick).toBe(2);
    expect(w.time.now).toBe(2 * TICK);
    expect(() => tick(w)).toThrow(/inside a tick/);
    // The scheduler recovers after the throw
    w.systems.remove("again");
    tick(w);
    expect(w.time.tick).toBe(3);
  });

  it("reports system errors to world.onError and carries on, or propagates without a handler", () => {
    const w = bare();
    let after = 0;
    w.addSystem(
      sys("bad", "ai", () => {
        throw new Error("bad system");
      })
    );
    w.addSystem(sys("after", "late", () => after++));
    expect(() => tick(w)).toThrow("bad system");
    const errors: string[] = [];
    w.onError = (err, where) => errors.push(`${where}: ${(err as Error).message}`);
    tick(w);
    tick(w);
    expect(errors).toEqual(["bad: bad system", "bad: bad system"]);
    expect(after).toBe(2);
    expect(w.stats.system("bad")?.errors).toBe(3);
    // Listener errors take the same route
    w.events.on("test:ping", () => {
      throw new Error("bad listener");
    });
    w.events.emit("test:ping", "x");
    w.flush();
    expect(errors[2]).toBe("event:test:ping: bad listener");
    // And so do destroy hooks, without leaving the entity half-destroyed
    w.onDestroy(() => {
      throw new Error("bad hook");
    });
    const e = w.create();
    w.destroy(e);
    tick(w);
    expect(errors[errors.length - 1]).toBe("destroy: bad hook");
    expect(w.alive(e)).toBe(false);
  });

  it("keeps per-system timings, entity counts per component and tick stats", () => {
    const w = new World({ seed: 2 });
    w.addSystem(sys("busy", "ai", () => {}));
    for (let k = 0; k < 5; k++) w.spawn(w.core.pos, w.core.vel);
    for (let k = 0; k < 70; k++) tick(w);
    const names = w.stats.systems.map((s) => s.name);
    expect(names).toEqual(["busy", "core.spatial", "core.life"]);
    const busy = w.stats.system("busy")!;
    expect(busy.samples).toBe(70);
    expect(busy.ms).toBeGreaterThanOrEqual(0);
    expect(busy.max).toBeGreaterThanOrEqual(busy.last);
    expect(w.stats.ticks).toBe(70);
    expect(w.stats.entities).toBe(5);
    expect(w.stats.components.find((c) => c.name === "vel")?.count).toBe(5);
    expect(w.stats.components.find((c) => c.name === "life")?.count).toBe(0);
    expect(w.stats.tick.samples).toBe(70);
    w.stats.timing = false;
    tick(w);
    expect(busy.samples).toBe(70);
  });
});

describe("fixed step", () => {
  it("turns frame time into whole ticks and keeps the remainder as alpha", () => {
    const w = bare();
    let n = 0;
    for (let f = 0; f < 60; f++) n += stepFixed(w, 1 / 60);
    expect(n).toBe(30);
    expect(w.time.tick).toBe(30);
    expect(stepFixed(w, TICK * 0.25)).toBe(0);
    expect(w.time.alpha).toBeCloseTo(0.25);
    expect(stepFixed(w, TICK * 0.75)).toBe(1);
    expect(w.time.alpha).toBeCloseTo(0, 5);
    // At 144 Hz ticks land on the right frames over a second
    let m = 0;
    for (let f = 0; f < 144; f++) m += stepFixed(w, 1 / 144);
    expect(m).toBe(30);
  });

  it("caps the ticks per call and drops the backlog instead of spiralling", () => {
    const w = bare();
    expect(stepFixed(w, 5)).toBe(MAX_TICKS_PER_STEP);
    expect(w.stats.droppedTicks).toBe(150 - MAX_TICKS_PER_STEP);
    expect(w.time.acc).toBeLessThan(TICK);
    expect(stepFixed(w, TICK)).toBe(1);
  });

  it("ignores zero, negative and non-finite frame times", () => {
    const w = bare();
    expect(stepFixed(w, 0)).toBe(0);
    expect(stepFixed(w, -1)).toBe(0);
    expect(stepFixed(w, NaN)).toBe(0);
    expect(stepFixed(w, Infinity)).toBe(0);
    expect(w.time.acc).toBe(0);
  });

  it("time scale is speed × slow motion, 0 when paused", () => {
    const w = bare();
    w.time.speed = 2;
    w.time.slowmo = 0.25;
    expect(w.time.scale).toBe(0.5);
    w.time.paused = true;
    expect(w.time.scale).toBe(0);
  });
});

describe("core systems", () => {
  it("life counts down and destroys at 0", () => {
    const w = new World({ seed: 1 });
    const { life } = w.core;
    const e = w.spawn(life);
    life.t[indexOf(e)] = 0.1;
    tick(w);
    tick(w);
    expect(w.alive(e)).toBe(true);
    tick(w);
    expect(w.alive(e)).toBe(false);
  });

  it("the optional motion system integrates free bodies only", () => {
    const w = new World({ seed: 1, coreSystems: { motion: true, spatial: false } });
    const { vel, body, pos } = w.core;
    const free = w.spawn(vel);
    place(w, free, 0, 0);
    vel.vx[indexOf(free)] = 30;
    const walker = w.spawn(vel, body);
    place(w, walker, 0, 0);
    vel.vx[indexOf(walker)] = 30;
    body.flags[indexOf(walker)] = 1 << 4;
    for (let k = 0; k < 30; k++) tick(w);
    expect(pos.x[indexOf(free)]).toBeCloseTo(30, 3);
    expect(pos.x[indexOf(walker)]).toBe(0);
    expect(w.systems.order).toEqual(["core.motion", "core.life"]);
  });
});
