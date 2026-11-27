import { describe, expect, it } from "vitest";
import { World } from "@/engine/core";
import { Loop, type LoopHost, type LoopOptions } from "@/engine/loop";

/** A hand-cranked browser: frames and timers fire only when the test says so. */
function fakeHost() {
  let nextId = 1;
  const frames = new Map<number, (ms: number) => void>();
  const timers = new Map<number, { cb: () => void; ms: number }>();
  const listeners: (() => void)[] = [];
  const doc = {
    hidden: false,
    addEventListener: (_: "visibilitychange", fn: () => void) => listeners.push(fn),
    removeEventListener: (_: "visibilitychange", fn: () => void) => listeners.splice(listeners.indexOf(fn), 1),
  };
  const host: LoopHost = {
    now: () => 0,
    requestFrame: (cb) => {
      frames.set(nextId, cb);
      return nextId++;
    },
    cancelFrame: (id) => frames.delete(id),
    setTimer: (cb, ms) => {
      timers.set(nextId, { cb, ms });
      return nextId++;
    },
    clearTimer: (id) => timers.delete(id),
    doc,
  };
  return {
    host,
    doc,
    frames,
    timers,
    listeners,
    /** Fire every pending animation frame at time `ms`. */
    frame(ms: number) {
      const due = [...frames.values()];
      frames.clear();
      for (const cb of due) cb(ms);
    },
    /** Fire every pending timer; returns their delays. */
    timer() {
      const due = [...timers.values()];
      timers.clear();
      for (const t of due) t.cb();
      return due.map((t) => t.ms);
    },
    setHidden(h: boolean) {
      doc.hidden = h;
      for (const l of [...listeners]) l();
    },
  };
}

function setup(opts: LoopOptions = {}) {
  const fake = fakeHost();
  const world = new World({ seed: 1, coreSystems: false });
  const renders: number[] = [];
  const loop = new Loop(world, {
    render: (alpha) => renders.push(alpha),
    ...opts,
    host: fake.host,
  });
  return { fake, world, renders, loop };
}

describe("frame loop", () => {
  it("runs on animation frames, steps fixed ticks and renders with the leftover alpha", () => {
    const { fake, world, renders, loop } = setup();
    loop.start();
    expect(fake.frames.size).toBe(1);
    fake.frame(1000);
    // The first frame only sets the clock
    expect(world.time.tick).toBe(0);
    for (let k = 1; k <= 60; k++) fake.frame(1000 + (k * 1000) / 60);
    expect(world.time.tick).toBe(30);
    expect(renders.length).toBe(61);
    expect(renders[1]).toBeCloseTo(0.5);
    expect(loop.frames).toBe(61);
    expect(world.stats.fps).toBeCloseTo(60, 0);
    expect(world.stats.render.samples).toBe(61);
    loop.stop();
    expect(fake.frames.size).toBe(0);
    expect(fake.listeners.length).toBe(0);
  });

  it("with maxFps, waits on a timer until a frame is nearly due, then asks for an animation frame", () => {
    const { fake, world, loop } = setup({ maxFps: 30 });
    loop.start();
    expect(fake.frames.size).toBe(0);
    expect(fake.timer()).toEqual([1000 / 30 - 6]);
    expect(fake.frames.size).toBe(1);
    fake.frame(0);
    for (let k = 1; k <= 30; k++) {
      expect(fake.timer().length).toBe(1);
      fake.frame((k * 1000) / 30);
    }
    expect(world.time.tick).toBe(30);
    // Switching to uncapped at runtime goes back to plain animation frames
    loop.maxFps = 0;
    fake.timer();
    fake.frame(1100);
    expect(fake.frames.size).toBe(1);
    expect(fake.timers.size).toBe(0);
    loop.stop();
    expect(fake.frames.size + fake.timers.size).toBe(0);
  });

  it("speed, slow motion and the hook scale simulation time", () => {
    let hook = 1;
    const { fake, world, loop } = setup({ slowmo: () => hook });
    loop.start();
    fake.frame(0);
    loop.speed = 2;
    for (let k = 1; k <= 30; k++) fake.frame((k * 1000) / 30);
    expect(world.time.tick).toBe(60);
    loop.speed = 1;
    world.time.slowmo = 0.5;
    for (let k = 31; k <= 60; k++) fake.frame((k * 1000) / 30);
    expect(world.time.tick).toBe(75);
    world.time.slowmo = 1;
    hook = 0.25;
    for (let k = 61; k <= 90; k++) fake.frame((k * 1000) / 30);
    expect(world.time.tick).toBeGreaterThanOrEqual(82);
    expect(world.time.tick).toBeLessThanOrEqual(83);
    loop.speed = -3;
    expect(loop.speed).toBe(0);
  });

  it("pause holds the last frame (one render), resume carries on without a jump", () => {
    const { fake, world, renders, loop } = setup();
    loop.start();
    fake.frame(0);
    fake.frame(100);
    const ticks = world.time.tick;
    loop.pause();
    expect(loop.paused).toBe(true);
    const before = renders.length;
    for (let k = 1; k <= 10; k++) fake.frame(100 + k * 16);
    expect(renders.length).toBe(before + 1);
    expect(world.time.tick).toBe(ticks);
    loop.invalidate();
    fake.frame(400);
    expect(renders.length).toBe(before + 2);
    loop.resume();
    fake.frame(400 + 1000 / 30);
    expect(world.time.tick).toBe(ticks + 1);
  });

  it("pauses while the page is hidden or the host says sleep, polling slowly, and doesn't fast-forward after", () => {
    let asleep = false;
    const { fake, world, renders, loop } = setup({ sleep: () => asleep });
    loop.start();
    fake.frame(0);
    fake.frame(100);
    const t = world.time.tick,
      r = renders.length;
    fake.setHidden(true);
    fake.frame(200);
    fake.frame(5000);
    expect(world.time.tick).toBe(t);
    expect(renders.length).toBe(r);
    fake.setHidden(false);
    fake.frame(5100);
    fake.frame(5100 + 1000 / 30);
    expect(world.time.tick).toBe(t + 1);
    asleep = true;
    fake.frame(6000);
    // Asleep: no animation frame requested, just a slow poll
    expect(fake.frames.size).toBe(0);
    expect(fake.timer()).toEqual([100]);
    expect(fake.frames.size).toBe(1);
    asleep = false;
    fake.frame(9000);
    fake.frame(9000 + 1000 / 30);
    expect(world.time.tick).toBe(t + 2);
  });

  it("clamps a long stall to maxFrameDt and the step cap", () => {
    const { fake, world, loop } = setup({ maxFrameDt: 0.1 });
    loop.start();
    fake.frame(0);
    fake.frame(10_000);
    expect(world.time.tick).toBe(3);
    expect(world.stats.ticksLastFrame).toBe(3);
  });

  it("keeps running after a render throws", () => {
    let bad = true;
    const fake = fakeHost();
    const world = new World({ seed: 1, coreSystems: false });
    const loop = new Loop(world, {
      host: fake.host,
      render: () => {
        if (bad) throw new Error("render failed");
      },
    });
    loop.start();
    expect(() => fake.frame(0)).toThrow("render failed");
    bad = false;
    expect(fake.frames.size).toBe(1);
    fake.frame(16);
    expect(loop.frames).toBe(1);
  });
});
