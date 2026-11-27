import { describe, expect, it } from "vitest";
import { runHeadless } from "@/engine/headless";
import {
  Registry,
  TICK,
  World,
  angleDiff,
  angleTo,
  approach,
  approachAngle,
  clamp,
  clamp01,
  damp,
  dampAngle,
  dist,
  dist2,
  indexOf,
  invLerp,
  lerp,
  len,
  lerpAngle,
  mod,
  nextPow2,
  place,
  remap,
  resource,
  segmentDist2,
  shiftOrigin,
  sign,
  smoothstep,
  v2add,
  v2addScaled,
  v2copy,
  v2cross,
  v2dist,
  v2dot,
  v2fromAngle,
  v2len,
  v2lerp,
  v2limit,
  v2normalize,
  v2rotate,
  v2scale,
  v2set,
  v2sub,
  vec2,
  wrapAngle,
} from "@/engine/core";

describe("registry", () => {
  interface UnitDef {
    hp: number;
  }

  it("indexes defs from 1 so a zero kind means none", () => {
    const reg = new Registry<UnitDef>("units");
    const soldier = reg.register("soldier", { hp: 10 });
    const tank = reg.register("tank", { hp: 200 });
    expect(soldier).toBe(1);
    expect(tank).toBe(2);
    expect(reg.size).toBe(2);
    expect(reg.get(tank).hp).toBe(200);
    expect(reg.byId("soldier").hp).toBe(10);
    expect(reg.index("tank")).toBe(2);
    expect(reg.index("nope")).toBe(0);
    expect(reg.id(1)).toBe("soldier");
    expect(reg.has("tank")).toBe(true);
    expect(reg.tryGet(0)).toBeUndefined();
    expect(() => reg.get(0)).toThrow(RangeError);
    expect(() => reg.get(3)).toThrow(RangeError);
    expect(() => reg.id(0)).toThrow(RangeError);
    expect(() => reg.byId("nope")).toThrow(/unknown/);
    expect(() => reg.register("tank", { hp: 1 })).toThrow(/already/);
    expect([...reg].map((d) => d.hp)).toEqual([10, 200]);
    expect(reg.ids).toEqual(["soldier", "tank"]);
  });

  it("freezes into a read-only view after boot", () => {
    const reg = new Registry<UnitDef>("units");
    reg.register("a", { hp: 1 });
    const view = reg.freeze();
    expect(view.frozen).toBe(true);
    expect(() => reg.register("b", { hp: 2 })).toThrow(/frozen/);
    expect(Object.isFrozen(view.get(1))).toBe(true);
    expect(() => {
      (view.get(1) as { hp: number }).hp = 5;
    }).toThrow();
    expect(reg.freeze()).toBe(view);
  });
});

describe("math", () => {
  it("scalars", () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
    expect(lerp(2, 4, 0.5)).toBe(3);
    expect(invLerp(2, 4, 3)).toBe(0.5);
    expect(invLerp(2, 2, 3)).toBe(0);
    expect(remap(5, 0, 10, 100, 200)).toBe(150);
    expect(smoothstep(0, 1, 0.5)).toBe(0.5);
    expect(smoothstep(0, 1, 2)).toBe(1);
    expect(mod(-1, 5)).toBe(4);
    expect(approach(0, 10, 3)).toBe(3);
    expect(approach(9, 10, 3)).toBe(10);
    expect(approach(10, 0, 4)).toBe(6);
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(5)).toBe(8);
    expect(nextPow2(1024)).toBe(1024);
    expect(nextPow2(1025)).toBe(2048);
  });

  it("damp is frame-rate independent", () => {
    let a = 0,
      b = 0;
    for (let k = 0; k < 60; k++) a = damp(a, 100, 3, 1 / 60);
    for (let k = 0; k < 30; k++) b = damp(b, 100, 3, 1 / 30);
    expect(a).toBeCloseTo(b, 6);
    expect(a).toBeCloseTo(100 * (1 - Math.exp(-3)), 6);
  });

  it("angles wrap and take the short way round", () => {
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(-Math.PI);
    expect(wrapAngle(-Math.PI / 2)).toBeCloseTo(-Math.PI / 2);
    expect(angleDiff(0.1, -0.1)).toBeCloseTo(-0.2);
    expect(angleDiff(Math.PI - 0.1, -Math.PI + 0.1)).toBeCloseTo(0.2);
    expect(lerpAngle(Math.PI - 0.1, -Math.PI + 0.1, 0.5)).toBeCloseTo(Math.PI);
    expect(approachAngle(0, 1, 0.25)).toBeCloseTo(0.25);
    expect(approachAngle(0, 0.1, 0.25)).toBe(0.1);
  });

  it("distances, signs and the remaining vector helpers", () => {
    expect(clamp01(1.5)).toBe(1);
    expect(clamp01(-0.5)).toBe(0);
    expect(sign(-3)).toBe(-1);
    expect(sign(0)).toBe(0);
    expect(len(3, 4)).toBe(5);
    expect(dist2(1, 1, 4, 5)).toBe(25);
    expect(dist(1, 1, 4, 5)).toBe(5);
    expect(angleTo(0, 0, 0, 2)).toBeCloseTo(Math.PI / 2);
    expect(dampAngle(Math.PI - 0.1, -Math.PI + 0.1, 1000, 1)).toBeCloseTo(-Math.PI + 0.1 + 2 * Math.PI, 5);
    const a = vec2(1, 2),
      b = vec2(3, 5),
      o = vec2();
    expect(v2set(o, 7, 8)).toEqual({ x: 7, y: 8 });
    expect(v2copy(o, a)).toEqual({ x: 1, y: 2 });
    expect(v2add(o, a, b)).toEqual({ x: 4, y: 7 });
    expect(v2sub(o, b, a)).toEqual({ x: 2, y: 3 });
    expect(v2scale(o, a, 3)).toEqual({ x: 3, y: 6 });
    expect(v2addScaled(o, a, b, 2)).toEqual({ x: 7, y: 12 });
    expect(v2lerp(o, a, b, 0.5)).toEqual({ x: 2, y: 3.5 });
    expect(v2dot(a, b)).toBe(13);
    expect(v2cross(a, b)).toBe(-1);
    expect(v2len(vec2(6, 8))).toBe(10);
    expect(v2dist(a, b)).toBeCloseTo(Math.sqrt(13));
    expect(v2limit(o, vec2(1, 1), 5)).toEqual({ x: 1, y: 1 });
  });

  it("vectors write into out without allocating", () => {
    const v = vec2(3, 4),
      out = vec2();
    expect(v2normalize(out, v)).toBe(5);
    expect(out).toEqual({ x: 0.6, y: 0.8 });
    expect(v2normalize(out, vec2())).toBe(0);
    expect(out).toEqual({ x: 0, y: 0 });
    v2rotate(out, vec2(1, 0), Math.PI / 2);
    expect(out.x).toBeCloseTo(0);
    expect(out.y).toBeCloseTo(1);
    v2rotate(v, v, Math.PI);
    expect(v.x).toBeCloseTo(-3);
    expect(v.y).toBeCloseTo(-4);
    v2limit(out, vec2(30, 40), 5);
    expect(out.x).toBeCloseTo(3);
    v2fromAngle(out, 0, 2);
    expect(out).toEqual({ x: 2, y: 0 });
    expect(segmentDist2(5, 3, 0, 0, 10, 0)).toBe(9);
    expect(segmentDist2(-3, 4, 0, 0, 10, 0)).toBe(25);
  });
});

describe("helpers", () => {
  it("place sets pos and prev together; shiftOrigin moves both", () => {
    const w = new World({ coreSystems: false });
    const e = w.create();
    expect(place(w, e, 5, 6)).toBe(true);
    const i = indexOf(e);
    expect([w.core.pos.x[i], w.core.prev.x[i], w.core.pos.y[i], w.core.prev.y[i]]).toEqual([5, 5, 6, 6]);
    shiftOrigin(w, -4096, 10);
    expect([w.core.pos.x[i], w.core.prev.x[i], w.core.pos.y[i]]).toEqual([5 - 4096, 5 - 4096, 16]);
    w.destroy(e);
    w.flush();
    expect(place(w, e, 0, 0)).toBe(false);
  });

  it("typed resource keys", () => {
    const CAMERA = resource<{ x: number }>("camera");
    const w = new World();
    w.setRes(CAMERA, { x: 3 });
    expect(w.res(CAMERA).x).toBe(3);
    expect(w.res<{ x: number }>("camera").x).toBe(3);
  });
});

describe("headless", () => {
  it("runs whole ticks for the given sim time and reports timings", () => {
    const w = new World({ seed: 1 });
    let calls = 0;
    w.addSystem({ name: "count", phase: "late", run: () => calls++ });
    w.time.paused = true;
    const rep = runHeadless(w, 2);
    expect(rep.ticks).toBe(60);
    expect(calls).toBe(60);
    expect(rep.simSeconds).toBeCloseTo(2);
    expect(rep.msPerTick).toBeGreaterThanOrEqual(0);
    expect(rep.maxTickMs).toBeGreaterThanOrEqual(rep.p95TickMs);
    expect(rep.stoppedEarly).toBe(false);
    expect(w.time.now).toBeCloseTo(60 * TICK);
  });

  it("stops early when onTick says so", () => {
    const w = new World({ seed: 1 });
    const rep = runHeadless(w, 10, { onTick: (_w, t) => t < 5 });
    expect(rep.ticks).toBe(5);
    expect(rep.stoppedEarly).toBe(true);
    expect(runHeadless(w, 0).ticks).toBe(0);
  });
});
