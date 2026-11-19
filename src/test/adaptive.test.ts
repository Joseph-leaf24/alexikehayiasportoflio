import { describe, expect, it, vi } from "vitest";
import { buildLadder, createController, createPacer, rememberedRung, rememberRung, startRung, TARGET_MS } from "@/lib/adaptive";

describe("quality ladder", () => {
  it("goes from 0.5× without effects up to the screen's own ratio (at most 2×)", () => {
    const one = buildLadder(1);
    expect(one[0]).toMatchObject({ dpr: 0.5, half: true });
    expect(one.at(-1)).toEqual({ dpr: 1, fx: true, bloom: true, snow: 1, half: false });
    expect(buildLadder(3).at(-1)!.dpr).toBe(2);
    expect(buildLadder(1.6).at(-1)!.dpr).toBe(1.6);
    expect(buildLadder(0).at(-1)!.dpr).toBe(1);
    // Cheapest first: resolution and effects never get worse going up
    const l = buildLadder(2);
    for (let i = 1; i < l.length; i++) {
      expect(l[i].dpr).toBeGreaterThanOrEqual(l[i - 1].dpr);
      expect(Number(l[i].bloom)).toBeGreaterThanOrEqual(Number(l[i - 1].bloom));
    }
  });

  it("starts each device tier at a sensible rung", () => {
    const l = buildLadder(2);
    expect(l[startRung(l, "low")]).toMatchObject({ dpr: 0.65, fx: false, bloom: false });
    expect(l[startRung(l, "mid")]).toMatchObject({ dpr: 1, bloom: true });
    expect(l[startRung(l, "high")]).toMatchObject({ dpr: 1.5, bloom: true });
    expect(buildLadder(1)[startRung(buildLadder(1), "high")].dpr).toBe(1);
  });
});

describe("display pacing", () => {
  const settle = (ms: number) => {
    const p = createPacer();
    let d = 0;
    for (let i = 0; i < 60; i++) d = p.sample(ms);
    return d;
  };
  it("renders every refresh up to 100 Hz and about 60 fps on faster screens", () => {
    expect(settle(1000 / 60)).toBe(1);
    expect(settle(1000 / 90)).toBe(1);
    expect(settle(1000 / 120)).toBe(2);
    expect(settle(1000 / 144)).toBe(2);
    expect(settle(1000 / 240)).toBe(4);
  });
  it("ignores pauses (a hidden tab) when estimating the refresh rate", () => {
    const p = createPacer();
    for (let i = 0; i < 60; i++) p.sample(i % 10 === 0 ? 5000 : 1000 / 120);
    expect(p.sample(1000 / 120)).toBe(2);
  });
});

describe("quality controller", () => {
  const ladder = buildLadder(2);
  /** Feed `count` frames of `ms` each, starting at time `t`; returns the end time. */
  const feed = (c: ReturnType<typeof createController>, ms: number, count: number, t = 0) => {
    for (let i = 0; i < count; i++) c.frame(ms, (t += ms));
    return t;
  };

  it("doesn't count one-off hitches (route changes, GC) against the device", () => {
    const onChange = vi.fn();
    const c = createController(ladder, { start: 5, warmup: 0, onChange });
    let t = 0;
    for (let i = 0; i < 400; i++) t = feed(c, i % 20 === 0 ? 400 : TARGET_MS, 1, t);
    // Only clean windows remain, so it climbs, never descends
    expect(onChange.mock.calls.every(([lvl], i, all) => i === 0 || lvl > all[i - 1][0])).toBe(true);
    expect(c.level).toBeGreaterThan(5);
  });

  it("still measures a device whose every frame is slow (not mistaking them all for hitches)", () => {
    const c = createController(ladder, { start: 5, onChange: () => {} });
    // Every frame 160–400 ms (a software renderer), cheaper on lower rungs
    let t = 0;
    for (let i = 0; i < 80; i++) {
      const ms = 100 + 60 * c.level;
      c.frame(ms, (t += ms));
    }
    expect(c.level).toBe(0);
  });

  it("waits out the start-up, and starts measuring afresh after reset()", () => {
    const onChange = vi.fn();
    const c = createController(ladder, { start: 5, onChange });
    // 1 s of terrible frames during warm-up: ignored
    let t = feed(c, 50, 20);
    expect(onChange).not.toHaveBeenCalled();
    t = feed(c, 50, 60, t);
    expect(onChange).toHaveBeenCalled();
    expect(c.level).toBeLessThan(5);
    const at = c.level;
    c.reset();
    feed(c, TARGET_MS, 5, t);
    expect(c.level).toBe(at);
    expect(c.rung).toBe(ladder[at]);
  });
});

describe("remembered rung", () => {
  it("round-trips through this session's storage and rejects nonsense", () => {
    expect(rememberedRung(10)).toBeNull();
    rememberRung(4);
    expect(rememberedRung(10)).toBe(4);
    expect(rememberedRung(3)).toBeNull();
    sessionStorage.setItem("ak:rung", "abc");
    expect(rememberedRung(10)).toBeNull();
  });

  it("survives blocked storage", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("denied");
    });
    expect(() => rememberRung(2)).not.toThrow();
    expect(rememberedRung(10)).toBeNull();
  });
});
