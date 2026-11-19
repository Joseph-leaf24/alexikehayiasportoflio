import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { sceneState, ui } from "@/lib/store";
import { resetQualityCache } from "@/lib/device";
import { resetState, useAnimationTimers, advance } from "./utils";
import { media, setScrollY } from "./setup";

/* GPU-facing classes are stubbed; geometry, materials and maths are real three.js */
const gl = vi.hoisted(() => ({ fail: false, renders: 0, dpr: 1, passes: [] as { enabled?: boolean }[] }));

vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof import("three")>();
  class FakeRenderer {
    outputColorSpace = "";
    toneMapping = 0;
    constructor() {
      if (gl.fail) throw new Error("WebGL unavailable");
    }
    setPixelRatio(d: number) {
      gl.dpr = d;
    }
    setSize() {}
    setClearColor() {}
    compile() {}
    dispose() {}
  }
  return { ...actual, WebGLRenderer: FakeRenderer };
});
vi.mock("three/examples/jsm/postprocessing/EffectComposer.js", () => ({
  EffectComposer: class {
    addPass(p: { enabled?: boolean }) {
      gl.passes.push(p);
    }
    setPixelRatio() {}
    setSize() {}
    render() {
      gl.renders++;
    }
    dispose() {}
  },
}));
vi.mock("three/examples/jsm/postprocessing/RenderPass.js", () => ({ RenderPass: class {} }));
vi.mock("three/examples/jsm/postprocessing/UnrealBloomPass.js", () => ({
  UnrealBloomPass: class {
    enabled = true;
    setSize() {}
    dispose() {}
  },
}));
vi.mock("three/examples/jsm/postprocessing/ShaderPass.js", () => ({
  ShaderPass: class {
    enabled = true;
    uniforms = { uTime: { value: 0 }, uAberr: { value: 0 }, uGlitch: { value: 0 }, uRes: { value: [1, 1] } };
  },
}));

const { default: IceScene } = await import("@/components/gl/IceScene");

beforeEach(() => {
  resetState();
  gl.fail = false;
  gl.renders = 0;
  gl.passes = [];
});
afterEach(() => {
  document.documentElement.classList.remove("no-webgl");
  delete document.documentElement.dataset.perf;
  Object.defineProperty(window, "devicePixelRatio", { configurable: true, value: 1 });
  window.history.replaceState({}, "", "/");
});

type Settings = { dpr: number; bloom: boolean; fx: boolean };

/**
 * Runs the scene on a simulated display: rAF fires once per refresh, and a rendered frame
 * whose GPU cost (ms, from the settings in use) is longer than a refresh pushes the next
 * callback back to a later refresh, the way the browser waits for the GPU.
 */
function display(refreshMs: number, cost: (s: Settings) => number) {
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => now);
  const frames: FrameRequestCallback[] = [];
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
    frames.push(cb);
    return frames.length;
  });
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
  const settings = (): Settings => {
    const [bloom, fx] = gl.passes.filter((p) => "enabled" in p);
    return { dpr: gl.dpr, bloom: !!bloom?.enabled, fx: !!fx?.enabled };
  };
  return {
    settings,
    /** Simulated seconds */
    run(seconds: number) {
      const end = now + seconds * 1000;
      while (now < end && frames.length) {
        const before = gl.renders;
        frames.shift()!(now);
        const busy = gl.renders > before ? Math.max(refreshMs, cost(settings())) : refreshMs;
        now += Math.ceil(busy / refreshMs - 1e-9) * refreshMs;
      }
    },
  };
}

const setScreenDpr = (value: number) => Object.defineProperty(window, "devicePixelRatio", { configurable: true, value });

const stops = [
  { y: 0, layout: "village" as const },
  { y: 1000, layout: "shatter" as const },
  { y: 2000, layout: "clusters" as const },
  { y: 3000, layout: "network" as const },
  { y: 4000, layout: "bars" as const },
  { y: 5000, layout: "villageTop" as const },
];

describe("IceScene", () => {
  it("builds the whole cube world and renders through every formation", () => {
    window.history.replaceState({}, "", "/?quality=high");
    useAnimationTimers();
    const { container, unmount } = render(<IceScene />);
    const canvas = container.querySelector("canvas")!;
    expect(canvas).toHaveClass("gl-canvas");

    advance(40);
    expect(sceneState.ready).toBe(true);
    expect(sceneState.blocks).toBeGreaterThan(10000);
    expect(gl.renders).toBeGreaterThan(0);

    // Loader lifts → village drops in
    act(() => ui.set({ ready: true }));
    advance(500);

    // Scroll through every section, including mid-transition frames
    sceneState.stops = stops;
    for (const y of [0, 300, 700, 1000, 1500, 2000, 2400, 2800, 3000, 3500, 4000, 4500, 5000, 6000]) {
      // hovering a domain card while the clusters are on screen lights one up
      sceneState.highlight = y >= 1500 && y <= 2400 ? 2 : -1;
      setScrollY(y);
      advance(120);
    }
    sceneState.highlight = -1;
    expect(["villageTop", "bars"]).toContain(sceneState.formation);

    // Pointer, click burst, hold-to-shatter and a fast scroll
    sceneState.pointer.x = 0.2;
    sceneState.pointer.y = -0.1;
    sceneState.pointer.lastMove = performance.now();
    sceneState.burst++;
    sceneState.press = true;
    sceneState.velocity = 6000;
    setScrollY(0);
    advance(400);
    sceneState.press = false;
    sceneState.pointer.x = 0.4;
    sceneState.pointer.lastMove = performance.now();
    advance(400);

    // Detail page + 404 overrides
    sceneState.override = "cube";
    advance(200);
    sceneState.override = "lost";
    advance(200);

    // Window + context events
    window.dispatchEvent(new Event("resize"));
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
    const before = gl.renders;
    advance(200);
    expect(gl.renders).toBe(before);
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
    document.dispatchEvent(new Event("visibilitychange"));
    canvas.dispatchEvent(new Event("webglcontextlost"));
    canvas.dispatchEvent(new Event("webglcontextrestored"));
    advance(100);
    expect(gl.renders).toBeGreaterThan(before);
    unmount();
  });

  describe("live quality (a steady 60 fps at the best settings the device holds)", () => {
    it("climbs to the richest settings on a fast device, and remembers a bigger budget", () => {
      window.history.replaceState({}, "", "/?quality=mid");
      resetQualityCache();
      setScreenDpr(2);
      const d = display(1000 / 60, () => 4);
      const { unmount } = render(<IceScene />);
      // Starts at 1× with bloom (the mid tier's rung)…
      d.run(1);
      expect(d.settings()).toEqual({ dpr: 1, bloom: true, fx: true });
      // …and climbs to the screen's own 2× within ~15 s, then flags the tier as too small
      d.run(40);
      expect(d.settings()).toEqual({ dpr: 2, bloom: true, fx: true });
      expect(sceneState.res).toBe(2);
      expect(Math.round(sceneState.fps)).toBe(60);
      expect(sessionStorage.getItem("ak:tier")).toBe("high");
      unmount();
    });

    it("steps down to the best settings a slower GPU holds at 60 fps, and stops re-trying richer ones", () => {
      window.history.replaceState({}, "", "/?quality=high");
      resetQualityCache();
      setScreenDpr(1.5);
      // 1× with grain fits a 60 fps frame (16 ms); adding bloom doesn't (20 ms)
      const d = display(1000 / 60, ({ dpr, bloom, fx }) => 3 + 12 * dpr * dpr + (bloom ? 4 : 0) + (fx ? 1 : 0));
      const { unmount } = render(<IceScene />);
      d.run(50);
      expect(d.settings()).toEqual({ dpr: 1, bloom: false, fx: true });
      // Three failed tries at bloom and it stops trying: the settings hold from here on
      const rung = sessionStorage.getItem("ak:rung");
      const renders = gl.renders;
      d.run(30);
      expect(sessionStorage.getItem("ak:rung")).toBe(rung);
      expect(gl.renders - renders).toBeGreaterThan(30 * 58);
      unmount();
    });

    it("drops to 30 fps as a last resort, then remembers a smaller budget", () => {
      window.history.replaceState({}, "", "/?quality=mid");
      resetQualityCache();
      const d = display(1000 / 60, ({ dpr, bloom }) => 30 + 40 * dpr * dpr + (bloom ? 10 : 0));
      const { unmount } = render(<IceScene />);
      d.run(20);
      expect(d.settings()).toEqual({ dpr: 0.5, bloom: false, fx: false });
      expect(document.documentElement.dataset.perf).toBe("degraded");
      expect(sessionStorage.getItem("ak:tier")).toBe("low");
      unmount();
    });

    it("keeps its quality when lowering it doesn't speed frames up (a CPU limit, not the GPU)", () => {
      window.history.replaceState({}, "", "/?quality=high");
      resetQualityCache();
      const d = display(1000 / 60, () => 30);
      const { unmount } = render(<IceScene />);
      d.run(10);
      expect(d.settings()).toEqual({ dpr: 1, bloom: true, fx: true });
      // It tries once more later, finds the same, and then leaves the settings alone for good
      d.run(45);
      expect(d.settings()).toEqual({ dpr: 1, bloom: true, fx: true });
      expect(document.documentElement.dataset.perf).toBeUndefined();
      unmount();
    });

    it("renders every other refresh on a 120 Hz screen (≈60 fps), starting low-end devices without effects", () => {
      window.history.replaceState({}, "", "/?quality=low");
      resetQualityCache();
      const d = display(1000 / 120, () => 2);
      const { unmount } = render(<IceScene />);
      d.run(1);
      expect(d.settings()).toEqual({ dpr: 0.65, bloom: false, fx: false });
      const before = gl.renders;
      d.run(3);
      // 3 s of a 120 Hz screen is 360 refreshes; ~180 of them are drawn
      expect(gl.renders - before).toBeGreaterThan(170);
      expect(gl.renders - before).toBeLessThan(190);
      unmount();
    });

    it("starts a reload where this session's last visit settled", () => {
      resetQualityCache();
      sessionStorage.setItem("ak:rung", "3");
      const d = display(1000 / 60, () => 4);
      const { unmount } = render(<IceScene />);
      d.run(0.2);
      expect(d.settings()).toEqual({ dpr: 0.8, bloom: false, fx: true });
      unmount();
    });
  });

  it("stops drawing during page transitions when frames are already slow", () => {
    window.history.replaceState({}, "", "/?quality=low");
    resetQualityCache();
    const d = display(1000 / 60, () => 120);
    const { unmount } = render(<IceScene />);
    d.run(3);
    sceneState.transition = true;
    const before = gl.renders;
    d.run(2);
    expect(gl.renders).toBe(before);
    sceneState.transition = false;
    d.run(1);
    expect(gl.renders).toBeGreaterThan(before);
    unmount();
  });

  it("lights the scene for the real time in Amsterdam, re-reads the clock, and obeys ?time=", () => {
    window.history.replaceState({}, "", "/");
    useAnimationTimers();
    vi.setSystemTime(new Date("2026-06-21T11:00:00Z")); // 13:00 CEST
    const first = render(<IceScene />);
    advance(40);
    expect(document.documentElement.dataset.sky).toBe("day");
    // An evening later the next refresh (every 10 s) finds the sun down
    vi.setSystemTime(new Date("2026-06-21T21:30:00Z")); // 23:30 CEST
    advance(10_200);
    expect(document.documentElement.dataset.sky).toBe("night");
    first.unmount();
    expect(document.documentElement.dataset.sky).toBeUndefined();

    // ?time= pins the wall-clock time whatever the real one is
    vi.setSystemTime(new Date("2026-06-21T11:00:00Z"));
    window.history.replaceState({}, "", "/?time=23:00");
    const pinned = render(<IceScene />);
    advance(40);
    expect(document.documentElement.dataset.sky).toBe("night");
    pinned.unmount();
    window.history.replaceState({}, "", "/");
  });

  it("falls back to a CSS background when WebGL is unavailable", () => {
    gl.fail = true;
    render(<IceScene />);
    expect(sceneState.webgl).toBe(false);
    expect(sceneState.ready).toBe(true);
    expect(document.documentElement).toHaveClass("no-webgl");
  });
});
