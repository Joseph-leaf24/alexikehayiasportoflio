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
    uniforms = {
      uTime: { value: 0 },
      uAberr: { value: 0 },
      uGlitch: { value: 0 },
      uRes: { value: [1, 1] },
      uDim: { value: 1 },
    };
    dispose() {}
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

const setScreenDpr = (value: number) =>
  Object.defineProperty(window, "devicePixelRatio", { configurable: true, value });

/** The final pass's backdrop dimming, for the first scene mounted in the test */
const dim = () =>
  (gl.passes.find((p) => "uniforms" in p) as unknown as { uniforms: { uDim: { value: number } } }).uniforms.uDim.value;

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
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    document.dispatchEvent(new Event("visibilitychange"));
    const before = gl.renders;
    advance(200);
    expect(gl.renders).toBe(before);
    hidden.mockReturnValue(false);
    document.dispatchEvent(new Event("visibilitychange"));
    canvas.dispatchEvent(new Event("webglcontextlost"));
    canvas.dispatchEvent(new Event("webglcontextrestored"));
    advance(100);
    expect(gl.renders).toBeGreaterThan(before);
    unmount();
  });

  it("always renders at full quality: the screen's own resolution (up to 2×), bloom and grain, even on a struggling device", () => {
    setScreenDpr(3);
    media["(pointer: coarse)"] = true;
    resetQualityCache();
    // A phone that only manages ~12 fps
    const d = display(1000 / 60, () => 80);
    const { unmount } = render(<IceScene />);
    d.run(30);
    expect(d.settings()).toEqual({ dpr: 2, bloom: true, fx: true });
    expect(document.documentElement.dataset.tier).toBe("high");
    unmount();
  });

  it("paces itself to spare the GPU: ~60 fps on a 120 Hz screen, 30 when idle or in the background, none behind an overlay", () => {
    resetQualityCache();
    const focus = vi.spyOn(document, "hasFocus").mockReturnValue(true);
    const d = display(1000 / 120, () => 2);
    const { unmount } = render(<IceScene />);
    /** Frames drawn over `s` simulated seconds */
    const drawn = (s: number) => {
      const before = gl.renders;
      d.run(s);
      return gl.renders - before;
    };
    d.run(0.5); // the refresh rate is measured first
    // While the intro plays: every other refresh, 60 fps
    expect(drawn(2)).toBeGreaterThan(110);
    expect(drawn(2)).toBeLessThan(130);
    // Nobody touching anything for a while: 30 fps
    ui.set({ ready: true });
    d.run(13);
    const idle = drawn(2);
    expect(idle).toBeGreaterThan(55);
    expect(idle).toBeLessThan(65);
    // Any input brings it straight back to 60
    window.dispatchEvent(new Event("pointermove"));
    expect(drawn(2)).toBeGreaterThan(110);
    // In a background window: 30 again
    window.dispatchEvent(new Event("pointermove"));
    focus.mockReturnValue(false);
    expect(drawn(2)).toBeLessThan(65);
    // Behind the image viewer / phone menu: nothing, then back
    focus.mockReturnValue(true);
    sceneState.covered = true;
    expect(drawn(1)).toBe(0);
    sceneState.covered = false;
    expect(drawn(1)).toBeGreaterThan(25);
    // The quality itself never changes
    expect(d.settings()).toEqual({ dpr: 1, bloom: true, fx: true });
    unmount();
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
    expect(dim()).toBeCloseTo(0.76, 2);
    // An evening later the next refresh (every 10 s) finds the sun down
    vi.setSystemTime(new Date("2026-06-21T21:30:00Z")); // 23:30 CEST
    advance(10_200);
    expect(document.documentElement.dataset.sky).toBe("night");
    // …and the backdrop dims less at night, when it is dark already, than by day
    expect(dim()).toBeCloseTo(0.88, 2);
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
