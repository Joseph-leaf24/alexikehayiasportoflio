import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { sceneState, ui } from "@/lib/store";
import { resetQualityCache } from "@/lib/device";
import { resetState, useAnimationTimers, advance } from "./utils";
import { media, setScrollY } from "./setup";

/* GPU-facing classes are stubbed; geometry, materials and maths are real three.js */
const gl = vi.hoisted(() => ({ fail: false, renders: 0, passes: [] as { enabled?: boolean }[] }));

vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof import("three")>();
  class FakeRenderer {
    outputColorSpace = "";
    toneMapping = 0;
    constructor() {
      if (gl.fail) throw new Error("WebGL unavailable");
    }
    setPixelRatio() {}
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
});

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

  it("steps quality down when a device can't keep up", () => {
    window.history.replaceState({}, "", "/?quality=high");
    resetQualityCache();
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => (now += 30));
    const frames: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      frames.push(cb);
      return frames.length;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
    const { unmount } = render(<IceScene />);
    ui.set({ ready: true });
    // ~24 s of 30 ms frames: bloom → post-fx → remembered downgrade
    for (let i = 0; i < 800 && frames.length; i++) frames.shift()!(now);
    expect(gl.passes.filter((p) => "enabled" in p).every((p) => p.enabled === false)).toBe(true);
    expect(document.documentElement.dataset.perf).toBe("degraded");
    expect(sessionStorage.getItem("ak:tier")).toBe("mid");
    unmount();
  });

  it("stops drawing during page transitions when frames are already slow", () => {
    window.history.replaceState({}, "", "/?quality=low");
    resetQualityCache();
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => (now += 120));
    const frames: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      frames.push(cb);
      return frames.length;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});
    const run = (n: number) => {
      for (let i = 0; i < n && frames.length; i++) frames.shift()!(now);
    };
    const { unmount } = render(<IceScene />);
    // Fast enough before the governor has measured anything; then ~8 fps
    run(40);
    sceneState.transition = true;
    const before = gl.renders;
    run(20);
    expect(gl.renders).toBe(before);
    sceneState.transition = false;
    run(5);
    expect(gl.renders).toBeGreaterThan(before);
    unmount();
    window.history.replaceState({}, "", "/");
  });

  it("runs the low tier at a capped frame rate on touch devices", () => {
    media["(pointer: coarse)"] = true;
    media["(prefers-reduced-motion: reduce)"] = true;
    useAnimationTimers();
    const { unmount } = render(<IceScene />);
    advance(16);
    const early = gl.renders;
    advance(1000);
    // 30 fps cap → roughly 30 frames, not 60
    expect(gl.renders - early).toBeLessThan(40);
    expect(gl.passes.filter((p) => "enabled" in p).every((p) => p.enabled === false)).toBe(true);
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
