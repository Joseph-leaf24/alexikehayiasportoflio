import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import PixelScene from "@/components/pixel/PixelScene";
import { sceneState } from "@/lib/store";
import { advance, resetState, useAnimationTimers } from "./utils";
import { media, setScrollY } from "./setup";

/**
 * jsdom has no 2D canvas, so these tests give every canvas a stand-in context that accepts
 * every drawing call and counts them. The level, the game and the camera are the real thing.
 */
type FakeCtx = Record<string, unknown> & { calls: Record<string, number> };
const contexts = new Map<HTMLCanvasElement, FakeCtx>();

function fakeContext(canvas: HTMLCanvasElement): FakeCtx {
  const calls: Record<string, number> = {};
  const state: Record<string, unknown> = { canvas, calls };
  return new Proxy(state, {
    get(target, key: string) {
      if (key in target) return target[key];
      if (key === "createImageData") return (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
      return () => {
        calls[key] = (calls[key] ?? 0) + 1;
      };
    },
    set(target, key: string, value) {
      target[key] = value;
      return true;
    },
  }) as FakeCtx;
}

/** Render the scene with working canvases; returns the on-screen canvas's context. */
function renderScene() {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    let ctx = contexts.get(this);
    if (!ctx) contexts.set(this, (ctx = fakeContext(this)));
    return ctx as unknown as CanvasRenderingContext2D;
  } as unknown as typeof HTMLCanvasElement.prototype.getContext);
  const view = render(<PixelScene />);
  const canvas = view.container.querySelector("canvas")!;
  return { ...view, canvas, main: () => contexts.get(canvas)! };
}

/** Point the (mouse) pointer at a place on the screen, in CSS px. */
function pointAt(clientX: number, clientY: number) {
  sceneState.pointer.x = (clientX / window.innerWidth) * 2 - 1;
  sceneState.pointer.y = -(clientY / window.innerHeight) * 2 + 1;
}

beforeEach(() => {
  resetState();
  contexts.clear();
});

describe("PixelScene", () => {
  it("without a 2D canvas, steps aside: marks <html>, lets the loader finish and draws nothing", () => {
    render(<PixelScene />);
    expect(sceneState.canvas).toBe(false);
    expect(sceneState.ready).toBe(true);
    expect(document.documentElement).toHaveClass("no-canvas");
  });

  it("opens mid-fight on a chunky, upscaled canvas and reports the squad to the HUD", () => {
    useAnimationTimers();
    const { canvas, main } = renderScene();
    expect(sceneState.ready).toBe(false);
    advance(50);
    expect(sceneState.ready).toBe(true);
    // jsdom's 1024×768 window at 1× → 2 screen pixels per scene pixel (a wide view of the battle)
    expect(canvas.width).toBe(512);
    expect(canvas.height).toBe(384);
    expect(canvas.style.width).toBe("1024px");
    expect(main().imageSmoothingEnabled).toBe(false);
    expect(main().calls.drawImage).toBeGreaterThan(0);
    expect(sceneState.cam.z).toBe(2);
    expect(sceneState.level).toBe(1);
    expect(sceneState.stage.length).toBeGreaterThan(3);
    expect(sceneState.players).toHaveLength(8);
    // The warm-up already moved the squad on from the start
    expect(sceneState.cam.x).toBeGreaterThan(0);
    for (const p of sceneState.players) {
      expect(p.hp).toBeGreaterThanOrEqual(0);
      expect(p.hp).toBeLessThanOrEqual(1);
    }
  });

  it("follows the squad as it pushes through the level", () => {
    useAnimationTimers();
    renderScene();
    advance(50);
    const start = sceneState.cam.x;
    // (fights hold the squad up for a few seconds at a time, so give it a while)
    advance(10_000);
    expect(sceneState.cam.x).toBeGreaterThan(start + 80);
  });

  it("opens on the level named in ?level=, if there is one", () => {
    useAnimationTimers();
    window.history.replaceState({}, "", "/?level=snow");
    renderScene();
    advance(50);
    expect(sceneState.stage).toBe("Frozen peaks");
  });

  it("starts a new level on every page change, and drops into the caves on the 404", () => {
    useAnimationTimers();
    window.history.replaceState({}, "", "/?level=desert");
    renderScene();
    advance(50);
    advance(4000);
    const before = sceneState.cam.x;
    sceneState.override = "cube";
    advance(100);
    expect(sceneState.level).toBe(2);
    expect(sceneState.stage).not.toBe("Desert outpost");
    // A new level, so the camera is back near the start
    expect(sceneState.cam.x).toBeLessThan(before);
    sceneState.override = "lost";
    advance(100);
    expect(sceneState.level).toBe(3);
    expect(sceneState.stage).toBe("Underground");
  });

  it("freezes the fight mid-action with reduced motion", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    useAnimationTimers();
    renderScene();
    advance(50);
    const x = sceneState.cam.x;
    advance(2000);
    expect(sceneState.cam.x).toBe(x);
  });

  it("calls in a blast where you click", () => {
    useAnimationTimers();
    renderScene();
    advance(50);
    const before = sceneState.booms;
    pointAt(512, 768 * 0.72);
    sceneState.burst++;
    advance(20);
    expect(sceneState.booms).toBe(before + 1);
  });

  it("calls in an airstrike while the hero is held", () => {
    useAnimationTimers();
    renderScene();
    advance(50);
    const before = sceneState.booms;
    sceneState.press = true;
    advance(2000);
    sceneState.press = false;
    advance(1500);
    expect(sceneState.booms).toBeGreaterThan(before + 6);
  });

  it("stops drawing while an overlay covers it", () => {
    useAnimationTimers();
    const { main } = renderScene();
    advance(50);
    sceneState.covered = true;
    const draws = main().calls.drawImage;
    advance(500);
    expect(main().calls.drawImage).toBe(draws);
    sceneState.covered = false;
    advance(50);
    expect(main().calls.drawImage).toBeGreaterThan(draws);
  });

  it("settles the scroll velocity it is fed", () => {
    useAnimationTimers();
    renderScene();
    advance(50);
    sceneState.velocity = 2000;
    advance(1500);
    expect(Math.abs(sceneState.velocity)).toBeLessThan(20);
  });

  it("reports the level in play, in detail, for the spectator panel", () => {
    useAnimationTimers();
    window.history.replaceState({}, "", "/?level=monsoon");
    renderScene();
    advance(50);
    const intel = sceneState.intel;
    expect(intel.objective.length).toBeGreaterThan(5);
    expect(intel.progress).toBeGreaterThanOrEqual(0);
    expect(intel.progress).toBeLessThanOrEqual(1);
    expect(intel.weather).toBe("Monsoon storm");
    expect(intel.light).toBe("Dusk");
    expect(intel.uptime).toBeGreaterThan(4);
    expect(["none", "ahead"]).toContain(intel.bossState);
    expect(sceneState.white).toBe(0);
  });

  it("drops to about half the frame rate when left idle, but not while someone is spectating", () => {
    useAnimationTimers();
    const { main } = renderScene();
    // Nobody has touched anything for a while: the idle frame rate
    advance(9000);
    let before = main().calls.drawImage;
    advance(1000);
    const idle = main().calls.drawImage - before;
    sceneState.watching = true;
    before = main().calls.drawImage;
    advance(1000);
    const watching = main().calls.drawImage - before;
    expect(watching).toBeGreaterThan(idle * 1.5);
  });
});
