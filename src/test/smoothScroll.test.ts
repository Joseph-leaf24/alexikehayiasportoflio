import { afterEach, describe, expect, it, vi } from "vitest";
import { sceneState } from "@/lib/store";
import { media, setScrollY } from "./setup";

const lenisInstances: FakeLenis[] = [];
class FakeLenis {
  handlers: Record<string, (l: FakeLenis) => void> = {};
  velocity = 2;
  scrollTo = vi.fn();
  resize = vi.fn();
  stop = vi.fn();
  start = vi.fn();
  destroy = vi.fn();
  raf = vi.fn();
  constructor(public opts: Record<string, unknown>) {
    lenisInstances.push(this);
  }
  on(event: string, cb: (l: FakeLenis) => void) {
    this.handlers[event] = cb;
  }
}
vi.mock("lenis", () => ({ default: FakeLenis }));

const mod = await import("@/lib/smoothScroll");

afterEach(() => {
  document.documentElement.classList.remove("scroll-locked");
});

describe("smooth scrolling", () => {
  it("drives Lenis from the GSAP ticker and feeds scroll velocity to the scene", () => {
    const stop = mod.initSmoothScroll();
    expect(mod.initSmoothScroll()).toBe(stop); // idempotent
    const lenis = lenisInstances.at(-1)!;
    expect(typeof (lenis.opts.easing as (t: number) => number)(0.5)).toBe("number");
    lenis.handlers.scroll(lenis);
    expect(sceneState.velocity).toBe(120);

    mod.scrollToTarget("#projects", { offset: -10 });
    expect(lenis.resize).toHaveBeenCalled();
    expect(lenis.scrollTo).toHaveBeenCalledWith("#projects", expect.objectContaining({ offset: -10, lock: true }));
    mod.scrollToTarget(0, { immediate: true });
    expect(lenis.scrollTo).toHaveBeenLastCalledWith(0, expect.objectContaining({ immediate: true, lock: false }));

    mod.setScrollLocked(true);
    expect(lenis.stop).toHaveBeenCalled();
    expect(document.documentElement).toHaveClass("scroll-locked");
    // …and tells the 3D scene it's hidden behind the overlay
    expect(sceneState.covered).toBe(true);
    mod.setScrollLocked(false);
    expect(lenis.start).toHaveBeenCalled();
    expect(sceneState.covered).toBe(false);
    mod.refreshScroll();

    stop();
    expect(lenis.destroy).toHaveBeenCalled();
  });

  it("falls back to native scrolling when reduced motion is preferred", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    expect(mod.prefersReducedMotion()).toBe(true);
    const stop = mod.initSmoothScroll();
    setScrollY(300);
    window.dispatchEvent(new Event("scroll"));
    expect(sceneState.velocity).not.toBe(0);

    const el = document.createElement("section");
    el.id = "target";
    document.body.appendChild(el);
    mod.scrollToTarget("#target", { offset: 5 });
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 305, behavior: "auto" });
    mod.scrollToTarget(el);
    mod.scrollToTarget(120, { immediate: true });
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 120, behavior: "auto" });
    const calls = vi.mocked(window.scrollTo).mock.calls.length;
    mod.scrollToTarget("#missing");
    expect(vi.mocked(window.scrollTo).mock.calls.length).toBe(calls);
    mod.setScrollLocked(true);
    expect(document.documentElement).toHaveClass("scroll-locked");
    el.remove();
    stop();
  });

  it("uses smooth native scrolling when motion is allowed and Lenis is off", () => {
    mod.scrollToTarget(50);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 50, behavior: "smooth" });
  });
});
