/**
 * Page scrolling: Lenis smooth scroll kept in sync with GSAP ScrollTrigger, plus helpers
 * to scroll to a target, lock scrolling and re-measure. With reduced motion it falls back
 * to native scrolling. Either way the scroll velocity is fed to the WebGL scene.
 */
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { sceneState } from "./store";

gsap.registerPlugin(ScrollTrigger);

let lenis: Lenis | null = null;
let teardown: (() => void) | null = null;

/** The visitor asked the OS for reduced motion. */
export const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Start Lenis smooth scrolling driven by the GSAP ticker (falls back to native scroll). Idempotent; returns the teardown. */
export function initSmoothScroll() {
  if (teardown) return teardown;
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";

  if (prefersReducedMotion()) {
    // Native scroll: still feed velocity to the scene + ScrollTrigger.
    let lastY = window.scrollY;
    let lastT = performance.now();
    const onScroll = () => {
      const now = performance.now();
      const dt = Math.max(1, now - lastT);
      sceneState.velocity = ((window.scrollY - lastY) / dt) * 1000;
      lastY = window.scrollY;
      lastT = now;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    teardown = () => {
      window.removeEventListener("scroll", onScroll);
      teardown = null;
    };
    return teardown;
  }

  lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
    touchMultiplier: 1.3,
  });

  lenis.on("scroll", (l: Lenis) => {
    // Lenis velocity is px per frame; convert to roughly px/s.
    sceneState.velocity = l.velocity * 60;
    ScrollTrigger.update();
  });

  const raf = (time: number) => lenis?.raf(time * 1000);
  gsap.ticker.add(raf);
  gsap.ticker.lagSmoothing(0);

  teardown = () => {
    gsap.ticker.remove(raf);
    lenis?.destroy();
    lenis = null;
    teardown = null;
  };
  return teardown;
}

/** An element, a selector or a y position in px. */
type Target = HTMLElement | number | string;

/** Scroll to an element, selector or y position (through Lenis when it is running). */
export function scrollToTarget(target: Target, opts: { offset?: number; immediate?: boolean; duration?: number } = {}) {
  const { offset = 0, immediate = false, duration } = opts;
  if (lenis) {
    // After a route change Lenis still holds the previous page's height and would
    // clamp the target to it — re-measure first.
    lenis.resize();
    lenis.scrollTo(target, { offset, immediate, duration, force: true, lock: !immediate });
    return;
  }
  let top = 0;
  if (typeof target === "number") top = target;
  else {
    const el = typeof target === "string" ? document.querySelector<HTMLElement>(target) : target;
    if (!el) return;
    top = el.getBoundingClientRect().top + window.scrollY;
  }
  window.scrollTo({ top: top + offset, behavior: immediate || prefersReducedMotion() ? "auto" : "smooth" });
}

/** Freeze page scrolling (lightbox, mobile menu). */
export function setScrollLocked(locked: boolean) {
  if (lenis) {
    if (locked) lenis.stop();
    else lenis.start();
  }
  document.documentElement.classList.toggle("scroll-locked", locked);
}

/** Re-measure page height and ScrollTrigger positions after the layout changes. */
export function refreshScroll() {
  lenis?.resize();
  ScrollTrigger.refresh();
}

/** Components import gsap from here so ScrollTrigger is always registered first. */
export { gsap };
