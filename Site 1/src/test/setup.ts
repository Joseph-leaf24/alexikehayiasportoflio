/** Vitest setup: jsdom stand-ins for the browser APIs the site uses, reset after every test. */
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/* ------------------------------------------------------------------ */
/* Browser APIs jsdom doesn't implement                                */
/* ------------------------------------------------------------------ */

/** Media queries that should match — tests flip these per case. */
export const media: Record<string, boolean> = {};
Object.defineProperty(window, "matchMedia", {
  writable: true,
  configurable: true,
  value: (query: string) => ({
    matches: !!media[query],
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }),
});

/** Everything observed is reported as in view (unless a test turns this off). */
export const io = { autoIntersect: true, instances: [] as MockIntersectionObserver[] };
class MockIntersectionObserver {
  root = null;
  rootMargin = "";
  thresholds = [];
  targets: Element[] = [];
  constructor(public cb: IntersectionObserverCallback) {
    io.instances.push(this);
  }
  observe(el: Element) {
    this.targets.push(el);
    if (io.autoIntersect)
      queueMicrotask(() =>
        this.cb(
          [{ isIntersecting: true, target: el, intersectionRatio: 1 } as IntersectionObserverEntry],
          this as unknown as IntersectionObserver
        )
      );
  }
  /** Manually report an entry. */
  fire(isIntersecting: boolean, target = this.targets[0]) {
    this.cb(
      [{ isIntersecting, target, intersectionRatio: isIntersecting ? 1 : 0 } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver
    );
  }
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}
Object.defineProperty(window, "IntersectionObserver", {
  writable: true,
  configurable: true,
  value: MockIntersectionObserver,
});

class MockResizeObserver {
  constructor(public cb: ResizeObserverCallback) {}
  observe() {}
  unobserve() {}
  disconnect() {}
}
Object.defineProperty(window, "ResizeObserver", { writable: true, configurable: true, value: MockResizeObserver });

// Web Animations: resolve immediately
Element.prototype.animate = function animate() {
  return { finished: Promise.resolve(), cancel: () => {}, play: () => {}, pause: () => {} } as unknown as Animation;
};

// Scrolling: keep a real scrollY so code reading it sees the change
let scrollY = 0;
Object.defineProperty(window, "scrollY", { configurable: true, get: () => scrollY });
Object.defineProperty(window, "pageYOffset", { configurable: true, get: () => scrollY });
export const setScrollY = (y: number) => {
  scrollY = y;
};
window.scrollTo = vi.fn((x?: number | ScrollToOptions, y?: number) => {
  scrollY = typeof x === "object" ? (x.top ?? scrollY) : (y ?? scrollY);
}) as unknown as typeof window.scrollTo;
Element.prototype.scrollIntoView = () => {};

// No WebGL in jsdom: getContext returns null instead of logging "not implemented"
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as typeof HTMLCanvasElement.prototype.getContext;

Object.defineProperty(document, "fonts", { configurable: true, value: { ready: Promise.resolve() } });

if (typeof window.PointerEvent === "undefined") {
  class PointerEventPolyfill extends MouseEvent {
    pointerType: string;
    pointerId: number;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerType = init.pointerType ?? "mouse";
      this.pointerId = init.pointerId ?? 1;
    }
  }
  Object.defineProperty(window, "PointerEvent", { writable: true, configurable: true, value: PointerEventPolyfill });
}

afterEach(() => {
  cleanup();
  for (const k of Object.keys(media)) delete media[k];
  io.autoIntersect = true;
  io.instances.length = 0;
  setScrollY(0);
  sessionStorage.clear();
  vi.useRealTimers();
});
