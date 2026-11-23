import type { ReactElement } from "react";
import { act, render } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { vi } from "vitest";
import { TransitionProvider } from "@/components/chrome/Transition";
import { blankIntel, nav, sceneState, ui } from "@/lib/store";
import { resetQualityCache } from "@/lib/device";

/** Fake timers that also drive requestAnimationFrame + performance.now. */
export function useAnimationTimers() {
  vi.useFakeTimers({
    toFake: [
      "setTimeout",
      "clearTimeout",
      "setInterval",
      "clearInterval",
      "requestAnimationFrame",
      "cancelAnimationFrame",
      "performance",
      "Date",
    ],
  });
}

/** Advance fake timers (and animation frames) by `ms`, inside act(). */
export const advance = (ms: number) =>
  act(() => {
    vi.advanceTimersByTime(ms);
  });

/** Let pending microtasks (e.g. the mock IntersectionObserver callbacks) run, inside act(). */
export const flush = () => act(async () => {
  await Promise.resolve();
});

/** Put every shared store back to its initial state. */
export function resetState() {
  ui.set({ ready: false, sectionIndex: "00", sectionLabel: "INTRO", menuOpen: false });
  nav.inApp = false;
  Object.assign(sceneState, {
    stops: [],
    override: null,
    press: false,
    transition: false,
    covered: false,
    pulse: -100,
    velocity: 0,
    ready: false,
    canvas: true,
    kills: 0,
    booms: 0,
    level: 1,
    stage: "",
    intel: blankIntel(),
    watching: false,
    white: 0,
    burst: 0,
    highlight: -1,
    formation: "village",
  });
  document.documentElement.classList.remove("is-whiteout");
  sceneState.pointer.x = sceneState.pointer.y = 0;
  sceneState.pointer.lastMove = -1e9;
  sceneState.cam.x = sceneState.cam.y = sceneState.cam.z = 0;
  for (const p of sceneState.players) Object.assign(p, { score: 0, hp: 1, alive: true });
  resetQualityCache();
  delete document.documentElement.dataset.tier;
  delete document.documentElement.dataset.sky;
  document.documentElement.classList.remove("no-canvas");
  window.history.replaceState({}, "", "/");
}

/** Finish the loader: everything waiting on onAppReady starts. */
export function markReady() {
  act(() => ui.set({ ready: true }));
}

/** Shows the current location so tests can assert navigation. */
export function LocationProbe() {
  const loc = useLocation();
  return <output data-testid="location">{loc.pathname}</output>;
}

/** Render inside a router + page-transition provider. */
export function renderRoute(element: ReactElement, { path = "/", route = "/", state }: { path?: string; route?: string; state?: unknown } = {}) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: route, state }]}>
      <TransitionProvider>
        <Routes>
          <Route path={path} element={element} />
          <Route path="*" element={<p>elsewhere</p>} />
        </Routes>
        <LocationProbe />
      </TransitionProvider>
    </MemoryRouter>
  );
}
