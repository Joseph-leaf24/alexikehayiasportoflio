import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import Watch from "@/components/Watch";
import { sceneState } from "@/lib/store";
import { advance, flush, resetState, useAnimationTimers } from "./utils";
import { io } from "./setup";

/** The observer watching the open ground. */
const stageObserver = () => io.instances.find((o) => o.targets.some((t) => t.classList.contains("watch__stage")))!;

/** A stand-in Screen Wake Lock API. */
function fakeWakeLock() {
  const sentinel = {
    released: false,
    release: vi.fn(async () => {
      sentinel.released = true;
    }),
    addEventListener: vi.fn(),
  };
  const request = vi.fn(async () => sentinel);
  Object.defineProperty(navigator, "wakeLock", { configurable: true, value: { request } });
  return { sentinel, request };
}

beforeEach(resetState);
afterEach(() => {
  delete (navigator as unknown as { wakeLock?: unknown }).wakeLock;
});

describe("Watch (the spectator space)", () => {
  it("is open ground with nothing over it, and lets the scene keep its full frame rate while in view", async () => {
    useAnimationTimers();
    render(<Watch />);
    expect(document.querySelector("#watch")).toHaveAttribute("data-scene", "watch");
    expect(document.querySelector("#watch")?.textContent).toBe("");
    expect(screen.queryByRole("heading")).toBeNull();
    await flush();
    advance(300);
    expect(sceneState.watching).toBe(true);
    // Scrolled away: back to normal
    act(() => stageObserver().fire(false));
    expect(sceneState.watching).toBe(false);
  });

  it("keeps the screen awake while you watch, and lets go when you leave", async () => {
    const { sentinel, request } = fakeWakeLock();
    const view = render(<Watch />);
    await flush();
    await flush();
    expect(request).toHaveBeenCalledWith("screen");
    view.unmount();
    expect(sentinel.release).toHaveBeenCalled();
    expect(sceneState.watching).toBe(false);
  });

  it("fades the HUD out with the end-of-level white-out, so the whole screen goes white", async () => {
    useAnimationTimers();
    render(<Watch />);
    await flush();
    sceneState.white = 1;
    advance(300);
    expect(document.documentElement).toHaveClass("is-whiteout");
    sceneState.white = 0.2;
    advance(300);
    expect(document.documentElement).not.toHaveClass("is-whiteout");
  });
});
