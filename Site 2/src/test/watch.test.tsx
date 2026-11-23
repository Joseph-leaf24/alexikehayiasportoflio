import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
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
  it("follows the level in play while in view, and lets the scene keep its full frame rate", async () => {
    useAnimationTimers();
    Object.assign(sceneState, { level: 7, stage: "Deep jungle", kills: 4210 });
    Object.assign(sceneState.intel, {
      progress: 0.42,
      objective: "Push on - Gunship ahead",
      boss: "Gunship",
      bossState: "ahead",
      levelKills: 128,
      hostiles: 23,
      levelTime: 95,
      cleared: 6,
      uptime: 4335,
      weather: "Fireflies",
      light: "Dusk",
      log: [{ no: 6, name: "Missile silos", kills: 140, time: 92 }],
    });
    render(<Watch />);
    expect(screen.getByRole("heading", { name: "Watch the battle" })).toBeInTheDocument();
    expect(document.querySelector("#watch")).toHaveAttribute("data-scene", "watch");
    await flush();
    advance(300);
    expect(sceneState.watching).toBe(true);
    const panel = screen.getByRole("complementary", { name: "Level intel" });
    for (const text of ["Level 07", "Deep jungle", "Push on - Gunship ahead", "42% to the stronghold", "Gunship: Ahead", "128", "23", "1:35", "Dusk, fireflies", "4,210", "01:12:15", "Missile silos"])
      expect(panel).toHaveTextContent(text);
    // A new level comes in: the panel catches up
    sceneState.level = 8;
    sceneState.stage = "Lost temple";
    advance(300);
    expect(panel).toHaveTextContent("Level 08");
    expect(panel).toHaveTextContent("Lost temple");
    // Scrolled away: back to normal
    act(() => stageObserver().fire(false));
    expect(sceneState.watching).toBe(false);
  });

  it("folds the panel away and back", async () => {
    render(<Watch />);
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "Hide intel" }));
    expect(screen.queryByRole("complementary", { name: "Level intel" })).toBeNull();
    const show = screen.getByRole("button", { name: "Show level intel" });
    expect(show).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(show);
    expect(screen.getByRole("complementary", { name: "Level intel" })).toBeInTheDocument();
  });

  it("starts folded on a phone, where the panel would cover the battle", () => {
    const width = window.innerWidth;
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });
    try {
      render(<Watch />);
      expect(screen.getByRole("button", { name: "Show level intel" })).toBeInTheDocument();
    } finally {
      Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
    }
  });

  it("keeps the screen awake while you watch, and lets go when you leave", async () => {
    const { sentinel, request } = fakeWakeLock();
    const view = render(<Watch />);
    await flush();
    await flush();
    expect(request).toHaveBeenCalledWith("screen");
    expect(screen.getByRole("complementary", { name: "Level intel" })).toHaveTextContent("Kept awake");
    view.unmount();
    expect(sentinel.release).toHaveBeenCalled();
    expect(sceneState.watching).toBe(false);
  });

  it("says the screen may sleep where there's no wake lock", async () => {
    render(<Watch />);
    await flush();
    expect(screen.getByRole("complementary", { name: "Level intel" })).toHaveTextContent("May sleep");
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
