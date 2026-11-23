import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { createStore, onAppReady, sceneState, triggerPulse, ui, useStore } from "@/lib/store";
import { cn, fitRect, wipeGrid } from "@/lib/utils";
import { getQuality, qualityFor, resetQualityCache } from "@/lib/device";
import { resetState } from "./utils";
import { media } from "./setup";

beforeEach(resetState);

describe("createStore", () => {
  it("notifies subscribers only when a value actually changes", () => {
    const store = createStore({ a: 1, b: "x" });
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    store.set({ a: 1 });
    expect(listener).not.toHaveBeenCalled();
    store.set({ a: 2 });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.get()).toEqual({ a: 2, b: "x" });
    unsubscribe();
    store.set({ b: "y" });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("useStore re-renders a component with the selected slice", () => {
    const store = createStore({ count: 0 });
    const Probe = () => <span>count:{useStore(store, (s) => s.count)}</span>;
    render(<Probe />);
    expect(screen.getByText("count:0")).toBeInTheDocument();
    act(() => store.set({ count: 5 }));
    expect(screen.getByText("count:5")).toBeInTheDocument();
  });
});

describe("app-ready signal", () => {
  it("fires immediately when already ready", () => {
    ui.set({ ready: true });
    const cb = vi.fn();
    onAppReady(cb)();
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it("fires once when the loader finishes and can be cancelled", () => {
    const cb = vi.fn();
    const cancelled = vi.fn();
    onAppReady(cb);
    const cancel = onAppReady(cancelled);
    cancel();
    ui.set({ sectionLabel: "unrelated" });
    expect(cb).not.toHaveBeenCalled();
    ui.set({ ready: true });
    ui.set({ sectionLabel: "again" });
    expect(cb).toHaveBeenCalledTimes(1);
    expect(cancelled).not.toHaveBeenCalled();
  });
});

describe("triggerPulse", () => {
  it("stamps the current time", () => {
    triggerPulse();
    expect(sceneState.pulse).toBeGreaterThan(0);
  });
});

describe("cn", () => {
  it("merges and de-duplicates tailwind classes", () => {
    const hidden = Math.random() > 2;
    expect(cn("p-2", hidden && "hidden", "p-4", "text-sm")).toBe("p-4 text-sm");
  });
});

describe("rendering budget", () => {
  beforeEach(() => {
    resetQualityCache();
    window.history.replaceState({}, "", "/");
  });

  it("gives every device the full budget, phones included, without probing the hardware", () => {
    const probe = vi.spyOn(HTMLCanvasElement.prototype, "getContext");
    expect(getQuality()).toEqual(qualityFor("high"));
    expect(document.documentElement.dataset.tier).toBe("high");
    resetQualityCache();
    media["(pointer: coarse)"] = true;
    const phone = getQuality();
    expect(phone.tier).toBe("high");
    // Touch screens get exactly the same scene
    expect(phone).toEqual(qualityFor("high"));
    expect(probe).not.toHaveBeenCalled();
  });

  it("picks a lighter budget only when asked to in the URL (?quality=), for testing", () => {
    window.history.replaceState({}, "", "/?quality=low");
    expect(getQuality().tier).toBe("low");
    expect(getQuality()).toBe(getQuality()); // decided once per page load
    resetQualityCache();
    window.history.replaceState({}, "", "/?quality=nonsense");
    expect(getQuality().tier).toBe("high");
    window.history.replaceState({}, "", "/");
  });

  it("keeps the hand-picked tiers progressively lighter", () => {
    const [high, mid, low] = (["high", "mid", "low"] as const).map((t) => qualityFor(t));
    expect(high.enemies).toBeGreaterThan(mid.enemies);
    expect(mid.enemies).toBeGreaterThan(low.enemies);
    expect(high.particles).toBeGreaterThan(mid.particles);
    expect(mid.particles).toBeGreaterThan(low.particles);
    expect(high.wipeCols).toBeGreaterThanOrEqual(low.wipeCols);
  });
});

describe("HUD and layout helpers", () => {
  it("covers any viewport with roughly square wipe blocks", () => {
    expect(wipeGrid(12, 1440, 900)).toEqual({ cols: 12, rows: 8 });
    // A portrait phone: the columns go along the long (vertical) side
    expect(wipeGrid(12, 390, 844)).toEqual({ cols: 6, rows: 12 });
    expect(wipeGrid(12, 0, 0)).toEqual({ cols: 1, rows: 1 });
  });

  it("finds the part of a picture that cover / contain show", () => {
    // A 2:1 picture in a square box: cover crops the sides
    expect(fitRect(200, 100, 50, 50, "cover")).toEqual({ sx: 50, sy: 0, sw: 100, sh: 100 });
    expect(fitRect(200, 100, 50, 50, "contain")).toEqual({ sx: 0, sy: 0, sw: 200, sh: 100 });
  });
});
