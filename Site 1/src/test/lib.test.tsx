import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { createStore, onAppReady, sceneState, triggerPulse, ui, useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { getQuality, MAX_DPR, qualityFor, resetQualityCache } from "@/lib/device";
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
    expect(getQuality()).toEqual(qualityFor("high", false));
    expect(document.documentElement.dataset.tier).toBe("high");
    resetQualityCache();
    media["(pointer: coarse)"] = true;
    const phone = getQuality();
    expect(phone.tier).toBe("high");
    // Touch screens differ only where it can't show: no MSAA (pixel density) and no cursor dust
    expect({ ...phone, msaa: 2, dust: 480 }).toEqual(qualityFor("high", false));
    expect(probe).not.toHaveBeenCalled();
    expect(MAX_DPR).toBe(2);
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
    const [high, mid, low] = (["high", "mid", "low"] as const).map((t) => qualityFor(t, false));
    expect(high.msaa).toBeGreaterThanOrEqual(mid.msaa);
    expect(mid.particles).toBeGreaterThan(low.particles);
    expect(high.terrain.xMax - high.terrain.xMin).toBeGreaterThan(low.terrain.xMax - low.terrain.xMin);
  });
});
