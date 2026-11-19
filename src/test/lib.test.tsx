import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { createStore, onAppReady, sceneState, triggerPulse, ui, useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  getQuality,
  isIntegratedGpu,
  qualityFor,
  readSignals,
  rememberDowngrade,
  resetQualityCache,
  scoreDevice,
  type DeviceSignals,
} from "@/lib/device";
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

describe("device tiers", () => {
  const base: DeviceSignals = { cores: 8, memory: 8, dpr: 1, width: 1440, coarse: false, saveData: false, webgl2: true };

  it("rates strong desktops high", () => {
    expect(scoreDevice({ ...base, gpu: "NVIDIA GeForce RTX 4070" })).toBe("high");
    expect(scoreDevice(base)).toBe("high");
  });

  it("never rates integrated laptop graphics high (browsers often run on them)", () => {
    const iris = "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics (0x0000A7A0) Direct3D11 vs_5_0 ps_5_0, D3D11)";
    const apu = "ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001638) Direct3D11 vs_5_0 ps_5_0, D3D11)";
    expect(scoreDevice({ ...base, gpu: iris })).toBe("mid");
    expect(scoreDevice({ ...base, gpu: apu })).toBe("mid");
    expect(scoreDevice({ ...base, gpu: "AMD Radeon 780M Graphics" })).toBe("mid");
    expect(isIntegratedGpu("Intel(R) UHD Graphics 620")).toBe(true);
    // Dedicated laptop GPUs keep the full effect
    expect(isIntegratedGpu("NVIDIA GeForce RTX 4080 Laptop GPU")).toBe(false);
    expect(isIntegratedGpu("AMD Radeon RX 7600M XT")).toBe(false);
    expect(isIntegratedGpu(undefined)).toBe(false);
    expect(scoreDevice({ ...base, gpu: "NVIDIA GeForce RTX 4080 Laptop GPU" })).toBe("high");
  });

  it("rates average hardware mid", () => {
    expect(scoreDevice({ ...base, cores: 4, memory: 4 })).toBe("mid");
    expect(scoreDevice({ ...base, coarse: true, width: 390, gpu: "Apple GPU", cores: 6, memory: undefined })).toBe("mid");
  });

  it("rates weak, data-saving or software-rendered devices low", () => {
    expect(scoreDevice({ ...base, webgl2: false })).toBe("low");
    expect(scoreDevice({ ...base, saveData: true })).toBe("low");
    expect(scoreDevice({ ...base, gpu: "Google SwiftShader" })).toBe("low");
    expect(scoreDevice({ ...base, cores: 2, memory: 2, coarse: true, width: 360 })).toBe("low");
    expect(scoreDevice({ ...base, cores: undefined, memory: undefined, coarse: true, width: 360 })).toBe("low");
  });

  it("gives each tier a progressively lighter budget", () => {
    const high = qualityFor("high", false);
    const mid = qualityFor("mid", false);
    const low = qualityFor("low", true);
    expect(high.msaa).toBeGreaterThan(mid.msaa);
    expect(mid.particles).toBeGreaterThan(low.particles);
    expect(low.bloom).toBe(false);
    expect(low.fpsCap).toBe(30);
    expect(high.dprCap).toBeLessThanOrEqual(1.5);
    expect(mid.dprCap).toBeLessThan(high.dprCap);
    expect(qualityFor("high", true).msaa).toBeLessThan(high.msaa);
    expect(qualityFor("mid", true).particles).toBeLessThan(mid.particles);
    expect(high.terrain.xMax - high.terrain.xMin).toBeGreaterThan(low.terrain.xMax - low.terrain.xMin);
  });

  it("reads signals from the browser (no WebGL in jsdom → low)", () => {
    const s = readSignals();
    expect(s.webgl2).toBe(false);
    expect(getQuality(s).tier).toBe("low");
    expect(document.documentElement.dataset.tier).toBe("low");
    // cached
    expect(getQuality({ ...base, gpu: "RTX" }).tier).toBe("low");
  });

  it("reads the GPU name when WebGL2 is available", () => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    const lose = vi.fn();
    HTMLCanvasElement.prototype.getContext = (() => ({
      getExtension: (name: string) =>
        name === "WEBGL_debug_renderer_info" ? { UNMASKED_RENDERER_WEBGL: 1 } : { loseContext: lose },
      getParameter: () => "Apple M2",
    })) as unknown as typeof getContext;
    media["(pointer: coarse)"] = false;
    const s = readSignals();
    expect(s).toMatchObject({ webgl2: true, gpu: "Apple M2" });
    expect(lose).toHaveBeenCalled();
    HTMLCanvasElement.prototype.getContext = (() => {
      throw new Error("blocked");
    }) as unknown as typeof getContext;
    expect(readSignals().webgl2).toBe(false);
    HTMLCanvasElement.prototype.getContext = getContext;
  });

  it("honours ?quality= and remembered downgrades", () => {
    window.history.replaceState({}, "", "/?quality=high");
    expect(getQuality({ ...base, webgl2: false }).tier).toBe("high");
    resetQualityCache();
    window.history.replaceState({}, "", "/");
    expect(rememberDowngrade("high")).toBe("mid");
    expect(document.documentElement.dataset.perf).toBe("degraded");
    expect(getQuality(base).tier).toBe("mid");
    resetQualityCache();
    expect(rememberDowngrade("low")).toBe("low");
  });

  it("survives blocked storage", () => {
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied");
    });
    const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("denied");
    });
    expect(getQuality(base).tier).toBe("high");
    expect(rememberDowngrade("mid")).toBe("low");
    get.mockRestore();
    set.mockRestore();
  });
});
