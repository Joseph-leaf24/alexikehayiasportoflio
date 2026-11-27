import { describe, expect, it } from "vitest";
import {
  dayCycle,
  parseTimeOverride,
  sceneSun,
  solarPosition,
  sunDirection,
  VIEW_HEADING,
  wallClockToday,
} from "@/lib/sun";

const deg = (r: number) => (r * 180) / Math.PI;
const AMS = "Europe/Amsterdam";

describe("sun position (Netherlands)", () => {
  it("matches the textbook noon heights at the solstices", () => {
    // Local solar noon is ~11:39 UTC at 5.3° E
    expect(deg(solarPosition(new Date("2026-06-21T11:39:00Z")).elevation)).toBeCloseTo(61.2, 0);
    expect(deg(solarPosition(new Date("2026-12-21T11:39:00Z")).elevation)).toBeCloseTo(14.4, 0);
    // …due south at noon
    expect(deg(solarPosition(new Date("2026-06-21T11:39:00Z")).azimuth)).toBeCloseTo(180, -1);
  });

  it("is below the horizon at midnight and rises in the east", () => {
    expect(solarPosition(new Date("2026-06-21T23:39:00Z")).elevation).toBeLessThan(0);
    const morning = solarPosition(new Date("2026-03-20T06:30:00Z"));
    expect(deg(morning.azimuth)).toBeGreaterThan(60);
    expect(deg(morning.azimuth)).toBeLessThan(120);
  });

  it("turns into day factors that fade through twilight", () => {
    const noon = dayCycle(new Date("2026-06-21T11:39:00Z"));
    const night = dayCycle(new Date("2026-12-21T23:00:00Z"));
    expect(noon.day).toBe(1);
    expect(noon.golden).toBeLessThan(0.01);
    expect(night.day).toBe(0);
    expect(night.golden).toBe(0);
    // Winter sunrise in Amsterdam is ~07:48 UTC: half-light and golden around then
    const sunrise = dayCycle(new Date("2026-12-21T07:48:00Z"));
    expect(sunrise.day).toBeGreaterThan(0.3);
    expect(sunrise.day).toBeLessThan(0.7);
    expect(sunrise.golden).toBeGreaterThan(0.5);
  });

  it("maps compass bearings into the scene (south ahead, east left)", () => {
    const [sx, , sz] = sunDirection({ elevation: 0, azimuth: Math.PI });
    expect(sx).toBeCloseTo(0);
    expect(sz).toBeCloseTo(-1);
    const [ex] = sunDirection({ elevation: 0, azimuth: Math.PI / 2 });
    expect(ex).toBeCloseTo(-1);
    const [, y] = sunDirection({ elevation: Math.PI / 2, azimuth: 0 });
    expect(y).toBeCloseTo(1);
  });
});

describe("the sun as drawn in the scene", () => {
  const spread = 26;
  const at = (iso: string) => sceneSun(new Date(iso));

  it("crosses the frame: up on the left, centred at solar noon, down on the right", () => {
    const rise = at("2026-12-21T07:48:00Z");
    const noon = at("2026-12-21T11:39:00Z");
    const set = at("2026-12-21T15:30:00Z");
    expect(deg(rise.azimuth - VIEW_HEADING)).toBeCloseTo(-spread, -1);
    expect(deg(noon.azimuth - VIEW_HEADING)).toBeCloseTo(0, 0);
    expect(deg(set.azimuth - VIEW_HEADING)).toBeCloseTo(spread, -1);
    // Noon sits at the peak; sunrise / sunset just above the land (the lift)
    expect(deg(noon.elevation)).toBeCloseTo(12.5, 0);
    expect(deg(rise.elevation)).toBeGreaterThan(5);
    expect(deg(rise.elevation)).toBeLessThan(8);
  });

  it("peaks at the same height in summer and winter, and is well hidden at night", () => {
    expect(deg(at("2026-06-21T11:39:00Z").elevation)).toBeCloseTo(12.5, 0);
    const night = at("2026-12-21T23:00:00Z");
    expect(deg(night.elevation)).toBeLessThan(-30);
    // Night azimuths stay clamped near the frame instead of wrapping round behind the camera
    expect(Math.abs(deg(night.azimuth - VIEW_HEADING))).toBeLessThanOrEqual(spread * 1.4 + 1e-6);
  });

  it("takes a narrower, lower arc when asked (portrait screens)", () => {
    const d = new Date("2026-12-21T11:39:00Z");
    const tall = sceneSun(d, (8 * Math.PI) / 180, (9.5 * Math.PI) / 180, (6.5 * Math.PI) / 180);
    expect(deg(tall.elevation)).toBeCloseTo(9.5, 0);
    const morning = sceneSun(new Date("2026-12-21T09:00:00Z"), (8 * Math.PI) / 180);
    expect(Math.abs(deg(morning.azimuth - VIEW_HEADING))).toBeLessThan(8);
  });
});

describe("?time= override", () => {
  it("parses HH:MM and rejects nonsense", () => {
    expect(parseTimeOverride("?time=07:30")).toEqual([7, 30]);
    expect(parseTimeOverride("?quality=low&time=23%3A05")).toEqual([23, 5]);
    expect(parseTimeOverride("?time=25:00")).toBeNull();
    expect(parseTimeOverride("?time=noon")).toBeNull();
    expect(parseTimeOverride("")).toBeNull();
  });

  it("builds today's date at that Amsterdam wall-clock time (CET and CEST)", () => {
    const winter = wallClockToday(13, 0, AMS, new Date("2026-01-15T09:00:00Z"));
    expect(winter.toISOString()).toBe("2026-01-15T12:00:00.000Z");
    const summer = wallClockToday(13, 0, AMS, new Date("2026-07-15T09:00:00Z"));
    expect(summer.toISOString()).toBe("2026-07-15T11:00:00.000Z");
  });
});
