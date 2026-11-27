import { describe, expect, it } from "vitest";
import * as old from "@/components/pixel/themes";
import { lightOf, ROTATION, THEMES, themeDeck, variant, WEATHER } from "@/battle/level/themes";

/** A small seeded generator, so both decks see the same numbers. */
const seeded = (seed: number) => () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;

describe("battle themes", () => {
  it("every theme has a name, a palette and at least one feature", () => {
    for (const [key, t] of Object.entries(THEMES)) {
      expect(t.key).toBe(key);
      expect(t.name).toBeTruthy();
      expect(t.sky).toHaveLength(3);
      for (const c of t.sky) expect(c).toMatch(/^#[0-9a-f]{6}$/);
      expect(Object.values(t.features).some((w) => (w ?? 0) > 0)).toBe(true);
      expect(WEATHER[t.ambient]).toBeTruthy();
      expect(lightOf(t)).toBeTruthy();
    }
  });

  it("matches the old engine's themes key for key and value for value", () => {
    expect(Object.keys(THEMES)).toEqual(Object.keys(old.THEMES));
    expect(THEMES).toEqual(old.THEMES);
    expect(ROTATION).toEqual(old.ROTATION);
  });

  it("deals the same themes and variants as the old engine from the same seed", () => {
    const deal = (deck: typeof themeDeck, vary: typeof variant) => {
      const rand = seeded(7);
      const next = deck(rand);
      let key: string | undefined;
      return Array.from({ length: 40 }, () => {
        const t = vary(next(key), rand);
        key = t.key;
        return t;
      });
    };
    const dealt = deal(themeDeck, variant);
    expect(dealt).toEqual(deal(old.themeDeck, old.variant));
    expect(dealt.every((t, i) => i === 0 || t.key !== dealt[i - 1].key)).toBe(true);
  });
});
