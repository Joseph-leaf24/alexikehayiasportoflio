import { describe, expect, it } from "vitest";
import { BASE_ROW, TILE } from "@/components/pixel/level";
import { THEMES } from "@/components/pixel/themes";
import { Game, type View } from "@/components/pixel/game";

/** Step a game for `seconds`, the camera on the squad or, all aboard, their gunship. `each` runs every step. */
function play(g: Game, seconds: number, each?: (t: number) => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    const up = g.players.filter((p) => !p.dead && !p.chute && !p.aboard);
    view.y =
      (up.length ? up.reduce((s, p) => s + p.y, 0) / up.length : g.ship ? g.ship.y + 58 : BASE_ROW * TILE) -
      view.h * 0.68;
    g.step(1 / 30, view);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.(i / 30);
  }
}

describe("the squad's gunship", () => {
  it("comes in, sets down for the squad, carries them on far and fast until it's shot down, and they bail out", () => {
    const g = new Game(THEMES.jungle, { seed: 1, waves: false, level: { boss: null, length: 4000 } });
    const modes = new Set<string>();
    let crew = 0;
    let from = -1;
    let flown = 0;
    let bailed = false;
    let hidden = true;
    play(g, 75, () => {
      const s = g.ship;
      if (!s) return;
      if (!s.dead) modes.add(g.shipMode);
      crew = Math.max(crew, g.crew.length);
      if (g.shipMode === "fly" && !s.dead) {
        if (from < 0) from = s.x;
        flown = s.x - from;
        // Far enough: then it's brought down, if the enemy hasn't managed it yet
        if (flown > 1400) g.hurtAlly(s, 999);
        // The squad's advance goes along with it, and those aboard are out of the fight on the ground
        expect(g.advance).toBeGreaterThanOrEqual(s.x - 1);
        for (const id of g.crew) if (Math.abs(g.players[id].y - s.y) > 12) hidden = false;
      }
      if (s.dead && from >= 0 && !bailed) {
        bailed = true;
        // Everyone aboard out, on a parachute
        expect(g.crew).toHaveLength(0);
        expect(g.players.filter((p) => !p.dead && p.chute).length).toBeGreaterThan(3);
        expect(g.players.some((p) => p.aboard)).toBe(false);
      }
    });
    expect([...modes]).toEqual(expect.arrayContaining(["inbound", "board", "fly"]));
    expect(crew).toBe(8);
    expect(hidden).toBe(true);
    // Several screens on in the air, well over the squad's pace on foot
    expect(flown).toBeGreaterThan(1200);
    expect(bailed).toBe(true);
  });

  it("flies them on over the land, never into it, and its door gunners score for the squad", () => {
    // Over a couple of battles: how long it lasts up there is down to the enemy
    let clipped = 0;
    let flying = 0;
    let killsAboard = 0;
    for (const seed of [2, 3]) {
      const g = new Game(THEMES.snow, { seed, waves: false, level: { boss: null, length: 4000 } });
      const before = g.players.reduce((s, p) => s + p.kills, 0);
      let kills = 0;
      play(g, 75, () => {
        const s = g.ship;
        if (!s || s.dead || g.shipMode !== "fly") return;
        flying++;
        if (g.solid(s.x, s.y) || g.solid(s.x - 20, s.y + 6) || g.solid(s.x + 20, s.y + 6)) clipped++;
        kills = g.players.reduce((n, p) => n + p.kills, 0) - before;
      });
      killsAboard += kills;
    }
    expect(flying).toBeGreaterThan(30 * 10);
    expect(clipped).toBe(0);
    expect(killsAboard).toBeGreaterThan(0);
  });

  it("can be kept away, for the squad's tests on foot", () => {
    const g = new Game(THEMES.jungle, { seed: 1, gunship: false, level: { boss: null, length: 4000 } });
    let seen = false;
    play(g, 40, () => {
      if (g.ship || g.allies.some((a) => a.kind === "gunship")) seen = true;
    });
    expect(seen).toBe(false);
  });
});
