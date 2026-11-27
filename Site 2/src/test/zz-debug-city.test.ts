import { expect, it } from "vitest";
import { F_STRUCTURAL, ROWS, runHeadless } from "@/engine";
import { BattleGame, type Feature } from "@/battle";
import { install as art } from "@/battle/art";
import { install as flow } from "@/battle/flow";
import { install as level } from "@/battle/level";
import { install as physics } from "@/battle/physics";
import { install as render } from "@/battle/render";

it("debug city", () => {
  const msgs: string[] = [];
  for (const mode of ["all"]) {
    const features: Feature[] = [
      { name: "art", install: art },
      { name: "render", install: render },
      { name: "physics", install: physics },
      { name: "level", install: level },
      { name: "flow", install: flow },
    ];
    const g = new BattleGame({ seed: 3, features, feature: { level: { theme: "city" } } });
    const t = g.terrain;
    const fg0 = t.fg.slice();
    const fl0 = t.flags.slice();
    const snap = (c0: number, c1: number, r0: number, r1: number) => {
      const lines: string[] = [];
      for (let r = r0; r < r1; r++) {
        let s = String(r).padStart(3) + " ";
        for (let c = c0; c < c1; c++) {
          const m = fg0[t.index(c, r)];
          const st = fl0[t.index(c, r)] & F_STRUCTURAL;
          s += m === 0 ? "." : st ? String.fromCharCode(65 + (m % 26)) : String.fromCharCode(97 + (m % 26));
        }
        lines.push(s);
      }
      return lines.join("\n");
    };
    const qs: [number, number][] = [];
    for (let c = t.startCol; c < t.endCol; c++)
      for (let r = 0; r < ROWS; r++)
        if (t.flags[t.index(c, r)] & F_STRUCTURAL) {
          qs.push([c, r]);
          break;
        }
    const groups = mode === "all" ? [qs] : mode === "each" ? qs.map((q) => [q]) : [[]];
    let reported = false;
    for (const grp of groups) {
      for (const [c, r] of grp) t.structure.queue(c, r);
      runHeadless(g.world, mode === "none" ? 10 : 4, {
        onTick: (_w, tick) => {
          const ch = t.structure.falling.chunks[0];
          if (ch && !reported) {
            reported = true;
            msgs.push(`${mode}: tick ${tick} seeds ${JSON.stringify(grp.slice(0, 3))} chunk ${ch.col},${ch.row} ${ch.w}x${ch.h}`);
            msgs.push(snap(ch.col - 12, ch.col + ch.w + 30, ch.row - 12, Math.min(ROWS, ch.row + ch.h + 14)));
          }
          return mode === "none" || t.structure.pending;
        },
      });
      if (reported) break;
    }
    if (!reported) msgs.push(`${mode}: nothing fell (${qs.length} seeds)`);
  }
  expect(msgs.join("\n")).toBe("");
}, 120000);
