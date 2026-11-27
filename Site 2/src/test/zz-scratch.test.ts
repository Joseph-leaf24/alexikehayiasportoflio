import { it } from "vitest";
import { cell } from "@/components/pixel/level";
import { BattleGame } from "@/battle";
import { LEVEL } from "@/battle/level";

it("dump", () => {
  const g = new BattleGame({ seed: 3, feature: { level: { theme: "city" } } });
  const L = g.world.res(LEVEL).old;
  const off = g.origin.cols;
  const lines: string[] = [`off ${off} start ${L.start} next ${L.next} h ${L.h}`];
  for (let r = 130; r < 150; r++) {
    let s = String(r).padStart(3) + " ";
    for (let c = 36; c < 56; c++) s += String(L.fg[cell(c + off, r)]).padStart(3);
    s += "   |";
    for (let c = 36; c < 56; c++) s += String(g.terrain.get(c, r)).padStart(3);
    lines.push(s);
  }
  console.log(lines.join("\n"));
});
