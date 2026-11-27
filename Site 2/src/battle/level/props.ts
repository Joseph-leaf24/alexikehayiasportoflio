/**
 * The old generator's set dressing (`Level.props`, in true-column px), drawn as the old scene's `drawWorld` did: the
 * sprite props lit in front of the blocks or in shadow behind them, and the hand-drawn kinds (`drawn.ts`) in front,
 * each followed by its glow from the old lighting pass (`glows.ts`).
 */
import { TILE } from "@/engine";
import { assetArt } from "../art";
import type { BattleGame } from "../game/BattleGame";
import { artLookup, LAYER, type RenderPass } from "../render";
import { drawProp, Pen, reach } from "./drawn";
import type { LevelSource } from "./generator";
import { drawGlow, glowLead } from "./glows";

/** How far past the view's sides a prop is still drawn: no sprite prop is wider than this. */
const MARGIN = 60;

export function propsPass(game: BattleGame, level: LevelSource): RenderPass {
  const art = artLookup(game);
  const pen = new Pen(game, level);
  return {
    name: "props",
    order: 150,
    draw(g, f, v) {
      const off = g.origin.cols * TILE;
      pen.begin(f, v.time);
      for (const p of level.old.props) {
        const x = p.x - off;
        if (x + reach(p) < v.left - MARGIN || x - glowLead(p) > v.right + MARGIN) continue;
        if (p.kind !== "sprite") drawProp(pen, p, x);
        else {
          const frame = g.art.frame(art(assetArt(p.id)), p.back ? 1 : 0, p.flip ? -1 : 1);
          if (frame >= 0) f.sprites.push(frame, x, p.y, p.back ? LAYER.BACK : LAYER.PROPS, 0, 0xffffffff, 1, 0);
        }
        drawGlow(pen, p, x);
      }
      pen.end();
    },
  };
}
