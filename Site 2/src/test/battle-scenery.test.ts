import { describe, expect, it, vi } from "vitest";
import { lightState as oldLightState } from "@/components/pixel/game";
import { createFrame, materials, SPRITE_EMISSIVE, TILE, type SpriteList } from "@/engine";
import { BattleGame, type Feature } from "@/battle";
import { assetArt, install as art } from "@/battle/art";
import { backdropLayers, install as level, LEVEL, type LevelSource } from "@/battle/level";
import { signArt } from "@/battle/level/drawn";
import { THEMES } from "@/battle/level/themes";
import { install as render, LAYER } from "@/battle/render";

/** Just the land and its sky: nobody to move the camera. */
function sky(theme: string, seed = 4): BattleGame {
  const features: Feature[] = [
    { name: "art", install: art },
    { name: "render", install: render },
    { name: "level", install: level },
  ];
  return new BattleGame({ seed, features, feature: { level: { theme } } });
}

describe("battle backdrops", () => {
  it("show the level theme's backdrop at once, with every theme's backdrop and uniforms registered", () => {
    const g = sky("desert");
    expect(g.sky.backdrop.map((b) => b.art)).toEqual(backdropLayers("desert").map((b) => b.art));
    expect(g.sky.previous).toEqual([]);
    expect(g.sky.fade).toBe(1);
    expect(g.sky.clouds).toBe(true);
    for (const t of Object.values(THEMES)) {
      expect(g.content.art.has(`backdrop.${t.backdrop}.far`)).toBe(true);
      expect(g.content.art.has(`troop.${t.look}.mook`)).toBe(true);
    }
  });

  it("have no clouds under a roof, and the cave's far rock hangs from the top of the view", () => {
    const g = sky("cave");
    expect(g.sky.clouds).toBe(false);
    expect(g.sky.backdrop.filter((b) => b.top).map((b) => b.art)).toEqual(["backdrop.cave.far"]);
  });

  it("fade into the next biome's backdrop once the camera passes a border", () => {
    const g = sky("jungle");
    const src = g.world.res(LEVEL);
    while (src.old.borders.length === 0) g.terrain.ensure(g.terrain.endCol + 64);
    const border = src.old.borders[0];
    const before = g.sky.backdrop;
    g.camera.place((border.col - g.origin.cols) * TILE + TILE, g.camera.y);
    g.step(1 / 30);
    expect(src.biome).toBe(border.theme);
    expect(g.sky.previous).toBe(before);
    expect(g.sky.backdrop.map((b) => b.art)).toEqual(backdropLayers(border.theme.backdrop).map((b) => b.art));
    expect(g.sky.fade).toBeLessThan(0.1);
    for (let k = 0; k < 45; k++) g.step(1 / 30);
    expect(g.sky.fade).toBe(1);
  });
});

type Prop = LevelSource["old"]["props"][number];

/**
 * A level with the camera on its first prop of a kind (that `where` accepts), and a frame drawer. Headless there is
 * no canvas to paint on, so a stand-in frame number encodes the art, pose and facing: art × 100 + pose × 2 + (1 when
 * facing left).
 */
function propsAt<K extends Prop["kind"]>(
  theme: string,
  seed: number,
  kind: K,
  where: (p: Extract<Prop, { kind: K }>, g: BattleGame) => boolean = () => true
) {
  const g = sky(theme, seed);
  vi.spyOn(g.art, "frame").mockImplementation((art, pose, facing) => art * 100 + pose * 2 + (facing < 0 ? 1 : 0));
  g.terrain.ensure(400);
  const src = g.world.res(LEVEL);
  type P = Extract<Prop, { kind: K }>;
  const p = src.old.props.find((p): p is P => p.kind === kind && where(p as P, g));
  if (!p) throw new Error(`no ${kind} in ${theme} ${seed}`);
  const off = g.origin.cols * TILE;
  g.camera.place(p.x - off, g.camera.y);
  const draw = () => {
    const f = createFrame();
    g.buildFrame(f, 1);
    return f.sprites as SpriteList;
  };
  return { g, src, p, x: p.x - off, off, draw };
}

/** Where a drawn prop's art is in a frame, as [x, y, pose]. */
function placed(g: BattleGame, s: SpriteList, id: string): number[][] {
  const art = g.art.index(id);
  const out: number[][] = [];
  for (let i = 0; i < s.count; i++)
    if (Math.floor(s.frame[i] / 100) === art) {
      expect(s.layer[i]).toBe(LAYER.PROPS);
      out.push([s.x[i], s.y[i], s.frame[i] % 100 >> 1]);
    }
  return out;
}

/** A strip's runs in a frame, as [x, y, length] (a strip's pose is its length - 1). */
const runs = (g: BattleGame, s: SpriteList, id: string) => placed(g, s, id).map(([x, y, pose]) => [x, y, pose + 1]);

describe("battle props", () => {
  it("draw the old generator's sprite props in view: lit in front of the blocks, in shadow behind them", () => {
    const { g, src, off, draw } = propsAt("city", 3, "sprite");
    const s = draw();
    const asset = (p: Prop) => (p.kind === "sprite" ? g.art.index(assetArt(p.id)) : 0);
    const want = src.old.props
      .filter((p) => p.kind === "sprite" && p.x - off > g.camera.left - 60 && p.x - off < g.camera.right + 60)
      .map((p) => (p.kind === "sprite" ? [asset(p) * 100 + (p.back ? 2 : 0) + (p.flip ? 1 : 0), p.x - off, p.y] : []));
    expect(want.length).toBeGreaterThan(0);
    const got: number[][] = [];
    for (let i = 0; i < s.count; i++)
      if (g.content.art.id(Math.floor(s.frame[i] / 100)).startsWith("asset.")) {
        expect(s.layer[i]).toBe(s.frame[i] % 100 >= 2 ? LAYER.BACK : LAYER.PROPS);
        got.push([s.frame[i], s.x[i], s.y[i]]);
      }
    expect(got).toEqual(want);
  });

  it("draw a hand-drawn prop of any size as runs of whole strip pieces plus one", () => {
    const { g, p, x, draw } = propsAt("jungle", 1, "antenna");
    const s = draw();
    const mast = runs(g, s, "strip.mast");
    // The mast, top down: 8-px pieces, then what's left
    const want: number[][] = [];
    for (let k = 0; k < p.h; k += 8) want.push([x, p.y - p.h + k, Math.min(8, p.h - k)]);
    expect(mast).toEqual(want);
    expect(runs(g, s, "strip.rung").length).toBe(Math.ceil((p.h - 6) / 8));
    expect(runs(g, s, "strip.beacon.on").length + runs(g, s, "strip.beacon.off").length).toBe(1);
  });

  it("draw a stilt's legs while their beams stand, and drop a leg whose beams are gone", () => {
    // Legs on beams at both ends (legs into water have none, and are drawn regardless)
    const beamAt = (g: BattleGame, tc: number, y: number) =>
      g.terrain.get(tc - g.origin.cols, Math.round(y / TILE)) === materials.BEAM;
    const { g, p, x, draw } = propsAt(
      "jungle",
      1,
      "legs",
      (p, g) => beamAt(g, Math.floor(p.x / TILE), p.y) && beamAt(g, Math.floor((p.x + p.w - 1) / TILE), p.y)
    );
    const mine = ([rx]: number[]) => rx >= x && rx < x + p.w;
    const legs = () => runs(g, draw(), "strip.leg").filter((r) => mine(r) && r[1] === p.y);
    const braces = () => runs(g, draw(), "strip.brace").filter(mine);
    expect(legs().map(([lx]) => lx)).toEqual([x, x + p.w - 2]);
    expect(braces().length).toBeGreaterThan(0);
    // Blast the left leg's beams away
    const col = Math.floor(p.x / TILE) - g.origin.cols;
    for (let r = Math.round(p.y / TILE); g.terrain.get(col, r) === materials.BEAM; r++)
      g.terrain.set(col, r, materials.AIR);
    expect(legs().map(([lx]) => lx)).toEqual([x + p.w - 2]);
    expect(braces()).toEqual([]);
  });

  it("draw a curve as one run per stretch at a height, covering every column once", () => {
    const { g, p, x, draw } = propsAt("city", 3, "rope");
    const rope = runs(g, draw(), "strip.rope").filter(([rx]) => rx >= x && rx <= x + p.w);
    const cols = rope.flatMap(([rx, , len]) => Array.from({ length: len }, (_, k) => rx + k));
    expect(cols).toEqual(Array.from({ length: p.w + 1 }, (_, k) => x + k));
    for (const [rx, ry] of rope) expect(ry).toBe(p.y - 10 + Math.round(Math.sin(((rx - x) / p.w) * Math.PI) * 4));
    expect(rope.length).toBeLessThan(cols.length / 2);
  });

  it("draw fixed shapes at their pivot, and animated ones in poses picked by time", () => {
    const { g, p, x, draw } = propsAt("deepjungle", 1, "flag");
    const tents = g.world.res(LEVEL).old.props.filter((t) => t.kind === "tent" && Math.abs(t.x - p.x) < 300);
    expect(tents.length).toBeGreaterThan(0);
    const poses = new Set<number>();
    for (let k = 0; k < 30; k++) {
      const s = draw();
      const flag = placed(g, s, "shape.flag");
      expect(flag.map(([fx, fy]) => [fx, fy])).toEqual([[x, p.y]]);
      poses.add(flag[0][2]);
      expect(placed(g, s, "shape.tent")).toEqual(tents.map((t) => [t.x - (p.x - x), t.y, 0]));
      g.step(1 / 30);
    }
    expect(poses.size).toBeGreaterThan(4);
  });

  it("draw a silo's hatch, braziers, and the fittings of lamps and alarms", () => {
    const { g, p, x, draw } = propsAt("jungle", 1, "silo");
    const s = draw();
    expect(runs(g, s, "strip.silo")).toEqual([
      [x - 14, p.y, 8],
      [x - 6, p.y, 8],
      [x + 2, p.y, 8],
      [x + 10, p.y, 4],
    ]);
    // The launch site's alarm stands by its silo
    const alarm = g.world.res(LEVEL).old.props.find((a) => a.kind === "alarm" && Math.abs(a.x - p.x) < 200);
    if (!alarm) throw new Error("no alarm by the silo");
    expect(runs(g, s, "strip.mount")).toContainEqual([alarm.x - (p.x - x) - 1, alarm.y - 2, 4]);
    const b = propsAt("jungle", 1, "brazier");
    expect(placed(b.g, b.draw(), "shape.brazier")).toContainEqual([b.x, b.p.y, 0]);
    const l = propsAt("jungle", 1, "lamp");
    expect(runs(l.g, l.draw(), "strip.lamp")).toContainEqual([l.x - 1, l.p.y, 4]);
  });

  it("draw a traffic light in the old one's state through its cycle, and streetlights lit only after dusk", () => {
    const { g, p, x, draw } = propsAt("city", 1, "stoplight");
    const states = new Set<number>();
    // 15 s in half-second hops: past one 14 s cycle, so the short amber shows too
    for (let k = 0; k < 30; k++) {
      const s = draw();
      const st = oldLightState(g.frames.view.time, p.x);
      expect(placed(g, s, "shape.stoplight").filter(([lx]) => lx === x)).toEqual([[x, p.y, st]]);
      states.add(st);
      for (let n = 0; n < 15; n++) g.step(1 / 30);
    }
    expect(states).toEqual(new Set([0, 1, 2]));

    const sl = propsAt("city", 1, "streetlight");
    const lamp = (id: string) => {
      const s = sl.draw();
      expect(placed(sl.g, s, "shape.streetlight")).toContainEqual([sl.x, sl.p.y, 0]);
      const art = sl.g.art.index(id);
      for (let i = 0; i < s.count; i++)
        if (Math.floor(s.frame[i] / 100) === art) expect(s.flags[i]).toBe(id.endsWith(".on") ? SPRITE_EMISSIVE : 0);
      return runs(sl.g, s, id);
    };
    sl.g.sky.clock = 0;
    expect(lamp("strip.streetlamp.off")).toContainEqual([sl.x + 4, sl.p.y - 23, 3]);
    expect(lamp("strip.streetlamp.on")).toEqual([]);
    sl.g.sky.clock = 0.5;
    expect(lamp("strip.streetlamp.on")).toContainEqual([sl.x + 4, sl.p.y - 23, 3]);
    expect(lamp("strip.streetlamp.off")).toEqual([]);
  });

  it("turn a radar's dish as the old one did, and nod a derrick's pumpjack beside its tower", () => {
    const r = propsAt("snow", 6, "radar");
    const d = propsAt("desert", 6, "derrick");
    const widths = new Set<number>();
    const nods = new Set<number>();
    // 4 s: past a turn of the dish (2.1 s) and a nod (3.1 s)
    for (let k = 0; k < 30; k++) {
      const s = r.draw();
      const w = Math.round(Math.abs(Math.cos(r.g.frames.view.time * 1.5)) * 10);
      expect(placed(r.g, s, "shape.radar")).toContainEqual([r.x, r.p.y, w]);
      widths.add(w);
      const sd = d.draw();
      expect(placed(d.g, sd, "shape.derrick")).toEqual([[d.x, d.p.y, 0]]);
      const pump = placed(d.g, sd, "shape.pumpjack");
      expect(pump.map(([px, py]) => [px, py])).toEqual([[d.x, d.p.y]]);
      nods.add(pump[0][2]);
      r.g.step(4 / 30);
      d.g.step(4 / 30);
    }
    expect(widths.size).toBeGreaterThan(5);
    expect(nods.size).toBe(8);
  });

  it("draw a waterfall as columns of bands falling down its height, with foam at its foot", () => {
    const { g, p, x, draw } = propsAt("deepjungle", 3, "waterfall");
    // Each column: pieces down its whole height in one look (which band is on top), pose = look × 12 + length - 1
    const looks = () => {
      const pieces = placed(g, draw(), "shape.waterfall").filter(([fx]) => fx >= x && fx < x + p.w);
      return Array.from({ length: p.w }, (_, c) => {
        const col = pieces.filter(([fx]) => fx === x + c);
        expect(col.map(([, fy, pose]) => [fy, (pose % 12) + 1])).toEqual(
          Array.from({ length: Math.ceil(p.h / 12) }, (_, j) => [p.y + j * 12, Math.min(12, p.h - j * 12)])
        );
        const look = Math.floor(col[0][2] / 12);
        for (const [, , pose] of col) expect(Math.floor(pose / 12)).toBe(look);
        return look;
      });
    };
    // The bands move down: a frame later each column's top band is the same or the one before it
    let last = looks();
    let moved = 0;
    for (let k = 0; k < 10; k++) {
      g.step(1 / 30);
      const now = looks();
      now.forEach((look, c) => {
        expect([0, 3]).toContain((look - last[c] + 4) % 4);
        if (look !== last[c]) moved++;
      });
      last = now;
    }
    expect(moved).toBeGreaterThan(p.w * 5);
    expect(new Set(last).size).toBeGreaterThan(1);
    const foam = runs(g, draw(), "strip.foam").filter(([fx]) => fx >= x - 3 && fx < x + p.w + 3);
    expect(foam.length).toBeGreaterThan(0);
    for (const [, fy, len] of foam) {
      expect(len).toBe(1);
      expect(fy).toBeGreaterThanOrEqual(p.y + p.h - 4);
      expect(fy).toBeLessThanOrEqual(p.y + p.h - 2);
    }
  });

  it("draw an awning's stripes out left of it, a px lower every 5, in its colour and white, over an ink edge", () => {
    const { g, p, x, draw } = propsAt("city", 2, "awning");
    const s = draw();
    const mine = ([rx]: number[]) => rx >= x - p.w && rx < x;
    const cols = new Map<number, string>();
    for (const color of [p.color, "#f4f0e8"])
      for (const [rx, ry, len] of runs(g, s, `strip.awning.${color.slice(1)}`).filter(mine))
        for (let c = rx; c < rx + len; c++) {
          const k = x - 1 - c;
          expect(cols.has(k)).toBe(false);
          expect(ry).toBe(p.y + Math.floor(k / 5));
          cols.set(k, color);
        }
    expect([...cols.keys()].sort((a, b) => a - b)).toEqual(Array.from({ length: p.w }, (_, k) => k));
    for (const [k, color] of cols) expect(color).toBe((k >> 1) % 2 ? "#f4f0e8" : p.color);
    // The edge, w + 1 px (every awning is 14 wide): a whole piece and the rest
    const ey = p.y + Math.floor(p.w / 5) + 3;
    const edge = runs(g, s, "strip.awning.edge").filter(([rx]) => rx >= x - p.w - 1 && rx < x);
    expect(edge).toEqual([
      [x - p.w - 1, ey, 8],
      [x - p.w + 7, ey, p.w - 7],
    ]);
  });

  it("draw every sign the old generator places, each a shape registered at install", () => {
    let signs = 0;
    for (const theme of Object.keys(THEMES)) {
      const g = sky(theme, 2);
      g.terrain.ensure(400);
      for (const p of g.world.res(LEVEL).old.props)
        if (p.kind === "sign") {
          signs++;
          expect(g.content.art.has(signArt(p)), `${theme}: ${p.text} ${p.board} ${p.ink}`).toBe(true);
        }
    }
    expect(signs).toBeGreaterThan(20);
    const { g, p, x, draw } = propsAt("city", 2, "sign");
    expect(placed(g, draw(), signArt(p))).toContainEqual([x, p.y, 0]);
  });
});
