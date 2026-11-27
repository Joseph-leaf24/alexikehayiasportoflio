/**
 * The built-in draw passes, in order: sky and backdrop, falling terrain chunks, units, projectiles, particles,
 * lights, screen effects, text. Each reads simulation state, interpolates by the view's alpha and pushes into the
 * frame; none of them changes the game.
 */
import {
  measureText,
  SPRITE_FLIP_X,
  SPRITE_SILHOUETTE,
  VARIANTS,
  mixRgb,
  type ParallaxLayer,
  type RenderFrame,
} from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, MODE_TUMBLE, UNIT_DEAD, UNIT_HIDDEN } from "../game/components";
import type { DrawSink } from "../game/defs";
import { TEAM_SQUAD } from "../game/teams";
import { BANNER_FADE, POPUP_LIFE } from "./banners";
import type { RenderPass } from "./extract";
import { LAYER, TEXT_LAYER } from "./layers";
import { drawBurning } from "./lights";
import { humanoidPose } from "./poses";

const hash01 = (a: number, b: number) => {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const mod = (a: number, n: number) => ((a % n) + n) % n;

/** Art ids resolved once per game (the art book is per game; ids are fixed after boot). */
function artLookup(game: BattleGame) {
  const cache = new Map<string, number>();
  return (id: string) => {
    let k = cache.get(id);
    if (k === undefined) cache.set(id, (k = game.art.index(id)));
    return k;
  };
}

/** Sky gradient, ambient light, colour grade, the backdrop's parallax layers and the clouds. */
export function skyPass(game: BattleGame): RenderPass {
  const art = artLookup(game);
  const layers: ParallaxLayer[] = [];
  return {
    name: "sky",
    order: 0,
    draw(g, f, v) {
      const sky = g.sky;
      const night = sky.night;
      sky.colors(f.sky);
      f.ambient = sky.ambient;
      const grade = f.effects.grade;
      grade.saturation = 1.1 - night * 0.25;
      grade.contrast = 1.05;
      grade.tint = mixRgb(0xfff4e8, 0xd8e4ff, night);
      // The backdrop stands on the horizon as the old scene placed it (`parallaxBase`), for this view height
      const H = v.h / v.zoom;
      const hz = sky.horizon;
      const B = sky.horizonY;
      let used = 0;
      for (const b of sky.backdrop) {
        const k = art(b.art);
        const frame = k > 0 ? g.art.frame(k, 0, 1) : -1;
        if (frame < 0) continue;
        const l = (layers[used] ??= { frame: 0, factor: 0, y: 0, tint: 0xffffffff, repeatX: true });
        l.frame = frame;
        l.factor = b.factor;
        l.y = H * hz - H / 2 + (H / 2 + B - H * hz) * b.factor + b.lift;
        l.tint = night > 0.5 ? 0xc0c8e8ff : 0xffffffff;
        used++;
      }
      layers.length = used;
      f.parallax = layers;
      // Clouds: drifting sprites behind the terrain, faded out at night
      const cloud = art("sky.cloud");
      const poses = cloud > 0 ? g.art.poses(cloud) : 0;
      const fade = 1 - Math.min(1, night * 2);
      if (poses === 0 || fade < 0.05) return;
      const W = v.w / v.zoom;
      const span = Math.max(W + 160, 1000);
      const base = H * hz - (v.top - (B - H * hz)) * 0.03;
      const tint = (0xffffff00 | Math.round(fade * 255)) >>> 0;
      for (let c = 0; c < g.budgets.clouds; c++) {
        const sx = mod(hash01(c, 1) * 1600 + v.time * (sky.cloudSpeed + hash01(c, 3) * 5) - v.left * 0.05, span) - 80;
        const sy = base - 60 - (40 + hash01(c, 2) * 110);
        const frame = g.art.frame(cloud, c % poses, 1);
        if (frame >= 0) f.sprites.push(frame, v.left + sx, v.top + sy, LAYER.SKY, 0, tint, 1, 0);
      }
    },
  };
}

/** Rigid pieces of collapsing structures, drawn from their cells with the terrain's own tiles. */
export function chunksPass(): RenderPass {
  return {
    name: "chunks",
    order: 100,
    draw(g, f, v) {
      const t = g.terrain;
      const tiles = t.tiles;
      for (const ch of t.falling) {
        const y0 = ch.prevY + (ch.y - ch.prevY) * v.alpha;
        const x0 = ch.col * 8;
        if (x0 > v.right || x0 + ch.pxW < v.left || y0 > v.bottom || y0 + ch.pxH < v.top) continue;
        for (let lc = 0; lc < ch.w; lc++)
          for (let lr = 0; lr < ch.h; lr++) {
            const k = lc * ch.h + lr;
            const m = ch.fg[k];
            if (!m) continue;
            const frame = tiles[m * VARIANTS + (ch.variant[k] & (VARIANTS - 1))];
            if (frame >= 0) f.sprites.push(frame, x0 + lc * 8, y0 + lr * 8, LAYER.CHUNKS, 0, 0xffffffff, 1, 0);
          }
      }
    },
  };
}

/** Every unit: its pose from its art set, facing, hit flash, corpse tumble and fade, parachute, and its def's extras. */
export function unitsPass(game: BattleGame): RenderPass {
  const { pos, prev, kind, team } = game.core;
  const { unit, motor, player } = game.c;
  const q = game.world.query(pos, prev, unit, kind);
  const art = artLookup(game);
  let frame: RenderFrame | null = null;
  const sink: DrawSink = {
    sprite(a, pose, facing, x, y, layer, tint = 0xffffffff) {
      const fr = game.art.frame(a, pose, facing);
      if (fr >= 0 && frame)
        frame.sprites.push(fr, x, y, layer, facing < 0 && game.art.mirrors(a) ? SPRITE_FLIP_X : 0, tint, 1, 0);
    },
    particle(x, y, size, color, flags) {
      frame?.particles.push(x, y, size, color, flags);
    },
    light(x, y, radius, color, intensity) {
      frame?.lights.push(x, y, radius, color, intensity);
    },
  };
  return {
    name: "units",
    order: 200,
    draw(g, f, v) {
      frame = f;
      const chute = art("fx.chute");
      const defs = g.content.units.defs;
      const links = g.links;
      const n = q.collect();
      const idx = q.idx;
      const a = v.alpha;
      const blink = Math.floor(v.time * 18) & 1;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const fl = unit.flags[i];
        if (fl & UNIT_HIDDEN) continue;
        const x = prev.x[i] + (pos.x[i] - prev.x[i]) * a;
        if (x < v.left - 48 || x > v.right + 48) continue;
        const y = prev.y[i] + (pos.y[i] - prev.y[i]) * a;
        if (y < v.top - 64 || y > v.bottom + 48) continue;
        const d = kind.def[i];
        const def = defs[d - 1];
        const isPlayer = player.$.sparse[i] >= 0;
        if (isPlayer && player.inv[i] > 0 && !(fl & UNIT_DEAD) && blink) continue;
        const set = unit.art[i] || links.unitArt[d];
        const facing = unit.facing[i];
        const fr = g.art.frame(set, def.pose ? def.pose(g, i) : humanoidPose(g, i), facing);
        if (fr < 0) continue;
        let flags = facing < 0 && g.art.mirrors(set) ? SPRITE_FLIP_X : 0;
        let tint = 0xffffffff;
        let rot = 0;
        let dy = 0;
        if (fl & UNIT_DEAD) {
          // A corpse turns in quarter turns as it tumbles, lies flat once down, and fades out at the end
          const grounded = motor.mode[i] === MODE_TUMBLE && g.core.vel.vy[i] === 0;
          rot = (grounded ? -facing : Math.floor(unit.spin[i]) & 3) * (Math.PI / 2);
          dy = -6;
          const left = def.corpse ?? 1.2;
          const fade = isPlayer ? 1 : Math.min(1, (left - unit.deadT[i]) / 0.6);
          tint = (0xffffff00 | Math.max(0, Math.round(fade * 255))) >>> 0;
        } else if (unit.hitT[i] > 0) {
          flags |= SPRITE_SILHOUETTE;
          tint = 0xfff4e0ff;
        }
        const layer = def.layer ?? (team.id[i] === TEAM_SQUAD ? LAYER.PLAYERS : LAYER.ENEMIES);
        f.sprites.push(fr, x, y + dy, layer, flags, tint, 1, rot);
        if (motor.mode[i] === MODE_CHUTE && chute > 0) {
          const cf = g.art.frame(chute, isPlayer ? player.id[i] : 8, 1);
          if (cf >= 0) f.sprites.push(cf, x, y, LAYER.FX, 0, 0xffffffff, 1, 0);
        }
        def.draw?.(g, i, x, y, sink);
      }
      frame = null;
    },
  };
}

/** Shots in flight: a rotated sprite, or the old engine's pixel streaks; glowing ones light the night. */
export function projectilesPass(game: BattleGame): RenderPass {
  const { pos, prev, vel } = game.core;
  const { projectile } = game.c;
  const q = game.world.query(pos, prev, projectile);
  return {
    name: "projectiles",
    order: 300,
    draw(g, f, v) {
      const defs = g.content.projectiles.defs;
      const links = g.links;
      const n = q.collect();
      const idx = q.idx;
      const a = v.alpha;
      const budget = g.budgets.maxLights;
      const blink = Math.floor(v.time * 6) & 1;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const x = prev.x[i] + (pos.x[i] - prev.x[i]) * a;
        const y = prev.y[i] + (pos.y[i] - prev.y[i]) * a;
        if (x < v.left - 16 || x > v.right + 16 || y < v.top - 16 || y > v.bottom + 16) continue;
        const d = projectile.def[i];
        const def = defs[d - 1];
        const vx = vel.vx[i],
          vy = vel.vy[i];
        const sp = Math.hypot(vx, vy) || 1;
        const ux = vx / sp,
          uy = vy / sp;
        const set = links.projectileArt[d];
        if (set > 0) {
          const fr = g.art.frame(set, 0, 1);
          if (fr >= 0) f.sprites.push(fr, x, y, LAYER.PROJECTILES, 0, 0xffffffff, 1, Math.atan2(vy, vx));
        } else if (def.look) {
          const L = def.look;
          const s = L.size ?? 1;
          const gap = L.spacing ?? 2;
          for (let c = 0; c < L.colors.length; c++) {
            const col = c === 0 && L.blink !== undefined && blink ? L.blink : L.colors[c];
            f.particles.push(x - ux * c * gap, y - uy * c * gap, s, col, L.flags ?? 0);
          }
        }
        if (def.glow && f.lights.count < budget)
          f.lights.push(x, y, def.glow.radius, def.glow.color, def.glow.intensity);
      }
    },
  };
}

export function particlesPass(): RenderPass {
  return {
    name: "particles",
    order: 400,
    draw(g, f, v) {
      g.particles.draw(f.particles, v.alpha, v.left - 4, v.top - 4, v.right + 4, v.bottom + 4);
    },
  };
}

/** Flashes from the simulation, then burning terrain, within the light budget. */
export function lightsPass(): RenderPass {
  return {
    name: "lights",
    order: 500,
    draw(g, f, v) {
      const budget = g.budgets.maxLights;
      const m = 64;
      g.flashes.draw(f.lights, budget - f.lights.count, v.left - m, v.top - m, v.right + m, v.bottom + m);
      const left = budget - f.lights.count;
      if (left > 0)
        drawBurning(f.lights, g.terrain, left, Math.floor(v.left / 8) - 4, Math.ceil(v.right / 8) + 4, v.time);
    },
  };
}

export function screenPass(): RenderPass {
  return {
    name: "screen",
    order: 600,
    draw(g, f) {
      g.screen.apply(f);
    },
  };
}

/** Banners across the upper third, and popups floating up from their world points. */
export function textPass(): RenderPass {
  return {
    name: "text",
    order: 700,
    draw(g, f, v) {
      const W = v.w;
      const b = g.banners.current;
      const alpha = g.banners.alpha;
      if (b && alpha > 0) {
        const a = Math.round(alpha * 255);
        const big = measureText(b.text, 3) < W - 16 ? 3 : 2;
        const y = Math.round(v.h * 0.26) - (b.t < BANNER_FADE ? Math.round((1 - alpha) * 6) : 0);
        f.text.push(b.text, W / 2, y, ((b.color & 0xffffff00) | a) >>> 0, big, 1, TEXT_LAYER.BANNERS);
        if (b.sub) f.text.push(b.sub, W / 2, y + 6 * big, (0xe8e4dc00 | a) >>> 0, 1, 1, TEXT_LAYER.BANNERS);
      }
      const p = g.popups;
      for (let k = 0; k < p.n; k++) {
        const u = p.t[k] / POPUP_LIFE;
        const sx = (p.x[k] - v.left) * v.zoom;
        const sy = (p.y[k] - v.top) * v.zoom - u * 12;
        if (sx < -40 || sx > W + 40 || sy < -10 || sy > v.h + 10) continue;
        const a = Math.round(255 * (u < 0.7 ? 1 : (1 - u) / 0.3));
        f.text.push(p.text[k], sx, sy, ((p.color[k] & 0xffffff00) | a) >>> 0, 1, 1, TEXT_LAYER.POPUPS);
      }
    },
  };
}
