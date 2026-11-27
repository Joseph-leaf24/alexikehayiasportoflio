/**
 * Battle benchmark: `node scripts/bench-battle.mjs [--seconds 20] [--enemies 300] [--particles 3000] [--seed 7]`.
 *
 * Bundles the battle (and the engine under it) with esbuild, Vite's own dependency, into memory, then runs the
 * vertical slice headless at the target load: the 8 players and 300 Regime troopers spread over the view and the
 * ground ahead of it (replaced as they fall), all thinking, moving and shooting, and 3,000 live particles. Reports ms per
 * tick (mean, 95th percentile, worst) and the slowest systems, with and without the per-system timing.
 */

import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const flag = (name, d) => {
  const k = process.argv.indexOf(`--${name}`);
  return k >= 0 ? Number(process.argv[k + 1]) : d;
};
const SECONDS = flag("seconds", 20);
const ENEMIES = flag("enemies", 300);
const PARTICLES = flag("particles", 3000);
const SEED = flag("seed", 7);

let esbuild;
try {
  esbuild = await import("esbuild");
} catch {
  console.error("bench-battle: esbuild not found (it ships with Vite); run `npm install` in Site 2 first.");
  process.exit(1);
}
const bundle = await esbuild.build({
  stdin: {
    contents:
      'export * from "./src/battle/index.ts"; export { runHeadless } from "./src/engine/headless.ts";' +
      'export { SQUAD } from "./src/battle/squad/index.ts";',
    resolveDir: root,
    loader: "ts",
    sourcefile: "bench-battle-entry.ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  target: "es2022",
  tsconfig: path.join(root, "tsconfig.app.json"),
  write: false,
  logLevel: "error",
});
const B = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`);
const { BattleGame, runHeadless, TEAM_SQUAD } = B;

const fmt = (v, d = 3) => v.toFixed(d);

function setup(timing) {
  const game = new BattleGame({
    seed: SEED,
    viewW: 640,
    viewH: 360,
    debug: timing,
    budgets: { maxEnemies: ENEMIES + 20, maxParticles: PARTICLES + 500, maxProjectiles: 2000 },
    feature: { enemies: { spawn: false } },
  });
  // Let the squad land and the camera settle
  runHeadless(game.world, 3);
  const cam = game.camera;
  const r = game.world.rng.get("bench");
  for (let k = 0; k < ENEMIES; k++) {
    const x = cam.left + 40 + r.next() * (cam.right - cam.left + 700);
    game.spawn("regime.trooper", x, game.terrain.surfaceY(x, 0), { facing: -1 });
  }
  return game;
}

/** Keep the load at the target: troopers replaced ahead as they fall, the particle pool full of embers and dust. */
function topUp(game) {
  const p = game.particles;
  const cam = game.camera;
  const r = game.rng.fx;
  for (let n = 0; game.stats.enemies < ENEMIES && n < 20; n++) {
    const x = cam.right + 20 + r.next() * 700;
    game.spawn("regime.trooper", x, game.terrain.surfaceY(x, 0), { facing: -1 });
  }
  while (p.n < PARTICLES) {
    const x = cam.left + r.next() * (cam.right - cam.left);
    const y = cam.top + r.next() * (cam.bottom - cam.top);
    p.spawn(
      x,
      y,
      (r.next() - 0.5) * 20,
      -10 - r.next() * 20,
      2 + r.next() * 2,
      1,
      0xffb060ff,
      0x5b5866ff,
      20,
      0.2,
      0,
      1
    );
  }
}

function run(timing) {
  const game = setup(timing);
  const rep = runHeadless(game.world, SECONDS, { onTick: () => topUp(game) });
  const teams = { squad: 0, enemy: 0 };
  const u = game.c.unit;
  for (let k = 0; k < u.$.size; k++) {
    const i = u.$.dense[k];
    if (game.core.team.id[i] === TEAM_SQUAD) teams.squad++;
    else teams.enemy++;
  }
  return { game, rep, teams };
}

console.log(`Battle benchmark: 8 players + ${ENEMIES} troopers + ${PARTICLES} particles, ${SECONDS} s (seed ${SEED})`);
// One untimed run first, so the numbers below are for JIT-compiled code
run(false);
for (const timing of [false, true]) {
  const { game, rep, teams } = run(timing);
  console.log(
    `\n${timing ? "with" : "without"} per-system timing: ${fmt(rep.msPerTick)} ms/tick mean, ` +
      `p95 ${fmt(rep.p95TickMs)}, worst ${fmt(rep.maxTickMs, 2)} (${rep.ticks} ticks, ${fmt(rep.wallMs, 0)} ms wall)`
  );
  console.log(
    `  at the end: ${teams.squad} squad, ${teams.enemy} other units, ${game.particles.n} particles, ` +
      `${game.stats.projectiles} projectiles, ${game.stats.kills} kills, ${game.stats.booms} blasts, ${game.world.count} entities`
  );
  if (timing) {
    const top = [...game.world.stats.systems].sort((a, b) => b.ms - a.ms).slice(0, 10);
    for (const s of top) console.log(`  ${s.name.padEnd(22)} ${fmt(s.ms)} ms  (max ${fmt(s.max)})`);
  }
}
