/**
 * Engine core benchmark: `node scripts/bench-engine.mjs [--ticks 3000] [--units 400] [--only battle|spatial|ecs]`.
 *
 * Bundles the core with esbuild (Vite's own dependency) into memory and runs:
 *   1. a battle of 400 units (AI targeting, separation, hitscan shots, bullets as short-lived entities, deaths and
 *      respawns) for thousands of ticks, once on the spatial hash and once on brute-force scans with identical
 *      semantics, checking both end in the same world state;
 *   2. the spatial queries alone at several unit counts, grid against brute force;
 *   3. ECS and event-bus microbenchmarks.
 * Allocation is measured as heap growth over a run with no garbage collection in between.
 */

import { PerformanceObserver } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import path from "node:path";
import v8 from "node:v8";
import vm from "node:vm";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const flag = (name) => {
  const k = process.argv.indexOf(`--${name}`);
  return k >= 0 ? process.argv[k + 1] : undefined;
};
const TICKS = Number(flag("ticks") ?? 3000);
const UNITS = Number(flag("units") ?? 400);
const ONLY = flag("only") ?? "";
const section = (name) => !ONLY || ONLY === name;

let esbuild;
try {
  esbuild = await import("esbuild");
} catch {
  console.error("bench-engine: esbuild not found (it ships with Vite); run `npm install` in Site 2 first.");
  process.exit(1);
}
const bundle = await esbuild.build({
  stdin: {
    contents: 'export * from "./src/engine/core/index.ts"; export * from "./src/engine/headless.ts";',
    resolveDir: root,
    loader: "ts",
    sourcefile: "bench-entry.ts",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  target: "es2022",
  write: false,
  logLevel: "error",
});
const E = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`);
const { World, Rng, SpatialHash, EventBus, runHeadless, hashWorld, enemiesOf, segBox, place, tick } = E;
const { BODY_UNINDEXED, TEAM_ALL, ENTITY_DOOMED, INDEX_MASK, INDEX_BITS } = E;

/* Allocation probe: force a full GC, run synchronously, read the heap growth; valid when no GC ran meanwhile */
v8.setFlagsFromString("--expose-gc");
const gc = vm.runInNewContext("gc");
let gcs = 0;
const gcObserver = new PerformanceObserver((list) => (gcs += list.getEntries().length));
gcObserver.observe({ entryTypes: ["gc"] });
const settle = () => new Promise((r) => setTimeout(r, 30));
async function allocated(fn) {
  gc();
  await settle();
  const g0 = gcs;
  const h0 = process.memoryUsage().heapUsed;
  fn();
  const bytes = process.memoryUsage().heapUsed - h0;
  await settle();
  return { bytes, gcs: gcs - g0 };
}
const fmt = (v, d = 3) => v.toFixed(d);
const pad = (v, n) => String(v).padStart(n);

/* ------------------------------------------------------------------ */
/* Brute force with exactly the spatial hash's semantics               */
/* ------------------------------------------------------------------ */

class Brute {
  n = 0;
  ecs = null;
  constructor() {
    this.alloc(1024);
  }
  alloc(c) {
    const old = this.id;
    this.id = new Int32Array(c);
    this.x0 = new Float64Array(c);
    this.y0 = new Float64Array(c);
    this.x1 = new Float64Array(c);
    this.y1 = new Float64Array(c);
    this.team = new Int32Array(c);
    if (old) throw new Error("Brute: size the index up front");
  }
  begin(ecs) {
    this.n = 0;
    this.ecs = ecs;
  }
  insert(id, x0, y0, x1, y1, team) {
    const k = this.n++;
    if (k >= this.id.length) throw new Error("Brute: too many boxes");
    this.id[k] = id;
    this.x0[k] = x0;
    this.y0[k] = y0;
    this.x1[k] = x1;
    this.y1[k] = y1;
    this.team[k] = 1 << (team & 31);
  }
  end() {}
  rebuild(world) {
    const { pos, body, team } = world.core;
    this.q ??= world.query(pos, body);
    const n = this.q.collect(true),
      idx = this.q.idx;
    this.begin(world);
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      if (body.flags[i] & BODY_UNINDEXED) continue;
      const x = pos.x[i],
        y = pos.y[i],
        w = body.hw[i];
      this.insert(
        (world.gen[i] << INDEX_BITS) | i,
        x - w,
        y - body.h[i],
        x + w,
        y,
        team.$.sparse[i] >= 0 ? team.id[i] : 0
      );
    }
  }
  d2(k, x, y) {
    const dx = x < this.x0[k] ? this.x0[k] - x : x > this.x1[k] ? x - this.x1[k] : 0;
    const dy = y < this.y0[k] ? this.y0[k] - y : y > this.y1[k] ? y - this.y1[k] : 0;
    return dx * dx + dy * dy;
  }
  skip(k, mask) {
    return (
      (this.team[k] & mask) === 0 ||
      (this.ecs !== null && (this.ecs.state[this.id[k] & INDEX_MASK] & ENTITY_DOOMED) !== 0)
    );
  }
  nearest(x, y, r, mask, _filter, exclude = 0) {
    let best = r * r,
      bestId = 0;
    for (let k = 0; k < this.n; k++) {
      const id = this.id[k];
      if (id === exclude) continue;
      const d = this.d2(k, x, y);
      if (d > best || (d === best && bestId !== 0 && id > bestId)) continue;
      if (this.skip(k, mask)) continue;
      best = d;
      bestId = id;
    }
    return bestId;
  }
  inRadius(x, y, r, mask, out) {
    let c = 0;
    const r2 = r * r;
    for (let k = 0; k < this.n; k++) if (this.d2(k, x, y) <= r2 && !this.skip(k, mask)) out[c++] = this.id[k];
    return c;
  }
  segmentFirst(x0, y0, x1, y1, width, mask) {
    const p = width / 2,
      dx = x1 - x0,
      dy = y1 - y0;
    let bestT = Infinity,
      bestId = 0;
    for (let k = 0; k < this.n; k++) {
      const t = segBox(x0, y0, dx, dy, this.x0[k] - p, this.y0[k] - p, this.x1[k] + p, this.y1[k] + p);
      if (t < 0) continue;
      const id = this.id[k];
      if (t > bestT || (t === bestT && id > bestId)) continue;
      if (this.skip(k, mask)) continue;
      bestT = t;
      bestId = id;
    }
    return bestId;
  }
  segment(x0, y0, x1, y1, width, mask, out) {
    const p = width / 2,
      dx = x1 - x0,
      dy = y1 - y0;
    let c = 0;
    for (let k = 0; k < this.n; k++) {
      const t = segBox(x0, y0, dx, dy, this.x0[k] - p, this.y0[k] - p, this.x1[k] + p, this.y1[k] + p);
      if (t >= 0 && !this.skip(k, mask)) out[c++] = this.id[k];
    }
    return c;
  }
}

/* ------------------------------------------------------------------ */
/* 1. The battle                                                       */
/* ------------------------------------------------------------------ */

const W = 4096,
  H = 1900;

function battle(mode, units, seed = 1) {
  const brute = mode === "brute";
  const w = new World({ seed, capacity: 8192, coreSystems: { spatial: !brute } });
  const { pos, vel, body, team, health, life } = w.core;
  const aim = w.defineComponent("aim", { target: "u32", cd: "f32" });
  const unit = w.defineTag("unit");
  const qUnits = w.query(unit, aim);
  const qMoving = w.query(pos, vel);
  const index = brute ? new Brute() : w.spatial;
  const out = [];
  let shots = 0,
    deaths = 0;
  const sortSmall = (n) => {
    for (let a = 1; a < n; a++) {
      const v = out[a];
      let b = a;
      while (b > 0 && out[b - 1] > v) out[b] = out[--b];
      out[b] = v;
    }
  };
  if (brute)
    w.addSystem({ name: "brute.rebuild", phase: "physics", order: 1_000_000, run: (world) => index.rebuild(world) });

  w.addSystem({
    name: "spawn",
    phase: "spawn",
    run(world) {
      const r = world.rng.get("spawn");
      while (unit.$.size < units) {
        const e = world.create();
        world.add(e, vel);
        world.add(e, body);
        world.add(e, team);
        world.add(e, health);
        world.add(e, aim);
        world.add(e, unit);
        place(world, e, r.range(0, W), r.range(100, H - 100));
        const i = e & INDEX_MASK;
        team.id[i] = 1 + r.int(4);
        body.hw[i] = 4;
        body.h[i] = 12;
        health.hp[i] = health.max[i] = r.range(30, 60);
        aim.cd[i] = r.range(0, 1);
      }
    },
  });
  w.addSystem({
    name: "ai",
    phase: "ai",
    run(world, dt) {
      const n = qUnits.collect(true),
        idx = qUnits.idx;
      const r = world.rng.get("ai");
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const x = pos.x[i],
          y = pos.y[i];
        const id = world.idOf(i);
        const foe = index.nearest(x, y - 6, 400, enemiesOf(team.id[i]), null, id);
        aim.target[i] = foe;
        let ax = r.range(-20, 20),
          ay = r.range(-20, 20);
        if (foe) {
          const f = foe & INDEX_MASK;
          const dx = pos.x[f] - x,
            dy = pos.y[f] - y;
          const l = Math.sqrt(dx * dx + dy * dy) || 1;
          // Close to about 150 px, then back off
          const s = l > 150 ? 50 : -20;
          ax += (dx / l) * s;
          ay += (dy / l) * s;
        }
        // Separation from anyone within 16 px (sorted so the sum doesn't depend on query order)
        const c = index.inRadius(x, y, 16, TEAM_ALL, out);
        sortSmall(c);
        for (let j = 0; j < c; j++) {
          if (out[j] === id) continue;
          const o = out[j] & INDEX_MASK;
          ax += (x - pos.x[o]) * 2;
          ay += (y - pos.y[o]) * 2;
        }
        vel.vx[i] += (ax - vel.vx[i]) * Math.min(1, dt * 4);
        vel.vy[i] += (ay - vel.vy[i]) * Math.min(1, dt * 4);
      }
    },
  });
  w.addSystem({
    name: "weapons",
    phase: "weapons",
    run(world, dt) {
      const n = qUnits.collect(true),
        idx = qUnits.idx;
      const r = world.rng.get("weapons");
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        aim.cd[i] -= dt;
        const foe = aim.target[i];
        if (aim.cd[i] > 0 || !world.alive(foe)) continue;
        aim.cd[i] = r.range(0.3, 0.7);
        const f = foe & INDEX_MASK;
        const x = pos.x[i],
          y = pos.y[i] - 6;
        const hit = index.segmentFirst(x, y, pos.x[f], pos.y[f] - 6, 2, enemiesOf(team.id[i]));
        shots++;
        if (hit) health.hp[hit & INDEX_MASK] -= r.range(4, 10);
        // The tracer: a short-lived entity, so creation and destruction churn every tick
        const b = world.create();
        world.add(b, vel);
        world.add(b, body);
        world.add(b, life);
        place(world, b, x, y);
        const bi = b & INDEX_MASK;
        body.flags[bi] = BODY_UNINDEXED;
        vel.vx[bi] = (pos.x[f] - x) * 3;
        vel.vy[bi] = (pos.y[f] - 6 - y) * 3;
        life.t[bi] = 0.3;
      }
    },
  });
  w.addSystem({
    name: "motion",
    phase: "motion",
    run(_world, dt) {
      const n = qMoving.collect(),
        idx = qMoving.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        let x = pos.x[i] + vel.vx[i] * dt,
          y = pos.y[i] + vel.vy[i] * dt;
        if (x < 0 || x > W) {
          vel.vx[i] = -vel.vx[i];
          x = x < 0 ? 0 : W;
        }
        if (y < 40 || y > H) {
          vel.vy[i] = -vel.vy[i];
          y = y < 40 ? 40 : H;
        }
        pos.x[i] = x;
        pos.y[i] = y;
      }
    },
  });
  w.addSystem({
    name: "death",
    phase: "death",
    run(world) {
      const n = qUnits.collect(true),
        idx = qUnits.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (health.hp[i] > 0) continue;
        deaths++;
        world.destroy(world.idOf(i));
      }
    },
  });
  return { w, shots: () => shots, deaths: () => deaths };
}

async function runBattle(mode) {
  const b = battle(mode, UNITS);
  runHeadless(b.w, 300 / 30); // warm-up: the JIT and the pools settle
  const names = b.w.stats.systems.map((s) => s.name);
  const totals = new Float64Array(names.length);
  const shots0 = b.shots(),
    deaths0 = b.deaths();
  const rep = runHeadless(b.w, TICKS / 30, {
    onTick: (world) => {
      const list = world.stats.systems;
      for (let k = 0; k < list.length; k++) totals[k] += list[k].last;
    },
  });
  const shots = b.shots() - shots0,
    deaths = b.deaths() - deaths0;
  // Allocation over 200 more ticks, with and without per-system timing (Node's clock boxes its result)
  const PROBE = 200;
  const timed = await allocated(() => {
    for (let k = 0; k < PROBE; k++) tick(b.w);
  });
  b.w.stats.timing = false;
  const untimed = await allocated(() => {
    for (let k = 0; k < PROBE; k++) tick(b.w);
  });
  const byName = new Map(names.map((n, k) => [n, totals[k] / rep.ticks]));
  return { mode, rep, byName, shots, deaths, timed, untimed, probe: PROBE, hash: hashWorld(b.w), entities: b.w.count };
}

const perTick = (a, ticks) => (a.gcs > 0 ? `GC ran (${a.gcs})` : `${fmt(Math.max(0, a.bytes) / ticks, 0)} B/tick`);

console.log(`Engine core benchmark: Node ${process.version}, ${UNITS} units, ${TICKS} measured ticks\n`);

if (section("battle")) {
  const grid = await runBattle("grid");
  const brute = await runBattle("brute");
  console.log(`1. Battle, ${UNITS} units in ${W}×${H} px (ms per tick)`);
  console.log("   index   mean    p95     max     shots/s  deaths/s  alloc (timing on | off)");
  for (const r of [grid, brute]) {
    const secs = r.rep.simSeconds;
    console.log(
      `   ${r.mode.padEnd(6)} ${fmt(r.rep.msPerTick).padStart(6)} ${fmt(r.rep.p95TickMs).padStart(6)} ${fmt(
        r.rep.maxTickMs
      ).padStart(7)}  ${pad(fmt(r.shots / secs, 0), 7)}  ${pad(fmt(r.deaths / secs, 1), 8)}  ${perTick(
        r.timed,
        r.probe
      )} | ${perTick(r.untimed, r.probe)}`
    );
  }
  console.log("   per system, mean ms per tick (grid | brute):");
  for (const name of new Set([...grid.byName.keys(), ...brute.byName.keys()])) {
    const a = grid.byName.get(name),
      b = brute.byName.get(name);
    const cell = (v) => (v === undefined ? "      -" : fmt(v).padStart(7));
    console.log(`     ${name.padEnd(14)} ${cell(a)} | ${cell(b)}`);
  }
  const spatialMs = (r) =>
    r.byName.get("ai") + r.byName.get("weapons") + (r.byName.get("core.spatial") ?? r.byName.get("brute.rebuild"));
  console.log(
    `   queries + index: ${fmt(spatialMs(grid))} ms vs ${fmt(spatialMs(brute))} ms brute = ${fmt(
      spatialMs(brute) / spatialMs(grid),
      1
    )}x; whole tick ${fmt(brute.rep.msPerTick / grid.rep.msPerTick, 1)}x`
  );
  console.log(
    `   same final world on both indexes: ${grid.hash === brute.hash ? "yes" : "NO"} (hash ${grid.hash >>> 0})`
  );

  // The engine's own share: 400 resting bodies, core systems only (prev snapshot, spatial rebuild, life, stats)
  const idle = new World({ seed: 1, capacity: 8192 });
  const { body, team } = idle.core;
  const r = new Rng(3);
  for (let k = 0; k < 400; k++) {
    const e = idle.create();
    idle.add(e, body);
    idle.add(e, team);
    place(idle, e, r.range(0, W), r.range(0, H));
    body.hw[e & INDEX_MASK] = 4;
    body.h[e & INDEX_MASK] = 12;
    team.id[e & INDEX_MASK] = 1 + (k & 3);
  }
  runHeadless(idle, 100);
  const IDLE = 300;
  const idleOn = await allocated(() => {
    for (let k = 0; k < IDLE; k++) tick(idle);
  });
  idle.stats.timing = false;
  const idleOff = await allocated(() => {
    for (let k = 0; k < IDLE; k++) tick(idle);
  });
  const idleRep = runHeadless(idle, 100);
  console.log(
    `   engine alone (400 bodies, core systems): ${fmt(idleRep.msPerTick)} ms per tick; alloc ${perTick(
      idleOn,
      IDLE
    )} timed | ${perTick(idleOff, IDLE)} untimed`
  );
  console.log("   (game code passing fresh floats to engine calls pays V8's 16 B box per float, per call)\n");
}

/* ------------------------------------------------------------------ */
/* 2. Spatial queries alone                                            */
/* ------------------------------------------------------------------ */

async function spatialBench(n, reps) {
  const r = new Rng(n);
  const boxes = [];
  for (let k = 0; k < n; k++) {
    const x = r.range(0, W),
      y = r.range(100, H);
    boxes.push([k + (1 << 20), x - 4, y - 12, x + 4, y, 1 + r.int(4)]);
  }
  const queries = [];
  for (let k = 0; k < n; k++) {
    const b = boxes[k];
    const a = r.angle();
    queries.push([b[1] + 4, b[4] - 6, b[5], Math.cos(a) * 600, Math.sin(a) * 600]);
  }
  const out = [];
  const run = (index) => {
    index.begin(null);
    for (let k = 0; k < boxes.length; k++) {
      const b = boxes[k];
      index.insert(b[0], b[1], b[2], b[3], b[4], b[5]);
    }
    index.end();
    let acc = 0;
    for (let k = 0; k < queries.length; k++) {
      const q = queries[k];
      const mask = enemiesOf(q[2]);
      acc += index.nearest(q[0], q[1], 400, mask, null, boxes[k][0]) & 7;
      acc += index.inRadius(q[0], q[1], 48, TEAM_ALL, out);
      if (k % 4 === 0) acc += index.segment(q[0], q[1], q[0] + q[3], q[1] + q[4], 2, mask, out);
    }
    return acc;
  };
  const gridIndex = new SpatialHash(),
    bruteIndex = new Brute();
  if (n > 1024) {
    bruteIndex.id = null;
    bruteIndex.alloc(n);
  }
  for (let k = 0; k < 30; k++) {
    run(gridIndex);
    run(bruteIndex);
  }
  if (run(gridIndex) !== run(bruteIndex)) throw new Error(`spatial bench: grid and brute disagree at n=${n}`);
  const time = (index) => {
    const t0 = performance.now();
    for (let k = 0; k < reps; k++) run(index);
    return (performance.now() - t0) / reps;
  };
  const g = time(gridIndex),
    b = time(bruteIndex);
  const a = await allocated(() => run(gridIndex));
  return { g, b, a };
}

if (section("spatial")) {
  console.log(
    "2. Spatial queries per tick: rebuild + n × nearest(400 px) + n × inRadius(48 px) + n/4 × segment(600 px)"
  );
  console.log("   units   grid ms   brute ms   speedup   grid alloc per pass (the float arguments, boxed by V8)");
  for (const n of [100, 400, 1000, 2000, 5000]) {
    const s = await spatialBench(n, n >= 2000 ? 20 : n >= 1000 ? 60 : 300);
    const alloc = s.a.gcs > 0 ? "GC ran" : `${Math.max(0, s.a.bytes)} B`;
    console.log(
      `   ${pad(n, 5)}   ${fmt(s.g).padStart(7)}   ${fmt(s.b).padStart(8)}   ${pad(fmt(s.b / s.g, 1), 6)}x   ${alloc}`
    );
  }
  console.log();
}

/* ------------------------------------------------------------------ */
/* 3. ECS and events                                                   */
/* ------------------------------------------------------------------ */

if (section("ecs")) {
  const w = new World({ seed: 1, capacity: 16384, coreSystems: false });
  const { pos, vel } = w.core;
  const N = 10_000;
  for (let k = 0; k < N; k++) {
    const e = w.create();
    w.add(e, pos);
    w.add(e, vel);
    vel.vx[e & INDEX_MASK] = 1;
  }
  const q = w.query(pos, vel);
  const step = () => {
    const n = q.collect(),
      idx = q.idx;
    const x = pos.x,
      vx = vel.vx;
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      x[i] += vx[i] * 0.033;
    }
  };
  for (let k = 0; k < 200; k++) step();
  const R = 2000;
  let t0 = performance.now();
  for (let k = 0; k < R; k++) step();
  const iter = ((performance.now() - t0) * 1e6) / (R * N);
  const iterAlloc = await allocated(() => {
    for (let k = 0; k < 100; k++) step();
  });

  const CY = 200_000;
  const churn = () => {
    for (let k = 0; k < CY; k++) {
      const e = w.create();
      w.add(e, pos);
      w.destroy(e);
      if ((k & 63) === 63) w.flushDestroyed();
    }
    w.flushDestroyed();
  };
  churn();
  t0 = performance.now();
  churn();
  const life = ((performance.now() - t0) * 1e6) / CY;
  const churnAlloc = await allocated(churn);

  const bus = new EventBus();
  bus.define("hit", () => ({ a: 0, b: 0 }));
  let sum = 0;
  bus.on("hit", (h) => (sum = (sum + h.a) | 0));
  const flushOnce = () => {
    for (let k = 0; k < 1000; k++) bus.claim("hit").a = k;
    bus.flush();
  };
  for (let k = 0; k < 200; k++) flushOnce();
  t0 = performance.now();
  for (let k = 0; k < 2000; k++) flushOnce();
  const ev = ((performance.now() - t0) * 1e6) / (2000 * 1000);
  const evAlloc = await allocated(() => {
    for (let k = 0; k < 200; k++) flushOnce();
  });
  const bytes = (a) => (a.gcs > 0 ? "GC ran" : `${Math.max(0, a.bytes)} B`);
  console.log("3. ECS and events");
  console.log(
    `   query collect + pos += vel, ${N} entities: ${fmt(iter, 2)} ns per entity (alloc over 100 passes: ${bytes(iterAlloc)})`
  );
  console.log(
    `   create + add + destroy + flush: ${fmt(life, 1)} ns per entity (alloc over ${CY}: ${bytes(churnAlloc)})`
  );
  console.log(
    `   pooled event claim + deliver: ${fmt(ev, 1)} ns per event (alloc over 200k: ${bytes(evAlloc)}; checksum ${sum % 7})`
  );
}

gcObserver.disconnect();
