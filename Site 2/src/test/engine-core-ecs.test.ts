import { describe, expect, it } from "vitest";
import {
  Ecs,
  GEN_MAX,
  INDEX_BITS,
  INDEX_MASK,
  NULL_ENTITY,
  World,
  genOf,
  indexOf,
  type ComponentBase,
} from "@/engine/core";

describe("entities", () => {
  it("hands out generational ids that are never 0", () => {
    const ecs = new Ecs();
    const a = ecs.create();
    const b = ecs.create();
    expect(a).not.toBe(NULL_ENTITY);
    expect(indexOf(a)).toBe(0);
    expect(indexOf(b)).toBe(1);
    expect(genOf(a)).toBe(1);
    expect(ecs.alive(a) && ecs.alive(b)).toBe(true);
    expect(ecs.count).toBe(2);
    expect(ecs.alive(0)).toBe(false);
    expect(ecs.alive(-1)).toBe(false);
    expect(ecs.alive((1 << INDEX_BITS) | 12345)).toBe(false);
  });

  it("defers destroy until the flush, then the id is dead and the slot's generation moves on", () => {
    const ecs = new Ecs();
    const pos = ecs.defineComponent("pos", { x: "f32", y: "f32" });
    const e = ecs.create();
    ecs.add(e, pos);
    pos.x[indexOf(e)] = 7;
    expect(ecs.destroy(e)).toBe(true);
    // Still alive and readable until the flush
    expect(ecs.alive(e)).toBe(true);
    expect(ecs.doomed(e)).toBe(true);
    expect(ecs.has(e, pos)).toBe(true);
    expect(pos.x[indexOf(e)]).toBe(7);
    expect(ecs.destroy(e)).toBe(false);
    expect(ecs.pendingDestroys).toBe(1);
    expect(ecs.flushDestroyed()).toBe(1);
    expect(ecs.alive(e)).toBe(false);
    expect(ecs.doomed(e)).toBe(false);
    expect(ecs.has(e, pos)).toBe(false);
    expect(pos.$.size).toBe(0);
    expect(ecs.count).toBe(0);
    expect(ecs.destroy(e)).toBe(false);
    expect(ecs.add(e, pos)).toBe(false);
  });

  it("reuses an index only after many others are free, with a new generation", () => {
    const ecs = new Ecs();
    const first = ecs.create();
    ecs.destroy(first);
    ecs.flushDestroyed();
    // Too few free slots yet: a fresh index
    const second = ecs.create();
    expect(indexOf(second)).toBe(1);
    const ids: number[] = [];
    for (let i = 0; i < 1100; i++) ids.push(ecs.create());
    for (const id of ids) ecs.destroy(id);
    ecs.flushDestroyed();
    // Now over 1024 free: the oldest freed slot comes back first, one generation on
    const reused = ecs.create();
    expect(indexOf(reused)).toBe(indexOf(first));
    expect(genOf(reused)).toBe(genOf(first) + 1);
    expect(ecs.alive(first)).toBe(false);
    expect(ecs.alive(reused)).toBe(true);
  });

  it("wraps a slot's generation from GEN_MAX back to 1, never producing id 0", () => {
    const ecs = new Ecs();
    // Keep the free queue just above the reuse threshold so slot 0 cycles every 1025 creates
    const ring: number[] = [];
    for (let i = 0; i < 1026; i++) ring.push(ecs.create());
    for (const id of ring) ecs.destroy(id);
    ecs.flushDestroyed();
    const seen = new Set<number>();
    let wrapped = false,
      minId = Infinity;
    for (let n = 0; n < 1026 * (GEN_MAX + 2); n++) {
      const id = ecs.create();
      if (id < minId) minId = id;
      if (indexOf(id) === 0) {
        if (genOf(id) === 1 && seen.size > 0) wrapped = true;
        seen.add(genOf(id));
      }
      ecs.destroy(id);
      ecs.flushDestroyed();
      if (wrapped) break;
    }
    expect(wrapped).toBe(true);
    expect(minId).toBeGreaterThan(0);
    expect(Math.max(...seen)).toBe(GEN_MAX);
    expect(seen.has(0)).toBe(false);
  });

  it("idOf, destroyAll and the index limit", () => {
    const ecs = new Ecs();
    const a = ecs.create();
    const b = ecs.create();
    expect(ecs.idOf(indexOf(b))).toBe(b);
    expect(ecs.idOf(99)).toBe(0);
    expect(ecs.indexLimit).toBe(2);
    expect(ecs.destroyAll()).toBe(2);
    expect(ecs.destroyAll()).toBe(0);
    ecs.flushDestroyed();
    expect(ecs.alive(a) || ecs.alive(b)).toBe(false);
    expect(ecs.idOf(indexOf(a))).toBe(0);
  });

  it("destroy hooks see the components, can destroy more in the same flush, and can't wedge it by throwing", () => {
    const ecs = new Ecs();
    const hp = ecs.defineComponent("hp", { v: "f32" });
    const a = ecs.create(),
      b = ecs.create(),
      c = ecs.create();
    ecs.add(a, hp);
    hp.v[indexOf(a)] = 42;
    const seen: [number, number][] = [];
    const off = ecs.onDestroy((id, i) => {
      seen.push([id, hp.$.hasIndex(i) ? hp.v[i] : -1]);
      if (id === a) ecs.destroy(b);
      if (id === b) throw new Error("boom");
    });
    ecs.destroy(a);
    ecs.destroy(c);
    expect(() => ecs.flushDestroyed()).toThrow("boom");
    expect(seen).toEqual([
      [a, 42],
      [c, -1],
      [b, -1],
    ]);
    expect(ecs.count).toBe(0);
    off();
    const d = ecs.create();
    ecs.destroy(d);
    expect(ecs.flushDestroyed()).toBe(1);
    expect(seen.length).toBe(3);
  });
});

describe("components", () => {
  it("numeric columns start at their schema values, keep values when added twice, and reset when re-added", () => {
    const ecs = new Ecs();
    const spr = ecs.defineComponent("sprite", { frame: "u16", tint: ["u32", 0xffffffff], scale: ["f32", 1] });
    const e = ecs.create();
    expect(ecs.add(e, spr)).toBe(true);
    const i = indexOf(e);
    expect(spr.tint[i]).toBe(0xffffffff);
    expect(spr.scale[i]).toBe(1);
    expect(spr.frame[i]).toBe(0);
    spr.frame[i] = 9;
    ecs.add(e, spr);
    expect(spr.frame[i]).toBe(9);
    ecs.remove(e, spr);
    expect(ecs.has(e, spr)).toBe(false);
    ecs.add(e, spr);
    expect(spr.frame[i]).toBe(0);
    expect(spr.frame).toBeInstanceOf(Uint16Array);
    expect(spr.tint).toBeInstanceOf(Uint32Array);
  });

  it("columns grow by doubling past capacity and keep their contents", () => {
    const ecs = new Ecs(16);
    const pos = ecs.defineComponent("pos", { x: "f32", y: "f32" });
    const tag = ecs.defineTag("t");
    const before = pos.x;
    expect(ecs.capacity).toBe(16);
    const ids: number[] = [];
    for (let k = 0; k < 1000; k++) {
      const e = ecs.spawn(pos);
      if (k % 3 === 0) ecs.add(e, tag);
      pos.x[indexOf(e)] = k;
      pos.y[indexOf(e)] = -k;
      ids.push(e);
    }
    expect(ecs.capacity).toBe(1024);
    expect(pos.x).not.toBe(before);
    expect(pos.x.length).toBe(1024);
    for (let k = 0; k < 1000; k++) {
      expect(pos.x[indexOf(ids[k])]).toBe(k);
      expect(pos.y[indexOf(ids[k])]).toBe(-k);
      expect(ecs.has(ids[k], tag)).toBe(k % 3 === 0);
    }
    expect(pos.$.size).toBe(1000);
    expect(tag.$.size).toBe(334);
    ecs.reserve(5000);
    expect(ecs.capacity).toBe(8192);
    expect(pos.x[indexOf(ids[999])]).toBe(999);
  });

  it("object components hold values by entity and drop them on destroy", () => {
    const ecs = new Ecs();
    const brain = ecs.defineObject<{ goal: string }>("brain");
    const a = ecs.create(),
      b = ecs.create();
    expect(ecs.set(a, brain, { goal: "flank" })).toBe(true);
    ecs.set(b, brain, { goal: "hold" });
    expect(brain.get(a)?.goal).toBe("flank");
    ecs.set(a, brain, { goal: "charge" });
    expect(brain.get(a)?.goal).toBe("charge");
    expect(brain.$.size).toBe(2);
    ecs.remove(a, brain);
    expect(brain.get(a)).toBeUndefined();
    expect(brain.get(b)?.goal).toBe("hold");
    expect(brain.at(indexOf(b))?.goal).toBe("hold");
    ecs.destroy(b);
    ecs.flushDestroyed();
    expect(brain.get(b)).toBeUndefined();
    expect(brain.$.size).toBe(0);
    expect(ecs.set(b, brain, { goal: "x" })).toBe(false);
  });

  it("rejects duplicate names and the reserved $ field, and finds components by name", () => {
    const ecs = new Ecs();
    const pos = ecs.defineComponent("pos", { x: "f32" });
    expect(() => ecs.defineTag("pos")).toThrow(/already defined/);
    expect(() => ecs.defineComponent("bad", { $: "f32" })).toThrow(/reserved/);
    expect(ecs.component("pos")).toBe(pos);
    expect(ecs.component("nope")).toBeUndefined();
  });

  it("supports more than 32 components (multi-word signatures), defined after entities exist", () => {
    const ecs = new Ecs();
    const early = ecs.defineTag("early");
    const e = ecs.spawn(early);
    const tags: ComponentBase[] = [];
    for (let k = 0; k < 70; k++) tags.push(ecs.defineTag(`t${k}`));
    expect(ecs.words).toBe(3);
    expect(ecs.has(e, early)).toBe(true);
    ecs.add(e, tags[40]);
    ecs.add(e, tags[69]);
    const f = ecs.spawn(tags[40]);
    expect(ecs.query(tags[40], tags[69]).collect()).toBe(1);
    expect(ecs.query(tags[40], tags[69]).idx[0]).toBe(indexOf(e));
    expect(ecs.query(tags[40]).collect()).toBe(2);
    expect(ecs.queryWithout([tags[40]], [early]).collect()).toBe(1);
    expect(ecs.queryWithout([tags[40]], [early]).idx[0]).toBe(indexOf(f));
    ecs.destroy(e);
    ecs.flushDestroyed();
    expect(tags[69].$.size).toBe(0);
    expect(early.$.size).toBe(0);
    for (let w = 0; w < ecs.words; w++) expect(ecs.sig[indexOf(e) * ecs.words + w]).toBe(0);
  });
});

describe("queries", () => {
  const setup = () => {
    const ecs = new Ecs();
    const pos = ecs.defineComponent("pos", { x: "f32", y: "f32" });
    const vel = ecs.defineComponent("vel", { vx: "f32", vy: "f32" });
    const dead = ecs.defineTag("dead");
    return { ecs, pos, vel, dead };
  };

  it("are cached by component set regardless of order", () => {
    const { ecs, pos, vel, dead } = setup();
    expect(ecs.query(pos, vel)).toBe(ecs.query(vel, pos));
    expect(ecs.query(pos)).not.toBe(ecs.query(pos, vel));
    expect(ecs.queryWithout([pos], [dead])).toBe(ecs.queryWithout([pos], [dead]));
    expect(ecs.queryWithout([pos], [dead])).not.toBe(ecs.query(pos));
    expect(() => ecs.query()).toThrow();
  });

  it("match exactly the entities with all components and none of the excluded ones", () => {
    const { ecs, pos, vel, dead } = setup();
    const want = new Set<number>();
    const wantAlive = new Set<number>();
    for (let k = 0; k < 300; k++) {
      const e = ecs.create();
      if (k % 2 === 0) ecs.add(e, pos);
      if (k % 3 === 0) ecs.add(e, vel);
      if (k % 5 === 0) ecs.add(e, dead);
      if (k % 6 === 0) want.add(indexOf(e));
      if (k % 6 === 0 && k % 5 !== 0) wantAlive.add(indexOf(e));
    }
    const q = ecs.query(pos, vel);
    const n = q.collect();
    expect(n).toBe(want.size);
    expect(q.length).toBe(n);
    expect(new Set(Array.from(q.idx.subarray(0, n)))).toEqual(want);
    expect(q.count()).toBe(want.size);
    const qa = ecs.queryWithout([pos, vel], [dead]);
    expect(new Set(Array.from(qa.idx.subarray(0, qa.collect())))).toEqual(wantAlive);
    expect(qa.matches(q.idx[0])).toBe(wantAlive.has(q.idx[0]));
    expect(wantAlive.has(qa.first())).toBe(true);
  });

  it("iterate the smallest set", () => {
    const { ecs, pos, vel } = setup();
    for (let k = 0; k < 200; k++) ecs.spawn(pos);
    const few = [ecs.spawn(pos, vel), ecs.spawn(pos, vel), ecs.spawn(vel)];
    const q = ecs.query(pos, vel);
    expect(q.collect()).toBe(2);
    // In vel's dense order, which is creation order here
    expect(Array.from(q.idx.subarray(0, 2))).toEqual([indexOf(few[0]), indexOf(few[1])]);
  });

  it("snapshot, so structural changes while looping are safe and new entities wait for the next collect", () => {
    const { ecs, pos, vel } = setup();
    for (let k = 0; k < 50; k++) ecs.spawn(pos, vel);
    const q = ecs.query(pos, vel);
    const n = q.collect(),
      idx = q.idx;
    let visited = 0;
    for (let k = 0; k < n; k++) {
      visited++;
      const id = ecs.idOf(idx[k]);
      ecs.remove(id, vel);
      ecs.spawn(pos, vel);
    }
    expect(visited).toBe(50);
    expect(q.collect()).toBe(50);
  });

  it("can skip entities already queued for destruction", () => {
    const { ecs, pos } = setup();
    const ids = [ecs.spawn(pos), ecs.spawn(pos), ecs.spawn(pos)];
    ecs.destroy(ids[1]);
    const q = ecs.query(pos);
    expect(q.collect()).toBe(3);
    expect(q.collect(true)).toBe(2);
    const seen: number[] = [];
    q.each((i) => seen.push(i), true);
    expect(seen).toEqual([indexOf(ids[0]), indexOf(ids[2])]);
  });

  it("grow their buffer as the matching set grows", () => {
    const { ecs, pos } = setup();
    const q = ecs.query(pos);
    q.collect();
    for (let k = 0; k < 5000; k++) ecs.spawn(pos);
    expect(q.collect()).toBe(5000);
    expect(q.idx.length).toBeGreaterThanOrEqual(5000);
    expect(q.idx[4999]).toBe(4999);
  });
});

describe("world", () => {
  it("has the core components with the contract's column types", () => {
    const w = new World({ seed: 3 });
    const c = w.core;
    expect(c.pos.x).toBeInstanceOf(Float32Array);
    expect(c.prev.y).toBeInstanceOf(Float32Array);
    expect(c.vel.vx).toBeInstanceOf(Float32Array);
    expect(c.body.flags).toBeInstanceOf(Uint32Array);
    expect(c.health.max).toBeInstanceOf(Float32Array);
    expect(c.team.id).toBeInstanceOf(Uint8Array);
    expect(c.kind.def).toBeInstanceOf(Uint16Array);
    expect(c.sprite.frame).toBeInstanceOf(Uint16Array);
    expect(c.sprite.layer).toBeInstanceOf(Uint8Array);
    expect(c.sprite.tint).toBeInstanceOf(Uint32Array);
    expect(c.sprite.scale).toBeInstanceOf(Float32Array);
    expect(c.life.t).toBeInstanceOf(Float32Array);
    const e = w.spawn(c.sprite);
    expect(c.sprite.scale[e & INDEX_MASK]).toBe(1);
    expect(c.sprite.tint[e & INDEX_MASK]).toBe(0xffffffff);
  });

  it("keeps typed resources", () => {
    const w = new World();
    w.setRes("gravity", 900);
    expect(w.res<number>("gravity")).toBe(900);
    expect(w.hasRes("gravity")).toBe(true);
    expect(() => w.res("missing")).toThrow(/no resource/);
    expect(w.tryRes("missing")).toBeUndefined();
    expect(w.deleteRes("gravity")).toBe(true);
    expect(w.hasRes("gravity")).toBe(false);
  });
});
