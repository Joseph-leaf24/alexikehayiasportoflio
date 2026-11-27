/**
 * A data-oriented ECS. Entities are generational ids; hot components are struct-of-arrays typed columns indexed
 * by entity index (`pos.x[index]`); cold components (tags and plain objects) only keep membership and values.
 * Every component's membership is a sparse set, and each entity carries a bitmask signature, so a query walks the
 * smallest set and tests the others with one AND per 32 components.
 */

import { nextPow2 } from "./math";

/** Index in the low 20 bits, generation above: up to 1,048,576 live entities. */
export const INDEX_BITS = 20;
export const INDEX_MASK = (1 << INDEX_BITS) - 1;
export const MAX_ENTITIES = 1 << INDEX_BITS;
/** Ten generation bits keep every id below 2^30, a small integer in V8 even with pointer compression. */
export const GEN_BITS = 10;
/** Generations run 1..GEN_MAX, so no live id is ever 0. */
export const GEN_MAX = (1 << GEN_BITS) - 1;
/** "No entity": safe to store in any integer column. */
export const NULL_ENTITY = 0;

export const indexOf = (id: number): number => id & INDEX_MASK;
export const genOf = (id: number): number => id >>> INDEX_BITS;

/** Bits of `Ecs.state`. */
export const ENTITY_ALIVE = 1;
export const ENTITY_DOOMED = 2;
const ALIVE = ENTITY_ALIVE;
const DOOMED = ENTITY_DOOMED;
/** A freed index is reused only once this many are free, so a stale id needs ~a million destroys to alias. */
const REUSE_AFTER = 1024;

export type FieldType = "f32" | "f64" | "i8" | "u8" | "i16" | "u16" | "i32" | "u32";
interface ArrayTypes {
  f32: Float32Array;
  f64: Float64Array;
  i8: Int8Array;
  u8: Uint8Array;
  i16: Int16Array;
  u16: Uint16Array;
  i32: Int32Array;
  u32: Uint32Array;
}
export type TypedColumn = ArrayTypes[FieldType];
/** A field type, or a type with the value a fresh component starts with (otherwise 0). */
export type FieldSpec = FieldType | readonly [FieldType, number];
export type Schema = { readonly [field: string]: FieldSpec };
type TypeOfField<F> = F extends FieldType ? F : F extends readonly [infer T, number] ? T : never;
export type Columns<S extends Schema> = { [K in keyof S]: ArrayTypes[TypeOfField<S[K]> & FieldType] };

const CTORS: { [K in FieldType]: new (n: number) => ArrayTypes[K] } = {
  f32: Float32Array,
  f64: Float64Array,
  i8: Int8Array,
  u8: Uint8Array,
  i16: Int16Array,
  u16: Uint16Array,
  i32: Int32Array,
  u32: Uint32Array,
};

export type ComponentKind = "num" | "tag" | "obj";

/** One component's membership (a sparse set over entity indices) and storage bookkeeping. */
export class ComponentMeta {
  /** Member entity indices, packed in [0, size). */
  dense: Int32Array;
  /** Position in `dense` per entity index, or -1. */
  sparse: Int32Array;
  size = 0;
  readonly word: number;
  readonly bit: number;
  /** Numeric components: field names, types, start values and the live columns, in schema order. */
  readonly fields: string[] = [];
  readonly types: FieldType[] = [];
  readonly defaults: number[] = [];
  columns: TypedColumn[] = [];
  /** Object components: values parallel to `dense`. */
  readonly values: unknown[] | null;
  /** The public handle whose column properties are swapped when storage grows. */
  handle: Record<string, unknown> | null = null;

  constructor(
    readonly name: string,
    readonly cid: number,
    readonly kind: ComponentKind,
    capacity: number
  ) {
    this.word = cid >>> 5;
    this.bit = (1 << (cid & 31)) | 0;
    this.dense = new Int32Array(capacity);
    this.sparse = new Int32Array(capacity).fill(-1);
    this.values = kind === "obj" ? [] : null;
  }

  /** Membership by entity index (no generation check; use `world.has` for ids). */
  hasIndex(index: number): boolean {
    return this.sparse[index] >= 0;
  }

  insert(index: number): void {
    const p = this.size++;
    this.dense[p] = index;
    this.sparse[index] = p;
    const cols = this.columns;
    for (let k = 0; k < cols.length; k++) cols[k][index] = this.defaults[k];
  }

  removeIndex(index: number): void {
    const p = this.sparse[index];
    const last = --this.size;
    const moved = this.dense[last];
    this.dense[p] = moved;
    this.sparse[moved] = p;
    this.sparse[index] = -1;
    const values = this.values;
    if (values) {
      values[p] = values[last];
      values[last] = undefined;
    }
  }

  grow(capacity: number): void {
    const dense = new Int32Array(capacity);
    dense.set(this.dense);
    this.dense = dense;
    const sparse = new Int32Array(capacity).fill(-1);
    sparse.set(this.sparse);
    this.sparse = sparse;
    for (let k = 0; k < this.columns.length; k++) {
      const col = new CTORS[this.types[k]](capacity);
      col.set(this.columns[k]);
      this.columns[k] = col;
      if (this.handle) this.handle[this.fields[k]] = col;
    }
  }
}

export interface ComponentBase {
  readonly $: ComponentMeta;
}
/** A hot component: `pos.x[index]`. Column arrays are replaced when the world grows, so re-read them per run. */
export type Component<S extends Schema = Schema> = ComponentBase & Columns<S>;
export type Tag = ComponentBase;
export interface ObjectComponent<T> extends ComponentBase {
  /** The value for a live entity id, or undefined. */
  get(id: number): T | undefined;
  /** The value by entity index (no generation check). */
  at(index: number): T | undefined;
}

type DestroyHook = (id: number, index: number) => void;

/** Entity and component storage, membership and queries. `World` extends it with time, events and systems. */
export class Ecs {
  capacity: number;
  /** Current generation per entity index. */
  gen: Uint16Array;
  /** ALIVE / DOOMED bits per entity index. */
  state: Uint8Array;
  /** Component bitmask per entity: `sig[index * words + word]`. */
  sig: Int32Array;
  words = 1;
  readonly components: ComponentBase[] = [];

  private free: Int32Array;
  private freeHead = 0;
  private freeCount = 0;
  private top = 0;
  private live = 0;
  private pending: number[] = [];
  private pendingCount = 0;
  private flushing = false;
  private readonly byName = new Map<string, ComponentBase>();
  private readonly queryCache = new Map<string, Query>();
  private readonly destroyHooks: DestroyHook[] = [];

  constructor(capacity = 1024) {
    this.capacity = Math.min(MAX_ENTITIES, nextPow2(Math.max(16, capacity)));
    this.gen = new Uint16Array(this.capacity);
    this.state = new Uint8Array(this.capacity);
    this.sig = new Int32Array(this.capacity);
    this.free = new Int32Array(this.capacity);
  }

  /* Entities */

  /** A new entity with no components. */
  create(): number {
    let i: number;
    if (this.freeCount > REUSE_AFTER || (this.top >= MAX_ENTITIES && this.freeCount > 0)) {
      i = this.free[this.freeHead];
      this.freeHead = (this.freeHead + 1) & (this.capacity - 1);
      this.freeCount--;
    } else {
      if (this.top >= MAX_ENTITIES) throw new RangeError(`ECS: more than ${MAX_ENTITIES} live entities`);
      if (this.top === this.capacity) this.grow(this.top + 1);
      i = this.top++;
      this.gen[i] = 1;
    }
    this.state[i] = ALIVE;
    this.live++;
    return (this.gen[i] << INDEX_BITS) | i;
  }

  /** A new entity with these components (numeric ones start at their schema values). */
  spawn(...components: ComponentBase[]): number {
    const id = this.create();
    for (let k = 0; k < components.length; k++) this.add(id, components[k]);
    return id;
  }

  /** Queue an entity for destruction at the end of the tick; it stays alive and readable until then. */
  destroy(id: number): boolean {
    if (!this.alive(id)) return false;
    const i = id & INDEX_MASK;
    if (this.state[i] & DOOMED) return false;
    this.state[i] |= DOOMED;
    this.pending[this.pendingCount++] = id;
    return true;
  }

  /** Doom every live entity (a level reset); they go at the next flush. */
  destroyAll(): number {
    let n = 0;
    for (let i = 0; i < this.top; i++)
      if (this.state[i] === ALIVE && this.destroy((this.gen[i] << INDEX_BITS) | i)) n++;
    return n;
  }

  alive(id: number): boolean {
    const i = id & INDEX_MASK;
    return id > 0 && i < this.top && this.gen[i] === id >>> INDEX_BITS && (this.state[i] & ALIVE) !== 0;
  }

  /** Alive but queued for destruction this tick. */
  doomed(id: number): boolean {
    return this.alive(id) && (this.state[id & INDEX_MASK] & DOOMED) !== 0;
  }

  /** The live id at an entity index, or NULL_ENTITY. */
  idOf(index: number): number {
    return index >= 0 && index < this.top && this.state[index] & ALIVE ? (this.gen[index] << INDEX_BITS) | index : 0;
  }

  /** Live entities (including doomed ones not yet flushed). */
  get count(): number {
    return this.live;
  }

  /** One past the highest entity index ever used: iterate [0, indexLimit) to visit every slot. */
  get indexLimit(): number {
    return this.top;
  }

  get pendingDestroys(): number {
    return this.pendingCount;
  }

  /** Runs for each entity as it is destroyed, before its components are removed, so it can read them. */
  onDestroy(hook: DestroyHook): () => void {
    this.destroyHooks.push(hook);
    return () => {
      const k = this.destroyHooks.indexOf(hook);
      if (k >= 0) this.destroyHooks.splice(k, 1);
    };
  }

  /**
   * Destroy everything queued (hooks may queue more; they go in the same flush). Returns how many went. A hook
   * that throws doesn't stop the flush: every queued entity is still destroyed, then the first error is rethrown.
   */
  flushDestroyed(): number {
    if (this.flushing) return 0;
    this.flushing = true;
    let n = 0;
    let failed = false;
    let failure: unknown;
    try {
      for (let k = 0; k < this.pendingCount; k++) {
        const id = this.pending[k];
        const i = id & INDEX_MASK;
        for (let h = 0; h < this.destroyHooks.length; h++) {
          try {
            this.destroyHooks[h](id, i);
          } catch (err) {
            if (!failed) failure = err;
            failed = true;
          }
        }
        const W = this.words,
          sig = this.sig,
          base = i * W;
        for (let w = 0; w < W; w++) {
          let bits = sig[base + w];
          while (bits !== 0) {
            const low = bits & -bits;
            this.components[(w << 5) + 31 - Math.clz32(low)].$.removeIndex(i);
            bits ^= low;
          }
          sig[base + w] = 0;
        }
        this.state[i] = 0;
        this.gen[i] = this.gen[i] >= GEN_MAX ? 1 : this.gen[i] + 1;
        this.free[(this.freeHead + this.freeCount) & (this.capacity - 1)] = i;
        this.freeCount++;
        this.live--;
        n++;
      }
    } finally {
      this.pendingCount = 0;
      this.flushing = false;
    }
    if (failed) throw failure;
    return n;
  }

  /** Make room for at least `n` entity indices up front, so columns never reallocate mid-run. */
  reserve(n: number): void {
    if (n > this.capacity) this.grow(n);
  }

  private grow(min: number): void {
    let cap = this.capacity;
    while (cap < min) cap *= 2;
    cap = Math.min(cap, MAX_ENTITIES);
    const gen = new Uint16Array(cap);
    gen.set(this.gen);
    this.gen = gen;
    const state = new Uint8Array(cap);
    state.set(this.state);
    this.state = state;
    const sig = new Int32Array(cap * this.words);
    sig.set(this.sig);
    this.sig = sig;
    // Unroll the free ring so it starts at 0 in the bigger buffer
    const free = new Int32Array(cap);
    for (let k = 0; k < this.freeCount; k++) free[k] = this.free[(this.freeHead + k) & (this.capacity - 1)];
    this.free = free;
    this.freeHead = 0;
    for (const c of this.components) c.$.grow(cap);
    for (const q of this.queryCache.values()) q.idx = new Int32Array(cap);
    this.capacity = cap;
  }

  /* Components */

  private register(name: string, kind: ComponentKind): ComponentMeta {
    if (this.byName.has(name)) throw new Error(`ECS: component "${name}" is already defined`);
    const cid = this.components.length;
    const words = (cid >>> 5) + 1;
    if (words > this.words) {
      // Re-stride the signatures for one more word per entity
      const sig = new Int32Array(this.capacity * words);
      for (let i = 0; i < this.top; i++)
        for (let w = 0; w < this.words; w++) sig[i * words + w] = this.sig[i * this.words + w];
      this.sig = sig;
      this.words = words;
    }
    return new ComponentMeta(name, cid, kind, this.capacity);
  }

  private publish<C extends ComponentBase>(c: C): C {
    this.components.push(c);
    this.byName.set(c.$.name, c);
    return c;
  }

  /** A hot component of typed columns: `defineComponent("pos", { x: "f32", y: "f32" })` gives `pos.x[index]`. */
  defineComponent<const S extends Schema>(name: string, schema: S): Component<S> {
    const meta = this.register(name, "num");
    const handle: Record<string, unknown> = { $: meta };
    for (const field of Object.keys(schema)) {
      if (field === "$") throw new Error(`ECS: "$" is reserved (component "${name}")`);
      const spec = schema[field];
      const type: FieldType = typeof spec === "string" ? spec : spec[0];
      const col = new CTORS[type](this.capacity);
      meta.fields.push(field);
      meta.types.push(type);
      meta.defaults.push(typeof spec === "string" ? 0 : spec[1]);
      meta.columns.push(col);
      handle[field] = col;
    }
    meta.handle = handle;
    return this.publish(handle as unknown as Component<S>);
  }

  /** A membership-only component (a flag such as "boss" or "marked"). */
  defineTag(name: string): Tag {
    return this.publish({ $: this.register(name, "tag") });
  }

  /** A cold component holding one plain object per entity, set with `world.set`. */
  defineObject<T>(name: string): ObjectComponent<T> {
    const meta = this.register(name, "obj");
    const values = meta.values as unknown[];
    return this.publish<ObjectComponent<T>>({
      $: meta,
      get: (id: number) =>
        this.alive(id) && meta.sparse[id & INDEX_MASK] >= 0
          ? (values[meta.sparse[id & INDEX_MASK]] as T | undefined)
          : undefined,
      at: (index: number) => (meta.sparse[index] >= 0 ? (values[meta.sparse[index]] as T | undefined) : undefined),
    });
  }

  /** A component by name (for content that refers to components in data). */
  component<C extends ComponentBase = ComponentBase>(name: string): C | undefined {
    return this.byName.get(name) as C | undefined;
  }

  /** Give an entity a component. Numeric fields start at their schema values; adding twice keeps the values. */
  add(id: number, c: ComponentBase): boolean {
    if (!this.alive(id)) return false;
    const i = id & INDEX_MASK,
      m = c.$;
    if (m.sparse[i] >= 0) return true;
    m.insert(i);
    this.sig[i * this.words + m.word] |= m.bit;
    return true;
  }

  /** Set an object component's value, adding the component if needed. */
  set<T>(id: number, c: ObjectComponent<T>, value: T): boolean {
    if (!this.add(id, c)) return false;
    const m = c.$;
    (m.values as unknown[])[m.sparse[id & INDEX_MASK]] = value;
    return true;
  }

  remove(id: number, c: ComponentBase): boolean {
    if (!this.alive(id)) return false;
    const i = id & INDEX_MASK,
      m = c.$;
    if (m.sparse[i] < 0) return false;
    m.removeIndex(i);
    this.sig[i * this.words + m.word] &= ~m.bit;
    return true;
  }

  has(id: number, c: ComponentBase): boolean {
    return this.alive(id) && c.$.sparse[id & INDEX_MASK] >= 0;
  }

  /* Queries */

  /** The cached query over entities that have every listed component. Hold on to it: it is reusable. */
  query(...all: ComponentBase[]): Query {
    return this.queryWithout(all, []);
  }

  /** The cached query over entities with all of `all` and none of `none`. */
  queryWithout(all: readonly ComponentBase[], none: readonly ComponentBase[]): Query {
    if (all.length === 0) throw new Error("ECS: a query needs at least one component");
    const key =
      all
        .map((c) => c.$.cid)
        .sort((a, b) => a - b)
        .join(",") +
      "|" +
      none
        .map((c) => c.$.cid)
        .sort((a, b) => a - b)
        .join(",");
    let q = this.queryCache.get(key);
    if (!q) {
      q = new Query(this, all, none);
      this.queryCache.set(key, q);
    }
    return q;
  }
}

/**
 * Entities matching a component set. `collect()` snapshots the matching entity indices into `idx`, so systems can
 * add, remove and destroy freely while looping; entities created meanwhile are not visited.
 */
export class Query {
  /**
   * Matching entity indices from the last `collect()`. Sized to the world's capacity, so like the component
   * columns it is only replaced when the world grows: read it per run, don't keep it across ticks.
   */
  idx: Int32Array;
  /** How many entries of `idx` the last `collect()` filled. */
  length = 0;
  private readonly all: ComponentMeta[];
  private readonly reqWords: number[] = [];
  private readonly reqMasks: number[] = [];
  private readonly noneWords: number[] = [];
  private readonly noneMasks: number[] = [];

  constructor(
    private readonly ecs: Ecs,
    all: readonly ComponentBase[],
    none: readonly ComponentBase[]
  ) {
    this.idx = new Int32Array(ecs.capacity);
    this.all = all.map((c) => c.$);
    Query.masks(all, this.reqWords, this.reqMasks);
    Query.masks(none, this.noneWords, this.noneMasks);
  }

  private static masks(list: readonly ComponentBase[], words: number[], masks: number[]): void {
    for (const c of list) {
      const k = words.indexOf(c.$.word);
      if (k >= 0) masks[k] |= c.$.bit;
      else {
        words.push(c.$.word);
        masks.push(c.$.bit);
      }
    }
  }

  /** Whether an entity index matches. */
  matches(index: number): boolean {
    const sig = this.ecs.sig,
      base = index * this.ecs.words;
    for (let k = 0; k < this.reqWords.length; k++) {
      const m = this.reqMasks[k];
      if ((sig[base + this.reqWords[k]] & m) !== m) return false;
    }
    for (let k = 0; k < this.noneWords.length; k++)
      if ((sig[base + this.noneWords[k]] & this.noneMasks[k]) !== 0) return false;
    return true;
  }

  private smallest(): ComponentMeta {
    let s = this.all[0];
    for (let k = 1; k < this.all.length; k++) if (this.all[k].size < s.size) s = this.all[k];
    return s;
  }

  /** Fill `idx` with the matching entity indices (in the smallest set's order) and return how many. */
  collect(skipDoomed = false): number {
    const set = this.smallest();
    const n = set.size,
      dense = set.dense;
    const out = this.idx,
      ecs = this.ecs,
      sig = ecs.sig,
      W = ecs.words,
      state = ecs.state;
    let count = 0;
    if (this.reqWords.length === 1 && this.noneWords.length <= 1) {
      // One signature word to test (the common case): no inner loop
      const rw = this.reqWords[0],
        rm = this.reqMasks[0];
      const nw = this.noneWords.length ? this.noneWords[0] : 0,
        nm = this.noneWords.length ? this.noneMasks[0] : 0;
      for (let k = 0; k < n; k++) {
        const i = dense[k],
          base = i * W;
        if ((sig[base + rw] & rm) !== rm || (sig[base + nw] & nm) !== 0) continue;
        if (skipDoomed && state[i] & DOOMED) continue;
        out[count++] = i;
      }
    } else {
      for (let k = 0; k < n; k++) {
        const i = dense[k];
        if (!this.matches(i) || (skipDoomed && state[i] & DOOMED)) continue;
        out[count++] = i;
      }
    }
    this.length = count;
    return count;
  }

  /** Call `fn` with each matching entity index (a snapshot, as in `collect`). */
  each(fn: (index: number) => void, skipDoomed = false): number {
    const n = this.collect(skipDoomed),
      idx = this.idx;
    for (let k = 0; k < n; k++) fn(idx[k]);
    return n;
  }

  /** How many entities match right now (no snapshot). */
  count(): number {
    const set = this.smallest();
    let c = 0;
    for (let k = 0; k < set.size; k++) if (this.matches(set.dense[k])) c++;
    return c;
  }

  /** The first matching entity index, or -1. */
  first(): number {
    const set = this.smallest();
    for (let k = 0; k < set.size; k++) if (this.matches(set.dense[k])) return set.dense[k];
    return -1;
  }
}
