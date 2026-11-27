/**
 * Content definitions by string id, with a dense numeric index for the `kind` column. Indices start at 1, so a
 * `kind` left at 0 means "none" instead of silently pointing at the first def. Freeze after boot: the defs become
 * read-only, which keeps runs deterministic and catches content mutated at runtime.
 */

/** The read-only view handed out after boot. */
export interface ReadonlyRegistry<Def> {
  readonly name: string;
  /** Number of defs; valid indices are 1..size. */
  readonly size: number;
  readonly frozen: boolean;
  /** The def at an index; throws for 0 or an unknown index. */
  get(index: number): Def;
  /** The def at an index, or undefined. */
  tryGet(index: number): Def | undefined;
  /** The def for an id; throws when unknown. */
  byId(id: string): Def;
  /** The index for an id, or 0 when unknown. */
  index(id: string): number;
  /** The id at an index; throws for an unknown index. */
  id(index: number): string;
  has(id: string): boolean;
  /** Defs in index order (index k + 1 is at position k). */
  readonly defs: readonly Def[];
  readonly ids: readonly string[];
}

export class Registry<Def> implements ReadonlyRegistry<Def> {
  private readonly list: Def[] = [];
  private readonly names: string[] = [];
  private readonly lookup = new Map<string, number>();
  private isFrozen = false;

  constructor(readonly name: string) {}

  /** Add a def and return its index (1-based). Ids are unique; nothing can be added once frozen. */
  register(id: string, def: Def): number {
    if (this.isFrozen) throw new Error(`Registry ${this.name}: frozen; "${id}" was registered after boot`);
    if (this.lookup.has(id)) throw new Error(`Registry ${this.name}: "${id}" is already registered`);
    if (this.list.length >= 0xffff) throw new Error(`Registry ${this.name}: full (the kind column is u16)`);
    this.list.push(def);
    this.names.push(id);
    this.lookup.set(id, this.list.length);
    return this.list.length;
  }

  get size(): number {
    return this.list.length;
  }

  get frozen(): boolean {
    return this.isFrozen;
  }

  get defs(): readonly Def[] {
    return this.list;
  }

  get ids(): readonly string[] {
    return this.names;
  }

  get(index: number): Def {
    const d = this.list[index - 1];
    if (d === undefined || index < 1) throw new RangeError(`Registry ${this.name}: no def at index ${index}`);
    return d;
  }

  tryGet(index: number): Def | undefined {
    return index >= 1 ? this.list[index - 1] : undefined;
  }

  byId(id: string): Def {
    const k = this.lookup.get(id);
    if (k === undefined) throw new Error(`Registry ${this.name}: unknown id "${id}"`);
    return this.list[k - 1];
  }

  index(id: string): number {
    return this.lookup.get(id) ?? 0;
  }

  id(index: number): string {
    const n = this.names[index - 1];
    if (n === undefined || index < 1) throw new RangeError(`Registry ${this.name}: no def at index ${index}`);
    return n;
  }

  has(id: string): boolean {
    return this.lookup.has(id);
  }

  /** Stop registration and freeze every def (shallowly); returns the read-only view. */
  freeze(): ReadonlyRegistry<Def> {
    if (!this.isFrozen) {
      this.isFrozen = true;
      for (const d of this.list) if (typeof d === "object" && d !== null) Object.freeze(d);
      Object.freeze(this.list);
      Object.freeze(this.names);
    }
    return this;
  }

  [Symbol.iterator](): IterableIterator<Def> {
    return this.list[Symbol.iterator]();
  }
}
