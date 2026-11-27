/**
 * A typed event bus. `emit` queues and `flush` delivers in emission order (the scheduler flushes between
 * phases), so a system's events reach listeners at a predictable point in the tick. Events nobody listens to are
 * dropped on the spot, and `claim` hands out pooled payloads, so steady-state traffic allocates nothing.
 */

/**
 * Payloads by event type. Game code adds its own types with declaration merging:
 * `declare module "@/engine/core/events" { interface WorldEvents { kill: KillEvent } }`.
 */
export interface WorldEvents {
  /** An entity was destroyed at the end of a tick (its id, no longer alive). */
  destroyed: number;
}

type Listener = (payload: never) => void;

interface Channel {
  /** Copy-on-write, so (un)subscribing while a flush is delivering is safe. */
  listeners: Listener[];
  make: (() => unknown) | null;
  pool: unknown[];
}

export class EventBus<M extends object = WorldEvents> {
  /** Events delivered per flush at most; the rest wait, so a listener feedback loop can't stall a tick. */
  maxPerFlush = 100_000;
  /** When set, a throwing listener is reported here and the others still run; otherwise the error propagates. */
  onError: ((err: unknown, type: keyof M) => void) | null = null;
  /** Counters for the debug overlay. */
  emitted = 0;
  delivered = 0;

  private readonly channels = new Map<keyof M, Channel>();
  private qType: (keyof M)[] = [];
  private qChan: (Channel | null)[] = [];
  private qPayload: unknown[] = [];
  private qPooled: boolean[] = [];
  private head = 0;
  private tail = 0;
  private flushing = false;

  private channel(type: keyof M): Channel {
    let ch = this.channels.get(type);
    if (!ch) {
      ch = { listeners: [], make: null, pool: [] };
      this.channels.set(type, ch);
    }
    return ch;
  }

  /** Subscribe; returns the unsubscribe function. */
  on<K extends keyof M>(type: K, fn: (payload: M[K]) => void): () => void {
    const ch = this.channel(type);
    ch.listeners = [...ch.listeners, fn as Listener];
    return () => {
      ch.listeners = ch.listeners.filter((l) => l !== fn);
    };
  }

  /** Whether anyone listens: guard payload construction with it for hot event types. */
  has(type: keyof M): boolean {
    const ch = this.channels.get(type);
    return ch !== undefined && ch.listeners.length > 0;
  }

  /** Queue an event for the next flush. Returns false (and keeps nothing) when nobody listens. */
  emit<K extends keyof M>(type: K, payload: M[K]): boolean {
    const ch = this.channels.get(type);
    if (ch === undefined || ch.listeners.length === 0) return false;
    this.push(type, ch, payload, false);
    return true;
  }

  /** Let `claim` pool payloads of this type, built by `make`. */
  define<K extends keyof M>(type: K, make: () => M[K]): void {
    this.channel(type).make = make;
  }

  /**
   * A pooled payload already queued for delivery: fill in its fields. Null when nobody listens. The object goes
   * back to the pool after delivery, so listeners must copy what they keep.
   */
  claim<K extends keyof M>(type: K): M[K] | null {
    const ch = this.channels.get(type);
    if (ch === undefined || ch.listeners.length === 0) return null;
    if (!ch.make) throw new Error(`EventBus: define("${String(type)}") before claiming it`);
    const p = ch.pool.length > 0 ? ch.pool.pop() : ch.make();
    this.push(type, ch, p, true);
    return p as M[K];
  }

  /** Deliver immediately, bypassing the queue. */
  dispatch<K extends keyof M>(type: K, payload: M[K]): void {
    const ch = this.channels.get(type);
    if (ch !== undefined && ch.listeners.length > 0) this.deliver(type, ch.listeners, payload);
  }

  private push(type: keyof M, ch: Channel, payload: unknown, pooled: boolean): void {
    const k = this.tail++;
    this.qType[k] = type;
    this.qChan[k] = ch;
    this.qPayload[k] = payload;
    this.qPooled[k] = pooled;
    this.emitted++;
  }

  private deliver(type: keyof M, listeners: Listener[], payload: unknown): void {
    for (let j = 0; j < listeners.length; j++) {
      if (this.onError === null) (listeners[j] as (p: unknown) => void)(payload);
      else {
        try {
          (listeners[j] as (p: unknown) => void)(payload);
        } catch (err) {
          this.onError(err, type);
        }
      }
    }
  }

  /** Events waiting for the next flush. */
  get pending(): number {
    return this.tail - this.head;
  }

  /**
   * Deliver queued events in order, including ones listeners emit meanwhile, up to `maxPerFlush`. Returns how many
   * were delivered. A nested call from inside a listener does nothing: the outer flush carries on.
   */
  flush(): number {
    if (this.flushing || this.head === this.tail) return 0;
    this.flushing = true;
    let n = 0;
    try {
      while (this.head < this.tail && n < this.maxPerFlush) {
        const k = this.head++;
        const ch = this.qChan[k] as Channel;
        const payload = this.qPayload[k];
        this.qChan[k] = null;
        this.qPayload[k] = undefined;
        n++;
        this.deliver(this.qType[k], ch.listeners, payload);
        if (this.qPooled[k]) ch.pool.push(payload);
      }
    } finally {
      this.compact();
      this.delivered += n;
      this.flushing = false;
    }
    return n;
  }

  /** Drop everything queued. */
  clear(): void {
    for (let k = this.head; k < this.tail; k++) {
      if (this.qPooled[k]) (this.qChan[k] as Channel).pool.push(this.qPayload[k]);
      this.qChan[k] = null;
      this.qPayload[k] = undefined;
    }
    this.head = this.tail = 0;
  }

  private compact(): void {
    if (this.head === this.tail) {
      this.head = this.tail = 0;
      return;
    }
    // Over the cap: slide the leftovers to the front so the queue doesn't creep
    const n = this.tail - this.head;
    for (let k = 0; k < n; k++) {
      const from = this.head + k;
      this.qType[k] = this.qType[from];
      this.qChan[k] = this.qChan[from];
      this.qPayload[k] = this.qPayload[from];
      this.qPooled[k] = this.qPooled[from];
    }
    for (let k = n; k < this.tail; k++) {
      this.qChan[k] = null;
      this.qPayload[k] = undefined;
    }
    this.head = 0;
    this.tail = n;
  }
}
