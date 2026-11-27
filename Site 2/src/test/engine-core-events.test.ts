import { describe, expect, it } from "vitest";
import { EventBus } from "@/engine/core";

interface Hit {
  target: number;
  dmg: number;
}
interface Events {
  hit: Hit;
  kill: number;
  note: string;
}

describe("event bus", () => {
  it("queues until flush and delivers in emission order across types", () => {
    const bus = new EventBus<Events>();
    const log: string[] = [];
    bus.on("kill", (id) => log.push(`kill ${id}`));
    bus.on("note", (s) => log.push(`note ${s}`));
    bus.emit("kill", 1);
    bus.emit("note", "a");
    bus.emit("kill", 2);
    expect(log).toEqual([]);
    expect(bus.pending).toBe(3);
    expect(bus.flush()).toBe(3);
    expect(log).toEqual(["kill 1", "note a", "kill 2"]);
    expect(bus.pending).toBe(0);
    expect(bus.flush()).toBe(0);
  });

  it("drops events nobody listens to and says so", () => {
    const bus = new EventBus<Events>();
    expect(bus.has("kill")).toBe(false);
    expect(bus.emit("kill", 1)).toBe(false);
    expect(bus.pending).toBe(0);
    const off = bus.on("kill", () => {});
    expect(bus.has("kill")).toBe(true);
    expect(bus.emit("kill", 1)).toBe(true);
    off();
    expect(bus.has("kill")).toBe(false);
    // Queued before the unsubscribe: nobody is left to receive it
    expect(bus.flush()).toBe(1);
  });

  it("pools claimed payloads: no new objects once warm, and null without listeners", () => {
    const bus = new EventBus<Events>();
    let made = 0;
    bus.define("hit", () => {
      made++;
      return { target: 0, dmg: 0 };
    });
    expect(bus.claim("hit")).toBeNull();
    const got: string[] = [];
    bus.on("hit", (h) => got.push(`${h.target}:${h.dmg}`));
    for (let round = 0; round < 50; round++) {
      for (let k = 0; k < 4; k++) {
        const h = bus.claim("hit")!;
        h.target = k;
        h.dmg = round;
      }
      bus.flush();
    }
    expect(made).toBe(4);
    expect(got.length).toBe(200);
    expect(got[199]).toBe("3:49");
    expect(() => new EventBus<Events>().claim("kill")).not.toThrow();
    const b2 = new EventBus<Events>();
    b2.on("kill", () => {});
    expect(() => b2.claim("kill")).toThrow(/define/);
  });

  it("delivers events emitted by listeners in the same flush, up to the per-flush cap", () => {
    const bus = new EventBus<Events>();
    const log: number[] = [];
    bus.on("kill", (id) => {
      log.push(id);
      if (id < 5) bus.emit("kill", id + 1);
    });
    bus.emit("kill", 0);
    expect(bus.flush()).toBe(6);
    expect(log).toEqual([0, 1, 2, 3, 4, 5]);

    // A feedback loop is cut at the cap and carries over, in order
    const loop = new EventBus<Events>();
    loop.maxPerFlush = 10;
    let seen = 0;
    loop.on("kill", (id) => {
      seen++;
      loop.emit("kill", id + 1);
    });
    loop.emit("kill", 0);
    expect(loop.flush()).toBe(10);
    expect(loop.pending).toBe(1);
    expect(loop.flush()).toBe(10);
    expect(seen).toBe(20);
  });

  it("lets listeners unsubscribe or subscribe mid-flush without skipping anyone", () => {
    const bus = new EventBus<Events>();
    const log: string[] = [];
    const offA = bus.on("note", (s) => {
      log.push(`a ${s}`);
      offA();
      bus.on("note", (t) => log.push(`c ${t}`));
    });
    bus.on("note", (s) => log.push(`b ${s}`));
    bus.emit("note", "1");
    bus.emit("note", "2");
    bus.flush();
    expect(log).toEqual(["a 1", "b 1", "b 2", "c 2"]);
  });

  it("dispatch delivers synchronously; clear drops the queue; nested flush is a no-op", () => {
    const bus = new EventBus<Events>();
    const log: string[] = [];
    bus.on("note", (s) => {
      log.push(s);
      expect(bus.flush()).toBe(0);
    });
    bus.dispatch("note", "now");
    expect(log).toEqual(["now"]);
    bus.emit("note", "later");
    bus.clear();
    expect(bus.flush()).toBe(0);
    expect(log).toEqual(["now"]);
    bus.emit("note", "x");
    bus.flush();
    expect(bus.emitted).toBe(2);
    expect(bus.delivered).toBe(1);
  });

  it("isolates throwing listeners when onError is set", () => {
    const bus = new EventBus<Events>();
    const log: string[] = [];
    bus.on("note", () => {
      throw new Error("x");
    });
    bus.on("note", (s) => log.push(s));
    bus.emit("note", "a");
    expect(() => bus.flush()).toThrow("x");
    bus.onError = (err, type) => log.push(`${String(type)}: ${(err as Error).message}`);
    bus.emit("note", "b");
    bus.flush();
    expect(log).toEqual(["note: x", "b"]);
  });
});
