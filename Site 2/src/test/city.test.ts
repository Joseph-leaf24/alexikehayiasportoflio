import { describe, expect, it } from "vitest";
import {
  BASE_ROW,
  createLevel,
  crustOf,
  extend,
  kindAt,
  LADDER,
  RAIL,
  REINFORCED,
  TILE,
  TILE_HP,
  WINDOW,
  cell,
} from "@/components/pixel/level";
import { ROTATION, THEMES, type Theme } from "@/components/pixel/themes";
import { CAR_LEN, CAR_SIZE, Game, lightState, WAVES, type View } from "@/components/pixel/game";

/** Step a game for `seconds` with the camera on the squad or whatever they're riding in. `each` runs every step. */
function play(g: Game, seconds: number, each?: (view: View) => void) {
  const view: View = { x: 0, y: 0, w: 480, h: 300 };
  for (let i = 0; i < seconds * 30; i++) {
    view.x = Math.max(g.level.start * TILE, g.advance - view.w * 0.42);
    const ride = g.carrier;
    const up = g.players.filter((p) => !p.dead && !p.chute && !p.aboard);
    view.y =
      (ride
        ? ride.y + (ride.air ? 58 : 0)
        : up.length
          ? up.reduce((s, p) => s + p.y, 0) / up.length
          : BASE_ROW * TILE) -
      view.h * 0.68;
    g.step(1 / 30, view);
    g.fresh.length = 0;
    g.dirty.length = 0;
    each?.(view);
  }
}

/** A city of nothing but metro lines and some flat ground. */
const metroCity: Theme = { ...THEMES.city, features: { metro: 3, flat: 1 } };
const yards: Theme = { ...THEMES.industry, features: { railyard: 3, flat: 1 } };

describe("the cities and the industrial lands", () => {
  it("go on much longer than the rest, and the Industrial zone is in the rotation", () => {
    const len = (key: string) =>
      [1, 2, 3, 4, 5].reduce((s, seed) => s + createLevel(THEMES[key], seed).plan.length, 0) / 5;
    expect(len("city")).toBeGreaterThan(len("jungle") * 2.2);
    expect(len("megacity")).toBeGreaterThan(len("jungle") * 2.2);
    expect(len("industry")).toBeGreaterThan(len("jungle") * 2.4);
    expect(len("refinery")).toBeGreaterThan(len("jungle") * 1.8);
    expect(ROTATION).toContain("industry");
  });

  it("build the city up block after block, tall towers standing behind the streets", () => {
    const L = createLevel({ ...THEMES.city, features: { metropolis: 1 } }, 2, { boss: null, length: 4000 });
    extend(L, 300);
    const signs = L.props.filter((p) => p.kind === "sign").length;
    expect(signs).toBeGreaterThan(8);
    // Windows lit in the back walls of the towers behind
    let back = 0;
    for (let c = L.start; c < L.next; c++)
      for (let r = 0; r < BASE_ROW; r++) if (L.bg[cell(c, r)] === WINDOW && !L.fg[cell(c, r)]) back++;
    expect(back).toBeGreaterThan(30);
  });
});

describe("the metro", () => {
  it("runs a tunnel under the city from station to station, a train in it, stairwells up to the street", () => {
    const L = createLevel(metroCity, 3, { boss: null, length: 6000 });
    // A set piece at a time, up to the first metro line: they're long, and the ring only holds so much
    const spawns: ReturnType<typeof extend> = [];
    while (!spawns.some((s) => s.kind === "metro") && L.next < 2000) spawns.push(...extend(L, L.next + 1));
    const trains = spawns.filter((s) => s.kind === "metro");
    expect(trains.length).toBeGreaterThan(0);
    const [lo, hi, ...stops] = trains[0].extra!;
    expect(stops.length).toBeGreaterThanOrEqual(2);
    expect(hi - lo).toBeGreaterThan(300);
    // A tunnel with its rails all the way along, under the street
    const tf = trains[0].y / TILE;
    for (let x = lo / TILE + 1; x < hi / TILE - 1; x++) {
      expect(kindAt(L, x, tf - 1), `column ${x}`).toBe(RAIL);
      expect(tf - crustOf(L, x)).toBeGreaterThan(10);
    }
    // Every station's stairwell has its ladder up to the street; signs with their names
    for (const s of stops) {
      let ladder = 0;
      for (let x = s / TILE - 9; x < s / TILE + 9; x++)
        for (let r = tf - 12; r < tf; r++) if (kindAt(L, x, r) === LADDER) ladder++;
      expect(ladder).toBeGreaterThan(5);
    }
    const names = L.props
      .filter((p) => p.kind === "sign" && Math.abs(p.y - (tf - 4) * TILE) < 2 * TILE)
      .map((p) => (p as { text: string }).text);
    expect(names.length).toBeGreaterThanOrEqual(stops.length);
  });

  it("has its trains running from station to station, stopping at each with the doors open, and back from the end of the line", () => {
    const g = new Game(metroCity, { seed: 3, waves: false, gunship: false, level: { boss: null, length: 6000 } });
    let moving = 0;
    let open = 0;
    const turned = new Set<object>();
    const dirs = new Map<object, number>();
    play(g, 60, () => {
      for (const m of g.metros) {
        if (m.v > 40) moving++;
        if (m.doors >= 1 && m.v === 0) open++;
        if (dirs.has(m) && dirs.get(m) !== m.dir) turned.add(m);
        dirs.set(m, m.dir);
      }
    });
    expect(moving).toBeGreaterThan(30 * 20);
    expect(open).toBeGreaterThan(30 * 5);
    expect(turned.size).toBeGreaterThan(0);
  });

  it("takes the squad down into it on some lines and along the street on others, and they ride its trains", () => {
    let under = 0;
    let street = 0;
    let rides = 0;
    let riding = 0;
    for (const seed of [2, 3]) {
      const g = new Game(metroCity, { seed, waves: false, gunship: false, level: { boss: null, length: 6000 } });
      let was = false;
      play(g, 110, () => {
        const up = g.players.filter((p) => !p.dead && !p.chute && !p.aboard);
        const lines = g.metros.filter((m) => up.length && up[0].x > m.lo && up[0].x < m.hi);
        if (lines.length) {
          if (up.filter((p) => p.y > crustOf(g.level, Math.floor(p.x / TILE)) * TILE + 60).length > up.length / 2)
            under++;
          else street++;
        }
        if (g.ride) {
          riding++;
          // All hidden aboard it, and the squad's advance goes with it
          for (const id of g.crew) expect(Math.abs(g.players[id].x - g.ride.x)).toBeLessThan(70);
        }
        if (g.ride && !was) rides++;
        was = !!g.ride;
      });
    }
    expect(under).toBeGreaterThan(30 * 10);
    expect(street).toBeGreaterThan(30 * 10);
    expect(rides).toBeGreaterThan(0);
    expect(riding).toBeGreaterThan(30 * 4);
  }, 40_000);
});

describe("the railways", () => {
  it("run freight trains along the main line, out of one tunnel and into the other, one after another", () => {
    const g = new Game(yards, { seed: 2, waves: false, gunship: false, level: { boss: null, length: 6000 } });
    let moved = 0;
    let gone = 0;
    const seen = new Map<object, number>();
    play(g, 60, () => {
      for (const f of g.freights) {
        if (f.wait > 0) gone++;
        const x = seen.get(f);
        if (x !== undefined && Math.abs(f.x - x) > 200) moved++;
        if (x === undefined) seen.set(f, f.x);
        // A locomotive at the front, wagons behind
        expect(f.cars[0]).toBe(0);
        expect(f.cars.length).toBeGreaterThan(4);
      }
    });
    expect(g.freights.length).toBeGreaterThan(0);
    expect(moved).toBeGreaterThan(0);
    expect(gone).toBeGreaterThan(0);
    // Adjoining stretches of line make one long line
    for (const f of g.freights) expect(f.hi - f.lo).toBeGreaterThan(CAR_LEN * 6);
  });

  it("send an armoured train up and down the line, running in on the squad and backing off", () => {
    const g = new Game(
      { ...yards, enemies: 3 },
      { seed: 2, waves: false, gunship: false, level: { boss: null, length: 6000 } }
    );
    let left = 0;
    let right = 0;
    play(g, 60, () => {
      for (const e of g.enemies) {
        if (e.kind !== "train" || !e.alive) continue;
        if (e.vx < 0) left++;
        else if (e.vx > 0) right++;
      }
    });
    expect(left).toBeGreaterThan(30);
    expect(right).toBeGreaterThan(30);
  });
});

describe("the battle", () => {
  it("comes in waves: lulls, skirmishes, assaults and onslaughts with three times the enemy on screen", () => {
    // Over a couple of long battles: how the tide turns is down to chance
    const seen = WAVES.map(() => ({ t: 0, n: 0 }));
    for (const seed of [1, 2]) {
      const g = new Game(THEMES.city, { seed, gunship: false, level: { boss: null, length: 6000 } });
      play(g, 240, (view) => {
        const here = g.enemies.filter((e) => e.alive && e.x > view.x && e.x < view.x + view.w).length;
        seen[g.wave].t++;
        seen[g.wave].n += here;
      });
    }
    for (const [i, w] of seen.entries()) expect(w.t, WAVES[i].name).toBeGreaterThan(0);
    const avg = seen.map((w) => w.n / Math.max(1, w.t));
    expect(avg[0]).toBeLessThan(avg[1]);
    expect(avg[2]).toBeGreaterThan(avg[1] * 1.4);
    expect(avg[3]).toBeGreaterThan(avg[1] * 2.4);
  }, 40_000);

  it("brings the big missiles in on a curve, trailing smoke, and goes up in a mushroom cloud that rises, spreads and fades", () => {
    const g = new Game(THEMES.jungle, { seed: 1, waves: false, gunship: false, level: { boss: null, length: 4000 } });
    play(g, 1 / 30);
    const x0 = g.advance + 300;
    const ty = BASE_ROW * TILE;
    g.missiles.push({ x: x0, y: ty - 500, tx: g.advance + 100, ty, speed: 200, up: false, r: 56 });
    const m = g.missiles[0];
    let bent = 0;
    let trail = 0;
    let landed = false;
    let cloud: object | undefined;
    play(g, 4, () => {
      if (g.missiles.includes(m)) {
        // Off the straight line between where it started and its target
        const u = (m.y - (ty - 500)) / 500;
        if (Math.abs(m.x - (x0 + (g.advance + 100 - x0) * u)) > 6) bent++;
        trail = Math.max(trail, (m.trail ?? []).length);
      } else if (!landed) {
        landed = true;
        cloud = g.mushrooms.find((u) => Math.abs(u.x - (g.advance + 100)) < 80);
        expect(cloud).toBeDefined();
        expect((cloud as { s: number }).s).toBeGreaterThan(0.5);
      }
    });
    expect(landed).toBe(true);
    expect(bent).toBeGreaterThan(3);
    expect(trail).toBeGreaterThan(20);
    // ... and in a while it has drifted apart and gone
    play(g, 10);
    expect(g.mushrooms.includes(cloud as (typeof g.mushrooms)[number])).toBe(false);
  });
});

describe("the city's streets", () => {
  /** A city of just one set piece and some flat ground: its stretch and its spawns. */
  const cityOf = (feature: string, seed = 2) => {
    const L = createLevel({ ...THEMES.city, features: { [feature]: 3, flat: 1 } }, seed, { boss: null, length: 6000 });
    const spawns: ReturnType<typeof extend> = [];
    while (L.next < 420) spawns.push(...extend(L, L.next + 1));
    return { L, spawns };
  };

  it("have boulevards with traffic lights, bus stops and street lights, parks, bridges, waterworks, stations, expressways and power stations", () => {
    const blv = cityOf("boulevard");
    expect(blv.spawns.some((s) => s.kind === "road" && (s.extra?.[2] ?? 0) > 0)).toBe(true);
    expect(blv.L.props.filter((p) => p.kind === "stoplight").length).toBeGreaterThan(2);
    expect(blv.L.props.filter((p) => p.kind === "streetlight").length).toBeGreaterThan(8);
    expect(blv.L.props.some((p) => p.kind === "sprite" && p.id === "busstop")).toBe(true);
    const park = cityOf("park");
    const ids = new Set(park.L.props.map((p) => (p.kind === "sprite" ? p.id : p.kind)));
    expect(
      ["fountain", "statue", "gazebo", "playground", "bench", "flowerbed", "picnictable", "icecreamcart"].filter((id) =>
        ids.has(id)
      ).length
    ).toBeGreaterThan(3);
    const bridge = cityOf("citybridge");
    expect(bridge.L.props.filter((p) => p.kind === "cable").length).toBeGreaterThan(2);
    expect(bridge.spawns.some((s) => s.kind === "boat")).toBe(true);
    const water = cityOf("waterplant");
    expect(water.L.props.some((p) => p.kind === "sprite" && p.id === "clarifier")).toBe(true);
    const station = cityOf("station");
    expect(station.spawns.some((s) => s.kind === "freight" && s.extra?.[3] === 1)).toBe(true);
    const hwy = cityOf("highway");
    expect(hwy.spawns.some((s) => s.kind === "motorway")).toBe(true);
    const nuke = cityOf("nuclear");
    expect(nuke.L.props.some((p) => p.kind === "sprite" && p.id === "reactordome")).toBe(true);
  });

  it("build some of the buildings hard: reinforced concrete through and through, all but unbreakable", () => {
    const { L } = cityOf("metropolis", 3);
    let hard = 0;
    for (let c = L.start; c < L.next; c++)
      for (let r = 0; r < BASE_ROW + 4; r++) if (L.fg[cell(c, r)] === REINFORCED) hard++;
    expect(hard).toBeGreaterThan(20);
    expect(TILE_HP[REINFORCED]).toBeGreaterThan(25);
    expect(TILE_HP[REINFORCED]).toBeLessThan(255);
  });

  it("are full of traffic that stops at red lights, buses calling at their stops, fire engines putting fires out", () => {
    const quietCity: Theme = {
      ...THEMES.city,
      features: { boulevard: 3 },
      enemies: 0,
      barrels: 0,
      artillery: 0,
      helis: 0,
      paras: 0,
      jets: 0,
      drones: 0,
    };
    const g = new Game(quietCity, { seed: 2, waves: false, gunship: false, level: { boss: null, length: 6000 } });
    let moving = 0;
    let redRunners = 0;
    let atRed = 0;
    let busStops = 0;
    let fought = 0;
    play(g, 1);
    const rd = g.roads[0];
    expect(rd).toBeDefined();
    // A fire in the street, for the fire engines
    g.fires.push({ x: (rd.lo + rd.hi) / 2, y: rd.y, t: 300 });
    const fire = g.fires[g.fires.length - 1];
    let burning = 0;
    play(g, 60, () => {
      if (g.fires.includes(fire)) burning += 1 / 30;
      for (const road of g.roads)
        for (const car of road.cars) {
          if (car.v > 20) moving++;
          if (car.kind === "bus" && car.wait > 0 && Number.isNaN(car.spray)) busStops++;
          if (!Number.isNaN(car.spray)) fought++;
          for (const lx of road.lights) {
            const d = (lx - car.x) * car.dir;
            if (lightState(g.t, lx) === 2 && d > CAR_SIZE[car.kind] / 2 + 2 && d < CAR_SIZE[car.kind] / 2 + 10) {
              if (car.v < 5) atRed++;
              else if (!car.siren) redRunners++;
            }
          }
        }
    });
    expect(moving).toBeGreaterThan(30 * 20);
    expect(atRed).toBeGreaterThan(30);
    expect(redRunners).toBeLessThan(atRed / 4);
    expect(busStops).toBeGreaterThan(20);
    expect(fought).toBeGreaterThan(0);
    // The hoses knocked it down faster than it burns down on its own, while it was there
    expect(fire.t).toBeLessThan(300 - burning - 1);
  });

  it("have their traffic wrecked where a blast catches it, the wreck left burning", () => {
    const g = new Game(
      { ...THEMES.city, features: { boulevard: 3 }, enemies: 0 },
      { seed: 2, waves: false, gunship: false, level: { boss: null, length: 6000 } }
    );
    const road = (c: object) => g.roads.find((rd) => rd.cars.includes(c as (typeof rd.cars)[number]))!;
    // One of the cars near the squad, blown up there and then
    let car: { x: number } | undefined;
    let blown = false;
    let gone = false;
    let wrecks = 0;
    let fires = 0;
    play(g, 30, () => {
      if (!car)
        car = g.roads
          .flatMap((rd) => rd.cars)
          .find(
            (c) =>
              c.x > g.advance - 100 &&
              c.x < g.advance + 300 &&
              c.v > 0 &&
              (c.dir > 0 ? c.x < road(c).hi - 120 : c.x > road(c).lo + 120)
          );
      // Blow it up the moment it's found, while it's still on its road
      if (car && !blown) {
        blown = true;
        const rd = road(car);
        wrecks = g.level.props.filter((p) => p.kind === "sprite" && /wreck/.test(p.id)).length;
        fires = g.fires.length;
        g.explode(car.x, rd.y - 6, 20, -2, false);
        gone = !rd.cars.includes(car as (typeof rd.cars)[number]);
      }
    });
    expect(car).toBeDefined();
    expect(gone).toBe(true);
    expect(g.level.props.filter((p) => p.kind === "sprite" && /wreck/.test(p.id)).length).toBeGreaterThan(wrecks);
    expect(g.fires.length).toBeGreaterThan(fires);
  });

  it("have their passenger trains calling at the station, standing there a while, then pulling out", () => {
    const g = new Game(
      { ...THEMES.city, features: { station: 3 }, enemies: 0 },
      { seed: 3, waves: false, gunship: false, level: { boss: null, length: 6000 } }
    );
    let halted = 0;
    let coaches = 0;
    play(g, 90, () => {
      for (const f of g.freights) {
        if (f.halt && f.halt > 0) halted++;
        if (f.cars.includes(5)) coaches++;
      }
    });
    expect(coaches).toBeGreaterThan(0);
    expect(halted).toBeGreaterThan(30 * 3);
  });
});

describe("the city's sky", () => {
  it("turns from day to night and back, and its weather comes and goes", () => {
    // Over a couple of long battles: how the weather goes is down to chance
    let lo = 1;
    let hi = 0;
    const skies = new Set<string>();
    for (const seed of [4, 5]) {
      const g = new Game(THEMES.city, { seed, gunship: false, level: { boss: null, length: 6000 } });
      play(g, 260, () => {
        lo = Math.min(lo, g.night);
        hi = Math.max(hi, g.night);
        skies.add(g.sky);
      });
    }
    expect(lo).toBeLessThan(0.1);
    expect(hi).toBeGreaterThan(0.6);
    expect(skies.size).toBeGreaterThanOrEqual(3);
    // A jungle keeps its own light
    expect(new Game(THEMES.jungle, { seed: 4 }).night).toBe(THEMES.jungle.night);
  }, 60_000);

  it("fills with smoke over the big blasts, a column of it rising off the spot for a while", () => {
    const g = new Game(THEMES.jungle, { seed: 1, waves: false, gunship: false, level: { boss: null, length: 4000 } });
    play(g, 1 / 30);
    const smoke = () => g.parts.filter((p) => p.k === 2 || p.k === 12).length;
    const before = smoke();
    g.explode(g.advance + 120, BASE_ROW * TILE - 10, 36, -2, false);
    expect(g.plumes.length).toBeGreaterThan(0);
    expect(smoke()).toBeGreaterThan(before + 8);
    play(g, 2);
    expect(g.plumes.length).toBeGreaterThan(0);
    expect(smoke()).toBeGreaterThan(before + 8);
  });
});
