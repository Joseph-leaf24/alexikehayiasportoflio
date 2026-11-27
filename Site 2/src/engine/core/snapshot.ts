/** A fingerprint of simulation state, for determinism tests, replays and soak-run checks. */

import { INDEX_BITS } from "./ecs";
import { hashCombine } from "./rng";
import type { World } from "./world";

const f64 = new Float64Array(1);
const words = new Uint32Array(f64.buffer);

export interface HashOptions {
  /** Include every random stream's state (default true); off compares entity state alone. */
  rng?: boolean;
}

/**
 * A 32-bit hash of the tick, every random stream, and every live entity's id, component membership and numeric
 * column values (exact bits), visited by entity index. Object components are opaque and left out.
 */
export function hashWorld(world: World, opts: HashOptions = {}): number {
  let h = hashCombine(world.time.tick, opts.rng === false ? 0 : world.rng.hash());
  h = hashCombine(h, world.count);
  const comps = world.components;
  const W = world.words,
    sig = world.sig,
    gen = world.gen,
    state = world.state;
  for (let i = 0; i < world.indexLimit; i++) {
    if (state[i] === 0) continue;
    h = hashCombine(h, (gen[i] << INDEX_BITS) | i);
    h = hashCombine(h, state[i]);
    for (let w = 0; w < W; w++) h = hashCombine(h, sig[i * W + w]);
    for (let c = 0; c < comps.length; c++) {
      const m = comps[c].$;
      if (m.sparse[i] < 0) continue;
      const cols = m.columns;
      for (let k = 0; k < cols.length; k++) {
        f64[0] = cols[k][i];
        h = hashCombine(hashCombine(h, words[0]), words[1]);
      }
    }
  }
  return h;
}
