/**
 * The battle game on Engine 2. `new BattleGame(options)` builds a whole battle from the feature list in
 * `game/installs.ts`; hosts step it (or hand its world to the engine's `Loop`) and draw `buildFrame` output.
 */
export { BattleGame, type BattleStats, type GameStreams, type LevelState } from "./game/BattleGame";
export { Content, type Links } from "./game/content";
export * from "./game/components";
export * from "./game/defs";
export type * from "./game/events";
export { machine, Timers } from "./game/core";
export { INSTALLS } from "./game/installs";
export { ORIGIN_STEP, Origin } from "./game/origin";
export {
  budgetsFromQuality,
  resolveOptions,
  type BattleOptions,
  type Budgets,
  type Feature,
  type FeatureOptions,
  type ResolvedOptions,
} from "./game/options";
export * from "./game/teams";
