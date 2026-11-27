/** Teams and their spatial masks. Squad = 1, the classic enemy = 2; the remaster's factions take 3 and up. */
import { enemiesOf, teamBit, TEAM_NEUTRAL, TEAM_SQUAD } from "@/engine";

export { TEAM_NEUTRAL, TEAM_SQUAD };
/** The Regime: today's army, the baseline enemy. */
export const TEAM_REGIME = 2;
/** First team number for the remaster's factions. */
export const TEAM_FACTION0 = 3;

/** Spatial mask of every team hostile to `team` (never neutral, never itself). */
export const hostileTo = (team: number): number => enemiesOf(team);
/** Spatial mask of `team` alone. */
export const only = (team: number): number => teamBit(team);
