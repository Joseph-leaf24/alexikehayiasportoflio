/* Session storage helpers that never throw: storage can be blocked, e.g. in some private windows. */

/** Key for where the home page was scrolled to. */
export const HOME_Y = "ak:homeY";
/** Key for which Work tab was open. */
export const SPEC = "ak:spec";

/** A number saved this session; 0 when it is missing or invalid, or storage is blocked. */
export function readNumber(key: string) {
  try {
    const v = Number(sessionStorage.getItem(key));
    return Number.isFinite(v) ? v : 0;
  } catch {
    return 0;
  }
}

/** Saves a value for this session; does nothing when storage is blocked. */
export function save(key: string, value: string | number) {
  try {
    sessionStorage.setItem(key, String(value));
  } catch {
    /* storage blocked: nothing to remember */
  }
}
