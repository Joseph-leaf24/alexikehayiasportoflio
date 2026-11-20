/**
 * Session storage that never throws (it can be blocked, e.g. in some private windows), and the
 * keys the site keeps in it: where the home page was scrolled to, and which Work tab was open.
 */
export const HOME_Y = "ak:homeY";
export const SPEC = "ak:spec";

/** A number saved this session (0 when missing, invalid or storage is blocked). */
export function readNumber(key: string) {
  try {
    const v = Number(sessionStorage.getItem(key));
    return Number.isFinite(v) ? v : 0;
  } catch {
    return 0;
  }
}

export function save(key: string, value: string | number) {
  try {
    sessionStorage.setItem(key, String(value));
  } catch {
    /* storage blocked: nothing to remember */
  }
}
