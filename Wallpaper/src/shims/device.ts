import { qualityFor, type Quality } from "../../../Site 2/src/lib/device";

/**
 * The wallpaper's budget: lighter than the site's (it runs all day, behind everything else).
 * Fewer enemies at once and fewer particles, a few fewer clouds; the battle looks the same.
 */
export function getQuality(): Quality {
  return { ...qualityFor("mid"), enemies: 80, particles: 1200, clouds: 8 };
}

export { qualityFor };
