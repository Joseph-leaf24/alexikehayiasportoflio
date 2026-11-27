import { qualityFor, type Quality } from "../../../../Site 2/src/lib/device";

/** A lighter budget than the site's: the wallpaper runs all day behind everything else. */
export function getQuality(): Quality {
  return { ...qualityFor("mid"), enemies: 80, particles: 1200, clouds: 8 };
}

export { qualityFor };
