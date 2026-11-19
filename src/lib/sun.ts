/**
 * Real-time day / night for the 3D scene: where the sun is right now (as seen from the
 * Netherlands) and how much daylight that gives. Pure functions, so they're unit-tested.
 *
 * The site never displays a location; these approximate constants only aim the sun.
 */
const LAT = (52.2 * Math.PI) / 180;
const LON_DEG = 5.3;
const DEG = Math.PI / 180;

export interface SunPosition {
  /** Angle above the horizon, radians (negative at night) */
  elevation: number;
  /** Compass bearing, radians clockwise from north */
  azimuth: number;
}

interface Solar extends SunPosition {
  /** Hour angle, radians in −π..π: negative before solar noon, positive after */
  hourAngle: number;
  declination: number;
}

/** Low-precision almanac formulas (~0.5° accuracy). */
function solar(date: Date): Solar {
  const d = date.getTime() / 86_400_000 - 10_957.5; // days since J2000.0
  const g = (357.529 + 0.98560028 * d) * DEG; // mean anomaly
  const q = 280.459 + 0.98564736 * d; // mean longitude, degrees
  const lambda = (q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * DEG; // ecliptic longitude
  const eps = (23.439 - 0.00000036 * d) * DEG; // obliquity of the ecliptic
  const ra = Math.atan2(Math.cos(eps) * Math.sin(lambda), Math.cos(lambda));
  const dec = Math.asin(Math.sin(eps) * Math.sin(lambda));
  const gmst = (((18.697374558 + 24.06570982441908 * d) % 24) + 24) % 24; // hours
  const ha = (gmst * 15 + LON_DEG) * DEG - ra;
  const hourAngle = Math.atan2(Math.sin(ha), Math.cos(ha));
  const sinEl = Math.sin(LAT) * Math.sin(dec) + Math.cos(LAT) * Math.cos(dec) * Math.cos(hourAngle);
  const elevation = Math.asin(sinEl);
  const azimuth = Math.atan2(-Math.sin(hourAngle) * Math.cos(dec), (Math.sin(dec) - sinEl * Math.sin(LAT)) / Math.cos(LAT));
  return { elevation, azimuth: (azimuth + 2 * Math.PI) % (2 * Math.PI), hourAngle, declination: dec };
}

/** Solar position for a moment in time. */
export function solarPosition(date: Date): SunPosition {
  const { elevation, azimuth } = solar(date);
  return { elevation, azimuth };
}

/** Compass bearing the home-page camera looks along (south-south-west). */
export const VIEW_HEADING = 200 * DEG;

/**
 * Where the scene *draws* the sun. The camera sees only a slice of the sky (about
 * south-south-west, up to ~16° high) and the real sun would be out of shot most of the
 * day, so the day's arc is fitted into that view: it comes up `spread` left of centre,
 * passes the middle at solar noon and goes down `spread` to the right. Its height follows
 * the real one, from `lift` at sunrise / sunset (the land hides the true horizon, so it
 * rises over the village and sets behind the hills) up to `peak` at noon; once the real
 * sun is down it sinks quickly out of sight. Rise and set times are real. (The scene's
 * light still comes from the real sun direction; see sunDirection.)
 */
export function sceneSun(date: Date, spread = 26 * DEG, peak = 12.5 * DEG, lift = 7 * DEG): SunPosition {
  const s = solar(date);
  const halfDay = Math.acos(Math.min(1, Math.max(-1, -Math.tan(LAT) * Math.tan(s.declination))));
  const progress = Math.min(1.4, Math.max(-1.4, s.hourAngle / halfDay));
  const noon = Math.PI / 2 - Math.abs(LAT - s.declination);
  return {
    azimuth: VIEW_HEADING + progress * spread,
    elevation: s.elevation > 0 ? lift + (s.elevation / noon) * (peak - lift) : lift + s.elevation * 1.6,
  };
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export interface DayCycle extends SunPosition {
  /** 0 = night … 1 = full day (eases through civil twilight, −6° to +6°) */
  day: number;
  /** 0 … 1, peaks while the sun is near the horizon (sunrise / sunset colours) */
  golden: number;
}

/** Daylight factors for a sun position. */
export function dayCycle(date: Date): DayCycle {
  const sun = solarPosition(date);
  const el = sun.elevation / DEG;
  return {
    ...sun,
    day: smooth(-6, 6, el),
    golden: Math.exp(-(((el - 2) / 6) ** 2)) * smooth(-7, -2, el),
  };
}

/**
 * Scene direction of the sun (unit vector). The camera looks roughly south, so south is
 * −z (straight ahead), east is −x (left) and west +x (right): the sun rises on the left,
 * crosses in front of the village and sets on the right.
 */
export function sunDirection({ elevation, azimuth }: SunPosition): [number, number, number] {
  const c = Math.cos(elevation);
  return [-Math.sin(azimuth) * c, Math.sin(elevation), Math.cos(azimuth) * c];
}

/** `?time=HH:MM` (24 h, Amsterdam time) → [hours, minutes], for previews and tests. */
export function parseTimeOverride(search: string): [number, number] | null {
  const m = /[?&]time=(\d{1,2})(?::|%3A)(\d{2})\b/i.exec(search);
  if (!m) return null;
  const [h, min] = [Number(m[1]), Number(m[2])];
  return h < 24 && min < 60 ? [h, min] : null;
}

/** Today's date at `hours:minutes` wall-clock time in `timeZone`. */
export function wallClockToday(hours: number, minutes: number, timeZone: string, now = new Date()): Date {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-GB", { timeZone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
      .formatToParts(now)
      .map((p) => [p.type, Number(p.value)])
  );
  // The zone's current offset from UTC, then today's wall time converted back to UTC
  const offsetMs = Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute) - Math.floor(now.getTime() / 60_000) * 60_000;
  return new Date(Date.UTC(parts.year, parts.month - 1, parts.day, hours, minutes) - offsetMs);
}
