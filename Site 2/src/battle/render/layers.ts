/**
 * Sprite layers (0–15). Layers below 4 draw behind the terrain; world particles draw after layer 11, so layers
 * 12–15 cover them. Every draw picks one of these so features stack consistently.
 */
export const LAYER = {
  /** Clouds and anything in the sky. */
  SKY: 0,
  /** The war in the distance, titans, far aircraft. */
  DISTANT: 1,
  /** Behind-the-ground set dressing (back props, ships far out). */
  BACK: 2,
  /** Things the terrain covers: tunnels' contents, submarines. */
  BEHIND: 3,
  /** Falling terrain chunks: drawn like terrain, right in front of it. */
  CHUNKS: 4,
  PROPS: 5,
  VEHICLES: 6,
  ENEMIES: 7,
  PLAYERS: 8,
  PROJECTILES: 9,
  /** Sprite effects over units (muzzle flashes, parachutes). */
  FX: 10,
  /** Last layer under the particles. */
  UNDER_PARTICLES: 11,
  /** Foreground: foliage and water over everything. */
  FRONT: 12,
  /** Markers, health bars, selection outlines. */
  MARKERS: 13,
} as const;

/** Screen-space text layers (the overlay pass). */
export const TEXT_LAYER = {
  POPUPS: 10,
  BANNERS: 12,
  DEBUG: 15,
} as const;
