# Site 2 remaster: the plan (lean version)

The owner asked for a remaster of Site 2's pixel battle on a new engine of our own, then asked to **keep it
simple, because tokens are limited**. This is the agreed scope. Anything not listed here is out of scope unless
the owner asks for it.

## Constraints

- **What we're building: a self-playing battle scene, not a game you play.** It runs behind the Site 2 portfolio and
  as the Wallpaper. Every unit, the 8-player squad included, is AI-driven, and visitors only watch: never add player
  controls (keyboard, gamepad or click-to-move). Their only influence stays the old `sceneState` one: the pointer, a
  click for a strike, holding for a barrage, zoom, scroll speed and the watching mode. "Playable" in B0's "playable
  slice" only means a runnable demo page; its keys are for testing.
- **Features first, tests last.** No new tests until phase T, which Claude writes alone (about 80% coverage is
  enough); the local model only builds features. Until then, keep the existing checks passing.
- **Good enough, not perfect.** Keep to the tasks as written: no extra detail, no splitting tasks further, no new
  tasks beyond this plan. Port each feature only as far as it shows in the scene; a close match to the old look is
  fine, and polish can wait.
- **Keep the old engine working.** It lives in `src/components/pixel/` and keeps running the site and the
  Wallpaper, with all its tests green, until the new engine reaches the scope below.
- **Original pixel art only.** Reuse the old engine's own painters (`components/pixel/art.ts`, `catalog.ts`,
  `buildings.ts`).
- **Code standard:** Prettier at 120 columns and ESLint clean (no unused variables); no `any`; short "why"
  comments; files under about 800 lines.
- **The owner commits.** Agents and Claude never commit or push. Only the `main` branch is used; no other
  branches or pull requests.

## Phases

| Phase | What | Who | Status |
|---|---|---|---|
| A | **Engine** in `src/engine/`: core (ECS, fixed 1/30 s tick, rng streams, spatial hash, events, loop, headless), terrain (destructible cells, physics, structural collapse) and a WebGL2 renderer (atlas, sprites, GPU terrain, lights, post FX, text, Canvas 2D fallback, debug overlay, demo) | 3 agents | Done |
| B0 | **Game skeleton** in `src/battle/`: architecture, locomotion, projectiles, explosions, damage, render extraction, camera, art book, a playable slice (`battle-dev.html`) | 1 agent | Done |
| B1 | **World:** port the level generator and themes, terrain tiles and props art, parallax backdrops, weather and day/night | 1 agent | Done |
| B2 | **Squad and flow:** the 8 players, the 17 weapons, supply crates, respawns, intensity waves, banners, two finales (salvo and nuke) with the white-out and level card | 1 agent | In progress |
| B3 | **Army:** the core Regime roster (infantry variants, snipers, riot shields, flamers, grenadiers, dogs, jeeps, trucks, tanks, helicopters, jets, drones) and 3 bosses (siege tank, walker, gunship) | 1 agent | Planned |
| C1 | **Factions:** Iron Legion and the Brood, each with 4 units and a boss, plus faction choice per level | 1 agent | Planned |
| C2 | **Squad specials and missions:** one special per player; 4 mission types (assault, sabotage, rescue, defend) with a mission card; boss-intro cinematics (letterbox and title); an orbital strike | 1 agent | Planned |
| T | **Tests:** tests for the new game code, about 80% coverage, with fixed seeds; written once the features are done, none before | Claude | Planned |
| D | **Switch-over:** a `BattleScene.tsx` replacing `PixelScene.tsx` behind the same `sceneState` contract; e2e and unit tests updated; a performance check; Wallpaper rebuild; delete the old engine | Claude | Planned |

## Deferred, cut for budget (only if the owner asks later)

- From the old game: metro, freight and armoured trains, mine carts, road traffic, ships, submarines, SAMs, the
  gunship ferry, the worm and airship bosses, and the three other finales.
- From the big remaster design: the other three factions (Black Tide, Ashen Cult, Hollow Signal), faction wars,
  ultimates, ranks, medals, callouts, revives, vehicles, the war map and tours, the Director, the other seven
  mission types, titans, new weather events.

## The two new factions (C1)

1. **Iron Legion** (machines; gunmetal with red optics):
   - Units: scuttler (a leaping spider drone), sentry (a turret with a laser burst), hover tank (a beam), arc
     trooper (chaining shots).
   - Boss: **the Colossus** (a walking siege engine with a core weak point).
2. **The Brood** (creatures; sickly green and violet):
   - Units: crawler (pounces), spitter (acid that eats terrain), burrower (erupts from the ground), brute
     (armoured, charges).
   - Boss: **the Brood Queen** (births crawlers, burrows).

The Regime (the old army) is the third faction. A level's faction is picked to suit its theme.

## Squad specials (C2)

One each, on a cooldown, with a short tag over the player when it fires:

| Player | Special | Effect |
|---|---|---|
| P1 Engineer | Pipeline | A barricade as cover |
| P2 Analyst | Insight | Marks enemies, which take extra damage |
| P3 Scientist | Clustering | A cluster bomb |
| P4 ML Engineer | Deploy | An auto-turret |
| P5 Vision | Lock-on | Snipes 3 targets marked with detection boxes |
| P6 Language | Persuade | Turns a few enemies to the squad's side for a while |
| P7 Research | Experiment | A gravity well |
| P8 Agents | Agents | 3 orbiting drones |

## Missions (C2)

The level card shows `MISSION · SABOTAGE — Destroy the 3 relay masts`.

| Mission | Objective | Ends |
|---|---|---|
| Assault | Push to the stronghold | The finale |
| Sabotage | Destroy N targets | Chain blasts |
| Rescue | Free the caged hostages, who follow the squad | Extraction |
| Defend | Hold a beacon for T seconds | Fails if the beacon dies; a "Mission failed" card, then the next level |
