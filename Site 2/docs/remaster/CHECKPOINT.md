# Remaster checkpoint

**Read this first when continuing the Site 2 remaster in a new chat or on another computer.** It's kept up to
date after every phase.

**What we're building: a self-playing battle scene, not a game you play.** It runs behind the Site 2 portfolio and as
the Wallpaper. Every unit, the 8-player squad included, is AI-driven, and visitors only watch: never add player controls
(keyboard, gamepad or click-to-move). Their only influence stays the old `sceneState` one: the pointer, a click for a
strike, holding for a barrage, zoom, scroll speed and the watching mode. "Playable" in B0's "playable slice" only means
a runnable demo page; its keys are for testing.

_Last updated: 2026-11-27, after B1 (the world) and B2's 17 weapons (Claude, autopilot run)._

## Where things stand

| Phase | Status | Where |
|---|---|---|
| A: Engine | ✅ Done | `src/engine/`, about 10k lines and 200 tests. See [engine-notes.md](engine-notes.md) and [engine-spec.md](engine-spec.md). |
| B0: Game skeleton | ✅ Done | `src/battle/`. A playable slice at `/battle-dev.html?debug=1`; 450 tests in all. The map is [battle-notes.md](battle-notes.md). |
| B1: World | ✅ Done | `src/battle/level/`: themes, generator, backdrops, props and their glows, flames and mist, weather and day/night |
| B2: Squad and flow | In progress | The 17 weapons done (`src/battle/squad/`) |
| B3, C1, C2, T, D | Not started | Scope in [PLAN.md](PLAN.md) |

## Known bugs

Every Claude session starts by fixing everything listed here and anything the checks report, then reviews the local
model's unreviewed work, and only then continues with new tasks, so bugs never pile up. Whoever finds a bug adds one
line: where it is and what goes wrong. Remove it once it's fixed and verified.

_(none known)_

## In progress

Before starting a task, each model adds one line here, `**Claude:** <task>. Files: ... Left: ...` or `**Local model:**
<task>. Files: ...`, and removes it when the task is done; Claude gives each subagent's task its own line and keeps
its "Left:" note current. A line still here when a session starts was cut off (a run can stop at any moment). The
local model then works through them one at a time, one per session: its own first, then Claude's in order (relabelled
`**Local model:**` when it takes one over). Claude finishes and reviews any kind. Nobody touches the files on another
line, and a line marked `(blocked)` waits for Claude.

**Claude:** Squad AI: jetpack, supply crates, barrels; tune lethality. Files: `squad/` (`ai.ts`, `squad.ts`,
`index.ts`, new `supply.ts` and `barrels.ts`), `game/components.ts`, `game/defs.ts`, `physics/spawn.ts`,
`physics/damage.ts`, `physics/explosions.ts`, `physics/terrain.ts`, `enemies/index.ts`. Left: all of it.

## Next tasks, in order

Each line is one unit of work. Tick it (`[x]`) when it's done and verified.

- **Claude runs** may take several tasks, or a whole phase, per run, using foreground subagents for the big ones.
- **The local model** takes ONE small task at a time: the ones marked *(small)*. Claude marks a task *(small)* only
  once what it builds on is done; small ones still waiting say *(local model, after …)* instead. A task marked
  *(blocked)* waits for Claude to answer the local model's question in the log, and the local model skips it.

- **Features first, tests last.** No new tests until phase T, which Claude writes alone at the end (about 80%
  coverage is enough). Until then both models build features only, and keep the existing checks passing.
- **Good enough, not perfect.** Keep to the tasks as written: no extra detail, no splitting tasks further, no new
  tasks beyond PLAN.md. Port each feature only as far as it shows in the scene; a close match to the old look is
  fine, and polish can wait.

### B1: World (`src/battle/level/`, `src/battle/world/`)

- [x] *(small)* Port the theme data from `src/components/pixel/themes.ts` into `src/battle/level/themes.ts`, as typed
  data with the same keys and values. Add a test that every theme has a name, a palette and at least one feature.
  _Done: `level/themes.ts` (deck, `variant`, labels) plus the data in `level/themes/` (split in the old order to
  stay under 800 lines). `src/test/battle-themes.test.ts` checks it deep-equals the old themes and deals the same
  seeded deck._
- [x] Port the level generator (`components/pixel/level.ts` `createLevel`, `extend`, `feature`) onto `Terrain`, using
  the true column (`col + game.origin.cols`). Keep the old block ids. Buildings need door frames down to the ground
  (see battle-notes.md).
  _Done: `level/generator.ts` runs the old generator, a feature at a time 16 columns ahead, and copies cells by
  true column, making everything stand up to the engine's support checks (door leaves, stone-walled buildings made
  structures, `BEAM` cells under legs, posts and motorway pillars, corner joints). New engine material `BEAM` (94);
  ground on settled loose cells is anchored (`structure.ts` `runGround`). One old-generator fix: the airfield takes
  in its hangar's full width (the next feature cleared its far wall). `src/test/battle-level.test.ts` checks the
  cells, that everything resting on something stands in every theme, and that a stilt hut falls when its legs go._
- [x] Register the terrain tiles, props and parallax backdrops per theme (`registerBackdropArt`, `sky.backdrop`).
  _Done: the level feature registers every theme's backdrop (lazy; the first level's at boot) and uniforms, shows
  the biome's backdrop and fades to the next at a border (`level.biome`, `LevelSource.biome`, `sky.show`), with no
  clouds under a roof and the cave's far rock hung from the top. The old sprite props (`asset.<id>` art) are drawn
  by `level/props.ts`, lit in front or shadowed behind. Tiles need nothing per theme (one old block sheet). Tested in
  `src/test/battle-scenery.test.ts`; seen in Chrome on `battle-dev.html`._
- [x] The old scene's hand-drawn props, first group and the pattern: `legs`, `support`, `wire`, `rope`, `antenna`,
  `cable`, `tent`, `flag`.
  _Done: `level/drawn.ts`. Sized things are runs of a strip (`strip(name, axis, ...colours)`: plain colour painted
  at every length 1–8 px, `pen.run`, `pen.curve` for sagging lines); fixed shapes are painted once from the old
  case's own code (`shape(name, { w, h, ox, oy, poses, paint })`, `pen.shape`); animated ones are poses picked by
  `pen.cycle`; glows are `SPRITE_EMISSIVE` sprites plus `pen.light` (capped at a quarter of the light budget). A
  kind is one draw function in `DRAWN`. Legs and posts are drawn while their beams stand (`LevelSource.stands`;
  legs that never had beams, into water, are drawn as the old game did). Tested in `battle-scenery.test.ts`._

The rest of the old scene's prop drawing, one *(small)* task each. The old code: `PixelScene.tsx` `drawWorld`, the
`switch (p.kind)` after the blocks (about line 1135), and its night glows in the lighting pass (about line 2136).
Follow `level/drawn.ts` exactly (same colours and offsets, `x`, `y` as the old case's). If `drawn.ts` passes about
700 lines, move the kinds into a `level/drawn/` folder first.

- [x] *(small)* `brazier` (a shape), `silo` (a 28×2 strip), and the housings of `lamp`, `alarm`, `searchlight` (one
  `#3a3f4f` strip each; their glows come later). _Done (Claude), tested._
- [x] *(small)* `stoplight` (pole strip, and a shape with 3 poses for `lightState(time, p.x)` from
  `components/pixel/game.ts`) and `streetlight` (pole and arm strips; the lamp head a 2-pose shape: lit when
  `pen.night > 0.3`, drawn `SPRITE_EMISSIVE` then). _Done (Claude), tested. Both are fixed sizes, so each body is
  one shape (pole included); the streetlamp is an on/off strip pair like the antenna's beacon; `lightState` is ported
  into `drawn.ts` (the old engine goes in phase D)._
- [x] *(small)* `radar` (a shape with a pose per dish width 2–12, the width from time as the old case) and `derrick`
  (the tower as one shape, the pumpjack as a shape with 8 nod poses from `pen.cycle`). _Done (Claude), tested. The
  pumpjack is pivoted at the derrick's foot, which lies left of its canvas (`ox: -7`; sprite pivots are signed)._
- [x] *(small)* `waterfall` (sized w × h: per column, a 1×12 strip-like shape of the 4 bands with 12 scroll poses,
  repeated down `h`, the column's phase from the old `hash2(k, 99)` and time; the foam pixels at its foot) and
  `awning` (a run per stripe: register one strip per awning colour the old building specs use, found at install).
  _Done (Claude), tested. The old bands step on a fixed 3-px grid, so a column has 4 looks, not 12 (a shape posed by
  look and length 1–12); the phase uses the engine's `hash2`. Awning strips come from `BUILDINGS` at module load._
- [x] *(small)* `sign`: one shape per (text, board, ink) the old generator can place, all known at install (the
  building specs' signs and the station names: find every `kind: "sign"` push in `components/pixel/level.ts`), drawn
  with `pixelText` from `components/pixel/art.ts`. _Done (Claude), tested (every theme's signs have a shape;
  `signArt(p)` gives the id). `STATIONS` is now exported from the old `level.ts`._
- [x] The night glows from the old lighting pass: alarm blink, lamp, streetlight cone
  and pool, stoplight lamp, neon signs, the sprite props' `lights` (`asset(id).lights` in
  `components/pixel/assets.ts`), fire and brazier glow: emissive pixels or shapes plus `pen.light`.
  _Done (Claude): `level/glows.ts`, a draw per kind in `GLOWS`, called after each prop by `props.ts`. Lit lamps are
  emissive runs faded by `alpha(a)` tints; soft glows (streetlight cone and pool, neon halo, the searchlight's beam,
  turned with `pen.shape`'s `rot`) are `SPRITE_EMISSIVE | SPRITE_ADDITIVE` shapes. The searchlight was missing from
  the list and is ported too. Seen at night in Chrome; the existing checks pass._
- [x] *(small)* A `level.ambient` system (phase `effects`, `every: 2`), as the old `game.ts` did (about line 7203):
  flames over `fire` and `brazier` props and over sprite props' `flames`, smoke from their `smoke`, mist at a
  waterfall's foot. Use `emitEffect` on the `fx` stream as `fx.fire` does in `render/index.ts`.
  _Done (Claude): `level/ambient.ts`, with the level's own effects `level.flame`, `level.smoke`, `level.mist` (the old
  `flame`, `puff` and mist particles); four times the old per-frame amounts per run, as it runs at 15 Hz. Seen in
  Chrome; the checks pass._
- [x] Weather (rain, snow, fog, lightning) and day/night per theme.
  _Done (Claude): `level/weather.ts`. A biome's `SkyLook` (`skyLook(theme)`, given to `sky.show`) sets its darkness
  and sky colours, fixed or the cycle's, and its tint; `sky.night` is now the old scale (0.7 at the cycle's
  midnight). Weather particles at the old rates (`weather.*` effects; new particle bit `P_POP`, `EffectDef.pop`:
  dies on terrain), the cycle themes' changing sky (overcast, fog and wet as the grade in the sky pass), storm
  lightning (flash, shake, a bolt shape). Sun and moon not ported. Seen in Chrome; the checks pass._

### B2: Squad and flow (`src/battle/squad/`, `src/battle/flow/`)

- [x] The 17 weapons as `WeaponDef` + `ProjectileDef` (old `fire()` in `components/pixel/game.ts`), with beam visuals.
  _Done (Claude, with a subagent; reviewed): `squad/arms.ts` (`ARSENAL`, the old order; `equip(game, id, w)`),
  `squad/guns.ts` (the 13 projectile guns), `squad/rays.ts` (laser, tesla, prism, and the held death ray with its
  `squad.rays` system), `render/beams.ts` (beams, lightning and muzzle crosses: the `fx.beams` system and the beams
  pass, 350). Player n carries weapon n. Not ported: barrels, riot shields blocking shots (later tasks). Seen in
  Chrome; the checks pass._
- [x] *(small)* Weapon sprites per player through `unit.art`, using `playerArt(p, w)`.
  _Done (Claude): `equip` sets a player's sprite for its weapon, so crates get it for free._
- [ ] Squad AI: jetpack, supply crates (random weapon for 16 s), barrels; tune lethality (it's too high now).
- [ ] The respawn transport jet.
- [ ] Flow: intensity waves (`WAVES`), banners, the salvo and nuke finales, the white-out and level card.

### B3: Army (`src/battle/enemies/`)

- [ ] *(small)* The attack dog as a `UnitDef`, `regime.dog`, in a new `src/battle/enemies/dog.ts`: port the old
  `Game.updateDog` (`components/pixel/game.ts`) as a `machine()` AI, following `enemies/infantry.ts` exactly, and
  register it in `enemies/index.ts` next to `regime.trooper`. Its art is already registered (`art/index.ts`).
- [ ] The rest of the core Regime roster as `UnitDef`s: infantry variants, snipers, riot shields, flamers,
  grenadiers, jeeps, trucks, tanks, helicopters, jets, drones. Claude splits off one *(small)* task per unit.
- [ ] 3 bosses: siege tank, walker, gunship.
- [ ] The view-based enemy cap, and allies.

### C1, C2, D

See PLAN.md. Write a brief for each into this folder before starting it.

### T: Tests (Claude only, after C2, before D)

- [ ] Tests for the new game code, aiming at about 80% coverage of `src/battle/` (`npx vitest run --coverage`), with
  fixed seeds: the generator (every theme generates 2,000 columns without error, and the landing zone, true columns
  0–39, is flat), props, weapons, squad, flow and finales, the army, factions, specials and missions. Every test
  must pass.

## Checks (run from `Site 2/`)

```sh
npm run typecheck
npm run lint
npm run format:check
npx vitest run            # all tests (470 at the end of B1)
node scripts/bench-battle.mjs --seconds 30
```

E2E: `npm run build`, then `npx playwright test`.

## Running unattended

The autopilot (`local-llm/autopilot.ps1`) runs Claude Code headless on these tasks. When the usage limit is hit, it
switches to the local model until Claude's usage window resets, then hands back to Claude, which reviews the local
work first. See
[local-llm/LOCAL-MODEL.md](local-llm/LOCAL-MODEL.md).

Run the VS Code task "Remaster: dashboard": it opens http://localhost:8090, which shows the progress, which model is
working, the hand-overs, the live activity and the local model's tokens/s. Its Start button starts the autopilot;
Stop ends it within about 10 seconds (the next start carries on from "In progress").

## Local model log

The local model adds one dated line here per step: what changed, and whether the checks passed. Claude reviews new
entries when it takes over, marks them `(reviewed)`, and fixes what's needed.

_(no entries yet)_

## Reference docs in this folder

| File | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The agreed lean scope and phases |
| [engine-spec.md](engine-spec.md) | The engine's design contract |
| [engine-notes.md](engine-notes.md) | Engine facts game code must follow |
| [battle-notes.md](battle-notes.md) | The battle layer's structure, defs, API, and the site contract |
| [old-engine-map.md](old-engine-map.md) | Where everything lives in the old engine (`src/components/pixel/`), for porting |
| [b0-brief.md](b0-brief.md) | The skeleton brief, for reference |
| [local-llm/LOCAL-MODEL.md](local-llm/LOCAL-MODEL.md) | The local model and the autopilot |

## Working rules for this project

- **The owner commits and pushes themselves.** Never commit or push. Only `main` exists: no other branches or
  pull requests. Pushing to `main` deploys Site 1; Site 2 changes don't alter it.
- **Save a checkpoint after every task.** Tick it here. Tokens are limited and the limits hit often, so keep scopes
  lean.
- **Keep the old engine and all its tests green** until phase D.
- **All art stays original.** Site 1 is the live site; Site 2 is local-only.
