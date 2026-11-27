# Remaster checkpoint

**Read this first when continuing the Site 2 remaster in a new chat or on another computer.** It's kept up to
date after every phase.

_Last updated: 2026-11-27, after B1 task 1, the themes (Claude, autopilot run)._

## Where things stand

| Phase | Status | Where |
|---|---|---|
| A: Engine | ✅ Done | `src/engine/`, about 10k lines and 200 tests. See [engine-notes.md](engine-notes.md) and [engine-spec.md](engine-spec.md). |
| B0: Game skeleton | ✅ Done | `src/battle/`. A playable slice at `/battle-dev.html?debug=1`; 450 tests in all. The map is [battle-notes.md](battle-notes.md). |
| B1, B2, B3, C1, C2, D | Not started | Scope in [PLAN.md](PLAN.md) |

## Known bugs

Every Claude session starts by fixing everything listed here and anything the checks report, then reviews the local
model's unreviewed work, and only then continues with new tasks, so bugs never pile up. Whoever finds a bug adds one
line: where it is and what goes wrong. Remove it once it's fixed and verified.

_(none known)_

## In progress

Before starting a task, each model adds one line here, `**Claude:** <task>. Files: ...` or `**Local model:** <task>.
Files: ...`, and removes it when the task is done. A line still here when a session starts was cut off (a run can
stop at any moment): the local model continues its own; Claude finishes either kind, reviewing the local model's part
first. Neither model touches the files on the other's line.

**Claude:** B1 task 2, the level generator. Files: `src/battle/level/` (generator files and `index.ts`),
`src/test/battle-level.test.ts`.

## Next tasks, in order

Each line is one unit of work. Tick it (`[x]`) when it's done and verified.

- **Claude runs** may take several tasks, or a whole phase, per run, using foreground subagents for the big ones.
- **The local model** takes ONE small task at a time: the ones marked *(small)*. Claude marks a task *(small)* only
  once what it builds on is done; small ones still waiting say *(local model, after …)* instead. A task marked
  *(blocked)* waits for Claude to answer the local model's question in the log, and the local model skips it.

### B1: World (`src/battle/level/`, `src/battle/world/`)

- [x] *(small)* Port the theme data from `src/components/pixel/themes.ts` into `src/battle/level/themes.ts`, as typed
  data with the same keys and values. Add a test that every theme has a name, a palette and at least one feature.
  _Done: `level/themes.ts` (deck, `variant`, labels) plus the data in `level/themes/` (split in the old order to
  stay under 800 lines). `src/test/battle-themes.test.ts` checks it deep-equals the old themes and deals the same
  seeded deck._
- [ ] Port the level generator (`components/pixel/level.ts` `createLevel`, `extend`, `feature`) onto `Terrain`, using
  the true column (`col + game.origin.cols`). Keep the old block ids. Buildings need door frames down to the ground
  (see battle-notes.md).
- [ ] Register the terrain tiles, props and parallax backdrops per theme (`registerBackdropArt`, `sky.backdrop`).
- [ ] Weather (rain, snow, fog, lightning) and day/night per theme.
- [ ] *(local model, after the generator)* Tests: every theme generates 2,000 columns without error, and the landing
  zone is flat.

### B2: Squad and flow (`src/battle/squad/`, `src/battle/flow/`)

- [ ] The 17 weapons as `WeaponDef` + `ProjectileDef` (old `fire()` in `components/pixel/game.ts`), with beam visuals.
- [ ] *(local model, after the 17 weapons)* Weapon sprites per player through `unit.art`, using `playerArt(p, w)`.
- [ ] Squad AI: jetpack, supply crates (random weapon for 16 s), barrels; tune lethality (it's too high now).
- [ ] The respawn transport jet.
- [ ] Flow: intensity waves (`WAVES`), banners, the salvo and nuke finales, the white-out and level card.
- [ ] *(local model, after the tasks above)* Tests for each of the above, with fixed seeds.

### B3: Army (`src/battle/enemies/`)

- [ ] *(small)* The attack dog as a `UnitDef`, `regime.dog`, in a new `src/battle/enemies/dog.ts`: port the old
  `Game.updateDog` (`components/pixel/game.ts`) as a `machine()` AI, following `enemies/infantry.ts` exactly, and
  register it in `enemies/index.ts` next to `regime.trooper`. Its art is already registered (`art/index.ts`). Add
  `src/test/battle-dog.test.ts` with a fixed seed: it runs at the squad and bites.
- [ ] The rest of the core Regime roster as `UnitDef`s: infantry variants, snipers, riot shields, flamers,
  grenadiers, jeeps, trucks, tanks, helicopters, jets, drones. Claude splits off one *(small)* task per unit.
- [ ] 3 bosses: siege tank, walker, gunship.
- [ ] The view-based enemy cap, and allies.

### C1, C2, D

See PLAN.md. Write a brief for each into this folder before starting it.

## Checks (run from `Site 2/`)

```sh
npm run typecheck
npm run lint
npm run format:check
npx vitest run            # all tests (453 after the themes)
node scripts/bench-battle.mjs --seconds 30
```

E2E: `npm run build`, then `npx playwright test`.

## Running unattended

The autopilot (`local-llm/autopilot.ps1`) runs Claude Code headless on these tasks. When the usage limit is hit, it
switches to the local model for 5 hours, then hands back to Claude, which reviews the local work first. See
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
