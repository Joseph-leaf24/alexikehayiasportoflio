# Notes for Claude

## Repo layout

| Folder | What it is |
|---|---|
| `Site 1/` | The live portfolio, deployed to GitHub Pages on every push to `main` |
| `Site 2/` | A local-only pixel-art copy of the site, with a battle game behind it |
| `Wallpaper/` | Builds `PixelBattle.exe`, a Windows live wallpaper made from Site 2's game |
| `Projects/` | Old project code; out of scope |

## Rules

- **The owner commits and pushes.** Never run `git commit` or `git push`.
- **Only the `main` branch exists.** Don't create other branches or pull requests. Pushing to `main` redeploys the live site (Site 1).
- **All art must stay original.** No copied sprites, characters or designs.

## Site 2 remaster: in progress

Site 2's game is being remastered onto a new engine of our own: `Site 2/src/engine/` and `Site 2/src/battle/`.
The old engine in `Site 2/src/components/pixel/` keeps running the site until the switch-over.

**What we're building: a self-playing battle scene, not a game you play.** It runs behind the Site 2 portfolio and as
the Wallpaper. Every unit, the 8-player squad included, is AI-driven, and visitors only watch: never add player controls
(keyboard, gamepad or click-to-move). Their only influence stays the old `sceneState` one: the pointer, a click for a
strike, holding for a barrage, zoom, scroll speed and the watching mode. "Playable" in B0's "playable slice" only means
a runnable demo page; its keys are for testing.

**Before working on it, read `Site 2/docs/remaster/CHECKPOINT.md`.** It has the current status, next steps and
checks. The scope is deliberately lean, because tokens are limited (see `PLAN.md` in the same folder).

Update `CHECKPOINT.md` after every phase.

**Every coding session on the remaster, chat or autopilot, works in this order, so bugs never pile up:**
1. fix every bug under "Known bugs" in `CHECKPOINT.md`, and anything the checks report;
2. review the local model's unreviewed entries in its "Local model log", fix what's wrong, and mark them reviewed;
3. only then continue with the next task.

**Features first, tests last:** no new tests until phase T, which Claude writes alone at the end (about 80% coverage
is enough). The local model only builds features. **Good enough, not perfect:** keep to the tasks as written, with no
extra detail and no new or further-split tasks; port each feature only as far as it shows in the scene.

**An autopilot may be doing the work unattended**: `Site 2/docs/remaster/local-llm/autopilot.ps1`. It hands over
between Claude and a local LM Studio model (see `LOCAL-MODEL.md` there).

If `local-llm/logs/autopilot-live.log` was written in the last few minutes, the autopilot is running. Don't edit
remaster files from a chat then: stop it first (the Stop button on the dashboard, http://localhost:8090 from the task "Remaster: dashboard", or
create `local-llm/autopilot.stop`).

