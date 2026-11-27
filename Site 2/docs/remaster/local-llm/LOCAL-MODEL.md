# The autopilot: Claude and a local model, working in turns

The remaster runs unattended. Claude does the work. When Claude's 5-hour usage reaches 95%, a local model in LM
Studio takes over until Claude's usage window resets. Then Claude takes back over, reviews and fixes the local model's work, and
carries on. Nobody needs to be in the loop.

## How it works (`autopilot.ps1`)

- **Claude's runs.** Headless Claude Code sessions with no chat window, working through the tasks in
  `../CHECKPOINT.md`.
  - **Usage meter.** Claude Code streams a meter during each run (`rate_limit_event`, with the 5-hour window's
    utilization and reset time). At **95%** the autopilot stops the run and hands over.
- **The local model's turn.** It starts LM Studio's server and loads the model. The model then does one small task
  per session, again and again, until **Claude's window resets** (the reset time the meter reported; 5 hours if
  it didn't report one). Each step ends with the checks and a dated line in the checkpoint's
  "Local model log".
- **Handing back.**
  - At the reset time (after the local model's current step), the autopilot asks Claude for one word.
  - Once Claude answers, and the meter shows the window has reset (under 10%), it unloads the local model and
    Claude takes over. Until then, it asks again every hour.
- **Claude's review.** The first thing Claude does is review everything the local model did: fix bugs, remove
  hacks and duplication, match the project's patterns, and run all the checks. Then it continues.
- **The end.** It stops by itself when CHECKPOINT.md says `REMASTER COMPLETE`, or when the file `autopilot.stop`
  appears in this folder.

## How Claude and the local model share state

They never talk directly. They both work from the same files, and follow the same rules:

- **The shared files** in `Site 2/docs/remaster/`:
  - `CHECKPOINT.md`: the task list (ticked when done), "In progress" (what Claude is in the middle of, and the files
    it's touching), and the "Local model log";
  - `PLAN.md`: the scope;
  - `battle-notes.md`: the architecture and patterns;
  - and `git diff`.
- **"In progress" says who is doing what.** Before touching code, each model adds one line there (`**Claude:**` or
  `**Local model:**`, the task and its files) and removes it when the task is done. Claude gives each subagent's task
  its own line, with a "Left:" note it keeps current.
- **One task at a time for the local model.** A run can be cut off at any moment, so a line still there means
  unfinished work. The autopilot hands the local model exactly one task per session, in order: its own cut-off task,
  then Claude's unfinished tasks one by one (it relabels a line `**Local model:**` when it takes it over), then new
  *(small)* tasks. Nobody touches the files on another line. Claude finishes and reviews any kind.
- **The local model takes only tasks marked *(small)*, and does exactly what the task says:** no extra features, no
  refactoring, no files it didn't list. It follows the existing patterns, with no hacks or TODOs. When unsure, it
  writes a question for Claude in the log and marks the task *(blocked)*, so it never loops on it; Claude resolves
  blocked tasks.
- **Claude reviews every local entry before new work**, and marks it `(reviewed)`. Avoiding technical debt comes
  before speed.
- **Everyone stays within bounds:**
  - neither model may commit, push, branch or reset (blocked in the tool permissions too);
  - the local model may not edit `src/engine/` or the old engine.

## The dashboard: one task, one button

Run **Terminal > Run Task... > "Remaster: dashboard"** (the only task, in `.vscode/tasks.json`). It starts a small
local server (`dashboard.mjs`, no dependencies) and opens **http://localhost:8090**. Press **Start** there: the rest is
automatic. **Stop** ends the autopilot within about 10 seconds (the next start carries on from "In progress"), and unloads the local model if it was working.

The page refreshes every 2 seconds and shows:
- **Development:** overall progress, each phase and its ticked tasks, Claude's "In progress" entry, and how many
  local model log entries Claude hasn't reviewed yet (all from `CHECKPOINT.md`);
- **Models:** who is working now and for how long, when the local turn ends, and Claude's 5-hour usage;
- **Local model speed:** tokens/s per reply, from LM Studio's own stats;
- **Hand-overs and events**, and the **live activity** of both models: files read and edited, commands run, what
  each says.

The autopilot runs in a hidden window (a click in a visible console window would freeze it), so closing the dashboard (or VS Code) doesn't stop it. It writes
`logs/state.json` for the dashboard.

Logs are in `logs/` (kept out of git):
- `autopilot-live.log`, the live activity;
- `autopilot.log`, the hand-overs and events;
- one file per session, with the full stream.

Every session is also saved as a normal Claude Code conversation. From a command line, `autopilot.ps1 -Start` and
`-Stop` do the same as the buttons.

## The local model and LM Studio

- **Model.** Qwen3.5-35B-A3B (`qwen3.5-35b-a3b`, unsloth Q4_K_L), already downloaded.
  - It's a mixture-of-experts model, with about 3B active parameters per token, so it's usable on this laptop
    (RTX 4070 Laptop with 8 GB, plus 31 GB RAM).
  - It uses about 20.5 GB at the 131k context that Claude Code needs; Claude Code's opening prompt alone is about
    40k tokens.
  - Expect it to be slower and weaker than Claude: it writes about 24 tokens/s, and reading Claude Code's prompt
    at the start of each session takes about 2.5 minutes.
- **GPU tuning (2026-11-27), about 8 times faster than LM Studio's automatic setup (3 tokens/s).** The model's large
  "expert" weights stay in RAM; everything else, including the prompt memory, is on the GPU (about 5.7 GB of 8).
  These settings live in LM Studio, not in this repo:
  - `~/.lmstudio/.internal/user-concrete-model-default-config/unsloth/Qwen3.5-35B-A3B-GGUF/...gguf.json`, under
    `load`: GPU offload 1 (all layers), `numCpuExpertLayersRatio` 1 (experts in RAM), flash attention on, KV cache
    on the GPU, and K and V cache quantization `q8_0` (without it the GPU runs out of memory and the engine crashes);
  - `~/.lmstudio/.internal/hardware-config.json`: `load.gpuStrictVramCap` false. LM Studio's memory estimate wrongly
    counts the experts as going to the GPU, so its cap left 29 of the layers on the CPU;
  - the autopilot loads with `--parallel 1`, because each parallel slot needs its own GPU memory.

  Changing the model's settings in LM Studio's app can overwrite these; re-apply them if the speed drops back.
- **The autopilot controls LM Studio** through `lms` (server, load, unload). If the model keeps failing, it restarts
  the server.
- **Template fix.** The model's built-in chat template refused the system messages Claude Code sends later in a
  conversation. LM Studio's API uses the template inside the model file, so `patch_template.py` fixed it there, in
  place: the replacement is the same length, so only a few bytes change.
  - `python patch_template.py <model.gguf> --revert` undoes it.
  - Re-run the patch if the model is ever re-downloaded.
  - The fixed template is also saved as `qwen3.5-claude-code.jinja`.

## Other scripts

| Script | What it does |
|---|---|
| `autopilot.ps1 -Test` | One cheap cycle: Claude, a simulated limit, the local model, a simulated reset, Claude again |
| `autopilot.ps1 -LocalTask` | The local model does one real small task, then exits |
| `start-local.ps1` | An interactive Claude Code session on the local model in a terminal, by hand |

## Limits to know

- Anthropic doesn't officially support Claude Code on non-Claude models; it works here, but the quality is the
  local model's.
- The 95% switch relies on the usage meter Claude Code streams today. If a run ends without one, the switch falls
  back to Claude reporting the limit itself.
- The monthly spend limit doesn't reset after 5 hours. If that's the limit, the hourly checks keep the local model
  working until it does.
- Claude's runs are billed to the account as usual.
