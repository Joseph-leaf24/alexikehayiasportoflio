# Remaster autopilot: keeps work going on Claude, hands over to the local model when Claude's 5-hour usage reaches
# 95%, and hands back after 5 hours once the usage window has reset. Claude then reviews and fixes the local
# model's work before carrying on. Shared state lives in Site 2/docs/remaster/CHECKPOINT.md.
#
# Normally driven from the dashboard (dashboard.mjs, the VS Code task "Remaster: dashboard"): its Start button runs
# -Start, and Stop creates autopilot.stop in this folder, which ends the current session within about 10 seconds.
#
#   -Test        one short, cheap cycle (Claude -> simulated limit -> local model -> simulated reset -> Claude)
#   -LocalTask   the local model does one real small task, then the script exits
#   -Start       starts the autopilot in its own minimised window and returns
#   -Stop        stops a running autopilot within about 10 seconds
param([switch]$Test, [switch]$LocalTask, [switch]$Start, [switch]$Stop)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$model = "qwen3.5-35b-a3b"
$LocalHours = 5          # how long the local model works once Claude's limit is near
$SwitchAt = 0.95         # Claude's 5-hour utilization that triggers the hand-over
$BackBelow = 0.10        # utilization that counts as "reset" when handing back
$StallMinutes = 90       # a session silent this long is hung (the local model can take ~1 h to write a big file)
$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here "..\..\..\..")).Path
$checkpoint = Join-Path $repo "Site 2\docs\remaster\CHECKPOINT.md"
$stopFile = Join-Path $here "autopilot.stop"
$logDir = Join-Path $here "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "autopilot.log"
$live = Join-Path $logDir "autopilot-live.log"
$lockFile = Join-Path $logDir "autopilot.lock"
$lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
$ext = Get-ChildItem (Join-Path $env:USERPROFILE ".vscode\extensions") -Directory -Filter "anthropic.claude-code-*" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$claude = Join-Path $ext.FullName "resources\native-binary\claude.exe"

# A running autopilot holds autopilot.lock open; Windows releases it the moment the process ends, however it ends.
function Test-Running {
  try { [IO.File]::Open($lockFile, "OpenOrCreate", "ReadWrite", "None").Close(); return $false } catch { return $true }
}

function Start-Autopilot {
  Remove-Item $stopFile -ErrorAction SilentlyContinue
  Start-Process powershell -WindowStyle Minimized -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
}

if ($Stop) {
  New-Item -ItemType File -Force $stopFile | Out-Null
  Write-Host "Stop requested: the autopilot ends its current session within about 10 seconds."
  exit 0
}
if ($Start) {
  if (Test-Running) { Write-Host "The autopilot is already running."; exit 1 }
  Start-Autopilot
  Write-Host "Autopilot started (minimised window)."
  exit 0
}

# One autopilot at a time. (A few tries: the dashboard's check opens the file for an instant every 2 seconds.)
$lock = $null
foreach ($try in 1..5) {
  try { $lock = [IO.File]::Open($lockFile, "OpenOrCreate", "ReadWrite", "None"); break } catch { Start-Sleep -Milliseconds 300 }
}
if (-not $lock) { Write-Host "The autopilot is already running."; exit 1 }

# Keep the PC awake while this process runs (released automatically when it exits).
Add-Type -Namespace Autopilot -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[void][Autopilot.Power]::SetThreadExecutionState([uint32]"0x80000001")

$guard = "Never run git commit, push, branch, checkout, switch, reset, rebase, merge or stash. Never create branches or pull requests."

$claudePrompt = @"
You are continuing the Site 2 remaster unattended, working hand in hand with a weaker local model that takes over
while your usage limit is near. Read CLAUDE.md, Site 2/docs/remaster/CHECKPOINT.md, PLAN.md and battle-notes.md.

Every run works in this order, so bugs never pile up:
1. Fix bugs first: everything under "Known bugs" in CHECKPOINT.md, and anything the checks report (run them all in
   Site 2/). Remove each bug from the list once it's fixed and verified. If the checks fail only because the task
   under "In progress" is unfinished, finishing that task is the fix.
2. Review the local model's work: for every "Local model log" entry not marked (reviewed), check those changes (git
   diff and the files they name). Fix bugs, remove anything hacky or duplicated, make it follow the existing patterns
   and code standard, run the checks again, and mark the entries (reviewed). A "**Local model:**" line under "In
   progress" is a local task that was cut off: review its partial work in git diff, finish it, and remove the line.
   A task marked (blocked) waits on the local model's question in the log: resolve it and remove the mark. Avoiding
   technical debt matters more than speed.
3. Then continue: if "In progress" has a "**Claude:**" line, finish that task; otherwise take the next unticked task
   in "Next tasks". Before starting a task, add one line under "In progress": **Claude:** <task>. Files: <files>.
   When it's done and verified, tick it, remove your line, and update the notes (battle-notes.md and so on) if the
   architecture changed. Your run can be stopped at any moment, so keep CHECKPOINT.md current: it is how you and the
   local model share state. Work only on the checkpoint's tasks, within PLAN.md's scope. A bug you find but don't fix
   straight away goes under "Known bugs".
4. Keep scope lean (tokens are limited). When every phase in PLAN.md is done and verified, add the line
   "REMASTER COMPLETE" to CHECKPOINT.md.

This run is unattended: do the work yourself or with foreground subagents, and don't leave background agents running
when you finish. $guard
"@

$localPrompt = @"
You are a local model helping Claude with the Site 2 remaster while Claude's usage limit is near. Claude reviews all
your work later, so be careful and conservative. Read CLAUDE.md, then Site 2/docs/remaster/CHECKPOINT.md and
Site 2/docs/remaster/battle-notes.md.

Choose your ONE task:
- If "In progress" has a "**Local model:**" line, your previous session was cut off: continue exactly that task (git
  diff shows what it already changed).
- Otherwise take the FIRST unticked task in "Next tasks" marked (small) and not marked (blocked). Before changing any
  code, add one line under "In progress": **Local model:** <the task, as written>. Files: <files you'll create or
  change>.
Do only that task, exactly as written: no extra features, no refactoring, no other files (if you need another file,
add it to your line first). Never touch the files on a "**Claude:**" line, Site 2/src/engine/ or
Site 2/src/components/pixel/, and don't delete files.
Study the existing code it builds on first and follow its patterns exactly: no hacks, no TODOs, no duplicated code,
small files. You are slow, so keep each reply short: to move existing code or data, copy the file with a node command
and then edit the copy, rather than retyping it; write a new big file in parts of about 150 lines. Then run npm run
typecheck, npm run lint, npm run format:check and npx vitest run in Site 2/, and fix what fails.

When done, tick the task, remove your "**Local model:**" line, and add a dated line under "Local model log" in
CHECKPOINT.md: what you changed, the files, and whether every check passed. If you notice a bug outside your task,
add one line about it under "Known bugs" instead of fixing it. If you get stuck or are unsure, don't guess: write the
question for Claude in the log (with the files you changed), mark the task (blocked), remove your "In progress" line,
and stop. $guard
"@

$allowed = @("Bash(npm *)", "Bash(npx *)", "Bash(node *)", "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)")
$denied = @("Bash(git commit*)", "Bash(git push*)", "Bash(git branch*)", "Bash(git checkout*)", "Bash(git switch*)",
  "Bash(git reset*)", "Bash(git rebase*)", "Bash(git merge*)", "Bash(git stash*)", "Bash(rm -rf*)")
$localEnv = [ordered]@{
  ANTHROPIC_BASE_URL = "http://localhost:1234"; ANTHROPIC_AUTH_TOKEN = "lmstudio"
  ANTHROPIC_MODEL = $model; ANTHROPIC_DEFAULT_OPUS_MODEL = $model; ANTHROPIC_DEFAULT_SONNET_MODEL = $model
  ANTHROPIC_DEFAULT_HAIKU_MODEL = $model; CLAUDE_CODE_SUBAGENT_MODEL = $model
  CLAUDE_CODE_ATTRIBUTION_HEADER = "0"; CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS = "1"
  CLAUDE_CODE_AUTO_COMPACT_WINDOW = "120000"  # the model's 131k window; Claude Code assumes 1M for unknown models
  # The local model is slower than Claude (about 24 tokens/s, after 2+ minutes of silent prompt reading per
  # session): lift Claude Code's stream limits to their maximums so a long reply isn't cut off and retried.
  API_TIMEOUT_MS = "3600000"; CLAUDE_STREAM_IDLE_TIMEOUT_MS = "1800000"; CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS = "1800000"
}

# What the dashboard shows: which model is working, since when, until when, and Claude's last usage reading.
$stateFile = Join-Path $logDir "state.json"
$state = [ordered]@{ pid = $PID; mode = "starting"; since = ""; localUntil = ""; util = -1; resetsAt = "" }

function Set-State([hashtable]$changes) {
  foreach ($name in $changes.Keys) { $state[$name] = $changes[$name] }
  $tmp = "$stateFile.tmp"
  [IO.File]::WriteAllText($tmp, ($state | ConvertTo-Json -Compress))
  Move-Item -Force $tmp $stateFile
}

function Set-Mode([string]$mode, $until = "") {
  $untilText = if ($until) { ([datetime]$until).ToString("o") } else { "" }
  Set-State @{ mode = $mode; since = (Get-Date).ToString("o"); localUntil = $untilText }
}

# Appends a line while letting the dashboard read the file at the same time (Add-Content blocks readers).
$utf8 = New-Object Text.UTF8Encoding($false)
function Add-Line([string]$path, [string]$line) {
  $bytes = $utf8.GetBytes($line + "`r`n")
  foreach ($try in 1..20) {
    try {
      $file = [IO.File]::Open($path, "Append", "Write", "ReadWrite")
      try { $file.Write($bytes, 0, $bytes.Length) } finally { $file.Close() }
      return
    } catch [IO.IOException] { Start-Sleep -Milliseconds 100 }
  }
}

function Write-Live($text) {
  $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $text
  Add-Line $live $line
  Write-Host $line
}

function Write-Log($text) {
  Add-Line $log ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text)
  Write-Live "== $text"
}

# Waits, but returns as soon as a stop is requested.
function Wait-UnlessStopped([int]$seconds) {
  $until = (Get-Date).AddSeconds($seconds)
  while ((Get-Date) -lt $until -and -not (Test-Path $stopFile)) { Start-Sleep -Seconds 5 }
}

function Get-Short([string]$text, [int]$max) {
  $one = ($text -replace "\s+", " ").Trim()
  if ($one.Length -le $max) { return $one }
  return $one.Substring(0, $max) + "..."
}

function Get-InputSummary($inputs) {
  foreach ($name in "file_path", "command", "pattern", "path", "description", "prompt", "url") {
    $value = $inputs.$name
    if ($value) { return (Get-Short ([string]$value) 140) }
  }
  return ""
}

function Format-Arg([string]$a) { if ($a -match '[\s()*]') { return '"' + $a + '"' } return $a }

# Highest utilization across the usage windows in a rate_limit_event (0..1), or -1 if it has none.
function Get-Utilization($info) {
  $max = -1.0
  if ($info.unifiedWindows) {
    foreach ($w in $info.unifiedWindows.PSObject.Properties) {
      if ($null -ne $w.Value.utilization) { $max = [Math]::Max($max, [double]$w.Value.utilization) }
    }
  }
  return $max
}

# One headless Claude Code session on Anthropic ("claude") or LM Studio ("local"). Streams its actions to the live
# log. A Claude session is stopped as soon as its usage reaches $SwitchAt or the limit is reported.
function Invoke-Session([string]$backend, [string]$prompt, [int]$maxTurns) {
  $tag = if ($backend -eq "local") { "[Local]" } else { "[Claude]" }
  $psi = New-Object Diagnostics.ProcessStartInfo
  $psi.FileName = $claude
  $argv = @("-p", "--max-turns", "$maxTurns", "--permission-mode", "acceptEdits", "--output-format", "stream-json", "--verbose")
  if ($backend -eq "local") { $argv += @("--model", $model) }
  $argv += @("--allowedTools") + $allowed + @("--disallowedTools") + $denied
  $psi.Arguments = ($argv | ForEach-Object { Format-Arg $_ }) -join " "
  $psi.WorkingDirectory = $repo
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
  $psi.StandardErrorEncoding = [Text.Encoding]::UTF8
  foreach ($name in $localEnv.Keys) { [void]$psi.EnvironmentVariables.Remove($name) }
  if ($backend -eq "local") { foreach ($name in $localEnv.Keys) { $psi.EnvironmentVariables[$name] = $localEnv[$name] } }

  $proc = [Diagnostics.Process]::Start($psi)
  $errTask = $proc.StandardError.ReadToEndAsync()
  # Raw UTF-8 bytes: the StreamWriter would put a byte-order mark in front of the prompt.
  $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($prompt)
  $proc.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
  $proc.StandardInput.Close()

  $lines = New-Object System.Collections.Generic.List[string]
  $limited = $false
  $util = -1.0
  $resetsAt = $null
  $lastOutput = Get-Date
  $next = $proc.StandardOutput.ReadLineAsync()
  while ($true) {
    if (Test-Path $stopFile) {
      Write-Live "$tag stop requested: ending this session"
      & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
      break
    }
    if (-not $next.Wait(5000)) {
      if (((Get-Date) - $lastOutput).TotalMinutes -lt $StallMinutes) { continue }
      Write-Live "$tag no output for $StallMinutes minutes: stopping this session"
      & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
      break
    }
    $line = $next.Result
    if ($null -eq $line) { break }
    $lastOutput = Get-Date
    $next = $proc.StandardOutput.ReadLineAsync()
    $lines.Add($line)
    if (-not $line.Trim()) { continue }
    try { $evt = $line | ConvertFrom-Json } catch { continue }
    switch ($evt.type) {
      "rate_limit_event" {
        $info = $evt.rate_limit_info
        $util = Get-Utilization $info
        if ($info.resetsAt) { $resetsAt = [DateTimeOffset]::FromUnixTimeSeconds([int64]$info.resetsAt).LocalDateTime }
        if ($util -ge 0) { Set-State @{ util = $util; resetsAt = $(if ($resetsAt) { $resetsAt.ToString("o") } else { "" }) } }
        if ($info.status -notlike "allowed*" -or $util -ge $SwitchAt) { $limited = $true }
      }
      "assistant" {
        foreach ($part in $evt.message.content) {
          if ($part.type -eq "text" -and $part.text.Trim()) { Write-Live "$tag says: $(Get-Short $part.text 300)" }
          elseif ($part.type -eq "tool_use") { Write-Live ("$tag {0}: {1}" -f $part.name, (Get-InputSummary $part.input)) }
        }
      }
      "result" {
        Write-Live ("$tag finished ({0}, {1} turns): {2}" -f $evt.subtype, $evt.num_turns, (Get-Short ([string]$evt.result) 200))
        if ($evt.is_error -and ([string]$evt.result) -match "usage limit|spend limit|rate.?limit|limit reached|hit your .*limit") { $limited = $true }
      }
    }
    if ($limited -and $backend -eq "claude") {
      Write-Live ("$tag usage at {0:P0}: stopping this run to hand over" -f [Math]::Max($util, 0))
      & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
      break
    }
  }
  $proc.WaitForExit()
  $stderr = $errTask.Result
  if ($stderr -match "usage limit|spend limit|rate.?limit|limit reached|hit your .*limit|\b429\b") { $limited = $true }
  $runLog = Join-Path $logDir ("{0}-{1}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $backend)
  ($lines + @("--- stderr ---", $stderr)) | Set-Content -Path $runLog -Encoding UTF8
  if ($util -ge 0) {
    Write-Live ("$tag usage: {0:P0} of the 5-hour window" -f $util)
    Set-State @{ util = $util; resetsAt = $(if ($resetsAt) { $resetsAt.ToString("o") } else { "" }) }
  }
  return @{ Code = $proc.ExitCode; Limited = $limited; Util = $util; ResetsAt = $resetsAt; Log = $runLog }
}

# LM Studio is driven entirely from here: server, model, context.
function Start-Local {
  & $lms server start --port 1234 2>&1 | Out-Null
  if (-not ((& $lms ps 2>&1 | Out-String) -match [regex]::Escape($model))) {
    Write-Log "loading the local model ($model, 131k context)"
    & $lms unload --all 2>&1 | Out-Null
    # One request at a time: each parallel slot needs its own GPU memory (LM Studio's tuned settings: LOCAL-MODEL.md).
    & $lms load $model --context-length 131072 --parallel 1 --identifier $model -y 2>&1 | Out-Null
  }
}

function Stop-Local { & $lms unload --all 2>&1 | Out-Null }

if ($Test) {
  Set-Mode "test"
  Write-Log "TEST 1/4: Claude answers and reports its usage"
  $r = Invoke-Session "claude" "Reply with the single word OK." 1
  Write-Log ("TEST: Claude exit {0}, limit: {1}, usage {2:P0}, window resets {3}" -f $r.Code, $r.Limited, $r.Util, $r.ResetsAt)
  Write-Log "TEST 2/4: simulated limit, starting the local model"
  Start-Local
  Write-Log "TEST 3/4: the local model reads the checkpoint with Claude Code's tools"
  $read = "Use the Read tool to read Site 2/docs/remaster/CHECKPOINT.md, then reply with one line: the first unticked task marked (small). Don't change any files."
  $r = Invoke-Session "local" $read 4
  Write-Log ("TEST: local exit {0}" -f $r.Code)
  Write-Log "TEST 4/4: simulated reset, Claude probe, then unload"
  $p = Invoke-Session "claude" "Reply with the single word OK." 1
  Stop-Local
  Write-Log ("TEST: Claude exit {0}, limit: {1}; local model unloaded" -f $p.Code, $p.Limited)
  Set-Mode "stopped"
  exit 0
}

if ($LocalTask) {
  Set-Mode "local"
  Write-Log "LOCAL TASK: one real small task on the local model"
  Start-Local
  $r = Invoke-Session "local" $localPrompt 80
  Stop-Local
  Write-Log ("LOCAL TASK: exit {0} (log: {1})" -f $r.Code, $r.Log)
  Set-Mode "stopped"
  exit 0
}

Write-Log "autopilot started (Claude Code $($ext.Name)); hand-over at $($SwitchAt.ToString('P0')) usage; a session silent for $StallMinutes min counts as hung"
$mode = "claude"
Set-Mode "claude"
$resetAt = Get-Date
$localFailures = 0

while ($true) {
  if (Test-Path $stopFile) { Remove-Item $stopFile; Write-Log "stop file found: stopping"; break }
  if ((Get-Content $checkpoint -Raw) -match "REMASTER COMPLETE") { Write-Log "the remaster is complete: stopping"; break }

  if ($mode -eq "claude") {
    Write-Log "Claude run starting"
    $r = Invoke-Session "claude" $claudePrompt 400
    if (Test-Path $stopFile) { continue }
    if ($r.Limited) {
      $resetAt = (Get-Date).AddHours($LocalHours)
      Write-Log "Claude's usage is at the limit; the local model works until $resetAt"
      Start-Local
      $mode = "local"
      Set-Mode "local" $resetAt
      $localFailures = 0
    } elseif ($r.Code -ne 0) {
      Write-Log "Claude run ended with an error (exit $($r.Code)); retrying in 10 minutes (log: $($r.Log))"
      Wait-UnlessStopped 600
    } else {
      Write-Log "Claude run finished (log: $($r.Log))"
    }
    continue
  }

  # Local mode. After the 5 hours, hand back once Claude answers and its usage window has reset (else check hourly).
  if ((Get-Date) -ge $resetAt) {
    $probe = Invoke-Session "claude" "Reply with the single word OK." 1
    $reset = $probe.Code -eq 0 -and -not $probe.Limited -and ($probe.Util -lt 0 -or $probe.Util -lt $BackBelow)
    if ($reset) {
      Write-Log "Claude's usage has reset: unloading the local model; Claude reviews the local work next"
      Stop-Local
      $mode = "claude"
      Set-Mode "claude"
      continue
    }
    $resetAt = (Get-Date).AddHours(1)
    Set-State @{ localUntil = $resetAt.ToString("o") }
    Write-Log ("Claude not reset yet (usage {0:P0}); next check at {1}" -f [Math]::Max($probe.Util, 0), $resetAt)
  }

  Write-Log "local model step starting"
  Start-Local
  $r = Invoke-Session "local" $localPrompt 80
  if (Test-Path $stopFile) { continue }
  if ($r.Code -eq 0) {
    $localFailures = 0
    Write-Log "local model step finished (log: $($r.Log))"
  } else {
    $localFailures++
    Write-Log "local model step failed (exit $($r.Code), $localFailures in a row; log: $($r.Log))"
    if ($localFailures -ge 3) {
      Write-Log "the local model keeps failing; restarting LM Studio's server and waiting 15 minutes"
      & $lms server stop 2>&1 | Out-Null
      Stop-Local
      Wait-UnlessStopped 900
      $localFailures = 0
    } else {
      Wait-UnlessStopped 60
    }
  }
}
if ($mode -eq "local") { Stop-Local }
Write-Log "autopilot stopped"
Set-Mode $(if ((Get-Content $checkpoint -Raw) -match "REMASTER COMPLETE") { "complete" } else { "stopped" })
$lock.Close()
