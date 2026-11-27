# Starts LM Studio's server with the local model and opens Claude Code on it, in this terminal only.
# Other Claude Code chats (the VS Code panel) keep using Anthropic.
# Run from any PowerShell terminal:
#   powershell -ExecutionPolicy Bypass -File "Site 2\docs\remaster\local-llm\start-local.ps1"

$ErrorActionPreference = "Stop"
$model = "qwen3.5-35b-a3b"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"

& $lms server start --port 1234
& $lms unload --all
# Claude Code's first request alone is about 40k tokens, so it needs a large window.
& $lms load $model --context-length 131072 --identifier $model -y

# Claude Code ships inside the VS Code extension; use the newest installed copy.
$ext = Get-ChildItem (Join-Path $env:USERPROFILE ".vscode\extensions") -Directory -Filter "anthropic.claude-code-*" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$claude = Join-Path $ext.FullName "resources\native-binary\claude.exe"

# Route this process (only) to LM Studio's Anthropic-compatible endpoint.
$env:ANTHROPIC_BASE_URL = "http://localhost:1234"
$env:ANTHROPIC_AUTH_TOKEN = "lmstudio"
$env:ANTHROPIC_MODEL = $model
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = $model
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = $model
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $model
$env:CLAUDE_CODE_SUBAGENT_MODEL = $model
$env:CLAUDE_CODE_ATTRIBUTION_HEADER = "0"
$env:CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS = "1"
$env:CLAUDE_CODE_AUTO_COMPACT_WINDOW = "120000"

Set-Location $repo
# Edits are accepted automatically; only the project's check commands run without asking.
& $claude --model $model --permission-mode acceptEdits `
  --allowedTools "Bash(npm run typecheck*)" "Bash(npm run lint*)" "Bash(npm run format*)" "Bash(npx vitest*)" "Bash(npx tsc*)" "Bash(npx eslint*)" "Bash(npx prettier*)"
