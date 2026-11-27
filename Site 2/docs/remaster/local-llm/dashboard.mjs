// The remaster dashboard: `node dashboard.mjs` (the VS Code task "Remaster: dashboard") serves
// http://localhost:8090 and opens it. It shows the development progress (from CHECKPOINT.md), which model is working,
// the hand-overs, the live activity and the local model's speed; its buttons start and stop the autopilot.
import { spawn } from "node:child_process";
import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = 8090;
const ADDRESS = `http://localhost:${PORT}`;
const HOSTS = new Set([`localhost:${PORT}`, `127.0.0.1:${PORT}`]);
const here = dirname(fileURLToPath(import.meta.url));
const logs = join(here, "logs");
const checkpoint = join(here, "..", "CHECKPOINT.md");
const stopFile = join(here, "autopilot.stop");
const lockFile = join(logs, "autopilot.lock");
const lms = join(homedir(), ".lmstudio", "bin", "lms.exe");

// The autopilot writes these files while we read them. A read that fails returns the last good copy.
const lastGood = new Map();
const clean = (text) => text.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
function read(file) {
  try {
    lastGood.set(file, clean(readFileSync(file, "utf8")));
  } catch {
    // Missing, or busy for a moment.
  }
  return lastGood.get(file) ?? "";
}

// The last lines of a log, reading only its end: the live log grows by megabytes a day.
function lastLines(file, n) {
  try {
    const fd = openSync(file, "r");
    try {
      const size = fstatSync(fd).size;
      const start = Math.max(0, size - 64 * 1024);
      const buffer = Buffer.alloc(size - start);
      readSync(fd, buffer, 0, buffer.length, start);
      const lines = clean(buffer.toString("utf8")).split("\n").filter(Boolean);
      lastGood.set(file, (start > 0 ? lines.slice(1) : lines).slice(-n).join("\n"));
    } finally {
      closeSync(fd);
    }
  } catch {
    // Missing, or busy for a moment.
  }
  return (lastGood.get(file) ?? "").split("\n").filter(Boolean);
}

// Only what happened since the latest start (a run, a test or a single local task), not older runs.
// (The live feed writes the same lines as autopilot.log, with "== " in front.)
const START = / {2}(== )?(autopilot started|TEST 1\/4|LOCAL TASK:)/;
function sinceLatestStart(lines) {
  const at = lines.findLastIndex((line) => START.test(line));
  return at < 0 ? lines : lines.slice(at);
}

// A running autopilot holds autopilot.lock open, so opening it fails until the process ends.
function isRunning() {
  try {
    closeSync(openSync(lockFile, "r+"));
    return false;
  } catch (err) {
    return err.code === "EBUSY";
  }
}

// The local model's speed: LM Studio reports stats for every reply it finishes.
const replies = [];
let lmsLog = null;
function watchLmStudio() {
  lmsLog = spawn(lms, ["log", "stream", "--source", "model", "--filter", "output", "--stats", "--json"]);
  lmsLog.stdout.setEncoding("utf8");
  let buffer = "";
  lmsLog.stdout.on("data", (chunk) => {
    const lines = (buffer + chunk).split(/\r?\n/);
    buffer = lines.pop();
    for (const line of lines) {
      let stats;
      try {
        stats = JSON.parse(line).data?.stats;
      } catch {
        continue; // not JSON ("Streaming logs from LM Studio")
      }
      if (typeof stats?.tokensPerSecond !== "number") continue;
      replies.push({
        at: Date.now(),
        tps: stats.tokensPerSecond,
        tokens: stats.predictedTokensCount ?? 0,
        prompt: stats.promptTokensCount ?? 0,
        ttft: stats.timeToFirstTokenSec ?? 0,
      });
      if (replies.length > 200) replies.shift();
    }
  });
  lmsLog.on("error", () => {});
  lmsLog.on("exit", () => setTimeout(watchLmStudio, 30_000));
}

// Progress from CHECKPOINT.md: the phase table, and the ticked tasks under each "### <phase>" heading.
function progress(md) {
  const section = (title) => md.split(/^## /m).find((s) => s.startsWith(title)) ?? "";
  // A section's body after its title and the paragraph explaining it.
  const entries = (title) =>
    section(title)
      .split(/\n\s*\n/)
      .slice(2)
      .join("\n")
      .trim();
  const phases = [];
  for (const row of section("Where things stand").split("\n")) {
    const cells = row.split("|").map((c) => c.trim());
    if (cells.length < 4 || cells[1] === "Phase" || /^-+$/.test(cells[1])) continue;
    const done = /done|✅/i.test(cells[2]);
    for (const name of cells[1].split(",")) phases.push({ name: name.split(":")[0].trim(), done, tasks: 0, ticked: 0 });
  }
  let current = null;
  for (const line of section("Next tasks").split("\n")) {
    const heading = line.match(/^### ([A-Z]\d?)/);
    if (heading) current = phases.find((p) => p.name === heading[1]);
    const box = line.match(/^- \[([ xX])\]/);
    if (box && current) {
      current.tasks++;
      if (box[1] !== " ") current.ticked++;
    }
  }
  const share = (p) => (p.done ? 1 : p.tasks ? p.ticked / p.tasks : 0);
  const localLog = section("Local model log")
    .split("\n")
    .filter((l) => /^- /.test(l));
  return {
    phases,
    overall: phases.length ? phases.reduce((sum, p) => sum + share(p), 0) / phases.length : 0,
    complete: /REMASTER COMPLETE/.test(md),
    bugs: section("Known bugs")
      .split("\n")
      .filter((l) => /^- /.test(l)).length,
    inProgress: entries("In progress"),
    localLog: { entries: localLog.length, unreviewed: localLog.filter((l) => !/\(reviewed\)/.test(l)).length },
  };
}

function snapshot() {
  let state = {};
  try {
    state = JSON.parse(read(join(logs, "state.json")) || "{}");
  } catch {
    // Mid-write: the next poll gets it.
  }
  const running = isRunning();
  return {
    running,
    stopping: running && existsSync(stopFile),
    state,
    progress: progress(read(checkpoint)),
    events: sinceLatestStart(lastLines(join(logs, "autopilot.log"), 200)),
    feed: sinceLatestStart(lastLines(join(logs, "autopilot-live.log"), 400)).slice(-150),
    replies: replies.slice(-60),
  };
}

function startAutopilot() {
  const script = join(here, "autopilot.ps1");
  spawn("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Start"], { stdio: "ignore" });
}

function handle(req, res) {
  const send = (code, type, body) => {
    res.writeHead(code, { "content-type": type, "cache-control": "no-store" });
    res.end(body);
  };
  // Only this machine's browser: blocks other websites reaching it through DNS tricks.
  if (!HOSTS.has(req.headers.host)) return send(403, "text/plain", "forbidden");
  if (req.method === "GET" && req.url === "/")
    return send(200, "text/html; charset=utf-8", read(join(here, "dashboard.html")));
  if (req.method === "GET" && req.url === "/api/state")
    return send(200, "application/json", JSON.stringify(snapshot()));
  // Buttons send this header; other websites can't, so they can't press them.
  if (req.method === "POST" && req.headers["x-dashboard"] === "1") {
    if (req.url === "/api/start") {
      if (!isRunning()) startAutopilot();
      return send(204, "text/plain", "");
    }
    if (req.url === "/api/stop") {
      if (isRunning()) writeFileSync(stopFile, "");
      return send(204, "text/plain", "");
    }
  }
  send(404, "text/plain", "not found");
}

const server = createServer((req, res) => {
  try {
    handle(req, res);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) res.writeHead(500, { "content-type": "text/plain" });
    res.end("error");
  }
});

const open = () => spawn("cmd", ["/c", "start", "", ADDRESS], { stdio: "ignore", detached: true }).unref();
server.on("error", (err) => {
  if (err.code !== "EADDRINUSE") throw err;
  console.log(`The dashboard is already running: ${ADDRESS}`);
  open();
});
server.listen(PORT, "127.0.0.1", () => {
  console.log(`Remaster dashboard: ${ADDRESS} (closing this terminal stops the dashboard, not the autopilot)`);
  watchLmStudio();
  open();
});

const quit = () => {
  lmsLog?.kill();
  process.exit(0);
};
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(signal, quit);
