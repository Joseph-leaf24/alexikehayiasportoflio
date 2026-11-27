/**
 * README badge data. CI collects the numbers (unit tests, coverage, npm audit,
 * Lighthouse) and the deploy publishes them under /badges/ on the live site as
 * shields.io "endpoint" JSON files, which the README badges read.
 *
 *   node scripts/badges.mjs collect            prints tests / coverage / vulnerabilities lines for $GITHUB_OUTPUT
 *                                              (run after `vitest --coverage` with the JSON reporter)
 *   node scripts/badges.mjs lighthouse <dir>   prints a lighthouse=<json> line from an LHCI manifest.json
 *   node scripts/badges.mjs write <outDir>     writes the badge files from the BADGE_* environment variables
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

/** Lighthouse's own colour bands: 90–100 good, 50–89 needs work, below 50 poor. */
export const scoreColor = (score) => (score >= 0.9 ? "brightgreen" : score >= 0.5 ? "orange" : "red");

/** Coverage badge colour by percentage band. */
export const coverageColor = (pct) => (pct >= 90 ? "brightgreen" : pct >= 80 ? "green" : pct >= 70 ? "yellow" : "red");

/** One shields.io endpoint badge (https://shields.io/badges/endpoint-badge). */
export const badge = (label, message, color) => ({ schemaVersion: 1, label, message: String(message), color });

/** Unit-test counts and statement coverage from Vitest's JSON results + the v8 coverage summary. */
export function summariseUnit(results, coverageSummary) {
  return {
    passed: results.numPassedTests,
    failed: results.numFailedTests,
    coverage: Math.round(coverageSummary.total.statements.pct * 10) / 10,
  };
}

/** Number of known vulnerabilities in an `npm audit --json` report; undefined if the audit itself failed, so no badge claims 0. */
export const countVulnerabilities = (audit) => {
  const counts = audit?.metadata?.vulnerabilities;
  return counts ? Object.values(counts).reduce((sum, n) => sum + n, 0) : undefined;
};

/** Lowest score per Lighthouse category across all audited pages, the same rule CI asserts with. */
export function summariseLighthouse(manifest) {
  const out = {};
  for (const run of manifest) {
    for (const [category, score] of Object.entries(run.summary ?? {})) {
      if (typeof score === "number") out[category] = Math.min(out[category] ?? 1, score);
    }
  }
  return out;
}

const LIGHTHOUSE_LABELS = {
  performance: "performance",
  accessibility: "accessibility",
  "best-practices": "best practices",
  seo: "SEO",
};

/** File name → badge, for every value that is present. */
export function buildBadges({ passed, failed, coverage, vulnerabilities, lighthouse }) {
  const files = {};
  if (Number.isFinite(passed)) {
    files["tests.json"] =
      failed > 0 ? badge("tests", `${failed} failed`, "red") : badge("tests", `${passed} passed`, "brightgreen");
  }
  if (Number.isFinite(coverage)) files["coverage.json"] = badge("coverage", `${coverage}%`, coverageColor(coverage));
  if (Number.isFinite(vulnerabilities)) {
    files["vulnerabilities.json"] = badge(
      "vulnerabilities",
      vulnerabilities,
      vulnerabilities === 0 ? "brightgreen" : "red"
    );
  }
  for (const [category, label] of Object.entries(LIGHTHOUSE_LABELS)) {
    const score = lighthouse?.[category];
    if (Number.isFinite(score))
      files[`lighthouse-${category}.json`] = badge(label, Math.round(score * 100), scoreColor(score));
  }
  return files;
}

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const num = (v) => (v === undefined || v === "" ? undefined : Number(v));

function collect() {
  const unit = summariseUnit(readJson("coverage/test-results.json"), readJson("coverage/coverage-summary.json"));
  // npm exits non-zero when it finds anything, so read the JSON whatever the status
  const audit = spawnSync("npm", ["audit", "--omit=dev", "--json"], {
    encoding: "utf8",
    shell: process.platform === "win32",
  });
  let report = {};
  try {
    report = JSON.parse(audit.stdout);
  } catch {
    /* no usable report: that badge keeps its last value */
  }
  const vulnerabilities = countVulnerabilities(report) ?? "";
  return `passed=${unit.passed}\nfailed=${unit.failed}\ncoverage=${unit.coverage}\nvulnerabilities=${vulnerabilities}\n`;
}

function lighthouse(dir) {
  const path = join(dir, "manifest.json");
  if (!existsSync(path)) throw new Error(`No Lighthouse manifest at ${path}`);
  return `lighthouse=${JSON.stringify(summariseLighthouse(readJson(path)))}\n`;
}

function write(outDir) {
  const env = process.env;
  const files = buildBadges({
    passed: num(env.BADGE_PASSED),
    failed: num(env.BADGE_FAILED),
    coverage: num(env.BADGE_COVERAGE),
    vulnerabilities: num(env.BADGE_VULNERABILITIES),
    lighthouse: env.BADGE_LIGHTHOUSE ? JSON.parse(env.BADGE_LIGHTHOUSE) : undefined,
  });
  mkdirSync(outDir, { recursive: true });
  for (const [name, data] of Object.entries(files)) writeFileSync(join(outDir, name), JSON.stringify(data) + "\n");
  return `badges: wrote ${Object.keys(files).join(", ") || "nothing"} to ${outDir}\n`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [command, arg] = process.argv.slice(2);
  const run = {
    collect,
    lighthouse: () => lighthouse(arg ?? ".lighthouseci"),
    write: () => write(arg ?? "dist/badges"),
  }[command];
  if (!run) {
    console.error("usage: node scripts/badges.mjs collect | lighthouse <dir> | write <outDir>");
    process.exit(1);
  }
  process.stdout.write(run());
}
