// Fails if credentials (API keys, tokens, private keys, hard-coded passwords) are
// present in the working tree. Values are redacted in the output.
// Usage: node scripts/scan-secrets.mjs [dir]
import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";

const root = process.argv[2] ?? ".";
const SKIP_DIRS = new Set([".git", "node_modules", "dist", "coverage", ".ipynb_checkpoints", "__pycache__"]);
const TEXT = new Set([
  ".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".json", ".md", ".yml", ".yaml",
  ".toml", ".cfg", ".ini", ".env", ".txt", ".html", ".css", ".sh", ".urdf",
]);
const MAX_BYTES = 3 * 1024 * 1024;

const RULES = [
  { name: "OpenAI / Stripe secret key", re: /\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}/g },
  { name: "AWS access key", re: /\bAKIA[0-9A-Z]{16}\b/g },
  { name: "GitHub token", re: /\bgh[pousr]_[A-Za-z0-9]{36,}\b/g },
  { name: "Google API key", re: /\bAIza[0-9A-Za-z_-]{35}\b/g },
  { name: "Slack token", re: /\bxox[baprs]-[A-Za-z0-9-]{10,}/g },
  { name: "Private key", re: /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/g },
  {
    name: "Hard-coded password / secret",
    re: /\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\b['"]?\s*[:=]\s*['"]([^'"\s]{6,})['"]/gi,
    value: 1,
  },
];
// Obvious placeholders and test fixtures are fine
const ALLOW = /^(?:test|dummy|example|fake|mock|sample|changeme|your[_-]|<|\$\{|\{\{|x{3,}|\*{3,}|os\.environ)/i;

const findings = [];
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const st = statSync(path);
    if (st.isDirectory()) {
      if (!SKIP_DIRS.has(name)) walk(path);
    } else if (TEXT.has(extname(name).toLowerCase()) && st.size <= MAX_BYTES && !name.endsWith("package-lock.json")) {
      scan(path);
    }
  }
}
function scan(path) {
  const lines = readFileSync(path, "utf8").split(/\r?\n/);
  lines.forEach((line, i) => {
    for (const rule of RULES) {
      for (const m of line.matchAll(rule.re)) {
        const secret = rule.value ? m[rule.value] : m[0];
        if (rule.value && ALLOW.test(secret)) continue;
        findings.push({ file: relative(root, path), line: i + 1, rule: rule.name, preview: `${secret.slice(0, 4)}…(${secret.length} chars)` });
      }
    }
  });
}

walk(root);
if (findings.length) {
  console.error(`✖ ${findings.length} possible secret(s) found:`);
  for (const f of findings) console.error(`  ${f.file}:${f.line}  ${f.rule}  ${f.preview}`);
  console.error("Move them to environment variables / repository secrets, and rotate any that were pushed.");
  process.exit(1);
}
console.log("✓ No secrets found.");
