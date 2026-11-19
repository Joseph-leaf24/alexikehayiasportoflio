// Turns the Vitest JSON results + v8 coverage summary into src/data/test-report.json,
// which the site renders at the bottom of the home page.
import { readFileSync, writeFileSync } from "node:fs";

const coverage = JSON.parse(readFileSync("coverage/coverage-summary.json", "utf8")).total;
const results = JSON.parse(readFileSync("coverage/test-results.json", "utf8"));

const report = {
  generatedAt: new Date().toISOString(),
  threshold: 80,
  tests: {
    total: results.numTotalTests,
    passed: results.numPassedTests,
    failed: results.numFailedTests,
    files: results.testResults.length,
    suites: results.numTotalTestSuites,
  },
  coverage: {
    statements: coverage.statements.pct,
    branches: coverage.branches.pct,
    functions: coverage.functions.pct,
    lines: coverage.lines.pct,
  },
};

writeFileSync("src/data/test-report.json", JSON.stringify(report, null, 2) + "\n");
console.log(
  `test-report: ${report.tests.passed}/${report.tests.total} tests · ` +
    `statements ${report.coverage.statements}% · branches ${report.coverage.branches}% · ` +
    `functions ${report.coverage.functions}% · lines ${report.coverage.lines}%`
);
