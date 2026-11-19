import { describe, expect, it } from "vitest";
import { badge, buildBadges, countVulnerabilities, coverageColor, scoreColor, summariseLighthouse, summariseUnit } from "../../scripts/badges.mjs";

describe("README badge data (scripts/badges.mjs)", () => {
  it("colours scores like Lighthouse and coverage by threshold", () => {
    expect([scoreColor(1), scoreColor(0.9), scoreColor(0.62), scoreColor(0.3)]).toEqual(["brightgreen", "brightgreen", "orange", "red"]);
    expect([coverageColor(98), coverageColor(85), coverageColor(75), coverageColor(40)]).toEqual(["brightgreen", "green", "yellow", "red"]);
    expect(badge("coverage", 98.1, "brightgreen")).toEqual({ schemaVersion: 1, label: "coverage", message: "98.1", color: "brightgreen" });
  });

  it("summarises Vitest results, coverage and npm audit", () => {
    const unit = summariseUnit({ numPassedTests: 106, numFailedTests: 0 }, { total: { statements: { pct: 98.0912 } } });
    expect(unit).toEqual({ passed: 106, failed: 0, coverage: 98.1 });
    expect(countVulnerabilities({ metadata: { vulnerabilities: { info: 0, low: 1, moderate: 2, high: 0, critical: 0 } } })).toBe(3);
    expect(countVulnerabilities({})).toBe(0);
  });

  it("keeps the lowest Lighthouse score per category across pages", () => {
    const manifest = [
      { summary: { performance: 0.58, accessibility: 1, "best-practices": 1, seo: 1 } },
      { summary: { performance: 0.67, accessibility: 0.96, "best-practices": 1, seo: 1 } },
      {},
    ];
    expect(summariseLighthouse(manifest)).toEqual({ performance: 0.58, accessibility: 0.96, "best-practices": 1, seo: 1 });
  });

  it("builds one badge file per value that is present", () => {
    const files = buildBadges({ passed: 106, failed: 0, coverage: 98.1, vulnerabilities: 0, lighthouse: { performance: 0.58, accessibility: 1, "best-practices": 1, seo: 1 } });
    expect(Object.keys(files).sort()).toEqual([
      "coverage.json",
      "lighthouse-accessibility.json",
      "lighthouse-best-practices.json",
      "lighthouse-performance.json",
      "lighthouse-seo.json",
      "tests.json",
      "vulnerabilities.json",
    ]);
    expect(files["tests.json"].message).toBe("106 passed");
    expect(files["lighthouse-performance.json"]).toMatchObject({ label: "performance", message: "58", color: "orange" });
    expect(files["lighthouse-best-practices.json"].label).toBe("best practices");
    expect(buildBadges({ passed: 100, failed: 2, vulnerabilities: 3 })["tests.json"]).toMatchObject({ message: "2 failed", color: "red" });
    expect(buildBadges({ vulnerabilities: 3 })["vulnerabilities.json"].color).toBe("red");
    expect(buildBadges({})).toEqual({});
  });
});
