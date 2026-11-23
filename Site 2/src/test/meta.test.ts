import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/** The page <head> as shipped (index.html), parsed like a browser would. */
const html = readFileSync(resolve(process.cwd(), "index.html"), "utf8");
const head = new DOMParser().parseFromString(html, "text/html");
const meta = (key: string) => head.querySelector(`meta[name="${key}"]`)?.getAttribute("content") ?? "";

describe("page head (index.html)", () => {
  it("describes the site by its three disciplines", () => {
    expect(meta("description")).toMatch(/data science, data engineering and data analytics/i);
    expect(head.title).toBe("Alexi Kehayias — Data Science & AI Portfolio");
  });

  it("is a local-only copy: no links to the deployed site, no link-preview or search-profile tags", () => {
    expect(html).not.toMatch(/github\.io|alexikehayiasportoflio/);
    expect(head.querySelector('link[rel="canonical"], meta[property^="og:"], script[type="application/ld+json"]')).toBeNull();
  });

  it("loads the three pixel fonts and an inline pixel-art icon", () => {
    const fonts = head.querySelector('link[rel="stylesheet"]')?.getAttribute("href") ?? "";
    for (const family of ["Press+Start+2P", "Silkscreen", "Pixelify+Sans"]) expect(fonts).toContain(family);
    const icon = head.querySelector('link[rel="icon"]')?.getAttribute("href") ?? "";
    expect(icon).toMatch(/^data:image\/svg\+xml,/);
    expect(decodeURIComponent(icon)).toMatch(/shape-rendering='crispEdges'/);
  });

  it("keeps the <meta charset> the build's CSP injection looks for", () => {
    expect(html).toContain('<meta charset="UTF-8" />');
  });
});
