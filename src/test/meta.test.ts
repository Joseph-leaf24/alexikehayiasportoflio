import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/** The page <head> as shipped (index.html), parsed like a browser would. */
const head = new DOMParser().parseFromString(readFileSync(resolve(process.cwd(), "index.html"), "utf8"), "text/html");
const meta = (key: string) =>
  head.querySelector(`meta[property="${key}"], meta[name="${key}"]`)?.getAttribute("content") ?? "";

const SITE = "https://joseph-leaf24.github.io/alexikehayiasportoflio/";

/** [width, height] from a JPEG's start-of-frame segment. */
function jpegSize(buf: Buffer): [number, number] | null {
  for (let i = 2; i + 9 < buf.length; ) {
    if (buf[i] !== 0xff) return null;
    const marker = buf[i + 1];
    // SOF0–SOF15, except DHT (C4), JPG (C8) and DAC (CC)
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return [buf.readUInt16BE(i + 7), buf.readUInt16BE(i + 5)];
    i += 2 + buf.readUInt16BE(i + 2);
  }
  return null;
}

describe("link previews and search profile (index.html)", () => {
  it("describes the site by its three disciplines", () => {
    for (const key of ["description", "og:description", "twitter:description"]) {
      expect(meta(key)).toMatch(/data science, data engineering and data analytics/i);
    }
  });

  it("has a full Open Graph + large-image card", () => {
    expect(meta("og:url")).toBe(SITE);
    expect(head.querySelector('link[rel="canonical"]')?.getAttribute("href")).toBe(SITE);
    expect(meta("og:image")).toBe(`${SITE}og-image.jpg`);
    expect(meta("og:image:type")).toBe("image/jpeg");
    expect(meta("twitter:image")).toBe(meta("og:image"));
    expect([meta("og:image:width"), meta("og:image:height")]).toEqual(["1200", "630"]);
    expect(meta("og:image:alt").length).toBeGreaterThan(10);
    expect(meta("twitter:card")).toBe("summary_large_image");
  });

  it("publishes a Person profile without location or GitHub", () => {
    const block = head.querySelector('script[type="application/ld+json"]')?.textContent ?? "";
    const person = JSON.parse(block);
    expect(person).toMatchObject({ "@type": "Person", name: "Alexi Kehayias", url: SITE });
    expect(person.sameAs).toEqual(["https://www.linkedin.com/in/alexi-kehayias/"]);
    // (the site itself lives on github.io; what must not appear is a GitHub profile link)
    expect(block).not.toMatch(/breda|address|"geo|github\.com/i);
  });

  it("ships the preview image it points to", () => {
    // (a JPEG: the render's film grain makes a PNG of it over 1 MB)
    const jpg = readFileSync(resolve(process.cwd(), "public/og-image.jpg"));
    expect([jpg[0], jpg[1]]).toEqual([0xff, 0xd8]);
    expect(jpegSize(jpg)).toEqual([1200, 630]);
    expect(jpg.length).toBeLessThan(300_000);
  });
});
