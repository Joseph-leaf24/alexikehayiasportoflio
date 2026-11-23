import { describe, expect, it } from "vitest";
import { projectData } from "@/data/projects";
import { allProjects, profile, projectPath, specializations } from "@/data/profile";
import { SKILLS } from "@/data/skills";
import { resolveBlend } from "@/components/pixel/blend";
import { sceneState } from "@/lib/store";

describe("portfolio content", () => {
  it("has a complete profile without any email address", () => {
    expect(profile.name).toBe("Alexi Kehayias");
    expect(profile.linkedin).toMatch(/^https:\/\/www\.linkedin\.com\//);
    expect(profile.cv).toMatch(/Alexi_Kehayias_CV\.pdf$/);
    expect(JSON.stringify(profile)).not.toMatch(/@[a-z0-9-]+\.[a-z]{2,}/i);
  });

  it("focuses on the three data disciplines", () => {
    expect(profile.focus).toEqual(["Data Science", "Data Engineering", "Data Analytics"]);
  });

  it("maps every specialization to a category with projects", () => {
    expect(specializations).toHaveLength(6);
    for (const s of specializations) {
      expect(projectData[s.id].projects.length).toBeGreaterThan(0);
    }
    expect(allProjects).toHaveLength(9);
    expect(projectPath("nlp", 1)).toBe("/project/nlp/1");
  });

  it("gives every project the fields the cards and detail pages need", () => {
    for (const { project } of allProjects) {
      expect(project.title.length).toBeGreaterThan(3);
      expect(project.description.length).toBeGreaterThan(20);
      expect(project.image).toBeTruthy();
      expect(project.tags.length).toBeGreaterThan(0);
      expect(project.concepts.length).toBeGreaterThan(0);
      // Covers are real project screenshots, never the old stock photos
      expect(project.image).not.toMatch(/(computer-vision|machine-learning|nlp|research)\.jpg$/);
      // No per-project source links
      expect(project).not.toHaveProperty("githubUrl");
      // Highlights are optional, short and non-empty
      for (const h of project.highlights ?? []) {
        expect(h.trim().length).toBeGreaterThan(10);
        expect(h.length).toBeLessThan(120);
      }
    }
  });

  it("lists the toolbox", () => {
    expect(SKILLS).toHaveLength(18);
    expect(new Set(SKILLS.map((s) => s.name)).size).toBe(18);
  });
});

describe("scroll blend", () => {
  it("holds the first zone, crossfades through each transition window and lets a page override win", () => {
    sceneState.override = null;
    sceneState.stops = [
      { y: 0, layout: "village" },
      { y: 1000, layout: "shatter" },
    ];
    // The next section's top moves from 95% to 25% of an 800 px viewport
    expect(resolveBlend(0, 800)).toEqual({ a: "village", b: "village", t: 0 });
    expect(resolveBlend(1000 - 800 * 0.6, 800)).toEqual({ a: "village", b: "shatter", t: 0.5 });
    expect(resolveBlend(900, 800)).toEqual({ a: "shatter", b: "shatter", t: 0 });
    sceneState.override = "lost";
    expect(resolveBlend(0, 800)).toEqual({ a: "lost", b: "lost", t: 0 });
    sceneState.override = null;
    sceneState.stops = [];
    expect(resolveBlend(500, 800).a).toBe("village");
  });
});
