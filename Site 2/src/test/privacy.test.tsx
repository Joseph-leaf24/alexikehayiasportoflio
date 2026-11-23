import { beforeEach, describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import Hud from "@/components/chrome/Hud";
import Loader from "@/components/chrome/Loader";
import About from "@/components/About";
import Contact from "@/components/Contact";
import { profile } from "@/data/profile";
import { renderRoute, resetState } from "./utils";

beforeEach(resetState);

/** No city, country code or map coordinates anywhere the visitor can see. */
const LOCATION = /breda|\bNL\b|°\s*[NSEW]/i;

describe("no personal location on the site", () => {
  it("is absent from the profile data", () => {
    expect(JSON.stringify(profile)).not.toMatch(LOCATION);
    expect(profile).not.toHaveProperty("location");
    expect(profile).not.toHaveProperty("coords");
  });

  it("is absent from the HUD, loader, About and Contact sections", () => {
    for (const ui of [<Hud key="hud" />, <About key="about" />, <Contact key="contact" />]) {
      const { container, unmount } = renderRoute(ui);
      expect(container.textContent).not.toMatch(LOCATION);
      unmount();
    }
    const { container } = render(<Loader />);
    expect(container.textContent).not.toMatch(LOCATION);
    expect(container.textContent).toContain(profile.focus.join(" · "));
  });
});
