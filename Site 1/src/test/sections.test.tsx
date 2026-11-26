import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Skills from "@/components/Skills";
import Work from "@/components/Work";
import Contact from "@/components/Contact";
import Lightbox from "@/components/Lightbox";
import ProjectCard from "@/components/ProjectCard";
import { projectData } from "@/data/projects";
import { profile, specializations } from "@/data/profile";
import { sceneState } from "@/lib/store";
import { gsap } from "@/lib/smoothScroll";
import { advance, flush, renderRoute, resetState, useAnimationTimers } from "./utils";
import { io, media } from "./setup";

beforeEach(resetState);

describe("Hero", () => {
  it("lets its copy drift and fade on scroll, but not with reduced motion", () => {
    const to = vi.spyOn(gsap, "to");
    const drifts = () => to.mock.calls.some(([target]) => target === ".hero__inner");
    const { unmount } = renderRoute(<Hero />);
    expect(drifts()).toBe(true);
    unmount();
    to.mockClear();
    media["(prefers-reduced-motion: reduce)"] = true;
    renderRoute(<Hero />);
    expect(drifts()).toBe(false);
    to.mockRestore();
  });

  it("shows the name, focus areas and the call to action", () => {
    const projects = document.createElement("section");
    projects.id = "projects";
    document.body.appendChild(projects);
    renderRoute(<Hero />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Alexi Kehayias");
    // The CV is offered once, in Contact
    expect(screen.queryByRole("link", { name: /download cv/i })).toBeNull();
    expect(screen.getByText("Data Engineering")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /view projects/i }));
    expect(window.scrollTo).toHaveBeenCalled();
    projects.remove();
  });

  it("click-and-hold shatters the village; releasing restores it", () => {
    const { container } = renderRoute(<Hero />);
    const hero = container.querySelector(".hero")!;
    fireEvent(hero, new PointerEvent("pointerdown", { pointerType: "mouse", clientX: 100, clientY: 100, bubbles: true }));
    expect(sceneState.press).toBe(true);
    fireEvent(window, new PointerEvent("pointerup"));
    expect(sceneState.press).toBe(false);

    // A right-click doesn't start a hold, and a context menu ends one
    fireEvent(hero, new PointerEvent("pointerdown", { pointerType: "mouse", button: 2, bubbles: true }));
    expect(sceneState.press).toBe(false);
    fireEvent(hero, new PointerEvent("pointerdown", { pointerType: "mouse", bubbles: true }));
    fireEvent(window, new MouseEvent("contextmenu"));
    expect(sceneState.press).toBe(false);

    // dragging a finger cancels (so scrolling on phones doesn't shatter)
    fireEvent(hero, new PointerEvent("pointerdown", { pointerType: "touch", clientX: 0, clientY: 0, bubbles: true }));
    fireEvent(window, new PointerEvent("pointermove", { pointerType: "touch", clientX: 0, clientY: 40 }));
    expect(sceneState.press).toBe(false);

    // pressing a button does not shatter
    fireEvent(screen.getByRole("button", { name: /view projects/i }), new PointerEvent("pointerdown", { bubbles: true }));
    expect(sceneState.press).toBe(false);
  });

  it("nudges the letters of the name away from the cursor on capable devices", () => {
    window.history.replaceState({}, "", "/?quality=high");
    useAnimationTimers();
    const { container } = renderRoute(<Hero />);
    const ch = container.querySelector(".hero__title .sc-ch") as HTMLElement;
    Object.defineProperty(ch, "offsetParent", { configurable: true, value: container.querySelector(".hero__inner") });
    fireEvent(window, new PointerEvent("pointermove", { pointerType: "mouse", clientX: 5, clientY: 5 }));
    advance(20);
    expect(ch.style.transform).toMatch(/translate3d/);
    fireEvent(window, new PointerEvent("pointermove", { pointerType: "touch", clientX: 900, clientY: 900 }));
    fireEvent.mouseLeave(document.documentElement);
    advance(20);
    expect(ch.style.transform).toBe("");
  });

  it("keeps the top plain: no role, year or shatter hint (holding still works, see above)", () => {
    media["(pointer: coarse)"] = true;
    const { container } = renderRoute(<Hero />);
    expect(container.querySelector(".hero__top")?.textContent).toBe("[ 00 ] — Portfolio");
    expect(container.textContent).not.toMatch(/student|hold|\b20\d\d\b/i);
  });
});

describe("About", () => {
  it("shows the intro, portrait, stats, focus areas and LinkedIn — but no email or CV", () => {
    renderRoute(<About />);
    expect(screen.getByText(profile.intro)).toBeInTheDocument();
    expect(screen.getByAltText(`Portrait of ${profile.name}`)).toBeInTheDocument();
    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /linkedin/i })).toHaveAttribute("href", profile.linkedin);
    expect(screen.queryByRole("link", { name: /email/i })).toBeNull();
    expect(screen.queryByRole("link", { name: /download cv/i })).toBeNull();
    expect(screen.getAllByText("Data Analytics").length).toBeGreaterThan(0);
  });
});

describe("Toolbox", () => {
  it("lists every tool, skews the marquee with scroll speed and runs it only while on screen", async () => {
    useAnimationTimers();
    const { container } = renderRoute(<Skills />);
    await flush();
    expect(within(screen.getByRole("list", { name: "Skills" })).getAllByRole("listitem")).toHaveLength(18);
    expect(container.querySelector('[data-scene="network"]')).toBeInTheDocument();
    sceneState.velocity = 2600;
    advance(200);
    const marquee = container.querySelector(".marquee") as HTMLElement;
    expect(marquee).toHaveAttribute("data-live");
    expect(parseFloat(marquee.style.getPropertyValue("--skew"))).toBeGreaterThan(0);
    act(() => io.instances.find((o) => o.targets.includes(marquee))!.fire(false));
    expect(marquee).not.toHaveAttribute("data-live");
  });
});

describe("Work", () => {
  function Harness() {
    const [active, setActive] = useState(0);
    return <Work active={active} setActive={setActive} />;
  }

  it("switches specialization with clicks and arrow keys", () => {
    renderRoute(<Harness />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(6);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent(projectData.dataAnalytics.title);

    fireEvent.click(tabs[1]);
    expect(screen.getAllByRole("tab")[1]).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent(projectData.computerVision.projects[0].title);

    const list = screen.getByRole("tablist");
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(screen.getAllByRole("tab")[2]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "End" });
    expect(screen.getAllByRole("tab")[5]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowDown" });
    expect(screen.getAllByRole("tab")[0]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowLeft" });
    expect(screen.getAllByRole("tab")[5]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "Home" });
    fireEvent.keyDown(list, { key: "ArrowUp" });
    expect(screen.getAllByRole("tab")[5]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "a" });
    fireEvent.click(screen.getAllByRole("tab")[5]); // already active: no-op
    expect(screen.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", `tab-${specializations[5].id}`);
  });

  it("brings the project list into view when it is off-screen", () => {
    const rafs: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      rafs.push(cb);
      return 1;
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top: 5000 } as DOMRect);
    renderRoute(<Harness />);
    fireEvent.click(screen.getAllByRole("tab")[3]);
    rafs.forEach((cb) => cb(0));
    expect(window.scrollTo).toHaveBeenCalled();
  });
});

describe("ProjectCard", () => {
  it("links to the project and tilts toward the pointer", () => {
    const project = projectData.dataAnalytics.projects[0];
    renderRoute(<ProjectCard project={project} category="dataAnalytics" index={0} />);
    // The title is the link (stretched over the card by CSS); the card is the tilting article
    const link = screen.getByRole("link", { name: project.title });
    expect(link).toHaveAttribute("href", "#/project/dataAnalytics/0");
    const card = link.closest(".project-card") as HTMLElement;
    expect(card.tagName).toBe("ARTICLE");
    // No "Live dashboard" badge on the cards (the dashboards are on the project pages)
    expect(screen.queryByText(/live dashboard/i)).toBeNull();
    vi.spyOn(card, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 200, height: 100 } as DOMRect);
    fireEvent(card, new PointerEvent("pointermove", { pointerType: "mouse", clientX: 150, clientY: 25, bubbles: true }));
    expect(card.style.getPropertyValue("--ry")).toBe("2.25deg");
    fireEvent(card, new PointerEvent("pointermove", { pointerType: "touch", clientX: 0, clientY: 0, bubbles: true }));
    expect(card.style.getPropertyValue("--ry")).toBe("2.25deg");
    fireEvent.pointerLeave(card);
    expect(card.style.getPropertyValue("--rx")).toBe("0deg");
    fireEvent.pointerEnter(card);
    expect(sceneState.pulse).toBeGreaterThan(0);
  });

  it("omits tools when a project has none", () => {
    const project = { ...projectData.nlp.projects[0], tools: undefined, powerBiUrl: undefined };
    renderRoute(<ProjectCard project={project} category="nlp" index={0} />);
    expect(screen.queryByText("Tools & Languages")).toBeNull();
  });
});

describe("Contact", () => {
  it("offers LinkedIn and the CV — no GitHub, email or test report", () => {
    const { container } = renderRoute(<Contact />);
    expect(screen.getByRole("link", { name: /linkedin/i })).toHaveAttribute("href", profile.linkedin);
    expect(screen.getByRole("link", { name: /download cv/i })).toHaveAttribute("download");
    expect(screen.queryByRole("link", { name: /github/i })).toBeNull();
    expect(container.innerHTML).not.toMatch(/mailto:|@gmail|github\.com/);
    expect(screen.queryByRole("region", { name: "Automated test results" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /back to top/i }));
    expect(window.scrollTo).toHaveBeenCalled();
  });
});

describe("Lightbox", () => {
  it("pages through images with buttons and keys, and closes", () => {
    const onClose = vi.fn();
    const onIndex = vi.fn();
    render(<Lightbox images={["a.png", "b.png", "c.png"]} index={0} title="Demo" onClose={onClose} onIndex={onIndex} />);
    const dialog = screen.getByRole("dialog", { name: "Demo — image viewer" });
    expect(within(dialog).getByText("01 / 03")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close image viewer" })).toHaveFocus();
    fireEvent.click(screen.getByRole("button", { name: "Next image" }));
    expect(onIndex).toHaveBeenLastCalledWith(1);
    fireEvent.click(screen.getByRole("button", { name: "Previous image" }));
    expect(onIndex).toHaveBeenLastCalledWith(2);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(onIndex).toHaveBeenLastCalledWith(1);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(onIndex).toHaveBeenLastCalledWith(2);
    fireEvent.click(screen.getByAltText("Demo — image 1 of 3"));
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "Escape" });
    fireEvent.click(dialog);
    expect(onClose).toHaveBeenCalledTimes(2);
    expect(document.documentElement).toHaveClass("scroll-locked");
  });

  it("keeps Tab focus inside the dialog", () => {
    render(<Lightbox images={["a.png", "b.png"]} index={0} title="Demo" onClose={() => {}} onIndex={() => {}} />);
    const close = screen.getByRole("button", { name: "Close image viewer" });
    const next = screen.getByRole("button", { name: "Next image" });
    next.focus();
    fireEvent.keyDown(window, { key: "Tab" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(window, { key: "Tab", shiftKey: true });
    expect(next).toHaveFocus();
    // Focus that escaped the dialog is pulled back in
    (document.activeElement as HTMLElement).blur();
    fireEvent.keyDown(window, { key: "Tab" });
    expect(close).toHaveFocus();
    // Moving between buttons inside the dialog is left to the browser
    screen.getByRole("button", { name: "Previous image" }).focus();
    fireEvent.keyDown(window, { key: "Tab" });
    expect(screen.getByRole("button", { name: "Previous image" })).toHaveFocus();
  });

  it("hides paging for a single image", () => {
    const onIndex = vi.fn();
    const { unmount } = render(<Lightbox images={["a.png"]} index={0} title="One" onClose={() => {}} onIndex={onIndex} />);
    expect(screen.queryByRole("button", { name: "Next image" })).toBeNull();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(onIndex).not.toHaveBeenCalled();
    unmount();
    expect(document.documentElement).not.toHaveClass("scroll-locked");
  });
});
