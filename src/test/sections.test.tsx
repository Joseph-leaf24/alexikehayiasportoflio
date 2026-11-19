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
import QualityReport, { type TestReport } from "@/components/QualityReport";
import { projectData } from "@/data/projects";
import { profile, specializations } from "@/data/profile";
import { sceneState } from "@/lib/store";
import { advance, flush, markReady, renderRoute, resetState, useAnimationTimers } from "./utils";
import { io, media } from "./setup";

beforeEach(resetState);

describe("Hero", () => {
  it("shows the name and the calls to action", () => {
    const projects = document.createElement("section");
    projects.id = "projects";
    document.body.appendChild(projects);
    renderRoute(<Hero />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Alexi Kehayias");
    expect(screen.getByRole("link", { name: /download cv/i })).toHaveAttribute("href", profile.cv);
    fireEvent.click(screen.getByRole("button", { name: /view projects/i }));
    expect(window.scrollTo).toHaveBeenCalled();
    projects.remove();
  });

  it("click-and-hold shatters the village; releasing restores it", () => {
    const { container } = renderRoute(<Hero />);
    const hero = container.querySelector(".hero")!;
    fireEvent(hero, new PointerEvent("pointerdown", { pointerType: "mouse", clientX: 100, clientY: 100, bubbles: true }));
    expect(sceneState.press).toBe(true);
    expect(hero).toHaveClass("is-pressing");
    fireEvent(window, new PointerEvent("pointerup"));
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

  it("shows a touch hint on phones", () => {
    media["(pointer: coarse)"] = true;
    renderRoute(<Hero />);
    expect(screen.getByText("Touch & hold")).toBeInTheDocument();
  });
});

describe("About", () => {
  it("shows the statement, intro, stats and links — but no email", () => {
    renderRoute(<About />);
    expect(screen.getByText(profile.intro)).toBeInTheDocument();
    expect(screen.getByAltText(`Portrait of ${profile.name}`)).toBeInTheDocument();
    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /linkedin/i })).toHaveAttribute("href", profile.linkedin);
    expect(screen.queryByRole("link", { name: /email/i })).toBeNull();
  });
});

describe("Toolbox", () => {
  it("lists every tool and skews the marquee with scroll speed", async () => {
    useAnimationTimers();
    const { container } = renderRoute(<Skills />);
    await flush();
    expect(within(screen.getByRole("list", { name: "Skills" })).getAllByRole("listitem")).toHaveLength(18);
    expect(container.querySelector('[data-scene="network"]')).toBeInTheDocument();
    sceneState.velocity = 2600;
    advance(200);
    const marquee = container.querySelector(".marquee") as HTMLElement;
    expect(parseFloat(marquee.style.getPropertyValue("--skew"))).toBeGreaterThan(0);
    act(() => io.instances.find((o) => o.targets.includes(marquee))!.fire(false));
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
    renderRoute(<ProjectCard project={project} category="dataAnalytics" index={0} order={0} />);
    const card = screen.getByRole("link", { name: `${project.title} — view project` });
    expect(card).toHaveAttribute("href", "#/project/dataAnalytics/0");
    expect(screen.getByText("Live dashboard")).toBeInTheDocument();
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
    renderRoute(<ProjectCard project={project} category="nlp" index={0} order={1} />);
    expect(screen.queryByText("Tools & Languages")).toBeNull();
  });
});

describe("Contact", () => {
  it("offers LinkedIn, GitHub and the CV — no email anywhere", () => {
    const { container } = renderRoute(<Contact />);
    expect(screen.getByRole("link", { name: /linkedin/i })).toHaveAttribute("href", profile.linkedin);
    expect(screen.getByRole("link", { name: /github/i })).toHaveAttribute("href", profile.github);
    expect(screen.getByRole("link", { name: /download cv/i })).toHaveAttribute("download");
    expect(container.innerHTML).not.toMatch(/mailto:|@gmail/);
    fireEvent.click(screen.getByRole("button", { name: /back to top/i }));
    expect(window.scrollTo).toHaveBeenCalled();
    expect(screen.getByRole("region", { name: "Automated test results" })).toBeInTheDocument();
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

describe("QualityReport", () => {
  const passing: TestReport = {
    generatedAt: "2026-11-19T10:00:00.000Z",
    threshold: 80,
    tests: { total: 120, passed: 120, failed: 0, files: 10 },
    coverage: { statements: 91.2, branches: 84.5, functions: 88.8, lines: 92.1 },
  };

  it("shows passing tests and coverage meters", () => {
    render(<QualityReport data={passing} />);
    expect(screen.getByText("All checks passing")).toBeInTheDocument();
    expect(screen.getByText("120/120")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Branches coverage" })).toHaveAttribute("aria-valuenow", "84.5");
    expect(screen.getByText(/verified 19 Nov 2026/)).toBeInTheDocument();
  });

  it("flags failures and low coverage", () => {
    render(<QualityReport data={{ ...passing, generatedAt: "nope", tests: { ...passing.tests, failed: 2 }, coverage: { ...passing.coverage, branches: 61 } }} />);
    expect(screen.getByText("Checks failing")).toBeInTheDocument();
    expect(screen.getByText("61.0%")).toHaveClass("is-low");
    expect(screen.getByText(/verified —/)).toBeInTheDocument();
  });

  it("renders the generated report by default", () => {
    markReady();
    render(<QualityReport />);
    expect(screen.getByRole("region", { name: "Automated test results" })).toBeInTheDocument();
  });
});
