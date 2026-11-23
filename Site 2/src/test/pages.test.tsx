import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Index from "@/pages/Index";
import ProjectDetails from "@/pages/ProjectDetails";
import NotFound from "@/pages/NotFound";
import { TransitionProvider } from "@/components/chrome/Transition";
import { projectData } from "@/data/projects";
import { allProjects } from "@/data/profile";
import { nav, sceneState, ui } from "@/lib/store";
import { flush, renderRoute, resetState } from "./utils";
import { setScrollY } from "./setup";

beforeEach(resetState);

function renderIndex(state?: unknown) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: "/", state }]} initialIndex={0}>
      <TransitionProvider>
        <Routes>
          <Route path="/" element={<Index />} />
        </Routes>
      </TransitionProvider>
    </MemoryRouter>
  );
}

describe("home page", () => {
  it("renders every section in order and feeds their positions to the 3D scene", async () => {
    const { unmount } = renderIndex();
    for (const id of ["top", "about", "domains", "toolbox", "projects", "contact"]) {
      expect(document.getElementById(id)).toBeInTheDocument();
    }
    expect(sceneState.stops.map((s) => s.layout)).toEqual(["village", "shatter", "clusters", "network", "bars", "villageTop", "watch"]);
    expect(sceneState.override).toBeNull();
    await flush();
    // the section label follows whatever crosses the middle of the screen
    expect(ui.get().sectionLabel).not.toBe("INTRO");
    window.dispatchEvent(new Event("resize"));
    unmount();
    expect(sceneState.stops).toEqual([]);
  });

  it("a fresh load or reload opens on the intro, whatever position or target is left over", () => {
    // A reload looks like this: React Router says "POP", the history entry still carries the
    // last link's state, and session storage still holds the last scroll position
    sessionStorage.setItem("ak:homeY", "1234");
    render(
      <MemoryRouter initialEntries={["/x", { pathname: "/", state: { scrollTo: "projects" } }]} initialIndex={1}>
        <TransitionProvider>
          <Routes>
            <Route path="/" element={<Index />} />
          </Routes>
        </TransitionProvider>
      </MemoryRouter>
    );
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: "auto" });
  });

  it("counts the browser's back / forward buttons as moving within the site", () => {
    expect(nav.inApp).toBe(false);
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(nav.inApp).toBe(true);
  });

  it("after moving within the site, opens on the requested section with the specialization that was being viewed", () => {
    nav.inApp = true;
    renderIndex({ scrollTo: "projects", category: "nlp" });
    // #projects is at 0 in jsdom, so the -10 offset gives -10 (not the plain top, 0)
    expect(window.scrollTo).toHaveBeenCalledWith({ top: -10, behavior: "auto" });
    const selected = screen.getAllByRole("tab").find((t) => t.getAttribute("aria-selected") === "true")!;
    expect(selected).toHaveAttribute("id", "tab-nlp");
    expect(sessionStorage.getItem("ak:spec")).toBe("3");
  });

  it("restores the last specialization from this session and remembers scroll position", () => {
    vi.useFakeTimers();
    sessionStorage.setItem("ak:spec", "4");
    renderIndex();
    const selected = screen.getAllByRole("tab").find((t) => t.getAttribute("aria-selected") === "true")!;
    expect(selected).toHaveAttribute("id", "tab-research");
    setScrollY(900);
    window.dispatchEvent(new Event("scroll"));
    act(() => {
      vi.advanceTimersByTime(200);
    });
    expect(sessionStorage.getItem("ak:homeY")).toBe("900");
  });

  it("ignores corrupt or blocked session storage", () => {
    sessionStorage.setItem("ak:spec", "99");
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    renderIndex();
    expect(screen.getAllByRole("tab")[0]).toHaveAttribute("aria-selected", "true");
    spy.mockRestore();
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    renderIndex();
    get.mockRestore();
  });

  it("restores the scroll position on back/forward navigation within the site", () => {
    nav.inApp = true;
    sessionStorage.setItem("ak:homeY", "1234");
    render(
      <MemoryRouter initialEntries={["/x", "/"]} initialIndex={1}>
        <TransitionProvider>
          <Routes>
            <Route path="/" element={<Index />} />
          </Routes>
        </TransitionProvider>
      </MemoryRouter>
    );
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 1234, behavior: "auto" });
  });
});

describe("project page", () => {
  it("shows the project with its specs, network cover, gallery and pager", () => {
    const project = projectData.computerVision.projects[0];
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/computerVision/0" });
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(project.title);
    expect(document.title).toBe(`${project.title} — Alexi Kehayias`);
    expect(screen.queryByRole("link", { name: /github/i })).toBeNull();
    expect(screen.getByText("Technologies Used")).toBeInTheDocument();
    // The cover is the project drawn as a neural network, not a picture
    expect(screen.getByRole("img", { name: /neural network diagram/i })).toBeInTheDocument();
    expect(screen.queryByAltText(`${project.title} — cover`)).toBeNull();
    expect(document.querySelector(".detail__actions")).toBeNull();
    expect(sceneState.override).toBe("cube");
    expect(ui.get().sectionLabel).toBe("Computer Vision");
    expect(within(screen.getByRole("list", { name: "Domains" })).getByText("Data Science & AI")).toBeInTheDocument();
    expect(document.querySelectorAll(".gallery .pximg--contain")).toHaveLength(project.images!.length);

    const images = project.images!;
    fireEvent.click(screen.getByRole("button", { name: `Open image 2 of ${images.length}` }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(`02 / 0${images.length}`)).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(within(screen.getByRole("dialog")).getByText(`03 / 0${images.length}`)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close image viewer" }));
    expect(screen.queryByRole("dialog")).toBeNull();

    const idx = allProjects.findIndex((r) => r.category === "computerVision" && r.index === 0);
    expect(screen.getByText(allProjects[idx + 1].project.title, { selector: ".sr-only" })).toBeInTheDocument();
  });

  it("keeps the highlights in a drop-down that starts closed", () => {
    const project = projectData.nlp.projects[1];
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/nlp/1" });
    const details = document.querySelector("details.highlights") as HTMLDetailsElement;
    expect(details).toBeInTheDocument();
    expect(details.open).toBe(false);
    const summary = within(details).getByText("Highlights");
    expect(details.querySelector("summary")).toHaveTextContent(`Highlights 0${project.highlights!.length}`);
    fireEvent.click(summary);
    expect(details.open).toBe(true);
    for (const h of project.highlights!) expect(within(details).getByText(h)).toBeInTheDocument();
  });

  it("embeds the Power BI dashboard for analytics projects", () => {
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/dataAnalytics/1" });
    expect(screen.getByTitle(projectData.dataAnalytics.projects[1].title)).toHaveAttribute("src", projectData.dataAnalytics.projects[1].powerBiUrl);
    expect(screen.getByRole("link", { name: /open dashboard/i })).toBeInTheDocument();
  });

  it("goes back to the work section from the back link", async () => {
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/mlops/0" });
    fireEvent.click(screen.getByRole("link", { name: /back to portfolio/i }));
    expect(await screen.findByText("elsewhere")).toBeInTheDocument();
    expect(screen.getByTestId("location").textContent).toBe("/");
  });

  it("handles unknown projects gracefully", () => {
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/nope/7" });
    expect(screen.getByText("Project not found", { selector: ".sr-only" })).toBeInTheDocument();
    renderRoute(<ProjectDetails />, { path: "/project/:category/:projectIndex", route: "/project/nlp/abc" });
    expect(screen.getAllByText("Project not found", { selector: ".sr-only" }).length).toBe(2);
  });
});

describe("404", () => {
  it("sends the squad underground, calls it a failed mission and offers a way home", () => {
    const { unmount } = renderRoute(<NotFound />, { path: "*", route: "/missing/page" });
    expect(screen.getByText("Mission failed", { selector: ".sr-only" })).toBeInTheDocument();
    expect(document.querySelector(".notfound__path")?.textContent).toBe("/missing/page");
    expect(sceneState.override).toBe("lost");
    expect(screen.getByRole("link", { name: /return to home/i })).toHaveAttribute("href", "#/");
    unmount();
    expect(sceneState.override).toBeNull();
  });
});

describe("App", () => {
  it("boots the whole site without a canvas", async () => {
    const { default: App } = await import("@/App");
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    render(<App />);
    expect(log.mock.calls[0][0]).toMatch(/Alexi Kehayias/);
    expect(JSON.stringify(log.mock.calls)).not.toMatch(/@gmail/);
    expect(screen.getByRole("status", { name: "Loading portfolio" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Skip to content" })).toBeInTheDocument();
    expect(sceneState.canvas).toBe(false);
    expect(document.documentElement).toHaveClass("no-canvas");
    fireEvent(window, new PointerEvent("pointermove", { clientX: 512, clientY: 384, pointerType: "mouse" }));
    expect(sceneState.pointer.x).toBeCloseTo(0, 1);
    const before = sceneState.burst;
    fireEvent(window, new PointerEvent("pointerdown", { clientX: 0, clientY: 0, pointerType: "mouse" }));
    expect(sceneState.burst).toBe(before + 1);
    fireEvent(window, new PointerEvent("pointerdown", { pointerType: "touch" }));
    expect(sceneState.burst).toBe(before + 1);
    // Clicking a control is for using the site: nothing blows up
    fireEvent(screen.getByRole("link", { name: "Skip to content" }), new PointerEvent("pointerdown", { pointerType: "mouse", bubbles: true }));
    expect(sceneState.burst).toBe(before + 1);
    fireEvent.click(screen.getByRole("link", { name: "Skip to content" }));
    expect(document.activeElement).toHaveAttribute("id", "main");
  });
});
