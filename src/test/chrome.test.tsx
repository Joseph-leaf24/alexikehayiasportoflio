import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Hud from "@/components/chrome/Hud";
import Cursor from "@/components/chrome/Cursor";
import Loader from "@/components/chrome/Loader";
import { TLink, TransitionProvider } from "@/components/chrome/Transition";
import { sceneState, ui } from "@/lib/store";
import { advance, LocationProbe, renderRoute, resetState, useAnimationTimers } from "./utils";
import { media, setScrollY } from "./setup";

beforeEach(resetState);

describe("HUD", () => {
  it("scrolls to sections on the home page and has no sound or email controls", () => {
    const target = document.createElement("section");
    target.id = "toolbox";
    document.body.appendChild(target);
    renderRoute(<Hud />);
    const nav = screen.getByRole("navigation", { name: "Primary" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual(["About", "Domains", "Toolbox", "Work", "Contact"].map((t) => expect.stringContaining(t)));
    fireEvent.click(within(nav).getAllByRole("link")[2]);
    expect(window.scrollTo).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("link", { name: /home/i }));
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 0, behavior: "smooth" });
    expect(screen.queryByRole("button", { name: /sound/i })).toBeNull();
    expect(document.body.innerHTML).not.toMatch(/mailto:/);
    target.remove();
  });

  it("opens and closes the mobile menu", () => {
    renderRoute(<Hud />);
    const burger = screen.getByRole("button", { name: "Open menu" });
    fireEvent.click(burger);
    expect(ui.get().menuOpen).toBe(true);
    expect(document.getElementById("mobile-menu")).toHaveClass("is-open");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(ui.get().menuOpen).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Open menu" }));
    const mobileNav = screen.getByRole("navigation", { name: "Mobile" });
    fireEvent.click(within(mobileNav).getAllByRole("link")[0]);
    expect(ui.get().menuOpen).toBe(false);
  });

  it("navigates home with a page transition from other pages", async () => {
    renderRoute(<Hud />, { path: "/project/:c/:i", route: "/project/nlp/0" });
    const nav = screen.getByRole("navigation", { name: "Primary" });
    fireEvent.click(within(nav).getAllByRole("link")[2]);
    expect(await screen.findByText("elsewhere")).toBeInTheDocument();
    expect(screen.getByTestId("location").textContent).toBe("/");
  });

  it("shows live telemetry, the clock, the scroll meter and the current section", () => {
    useAnimationTimers();
    sceneState.blocks = 12941;
    sceneState.formation = "network";
    sceneState.cam.x = -1.5;
    renderRoute(<Hud />);
    advance(300);
    expect(screen.getAllByText("Neural net").length).toBeGreaterThan(0);
    expect(screen.getByText("12,941")).toBeInTheDocument();
    expect(document.querySelector(".hud-telemetry")?.textContent).toMatch(/−01\.50/);
    expect(document.querySelector(".hud-bottom__left")?.textContent).toMatch(/\d{2}:\d{2}:\d{2} CES?T/);
    Object.defineProperty(document.documentElement, "scrollHeight", { configurable: true, value: 3000 });
    act(() => {
      window.dispatchEvent(new Event("resize"));
      setScrollY(1116);
      window.dispatchEvent(new Event("scroll"));
    });
    advance(50);
    expect(document.querySelector(".scroll-meter")?.textContent).toMatch(/050%/);
    act(() => ui.set({ sectionLabel: "Toolbox", sectionIndex: "02" }));
    expect(document.querySelector(".hud-section__idx")?.textContent).toBe("02");
  });

  it("hides telemetry when WebGL is unavailable", () => {
    sceneState.webgl = false;
    renderRoute(<Hud />);
    expect(document.querySelector(".hud-telemetry")).toBeNull();
  });
});

describe("Cursor", () => {
  it("stays off on touch devices", () => {
    const { container } = render(<Cursor />);
    expect(container.firstChild).toBeNull();
  });

  it("follows the mouse, labels hover targets and pulls magnetic buttons", () => {
    media["(hover: hover) and (pointer: fine)"] = true;
    useAnimationTimers();
    const { container, unmount } = render(
      <div>
        <Cursor />
        <button className="btn-ice" data-cursor="OPEN">
          Go
        </button>
      </div>
    );
    const cursor = container.querySelector(".cursor")!;
    const btn = screen.getByRole("button", { name: "Go" });
    expect(document.documentElement).toHaveClass("has-cursor");
    fireEvent(window, new PointerEvent("pointermove", { clientX: 10, clientY: 10, pointerType: "touch" }));
    expect(cursor).not.toHaveClass("is-visible");
    fireEvent(btn, new PointerEvent("pointermove", { clientX: 40, clientY: 20, pointerType: "mouse", bubbles: true }));
    expect(cursor).toHaveClass("is-visible");
    expect(btn.style.transform).toMatch(/translate3d/);
    fireEvent.mouseOver(btn);
    expect(cursor).toHaveClass("is-hover", "has-label");
    expect(cursor.querySelector(".cursor__label")?.textContent).toBe("OPEN");
    advance(50);
    fireEvent(window, new PointerEvent("pointerdown", { pointerType: "mouse" }));
    expect(cursor).toHaveClass("is-down");
    fireEvent(window, new PointerEvent("pointerup", { pointerType: "mouse" }));
    fireEvent(window, new PointerEvent("pointermove", { clientX: 400, clientY: 400, pointerType: "mouse" }));
    expect(btn.style.transform).toBe("");
    fireEvent.mouseOver(document.body);
    expect(cursor).not.toHaveClass("has-label");
    fireEvent.mouseLeave(document.documentElement);
    expect(cursor).not.toHaveClass("is-visible");
    unmount();
    expect(document.documentElement).not.toHaveClass("has-cursor");
  });
});

describe("Loader", () => {
  it("counts to 100 once the scene is ready, then lifts and marks the app ready", async () => {
    useAnimationTimers();
    sceneState.blocks = 8000;
    render(<Loader />);
    expect(screen.getByRole("status", { name: "Loading portfolio" })).toBeInTheDocument();
    await act(async () => {
      await Promise.resolve();
    });
    advance(600);
    expect(screen.getByText(/BOOTING RENDERER/)).toBeInTheDocument();
    sceneState.ready = true;
    advance(8000);
    expect(screen.getByText(/8,000 CUBES/)).toBeInTheDocument();
    advance(2000);
    expect(ui.get().ready).toBe(true);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("is quick with reduced motion", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    useAnimationTimers();
    sceneState.ready = true;
    const { unmount } = render(<Loader />);
    advance(9000);
    expect(ui.get().ready).toBe(true);
    unmount();
  });
});

describe("page transitions", () => {
  const app = (
    <MemoryRouter initialEntries={["/"]}>
      <TransitionProvider>
        <Routes>
          <Route
            path="/"
            element={
              <TLink to="/next" onClick={() => {}}>
                go
              </TLink>
            }
          />
          <Route path="/next" element={<p>next page</p>} />
        </Routes>
        <LocationProbe />
      </TransitionProvider>
    </MemoryRouter>
  );

  it("wipes ice sheets over the page and navigates", async () => {
    render(app);
    const link = screen.getByRole("link", { name: "go" });
    expect(link).toHaveAttribute("href", "#/next");
    expect(document.querySelectorAll(".page-wipe__shard").length).toBeGreaterThan(0);
    fireEvent.click(link);
    expect(await screen.findByText("next page")).toBeInTheDocument();
    expect(sceneState.pulse).toBeGreaterThan(0);
  });

  it("keeps to schedule when the browser stalls the wipe's animations (very slow devices)", async () => {
    const animate = Element.prototype.animate;
    // Animations that never finish, as on a device drawing about one frame a second
    Element.prototype.animate = function stalled() {
      return { finished: new Promise(() => {}), cancel: () => {} } as unknown as Animation;
    };
    try {
      render(app);
      fireEvent.click(screen.getByRole("link", { name: "go" }));
      expect(sceneState.transition).toBe(true);
      expect(await screen.findByText("next page", undefined, { timeout: 3000 })).toBeInTheDocument();
      await waitFor(() => expect(sceneState.transition).toBe(false), { timeout: 3000 });
    } finally {
      Element.prototype.animate = animate;
    }
  });

  it("queues a navigation requested while a transition is still running", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <TransitionProvider>
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <TLink to="/a">a</TLink>
                  <TLink to="/b">b</TLink>
                </>
              }
            />
            <Route path="/a" element={<p>page a</p>} />
            <Route path="/b" element={<p>page b</p>} />
          </Routes>
          <LocationProbe />
        </TransitionProvider>
      </MemoryRouter>
    );
    fireEvent.click(screen.getByRole("link", { name: "a" }));
    // Second click lands mid-transition: it must not be dropped
    fireEvent.click(screen.getByRole("link", { name: "b" }));
    expect(await screen.findByText("page b")).toBeInTheDocument();
    expect(screen.getByTestId("location").textContent).toBe("/b");
  });

  it("lets modified clicks through (open in new tab)", () => {
    render(app);
    fireEvent.click(screen.getByRole("link", { name: "go" }), { ctrlKey: true });
    expect(screen.getByTestId("location").textContent).toBe("/");
  });

  it("navigates instantly with reduced motion", async () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    render(app);
    fireEvent.click(screen.getByRole("link", { name: "go" }));
    expect(await screen.findByText("next page")).toBeInTheDocument();
  });

  it("remembers the home scroll position when leaving (and copes with blocked storage)", async () => {
    setScrollY(640);
    render(app);
    fireEvent.click(screen.getByRole("link", { name: "go" }));
    await screen.findByText("next page");
    expect(sessionStorage.getItem("ak:homeY")).toBe("640");
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    render(app);
    fireEvent.click(screen.getAllByRole("link", { name: "go" })[0]);
    spy.mockRestore();
  });
});
