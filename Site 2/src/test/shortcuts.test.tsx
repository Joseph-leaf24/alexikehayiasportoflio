import { beforeEach, describe, expect, it } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import Shortcuts from "@/components/chrome/Shortcuts";
import ProjectDetails from "@/pages/ProjectDetails";
import { allProjects, projectPath } from "@/data/profile";
import { ui } from "@/lib/store";
import { hasModifier, isModalOpen, isTypingTarget, trapTab } from "@/lib/keys";
import { renderRoute, resetState } from "./utils";

beforeEach(resetState);

const PROJECT = "/project/:category/:projectIndex";
const location = () => screen.getByTestId("location").textContent;

describe("keyboard helpers", () => {
  it("recognise text fields, modifiers and open dialogs", () => {
    const input = document.createElement("input");
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    Object.defineProperty(editable, "isContentEditable", { value: true });
    expect([isTypingTarget(input), isTypingTarget(editable), isTypingTarget(document.body), isTypingTarget(null)]).toEqual([true, true, false, false]);
    expect(hasModifier(new KeyboardEvent("keydown", { ctrlKey: true }))).toBe(true);
    expect(hasModifier(new KeyboardEvent("keydown", { shiftKey: true }))).toBe(false);
    expect(isModalOpen()).toBe(false);
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    document.body.appendChild(dialog);
    expect(isModalOpen()).toBe(true);
    dialog.remove();
  });

  it("trapTab cycles focus inside a container", () => {
    const root = document.createElement("div");
    root.innerHTML = '<button id="a">a</button><a id="b" href="#x">b</a>';
    document.body.appendChild(root);
    const [a, b] = [root.querySelector<HTMLElement>("#a")!, root.querySelector<HTMLElement>("#b")!];
    b.focus();
    trapTab(new KeyboardEvent("keydown", { key: "Tab", cancelable: true }), root);
    expect(a).toHaveFocus();
    trapTab(new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, cancelable: true }), root);
    expect(b).toHaveFocus();
    trapTab(new KeyboardEvent("keydown", { key: "Enter" }), root);
    expect(b).toHaveFocus();
    trapTab(new KeyboardEvent("keydown", { key: "Tab" }), document.createElement("div"));
    root.remove();
  });
});

describe("shortcuts help (?)", () => {
  // The loader has finished (the help stays out of its way, see below)
  beforeEach(() => ui.set({ ready: true }));

  it("never opens on top of the loader, the image viewer or the phone menu", () => {
    render(<Shortcuts />);
    act(() => ui.set({ ready: false }));
    fireEvent.keyDown(window, { key: "?" });
    act(() => ui.set({ ready: true, menuOpen: true }));
    fireEvent.keyDown(window, { key: "?" });
    act(() => ui.set({ menuOpen: false }));
    const viewer = document.createElement("div");
    viewer.setAttribute("role", "dialog");
    viewer.setAttribute("aria-modal", "true");
    document.body.appendChild(viewer);
    fireEvent.keyDown(window, { key: "?" });
    viewer.remove();
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
    fireEvent.keyDown(window, { key: "?" });
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();
  });

  it("opens with ?, focuses Close, traps Tab and closes with Esc", () => {
    render(<Shortcuts />);
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.keyDown(window, { key: "?" });
    const dialog = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    const close = screen.getByRole("button", { name: "Close keyboard shortcuts" });
    expect(close).toHaveFocus();
    expect(dialog).toHaveTextContent("Previous / next project");
    expect(document.documentElement).toHaveClass("scroll-locked");
    fireEvent.keyDown(window, { key: "Tab" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.documentElement).not.toHaveClass("scroll-locked");
  });

  it("toggles with ?, closes from the backdrop, and ignores modifiers and text fields", () => {
    render(
      <>
        <input aria-label="search" />
        <Shortcuts />
      </>
    );
    fireEvent.keyDown(window, { key: "?", ctrlKey: true });
    fireEvent.keyDown(screen.getByLabelText("search"), { key: "?" });
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.keyDown(window, { key: "?" });
    fireEvent.keyDown(window, { key: "?" });
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.keyDown(window, { key: "?" });
    fireEvent.click(screen.getByText(/Previous \/ next image/));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("dialog"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("← / → on project pages", () => {
  const at = allProjects.findIndex((r) => r.category === "nlp" && r.index === 1);
  const route = projectPath("nlp", 1);
  const nextPath = projectPath(allProjects[at + 1].category, allProjects[at + 1].index);
  const prevPath = projectPath(allProjects[at - 1].category, allProjects[at - 1].index);

  it("step to the next and previous project", async () => {
    renderRoute(<ProjectDetails />, { path: PROJECT, route });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => expect(location()).toBe(nextPath));
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    await waitFor(() => expect(location()).toBe(route));
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    await waitFor(() => expect(location()).toBe(prevPath));
  });

  it("leave other keys, modifiers, text fields, the image viewer and the menu alone", async () => {
    renderRoute(<ProjectDetails />, { path: PROJECT, route });
    fireEvent.keyDown(window, { key: "ArrowDown" });
    fireEvent.keyDown(window, { key: "ArrowRight", altKey: true });
    fireEvent.keyDown(window, { key: "ArrowRight", shiftKey: true });
    const input = document.createElement("input");
    document.body.appendChild(input);
    fireEvent.keyDown(input, { key: "ArrowRight" });
    input.remove();
    act(() => ui.set({ menuOpen: true }));
    fireEvent.keyDown(window, { key: "ArrowRight" });
    act(() => ui.set({ menuOpen: false }));
    // In the image viewer the arrows page through images instead
    fireEvent.click(screen.getByRole("button", { name: /open image 1 of/i }));
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByRole("dialog")).toHaveTextContent(/02 \//);
    await new Promise((r) => setTimeout(r, 50));
    expect(location()).toBe(route);
  });
});
