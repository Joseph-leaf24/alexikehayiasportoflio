import { beforeEach, describe, expect, it } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Counter from "@/components/fx/Counter";
import Corners from "@/components/fx/Corners";
import SectionHead from "@/components/fx/SectionHead";
import { advance, flush, markReady, resetState, useAnimationTimers } from "./utils";
import { io, media } from "./setup";

beforeEach(resetState);

const chars = (root: HTMLElement) => Array.from(root.querySelectorAll<HTMLElement>("[data-c]"));

describe("ScrambleText", () => {
  it("keeps the real text for screen readers and splits glyphs per character", () => {
    const { container } = render(<ScrambleText text="Hi there" as="h2" trigger="none" />);
    expect(screen.getByText("Hi there")).toHaveClass("sr-only");
    expect(container.querySelector("h2")).toHaveClass("scramble");
    expect(chars(container)).toHaveLength(7);
    expect(container.querySelectorAll(".sc-word")).toHaveLength(2);
  });

  it("hides characters until the loader finishes, then decodes them", () => {
    useAnimationTimers();
    const { container } = render(<ScrambleText text="Alexi" trigger="mount" duration={400} />);
    expect(chars(container).every((c) => c.hasAttribute("data-h"))).toBe(true);
    markReady();
    advance(150);
    expect(chars(container).some((c) => c.hasAttribute("data-g"))).toBe(true);
    advance(800);
    expect(chars(container).some((c) => c.hasAttribute("data-g") || c.hasAttribute("data-h"))).toBe(false);
  });

  it("decodes when scrolled into view and re-scrambles on hover", async () => {
    io.autoIntersect = false;
    useAnimationTimers();
    const { container } = render(
      <a href="#x">
        <ScrambleText text="Work" hover />
      </a>
    );
    markReady();
    const observer = io.instances.at(-1)!;
    act(() => observer.fire(false));
    expect(chars(container)[0]).toHaveAttribute("data-h");
    act(() => observer.fire(true));
    advance(2000);
    expect(chars(container)[0]).not.toHaveAttribute("data-h");

    fireEvent.mouseEnter(container.querySelector("a")!);
    advance(30);
    expect(chars(container).some((c) => c.hasAttribute("data-g"))).toBe(true);
    advance(1000);
    fireEvent.focus(container.querySelector("a")!);
    advance(1000);
    expect(chars(container).some((c) => c.hasAttribute("data-g"))).toBe(false);
    await flush();
  });

  it("does nothing fancy when reduced motion is preferred", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    const { container, unmount } = render(<ScrambleText text="Calm" trigger="mount" hover />);
    expect(chars(container).some((c) => c.hasAttribute("data-h"))).toBe(false);
    unmount();
  });
});

describe("Reveal", () => {
  it("adds is-in once in view, after the loader", async () => {
    const { container } = render(
      <div>
        <Reveal className="box" delay={120} data-extra="1">
          hello
        </Reveal>
        <Reveal variant="clip">clipped</Reveal>
      </div>
    );
    const box = container.querySelector(".box") as HTMLElement;
    expect(box).toHaveClass("reveal", "reveal-up");
    expect(box.style.getPropertyValue("--d")).toBe("120ms");
    expect(box).toHaveAttribute("data-extra", "1");
    expect(box).not.toHaveClass("is-in");
    markReady();
    await flush();
    expect(box).toHaveClass("is-in");
    expect(screen.getByText("clipped")).toHaveClass("reveal-clip", "is-in");
  });
});

describe("Counter", () => {
  it("counts up to its value once visible, while screen readers get the value itself", async () => {
    useAnimationTimers();
    const { container } = render(<Counter value={18} className="n" />);
    const el = container.querySelector(".n [aria-hidden]")!;
    expect(el.textContent).toBe("00");
    expect(container.querySelector(".n .sr-only")).toHaveTextContent("18");
    markReady();
    await flush();
    advance(2000);
    expect(el.textContent).toBe("18");
  });

  it("shows the final value straight away with reduced motion", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    const { container } = render(<Counter value={7} />);
    expect(container.querySelector("[aria-hidden]")?.textContent).toBe("07");
  });
});

describe("decorations", () => {
  it("renders four corner brackets", () => {
    const { container } = render(<Corners className="x" />);
    expect(container.querySelectorAll(".corners.x i")).toHaveLength(4);
  });

  it("renders a section header with a ghost index", () => {
    const { container, unmount } = render(
      <section>
        <SectionHead index="03" label="Selected Work" />
      </section>
    );
    expect(screen.getByText("[ 03 ]")).toBeInTheDocument();
    expect(container.querySelector(".ghost-num")?.textContent).toBe("03");
    expect(screen.getByText("Selected Work")).toHaveClass("sr-only");
    unmount();
  });

  it("still renders the ghost index with reduced motion", () => {
    media["(prefers-reduced-motion: reduce)"] = true;
    const { container } = render(<SectionHead index="01" label="Who" />);
    expect(container.querySelector(".ghost-num")).toBeInTheDocument();
  });
});
