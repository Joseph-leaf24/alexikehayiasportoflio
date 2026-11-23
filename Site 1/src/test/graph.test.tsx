import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import ProjectGraph from "@/components/fx/ProjectGraph";
import { layoutGraph, wrapLabel } from "@/components/fx/graphLayout";
import { allProjects } from "@/data/profile";
import { projectData } from "@/data/projects";
import type { Project } from "@/types/project";
import { flush, resetState } from "./utils";
import { io } from "./setup";

beforeEach(resetState);

const core = projectData.nlp.projects[1]; // 7 tools, 4 tags, 6 concepts
const layer = (g: ReturnType<typeof layoutGraph>, l: number) => g.nodes.filter((n) => n.layer === l);

describe("project network layout", () => {
  it("maps tools → technologies → hidden neurons → concepts on wide covers", () => {
    const g = layoutGraph(core, 1200, 514);
    expect(layer(g, 0).map((n) => n.full)).toEqual(core.tools);
    expect(layer(g, 1).map((n) => n.full)).toEqual(core.tags);
    expect(layer(g, 2)).toHaveLength(5);
    expect(layer(g, 3).map((n) => n.full)).toEqual(core.concepts);
    const [i, h, o] = [core.tools!.length, core.tags.length, core.concepts.length];
    expect(g.edges).toHaveLength(i * h + h * 5 + 5 * o);
    expect(g.counts).toEqual({ inputs: i, hidden: h, outputs: o });
    // Columns run left → right and every node sits inside the frame
    for (const n of g.nodes) {
      expect(n.x).toBeGreaterThan(0);
      expect(n.x).toBeLessThan(1200);
      expect(n.y).toBeGreaterThan(0);
      expect(n.y).toBeLessThan(514);
    }
    expect(layer(g, 0)[0].x).toBeLessThan(layer(g, 1)[0].x);
    expect(layer(g, 2)[0].x).toBeLessThan(layer(g, 3)[0].x);
  });

  it("drops the anonymous layer and shortens labels on narrow screens", () => {
    const g = layoutGraph(core, 360, 270);
    expect(layer(g, 2)).toHaveLength(0);
    const [i, h, o] = [core.tools!.length, core.tags.length, core.concepts.length];
    expect(g.edges).toHaveLength(i * h + h * o);
    // Signals still arrive exactly when the next column fires
    expect(layer(g, 3)[0].step).toBe(2);
    expect(Math.max(...g.edges.map((e) => e.step))).toBe(1);
    // Long names wrap onto two lines instead of being cut
    const pill = layer(g, 1).find((n) => n.full === "Email Processing")!;
    expect(pill.lines).toEqual(["Email", "Processing"]);
    expect(pill.h).toBeGreaterThan(22);
    for (const n of g.nodes.filter((n) => n.lines)) expect(n.lines!.length).toBeLessThanOrEqual(2);
    expect(g.heads.text).toEqual(["TOOLS", "STACK", "CONCEPTS"]);
  });

  it("fires a stable, per-project subset of connections", () => {
    const a = layoutGraph(core, 1200, 514);
    const b = layoutGraph(core, 1200, 514);
    const pulses = a.edges.filter((e) => e.pulse);
    expect(pulses.length).toBeGreaterThan(0);
    expect(pulses.length).toBeLessThan(a.edges.length);
    expect(b.edges.filter((e) => e.pulse).map((e) => e.key)).toEqual(pulses.map((e) => e.key));
    const other = layoutGraph(allProjects[0].project, 1200, 514);
    expect(other.edges.filter((e) => e.pulse).map((e) => e.key)).not.toEqual(pulses.map((e) => e.key));
  });

  it("wraps labels at spaces and slashes, cutting only what cannot fit", () => {
    expect(wrapLabel("Python", 10, 2)).toEqual(["Python"]);
    expect(wrapLabel("TensorFlow/Keras", 12, 2)).toEqual(["TensorFlow/", "Keras"]);
    expect(wrapLabel("Retrieval Augmented Generation", 12, 2)).toEqual(["Retrieval", "Augmented G…"]);
    expect(wrapLabel("Retrieval Augmented Generation", 40, 1)).toEqual(["Retrieval Augmented Generation"]);
    expect(wrapLabel("Retrieval Augmented Generation", 14, 1)).toEqual(["Retrieval Aug…"]);
    expect(wrapLabel("Supercalifragilistic", 8, 2)).toEqual(["Superca…"]);
  });

  it("uses the tags as inputs when a project lists no tools", () => {
    const p: Project = { ...core, tools: undefined };
    const g = layoutGraph(p, 1200, 514);
    expect(layer(g, 0).map((n) => n.full)).toEqual(core.tags);
  });

  it("lays out every project without overflowing", () => {
    for (const { project } of allProjects) {
      const g = layoutGraph(project, 1200, 514);
      expect(g.nodes.length).toBeGreaterThan(6);
      for (const e of g.edges) expect(e.d).toMatch(/^M[\d.]+ [\d.]+C/);
    }
  });
});

describe("ProjectGraph", () => {
  it("describes the network for screen readers", () => {
    render(<ProjectGraph project={core} />);
    expect(screen.getByRole("img", { name: /7 tools feed 4 technologies, producing 6 concepts/ })).toBeInTheDocument();
    // LangGraph is both a tool (input) and a technology (hidden pill)
    expect(screen.getAllByText("LangGraph", { selector: "tspan" })).toHaveLength(2);
    expect(screen.getByText("Agentic Pipelines", { selector: "tspan" })).toBeInTheDocument();
  });

  it("only animates while on screen", async () => {
    const { container } = render(<ProjectGraph project={core} />);
    const root = container.querySelector(".pgraph") as HTMLElement;
    await flush();
    expect(root).toHaveAttribute("data-live");
    const observer = io.instances.find((o) => o.targets.includes(root))!;
    act(() => observer.fire(false, root));
    expect(root).not.toHaveAttribute("data-live");
  });

  it("lights up a node's connections on hover", () => {
    const { container } = render(<ProjectGraph project={core} />);
    const root = container.querySelector(".pgraph") as HTMLElement;
    const node = container.querySelector(".pgraph__node--l1") as SVGGElement;
    fireEvent.pointerEnter(node);
    expect(root).toHaveAttribute("data-hot");
    expect(node).toHaveClass("is-hot");
    // Fully connected: every input and every next-layer neuron is wired to it
    const hotEdges = container.querySelectorAll(".pgraph__edges path.is-hot");
    expect(hotEdges).toHaveLength(core.tools!.length + 5);
    expect(container.querySelectorAll(".pgraph__node.is-hot")).toHaveLength(1 + core.tools!.length + 5);
    fireEvent.pointerLeave(node);
    expect(root).not.toHaveAttribute("data-hot");
  });

  it("fits a narrow frame, with full names as tooltips", () => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ width: 360, height: 270, top: 0, left: 0 } as DOMRect);
    const { container } = render(<ProjectGraph project={core} />);
    expect(container.querySelector("svg")).toHaveAttribute("viewBox", "0 0 360 270");
    expect(container.querySelectorAll(".pgraph__node--l2")).toHaveLength(0);
    const titles = Array.from(container.querySelectorAll("title")).map((t) => t.textContent);
    expect(titles).toContain("Retrieval Augmented Generation");
  });
});
