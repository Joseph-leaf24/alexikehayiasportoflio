import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import Domains from "@/components/Domains";
import DepthImage from "@/components/fx/DepthImage";
import ProjectCard from "@/components/ProjectCard";
import { domainIntro, domains, domainsFor, projectsOf } from "@/data/domains";
import { allProjects, projectPath, specializations } from "@/data/profile";
import { projectData } from "@/data/projects";
import { buildLayouts } from "@/components/gl/layouts";
import { sceneState } from "@/lib/store";
import { flush, renderRoute, resetState } from "./utils";
import { io } from "./setup";

beforeEach(resetState);

describe("domain data", () => {
  it("covers the four disciplines of the data lifecycle, in order", () => {
    expect(domains.map((d) => d.id)).toEqual(["engineering", "analytics", "science", "deployment"]);
    expect(domainIntro.title).toBe("Domains");
    for (const d of domains) {
      expect(d.capabilities.length).toBeGreaterThanOrEqual(3);
      expect(d.tools.length).toBeGreaterThanOrEqual(3);
      expect(specializations.some((s) => s.id === d.explore)).toBe(true);
    }
  });

  it("only links to projects that exist, and every project belongs to a domain", () => {
    for (const d of domains) {
      for (const { project } of projectsOf(d)) expect(project?.title).toBeTruthy();
    }
    for (const { category, index } of allProjects) {
      expect(domainsFor(category, index).length).toBeGreaterThan(0);
    }
    expect(domainsFor("nlp", 1).map((d) => d.id)).toEqual(["engineering", "deployment"]);
    expect(domainsFor("nope", 0)).toEqual([]);
  });

  it("gives the clusters formation one anchor per domain", () => {
    const set = buildLayouts();
    expect(set.layouts.clusters.anchors).toHaveLength(domains.length);
    expect(set.layouts.clusters.pos).toHaveLength(set.count);
  });
});

describe("Domains section", () => {
  it("lays out the lifecycle and a card per domain with related projects", () => {
    renderRoute(<Domains onExplore={() => {}} />);
    expect(document.getElementById("domains")).toHaveAttribute("data-scene", "clusters");
    expect(document.querySelectorAll(".lifecycle__step")).toHaveLength(4);
    const cards = screen.getAllByRole("article");
    expect(cards).toHaveLength(4);
    const science = within(screen.getByRole("article", { name: "Data Science & AI" }));
    const cv = projectData.computerVision.projects[0];
    expect(science.getByRole("link", { name: new RegExp(cv.title) })).toHaveAttribute("href", `#${projectPath("computerVision", 0)}`);
    expect(science.getByRole("list", { name: "Data Science & AI tools" })).toHaveTextContent("PyTorch");
  });

  it("lights up the matching 3D cluster on hover or focus", () => {
    renderRoute(<Domains onExplore={() => {}} />);
    const card = screen.getByRole("article", { name: "Data Analytics" });
    fireEvent.mouseEnter(card);
    expect(sceneState.highlight).toBe(1);
    expect(sceneState.pulse).toBeGreaterThan(0);
    fireEvent.mouseLeave(card);
    expect(sceneState.highlight).toBe(-1);
    fireEvent.focus(card);
    expect(sceneState.highlight).toBe(1);
    fireEvent.blur(card);
    expect(sceneState.highlight).toBe(-1);
  });

  it("opens the matching specialization in the Work section", () => {
    const onExplore = vi.fn();
    const rafs: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      rafs.push(cb);
      return 1;
    });
    const projects = document.createElement("section");
    projects.id = "projects";
    document.body.appendChild(projects);
    renderRoute(<Domains onExplore={onExplore} />);
    const deployment = within(screen.getByRole("article", { name: "MLOps & Deployment" }));
    fireEvent.click(deployment.getByRole("button", { name: /explore the work/i }));
    expect(onExplore).toHaveBeenCalledWith(specializations.findIndex((s) => s.id === "mlops"));
    rafs.forEach((cb) => cb(0));
    expect(window.scrollTo).toHaveBeenCalled();
    projects.remove();
  });
});

describe("DepthImage", () => {
  it("builds a picture and a lifted focal layer for cover images", () => {
    const { container } = render(<DepthImage src="a.png" alt="Pipeline diagram" />);
    const root = container.querySelector(".depth") as HTMLElement;
    expect(root).toHaveClass("depth--cover");
    expect(container.querySelectorAll("img")).toHaveLength(2);
    expect(container.querySelector(".depth__back")).toBeNull();
    expect(screen.getByAltText("Pipeline diagram")).toHaveAttribute("loading", "lazy");
    expect(container.querySelectorAll(".depth__particles i").length).toBeGreaterThan(3);
  });

  it("keeps the whole picture for contain-fit images and fills the space behind it", () => {
    const { container } = render(<DepthImage src="chart.png" alt="Loss curve" fit="contain" />);
    expect(container.querySelector(".depth")).toHaveClass("depth--contain");
    expect(container.querySelectorAll("img")).toHaveLength(2);
    const back = container.querySelector(".depth__back")!;
    expect(back).toHaveAttribute("aria-hidden", "true");
    // The blurred fill is static: it sits outside the animated 3D stage
    expect(back.closest(".depth__stage")).toBeNull();
    expect(screen.getByAltText("Loss curve")).toHaveAttribute("loading", "lazy");
  });

  it("only animates while on screen", async () => {
    const { container } = render(<DepthImage src="a.png" alt="" />);
    const root = container.querySelector(".depth") as HTMLElement;
    await flush();
    expect(root).toHaveAttribute("data-live");
    const observer = io.instances.find((o) => o.targets.includes(root))!;
    act(() => observer.fire(false, root));
    expect(root).not.toHaveAttribute("data-live");
  });

  it("tilts toward a mouse pointer and settles when it leaves", () => {
    const { container } = render(<DepthImage src="a.png" alt="" />);
    const root = container.querySelector(".depth") as HTMLElement;
    vi.spyOn(root, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 200, height: 100 } as DOMRect);
    fireEvent(root, new PointerEvent("pointermove", { pointerType: "mouse", clientX: 150, clientY: 25, bubbles: true }));
    expect(root.style.getPropertyValue("--dy")).toBe("3.5deg");
    expect(root.style.getPropertyValue("--dx")).toBe("2.5deg");
    fireEvent(root, new PointerEvent("pointermove", { pointerType: "touch", clientX: 0, clientY: 0, bubbles: true }));
    expect(root.style.getPropertyValue("--dy")).toBe("3.5deg");
    fireEvent.pointerLeave(root);
    expect(root.style.getPropertyValue("--dx")).toBe("0deg");
  });

  it("copes with an element that has no size yet", () => {
    const { container } = render(<DepthImage src="a.png" alt="" />);
    const root = container.querySelector(".depth") as HTMLElement;
    fireEvent(root, new PointerEvent("pointermove", { pointerType: "mouse", clientX: 10, clientY: 10, bubbles: true }));
    expect(root.style.getPropertyValue("--dx")).toBe("0deg");
  });
});

describe("project cards", () => {
  it("show which domains a project belongs to", () => {
    const project = projectData.mlops.projects[0];
    renderRoute(<ProjectCard project={project} category="mlops" index={0} />);
    expect(document.querySelector(".project-card__domains")?.textContent).toBe("Data Engineering · MLOps & Deployment");
    expect(document.querySelector(".project-card__media .depth")).toBeInTheDocument();
  });
});
