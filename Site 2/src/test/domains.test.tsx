import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import Domains from "@/components/Domains";
import PixelImage from "@/components/fx/PixelImage";
import ProjectCard from "@/components/ProjectCard";
import { domainIntro, domains, domainsFor, projectsOf } from "@/data/domains";
import { allProjects, projectPath, specializations } from "@/data/profile";
import { projectData } from "@/data/projects";
import { SQUAD } from "@/data/squad";
import { sceneState } from "@/lib/store";
import { renderRoute, resetState } from "./utils";
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

  it("gives every domain a player in the squad, P1–P4 in lifecycle order, each in its own colour", () => {
    expect(SQUAD.slice(0, 4).map((m) => m.domain)).toEqual(domains.map((d) => d.id));
    expect(SQUAD.map((m) => m.player)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(new Set(SQUAD.map((m) => m.color)).size).toBe(8);
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
    // Squad select: each card names its player
    expect(science.getByText("Data Scientist", { exact: false })).toBeInTheDocument();
    expect(science.getByText("Plasma")).toBeInTheDocument();
    const cv = projectData.computerVision.projects[0];
    expect(science.getByRole("link", { name: new RegExp(cv.title) })).toHaveAttribute(
      "href",
      `#${projectPath("computerVision", 0)}`
    );
    expect(science.getByRole("list", { name: "Data Science & AI tools" })).toHaveTextContent("PyTorch");
  });

  it("points out the matching player on hover or focus", () => {
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

describe("PixelImage", () => {
  it("keeps one real, lazy picture for screen readers and as the fallback, under a hidden mosaic canvas", () => {
    const { container } = render(<PixelImage src="a.png" alt="Pipeline diagram" />);
    const root = container.querySelector(".pximg") as HTMLElement;
    expect(root).toHaveClass("pximg--cover");
    expect(container.querySelectorAll("img")).toHaveLength(1);
    expect(screen.getByAltText("Pipeline diagram")).toHaveAttribute("loading", "lazy");
    expect(container.querySelector(".pximg__mosaic")).toHaveAttribute("aria-hidden", "true");
    // No canvas in jsdom: the picture just shows, no mosaic
    expect(root).not.toHaveAttribute("data-mosaic");
  });

  it("keeps the whole picture for contain-fit images", () => {
    const { container } = render(<PixelImage src="chart.png" alt="Loss curve" fit="contain" />);
    expect(container.querySelector(".pximg")).toHaveClass("pximg--contain");
  });

  it("resolves from blocks to sharp once it scrolls into view", async () => {
    // A stand-in 2D context, and a picture that has already loaded
    const draws: number[] = [];
    const ctx = {
      imageSmoothingEnabled: false,
      clearRect() {},
      drawImage() {
        draws.push(1);
      },
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLImageElement.prototype, "complete", "get").mockReturnValue(true);
    vi.spyOn(HTMLImageElement.prototype, "naturalWidth", "get").mockReturnValue(400);
    vi.spyOn(HTMLImageElement.prototype, "naturalHeight", "get").mockReturnValue(300);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(320);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(180);
    vi.useFakeTimers();
    io.autoIntersect = false;
    const { container } = render(<PixelImage src="a.png" alt="" />);
    const root = container.querySelector(".pximg") as HTMLElement;
    const canvas = container.querySelector("canvas") as HTMLCanvasElement;
    // Off screen it waits behind the coarsest mosaic
    expect(root).toHaveAttribute("data-mosaic");
    expect(canvas.width).toBe(Math.ceil(320 / 28));
    act(() => io.instances.find((o) => o.targets.includes(root))!.fire(true, root));
    act(() => vi.advanceTimersByTime(90));
    expect(canvas.width).toBe(Math.ceil(320 / 16));
    act(() => vi.advanceTimersByTime(1000));
    expect(root).not.toHaveAttribute("data-mosaic");
    expect(draws.length).toBeGreaterThanOrEqual(4);
  });
});

describe("project cards", () => {
  it("show which domains a project belongs to", () => {
    const project = projectData.mlops.projects[0];
    renderRoute(<ProjectCard project={project} category="mlops" index={0} />);
    expect(document.querySelector(".project-card__domains")?.textContent).toBe("Data Engineering · MLOps & Deployment");
    expect(document.querySelector(".project-card__media .pximg")).toBeInTheDocument();
  });
});
