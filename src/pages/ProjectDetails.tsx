import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ArrowUpRight, Github } from "lucide-react";
import { projectData } from "@/data/projects";
import { allProjects, projectPath, specializations, type CategoryId } from "@/data/profile";
import { sceneState, ui } from "@/lib/store";
import { gsap, prefersReducedMotion, scrollToTarget } from "@/lib/smoothScroll";
import { TLink } from "@/components/chrome/Transition";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";
import Lightbox from "@/components/Lightbox";

const pad = (n: number) => String(n).padStart(2, "0");

const ProjectDetails = () => {
  const { category, projectIndex } = useParams();
  const [lightbox, setLightbox] = useState<number | null>(null);
  const coverRef = useRef<HTMLDivElement>(null);

  const categoryData = category && category in projectData ? projectData[category] : null;
  const idx = Number(projectIndex);
  const project = categoryData && Number.isInteger(idx) ? categoryData.projects[idx] : undefined;
  const spec = specializations.find((s) => s.id === category);
  const flatIndex = allProjects.findIndex((r) => r.category === category && r.index === idx);
  const total = allProjects.length;
  const prev = flatIndex >= 0 ? allProjects[(flatIndex - 1 + total) % total] : null;
  const next = flatIndex >= 0 ? allProjects[(flatIndex + 1) % total] : null;

  // Always open a project at the top (HashRouter doesn't reset scroll)
  useLayoutEffect(() => {
    scrollToTarget(0, { immediate: true });
    setLightbox(null);
  }, [category, projectIndex]);

  useEffect(() => {
    sceneState.override = "cube";
    ui.set({ sectionIndex: flatIndex >= 0 ? pad(flatIndex + 1) : "--", sectionLabel: spec?.title ?? "Project" });
    const prevTitle = document.title;
    if (project) document.title = `${project.title} — Alexi Kehayias`;
    return () => {
      sceneState.override = null;
      document.title = prevTitle;
    };
  }, [project, spec, flatIndex]);

  const closeLightbox = useCallback(() => setLightbox(null), []);

  // Cover image drifts inside its frame while scrolling
  useEffect(() => {
    const el = coverRef.current;
    if (!el || prefersReducedMotion()) return;
    const img = el.querySelector("img");
    const tween = gsap.fromTo(
      img,
      { yPercent: -7 },
      { yPercent: 7, ease: "none", scrollTrigger: { trigger: el, start: "top bottom", end: "bottom top", scrub: true } }
    );
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
  }, [category, projectIndex]);

  const backState = { scrollTo: "projects", category: category as CategoryId };

  if (!project || !categoryData) {
    return (
      <section className="notfound container-x">
        <span className="mono-label">[ 404 ] — Project</span>
        <h1 className="display-2">
          <ScrambleText text="Project not found" trigger="mount" />
        </h1>
        <TLink to="/" state={{ scrollTo: "projects" }} className="btn-ice">
          <ArrowLeft className="h-4 w-4" /> Back to Home
        </TLink>
      </section>
    );
  }

  const images = project.images ?? [];

  return (
    <article className="detail container-x">
      <div className="detail__top">
        <TLink to="/" state={backState} className="back-link" data-cursor="BACK">
          <ArrowLeft className="h-4 w-4" />
          <ScrambleText text="Back to Portfolio" trigger="none" hover />
        </TLink>
        <span className="mono-label">
          {pad(flatIndex + 1)} / {pad(total)}
        </span>
      </div>

      <header className="detail__head">
        <Reveal variant="fade" className="mono-label detail__cat">
          [ {pad(idx + 1)} ] — {categoryData.title}
          {spec && spec.title !== categoryData.title ? ` · ${spec.title}` : ""}
        </Reveal>
        <h1 className="detail__title">
          <ScrambleText text={project.title} trigger="mount" duration={1100} />
        </h1>
        <Reveal delay={200}>
          <p className="detail__desc">{project.description}</p>
        </Reveal>
        <Reveal delay={320} className="detail__actions">
          {project.githubUrl ? (
            <a className="btn-ice btn-ice--solid" href={project.githubUrl} target="_blank" rel="noopener noreferrer" data-cursor="GITHUB">
              <Github className="h-4 w-4" />
              <ScrambleText text="View on GitHub" trigger="none" hover />
              <ArrowUpRight className="h-4 w-4" />
            </a>
          ) : null}
          {project.demoUrl ? (
            <a className="btn-ice" href={project.demoUrl} target="_blank" rel="noopener noreferrer" data-cursor="OPEN">
              <ScrambleText text="Live Demo" trigger="none" hover />
              <ArrowUpRight className="h-4 w-4" />
            </a>
          ) : null}
          {project.powerBiUrl ? (
            <a className="btn-ice" href={project.powerBiUrl} target="_blank" rel="noopener noreferrer" data-cursor="OPEN">
              <ScrambleText text="Open Dashboard" trigger="none" hover />
              <ArrowUpRight className="h-4 w-4" />
            </a>
          ) : null}
        </Reveal>
      </header>

      <Reveal variant="clip" className="detail__cover">
        <div ref={coverRef} className="detail__cover-frame">
          <img src={project.image} alt={`${project.title} — cover`} decoding="async" />
        </div>
        <Corners />
        <span className="detail__cover-tag">Fig. 01 — {categoryData.title}</span>
      </Reveal>

      <div className="detail__specs">
        <Reveal className="spec-panel hc">
          <Corners />
          <h3 className="mono-label">Technologies Used</h3>
          <ul className="chips chips--lg">
            {project.tags.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </Reveal>
        {project.tools && project.tools.length > 0 ? (
          <Reveal className="spec-panel hc" delay={100}>
            <Corners />
            <h3 className="mono-label">Tools &amp; Languages</h3>
            <ul className="chips chips--lg">
              {project.tools.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </Reveal>
        ) : null}
        <Reveal className="spec-panel hc" delay={200}>
          <Corners />
          <h3 className="mono-label">AI/Data Science Concepts</h3>
          <ul className="chips chips--lg chips--accent">
            {project.concepts.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </Reveal>
      </div>

      {project.powerBiUrl ? (
        <section className="detail__block">
          <div className="detail__block-head">
            <h2 className="mono-label">Interactive Dashboard</h2>
            <span className="mono-label">Microsoft Power BI</span>
          </div>
          <Reveal variant="clip" className="embed-frame">
            <Corners />
            <iframe title={project.title} src={project.powerBiUrl} allowFullScreen loading="lazy" />
          </Reveal>
        </section>
      ) : null}

      {images.length > 0 ? (
        <section className="detail__block">
          <div className="detail__block-head">
            <h2 className="mono-label">Project Gallery</h2>
            <span className="mono-label">{pad(images.length)} assets</span>
          </div>
          <div className="gallery">
            {images.map((src, i) => (
              <Reveal key={src + i} variant="clip" delay={(i % 3) * 90}>
                <button type="button" className="gallery__item hc" onClick={() => setLightbox(i)} data-cursor="EXPAND" aria-label={`Open image ${i + 1} of ${images.length}`}>
                  <img src={src} alt={`${project.title} - Image ${i + 1}`} loading="lazy" decoding="async" />
                  <Corners />
                  <span className="gallery__idx">IMG/{pad(i + 1)}</span>
                </button>
              </Reveal>
            ))}
          </div>
        </section>
      ) : null}

      {prev && next ? (
        <nav className="pager" aria-label="More projects">
          <TLink to={projectPath(prev.category, prev.index)} className="pager__link" data-cursor="PREV">
            <span className="mono-label">
              <ArrowLeft className="h-3.5 w-3.5" /> Previous project
            </span>
            <ScrambleText text={prev.project.title} trigger="none" hover className="pager__title" />
          </TLink>
          <TLink to={projectPath(next.category, next.index)} className="pager__link pager__link--next" data-cursor="NEXT">
            <span className="mono-label">
              Next project <ArrowRight className="h-3.5 w-3.5" />
            </span>
            <ScrambleText text={next.project.title} trigger="none" hover className="pager__title" />
          </TLink>
        </nav>
      ) : null}

      {lightbox !== null ? (
        <Lightbox images={images} index={lightbox} title={project.title} onClose={closeLightbox} onIndex={setLightbox} />
      ) : null}
    </article>
  );
};

export default ProjectDetails;
