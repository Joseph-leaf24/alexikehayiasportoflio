import { useRef, type PointerEvent } from "react";
import { ArrowUpRight } from "lucide-react";
import type { Project } from "@/types/project";
import { projectPath } from "@/data/profile";
import { triggerPulse } from "@/lib/store";
import { TLink } from "@/components/chrome/Transition";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";
import DepthImage from "@/components/fx/DepthImage";
import { domainsFor } from "@/data/domains";

interface ProjectCardProps {
  project: Project;
  category: string;
  /** Index within the category (builds the link and the P/xx label) */
  index: number;
  /** Position in the grid (staggers the reveal) */
  order: number;
}

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * Project card linking to its detail page: the project image with the depth effect, domains, chips.
 * The link is the title, stretched over the whole card by CSS (so its accessible name is just the
 * title). Tilts toward the mouse (--rx/--ry) with a glare that follows it (--mx/--my).
 */
const ProjectCard = ({ project, category, index, order }: ProjectCardProps) => {
  const ref = useRef<HTMLElement>(null);
  const { title, description, image, tags, tools, concepts } = project;

  const onMove = (e: PointerEvent<HTMLElement>) => {
    const el = ref.current;
    if (!el || e.pointerType !== "mouse") return;
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    el.style.setProperty("--rx", `${(0.5 - y) * 7}deg`);
    el.style.setProperty("--ry", `${(x - 0.5) * 9}deg`);
    el.style.setProperty("--mx", `${x * 100}%`);
    el.style.setProperty("--my", `${y * 100}%`);
  };
  const onLeave = () => {
    const el = ref.current;
    if (!el) return;
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  };

  return (
    <Reveal delay={order * 110} className="card-wrap">
      <article
        ref={ref}
        className="project-card hc"
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        onPointerEnter={() => triggerPulse()}
        data-scramble-host=""
      >
        <Corners />
        <div className="project-card__media">
          <DepthImage src={image} alt="" />
          <span className="project-card__tint" aria-hidden="true" />
          <span className="project-card__idx">P/{pad(index + 1)}</span>
        </div>

        <div className="project-card__body">
          <span className="project-card__domains">{domainsFor(category, index).map((d) => d.title).join(" · ")}</span>
          <h4 className="project-card__title">
            <TLink to={projectPath(category, index)} className="project-card__link" data-cursor="OPEN">
              <ScrambleText text={title} hover trigger="none" />
            </TLink>
          </h4>
          <p className="project-card__desc">{description}</p>

          <div className="project-card__meta">
            <div>
              <span className="chip-label">Technologies</span>
              <ul className="chips">
                {tags.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </div>
            {tools && tools.length > 0 ? (
              <div>
                <span className="chip-label">Tools &amp; Languages</span>
                <ul className="chips">
                  {tools.map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            <div>
              <span className="chip-label">AI/Data Science Concepts</span>
              <ul className="chips chips--accent">
                {concepts.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          </div>

          <span className="project-card__cta" aria-hidden="true">
            View project <ArrowUpRight className="h-4 w-4" />
          </span>
        </div>
      </article>
    </Reveal>
  );
};

export default ProjectCard;
