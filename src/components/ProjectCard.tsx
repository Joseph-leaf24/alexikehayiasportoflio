import { useRef, type PointerEvent } from "react";
import { ArrowUpRight } from "lucide-react";
import type { Project } from "@/types/project";
import { projectPath } from "@/data/profile";
import { triggerPulse } from "@/lib/store";
import { TLink } from "@/components/chrome/Transition";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";

interface ProjectCardProps {
  project: Project;
  category: string;
  index: number;
  order: number;
}

const pad = (n: number) => String(n).padStart(2, "0");

const ProjectCard = ({ project, category, index, order }: ProjectCardProps) => {
  const ref = useRef<HTMLAnchorElement>(null);
  const { title, description, image, tags, tools, concepts, powerBiUrl } = project;

  const onMove = (e: PointerEvent<HTMLAnchorElement>) => {
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
      <TLink
        ref={ref}
        to={projectPath(category, index)}
        className="project-card hc"
        data-cursor="OPEN"
        aria-label={`${title} — view project`}
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        onPointerEnter={() => triggerPulse()}
        data-scramble-host=""
      >
        <Corners />
        <div className="project-card__media">
          <img src={image} alt="" loading="lazy" decoding="async" />
          <span className="project-card__tint" aria-hidden="true" />
          <span className="project-card__idx">P/{pad(index + 1)}</span>
          {powerBiUrl ? <span className="project-card__flag">Live dashboard</span> : null}
        </div>

        <div className="project-card__body">
          <h4 className="project-card__title">
            <ScrambleText text={title} hover trigger="none" />
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

          <span className="project-card__cta">
            View project <ArrowUpRight className="h-4 w-4" />
          </span>
        </div>
      </TLink>
    </Reveal>
  );
};

export default ProjectCard;
