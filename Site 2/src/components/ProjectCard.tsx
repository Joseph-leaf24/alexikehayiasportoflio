import { ArrowUpRight } from "lucide-react";
import type { Project } from "@/types/project";
import { projectPath } from "@/data/profile";
import { triggerPulse } from "@/lib/store";
import { TLink } from "@/components/chrome/Transition";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";
import PixelImage from "@/components/fx/PixelImage";
import { domainsFor } from "@/data/domains";
import { pad } from "@/lib/utils";

interface ProjectCardProps {
  project: Project;
  category: string;
  /** Index within the category (builds the link and the P/xx label, and staggers the reveal) */
  index: number;
}

/**
 * Project card ("mission card") linking to its detail page: the project image with the
 * mosaic reveal, domains, chips. The link is the title, stretched over the whole card by
 * CSS (so its accessible name is just the title). On hover the card hops up a few pixels.
 */
const ProjectCard = ({ project, category, index }: ProjectCardProps) => {
  const { title, description, image, tags, tools, concepts } = project;

  return (
    <Reveal delay={index * 110} className="card-wrap">
      <article className="project-card hc" onPointerEnter={() => triggerPulse()} data-scramble-host="">
        <Corners />
        <div className="project-card__media">
          <PixelImage src={image} alt="" />
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
