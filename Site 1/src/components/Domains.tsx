import { Fragment, useEffect } from "react";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { domainIntro, domains, projectsOf } from "@/data/domains";
import { projectPath, specializations } from "@/data/profile";
import { sceneState, triggerPulse } from "@/lib/store";
import { scrollToTarget } from "@/lib/smoothScroll";
import { TLink } from "@/components/chrome/Transition";
import SectionHead from "@/components/fx/SectionHead";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";
import { pad } from "@/lib/utils";

interface DomainsProps {
  /** Open a specialization tab in the Work section */
  onExplore: (specializationIndex: number) => void;
}

/**
 * Domains section: the data lifecycle strip and one card per domain with its related
 * projects. Hovering or focusing a card lights up its cluster in the 3D scene; "Explore
 * the work" opens the matching specialization in Work and scrolls there.
 */
export default function Domains({ onExplore }: DomainsProps) {
  // Following a card's project link leaves the page before mouseleave / blur fire: clear
  // the highlight on the way out, or the cluster would still glow on coming back
  useEffect(
    () => () => {
      sceneState.highlight = -1;
    },
    []
  );

  const explore = (categoryId: string) => {
    const i = specializations.findIndex((s) => s.id === categoryId);
    onExplore(Math.max(0, i));
    requestAnimationFrame(() => scrollToTarget("#projects", { offset: -10 }));
  };

  return (
    <section id="domains" className="section section--domains" data-scene="clusters" data-section="02|Domains">
      <div className="container-x">
        <SectionHead index="02" label="Domains" />
        <div className="work-head">
          <h2 className="display-2">
            <ScrambleText text={domainIntro.title} />
          </h2>
          <Reveal as="p" className="work-head__text" delay={150}>
            {domainIntro.text}
          </Reveal>
        </div>

        {/* The data lifecycle: each domain hands over to the next */}
        <Reveal className="lifecycle" aria-hidden="true">
          {domains.map((d, i) => (
            <Fragment key={d.id}>
              <span className="lifecycle__step">
                <b>{pad(i + 1)}</b> {d.step}
              </span>
              {i < domains.length - 1 ? <span className="lifecycle__link" /> : null}
            </Fragment>
          ))}
        </Reveal>

        <div className="domain-grid">
          {domains.map((d, i) => {
            const Icon = d.icon;
            const related = projectsOf(d);
            return (
              <Reveal
                key={d.id}
                as="article"
                className="domain hc"
                delay={i * 90}
                aria-labelledby={`domain-${d.id}`}
                onMouseEnter={() => {
                  sceneState.highlight = i;
                  triggerPulse();
                }}
                onMouseLeave={() => {
                  sceneState.highlight = -1;
                }}
                onFocus={() => {
                  sceneState.highlight = i;
                }}
                onBlur={() => {
                  sceneState.highlight = -1;
                }}
              >
                <Corners />
                <div className="domain__top">
                  <span>
                    {pad(i + 1)} · {d.step}
                  </span>
                  <Icon className="domain__icon" aria-hidden="true" />
                </div>
                <h3 id={`domain-${d.id}`} className="domain__title">
                  <ScrambleText text={d.title} trigger="none" hover />
                </h3>
                <p className="domain__tagline">{d.tagline}</p>
                <p className="domain__desc">{d.description}</p>

                <ul className="domain__caps">
                  {d.capabilities.map((c) => (
                    <li key={c}>{c}</li>
                  ))}
                </ul>

                <ul className="chips" aria-label={`${d.title} tools`}>
                  {d.tools.map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>

                <div className="domain__projects">
                  <span className="chip-label">{pad(related.length)} related projects</span>
                  <ul>
                    {related.map(({ category, index, project }) => (
                      <li key={`${category}-${index}`}>
                        <TLink to={projectPath(category, index)} data-cursor="OPEN">
                          {project.title}
                          <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
                        </TLink>
                      </li>
                    ))}
                  </ul>
                </div>

                <button
                  type="button"
                  className="btn-ice btn-ice--sm domain__explore"
                  onClick={() => explore(d.explore)}
                >
                  <ScrambleText text="Explore the work" trigger="none" hover />
                  <ArrowRight className="h-4 w-4" />
                </button>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
