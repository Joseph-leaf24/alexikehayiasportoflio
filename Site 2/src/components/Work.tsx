import { useRef, type CSSProperties, type KeyboardEvent } from "react";
import { projectData } from "@/data/projects";
import { profile, specializations } from "@/data/profile";
import { sceneState, triggerPulse } from "@/lib/store";
import { playerFor } from "@/data/squad";
import PlayerPortrait from "@/components/fx/PlayerPortrait";
import { scrollToTarget } from "@/lib/smoothScroll";
import { cn, pad } from "@/lib/utils";
import SectionHead from "@/components/fx/SectionHead";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import ProjectCard from "./ProjectCard";

interface WorkProps {
  active: number;
  setActive: (i: number) => void;
}

/**
 * Work section ("mission select"): specialization tabs (ARIA tabs pattern with arrow / Home /
 * End keys and a roving tabindex), each with the player who stands for it (hovering points
 * them out in the scene), above the selected category's project cards.
 */
export default function Work({ active, setActive }: WorkProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const spec = specializations[active];
  const category = projectData[spec.id];

  const select = (i: number, focus = false) => {
    if (focus) tabRefs.current[i]?.focus();
    if (i === active) return;
    setActive(i);
    triggerPulse();
    // Bring the project list into view once it has rendered (only if it's mostly off-screen)
    requestAnimationFrame(() => {
      const el = panelRef.current;
      if (el && el.getBoundingClientRect().top > window.innerHeight * 0.65) scrollToTarget(el, { offset: -90 });
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const last = specializations.length - 1;
    const map: Record<string, number> = {
      ArrowRight: active === last ? 0 : active + 1,
      ArrowDown: active === last ? 0 : active + 1,
      ArrowLeft: active === 0 ? last : active - 1,
      ArrowUp: active === 0 ? last : active - 1,
      Home: 0,
      End: last,
    };
    if (e.key in map) {
      e.preventDefault();
      select(map[e.key], true);
    }
  };

  return (
    <section id="projects" className="section section--work" data-scene="bars" data-section="04|Work">
      <div className="container-x">
        <SectionHead index="04" label="Selected Work" />
        <div className="work-head">
          <h2 className="display-2">
            <ScrambleText text={profile.work.title} />
          </h2>
          <Reveal as="p" className="work-head__text" delay={150}>
            {profile.work.text}
          </Reveal>
        </div>

        <div role="tablist" aria-label="Specializations" className="spec-grid" onKeyDown={onKeyDown}>
          {specializations.map((s, i) => {
            const Icon = s.icon;
            const count = projectData[s.id].projects.length;
            const isActive = i === active;
            const member = playerFor(s.id);
            const point = (on: boolean) => {
              if (member) sceneState.highlight = on ? member.player - 1 : -1;
            };
            return (
              <Reveal key={s.id} delay={i * 70} variant="fade" className="spec-grid__cell">
                <button
                  ref={(el) => (tabRefs.current[i] = el)}
                  type="button"
                  role="tab"
                  id={`tab-${s.id}`}
                  aria-selected={isActive}
                  aria-controls="work-panel"
                  tabIndex={isActive ? 0 : -1}
                  className={cn("spec-tab", isActive && "is-active")}
                  onClick={() => select(i)}
                  onMouseEnter={() => point(true)}
                  onMouseLeave={() => point(false)}
                  onFocus={() => point(true)}
                  onBlur={() => point(false)}
                  data-cursor={isActive ? "" : "SELECT"}
                >
                  <span className="spec-tab__top">
                    <span>{pad(i + 1)}</span>
                    <span>
                      {pad(count)} {count === 1 ? "Project" : "Projects"}
                    </span>
                  </span>
                  <span className="spec-tab__art">
                    <Icon className="spec-tab__icon" aria-hidden="true" />
                    {member ? (
                      <span className="spec-tab__player" style={{ "--pc": member.color } as CSSProperties}>
                        <PlayerPortrait player={member.player} />
                        <span aria-hidden="true">P{member.player}</span>
                      </span>
                    ) : null}
                  </span>
                  <ScrambleText text={s.title} hover trigger="none" className="spec-tab__title" />
                  <span className="spec-tab__desc">{s.description}</span>
                  <span className="spec-tab__bar" aria-hidden="true" />
                </button>
              </Reveal>
            );
          })}
        </div>

        <div
          ref={panelRef}
          id="work-panel"
          role="tabpanel"
          aria-labelledby={`tab-${spec.id}`}
          className="work-panel"
          key={spec.id}
        >
          <div className="work-panel__head">
            <span className="mono-label">
              {pad(active + 1)} / {pad(specializations.length)} — {spec.title}
            </span>
            <h3 className="display-3">
              <ScrambleText text={category.title} trigger="mount" duration={800} />
            </h3>
            <p className="work-panel__desc">{category.description}</p>
          </div>

          <div className="project-grid">
            {category.projects.map((project, i) => (
              <ProjectCard key={project.title} project={project} category={spec.id} index={i} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
