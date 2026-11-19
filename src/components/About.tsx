import { useEffect, useRef } from "react";
import { ArrowUpRight, Linkedin } from "lucide-react";
import profilePhoto from "@/assets/profile_photo.jpg";
import { allProjects, profile, specializations } from "@/data/profile";
import { SKILLS } from "@/data/skills";
import { gsap, prefersReducedMotion } from "@/lib/smoothScroll";
import SectionHead from "@/components/fx/SectionHead";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";
import Counter from "@/components/fx/Counter";

const STATS = [
  { value: allProjects.length, label: "Projects" },
  { value: specializations.length, label: "Specializations" },
  { value: SKILLS.length, label: "Tools in the toolbox" },
  { value: profile.focus.length, label: "Focus areas" },
];

export default function About() {
  const manifestoRef = useRef<HTMLParagraphElement>(null);

  // Words light up one by one as the statement scrolls through the viewport
  useEffect(() => {
    if (prefersReducedMotion()) return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".mw",
        { opacity: 0.12 },
        {
          opacity: 1,
          stagger: 0.08,
          ease: "none",
          scrollTrigger: { trigger: manifestoRef.current, start: "top 85%", end: "bottom 45%", scrub: true },
        }
      );
    }, manifestoRef);
    return () => ctx.revert();
  }, []);

  return (
    <section id="about" className="section" data-scene="shatter" data-section="01|Who I Am">
      <div className="container-x">
        <SectionHead index="01" label="Who I Am" />

        <p ref={manifestoRef} className="manifesto">
          {profile.statement.split(" ").map((w, i) => (
            <span key={i} className="mw">
              {w}{" "}
            </span>
          ))}
        </p>

        <div className="stats">
          {STATS.map((s, i) => (
            <Reveal key={s.label} className="stat" delay={i * 90}>
              <Counter value={s.value} className="stat__num" />
              <span className="stat__label">{s.label}</span>
            </Reveal>
          ))}
        </div>

        <div className="about-grid">
          <Reveal variant="clip" className="portrait hc">
            <img src={profilePhoto} alt={`Portrait of ${profile.name}`} width={200} height={200} />
            <span className="portrait__scan" aria-hidden="true" />
            <Corners />
            <dl className="portrait__meta">
              <div>
                <dt>Subject</dt>
                <dd>{profile.name}</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd>{profile.role}</dd>
              </div>
              <div>
                <dt>Base</dt>
                <dd>{profile.location}</dd>
              </div>
            </dl>
          </Reveal>

          <div className="about-copy">
            <Reveal>
              <p className="lead">{profile.intro}</p>
            </Reveal>

            <div className="focus-list">
              <Reveal variant="fade" className="mono-label">
                Focus areas
              </Reveal>
              {profile.focus.map((f, i) => (
                <Reveal key={f} delay={i * 90} className="focus-row" as="div">
                  <span className="focus-row__idx">0{i + 1}</span>
                  <ScrambleText text={f} className="focus-row__name" />
                  <span className="focus-row__rule" />
                </Reveal>
              ))}
            </div>

            <Reveal className="about-links" delay={200}>
              <a className="btn-ice" href={profile.linkedin} target="_blank" rel="noopener noreferrer" data-cursor="OPEN">
                <Linkedin className="h-4 w-4" />
                <ScrambleText text="LinkedIn" trigger="none" hover />
              </a>
              <a className="btn-ice" href={profile.cv} download data-cursor="PDF">
                <ScrambleText text="Download CV" trigger="none" hover />
                <ArrowUpRight className="h-4 w-4" />
              </a>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
