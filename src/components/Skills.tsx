import { useEffect, useRef } from "react";
import { SKILLS } from "@/data/skills";
import { profile } from "@/data/profile";
import { sceneState } from "@/lib/store";
import SectionHead from "@/components/fx/SectionHead";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import Corners from "@/components/fx/Corners";

const pad = (n: number) => String(n).padStart(2, "0");

/** One marquee row. The set is rendered twice (the copy hidden from screen readers) so the CSS loop is seamless. */
function MarqueeRow({ reverse = false, outline = false }: { reverse?: boolean; outline?: boolean }) {
  const items = reverse ? [...SKILLS].reverse() : SKILLS;
  const set = (key: string) => (
    <div className="marquee__set" key={key} aria-hidden={key === "b"}>
      {items.map(({ name, Icon }) => (
        <span key={name} className="marquee__item">
          <Icon className="marquee__icon" />
          <span className={outline ? "text-outline" : undefined}>{name}</span>
          <span className="marquee__sep">/</span>
        </span>
      ))}
    </div>
  );
  return <div className={`marquee__row${reverse ? " marquee__row--rev" : ""}`}>{[set("a"), set("b")]}</div>;
}

/** Toolbox section: a two-row marquee that skews with scroll velocity, then a grid of skill tiles. */
export default function Skills() {
  const marqueeRef = useRef<HTMLDivElement>(null);

  // Skew the marquee with scroll velocity (only while it's on screen)
  useEffect(() => {
    const el = marqueeRef.current;
    if (!el) return;
    let raf = 0;
    let skew = 0;
    let written = "";
    let visible = false;
    const io = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible) raf = requestAnimationFrame(loop);
      else cancelAnimationFrame(raf);
    });
    const loop = () => {
      if (!visible) return;
      raf = requestAnimationFrame(loop);
      const target = Math.max(-9, Math.min(9, sceneState.velocity / 260));
      skew += (target - skew) * 0.12;
      const next = `${(Math.abs(skew) < 0.01 ? 0 : skew).toFixed(2)}deg`;
      if (next !== written) el.style.setProperty("--skew", (written = next));
    };
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section id="toolbox" className="section section--toolbox" data-scene="network" data-section="03|Toolbox">
      <div className="container-x">
        <SectionHead index="03" label="Toolbox" />
        <div className="toolbox-head">
          <h2 className="display-2">
            <ScrambleText text={profile.toolbox.title} />
          </h2>
          <Reveal as="p" className="toolbox-head__text" delay={150}>
            {profile.toolbox.text}
          </Reveal>
        </div>
      </div>

      <div ref={marqueeRef} className="marquee" aria-hidden="true">
        <MarqueeRow />
        <MarqueeRow reverse outline />
      </div>

      <div className="container-x">
        <ul className="skills-grid" aria-label="Skills">
          {SKILLS.map(({ name, Icon, hint }, i) => (
            <Reveal as="li" key={name} className="skill-tile hc" delay={(i % 6) * 60} data-scramble-host="">
              <Corners />
              <span className="skill-tile__idx">S/{pad(i + 1)}</span>
              <Icon className="skill-tile__icon" />
              <div>
                <ScrambleText text={name} hover trigger="none" className="skill-tile__name" />
                <span className="skill-tile__hint">{hint ?? "—"}</span>
              </div>
            </Reveal>
          ))}
        </ul>
      </div>
    </section>
  );
}
