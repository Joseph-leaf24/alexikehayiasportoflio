import { useEffect, useRef, type CSSProperties } from "react";
import { ArrowDown } from "lucide-react";
import { profile } from "@/data/profile";
import { sceneState } from "@/lib/store";
import { gsap, prefersReducedMotion, scrollToTarget } from "@/lib/smoothScroll";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";
import PlayerPortrait from "@/components/fx/PlayerPortrait";
import { SQUAD } from "@/data/squad";

/**
 * Hero section, styled as a title screen: the player line-up, focus areas and the "View Projects"
 * button, with the rest left open for the battle behind. Pressing and holding on it calls in an
 * airstrike on the scene.
 */
const Hero = () => {
  const ref = useRef<HTMLElement>(null);

  // Click or touch and hold anywhere on the hero except its controls to call in the airstrike
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let start: { x: number; y: number } | null = null;
    const down = (e: PointerEvent) => {
      // Main button / touch only: a right-click's menu would swallow the release
      if (e.button !== 0 || (e.target instanceof Element && e.target.closest("a,button"))) return;
      start = { x: e.clientX, y: e.clientY };
      sceneState.press = true;
    };
    // A touch that turns into a scroll or drag cancels the hold
    const move = (e: PointerEvent) => {
      if (start && e.pointerType !== "mouse" && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 12) up();
    };
    const up = () => {
      start = null;
      sceneState.press = false;
    };
    const releases = ["pointerup", "pointercancel", "blur", "contextmenu"] as const;
    el.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move, { passive: true });
    for (const ev of releases) window.addEventListener(ev, up);
    return () => {
      up();
      el.removeEventListener("pointerdown", down);
      window.removeEventListener("pointermove", move);
      for (const ev of releases) window.removeEventListener(ev, up);
    };
  }, []);

  // Hero copy drifts up and fades as you scroll away, except with reduced motion. The GSAP scope is
  // the element, not the ref: React nulls the ref before this cleanup runs, and ScrollTrigger re-reads
  // the scope while tearing down, which made GSAP warn "Invalid scope" every time the page was left.
  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion()) return;
    const ctx = gsap.context(() => {
      gsap.to(".hero__inner", {
        yPercent: -14,
        opacity: 0,
        ease: "none",
        scrollTrigger: { trigger: el, start: "top top", end: "bottom top", scrub: true },
      });
    }, el);
    return () => ctx.revert();
  }, []);

  return (
    <section id="top" ref={ref} className="hero" data-scene="village" data-section="00|Intro">
      <div className="hero__inner container-x">
        <div className="hero__top">
          <Reveal variant="fade" className="mono-label">
            Portfolio Website
          </Reveal>
          <Reveal className="hero__squad" delay={500} aria-hidden="true">
            {SQUAD.map((m) => (
              <span key={m.player} className="hero__player" style={{ "--pc": m.color } as CSSProperties}>
                <PlayerPortrait player={m.player} />
                <b>P{m.player}</b>
                <span>{m.short}</span>
              </span>
            ))}
          </Reveal>
        </div>

        <div className="hero__main">
          {/* The page heading for screen readers and search, hidden to leave the room to the scene */}
          <h1 className="sr-only">{profile.name}</h1>

          <div className="hero__meta">
            <Reveal className="hero__focus" delay={700}>
              <span className="mono-label">Focus</span>
              <ul>
                {profile.focus.map((f, i) => (
                  <li key={f}>
                    <span>0{i + 1}</span>
                    {f}
                  </li>
                ))}
              </ul>
            </Reveal>
            <Reveal className="hero__ctas" delay={850}>
              <button
                type="button"
                className="btn-px btn-px--solid"
                onClick={() => scrollToTarget("#projects", { offset: -10 })}
                data-cursor="SCROLL"
              >
                <span className="btn-px__caret" aria-hidden="true" />
                <ScrambleText text="View Projects" trigger="none" hover />
                <ArrowDown className="h-4 w-4" />
              </button>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
