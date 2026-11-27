import { useEffect, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import { profile } from "@/data/profile";
import { sceneState } from "@/lib/store";
import { gsap, prefersReducedMotion, scrollToTarget } from "@/lib/smoothScroll";
import { getQuality } from "@/lib/device";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";

/**
 * Hero: giant name, focus areas and the "View Projects" CTA. Pressing and holding on it
 * shatters the 3D scene. With a mouse, the letters of the name shy away from the cursor, unless
 * motion is reduced or the tier is low.
 */
const Hero = () => {
  const ref = useRef<HTMLElement>(null);
  const [coarse] = useState(() => window.matchMedia("(pointer: coarse)").matches);

  // Click / touch & hold anywhere on the hero (not on controls) to blow the village apart
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

  // Giant name: each letter leans away from the cursor like it's being pushed through ice
  useEffect(() => {
    const title = ref.current?.querySelector<HTMLElement>(".hero__title");
    if (!title || coarse || prefersReducedMotion() || getQuality().tier === "low") return;
    let raf = 0;
    let px = -9999;
    let py = -9999;
    const R = 240;
    const update = () => {
      raf = 0;
      if (window.scrollY > window.innerHeight) return;
      const chars = Array.from(title.querySelectorAll<HTMLElement>(".sc-ch"));
      // Read every letter's resting centre first, then write the transforms: reading after each
      // write would make the browser recompute styles once per letter. offset* ignores
      // transforms, so the centres stay put while the letters move.
      const parentRects = new Map<HTMLElement, DOMRect>();
      const centres = chars.map((ch) => {
        const parent = ch.offsetParent as HTMLElement | null;
        if (!parent) return null;
        let pr = parentRects.get(parent);
        if (!pr) parentRects.set(parent, (pr = parent.getBoundingClientRect()));
        return { x: pr.left + ch.offsetLeft + ch.offsetWidth / 2, y: pr.top + ch.offsetTop + ch.offsetHeight / 2 };
      });
      chars.forEach((ch, i) => {
        const c = centres[i];
        if (!c) return;
        const dx = c.x - px;
        const dy = c.y - py;
        const d = Math.hypot(dx, dy);
        if (d < R && d > 0.001) {
          const f = (1 - d / R) ** 2 * 34;
          ch.style.transform = `translate3d(${(dx / d) * f}px, ${(dy / d) * f}px, 0) rotate(${(dx / d) * f * 0.35}deg)`;
        } else if (ch.style.transform) ch.style.transform = "";
      });
    };
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      px = e.clientX;
      py = e.clientY;
      if (!raf) raf = requestAnimationFrame(update);
    };
    const onLeave = () => {
      px = py = -9999;
      if (!raf) raf = requestAnimationFrame(update);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    document.documentElement.addEventListener("mouseleave", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("mouseleave", onLeave);
    };
  }, [coarse]);

  // Hero copy drifts up and fades as you scroll away, unless motion is reduced. The scope is the
  // element itself, not the ref: React nulls the ref before this cleanup runs and ScrollTrigger
  // re-reads the scope while tearing down, which makes GSAP warn "Invalid scope".
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
            [ 00 ] — Portfolio
          </Reveal>
        </div>

        <div>
          <h1 className="hero__title">
            <span className="sr-only">{profile.name}</span>
            <span aria-hidden="true">
              <ScrambleText
                text={profile.firstName}
                trigger="mount"
                duration={1300}
                delay={120}
                className="hero__line"
              />
              <ScrambleText
                text={profile.lastName}
                trigger="mount"
                duration={1500}
                delay={320}
                className="hero__line hero__line--r"
              />
            </span>
          </h1>

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
                className="btn-ice btn-ice--solid"
                onClick={() => scrollToTarget("#projects", { offset: -10 })}
                data-cursor="SCROLL"
              >
                <ScrambleText text="View Projects" trigger="none" hover />
                <ArrowDown className="h-4 w-4" />
              </button>
            </Reveal>
          </div>
        </div>

        <div className="scroll-cue" aria-hidden="true">
          <span>Scroll</span>
          <i />
        </div>
      </div>
    </section>
  );
};

export default Hero;
