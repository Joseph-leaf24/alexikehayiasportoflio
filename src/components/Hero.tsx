import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUpRight } from "lucide-react";
import { profile } from "@/data/profile";
import { sceneState } from "@/lib/store";
import { gsap, prefersReducedMotion, scrollToTarget } from "@/lib/smoothScroll";
import { sound } from "@/lib/sound";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";

const Hero = () => {
  const ref = useRef<HTMLElement>(null);
  const [coarse] = useState(() => window.matchMedia("(pointer: coarse)").matches);

  // Click / touch & hold anywhere on the hero (not on controls) to blow the village apart
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let start: { x: number; y: number } | null = null;
    const down = (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest("a,button")) return;
      start = { x: e.clientX, y: e.clientY };
      sceneState.press = true;
      el.classList.add("is-pressing");
      sound.blip("click");
    };
    const move = (e: PointerEvent) => {
      if (start && e.pointerType !== "mouse" && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 12) up();
    };
    const up = () => {
      start = null;
      sceneState.press = false;
      el.classList.remove("is-pressing");
    };
    el.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move, { passive: true });
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    window.addEventListener("blur", up);
    return () => {
      up();
      el.removeEventListener("pointerdown", down);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      window.removeEventListener("blur", up);
    };
  }, []);

  // Giant name: each letter leans away from the cursor like it's being pushed through ice
  useEffect(() => {
    const title = ref.current?.querySelector<HTMLElement>(".hero__title");
    if (!title || coarse || prefersReducedMotion()) return;
    let raf = 0;
    let px = -9999;
    let py = -9999;
    const R = 240;
    const update = () => {
      raf = 0;
      if (window.scrollY > window.innerHeight) return;
      const chars = title.querySelectorAll<HTMLElement>(".sc-ch");
      chars.forEach((ch) => {
        const parent = ch.offsetParent as HTMLElement | null;
        if (!parent) return;
        const pr = parent.getBoundingClientRect();
        // offset* ignores transforms, so the letter's resting centre stays stable
        const cx = pr.left + ch.offsetLeft + ch.offsetWidth / 2;
        const cy = pr.top + ch.offsetTop + ch.offsetHeight / 2;
        const dx = cx - px;
        const dy = cy - py;
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

  // Hero copy drifts up and fades as you leave it
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.to(".hero__inner", {
        yPercent: -14,
        opacity: 0,
        ease: "none",
        scrollTrigger: { trigger: ref.current, start: "top top", end: "bottom top", scrub: true },
      });
    }, ref);
    return () => ctx.revert();
  }, []);

  return (
    <section id="top" ref={ref} className="hero" data-scene="village" data-section="00|Intro">
      <div className="hero__inner container-x">
        <div className="hero__top">
          <Reveal variant="fade" className="mono-label">
            [ 00 ] — Portfolio · {new Date().getFullYear()}
          </Reveal>
          <ScrambleText text={profile.role} trigger="mount" delay={500} className="mono-label hero__role" />
        </div>

        <div className="hero__hint" aria-hidden="true">
          <span className="hero__hint-ring" />
          <ScrambleText text={coarse ? "Touch & hold" : "Click & hold to shatter"} trigger="mount" delay={1400} className="mono-label" />
        </div>

        <div className="hero__bottom">
          <h1 className="hero__title">
            <span className="sr-only">{profile.name}</span>
            <span aria-hidden="true">
              <ScrambleText text={profile.firstName} trigger="mount" duration={1300} delay={120} className="hero__line" />
              <ScrambleText text={profile.lastName} trigger="mount" duration={1500} delay={320} className="hero__line hero__line--r" />
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
              <button type="button" className="btn-ice" onClick={() => scrollToTarget("#projects", { offset: -10 })} data-cursor="SCROLL">
                <ScrambleText text="View Projects" trigger="none" hover />
                <ArrowDown className="h-4 w-4" />
              </button>
              <a className="btn-ice btn-ice--solid" href={profile.cv} download data-cursor="PDF">
                <ScrambleText text="Download CV" trigger="none" hover />
                <ArrowUpRight className="h-4 w-4" />
              </a>
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
