import { useEffect, useRef } from "react";
import { gsap, prefersReducedMotion } from "@/lib/smoothScroll";
import ScrambleText from "./ScrambleText";
import Reveal from "./Reveal";

/** `[ 01 ] ———————— WHO I AM` plus a giant outlined index drifting behind the section. */
export default function SectionHead({ index, label }: { index: string; label: string }) {
  const ghostRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ghostRef.current;
    if (!el || prefersReducedMotion()) return;
    const section = el.closest("section") ?? el.parentElement;
    const tween = gsap.fromTo(
      el,
      { yPercent: 30 },
      {
        yPercent: -30,
        ease: "none",
        scrollTrigger: { trigger: section, start: "top bottom", end: "bottom top", scrub: true },
      }
    );
    return () => {
      tween.scrollTrigger?.kill();
      tween.kill();
    };
  }, []);

  return (
    <div className="section-head">
      <span ref={ghostRef} className="ghost-num" aria-hidden="true">
        {index}
      </span>
      <span className="section-head__idx">[ {index} ]</span>
      <Reveal variant="line" className="section-head__rule" />
      <ScrambleText text={label} className="section-head__lbl" hover={false} />
    </div>
  );
}
