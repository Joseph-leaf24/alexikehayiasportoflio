import { useEffect, useRef, useState } from "react";
import { sceneState, triggerPulse, ui } from "@/lib/store";
import { prefersReducedMotion, setScrollLocked } from "@/lib/smoothScroll";
import { profile } from "@/data/profile";
import profilePhoto from "@/assets/profile_photo.jpg";

/** Boot log lines and the progress (0..1) at which each one appears. */
const LOG = [
  { at: 0, text: "LOADING LEVEL 1" },
  { at: 0.16, text: "SPAWNING SQUAD · 4 PLAYERS" },
  { at: 0.34, text: "ARMING EXPLOSIVES" },
  { at: 0.52, text: "BRIEFING MISSIONS" },
  { at: 0.72, text: "LOCK AND LOAD" },
  { at: 0.97, text: "READY" },
];

/** Boot screen: tracks real readiness (fonts, the level, portrait) with a minimum show time. */
export default function Loader() {
  const [gone, setGone] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const countRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const [lines, setLines] = useState(1);

  useEffect(() => {
    const reduced = prefersReducedMotion();
    const minDur = reduced ? 250 : 2000;
    const t0 = performance.now();
    let fonts = false;
    let img = false;
    let shown = 0;
    let raf = 0;
    let finished = false;
    const timers: number[] = [];
    setScrollLocked(true);

    (document.fonts?.ready ?? Promise.resolve()).then(() => (fonts = true)).catch(() => (fonts = true));
    const im = new Image();
    im.onload = im.onerror = () => (img = true);
    im.src = profilePhoto;

    const finish = () => {
      finished = true;
      rootRef.current?.classList.add("is-done");
      setScrollLocked(false);
      timers.push(
        window.setTimeout(() => {
          ui.set({ ready: true });
          triggerPulse();
        }, reduced ? 0 : 380),
        window.setTimeout(() => setGone(true), reduced ? 300 : 1500)
      );
    };

    let lastLines = 1;
    let lastT = t0;
    const loop = () => {
      // (stops for good once finished: the loader stays mounted, hidden, for the whole visit)
      if (finished) return;
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min(0.25, (now - lastT) / 1000);
      lastT = now;
      const elapsed = now - t0;
      const timeout = elapsed > 7000; // never hang the visitor
      const real = timeout ? 1 : (fonts ? 0.3 : 0) + (sceneState.ready ? 0.5 : 0) + (img ? 0.2 : 0);
      const target = Math.min(real, elapsed / minDur);
      // Time-based easing so slow devices don't crawl
      shown += (target - shown) * (1 - Math.exp(-dt * 6)) + dt * 0.15;
      shown = Math.min(shown, target);
      const pct = Math.round(shown * 100);
      if (countRef.current) countRef.current.textContent = String(pct).padStart(3, "0");
      // The bar fills in 20 chunky steps
      if (barRef.current) barRef.current.style.transform = `scaleX(${Math.floor(shown * 20) / 20})`;
      const n = LOG.filter((l) => shown >= l.at).length;
      if (n !== lastLines) {
        lastLines = n;
        setLines(n);
      }
      if (real >= 1 && shown >= 0.995) finish();
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      timers.forEach((t) => window.clearTimeout(t));
      if (!finished) setScrollLocked(false);
    };
  }, []);

  if (gone) return null;
  return (
    <div ref={rootRef} className="loader" role="status" aria-live="polite" aria-label="Loading portfolio">
      {/* Screen readers hear this once; the counter and boot log below change every frame */}
      <span className="sr-only">Loading portfolio…</span>
      <div className="loader__panel loader__panel--top" />
      <div className="loader__panel loader__panel--bottom" />
      <div className="loader__inner" aria-hidden="true">
        <div className="loader__row">
          <span>[ AK ] — Portfolio</span>
        </div>
        <div className="loader__center">
          <span className="loader__title">Press start</span>
          <span ref={countRef} className="loader__count">
            000
          </span>
          <span className="loader__bar">
            <span ref={barRef} />
          </span>
          <ul className="loader__log">
            {LOG.slice(0, lines).map((l, i) => (
              <li key={l.text} className={i === lines - 1 ? "is-current" : undefined}>
                <span>&gt;</span> {l.text}
              </li>
            ))}
          </ul>
        </div>
        <div className="loader__row">
          <span>{profile.focus.join(" · ")}</span>
          <span>{new Date().getFullYear()}</span>
        </div>
      </div>
    </div>
  );
}
