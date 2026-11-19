import { useEffect, useRef, useState } from "react";
import { sceneState, triggerPulse, ui } from "@/lib/store";
import { prefersReducedMotion, setScrollLocked } from "@/lib/smoothScroll";
import { profile } from "@/data/profile";
import profilePhoto from "@/assets/profile_photo.jpg";

/** Boot log lines and the progress (0..1) at which each one appears. */
const LOG = [
  { at: 0, text: "BOOTING RENDERER" },
  { at: 0.16, text: "RAISING THE VILLAGE" },
  { at: 0.34, text: "COMPILING SHADERS" },
  { at: 0.52, text: "INDEXING PROJECTS" },
  { at: 0.72, text: "CALIBRATING CAMERA" },
  { at: 0.97, text: "READY" },
];

/** Boot screen: tracks real readiness (fonts, WebGL, portrait) with a minimum show time. */
export default function Loader() {
  const [gone, setGone] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const countRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const [lines, setLines] = useState(1);
  const [blocks, setBlocks] = useState(0);

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
      raf = requestAnimationFrame(loop);
      if (finished) return;
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
      if (barRef.current) barRef.current.style.transform = `scaleX(${shown})`;
      const n = LOG.filter((l) => shown >= l.at).length;
      if (n !== lastLines) {
        lastLines = n;
        setLines(n);
        setBlocks(sceneState.blocks);
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
      <div className="loader__panel loader__panel--top" />
      <div className="loader__panel loader__panel--bottom" />
      <div className="loader__inner">
        <div className="loader__row">
          <span>[ AK ] — Portfolio</span>
        </div>
        <div className="loader__center">
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
                {l.text === "RAISING THE VILLAGE" && blocks ? ` · ${blocks.toLocaleString("en-GB")} CUBES` : ""}
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
