import { createElement, Fragment, useCallback, useEffect, useMemo, useRef, type ElementType } from "react";
import { cn } from "@/lib/utils";
import { onAppReady } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*+=<>/\\|[]{}_-~^$@!?";
const randGlyph = () => GLYPHS[(Math.random() * GLYPHS.length) | 0];

interface Props {
  text: string;
  as?: ElementType;
  className?: string;
  /** "view": decode when scrolled into view · "mount": decode right after the loader · "none": static */
  trigger?: "view" | "mount" | "none";
  /** Re-scramble on hover or focus of the closest link, button or [data-scramble-host], else of the element itself. */
  hover?: boolean;
  /** Wait before the first decode starts, ms. */
  delay?: number;
  /** Total time for the decode, ms. */
  duration?: number;
}

/**
 * Decoding "scramble" text. Every character is its own inline-block holding the final glyph, hidden
 * while scrambling, so nothing reflows. Screen readers get the plain text.
 */
export default function ScrambleText({
  text,
  as = "span",
  className,
  trigger = "view",
  hover = false,
  delay = 0,
  duration = 900,
}: Props) {
  const rootRef = useRef<HTMLElement | null>(null);
  const rafRef = useRef(0);
  const words = useMemo(() => text.split(" "), [text]);

  const chars = () => Array.from(rootRef.current?.querySelectorAll<HTMLSpanElement>("[data-c]") ?? []);

  const run = useCallback((opts: { fromHidden: boolean; dur: number; delay: number }) => {
    const els = chars();
    if (!els.length) return;
    cancelAnimationFrame(rafRef.current);
    const n = els.length;
    const t0 = performance.now() + opts.delay;
    const spread = opts.dur * 0.55;
    const plan = els.map((_, i) => {
      const start = (i / Math.max(1, n - 1)) * spread;
      return { start, end: start + opts.dur * 0.25 + Math.random() * opts.dur * 0.3 };
    });
    if (opts.fromHidden) els.forEach((el) => el.setAttribute("data-h", ""));
    const step = (now: number) => {
      const t = now - t0;
      let done = 0;
      // Random glyphs change every 55 ms, not every frame
      const tick = Math.floor(now / 55);
      for (let i = 0; i < n; i++) {
        const el = els[i];
        const p = plan[i];
        if (t < p.start) continue;
        if (t >= p.end) {
          if (el.hasAttribute("data-g") || el.hasAttribute("data-h")) {
            el.removeAttribute("data-g");
            el.removeAttribute("data-h");
          }
          done++;
          continue;
        }
        el.removeAttribute("data-h");
        if (el.dataset.k !== String(tick)) {
          el.dataset.k = String(tick);
          el.setAttribute("data-g", randGlyph());
        }
      }
      if (done < n) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
  }, []);

  // Initial reveal
  useEffect(() => {
    const el = rootRef.current;
    if (!el || trigger === "none" || prefersReducedMotion()) return;
    chars().forEach((c) => c.setAttribute("data-h", ""));
    let io: IntersectionObserver | null = null;
    const unsub = onAppReady(() => {
      if (trigger === "mount") {
        run({ fromHidden: true, dur: duration, delay });
        return;
      }
      io = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            io?.disconnect();
            run({ fromHidden: true, dur: duration, delay });
          }
        },
        { threshold: 0.15, rootMargin: "0px 0px -8% 0px" }
      );
      io.observe(el);
    });
    return () => {
      unsub();
      io?.disconnect();
      cancelAnimationFrame(rafRef.current);
      chars().forEach((c) => {
        c.removeAttribute("data-h");
        c.removeAttribute("data-g");
      });
    };
  }, [text, trigger, delay, duration, run]);

  // Hover re-scramble
  useEffect(() => {
    const el = rootRef.current;
    if (!el || !hover || prefersReducedMotion()) return;
    const host = (el.closest("a,button,[data-scramble-host]") as HTMLElement | null) ?? el;
    const onEnter = () => {
      run({ fromHidden: false, dur: Math.min(520, 160 + text.length * 18), delay: 0 });
    };
    host.addEventListener("mouseenter", onEnter);
    host.addEventListener("focus", onEnter);
    return () => {
      host.removeEventListener("mouseenter", onEnter);
      host.removeEventListener("focus", onEnter);
    };
  }, [hover, run, text]);

  return createElement(
    as,
    { ref: rootRef, className: cn("scramble", className) },
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {words.map((w, wi) => (
          <Fragment key={wi}>
            <span className="sc-word">
              {Array.from(w).map((ch, ci) => (
                <span key={ci} data-c="" className="sc-ch">
                  {ch}
                </span>
              ))}
            </span>
            {wi < words.length - 1 ? " " : null}
          </Fragment>
        ))}
      </span>
    </>
  );
}
