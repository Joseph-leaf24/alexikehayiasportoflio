import { useEffect, useRef } from "react";
import { onAppReady } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { pad } from "@/lib/utils";

/** Counts up from 00 to `value` the first time it scrolls into view. */
export default function Counter({ value, className }: { value: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (prefersReducedMotion()) {
      el.textContent = pad(value);
      return;
    }
    el.textContent = "00";
    let raf = 0;
    let io: IntersectionObserver | null = null;
    const run = () => {
      const t0 = performance.now();
      const dur = 1500;
      const step = (now: number) => {
        const p = Math.min(1, (now - t0) / dur);
        const eased = 1 - Math.pow(2, -10 * p);
        el.textContent = pad(Math.round(eased * value));
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    const unsub = onAppReady(() => {
      io = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            io?.disconnect();
            run();
          }
        },
        { threshold: 0.4 }
      );
      io.observe(el);
    });
    return () => {
      unsub();
      io?.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [value]);

  // Screen readers get the final number from the sr-only copy: most ignore aria-label on a plain span
  return (
    <span className={className}>
      <span ref={ref} aria-hidden="true">
        {pad(value)}
      </span>
      <span className="sr-only">{value}</span>
    </span>
  );
}
