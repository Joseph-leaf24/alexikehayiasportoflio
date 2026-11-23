import { createElement, useEffect, useRef, type CSSProperties, type ElementType, type HTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { onAppReady } from "@/lib/store";

interface Props extends HTMLAttributes<HTMLElement> {
  children?: ReactNode;
  as?: ElementType;
  /** up: rise + fade in · fade · clip: wipe from the bottom · line: draw horizontally */
  variant?: "up" | "fade" | "clip" | "line";
  /** Transition delay, ms */
  delay?: number;
  [dataAttr: `data-${string}`]: string | undefined;
}

/** Adds `.is-in` once the element scrolls into view (after the loader). CSS does the rest. */
export default function Reveal({ children, as = "div", className, variant = "up", delay = 0, style, ...rest }: Props) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let io: IntersectionObserver | null = null;
    // Chromium treats an element fully hidden by its own clip-path as "not intersecting",
    // so clip reveals watch their parent instead.
    const clip = variant === "clip";
    const target = clip ? el.parentElement ?? el : el;
    const unsub = onAppReady(() => {
      io = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            el.classList.add("is-in");
            io?.disconnect();
          }
        },
        { threshold: clip ? 0 : 0.12, rootMargin: clip ? "0px 0px -12% 0px" : "0px 0px -6% 0px" }
      );
      io.observe(target);
    });
    return () => {
      unsub();
      io?.disconnect();
    };
  }, [variant]);

  return createElement(
    as,
    {
      ...rest,
      ref,
      className: cn("reveal", `reveal-${variant}`, className),
      style: { ...style, "--d": `${delay}ms` } as CSSProperties,
    },
    children
  );
}
