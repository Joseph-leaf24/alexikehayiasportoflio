import { useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import { cn } from "@/lib/utils";

interface Props {
  src: string;
  alt: string;
  /** "cover" crops to fill; "contain" keeps the whole picture and fills the space behind it */
  fit?: "cover" | "contain";
}

const PARTICLES = 5;

/**
 * Turns a flat picture into a small 3D scene with a HUD layer (grid, glint, particles)
 * hovering above it. The stage drifts on its own and tilts toward the mouse, so the
 * layers parallax. Cover-fit pictures get a sharpened focal region floating in front;
 * contain-fit pictures float whole over a static, blurred copy that fills the frame.
 *
 * Every animation is paused while the picture is off screen (data-live), so a grid
 * of these costs nothing until you scroll to it.
 */
export default function DepthImage({ src, alt, fit = "cover" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting), { rootMargin: "80px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el || e.pointerType !== "mouse") return;
    const r = el.getBoundingClientRect();
    const x = r.width ? (e.clientX - r.left) / r.width : 0.5;
    const y = r.height ? (e.clientY - r.top) / r.height : 0.5;
    el.style.setProperty("--dx", `${(0.5 - y) * 10}deg`);
    el.style.setProperty("--dy", `${(x - 0.5) * 14}deg`);
  };
  const onLeave = () => {
    const el = ref.current;
    if (!el) return;
    el.style.setProperty("--dx", "0deg");
    el.style.setProperty("--dy", "0deg");
  };

  return (
    <div
      ref={ref}
      className={cn("depth", `depth--${fit}`)}
      data-live={live ? "" : undefined}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
    >
      {fit === "contain" ? <img className="depth__back" src={src} alt="" aria-hidden="true" loading="lazy" decoding="async" /> : null}
      <div className="depth__stage">
        <div className="depth__drift">
          <img className="depth__mid" src={src} alt={alt} loading="lazy" decoding="async" />
          {fit === "cover" ? <img className="depth__front" src={src} alt="" aria-hidden="true" loading="lazy" decoding="async" /> : null}
          <span className="depth__grid" aria-hidden="true" />
          <span className="depth__glint" aria-hidden="true" />
          <span className="depth__particles" aria-hidden="true">
            {Array.from({ length: PARTICLES }, (_, i) => (
              <i key={i} style={{ "--i": i } as CSSProperties} />
            ))}
          </span>
        </div>
      </div>
    </div>
  );
}
