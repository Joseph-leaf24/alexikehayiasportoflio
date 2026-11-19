import { useRef, type CSSProperties, type PointerEvent } from "react";
import { cn } from "@/lib/utils";

interface Props {
  src: string;
  alt: string;
  className?: string;
  /** "cover" crops to fill; "contain" keeps the whole picture and fills the space behind it */
  fit?: "cover" | "contain";
  /** Centre of the part that pops forward, in % of the image */
  focus?: [number, number];
  /** Load eagerly (above-the-fold covers) */
  eager?: boolean;
}

const PARTICLES = 7;

/**
 * Turns a flat picture into a small 3D scene: a blurred backdrop sits behind it, a
 * sharpened focal region floats in front, and a HUD layer hovers above. The stage
 * drifts on its own and tilts toward the pointer, so the layers parallax.
 */
export default function DepthImage({ src, alt, className, fit = "cover", focus = [50, 45], eager = false }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el || e.pointerType !== "mouse") return;
    const r = el.getBoundingClientRect();
    const x = r.width ? (e.clientX - r.left) / r.width : 0.5;
    const y = r.height ? (e.clientY - r.top) / r.height : 0.5;
    el.style.setProperty("--dx", `${(0.5 - y) * 10}deg`);
    el.style.setProperty("--dy", `${(x - 0.5) * 14}deg`);
    el.style.setProperty("--lx", `${x * 100}%`);
  };
  const onLeave = () => {
    const el = ref.current;
    if (!el) return;
    el.style.setProperty("--dx", "0deg");
    el.style.setProperty("--dy", "0deg");
  };

  const style = { "--fx": `${focus[0]}%`, "--fy": `${focus[1]}%` } as CSSProperties;
  const loading = eager ? "eager" : "lazy";

  return (
    <div ref={ref} className={cn("depth", `depth--${fit}`, className)} style={style} onPointerMove={onMove} onPointerLeave={onLeave}>
      <div className="depth__stage">
        <div className="depth__drift">
          <img className="depth__back" src={src} alt="" aria-hidden="true" loading={loading} decoding="async" />
          <img className="depth__mid" src={src} alt={alt} loading={loading} decoding="async" />
          {fit === "cover" ? (
            <img className="depth__front" src={src} alt="" aria-hidden="true" loading={loading} decoding="async" />
          ) : null}
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
