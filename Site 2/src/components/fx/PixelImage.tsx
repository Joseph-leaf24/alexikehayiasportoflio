import { useEffect, useRef } from "react";
import { cn, fitRect } from "@/lib/utils";
import { prefersReducedMotion } from "@/lib/smoothScroll";

interface Props {
  src: string;
  alt: string;
  /** "cover" crops to fill; "contain" keeps the whole picture */
  fit?: "cover" | "contain";
}

/** Block sizes (CSS px) the picture passes through on its way to sharp. */
const REVEAL_STEPS = [28, 16, 9, 5, 0];
const HOVER_STEPS = [7, 4, 0];
const STEP_MS = 85;

/**
 * A picture that "loads in" like an old game: the first time it scrolls into view it
 * resolves from big blocks to sharp in a few steps, and on hover it flickers back through
 * the mosaic. The mosaic is a tiny canvas scaled up with hard edges over the real image,
 * which stays in the page for screen readers and as the fallback (no canvas, reduced motion).
 */
export default function PixelImage({ src, alt, fit = "cover" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const root = ref.current;
    const img = imgRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!root || !img || !canvas || !ctx || prefersReducedMotion()) return;
    let timer = 0;
    let shown = false;

    const draw = (block: number) => {
      if (!block || !img.naturalWidth) {
        root.removeAttribute("data-mosaic");
        return;
      }
      const w = root.clientWidth;
      const h = root.clientHeight;
      if (!w || !h) return;
      canvas.width = Math.max(1, Math.ceil(w / block));
      canvas.height = Math.max(1, Math.ceil(h / block));
      ctx.imageSmoothingEnabled = true;
      const { sx, sy, sw, sh } = fitRect(img.naturalWidth, img.naturalHeight, w, h, fit);
      if (fit === "contain") {
        const s = Math.min(w / sw, h / sh);
        const dw = (sw * s) / block;
        const dh = (sh * s) / block;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, sx, sy, sw, sh, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
      } else ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      root.setAttribute("data-mosaic", "");
    };
    const play = (steps: number[]) => {
      window.clearTimeout(timer);
      let i = 0;
      const next = () => {
        draw(steps[i]);
        if (++i < steps.length) timer = window.setTimeout(next, STEP_MS);
      };
      next();
    };
    const whenLoaded = (cb: () => void) => {
      if (img.complete && img.naturalWidth) cb();
      else img.addEventListener("load", cb, { once: true });
    };

    // Hidden behind the first mosaic step until it scrolls into view
    whenLoaded(() => {
      if (!shown) draw(REVEAL_STEPS[0]);
    });
    const io = new IntersectionObserver(
      ([e]) => {
        if (!e.isIntersecting || shown) return;
        shown = true;
        io.disconnect();
        whenLoaded(() => play(REVEAL_STEPS));
      },
      { rootMargin: "0px 0px -10% 0px" }
    );
    io.observe(root);
    const host = root.closest<HTMLElement>("a, button, article") ?? root;
    const onEnter = () => shown && whenLoaded(() => play(HOVER_STEPS));
    host.addEventListener("mouseenter", onEnter);
    return () => {
      window.clearTimeout(timer);
      io.disconnect();
      host.removeEventListener("mouseenter", onEnter);
    };
  }, [src, fit]);

  return (
    <div ref={ref} className={cn("pximg", `pximg--${fit}`)}>
      <img ref={imgRef} src={src} alt={alt} loading="lazy" decoding="async" />
      <canvas ref={canvasRef} className="pximg__mosaic" aria-hidden="true" />
    </div>
  );
}
