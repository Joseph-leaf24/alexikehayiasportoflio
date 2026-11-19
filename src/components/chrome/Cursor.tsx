import { useEffect, useRef, useState } from "react";

/** The element an event happened on (events can also target window or text nodes). */
const elementOf = (t: EventTarget | null) => (t instanceof Element ? t : null);

const MAGNETIC = ".btn-ice, .hud-nav a, .back-link, .site-footer__top, .hud-burger, .hud-brand, [data-magnetic]";

/** Dot + lagging ring cursor with contextual labels (`data-cursor="OPEN"`). Mouse only. */
export default function Cursor() {
  const [enabled] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(hover: hover) and (pointer: fine)").matches
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!enabled) return;
    const root = rootRef.current!;
    document.documentElement.classList.add("has-cursor");
    const pos = { x: -100, y: -100 };
    const ring = { x: -100, y: -100 };
    let raf = 0;
    let visible = false;
    // Magnetic controls: they lean toward the pointer and the ring sticks to them
    let magnet: HTMLElement | null = null;
    const stick = { x: 0, y: 0 };
    const release = (el: HTMLElement) => {
      el.style.transition = "transform 0.7s cubic-bezier(0.16, 1, 0.3, 1)";
      el.style.transform = "";
    };

    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      pos.x = e.clientX;
      pos.y = e.clientY;
      if (!visible) {
        visible = true;
        ring.x = pos.x;
        ring.y = pos.y;
        root.classList.add("is-visible");
      }
      const next = elementOf(e.target)?.closest<HTMLElement>(MAGNETIC) ?? null;
      if (next !== magnet) {
        if (magnet) release(magnet);
        magnet = next;
      }
      if (magnet) {
        const r = magnet.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const dx = Math.max(-14, Math.min(14, (pos.x - cx) * 0.28));
        const dy = Math.max(-10, Math.min(10, (pos.y - cy) * 0.38));
        magnet.style.transition = "transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)";
        magnet.style.transform = `translate3d(${dx}px, ${dy}px, 0)`;
        stick.x = cx + dx;
        stick.y = cy + dy;
      }
    };
    const onOver = (e: MouseEvent) => {
      const t = elementOf(e.target)?.closest<HTMLElement>("[data-cursor], a, button, [role='tab'], label");
      root.classList.toggle("is-hover", !!t);
      const label = t?.dataset.cursor ?? "";
      if (labelRef.current && labelRef.current.textContent !== label) labelRef.current.textContent = label;
      root.classList.toggle("has-label", !!label);
    };
    const onDown = () => root.classList.add("is-down");
    const onUp = () => root.classList.remove("is-down");
    const onLeave = () => {
      visible = false;
      root.classList.remove("is-visible");
      if (magnet) release(magnet);
      magnet = null;
    };

    const loop = () => {
      raf = requestAnimationFrame(loop);
      const tx = magnet ? stick.x + (pos.x - stick.x) * 0.3 : pos.x;
      const ty = magnet ? stick.y + (pos.y - stick.y) * 0.3 : pos.y;
      ring.x += (tx - ring.x) * 0.18;
      ring.y += (ty - ring.y) * 0.18;
      if (dotRef.current) dotRef.current.style.transform = `translate3d(${pos.x}px, ${pos.y}px, 0)`;
      if (ringRef.current) ringRef.current.style.transform = `translate3d(${ring.x}px, ${ring.y}px, 0)`;
    };
    raf = requestAnimationFrame(loop);

    window.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("mouseover", onOver, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    document.documentElement.addEventListener("mouseleave", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      if (magnet) release(magnet);
      document.documentElement.classList.remove("has-cursor");
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("mouseover", onOver);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      document.documentElement.removeEventListener("mouseleave", onLeave);
    };
  }, [enabled]);

  if (!enabled) return null;
  return (
    <div ref={rootRef} className="cursor" aria-hidden="true">
      <div ref={ringRef} className="cursor__ring">
        <span ref={labelRef} className="cursor__label" />
      </div>
      <div ref={dotRef} className="cursor__dot" />
    </div>
  );
}
