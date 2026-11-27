import { useEffect, useRef, useState } from "react";

/** The element an event happened on, or null when it targets the window or a text node. */
const elementOf = (t: EventTarget | null) => (t instanceof Element ? t : null);

/** Controls that lean toward the pointer while hovered; the ring snaps to them. */
const MAGNETIC = ".btn-ice, .hud-nav a, .back-link, .site-footer__top, .hud-burger, .hud-brand, [data-magnetic]";

/** Dot + lagging ring cursor with contextual labels (`data-cursor="OPEN"`) and magnetic controls. Mouse only. */
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
    let magnet: HTMLElement | null = null;
    const stick = { x: 0, y: 0 };
    // The pull's transform transition is appended to the control's own transitions, so its colour
    // fill keeps animating, and taken off again once the control has settled back
    const own = new WeakMap<HTMLElement, string>();
    const glide = (el: HTMLElement, seconds: number) => {
      if (!own.has(el)) own.set(el, getComputedStyle(el).transition);
      const base = own.get(el);
      el.style.transition = `${base ? `${base}, ` : ""}transform ${seconds}s cubic-bezier(0.16, 1, 0.3, 1)`;
    };
    const release = (el: HTMLElement) => {
      glide(el, 0.7);
      el.style.transform = "";
      window.setTimeout(() => {
        if (el !== magnet) el.style.transition = "";
      }, 700);
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
        glide(magnet, 0.25);
        magnet.style.transform = `translate3d(${dx}px, ${dy}px, 0)`;
        stick.x = cx + dx;
        stick.y = cy + dy;
      }
      kick();
    };
    // Scrolling moves the control away from under the pointer: let it go
    const onScroll = () => {
      if (!magnet) return;
      release(magnet);
      magnet = null;
      kick();
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

    // Runs only while the ring is catching up: it stops once settled and the next move restarts it
    const drawn = { x: NaN, y: NaN, rx: NaN, ry: NaN };
    const loop = () => {
      raf = 0;
      const tx = magnet ? stick.x + (pos.x - stick.x) * 0.3 : pos.x;
      const ty = magnet ? stick.y + (pos.y - stick.y) * 0.3 : pos.y;
      ring.x += (tx - ring.x) * 0.18;
      ring.y += (ty - ring.y) * 0.18;
      if (
        pos.x === drawn.x &&
        pos.y === drawn.y &&
        Math.abs(ring.x - drawn.rx) < 0.05 &&
        Math.abs(ring.y - drawn.ry) < 0.05
      )
        return;
      drawn.x = pos.x;
      drawn.y = pos.y;
      drawn.rx = ring.x;
      drawn.ry = ring.y;
      if (dotRef.current) dotRef.current.style.transform = `translate3d(${pos.x}px, ${pos.y}px, 0)`;
      if (ringRef.current) ringRef.current.style.transform = `translate3d(${ring.x}px, ${ring.y}px, 0)`;
      raf = requestAnimationFrame(loop);
    };
    function kick() {
      if (!raf) raf = requestAnimationFrame(loop);
    }

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("mouseover", onOver, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });
    window.addEventListener("pointerup", onUp, { passive: true });
    document.documentElement.addEventListener("mouseleave", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      if (magnet) release(magnet);
      document.documentElement.classList.remove("has-cursor");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("scroll", onScroll);
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
