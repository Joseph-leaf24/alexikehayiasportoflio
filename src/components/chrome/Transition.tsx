import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useRef,
  type AnchorHTMLAttributes,
  type MouseEvent,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { sound } from "@/lib/sound";
import { triggerPulse } from "@/lib/store";

type Go = (to: string, state?: unknown) => void;
const TransitionCtx = createContext<Go>(() => {});

export const usePageTransition = () => useContext(TransitionCtx);

const EASE = "cubic-bezier(.76,0,.24,1)";
const SHARDS = 7;

/** Wipes an icy panel over the page, swaps the route underneath, then wipes it away. */
export function TransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const overlayRef = useRef<HTMLDivElement>(null);
  const busy = useRef(false);
  const pathRef = useRef(location.pathname);
  pathRef.current = location.pathname;

  const go = useCallback<Go>(
    async (to, state) => {
      if (busy.current) return;
      // Remember where we were on the home page so "back" returns to the same spot
      if (pathRef.current === "/") {
        try {
          sessionStorage.setItem("ak:homeY", String(window.scrollY));
        } catch {
          /* ignore */
        }
      }
      const el = overlayRef.current;
      sound.blip("click");
      triggerPulse();
      if (!el || prefersReducedMotion() || typeof el.animate !== "function") {
        navigate(to, { state });
        return;
      }
      busy.current = true;
      el.style.visibility = "visible";
      const shards = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__shard"));
      const deco = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__grid, .page-wipe__label"));
      const sweep = (from: string, to: string, duration: number) =>
        Promise.all(
          shards.map(
            (sh, i) =>
              sh.animate([{ transform: from }, { transform: to }], { duration, delay: i * 45, easing: EASE, fill: "forwards" }).finished
          )
        );
      const fade = (from: number, to: number) =>
        deco.forEach((d) => d.animate([{ opacity: from }, { opacity: to }], { duration: 260, fill: "forwards" }));
      try {
        // Ice sheets slide up over the page…
        await sweep("translateY(110%) skewY(-7deg)", "translateY(0) skewY(0deg)", 620);
        fade(0, 1);
        navigate(to, { state });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        await new Promise((r) => setTimeout(r, 140));
        fade(1, 0);
        // …and carry on upwards to reveal the new route
        await sweep("translateY(0) skewY(0deg)", "translateY(-110%) skewY(7deg)", 720);
      } finally {
        el.style.visibility = "hidden";
        busy.current = false;
      }
    },
    [navigate]
  );

  return (
    <TransitionCtx.Provider value={go}>
      {children}
      <div ref={overlayRef} className="page-wipe" aria-hidden="true">
        {Array.from({ length: SHARDS }, (_, i) => (
          <div
            key={i}
            className="page-wipe__shard"
            style={{ left: `${(i * 100) / SHARDS}%`, width: `${100 / SHARDS + 0.5}%` }}
          />
        ))}
        <div className="page-wipe__grid" />
        <div className="page-wipe__label">
          <span>// TRANSMITTING</span>
          <span className="page-wipe__bar">
            <i />
          </span>
        </div>
      </div>
    </TransitionCtx.Provider>
  );
}

interface TLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: string;
  state?: unknown;
  children: ReactNode;
}

/** A real `<a href="#/...">` (so open-in-new-tab works) that plays the page wipe. */
export const TLink = forwardRef<HTMLAnchorElement, TLinkProps>(function TLink({ to, state, onClick, children, ...rest }, ref) {
  const go = usePageTransition();
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    go(to, state);
  };
  return (
    <a ref={ref} href={`#${to}`} onClick={handle} {...rest}>
      {children}
    </a>
  );
});
