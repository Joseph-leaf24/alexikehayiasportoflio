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
import { nav, sceneState, triggerPulse } from "@/lib/store";
import { getQuality } from "@/lib/device";
import { HOME_Y, save } from "@/lib/session";

type Go = (to: string, state?: unknown) => void;
const TransitionCtx = createContext<Go>(() => {});

/**
 * Settle `p` after `ms` at the latest. The wipe's animations only advance as the browser
 * draws frames, so on a device that manages about one frame a second they could hold
 * navigation up for many seconds; this keeps the wipe on schedule regardless.
 */
const within = (p: Promise<unknown>, ms: number) => Promise.race([p, new Promise((r) => setTimeout(r, ms))]);

/** After a route change the link that was clicked is gone: move focus to the new page's content. */
const focusMain = () => document.getElementById("main")?.focus({ preventScroll: true });

/**
 * `go(to, state?)`: navigate behind the page-wipe transition. A request made while a transition
 * is running is queued and plays as soon as it ends; only the latest is kept.
 */
export const usePageTransition = () => useContext(TransitionCtx);

const EASE = "cubic-bezier(.76,0,.24,1)";

/**
 * Wipes slanted ice sheets over the page, swaps the route underneath, then wipes them away.
 * With reduced motion, or without the Web Animations API, it navigates straight away.
 */
export function TransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const overlayRef = useRef<HTMLDivElement>(null);
  const busy = useRef(false);
  /** Navigation requested mid-transition; plays when the current one finishes */
  const queued = useRef<{ to: string; state?: unknown } | null>(null);
  const quality = getQuality();
  const shardCount = quality.shards;
  const speed = quality.tier === "low" ? 0.7 : 1;
  const pathRef = useRef(location.pathname);
  pathRef.current = location.pathname;

  const go = useCallback<Go>(
    async (to, state) => {
      nav.inApp = true;
      if (busy.current) {
        queued.current = { to, state };
        return;
      }
      // Remember where we were on the home page so "back" returns to the same spot
      if (pathRef.current === "/") save(HOME_Y, window.scrollY);
      const el = overlayRef.current;
      triggerPulse();
      if (!el || prefersReducedMotion() || typeof el.animate !== "function") {
        navigate(to, { state });
        focusMain();
        return;
      }
      busy.current = true;
      sceneState.transition = true;
      el.style.visibility = "visible";
      el.classList.add("is-active");
      const shards = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__shard"));
      const deco = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__grid, .page-wipe__label"));
      const STAGGER = 45;
      // Each sweep, with a little slack past its last sheet before moving on regardless
      const sweep = (from: string, to: string, duration: number) =>
        within(
          Promise.all(
            shards.map(
              (sh, i) =>
                sh.animate([{ transform: from }, { transform: to }], {
                  duration,
                  delay: i * STAGGER,
                  easing: EASE,
                  fill: "forwards",
                }).finished
            )
          ),
          duration + (shards.length - 1) * STAGGER + 300
        );
      const fade = (from: number, to: number) =>
        deco.forEach((d) => d.animate([{ opacity: from }, { opacity: to }], { duration: 260, fill: "forwards" }));
      try {
        // Ice sheets slide up over the page…
        await sweep("translateY(110%) skewY(-7deg)", "translateY(0) skewY(0deg)", 620 * speed);
        fade(0, 1);
        navigate(to, { state });
        focusMain();
        await within(new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))), 300);
        await new Promise((r) => setTimeout(r, 140));
        fade(1, 0);
        // …and carry on upwards to reveal the new route
        await sweep("translateY(0) skewY(0deg)", "translateY(-110%) skewY(7deg)", 720 * speed);
      } finally {
        el.style.visibility = "hidden";
        el.classList.remove("is-active");
        busy.current = false;
        sceneState.transition = false;
        const next = queued.current;
        queued.current = null;
        if (next) goRef.current(next.to, next.state);
      }
    },
    [navigate, speed]
  );
  const goRef = useRef(go);
  goRef.current = go;

  return (
    <TransitionCtx.Provider value={go}>
      {children}
      <div ref={overlayRef} className="page-wipe" aria-hidden="true">
        {Array.from({ length: shardCount }, (_, i) => (
          <div
            key={i}
            className="page-wipe__shard"
            style={{ left: `${(i * 100) / shardCount}%`, width: `${100 / shardCount + 0.5}%` }}
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

/** A link that plays the page wipe. It is a real `<a href="#/...">`, so open-in-new-tab still works. */
export const TLink = forwardRef<HTMLAnchorElement, TLinkProps>(function TLink(
  { to, state, onClick, children, ...rest },
  ref
) {
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
