import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type AnchorHTMLAttributes,
  type MouseEvent,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { nav, sceneState, triggerPulse } from "@/lib/store";
import { getQuality } from "@/lib/device";
import { HOME_Y, save } from "@/lib/session";
import { wipeGrid } from "@/lib/utils";

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
 * `go(to, state?)`: navigate behind the page-wipe transition. A request made mid-transition is
 * queued, the latest one winning, and plays as soon as the transition ends.
 */
export const usePageTransition = () => useContext(TransitionCtx);

/** Chunky stepped growth, like an old console's screen wipe. */
const EASE = "steps(4, end)";
/** Delay between one diagonal of blocks and the next, ms. */
const STAGGER = 24;

/**
 * Fills the screen with blocks in a diagonal sweep from the top left, swaps the route underneath,
 * then clears them in the same order. With reduced motion or without the Web Animations API it
 * navigates straight away.
 */
export function TransitionProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const overlayRef = useRef<HTMLDivElement>(null);
  const busy = useRef(false);
  /** Navigation requested mid-transition; plays when the current one finishes */
  const queued = useRef<{ to: string; state?: unknown } | null>(null);
  const quality = getQuality();
  const speed = quality.tier === "low" ? 0.7 : 1;
  const pathRef = useRef(location.pathname);
  pathRef.current = location.pathname;

  const [grid, setGrid] = useState(() => wipeGrid(quality.wipeCols, window.innerWidth, window.innerHeight));
  useEffect(() => {
    let t = 0;
    const onResize = () => {
      window.clearTimeout(t);
      t = window.setTimeout(() => setGrid(wipeGrid(quality.wipeCols, window.innerWidth, window.innerHeight)), 200);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("resize", onResize);
    };
  }, [quality.wipeCols]);

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
      const cells = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__cell"));
      const deco = Array.from(el.querySelectorAll<HTMLElement>(".page-wipe__label"));
      const diag = (c: HTMLElement) => Number(c.dataset.d ?? 0);
      const longest = cells.reduce((m, c) => Math.max(m, diag(c)), 0);
      const sweep = (from: string, to: string, duration: number) =>
        within(
          Promise.all(
            cells.map(
              (c) =>
                c.animate([{ transform: from }, { transform: to }], {
                  duration,
                  delay: diag(c) * STAGGER,
                  easing: EASE,
                  fill: "forwards",
                }).finished
            )
          ),
          duration + longest * STAGGER + 300
        );
      const fade = (from: number, to: number) =>
        deco.forEach((d) => d.animate([{ opacity: from }, { opacity: to }], { duration: 200, fill: "forwards" }));
      try {
        // Blocks pop in over the page…
        await sweep("scale(0)", "scale(1.02)", 300 * speed);
        fade(0, 1);
        navigate(to, { state });
        focusMain();
        await within(new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))), 300);
        await new Promise((r) => setTimeout(r, 160));
        fade(1, 0);
        // …and pop out again to reveal the new route
        await sweep("scale(1.02)", "scale(0)", 300 * speed);
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

  const cells = [];
  for (let r = 0; r < grid.rows; r++)
    for (let c = 0; c < grid.cols; c++)
      cells.push(<div key={`${c}-${r}`} className="page-wipe__cell" data-d={c + r} data-t={(c * 7 + r * 3) % 3} />);

  return (
    <TransitionCtx.Provider value={go}>
      {children}
      <div
        ref={overlayRef}
        className="page-wipe"
        aria-hidden="true"
        style={{ gridTemplateColumns: `repeat(${grid.cols}, 1fr)`, gridTemplateRows: `repeat(${grid.rows}, 1fr)` }}
      >
        {cells}
        <div className="page-wipe__label">
          <span>Next stage</span>
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

/** A real `<a href="#/...">`, so open-in-new-tab works, that plays the page wipe on a plain click. */
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
