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
      try {
        await el.animate([{ clipPath: "inset(100% 0 0 0)" }, { clipPath: "inset(0 0 0 0)" }], {
          duration: 560,
          easing: EASE,
          fill: "forwards",
        }).finished;
        navigate(to, { state });
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        await el.animate([{ clipPath: "inset(0 0 0 0)" }, { clipPath: "inset(0 0 100% 0)" }], {
          duration: 680,
          easing: EASE,
          fill: "forwards",
        }).finished;
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
