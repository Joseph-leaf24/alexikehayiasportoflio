import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { sceneState, ui, useStore, type LayoutKey } from "@/lib/store";
import { scrollToTarget, setScrollLocked } from "@/lib/smoothScroll";
import { profile } from "@/data/profile";
import { cn } from "@/lib/utils";
import { trapTab } from "@/lib/keys";
import ScrambleText from "@/components/fx/ScrambleText";
import { usePageTransition } from "./Transition";

const NAV = [
  { id: "about", label: "About" },
  { id: "domains", label: "Domains" },
  { id: "toolbox", label: "Toolbox" },
  { id: "projects", label: "Work" },
  { id: "contact", label: "Contact" },
];

/** Local time at the profile's time zone, ticking every second. */
function Clock() {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const fmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: profile.timezone,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      // "CET" in winter, "CEST" in summer
      timeZoneName: "short",
    });
    const update = () => {
      if (ref.current) ref.current.textContent = fmt.format(new Date());
    };
    update();
    const id = window.setInterval(update, 1000);
    return () => window.clearInterval(id);
  }, []);
  return <span ref={ref} className="tabular-nums" />;
}

/** HUD names for the scene's formations. */
const FORMATIONS: Record<LayoutKey, string> = {
  village: "Cycladic village",
  shatter: "Fracture",
  clusters: "Domain clusters",
  network: "Neural net",
  bars: "Data bars",
  villageTop: "Village · aerial",
  cube: "Voxel cube",
  lost: "Drift",
};

/** Fixed-width signed coordinate for the camera readout (typographic minus). */
const signed = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2).padStart(5, "0")}`;

/** Live readout of what the 3D scene is doing, polled about 8 times a second. Hidden without WebGL. */
function Telemetry() {
  const [formation, setFormation] = useState<LayoutKey>(sceneState.formation);
  // The scene only finds out WebGL is missing after this first renders, so keep checking
  const [webgl, setWebgl] = useState(sceneState.webgl);
  const rootRef = useRef<HTMLDivElement>(null);
  const camRef = useRef<HTMLSpanElement>(null);
  const blocksRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let current = sceneState.formation;
    const id = window.setInterval(() => {
      if (!sceneState.webgl) {
        setWebgl(false);
        window.clearInterval(id);
        return;
      }
      // Hidden by CSS below 1280 px wide: nothing to update
      const root = rootRef.current;
      if (!root || getComputedStyle(root).display === "none") return;
      if (sceneState.formation !== current) {
        current = sceneState.formation;
        setFormation(current);
      }
      const c = sceneState.cam;
      if (camRef.current) camRef.current.textContent = `${signed(c.x)} ${signed(c.y)} ${signed(c.z)}`;
      if (blocksRef.current && sceneState.blocks)
        blocksRef.current.textContent = sceneState.blocks.toLocaleString("en-GB");
    }, 120);
    return () => window.clearInterval(id);
  }, []);
  if (!webgl) return null;
  return (
    <div ref={rootRef} className="hud-telemetry">
      <span>Formation</span>
      <ScrambleText
        key={formation}
        text={FORMATIONS[formation]}
        trigger="mount"
        duration={500}
        className="hud-telemetry__val"
      />
      <span className="hud-dot" />
      <span>
        <span ref={blocksRef}>—</span> cubes
      </span>
      <span className="hud-dot" />
      <span>
        Cam <span ref={camRef} className="tabular-nums" />
      </span>
    </div>
  );
}

/** Scroll progress: a vertical bar plus a 000–100 % counter. */
function ScrollMeter() {
  const barRef = useRef<HTMLSpanElement>(null);
  const numRef = useRef<HTMLSpanElement>(null);
  // Redraws only when the page scrolls or resizes; the page height is cached so a
  // scroll never forces a layout read.
  useEffect(() => {
    let raf = 0;
    let max = 0;
    let lastP = -1;
    let lastPct = -1;
    const draw = () => {
      raf = 0;
      const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      if (p === lastP) return;
      lastP = p;
      const pct = Math.round(p * 100);
      if (barRef.current) barRef.current.style.transform = `scaleY(${p})`;
      if (numRef.current && pct !== lastPct) {
        numRef.current.textContent = String(pct).padStart(3, "0");
        lastPct = pct;
      }
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(draw);
    };
    const measure = () => {
      max = document.documentElement.scrollHeight - window.innerHeight;
      schedule();
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(document.body);
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", measure);
    };
  }, []);
  return (
    <div className="scroll-meter" aria-hidden="true">
      <span className="scroll-meter__track">
        <span ref={barRef} className="scroll-meter__bar" />
      </span>
      <span>
        <span ref={numRef}>000</span>%
      </span>
    </div>
  );
}

/**
 * Fixed chrome on every page: frame corners, a top bar whose nav becomes a full-screen menu on
 * small screens, and a bottom bar of clock, telemetry and scroll progress. Off the home page,
 * nav links go home first.
 */
export default function Hud() {
  const { sectionIndex, sectionLabel, menuOpen } = useStore(ui, (s) => s);
  const location = useLocation();
  const go = usePageTransition();
  const onHome = location.pathname === "/";

  const navTo = (id: string) => {
    // From the phone menu, unlock scrolling first: restarting the smooth scroller later, in the
    // menu-closed effect, would cancel the scroll that had just begun
    if (ui.get().menuOpen) setScrollLocked(false);
    ui.set({ menuOpen: false });
    if (onHome) scrollToTarget(id === "top" ? 0 : `#${id}`, { offset: id === "top" ? 0 : -10 });
    else go("/", { scrollTo: id === "top" ? undefined : id });
  };

  const burgerRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setScrollLocked(menuOpen);
    if (!menuOpen) {
      // Closing with focus on one of its now-hidden links would drop focus to the page: hand it back to the burger
      if (document.getElementById("mobile-menu")?.contains(document.activeElement)) burgerRef.current?.focus();
      return;
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") ui.set({ menuOpen: false });
      // The menu covers the page: Tab cycles between its links and the burger, which closes it
      else
        trapTab(e, [burgerRef.current, ...Array.from(document.querySelectorAll<HTMLElement>("#mobile-menu a[href]"))]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  // Close the menu on route change
  useEffect(() => {
    ui.set({ menuOpen: false });
  }, [location.pathname]);

  return (
    <>
      <div className="hud-frame" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>

      <header className="hud-top">
        <a
          href="#/"
          className="hud-brand"
          onClick={(e) => {
            e.preventDefault();
            navTo("top");
          }}
        >
          <span className="hud-brand__mark" aria-hidden="true" />
          <ScrambleText text="Alexi Kehayias" trigger="mount" hover />
          {/* Accessible name: the visible name plus "home" (WCAG 2.5.3, label in name) */}
          <span className="sr-only"> — home</span>
        </a>

        <div className="hud-section" aria-hidden="true">
          <span className="hud-section__idx">{sectionIndex}</span>
          <span className="hud-section__rule" />
          <ScrambleText key={sectionLabel} text={sectionLabel} trigger="mount" duration={600} />
        </div>

        <div className="hud-right">
          <nav className="hud-nav" aria-label="Primary">
            {NAV.map((n) => (
              <a
                key={n.id}
                href="#/"
                onClick={(e) => {
                  e.preventDefault();
                  navTo(n.id);
                }}
              >
                <ScrambleText text={n.label} trigger="none" hover />
              </a>
            ))}
          </nav>
          <button
            ref={burgerRef}
            type="button"
            className={cn("hud-burger", menuOpen && "is-open")}
            onClick={() => ui.set({ menuOpen: !menuOpen })}
            aria-expanded={menuOpen}
            aria-controls="mobile-menu"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
          >
            <i />
            <i />
          </button>
        </div>
      </header>

      <div className="hud-bottom" aria-hidden="true">
        <div className="hud-bottom__left">
          <Clock />
          {/* Keyboard hint: CSS hides it on touch screens */}
          <span className="hud-keys">
            <span className="hud-dot" />
            Press <kbd>?</kbd> for shortcuts
          </span>
        </div>
        <Telemetry />
        <ScrollMeter />
      </div>

      <div id="mobile-menu" className={cn("mobile-menu", menuOpen && "is-open")} aria-hidden={!menuOpen}>
        <nav aria-label="Mobile">
          {NAV.map((n, i) => (
            <a
              key={n.id}
              href="#/"
              tabIndex={menuOpen ? 0 : -1}
              style={{ transitionDelay: menuOpen ? `${120 + i * 70}ms` : "0ms" }}
              onClick={(e) => {
                e.preventDefault();
                navTo(n.id);
              }}
            >
              <span className="mobile-menu__idx">0{i + 1}</span>
              {n.label}
            </a>
          ))}
        </nav>
        <div className="mobile-menu__foot">
          <a href={profile.linkedin} target="_blank" rel="noopener noreferrer" tabIndex={menuOpen ? 0 : -1}>
            LinkedIn <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </>
  );
}
