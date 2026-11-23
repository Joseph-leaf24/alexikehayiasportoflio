import { useEffect, useRef, useState, type CSSProperties } from "react";
import { useLocation } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { sceneState, ui, useStore } from "@/lib/store";
import { scrollToTarget, setScrollLocked } from "@/lib/smoothScroll";
import { profile } from "@/data/profile";
import { SQUAD } from "@/data/squad";
import { cn } from "@/lib/utils";
import ScrambleText from "@/components/fx/ScrambleText";
import { usePageTransition } from "./Transition";

const NAV = [
  { id: "about", label: "About" },
  { id: "domains", label: "Domains" },
  { id: "toolbox", label: "Toolbox" },
  { id: "projects", label: "Work" },
  { id: "contact", label: "Contact" },
  { id: "watch", label: "Watch" },
];

/** A score in six characters, also after hours of play (1.25M). */
const scoreText = (n: number) => (n < 1e6 ? String(n).padStart(6, "0") : `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`.padStart(6, " "));

/** The level being played and the squad's running totals (polled ~8×/s; hidden without a canvas). */
function Telemetry() {
  const [level, setLevel] = useState({ no: sceneState.level, name: sceneState.stage });
  const rootRef = useRef<HTMLDivElement>(null);
  const killsRef = useRef<HTMLSpanElement>(null);
  const boomsRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let current = "";
    const id = window.setInterval(() => {
      // Hidden by CSS below 1280 px wide: nothing to update
      const root = rootRef.current;
      if (!root || getComputedStyle(root).display === "none") return;
      const key = `${sceneState.level}:${sceneState.stage}`;
      if (key !== current) {
        current = key;
        setLevel({ no: sceneState.level, name: sceneState.stage });
      }
      if (killsRef.current) killsRef.current.textContent = String(sceneState.kills).padStart(4, "0");
      if (boomsRef.current) boomsRef.current.textContent = String(sceneState.booms).padStart(4, "0");
    }, 120);
    return () => window.clearInterval(id);
  }, []);
  if (!sceneState.canvas) return null;
  return (
    <div ref={rootRef} className="hud-telemetry">
      <span>Level {String(level.no).padStart(2, "0")}</span>
      {level.name ? <ScrambleText key={`${level.no}-${level.name}`} text={level.name} trigger="mount" duration={500} className="hud-telemetry__val" /> : null}
      <span className="hud-telemetry__extra">
        <span className="hud-dot" />
        Kills <span ref={killsRef} className="tabular-nums">0000</span>
        <span className="hud-dot" />
        Booms <span ref={boomsRef} className="tabular-nums">0000</span>
      </span>
    </div>
  );
}

/** P1–P8: badge, role, health in six segments and score, live from the scene. */
function Squad() {
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const id = window.setInterval(() => {
      const root = rootRef.current;
      if (!root || getComputedStyle(root).display === "none") return;
      root.querySelectorAll<HTMLElement>(".pcard").forEach((card, i) => {
        const p = sceneState.players[i];
        card.toggleAttribute("data-down", !p.alive);
        const hp = card.querySelector<HTMLElement>(".pcard__hp i");
        if (hp) hp.style.transform = `scaleX(${Math.round(p.hp * 6) / 6})`;
        const score = card.querySelector<HTMLElement>(".pcard__score");
        const text = scoreText(p.score);
        if (score && score.textContent !== text) score.textContent = text;
      });
    }, 150);
    return () => window.clearInterval(id);
  }, []);
  if (!sceneState.canvas) return null;
  return (
    <div ref={rootRef} className="hud-squad">
      {SQUAD.map((m) => (
        <div key={m.player} className="pcard" style={{ "--pc": m.color } as CSSProperties}>
          <span className="pcard__badge">P{m.player}</span>
          <span className="pcard__name">{m.short}</span>
          <span className="pcard__hp">
            <i />
          </span>
          <span className="pcard__score tabular-nums">000000</span>
        </div>
      ))}
    </div>
  );
}

/** Scroll progress: a ten-segment bar plus a 000–100 % counter. */
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
      // Fills one whole segment at a time
      if (barRef.current) barRef.current.style.transform = `scaleX(${Math.round(p * 10) / 10})`;
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
      <span>
        <span ref={numRef}>000</span>%
      </span>
      <span className="scroll-meter__track">
        <span ref={barRef} className="scroll-meter__bar" />
      </span>
    </div>
  );
}

/**
 * Fixed chrome on every page: frame corners; a top bar with the brand, current section
 * and nav (a full-screen menu on small screens); and a bottom bar like a co-op game's: the
 * stage with the squad's totals, the four player cards and the scroll progress. Off the home
 * page, nav links go home first.
 */
export default function Hud() {
  const { sectionIndex, sectionLabel, menuOpen } = useStore(ui, (s) => s);
  const location = useLocation();
  const go = usePageTransition();
  const onHome = location.pathname === "/";

  const navTo = (id: string) => {
    // From the phone menu: unlock scrolling before scrolling. (Left to the menu-closed effect,
    // restarting the smooth scroller would cancel the scroll that had just begun, so the
    // menu closed but the page stayed put.)
    if (ui.get().menuOpen) setScrollLocked(false);
    ui.set({ menuOpen: false });
    if (onHome) scrollToTarget(id === "top" ? 0 : `#${id}`, { offset: id === "top" ? 0 : -10 });
    else go("/", { scrollTo: id === "top" ? undefined : id });
  };

  const burgerRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setScrollLocked(menuOpen);
    if (!menuOpen) {
      // Closing with focus on one of its links (now hidden) would drop focus to the page: hand it back to the burger
      if (document.getElementById("mobile-menu")?.contains(document.activeElement)) burgerRef.current?.focus();
      return;
    }
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && ui.set({ menuOpen: false });
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
          {/* The name stays the visible text plus "home" (WCAG 2.5.3 label in name) */}
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
                href={`#/`}
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
          <Telemetry />
          {/* Keyboard users only (hidden on touch screens by CSS) */}
          <span className="hud-keys">
            <kbd>?</kbd> Shortcuts
          </span>
        </div>
        <Squad />
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
