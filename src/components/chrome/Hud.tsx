import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { ui, useStore } from "@/lib/store";
import { scrollToTarget, setScrollLocked } from "@/lib/smoothScroll";
import { sound } from "@/lib/sound";
import { profile } from "@/data/profile";
import { cn } from "@/lib/utils";
import ScrambleText from "@/components/fx/ScrambleText";
import { usePageTransition } from "./Transition";

const NAV = [
  { id: "about", label: "About" },
  { id: "toolbox", label: "Toolbox" },
  { id: "projects", label: "Work" },
  { id: "contact", label: "Contact" },
];

function SoundToggle() {
  const on = useStore(ui, (s) => s.soundOn);
  return (
    <button
      type="button"
      className={cn("sound-toggle", on && "is-on")}
      onClick={() => sound.toggle()}
      aria-pressed={on}
      aria-label={on ? "Mute ambient sound" : "Play ambient sound"}
      data-cursor={on ? "MUTE" : "SOUND"}
    >
      <span className="sound-toggle__bars" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </span>
      <span className="sound-toggle__lbl">{on ? "Sound on" : "Sound off"}</span>
    </button>
  );
}

function Clock() {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const fmt = new Intl.DateTimeFormat("en-GB", {
      timeZone: profile.timezone,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    const update = () => {
      if (ref.current) ref.current.textContent = `${fmt.format(new Date())} CET`;
    };
    update();
    const id = window.setInterval(update, 1000);
    return () => window.clearInterval(id);
  }, []);
  return <span ref={ref} className="tabular-nums" />;
}

function ScrollMeter() {
  const barRef = useRef<HTMLSpanElement>(null);
  const numRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    let lastPct = -1;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      const pct = Math.round(p * 100);
      if (barRef.current) barRef.current.style.transform = `scaleY(${p})`;
      if (numRef.current && pct !== lastPct) {
        numRef.current.textContent = String(pct).padStart(3, "0");
        lastPct = pct;
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
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

export default function Hud() {
  const { sectionIndex, sectionLabel, menuOpen } = useStore(ui, (s) => s);
  const location = useLocation();
  const go = usePageTransition();
  const onHome = location.pathname === "/";

  const navTo = (id: string) => {
    ui.set({ menuOpen: false });
    if (onHome) scrollToTarget(id === "top" ? 0 : `#${id}`, { offset: id === "top" ? 0 : -10 });
    else go("/", { scrollTo: id === "top" ? undefined : id });
  };

  useEffect(() => {
    setScrollLocked(menuOpen);
    if (!menuOpen) return;
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
          aria-label="Alexi Kehayias — home"
        >
          <span className="hud-brand__mark" aria-hidden="true" />
          <ScrambleText text="Alexi Kehayias" trigger="mount" hover className="hud-brand__name" />
          <span className="hud-brand__tag">[ DS &amp; AI ]</span>
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
          <SoundToggle />
          <button
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
          <span>{profile.coords}</span>
          <span className="hud-dot" />
          <span>{profile.location}</span>
          <span className="hud-dot" />
          <Clock />
        </div>
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
          <a href={`mailto:${profile.email}`} tabIndex={menuOpen ? 0 : -1}>
            {profile.email} <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
          <a href={profile.linkedin} target="_blank" rel="noopener noreferrer" tabIndex={menuOpen ? 0 : -1}>
            LinkedIn <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </>
  );
}
