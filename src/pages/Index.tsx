import { useEffect, useLayoutEffect, useState } from "react";
import { useLocation, useNavigationType } from "react-router-dom";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Domains from "@/components/Domains";
import Skills from "@/components/Skills";
import Work from "@/components/Work";
import Contact from "@/components/Contact";
import { specializations } from "@/data/profile";
import { nav, sceneState, ui, type LayoutKey } from "@/lib/store";
import { refreshScroll, scrollToTarget } from "@/lib/smoothScroll";

/** Router state other pages send when linking home. */
interface HomeState {
  /** Section id to land on */
  scrollTo?: string;
  /** Specialization to open in Work */
  category?: string;
}

/** A number from sessionStorage (0 when missing or storage is blocked). */
const readNumber = (key: string) => {
  try {
    const v = Number(sessionStorage.getItem(key));
    return Number.isFinite(v) ? v : 0;
  } catch {
    return 0;
  }
};

/**
 * Home page: every section in scroll order. Each section's data-scene / data-section
 * attributes drive the 3D formation and the HUD label. Coming back from a project
 * restores the open specialization and scroll position.
 */
const Index = () => {
  const location = useLocation();
  const navType = useNavigationType();
  const state = (location.state ?? {}) as HomeState;

  // Coming back from a project restores the specialization that was open
  const [active, setActive] = useState(() => {
    const fromState = specializations.findIndex((s) => s.id === state.category);
    if (fromState >= 0) return fromState;
    const saved = readNumber("ak:spec");
    return saved >= 0 && saved < specializations.length ? saved : 0;
  });

  useEffect(() => {
    try {
      sessionStorage.setItem("ak:spec", String(active));
    } catch {
      /* ignore */
    }
  }, [active]);

  // Scroll position on arrival. A fresh page load or reload always opens on the intro (React
  // Router reports it as "POP", and the router state survives a reload). After moving within
  // the site: an explicit target > browser back/forward (the saved position) > top.
  useLayoutEffect(() => {
    if (!nav.inApp) {
      scrollToTarget(0, { immediate: true });
    } else if (state.scrollTo) {
      scrollToTarget(`#${state.scrollTo}`, { immediate: true, offset: -10 });
    } else if (navType === "POP") {
      scrollToTarget(readNumber("ak:homeY"), { immediate: true });
    } else {
      scrollToTarget(0, { immediate: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.key]);

  // Remember the scroll position for "back"
  useEffect(() => {
    let t = 0;
    const onScroll = () => {
      window.clearTimeout(t);
      t = window.setTimeout(() => {
        try {
          sessionStorage.setItem("ak:homeY", String(window.scrollY));
        } catch {
          /* ignore */
        }
      }, 120);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  // Feed section positions to the 3D scene and keep the HUD section label in sync
  useEffect(() => {
    sceneState.override = null;
    const sections = Array.from(document.querySelectorAll<HTMLElement>("[data-scene]"));
    const measure = () => {
      sceneState.stops = sections.map((el) => ({
        y: el.getBoundingClientRect().top + window.scrollY,
        layout: el.dataset.scene as LayoutKey,
      }));
    };
    measure();
    const ro = new ResizeObserver(() => {
      measure();
      refreshScroll();
    });
    ro.observe(document.body);
    window.addEventListener("resize", measure);

    const labelled = Array.from(document.querySelectorAll<HTMLElement>("[data-section]"));
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const [idx, label] = (e.target as HTMLElement).dataset.section!.split("|");
          ui.set({ sectionIndex: idx, sectionLabel: label });
        }
      },
      { rootMargin: "-50% 0px -50% 0px" }
    );
    labelled.forEach((el) => io.observe(el));

    return () => {
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("resize", measure);
      sceneState.stops = [];
    };
  }, []);

  return (
    <>
      <Hero />
      <About />
      <Domains onExplore={setActive} />
      <Skills />
      <Work active={active} setActive={setActive} />
      <Contact />
    </>
  );
};

export default Index;
