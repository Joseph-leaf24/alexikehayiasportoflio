import { useEffect, useLayoutEffect, useState } from "react";
import { useLocation, useNavigationType } from "react-router-dom";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Skills from "@/components/Skills";
import Work from "@/components/Work";
import Contact from "@/components/Contact";
import { specializations } from "@/data/profile";
import { sceneState, ui, type LayoutKey } from "@/lib/store";
import { refreshScroll, scrollToTarget } from "@/lib/smoothScroll";

interface HomeState {
  scrollTo?: string;
  category?: string;
}

const readNumber = (key: string) => {
  try {
    const v = Number(sessionStorage.getItem(key));
    return Number.isFinite(v) ? v : 0;
  } catch {
    return 0;
  }
};

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

  // Scroll position on arrival: explicit target > browser back/forward > top
  useLayoutEffect(() => {
    if (state.scrollTo) {
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
      <Skills />
      <Work active={active} setActive={setActive} />
      <Contact />
    </>
  );
};

export default Index;
