import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  BarChart3,
  Eye,
  Brain,
  MessageSquare,
  Search,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Specialization {
  id: string;
  title: string;
  icon: any;
  description: string;
}

interface SpecializationSliderProps {
  activeSpecialization: number;
  setActiveSpecialization: (index: number) => void;
}

const SpecializationSlider = ({
  activeSpecialization,
  setActiveSpecialization,
}: SpecializationSliderProps) => {
  const sliderRef = useRef<HTMLDivElement | null>(null);
  const autoScrollRef = useRef<number | null>(null);
  const isAutoScrolling = useRef(true);

  // Lock when user selects; unlock when clicking off the tiles/strip
  const userLockedRef = useRef(false);

  // used for robust measurement/wrapping
  const singleSetWidthRef = useRef<number>(0);

  const specializations: Specialization[] = [
    {
      id: "dataAnalytics",
      title: "Data Analytics",
      icon: BarChart3,
      description: "Interactive dashboards and business intelligence solutions",
    },
    {
      id: "computerVision",
      title: "Computer Vision",
      icon: Eye,
      description: "Image processing and visual recognition systems",
    },
    {
      id: "machineLearning",
      title: "Machine Learning",
      icon: Brain,
      description: "Predictive modeling and algorithmic solutions",
    },
    {
      id: "nlp",
      title: "Natural Language Processing",
      icon: MessageSquare,
      description: "Text analysis and language understanding",
    },
    {
      id: "research",
      title: "Research Methods",
      icon: Search,
      description: "Scientific research and statistical analysis",
    },
    {
      id: "mlops",
      title: "MLOps & Deployment",
      icon: Settings,
      description: "Production ML systems and infrastructure",
    },
  ];

  // doubled for seamless loop
  const doubled = [...specializations, ...specializations];

  // Auto-scroll params (slightly slower than before)
  const step = 1.2; // px per tick
  const intervalMs = 25; // ms between ticks

  const measureLoopWidth = () => {
    const el = sliderRef.current;
    if (!el) return;
    const childCount = el.children.length;
    const singleSetCount = specializations.length;
    if (childCount >= singleSetCount + 1) {
      const secondCopyFirst = el.children[singleSetCount] as HTMLElement;
      singleSetWidthRef.current = secondCopyFirst.offsetLeft;
    } else {
      let w = 0;
      for (let i = 0; i < Math.min(childCount, singleSetCount); i++) {
        const c = el.children[i] as HTMLElement;
        w += c.offsetWidth;
        const style = window.getComputedStyle(c);
        const marginRight = parseFloat(style.marginRight || "0");
        w += marginRight;
      }
      singleSetWidthRef.current = w;
    }
  };

  const startAutoScroll = () => {
    if (userLockedRef.current) return;
    if (autoScrollRef.current != null) return;
    if (
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    )
      return;

    const el = sliderRef.current;
    if (!el) return;

    measureLoopWidth();
    if (!singleSetWidthRef.current) return;

    const prevBehavior = el.style.scrollBehavior;
    el.style.scrollBehavior = "auto";

    autoScrollRef.current = window.setInterval(() => {
      const el = sliderRef.current;
      const loopWidth = singleSetWidthRef.current || 0;
      if (!el || !loopWidth) return;

      let next = el.scrollLeft + step;
      if (next >= loopWidth) next -= loopWidth;
      else if (next < 0) next += loopWidth;

      el.scrollLeft = next;
    }, intervalMs);

    isAutoScrolling.current = true;

    const restore = () => {
      if (el) el.style.scrollBehavior = prevBehavior;
    };
    (startAutoScroll as any).__restore = restore;
  };

  const stopAutoScroll = () => {
    if (autoScrollRef.current != null) {
      clearInterval(autoScrollRef.current);
      autoScrollRef.current = null;
    }
    const restore = (startAutoScroll as any).__restore;
    if (typeof restore === "function") restore();
    isAutoScrolling.current = false;
  };

  useEffect(() => {
    const t = window.setTimeout(() => {
      measureLoopWidth();
      startAutoScroll();
    }, 50);

    const onResize = () => {
      window.clearTimeout((onResize as any).__t);
      (onResize as any).__t = window.setTimeout(() => {
        measureLoopWidth();
      }, 120);
    };
    window.addEventListener("resize", onResize);

    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      stopAutoScroll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth" });
  };

  const scrollToActiveCard = (index: number) => {
    const el = sliderRef.current;
    if (!el) return;
    measureLoopWidth();
    const loopWidth = singleSetWidthRef.current || 0;

    const firstCopyEl = el.children[index] as HTMLElement | undefined;
    if (!firstCopyEl) return;
    const basePos = firstCopyEl.offsetLeft;

    const current = el.scrollLeft;
    const optionA = basePos;
    const optionB = basePos + loopWidth;
    const target =
      Math.abs(optionA - current) <= Math.abs(optionB - current)
        ? optionA
        : optionB;

    el.scrollTo({ left: target, behavior: "smooth" });
  };

  const handleSpecializationClick = (index: number) => {
    userLockedRef.current = true;
    stopAutoScroll();
    setActiveSpecialization(index);
    scrollToSection(specializations[index].id);
    scrollToActiveCard(index);
  };

  const nudge = (delta: number) => {
    userLockedRef.current = true;
    stopAutoScroll();
    const el = sliderRef.current;
    if (!el) return;
    measureLoopWidth();
    const loopWidth = singleSetWidthRef.current || 0;

    const prevBehavior = el.style.scrollBehavior;
    el.style.scrollBehavior = "smooth";

    let next = (el.scrollLeft || 0) + delta;
    if (loopWidth > 0) {
      if (next >= loopWidth) next -= loopWidth;
      else if (next < 0) next += loopWidth;
    }
    el.scrollLeft = next;

    window.setTimeout(() => {
      el.style.scrollBehavior = prevBehavior;
    }, 300);
  };

  const scrollLeft = () => nudge(-300);
  const scrollRight = () => nudge(300);

  useEffect(() => {
    if (isAutoScrolling.current) return;
    scrollToActiveCard(activeSpecialization);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSpecialization]);

  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;

    let ticking = false;
    const arrowOverlap = 44;

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const children = el.children;
        if (!children || children.length === 0) {
          ticking = false;
          return;
        }

        if (userLockedRef.current) {
          ticking = false;
          return;
        }

        const leftEdge = el.scrollLeft + arrowOverlap;

        let lastFullyLeftIndex = -1;
        for (let i = 0; i < children.length; i++) {
          const c = children[i] as HTMLElement;
          const childRight = c.offsetLeft + c.offsetWidth;
          if (childRight <= leftEdge + 1) {
            lastFullyLeftIndex = i;
          } else {
            break;
          }
        }

        const nextDoubledIndex = lastFullyLeftIndex + 1;
        if (nextDoubledIndex >= 0) {
          const normalized = nextDoubledIndex % specializations.length;
          if (isAutoScrolling.current) {
            setActiveSpecialization(normalized);
          }
        }

        ticking = false;
      });
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    return () => {
      el.removeEventListener("scroll", onScroll);
    };
  }, [specializations.length, setActiveSpecialization]);

  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;
    const onEnter = () => stopAutoScroll();
    const onLeave = () => {
      if (!userLockedRef.current) startAutoScroll();
    };
    el.addEventListener("mouseenter", onEnter);
    el.addEventListener("touchstart", onEnter, { passive: true });
    el.addEventListener("mouseleave", onLeave);
    el.addEventListener("touchend", onLeave);
    return () => {
      el.removeEventListener("mouseenter", onEnter);
      el.removeEventListener("touchstart", onEnter);
      el.removeEventListener("mouseleave", onLeave);
      el.removeEventListener("touchend", onLeave);
    };
  }, []);

  // Resume auto-scroll when clicking OFF a tile or outside the strip
  useEffect(() => {
    const handleDocPointerDown = (e: PointerEvent) => {
      const root = sliderRef.current;
      const target = e.target as HTMLElement | null;
      if (!root || !target) return;

      if (root.contains(target)) {
        const onTile = !!target.closest('[data-role="tile"]');
        const onDot = !!target.closest('[data-role="dot"]');
        const onNav = !!target.closest('[data-role="nav"]');
        if (!onTile && !onDot && !onNav) {
          userLockedRef.current = false;
          startAutoScroll();
        }
      } else {
        userLockedRef.current = false;
        startAutoScroll();
      }
    };

    document.addEventListener("pointerdown", handleDocPointerDown, { passive: true });
    return () => {
      document.removeEventListener("pointerdown", handleDocPointerDown);
    };
  }, []);

  return (
    <section className="py-20 bg-gradient-surface">
      {/* Header stays centered in container */}
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto text-center mb-12">
          {/* Made header black (removed gradient text) */}
          <h2 className="text-4xl font-bold mb-6 font-heading text-black">
            Explore My Work
          </h2>
          {/* Made intro text black */}
          <p className="text-lg mb-8 max-w-2xl mx-auto text-black">
            Navigate through my portfolio by selecting a specialization to view related projects
          </p>
        </div>
      </div>

      {/* Full-bleed slider wrapper */}
      <div className="relative w-screen left-1/2 -ml-[50vw] px-4 sm:px-6 lg:px-8">
        {/* Arrow buttons pinned near screen edges */}
        <Button
          variant="outline"
          size="sm"
          data-role="nav"
          className="absolute left-4 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
          onClick={scrollLeft}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <Button
          variant="outline"
          size="sm"
          data-role="nav"
          className="absolute right-4 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
          onClick={scrollRight}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        {/* The autoscroll strip now spans the full screen width */}
        <div
          ref={sliderRef}
          className="flex overflow-x-auto gap-5 pb-4 mb-8 scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
        >
          {doubled.map((spec, idx) => {
            const originalIndex = idx % specializations.length;
            const isActive = originalIndex === activeSpecialization;
            const IconComponent = spec.icon;
            return (
              <Button
                key={`${spec.id}-${idx}`}
                data-role="tile"
                variant={isActive ? "default" : "outline"}
                className={cn(
                  "flex-shrink-0 h-auto p-6 min-w-[280px] flex flex-col items-center text-center transition-all duration-500 ease-out",
                  isActive
                    ? "bg-primary text-primary-foreground shadow-glow scale-105 animate-scale-in"
                    : "bg-card/50 backdrop-blur border-border hover:border-primary/30 hover:shadow-soft hover:scale-105 text-black"
                )}
                onClick={() => handleSpecializationClick(originalIndex)}
              >
                <div
                  className={cn(
                    "p-3 rounded-lg mb-3 transition-all duration-300",
                    isActive ? "bg-primary-foreground/20" : "bg-gradient-primary"
                  )}
                >
                  <IconComponent
                    className={cn(
                      "h-6 w-6 transition-transform duration-300",
                      isActive ? "text-primary-foreground animate-pulse" : "text-background"
                    )}
                  />
                </div>
                {/* Title: black when inactive, white when active for contrast */}
                <h3
                  className={cn(
                    "text-sm font-semibold mb-2 font-heading",
                    isActive ? "text-primary-foreground" : "text-black"
                  )}
                >
                  {spec.title}
                </h3>
                {/* Description: black when inactive, slightly translucent white when active */}
                <p
                  className={cn(
                    "text-xs leading-relaxed",
                    isActive ? "text-primary-foreground/80" : "text-black/80"
                  )}
                >
                  {spec.description}
                </p>
              </Button>
            );
          })}
        </div>

        <div className="flex justify-center gap-3">
          {specializations.map((_, index) => (
            <button
              key={index}
              data-role="dot"
              className={cn(
                "w-2 h-2 rounded-full transition-all duration-300 hover:scale-125",
                activeSpecialization === index
                  ? "bg-primary w-8 h-2 animate-pulse"
                  : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
              )}
              onClick={() => {
                handleSpecializationClick(index);
              }}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default SpecializationSlider;
