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

  // Auto-scroll params
  const step = 1.8; // px per tick
  const intervalMs = 17; // ~60fps

  const measureLoopWidth = () => {
    const el = sliderRef.current;
    if (!el) return;
    // We expect first copy size = offsetLeft of the first element of second copy.
    const childCount = el.children.length;
    const singleSetCount = specializations.length;
    if (childCount >= singleSetCount + 1) {
      const secondCopyFirst = el.children[singleSetCount] as HTMLElement;
      singleSetWidthRef.current = secondCopyFirst.offsetLeft;
    } else {
      // fallback: sum widths of first set
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
    if (autoScrollRef.current != null) return;
    if (
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    )
      return;

    const el = sliderRef.current;
    if (!el) return;

    // ensure loopWidth measured
    measureLoopWidth();
    if (!singleSetWidthRef.current) return;

    // prevent smooth behavior from interfering during interval ticks
    const prevBehavior = el.style.scrollBehavior;
    el.style.scrollBehavior = "auto";

    autoScrollRef.current = window.setInterval(() => {
      const el = sliderRef.current;
      const loopWidth = singleSetWidthRef.current || 0;
      if (!el || !loopWidth) return;

      let next = el.scrollLeft + step;

      // wrap both directions BEFORE the browser clamps to max scrollLeft
      if (next >= loopWidth) next -= loopWidth;
      else if (next < 0) next += loopWidth;

      el.scrollLeft = next;
    }, intervalMs);

    isAutoScrolling.current = true;

    // store a restore fn so stop can bring back previous behavior
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
    // measure after first paint and start auto-scroll
    const t = window.setTimeout(() => {
      measureLoopWidth();
      startAutoScroll();
    }, 50);

    // recalc on resize
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

  // Robust centering: pick the copy nearest to current scroll
  const scrollToActiveCard = (index: number) => {
    const el = sliderRef.current;
    if (!el) return;
    measureLoopWidth();
    const loopWidth = singleSetWidthRef.current || 0;

    // find target element in first copy
    const firstCopyEl = el.children[index] as HTMLElement | undefined;
    if (!firstCopyEl) return;
    const basePos = firstCopyEl.offsetLeft;

    // choose closest between basePos and basePos + loopWidth
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
    stopAutoScroll();
    setActiveSpecialization(index);
    scrollToSection(specializations[index].id);
    scrollToActiveCard(index);
  };

  // Smooth user nudge that also wraps
  const nudge = (delta: number) => {
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

    // restore after smooth finishes
    window.setTimeout(() => {
      el.style.scrollBehavior = prevBehavior;
    }, 300);
  };

  const scrollLeft = () => nudge(-300);
  const scrollRight = () => nudge(300);

  // When parent changes, center nearest copy — but don't fight auto-scroll
  useEffect(() => {
    if (isAutoScrolling.current) return;
    scrollToActiveCard(activeSpecialization);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSpecialization]);

  // Automated selection based on left-edge crossing.
  // When a tile fully passes the left edge (consider left nav overlap), the next tile becomes selected.
  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;

    let ticking = false;
    const arrowOverlap = 44; // px: approximate width of left arrow button overlap

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const children = el.children;
        if (!children || children.length === 0) {
          ticking = false;
          return;
        }

        // leftEdge threshold (account for left arrow)
        const leftEdge = el.scrollLeft + arrowOverlap;

        // find the last child whose right edge is <= leftEdge
        let lastFullyLeftIndex = -1;
        for (let i = 0; i < children.length; i++) {
          const c = children[i] as HTMLElement;
          const childRight = c.offsetLeft + c.offsetWidth;
          if (childRight <= leftEdge + 1) {
            lastFullyLeftIndex = i;
          } else {
            break; // children are in order; once a child is not fully left we can stop
          }
        }

        // selected becomes the next tile after lastFullyLeftIndex
        const nextDoubledIndex = lastFullyLeftIndex + 1;
        if (nextDoubledIndex >= 0) {
          // Use modulo to get the original specialization index
          const normalized = nextDoubledIndex % specializations.length;
          if (isAutoScrolling.current) {
            setActiveSpecialization(normalized);
          }
        }

        ticking = false;
      });
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    // call once to initialize
    onScroll();

    return () => {
      el.removeEventListener("scroll", onScroll);
    };
  }, [specializations.length, setActiveSpecialization]);

  // Optional: pause auto-scroll when hovering/touching the strip
  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;
    const onEnter = () => stopAutoScroll();
    const onLeave = () => startAutoScroll();
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

  return (
    <section className="py-20 bg-gradient-surface">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto text-center mb-12">
          <h2 className="text-4xl font-bold mb-6 bg-gradient-accent bg-clip-text text-transparent font-heading">
            Explore My Work
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Navigate through my portfolio by selecting a specialization to view related projects
          </p>
        </div>

        <div className="max-w-6xl mx-auto relative">
          <Button
            variant="outline"
            size="sm"
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
            onClick={scrollLeft}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
            onClick={scrollRight}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          <div
            ref={sliderRef}
            className="flex overflow-x-auto gap-5 pb-4 mb-8 mx-12 scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
          >
            {doubled.map((spec, idx) => {
              const originalIndex = idx % specializations.length;
              const isActive = originalIndex === activeSpecialization;
              const IconComponent = spec.icon;
              return (
                <Button
                  key={`${spec.id}-${idx}`}
                  variant={isActive ? "default" : "outline"}
                  className={cn(
                    "flex-shrink-0 h-auto p-6 min-w-[280px] flex flex-col items-center text-center transition-all duration-500 ease-out",
                    isActive
                      ? "bg-primary text-primary-foreground shadow-glow scale-105 animate-scale-in"
                      : "bg-card/50 backdrop-blur border-border hover:border-primary/30 hover:shadow-soft hover:scale-105"
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
                  <h3 className="text-sm font-semibold mb-2 font-heading">{spec.title}</h3>
                  <p
                    className={cn(
                      "text-xs leading-relaxed",
                      isActive ? "text-primary-foreground/80" : "text-muted-foreground"
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
                className={cn(
                  "w-2 h-2 rounded-full transition-all duration-300 hover:scale-125",
                  activeSpecialization === index ? "bg-primary w-8 h-2 animate-pulse" : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
                )}
                onClick={() => {
                  stopAutoScroll();
                  handleSpecializationClick(index);
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default SpecializationSlider;
