"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import * as SI from "react-icons/si";
import { Icon as Iconify } from "@iconify/react";
import {
  SiPython,
  SiPytorch,
  SiTensorflow,
  SiScikitlearn,
  SiApacheairflow,
  SiMlflow,
  SiDocker,
  SiFastapi,
  SiPandas,
  SiPostgresql,
  SiOpencv,
  SiHuggingface,
  SiSnowflake,
  SiTrello,
} from "react-icons/si";

const AzureDevOpsIcon: React.ComponentType<{ className?: string }> =
  // @ts-ignore – we intentionally probe the namespace at runtime
  (SI as any).SiAzuredevops || ((props) => (
    <Iconify icon="simple-icons:azuredevops" className={props.className} />
  ));

type Skill = {
  name: string;
  Icon: React.ComponentType<{ className?: string }>;
  hint?: string;
};

const SKILLS: Skill[] = [
  { name: "Python", Icon: SiPython, hint: "5+ yrs · data & backend" },
  { name: "PyTorch", Icon: SiPytorch, hint: "CV/NLP training & serving" },
  { name: "TensorFlow", Icon: SiTensorflow },
  { name: "scikit-learn", Icon: SiScikitlearn },
  { name: "Airflow", Icon: SiApacheairflow, hint: "40+ DAGs · SLAs" },
  { name: "MLflow", Icon: SiMlflow, hint: "tracking & model registry" },
  { name: "Docker", Icon: SiDocker, hint: "multi-stage builds" },
  { name: "FastAPI", Icon: SiFastapi, hint: "<100ms p95 inference" },
  { name: "Pandas", Icon: SiPandas },
  { name: "PostgreSQL", Icon: SiPostgresql },
  { name: "OpenCV", Icon: SiOpencv },
  { name: "Hugging Face", Icon: SiHuggingface },
  { name: "Snowflake", Icon: SiSnowflake, hint: "ELT & analytics" },
  { name: "Trello", Icon: SiTrello, hint: "Kanban · roadmaps" },
  { name: "Azure DevOps", Icon: AzureDevOpsIcon, hint: "Boards · Pipelines" },
];

type UISize = "xs" | "sm" | "md" | "lg" | "xl";

const UI_PRESETS: Record<
  UISize,
  { gap: string; pad: string; icon: string; text: string; card: string; step: number; intervalMs: number }
> = {
  xs: { gap: "gap-2", pad: "px-3 py-2", icon: "h-4 w-4", text: "text-xs", card: "rounded-lg border border-border bg-card/60 backdrop-blur shadow-sm hover:shadow", step: 0.8, intervalMs: 22 },
  sm: { gap: "gap-3", pad: "px-3.5 py-2.5", icon: "h-5 w-5", text: "text-sm", card: "rounded-lg border border-border bg-card/60 backdrop-blur shadow-sm hover:shadow", step: 1.1, intervalMs: 22 },
  md: { gap: "gap-4", pad: "px-4 py-3", icon: "h-5 w-5", text: "text-sm", card: "rounded-xl border border-border bg-card/60 backdrop-blur shadow-sm hover:shadow", step: 1.4, intervalMs: 20 },
  lg: { gap: "gap-5", pad: "px-5 py-3.5", icon: "h-6 w-6", text: "text-base", card: "rounded-xl border border-border bg-card/60 backdrop-blur shadow-sm hover:shadow", step: 1.8, intervalMs: 20 },
  xl: { gap: "gap-6", pad: "px-6 py-4", icon: "h-7 w-7", text: "text-base", card: "rounded-2xl border border-border bg-card/60 backdrop-blur shadow-sm hover:shadow", step: 2.2, intervalMs: 18 },
};

const getUISize = (w: number): UISize =>
  w < 380 ? "xs" : w < 640 ? "sm" : w < 1024 ? "md" : w < 1440 ? "lg" : "xl";

const SkillsStrip = () => {
  const railRef = useRef<HTMLDivElement | null>(null);
  const autoRef = useRef<number | null>(null);
  const loopWidthRef = useRef<number>(0);
  const runningRef = useRef(true);
  const copiesRef = useRef<number>(3);

  const [ui, setUi] = useState<UISize>(() => (typeof window === "undefined" ? "md" : getUISize(window.innerWidth)));
  const [copies, setCopies] = useState<number>(3);

  const preset = UI_PRESETS[ui];

  const REPEATED: Skill[] = useMemo(() => {
    const arr: Skill[] = [];
    for (let i = 0; i < copies; i++) arr.push(...SKILLS);
    return arr;
  }, [copies]);

  const measureLoopWidth = () => {
    const el = railRef.current;
    if (!el) return;

    const single = SKILLS.length;

    if (el.children.length >= single + 1) {
      const anchor = el.children[single] as HTMLElement;
      loopWidthRef.current = anchor.offsetLeft;

      const containerW = el.clientWidth;
      const singleWidth = loopWidthRef.current;
      if (singleWidth > 0) {
        const needed = Math.max(3, Math.ceil((containerW * 3.2) / singleWidth));
        if (needed !== copiesRef.current) {
          copiesRef.current = needed;
          setCopies(needed);
        }
      }
    } else {
      let w = 0;
      for (let i = 0; i < Math.min(single, el.children.length); i++) {
        const c = el.children[i] as HTMLElement;
        const style = window.getComputedStyle(c);
        w += c.offsetWidth + parseFloat(style.marginRight || "0");
      }
      loopWidthRef.current = w;
    }
  };

  const start = () => {
    if (autoRef.current != null) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const el = railRef.current;
    if (!el) return;

    measureLoopWidth();
    if (!loopWidthRef.current) return;

    const prev = el.style.scrollBehavior;
    el.style.scrollBehavior = "auto";

    autoRef.current = window.setInterval(() => {
      const loop = loopWidthRef.current;
      if (!railRef.current || !loop) return;
      let next = railRef.current.scrollLeft + preset.step;
      if (next >= loop) next -= loop;
      else if (next < 0) next += loop;
      railRef.current.scrollLeft = next;
    }, preset.intervalMs);

    runningRef.current = true;
    (start as any).__restore = () => (el.style.scrollBehavior = prev);
  };

  const stop = () => {
    if (autoRef.current != null) {
      clearInterval(autoRef.current);
      autoRef.current = null;
    }
    const restore = (start as any).__restore;
    if (typeof restore === "function") restore();
    runningRef.current = false;
  };

  useEffect(() => {
    const t = window.setTimeout(() => {
      measureLoopWidth();
      start();
    }, 50);

    const onResize = () => {
      setUi(getUISize(window.innerWidth));
      window.clearTimeout((onResize as any).__t);
      (onResize as any).__t = window.setTimeout(() => {
        measureLoopWidth();
      }, 100);
    };
    window.addEventListener("resize", onResize);

    // Fonts/icons settling can change widths
    // @ts-ignore - fonts may not be typed in older TS DOM libs
    document.fonts?.ready?.then(() => measureLoopWidth()).catch(() => {});

    // ✅ FIX: you can’t do `new ResizeObserver?.()`. Guard first, then construct.
    const RO: any = (window as any).ResizeObserver;
    const ro = RO ? new RO(() => measureLoopWidth()) : null;
    if (ro && railRef.current) ro.observe(railRef.current);

    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      ro?.disconnect?.();
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    measureLoopWidth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [copies]);

  useEffect(() => {
    if (!railRef.current) return;
    if (autoRef.current != null) {
      stop();
      measureLoopWidth();
      start();
    } else {
      measureLoopWidth();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ui]);

  useEffect(() => {
    const el = railRef.current;
    if (!el) return;
    const onEnter = () => stop();
    const onLeave = () => start();
    el.addEventListener("mouseenter", onEnter);
    el.addEventListener("touchstart", onEnter, { passive: true });
    el.addEventListener("mouseleave", onLeave);
    el.addEventListener("touchend", onLeave);
    return () => {
      el.removeEventListener("mouseenter", onEnter);
      el.removeEventListener("touchstart", onEnter as any);
      el.removeEventListener("mouseleave", onLeave);
      el.removeEventListener("touchend", onLeave as any);
    };
  }, []);

  return (
    <section className="py-12 bg-background">
      <div className="container mx-auto px-4">
        <div className="mb-6 flex items-end justify-between">
          <div>
            <h2 className="text-3xl font-bold font-heading text-foreground">Toolbox</h2>
            <p className="text-muted-foreground text-sm">The technologies I use in production.</p>
          </div>
        </div>

        <div
          ref={railRef}
          className={cn(
            "flex items-stretch overflow-x-auto pb-3",
            "scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]",
            // optional edge fade; remove if you don't use arbitrary properties in Tailwind
            "mask-image-[linear-gradient(to_right,transparent,black_24px,black_calc(100%-24px),transparent)]",
            UI_PRESETS[ui].gap
          )}
          aria-label="Skills"
          tabIndex={0}
        >
          {REPEATED.map(({ name, Icon, hint }, i) => (
            <div
              key={`${name}-${i}`}
              className={cn(
                "group flex-shrink-0 select-none transition",
                UI_PRESETS[ui].card,
                UI_PRESETS[ui].pad
              )}
              title={hint || name}
              role="img"
              aria-label={name}
            >
              <div className="flex items-center gap-3">
                <Icon className={cn(UI_PRESETS[ui].icon, "opacity-80 group-hover:opacity-100 transition")} />
                <span className={cn("font-medium", UI_PRESETS[ui].text)}>{name}</span>
              </div>
            </div>
          ))}
        </div>

        <p className="sr-only">The toolbar auto-scales and duplicates its content to fill very large screens.</p>
      </div>
    </section>
  );
};

export default SkillsStrip;
