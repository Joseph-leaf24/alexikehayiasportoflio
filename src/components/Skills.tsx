import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
"use client";
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

// Duplicate for seamless loop
const DOUBLED = [...SKILLS, ...SKILLS];

const SkillsStrip = () => {
  const railRef = useRef<HTMLDivElement | null>(null);
  const autoRef = useRef<number | null>(null);
  const loopWidthRef = useRef<number>(0);
  const runningRef = useRef(true);

  const step = 1.4; // px per tick
  const intervalMs = 23; // ~60fps

  const measureLoopWidth = () => {
    const el = railRef.current;
    if (!el) return;
    // distance from first item of second copy
    const single = SKILLS.length;
    if (el.children.length >= single + 1) {
      const anchor = el.children[single] as HTMLElement;
      loopWidthRef.current = anchor.offsetLeft;
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
      let next = railRef.current.scrollLeft + step;
      if (next >= loop) next -= loop;
      else if (next < 0) next += loop;
      railRef.current.scrollLeft = next;
    }, intervalMs);

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
      window.clearTimeout((onResize as any).__t);
      (onResize as any).__t = window.setTimeout(() => {
        measureLoopWidth();
      }, 120);
    };
    window.addEventListener("resize", onResize);

    return () => {
      clearTimeout(t);
      window.removeEventListener("resize", onResize);
      stop();
    };
  }, []);

  // Pause on hover/touch
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
      el.removeEventListener("touchstart", onEnter);
      el.removeEventListener("mouseleave", onLeave);
      el.removeEventListener("touchend", onLeave);
    };
  }, []);

  return (
    <section className="py-12 bg-background">
      <div className="container mx-auto px-4">
        <div className="mb-6 flex items-end justify-between">
          <div>
            <h2 className="text-3xl font-bold font-heading text-foreground">Toolbox</h2>
            <p className="text-muted-foreground text-sm">
              The technologies I use in production.
            </p>
          </div>
         
        </div>

        <div
          ref={railRef}
          className={cn(
            "flex items-stretch gap-4 overflow-x-auto pb-3",
            "scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
          )}
          aria-label="Skills"
        >
          {DOUBLED.map(({ name, Icon, hint }, i) => (
            <div
              key={`${name}-${i}`}
              className={cn(
                "group flex-shrink-0 select-none",
                "rounded-xl border border-border bg-card/60 backdrop-blur",
                "px-4 py-3 shadow-sm hover:shadow transition"
              )}
              title={hint || name}
              role="img"
              aria-label={name}
            >
              <div className="flex items-center gap-3">
                <Icon className="h-5 w-5 opacity-80 group-hover:opacity-100 transition" />
                <span className="text-sm font-medium">{name}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default SkillsStrip;
