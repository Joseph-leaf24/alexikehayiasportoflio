import { useEffect, useState } from "react";
import { Infinity as Endless, Minus, Pause, Play, Plus } from "lucide-react";
import { sceneState } from "@/lib/store";
import { hasModifier, isModalOpen, isTypingTarget } from "@/lib/keys";
import { cn } from "@/lib/utils";

const SPEEDS = [0.5, 1, 2, 4, 8] as const;

/**
 * Controls for the battle behind the page, shown while the spectator space at the bottom is in view:
 * pause, endless levels, speed from ½× to 8×, and zoom. On the keyboard, P pauses or resumes, [ and ]
 * change the speed, and - and + zoom out and in. Hidden when the scene can't draw.
 */
export default function SceneControls() {
  const [paused, setPaused] = useState(sceneState.paused);
  const [speed, setSpeed] = useState(sceneState.speed);
  const [watching, setWatching] = useState(sceneState.watching);
  const [endless, setEndless] = useState(sceneState.endless);
  useEffect(() => {
    const on = () => setWatching(sceneState.watching);
    window.addEventListener("scenewatch", on);
    return () => window.removeEventListener("scenewatch", on);
  }, []);
  // The scene reports how far it can zoom after each resize
  const [zoom, setZoom] = useState({ pct: 100, canIn: sceneState.zoom > 0, canOut: true });
  const zoomBy = (d: number) => {
    sceneState.zoom = Math.max(0, Math.min(8, sceneState.zoom + d));
    setZoom((z) => ({ ...z, canIn: sceneState.zoom > 0 }));
  };
  useEffect(() => {
    const on = (e: Event) => {
      setZoom((e as CustomEvent<{ pct: number; canIn: boolean; canOut: boolean }>).detail);
    };
    window.addEventListener("scenezoom", on);
    return () => window.removeEventListener("scenezoom", on);
  }, []);
  const toggle = () => {
    sceneState.paused = !sceneState.paused;
    setPaused(sceneState.paused);
  };
  const pick = (s: number) => {
    sceneState.speed = s;
    setSpeed(s);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (hasModifier(e) || isTypingTarget(e.target) || isModalOpen()) return;
      if (e.key === "p" || e.key === "P") toggle();
      else if (e.key === "-" || e.key === "_") zoomBy(1);
      else if (e.key === "=" || e.key === "+") zoomBy(-1);
      else if (e.key === "[" || e.key === "]") {
        const i = SPEEDS.indexOf(sceneState.speed as (typeof SPEEDS)[number]);
        pick(SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, (i < 0 ? 1 : i) + (e.key === "]" ? 1 : -1)))]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!sceneState.canvas || !watching) return null;
  // A landmark of its own, as it sits outside <main>. A toggle either names what it will do, like
  // Pause and Resume, or keeps its name and is pressed or not, like Endless and the speeds. Never
  // both: "Resume, pressed" would read as resuming being on.
  return (
    <div className="scene-controls" role="region" aria-label="Background animation">
      <button
        type="button"
        className={cn("scene-controls__play", paused && "is-on")}
        onClick={toggle}
        aria-label={paused ? "Resume the background" : "Pause the background"}
      >
        {paused ? (
          <Play className="h-3.5 w-3.5" aria-hidden="true" />
        ) : (
          <Pause className="h-3.5 w-3.5" aria-hidden="true" />
        )}
        <span>{paused ? "Resume" : "Pause"}</span>
      </button>
      <button
        type="button"
        className="scene-controls__endless"
        aria-pressed={endless}
        aria-label="Endless levels: no finale every 20"
        onClick={() => {
          sceneState.endless = !sceneState.endless;
          setEndless(sceneState.endless);
        }}
      >
        <Endless className="h-3.5 w-3.5" aria-hidden="true" />
        <span>Endless</span>
      </button>
      <span className="scene-controls__rule" aria-hidden="true" />
      <span className="scene-controls__speeds" role="group" aria-label="Speed">
        <span className="scene-controls__label" aria-hidden="true">
          Speed
        </span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            className={cn("scene-controls__speed", speed === s && "is-on")}
            onClick={() => pick(s)}
            aria-pressed={speed === s}
          >
            {s === 0.5 ? "½×" : `${s}×`}
          </button>
        ))}
      </span>
      <span className="scene-controls__rule" aria-hidden="true" />
      <button
        type="button"
        onClick={() => zoomBy(1)}
        disabled={!zoom.canOut}
        aria-label="Zoom out: see more of the battlefield"
      >
        <Minus className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
      <span className="scene-controls__label scene-controls__zoom" aria-live="polite">
        {zoom.pct}%
      </span>
      <button type="button" onClick={() => zoomBy(-1)} disabled={!zoom.canIn} aria-label="Zoom in">
        <Plus className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
    </div>
  );
}
