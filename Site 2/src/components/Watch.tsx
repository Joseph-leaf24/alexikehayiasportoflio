import { useEffect, useRef, useState } from "react";
import { sceneState } from "@/lib/store";

type Awake = "on" | "off" | "unsupported";


/**
 * Keep the screen from dimming and locking while `active`, with the Screen Wake Lock API where
 * the browser has one (the browser drops the lock whenever the tab is hidden: it is taken again
 * on return). Says whether the screen is being kept awake.
 */
function useWakeLock(active: boolean): Awake {
  const supported = typeof navigator !== "undefined" && "wakeLock" in navigator;
  const [state, setState] = useState<Awake>(supported ? "off" : "unsupported");
  useEffect(() => {
    if (!active || !supported) return;
    let lock: WakeLockSentinel | null = null;
    let done = false;
    const request = () => {
      if (done || document.visibilityState !== "visible" || (lock && !lock.released)) return;
      navigator.wakeLock.request("screen").then(
        (l) => {
          if (done) {
            l.release().catch(() => {});
            return;
          }
          lock = l;
          setState("on");
          l.addEventListener("release", () => {
            if (!done) setState("off");
          });
        },
        () => setState("off")
      );
    };
    request();
    document.addEventListener("visibilitychange", request);
    return () => {
      done = true;
      document.removeEventListener("visibilitychange", request);
      if (lock && !lock.released) lock.release().catch(() => {});
      setState("off");
    };
  }, [active, supported]);
  return state;
}

/**
 * The spectator space at the bottom of the page: open ground to leave the battle running in,
 * nothing over it. While it's in view the scene keeps its full frame rate, the screen is kept
 * awake (where the browser can), the scene controls (pause, speed, zoom) appear, and at the end
 * of each level the HUD fades with the white-out.
 */
export default function Watch() {
  const stageRef = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);
  useWakeLock(inView);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    sceneState.watching = inView;
    window.dispatchEvent(new Event("scenewatch"));
    if (!inView) return;
    const root = document.documentElement;
    const tick = () => root.classList.toggle("is-whiteout", sceneState.white > 0.6);
    tick();
    const id = window.setInterval(tick, 250);
    return () => {
      window.clearInterval(id);
      sceneState.watching = false;
      window.dispatchEvent(new Event("scenewatch"));
      root.classList.remove("is-whiteout");
    };
  }, [inView]);

  return (
    <section id="watch" className="section section--watch" data-scene="watch" data-section="06|Spectate" aria-label="The battle">
      <div ref={stageRef} className="watch__stage" />
    </section>
  );
}
