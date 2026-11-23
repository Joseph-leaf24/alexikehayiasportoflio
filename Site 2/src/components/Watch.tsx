import { useEffect, useRef, useState } from "react";
import SectionHead from "@/components/fx/SectionHead";
import { sceneState } from "@/lib/store";

type Awake = "on" | "off" | "unsupported";

const pad = (n: number) => String(n).padStart(2, "0");
const clock = (s: number) => `${Math.floor(s / 60)}:${pad(Math.floor(s % 60))}`;
const hms = (s: number) => `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(Math.floor(s % 60))}`;
const BOSS_STATE = { none: "None", ahead: "Ahead", engaged: "Engaged", destroyed: "Destroyed", bypassed: "Slipped past" };
const AWAKE: Record<Awake, string> = { on: "Kept awake", off: "May sleep", unsupported: "May sleep" };

/** What the panel shows, read from the scene. */
const snapshot = () => ({ ...sceneState.intel, level: sceneState.level, stage: sceneState.stage, kills: sceneState.kills });

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
 * The spectator space at the very bottom of the home page: open ground where the scene fills
 * the screen and you can leave it running. While it's in view the scene keeps its full frame
 * rate, the screen is kept awake, a panel follows the level in play (objective, progress, the
 * boss, the numbers, the last few levels cleared), and the end-of-level white-out takes the
 * HUD with it so the whole screen goes white.
 */
export default function Watch() {
  const stageRef = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);
  // (on phones the panel would cover the battle: it starts folded away there)
  const [open, setOpen] = useState(() => window.innerWidth >= 720);
  const [intel, setIntel] = useState(snapshot);
  const awake = useWakeLock(inView);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    sceneState.watching = inView;
    if (!inView) return;
    const root = document.documentElement;
    const tick = () => {
      setIntel(snapshot());
      root.classList.toggle("is-whiteout", sceneState.white > 0.6);
    };
    tick();
    const id = window.setInterval(tick, 250);
    return () => {
      window.clearInterval(id);
      sceneState.watching = false;
      root.classList.remove("is-whiteout");
    };
  }, [inView]);

  const pct = Math.round(intel.progress * 100);
  return (
    <section id="watch" className="section section--watch" data-scene="watch" data-section="06|Spectate" aria-labelledby="watch-title">
      <div className="container-x">
        <SectionHead index="06" label="Spectate" />
        <h2 id="watch-title" className="display-2">
          Watch the battle
        </h2>
        <p className="watch__text">
          This is the end of the page, but not of the fight. Scroll on into the open ground below and leave it running: the squad keeps fighting
          through random levels (jungles, temples, rivers, strongholds) for as long as you like.
        </p>
      </div>

      <div ref={stageRef} className="watch__stage">
        <div className="watch__dock container-x">
          {open ? (
            <aside className="watch-intel" aria-label="Level intel">
              <header className="watch-intel__head">
                <span className="watch-intel__live">Live</span>
                <span className="watch-intel__level">Level {pad(intel.level)}</span>
                <span className="watch-intel__stage">{intel.stage}</span>
              </header>
              <p className="watch-intel__objective">{intel.objective}</p>
              <div className="watch-intel__progress">
                <span className="watch-intel__bar" aria-hidden="true">
                  <i style={{ transform: `scaleX(${Math.round(intel.progress * 10) / 10})` }} />
                </span>
                <span>{pct}% to the stronghold</span>
              </div>
              <dl className="watch-intel__stats">
                <div>
                  <dt>Boss</dt>
                  <dd>{intel.boss ? `${intel.boss}: ${BOSS_STATE[intel.bossState]}` : "None"}</dd>
                </div>
                <div>
                  <dt>Down this level</dt>
                  <dd>{intel.levelKills}</dd>
                </div>
                <div>
                  <dt>Hostiles in sight</dt>
                  <dd>{intel.hostiles}</dd>
                </div>
                <div>
                  <dt>Level time</dt>
                  <dd>{clock(intel.levelTime)}</dd>
                </div>
                <div>
                  <dt>Conditions</dt>
                  <dd>
                    {intel.light}, {intel.weather.toLowerCase()}
                  </dd>
                </div>
                <div>
                  <dt>Levels cleared</dt>
                  <dd>{intel.cleared}</dd>
                </div>
                <div>
                  <dt>Enemies down</dt>
                  <dd>{intel.kills.toLocaleString("en-GB")}</dd>
                </div>
                <div>
                  <dt>Battle time</dt>
                  <dd>{hms(intel.uptime)}</dd>
                </div>
                <div>
                  <dt>Screen</dt>
                  <dd>{AWAKE[awake]}</dd>
                </div>
              </dl>
              {intel.log.length > 0 && (
                <ol className="watch-intel__log" aria-label="Levels cleared recently">
                  {intel.log.map((l) => (
                    <li key={l.no}>
                      <b>{pad(l.no)}</b> {l.name}
                      <span>
                        {l.kills} down, {clock(l.time)}
                      </span>
                    </li>
                  ))}
                </ol>
              )}
              <button type="button" className="btn-px btn-px--sm watch-intel__toggle" aria-expanded="true" onClick={() => setOpen(false)}>
                Hide intel
              </button>
            </aside>
          ) : (
            <button type="button" className="btn-px btn-px--sm watch-intel__show" aria-expanded="false" onClick={() => setOpen(true)}>
              Show level intel
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
