import { useEffect } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import ProjectDetails from "./pages/ProjectDetails";
import PixelScene from "./components/pixel/PixelScene";
import Hud from "./components/chrome/Hud";
import Cursor from "./components/chrome/Cursor";
import Loader from "./components/chrome/Loader";
import Shortcuts from "./components/chrome/Shortcuts";
import { TransitionProvider } from "./components/chrome/Transition";
import { initSmoothScroll } from "./lib/smoothScroll";
import { sceneState } from "./lib/store";

/** Clicks on these don't blow anything up: they're for using the site. */
const CONTROLS = "a, button, input, select, textarea, summary, label, iframe, [role='tab'], [role='dialog']";

/**
 * Everything around the routed page: the pixel-art backdrop, HUD, cursor and loader. Also
 * starts smooth scrolling and feeds the pointer (and mouse clicks) to the scene.
 */
function Shell() {
  useEffect(() => {
    const stopScroll = initSmoothScroll();

    const onPointer = (e: PointerEvent) => {
      sceneState.pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
      sceneState.pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
      if (e.pointerType === "mouse") sceneState.pointer.lastMove = performance.now();
    };
    const onDown = (e: PointerEvent) => {
      onPointer(e);
      // A click on the page itself (not on a control) sets off a blast under the cursor
      const onControl = e.target instanceof Element && !!e.target.closest(CONTROLS);
      if (e.pointerType === "mouse" && e.button === 0 && !onControl) sceneState.burst++;
    };
    window.addEventListener("pointermove", onPointer, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });

    // Ownership notice for anyone poking around in DevTools
    console.log(
      "%c▣ Alexi Kehayias%c\n© " +
        new Date().getFullYear() +
        " Alexi Kehayias. All rights reserved.\nThe design, code and pixel art on this site are proprietary — please don't copy or redistribute them.",
      "font: 600 16px/1.6 monospace; color: #ffcc33",
      "font: 12px/1.6 ui-monospace, monospace; color: #a9a3b8"
    );

    return () => {
      stopScroll();
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("pointerdown", onDown);
    };
  }, []);

  return (
    <>
      <a href="#main" className="skip-link" onClick={(e) => {
        // HashRouter would read "#main" as a route, so move focus to the landmark by hand
        e.preventDefault();
        document.getElementById("main")?.focus();
      }}>
        Skip to content
      </a>
      <PixelScene />
      <Hud />
      <main id="main" tabIndex={-1} className="site-main">
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/project/:category/:projectIndex" element={<ProjectDetails />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Cursor />
      <Loader />
      <Shortcuts />
    </>
  );
}

/** Root component: hash routing (works on any static file server, no rewrites) plus the page-wipe transition. */
const App = () => (
  <HashRouter>
    <TransitionProvider>
      <Shell />
    </TransitionProvider>
  </HashRouter>
);

export default App;
