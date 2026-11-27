import { useEffect } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import ProjectDetails from "./pages/ProjectDetails";
import IceScene from "./components/gl/IceScene";
import Hud from "./components/chrome/Hud";
import Cursor from "./components/chrome/Cursor";
import Loader from "./components/chrome/Loader";
import Shortcuts from "./components/chrome/Shortcuts";
import { TransitionProvider } from "./components/chrome/Transition";
import PageBoundary from "./components/chrome/PageBoundary";
import { initSmoothScroll } from "./lib/smoothScroll";
import { sceneState } from "./lib/store";

/**
 * Everything around the routed page: the WebGL backdrop, HUD, cursor and loader. Also
 * starts smooth scrolling and feeds pointer moves and mouse clicks to the 3D scene.
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
      if (e.pointerType === "mouse" && e.button === 0) sceneState.burst++;
    };
    window.addEventListener("pointermove", onPointer, { passive: true });
    window.addEventListener("pointerdown", onDown, { passive: true });

    // Ownership notice for anyone poking around in DevTools
    console.log(
      "%c❄ Alexi Kehayias%c\n© " +
        new Date().getFullYear() +
        " Alexi Kehayias. All rights reserved.\nThe design, code and 3D work on this site are proprietary — please don't copy or redistribute them.",
      "font: 600 16px/1.6 system-ui; color: #7fd4ff",
      "font: 12px/1.6 ui-monospace, monospace; color: #8e98aa"
    );

    return () => {
      stopScroll();
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("pointerdown", onDown);
    };
  }, []);

  return (
    <>
      <a
        href="#main"
        className="skip-link"
        onClick={(e) => {
          // HashRouter would read "#main" as a route, so move focus to the landmark by hand
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <IceScene />
      <Hud />
      <main id="main" tabIndex={-1} className="site-main">
        <PageBoundary>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/project/:category/:projectIndex" element={<ProjectDetails />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </PageBoundary>
      </main>
      <Cursor />
      <Loader />
      <Shortcuts />
    </>
  );
}

/** Root component: hash routing, which works on GitHub Pages without server rewrites, plus the page-wipe transition. */
const App = () => (
  <HashRouter>
    <TransitionProvider>
      <Shell />
    </TransitionProvider>
  </HashRouter>
);

export default App;
