import { useEffect } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import ProjectDetails from "./pages/ProjectDetails";
import IceScene from "./components/gl/IceScene";
import Hud from "./components/chrome/Hud";
import Cursor from "./components/chrome/Cursor";
import Loader from "./components/chrome/Loader";
import { TransitionProvider } from "./components/chrome/Transition";
import { initSmoothScroll } from "./lib/smoothScroll";
import { sceneState } from "./lib/store";

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
      if (e.pointerType === "mouse") sceneState.burst++;
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
      <a href="#main" className="skip-link" onClick={(e) => {
        e.preventDefault();
        document.getElementById("main")?.focus();
      }}>
        Skip to content
      </a>
      <IceScene />
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
    </>
  );
}

const App = () => (
  <HashRouter>
    <TransitionProvider>
      <Shell />
    </TransitionProvider>
  </HashRouter>
);

export default App;
