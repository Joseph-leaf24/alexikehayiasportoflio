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
import { sound } from "./lib/sound";

function Shell() {
  useEffect(() => {
    const stopScroll = initSmoothScroll();
    sound.restorePreference();

    const onPointer = (e: PointerEvent) => {
      sceneState.pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
      sceneState.pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
      if (e.pointerType === "mouse") sceneState.pointer.lastMove = performance.now();
    };
    window.addEventListener("pointermove", onPointer, { passive: true });
    return () => {
      stopScroll();
      window.removeEventListener("pointermove", onPointer);
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
