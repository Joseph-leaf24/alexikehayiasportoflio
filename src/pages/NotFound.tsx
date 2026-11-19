import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { sceneState, ui } from "@/lib/store";
import { TLink } from "@/components/chrome/Transition";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";

/** 404 page: the scene drifts into its scattered "lost" formation. */
const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    sceneState.override = "lost";
    ui.set({ sectionIndex: "404", sectionLabel: "Lost" });
    return () => {
      sceneState.override = null;
    };
  }, []);

  return (
    <section className="notfound container-x">
      <span className="mono-label">[ Error 404 ]</span>
      <h1 className="display-xl">
        <ScrambleText text="Lost in the ice" trigger="mount" duration={1200} />
      </h1>
      <Reveal as="p" className="notfound__text">
        Oops! Page not found
      </Reveal>
      <Reveal as="p" className="mono-label notfound__path" delay={120}>
        {location.pathname}
      </Reveal>
      <Reveal delay={240}>
        <TLink to="/" className="btn-ice btn-ice--solid">
          <ArrowLeft className="h-4 w-4" />
          <ScrambleText text="Return to Home" trigger="none" hover />
        </TLink>
      </Reveal>
    </section>
  );
};

export default NotFound;
