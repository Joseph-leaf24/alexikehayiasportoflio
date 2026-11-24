import { Component, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { TLink } from "./Transition";

interface State {
  failed: boolean;
}

/** Catches a render error in the routed page so the rest of the site (3D scene, HUD) stays up. */
class Boundary extends Component<{ children: ReactNode }, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error("Page failed to render:", error);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <section className="notfound container-x" role="alert">
        <span className="mono-label">[ Error ]</span>
        <h1 className="display-2">Something went wrong</h1>
        <p className="notfound__text">This page couldn't be shown.</p>
        <TLink to="/" className="btn-ice btn-ice--solid">
          <ArrowLeft className="h-4 w-4" /> Return to Home
        </TLink>
      </section>
    );
  }
}

/** Error boundary around the routes; it resets on every navigation, so moving on recovers. */
export default function PageBoundary({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  return <Boundary key={pathname}>{children}</Boundary>;
}
