/** Entry point: mounts the app into #root. */
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { getQuality } from "./lib/device";

// Resolve the render budget before the first paint, so <html data-tier> is already set
getQuality();

createRoot(document.getElementById("root")!).render(<App />);
