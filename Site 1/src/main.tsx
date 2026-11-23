/** Entry point: mounts the app into #root. */
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { getQuality } from './lib/device'

// Resolve the render budget (full quality, unless ?quality= says otherwise) so <html data-tier> is set before anything paints
getQuality();

createRoot(document.getElementById("root")!).render(<App />);
