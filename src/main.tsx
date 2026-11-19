import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { getQuality } from './lib/device'

getQuality();

createRoot(document.getElementById("root")!).render(<App />);
