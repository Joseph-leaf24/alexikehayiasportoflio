import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { setScrollLocked } from "@/lib/smoothScroll";
import { hasModifier, isModalOpen, isTypingTarget, trapTab } from "@/lib/keys";
import { ui } from "@/lib/store";

const SHORTCUTS: { keys: string[]; label: string }[] = [
  { keys: ["←", "→"], label: "Previous / next project (on a project page)" },
  { keys: ["←", "→"], label: "Previous / next image (in the image viewer)" },
  { keys: ["Esc"], label: "Close the image viewer, menu or this help" },
  { keys: ["?"], label: "Show or hide this help" },
  { keys: ["Click"], label: "Call in a strike where you click (anywhere off the buttons)" },
  { keys: ["Hold click"], label: "Call in an airstrike (on the intro)" },
];

/** The "Keyboard shortcuts" dialog: focuses Close, traps Tab, Esc or the backdrop closes it. */
function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    setScrollLocked(true);
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else trapTab(e, dialogRef.current);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      setScrollLocked(false);
      previouslyFocused?.focus?.();
    };
  }, [onClose]);

  return createPortal(
    <div ref={dialogRef} className="shortcuts" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title" onClick={onClose}>
      <div className="shortcuts__panel" onClick={(e) => e.stopPropagation()}>
        <div className="shortcuts__head">
          <h2 id="shortcuts-title" className="mono-label">
            Keyboard shortcuts
          </h2>
          <button ref={closeRef} type="button" className="btn-px btn-px--sm" onClick={onClose} aria-label="Close keyboard shortcuts">
            <X className="h-4 w-4" /> Close
          </button>
        </div>
        <dl className="shortcuts__list">
          {SHORTCUTS.map((s) => (
            <div key={s.label}>
              <dt>
                {s.keys.map((k) => (
                  <kbd key={k}>{k}</kbd>
                ))}
              </dt>
              <dd>{s.label}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>,
    document.body
  );
}

/**
 * Global `?` key: opens (or closes) the keyboard-shortcuts help. Ignored while typing in a
 * field, with Ctrl / Alt / Cmd held, and over another overlay (the image viewer, the phone
 * menu, the loader), so two overlays never stack.
 */
export default function Shortcuts() {
  const [open, setOpen] = useState(false);
  const openRef = useRef(open);
  openRef.current = open;
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "?" || hasModifier(e) || isTypingTarget(e.target)) return;
      const { ready, menuOpen } = ui.get();
      if (!openRef.current && (isModalOpen() || menuOpen || !ready)) return;
      e.preventDefault();
      setOpen((o) => !o);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return open ? <ShortcutsDialog onClose={close} /> : null;
}
