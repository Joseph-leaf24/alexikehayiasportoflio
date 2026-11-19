import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { setScrollLocked } from "@/lib/smoothScroll";

interface Props {
  images: string[];
  index: number;
  title: string;
  onClose: () => void;
  onIndex: (i: number) => void;
}

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * Full-screen gallery viewer (portalled to <body>). Locks page scroll, focuses Close, keeps
 * Tab focus inside the dialog and hands focus back on exit; Esc closes, ←/→ (and the side
 * buttons) cycle, clicking the backdrop closes.
 */
export default function Lightbox({ images, index, title, onClose, onIndex }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const count = images.length;
  const prev = () => onIndex((index - 1 + count) % count);
  const next = () => onIndex((index + 1) % count);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    setScrollLocked(true);
    closeRef.current?.focus();
    return () => {
      setScrollLocked(false);
      previouslyFocused?.focus?.();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight" && count > 1) onIndex((index + 1) % count);
      else if (e.key === "ArrowLeft" && count > 1) onIndex((index - 1 + count) % count);
      else if (e.key === "Tab") {
        // aria-modal: Tab / Shift+Tab cycle through the dialog's buttons only
        const buttons = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>("button") ?? []);
        if (!buttons.length) return;
        const first = buttons[0];
        const last = buttons[buttons.length - 1];
        const at = buttons.indexOf(document.activeElement as HTMLElement);
        if (at === -1 || (e.shiftKey && at === 0) || (!e.shiftKey && at === buttons.length - 1)) {
          e.preventDefault();
          (e.shiftKey ? last : first).focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, count, onClose, onIndex]);

  return createPortal(
    <div ref={dialogRef} className="lightbox" role="dialog" aria-modal="true" aria-label={`${title} — image viewer`} onClick={onClose}>
      <div className="lightbox__top" onClick={(e) => e.stopPropagation()}>
        <span>
          {pad(index + 1)} / {pad(count)}
        </span>
        <span className="lightbox__title">{title}</span>
        <button ref={closeRef} type="button" className="btn-ice btn-ice--sm" onClick={onClose} aria-label="Close image viewer">
          <X className="h-4 w-4" /> Close
        </button>
      </div>

      <img
        key={index}
        src={images[index]}
        alt={`${title} — image ${index + 1} of ${count}`}
        className="lightbox__img"
        onClick={(e) => e.stopPropagation()}
      />

      {count > 1 ? (
        <>
          <button
            type="button"
            className="lightbox__nav lightbox__nav--prev"
            onClick={(e) => {
              e.stopPropagation();
              prev();
            }}
            aria-label="Previous image"
          >
            <ChevronLeft className="h-6 w-6" />
          </button>
          <button
            type="button"
            className="lightbox__nav lightbox__nav--next"
            onClick={(e) => {
              e.stopPropagation();
              next();
            }}
            aria-label="Next image"
          >
            <ChevronRight className="h-6 w-6" />
          </button>
        </>
      ) : null}
    </div>,
    document.body
  );
}
