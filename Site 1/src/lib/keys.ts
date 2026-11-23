/**
 * Keyboard helpers shared by the shortcuts, the image viewer and the shortcuts help.
 */

/** True when a key event comes from a text field, where single-key shortcuts must not fire. */
export const isTypingTarget = (target: EventTarget | null) =>
  target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));

/** True while a modal dialog (image viewer, shortcuts help) is open. */
export const isModalOpen = () => !!document.querySelector('[role="dialog"][aria-modal="true"]');

/** True if any modifier is held (so browser / OS shortcuts are left alone). */
export const hasModifier = (e: KeyboardEvent) => e.altKey || e.ctrlKey || e.metaKey;

/**
 * For modal dialogs: keep Tab / Shift+Tab cycling through `root`'s buttons and links,
 * and pull focus back in if it has escaped. Call from a keydown handler.
 */
export function trapTab(e: KeyboardEvent, root: HTMLElement | null) {
  if (e.key !== "Tab") return;
  const items = Array.from(root?.querySelectorAll<HTMLElement>("button, a[href]") ?? []);
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  const at = items.indexOf(document.activeElement as HTMLElement);
  if (at === -1 || (e.shiftKey && at === 0) || (!e.shiftKey && at === items.length - 1)) {
    e.preventDefault();
    (e.shiftKey ? last : first).focus();
  }
}
