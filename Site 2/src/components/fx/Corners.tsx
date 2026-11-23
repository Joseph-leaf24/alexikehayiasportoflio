import { cn } from "@/lib/utils";

/** Four HUD corner brackets. Parent must be `position: relative`. */
export default function Corners({ className }: { className?: string }) {
  return (
    <span className={cn("corners", className)} aria-hidden="true">
      <i />
      <i />
      <i />
      <i />
    </span>
  );
}
