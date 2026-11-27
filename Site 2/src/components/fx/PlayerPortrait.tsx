import { useEffect, useRef } from "react";
import { playerSprite } from "@/components/pixel/art";
import { SQUAD } from "@/data/squad";
import { cn } from "@/lib/utils";

/**
 * One of the eight players (1–8), drawn from the same pixel art as the scene behind the page and
 * scaled up by CSS with hard edges. Decorative: the role is always written next to it.
 */
export default function PlayerPortrait({ player, className }: { player: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const img = playerSprite(player - 1, SQUAD[player - 1].color);
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  }, [player]);

  return <canvas ref={ref} width={12} height={14} className={cn("portrait-px", className)} aria-hidden="true" />;
}
