import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { Project } from "@/types/project";
import { cn } from "@/lib/utils";
import { DEFAULT_SIZE, layoutGraph, PERIOD, STEP, type GraphEdge } from "@/components/fx/graphLayout";

interface Props {
  project: Project;
}

/** A node label: one or two lines, vertically centred on the node. */
function Label({
  lines,
  x,
  anchor,
  lineH,
}: {
  lines: string[];
  x: number;
  anchor: "start" | "middle" | "end";
  lineH: number;
}) {
  // 0.35em drops a single line's baseline to the node's centre; extra lines shift it up
  const first = 0.35 * (lineH / 1.25) - ((lines.length - 1) * lineH) / 2;
  return (
    <text className="pgraph__label" textAnchor={anchor}>
      {lines.map((l, i) => (
        <tspan key={i} x={x} dy={i === 0 ? first : lineH}>
          {l}
        </tspan>
      ))}
    </text>
  );
}

/**
 * The project as a small neural network, laid out by layoutGraph. Signals fire left to right
 * in a loop while it is on screen, and hovering a node lights up its connections.
 */
export default function ProjectGraph({ project }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(DEFAULT_SIZE);
  const [live, setLive] = useState(false);
  const [hot, setHot] = useState<string | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0)
        setSize((s) => (Math.abs(s.w - r.width) < 1 && Math.abs(s.h - r.height) < 1 ? s : { w: r.width, h: r.height }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    // Only animate while the graph is on screen
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting));
    io.observe(el);
    return () => {
      ro.disconnect();
      io.disconnect();
    };
  }, []);

  const g = useMemo(() => layoutGraph(project, size.w, size.h), [project, size.w, size.h]);
  // The hovered node plus everything it is wired to
  const linked = useMemo(() => {
    if (!hot) return null;
    const set = new Set<string>([hot]);
    for (const e of g.edges) if (e.from === hot || e.to === hot) set.add(e.from).add(e.to);
    return set;
  }, [hot, g.edges]);
  const isHotEdge = (e: GraphEdge) => hot !== null && (e.from === hot || e.to === hot);

  const { inputs, hidden, outputs } = g.counts;
  const label = `Neural network diagram of the project: ${inputs} tools feed ${hidden} technologies, producing ${outputs} concepts.`;

  return (
    <div ref={ref} className="pgraph" data-live={live ? "" : undefined} data-hot={hot ? "" : undefined}>
      <svg
        role="img"
        aria-label={label}
        viewBox={`0 0 ${size.w.toFixed(0)} ${size.h.toFixed(0)}`}
        width="100%"
        height="100%"
        style={{ "--fs": `${g.font}px` } as CSSProperties}
      >
        <g className="pgraph__heads" aria-hidden="true">
          <text x={g.heads.in} y={g.heads.y} textAnchor="end">
            {g.heads.text[0]}
          </text>
          <text x={g.heads.hidden} y={g.heads.y} textAnchor="middle">
            {g.heads.text[1]}
          </text>
          <text x={g.heads.out} y={g.heads.y} textAnchor="start">
            {g.heads.text[2]}
          </text>
        </g>

        <g className="pgraph__edges" aria-hidden="true">
          {g.edges.map((e) => (
            <path key={e.key} d={e.d} className={isHotEdge(e) ? "is-hot" : undefined} />
          ))}
        </g>

        <g className="pgraph__pulses" aria-hidden="true">
          {g.edges
            .filter((e) => e.pulse)
            .map((e) => (
              <path
                key={e.key}
                d={e.d}
                pathLength={100}
                style={
                  {
                    animationDelay: `${(e.step * STEP + e.jitter * 0.18).toFixed(2)}s`,
                    animationDuration: `${PERIOD}s`,
                  } as CSSProperties
                }
              />
            ))}
        </g>

        <g>
          {g.nodes.map((n) => {
            const cls = cn("pgraph__node", `pgraph__node--l${n.layer}`, linked?.has(n.key) && "is-hot");
            const delay = {
              animationDelay: `${(n.step * STEP).toFixed(2)}s`,
              animationDuration: `${PERIOD}s`,
            } as CSSProperties;
            const enter = () => setHot(n.key);
            const leave = () => setHot((k) => (k === n.key ? null : k));
            return (
              <g
                key={n.key}
                className={cls}
                transform={`translate(${n.x.toFixed(1)} ${n.y.toFixed(1)})`}
                onPointerEnter={enter}
                onPointerLeave={leave}
              >
                {n.clipped ? <title>{n.full}</title> : null}
                {n.w && n.h ? (
                  <>
                    <rect className="pgraph__pill" x={-n.w / 2} y={-n.h / 2} width={n.w} height={n.h} rx={2} />
                    <rect
                      className="pgraph__glow"
                      x={-n.w / 2}
                      y={-n.h / 2}
                      width={n.w}
                      height={n.h}
                      rx={2}
                      style={delay}
                    />
                    <Label lines={n.lines!} x={0} anchor="middle" lineH={g.lineH} />
                  </>
                ) : (
                  <>
                    <circle className="pgraph__halo" r={n.layer === 2 ? 7 : 10} style={delay} />
                    <circle className="pgraph__dot" r={n.layer === 2 ? 2.5 : 4} />
                    {n.lines ? (
                      <Label
                        lines={n.lines}
                        x={n.layer === 0 ? -16 : 16}
                        anchor={n.layer === 0 ? "end" : "start"}
                        lineH={g.lineH}
                      />
                    ) : null}
                  </>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
