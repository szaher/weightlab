// One SVG bar chart for both histogram views: value bins (bars up from the baseline, at their true widths) and
// magnitude buckets (positive values up, negative values down, from a shared zero line, on one y-scale). Hover or
// the arrow keys select a column; the readout states its exact interval and counts. The table below each chart is
// the accessible equivalent, so nothing here is conveyed by the picture alone.
import { useId, useState, type KeyboardEvent } from "react";
import { formatInt } from "../../lib/format";
import { yFraction, yTicks, type YScale } from "./chart";

export interface Column {
  x0: number; // fractions of the plot width
  x1: number;
  up: number;
  down?: number;
}

interface Props {
  columns: Column[];
  scale: YScale;
  mirrored?: boolean;
  xLabels: { at: number; text: string }[];
  label: string;
  describe: (i: number) => string;
  testId: string;
}

const W = 640;
const L = 64; // y labels
const R = 12;
const T = 10;
const B = 26; // x labels
const PW = W - L - R;

export function BarChart({ columns, scale, mirrored = false, xLabels, label, describe, testId }: Props) {
  const [sel, setSel] = useState<number | null>(null);
  const readoutId = useId();
  const H = mirrored ? 260 : 210;
  const PH = H - T - B;
  const half = mirrored ? PH / 2 : PH;
  const base = T + half; // the zero line
  const max = Math.max(1, ...columns.map((c) => Math.max(c.up, c.down ?? 0)));
  // Gridlines closer than 10px to the zero line are dropped (log scale puts a count of 1 almost on it).
  const ticks = yTicks(max, scale).filter((t) => yFraction(t, max, scale) * half >= 10);

  function onKey(e: KeyboardEvent<HTMLDivElement>) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key) || columns.length === 0) return;
    e.preventDefault();
    const last = columns.length - 1;
    setSel((s) => {
      if (e.key === "Home") return 0;
      if (e.key === "End") return last;
      if (s === null) return e.key === "ArrowLeft" ? last : 0;
      return Math.min(last, Math.max(0, s + (e.key === "ArrowRight" ? 1 : -1)));
    });
  }

  return (
    <div className="hist-chart stack-sm">
      <div
        className="hist-plot"
        tabIndex={0}
        role="group"
        aria-label={`${label}. Use the left and right arrow keys to read each bar.`}
        aria-describedby={readoutId}
        onKeyDown={onKey}
        onBlur={() => setSel(null)}
        data-testid={testId}
      >
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="presentation" onMouseLeave={() => setSel(null)}>
          {ticks.map((t) => {
            const dy = yFraction(t, max, scale) * half;
            return (
              <g key={t} className="hist-grid">
                <line x1={L} x2={W - R} y1={base - dy} y2={base - dy} />
                <text x={L - 6} y={base - dy + 4} textAnchor="end">
                  {formatInt(t)}
                </text>
                {mirrored ? (
                  <>
                    <line x1={L} x2={W - R} y1={base + dy} y2={base + dy} />
                    <text x={L - 6} y={base + dy + 4} textAnchor="end">
                      {formatInt(t)}
                    </text>
                  </>
                ) : null}
              </g>
            );
          })}
          {columns.map((c, i) => {
            const x = L + c.x0 * PW;
            const w = (c.x1 - c.x0) * PW;
            const gap = w >= 4 ? 1 : 0;
            const bw = Math.max(w - gap, 0.75);
            const hu = yFraction(c.up, max, scale) * half;
            const hd = yFraction(c.down ?? 0, max, scale) * half;
            const on = sel === i;
            return (
              <g key={i} className={on ? "hist-col hist-col-on" : "hist-col"}>
                {hu > 0 ? <rect className="hist-bar" x={x} width={bw} y={base - hu} height={hu} /> : null}
                {hd > 0 ? <rect className="hist-bar hist-bar-neg" x={x} width={bw} y={base} height={hd} /> : null}
                <rect className="hist-hit" x={x} width={Math.max(w, 1)} y={T} height={PH} onMouseEnter={() => setSel(i)} />
              </g>
            );
          })}
          <line className="hist-axis" x1={L} x2={W - R} y1={base} y2={base} />
          {xLabels.map((l, k) => (
            <text key={k} className="hist-xlabel" x={L + l.at * PW} y={H - 8} textAnchor={l.at <= 0.02 ? "start" : l.at >= 0.98 ? "end" : "middle"}>
              {l.text}
            </text>
          ))}
        </svg>
      </div>
      <p id={readoutId} className="hist-readout mono small" aria-live="polite" data-testid={`${testId}-readout`}>
        {sel !== null && sel < columns.length ? describe(sel) : "Hover a bar, or focus the chart and use the arrow keys, to read its exact interval and count."}
      </p>
    </div>
  );
}
