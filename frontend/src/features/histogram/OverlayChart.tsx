// A and B histograms on one axis: A as filled bars, B as a 2px step outline (shape and colour both identify the
// series), one count scale. Only called with bins that are exactly the same intervals (overlay.sameBins). Hover or
// the arrow keys read a bin's interval, both counts and B − A; the table in the caller is the accessible equivalent.
import { useId, useState, type KeyboardEvent } from "react";
import { toNumber, type HistogramValue } from "../../lib/api/histogram";
import { formatInt } from "../../lib/format";
import { edgeFractions, shortValue, yFraction, yTicks, type YScale } from "./chart";

interface Props {
  edges: HistogramValue[];
  a: number[];
  b: number[];
  nameA: string;
  nameB: string;
  scale: YScale;
  describe: (i: number) => string;
}

const W = 640;
const H = 220;
const L = 64;
const R = 12;
const T = 10;
const B = 26;
const PW = W - L - R;
const PH = H - T - B;

export function OverlayChart({ edges, a, b, nameA, nameB, scale, describe }: Props) {
  const [sel, setSel] = useState<number | null>(null);
  const readoutId = useId();
  const fr = edgeFractions(edges);
  const max = Math.max(1, ...a, ...b);
  const base = T + PH;
  const y = (n: number) => base - yFraction(n, max, scale) * PH;
  const x = (i: number) => L + (fr[i] ?? 0) * PW;
  const ticks = yTicks(max, scale).filter((t) => yFraction(t, max, scale) * PH >= 10);
  const n = a.length;

  // B as one step path along the tops of its bins (down to the baseline at both ends).
  let path = `M ${x(0)} ${base}`;
  for (let i = 0; i < n; i++) path += ` L ${x(i)} ${y(b[i] ?? 0)} L ${x(i + 1)} ${y(b[i] ?? 0)}`;
  path += ` L ${x(n)} ${base}`;

  const first = edges[0] ?? 0;
  const last = edges[edges.length - 1] ?? first;
  const lo = toNumber(first);
  const hi = toNumber(last);
  const xLabels = [
    { at: 0, text: shortValue(first) },
    { at: 1, text: shortValue(last) },
  ];
  if (lo < 0 && hi > 0 && -lo / (hi - lo) > 0.12 && -lo / (hi - lo) < 0.88) xLabels.push({ at: -lo / (hi - lo), text: "0" });

  function onKey(e: KeyboardEvent<HTMLDivElement>) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key) || n === 0) return;
    e.preventDefault();
    setSel((s) => {
      if (e.key === "Home") return 0;
      if (e.key === "End") return n - 1;
      if (s === null) return e.key === "ArrowLeft" ? n - 1 : 0;
      return Math.min(n - 1, Math.max(0, s + (e.key === "ArrowRight" ? 1 : -1)));
    });
  }

  return (
    <div className="hist-chart stack-sm">
      <ul className="hist-legend" aria-label="Series">
        <li>
          <span className="hist-sw hist-sw-a" aria-hidden="true" /> A · {nameA} (filled bars)
        </li>
        <li>
          <span className="hist-sw hist-sw-b" aria-hidden="true" /> B · {nameB} (outline)
        </li>
      </ul>
      <div
        className="hist-plot"
        tabIndex={0}
        role="group"
        aria-label={`A and B histograms over the same ${n} bins, ${scale} count axis. Use the left and right arrow keys to read each bin.`}
        aria-describedby={readoutId}
        onKeyDown={onKey}
        onBlur={() => setSel(null)}
        data-testid="overlay-chart"
      >
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="presentation" onMouseLeave={() => setSel(null)}>
          {ticks.map((t) => (
            <g key={t} className="hist-grid">
              <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
              <text x={L - 6} y={y(t) + 4} textAnchor="end">
                {formatInt(t)}
              </text>
            </g>
          ))}
          {a.map((c, i) => {
            const w = x(i + 1) - x(i);
            const h = base - y(c);
            return h > 0 ? <rect key={i} className="hist-bar-a" x={x(i)} width={Math.max(w - (w >= 4 ? 1 : 0), 0.75)} y={y(c)} height={h} /> : null;
          })}
          <path className="hist-line-halo" d={path} />
          <path className="hist-line-b" d={path} data-testid="overlay-b" />
          {sel !== null ? <rect className="hist-sel" x={x(sel)} width={Math.max(x(sel + 1) - x(sel), 1)} y={T} height={PH} /> : null}
          {a.map((_, i) => (
            <rect key={i} className="hist-hit" x={x(i)} width={Math.max(x(i + 1) - x(i), 1)} y={T} height={PH} onMouseEnter={() => setSel(i)} />
          ))}
          <line className="hist-axis" x1={L} x2={W - R} y1={base} y2={base} />
          {xLabels.map((l, k) => (
            <text key={k} className="hist-xlabel" x={L + l.at * PW} y={H - 8} textAnchor={l.at <= 0.02 ? "start" : l.at >= 0.98 ? "end" : "middle"}>
              {l.text}
            </text>
          ))}
        </svg>
      </div>
      <p id={readoutId} className="hist-readout mono small" aria-live="polite" data-testid="overlay-readout">
        {sel !== null && sel < n ? describe(sel) : "Hover a bin, or focus the chart and use the arrow keys, to read A, B and B − A."}
      </p>
    </div>
  );
}
