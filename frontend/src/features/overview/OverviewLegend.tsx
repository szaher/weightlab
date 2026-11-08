import { useEffect, useRef } from "react";
import { drawPattern, readTheme } from "../roi/paint";
import { colorAt, type Palette, type Scale } from "../roi/scale";
import { floatText } from "../roi/values";
import type { Aggregation } from "../../lib/api/types";
import { AGG_LABEL } from "./labels";
import { NO_FINITE_RGB, NOT_REPR_RGB } from "./paintOverview";

export interface LegendCounts {
  drawable: number;
  noFinite: number;
  notRepresentable: number;
  mixed: number;
  withNan: number;
  withPosInf: number;
  withNegInf: number;
}

function Bar({ palette }: { palette: Palette }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    for (let x = 0; x < c.width; x++) {
      ctx.fillStyle = colorAt(x / (c.width - 1), palette);
      ctx.fillRect(x, 0, 1, c.height);
    }
  }, [palette]);
  return <canvas ref={ref} width={256} height={12} className="legend-bar" aria-hidden="true" />;
}

function Swatch({ kind }: { kind: "no_finite" | "not_repr" | "mixed" }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const rgb = kind === "not_repr" ? NOT_REPR_RGB : kind === "no_finite" ? NO_FINITE_RGB : [92, 200, 99];
    ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    ctx.fillRect(0, 0, c.width, c.height);
    const t = readTheme(c);
    if (kind === "not_repr") drawPattern(ctx, "overflow", 0, 0, c.width, c.height, t, 1);
    else if (kind === "no_finite") drawPattern(ctx, "nan", 0, 0, c.width, c.height, t, 1);
    else {
      ctx.fillStyle = t.nfFg;
      ctx.beginPath();
      ctx.moveTo(10, 0);
      ctx.lineTo(16, 0);
      ctx.lineTo(16, 6);
      ctx.fill();
    }
  }, [kind]);
  return <canvas ref={ref} width={16} height={16} className="legend-swatch" aria-hidden="true" />;
}

/** Colour legend of the overview: what the colours mean (aggregation and visible-region scale) and every
 * non-colour cell kind with its count in the visible tiles. */
export function OverviewLegend({ scale, palette, aggregation, counts }: { scale: Scale; palette: Palette; aggregation: Aggregation; counts: LegendCounts }) {
  const origin =
    scale.mode === "fixed"
      ? "fixed range you entered"
      : scale.mode === "symmetric"
        ? "visible-region scale: ± the largest |value| among the tiles on screen, not a whole-tensor statistic"
        : "visible-region scale: minimum to maximum of the tiles on screen, not a whole-tensor statistic";
  return (
    <div className="legend" role="group" aria-label="Overview colour legend" data-testid="overview-legend">
      {scale.degenerate === "empty" ? (
        <p className="legend-text">No colourable cell is loaded on screen yet.</p>
      ) : scale.degenerate === "constant" ? (
        <p className="legend-text">
          Every coloured cell on screen has {AGG_LABEL[aggregation]} <code>{floatText(scale.lo === -scale.hi ? 0 : scale.lo)}</code>; drawn in the middle colour.
        </p>
      ) : (
        <div className="legend-scale">
          <span className="mono" data-testid="overview-legend-lo">
            {floatText(scale.lo)}
          </span>
          <Bar palette={palette} />
          <span className="mono" data-testid="overview-legend-hi">
            {floatText(scale.hi)}
          </span>
          <span className="help">({AGG_LABEL[aggregation]} per block)</span>
        </div>
      )}
      <p className="legend-text help" data-testid="overview-scale-origin">
        Scale: {origin}.{scale.below + scale.above > 0 ? ` ${scale.below} cell(s) below and ${scale.above} above the range are drawn at the ends.` : ""}
      </p>
      <ul className="legend-special">
        <li className={counts.noFinite === 0 ? "muted" : undefined}>
          <Swatch kind="no_finite" /> no finite element in the block (NaN/±Inf only): {counts.noFinite}
        </li>
        <li className={counts.notRepresentable === 0 ? "muted" : undefined}>
          <Swatch kind="not_repr" /> not representable as float32: {counts.notRepresentable}
        </li>
        <li className={counts.mixed === 0 ? "muted" : undefined}>
          <Swatch kind="mixed" /> finite aggregate, block also holds NaN/±Inf (corner mark): {counts.mixed}
        </li>
      </ul>
      <p className="help">
        Blocks with NaN: {counts.withNan} · +Inf: {counts.withPosInf} · −Inf: {counts.withNegInf}. Patterns appear when cells are at least 6 px wide; hover or
        select a cell for its status in words.
      </p>
    </div>
  );
}
