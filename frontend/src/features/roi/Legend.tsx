import { useEffect, useRef } from "react";
import { drawPattern, readTheme } from "./paint";
import { colorAt, type Palette, type Scale } from "./scale";
import { floatText, STATE_LABEL, type RoiGrid } from "./values";

function scaleText(v: number, grid: RoiGrid): string {
  return grid.encoding === "float_json" ? floatText(v) : String(v);
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

type PatternKind = "nan" | "pos_inf" | "neg_inf" | "overflow" | "withheld";

function Swatch({ kind }: { kind: PatternKind }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    drawPattern(ctx, kind, 0, 0, c.width, c.height, readTheme(c), 1);
  }, [kind]);
  return <canvas ref={ref} width={16} height={16} className="legend-swatch" aria-hidden="true" />;
}

/** The colour legend: scale range and what it is computed from, plus every non-colour cell kind with its count. */
export function Legend({ grid, scale, palette }: { grid: RoiGrid; scale: Scale; palette: Palette }) {
  const origin =
    scale.mode === "fixed"
      ? "fixed range you entered"
      : scale.mode === "symmetric"
        ? "± the largest |finite value| in this region only, not the whole tensor"
        : "minimum to maximum of the finite values in this region only, not the whole tensor";
  const special: [PatternKind, number, string][] = [
    ["nan", grid.counts.nan, STATE_LABEL.nan],
    ["pos_inf", grid.counts.pos_inf, STATE_LABEL.pos_inf],
    ["neg_inf", grid.counts.neg_inf, STATE_LABEL.neg_inf],
    ["overflow", grid.counts.overflow, "overflow"],
    ["withheld", grid.withheld, "colour withheld: beyond ±(2^53 − 1), exact decimal text kept"],
  ];
  return (
    <div className="legend" aria-label="Colour legend" role="group" data-testid="legend">
      {scale.degenerate === "empty" ? (
        <p className="legend-text">No colourable values in this region (none finite, or all withheld): there is nothing to colour.</p>
      ) : scale.degenerate === "constant" ? (
        <p className="legend-text">
          Constant region: every coloured value here is <code>{scaleText(scale.lo === -scale.hi ? 0 : scale.lo, grid)}</code>; drawn in the
          middle colour.
        </p>
      ) : (
        <div className="legend-scale">
          <span className="mono" data-testid="legend-lo">
            {scaleText(scale.lo, grid)}
          </span>
          <Bar palette={palette} />
          <span className="mono" data-testid="legend-hi">
            {scaleText(scale.hi, grid)}
          </span>
        </div>
      )}
      <p className="legend-text help">
        Scale: {origin}.
        {scale.below + scale.above > 0 ? ` ${scale.below} cell(s) below and ${scale.above} above the range are drawn at the ends.` : ""}
      </p>
      <ul className="legend-special">
        {special.map(([kind, n, text]) => (
          <li key={kind} className={n === 0 ? "muted" : undefined}>
            <Swatch kind={kind} /> {text}: {n}
          </li>
        ))}
      </ul>
    </div>
  );
}
