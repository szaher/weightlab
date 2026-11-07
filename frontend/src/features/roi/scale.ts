// Colour scales for an exact ROI heatmap (doc 08 sections 5 and 9): no rainbow; a perceptually ordered
// sequential palette for magnitudes and a balanced diverging palette for signed values. "Auto" uses only the
// finite values of the region currently loaded: it is never a whole-tensor statistic and never scans further.
import type { RoiGrid } from "./values";

export type ScaleMode = "auto" | "symmetric" | "fixed";
export type Palette = "sequential" | "diverging";

export interface Scale {
  mode: ScaleMode;
  lo: number;
  hi: number;
  /** "empty": no colourable value; "constant": lo === hi (every coloured cell gets the middle colour). */
  degenerate: "none" | "constant" | "empty";
  below: number; // coloured cells clipped at lo (fixed mode)
  above: number; // coloured cells clipped at hi (fixed mode)
}

export interface FixedRange {
  min: string;
  max: string;
}

/** Validates the text of a fixed range. */
export function fixedRangeError(f: FixedRange): string | null {
  const lo = Number(f.min);
  const hi = Number(f.max);
  if (f.min.trim() === "" || f.max.trim() === "" || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    return "enter finite numbers for minimum and maximum";
  }
  if (!(lo < hi)) return "minimum must be less than maximum";
  return null;
}

/** The scale of `grid` for `mode`. Only finite, safely convertible values take part. */
export function computeScale(grid: RoiGrid, mode: ScaleMode, fixed?: FixedRange): Scale {
  let min = Infinity;
  let max = -Infinity;
  let n = 0;
  for (const c of grid.cells) {
    if (c.color === null) continue;
    n++;
    if (c.color < min) min = c.color;
    if (c.color > max) max = c.color;
  }
  if (mode === "fixed" && fixed && fixedRangeError(fixed) === null) {
    const lo = Number(fixed.min);
    const hi = Number(fixed.max);
    let below = 0;
    let above = 0;
    for (const c of grid.cells) {
      if (c.color === null) continue;
      if (c.color < lo) below++;
      else if (c.color > hi) above++;
    }
    return { mode, lo, hi, degenerate: n === 0 ? "empty" : "none", below, above };
  }
  if (n === 0) return { mode, lo: 0, hi: 0, degenerate: "empty", below: 0, above: 0 };
  if (mode === "symmetric") {
    const m = Math.max(Math.abs(min), Math.abs(max));
    return { mode, lo: -m, hi: m, degenerate: m === 0 ? "constant" : "none", below: 0, above: 0 };
  }
  return { mode: mode === "fixed" ? "auto" : mode, lo: min, hi: max, degenerate: min === max ? "constant" : "none", below: 0, above: 0 };
}

/** Position of `v` on the scale in [0, 1]. Halved operands keep the span finite even for ±1.8e308. */
export function normalize(v: number, s: Scale): number {
  if (s.degenerate !== "none") return 0.5;
  const t = (v / 2 - s.lo / 2) / (s.hi / 2 - s.lo / 2);
  return t <= 0 ? 0 : t >= 1 ? 1 : t;
}

type RGB = readonly [number, number, number];

// Viridis (sequential, perceptually uniform, colour-vision-deficiency friendly), 9 stops.
const SEQUENTIAL: readonly RGB[] = [
  [68, 1, 84],
  [71, 44, 122],
  [59, 81, 139],
  [44, 113, 142],
  [33, 144, 141],
  [39, 173, 129],
  [92, 200, 99],
  [170, 220, 50],
  [253, 231, 37],
];

// Blue - white - red (ColorBrewer RdBu, reversed so negative is blue), 11 stops.
const DIVERGING: readonly RGB[] = [
  [5, 48, 97],
  [33, 102, 172],
  [67, 147, 195],
  [146, 197, 222],
  [209, 229, 240],
  [247, 247, 247],
  [253, 219, 199],
  [244, 165, 130],
  [214, 96, 77],
  [178, 24, 43],
  [103, 0, 31],
];

function lut(stops: readonly RGB[]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const f = x - k;
    const [ar, ag, ab] = stops[k] ?? [0, 0, 0];
    const [br, bg, bb] = stops[k + 1] ?? [0, 0, 0];
    out[i * 3] = Math.round(ar + (br - ar) * f);
    out[i * 3 + 1] = Math.round(ag + (bg - ag) * f);
    out[i * 3 + 2] = Math.round(ab + (bb - ab) * f);
  }
  return out;
}

const LUTS: Record<Palette, Uint8ClampedArray> = { sequential: lut(SEQUENTIAL), diverging: lut(DIVERGING) };

/** CSS colour of position t in [0, 1]. */
export function colorAt(t: number, palette: Palette): string {
  const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
  const l = LUTS[palette];
  return `rgb(${l[i]}, ${l[i + 1]}, ${l[i + 2]})`;
}

export const PALETTE_LABEL: Record<Palette, string> = {
  sequential: "Sequential (viridis)",
  diverging: "Diverging (blue – white – red)",
};

export const MODE_LABEL: Record<ScaleMode, string> = {
  auto: "Auto: min–max of this region",
  symmetric: "Sign-symmetric: ±max |value| of this region",
  fixed: "Fixed minimum / maximum",
};
