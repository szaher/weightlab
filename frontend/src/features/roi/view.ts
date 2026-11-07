// The 2D view of a tensor and the exact window shown from it. These rules mirror the server's
// (weightlab/formats/selection.py: resolve_view / validate_roi) only so obviously invalid requests are not sent;
// the server re-validates everything and stays authoritative. No strides or element offsets are computed here.
import type { RoiBounds, RoiView } from "../../lib/api/types";

/** `fixed` holds the text of each fixed-index field (axis -> text), so an invalid entry is never "corrected". */
export interface ViewState {
  rowAxis: number | null;
  colAxis: number | null;
  fixed: Record<number, string>;
  transpose: boolean;
}

export interface Dims {
  rows: number;
  cols: number;
}

/** A requested window: top-left corner and size in view coordinates (size may be clipped at the edges). */
export interface Win {
  r0: number;
  c0: number;
  h: number;
  w: number;
}

export const DEFAULT_SIDE = 64;

/** Rank 0/1: no axes (the contract's null/null form). Rank 2: rows = axis 0. Rank > 2: the last two axes,
 * every other axis fixed at 0 (the server requires an explicit view above rank 2). */
export function defaultView(shape: readonly number[]): ViewState {
  const rank = shape.length;
  if (rank < 2) return { rowAxis: null, colAxis: null, fixed: {}, transpose: false };
  const fixed: Record<number, string> = {};
  for (let a = 0; a < rank - 2; a++) fixed[a] = "0";
  return { rowAxis: rank - 2, colAxis: rank - 1, fixed, transpose: false };
}

const INT = /^(0|[1-9][0-9]*)$/;

/** Why `view` cannot be requested for `shape`, or null when it can. */
export function viewError(shape: readonly number[], view: ViewState): string | null {
  const rank = shape.length;
  if (rank < 2) {
    if (view.rowAxis !== null || view.colAxis !== null || Object.keys(view.fixed).length > 0 || view.transpose) {
      return `a rank-${rank} tensor has no axes to choose`;
    }
    return null;
  }
  const { rowAxis: ra, colAxis: ca } = view;
  if (ra === null || ca === null || ra === ca || ra < 0 || ca < 0 || ra >= rank || ca >= rank) {
    return "choose two different axes for rows and columns";
  }
  for (let a = 0; a < rank; a++) {
    if (a === ra || a === ca) {
      if (a in view.fixed) return `axis ${a} is shown, so it cannot also be fixed`;
      continue;
    }
    const text = view.fixed[a];
    const size = shape[a] ?? 0;
    if (text === undefined) return `axis ${a} needs a fixed index`;
    if (!INT.test(text.trim())) return `axis ${a}: enter a whole number from 0 to ${size - 1}`;
    const i = Number(text.trim());
    if (i >= size) return `axis ${a}: index ${text.trim()} is outside 0–${size - 1}`;
  }
  return null;
}

/** Rows and columns of the (valid) view; transpose swaps them, as on the server. */
export function viewDims(shape: readonly number[], view: ViewState): Dims {
  if (shape.length === 0) return { rows: 1, cols: 1 };
  if (shape.length === 1) return { rows: 1, cols: shape[0] ?? 0 };
  const [ra, ca] = shownAxes(view);
  return { rows: shape[ra] ?? 0, cols: shape[ca] ?? 0 };
}

/** [axis shown as rows, axis shown as columns] after transpose. */
export function shownAxes(view: ViewState): [number, number] {
  const ra = view.rowAxis ?? 0;
  const ca = view.colAxis ?? 1;
  return view.transpose ? [ca, ra] : [ra, ca];
}

export function toRequestView(view: ViewState): RoiView {
  const fixed: Record<string, number> = {};
  for (const [axis, text] of Object.entries(view.fixed)) fixed[axis] = Number(text.trim());
  return { row_axis: view.rowAxis, col_axis: view.colAxis, fixed_indices: fixed, transpose: view.transpose };
}

/** The full tensor index of view cell (row, col): what the copied coordinates name. */
export function tensorIndex(shape: readonly number[], view: ViewState, row: number, col: number): number[] {
  if (shape.length === 0) return [];
  if (shape.length === 1) return [col];
  const [ra, ca] = shownAxes(view);
  return shape.map((_, a) => (a === ra ? row : a === ca ? col : Number(view.fixed[a] ?? 0)));
}

/** Largest default window: 64 x 64 within the tensor, halving the longer side until it fits `cap` cells. */
export function defaultWin(dims: Dims, cap: number): Win {
  let h = Math.max(1, Math.min(DEFAULT_SIDE, dims.rows));
  let w = Math.max(1, Math.min(DEFAULT_SIDE, dims.cols));
  while (h * w > cap) {
    if (w >= h && w > 1) w = Math.ceil(w / 2);
    else h = Math.ceil(h / 2);
  }
  return { r0: 0, c0: 0, h, w };
}

/** Keeps the window inside the view and the cell budget (size never below one cell). */
export function clampWin(win: Win, dims: Dims, cap: number): Win {
  let h = Math.max(1, Math.min(win.h, Math.max(1, dims.rows)));
  let w = Math.max(1, Math.min(win.w, Math.max(1, dims.cols)));
  while (h * w > cap) {
    if (w >= h && w > 1) w = Math.ceil(w / 2);
    else h = Math.ceil(h / 2);
  }
  const r0 = Math.max(0, Math.min(win.r0, dims.rows - h));
  const c0 = Math.max(0, Math.min(win.c0, dims.cols - w));
  return { r0, c0, h, w };
}

/** Half-open request bounds of a clamped window. */
export function toBounds(win: Win, dims: Dims): RoiBounds {
  return {
    r0: win.r0,
    r1: Math.min(dims.rows, win.r0 + win.h),
    c0: win.c0,
    c1: Math.min(dims.cols, win.c0 + win.w),
  };
}

/** Client-side bounds check before a request: the server's INVALID_ROI / RESOURCE_LIMIT rules. */
export function boundsError(b: RoiBounds, dims: Dims, cap: number): string | null {
  const all = [b.r0, b.r1, b.c0, b.c1];
  if (!all.every((v) => Number.isSafeInteger(v) && v >= 0)) return "coordinates must be whole numbers ≥ 0";
  if (!(b.r0 < b.r1 && b.r1 <= dims.rows && b.c0 < b.c1 && b.c1 <= dims.cols)) {
    return `region [${b.r0}, ${b.r1}) × [${b.c0}, ${b.c1}) is outside the ${dims.rows} × ${dims.cols} view`;
  }
  const cells = (b.r1 - b.r0) * (b.c1 - b.c0);
  if (cells > cap) return `region has ${cells} cells; the exact-read budget (roi_max_cells) is ${cap}`;
  return null;
}

/** Moves the window by whole cells. */
export function pan(win: Win, dims: Dims, cap: number, dr: number, dc: number): Win {
  return clampWin({ ...win, r0: win.r0 + dr, c0: win.c0 + dc }, dims, cap);
}

export type ZoomDir = "in" | "out";

/** Zoom in halves the window, zoom out doubles it (each side within the view); the anchor cell (the selected
 * one, or the centre) stays inside the window, so coordinates keep their meaning. Null when impossible. */
export function zoom(win: Win, dims: Dims, cap: number, dir: ZoomDir, anchor?: { row: number; col: number }): Win | null {
  const cur = clampWin(win, dims, cap);
  let h: number;
  let w: number;
  if (dir === "in") {
    h = Math.max(1, Math.ceil(cur.h / 2));
    w = Math.max(1, Math.ceil(cur.w / 2));
    if (h === cur.h && w === cur.w) return null;
  } else {
    const h2 = Math.min(dims.rows, cur.h * 2);
    const w2 = Math.min(dims.cols, cur.w * 2);
    // Both sides if the budget allows, else whichever single side still fits.
    const fit = ([
      [h2, w2],
      [cur.h, w2],
      [h2, cur.w],
    ] as const).find(([a, b]) => a * b <= cap && (a !== cur.h || b !== cur.w));
    if (!fit) return null;
    [h, w] = fit;
  }
  const ar = anchor?.row ?? cur.r0 + Math.floor(cur.h / 2);
  const ac = anchor?.col ?? cur.c0 + Math.floor(cur.w / 2);
  // Keep the anchor at the same relative position in the window.
  const fr = cur.h > 0 ? (ar - cur.r0) / cur.h : 0;
  const fc = cur.w > 0 ? (ac - cur.c0) / cur.w : 0;
  const next = clampWin({ r0: Math.round(ar - fr * h), c0: Math.round(ac - fc * w), h, w }, dims, cap);
  return ensureVisible(next, dims, cap, ar, ac);
}

/** The smallest shift of `win` that shows cell (row, col). */
export function ensureVisible(win: Win, dims: Dims, cap: number, row: number, col: number): Win {
  let { r0, c0 } = win;
  if (row < r0) r0 = row;
  else if (row >= r0 + win.h) r0 = row - win.h + 1;
  if (col < c0) c0 = col;
  else if (col >= c0 + win.w) c0 = col - win.w + 1;
  return clampWin({ ...win, r0, c0 }, dims, cap);
}

/** A window showing (row, col), placed so the cell is roughly centred. */
export function centreOn(win: Win, dims: Dims, cap: number, row: number, col: number): Win {
  return clampWin({ ...win, r0: row - Math.floor(win.h / 2), c0: col - Math.floor(win.w / 2) }, dims, cap);
}
