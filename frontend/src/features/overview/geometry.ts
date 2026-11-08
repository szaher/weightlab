// Whole-tensor coordinates (doc 09 section 3; the server's tile_geometry). Three coordinate systems, never mixed:
//
// * raw view coordinates: row/column indices of the 2D view (after axes, slice and transpose), half-open bounds;
// * level cells: at level L one output cell aggregates the raw block [i*2^L, (i+1)*2^L) x [j*2^L, (j+1)*2^L),
//   clipped to the view;
// * tiles: tile (x, y) at level L holds up to 256 x 256 level cells and covers raw rows from y*256*2^L and raw
//   columns from x*256*2^L. x is the column direction, y the row direction.
//
// The viewport is continuous: `ppe` CSS pixels per raw element and the raw coordinate at the canvas's top-left
// corner. Powers of two up to 2^52 and their products with tile indices are exact in float64, so every bound below
// is an exact integer for any tensor whose dimensions are safe integers.
import { TILE_SIZE } from "../../lib/api/tiles";

export interface Dims {
  rows: number;
  cols: number;
}

export interface RawRect {
  r0: number;
  r1: number;
  c0: number;
  c1: number;
}

export interface Viewport {
  ppe: number; // CSS px per raw element (1/ppe raw elements per pixel)
  top: number; // raw row at the canvas's top edge (may be fractional or negative)
  left: number; // raw column at the canvas's left edge
}

export interface Size {
  width: number; // CSS px
  height: number;
}

export const MAX_LEVEL = 52;
export const MAX_PPE = 64; // at most 64 CSS px per raw element
/** Draw cells no smaller than about one device pixel: the auto level is the finest with cells >= this many CSS px. */
export const MIN_CELL_PX = 1;

export const block = (level: number): number => 2 ** level;
export const tileSpan = (level: number, size = TILE_SIZE): number => size * 2 ** level;

/** Smallest level at which the whole view is one tile (the top of the pyramid). */
export function topLevel(d: Dims): number {
  const longest = Math.max(1, d.rows, d.cols);
  let l = 0;
  while (l < MAX_LEVEL && tileSpan(l) < longest) l++;
  return l;
}

/** Level cells of the whole view at `level`. */
export function levelCells(d: Dims, level: number): Dims {
  const b = block(level);
  return { rows: Math.ceil(d.rows / b), cols: Math.ceil(d.cols / b) };
}

/** Number of tiles across (x) and down (y) at `level`. */
export function tileGrid(d: Dims, level: number): { nx: number; ny: number } {
  const s = tileSpan(level);
  return { nx: Math.ceil(d.cols / s), ny: Math.ceil(d.rows / s) };
}

export interface TileBounds extends RawRect {
  level: number;
  x: number;
  y: number;
  rows: number; // level cells in this tile (edge tiles are clipped)
  cols: number;
}

/** Raw coverage of tile (x, y) at `level`, clipped to the view; null when the tile starts outside it. */
export function tileBounds(d: Dims, level: number, x: number, y: number): TileBounds | null {
  if (!Number.isSafeInteger(x) || !Number.isSafeInteger(y) || x < 0 || y < 0) return null;
  const s = tileSpan(level);
  const r0 = y * s;
  const c0 = x * s;
  if (r0 >= d.rows || c0 >= d.cols) return null;
  const r1 = Math.min(d.rows, r0 + s);
  const c1 = Math.min(d.cols, c0 + s);
  const b = block(level);
  return { level, x, y, r0, r1, c0, c1, rows: Math.ceil((r1 - r0) / b), cols: Math.ceil((c1 - c0) / b) };
}

/** Raw block of level cell (i, j) (global cell indices at `level`), clipped; null outside the view. */
export function cellBounds(d: Dims, level: number, i: number, j: number): RawRect | null {
  const b = block(level);
  const r0 = i * b;
  const c0 = j * b;
  if (i < 0 || j < 0 || r0 >= d.rows || c0 >= d.cols) return null;
  return { r0, r1: Math.min(d.rows, r0 + b), c0, c1: Math.min(d.cols, c0 + b) };
}

/** The level cell holding raw element (r, c). */
export function cellOf(level: number, r: number, c: number): { i: number; j: number } {
  const b = block(level);
  return { i: Math.floor(r / b), j: Math.floor(c / b) };
}

/** Raw bounds covered by the level cells [i0, i1] x [j0, j1] (inclusive cell indices, any order), clipped. */
export function cellRangeBounds(d: Dims, level: number, a: { i: number; j: number }, b: { i: number; j: number }): RawRect | null {
  const lo = cellBounds(d, level, Math.min(a.i, b.i), Math.min(a.j, b.j));
  const hi = cellBounds(d, level, Math.max(a.i, b.i), Math.max(a.j, b.j));
  if (!lo || !hi) return null;
  return { r0: lo.r0, r1: hi.r1, c0: lo.c0, c1: hi.c1 };
}

export const rectCells = (r: RawRect): number => (r.r1 - r.r0) * (r.c1 - r.c0);

// ---- viewport ---------------------------------------------------------------------------------------------

/** px per raw element that shows the whole view (the smaller of the two fits). */
export function fitPpe(d: Dims, s: Size): number {
  const w = Math.max(1, s.width);
  const h = Math.max(1, s.height);
  return Math.min(MAX_PPE, w / Math.max(1, d.cols), h / Math.max(1, d.rows));
}

/** Zoom limits: never further out than the whole view, never further in than MAX_PPE. */
export function ppeLimits(d: Dims, s: Size): { min: number; max: number } {
  const min = fitPpe(d, s);
  return { min, max: Math.max(min, MAX_PPE) };
}

/** Keeps the viewport inside its limits: on each axis the view is centred when it fits, else the viewport stays
 * within it (no empty margins), so extremely asymmetric views stay coherent. */
export function clampViewport(v: Viewport, d: Dims, s: Size): Viewport {
  const { min, max } = ppeLimits(d, s);
  const ppe = Math.min(max, Math.max(min, Number.isFinite(v.ppe) && v.ppe > 0 ? v.ppe : min));
  const visH = s.height / ppe;
  const visW = s.width / ppe;
  const top = visH >= d.rows ? (d.rows - visH) / 2 : Math.min(d.rows - visH, Math.max(0, v.top));
  const left = visW >= d.cols ? (d.cols - visW) / 2 : Math.min(d.cols - visW, Math.max(0, v.left));
  return { ppe, top, left };
}

/** The whole view, centred. */
export function fitViewport(d: Dims, s: Size): Viewport {
  return clampViewport({ ppe: fitPpe(d, s), top: 0, left: 0 }, d, s);
}

/** Finest level whose cells are at least MIN_CELL_PX wide at this zoom (cells are 2^L * ppe px), up to the top. */
export function autoLevel(v: Viewport, d: Dims): number {
  const top = topLevel(d);
  let l = 0;
  while (l < top && block(l) * v.ppe < MIN_CELL_PX) l++;
  return l;
}

/** Raw coordinate under canvas point (x, y). */
export function rawAt(v: Viewport, x: number, y: number): { r: number; c: number } {
  return { r: v.top + y / v.ppe, c: v.left + x / v.ppe };
}

/** Canvas point of raw coordinate (r, c). */
export function pointOf(v: Viewport, r: number, c: number): { x: number; y: number } {
  return { x: (c - v.left) * v.ppe, y: (r - v.top) * v.ppe };
}

/** Multiplies the zoom by `factor` keeping the raw point under (x, y) fixed (before clamping). */
export function zoomAt(v: Viewport, d: Dims, s: Size, factor: number, x: number, y: number): Viewport {
  const { r, c } = rawAt(v, x, y);
  const { min, max } = ppeLimits(d, s);
  const ppe = Math.min(max, Math.max(min, v.ppe * factor));
  return clampViewport({ ppe, top: r - y / ppe, left: c - x / ppe }, d, s);
}

/** Moves the viewport by a CSS-pixel drag (content follows the pointer). */
export function panBy(v: Viewport, d: Dims, s: Size, dx: number, dy: number): Viewport {
  return clampViewport({ ppe: v.ppe, top: v.top - dy / v.ppe, left: v.left - dx / v.ppe }, d, s);
}

/** Shows raw rectangle `r` as large as possible, centred. */
export function zoomToRect(d: Dims, s: Size, r: RawRect): Viewport {
  const h = Math.max(1, r.r1 - r.r0);
  const w = Math.max(1, r.c1 - r.c0);
  const ppe = Math.min(s.width / w, s.height / h);
  const v = { ppe, top: 0, left: 0 };
  const ppeC = clampViewport(v, d, s).ppe;
  return clampViewport({ ppe: ppeC, top: r.r0 + h / 2 - s.height / ppeC / 2, left: r.c0 + w / 2 - s.width / ppeC / 2 }, d, s);
}

/** Whole raw rows/columns at least partly visible, clipped to the view. */
export function visibleRect(v: Viewport, d: Dims, s: Size): RawRect {
  return {
    r0: Math.max(0, Math.floor(v.top)),
    r1: Math.min(d.rows, Math.ceil(v.top + s.height / v.ppe)),
    c0: Math.max(0, Math.floor(v.left)),
    c1: Math.min(d.cols, Math.ceil(v.left + s.width / v.ppe)),
  };
}

export interface TileRange {
  level: number;
  x0: number;
  x1: number; // inclusive
  y0: number;
  y1: number;
}

/** Tiles at `level` intersecting raw rectangle `r` (inclusive index ranges), or null if `r` is empty. */
export function tilesFor(r: RawRect, d: Dims, level: number): TileRange | null {
  if (r.r1 <= r.r0 || r.c1 <= r.c0) return null;
  const s = tileSpan(level);
  const { nx, ny } = tileGrid(d, level);
  return {
    level,
    x0: Math.max(0, Math.floor(r.c0 / s)),
    x1: Math.min(nx - 1, Math.floor((r.c1 - 1) / s)),
    y0: Math.max(0, Math.floor(r.r0 / s)),
    y1: Math.min(ny - 1, Math.floor((r.r1 - 1) / s)),
  };
}

export const rangeCount = (t: TileRange): number => (t.x1 - t.x0 + 1) * (t.y1 - t.y0 + 1);

/** The visible tiles plus a one-tile ring around them (prefetch), clipped to the grid. */
export function withRing(t: TileRange, d: Dims): TileRange {
  const { nx, ny } = tileGrid(d, t.level);
  return { level: t.level, x0: Math.max(0, t.x0 - 1), x1: Math.min(nx - 1, t.x1 + 1), y0: Math.max(0, t.y0 - 1), y1: Math.min(ny - 1, t.y1 + 1) };
}

/** Tile coordinates of a range, centre first (so the middle of the screen loads first). */
export function orderedTiles(t: TileRange): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  for (let y = t.y0; y <= t.y1; y++) for (let x = t.x0; x <= t.x1; x++) out.push({ x, y });
  const cx = (t.x0 + t.x1) / 2;
  const cy = (t.y0 + t.y1) / 2;
  return out.sort((a, b) => Math.hypot(a.x - cx, a.y - cy) - Math.hypot(b.x - cx, b.y - cy) || a.y - b.y || a.x - b.x);
}

/** Human description of a raw rectangle in half-open notation. */
export function rectText(r: RawRect): string {
  return `rows [${r.r0}, ${r.r1}) × columns [${r.c0}, ${r.c1})`;
}
