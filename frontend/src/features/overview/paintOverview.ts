// Canvas 2D drawing of the whole-tensor overview. Tiles become small bitmaps (one pixel per level cell) that are
// scaled with smoothing off, so adjacent tiles join into one continuous image and partial edge cells are clipped
// at the view's true bounds. Cells that cannot be coloured get their own flat colours in the bitmap and, when a
// cell is large enough to see, the same patterns as the exact ROI heatmap. Sampled tiles get a dashed outline in
// addition to the text badge, so exact and sampled are never told apart by colour alone.
import { DRAWABLE_MASK, HAS_NAN, HAS_NEG_INF, HAS_POS_INF, NO_FINITE, NOT_REPRESENTABLE, type DecodedTile } from "../../lib/api/tiles";
import { drawPattern, type Theme } from "../roi/paint";
import { fixedRangeError, normalize, paletteLut, type FixedRange, type Palette, type Scale, type ScaleMode } from "../roi/scale";
import { block, pointOf, type Dims, type RawRect, type Size, type Viewport } from "./geometry";

export const AXIS_LEFT = 64; // CSS px for row labels
export const AXIS_TOP = 22; // CSS px for column labels

// Bitmap colours of cells without a scale colour (RGB). Distinct from both palettes; patterns add the kind.
export const NO_FINITE_RGB = [140, 140, 140] as const;
export const NOT_REPR_RGB = [255, 79, 216] as const;

/** Visible-region scale: from the drawable values of the tiles on screen, never a whole-tensor statistic. */
export function tilesScale(tiles: readonly DecodedTile[], mode: ScaleMode, fixed?: FixedRange): Scale {
  let min = Infinity;
  let max = -Infinity;
  let n = 0;
  for (const t of tiles) {
    const { values, status } = t;
    for (let i = 0; i < values.length; i++) {
      if ((status[i] ?? 0) & DRAWABLE_MASK) continue;
      const v = (values[i] ?? 0);
      n++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (mode === "fixed" && fixed && fixedRangeError(fixed) === null) {
    const lo = Number(fixed.min);
    const hi = Number(fixed.max);
    let below = 0;
    let above = 0;
    for (const t of tiles) {
      for (let i = 0; i < t.values.length; i++) {
        if ((t.status[i] ?? 0) & DRAWABLE_MASK) continue;
        if ((t.values[i] ?? 0) < lo) below++;
        else if ((t.values[i] ?? 0) > hi) above++;
      }
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

/** RGBA pixels of a tile (rows x cols, one pixel per level cell). Pure, so it is unit-tested directly. */
export function tilePixels(tile: DecodedTile, scale: Scale, palette: Palette): Uint8ClampedArray {
  const n = tile.rows * tile.cols;
  const out = new Uint8ClampedArray(n * 4);
  const lut = paletteLut(palette);
  for (let i = 0; i < n; i++) {
    const s = (tile.status[i] ?? 0);
    let r: number;
    let g: number;
    let b: number;
    if (s & NO_FINITE) [r, g, b] = NO_FINITE_RGB;
    else if (s & NOT_REPRESENTABLE) [r, g, b] = NOT_REPR_RGB;
    else {
      const t = normalize((tile.values[i] ?? 0), scale);
      if (!Number.isFinite(t)) [r, g, b] = NO_FINITE_RGB;
      else {
        const k = Math.round(t * 255) * 3;
        r = (lut[k] ?? 0);
        g = (lut[k + 1] ?? 0);
        b = (lut[k + 2] ?? 0);
      }
    }
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** A drawable bitmap of a tile; released by setting its size to 0 when evicted. */
export interface Bitmap {
  canvas: HTMLCanvasElement;
  tile: DecodedTile;
  scaleKey: string;
}

export function scaleKey(s: Scale, palette: Palette): string {
  return `${palette}:${s.mode}:${s.lo}:${s.hi}:${s.degenerate}`;
}

export function makeBitmap(tile: DecodedTile, scale: Scale, palette: Palette, reuse?: HTMLCanvasElement): Bitmap | null {
  const canvas = reuse ?? document.createElement("canvas");
  canvas.width = tile.cols;
  canvas.height = tile.rows;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(tile.cols, tile.rows);
  img.data.set(tilePixels(tile, scale, palette));
  ctx.putImageData(img, 0, 0);
  return { canvas, tile, scaleKey: scaleKey(scale, palette) };
}

export function releaseBitmap(b: Bitmap): void {
  b.canvas.width = 0;
  b.canvas.height = 0;
}

const STEPS: number[] = [];
for (let e = 0; e <= 15; e++) for (const m of [1, 2, 5]) STEPS.push(m * 10 ** e);

/** A "nice" tick step in raw elements so labels are at least `minPx` apart. */
export function tickStep(ppe: number, minPx: number): number {
  return STEPS.find((s) => s * ppe >= minPx) ?? 1e16;
}

export interface OverviewPaint {
  size: Size; // CSS px of the plot area (without axes)
  dpr: number;
  view: Viewport;
  dims: Dims;
  level: number;
  tiles: readonly { bitmap: Bitmap | null; tile: DecodedTile }[];
  selection: RawRect | null;
  hover: RawRect | null;
  theme: Theme;
  showPatterns: boolean; // cells large enough for glyphs
}

export function paintOverview(ctx: CanvasRenderingContext2D, p: OverviewPaint): void {
  const { dpr, view: v, dims: d, theme: t } = p;
  const W = AXIS_LEFT + p.size.width;
  const H = AXIS_TOP + p.size.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, Math.round(W * dpr), Math.round(H * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Plot area clipped to the view's raw bounds: partial edge cells never spill past the tensor.
  const tl = pointOf(v, 0, 0);
  const br = pointOf(v, d.rows, d.cols);
  const clip = {
    x: AXIS_LEFT + Math.max(0, tl.x),
    y: AXIS_TOP + Math.max(0, tl.y),
    x1: AXIS_LEFT + Math.min(p.size.width, br.x),
    y1: AXIS_TOP + Math.min(p.size.height, br.y),
  };
  ctx.save();
  ctx.beginPath();
  ctx.rect(clip.x, clip.y, Math.max(0, clip.x1 - clip.x), Math.max(0, clip.y1 - clip.y));
  ctx.clip();
  ctx.fillStyle = t.nfBg;
  ctx.globalAlpha = 0.35;
  ctx.fillRect(clip.x, clip.y, clip.x1 - clip.x, clip.y1 - clip.y); // not yet loaded
  ctx.globalAlpha = 1;
  ctx.imageSmoothingEnabled = false;
  const b = block(p.level);
  const cellPx = b * v.ppe;
  for (const { bitmap, tile } of p.tiles) {
    const o = pointOf(v, tile.rowStart, tile.colStart);
    const x = AXIS_LEFT + o.x;
    const y = AXIS_TOP + o.y;
    const w = tile.cols * b * v.ppe;
    const h = tile.rows * b * v.ppe;
    const sb = 2 ** tile.level * v.ppe;
    if (bitmap) ctx.drawImage(bitmap.canvas, x, y, tile.cols * sb, tile.rows * sb);
    if (tile.level === p.level && p.showPatterns && cellPx >= 6) drawCellGlyphs(ctx, tile, x, y, cellPx, t);
    if (tile.quality === "row_sampled") {
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = t.text;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, Math.max(0, Math.min(w, (tile.colEnd - tile.colStart) * v.ppe) - 1), Math.max(0, Math.min(h, (tile.rowEnd - tile.rowStart) * v.ppe) - 1));
      ctx.restore();
    }
  }
  const outline = (r: RawRect, color: string, width: number, dash?: number[]) => {
    const a = pointOf(v, r.r0, r.c0);
    const z = pointOf(v, r.r1, r.c1);
    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.strokeRect(AXIS_LEFT + a.x + width / 2, AXIS_TOP + a.y + width / 2, Math.max(1, z.x - a.x - width), Math.max(1, z.y - a.y - width));
    ctx.restore();
  };
  if (p.hover) outline(p.hover, t.text, 1);
  if (p.selection) {
    outline(p.selection, t.surface, 4);
    outline(p.selection, t.focus, 2);
  }
  ctx.restore();

  // Frame and axes: raw view coordinates (row/column indices), never tile or cell indices.
  ctx.strokeStyle = t.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(AXIS_LEFT - 0.5, AXIS_TOP - 0.5, p.size.width + 1, p.size.height + 1);
  ctx.fillStyle = t.muted;
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  const rs = tickStep(v.ppe, 18);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let r = Math.max(0, Math.ceil(v.top / rs) * rs); r <= d.rows && (r - v.top) * v.ppe <= p.size.height; r += rs) {
    const y = AXIS_TOP + (r - v.top) * v.ppe;
    ctx.fillText(String(r), AXIS_LEFT - 6, y);
    ctx.fillRect(AXIS_LEFT - 3, y, 3, 1);
  }
  const cs = tickStep(v.ppe, Math.max(60, String(d.cols).length * 8 + 16));
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (let c = Math.max(0, Math.ceil(v.left / cs) * cs); c <= d.cols && (c - v.left) * v.ppe <= p.size.width; c += cs) {
    const x = AXIS_LEFT + (c - v.left) * v.ppe;
    ctx.fillText(String(c), Math.max(AXIS_LEFT + 10, Math.min(W - 16, x)), AXIS_TOP - 7);
    ctx.fillRect(x, AXIS_TOP - 3, 1, 3);
  }
}

/** Non-finite and non-representable glyphs, and a corner mark on finite cells whose block has non-finite elements. */
function drawCellGlyphs(ctx: CanvasRenderingContext2D, tile: DecodedTile, x: number, y: number, cell: number, t: Theme): void {
  for (let i = 0; i < tile.rows; i++) {
    for (let j = 0; j < tile.cols; j++) {
      const s = (tile.status[i * tile.cols + j] ?? 0);
      if (s === 0) continue;
      const cx = x + j * cell;
      const cy = y + i * cell;
      if (s & NO_FINITE) {
        const kind = s & HAS_NAN ? "nan" : s & HAS_POS_INF ? "pos_inf" : "neg_inf";
        drawPattern(ctx, kind, cx, cy, cell, cell, t, 1);
      } else if (s & NOT_REPRESENTABLE) {
        drawPattern(ctx, "overflow", cx, cy, cell, cell, t, 1);
      } else if (s & (HAS_NAN | HAS_POS_INF | HAS_NEG_INF)) {
        const m = Math.max(3, cell * 0.3);
        ctx.fillStyle = t.nfFg;
        ctx.beginPath();
        ctx.moveTo(cx + cell - m, cy);
        ctx.lineTo(cx + cell, cy);
        ctx.lineTo(cx + cell, cy + m);
        ctx.fill();
      }
    }
  }
}
