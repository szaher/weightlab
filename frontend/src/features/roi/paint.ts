// Canvas 2D drawing of an exact ROI. Geometry is computed in CSS pixels and every edge is snapped to device
// pixels, so cells stay crisp at any devicePixelRatio. Non-finite cells get a pattern (never a scale colour),
// cells whose colour is withheld get a dot; colour is never the only signal (the legend and readouts say it in text).
import { colorAt, normalize, type Palette, type Scale } from "./scale";
import type { Cell, CellState, RoiGrid } from "./values";

export const MIN_CELL = 4;
export const MAX_CELL = 32;
const MAX_HEIGHT = 640;
const CHAR_W = 7; // 11px monospace, generous
const TOP = 20;

export interface Layout {
  rows: number;
  cols: number;
  r0: number;
  c0: number;
  cell: number; // CSS px
  left: number; // CSS px reserved for row labels
  top: number; // CSS px reserved for column labels
  width: number; // CSS px of the whole canvas
  height: number;
  rowStep: number;
  colStep: number;
}

const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];

function tickStep(cell: number, minPx: number): number {
  return STEPS.find((s) => s * cell >= minPx) ?? 100000;
}

function digits(n: number): number {
  return String(Math.max(0, n)).length;
}

/** Cell size and margins for `rows` x `cols` cells in `avail` CSS px of width. */
export function computeLayout(rows: number, cols: number, r0: number, c0: number, avail: number): Layout {
  const left = Math.max(28, digits(r0 + rows - 1) * CHAR_W + 12);
  const fitW = Math.floor((Math.max(avail, 120) - left - 4) / Math.max(1, cols));
  const fitH = Math.floor(MAX_HEIGHT / Math.max(1, rows));
  const cell = Math.max(MIN_CELL, Math.min(MAX_CELL, fitW, fitH));
  const colLabel = digits(c0 + cols - 1) * CHAR_W + 10;
  return {
    rows,
    cols,
    r0,
    c0,
    cell,
    left,
    top: TOP,
    width: left + cols * cell + 2,
    height: TOP + rows * cell + 2,
    rowStep: tickStep(cell, 16),
    colStep: tickStep(cell, colLabel),
  };
}

/** The cell (window-relative) under a CSS-pixel point, or null outside the cell area. */
export function hitCell(l: Layout, x: number, y: number): { i: number; j: number } | null {
  const j = Math.floor((x - l.left) / l.cell);
  const i = Math.floor((y - l.top) / l.cell);
  if (x < l.left || y < l.top || i < 0 || j < 0 || i >= l.rows || j >= l.cols) return null;
  return { i, j };
}

export interface Theme {
  text: string;
  muted: string;
  surface: string;
  border: string;
  focus: string;
  nfBg: string;
  nfFg: string;
}

export const FALLBACK_THEME: Theme = {
  text: "#17202b",
  muted: "#4f5b69",
  surface: "#ffffff",
  border: "#d3d9e1",
  focus: "#b4530d",
  nfBg: "#e6e6e6",
  nfFg: "#222222",
};

/** What a cell is drawn as: a scale colour, or a pattern that says why it has none. */
export type Fill = { kind: "color"; color: string } | { kind: "pattern"; pattern: Exclude<CellState, "finite"> | "withheld" };

export function cellFill(cell: Cell, scale: Scale, palette: Palette): Fill {
  if (cell.state !== "finite") return { kind: "pattern", pattern: cell.state };
  if (cell.color === null) return { kind: "pattern", pattern: "withheld" };
  return { kind: "color", color: colorAt(normalize(cell.color, scale), palette) };
}

type Ctx = Pick<
  CanvasRenderingContext2D,
  | "fillStyle"
  | "strokeStyle"
  | "lineWidth"
  | "font"
  | "textAlign"
  | "textBaseline"
  | "fillRect"
  | "strokeRect"
  | "clearRect"
  | "beginPath"
  | "moveTo"
  | "lineTo"
  | "stroke"
  | "arc"
  | "fill"
  | "fillText"
  | "setTransform"
  | "save"
  | "restore"
>;

/** Draws a non-finite or withheld pattern into the device-pixel rectangle (x, y, w, h). */
export function drawPattern(ctx: Ctx, pattern: Exclude<Fill, { kind: "color" }>["pattern"], x: number, y: number, w: number, h: number, t: Theme, dpr: number): void {
  ctx.fillStyle = t.nfBg;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = t.nfFg;
  ctx.fillStyle = t.nfFg;
  ctx.lineWidth = Math.max(1, Math.round(dpr));
  const cx = x + w / 2;
  const cy = y + h / 2;
  const r = Math.max(1, Math.min(w, h) * 0.3);
  ctx.beginPath();
  switch (pattern) {
    case "nan": // two diagonals "//"
      ctx.moveTo(x, y + h);
      ctx.lineTo(x + w, y);
      ctx.moveTo(x, y + h / 2);
      ctx.lineTo(x + w / 2, y);
      ctx.moveTo(x + w / 2, y + h);
      ctx.lineTo(x + w, y + h / 2);
      ctx.stroke();
      return;
    case "pos_inf": // "+"
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.moveTo(cx, cy - r);
      ctx.lineTo(cx, cy + r);
      ctx.stroke();
      return;
    case "neg_inf": // "−"
      ctx.moveTo(cx - r, cy);
      ctx.lineTo(cx + r, cy);
      ctx.stroke();
      return;
    case "overflow": // "×"
      ctx.moveTo(cx - r, cy - r);
      ctx.lineTo(cx + r, cy + r);
      ctx.moveTo(cx + r, cy - r);
      ctx.lineTo(cx - r, cy + r);
      ctx.stroke();
      return;
    case "withheld": // a dot
      ctx.arc(cx, cy, Math.max(1, r / 2), 0, Math.PI * 2);
      ctx.fill();
      return;
  }
}

export interface PaintInput {
  layout: Layout;
  grid: RoiGrid;
  scale: Scale;
  palette: Palette;
  theme: Theme;
  dpr: number;
  selected: { i: number; j: number } | null; // window-relative
  hover: { i: number; j: number } | null;
}

/** Paints the whole heatmap. The canvas backing store must be layout size x dpr. */
export function paint(ctx: Ctx, p: PaintInput): void {
  const { layout: l, grid, dpr, theme: t } = p;
  const X = (css: number) => Math.round(css * dpr);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, X(l.width), X(l.height));
  for (let i = 0; i < l.rows; i++) {
    const y0 = X(l.top + i * l.cell);
    const y1 = X(l.top + (i + 1) * l.cell);
    for (let j = 0; j < l.cols; j++) {
      const x0 = X(l.left + j * l.cell);
      const x1 = X(l.left + (j + 1) * l.cell);
      const cell = grid.cells[i * grid.cols + j];
      if (!cell) continue;
      const f = cellFill(cell, p.scale, p.palette);
      if (f.kind === "color") {
        ctx.fillStyle = f.color;
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      } else {
        drawPattern(ctx, f.pattern, x0, y0, x1 - x0, y1 - y0, t, dpr);
      }
    }
  }
  // Frame and coordinate labels (global view indices).
  ctx.strokeStyle = t.border;
  ctx.lineWidth = Math.max(1, Math.round(dpr));
  ctx.strokeRect(X(l.left) - 0.5, X(l.top) - 0.5, X(l.left + l.cols * l.cell) - X(l.left) + 1, X(l.top + l.rows * l.cell) - X(l.top) + 1);
  ctx.fillStyle = t.muted;
  ctx.font = `${Math.round(11 * dpr)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < l.rows; i++) {
    const g = l.r0 + i;
    if (g % l.rowStep !== 0 && i !== 0) continue;
    ctx.fillText(String(g), X(l.left - 5), X(l.top + (i + 0.5) * l.cell));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (let j = 0; j < l.cols; j++) {
    const g = l.c0 + j;
    if (g % l.colStep !== 0 && j !== 0) continue;
    if (j !== 0 && (j * l.cell) < (String(l.c0).length * CHAR_W + 8)) continue; // no overlap with the first label
    ctx.fillText(String(g), X(l.left + (j + 0.5) * l.cell), X(l.top - 6));
  }
  // Hover and selection outlines (selection on top, two tones so it shows on any colour).
  const outline = (c: { i: number; j: number }, color: string, width: number) => {
    const x0 = X(l.left + c.j * l.cell);
    const y0 = X(l.top + c.i * l.cell);
    const w = X(l.left + (c.j + 1) * l.cell) - x0;
    const h = X(l.top + (c.i + 1) * l.cell) - y0;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.strokeRect(x0 + width / 2, y0 + width / 2, w - width, h - width);
  };
  if (p.hover && (!p.selected || p.hover.i !== p.selected.i || p.hover.j !== p.selected.j)) {
    outline(p.hover, t.text, Math.max(1, Math.round(dpr)));
  }
  if (p.selected) {
    outline(p.selected, t.focus, Math.max(2, Math.round(2 * dpr)));
    if (l.cell >= 8) {
      const inset = Math.max(2, Math.round(2 * dpr));
      const x0 = X(l.left + p.selected.j * l.cell) + inset;
      const y0 = X(l.top + p.selected.i * l.cell) + inset;
      const w = X(l.left + (p.selected.j + 1) * l.cell) - inset - x0;
      const h = X(l.top + (p.selected.i + 1) * l.cell) - inset - y0;
      ctx.strokeStyle = t.surface;
      ctx.lineWidth = Math.max(1, Math.round(dpr));
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    }
  }
}

/** Theme colours from the CSS custom properties (light/dark), with fallbacks outside a browser. */
export function readTheme(el: Element): Theme {
  const cs = getComputedStyle(el);
  const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
  return {
    text: v("--text", FALLBACK_THEME.text),
    muted: v("--muted", FALLBACK_THEME.muted),
    surface: v("--surface", FALLBACK_THEME.surface),
    border: v("--border", FALLBACK_THEME.border),
    focus: v("--focus", FALLBACK_THEME.focus),
    nfBg: v("--nf-bg", FALLBACK_THEME.nfBg),
    nfFg: v("--nf-fg", FALLBACK_THEME.nfFg),
  };
}
