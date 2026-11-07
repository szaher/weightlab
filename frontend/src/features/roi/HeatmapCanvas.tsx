import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type RefObject } from "react";
import { computeLayout, hitCell, paint, readTheme, type Layout } from "./paint";
import type { Palette, Scale } from "./scale";
import { cellText, type RoiGrid } from "./values";

export interface CellRef {
  row: number; // global view coordinates
  col: number;
}

interface Props {
  grid: RoiGrid;
  scale: Scale;
  palette: Palette;
  selected: CellRef | null;
  label: string;
  onSelect: (c: CellRef) => void;
  /** Keyboard: move the selection by (dr, dc) cells; the parent pans the window to keep it visible. */
  onMove: (dr: number, dc: number) => void;
  onZoom: (dir: "in" | "out") => void;
}

function useDevicePixelRatio(): number {
  const [dpr, setDpr] = useState(() => (typeof window === "undefined" ? 1 : window.devicePixelRatio || 1));
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(`(resolution: ${dpr}dppx)`);
    const onChange = () => setDpr(window.devicePixelRatio || 1);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [dpr]);
  return dpr;
}

function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      setWidth(Math.floor(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

/** The exact ROI as a Canvas 2D heatmap: pointer hover/click and keyboard selection, crisp at any DPR. */
export function HeatmapCanvas({ grid, scale, palette, selected, label, onSelect, onMove, onZoom }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const width = useWidth(wrap);
  const dpr = useDevicePixelRatio();
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const { r0, c0 } = grid.bounds;
  const layout: Layout = useMemo(() => computeLayout(grid.rows, grid.cols, r0, c0, width || 640), [grid.rows, grid.cols, r0, c0, width]);
  const sel = selected && selected.row >= r0 && selected.row < r0 + grid.rows && selected.col >= c0 && selected.col < c0 + grid.cols
    ? { i: selected.row - r0, j: selected.col - c0 }
    : null;

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const w = Math.round(layout.width * dpr);
    const h = Math.round(layout.height * dpr);
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    c.style.width = `${layout.width}px`;
    c.style.height = `${layout.height}px`;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    paint(ctx, { layout, grid, scale, palette, theme: readTheme(c), dpr, selected: sel, hover });
  });

  // Release the backing store when the heatmap goes away (no detached canvases holding pixels).
  useEffect(() => {
    const c = canvas.current;
    return () => {
      if (c) {
        c.width = 0;
        c.height = 0;
      }
    };
  }, []);

  function at(e: PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    return hitCell(layout, e.clientX - rect.left, e.clientY - rect.top);
  }

  function onKeyDown(e: KeyboardEvent<HTMLCanvasElement>) {
    const page = e.shiftKey ? 8 : 1;
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-page, 0],
      ArrowDown: [page, 0],
      ArrowLeft: [0, -page],
      ArrowRight: [0, page],
      PageUp: [-grid.rows, 0],
      PageDown: [grid.rows, 0],
      Home: [0, -Number.MAX_SAFE_INTEGER],
      End: [0, Number.MAX_SAFE_INTEGER],
    };
    const m = moves[e.key];
    if (m) {
      e.preventDefault();
      onMove(m[0], m[1]);
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      onZoom("in");
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault();
      onZoom("out");
    }
  }

  const hovered = hover ? grid.cells[hover.i * grid.cols + hover.j] : undefined;
  return (
    <div className="heatmap" ref={wrap}>
      <canvas
        ref={canvas}
        className="heatmap-canvas"
        role="application"
        aria-roledescription="heatmap"
        aria-label={label}
        aria-describedby="heatmap-keys"
        tabIndex={0}
        data-testid="heatmap-canvas"
        data-cell-px={layout.cell}
        data-left={layout.left}
        data-top={layout.top}
        onPointerMove={(e) => setHover(at(e))}
        onPointerLeave={() => setHover(null)}
        onPointerDown={(e) => {
          const c = at(e);
          if (c) onSelect({ row: r0 + c.i, col: c0 + c.j });
        }}
        onKeyDown={onKeyDown}
      />
      <p id="heatmap-keys" className="help">
        Click a cell, or focus the heatmap and use the arrow keys (Shift: 8 cells; Page Up/Down; Home/End) to select; + and −
        zoom. Moving past an edge pans the exact window.
      </p>
      <p className="heatmap-hover mono" aria-hidden="true" data-testid="heatmap-hover">
        {hover && hovered
          ? `Pointer: row ${r0 + hover.i}, column ${c0 + hover.j} = ${cellText(hovered)}${hovered.state !== "finite" ? "" : hovered.color === null ? " (colour withheld)" : ""}`
          : " "}
      </p>
    </div>
  );
}
