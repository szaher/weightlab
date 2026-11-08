// Builders for binary tile responses (the wire format of doc 09 section 3), written independently of the decoder:
// values via DataView little-endian, then one status byte per cell.
import type { TileRequest } from "../lib/api/types";

export const KEY = "a".repeat(64);

export function tileBody(values: number[], status: number[]): ArrayBuffer {
  const n = values.length;
  const buf = new ArrayBuffer(n * 5);
  const dv = new DataView(buf);
  values.forEach((v, i) => dv.setFloat32(i * 4, v, true));
  status.forEach((s, i) => dv.setUint8(n * 4 + i, s));
  return buf;
}

export interface TileOpts {
  rows: number;
  cols: number;
  level: number;
  x?: number;
  y?: number;
  viewRows: number;
  viewCols: number;
  quality?: "exact" | "row_sampled";
  aggregation?: string;
  sampledRows?: number;
  cacheKey?: string;
}

/** Headers of a correct tile for (level, x, y) in a viewRows x viewCols view. */
export function tileHeaders(o: TileOpts): Record<string, string> {
  const span = 256 * 2 ** o.level;
  const r0 = (o.y ?? 0) * span;
  const c0 = (o.x ?? 0) * span;
  const h: Record<string, string> = {
    "content-type": "application/vnd.weightlab.tile.v2",
    "x-weightlab-rows": String(o.rows),
    "x-weightlab-cols": String(o.cols),
    "x-weightlab-row-start": String(r0),
    "x-weightlab-row-end": String(Math.min(o.viewRows, r0 + span)),
    "x-weightlab-col-start": String(c0),
    "x-weightlab-col-end": String(Math.min(o.viewCols, c0 + span)),
    "x-weightlab-level": String(o.level),
    "x-weightlab-quality": o.quality ?? "exact",
    "x-weightlab-value-precision": "float32_visualization",
    "x-weightlab-aggregation": o.aggregation ?? "mean",
    "x-weightlab-cache-key": o.cacheKey ?? KEY,
  };
  if ((o.quality ?? "exact") === "row_sampled") h["x-weightlab-sampled-rows"] = String(o.sampledRows ?? 4);
  return h;
}

/** A complete 200 tile answering `req` in a viewRows x viewCols view, every cell = fill(i, j) (status 0). */
export function tileResponse(req: TileRequest, view: { rows: number; cols: number }, extra: Partial<TileOpts> = {}, fill: (i: number, j: number) => number = (i, j) => i + j / 1000): Response {
  const { level, x, y } = req.tile;
  const b = 2 ** level;
  const span = 256 * b;
  const rows = Math.ceil((Math.min(view.rows, y * span + span) - y * span) / b);
  const cols = Math.ceil((Math.min(view.cols, x * span + span) - x * span) / b);
  const values: number[] = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) values.push(fill(i, j));
  const headers = tileHeaders({ rows, cols, level, x, y, viewRows: view.rows, viewCols: view.cols, aggregation: req.aggregation, ...extra });
  return new Response(tileBody(values, new Array<number>(rows * cols).fill(0)), { status: 200, headers });
}
