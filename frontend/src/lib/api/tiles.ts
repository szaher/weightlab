// Binary heatmap tiles (doc 09 section 3) and the jobs that build exact pyramids (section 5).
//
// POST /tiles answers 200 with `application/vnd.weightlab.tile.v2` bytes, or 202 with a JSON Job. The status
// code and Content-Type are checked before the body is read, so a binary body is never parsed as JSON and a JSON
// body is never decoded as a tile. Every binary tile is validated completely (headers against the request, body
// length, reserved status bits, drawable/non-drawable rules) before anything can draw it; a tile that fails any
// check is a BAD_RESPONSE error, never a partial picture.
import { ApiRequestError, send } from "./client";
import {
  AGGREGATIONS,
  type Aggregation,
  type Job,
  type JobState,
  type SamplingPolicy,
  type TileMetadata,
  type TileQuality,
  type TileRequest,
} from "./types";

export const TILE_FORMAT = "application/vnd.weightlab.tile.v2";
export const TILE_SIZE = 256;
export const MAX_LEVEL = 52;
/** 256 x 256 cells x (4 value bytes + 1 status byte): no valid tile body is larger. */
export const MAX_TILE_BYTES = TILE_SIZE * TILE_SIZE * 5;

// Status bits (TileStatus).
export const NO_FINITE = 1;
export const HAS_NAN = 2;
export const HAS_POS_INF = 4;
export const HAS_NEG_INF = 8;
export const NOT_REPRESENTABLE = 16;
export const RESERVED_BITS = 32 | 64 | 128;
export const DRAWABLE_MASK = NO_FINITE | NOT_REPRESENTABLE;
const KIND_BITS = HAS_NAN | HAS_POS_INF | HAS_NEG_INF;
const LEVEL0_STATUSES = new Set([0, NO_FINITE | HAS_NAN, NO_FINITE | HAS_POS_INF, NO_FINITE | HAS_NEG_INF, NOT_REPRESENTABLE]);

/** A decoded, validated tile. `values[i]` is meaningful only where `(status[i] & DRAWABLE_MASK) === 0`. */
export interface DecodedTile {
  cacheKey: string;
  level: number;
  rows: number;
  cols: number;
  rowStart: number;
  rowEnd: number;
  colStart: number;
  colEnd: number;
  quality: TileQuality;
  valuePrecision: "float32_visualization";
  aggregation: Aggregation;
  sampledRows: number | null;
  values: Float32Array;
  status: Uint8Array;
  /** Cells by kind, for legends and disclosures (counted once at decode time). */
  counts: TileCounts;
  /** Bytes held by the decoded arrays (for the browser cache budget). */
  bytes: number;
}

export interface TileCounts {
  drawable: number;
  noFinite: number;
  notRepresentable: number;
  withNan: number; // any cell whose block has a NaN (drawable or not)
  withPosInf: number;
  withNegInf: number;
  mixed: number; // drawable cells whose block also holds non-finite elements
}

export type TileResult = { kind: "tile"; tile: DecodedTile } | { kind: "job"; job: Job };

/** Raw coordinates a request must produce in a `viewRows` x `viewCols` view (the server's tile_geometry). */
export interface ExpectedGeometry {
  viewRows: number;
  viewCols: number;
}

function bad(message: string): ApiRequestError {
  return new ApiRequestError({ status: 200, code: "BAD_RESPONSE", message: `invalid tile: ${message}` });
}

const UINT = /^(0|[1-9][0-9]{0,15})$/;

function intHeader(h: Headers, name: string): number {
  const v = h.get(name);
  if (v === null) throw bad(`missing ${name}`);
  const t = v.trim();
  if (!UINT.test(t)) throw bad(`${name} is not a whole number`);
  const n = Number(t);
  if (!Number.isSafeInteger(n)) throw bad(`${name} is out of range`);
  return n;
}

function mediaType(h: Headers): string {
  return ((h.get("content-type") ?? "").split(";")[0] ?? "").trim().toLowerCase();
}

/** Decodes and validates a binary tile body against its headers and the request that produced it. */
export function decodeTile(headers: Headers, body: ArrayBuffer, req: TileRequest, expect?: ExpectedGeometry): DecodedTile {
  if (mediaType(headers) !== TILE_FORMAT) throw bad(`content type ${headers.get("content-type") ?? "(none)"}`);
  const rows = intHeader(headers, "X-WeightLab-Rows");
  const cols = intHeader(headers, "X-WeightLab-Cols");
  const level = intHeader(headers, "X-WeightLab-Level");
  const rowStart = intHeader(headers, "X-WeightLab-Row-Start");
  const rowEnd = intHeader(headers, "X-WeightLab-Row-End");
  const colStart = intHeader(headers, "X-WeightLab-Col-Start");
  const colEnd = intHeader(headers, "X-WeightLab-Col-End");
  const quality = headers.get("X-WeightLab-Quality");
  const precision = headers.get("X-WeightLab-Value-Precision");
  const aggregation = headers.get("X-WeightLab-Aggregation");
  const cacheKey = headers.get("X-WeightLab-Cache-Key") ?? "";
  if (rows < 1 || rows > TILE_SIZE || cols < 1 || cols > TILE_SIZE) throw bad(`${rows} x ${cols} cells is outside 1..${TILE_SIZE}`);
  if (level > MAX_LEVEL) throw bad(`level ${level} is above ${MAX_LEVEL}`);
  if (level !== req.tile.level) throw bad(`level ${level} answers a request for level ${req.tile.level}`);
  if (quality !== "exact" && quality !== "row_sampled") throw bad(`quality ${quality ?? "(none)"}`);
  if (req.quality.mode === "exact" && quality !== "exact") throw bad("an exact request was answered with a sampled tile");
  if (precision !== "float32_visualization") throw bad(`value precision ${precision ?? "(none)"}`);
  if (aggregation !== req.aggregation || !(AGGREGATIONS as readonly string[]).includes(aggregation)) {
    throw bad(`aggregation ${aggregation ?? "(none)"} answers a request for ${req.aggregation}`);
  }
  if (!/^[0-9a-f]{64}$/.test(cacheKey)) throw bad("cache key is not a SHA-256 hex digest");
  let sampledRows: number | null = null;
  if (quality === "row_sampled") {
    sampledRows = intHeader(headers, "X-WeightLab-Sampled-Rows");
    if (sampledRows < 1) throw bad("a sampled tile must read at least one row per block");
  } else if (headers.get("X-WeightLab-Sampled-Rows") !== null) {
    throw bad("an exact tile carries X-WeightLab-Sampled-Rows");
  }
  // Raw coverage: half-open, at the block size of this level, and exactly where the request's tile lies.
  const block = 2 ** level;
  const span = req.tile.size * block;
  const h = rowEnd - rowStart;
  const w = colEnd - colStart;
  if (h <= 0 || w <= 0) throw bad("empty or inverted raw range");
  if (Math.ceil(h / block) !== rows || Math.ceil(w / block) !== cols) throw bad("raw ranges do not match rows/cols at this level");
  if (rowStart !== req.tile.y * span || colStart !== req.tile.x * span) throw bad("raw range does not start where the requested tile does");
  if (h > span || w > span) throw bad("raw range is larger than one tile");
  if (expect) {
    if (rowEnd !== Math.min(expect.viewRows, rowStart + span) || colEnd !== Math.min(expect.viewCols, colStart + span)) {
      throw bad(`raw range [${rowStart}, ${rowEnd}) x [${colStart}, ${colEnd}) does not match the ${expect.viewRows} x ${expect.viewCols} view`);
    }
  }
  const n = rows * cols;
  const declared = headers.get("content-length");
  if (declared !== null && UINT.test(declared.trim()) && Number(declared) !== n * 5) throw bad(`Content-Length ${declared} is not ${n * 5}`);
  if (body.byteLength > MAX_TILE_BYTES) throw bad(`body of ${body.byteLength} bytes exceeds ${MAX_TILE_BYTES}`);
  if (body.byteLength !== n * 5) throw bad(`body is ${body.byteLength} bytes, expected rows*cols*5 = ${n * 5}`);

  // Little-endian float32 values regardless of the platform's byte order, then the status bytes.
  const dv = new DataView(body);
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) values[i] = dv.getFloat32(i * 4, true);
  const status = new Uint8Array(body.slice(n * 4, n * 5));
  const counts: TileCounts = { drawable: 0, noFinite: 0, notRepresentable: 0, withNan: 0, withPosInf: 0, withNegInf: 0, mixed: 0 };
  for (let i = 0; i < n; i++) {
    const s = (status[i] ?? 0);
    const v = (values[i] ?? 0);
    if (s & RESERVED_BITS) throw bad(`cell ${i} sets reserved status bits (${s})`);
    const drawable = (s & DRAWABLE_MASK) === 0;
    if (drawable !== Number.isFinite(v)) {
      throw bad(drawable ? `drawable cell ${i} has no finite value` : `non-drawable cell ${i} carries a finite value`);
    }
    if (s & NO_FINITE) {
      if ((s & KIND_BITS) === 0) throw bad(`cell ${i}: NO_FINITE without a non-finite kind`);
      if (s & NOT_REPRESENTABLE) throw bad(`cell ${i}: NO_FINITE and NOT_REPRESENTABLE together`);
      counts.noFinite++;
    } else if (s & NOT_REPRESENTABLE) counts.notRepresentable++;
    else {
      counts.drawable++;
      if (s & KIND_BITS) counts.mixed++;
    }
    if (level === 0 && !LEVEL0_STATUSES.has(s)) throw bad(`level-0 cell ${i} has status ${s}`);
    if (s & HAS_NAN) counts.withNan++;
    if (s & HAS_POS_INF) counts.withPosInf++;
    if (s & HAS_NEG_INF) counts.withNegInf++;
  }
  return {
    cacheKey,
    level,
    rows,
    cols,
    rowStart,
    rowEnd,
    colStart,
    colEnd,
    quality,
    valuePrecision: "float32_visualization",
    aggregation: aggregation as Aggregation,
    sampledRows,
    values,
    status,
    counts,
    bytes: values.byteLength + status.byteLength,
  };
}

const JOB_STATES: readonly JobState[] = ["created", "queued", "running", "completed", "canceled", "failed", "interrupted"];

/** Minimal structural check of a Job body; unknown optional fields are tolerated (doc 09). */
export function parseJob(v: unknown): Job {
  const j = v as Partial<Job> | null;
  if (
    typeof j !== "object" ||
    j === null ||
    typeof j.job_id !== "string" ||
    typeof j.kind !== "string" ||
    !JOB_STATES.includes(j.state as JobState) ||
    typeof j.progress !== "number" ||
    !(j.progress >= 0 && j.progress <= 1)
  ) {
    throw new ApiRequestError({ status: 202, code: "BAD_RESPONSE", message: "the job response is not a valid Job" });
  }
  return j as Job;
}

/** POST /tiles: a decoded tile (200) or the pyramid job to wait for (202). */
export async function fetchTile(req: TileRequest, signal?: AbortSignal, expect?: ExpectedGeometry): Promise<TileResult> {
  const res = await send("/tiles", { method: "POST", body: req, signal, accept: TILE_FORMAT });
  if (res.status === 202) {
    if (mediaType(res.headers) !== "application/json") throw bad("a 202 answer must be a JSON job");
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      throw new ApiRequestError({ status: 202, code: "BAD_RESPONSE", message: "the job response was not JSON" });
    }
    return { kind: "job", job: parseJob(body) };
  }
  if (res.status !== 200) throw bad(`unexpected HTTP ${res.status}`);
  if (mediaType(res.headers) !== TILE_FORMAT) throw bad(`content type ${res.headers.get("content-type") ?? "(none)"}`);
  const declared = res.headers.get("content-length");
  if (declared !== null && Number(declared) > MAX_TILE_BYTES) throw bad(`Content-Length ${declared} exceeds ${MAX_TILE_BYTES}`);
  const body = await res.arrayBuffer();
  return { kind: "tile", tile: decodeTile(res.headers, body, req, expect) };
}

const SAMPLING: readonly (SamplingPolicy | null)[] = ["stride", "splitmix64_rows", null];

/** GET /tiles/{cache_key}/metadata, checked against the decoded tile it describes. */
export async function fetchTileMetadata(tile: DecodedTile, signal?: AbortSignal): Promise<TileMetadata> {
  const res = await send(`/tiles/${encodeURIComponent(tile.cacheKey)}/metadata`, { signal });
  let m: TileMetadata;
  try {
    m = (await res.json()) as TileMetadata;
  } catch {
    throw new ApiRequestError({ status: res.status, code: "BAD_RESPONSE", message: "tile metadata was not JSON" });
  }
  const same =
    m !== null &&
    typeof m === "object" &&
    m.schema_version === 2 &&
    m.cache_key === tile.cacheKey &&
    m.rows === tile.rows &&
    m.cols === tile.cols &&
    m.level === tile.level &&
    m.row_start === tile.rowStart &&
    m.row_end === tile.rowEnd &&
    m.col_start === tile.colStart &&
    m.col_end === tile.colEnd &&
    m.quality === tile.quality &&
    m.aggregation === tile.aggregation &&
    m.value_precision === "float32_visualization" &&
    (m.sampled_rows ?? null) === tile.sampledRows &&
    SAMPLING.includes(m.sampling_policy ?? null) &&
    (tile.quality === "exact") === (m.sampling_policy == null);
  if (!same) throw new ApiRequestError({ status: res.status, code: "BAD_RESPONSE", message: "tile metadata does not match the tile it describes" });
  return m;
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  const res = await send(`/jobs/${encodeURIComponent(jobId)}`, { signal });
  return parseJob(await res.json().catch(() => null));
}

export async function cancelJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  const res = await send(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", signal });
  return parseJob(await res.json().catch(() => null));
}
