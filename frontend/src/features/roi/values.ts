// Validation and exact text of a RoiResponse (doc 09 section 2). Nothing is drawn or listed until the whole
// response has been checked against the request: a malformed, partial or mismatched answer is an error, never data.
//
// Exactness:
// - float_json: the server emits the shortest round-trip repr of the float64 value, so JSON.parse yields exactly
//   that double; its text here is the shortest round-trip repr again (with "-0.0" kept). No rounding anywhere.
// - int_json (I8..I32, U8..U32): safe integers by contract; checked.
// - int_decimal_string (I64/U64): kept as the server's decimal string; never converted to Number for display or
//   copy. A colour is derived only when |v| <= 2^53-1 (otherwise colour is withheld and the text stays exact).
// - bool_json: true/false.
import { ApiRequestError } from "../../lib/api/client";
import type { Dtype, NonfiniteKind, NumericPrecision, RoiBounds, RoiRequest, RoiResponse, ValueEncoding } from "../../lib/api/types";

export type CellState = "finite" | NonfiniteKind;

export interface Cell {
  /** Exact value as text (the copy action copies exactly this). Empty for non-finite cells. */
  text: string;
  state: CellState;
  /** Value used only for colour, or null when the cell is not coloured (non-finite, or not safely convertible). */
  color: number | null;
}

export interface RoiGrid {
  bounds: RoiBounds;
  rows: number;
  cols: number;
  dtype: Dtype;
  encoding: ValueEncoding;
  precision: NumericPrecision;
  coverage: "complete";
  warnings: string[];
  cells: Cell[]; // row-major
  counts: Record<CellState, number>;
  /** Finite cells whose colour is withheld (64-bit integers beyond +/-(2^53-1)). */
  withheld: number;
}

const FLOAT: ReadonlySet<Dtype> = new Set(["F64", "F32", "F16", "BF16"]);
const SMALL_INT: ReadonlySet<Dtype> = new Set(["I32", "I16", "I8", "U32", "U16", "U8"]);
const MASK: ReadonlySet<string> = new Set(["nan", "pos_inf", "neg_inf", "overflow"]);
const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
const I64_MIN = -(2n ** 63n);
const I64_MAX = 2n ** 63n - 1n;
const U64_MAX = 2n ** 64n - 1n;

/** The encoding the server must use for `dtype` with value_encoding=auto. */
export function autoEncoding(dtype: Dtype): ValueEncoding {
  if (FLOAT.has(dtype)) return "float_json";
  if (dtype === "BOOL") return "bool_json";
  if (dtype === "I64" || dtype === "U64") return "int_decimal_string";
  return "int_json";
}

/** Shortest round-trip text of a float64, keeping the sign of zero and marking integral values as floats. */
export function floatText(x: number): string {
  if (Object.is(x, -0)) return "-0.0";
  const s = String(x);
  return /^-?\d+$/.test(s) ? `${s}.0` : s;
}

export const STATE_LABEL: Record<CellState, string> = {
  finite: "finite",
  nan: "NaN",
  pos_inf: "+Infinity",
  neg_inf: "−Infinity",
  overflow: "overflow (derived value out of float64 range)",
};

/** What "Copy exact value" puts on the clipboard: the exact text, or a plain-ASCII non-finite name. */
export function copyText(cell: Cell): string {
  if (cell.state === "finite") return cell.text;
  return { nan: "NaN", pos_inf: "Infinity", neg_inf: "-Infinity", overflow: "overflow" }[cell.state];
}

/** Text for a cell wherever a value is shown: the exact value, or the non-finite kind spelled out. */
export function cellText(cell: Cell): string {
  return cell.state === "finite" ? cell.text : STATE_LABEL[cell.state];
}

class Invalid extends Error {}
function check(cond: boolean, why: string): asserts cond {
  if (!cond) throw new Invalid(why);
}

function sameBounds(a: RoiBounds, b: RoiBounds): boolean {
  return a.r0 === b.r0 && a.r1 === b.r1 && a.c0 === b.c0 && a.c1 === b.c1;
}

function decimalCell(v: unknown, dtype: Dtype): { cell: Cell; unsafe: boolean } {
  check(typeof v === "string" && /^-?(0|[1-9][0-9]*)$/.test(v), "64-bit integers must be decimal strings");
  const big = BigInt(v);
  if (dtype === "U64") check(big >= 0n && big <= U64_MAX, "value outside the U64 range");
  else check(big >= I64_MIN && big <= I64_MAX, "value outside the I64 range");
  const safe = big <= MAX_SAFE && big >= -MAX_SAFE;
  return { cell: { text: v, state: "finite", color: safe ? Number(v) : null }, unsafe: !safe };
}

/** Validates `resp` against `req` and the tensor's dtype, returning the grid or throwing a BAD_RESPONSE error. */
export function parseRoi(resp: unknown, req: RoiRequest, dtype: Dtype): RoiGrid {
  try {
    return parse(resp, req, dtype);
  } catch (e) {
    if (!(e instanceof Invalid)) throw e;
    throw new ApiRequestError({
      status: 200,
      code: "BAD_RESPONSE",
      message: `The exact-value response failed validation (${e.message}); nothing from it is shown.`,
    });
  }
}

function parse(resp: unknown, req: RoiRequest, dtype: Dtype): RoiGrid {
  check(typeof resp === "object" && resp !== null, "not an object");
  const r = resp as Partial<RoiResponse>;
  check(r.schema_version === 2, "unknown schema_version");
  const src = r.source as { kind?: unknown; model_id?: unknown; tensor_id?: unknown } | undefined;
  check(
    src?.kind === "tensor" && src.model_id === req.source.model_id && src.tensor_id === req.source.tensor_id,
    "answer is for a different tensor",
  );
  check(typeof r.roi === "object" && r.roi !== null && sameBounds(r.roi, req.roi), "answer is for a different region");
  const rows = req.roi.r1 - req.roi.r0;
  const cols = req.roi.c1 - req.roi.c0;
  check(Array.isArray(r.shape) && r.shape.length === 2 && r.shape[0] === rows && r.shape[1] === cols, "shape does not match the region");
  check(r.source_dtype === dtype, `dtype ${String(r.source_dtype)} differs from the catalog's ${dtype}`);
  check(r.coverage === "complete", "coverage is not complete");
  check(r.numeric_precision === "lossless" || r.numeric_precision === "rounded" || r.numeric_precision === "overflowed", "unknown numeric_precision");
  // A raw tensor source is lossless by contract (doc 09 section 2): rounded/overflowed belongs to derived sources only.
  check(req.source.kind !== "tensor" || r.numeric_precision === "lossless", `a raw tensor read must be lossless, not ${String(r.numeric_precision)}`);
  const enc = r.value_encoding;
  check(enc === autoEncoding(dtype), `encoding ${String(enc)} is not the lossless encoding of ${dtype}`);
  check(Array.isArray(r.warnings) && r.warnings.every((w) => typeof w === "string"), "warnings malformed");
  const values = r.values;
  check(Array.isArray(values) && values.length === rows, "row count differs from the region");
  const mask = r.nonfinite_mask ?? null;
  check(mask === null || (Array.isArray(mask) && mask.length === rows), "non-finite mask malformed");
  check(mask === null || enc === "float_json", "non-finite mask on a non-float encoding");

  const counts: Record<CellState, number> = { finite: 0, nan: 0, pos_inf: 0, neg_inf: 0, overflow: 0 };
  const cells: Cell[] = new Array<Cell>(rows * cols);
  let withheld = 0;
  for (let i = 0; i < rows; i++) {
    const row = values[i];
    check(Array.isArray(row) && row.length === cols, `row ${i} has the wrong length`);
    const mrow = mask?.[i] ?? null;
    check(mask === null || (Array.isArray(mrow) && mrow.length === cols), `mask row ${i} has the wrong length`);
    for (let j = 0; j < cols; j++) {
      const v: unknown = row[j];
      let cell: Cell;
      if (enc === "float_json") {
        const m: unknown = mrow?.[j] ?? null;
        if (m === null) {
          check(typeof v === "number" && Number.isFinite(v), `cell ${i},${j} is not a finite number`);
          cell = { text: floatText(v), state: "finite", color: v };
        } else {
          check(typeof m === "string" && MASK.has(m) && v === null, `cell ${i},${j} has an inconsistent mask`);
          check(m !== "overflow" || r.numeric_precision === "overflowed", "overflow mask without overflowed precision");
          cell = { text: "", state: m as NonfiniteKind, color: null };
        }
      } else if (enc === "int_json") {
        check(SMALL_INT.has(dtype) && typeof v === "number" && Number.isSafeInteger(v), `cell ${i},${j} is not a safe integer`);
        cell = { text: String(v), state: "finite", color: v };
      } else if (enc === "int_decimal_string") {
        const d = decimalCell(v, dtype);
        cell = d.cell;
        if (d.unsafe) withheld++;
      } else {
        check(typeof v === "boolean", `cell ${i},${j} is not a boolean`);
        cell = { text: v ? "true" : "false", state: "finite", color: v ? 1 : 0 };
      }
      counts[cell.state]++;
      cells[i * cols + j] = cell;
    }
  }
  return {
    bounds: { ...req.roi },
    rows,
    cols,
    dtype,
    encoding: enc,
    precision: r.numeric_precision,
    coverage: "complete",
    warnings: r.warnings,
    cells,
    counts,
    withheld,
  };
}
