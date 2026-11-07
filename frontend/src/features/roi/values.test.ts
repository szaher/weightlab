// Exactness and validation of RoiResponse parsing. Float expectations come from the fixture generator's own
// output (examples/fixtures/expected_results.json, produced by numpy) and from Math.fround / literal bit patterns,
// not from WeightLab code.
import { describe, expect, it } from "vitest";
import { ApiRequestError } from "../../lib/api/client";
import type { Dtype, RoiRequest } from "../../lib/api/types";
import { copyText, floatText, parseRoi } from "./values";

import fixture from "../../../../examples/fixtures/expected_results.json";

const expected = fixture as unknown as {
  base_full: number[][];
  precision: { f64_a: string[]; i64: string[]; u64: string[]; i32: string[] };
  dtypes: { bf16_specials_bits: number[] };
};

function req(rows: number, cols: number, r0 = 0, c0 = 0): RoiRequest {
  return {
    source: { kind: "tensor", model_id: "mod_1", tensor_id: "tns_1" },
    view: { row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false },
    roi: { r0, r1: r0 + rows, c0, c1: c0 + cols },
    value_encoding: "auto",
  };
}

function resp(r: RoiRequest, dtype: Dtype, encoding: string, values: unknown[][], mask: unknown = null, extra: object = {}) {
  return {
    schema_version: 2,
    source: r.source,
    roi: r.roi,
    shape: [r.roi.r1 - r.roi.r0, r.roi.c1 - r.roi.c0],
    source_dtype: dtype,
    value_encoding: encoding,
    coverage: "complete",
    numeric_precision: "lossless",
    values,
    nonfinite_mask: mask,
    warnings: [],
    ...extra,
  };
}

/** The JSON text the server sends, parsed the way the browser parses it. */
function wire(o: unknown): unknown {
  return JSON.parse(JSON.stringify(o));
}

describe("parseRoi: exact values", () => {
  it("keeps the 8x8 ramp exact (F32 widened to float64), row-major and correctly oriented", () => {
    const r = req(8, 8);
    const g = parseRoi(wire(resp(r, "F32", "float_json", expected.base_full)), r, "F32");
    expect(g.rows).toBe(8);
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const cell = g.cells[i * 8 + j]!;
        const want = expected.base_full[i]![j]!;
        expect(Object.is(Number(cell.text), want)).toBe(true);
        expect(Math.fround(want)).toBe(want); // really an F32 value
        expect(cell.color).toBe(want);
      }
    }
    expect(g.cells[0 * 8 + 1]!.text).toBe("-0.949999988079071"); // (0,1), not (1,0)
    expect(g.cells[1 * 8 + 0]!.text).toBe("-0.6000000238418579");
  });

  it("preserves -0.0, subnormals and float64 extremes from the precision fixture", () => {
    const vals = expected.precision.f64_a.map(Number); // the generator's repr strings
    const r = req(1, vals.length);
    // The server writes "-0.0"; JSON.stringify would lose the sign, so build the wire text by hand.
    const text = JSON.stringify(resp(r, "F64", "float_json", [vals.map(() => 0)])).replace(
      /"values":\[\[[^\]]*\]\]/,
      `"values":[[${expected.precision.f64_a.join(",")}]]`,
    );
    const g = parseRoi(JSON.parse(text), r, "F64");
    g.cells.forEach((c, k) => expect(Object.is(Number(c.text), vals[k])).toBe(true));
    expect(g.cells[4]!.text).toBe("-0.0");
    expect(copyText(g.cells[4]!)).toBe("-0.0");
    expect(g.cells[3]!.text).toBe("5e-324");
    expect(g.cells[5]!.text).toBe("1.7976931348623157e+308");
    expect(g.cells[0]!.text).toBe("1.0");
  });

  it("keeps I64/U64 as decimal strings across the full range and withholds colour beyond 2^53-1", () => {
    const r = req(1, expected.precision.i64.length);
    const g = parseRoi(wire(resp(r, "I64", "int_decimal_string", [expected.precision.i64])), r, "I64");
    expect(g.cells.map((c) => c.text)).toEqual(expected.precision.i64);
    expect(g.cells.map((c) => copyText(c))).toEqual(expected.precision.i64);
    // 9007199254740993 would round to ...992 as a Number; it stays exact text and gets no colour.
    expect(g.cells[5]!.text).toBe("9007199254740993");
    expect(g.cells[5]!.color).toBeNull();
    expect(g.cells[3]!.color).toBe(9007199254740991);
    expect(g.withheld).toBe(4);
    const ru = req(1, 3);
    const u = parseRoi(wire(resp(ru, "U64", "int_decimal_string", [expected.precision.u64])), ru, "U64");
    expect(u.cells.map((c) => c.text)).toEqual(["0", "9007199254740993", "18446744073709551615"]);
  });

  it("I32 as safe integers and BOOL as booleans", () => {
    const r = req(1, 2);
    const g = parseRoi(wire(resp(r, "I32", "int_json", [expected.precision.i32.map(Number)])), r, "I32");
    expect(g.cells.map((c) => c.text)).toEqual(["-2147483648", "2147483647"]);
    const b = parseRoi(wire(resp(r, "BOOL", "bool_json", [[true, false]])), r, "BOOL");
    expect(b.cells.map((c) => [c.text, c.color])).toEqual([
      ["true", 1],
      ["false", 0],
    ]);
  });

  it("distinguishes NaN, +Inf and -Inf from the mask and never invents a number", () => {
    const r = req(2, 2);
    const g = parseRoi(wire(resp(r, "BF16", "float_json", [[null, 1.5], [null, null]], [[ "nan", null], ["pos_inf", "neg_inf"]])), r, "BF16");
    expect(g.cells.map((c) => c.state)).toEqual(["nan", "finite", "pos_inf", "neg_inf"]);
    expect(g.cells.filter((c) => c.state !== "finite").every((c) => c.color === null && c.text === "")).toBe(true);
    expect(g.counts).toMatchObject({ nan: 1, pos_inf: 1, neg_inf: 1, finite: 1 });
    expect(g.cells.map(copyText)).toEqual(["NaN", "1.5", "Infinity", "-Infinity"]);
  });

  it("decodes the BF16 specials' bit patterns to the same float64 values the text shows", () => {
    // bf16 bits -> float32 by shifting into the high half; independent of the app.
    const f = (bits: number) => new Float32Array(new Uint32Array([bits << 16]).buffer)[0]!;
    const vals = expected.dtypes.bf16_specials_bits.map(f);
    const finite = vals.filter((v) => Number.isFinite(v));
    const r = req(1, finite.length);
    const g = parseRoi(JSON.parse(JSON.stringify(resp(r, "BF16", "float_json", [finite])).replace('"values":[[1,-2,0,0,', '"values":[[1,-2,0,-0.0,')), r, "BF16");
    g.cells.forEach((c, k) => expect(Number(c.text)).toBe(finite[k]));
    expect(floatText(f(0x7f7f))).toBe("3.3895313892515355e+38");
  });
});

describe("parseRoi: anything malformed or mismatched is an error, never data", () => {
  const r = req(2, 2, 10, 20);
  const ok = () => resp(r, "F32", "float_json", [[1, 2], [3, 4]]);
  const cases: [string, unknown][] = [
    ["other region", { ...ok(), roi: { r0: 0, r1: 2, c0: 0, c1: 2 } }],
    ["other tensor", { ...ok(), source: { ...r.source, tensor_id: "tns_2" } }],
    ["shape", { ...ok(), shape: [2, 3] }],
    ["dtype", { ...ok(), source_dtype: "F16" }],
    ["encoding", { ...ok(), value_encoding: "hex_bits" }],
    ["coverage", { ...ok(), coverage: "row_sampled" }],
    ["short row", { ...ok(), values: [[1, 2], [3]] }],
    ["missing row", { ...ok(), values: [[1, 2]] }],
    ["null without mask", { ...ok(), values: [[1, null], [3, 4]] }],
    ["string float", { ...ok(), values: [[1, "2"], [3, 4]] }],
    ["mask with value", { ...ok(), nonfinite_mask: [["nan", null], [null, null]] }],
    ["unknown mask", { ...ok(), values: [[null, 2], [3, 4]], nonfinite_mask: [["huge", null], [null, null]] }],
    ["overflow on lossless", { ...ok(), values: [[null, 2], [3, 4]], nonfinite_mask: [["overflow", null], [null, null]] }],
    ["schema", { ...ok(), schema_version: 1 }],
    ["not an object", null],
  ];
  it.each(cases)("%s", (_, body) => {
    expect(() => parseRoi(body, r, "F32")).toThrow(ApiRequestError);
    try {
      parseRoi(body, r, "F32");
    } catch (e) {
      expect((e as ApiRequestError).code).toBe("BAD_RESPONSE");
    }
  });
  it("rejects bad 64-bit strings and out-of-range integers", () => {
    const one = req(1, 1);
    for (const v of ["01", "1.0", "1e3", "18446744073709551616", 5]) {
      expect(() => parseRoi(resp(one, "U64", "int_decimal_string", [[v]]), one, "U64")).toThrow(ApiRequestError);
    }
    expect(() => parseRoi(resp(one, "I64", "int_decimal_string", [["9223372036854775808"]]), one, "I64")).toThrow(ApiRequestError);
    expect(() => parseRoi(resp(one, "I32", "int_json", [[2 ** 53]]), one, "I32")).toThrow(ApiRequestError);
  });
});
