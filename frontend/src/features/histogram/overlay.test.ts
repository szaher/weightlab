import { describe, expect, it } from "vitest";
import type { TensorHistogram } from "../../lib/api/histogram";
import ab from "./fixtures/ab_f32.json";
import i8Json from "./fixtures/i8_symmetric.json";
import nonfiniteJson from "./fixtures/f32_nonfinite.json";
import { sameBins, sharedRange, signed } from "./overlay";

const H = (v: unknown) => JSON.parse(JSON.stringify(v)) as TensorHistogram;

describe("sharedRange", () => {
  it("is the union of both finite ranges (the backend's own min / max)", () => {
    expect(sharedRange(H(ab.a_minmax), H(ab.b_minmax))).toEqual({ lo: ab.lo, hi: ab.hi });
  });

  it("uses the side that has finite values when the other has none", () => {
    expect(sharedRange(H(ab.a_minmax), H(nonfiniteJson))).toEqual({ lo: ab.a_minmax.min, hi: ab.a_minmax.max });
    expect(sharedRange(H(nonfiniteJson), H(nonfiniteJson))).toEqual({ error: expect.stringContaining("neither A nor B") });
  });

  it("refuses mixed value domains and extremes beyond 2^53", () => {
    expect(sharedRange(H(ab.a_minmax), H(i8Json))).toEqual({ error: expect.stringContaining("different value domains") });
    const big = H(i8Json);
    big.max = "9007199254740993";
    expect(sharedRange(H(i8Json), big)).toEqual({ error: expect.stringContaining("explicit range") });
  });
});

describe("sameBins", () => {
  it("accepts the backend's two histograms over one explicit range", () => {
    expect(sameBins(H(ab.a_shared), H(ab.b_shared))).toBe(true);
  });

  it("refuses each side's own min-max bins, and any edge that differs", () => {
    expect(sameBins(H(ab.a_minmax), H(ab.b_minmax))).toBe(false);
    const moved = H(ab.b_shared);
    moved.bins!.edges[3] = (moved.bins!.edges[3] as number) + 1e-12;
    expect(sameBins(H(ab.a_shared), moved)).toBe(false);
    expect(sameBins(H(ab.a_shared), H(nonfiniteJson))).toBe(false);
  });

  it("signed counts", () => {
    expect([signed(3), signed(-1200), signed(0)]).toEqual(["+3", "−1,200", "0"]);
  });
});
