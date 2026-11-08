import { describe, expect, it } from "vitest";
import { AGG_LABEL, AGG_MEANING, blockText, PRECISION_TEXT, QUALITY_BADGE, qualityText, statusText } from "./labels";

describe("overview disclosures", () => {
  it("names every aggregation by what it computes", () => {
    expect(AGG_LABEL).toEqual({ mean: "mean", mean_abs: "mean |value|", rms: "RMS", min: "minimum", max: "maximum" });
    expect(AGG_MEANING.mean_abs).toContain("not signed weights");
    expect(AGG_MEANING.rms).toContain("not a standard deviation");
    expect(AGG_MEANING.min).toContain("smallest finite element");
  });

  it("labels coverage and precision", () => {
    expect(QUALITY_BADGE).toEqual({ exact: "EXACT coverage", row_sampled: "ROW-SAMPLED preview" });
    expect(qualityText("exact", null, null, null)).toBe("Exact coverage: every element of every block was aggregated.");
    expect(qualityText("row_sampled", 8, "stride", null)).toBe(
      "Row-sampled preview: only 8 row(s) of each block were read (evenly strided rows). Blocks are not fully covered; this is not exact.",
    );
    expect(qualityText("row_sampled", 2, "splitmix64_rows", 7)).toContain("seeded SplitMix64 (seed 7)");
    expect(PRECISION_TEXT).toContain("float32 visualizations");
    expect(PRECISION_TEXT).toContain("Exact ROI");
    expect(blockText(0)).toBe("L0: one cell per element");
    expect(blockText(7)).toBe("L7: each cell covers up to 128 × 128 elements");
  });

  it("describes status bits in words", () => {
    expect(statusText(0)).toBe("finite aggregate");
    expect(statusText(3)).toContain("no finite element in this block (NaN)");
    expect(statusText(15)).toContain("NaN, +Inf, −Inf");
    expect(statusText(6)).toBe("finite aggregate of the finite elements; the block also contains NaN, +Inf");
    expect(statusText(16)).toContain("cannot be shown as float32");
  });
});
