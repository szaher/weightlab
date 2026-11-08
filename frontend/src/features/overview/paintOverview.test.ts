import { describe, expect, it } from "vitest";
import type { DecodedTile } from "../../lib/api/tiles";
import { paletteLut } from "../roi/scale";
import { NO_FINITE_RGB, NOT_REPR_RGB, tilePixels, tilesScale } from "./paintOverview";

function tile(values: number[], status: number[]): DecodedTile {
  return { cacheKey: "a".repeat(64), level: 2, rows: 1, cols: values.length, rowStart: 0, rowEnd: 4, colStart: 0, colEnd: values.length * 4, quality: "exact", valuePrecision: "float32_visualization", aggregation: "mean", sampledRows: null, values: Float32Array.from(values), status: Uint8Array.from(status), counts: { drawable: 0, noFinite: 0, notRepresentable: 0, withNan: 0, withPosInf: 0, withNegInf: 0, mixed: 0 }, bytes: values.length * 5 };
}

describe("overview painting", () => {
  it("scales from drawable cells of the visible tiles only", () => {
    const t = tile([-1, 3, NaN, NaN, 2], [0, 0, 3, 16, 2]);
    expect(tilesScale([t], "auto")).toMatchObject({ lo: -1, hi: 3, degenerate: "none" });
    expect(tilesScale([t], "symmetric")).toMatchObject({ lo: -3, hi: 3 });
    expect(tilesScale([t], "fixed", { min: "0", max: "2" })).toMatchObject({ lo: 0, hi: 2, below: 1, above: 1 });
    expect(tilesScale([tile([NaN], [3])], "auto").degenerate).toBe("empty");
  });

  it("never colours a non-drawable cell from the palette, and keeps finite aggregates of mixed blocks", () => {
    const t = tile([0, 1, NaN, NaN, 0.5], [0, 0, 3, 16, 2]);
    const px = tilePixels(t, tilesScale([t], "auto"), "sequential");
    const lut = paletteLut("sequential");
    const rgb = (i: number) => [px[i * 4], px[i * 4 + 1], px[i * 4 + 2]];
    expect(rgb(0)).toEqual([lut[0], lut[1], lut[2]]);
    expect(rgb(1)).toEqual([lut[255 * 3], lut[255 * 3 + 1], lut[255 * 3 + 2]]);
    expect(rgb(2)).toEqual([...NO_FINITE_RGB]);
    expect(rgb(3)).toEqual([...NOT_REPR_RGB]);
    expect(rgb(4)).toEqual([lut[128 * 3], lut[128 * 3 + 1], lut[128 * 3 + 2]]); // NaN in the block, finite mean kept
    expect(px[3]).toBe(255);
  });
});
