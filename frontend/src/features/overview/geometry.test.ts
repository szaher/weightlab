import { describe, expect, it } from "vitest";
import {
  autoLevel,
  cellBounds,
  cellOf,
  cellRangeBounds,
  clampViewport,
  fitViewport,
  levelCells,
  panBy,
  rawAt,
  tileBounds,
  tileGrid,
  tilesFor,
  topLevel,
  visibleRect,
  withRing,
  zoomAt,
  zoomToRect,
} from "./geometry";

const S = { width: 600, height: 400 };

describe("tile and cell coordinates", () => {
  it("maps tiles to raw coordinates at L0, L3, L4 and high levels (half-open, x = columns)", () => {
    const d = { rows: 10_000, cols: 30_000 };
    expect(tileBounds(d, 0, 2, 1)).toMatchObject({ r0: 256, r1: 512, c0: 512, c1: 768, rows: 256, cols: 256 });
    expect(tileBounds(d, 3, 1, 0)).toMatchObject({ r0: 0, r1: 2048, c0: 2048, c1: 4096, rows: 256, cols: 256 });
    expect(tileBounds(d, 4, 0, 2)).toMatchObject({ r0: 8192, r1: 10_000, c0: 0, c1: 4096, rows: 113, cols: 256 });
    expect(tileBounds(d, 7, 0, 0)).toMatchObject({ r0: 0, r1: 10_000, c0: 0, c1: 30_000, rows: 79, cols: 235 });
    expect(tileBounds(d, 52, 0, 0)).toMatchObject({ r0: 0, r1: 10_000, c0: 0, c1: 30_000, rows: 1, cols: 1 });
    expect(tileBounds(d, 4, 0, 3)).toBeNull(); // starts outside the view
    expect(tileBounds(d, 4, -1, 0)).toBeNull();
  });

  it("clips the odd edges of a 257 x 263 view", () => {
    const d = { rows: 257, cols: 263 };
    expect(tileGrid(d, 0)).toEqual({ nx: 2, ny: 2 });
    expect(tileBounds(d, 0, 1, 1)).toMatchObject({ r0: 256, r1: 257, c0: 256, c1: 263, rows: 1, cols: 7 });
    expect(tileBounds(d, 3, 0, 0)).toMatchObject({ r1: 257, c1: 263, rows: 33, cols: 33 });
    expect(levelCells(d, 3)).toEqual({ rows: 33, cols: 33 });
    expect(cellBounds(d, 3, 32, 32)).toEqual({ r0: 256, r1: 257, c0: 256, c1: 263 });
    expect(cellBounds(d, 3, 33, 0)).toBeNull();
    expect(topLevel(d)).toBe(1);
  });

  it("handles extremely asymmetric views", () => {
    const d = { rows: 151_936, cols: 896 }; // an embedding matrix
    expect(topLevel(d)).toBe(10);
    expect(tileGrid(d, 4)).toEqual({ nx: 1, ny: 38 });
    const v = fitViewport(d, S);
    expect(v.ppe).toBeCloseTo(400 / 151_936, 12);
    expect(v.top).toBe(0);
    expect(v.left).toBeLessThan(0); // centred horizontally: the view is narrower than the canvas
    const vr = visibleRect(v, d, S);
    expect(vr).toEqual({ r0: 0, r1: 151_936, c0: 0, c1: 896 });
  });

  it("derives cells and rectangles from raw elements, never the reverse", () => {
    expect(cellOf(4, 165, 325)).toEqual({ i: 10, j: 20 });
    expect(cellRangeBounds({ rows: 1024, cols: 2304 }, 4, { i: 4, j: 4 }, { i: 0, j: 0 })).toEqual({ r0: 0, r1: 80, c0: 0, c1: 80 });
  });
});

describe("viewport", () => {
  const d = { rows: 4096, cols: 14336 };

  it("fits the whole view and picks the finest level whose cells are >= 1 px", () => {
    const v = fitViewport(d, S);
    expect(v.ppe).toBeCloseTo(600 / 14336, 12);
    expect(autoLevel(v, d)).toBe(5); // 2^5 * 0.0419 = 1.34 px; 2^4 would be 0.67 px
    expect(autoLevel({ ...v, ppe: 2 }, d)).toBe(0);
    expect(autoLevel({ ...v, ppe: 1e-9 }, d)).toBe(topLevel(d));
  });

  it("keeps the raw point under the pointer fixed while zooming", () => {
    const v0 = zoomAt(fitViewport(d, S), d, S, 8, 300, 200);
    const under = rawAt(v0, 123, 77);
    const v1 = zoomAt(v0, d, S, 2, 123, 77);
    const after = rawAt(v1, 123, 77);
    expect(v1.ppe).toBeCloseTo(v0.ppe * 2, 12);
    expect(after.r).toBeCloseTo(under.r, 6);
    expect(after.c).toBeCloseTo(under.c, 6);
  });

  it("never zooms out beyond the whole view or in beyond 64 px per element, and clamps pans", () => {
    const f = fitViewport(d, S);
    expect(zoomAt(f, d, S, 0.1, 0, 0).ppe).toBeCloseTo(f.ppe, 12);
    expect(zoomAt(f, d, S, 1e9, 0, 0).ppe).toBe(64);
    const z = zoomAt(f, d, S, 16, 0, 0);
    const far = panBy(z, d, S, 1e9, 1e9);
    expect(far.left).toBe(0);
    expect(far.top).toBe(0);
    const other = panBy(z, d, S, -1e9, -1e9);
    expect(other.left).toBeCloseTo(d.cols - S.width / z.ppe, 6);
    expect(clampViewport({ ppe: NaN, top: 0, left: 0 }, d, S).ppe).toBeCloseTo(f.ppe, 12);
  });

  it("zooms to a selected rectangle", () => {
    const v = zoomToRect(d, S, { r0: 160, r1: 176, c0: 320, c1: 336 });
    expect(v.ppe).toBeCloseTo(25, 6);
    const c = rawAt(v, S.width / 2, S.height / 2);
    expect(c.r).toBeCloseTo(168, 6);
    expect(c.c).toBeCloseTo(328, 6);
  });

  it("lists the visible tiles and a bounded prefetch ring", () => {
    const r = tilesFor({ r0: 300, r1: 900, c0: 0, c1: 10 }, d, 0)!;
    expect(r).toEqual({ level: 0, x0: 0, x1: 0, y0: 1, y1: 3 });
    expect(withRing(r, d)).toEqual({ level: 0, x0: 0, x1: 1, y0: 0, y1: 4 });
    expect(tilesFor({ r0: 0, r1: 0, c0: 0, c1: 5 }, d, 0)).toBeNull();
  });
});
