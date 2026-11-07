import { describe, expect, it } from "vitest";
import { cellFill, computeLayout, FALLBACK_THEME, hitCell, paint } from "./paint";
import { colorAt, computeScale, normalize } from "./scale";
import type { Cell, CellState, RoiGrid } from "./values";

function grid(rows: number, cols: number, vals: (number | CellState | null)[]): RoiGrid {
  const cells: Cell[] = vals.map((v) =>
    typeof v === "number" ? { text: String(v), state: "finite", color: v } : v === null ? { text: "9007199254740993", state: "finite", color: null } : { text: "", state: v, color: null },
  );
  const counts = { finite: 0, nan: 0, pos_inf: 0, neg_inf: 0, overflow: 0 };
  cells.forEach((c) => counts[c.state]++);
  return {
    bounds: { r0: 0, r1: rows, c0: 0, c1: cols },
    rows,
    cols,
    dtype: "F64",
    encoding: "float_json",
    precision: "lossless",
    coverage: "complete",
    warnings: [],
    cells,
    counts,
    withheld: cells.filter((c) => c.state === "finite" && c.color === null).length,
  };
}

describe("colour scales use only this region's colourable values", () => {
  it("auto, sign-symmetric and fixed", () => {
    const g = grid(1, 5, [-2, 0.5, 3, "nan", "pos_inf"]);
    expect(computeScale(g, "auto")).toMatchObject({ lo: -2, hi: 3, degenerate: "none" });
    expect(computeScale(g, "symmetric")).toMatchObject({ lo: -3, hi: 3 });
    expect(computeScale(g, "fixed", { min: "0", max: "1" })).toMatchObject({ lo: 0, hi: 1, below: 1, above: 1 });
    expect(computeScale(g, "fixed", { min: "1", max: "1" })).toMatchObject({ mode: "auto", lo: -2, hi: 3 }); // invalid range
  });

  it("all-zero, constant, empty and withheld-only regions", () => {
    expect(computeScale(grid(1, 3, [0, 0, -0]), "auto").degenerate).toBe("constant");
    expect(computeScale(grid(1, 3, [0, 0, 0]), "symmetric").degenerate).toBe("constant");
    expect(computeScale(grid(1, 2, [7, 7]), "auto")).toMatchObject({ lo: 7, hi: 7, degenerate: "constant" });
    expect(computeScale(grid(1, 2, ["nan", "neg_inf"]), "auto").degenerate).toBe("empty");
    expect(computeScale(grid(1, 2, [null, null]), "auto").degenerate).toBe("empty");
    expect(normalize(7, computeScale(grid(1, 2, [7, 7]), "auto"))).toBe(0.5);
  });

  it("extreme float64 values normalize without overflow", () => {
    const s = computeScale(grid(1, 3, [-1.7976931348623157e308, 0, 1.7976931348623157e308]), "auto");
    expect(normalize(-1.7976931348623157e308, s)).toBe(0);
    expect(normalize(0, s)).toBe(0.5);
    expect(normalize(1.7976931348623157e308, s)).toBe(1);
    const tiny = computeScale(grid(1, 2, [5e-324, 1e-323]), "auto");
    expect(normalize(5e-324, tiny)).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(normalize(1e-323, tiny))).toBe(true);
  });

  it("subnormal F64 ranges normalize to real positions (review P1: halving underflowed to NaN)", () => {
    const at = (vals: number[], mode: "auto" | "symmetric" = "auto") => {
      const s = computeScale(grid(1, vals.length, vals), mode);
      return vals.map((v) => normalize(v, s));
    };
    expect(at([0, 5e-324])).toEqual([0, 1]);
    expect(at([-5e-324, 5e-324])).toEqual([0, 1]);
    expect(at([-5e-324, 0, 5e-324])).toEqual([0, 0.5, 1]);
    expect(at([5e-324, 1e-323])).toEqual([0, 1]);
    expect(at([0, 5e-324], "symmetric")).toEqual([0.5, 1]);
    expect(at([5e-324, 1e-323, 1.5e-323])).toEqual([0, 0.5, 1]);
    // Large and tiny together: the subnormal is indistinguishable from 0 at this scale, which is correct.
    expect(at([-1.7976931348623157e308, 5e-324, 1.7976931348623157e308])).toEqual([0, 0.5, 1]);
  });

  it("every cell of extreme, subnormal and fixed-range regions gets a valid CSS colour", () => {
    const RGB = /^rgb\((\d{1,3}), (\d{1,3}), (\d{1,3})\)$/;
    const MAX = 1.7976931348623157e308;
    const regions: number[][] = [
      [0, 5e-324],
      [-5e-324, 5e-324],
      [5e-324, 1e-323],
      [-MAX, MAX],
      [-MAX, -5e-324, 0, 5e-324, MAX],
      [2.2250738585072014e-308, 2.225073858507201e-308],
      [1e300, -1e300, 1, -1],
      [0, 1e-320, 2e-320, 3e-320],
    ];
    for (const vals of regions) {
      const g = grid(1, vals.length, vals);
      const fixed = [
        { min: "0", max: "5e-324" },
        { min: "-1", max: "1" },
        { min: String(-MAX), max: String(MAX) },
      ];
      const scales = [computeScale(g, "auto"), computeScale(g, "symmetric"), ...fixed.map((f) => computeScale(g, "fixed", f))];
      for (const sc of scales) {
        for (const palette of ["sequential", "diverging"] as const) {
          for (const c of g.cells) {
            const f = cellFill(c, sc, palette);
            expect(f.kind, `${vals} ${sc.mode} ${sc.lo}..${sc.hi}`).toBe("color");
            if (f.kind !== "color") continue;
            const m = RGB.exec(f.color);
            expect(m, f.color).not.toBeNull();
            expect(m!.slice(1).every((x) => Number(x) <= 255)).toBe(true);
          }
        }
      }
    }
  });

  it("the palette refuses a non-finite position instead of inventing a colour", () => {
    expect(() => colorAt(NaN, "sequential")).toThrow(RangeError);
    expect(() => colorAt(Infinity, "diverging")).toThrow(RangeError);
  });

  it("palettes: sequential gets lighter monotonically; diverging is near white in the middle and dark at both ends", () => {
    const lum = (c: string) => {
      const [r, g, b] = c.match(/\d+/g)!.map(Number);
      return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
    };
    let prev = -1;
    for (let k = 0; k <= 16; k++) {
      const l = lum(colorAt(k / 16, "sequential"));
      expect(l).toBeGreaterThan(prev);
      prev = l;
    }
    expect(colorAt(0.5, "diverging").match(/\d+/g)!.map(Number).every((c) => c >= 245)).toBe(true); // near white
    expect(lum(colorAt(0, "diverging"))).toBeLessThan(80);
    expect(lum(colorAt(1, "diverging"))).toBeLessThan(80);
  });
});

describe("heatmap drawing", () => {
  it("non-finite and withheld cells get a pattern, never a scale colour", () => {
    const g = grid(1, 5, [1, "nan", "pos_inf", "neg_inf", null]);
    const s = computeScale(g, "auto");
    expect(g.cells.map((c) => cellFill(c, s, "sequential"))).toEqual([
      { kind: "color", color: colorAt(0.5, "sequential") },
      { kind: "pattern", pattern: "nan" },
      { kind: "pattern", pattern: "pos_inf" },
      { kind: "pattern", pattern: "neg_inf" },
      { kind: "pattern", pattern: "withheld" },
    ]);
  });

  it("hit-testing inverts the layout and ignores the margins", () => {
    const l = computeLayout(64, 64, 193, 199, 700);
    expect(hitCell(l, l.left + 0.5 * l.cell, l.top + 0.5 * l.cell)).toEqual({ i: 0, j: 0 });
    expect(hitCell(l, l.left + 63.9 * l.cell, l.top + 10.2 * l.cell)).toEqual({ i: 10, j: 63 });
    expect(hitCell(l, l.left - 1, l.top + 1)).toBeNull();
    expect(hitCell(l, l.left + 64 * l.cell + 1, l.top + 1)).toBeNull();
    expect(l.cell).toBeGreaterThanOrEqual(4);
  });

  for (const dpr of [1, 1.25, 1.5, 2, 3]) {
    it(`cells tile the device-pixel grid exactly at devicePixelRatio ${dpr}`, () => {
      const rows = 7;
      const cols = 13;
      const g = grid(rows, cols, Array.from({ length: rows * cols }, (_, k) => k));
      const l = computeLayout(rows, cols, 0, 0, 333);
      const rects: [number, number, number, number][] = [];
      const ctx = new Proxy(
        {},
        {
          get: (_t, prop) => (prop === "fillRect" ? (x: number, y: number, w: number, h: number) => rects.push([x, y, w, h]) : () => undefined),
          set: () => true,
        },
      ) as unknown as CanvasRenderingContext2D;
      paint(ctx, { layout: l, grid: g, scale: computeScale(g, "auto"), palette: "sequential", theme: FALLBACK_THEME, dpr, selected: null, hover: null });
      expect(rects).toHaveLength(rows * cols);
      for (const [x, y, w, h] of rects) expect([x, y, w, h].every(Number.isInteger)).toBe(true);
      // Adjacent cells share an edge: no gaps, no overlaps.
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j + 1 < cols; j++) {
          const a = rects[i * cols + j]!;
          const b = rects[i * cols + j + 1]!;
          expect(a[0] + a[2]).toBe(b[0]);
        }
      }
      const first = rects[0]!;
      const last = rects[rects.length - 1]!;
      expect(last[0] + last[2] - first[0]).toBe(Math.round((l.left + cols * l.cell) * dpr) - Math.round(l.left * dpr));
    });
  }
});
