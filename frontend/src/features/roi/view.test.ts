import { describe, expect, it } from "vitest";
import {
  boundsError,
  centreOn,
  clampWin,
  defaultView,
  defaultWin,
  ensureVisible,
  pan,
  tensorIndex,
  toBounds,
  toRequestView,
  viewDims,
  viewError,
  zoom,
} from "./view";

describe("views follow the server's selection rules", () => {
  it("rank 0/1 use the null/null view; rank 2 rows = axis 0; rank > 2 the last two axes", () => {
    expect(toRequestView(defaultView([]))).toEqual({ row_axis: null, col_axis: null, fixed_indices: {}, transpose: false });
    expect(toRequestView(defaultView([6]))).toEqual({ row_axis: null, col_axis: null, fixed_indices: {}, transpose: false });
    expect(toRequestView(defaultView([257, 263]))).toEqual({ row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false });
    expect(toRequestView(defaultView([2, 3, 4, 8]))).toEqual({ row_axis: 2, col_axis: 3, fixed_indices: { "0": 0, "1": 0 }, transpose: false });
    expect(viewDims([], defaultView([]))).toEqual({ rows: 1, cols: 1 });
    expect(viewDims([6], defaultView([6]))).toEqual({ rows: 1, cols: 6 });
  });

  it("transpose swaps the shown axes and the tensor index stays the same element", () => {
    const v = { ...defaultView([257, 263]), transpose: true };
    expect(viewDims([257, 263], v)).toEqual({ rows: 263, cols: 257 });
    expect(tensorIndex([257, 263], v, 262, 256)).toEqual([256, 262]);
    const cube = { rowAxis: 0, colAxis: 2, fixed: { 1: "3" }, transpose: false };
    expect(viewDims([2, 4, 8], cube)).toEqual({ rows: 2, cols: 8 });
    expect(tensorIndex([2, 4, 8], cube, 1, 5)).toEqual([1, 3, 5]);
    expect(tensorIndex([2, 4, 8], { ...cube, transpose: true }, 5, 1)).toEqual([1, 3, 5]);
  });

  it("rejects views the server would reject, without guessing a fix", () => {
    const shape = [2, 4, 8];
    expect(viewError(shape, defaultView(shape))).toBeNull();
    expect(viewError(shape, { rowAxis: 1, colAxis: 1, fixed: { 0: "0", 2: "0" }, transpose: false })).toMatch(/two different axes/);
    expect(viewError(shape, { rowAxis: 1, colAxis: 2, fixed: {}, transpose: false })).toMatch(/axis 0 needs a fixed index/);
    expect(viewError(shape, { rowAxis: 1, colAxis: 2, fixed: { 0: "2" }, transpose: false })).toMatch(/outside 0–1/);
    expect(viewError(shape, { rowAxis: 1, colAxis: 2, fixed: { 0: "-1" }, transpose: false })).toMatch(/whole number/);
    expect(viewError(shape, { rowAxis: 1, colAxis: 2, fixed: { 0: "1.5" }, transpose: false })).toMatch(/whole number/);
    expect(viewError([6], { rowAxis: null, colAxis: null, fixed: {}, transpose: true })).toMatch(/no axes/);
  });
});

describe("exact windows stay inside the view and the cell budget", () => {
  const dims = { rows: 257, cols: 263 };
  it("defaults to 64 x 64 and shrinks to fit roi_max_cells", () => {
    expect(defaultWin(dims, 4096)).toEqual({ r0: 0, c0: 0, h: 64, w: 64 });
    const small = defaultWin(dims, 1000);
    expect(small.h * small.w).toBeLessThanOrEqual(1000);
    expect(defaultWin({ rows: 2, cols: 4 }, 4096)).toEqual({ r0: 0, c0: 0, h: 2, w: 4 });
    expect(defaultWin({ rows: 1, cols: 100000 }, 4096)).toEqual({ r0: 0, c0: 0, h: 1, w: 64 });
  });

  it("pans by whole cells and clamps at the far edges (ragged 257 x 263)", () => {
    const w = defaultWin(dims, 4096);
    expect(pan(w, dims, 4096, 32, 32)).toMatchObject({ r0: 32, c0: 32 });
    const far = pan(w, dims, 4096, 10000, 10000);
    expect(toBounds(far, dims)).toEqual({ r0: 193, r1: 257, c0: 199, c1: 263 });
    expect(pan(w, dims, 4096, -5, -5)).toMatchObject({ r0: 0, c0: 0 });
  });

  it("zoom keeps the anchor cell at the same place and never exceeds the budget", () => {
    const w = { r0: 32, c0: 32, h: 64, w: 64 };
    const z = zoom(w, dims, 4096, "in", { row: 42, col: 52 })!;
    expect(z).toEqual({ r0: 37, c0: 42, h: 32, w: 32 });
    const back = zoom(z, dims, 4096, "out", { row: 42, col: 52 })!;
    expect(back.h * back.w).toBe(4096);
    expect(42 >= back.r0 && 42 < back.r0 + back.h && 52 >= back.c0 && 52 < back.c0 + back.w).toBe(true);
    expect(zoom(back, dims, 4096, "out")).toBeNull(); // 128 x 64 would be 8,192 cells
    expect(zoom(back, dims, 16384, "out")).toMatchObject({ h: 128, w: 128 });
    expect(zoom({ r0: 0, c0: 0, h: 1, w: 1 }, dims, 4096, "in")).toBeNull();
    for (let cap = 1; cap < 70000; cap = cap * 3 + 1) {
      let cur = defaultWin(dims, cap);
      for (let k = 0; k < 6; k++) cur = zoom(cur, dims, cap, "out") ?? cur;
      expect(cur.h * cur.w).toBeLessThanOrEqual(cap);
    }
  });

  it("ensureVisible and centreOn bring a cell into the window", () => {
    const w = defaultWin(dims, 4096);
    expect(ensureVisible(w, dims, 4096, 6, 262)).toMatchObject({ r0: 0, c0: 199 });
    expect(centreOn(w, dims, 4096, 256, 262)).toMatchObject({ r0: 193, c0: 199 });
    expect(clampWin({ r0: -4, c0: 0, h: 9999, w: 9999 }, dims, 4096).h * 1).toBeLessThanOrEqual(257);
  });

  it("bounds are checked before a request", () => {
    expect(boundsError({ r0: 0, r1: 64, c0: 0, c1: 64 }, dims, 4096)).toBeNull();
    expect(boundsError({ r0: 0, r1: 64, c0: 0, c1: 65 }, dims, 4096)).toMatch(/budget/);
    expect(boundsError({ r0: 250, r1: 258, c0: 0, c1: 1 }, dims, 4096)).toMatch(/outside/);
    expect(boundsError({ r0: 3, r1: 3, c0: 0, c1: 1 }, dims, 4096)).toMatch(/outside/);
    expect(boundsError({ r0: 0.5, r1: 3, c0: 0, c1: 1 }, dims, 4096)).toMatch(/whole numbers/);
  });
});
