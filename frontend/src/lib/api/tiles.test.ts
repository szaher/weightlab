import { describe, expect, it } from "vitest";
import { fakeApi, json, withToken } from "../../test/fakeApi";
import { KEY, tileBody, tileHeaders, tileResponse } from "../../test/tiles";
import { cancelJob, decodeTile, fetchTile, fetchTileMetadata, getJob, MAX_TILE_BYTES } from "./tiles";
import type { TileRequest } from "./types";

const req = (level = 0, x = 0, y = 0, mode: "auto" | "exact" | "row_sampled" = "auto"): TileRequest => ({
  source: { kind: "tensor", model_id: "mod_1", tensor_id: "tns_1" },
  view: { row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false },
  tile: { level, x, y, size: 256 },
  aggregation: "mean",
  quality: { mode },
});

const h = (o: Parameters<typeof tileHeaders>[0], patch: Record<string, string | null> = {}) => {
  const out = new Headers(tileHeaders(o));
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) out.delete(k);
    else out.set(k, v);
  }
  return out;
};

describe("binary tile decoding", () => {
  it("decodes little-endian float32 values and status bytes in row-major order", () => {
    const body = tileBody([1.5, -2.25, NaN, NaN], [0, 6, 3, 16]); // placeholders of non-drawable cells are NaN
    // Byte layout pinned: 1.5f = 0x3FC00000 little-endian.
    expect([...new Uint8Array(body, 0, 4)]).toEqual([0x00, 0x00, 0xc0, 0x3f]);
    const t = decodeTile(h({ rows: 2, cols: 2, level: 1, viewRows: 4, viewCols: 4 }), body, req(1), { viewRows: 4, viewCols: 4 });
    expect(t.rows).toBe(2);
    expect([...t.values.slice(0, 2)]).toEqual([1.5, -2.25]);
    expect([...t.status]).toEqual([0, 6, 3, 16]);
    expect(t.counts).toMatchObject({ drawable: 2, noFinite: 1, notRepresentable: 1, mixed: 1, withNan: 2, withPosInf: 1 });
    expect(t.bytes).toBe(20);
    expect(t.quality).toBe("exact");
    expect(t.sampledRows).toBeNull();
  });

  it("decodes clipped edge tiles of a 257 x 263 view at L0 and L3", () => {
    const l0 = decodeTile(h({ rows: 1, cols: 7, level: 0, x: 1, y: 1, viewRows: 257, viewCols: 263 }), tileBody(new Array(7).fill(1), new Array(7).fill(0)), req(0, 1, 1), { viewRows: 257, viewCols: 263 });
    expect([l0.rowStart, l0.rowEnd, l0.colStart, l0.colEnd]).toEqual([256, 257, 256, 263]);
    const l3 = decodeTile(h({ rows: 33, cols: 33, level: 3, viewRows: 257, viewCols: 263 }), tileBody(new Array(33 * 33).fill(1), new Array(33 * 33).fill(0)), req(3), { viewRows: 257, viewCols: 263 });
    expect([l3.rowEnd, l3.colEnd, l3.rows, l3.cols]).toEqual([257, 263, 33, 33]);
  });

  it("carries the sampling disclosure of row-sampled tiles", () => {
    const t = decodeTile(h({ rows: 1, cols: 1, level: 9, viewRows: 300, viewCols: 300, quality: "row_sampled", sampledRows: 8 }), tileBody([2], [0]), req(9), { viewRows: 300, viewCols: 300 });
    expect(t.quality).toBe("row_sampled");
    expect(t.sampledRows).toBe(8);
  });

  const o = { rows: 1, cols: 2, level: 0, viewRows: 1, viewCols: 2 };
  const ok = () => tileBody([1, 2], [0, 0]);
  it.each([
    ["missing rows header", { "x-weightlab-rows": null }, /missing X-WeightLab-Rows/],
    ["non-numeric cols", { "x-weightlab-cols": "2x" }, /not a whole number/],
    ["negative start", { "x-weightlab-row-start": "-1" }, /not a whole number/],
    ["rows above 256", { "x-weightlab-rows": "257" }, /outside 1..256/],
    ["zero cols", { "x-weightlab-cols": "0" }, /outside 1..256/],
    ["level mismatch", { "x-weightlab-level": "1" }, /answers a request for level 0/],
    ["unknown quality", { "x-weightlab-quality": "approximate" }, /quality/],
    ["wrong precision", { "x-weightlab-value-precision": "float64" }, /value precision/],
    ["aggregation mismatch", { "x-weightlab-aggregation": "max" }, /aggregation max/],
    ["bad cache key", { "x-weightlab-cache-key": "XYZ" }, /cache key/],
    ["wrong content type", { "content-type": "application/octet-stream" }, /content type/],
    ["ranges do not match cells", { "x-weightlab-col-end": "3" }, /do not match rows\/cols|does not match the 1 x 2 view/],
    ["exact tile with sampled rows", { "x-weightlab-sampled-rows": "4" }, /exact tile carries/],
    ["sampled tile without sampled rows", { "x-weightlab-quality": "row_sampled" }, /missing X-WeightLab-Sampled-Rows/],
    ["content-length disagrees", { "content-length": "11" }, /Content-Length/],
  ])("rejects %s", (_n, patch, re) => {
    expect(() => decodeTile(h(o, patch), ok(), req(0), { viewRows: 1, viewCols: 2 })).toThrow(re);
  });

  it("rejects a tile starting somewhere else than the requested tile, or not matching the view", () => {
    expect(() => decodeTile(h({ ...o }), ok(), req(0, 1, 0))).toThrow(/does not start where/);
    expect(() => decodeTile(h(o), ok(), req(0), { viewRows: 1, viewCols: 3 })).toThrow(/does not match the 1 x 3 view/);
  });

  it.each([
    ["short body", tileBody([1], [0]), /expected rows\*cols\*5 = 10/],
    ["long body", new ArrayBuffer(11), /expected rows\*cols\*5/],
    ["reserved bit 32", tileBody([1, 2], [0, 32]), /reserved status bits/],
    ["reserved bit 128", tileBody([1, 2], [128, 0]), /reserved status bits/],
    ["drawable NaN", tileBody([NaN, 2], [0, 0]), /drawable cell 0 has no finite value/],
    ["drawable Inf", tileBody([1, Infinity], [0, 0]), /drawable cell 1/],
    ["finite placeholder", tileBody([1, 2], [3, 0]), /non-drawable cell 0 carries a finite value/],
    ["NO_FINITE without a kind", tileBody([NaN, 2], [1, 0]), /NO_FINITE without a non-finite kind/],
    ["NO_FINITE with NOT_REPRESENTABLE", tileBody([NaN, 2], [19, 0]), /together/],
    ["level-0 mixed status", tileBody([1, 2], [6, 0]), /level-0 cell 0 has status 6/],
  ])("rejects %s", (_n, body, re) => {
    expect(() => decodeTile(h(o), body, req(0), { viewRows: 1, viewCols: 2 })).toThrow(re);
  });

  it("rejects a sampled answer to an exact request", () => {
    const hh = h({ ...o, quality: "row_sampled", sampledRows: 2 });
    expect(() => decodeTile(hh, ok(), req(0, 0, 0, "exact"))).toThrow(/exact request was answered with a sampled tile/);
  });

  it("bounds the payload", () => {
    expect(MAX_TILE_BYTES).toBe(256 * 256 * 5);
  });
});

describe("POST /tiles client", () => {
  it("sends the binary Accept header with the bearer token and decodes a 200 without parsing JSON", async () => {
    withToken();
    const calls = fakeApi((c) => tileResponse(c.body as TileRequest, { rows: 3, cols: 3 }));
    const r = await fetchTile(req(0), undefined, { viewRows: 3, viewCols: 3 });
    expect(r.kind).toBe("tile");
    const c = calls[0]!;
    expect(c.url.pathname).toBe("/api/v1/tiles");
    expect(c.headers.Accept).toBe("application/vnd.weightlab.tile.v2");
    expect(c.headers.Authorization).toMatch(/^Bearer /);
    expect(c.init.credentials).toBe("omit");
    expect(c.init.cache).toBe("no-store");
    expect(c.url.search).toBe("");
  });

  it("returns the job of a 202", async () => {
    withToken();
    fakeApi(() => json({ job_id: "job_1", kind: "pyramid_build", state: "queued", progress: 0, poll_after_ms: 500, created_at: "2026-01-01T00:00:00Z" }, 202));
    const r = await fetchTile(req(4));
    expect(r).toMatchObject({ kind: "job", job: { job_id: "job_1", state: "queued" } });
  });

  it("refuses a 202 that is not a job and a 200 that is JSON", async () => {
    withToken();
    fakeApi(() => json({ nope: 1 }, 202));
    await expect(fetchTile(req(4))).rejects.toMatchObject({ code: "BAD_RESPONSE" });
    fakeApi(() => json({ metadata: {} }, 200));
    await expect(fetchTile(req(0))).rejects.toMatchObject({ code: "BAD_RESPONSE", message: /content type application\/json/ });
  });

  it("refuses an oversized declared body before reading it", async () => {
    withToken();
    fakeApi(() => new Response(new ArrayBuffer(8), { status: 200, headers: { ...tileHeaders({ rows: 1, cols: 1, level: 0, viewRows: 1, viewCols: 1 }), "content-length": String(MAX_TILE_BYTES + 1) } }));
    await expect(fetchTile(req(0))).rejects.toMatchObject({ code: "BAD_RESPONSE", message: /exceeds/ });
  });

  it("keeps the error envelope of failures", async () => {
    withToken();
    fakeApi(() => json({ error: { code: "FILE_CHANGED", message: "changed", retryable: false }, request_id: "req_1" }, 409));
    await expect(fetchTile(req(0))).rejects.toMatchObject({ code: "FILE_CHANGED", status: 409 });
  });

  it("checks metadata against the tile it describes", async () => {
    withToken();
    fakeApi((c) => (c.url.pathname === "/api/v1/tiles" ? tileResponse(c.body as TileRequest, { rows: 2, cols: 2 }, { quality: "row_sampled", sampledRows: 2 }) : json({})));
    const r = await fetchTile(req(0));
    if (r.kind !== "tile") throw new Error("expected a tile");
    const meta = { schema_version: 2, cache_key: KEY, rows: 2, cols: 2, row_start: 0, row_end: 2, col_start: 0, col_end: 2, level: 0, quality: "row_sampled", value_precision: "float32_visualization", sampled_rows: 2, sampling_policy: "splitmix64_rows", seed: 7, aggregation: "mean", finite_count: 4, nonfinite_count: 0, source: {}, view: {} };
    fakeApi(() => json(meta));
    await expect(fetchTileMetadata(r.tile)).resolves.toMatchObject({ seed: 7 });
    fakeApi(() => json({ ...meta, quality: "exact", sampling_policy: null }));
    await expect(fetchTileMetadata(r.tile)).rejects.toMatchObject({ code: "BAD_RESPONSE" });
    fakeApi(() => json({ ...meta, cache_key: "b".repeat(64) }));
    await expect(fetchTileMetadata(r.tile)).rejects.toMatchObject({ code: "BAD_RESPONSE" });
  });

  it("polls and cancels jobs on their own routes", async () => {
    withToken();
    const calls = fakeApi((c) => json({ job_id: "job_1", kind: "pyramid_build", state: c.method === "POST" ? "canceled" : "running", progress: 0.5, created_at: "x" }));
    await expect(getJob("job_1")).resolves.toMatchObject({ state: "running" });
    await expect(cancelJob("job_1")).resolves.toMatchObject({ state: "canceled" });
    expect(calls.map((c) => `${c.method} ${c.url.pathname}`)).toEqual(["GET /api/v1/jobs/job_1", "POST /api/v1/jobs/job_1/cancel"]);
  });
});
