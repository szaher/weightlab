import { describe, expect, it } from "vitest";
import { ApiRequestError } from "../../lib/api/client";
import type { DecodedTile, TileResult } from "../../lib/api/tiles";
import type { TileRequest } from "../../lib/api/types";
import { DecodedTileCache, MAX_CONCURRENT, TileScheduler, type Outcome, type Want } from "./tileStore";

function tile(key = "a".repeat(64), bytes = 1000): DecodedTile {
  return { cacheKey: key, level: 0, rows: 1, cols: 1, rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 1, quality: "exact", valuePrecision: "float32_visualization", aggregation: "mean", sampledRows: null, values: new Float32Array(1), status: new Uint8Array(1), counts: { drawable: 1, noFinite: 0, notRepresentable: 0, withNan: 0, withPosInf: 0, withNegInf: 0, mixed: 0 }, bytes };
}

const req = (x: number): TileRequest => ({ source: { kind: "tensor", model_id: "m", tensor_id: "t" }, view: { row_axis: 0, col_axis: 1, fixed_indices: {}, transpose: false }, tile: { level: 0, x, y: 0, size: 256 }, aggregation: "mean", quality: { mode: "auto" } });
const want = (x: number, p = x): Want => ({ id: `0/${x}/0`, req: req(x), priority: p });

interface Pending {
  req: TileRequest;
  signal: AbortSignal;
  resolve: (r: TileResult) => void;
  reject: (e: unknown) => void;
}

function harness(cache = new DecodedTileCache()) {
  const pending: Pending[] = [];
  const results: [string, Outcome][] = [];
  const s = new TileScheduler(
    cache,
    (r, signal) =>
      new Promise<TileResult>((resolve, reject) => {
        pending.push({ req: r, signal, resolve, reject });
        signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
    (id, o) => results.push([id, o]),
  );
  return { s, pending, results, cache };
}
const flush = () => new Promise((r) => setTimeout(r, 0));

describe("decoded tile cache", () => {
  it("evicts least recently used tiles by bytes", () => {
    const c = new DecodedTileCache(3000);
    c.put("a", "ns", tile("1", 1000));
    c.put("b", "ns", tile("2", 1000));
    c.put("c", "ns", tile("3", 1000));
    c.get("a");
    c.put("d", "ns", tile("4", 1000));
    expect(c.peek("b")).toBeUndefined();
    expect(c.peek("a")).toBeDefined();
    expect(c.bytes).toBe(3000);
    expect(c.evictions).toBe(1);
    c.put("huge", "ns", tile("5", 4000)); // larger than the cache: not stored, nothing evicted
    expect(c.size).toBe(3);
    expect(c.purge((n) => n === "ns")).toBe(3);
    expect(c.bytes).toBe(0);
  });
});

describe("tile scheduler", () => {
  it("keeps at most the concurrency limit in flight, aborted requests included until they settle", async () => {
    const { s, pending } = harness();
    s.setNamespace("ns", { viewRows: 1, viewCols: 10_000 });
    s.setWanted(Array.from({ length: 20 }, (_, i) => want(i)));
    expect(pending.length).toBe(MAX_CONCURRENT);
    s.setLimit(2);
    s.setWanted(Array.from({ length: 20 }, (_, i) => want(i + 100))); // everything aborted
    expect(pending.length).toBe(MAX_CONCURRENT); // nothing new before the aborted ones settled
    await flush();
    expect(pending.length).toBe(MAX_CONCURRENT + 2);
    expect(s.inFlight).toBe(2);
    expect(s.peakConcurrent).toBeLessThanOrEqual(MAX_CONCURRENT);
  });

  it("shares one request per tile identity and serves it from the cache afterwards", async () => {
    const { s, pending, results } = harness();
    s.setNamespace("ns", { viewRows: 1, viewCols: 1000 });
    s.setWanted([want(0), want(0)]);
    s.setWanted([want(0)]);
    expect(pending.length).toBe(1);
    pending[0]!.resolve({ kind: "tile", tile: tile() });
    await flush();
    expect(results.map((r) => r[1].kind)).toEqual(["tile"]);
    s.setWanted([want(0)]);
    expect(pending.length).toBe(1);
    expect(s.cached("0/0/0")).toBeDefined();
  });

  it("drops late answers of a previous namespace or of tiles no longer wanted", async () => {
    const { s, pending, results, cache } = harness();
    s.setNamespace("tensor-A", { viewRows: 1, viewCols: 1000 });
    s.setWanted([want(0)]);
    const a = pending[0]!;
    s.setNamespace("tensor-B", { viewRows: 1, viewCols: 1000 });
    expect(a.signal.aborted).toBe(true);
    a.resolve({ kind: "tile", tile: tile() }); // the late answer (ignored even if the abort lost the race)
    await flush();
    expect(results).toEqual([]);
    expect(cache.size).toBe(0);
    expect(s.cached("0/0/0")).toBeUndefined();
  });

  it("re-validates a namespace before showing cached tiles; a different cache key purges it", async () => {
    const cache = new DecodedTileCache();
    const first = harness(cache);
    first.s.setNamespace("ns", { viewRows: 1, viewCols: 1000 });
    first.s.setWanted([want(0), want(1)]);
    first.pending[0]!.resolve({ kind: "tile", tile: tile("a".repeat(64)) });
    first.pending[1]!.resolve({ kind: "tile", tile: tile("c".repeat(64)) });
    await flush();
    expect(cache.size).toBe(2);
    // Reopened later (another overview of the same namespace): cached tiles are hidden until confirmed.
    const again = harness(cache);
    again.s.setNamespace("ns", { viewRows: 1, viewCols: 1000 });
    expect(again.s.cached("0/0/0")).toBeUndefined();
    again.s.setWanted([want(0)]);
    again.pending[0]!.resolve({ kind: "tile", tile: tile("b".repeat(64)) }); // the files changed: new key
    await flush();
    expect(cache.peek("ns|0/1/0")).toBeUndefined(); // the other old tile is gone too
    expect(again.s.cached("0/0/0")?.cacheKey).toBe("b".repeat(64));
  });

  it("recovers a re-opened namespace whose requested region was never cached, without exposing the old tiles", async () => {
    const cache = new DecodedTileCache();
    const first = harness(cache);
    first.s.setNamespace("ns", { viewRows: 1, viewCols: 100_000 });
    first.s.setWanted([want(0)]); // tensor A, region X
    first.pending[0]!.resolve({ kind: "tile", tile: tile("a".repeat(64)) });
    await flush();
    expect(cache.peek("ns|0/0/0")).toBeDefined();
    // Switch away (another tensor), then come back to A and go straight to region Y, never cached.
    first.s.setNamespace("other", { viewRows: 1, viewCols: 100_000 });
    const back = harness(cache);
    back.s.setNamespace("ns", { viewRows: 1, viewCols: 100_000 });
    back.s.setWanted([want(5), want(6)]);
    expect(back.s.cached("0/0/0")).toBeUndefined(); // hidden until confirmed
    back.pending[0]!.resolve({ kind: "tile", tile: tile("y".repeat(64)) });
    back.pending[1]!.resolve({ kind: "tile", tile: tile("z".repeat(64)) });
    await flush();
    // Y renders (before the fix it stayed hidden behind "Loading overview" forever).
    expect(back.s.cached("0/5/0")?.cacheKey).toBe("y".repeat(64));
    expect(back.s.cached("0/6/0")?.cacheKey).toBe("z".repeat(64));
    expect(back.results.map((r) => r[0])).toEqual(["0/5/0", "0/6/0"]);
    // The unconfirmed region-X tile was dropped, so going back to X re-reads it instead of showing old bytes.
    expect(back.s.cached("0/0/0")).toBeUndefined();
    expect(cache.peek("ns|0/0/0")).toBeUndefined();
    back.s.setWanted([want(0)]);
    expect(back.pending.length).toBe(3);
  });

  it("parks tiles answered with a job until the job ends, and retries 429 instead of failing", async () => {
    const { s, pending, results } = harness();
    s.setNamespace("ns", { viewRows: 1, viewCols: 1000 });
    s.setWanted([want(0)]);
    pending[0]!.resolve({ kind: "job", job: { job_id: "j", kind: "pyramid_build", state: "queued", progress: 0, created_at: "x" } });
    await flush();
    s.setWanted([want(0)]);
    expect(pending.length).toBe(1); // parked: no request storm while the job runs
    s.retryPending();
    expect(pending.length).toBe(2);
    pending[1]!.reject(new ApiRequestError({ status: 429, code: "RATE_LIMITED", message: "busy" }));
    await new Promise((r) => setTimeout(r, 250));
    expect(pending.length).toBe(3);
    expect(results.filter((r) => r[1].kind === "error")).toEqual([]);
  });
});
