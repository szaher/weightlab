// The browser's decoded-tile cache and request scheduler. This is NOT the server's pyramid cache (doc 10
// section 6): it only keeps tiles this tab already decoded, so panning back does not re-download them.
//
// * Bounded: least-recently-used eviction to DECODED_CACHE_BYTES of decoded arrays (values + status).
// * Scoped: every entry belongs to a namespace (model, tensor, view, aggregation, quality request). A namespace is
//   re-validated when an overview opens it again: its cached tiles are not shown until the server answers the
//   first tile request. Same cache key as the cached copy of that tile (the server's key covers the files'
//   fingerprints): the cached tiles are confirmed. A different key, no cached copy to compare with, an error or
//   FILE_CHANGED: the older tiles are dropped and the namespace continues fresh. Stale bytes are never redrawn.
// * Network: at most MAX_CONCURRENT requests in flight, identical tile identities share one request, and tiles
//   that leave the wanted set are aborted (or dropped from the queue) at once.
import { ApiRequestError, isAbort } from "../../lib/api/client";
import type { DecodedTile, ExpectedGeometry, TileResult } from "../../lib/api/tiles";
import type { Job, TileRequest } from "../../lib/api/types";

export const DECODED_CACHE_BYTES = 64 * 1024 * 1024;
export const MAX_CONCURRENT = 6;
export const MAX_RETRIES = 6;

export interface Entry {
  ns: string;
  tile: DecodedTile;
}

/** LRU of decoded tiles bounded by bytes. */
export class DecodedTileCache {
  private map = new Map<string, Entry>();
  private used = 0;
  readonly capacity: number;
  evictions = 0;

  constructor(capacity = DECODED_CACHE_BYTES) {
    this.capacity = capacity;
  }

  get bytes(): number {
    return this.used;
  }

  get size(): number {
    return this.map.size;
  }

  /** The entry, marked most recently used. */
  get(id: string): Entry | undefined {
    const e = this.map.get(id);
    if (e) {
      this.map.delete(id);
      this.map.set(id, e);
    }
    return e;
  }

  peek(id: string): Entry | undefined {
    return this.map.get(id);
  }

  put(id: string, ns: string, tile: DecodedTile): void {
    if (tile.bytes > this.capacity) return;
    this.delete(id);
    this.map.set(id, { ns, tile });
    this.used += tile.bytes;
    for (const [k, e] of this.map) {
      if (this.used <= this.capacity) break;
      this.map.delete(k);
      this.used -= e.tile.bytes;
      this.evictions++;
    }
  }

  delete(id: string): void {
    const e = this.map.get(id);
    if (e) {
      this.map.delete(id);
      this.used -= e.tile.bytes;
    }
  }

  /** Drops every entry of a namespace (or of every namespace `match` accepts). Returns how many. */
  purge(match: (ns: string) => boolean): number {
    let n = 0;
    for (const [k, e] of [...this.map]) {
      if (match(e.ns)) {
        this.delete(k);
        n++;
      }
    }
    return n;
  }

  hasNamespace(ns: string): boolean {
    for (const e of this.map.values()) if (e.ns === ns) return true;
    return false;
  }
}

/** One cache for the whole tab (tiles survive switching tensors; namespaces keep them apart). */
export const sharedTileCache = new DecodedTileCache();

export type Outcome =
  | { kind: "tile"; tile: DecodedTile; fromCache: boolean }
  | { kind: "job"; job: Job }
  | { kind: "error"; error: unknown };

export interface Want {
  id: string; // tile identity within the namespace (level/x/y)
  req: TileRequest;
  priority: number; // lower first
}

export type FetchTile = (req: TileRequest, signal: AbortSignal, expect?: ExpectedGeometry) => Promise<TileResult>;

interface Flight {
  ctl: AbortController;
  want: Want;
}

/** Schedules tile requests for one namespace at a time. Results arrive through `onResult`, only for the
 * namespace that is current when they settle and only for tiles still wanted; everything else is discarded. */
export class TileScheduler {
  private ns = "";
  private nsGen = 0;
  private validated = true;
  private wanted = new Map<string, Want>();
  private queue: Want[] = [];
  private flights = new Map<string, Flight>();
  private expect: ExpectedGeometry | undefined;
  /** Tile ids answered with 202 for the current job: re-requested only after `retryPending`. */
  private parked = new Set<string>();
  /** Requests issued and not yet settled, aborted ones included: an aborted fetch still counts until its promise
   * settles, so the bound holds for real network concurrency, not just for the wanted set. */
  private active = 0;
  /** Concurrency limit: MAX_CONCURRENT, or less when the server computes fewer tiles at once (max_active_tiles);
   * sending more would only queue them server-side or have them refused with 429. */
  private limit = MAX_CONCURRENT;
  /** RATE_LIMITED answers are retried with a growing delay instead of being shown as errors. */
  private retries = new Map<string, number>();
  private timers = new Set<ReturnType<typeof setTimeout>>();
  peakConcurrent = 0;
  requests = 0;
  aborted = 0;

  constructor(
    private readonly cache: DecodedTileCache,
    private readonly fetchTile: FetchTile,
    private readonly onResult: (id: string, o: Outcome) => void,
  ) {}

  get namespace(): string {
    return this.ns;
  }

  setLimit(n: number): void {
    this.limit = Math.max(1, Math.min(MAX_CONCURRENT, Math.floor(n) || 1));
    this.pump();
  }

  get inFlight(): number {
    return this.active;
  }

  /** Switches to namespace `ns` (a different tensor, view, aggregation or quality request). Every request of the
   * previous namespace is aborted; cached tiles of `ns` are held back until it re-validates. */
  setNamespace(ns: string, expect: ExpectedGeometry): void {
    if (ns === this.ns) {
      this.expect = expect;
      return;
    }
    this.abortAll();
    this.ns = ns;
    this.nsGen++;
    this.expect = expect;
    this.wanted.clear();
    this.queue = [];
    this.parked.clear();
    this.validated = !this.cache.hasNamespace(ns);
  }

  /** Re-validates the current namespace: cached tiles are hidden until the server confirms one again. */
  revalidate(): void {
    this.validated = !this.cache.hasNamespace(this.ns);
  }

  private key(id: string): string {
    return `${this.ns}|${id}`;
  }

  /** A cached tile of the current namespace, if it may be shown. */
  cached(id: string): DecodedTile | undefined {
    if (!this.validated) return undefined;
    return this.cache.get(this.key(id))?.tile;
  }

  /** Declares the tiles that should be loaded now; anything no longer wanted is aborted or dequeued. */
  setWanted(wants: Want[]): void {
    const next = new Map(wants.map((w) => [w.id, w]));
    for (const [id, f] of [...this.flights]) {
      if (!next.has(id)) {
        f.ctl.abort();
        this.flights.delete(id);
        this.aborted++;
      }
    }
    this.wanted = next;
    this.queue = wants
      .filter((w) => !this.flights.has(w.id) && !this.parked.has(w.id) && this.cached(w.id) === undefined)
      .sort((a, b) => a.priority - b.priority);
    this.pump();
  }

  /** After a pyramid job completed (or was canceled/failed): parked tiles may be requested again. */
  retryPending(): void {
    this.parked.clear();
    this.setWanted([...this.wanted.values()]);
  }

  /** Drops the current namespace's cached tiles (e.g. after FILE_CHANGED). */
  purgeNamespace(): void {
    const ns = this.ns;
    this.cache.purge((n) => n === ns);
    this.validated = true;
  }

  abortAll(): void {
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    for (const f of this.flights.values()) {
      f.ctl.abort();
      this.aborted++;
    }
    this.flights.clear();
    this.queue = [];
  }

  private pump(): void {
    while (this.active < this.limit && this.queue.length > 0) {
      const w = this.queue.shift();
      if (!w) break;
      if (this.flights.has(w.id) || !this.wanted.has(w.id)) continue;
      this.start(w);
    }
    this.peakConcurrent = Math.max(this.peakConcurrent, this.active);
  }

  private start(w: Want): void {
    const ctl = new AbortController();
    const gen = this.nsGen;
    const ns = this.ns;
    const flight: Flight = { ctl, want: w };
    this.flights.set(w.id, flight);
    this.requests++;
    this.active++;
    this.fetchTile(w.req, ctl.signal, this.expect)
      .then(
        (r) => this.settle(gen, ns, w.id, flight, r.kind === "tile" ? { kind: "tile", tile: r.tile, fromCache: false } : { kind: "job", job: r.job }),
        (error: unknown) => {
          if (isAbort(error)) return;
          this.settle(gen, ns, w.id, flight, { kind: "error", error });
        },
      )
      .finally(() => {
        this.active--;
        this.pump(); // a slot is free (also after an abort)
      });
  }

  private settle(gen: number, ns: string, id: string, flight: Flight, o: Outcome): void {
    // Late answers of another namespace, of an aborted request or of a tile nobody wants any more are dropped.
    if (gen !== this.nsGen || flight.ctl.signal.aborted || this.flights.get(id) !== flight) return;
    this.flights.delete(id);
    if (o.kind === "error" && o.error instanceof ApiRequestError && o.error.code === "RATE_LIMITED") {
      const n = (this.retries.get(id) ?? 0) + 1;
      if (n <= MAX_RETRIES) {
        this.retries.set(id, n);
        const t = setTimeout(() => {
          this.timers.delete(t);
          if (gen !== this.nsGen) return;
          const w = this.wanted.get(id);
          if (w && !this.flights.has(id)) {
            this.queue.unshift(w);
            this.pump();
          }
        }, 200 * n);
        this.timers.add(t);
        return;
      }
    }
    this.retries.delete(id);
    if (o.kind === "tile") {
      const key = this.key(id);
      if (!this.validated) {
        // The first fresh tile of a re-opened namespace decides what happens to its older cached tiles. If it has
        // a cached counterpart with the same key, the server confirmed them (same files): keep them. Otherwise
        // (a different key, or nothing cached for this tile, e.g. the user moved to a region not seen before)
        // they cannot be confirmed: drop them and continue as a fresh, validated session. Either way the
        // namespace never stays hidden once the server has answered with a tile.
        const old = this.cache.peek(key);
        if (!old || old.tile.cacheKey !== o.tile.cacheKey) this.cache.purge((n) => n === ns);
        this.validated = true;
      }
      this.cache.put(key, ns, o.tile);
    } else if (o.kind === "job") {
      this.parked.add(id);
    } else if (!this.validated) {
      this.cache.purge((n) => n === ns); // cannot confirm the cached tiles: never show them
      this.validated = true;
    }
    if (this.wanted.has(id)) this.onResult(id, o);
  }
}
