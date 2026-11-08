// The whole-tensor overview (Phase 1): a multiresolution heatmap of one 2D view, assembled from POST /tiles.
//
// * Coordinates: geometry.ts. Everything the user reads is a raw view coordinate (half-open), never a tile index.
// * Requests: TileScheduler (tileStore.ts): <= 6 in flight, identical tiles share one request, tiles that leave
//   the screen are aborted, and a late answer of an earlier tensor/view/aggregation/quality is discarded.
// * Viewport changes during a drag are kept in a ref and drawn on the next animation frame; React state (and with
//   it the wanted tiles) is committed at most every COMMIT_MS, so panning does not re-render per pointer event.
// * Jobs: a 202 parks the tile and shows the build's progress; polling follows poll_after_ms and stops on any
//   terminal state or unmount. Navigating away aborts polling but never cancels the (possibly shared) job; only
//   the Cancel button does, once.
// * Quality: every visible tile's coverage is disclosed (exact / row-sampled + rows, policy, seed); the values are
//   float32 visualizations of aggregates and never stand in for exact scalars (Exact ROI does that).
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent, type PointerEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError, isAbort } from "../../lib/api/client";
import { cancelJob, fetchTile, fetchTileMetadata, getJob, type DecodedTile } from "../../lib/api/tiles";
import { AGGREGATIONS, PENDING_JOB_STATES, type Aggregation, type Job, type Settings, type Tensor, type TileMetadata, type TileRequest } from "../../lib/api/types";
import { formatBytes, formatInt } from "../../lib/format";
import { useResource } from "../../lib/useResource";
import { readTheme } from "../roi/paint";
import { fixedRangeError, MODE_LABEL, PALETTE_LABEL, type FixedRange, type Palette, type ScaleMode } from "../roi/scale";
import { toRequestView, type ViewState } from "../roi/view";
import {
  autoLevel,
  block,
  cellBounds,
  cellOf,
  cellRangeBounds,
  clampViewport,
  fitViewport,
  orderedTiles,
  panBy,
  rangeCount,
  rawAt,
  rectCells,
  rectText,
  tileBounds,
  tilesFor,
  topLevel,
  visibleRect,
  withRing,
  zoomAt,
  zoomToRect,
  type Dims,
  type RawRect,
  type Size,
  type Viewport,
} from "./geometry";
import { AGG_LABEL, AGG_MEANING, blockText, PRECISION_TEXT, QUALITY_BADGE, qualityText, statusText } from "./labels";
import { OverviewLegend, type LegendCounts } from "./OverviewLegend";
import { AXIS_LEFT, AXIS_TOP, makeBitmap, paintOverview, releaseBitmap, scaleKey, tilesScale, type Bitmap } from "./paintOverview";
import { sharedTileCache, TileScheduler, type Outcome, type Want } from "./tileStore";

export const COMMIT_MS = 120;
export const PLOT_HEIGHT = 480;
export const MIN_PLOT_HEIGHT = 200;
export const MAX_VISIBLE_TILES = 48;
const MAX_BITMAPS = 96;
const ZOOM_STEP = 2;

interface Props {
  modelId: string;
  tensor: Tensor;
  view: ViewState;
  dims: Dims;
  settings: Settings | undefined;
  /** Kept by the parent so it survives a view change (which remounts the overview on the new view). */
  aggregation: Aggregation;
  onAggregation: (a: Aggregation) => void;
  onViewChange: (v: ViewState) => void;
  /** Open Exact ROI on raw bounds `r` of this view. */
  onInspect: (r: RawRect) => void;
}

interface QualityReq {
  mode: "auto" | "exact";
  seed: number | null;
}

type TileState = { kind: "pending" } | { kind: "error"; error: unknown };

function useWidth(): [(el: HTMLDivElement | null) => void, number] {
  const [width, setWidth] = useState(0);
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback((el: HTMLDivElement | null) => {
    ro.current?.disconnect();
    ro.current = null;
    if (!el) return;
    setWidth(Math.floor(el.getBoundingClientRect().width));
    if (typeof ResizeObserver === "undefined") return;
    ro.current = new ResizeObserver((e) => setWidth(Math.floor(e[0]?.contentRect.width ?? 0)));
    ro.current.observe(el);
  }, []);
  return [ref, width];
}

function useDpr(): number {
  const [dpr, setDpr] = useState(() => (typeof window === "undefined" ? 1 : window.devicePixelRatio || 1));
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(`(resolution: ${dpr}dppx)`);
    const on = () => setDpr(window.devicePixelRatio || 1);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [dpr]);
  return dpr;
}

interface JobBox {
  ns: string;
  job: Job | null;
  error: unknown;
  canceling: boolean;
  /** Consecutive failed status polls; polling backs off and stops after MAX_POLL_FAILURES. */
  failures: number;
  /** Bumped by "Check again" to restart polling after it gave up. */
  pollNonce: number;
}
const emptyBox = (ns: string): JobBox => ({ ns, job: null, error: null, canceling: false, failures: 0, pollNonce: 0 });
export const MAX_POLL_FAILURES = 5;
/** Delay before the next status poll: poll_after_ms normally, 1, 2, 4, 8 s after consecutive failures. */
export const pollBackoff = (j: Job, failures: number): number => (failures === 0 ? pollDelay(j) : Math.min(8_000, 1_000 * 2 ** (failures - 1)));

const pollDelay = (j: Job) => Math.min(10_000, Math.max(100, j.poll_after_ms ?? 500));
const isPending = (j: Job | null) => !!j && PENDING_JOB_STATES.includes(j.state);

export function Overview({ modelId, tensor, view, dims, settings, aggregation: agg, onAggregation: setAgg, onViewChange, onInspect }: Props) {
  const [wrapRef, width] = useWidth();
  const dpr = useDpr();
  // Plot height follows the view's aspect ratio within [MIN_PLOT_HEIGHT, PLOT_HEIGHT], so wide matrices do not
  // leave empty bands and tall ones (embeddings) still get a usable height.
  const size: Size = useMemo(() => {
    const w = Math.max(160, width - AXIS_LEFT - 2);
    return { width: w, height: Math.round(Math.min(PLOT_HEIGHT, Math.max(MIN_PLOT_HEIGHT, (w * dims.rows) / Math.max(1, dims.cols)))) };
  }, [width, dims.rows, dims.cols]);

  const [levelMode, setLevelMode] = useState<"auto" | number>("auto");
  const [quality, setQuality] = useState<QualityReq>({ mode: "auto", seed: null });
  const [seedText, setSeedText] = useState("");
  const [mode, setMode] = useState<ScaleMode>("auto");
  const [palette, setPalette] = useState<Palette>("sequential");
  const [fixed, setFixed] = useState<FixedRange>({ min: "0", max: "1" });
  const [selection, setSelection] = useState<RawRect | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ rect: RawRect; text: string } | null>(null);
  const [dragSel, setDragSel] = useState<RawRect | null>(null);
  const [confirmExact, setConfirmExact] = useState(false);
  // The pyramid job of the current namespace (tensor, view, aggregation, quality). Kept with the namespace it
  // belongs to, so a job of an earlier namespace is never shown for the current one.
  const [jobBox, setJobBox] = useState<JobBox>(() => emptyBox(""));
  const [tick, setTick] = useState(0); // a tile arrived / failed
  const [announce, setAnnounce] = useState("");

  const reqView = useMemo(() => toRequestView(view), [view]);
  const top = topLevel(dims);

  // ---- viewport: ref for drawing, committed state for everything React renders
  const [vp, setVp] = useState<Viewport>(() => fitViewport(dims, size));
  const vpRef = useRef(vp);
  const sizeRef = useRef(size);
  const fitted = useRef(false);
  useEffect(() => {
    sizeRef.current = size;
    if (width > 0 && !fitted.current) {
      fitted.current = true;
      const f = fitViewport(dims, size);
      vpRef.current = f;
      setVp(f);
    } else {
      const c = clampViewport(vpRef.current, dims, size);
      vpRef.current = c;
      setVp(c);
    }
  }, [size, width, dims]);
  const commitTimer = useRef<number | null>(null);
  const commit = useCallback((now = false) => {
    if (now) {
      if (commitTimer.current !== null) window.clearTimeout(commitTimer.current);
      commitTimer.current = null;
      setVp(vpRef.current);
      return;
    }
    if (commitTimer.current !== null) return;
    commitTimer.current = window.setTimeout(() => {
      commitTimer.current = null;
      setVp(vpRef.current);
    }, COMMIT_MS);
  }, []);
  useEffect(() => () => {
    if (commitTimer.current !== null) window.clearTimeout(commitTimer.current);
  }, []);

  const level = levelMode === "auto" ? autoLevel(vp, dims) : Math.min(levelMode, top);

  // ---- requests
  const ns = useMemo(
    () => JSON.stringify({ m: modelId, t: tensor.id, v: reqView, a: agg, q: quality }),
    [modelId, tensor.id, reqView, agg, quality],
  );
  const states = useRef(new Map<string, TileState>());
  const onResult = useCallback((id: string, o: Outcome) => {
    if (o.kind === "tile") states.current.delete(id);
    else if (o.kind === "job") {
      states.current.set(id, { kind: "pending" });
      // A terminal state already seen for this job is never replaced by an older pending view of it.
      const at = schedRef.current?.namespace ?? "";
      setJobBox((b) => {
        const cur = b.ns === at ? b : emptyBox(at);
        return cur.job && cur.job.job_id === o.job.job_id && !isPending(cur.job) ? cur : { ...cur, job: o.job, error: null };
      });
    } else {
      states.current.set(id, { kind: "error", error: o.error });
      if (o.error instanceof ApiRequestError && ["FILE_CHANGED", "MISSING_SHARD", "NOT_FOUND"].includes(o.error.code)) {
        schedRef.current?.purgeNamespace();
      }
    }
    setTick((t) => t + 1);
  }, []);
  const schedRef = useRef<TileScheduler | null>(null);
  if (schedRef.current === null) schedRef.current = new TileScheduler(sharedTileCache, fetchTile, onResult);
  const sched = schedRef.current;

  // A new namespace: abort the old one's requests and forget its per-tile states before anything is drawn, so
  // not even one frame can show tiles of the previous tensor, view, aggregation or quality request.
  if (sched.namespace !== ns) {
    sched.setNamespace(ns, { viewRows: dims.rows, viewCols: dims.cols });
    states.current.clear();
  }
  const activeTiles = typeof settings?.max_active_tiles === "number" ? settings.max_active_tiles : 2;
  useEffect(() => sched.setLimit(activeTiles), [sched, activeTiles]);
  const box = jobBox.ns === ns ? jobBox : emptyBox(ns);
  const job = box.job;
  const jobError = box.error;
  const canceling = box.canceling;
  const pollFailures = box.failures;
  const pollNonce = box.pollNonce;
  const patchJob = useCallback((at: string, patch: Partial<JobBox>) => {
    setJobBox((b) => ({ ...(b.ns === at ? b : emptyBox(at)), ...patch }));
  }, []);
  useEffect(() => () => sched.abortAll(), [sched]);

  const visible = useMemo(() => visibleRect(vp, dims, size), [vp, dims, size]);
  const range = useMemo(() => tilesFor(visible, dims, level), [visible, dims, level]);
  const tooMany = range !== null && rangeCount(range) > MAX_VISIBLE_TILES;
  const tileReq = useCallback(
    (x: number, y: number, l: number): TileRequest => {
      const q: TileRequest["quality"] = { mode: quality.mode };
      if (quality.seed !== null) q.seed = quality.seed;
      return { source: { kind: "tensor", model_id: modelId, tensor_id: tensor.id }, view: reqView, tile: { level: l, x, y, size: 256 }, aggregation: agg, quality: q };
    },
    [modelId, tensor.id, reqView, agg, quality],
  );
  useEffect(() => {
    if (!range || tooMany) {
      sched.setWanted([]);
      return;
    }
    const wants: Want[] = [];
    const vis = orderedTiles(range);
    vis.forEach((t, i) => wants.push({ id: `${level}/${t.x}/${t.y}`, req: tileReq(t.x, t.y, level), priority: i }));
    const seen = new Set(wants.map((w) => w.id));
    // Prefetch: the ring of neighbouring tiles at the same level, after everything visible.
    orderedTiles(withRing(range, dims)).forEach((t, i) => {
      const id = `${level}/${t.x}/${t.y}`;
      if (!seen.has(id)) wants.push({ id, req: tileReq(t.x, t.y, level), priority: 10_000 + i });
    });
    // While the user is on a fine level, the top-level tile (one request) gives the context underneath.
    if (level < top) {
      const id = `${top}/0/0`;
      if (!seen.has(id)) wants.push({ id, req: tileReq(0, 0, top), priority: 5_000 });
    }
    sched.setWanted(wants);
  }, [range, tooMany, level, top, dims, tileReq, sched, ns]);

  // ---- tiles available to draw: current level, with coarser cached tiles underneath while finer ones load
  const drawn = useMemo(() => {
    void tick;
    const cur: DecodedTile[] = [];
    const under: DecodedTile[] = [];
    let missing = 0;
    let pending = 0;
    const errors: unknown[] = [];
    if (range && !tooMany) {
      for (const t of orderedTiles(range)) {
        const id = `${level}/${t.x}/${t.y}`;
        const tile = sched.cached(id);
        if (tile) {
          cur.push(tile);
          continue;
        }
        missing++;
        const st = states.current.get(id);
        if (st?.kind === "pending") pending++;
        else if (st?.kind === "error") errors.push(st.error);
        const b = tileBounds(dims, level, t.x, t.y);
        for (let l = level + 1; b && l <= top; l++) {
          const s = 256 * 2 ** l;
          const c = sched.cached(`${l}/${Math.floor(b.c0 / s)}/${Math.floor(b.r0 / s)}`);
          if (c) {
            if (!under.includes(c)) under.push(c);
            break;
          }
        }
      }
    }
    return { cur, under, missing, pending, errors };
  }, [tick, range, tooMany, level, sched, dims, top]);

  const scale = useMemo(() => tilesScale(drawn.cur.length > 0 ? drawn.cur : drawn.under, mode, fixed), [drawn, mode, fixed]);
  const counts: LegendCounts = useMemo(() => {
    const c: LegendCounts = { drawable: 0, noFinite: 0, notRepresentable: 0, mixed: 0, withNan: 0, withPosInf: 0, withNegInf: 0 };
    for (const t of drawn.cur) for (const k of Object.keys(c) as (keyof LegendCounts)[]) c[k] += t.counts[k];
    return c;
  }, [drawn]);
  const qualities = new Set(drawn.cur.map((t) => t.quality));
  const sampledTile = drawn.cur.find((t) => t.quality === "row_sampled") ?? null;
  const firstTile = sampledTile ?? drawn.cur[0] ?? null;
  const meta = useResource<TileMetadata>(firstTile ? `${ns}|${firstTile.cacheKey}` : null, (signal) =>
    firstTile ? fetchTileMetadata(firstTile, signal) : Promise.reject(new Error("no tile")),
  );

  // ---- painting
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bitmaps = useRef(new Map<string, Bitmap>());
  const drawState = useRef({ drawn, scale, palette, level, selection, hover: null as RawRect | null });
  drawState.current = { drawn, scale, palette, level, selection: dragSel ?? selection, hover: hoverInfo?.rect ?? null };
  const frame = useRef<number | null>(null);
  const paintNow = useCallback(() => {
    frame.current = null;
    const c = canvasRef.current;
    if (!c) return;
    const s = sizeRef.current;
    const w = Math.round((AXIS_LEFT + s.width) * dpr);
    const h = Math.round((AXIS_TOP + s.height) * dpr);
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    c.style.width = `${AXIS_LEFT + s.width}px`;
    c.style.height = `${AXIS_TOP + s.height}px`;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const st = drawState.current;
    const key = scaleKey(st.scale, st.palette);
    const tiles = [...st.drawn.under, ...st.drawn.cur].map((tile) => {
      let b = bitmaps.current.get(tile.cacheKey);
      if (!b || b.tile !== tile || b.scaleKey !== key) {
        const fresh = makeBitmap(tile, st.scale, st.palette, b?.canvas);
        if (fresh) {
          bitmaps.current.delete(tile.cacheKey);
          bitmaps.current.set(tile.cacheKey, fresh);
          b = fresh;
        }
      }
      return { bitmap: b ?? null, tile };
    });
    // Bounded bitmap pool: release the backing stores of the least recently drawn ones.
    while (bitmaps.current.size > MAX_BITMAPS) {
      const [k, b] = bitmaps.current.entries().next().value as [string, Bitmap];
      releaseBitmap(b);
      bitmaps.current.delete(k);
    }
    paintOverview(ctx, {
      size: s,
      dpr,
      view: vpRef.current,
      dims,
      level: st.level,
      tiles,
      selection: st.selection,
      hover: st.hover,
      theme: readTheme(c),
      showPatterns: true,
    });
  }, [dpr, dims]);
  const requestPaint = useCallback(() => {
    if (frame.current === null) frame.current = requestAnimationFrame(paintNow);
  }, [paintNow]);
  useEffect(() => {
    requestPaint();
  });
  useEffect(() => {
    const pool = bitmaps.current;
    const canvas = canvasRef.current;
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      for (const b of pool.values()) releaseBitmap(b);
      pool.clear();
      if (canvas) {
        canvas.width = 0;
        canvas.height = 0;
      }
    };
  }, []);

  // ---- jobs: poll while pending; never cancel on unmount. A failed poll is retried with backoff (bounded);
  // after MAX_POLL_FAILURES the panel offers "Check again". Terminal states and unmount stop polling.
  useEffect(() => {
    if (!job || !isPending(job) || pollFailures >= MAX_POLL_FAILURES) return;
    const ctl = new AbortController();
    const t = window.setTimeout(() => {
      getJob(job.job_id, ctl.signal).then(
        (next) => {
          if (ctl.signal.aborted) return;
          patchJob(ns, { job: next, error: null, failures: 0 });
          if (next.state === "completed") {
            setAnnounce("Exact overview ready.");
            sched.retryPending();
          } else if (!isPending(next)) setAnnounce(`Exact overview build ${next.state}.`);
        },
        (e: unknown) => {
          if (ctl.signal.aborted || isAbort(e)) return;
          patchJob(ns, { error: e, failures: pollFailures + 1 });
        },
      );
    }, pollBackoff(job, pollFailures));
    return () => {
      ctl.abort();
      window.clearTimeout(t);
    };
  }, [job, sched, ns, patchJob, pollFailures, pollNonce]);
  function onCheckAgain() {
    patchJob(ns, { failures: 0, error: null, pollNonce: pollNonce + 1 });
  }

  function onCancelJob() {
    if (!job || canceling) return;
    const at = ns;
    patchJob(at, { canceling: true });
    cancelJob(job.job_id).then(
      (j) => {
        patchJob(at, { job: j });
        setAnnounce("Exact overview build canceled.");
      },
      (e: unknown) => patchJob(at, { canceling: false, error: e }),
    );
  }
  function onRetryBuild() {
    patchJob(ns, { job: null, error: null, canceling: false, failures: 0 });
    states.current.clear();
    sched.retryPending();
  }

  // ---- interaction
  const setViewport = useCallback(
    (next: Viewport, now = false) => {
      vpRef.current = next;
      requestPaint();
      commit(now);
    },
    [commit, requestPaint],
  );
  function zoomBy(factor: number, at?: { x: number; y: number }) {
    const p = at ?? (selection ? centreOfRect(selection) : { x: size.width / 2, y: size.height / 2 });
    setViewport(zoomAt(vpRef.current, dims, size, factor, p.x, p.y), true);
  }
  function centreOfRect(r: RawRect) {
    const v = vpRef.current;
    return { x: ((r.c0 + r.c1) / 2 - v.left) * v.ppe, y: ((r.r0 + r.r1) / 2 - v.top) * v.ppe };
  }
  function onFit() {
    setLevelMode("auto");
    setViewport(fitViewport(dims, size), true);
  }
  function onReset() {
    setSelection(null);
    setLevelMode("auto");
    setMode("auto");
    setViewport(fitViewport(dims, size), true);
  }

  const drag = useRef<{ x: number; y: number; moved: boolean; rect: boolean; id: number; anchor: { r: number; c: number } } | null>(null);
  function plotPoint(e: PointerEvent<HTMLCanvasElement>) {
    const b = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - b.left - AXIS_LEFT, y: e.clientY - b.top - AXIS_TOP };
  }
  function cellRectAt(px: number, py: number): RawRect | null {
    if (px < 0 || py < 0 || px > size.width || py > size.height) return null;
    const { r, c } = rawAt(vpRef.current, px, py);
    if (r < 0 || c < 0 || r >= dims.rows || c >= dims.cols) return null;
    const { i, j } = cellOf(level, Math.floor(r), Math.floor(c));
    return cellBounds(dims, level, i, j);
  }
  function cellReadout(rect: RawRect): string {
    const { i, j } = cellOf(level, rect.r0, rect.c0);
    for (const t of drawn.cur) {
      const ti = i - t.rowStart / block(level);
      const tj = j - t.colStart / block(level);
      if (ti < 0 || tj < 0 || ti >= t.rows || tj >= t.cols) continue;
      const k = ti * t.cols + tj;
      const s = (t.status[k] ?? 0);
      const v = (s & 17) === 0 ? `${AGG_LABEL[agg]} ≈ ${(t.values[k] ?? 0).toPrecision(6)} (float32 visualization)` : "no colour";
      return `${rectText(rect)} · ${v} · ${statusText(s)} · ${t.quality === "exact" ? "exact coverage" : `row-sampled (${t.sampledRows} rows/block)`}`;
    }
    return `${rectText(rect)} · not loaded yet`;
  }
  function onPointerDown(e: PointerEvent<HTMLCanvasElement>) {
    if (e.button !== 0) return;
    const p = plotPoint(e);
    const { r, c } = rawAt(vpRef.current, p.x, p.y);
    drag.current = { x: e.clientX, y: e.clientY, moved: false, rect: e.shiftKey, id: e.pointerId, anchor: { r, c } };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onPointerMove(e: PointerEvent<HTMLCanvasElement>) {
    const d = drag.current;
    const p = plotPoint(e);
    if (d && d.id === e.pointerId) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) < 4) return;
      d.moved = true;
      if (d.rect) {
        const { r, c } = rawAt(vpRef.current, p.x, p.y);
        const clampR = (v: number) => Math.min(dims.rows - 1, Math.max(0, Math.floor(v)));
        const clampC = (v: number) => Math.min(dims.cols - 1, Math.max(0, Math.floor(v)));
        const a = cellOf(level, clampR(d.anchor.r), clampC(d.anchor.c));
        const b = cellOf(level, clampR(r), clampC(c));
        const rect = cellRangeBounds(dims, level, a, b);
        if (rect) {
          setDragSel(rect);
          setHoverInfo({ rect, text: `Selecting ${rectText(rect)} (${formatInt(rectCells(rect))} elements)` });
        }
      } else {
        d.x = e.clientX;
        d.y = e.clientY;
        setViewport(panBy(vpRef.current, dims, size, dx, dy));
      }
      return;
    }
    const rect = cellRectAt(p.x, p.y);
    if (!rect) {
      if (hoverInfo) setHoverInfo(null);
      return;
    }
    if (!hoverInfo || hoverInfo.rect.r0 !== rect.r0 || hoverInfo.rect.c0 !== rect.c0 || hoverInfo.rect.r1 !== rect.r1 || hoverInfo.rect.c1 !== rect.c1) {
      setHoverInfo({ rect, text: cellReadout(rect) });
    }
  }
  function onPointerUp(e: PointerEvent<HTMLCanvasElement>) {
    const d = drag.current;
    drag.current = null;
    if (!d || d.id !== e.pointerId) return;
    if (d.moved && d.rect) {
      if (dragSel) select(dragSel);
      setDragSel(null);
      setHoverInfo(null);
      return;
    }
    if (d.moved) {
      commit(true);
      return;
    }
    const p = plotPoint(e);
    const rect = cellRectAt(p.x, p.y);
    if (rect) select(rect);
  }
  function select(rect: RawRect) {
    setSelection(rect);
    setAnnounce(`Selected block ${rectText(rect)}.`);
  }
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey || e.altKey)) return; // plain wheel scrolls the page
      e.preventDefault();
      const b = c.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0025);
      vpRef.current = zoomAt(vpRef.current, dims, sizeRef.current, factor, e.clientX - b.left - AXIS_LEFT, e.clientY - b.top - AXIS_TOP);
      requestPaint();
      commit();
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [dims, commit, requestPaint]);

  function onKeyDown(e: KeyboardEvent<HTMLCanvasElement>) {
    const arrows: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const a = arrows[e.key];
    if (a) {
      e.preventDefault();
      if (e.shiftKey) {
        setViewport(panBy(vpRef.current, dims, size, -a[1] * size.width * 0.25, -a[0] * size.height * 0.25), true);
        return;
      }
      const vis = visibleRect(vpRef.current, dims, size);
      const from = selection ? cellOf(level, selection.r0, selection.c0) : cellOf(level, vis.r0, vis.c0);
      const step = selection ? 1 : 0;
      const lc = { rows: Math.ceil(dims.rows / block(level)), cols: Math.ceil(dims.cols / block(level)) };
      const i = Math.min(lc.rows - 1, Math.max(0, from.i + a[0] * step));
      const j = Math.min(lc.cols - 1, Math.max(0, from.j + a[1] * step));
      const rect = cellBounds(dims, level, i, j);
      if (!rect) return;
      select(rect);
      // Keep the selected block on screen.
      const v = vpRef.current;
      const vr = visibleRect(v, dims, size);
      if (rect.r0 < vr.r0 || rect.r1 > vr.r1 || rect.c0 < vr.c0 || rect.c1 > vr.c1) {
        const dyRaw = rect.r0 < vr.r0 ? rect.r0 - v.top : rect.r1 > vr.r1 ? rect.r1 - (v.top + size.height / v.ppe) : 0;
        const dxRaw = rect.c0 < vr.c0 ? rect.c0 - v.left : rect.c1 > vr.c1 ? rect.c1 - (v.left + size.width / v.ppe) : 0;
        setViewport(clampViewport({ ...v, top: v.top + dyRaw, left: v.left + dxRaw }, dims, size), true);
      }
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      zoomBy(ZOOM_STEP);
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault();
      zoomBy(1 / ZOOM_STEP);
    } else if (e.key === "0") {
      e.preventDefault();
      onFit();
    } else if (e.key === "Escape") {
      setSelection(null);
    } else if (e.key === "Enter" && selection) {
      e.preventDefault();
      inspect(selection);
    }
  }

  // ---- exact drill-down
  const capRaw = settings?.roi_max_cells;
  const cap = typeof capRaw === "number" && Number.isSafeInteger(capRaw) && capRaw > 0 ? capRaw : null;
  function inspect(r: RawRect) {
    if (cap !== null && rectCells(r) <= cap) onInspect(r);
  }
  function shrinkToCap(r: RawRect): RawRect {
    let h = r.r1 - r.r0;
    let w = r.c1 - r.c0;
    while (cap !== null && h * w > cap) {
      if (w >= h && w > 1) w = Math.ceil(w / 2);
      else h = Math.ceil(h / 2);
    }
    return { r0: r.r0, r1: r.r0 + h, c0: r.c0, c1: r.c0 + w };
  }

  // ---- disclosures
  const limitMib = typeof settings?.pyramid_exact_max_mib === "number" ? settings.pyramid_exact_max_mib : null;
  const aboveLimit = limitMib !== null && tensor.byte_length > limitMib * 1024 * 1024;
  const pending = isPending(job);
  const transposeSampleError = drawn.errors.find(
    (e) => e instanceof ApiRequestError && e.code === "RESOURCE_LIMIT" && /row sampling/i.test(e.message),
  );
  const otherError = drawn.errors.find((e) => e !== transposeSampleError);
  const allLoaded = !!range && !tooMany && drawn.missing === 0;
  const coverage: "exact" | "row_sampled" | "mixed" | null =
    qualities.size === 0 ? null : qualities.size > 1 ? "mixed" : qualities.has("exact") ? "exact" : "row_sampled";
  const selInfo = selection ? cellReadout(selection) : null;
  const selCells = selection ? rectCells(selection) : 0;
  const selectedTooBig = selection !== null && cap !== null && selCells > cap;

  function onSeed(e: FormEvent) {
    e.preventDefault();
    const t = seedText.trim();
    if (t === "") return setQuality({ ...quality, seed: null });
    if (!/^(0|[1-9][0-9]{0,9})$/.test(t) || Number(t) > 4294967295) return;
    setQuality({ ...quality, seed: Number(t) });
  }
  const seedInvalid = seedText.trim() !== "" && (!/^(0|[1-9][0-9]{0,9})$/.test(seedText.trim()) || Number(seedText.trim()) > 4294967295);

  const levelOptions = Array.from({ length: top + 1 }, (_, l) => l);
  const vr = visible;
  const description = `Whole-tensor heatmap of ${formatInt(dims.rows)} × ${formatInt(dims.cols)} view, level ${level} (${blockText(level)}), ${AGG_LABEL[agg]}, ${
    coverage === "exact" ? "exact coverage" : coverage === "row_sampled" ? "row-sampled preview, not exact" : coverage === "mixed" ? "some tiles exact, some row-sampled" : "loading"
  }. Visible ${rectText(vr)}.`;

  return (
    <section className="overview" aria-labelledby="overview-h" data-testid="overview">
      <div className="roi-head">
        <h3 id="overview-h" className="h3">
          Whole-tensor overview
        </h3>
        <p className="roi-precision" data-testid="overview-badges">
          {coverage ? (
            <span className={`pill ${coverage === "exact" ? "q-exact" : "q-sampled"}`} data-testid="coverage-badge">
              {coverage === "mixed" ? "MIXED: exact and ROW-SAMPLED tiles" : QUALITY_BADGE[coverage]}
            </span>
          ) : null}
          {pending ? (
            <span className="pill q-building" data-testid="building-badge">
              BUILDING exact overview
            </span>
          ) : null}
          <span className="pill">float32 visualization</span>
          <span className="pill mono">{tensor.dtype} source</span>
        </p>
      </div>

      <div className="row overview-controls" role="group" aria-label="Overview settings">
        <label className="field field-num">
          <span className="label-inline">Aggregation</span>
          <select className="input input-small" value={agg} onChange={(e) => setAgg(e.target.value as Aggregation)} data-testid="agg-select">
            {AGGREGATIONS.map((a) => (
              <option key={a} value={a}>
                {a} ({AGG_LABEL[a]})
              </option>
            ))}
          </select>
        </label>
        <label className="field field-num">
          <span className="label-inline">Level</span>
          <select
            className="input input-small"
            value={levelMode === "auto" ? "auto" : String(levelMode)}
            onChange={(e) => setLevelMode(e.target.value === "auto" ? "auto" : Number(e.target.value))}
            data-testid="level-select"
          >
            <option value="auto">Auto (L{autoLevel(vp, dims)})</option>
            {levelOptions.map((l) => (
              <option key={l} value={l}>
                L{l} (block {formatInt(2 ** l)})
              </option>
            ))}
          </select>
        </label>
        <label className="field field-num">
          <span className="label-inline">Scale</span>
          <select
            className="input input-small"
            value={mode}
            onChange={(e) => {
              const m = e.target.value as ScaleMode;
              setMode(m);
              if (m === "symmetric") setPalette("diverging");
            }}
          >
            {(Object.keys(MODE_LABEL) as ScaleMode[]).map((m) => (
              <option key={m} value={m}>
                {m === "auto" ? "Auto: min–max of the visible region" : m === "symmetric" ? "Sign-symmetric: ± max |value| visible" : "Fixed minimum / maximum"}
              </option>
            ))}
          </select>
        </label>
        <label className="field field-num">
          <span className="label-inline">Palette</span>
          <select className="input input-small" value={palette} onChange={(e) => setPalette(e.target.value as Palette)}>
            {(Object.keys(PALETTE_LABEL) as Palette[]).map((p) => (
              <option key={p} value={p}>
                {PALETTE_LABEL[p]}
              </option>
            ))}
          </select>
        </label>
        {mode === "fixed" ? (
          <>
            <label className="field field-num">
              <span className="label-inline">Minimum</span>
              <input className="input input-small" inputMode="decimal" value={fixed.min} onChange={(e) => setFixed({ ...fixed, min: e.target.value })} />
            </label>
            <label className="field field-num">
              <span className="label-inline">Maximum</span>
              <input className="input input-small" inputMode="decimal" value={fixed.max} onChange={(e) => setFixed({ ...fixed, max: e.target.value })} />
            </label>
            {fixedRangeError(fixed) ? (
              <span className="field-error" role="alert">
                {fixedRangeError(fixed)}; using the visible min–max meanwhile.
              </span>
            ) : null}
          </>
        ) : null}
        <label className="radio field-end">
          <input type="checkbox" checked={view.transpose} onChange={() => onViewChange({ ...view, transpose: !view.transpose })} />
          Transpose
        </label>
      </div>

      <div className="row" role="group" aria-label="Zoom and pan the overview">
        <button type="button" className="btn btn-small" onClick={() => zoomBy(1 / ZOOM_STEP)} aria-label="Zoom out">
          Zoom −
        </button>
        <button type="button" className="btn btn-small" onClick={() => zoomBy(ZOOM_STEP)} aria-label="Zoom in">
          Zoom +
        </button>
        <button type="button" className="btn btn-small" onClick={onFit}>
          Fit
        </button>
        <button type="button" className="btn btn-small" onClick={onReset}>
          Reset
        </button>
        <span className="help mono" data-testid="overview-visible">
          Visible {rectText(vr)} · {blockText(level)}
        </span>
      </div>

      <div className="heatmap overview-plot" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="heatmap-canvas"
          role="application"
          aria-roledescription="heatmap"
          aria-label={description}
          aria-describedby="overview-keys"
          tabIndex={0}
          data-testid="overview-canvas"
          data-level={level}
          data-ppe={vp.ppe}
          data-top={vp.top}
          data-left={vp.left}
          data-axis-left={AXIS_LEFT}
          data-axis-top={AXIS_TOP}
          data-cache-bytes={sharedTileCache.bytes}
          data-cache-entries={sharedTileCache.size}
          data-requests={sched.requests}
          data-peak-inflight={sched.peakConcurrent}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={() => (drag.current = null)}
          onPointerLeave={() => {
            if (!drag.current) setHoverInfo(null);
          }}
          onKeyDown={onKeyDown}
        />
        <p id="overview-keys" className="help">
          Drag to pan; Ctrl/⌘/Alt + wheel or the buttons to zoom (toward the pointer). Click a cell to select its block; Shift + drag selects a rectangle.
          Keyboard: arrows move the selected block, Shift + arrows pan, + / − zoom, 0 fits, Enter inspects exact values, Escape clears.
        </p>
        <p className="heatmap-hover mono" aria-hidden="true" data-testid="overview-hover">
          {hoverInfo ? hoverInfo.text : " "}
        </p>
      </div>
      <p className="visually-hidden" role="status" aria-live="polite" data-testid="overview-announce">
        {announce}
      </p>

      {tooMany ? (
        <p className="notice notice-info" role="status">
          Level L{level} would need {formatInt(range ? rangeCount(range) : 0)} tiles for this screen (limit {MAX_VISIBLE_TILES}). Zoom in, or choose a coarser level or Auto.
        </p>
      ) : null}
      {!tooMany && !allLoaded && !pending && drawn.errors.length === 0 ? (
        <p className="muted" role="status" data-testid="overview-loading">
          Loading overview… {drawn.cur.length} of {drawn.cur.length + drawn.missing} tiles.
        </p>
      ) : null}

      {job ? (
        <JobPanel
          job={job}
          error={jobError}
          pollStopped={pollFailures >= MAX_POLL_FAILURES}
          retrying={pollFailures > 0 && pollFailures < MAX_POLL_FAILURES}
          canceling={canceling}
          onCancel={onCancelJob}
          onRetry={onRetryBuild}
          onCheckAgain={onCheckAgain}
        />
      ) : null}

      {transposeSampleError ? (
        <div className="notice notice-warn" role="alert" data-testid="sampling-unsupported">
          <p className="notice-title">No row-sampled preview for this view</p>
          <p>{(transposeSampleError as ApiRequestError).message}.</p>
          <div className="row">
            <button type="button" className="btn btn-small" onClick={() => setConfirmExact(true)}>
              Build the exact overview instead…
            </button>
            {view.transpose ? (
              <button type="button" className="btn btn-small" onClick={() => onViewChange({ ...view, transpose: false })}>
                Show the untransposed view
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
      {otherError ? <ErrorPanel error={otherError} onRetry={onRetryBuild} /> : null}

      {(coverage === "row_sampled" || coverage === "mixed" || transposeSampleError) && quality.mode === "auto" && !pending ? (
        <div className="notice notice-info" data-testid="sampling-panel">
          <p className="notice-title">This is a row-sampled preview, not exact</p>
          <p>
            The tensor is {formatBytes(tensor.byte_length)}, above pyramid_exact_max_mib ({limitMib ?? "?"} MiB), so Auto shows a labelled preview instead of reading
            everything. {meta.data ? qualityText("row_sampled", meta.data.sampled_rows ?? null, meta.data.sampling_policy, meta.data.seed) : sampledTile ? qualityText("row_sampled", sampledTile.sampledRows, null, null) : ""}
          </p>
          <form className="row" onSubmit={onSeed} aria-label="Sampling seed">
            <label className="field field-num">
              <span className="label-inline">Seed (empty: strided rows)</span>
              <input
                className="input input-small"
                inputMode="numeric"
                value={seedText}
                aria-invalid={seedInvalid || undefined}
                onChange={(e) => setSeedText(e.target.value)}
                data-testid="seed-input"
              />
            </label>
            <button type="submit" className="btn btn-small field-end" disabled={seedInvalid}>
              Apply seed
            </button>
            {seedInvalid ? <span className="field-error">Enter a whole number from 0 to 4,294,967,295.</span> : null}
          </form>
          {confirmExact ? null : (
            <button type="button" className="btn btn-small" onClick={() => setConfirmExact(true)} data-testid="request-exact">
              Build the exact overview…
            </button>
          )}
        </div>
      ) : null}
      {confirmExact && !pending && quality.mode === "auto" ? (
        <div className="notice notice-warn" role="group" aria-label="Confirm exact overview build" data-testid="confirm-exact">
          <p className="notice-title">Build the exact overview?</p>
          <p>
            Estimated read: <strong>{formatBytes(tensor.byte_length)}</strong> (the whole tensor, once) to build an exact pyramid. It runs as a background job you can cancel;
            the sampled preview stays until it completes.
          </p>
          <div className="row">
            <button
              type="button"
              className="btn btn-small btn-primary"
              onClick={() => {
                setConfirmExact(false);
                setQuality({ ...quality, mode: "exact" });
              }}
              data-testid="confirm-exact-build"
            >
              Start exact build
            </button>
            <button type="button" className="btn btn-small" onClick={() => setConfirmExact(false)}>
              Not now
            </button>
          </div>
        </div>
      ) : null}
      {quality.mode === "exact" && aboveLimit && !pending ? (
        <p className="help">
          Exact mode requested.{" "}
          <button type="button" className="btn btn-small" onClick={() => setQuality({ ...quality, mode: "auto" })}>
            Back to the sampled preview
          </button>
        </p>
      ) : null}

      <div className="overview-facts" data-testid="overview-facts">
        <dl className="facts facts-grid">
          <div>
            <dt>Coverage</dt>
            <dd data-testid="coverage-text">
              {coverage === "exact"
                ? qualityText("exact", null, null, null)
                : coverage === "row_sampled"
                  ? meta.data
                    ? qualityText("row_sampled", meta.data.sampled_rows ?? null, meta.data.sampling_policy, meta.data.seed)
                    : qualityText("row_sampled", sampledTile?.sampledRows ?? null, null, null)
                  : coverage === "mixed"
                    ? "Some visible tiles are exact, others are row-sampled previews (dashed outline)."
                    : pending
                      ? "Exact overview is being built; nothing is shown as exact until it completes."
                      : "—"}
            </dd>
          </div>
          <div>
            <dt>Value precision</dt>
            <dd>{PRECISION_TEXT}</dd>
          </div>
          <div>
            <dt>Aggregation</dt>
            <dd>{AGG_MEANING[agg]}</dd>
          </div>
          <div>
            <dt>Level and block</dt>
            <dd className="mono">{blockText(level)}</dd>
          </div>
          <div>
            <dt>Source dtype</dt>
            <dd className="mono">{tensor.dtype}</dd>
          </div>
        </dl>
        {meta.data?.warnings && meta.data.warnings.length > 0 ? (
          <div className="notice notice-warn">
            <p className="notice-title">Server warnings for this tile</p>
            <ul>
              {meta.data.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {meta.error ? <ErrorPanel error={meta.error} /> : null}
      </div>

      {drawn.cur.length > 0 || drawn.under.length > 0 ? <OverviewLegend scale={scale} palette={palette} aggregation={agg} counts={counts} /> : null}

      <div className="selected-cell" data-testid="overview-selection" aria-live="polite">
        {selection ? (
          <>
            <p className="label">Selected {selCells > 1 && rectCells(selection) > block(level) ** 2 ? "region" : "block"}</p>
            <p className="mono" data-testid="selection-bounds">
              rows [{selection.r0}, {selection.r1}) · columns [{selection.c0}, {selection.c1})
            </p>
            <p className="help">
              {formatInt(selCells)} elements · {selInfo}
            </p>
            {selectedTooBig ? (
              <div className="notice notice-warn" role="alert" data-testid="roi-too-big">
                <p>
                  This region has {formatInt(selCells)} elements; Exact ROI reads at most {formatInt(cap ?? 0)} (roi_max_cells). It is not truncated silently: choose a
                  smaller region.
                </p>
                <div className="row">
                  <button type="button" className="btn btn-small" onClick={() => setViewport(zoomToRect(dims, size, selection), true)}>
                    Zoom the overview to this region
                  </button>
                  <button type="button" className="btn btn-small" onClick={() => select(shrinkToCap(selection))} data-testid="shrink-selection">
                    Select its top-left {shrinkToCap(selection).r1 - selection.r0} × {shrinkToCap(selection).c1 - selection.c0} window
                  </button>
                </div>
              </div>
            ) : null}
            <div className="row">
              <button type="button" className="btn btn-small btn-primary" disabled={selectedTooBig || cap === null} onClick={() => inspect(selection)} data-testid="inspect-exact">
                Inspect exact values
              </button>
              <button type="button" className="btn btn-small" onClick={() => setSelection(null)}>
                Clear selection
              </button>
            </div>
          </>
        ) : (
          <p className="muted">Select a block (click or keyboard) or a rectangle (Shift + drag) to see its raw bounds and open its exact values.</p>
        )}
      </div>
    </section>
  );
}

interface JobPanelProps {
  job: Job;
  error: unknown;
  pollStopped: boolean;
  retrying: boolean;
  canceling: boolean;
  onCancel: () => void;
  onRetry: () => void;
  onCheckAgain: () => void;
}

function JobPanel({ job, error, pollStopped, retrying, canceling, onCancel, onRetry, onCheckAgain }: JobPanelProps) {
  const pct = Math.floor(job.progress * 100);
  const io =
    job.estimated_bytes !== undefined ? (
      <span>
        {" "}
        · read {formatBytes(job.bytes_scanned ?? 0)} of ~{formatBytes(job.estimated_bytes)}
      </span>
    ) : null;
  if (isPending(job)) {
    return (
      <div className="notice notice-info job-panel" data-testid="job-panel" data-state={job.state}>
        <p className="notice-title">Building the exact overview ({job.state})</p>
        <div className="row">
          <progress max={100} value={pct} aria-label="Exact overview build progress" data-testid="job-progress" />
          <span className="mono" role="status" aria-live="polite">
            {pct}%{io}
          </span>
        </div>
        <p className="help">
          Nothing is shown as exact until the build completes{job.deduplicated ? "; this build is shared with another request for the same pyramid" : ""}. Leaving this view
          keeps it running.
        </p>
        <button type="button" className="btn btn-small btn-danger-outline" onClick={onCancel} disabled={canceling} data-testid="job-cancel">
          {canceling ? "Canceling…" : "Cancel build"}
        </button>
        {error ? (
          <div data-testid="job-poll-error">
            <p className="help" role="status">
              {pollStopped
                ? "The build's status could not be read; WeightLab stopped checking. The build itself keeps running on the server."
                : retrying
                  ? "The build's status could not be read; checking again shortly."
                  : null}
            </p>
            <ErrorPanel error={error} />
            <button type="button" className="btn btn-small" onClick={onCheckAgain} data-testid="job-check-again">
              Check status again
            </button>
          </div>
        ) : null}
      </div>
    );
  }
  if (job.state === "completed") return null;
  const text =
    job.state === "canceled"
      ? "The exact overview build was canceled. Nothing was published; no tile is shown as exact."
      : job.state === "interrupted"
        ? "The build did not complete: the WeightLab server stopped or restarted while it ran. Nothing was published."
        : `The build failed${job.error ? ` (${job.error.code}): ${job.error.message}` : "."}`;
  return (
    <div className={`notice ${job.state === "failed" ? "notice-error" : "notice-warn"}`} role="alert" data-testid="job-panel" data-state={job.state}>
      <p className="notice-title">Exact overview {job.state}</p>
      <p>{text}</p>
      {job.error?.code === "RESOURCE_LIMIT" ? (
        <p className="notice-action">Raise memory_budget_mib or tile_cache_disk_mib in Settings, or inspect smaller exact regions instead.</p>
      ) : job.error?.code === "FILE_CHANGED" ? (
        <p className="notice-action">The model's files changed: re-import the model; relink it only if it was moved unchanged.</p>
      ) : null}
      <button type="button" className="btn btn-small" onClick={onRetry} data-testid="job-retry">
        Build again
      </button>
    </div>
  );
}
