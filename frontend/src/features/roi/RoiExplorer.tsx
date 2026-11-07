// Phase 1 Task 4A: the exact-window explorer. It shows one bounded region of one 2D view of a tensor, read exactly
// with POST /tensors/roi; it is never a whole-tensor overview (tiles/pyramids are a later milestone).
//
// Requests: at most one in flight. Every pan/zoom/slice change aborts the pending request immediately and the next
// one is sent after a short quiet period, so rapid panning sends one request for where the user stops. A response is
// drawn only if it validates against the request that is current *now*; anything else is discarded. There is no
// client-side value cache: every window shown is a fresh read checked by the server against the imported files, so
// a changed checkpoint can never be masked by stale values (FILE_CHANGED is shown instead).
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { RoiRequest, Tensor } from "../../lib/api/types";
import { formatInt, formatShape } from "../../lib/format";
import { useDebounced, useResource } from "../../lib/useResource";
import { HeatmapCanvas, type CellRef } from "./HeatmapCanvas";
import { Legend } from "./Legend";
import { computeScale, fixedRangeError, MODE_LABEL, PALETTE_LABEL, type FixedRange, type Palette, type ScaleMode } from "./scale";
import { SelectedCell } from "./SelectedCell";
import { parseRoi, type RoiGrid } from "./values";
import { ValuesTable } from "./ValuesTable";
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
  type ViewState,
  type Win,
} from "./view";

export const ROI_QUIET_MS = 120;

interface Props {
  modelId: string;
  tensor: Tensor;
}

type Tab = "heatmap" | "values";

export function RoiExplorer({ modelId, tensor }: Props) {
  const settings = useResource("settings", (signal) => api.settings(signal));
  const capRaw = settings.data?.roi_max_cells;
  const cap = typeof capRaw === "number" && Number.isSafeInteger(capRaw) && capRaw > 0 ? capRaw : null;

  const shape = tensor.shape;
  const rank = shape.length;
  const [view, setView] = useState<ViewState>(() => defaultView(shape));
  const [win, setWin] = useState<Win | null>(null);
  const [selected, setSelected] = useState<CellRef | null>(rank === 0 ? { row: 0, col: 0 } : null);
  const [tab, setTab] = useState<Tab>("heatmap");
  const [mode, setMode] = useState<ScaleMode>("auto");
  const [palette, setPalette] = useState<Palette>("sequential");
  const [fixed, setFixed] = useState<FixedRange>({ min: "-1", max: "1" });
  const [jump, setJump] = useState({ row: "", col: "", error: "" });

  const empty = tensor.numel === 0 || shape.some((d) => d === 0);
  const vErr = empty ? null : viewError(shape, view);
  const dims = !empty && !vErr ? viewDims(shape, view) : null;
  const eff = dims && cap ? clampWin(win ?? defaultWin(dims, cap), dims, cap) : null;
  const bounds = dims && eff ? toBounds(eff, dims) : null;
  const bErr = bounds && dims && cap ? boundsError(bounds, dims, cap) : null;

  const req: RoiRequest | null =
    bounds && !bErr
      ? {
          source: { kind: "tensor", model_id: modelId, tensor_id: tensor.id },
          view: toRequestView(view),
          roi: bounds,
          value_encoding: "auto",
        }
      : null;
  const key = req ? JSON.stringify(req) : null;
  const quiet = useDebounced(key, ROI_QUIET_MS);
  // While the user is still moving, nothing is in flight: the previous request was aborted on the change.
  const fetchKey = key !== null && key === quiet ? key : null;
  // Every read the explorer issues gets a new generation, also when the user returns to a region read before
  // (A -> B -> A is two reads of A). The resource key includes the generation, so a completed result is shown
  // only for the read that produced it: leaving a region and coming back never redisplays its earlier values,
  // which may come from files that have changed since; a fresh successful read is required (or FILE_CHANGED).
  const [issued, setIssued] = useState<{ key: string | null; gen: number }>({ key: null, gen: 0 });
  const gen = fetchKey !== issued.key ? issued.gen + 1 : issued.gen;
  if (fetchKey !== issued.key) setIssued({ key: fetchKey, gen }); // derived from the previous render, not an effect
  const readKey = fetchKey === null ? null : `${gen}:${fetchKey}`;
  const res = useResource<{ readKey: string; grid: RoiGrid }>(readKey, async (signal) => {
    const r = JSON.parse(fetchKey ?? "null") as RoiRequest;
    const body = await api.roi(r, signal);
    return { readKey: readKey ?? "", grid: parseRoi(body, r, tensor.dtype) };
  });
  // Only the settled answer of the current read is ever shown: not while it (or a retry) is loading, not next to
  // an error, and never a result of an earlier read of the same region.
  const grid = readKey !== null && !res.loading && !res.error && res.data?.readKey === readKey ? res.data.grid : null;
  const error = readKey !== null ? res.error : undefined;

  const scale = grid ? computeScale(grid, mode, fixed) : null;

  // ---- navigation (all in event handlers; the window is derived from state on every render)
  function select(c: CellRef) {
    setSelected(c);
  }
  function setWindow(next: Win, sel: CellRef | null = selected) {
    setWin(next);
    if (sel && dims) {
      // The selection follows the window.
      const row = Math.min(Math.max(sel.row, next.r0), Math.min(dims.rows, next.r0 + next.h) - 1);
      const col = Math.min(Math.max(sel.col, next.c0), Math.min(dims.cols, next.c0 + next.w) - 1);
      setSelected({ row, col });
    }
  }
  function onPan(dr: number, dc: number) {
    if (eff && dims && cap) setWindow(pan(eff, dims, cap, dr, dc));
  }
  function onZoom(dir: "in" | "out") {
    if (!eff || !dims || !cap) return;
    const next = zoom(eff, dims, cap, dir, selected ?? undefined);
    if (next) setWindow(next);
  }
  function onMove(dr: number, dc: number) {
    if (!eff || !dims || !cap) return;
    const from = selected ?? { row: eff.r0, col: eff.c0 };
    const row = Math.min(dims.rows - 1, Math.max(0, from.row + (selected ? dr : 0)));
    const col = Math.min(dims.cols - 1, Math.max(0, from.col + (selected ? dc : 0)));
    setSelected({ row, col });
    setWin(ensureVisible(eff, dims, cap, row, col));
  }
  function onReset() {
    if (!dims || !cap) return;
    setWin(defaultWin(dims, cap));
    setSelected(null);
  }
  function onJump(e: FormEvent) {
    e.preventDefault();
    if (!eff || !dims || !cap) return;
    const parse = (t: string, n: number, what: string): number | string => {
      const s = t.trim() === "" ? "0" : t.trim();
      if (!/^(0|[1-9][0-9]*)$/.test(s)) return `${what}: enter a whole number`;
      const v = Number(s);
      return v < n ? v : `${what} ${s} is outside 0–${n - 1}`;
    };
    const r = rank < 2 ? 0 : parse(jump.row, dims.rows, "Row");
    const c = parse(jump.col, dims.cols, "Column");
    if (typeof r === "string" || typeof c === "string") {
      setJump({ ...jump, error: typeof r === "string" ? r : String(c) });
      return;
    }
    setJump({ ...jump, error: "" });
    setWin(centreOn(eff, dims, cap, r, c));
    setSelected({ row: r, col: c });
  }
  function onTranspose() {
    setView({ ...view, transpose: !view.transpose });
    if (eff) setWin({ r0: eff.c0, c0: eff.r0, h: eff.w, w: eff.h });
    if (selected) setSelected({ row: selected.col, col: selected.row });
  }
  function onAxis(which: "rowAxis" | "colAxis", axis: number) {
    const other = which === "rowAxis" ? "colAxis" : "rowAxis";
    const prev = view[which];
    const next: ViewState = { ...view, [which]: axis };
    if (view[other] === axis) next[other] = prev; // choosing the other's axis swaps them
    const nextFixed: Record<number, string> = {};
    for (let a = 0; a < rank; a++) {
      if (a === next.rowAxis || a === next.colAxis) continue;
      nextFixed[a] = view.fixed[a] ?? "0";
    }
    next.fixed = nextFixed;
    setView(next);
    setWin(null);
    setSelected(null);
  }
  function onFixed(axis: number, text: string) {
    setView({ ...view, fixed: { ...view.fixed, [axis]: text } });
  }

  // ---- render
  const sliceText = Object.keys(view.fixed).length
    ? Object.entries(view.fixed)
        .map(([a, i]) => `axis ${a} = ${i}`)
        .join(", ")
    : null;

  if (empty) {
    return (
      <section className="roi" aria-labelledby="roi-h">
        <h3 id="roi-h" className="h3">
          Exact values
        </h3>
        <p className="notice notice-info" data-testid="roi-empty">
          This tensor has no elements (shape {formatShape(shape)}), so there are no values to show.
        </p>
      </section>
    );
  }
  if (settings.error) {
    return (
      <section className="roi" aria-labelledby="roi-h">
        <h3 id="roi-h" className="h3">
          Exact values
        </h3>
        <ErrorPanel error={settings.error} onRetry={settings.reload} />
      </section>
    );
  }

  const heading =
    rank === 0 ? (
      <p className="roi-bounds" data-testid="roi-bounds">
        Scalar (rank 0): the single exact value
      </p>
    ) : bounds && dims ? (
      <div className="roi-banner" data-testid="roi-banner">
        <p className="roi-bounds" data-testid="roi-bounds">
          {rank >= 2 ? (
            <>
              <span>
                Rows <strong className="mono">[{bounds.r0}, {bounds.r1})</strong> of {formatInt(dims.rows)}
              </span>
              <span>
                Columns <strong className="mono">[{bounds.c0}, {bounds.c1})</strong> of {formatInt(dims.cols)}
              </span>
            </>
          ) : (
            <span>
              Indices <strong className="mono">[{bounds.c0}, {bounds.c1})</strong> of {formatInt(dims.cols)} (rank 1, shown as one row)
            </span>
          )}
        </p>
        <p className="help" data-testid="roi-coverage-note">
          {(bounds.r1 - bounds.r0) * (bounds.c1 - bounds.c0) === dims.rows * dims.cols
            ? `Exact region covering the whole ${rank > 2 ? "slice" : "tensor"} (${formatInt(dims.rows * dims.cols)} cells).`
            : `Exact region of ${formatInt((bounds.r1 - bounds.r0) * (bounds.c1 - bounds.c0))} of ${formatInt(dims.rows * dims.cols)} cells (${(
                (100 * (bounds.r1 - bounds.r0) * (bounds.c1 - bounds.c0)) /
                (dims.rows * dims.cols)
              ).toPrecision(2)}%): this is a window, not the whole ${rank > 2 ? "slice" : "tensor"}.`}
          {sliceText ? ` Slice: ${sliceText}.` : ""}
        </p>
      </div>
    ) : null;

  const zoomIn = eff && dims && cap ? zoom(eff, dims, cap, "in") : null;
  const zoomOut = eff && dims && cap ? zoom(eff, dims, cap, "out") : null;
  const budgetBlocksZoomOut = !!(eff && dims && cap && !zoomOut && (eff.h < dims.rows || eff.w < dims.cols));
  const stepR = eff ? Math.max(1, Math.floor(eff.h / 2)) : 1;
  const stepC = eff ? Math.max(1, Math.floor(eff.w / 2)) : 1;
  const canUp = !!eff && eff.r0 > 0;
  const canDown = !!(eff && dims && eff.r0 + eff.h < dims.rows);
  const canLeft = !!eff && eff.c0 > 0;
  const canRight = !!(eff && dims && eff.c0 + eff.w < dims.cols);

  const status = !cap ? (
    <p className="muted" role="status">
      Loading settings…
    </p>
  ) : vErr ? (
    <p className="notice notice-warn" role="alert" data-testid="roi-view-error">
      <span>
        <strong>Unsupported view:</strong> {vErr}. Nothing is read until the view is valid.
      </span>
    </p>
  ) : bErr ? (
    <p className="notice notice-warn" role="alert">
      {bErr}
    </p>
  ) : error ? (
    <ErrorPanel error={error} context="roi" onRetry={res.reload} />
  ) : !grid ? (
    <p className="roi-loading muted" role="status" data-testid="roi-loading">
      Loading exact values…
    </p>
  ) : null;

  const tensorIdx = selected && rank > 0 ? tensorIndex(shape, view, selected.row, selected.col) : null;

  return (
    <section className="roi" aria-labelledby="roi-h" aria-busy={!grid && !error && !vErr}>
      <div className="roi-head">
        <h3 id="roi-h" className="h3">
          Exact values {rank > 0 ? "(exact region)" : ""}
        </h3>
        {grid ? (
          <p className="roi-precision" data-testid="roi-precision">
            <span className="pill">coverage: {grid.coverage}</span>
            <span className={`pill ${grid.precision === "lossless" ? "prec-lossless" : "prec-rounded"}`}>numeric precision: {grid.precision}</span>
            <span className="pill mono">{grid.dtype} · {grid.encoding}</span>
          </p>
        ) : null}
      </div>
      {heading}

      {rank > 2 ? (
        <fieldset className="roi-slice" data-testid="slice-controls">
          <legend className="label">View and slice (shape {formatShape(shape)})</legend>
          <div className="row">
            <label className="field field-num">
              <span className="label-inline">Row axis</span>
              <select className="input input-small" value={view.rowAxis ?? 0} onChange={(e) => onAxis("rowAxis", Number(e.target.value))}>
                {shape.map((d, a) => (
                  <option key={a} value={a}>
                    {a} (size {d})
                  </option>
                ))}
              </select>
            </label>
            <label className="field field-num">
              <span className="label-inline">Column axis</span>
              <select className="input input-small" value={view.colAxis ?? 1} onChange={(e) => onAxis("colAxis", Number(e.target.value))}>
                {shape.map((d, a) => (
                  <option key={a} value={a}>
                    {a} (size {d})
                  </option>
                ))}
              </select>
            </label>
            {Object.keys(view.fixed).map((k) => {
              const a = Number(k);
              return (
                <label key={a} className="field field-num">
                  <span className="label-inline">Axis {a} index (0–{(shape[a] ?? 0) - 1})</span>
                  <input
                    className="input input-small"
                    inputMode="numeric"
                    value={view.fixed[a] ?? ""}
                    aria-invalid={vErr?.startsWith(`axis ${a}`) ? true : undefined}
                    onChange={(e) => onFixed(a, e.target.value)}
                    data-testid={`fixed-${a}`}
                  />
                </label>
              );
            })}
          </div>
        </fieldset>
      ) : null}

      {rank > 0 ? (
        <div className="roi-toolbar" role="group" aria-label="Navigate the exact window">
          <div className="row">
            <button type="button" className="btn btn-small" disabled={!canLeft} onClick={() => onPan(0, -stepC)} aria-label={`Pan left by ${stepC} columns`}>
              ←
            </button>
            <button type="button" className="btn btn-small" disabled={!canRight} onClick={() => onPan(0, stepC)} aria-label={`Pan right by ${stepC} columns`}>
              →
            </button>
            {rank >= 2 ? (
              <>
                <button type="button" className="btn btn-small" disabled={!canUp} onClick={() => onPan(-stepR, 0)} aria-label={`Pan up by ${stepR} rows`}>
                  ↑
                </button>
                <button type="button" className="btn btn-small" disabled={!canDown} onClick={() => onPan(stepR, 0)} aria-label={`Pan down by ${stepR} rows`}>
                  ↓
                </button>
              </>
            ) : null}
            <button type="button" className="btn btn-small" disabled={!zoomIn} onClick={() => onZoom("in")} aria-label={zoomIn ? `Zoom in to ${zoomIn.h} × ${zoomIn.w} cells` : "Zoom in"}>
              Zoom +
            </button>
            <button type="button" className="btn btn-small" disabled={!zoomOut} onClick={() => onZoom("out")} aria-label={zoomOut ? `Zoom out to ${zoomOut.h} × ${zoomOut.w} cells` : "Zoom out"}>
              Zoom −
            </button>
            <button type="button" className="btn btn-small" onClick={onReset} disabled={!dims}>
              Reset to origin
            </button>
            {rank >= 2 ? (
              <label className="radio">
                <input type="checkbox" checked={view.transpose} onChange={onTranspose} />
                Transpose
              </label>
            ) : null}
          </div>
          {budgetBlocksZoomOut ? (
            <p className="help" data-testid="budget-note">
              A larger window would exceed the exact-read budget of {formatInt(cap ?? 0)} cells (the server's roi_max_cells setting,
              1–65,536).
            </p>
          ) : null}
          <form className="row" onSubmit={onJump} aria-label="Jump to a cell">
            {rank >= 2 ? (
              <label className="field field-num">
                <span className="label-inline">Row</span>
                <input className="input input-small" inputMode="numeric" value={jump.row} onChange={(e) => setJump({ ...jump, row: e.target.value })} />
              </label>
            ) : null}
            <label className="field field-num">
              <span className="label-inline">{rank >= 2 ? "Column" : "Index"}</span>
              <input className="input input-small" inputMode="numeric" value={jump.col} onChange={(e) => setJump({ ...jump, col: e.target.value })} />
            </label>
            <button type="submit" className="btn btn-small field-end">
              Go to cell
            </button>
            {jump.error ? (
              <span className="field-error" role="alert">
                {jump.error}
              </span>
            ) : null}
          </form>
        </div>
      ) : null}

      {status}

      {grid && scale ? (
        rank === 0 ? (
          <SelectedCell grid={grid} selected={selected} tensorIndex={null} rank={0} />
        ) : (
          <>
            <Tabs tab={tab} onTab={setTab} />
            <div role="tabpanel" id={`roi-panel-${tab}`} aria-labelledby={`roi-tab-${tab}`} className="roi-panel">
              {tab === "heatmap" ? (
                <div className="stack-sm">
                  <div className="row roi-scale" role="group" aria-label="Colour scale">
                    <label className="field field-num">
                      <span className="label-inline">Scale</span>
                      <select
                        className="input input-small"
                        value={mode}
                        onChange={(e) => {
                          const m = e.target.value as ScaleMode;
                          setMode(m);
                          setPalette(m === "symmetric" ? "diverging" : palette);
                        }}
                      >
                        {(Object.keys(MODE_LABEL) as ScaleMode[]).map((m) => (
                          <option key={m} value={m}>
                            {MODE_LABEL[m]}
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
                            {fixedRangeError(fixed)}; using the region's min–max meanwhile.
                          </span>
                        ) : null}
                      </>
                    ) : null}
                  </div>
                  <HeatmapCanvas
                    grid={grid}
                    scale={scale}
                    palette={palette}
                    selected={selected}
                    label={`Heatmap of exact values: rows ${grid.bounds.r0} to ${grid.bounds.r1 - 1}, columns ${grid.bounds.c0} to ${grid.bounds.c1 - 1}. The Values tab lists the same cells as a table.`}
                    onSelect={select}
                    onMove={onMove}
                    onZoom={onZoom}
                  />
                  <Legend grid={grid} scale={scale} palette={palette} />
                </div>
              ) : (
                <ValuesTable grid={grid} selected={selected} onSelect={select} />
              )}
            </div>
            <SelectedCell grid={grid} selected={selected} tensorIndex={tensorIdx} rank={rank} />
          </>
        )
      ) : null}
      {grid && grid.warnings.length > 0 ? (
        <div className="notice notice-warn">
          <p className="notice-title">Server warnings</p>
          <ul>
            {grid.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function Tabs({ tab, onTab }: { tab: Tab; onTab: (t: Tab) => void }) {
  const tabs: [Tab, string][] = [
    ["heatmap", "Heatmap"],
    ["values", "Values"],
  ];
  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    const i = tabs.findIndex(([t]) => t === tab);
    const n = e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    const next = tabs[n]?.[0] ?? "heatmap";
    onTab(next);
    document.getElementById(`roi-tab-${next}`)?.focus();
  }
  return (
    <div role="tablist" aria-label="Exact region views" className="tabs" onKeyDown={onKeyDown}>
      {tabs.map(([t, label]) => (
        <button
          key={t}
          type="button"
          role="tab"
          id={`roi-tab-${t}`}
          aria-selected={tab === t}
          aria-controls={tab === t ? `roi-panel-${t}` : undefined}
          tabIndex={tab === t ? 0 : -1}
          className="tab"
          onClick={() => onTab(t)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
