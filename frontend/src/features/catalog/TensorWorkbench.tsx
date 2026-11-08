// Overview and Exact ROI of one tensor, on one shared view (axes, slice, transpose): a block selected in the
// overview opens Exact ROI on exactly its raw bounds, read with POST /tensors/roi by the existing explorer (its
// decoder and precision rules are the only way exact values are ever shown). Rank 0/1 and empty tensors have no
// 2D overview and keep the exact representation only.
import { useState, type KeyboardEvent } from "react";
import { api } from "../../lib/api/endpoints";
import type { Aggregation, Tensor } from "../../lib/api/types";
import { formatShape } from "../../lib/format";
import { useResource } from "../../lib/useResource";
import { Overview } from "../overview/Overview";
import type { RawRect } from "../overview/geometry";
import { RoiExplorer } from "../roi/RoiExplorer";
import { defaultView, viewDims, viewError, type ViewState, type Win } from "../roi/view";

type Tab = "overview" | "roi";

export function TensorWorkbench({ modelId, tensor }: { modelId: string; tensor: Tensor }) {
  const shape = tensor.shape;
  const empty = tensor.numel === 0 || shape.some((d) => d === 0);
  const canOverview = shape.length >= 2 && !empty;
  const [tab, setTab] = useState<Tab>(canOverview ? "overview" : "roi");
  const [view, setView] = useState<ViewState>(() => defaultView(shape));
  const [aggregation, setAggregation] = useState<Aggregation>("mean_abs");
  const [focus, setFocus] = useState<{ win: Win; nonce: number; from: RawRect } | null>(null);
  const settings = useResource("settings", (signal) => api.settings(signal));
  const vErr = viewError(shape, view);
  const dims = vErr ? null : viewDims(shape, view);

  if (!canOverview) return <RoiExplorer modelId={modelId} tensor={tensor} />;

  function inspect(r: RawRect) {
    setFocus((f) => ({ win: { r0: r.r0, c0: r.c0, h: r.r1 - r.r0, w: r.c1 - r.c0 }, nonce: (f?.nonce ?? 0) + 1, from: r }));
    setTab("roi");
    requestAnimationFrame(() => document.getElementById("wb-tab-roi")?.focus());
  }

  const tabs: [Tab, string][] = [
    ["overview", "Overview"],
    ["roi", "Exact ROI"],
  ];
  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const next: Tab = tab === "overview" ? "roi" : "overview";
    setTab(next);
    document.getElementById(`wb-tab-${next}`)?.focus();
  }
  return (
    <div className="workbench stack-sm">
      <div role="tablist" aria-label="Tensor views" className="tabs" onKeyDown={onKeyDown}>
        {tabs.map(([t, label]) => (
          <button
            key={t}
            type="button"
            role="tab"
            id={`wb-tab-${t}`}
            aria-selected={tab === t}
            aria-controls={tab === t ? `wb-panel-${t}` : undefined}
            tabIndex={tab === t ? 0 : -1}
            className="tab"
            onClick={() => setTab(t)}
          >
            {label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`wb-panel-${tab}`} aria-labelledby={`wb-tab-${tab}`}>
        {tab === "overview" ? (
          <>
            {shape.length > 2 ? <SliceControls shape={shape} view={view} onView={setView} /> : null}
            {vErr || !dims ? (
              <p className="notice notice-warn" role="alert" data-testid="overview-view-error">
                <span>
                  <strong>Unsupported view:</strong> {vErr}. Nothing is read until the view is valid.
                </span>
              </p>
            ) : (
              <Overview
                key={JSON.stringify(view)}
                modelId={modelId}
                tensor={tensor}
                view={view}
                dims={dims}
                settings={settings.data}
                aggregation={aggregation}
                onAggregation={setAggregation}
                onViewChange={setView}
                onInspect={inspect}
              />
            )}
          </>
        ) : (
          <>
            {focus ? (
              <p className="notice notice-info" data-testid="roi-from-overview">
                Opened from the overview selection: rows [{focus.from.r0}, {focus.from.r1}) × columns [{focus.from.c0}, {focus.from.c1}). These are exact values read from the
                checkpoint, not the overview's aggregates.
              </p>
            ) : null}
            <RoiExplorer
              key={focus ? `focus-${focus.nonce}` : "roi"}
              modelId={modelId}
              tensor={tensor}
              initialView={view}
              initialWin={focus?.win}
              onViewChange={setView}
            />
          </>
        )}
      </div>
    </div>
  );
}

/** Axes and slice for rank > 2 views (the same rules as the exact explorer's controls). */
function SliceControls({ shape, view, onView }: { shape: number[]; view: ViewState; onView: (v: ViewState) => void }) {
  const rank = shape.length;
  function onAxis(which: "rowAxis" | "colAxis", axis: number) {
    const other = which === "rowAxis" ? "colAxis" : "rowAxis";
    const prev = view[which];
    const next: ViewState = { ...view, [which]: axis };
    if (view[other] === axis) next[other] = prev;
    const fixed: Record<number, string> = {};
    for (let a = 0; a < rank; a++) if (a !== next.rowAxis && a !== next.colAxis) fixed[a] = view.fixed[a] ?? "0";
    onView({ ...next, fixed });
  }
  return (
    <fieldset className="roi-slice" data-testid="overview-slice">
      <legend className="label">View and slice (shape {formatShape(shape)})</legend>
      <div className="row">
        {(["rowAxis", "colAxis"] as const).map((w) => (
          <label key={w} className="field field-num">
            <span className="label-inline">{w === "rowAxis" ? "Row axis" : "Column axis"}</span>
            <select className="input input-small" value={view[w] ?? 0} onChange={(e) => onAxis(w, Number(e.target.value))}>
              {shape.map((d, a) => (
                <option key={a} value={a}>
                  {a} (size {d})
                </option>
              ))}
            </select>
          </label>
        ))}
        {Object.keys(view.fixed).map((k) => {
          const a = Number(k);
          return (
            <label key={a} className="field field-num">
              <span className="label-inline">
                Axis {a} index (0–{(shape[a] ?? 0) - 1})
              </span>
              <input
                className="input input-small"
                inputMode="numeric"
                value={view.fixed[a] ?? ""}
                onChange={(e) => onView({ ...view, fixed: { ...view.fixed, [a]: e.target.value } })}
              />
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
