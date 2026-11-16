// The delta heatmap of one compared tensor (ROADMAP P1.1): POST /tiles and /tensors/roi with a `delta` source, so
// the server computes B − A (or |B − A|, or |B − A| / max(|A|, ε)) elementwise in a paired pass over both
// checkpoints. It reuses the whole-tensor overview (tile scheduling, pyramid jobs, coverage labels, legend) and the
// exact explorer (validation, exact text, non-finite and overflow masks) on one shared view; a block selected in the
// heatmap opens the exact paired values of exactly that region.
//
// Nothing is read until the section is opened. Signed B − A starts on a symmetric diverging scale centred on zero
// (FR-015), so the sign is never a colour accident; magnitudes use a sequential scale. Tile values are float32
// visualizations of block aggregates; the exact values are the Exact ROI tab's.
import { useState, type KeyboardEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import type { TensorRow } from "../../lib/api/compare";
import { api } from "../../lib/api/endpoints";
import type { Aggregation, DeltaSource, DeltaTransform, Tensor } from "../../lib/api/types";
import { formatBytes } from "../../lib/format";
import { useResource } from "../../lib/useResource";
import { SliceControls } from "../catalog/TensorWorkbench";
import { Overview, type Derived } from "../overview/Overview";
import type { RawRect } from "../overview/geometry";
import { RoiExplorer } from "../roi/RoiExplorer";
import { defaultView, viewDims, viewError, type ViewState, type Win } from "../roi/view";
import { findEntry, findTensor } from "./findTensor";

export const TRANSFORM_LABEL: Record<DeltaTransform, string> = {
  b_minus_a: "B − A (signed)",
  abs_b_minus_a: "|B − A|",
  relative_abs: "|B − A| / max(|A|, ε)",
};

/** ε of relative_abs as typed: a finite number with 0 < ε <= 1 (the contract's bounds), else an error text. */
export function epsilonError(text: string): string | null {
  const t = text.trim();
  const v = Number(t);
  if (t === "" || !Number.isFinite(v)) return "enter a number";
  if (!(v > 0 && v <= 1)) return "ε must be greater than 0 and at most 1";
  return null;
}

interface Subject {
  entryId: string;
  a: Tensor;
  b: Tensor;
}

interface Props {
  comparisonId: string;
  row: TensorRow;
  modelA: { id: string; name: string };
  modelB: { id: string; name: string };
}

type Tab = "heatmap" | "roi";

export function DeltaWorkbench({ comparisonId, row, modelA, modelB }: Props) {
  const [open, setOpen] = useState(false);
  const subject = useResource<Subject>(open ? `delta#${comparisonId}#${row.key}` : null, async (signal) => {
    const [entry, a, b] = await Promise.all([
      findEntry(comparisonId, row.key, signal),
      findTensor(modelA.id, row.key, signal),
      findTensor(modelB.id, row.key, signal),
    ]);
    return { entryId: entry.entry_id, a, b };
  });
  return (
    <section className="delta-heatmap stack-sm" aria-labelledby="dh-h" data-testid="delta-heatmap">
      <div className="pane-head">
        <h4 id="dh-h" className="h4">
          Delta heatmap (B − A)
        </h4>
        <button type="button" className="btn btn-small" aria-expanded={open} aria-controls="dh-body" onClick={() => setOpen((o) => !o)} data-testid="delta-toggle">
          {open ? "Hide delta heatmap" : "Show delta heatmap"}
        </button>
      </div>
      {!open ? (
        <p className="muted small">
          Where inside this tensor did values change? Computed by the server elementwise from both checkpoints (B = {modelB.name}, A = {modelA.name}); nothing is read
          until you open it.
        </p>
      ) : (
        <div id="dh-body" className="stack-sm">
          {subject.error ? <ErrorPanel error={subject.error} onRetry={subject.reload} /> : null}
          {subject.loading && !subject.data ? <p className="muted small">Finding the tensor in both checkpoints…</p> : null}
          {subject.data ? <DeltaBody key={subject.data.entryId} comparisonId={comparisonId} row={row} subject={subject.data} /> : null}
        </div>
      )}
    </section>
  );
}

function DeltaBody({ comparisonId, row, subject }: { comparisonId: string; row: TensorRow; subject: Subject }) {
  const { a, b } = subject;
  const shape = a.shape;
  const empty = a.numel === 0 || shape.some((d) => d === 0);
  const canHeatmap = shape.length >= 2 && !empty;
  const [transform, setTransform] = useState<DeltaTransform>("b_minus_a");
  const [epsText, setEpsText] = useState("1e-8");
  const [eps, setEps] = useState(1e-8);
  const [tab, setTab] = useState<Tab>(canHeatmap ? "heatmap" : "roi");
  const [view, setView] = useState<ViewState>(() => defaultView(shape));
  const [aggregation, setAggregation] = useState<Aggregation>("mean");
  const [focus, setFocus] = useState<{ win: Win; nonce: number; from: RawRect } | null>(null);
  const settings = useResource("settings", (signal) => api.settings(signal));
  const vErr = viewError(shape, view);
  const dims = vErr ? null : viewDims(shape, view);
  const epsErr = epsilonError(epsText);

  const source: DeltaSource = { kind: "delta", comparison_id: comparisonId, entry_id: subject.entryId, transform };
  if (transform === "relative_abs") source.epsilon = eps;
  const signed = transform === "b_minus_a";
  const inputs = a.dtype === b.dtype ? `${a.dtype} inputs` : `${a.dtype} → ${b.dtype} inputs`;
  const what = transform === "relative_abs" ? `|B − A| / max(|A|, ${eps})` : TRANSFORM_LABEL[transform].replace(" (signed)", "");
  const derived: Derived = {
    source,
    title: `Delta heatmap, ${what}`,
    sourceLabel: `${what} of ${inputs}, float64`,
    readBytes: a.byte_length + b.byte_length,
    limitBytes: Math.max(a.byte_length, b.byte_length),
    initialPalette: signed ? "diverging" : "sequential",
    initialMode: signed ? "symmetric" : "auto",
  };

  function inspect(r: RawRect) {
    setFocus((f) => ({ win: { r0: r.r0, c0: r.c0, h: r.r1 - r.r0, w: r.c1 - r.c0 }, nonce: (f?.nonce ?? 0) + 1, from: r }));
    setTab("roi");
    requestAnimationFrame(() => document.getElementById("dh-tab-roi")?.focus());
  }
  const tabs: Tab[] = canHeatmap ? ["heatmap", "roi"] : ["roi"];
  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const next = tabs[(tabs.indexOf(tab) + 1) % tabs.length] ?? tab;
    setTab(next);
    document.getElementById(`dh-tab-${next}`)?.focus();
  }

  if (empty) return <p className="muted small">This tensor has no elements; there is no delta to draw.</p>;
  // Remount the heatmap when the source or view changes (as the tensor workbench does) and the explorer when the
  // source changes, so each starts on the scale that suits what it draws.
  const key = JSON.stringify({ source, view });
  return (
    <div className="stack-sm">
      <div className="row" role="group" aria-label="Delta transform">
        <label className="field field-num">
          <span className="label-inline">Show</span>
          <select className="input input-small" value={transform} onChange={(e) => setTransform(e.target.value as DeltaTransform)} data-testid="delta-transform">
            {(Object.keys(TRANSFORM_LABEL) as DeltaTransform[]).map((t) => (
              <option key={t} value={t}>
                {TRANSFORM_LABEL[t]}
              </option>
            ))}
          </select>
        </label>
        {transform === "relative_abs" ? (
          <form
            className="row"
            aria-label="Relative change epsilon"
            onSubmit={(e) => {
              e.preventDefault();
              if (!epsErr) setEps(Number(epsText.trim()));
            }}
          >
            <label className="field field-num">
              <span className="label-inline">ε (denominator floor)</span>
              <input
                className="input input-small"
                inputMode="decimal"
                value={epsText}
                aria-invalid={epsErr ? true : undefined}
                aria-describedby={epsErr ? "dh-eps-err" : undefined}
                onChange={(e) => setEpsText(e.target.value)}
                data-testid="delta-epsilon"
              />
            </label>
            <button type="submit" className="btn btn-small field-end" disabled={epsErr !== null}>
              Apply ε
            </button>
            {epsErr ? (
              <span id="dh-eps-err" className="field-error">
                {epsErr}
              </span>
            ) : null}
          </form>
        ) : null}
      </div>
      <p className="muted small" data-testid="delta-explainer">
        {signed
          ? "Colour is centred on zero: one hue where B is above A, the other where it is below. A block’s mean can cancel opposite changes; use rms, min or max to see their size."
          : transform === "abs_b_minus_a"
            ? "Size of the change only; the sign is not shown."
            : "Change relative to A’s magnitude; where |A| < ε the denominator is ε, so near-zero weights do not explode."}{" "}
        It shows where values changed, not why or whether it matters. One full read is {formatBytes(derived.readBytes)} (both checkpoints).
      </p>
      {canHeatmap ? (
        <div role="tablist" aria-label="Delta views" className="tabs" onKeyDown={onKeyDown}>
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              id={`dh-tab-${t}`}
              aria-selected={tab === t}
              aria-controls={tab === t ? `dh-panel-${t}` : undefined}
              tabIndex={tab === t ? 0 : -1}
              className="tab"
              onClick={() => setTab(t)}
            >
              {t === "heatmap" ? "Heatmap" : "Exact ROI"}
            </button>
          ))}
        </div>
      ) : null}
      {tab === "heatmap" && canHeatmap ? (
        <div role="tabpanel" id="dh-panel-heatmap" aria-labelledby="dh-tab-heatmap">
          {shape.length > 2 ? <SliceControls shape={shape} view={view} onView={setView} /> : null}
          {vErr || !dims ? (
            <p className="notice notice-warn" role="alert">
              <span>
                <strong>Unsupported view:</strong> {vErr}. Nothing is read until the view is valid.
              </span>
            </p>
          ) : (
            <Overview
              key={key}
              modelId={a.model_id}
              tensor={a}
              view={view}
              dims={dims}
              settings={settings.data}
              aggregation={aggregation}
              onAggregation={setAggregation}
              onViewChange={setView}
              onInspect={inspect}
              derived={derived}
            />
          )}
        </div>
      ) : null}
      {tab === "roi" ? (
        <div role="tabpanel" id="dh-panel-roi" aria-labelledby={canHeatmap ? "dh-tab-roi" : undefined}>
          {focus ? (
            <p className="notice notice-info" data-testid="delta-roi-from-heatmap">
              Opened from the heatmap selection: rows [{focus.from.r0}, {focus.from.r1}) × columns [{focus.from.c0}, {focus.from.c1}). These are exact elementwise values of{" "}
              {what}, read from both checkpoints, not the heatmap’s aggregates.
            </p>
          ) : null}
          <RoiExplorer
            key={`${JSON.stringify(source)}#${focus?.nonce ?? 0}`}
            modelId={a.model_id}
            tensor={a}
            initialView={canHeatmap ? view : undefined}
            initialWin={focus?.win}
            onViewChange={canHeatmap ? setView : undefined}
            source={source}
            valueLabel={`${what} · float64 (${inputs})`}
            initialPalette={signed ? "diverging" : "sequential"}
            initialMode={signed ? "symmetric" : "auto"}
          />
        </div>
      ) : null}
      <p className="muted small">
        Entry of <span className="mono">{row.key}</span> in this comparison; tile and value warnings from the server (lineage, mixed dtypes) appear with the results.
      </p>
    </div>
  );
}
