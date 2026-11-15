// "Value distributions, A vs B" in the Compare drill-down (ROADMAP P0.2 overlay): exact histograms of the same
// tensor in A and B over one shared range, so both use the same bins. Counts are the backend's (tensor_histogram
// jobs); this section only chooses the range, runs the jobs and draws the two results on one axis, with every
// count and B − A in a table.
import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError } from "../../lib/api/client";
import { DEFAULT_BINS, INTEGER_DTYPES, MAX_BINS, parseBound, type TensorHistogram } from "../../lib/api/histogram";
import { api } from "../../lib/api/endpoints";
import type { TensorRow } from "../../lib/api/compare";
import { PENDING_JOB_STATES, type Job } from "../../lib/api/types";
import { formatBytes, formatInt } from "../../lib/format";
import { useResource } from "../../lib/useResource";
import { binInterval, exactValue, percent, type YScale } from "../histogram/chart";
import { HISTOGRAM_ADVICE } from "../histogram/HistogramPanel";
import { OverlayChart } from "../histogram/OverlayChart";
import { signed } from "../histogram/overlay";
import { useDistributionCompare, type OverlayResult, type RangeMode } from "./useDistributionCompare";

const ESIZE: Record<string, number> = { F64: 8, I64: 8, U64: 8, F32: 4, I32: 4, U32: 4, F16: 2, BF16: 2, I16: 2, U16: 2 }; // 1 byte otherwise
const BIN_CHOICES = [16, 32, 64, 128, 256, 512, MAX_BINS];
const PHASE_TEXT = {
  resolving: "Finding the tensor in A and B",
  range: "Step 1 of 2: reading both tensors' value ranges",
  bins: "Step 2 of 2: counting both tensors over the shared range",
} as const;
const STATE_TEXT: Record<Job["state"], string> = {
  created: "queued",
  queued: "queued",
  running: "counting",
  completed: "done",
  failed: "failed",
  canceled: "canceled",
  interrupted: "interrupted",
};

interface Props {
  row: TensorRow;
  modelA: { id: string; name: string } | null;
  modelB: { id: string; name: string } | null;
}

export function DistributionCompare({ row, modelA, modelB }: Props) {
  const settings = useResource("settings", (signal) => api.settings(signal));
  const dc = useDistributionCompare(row.key, modelA?.id ?? null, modelB?.id ?? null, settings.data);
  const [bins, setBins] = useState(DEFAULT_BINS);
  const [mode, setMode] = useState<RangeMode>("shared");
  const [loText, setLoText] = useState("");
  const [hiText, setHiText] = useState("");
  const [consistency, setConsistency] = useState<"best_effort" | "verified">("best_effort");

  const intA = INTEGER_DTYPES.has(row.dtype_a);
  const intB = INTEGER_DTYPES.has(row.dtype_b);
  const integer = intA && intB;
  const lo = parseBound(loText, integer);
  const hi = parseBound(hiText, integer);
  const formError =
    mode !== "explicit" ? null : lo.error ? `lo: ${lo.error}` : hi.error ? `hi: ${hi.error}` : (lo.value ?? 0) > (hi.value ?? 0) ? "lo must not exceed hi" : null;
  const running = dc.phase === "resolving" || dc.phase === "range" || dc.phase === "bins";
  const elements = typeof row.counts.elements === "number" ? row.counts.elements : null;
  const readBytes = elements === null ? null : (mode === "shared" ? 3 : 1) * elements * ((ESIZE[row.dtype_a] ?? 1) + (ESIZE[row.dtype_b] ?? 1));

  if (!modelA || !modelB) return null;
  if (intA !== intB)
    return (
      <section className="stack-sm" aria-labelledby="dist-h">
        <h4 id="dist-h" className="h4">
          Value distributions, A vs B
        </h4>
        <p className="notice notice-info">
          A is {row.dtype_a} and B is {row.dtype_b}: one is an integer and one a floating-point tensor, so their histograms cannot share bins. Open each side with “Inspect” to see its own histogram.
        </p>
      </section>
    );

  function run() {
    void dc.run({ bins, mode, consistency, ...(mode === "explicit" ? { lo: lo.value ?? 0, hi: hi.value ?? 0 } : {}) });
  }

  return (
    <section className="dist stack-sm" aria-labelledby="dist-h" data-testid="dist-compare">
      <h4 id="dist-h" className="h4">
        Value distributions, A vs B
      </h4>
      <p className="help">
        Exact histograms of this tensor in both checkpoints, counted by the backend over one shared range so both use the same bins. Element-wise change is in the metrics above; this shows
        how the overall value distribution moved.
      </p>
      <fieldset className="hist-controls" disabled={running}>
        <legend className="label">Binning</legend>
        <div className="row">
          <label className="field field-num">
            <span className="label-inline">Bins (at most)</span>
            <select className="input input-small" value={bins} onChange={(e) => setBins(Number(e.target.value))} data-testid="dist-bins">
              {BIN_CHOICES.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </label>
          <label className="field field-num">
            <span className="label-inline">Range</span>
            <select className="input input-small" value={mode} onChange={(e) => setMode(e.target.value as RangeMode)} data-testid="dist-range">
              <option value="shared">Min to max of A and B</option>
              <option value="explicit">Explicit range</option>
            </select>
          </label>
          {mode === "explicit" ? (
            <>
              <label className="field field-num">
                <span className="label-inline">Lower bound (lo)</span>
                <input className="input input-small" inputMode="decimal" value={loText} onChange={(e) => setLoText(e.target.value)} aria-invalid={!!formError} />
              </label>
              <label className="field field-num">
                <span className="label-inline">Upper bound (hi)</span>
                <input className="input input-small" inputMode="decimal" value={hiText} onChange={(e) => setHiText(e.target.value)} aria-invalid={!!formError} />
              </label>
            </>
          ) : null}
        </div>
        <div className="row">
          <span className="radio-row" role="radiogroup" aria-label="Consistency">
            <label className="radio">
              <input type="radio" name={`dist-consistency-${row.key}`} checked={consistency === "best_effort"} onChange={() => setConsistency("best_effort")} />
              Best effort
            </label>
            <label className="radio" title="Hashes the files holding each tensor before and after each scan">
              <input type="radio" name={`dist-consistency-${row.key}`} checked={consistency === "verified"} onChange={() => setConsistency("verified")} />
              Verified (SHA-256 before and after)
            </label>
          </span>
          <button type="button" className="btn btn-primary" disabled={!!formError} onClick={run} data-testid="dist-run">
            {dc.result ? "Recompute distributions" : "Compare distributions"}
          </button>
        </div>
        <p className="muted small" data-testid="dist-cost">
          {mode === "shared"
            ? "Three passes over each tensor: its value range (two passes), then both tensors binned over the union of their ranges (one pass)."
            : "One pass over each tensor: values outside [lo, hi] are counted as below / above, never dropped."}
          {readBytes !== null ? ` ${formatBytes(readBytes)} read in total.` : ""}
          {consistency === "verified" ? " Verified also hashes the files holding each tensor, twice per job." : ""}
        </p>
        {formError ? (
          <p className="field-error" role="alert" data-testid="dist-form-error">
            {formError}
          </p>
        ) : null}
      </fieldset>

      {running ? (
        <div role="status" aria-live="polite" className="stack-sm" data-testid="dist-progress">
          <p>
            <strong>{PHASE_TEXT[dc.phase as keyof typeof PHASE_TEXT]}</strong>
          </p>
          {(["a", "b"] as const).map((s) => {
            const j = dc.jobs[s];
            return j ? (
              <p key={s} className="small">
                {s.toUpperCase()}: {STATE_TEXT[j.state]}
                {j.state === "running" ? ` · ${Math.round(j.progress * 100)}%` : ""}
                {j.deduplicated ? " · joined an identical running job" : ""}
                <progress max={1} value={PENDING_JOB_STATES.includes(j.state) ? j.progress : 1} aria-label={`${s.toUpperCase()} progress`} className="dist-progress" />
              </p>
            ) : null;
          })}
          <button type="button" className="btn btn-small" onClick={() => void dc.cancel()}>
            Cancel
          </button>
        </div>
      ) : null}
      {dc.phase === "canceled" ? (
        <p className="notice notice-warn" role="status" data-testid="dist-canceled">
          Canceled. Both jobs were asked to stop; no partial overlay is shown.
        </p>
      ) : null}
      {dc.phase === "failed" && dc.error ? (
        <div data-testid="dist-error">
          <ErrorPanel error={dc.error} />
          {dc.error instanceof ApiRequestError && dc.error.status === 200 && HISTOGRAM_ADVICE[dc.error.code] ? <p className="notice-action">{HISTOGRAM_ADVICE[dc.error.code]}</p> : null}
          <p className="muted small">The other side's job was canceled; no partial overlay is shown.</p>
        </div>
      ) : null}
      {dc.result && dc.phase === "done" ? <OverlayView r={dc.result} nameA={modelA.name} nameB={modelB.name} /> : null}
    </section>
  );
}

function OverlayView({ r, nameA, nameB }: { r: OverlayResult; nameA: string; nameB: string }) {
  const [scale, setScale] = useState<YScale>("linear");
  const [table, setTable] = useState(false);
  const ba = r.a.bins;
  const bb = r.b.bins;
  if (!ba || !bb) return null;
  const interval = (i: number) => binInterval(ba.edges, i, r.a.value_domain, ba.last_bin_closed);
  const describe = (i: number) => {
    const a = ba.counts[i] ?? 0;
    const b = bb.counts[i] ?? 0;
    return `${interval(i)}: A ${formatInt(a)} (${percent(a, r.a.counts.finite)}), B ${formatInt(b)} (${percent(b, r.b.counts.finite)}), B − A ${signed(b - a)}`;
  };
  const rangeText = r.request.mode === "shared" ? "the union of A's and B's finite ranges" : "the explicit range";
  return (
    <div className="stack-sm" data-testid="dist-result">
      <p className="tensor-sub">
        {ba.counts.length} bins over [{exactValue(ba.lo)}, {exactValue(ba.hi)}], {rangeText} · {r.consistency === "verified" ? "verified" : "best effort"} · exact, every element of both tensors
      </p>
      <div className="table-wrap">
        <table className="table table-compact" data-testid="dist-counts">
          <caption className="visually-hidden">Counts of A and B</caption>
          <thead>
            <tr>
              <th scope="col">Count</th>
              <th scope="col" className="num">
                A
              </th>
              <th scope="col" className="num">
                B
              </th>
            </tr>
          </thead>
          <tbody>
            <CountRow label="Min" a={r.a.min === null ? "—" : exactValue(r.a.min)} b={r.b.min === null ? "—" : exactValue(r.b.min)} />
            <CountRow label="Max" a={r.a.max === null ? "—" : exactValue(r.a.max)} b={r.b.max === null ? "—" : exactValue(r.b.max)} />
            {countRows(r.a, r.b).map(([l, a, b]) => (
              <CountRow key={l} label={l} a={formatInt(a)} b={formatInt(b)} />
            ))}
            {r.request.mode === "explicit" ? (
              <>
                <CountRow label="Below lo" a={formatInt(ba.below)} b={formatInt(bb.below)} />
                <CountRow label="Above hi" a={formatInt(ba.above)} b={formatInt(bb.above)} />
              </>
            ) : null}
          </tbody>
        </table>
      </div>
      <div className="row hist-toolbar">
        <span className="radio-row" role="radiogroup" aria-label="Count axis">
          <label className="radio">
            <input type="radio" name="dist-scale" checked={scale === "linear"} onChange={() => setScale("linear")} />
            Linear counts
          </label>
          <label className="radio">
            <input type="radio" name="dist-scale" checked={scale === "log"} onChange={() => setScale("log")} />
            Log counts
          </label>
        </span>
        <label className="radio">
          <input type="checkbox" checked={table} onChange={(e) => setTable(e.target.checked)} data-testid="dist-table-toggle" />
          Show table
        </label>
      </div>
      <OverlayChart edges={ba.edges} a={ba.counts} b={bb.counts} nameA={nameA} nameB={nameB} scale={scale} describe={describe} />
      {table ? (
        <div className="table-wrap">
          <table className="table hist-table" data-testid="dist-bins-table">
            <caption className="visually-hidden">Bins: interval, A count, B count, B − A</caption>
            <thead>
              <tr>
                <th scope="col">Interval</th>
                <th scope="col" className="num">
                  A
                </th>
                <th scope="col" className="num">
                  B
                </th>
                <th scope="col" className="num">
                  B − A
                </th>
              </tr>
            </thead>
            <tbody>
              {ba.counts.map((a, i) => {
                const b = bb.counts[i] ?? 0;
                return (
                  <tr key={i}>
                    <td className="mono">{interval(i)}</td>
                    <td className="num mono">{formatInt(a)}</td>
                    <td className="num mono">{formatInt(b)}</td>
                    <td className="num mono">{signed(b - a)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="help small">{ba.rule}. Both sides use exactly these edges (checked before drawing).</p>
    </div>
  );
}

function countRows(a: TensorHistogram, b: TensorHistogram): [string, number, number][] {
  const rows: [string, number, number][] = [
    ["Elements", a.counts.elements, b.counts.elements],
    ["Finite", a.counts.finite, b.counts.finite],
    ["Zeros", a.counts.zeros, b.counts.zeros],
    ["Positive", a.counts.positives, b.counts.positives],
    ["Negative", a.counts.negatives, b.counts.negatives],
  ];
  if (a.value_domain === "float")
    rows.push(["NaN", a.counts.nan, b.counts.nan], ["+Inf", a.counts.pos_inf, b.counts.pos_inf], ["−Inf", a.counts.neg_inf, b.counts.neg_inf]);
  return rows;
}

function CountRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td className="num mono">{a}</td>
      <td className="num mono">{b}</td>
    </tr>
  );
}
