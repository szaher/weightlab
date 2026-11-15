// Exact value histogram of one tensor (ROADMAP P0.2; ADR-021). The backend counts every element in a bounded,
// cancellable job; this panel chooses the binning, runs the job and draws its validated result: value bins on a
// linear or log count axis, the magnitude (binary exponent) view for float tensors, and the special values that are
// counted but never binned (NaN, ±Inf, zeros, -0, values outside an explicit range). Every chart has a table view.
import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError } from "../../lib/api/client";
import {
  DEFAULT_BINS,
  HISTOGRAM_WORKING_SET_MIB,
  MAX_BINS,
  type Consistency,
  type HistogramOptions,
  type HistogramBins,
  type HistogramRange,
  type TensorHistogram,
  toNumber,
} from "../../lib/api/histogram";
import { PENDING_JOB_STATES, type ErrorBody, type Job, type Tensor } from "../../lib/api/types";
import { formatBytes, formatInt } from "../../lib/format";
import { BarChart, type Column } from "./BarChart";
import { binInterval, edgeFractions, exactValue, magnitudeColumns, percent, shortValue, type YScale } from "./chart";
import { MAX_POLL_FAILURES, useHistogramJob, type HistogramRun } from "./useHistogramJob";

const BIN_CHOICES = [16, 32, 64, 128, 256, 512, MAX_BINS];
const INTEGER_DTYPES = new Set(["I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL"]);
const MAX_SAFE = Number.MAX_SAFE_INTEGER;

/** What resolves each failure of a histogram job (submit errors and failed jobs). */
export const HISTOGRAM_ADVICE: Record<string, string> = {
  FILE_CHANGED:
    "The tensor's file changed since the model was imported. Re-import the model (or relink it in the Workspace if it was only moved), then compute the histogram again.",
  RESOURCE_LIMIT: `A histogram needs ${HISTOGRAM_WORKING_SET_MIB} MiB of working memory and enough time to read the whole tensor. Raise memory_budget_mib or max_job_seconds in Settings if this computer has room.`,
  VALIDATION_ERROR: "Check the binning: explicit bounds must satisfy lo ≤ hi, and integer tensors need whole-number bounds.",
  NOT_FOUND: "The model or tensor was deleted. Select it again from the catalog.",
  RATE_LIMITED: "The job queue is full. Wait for running jobs to finish, then try again.",
};

/** Submit errors whose generic explanation (ErrorPanel) lacks the histogram-specific remedy. */
const SUBMIT_ADVICE = new Set(["RESOURCE_LIMIT", "VALIDATION_ERROR"]);

const STATE_TEXT: Record<Job["state"], string> = {
  created: "Queued",
  queued: "Queued",
  running: "Counting",
  completed: "Completed",
  failed: "Failed",
  canceled: "Canceled",
  interrupted: "Interrupted",
};

const RANGE_TEXT: Record<HistogramRange, string> = {
  min_max: "Min to max",
  symmetric: "Symmetric around 0",
  explicit: "Explicit range",
};

interface Props {
  modelId: string;
  tensor: Tensor;
  settings: Record<string, number | boolean> | undefined;
}

/** Parses an explicit bound; null with a reason when it is not usable for this tensor. */
function parseBound(text: string, integer: boolean): { value: number | null; error: string | null } {
  const t = text.trim();
  if (t === "") return { value: null, error: "required" };
  const v = Number(t);
  if (!Number.isFinite(v)) return { value: null, error: "not a finite number" };
  if (integer && !Number.isInteger(v)) return { value: null, error: "must be a whole number for an integer tensor" };
  if (integer && Math.abs(v) > MAX_SAFE) return { value: null, error: "beyond ±(2^53 − 1)" };
  return { value: v, error: null };
}

export function HistogramPanel({ modelId, tensor, settings }: Props) {
  const integer = INTEGER_DTYPES.has(tensor.dtype);
  const [bins, setBins] = useState(DEFAULT_BINS);
  const [range, setRange] = useState<HistogramRange>("min_max");
  const [loText, setLoText] = useState("");
  const [hiText, setHiText] = useState("");
  const [consistency, setConsistency] = useState<Consistency>("best_effort");
  const hj = useHistogramJob(modelId, tensor.id, settings);

  const lo = parseBound(loText, integer);
  const hi = parseBound(hiText, integer);
  const orderError = range === "explicit" && lo.value !== null && hi.value !== null && lo.value > hi.value ? "lo must not exceed hi" : null;
  const formError = range === "explicit" ? (lo.error ? `lo: ${lo.error}` : hi.error ? `hi: ${hi.error}` : orderError) : null;
  const memMib = typeof settings?.memory_budget_mib === "number" ? settings.memory_budget_mib : null;
  const tooLittleMemory = memMib !== null && memMib < HISTOGRAM_WORKING_SET_MIB;

  const { job } = hj;
  const pending = !!job && PENDING_JOB_STATES.includes(job.state);
  const jobError: ErrorBody | null = job?.state === "failed" ? (job.error ?? null) : null;

  function compute() {
    const options: HistogramOptions = range === "explicit" ? { bins, range, lo: lo.value ?? 0, hi: hi.value ?? 0 } : { bins, range };
    void hj.submit({ options, consistency });
  }

  return (
    <section className="hist stack" aria-labelledby="hist-h" data-testid="histogram-panel">
      <h3 id="hist-h" className="h3">
        Value distribution
      </h3>
      <p className="help">
        Every element is counted exactly, in a background job that reads the tensor in bounded chunks. NaN and ±Inf are counted but never placed in a bin.
      </p>

      <fieldset className="hist-controls" disabled={pending || hj.busy}>
        <legend className="label">Binning</legend>
        <div className="row">
          <label className="field field-num">
            <span className="label-inline">Bins (at most)</span>
            <select className="input input-small" value={bins} onChange={(e) => setBins(Number(e.target.value))} data-testid="hist-bins">
              {BIN_CHOICES.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </label>
          <label className="field field-num">
            <span className="label-inline">Range</span>
            <select className="input input-small" value={range} onChange={(e) => setRange(e.target.value as HistogramRange)} data-testid="hist-range">
              {(Object.keys(RANGE_TEXT) as HistogramRange[]).map((r) => (
                <option key={r} value={r}>
                  {RANGE_TEXT[r]}
                </option>
              ))}
            </select>
          </label>
          {range === "explicit" ? (
            <>
              <label className="field field-num">
                <span className="label-inline">Lower bound (lo)</span>
                <input className="input input-small" inputMode="decimal" value={loText} onChange={(e) => setLoText(e.target.value)} aria-invalid={!!lo.error || !!orderError} />
              </label>
              <label className="field field-num">
                <span className="label-inline">Upper bound (hi)</span>
                <input className="input input-small" inputMode="decimal" value={hiText} onChange={(e) => setHiText(e.target.value)} aria-invalid={!!hi.error || !!orderError} />
              </label>
            </>
          ) : null}
        </div>
        <div className="row">
          <span className="radio-row" role="radiogroup" aria-label="Consistency">
            <label className="radio">
              <input type="radio" name={`hist-consistency-${tensor.id}`} checked={consistency === "best_effort"} onChange={() => setConsistency("best_effort")} />
              Best effort
            </label>
            <label className="radio" title="Hashes the tensor's files before and after the scan (two extra full reads)">
              <input type="radio" name={`hist-consistency-${tensor.id}`} checked={consistency === "verified"} onChange={() => setConsistency("verified")} />
              Verified (SHA-256 before and after)
            </label>
          </span>
          <button type="button" className="btn btn-primary" disabled={!!formError || tooLittleMemory} onClick={compute} data-testid="hist-compute">
            {hj.result ? "Recompute histogram" : "Compute histogram"}
          </button>
        </div>
        <p className="muted small" data-testid="hist-cost">
          {range === "explicit"
            ? `One pass over the tensor (${formatBytes(tensor.byte_length)}): values outside [lo, hi] are counted as below / above, never dropped.`
            : `Two passes over the tensor (${formatBytes(2 * tensor.byte_length)} read): the value range first, then the bins.`}
          {consistency === "verified" ? " Verified also hashes the files holding it, twice." : ""}
        </p>
        {formError ? (
          <p className="field-error" role="alert" data-testid="hist-form-error">
            {formError}
          </p>
        ) : null}
        {tooLittleMemory ? (
          <p className="notice notice-warn" role="alert">
            The memory budget in Settings ({memMib} MiB) is below the {HISTOGRAM_WORKING_SET_MIB} MiB a histogram needs.
          </p>
        ) : null}
      </fieldset>

      {pending && job ? (
        <div role="status" aria-live="polite" className="stack-sm" data-testid="hist-progress">
          <p>
            <strong>{STATE_TEXT[job.state]}</strong>
            {job.state === "running" ? ` · ${Math.round(job.progress * 100)}%` : ""}
            {job.estimated_bytes ? ` · ${formatBytes(job.bytes_scanned ?? 0)} of ${formatBytes(job.estimated_bytes)} read` : ""}
            {job.deduplicated ? " · joined the identical histogram that was already running" : ""}
          </p>
          <progress max={1} value={job.progress} aria-label="Histogram progress" />
          <button type="button" className="btn btn-small" disabled={hj.busy} onClick={() => void hj.cancel()}>
            Cancel
          </button>
          {hj.pollFailures >= MAX_POLL_FAILURES ? (
            <button type="button" className="btn btn-small" onClick={hj.checkAgain}>
              Check again
            </button>
          ) : null}
        </div>
      ) : null}

      {job && !pending && job.state !== "completed" ? (
        <div className={`notice ${job.state === "failed" ? "notice-error" : "notice-warn"}`} role="alert" data-testid="hist-job-ended">
          <p className="notice-title">
            <span aria-hidden="true">⚠ </span>
            {job.state === "failed" ? "The histogram failed" : job.state === "canceled" ? "The histogram was canceled" : "The histogram was interrupted (WeightLab stopped while it ran)"}
          </p>
          {jobError ? (
            <>
              <p>{jobError.message}</p>
              {HISTOGRAM_ADVICE[jobError.code] ? <p className="notice-action">{HISTOGRAM_ADVICE[jobError.code]}</p> : null}
              <p className="notice-meta">
                Code <code>{jobError.code}</code> · Job <code>{job.job_id}</code>
              </p>
            </>
          ) : null}
          <p>No partial histogram is ever shown.{hj.result ? ` The histogram below is the previous completed one (${RANGE_TEXT[hj.result.run.options.range].toLowerCase()}, ${hj.result.run.options.bins} bins).` : ""}</p>
        </div>
      ) : null}
      {hj.error ? (
        <>
          <ErrorPanel error={hj.error} />
          {hj.error instanceof ApiRequestError && SUBMIT_ADVICE.has(hj.error.code) ? <p className="notice-action">{HISTOGRAM_ADVICE[hj.error.code]}</p> : null}
        </>
      ) : null}

      {hj.result ? <HistogramResult h={hj.result.data.histogram} run={hj.result.run} verified={hj.result.data.consistency === "verified"} /> : null}
    </section>
  );
}

function HistogramResult({ h, run, verified }: { h: TensorHistogram; run: HistogramRun; verified: boolean }) {
  const [view, setView] = useState<"values" | "magnitude">("values");
  const [scale, setScale] = useState<YScale>("linear");
  const [table, setTable] = useState(false);
  const c = h.counts;
  const b = h.bins;
  const floatish = h.value_domain === "float";
  const merged = b ? b.counts.length < b.requested_bins : false;

  return (
    <div className="hist-result stack-sm" data-testid="hist-result">
      <p className="tensor-sub">
        {RANGE_TEXT[b?.range ?? run.options.range]}
        {b ? ` · ${b.counts.length} bin${b.counts.length === 1 ? "" : "s"}` : ""}
        {merged && b ? ` (of ${b.requested_bins} requested: the range has fewer distinct ${floatish ? "float64 edges" : "whole-number steps"})` : ""}
        {` · ${verified ? "verified" : "best effort"} · exact, all ${formatInt(c.elements)} elements`}
      </p>

      <dl className="facts facts-grid hist-counts" data-testid="hist-counts">
        <Fact label="Min" value={h.min === null ? "—" : exactValue(h.min)} mono />
        <Fact label="Max" value={h.max === null ? "—" : exactValue(h.max)} mono />
        <Fact label="Finite" value={`${formatInt(c.finite)} (${percent(c.finite, c.elements)})`} />
        <Fact label="Zeros" value={`${formatInt(c.zeros)}${floatish && c.negative_zeros > 0 ? ` (incl. ${formatInt(c.negative_zeros)} −0)` : ""}`} />
        <Fact label="Positive" value={formatInt(c.positives)} />
        <Fact label="Negative" value={formatInt(c.negatives)} />
        {floatish ? <Fact label="NaN" value={formatInt(c.nan)} warn={c.nan > 0} /> : null}
        {floatish ? <Fact label="+Inf" value={formatInt(c.pos_inf)} warn={c.pos_inf > 0} /> : null}
        {floatish ? <Fact label="−Inf" value={formatInt(c.neg_inf)} warn={c.neg_inf > 0} /> : null}
        {b && b.range === "explicit" ? <Fact label={`Below ${exactValue(b.lo)}`} value={formatInt(b.below)} /> : null}
        {b && b.range === "explicit" ? <Fact label={`Above ${exactValue(b.hi)}`} value={formatInt(b.above)} /> : null}
      </dl>

      {!b ? (
        <p className="notice notice-info" data-testid="hist-no-bins">
          This tensor has no finite values, so there is nothing to bin; the counts above are the whole result.
        </p>
      ) : (
        <>
          <div className="row hist-toolbar">
            {h.magnitude ? (
              <span className="radio-row" role="radiogroup" aria-label="Histogram view">
                <label className="radio">
                  <input type="radio" name="hist-view" checked={view === "values"} onChange={() => setView("values")} />
                  Values
                </label>
                <label className="radio">
                  <input type="radio" name="hist-view" checked={view === "magnitude"} onChange={() => setView("magnitude")} />
                  Magnitude (log₂ |x|)
                </label>
              </span>
            ) : null}
            <span className="radio-row" role="radiogroup" aria-label="Count axis">
              <label className="radio">
                <input type="radio" name="hist-scale" checked={scale === "linear"} onChange={() => setScale("linear")} />
                Linear counts
              </label>
              <label className="radio">
                <input type="radio" name="hist-scale" checked={scale === "log"} onChange={() => setScale("log")} />
                Log counts
              </label>
            </span>
            <label className="radio">
              <input type="checkbox" checked={table} onChange={(e) => setTable(e.target.checked)} data-testid="hist-table-toggle" />
              Show table
            </label>
          </div>
          {view === "magnitude" && h.magnitude ? <MagnitudeView h={h} m={h.magnitude} scale={scale} table={table} /> : <ValuesView h={h} b={b} scale={scale} table={table} />}
        </>
      )}
      <p className="help small">{h.method}.</p>
    </div>
  );
}

function Fact({ label, value, mono = false, warn = false }: { label: string; value: string; mono?: boolean; warn?: boolean }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className={`${mono ? "mono " : ""}${warn ? "hist-warn" : ""}`.trim() || undefined}>
        {warn ? <span aria-hidden="true">⚠ </span> : null}
        {value}
      </dd>
    </div>
  );
}

function ValuesView({ h, b, scale, table }: { h: TensorHistogram; b: HistogramBins; scale: YScale; table: boolean }) {
  const fr = edgeFractions(b.edges);
  const columns: Column[] = b.counts.map((n, i) => ({ x0: fr[i] ?? 0, x1: fr[i + 1] ?? 1, up: n }));
  const total = h.counts.finite;
  const first = b.edges[0] ?? 0;
  const last = b.edges[b.edges.length - 1] ?? first;
  const xLabels = [
    { at: 0, text: shortValue(first) },
    { at: 1, text: shortValue(last) },
  ];
  const lo = toNumber(first);
  const hi = toNumber(last);
  if (lo < 0 && hi > 0 && -lo / (hi - lo) > 0.12 && -lo / (hi - lo) < 0.88) xLabels.push({ at: -lo / (hi - lo), text: "0" });
  const interval = (i: number) => binInterval(b.edges, i, h.value_domain, b.last_bin_closed);
  return (
    <>
      <BarChart
        columns={columns}
        scale={scale}
        xLabels={xLabels}
        label={`Histogram of ${b.counts.length} value bins, ${scale} count axis`}
        describe={(i) => `${interval(i)}: ${formatInt(b.counts[i] ?? 0)} (${percent(b.counts[i] ?? 0, total)} of finite values)`}
        testId="hist-values"
      />
      {table ? (
        <div className="table-wrap">
          <table className="table hist-table" data-testid="hist-values-table">
            <caption className="visually-hidden">Value bins: interval and count</caption>
            <thead>
              <tr>
                <th scope="col">Interval</th>
                <th scope="col" className="num">
                  Count
                </th>
                <th scope="col" className="num">
                  Share of finite
                </th>
              </tr>
            </thead>
            <tbody>
              {b.range === "explicit" ? <Row label={`< ${exactValue(b.lo)} (below)`} n={b.below} total={total} /> : null}
              {b.counts.map((n, i) => (
                <Row key={i} label={interval(i)} n={n} total={total} />
              ))}
              {b.range === "explicit" ? <Row label={`> ${exactValue(b.hi)} (above)`} n={b.above} total={total} /> : null}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="help small">{b.rule}.</p>
    </>
  );
}

function Row({ label, n, total }: { label: string; n: number; total: number }) {
  return (
    <tr>
      <td className="mono">{label}</td>
      <td className="num mono">{formatInt(n)}</td>
      <td className="num mono">{percent(n, total)}</td>
    </tr>
  );
}

function MagnitudeView({ h, m, scale, table }: { h: TensorHistogram; m: NonNullable<TensorHistogram["magnitude"]>; scale: YScale; table: boolean }) {
  const cols = magnitudeColumns(m.buckets);
  const n = cols.length;
  const columns: Column[] = cols.map((c, i) => ({ x0: i / n, x1: (i + 1) / n, up: c.positive, down: c.negative }));
  const nonzero = h.counts.positives + h.counts.negatives;
  const picks = n > 2 ? [0, Math.floor(n / 2), n - 1] : n === 2 ? [0, 1] : n === 1 ? [0] : [];
  const xLabels = picks.map((i) => ({ at: (i + 0.5) / n, text: `2^${cols[i]?.exponent ?? 0}` }));
  const bucket = (e: number) => `2^${e} ≤ |x| < 2^${e + 1}`;
  if (n === 0)
    return (
      <p className="notice notice-info" data-testid="hist-magnitude-empty">
        Every finite value is zero, so there are no magnitudes to show.
      </p>
    );
  return (
    <>
      <p className="muted small">Positive values above the line, negative values below, on one count scale. Zeros are counted above, not here.</p>
      <BarChart
        columns={columns}
        scale={scale}
        mirrored
        xLabels={xLabels}
        label={`Magnitude histogram over ${n} binary exponents, ${scale} count axis`}
        describe={(i) => {
          const c = cols[i] ?? { exponent: 0, positive: 0, negative: 0 };
          return `${bucket(c.exponent)}: ${formatInt(c.positive)} positive, ${formatInt(c.negative)} negative (${percent(c.positive + c.negative, nonzero)} of nonzero values)`;
        }}
        testId="hist-magnitude"
      />
      {table ? (
        <div className="table-wrap">
          <table className="table hist-table" data-testid="hist-magnitude-table">
            <caption className="visually-hidden">Magnitude buckets: binary exponent, positive and negative counts</caption>
            <thead>
              <tr>
                <th scope="col">|x| range</th>
                <th scope="col" className="num">
                  Positive
                </th>
                <th scope="col" className="num">
                  Negative
                </th>
              </tr>
            </thead>
            <tbody>
              {m.buckets.map((k) => (
                <tr key={k.exponent}>
                  <td className="mono">{bucket(k.exponent)}</td>
                  <td className="num mono">{formatInt(k.positive)}</td>
                  <td className="num mono">{formatInt(k.negative)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="help small">{m.rule}; empty exponents between the smallest and largest are drawn as gaps.</p>
    </>
  );
}
