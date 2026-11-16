// One tensor of the report: identity, every metric with its method and precision, counts (non-finite values per
// side), caveats, the A/B value distributions, the delta heatmap (B − A, computed by the server in a paired pass;
// only for entries that have an elementwise difference), and the single-tensor inspection of A or B.
import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { TENSOR_METRICS, type TensorRow } from "../../lib/api/compare";
import { formatInt, formatShape } from "../../lib/format";
import { DeltaWorkbench } from "./DeltaWorkbench";
import { DistributionCompare } from "./DistributionCompare";
import { findTensor } from "./findTensor";
import { METRIC_INFO, metricVal, valText, valTitle } from "./metrics";

/** Report rows whose entry has an elementwise B − A (the server also checks that it is analyzable). */
const DELTA_ROW_STATUSES: ReadonlySet<string> = new Set(["matched", "dtype_mismatch"]);

const COUNT_LABELS: [string, string][] = [
  ["elements", "Elements"],
  ["finite_pairs", "Finite pairs"],
  ["changed", "Changed elements"],
  ["invalid_pairs", "Pairs with a non-finite side"],
  ["nan_a", "NaN in A"],
  ["pos_inf_a", "+Inf in A"],
  ["neg_inf_a", "−Inf in A"],
  ["nan_b", "NaN in B"],
  ["pos_inf_b", "+Inf in B"],
  ["neg_inf_b", "−Inf in B"],
  ["delta_overflow", "B − A overflowed float64"],
  ["b_overflow", "B overflowed float64"],
];

interface Props {
  comparisonId: string;
  row: TensorRow;
  modelA: { id: string; name: string } | null;
  modelB: { id: string; name: string } | null;
  onInspect: (modelId: string, tensorId: string) => void;
  onClose: () => void;
}

export function TensorDetail({ comparisonId, row, modelA, modelB, onInspect, onClose }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function inspect(modelId: string) {
    setBusy(modelId);
    setError(null);
    try {
      onInspect(modelId, (await findTensor(modelId, row.key)).id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  const dtype = row.dtype_a === row.dtype_b ? row.dtype_a : `${row.dtype_a} → ${row.dtype_b}`;
  return (
    <section className="tensor-detail card stack-sm" aria-labelledby="td-h">
      <div className="pane-head">
        <h3 id="td-h" className="h3 mono wrap">
          {row.key}
        </h3>
        <button type="button" className="btn btn-small" onClick={onClose} aria-label="Close tensor details">
          Close
        </button>
      </div>
      <dl className="facts">
        <div>
          <dt>Layer</dt>
          <dd>{row.layer ?? "— (no layer index)"}</dd>
        </div>
        <div>
          <dt>Role (heuristic)</dt>
          <dd className="mono">{row.role}</dd>
        </div>
        <div>
          <dt>Shape</dt>
          <dd className="mono">{formatShape(row.shape)}</dd>
        </div>
        <div>
          <dt>Dtype</dt>
          <dd className="mono">{dtype}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>{row.status.replace(/_/g, " ")}</dd>
        </div>
        <div>
          <dt>Rank in report</dt>
          <dd>{row.rank}</dd>
        </div>
      </dl>
      {row.caveat ? (
        <p className="notice notice-warn" role="note">
          {row.caveat}
        </p>
      ) : null}
      {row.withheld_metrics.length ? (
        <p className="notice notice-warn" role="note">
          Withheld (overflow or unreliable arithmetic): <span className="mono">{row.withheld_metrics.join(", ")}</span>
        </p>
      ) : null}
      <div className="table-wrap">
        <table className="table table-compact">
          <caption className="visually-hidden">Metrics of {row.key}, B − A</caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="num">
                Value
              </th>
              <th scope="col">Precision</th>
              <th scope="col">Method</th>
            </tr>
          </thead>
          <tbody>
            {[...TENSOR_METRICS, "norm_a" as const, "norm_b" as const].map((k) => {
              const m = row.metrics[k];
              const v = metricVal(m);
              const name = k === "norm_a" ? "‖A‖₂" : k === "norm_b" ? "‖B‖₂" : METRIC_INFO[k].label;
              return (
                <tr key={k}>
                  <th scope="row">{name}</th>
                  <td className={`num mono${v.kind === "value" ? "" : " val-na"}`} title={valTitle(v)}>
                    {v.kind === "value" ? String(v.v) : valText(v)}
                  </td>
                  <td>{m?.numeric_precision ?? "—"}</td>
                  <td className="small">
                    {m?.method}
                    {m?.numerical_warnings?.length ? <span className="muted"> · {m.numerical_warnings.join("; ")}</span> : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <details className="caveats">
        <summary>Element counts and non-finite values</summary>
        <dl className="facts">
          {COUNT_LABELS.filter(([k]) => row.counts[k] !== undefined).map(([k, l]) => (
            <div key={k}>
              <dt>{l}</dt>
              <dd className="mono">{typeof row.counts[k] === "number" ? formatInt(row.counts[k] as number) : String(row.counts[k])}</dd>
            </div>
          ))}
        </dl>
      </details>
      <DistributionCompare key={row.key} row={row} modelA={modelA} modelB={modelB} />
      {modelA && modelB && DELTA_ROW_STATUSES.has(row.status) ? (
        <DeltaWorkbench key={row.key} comparisonId={comparisonId} row={row} modelA={modelA} modelB={modelB} />
      ) : (
        <p className="muted small" data-testid="delta-unavailable">
          No delta heatmap: {modelA && modelB ? `this entry is ${row.status.replace(/_/g, " ")}, with no elementwise B − A` : "a checkpoint of this comparison is no longer in the project"}.
        </p>
      )}
      <div className="row">
        {modelA ? (
          <button type="button" className="btn btn-small" disabled={busy !== null} onClick={() => void inspect(modelA.id)}>
            {busy === modelA.id ? "Opening…" : `Inspect in A (${modelA.name})`}
          </button>
        ) : null}
        {modelB ? (
          <button type="button" className="btn btn-small" disabled={busy !== null} onClick={() => void inspect(modelB.id)}>
            {busy === modelB.id ? "Opening…" : `Inspect in B (${modelB.name})`}
          </button>
        ) : null}
      </div>
      {error ? <ErrorPanel error={error} /> : null}
    </section>
  );
}
