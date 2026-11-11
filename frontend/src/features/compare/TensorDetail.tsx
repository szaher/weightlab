// One tensor of the report: identity, every metric with its method and precision, counts (non-finite values per
// side), caveats, and the existing single-tensor inspection of A or B. A graphical delta heatmap is not
// implemented yet and is shown as unavailable, not as a working action.
import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError } from "../../lib/api/client";
import { TENSOR_METRICS, type TensorRow } from "../../lib/api/compare";
import { api } from "../../lib/api/endpoints";
import { DEFAULT_QUERY, MAX_SEARCH } from "../../lib/api/tensorQuery";
import { formatInt, formatShape } from "../../lib/format";
import { METRIC_INFO, metricVal, valText, valTitle } from "./metrics";

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
  row: TensorRow;
  modelA: { id: string; name: string } | null;
  modelB: { id: string; name: string } | null;
  onInspect: (modelId: string, tensorId: string) => void;
  onClose: () => void;
}

export function TensorDetail({ row, modelA, modelB, onInspect, onClose }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function inspect(modelId: string) {
    setBusy(modelId);
    setError(null);
    try {
      // The catalog search is a substring match; the exact raw key picks the one tensor (keys are unique per model).
      const page = await api.listTensors(modelId, { ...DEFAULT_QUERY, q: row.key.slice(0, MAX_SEARCH), limit: 200 });
      const t = page.items.find((x) => x.raw_key === row.key);
      if (!t) throw new ApiRequestError({ status: 404, code: "NOT_FOUND", message: `“${row.key}” is not in this model's catalog.` });
      onInspect(modelId, t.id);
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
        <button type="button" className="btn btn-small" disabled aria-describedby="td-delta-note">
          Delta heatmap (B − A)
        </button>
        <span id="td-delta-note" className="muted small">
          Not available yet: graphical delta heatmaps are planned (ROADMAP P1.1). Inspect opens one checkpoint’s own values.
        </span>
      </div>
      {error ? <ErrorPanel error={error} /> : null}
    </section>
  );
}
