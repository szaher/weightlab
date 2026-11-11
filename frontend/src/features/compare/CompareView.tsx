// The Compare section: the project's comparisons, a form to create one, and the dashboard of the selected one
// (compatibility first, then the analysis job, the layer x role grid, rankings and tensor details).
import { useMemo, useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { ApiRequestError } from "../../lib/api/client";
import { compareApi, TENSOR_METRICS, type Comparison, type Policy, type TensorMetric } from "../../lib/api/compare";
import { api } from "../../lib/api/endpoints";
import type { Model } from "../../lib/api/types";
import type { Resource } from "../../lib/useResource";
import { useResource } from "../../lib/useResource";
import { ChangeGrid } from "./ChangeGrid";
import { ClassBadge, ComparisonHeader } from "./ComparisonHeader";
import { buildGrid, METRIC_INFO } from "./metrics";
import { Rankings, type Filter } from "./Rankings";
import { REPORT_ADVICE, ReportJobPanel } from "./ReportJobPanel";
import { TensorDetail } from "./TensorDetail";
import { useReportJob } from "./useReportJob";

const SEL_KEY = "weightlab.compare.selected";
const readSel = (): string | null => {
  try {
    return sessionStorage.getItem(SEL_KEY);
  } catch {
    return null;
  }
};
const writeSel = (v: string) => {
  try {
    sessionStorage.setItem(SEL_KEY, v);
  } catch {
    /* per-tab convenience only */
  }
};

interface Props {
  projectId: string | null;
  projectName: string | null;
  models: Resource<Model[]>;
  onGoWorkspace: () => void;
  onInspect: (modelId: string, tensorId: string) => void;
}

export function CompareView({ projectId, projectName, models, onGoWorkspace, onInspect }: Props) {
  const [nonce, setNonce] = useState(0);
  const list = useResource(projectId ? `comparisons#${projectId}#${nonce}` : null, (s) => compareApi.list(projectId ?? "", s), true);
  const [chosen, setChosen] = useState<string | null>(readSel);
  // The list may still contain the previous response after creating a comparison.
  // Keep the just-created record selected until the refreshed list includes it.
  const [created, setCreated] = useState<Comparison | null>(null);
  const comparisons = list.data;
  const pendingCreated = created?.id === chosen && created.project_id === projectId ? created : null;
  const current = pendingCreated
    ? (comparisons?.find((c) => c.id === chosen) ?? pendingCreated)
    : (comparisons?.find((c) => c.id === chosen) ?? comparisons?.[0] ?? null);
  const modelList = models.data ?? [];
  const name = (id: string) => modelList.find((m) => m.id === id)?.display_name ?? id;

  if (!projectId) {
    return (
      <div className="empty">
        <h1 className="pane-title">No project selected</h1>
        <p className="muted">Choose or create a project in the Workspace, then import two checkpoints to compare.</p>
        <button type="button" className="btn btn-primary" onClick={onGoWorkspace}>
          Go to Workspace
        </button>
      </div>
    );
  }

  return (
    <div className="compare">
      <section className="pane pane-side stack-sm" aria-labelledby="cmp-list-h">
        <h2 id="cmp-list-h" className="pane-title">
          Comparisons{projectName ? ` · ${projectName}` : ""}
        </h2>
        {list.error ? <ErrorPanel error={list.error} onRetry={list.reload} /> : null}
        {comparisons === undefined && !list.error ? (
          <p className="muted" role="status">
            Loading comparisons…
          </p>
        ) : null}
        {comparisons?.length === 0 ? <p className="muted">No comparisons yet.</p> : null}
        {comparisons?.length ? (
          <ul className="list" aria-label="Comparisons">
            {comparisons.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  className="list-item"
                  aria-current={c.id === current?.id ? "true" : undefined}
                  onClick={() => {
                    setCreated(null);
                    setChosen(c.id);
                    writeSel(c.id);
                  }}
                >
                  <span className="list-item-title">
                    <ClassBadge cls={c.lineage.compatibility_class} /> {name(c.model_a_id)} → {name(c.model_b_id)}
                  </span>
                  <span className="list-item-sub">
                    {c.report ? (c.report.complete ? "report complete" : "report incomplete") : "no report yet"} ·{" "}
                    {new Date(c.created_at).toLocaleString()}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        <CreateComparison
          projectId={projectId}
          models={modelList}
          onGoWorkspace={onGoWorkspace}
          onCreated={(c) => {
            setCreated(c);
            setChosen(c.id);
            writeSel(c.id);
            setNonce((n) => n + 1);
          }}
        />
      </section>
      <section className="pane pane-main" aria-label="Comparison dashboard">
        {current ? (
          <Dashboard key={current.id} comparisonId={current.id} models={modelList} onListChanged={() => list.reload()} onInspect={onInspect} />
        ) : (
          <div className="empty">
            <h2 className="pane-title">Compare two checkpoints</h2>
            <p className="muted">
              Create a comparison of a base checkpoint (A) and a derived one (B), then run the analysis to see which layers, roles and
              tensors changed most (B − A).
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

function CreateComparison({ projectId, models, onCreated, onGoWorkspace }: { projectId: string; models: Model[]; onCreated: (c: Comparison) => void; onGoWorkspace: () => void }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [policy, setPolicy] = useState<Policy>("strict");
  const [attest, setAttest] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  if (models.length < 2) {
    return (
      <div className="card stack-sm">
        <p className="muted">Import at least two checkpoints in this project to compare them.</p>
        <button type="button" className="btn btn-small" onClick={onGoWorkspace}>
          Import models
        </button>
      </div>
    );
  }
  async function create() {
    setBusy(true);
    setError(null);
    try {
      const c = await compareApi.create({ project_id: projectId, model_a_id: a, model_b_id: b, policy, user_attestation: attest });
      onCreated(c);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <form
      className="card stack-sm"
      aria-labelledby="cmp-new-h"
      onSubmit={(e) => {
        e.preventDefault();
        void create();
      }}
    >
      <h3 id="cmp-new-h" className="h3">
        New comparison
      </h3>
      <div className="field">
        <label htmlFor="cmp-a" className="label">
          Base checkpoint (A)
        </label>
        <select id="cmp-a" className="input" value={a} onChange={(e) => setA(e.target.value)} required>
          <option value="">Choose…</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="cmp-b" className="label">
          Derived checkpoint (B)
        </label>
        <select id="cmp-b" className="input" value={b} onChange={(e) => setB(e.target.value)} required>
          <option value="">Choose…</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name}
            </option>
          ))}
        </select>
      </div>
      <fieldset className="stack-sm">
        <legend className="label">If lineage is unverified (Class S or X)</legend>
        <label className="radio">
          <input type="radio" name="cmp-policy" checked={policy === "strict"} onChange={() => setPolicy("strict")} />
          Strict: do not analyze
        </label>
        <label className="radio">
          <input type="radio" name="cmp-policy" checked={policy === "descriptive_only"} onChange={() => setPolicy("descriptive_only")} />
          Descriptive only: analyze, knowing coordinates may not be aligned
        </label>
      </fieldset>
      <label className="radio">
        <input type="checkbox" checked={attest} onChange={(e) => setAttest(e.target.checked)} />
        I attest that A is B’s training base (recorded with file hashes; never verified, at most Class L)
      </label>
      <button type="submit" className="btn btn-primary" disabled={busy || !a || !b}>
        {busy ? "Assessing…" : "Create comparison"}
      </button>
      {error ? <ErrorPanel error={error} /> : null}
      {error instanceof ApiRequestError && REPORT_ADVICE[error.code] ? <p className="notice-action">{REPORT_ADVICE[error.code]}</p> : null}
    </form>
  );
}

function Dashboard({ comparisonId, models, onListChanged, onInspect }: { comparisonId: string; models: Model[]; onListChanged: () => void; onInspect: (m: string, t: string) => void }) {
  const [tick, setTick] = useState(0);
  const comparison = useResource(`comparison#${comparisonId}#${tick}`, (s) => compareApi.get(comparisonId, s), true);
  const cmp = comparison.data;
  const reportJob = cmp?.report?.job_id ?? null;
  const report = useResource(reportJob ? `report#${comparisonId}#${reportJob}#${tick}` : null, (s) => compareApi.report(comparisonId, s), true);
  const settings = useResource("settings", (s) => api.settings(s));
  const rj = useReportJob(cmp, settings.data, () => {
    setTick((t) => t + 1);
    onListChanged();
  });
  const [metric, setMetric] = useState<TensorMetric>("relative_l2");
  const [log, setLog] = useState(true);
  const [filter, setFilter] = useState<Filter | null>(null);
  const [pick, setPick] = useState<{ report: string | null; key: string } | null>(null);
  // keepPrevious: while a newer report loads, the previous one stays on screen (never a blank flash).
  const r = reportJob ? report.data : undefined;
  const grid = useMemo(() => (r ? buildGrid(r.report.tensors, r.report.settings.epsilon_norm) : null), [r]);
  // A selection belongs to the report it was made in; a newer report starts with none.
  const selected = pick && pick.report === reportJob ? pick.key : null;
  const setSelected = (key: string | null) => setPick(key === null ? null : { report: reportJob, key });
  const row = r && selected ? (r.report.tensors.find((t) => t.key === selected) ?? null) : null;

  if (comparison.error && !cmp) return <ErrorPanel error={comparison.error} onRetry={comparison.reload} />;
  if (!cmp) {
    return (
      <p className="muted" role="status">
        Loading comparison…
      </p>
    );
  }
  const ma = models.find((m) => m.id === cmp.model_a_id);
  const mb = models.find((m) => m.id === cmp.model_b_id);
  return (
    <div className="dashboard stack">
      <ComparisonHeader comparison={cmp} modelA={ma} modelB={mb} report={r} />
      <ReportJobPanel rj={rj} hasReport={!!r} reportCreated={r?.created_at ?? null} />
      {reportJob && report.error && !r ? <ErrorPanel error={report.error} onRetry={report.reload} /> : null}
      {reportJob && !r && !report.error ? (
        <p className="muted" role="status">
          Loading report…
        </p>
      ) : null}
      {!reportJob && !rj.job ? (
        <p className="muted">No report yet. Run the analysis to compute it (the same numbers as <code>weightlab diff</code>).</p>
      ) : null}
      {r && grid ? (
        <>
          <section className="stack-sm" aria-labelledby="grid-h">
            <div className="pane-head">
              <h3 id="grid-h" className="h3">
                Change by layer and role
              </h3>
              <div className="row cmp-controls">
                <div className="field">
                  <label htmlFor="cmp-metric" className="label">
                    Metric
                  </label>
                  <select id="cmp-metric" className="input input-small" value={metric} onChange={(e) => setMetric(e.target.value as TensorMetric)}>
                    {TENSOR_METRICS.map((k) => (
                      <option key={k} value={k}>
                        {METRIC_INFO[k].label}
                      </option>
                    ))}
                  </select>
                </div>
                <label className="radio">
                  <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> Log colour scale
                </label>
              </div>
            </div>
            <p className="muted small">
              {METRIC_INFO[metric].definition} Each cell aggregates the tensors of exactly that layer and role (for example a projection’s
              weight and bias) as if concatenated; select a cell to filter the rankings.
            </p>
            <ChangeGrid grid={grid} metric={metric} log={log} selected={filter} onSelect={setFilter} />
          </section>
          {row ? (
            <TensorDetail
              row={row}
              modelA={ma ? { id: ma.id, name: ma.display_name } : null}
              modelB={mb ? { id: mb.id, name: mb.display_name } : null}
              onInspect={onInspect}
              onClose={() => setSelected(null)}
            />
          ) : null}
          <Rankings report={r.report} metric={metric} filter={filter} onFilter={setFilter} selectedKey={selected} onSelectTensor={setSelected} />
          {r.report.compatibility.not_compared.length ? (
            <details className="caveats">
              <summary>Entries not compared ({r.report.compatibility.not_compared.length})</summary>
              <div className="table-wrap">
                <table className="table table-compact">
                  <thead>
                    <tr>
                      <th scope="col">Tensor</th>
                      <th scope="col">Status</th>
                      <th scope="col">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.report.compatibility.not_compared.map((e) => (
                      <tr key={e.raw_key}>
                        <td className="mono">{e.raw_key}</td>
                        <td>{e.status.replace(/_/g, " ")}</td>
                        <td>{e.tied_to ? `tied to ${e.tied_to}` : e.reasons.join("; ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
