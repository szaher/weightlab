// Sortable rankings of tensors, layers and roles. Layer and role rows are the report's own rollups (computed by the
// backend with exact scaled sums), never re-derived here. Each table is the accessible alternative of its bars;
// unavailable values are labelled and always sort last.
import { useMemo, useState, type ReactNode } from "react";
import { TENSOR_METRICS, type ChangeReport, type RollupRow, type TensorMetric, type TensorRow } from "../../lib/api/compare";
import { formatInt } from "../../lib/format";

/** Tensor filter from the grid (one cell) or a layer / role ranking row (one axis). */
export interface Filter {
  layer: number | null | "any";
  role: string | "any";
}
import { changeOrder, METRIC_INFO, metricVal, rollupVal, valText, valTitle, type Val } from "./metrics";

type Tab = "tensors" | "layers" | "roles";
const PAGE = 100;

interface Props {
  report: ChangeReport;
  metric: TensorMetric;
  filter: Filter | null;
  onFilter: (f: Filter | null) => void;
  selectedKey: string | null;
  onSelectTensor: (key: string) => void;
}

const filterText = (f: Filter) =>
  [f.layer === "any" ? null : f.layer === null ? "no layer" : `layer ${f.layer}`, f.role === "any" ? null : f.role].filter(Boolean).join(" × ");

function Bar({ v, metric, max }: { v: Val; metric: TensorMetric; max: number }) {
  if (v.kind !== "value" || !(max > 0)) return <span className="bar bar-empty" aria-hidden="true" />;
  const x = metric === "cosine" ? 1 - v.v : v.v;
  return (
    <span className="bar" aria-hidden="true">
      <span style={{ width: `${Math.max(0, Math.min(100, (x / max) * 100))}%` }} />
    </span>
  );
}

function ValCell({ v }: { v: Val }) {
  return (
    <td className={`num mono${v.kind === "value" ? "" : " val-na"}`} title={valTitle(v)}>
      {valText(v)}
    </td>
  );
}

function SortTh({ id, label, sort, onSort, title }: { id: string; label: ReactNode; sort: [string, boolean]; onSort: (id: string) => void; title?: string }) {
  const active = sort[0] === id;
  return (
    <th scope="col" aria-sort={active ? (sort[1] ? "descending" : "ascending") : "none"} title={title}>
      <button type="button" className="sort-btn" onClick={() => onSort(id)}>
        {label}
        <span className="sort-ind" aria-hidden="true">
          {active ? (sort[1] ? "▼" : "▲") : ""}
        </span>
      </button>
    </th>
  );
}

function maxOf(vals: Val[], metric: TensorMetric): number {
  let m = 0;
  for (const v of vals) if (v.kind === "value") m = Math.max(m, metric === "cosine" ? 1 - v.v : v.v);
  return m;
}

export function Rankings({ report, metric, filter, onFilter, selectedKey, onSelectTensor }: Props) {
  const [tab, setTab] = useState<Tab>("tensors");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState<[string, boolean]>(["metric", true]); // [column, descending = most change first]
  const onSort = (id: string) => setSort((s) => (s[0] === id ? [id, !s[1]] : [id, true]));
  const info = METRIC_INFO[metric];

  const tensors = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const rows = report.tensors.filter(
      (t) =>
        (!filter || ((filter.layer === "any" || t.layer === filter.layer) && (filter.role === "any" || t.role === filter.role))) &&
        (!needle || t.key.toLowerCase().includes(needle)),
    );
    return sortRows(rows, sort, (t, k) => metricVal(t.metrics[k as TensorMetric]), (t) => t.key, metric);
  }, [report, q, filter, sort, metric]);
  const layers = useMemo(() => sortRows(report.layers, sort, (r, k) => rollupVal(r, k as TensorMetric), (r) => String(r.layer ?? "~"), metric), [report, sort, metric]);
  const roles = useMemo(() => sortRows(report.roles, sort, (r, k) => rollupVal(r, k as TensorMetric), (r) => r.role ?? "", metric), [report, sort, metric]);

  const tabs: [Tab, string][] = [
    ["tensors", `Tensors (${formatInt(report.tensors.length)})`],
    ["layers", `Layers (${report.layers.length})`],
    ["roles", `Roles (${report.roles.length})`],
  ];
  const pages = Math.max(1, Math.ceil(tensors.length / PAGE));
  const p = Math.min(page, pages - 1);
  const shown = tensors.slice(p * PAGE, (p + 1) * PAGE);
  const tMax = maxOf(tensors.map((t) => metricVal(t.metrics[metric])), metric);
  const metricCols = (exclude: TensorMetric[] = []) =>
    TENSOR_METRICS.filter((k) => !exclude.includes(k)).map((k) => (
      <SortTh key={k} id={k} label={METRIC_INFO[k].short} sort={sort} onSort={onSort} title={`${METRIC_INFO[k].label}: ${METRIC_INFO[k].definition}`} />
    ));

  return (
    <section className="rankings stack-sm" aria-labelledby="rank-h">
      <h3 id="rank-h" className="h3">
        Rankings by {info.label}
      </h3>
      <div role="tablist" aria-label="Ranking level" className="tabs">
        {tabs.map(([t, l]) => (
          <button key={t} type="button" role="tab" id={`rk-tab-${t}`} aria-selected={tab === t} aria-controls={`rk-panel-${t}`} tabIndex={tab === t ? 0 : -1} className="tab" onClick={() => setTab(t)}>
            {l}
          </button>
        ))}
      </div>

      {tab === "tensors" ? (
        <div role="tabpanel" id="rk-panel-tensors" aria-labelledby="rk-tab-tensors" className="stack-sm">
          <div className="row">
            <div className="field field-grow">
              <label htmlFor="rk-q" className="label">
                Filter tensor names
              </label>
              <input id="rk-q" className="input" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} placeholder="e.g. q_proj or layers.3." />
            </div>
            {filter ? (
              <p className="chip" role="status">
                Showing <span className="mono">{filterText(filter)}</span>
                <button type="button" className="btn btn-small" onClick={() => onFilter(null)} aria-label={`Clear filter ${filterText(filter)}`}>
                  ✕
                </button>
              </p>
            ) : null}
          </div>
          <p className="muted small" aria-live="polite">
            {formatInt(tensors.length)} tensor{tensors.length === 1 ? "" : "s"}
            {pages > 1 ? ` · page ${p + 1} of ${pages}` : ""}. Select a row for details.
          </p>
          <div className="table-wrap">
            <table className="table table-compact rank-table">
              <thead>
                <tr>
                  <SortTh id="metric" label="#" sort={sort} onSort={onSort} title={`Rank by ${info.label}`} />
                  <SortTh id="key" label="Tensor" sort={sort} onSort={onSort} />
                  <th scope="col">Layer</th>
                  <th scope="col">Role</th>
                  <th scope="col" title={`Bar: ${info.label}, relative to the largest shown`}>Change</th>
                  {metricCols()}
                </tr>
              </thead>
              <tbody>
                {shown.map((t, i) => (
                  <tr key={t.key} className={t.key === selectedKey ? "selected" : undefined}>
                    <td className="num">{p * PAGE + i + 1}</td>
                    <td className="mono key-cell">
                      <button type="button" className="link-btn" aria-pressed={t.key === selectedKey} onClick={() => onSelectTensor(t.key)}>
                        {t.key}
                      </button>
                    </td>
                    <td className="num">{t.layer ?? "—"}</td>
                    <td className="mono">{t.role}</td>
                    <td>
                      <Bar v={metricVal(t.metrics[metric])} metric={metric} max={tMax} />
                    </td>
                    {TENSOR_METRICS.map((k) => (
                      <ValCell key={k} v={metricVal(t.metrics[k])} />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {pages > 1 ? (
            <div className="pager">
              <button type="button" className="btn btn-small" disabled={p === 0} onClick={() => setPage(p - 1)}>
                Previous
              </button>
              <button type="button" className="btn btn-small" disabled={p >= pages - 1} onClick={() => setPage(p + 1)}>
                Next
              </button>
            </div>
          ) : null}
        </div>
      ) : (
        <RollupTable
          kind={tab}
          rows={tab === "layers" ? layers : roles}
          metric={metric}
          sort={sort}
          onSort={onSort}
          metricCols={metricCols(["cosine"])}
          onPick={(r) => {
            onFilter(tab === "layers" ? { layer: r.layer ?? null, role: "any" } : { layer: "any", role: r.role ?? "" });
            setTab("tensors");
            setPage(0);
          }}
        />
      )}
      <details className="caveats">
        <summary>Metric definitions</summary>
        <dl className="facts facts-1col">
          {TENSOR_METRICS.map((k) => (
            <div key={k}>
              <dt>{METRIC_INFO[k].label}</dt>
              <dd>{METRIC_INFO[k].definition}</dd>
            </div>
          ))}
          <div>
            <dt>Layer and role rows</dt>
            <dd>
              Rollups over all of their tensors as if concatenated (the CLI's layers.csv and roles.csv), not averages of tensor values.
              Cosine is not defined for rollups.
            </dd>
          </div>
        </dl>
      </details>
    </section>
  );
}

function RollupTable({ kind, rows, metric, sort, onSort, metricCols, onPick }: {
  kind: "layers" | "roles";
  rows: RollupRow[];
  metric: TensorMetric;
  sort: [string, boolean];
  onSort: (id: string) => void;
  metricCols: ReactNode;
  onPick: (r: RollupRow) => void;
}) {
  const max = maxOf(rows.map((r) => rollupVal(r, metric)), metric);
  return (
    <div role="tabpanel" id={`rk-panel-${kind}`} aria-labelledby={`rk-tab-${kind}`} className="stack-sm">
      {metric === "cosine" ? <p className="notice notice-info">Cosine is not defined for rollups; choose another metric to rank {kind}.</p> : null}
      <div className="table-wrap">
        <table className="table table-compact rank-table">
          <thead>
            <tr>
              <SortTh id="key" label={kind === "layers" ? "Layer" : "Role"} sort={sort} onSort={onSort} />
              <th scope="col">Tensors</th>
              <th scope="col">Withheld</th>
              <th scope="col" title={`Bar: ${METRIC_INFO[metric].label}, relative to the largest shown`}>Change</th>
              {metricCols}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const name = kind === "layers" ? (r.layer === null || r.layer === undefined ? "No layer" : `Layer ${r.layer}`) : (r.role ?? "");
              return (
                <tr key={name}>
                  <td className="mono">
                    <button type="button" className="link-btn" onClick={() => onPick(r)} title={`Show the tensors of ${name}`}>
                      {name}
                    </button>
                  </td>
                  <td className="num">{formatInt(r.tensors)}</td>
                  <td className="num">{r.withheld_tensors ? formatInt(r.withheld_tensors) : "—"}</td>
                  <td>
                    <Bar v={rollupVal(r, metric)} metric={metric} max={max} />
                  </td>
                  {TENSOR_METRICS.filter((k) => k !== "cosine").map((k) => (
                    <ValCell key={k} v={rollupVal(r, k)} />
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function sortRows<T>(rows: readonly T[], sort: [string, boolean], val: (r: T, k: string) => Val, name: (r: T) => string, metric: TensorMetric): T[] {
  const [col, desc] = sort;
  const out = [...rows];
  if (col === "key") {
    out.sort((a, b) => name(a).localeCompare(name(b), "en", { numeric: true }) * (desc ? -1 : 1));
    return out;
  }
  const k = col === "metric" ? metric : col;
  out.sort((a, b) => {
    const va = val(a, k);
    const vb = val(b, k);
    // Unavailable values last in either direction; ties by name for a stable, deterministic order.
    if (va.kind !== "value" || vb.kind !== "value") {
      if (va.kind === vb.kind) return name(a).localeCompare(name(b));
      return va.kind === "value" ? -1 : 1;
    }
    const d = changeOrder(vb, k as TensorMetric) - changeOrder(va, k as TensorMetric);
    if (d !== 0) return desc ? d : -d;
    return name(a).localeCompare(name(b));
  });
  return out;
}

export type { TensorRow };
