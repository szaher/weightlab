import { useId, useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { KeyText } from "../../components/KeyText";
import { api } from "../../lib/api/endpoints";
import {
  DEFAULT_QUERY,
  MAX_SEARCH,
  PAGE_SIZES,
  pageInfo,
  parseCount,
  toggleSort,
  toSearchParams,
  updateQuery,
  type TensorQuery,
} from "../../lib/api/tensorQuery";
import { DTYPES, ROLES, type SortKey, type Tensor } from "../../lib/api/types";
import { formatBytes, formatInt, formatShape } from "../../lib/format";
import { useDebounced, useResource } from "../../lib/useResource";

export const SEARCH_DEBOUNCE_MS = 250;

interface Props {
  modelId: string;
  selectedId: string | null;
  onSelect: (t: Tensor) => void;
}

const SORT_LABEL: Record<SortKey, string> = {
  raw_key: "name",
  numel: "parameter count",
  byte_length: "byte length",
  layer_index: "layer",
};

/** Server-side searched, filtered, sorted and paged catalog: one bounded page is in the browser at a time. */
export function TensorCatalog({ modelId, selectedId, onSelect }: Props) {
  const id = useId();
  const [query, setQuery] = useState<TensorQuery>(DEFAULT_QUERY);
  const [text, setText] = useState("");
  // The search box updates at once; the request uses the debounced text. Typing resets to the first page.
  const debounced = useDebounced(text, SEARCH_DEBOUNCE_MS);
  const effective: TensorQuery = { ...query, q: debounced };
  const key = `${modelId}?${toSearchParams(effective).toString()}`;
  const page = useResource(key, (signal) => api.listTensors(modelId, effective, signal), true);
  const set = (patch: Partial<TensorQuery>) => setQuery((q) => updateQuery(q, patch));
  const filtersActive =
    text !== "" || query.dtype !== "" || query.role !== "" || query.layer !== "" || query.rank !== "" || query.minNumel !== "";

  const numberError = (raw: string, max: number) => (raw.trim() !== "" && parseCount(raw, max) === null ? "Whole number ≥ 0" : null);
  const layerErr = numberError(query.layer, 100000);
  const rankErr = numberError(query.rank, 32);
  const minErr = numberError(query.minNumel, Number.MAX_SAFE_INTEGER);

  const data = page.data;
  const info = data ? pageInfo(data.total, query.offset, query.limit, data.items.length) : null;

  function header(label: string, sortKey: SortKey | null, className?: string) {
    if (!sortKey) {
      return (
        <th scope="col" className={className}>
          {label}
        </th>
      );
    }
    const active = query.sort === sortKey;
    const ariaSort = active ? (query.order === "asc" ? "ascending" : "descending") : "none";
    return (
      <th scope="col" aria-sort={ariaSort} className={className}>
        <button type="button" className="sort-btn" onClick={() => setQuery((q) => toggleSort(q, sortKey))}>
          {label}
          <span aria-hidden="true" className="sort-ind">
            {active ? (query.order === "asc" ? " ▲" : " ▼") : " ↕"}
          </span>
        </button>
      </th>
    );
  }

  return (
    <section className="catalog" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`} className="visually-hidden">
        Tensor catalog
      </h2>
      <div className="filters" role="search" aria-label="Search and filter tensors">
        <div className="field field-grow">
          <label htmlFor={`${id}-q`} className="label">
            Search tensor names
          </label>
          <input
            id={`${id}-q`}
            type="search"
            className="input mono"
            value={text}
            maxLength={MAX_SEARCH}
            placeholder="e.g. layers.0 or up_proj"
            onChange={(e) => {
              setText(e.target.value);
              set({ q: e.target.value });
            }}
            spellCheck={false}
            autoComplete="off"
          />
        </div>
        <div className="field">
          <label htmlFor={`${id}-dtype`} className="label">
            Dtype
          </label>
          <select id={`${id}-dtype`} className="input" value={query.dtype} onChange={(e) => set({ dtype: e.target.value as TensorQuery["dtype"] })}>
            <option value="">Any</option>
            {DTYPES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={`${id}-role`} className="label">
            Role (heuristic)
          </label>
          <select id={`${id}-role`} className="input" value={query.role} onChange={(e) => set({ role: e.target.value as TensorQuery["role"] })}>
            <option value="">Any</option>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        {(
          [
            ["layer", "Layer", query.layer, layerErr],
            ["rank", "Rank", query.rank, rankErr],
            ["minNumel", "Min. params", query.minNumel, minErr],
          ] as const
        ).map(([k, label, value, err]) => (
          <div className="field field-num" key={k}>
            <label htmlFor={`${id}-${k}`} className="label">
              {label}
            </label>
            <input
              id={`${id}-${k}`}
              className="input mono"
              inputMode="numeric"
              value={value}
              onChange={(e) => set({ [k]: e.target.value })}
              aria-invalid={err ? true : undefined}
              aria-describedby={err ? `${id}-${k}-err` : undefined}
            />
            {err ? (
              <p id={`${id}-${k}-err`} className="field-error">
                {err}
              </p>
            ) : null}
          </div>
        ))}
        <div className="field field-end">
          <button
            type="button"
            className="btn btn-small"
            disabled={!filtersActive}
            onClick={() => {
              setText("");
              setQuery((q) => ({ ...DEFAULT_QUERY, sort: q.sort, order: q.order, limit: q.limit }));
            }}
          >
            Clear filters
          </button>
        </div>
      </div>

      <p className="catalog-status" role="status" aria-live="polite">
        {page.loading ? "Loading… " : ""}
        {info && data
          ? data.total === 0
            ? "No tensors match."
            : `Rows ${formatInt(info.first)}–${formatInt(info.last)} of ${formatInt(data.total)} ${filtersActive ? "matching " : ""}tensors, sorted by ${SORT_LABEL[query.sort]} (${query.order === "asc" ? "ascending" : "descending"}) across the whole catalog.`
          : ""}
      </p>

      {page.error ? <ErrorPanel error={page.error} onRetry={page.reload} /> : null}

      <div className="table-wrap">
        <table className="table" aria-busy={page.loading} aria-describedby={`${id}-role-note`}>
          <caption className="visually-hidden">Tensors, one row per tensor. Select a tensor name to inspect it.</caption>
          <thead>
            <tr>
              {header("Tensor", "raw_key", "col-key")}
              {header("Shape", null)}
              {header("Rank", null, "num")}
              {header("Dtype", null)}
              {header("Params", "numel", "num")}
              {header("Bytes", "byte_length", "num")}
              {header("Role*", null)}
              {header("Layer", "layer_index", "num")}
            </tr>
          </thead>
          <tbody>
            {data?.items.map((t) => (
              <tr key={t.id} className={t.id === selectedId ? "selected" : undefined} aria-selected={t.id === selectedId}>
                <td className="col-key">
                  <button type="button" className="key-btn" onClick={() => onSelect(t)} aria-current={t.id === selectedId ? "true" : undefined}>
                    <KeyText value={t.raw_key} />
                  </button>
                </td>
                <td className="mono">{formatShape(t.shape)}</td>
                <td className="num">{t.rank}</td>
                <td className="mono">{t.dtype}</td>
                <td className="num">{formatInt(t.numel)}</td>
                <td className="num" title={`${formatInt(t.byte_length)} bytes`}>
                  {formatBytes(t.byte_length)}
                </td>
                <td>{t.role ?? "—"}</td>
                <td className="num">{t.layer_index ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && data.total === 0 ? (
          <div className="empty">
            <p>No tensors match these filters.</p>
            {filtersActive ? <p className="muted">Clear or change the search and filters above.</p> : null}
          </div>
        ) : null}
      </div>
      <p id={`${id}-role-note`} className="help">
        * Roles and layer numbers are guesses from tensor names, not verified architecture facts.
      </p>

      <nav className="pager" aria-label="Catalog pages">
        <label className="label-inline">
          Rows per page{" "}
          <select className="input input-small" value={query.limit} onChange={(e) => set({ limit: Number(e.target.value) })}>
            {PAGE_SIZES.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <div className="row">
          <button type="button" className="btn btn-small" disabled={!info?.hasPrev} onClick={() => set({ offset: 0 })}>
            First
          </button>
          <button
            type="button"
            className="btn btn-small"
            disabled={!info?.hasPrev}
            onClick={() => set({ offset: Math.max(0, query.offset - query.limit) })}
          >
            Previous
          </button>
          <span className="muted">{info ? `Page ${formatInt(info.page)} of ${formatInt(info.pages)}` : ""}</span>
          <button type="button" className="btn btn-small" disabled={!info?.hasNext} onClick={() => set({ offset: query.offset + query.limit })}>
            Next
          </button>
          <button type="button" className="btn btn-small" disabled={!info?.hasNext} onClick={() => info && set({ offset: info.lastOffset })}>
            Last
          </button>
        </div>
      </nav>
    </section>
  );
}
