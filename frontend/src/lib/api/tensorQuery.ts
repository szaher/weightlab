// Catalog query state <-> GET /models/{id}/tensors parameters (doc 09 section 1). Filtering, sorting and
// paging all happen on the server, so a sorted page is a slice of the globally sorted catalog.
import type { Dtype, Role, SortKey, SortOrder } from "./types";

export const PAGE_SIZES = [25, 50, 100, 200] as const;
export const MAX_SEARCH = 256; // contract maxLength of q

export interface TensorQuery {
  q: string;
  dtype: Dtype | "";
  role: Role | "";
  layer: string; // raw input; only a non-negative integer is sent
  rank: string;
  minNumel: string;
  sort: SortKey;
  order: SortOrder;
  limit: number;
  offset: number;
}

export const DEFAULT_QUERY: TensorQuery = {
  q: "",
  dtype: "",
  role: "",
  layer: "",
  rank: "",
  minNumel: "",
  sort: "raw_key",
  order: "asc",
  limit: 50,
  offset: 0,
};

/** A non-negative safe integer typed by the user, or null (empty or invalid input is not sent). */
export function parseCount(raw: string, max = Number.MAX_SAFE_INTEGER): number | null {
  const t = raw.trim();
  if (!/^\d{1,16}$/.test(t)) return null;
  const n = Number(t);
  return Number.isSafeInteger(n) && n <= max ? n : null;
}

export function toSearchParams(q: TensorQuery): URLSearchParams {
  const p = new URLSearchParams();
  const text = q.q.trim().slice(0, MAX_SEARCH);
  if (text) p.set("q", text);
  if (q.dtype) p.append("dtype", q.dtype);
  if (q.role) p.append("role", q.role);
  const layer = parseCount(q.layer, 100000);
  if (layer !== null) p.set("layer", String(layer));
  const rank = parseCount(q.rank, 32);
  if (rank !== null) p.set("rank", String(rank));
  const minNumel = parseCount(q.minNumel);
  if (minNumel !== null) p.set("min_numel", String(minNumel));
  p.set("sort", q.sort);
  p.set("order", q.order);
  p.set("limit", String(q.limit));
  p.set("offset", String(q.offset));
  return p;
}

/** Any change other than paging starts again at the first page. */
export function updateQuery(prev: TensorQuery, patch: Partial<TensorQuery>): TensorQuery {
  const paging = Object.keys(patch).every((k) => k === "offset");
  const next = { ...prev, ...patch };
  if (!paging) next.offset = 0;
  return next;
}

/** Clicking a column header: a new column sorts ascending, the same column flips the order. */
export function toggleSort(prev: TensorQuery, key: SortKey): TensorQuery {
  return updateQuery(prev, { sort: key, order: prev.sort === key && prev.order === "asc" ? "desc" : "asc" });
}

export interface PageInfo {
  first: number; // 1-based index of the first row shown, 0 when empty
  last: number;
  page: number; // 1-based
  pages: number;
  hasPrev: boolean;
  hasNext: boolean;
  lastOffset: number;
}

export function pageInfo(total: number, offset: number, limit: number, shown: number): PageInfo {
  const pages = Math.max(1, Math.ceil(total / limit));
  const page = Math.floor(offset / limit) + 1;
  return {
    first: shown === 0 ? 0 : offset + 1,
    last: offset + shown,
    page,
    pages,
    hasPrev: offset > 0,
    hasNext: offset + shown < total,
    lastOffset: Math.max(0, (pages - 1) * limit),
  };
}
