// A report row names a tensor by its raw key; the API addresses tensors by id. The catalog search is a substring
// match, so the exact raw key picks the one tensor (keys are unique per model).
import { ApiRequestError } from "../../lib/api/client";
import { compareApi, DELTA_STATUSES, type ComparisonEntry } from "../../lib/api/compare";
import { api } from "../../lib/api/endpoints";
import { DEFAULT_QUERY, MAX_SEARCH } from "../../lib/api/tensorQuery";
import type { Tensor } from "../../lib/api/types";

export async function findTensor(modelId: string, rawKey: string, signal?: AbortSignal): Promise<Tensor> {
  const query = rawKey.slice(0, MAX_SEARCH);
  let offset = 0;
  for (;;) {
    const page = await api.listTensors(modelId, { ...DEFAULT_QUERY, q: query, limit: 200, offset }, signal);
    const t = page.items.find((x) => x.raw_key === rawKey);
    if (t) return t;
    offset += page.items.length;
    if (!page.items.length || offset >= page.total) break;
  }
  throw new ApiRequestError({ status: 404, code: "NOT_FOUND", message: `“${rawKey}” is not in this model's catalog.` });
}

/** The comparison entry of `rawKey` among the entries that have an elementwise B − A, paged in storage order (the
 *  entries route has no key filter). NOT_FOUND when the key has no such entry. */
export async function findEntry(comparisonId: string, rawKey: string, signal?: AbortSignal): Promise<ComparisonEntry> {
  let offset = 0;
  for (;;) {
    const page = await compareApi.entries(comparisonId, { status: DELTA_STATUSES, limit: ENTRY_PAGE, offset }, signal);
    const e = page.items.find((x) => x.raw_key === rawKey);
    if (e) return e;
    offset += page.items.length;
    if (!page.items.length || offset >= page.total) break;
  }
  throw new ApiRequestError({ status: 404, code: "NOT_FOUND", message: `“${rawKey}” has no matched entry with an elementwise B − A in this comparison.` });
}

export const ENTRY_PAGE = 1000;
