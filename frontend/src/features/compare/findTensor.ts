// A report row names a tensor by its raw key; the API addresses tensors by id. The catalog search is a substring
// match, so the exact raw key picks the one tensor (keys are unique per model).
import { ApiRequestError } from "../../lib/api/client";
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
