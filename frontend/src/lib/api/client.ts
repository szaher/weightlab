// Same-origin JSON client for /api/v1. Every protected request carries `Authorization: Bearer <token>`;
// no cookies, no credentials mode, no cross-origin URL can be built here (paths must start with "/").
import { getToken } from "../auth/token";
import type { ErrorEnvelope } from "./types";

export const API_PREFIX = "/api/v1";

/** Codes produced by the client itself, outside the contract's error enum. */
export type ClientCode = "NETWORK" | "BAD_RESPONSE" | "NO_TOKEN";

export class ApiRequestError extends Error {
  readonly status: number; // 0 when no HTTP response arrived
  readonly code: string;
  readonly retryable: boolean;
  readonly recommendedAction: string | undefined;
  readonly requestId: string | undefined;
  readonly details: Record<string, string | number | boolean | null>;

  constructor(init: {
    status: number;
    code: string;
    message: string;
    retryable?: boolean;
    recommendedAction?: string | undefined;
    requestId?: string | undefined;
    details?: Record<string, string | number | boolean | null> | undefined;
  }) {
    super(init.message);
    this.name = "ApiRequestError";
    this.status = init.status;
    this.code = init.code;
    this.retryable = init.retryable ?? false;
    this.recommendedAction = init.recommendedAction;
    this.requestId = init.requestId;
    this.details = init.details ?? {};
  }
}

function isEnvelope(v: unknown): v is ErrorEnvelope {
  if (typeof v !== "object" || v === null) return false;
  const e = (v as { error?: unknown }).error;
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as { code?: unknown }).code === "string" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

/** Builds an ApiRequestError from a non-2xx response, using the doc 09 section 7 envelope when present. */
export async function errorFromResponse(res: Response): Promise<ApiRequestError> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  const headerId = res.headers.get("x-request-id") ?? undefined;
  if (isEnvelope(body)) {
    const e = body.error;
    return new ApiRequestError({
      status: res.status,
      code: e.code,
      message: e.message,
      retryable: e.retryable === true,
      recommendedAction: typeof e.recommended_action === "string" ? e.recommended_action : undefined,
      requestId: typeof body.request_id === "string" ? body.request_id : headerId,
      details: e.details,
    });
  }
  return new ApiRequestError({
    status: res.status,
    code: "BAD_RESPONSE",
    message: `unexpected response (HTTP ${res.status})`,
    requestId: headerId,
  });
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  query?: URLSearchParams;
  signal?: AbortSignal | undefined;
  auth?: boolean; // default true; only /health is public
}

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;
let fetchImpl: FetchLike = (input, init) => fetch(input, init);
export function setFetch(f: FetchLike): void {
  fetchImpl = f;
}

/** Listeners told about 401 responses, so the app can show the reconnect screen. */
const unauthorizedListeners = new Set<() => void>();
export function onUnauthorized(fn: () => void): () => void {
  unauthorizedListeners.add(fn);
  return () => unauthorizedListeners.delete(fn);
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  if (!path.startsWith("/") || path.startsWith("//")) throw new Error("API paths are same-origin and absolute");
  const headers: Record<string, string> = { Accept: "application/json" };
  if (opts.auth !== false) {
    const token = getToken();
    if (token === null) {
      throw new ApiRequestError({ status: 401, code: "NO_TOKEN", message: "no access token in this tab" });
    }
    headers.Authorization = `Bearer ${token}`;
  }
  let body: string | undefined;
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  const qs = opts.query && [...opts.query.keys()].length > 0 ? `?${opts.query.toString()}` : "";
  const init: RequestInit = {
    method: opts.method ?? "GET",
    headers,
    credentials: "omit",
    cache: "no-store",
    referrerPolicy: "no-referrer",
    redirect: "error",
  };
  if (body !== undefined) init.body = body;
  if (opts.signal) init.signal = opts.signal;
  let res: Response;
  try {
    res = await fetchImpl(API_PREFIX + path + qs, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiRequestError({
      status: 0,
      code: "NETWORK",
      message: "the local WeightLab server did not answer",
      retryable: true,
    });
  }
  if (!res.ok) {
    const err = await errorFromResponse(res);
    if (res.status === 401) unauthorizedListeners.forEach((fn) => fn());
    throw err;
  }
  if (res.status === 204) return undefined as T;
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiRequestError({ status: res.status, code: "BAD_RESPONSE", message: "response was not JSON" });
  }
}

export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}
