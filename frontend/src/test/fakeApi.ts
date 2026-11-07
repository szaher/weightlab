import { setFetch } from "../lib/api/client";
import { TOKEN_KEY } from "../lib/auth/token";

export const TEST_TOKEN = "t".repeat(43);

export interface Call {
  method: string;
  url: URL;
  headers: Record<string, string>;
  body: unknown;
  init: RequestInit;
}

type Handler = (call: Call) => Response | Promise<Response>;

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

export function apiError(status: number, code: string, message: string, extra: Record<string, unknown> = {}): Response {
  return json({ error: { code, message, retryable: code === "RATE_LIMITED", ...extra }, request_id: "req_0123456789abcdef" }, status);
}

/** Routes requests to `handler` and records them. Unmatched requests fail the test loudly. */
export function fakeApi(handler: Handler): Call[] {
  const calls: Call[] = [];
  setFetch(async (input, init) => {
    const url = new URL(input, "http://127.0.0.1:8765");
    const headers = { ...(init.headers as Record<string, string>) };
    const call: Call = {
      method: init.method ?? "GET",
      url,
      headers,
      body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
      init,
    };
    calls.push(call);
    if (init.signal?.aborted) throw new DOMException("aborted", "AbortError");
    return handler(call);
  });
  return calls;
}

export function withToken(): void {
  sessionStorage.setItem(TOKEN_KEY, TEST_TOKEN);
}
