// Per-launch access token (doc 09 header, doc 12 section 4, ADR-016).
// `weightlab start` prints http://127.0.0.1:PORT/#token=...; browsers never send a fragment to the server.
// The token is moved to sessionStorage (this tab only, gone when it closes) and removed from the address bar
// and history at once. It is never written to localStorage, cookies, query strings or logs.

export const TOKEN_KEY = "weightlab.session.token";
const TOKEN_SHAPE = /^[A-Za-z0-9_-]{32,256}$/;

function storage(win: Window): Storage | null {
  try {
    return win.sessionStorage;
  } catch {
    return null; // storage disabled: the token then lives only in memory for this page
  }
}

let memoryToken: string | null = null;

/** Reads `#token=` from the URL (a new launch replaces any stored token), strips the fragment from the
 * address bar and history, and returns the token for this tab, or null when there is none. */
export function initToken(win: Window = window): string | null {
  const hash = win.location.hash.startsWith("#") ? win.location.hash.slice(1) : win.location.hash;
  const params = new URLSearchParams(hash);
  const fromUrl = params.get("token");
  const store = storage(win);
  if (params.has("token")) {
    // Remove the fragment whatever its value, so a malformed token does not linger either.
    win.history.replaceState(win.history.state, "", win.location.pathname + win.location.search);
  }
  if (fromUrl !== null && TOKEN_SHAPE.test(fromUrl)) {
    memoryToken = fromUrl;
    try {
      store?.setItem(TOKEN_KEY, fromUrl);
    } catch {
      /* quota or privacy mode: keep the in-memory copy */
    }
    return fromUrl;
  }
  return getToken(win);
}

export function getToken(win: Window = window): string | null {
  try {
    const stored = storage(win)?.getItem(TOKEN_KEY) ?? null;
    if (stored !== null && TOKEN_SHAPE.test(stored)) return stored;
  } catch {
    /* fall through */
  }
  return memoryToken;
}

/** Forget the token, e.g. after the server rejected it (it was restarted with a new one). */
export function clearToken(win: Window = window): void {
  memoryToken = null;
  try {
    storage(win)?.removeItem(TOKEN_KEY);
  } catch {
    /* nothing stored */
  }
}
