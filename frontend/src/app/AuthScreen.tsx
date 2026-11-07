/** Shown when this tab holds no valid token: the only way in is the link printed by `weightlab start`. */
export function AuthScreen({ expired, onRetry }: { expired: boolean; onRetry: () => void }) {
  return (
    <section className="auth-screen" aria-labelledby="auth-h">
      <h1 id="auth-h">{expired ? "This WeightLab session has ended" : "Connect to WeightLab"}</h1>
      <p>
        {expired
          ? "The server no longer accepts this tab's access token, usually because WeightLab was restarted. Every launch creates a new token."
          : "This page needs the per-launch access token from the link that `weightlab start` printed in its terminal."}
      </p>
      <ol className="steps">
        <li>
          In the terminal, run <code>weightlab start</code> (or find the one already running).
        </li>
        <li>
          Open the link it prints, of the form <code>http://127.0.0.1:PORT/#token=…</code>. The token stays on this computer and
          is removed from the address bar once the page has it.
        </li>
      </ol>
      <button type="button" className="btn" onClick={onRetry}>
        I opened the new link — check again
      </button>
    </section>
  );
}
