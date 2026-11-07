import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { Health } from "../../lib/api/types";
import type { Theme } from "../../lib/prefs";
import { useResource } from "../../lib/useResource";

const DESCRIBE: Record<string, string> = {
  offline: "Offline mode (always on in this release)",
  telemetry_enabled: "Telemetry (always off in this release)",
  memory_budget_mib: "Backend memory budget (MiB)",
  max_threads: "Worker threads",
  max_active_tiles: "Concurrent heatmap tiles",
  max_heavy_jobs: "Concurrent heavy jobs",
  roi_max_cells: "Max cells per exact ROI read",
  max_tile_cells: "Max cells per tile (fixed)",
  direct_tile_max_mib: "Direct tile read limit (MiB)",
  pyramid_exact_max_mib: "Exact pyramid limit (MiB)",
  tile_cache_memory_mib: "Tile cache, memory (MiB)",
  tile_cache_disk_mib: "Tile cache, disk (MiB)",
  max_scan_read_mib_without_confirmation: "Scan size needing confirmation (MiB)",
  max_job_seconds: "Job time limit (s)",
  max_lora_scan_gmac: "LoRA scan limit (GMAC)",
  export_absolute_paths: "Include absolute paths in exports",
};

export function SettingsView({ theme, onTheme, health }: { theme: Theme; onTheme: (t: Theme) => void; health: Health | undefined }) {
  const settings = useResource("settings", (signal) => api.settings(signal));
  return (
    <div className="settings stack">
      <h1 className="pane-title">Settings</h1>
      <section className="card stack-sm" aria-labelledby="appearance-h">
        <h2 id="appearance-h" className="card-title">
          Appearance
        </h2>
        <fieldset className="radio-row">
          <legend className="label">Theme (saved in this browser only)</legend>
          {(["system", "light", "dark"] as const).map((t) => (
            <label key={t} className="radio">
              <input type="radio" name="theme" value={t} checked={theme === t} onChange={() => onTheme(t)} />
              {t === "system" ? "Follow system" : t === "light" ? "Light" : "Dark"}
            </label>
          ))}
        </fieldset>
      </section>
      <section className="card stack-sm" aria-labelledby="runtime-h">
        <h2 id="runtime-h" className="card-title">
          Local runtime
        </h2>
        <dl className="facts">
          <div>
            <dt>Version</dt>
            <dd className="mono">{health?.version ?? "—"}</dd>
          </div>
          <div>
            <dt>Network</dt>
            <dd>{health?.offline ? "Offline: no network requests are made" : "—"}</dd>
          </div>
          <div>
            <dt>Server status</dt>
            <dd>{health?.status ?? "unknown"}</dd>
          </div>
        </dl>
      </section>
      <section className="card stack-sm" aria-labelledby="limits-h">
        <h2 id="limits-h" className="card-title">
          Resource and privacy settings
        </h2>
        <p className="help">
          Current values stored by the backend. <code>roi_max_cells</code> bounds every exact-value window in Explore; the others
          govern tiled heatmaps, jobs and exports, which arrive in later milestones. Editing them here is not available yet.
        </p>
        {settings.error ? <ErrorPanel error={settings.error} onRetry={settings.reload} /> : null}
        {settings.data ? (
          <table className="table table-compact">
            <caption className="visually-hidden">Backend settings</caption>
            <thead>
              <tr>
                <th scope="col">Setting</th>
                <th scope="col" className="num">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(settings.data).map(([k, v]) => (
                <tr key={k}>
                  <th scope="row">
                    {DESCRIBE[k] ?? k} <code className="muted">{k}</code>
                  </th>
                  <td className="num mono">{typeof v === "boolean" ? (v ? "on" : "off") : v.toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : !settings.error ? (
          <p className="muted" role="status">
            Loading…
          </p>
        ) : null}
      </section>
    </div>
  );
}
