import type { Health } from "../lib/api/types";

export type Connection = "checking" | "connected" | "unavailable" | "unauthorized";

interface Props {
  projectName: string | null;
  modelName: string | null;
  connection: Connection;
  health: Health | undefined;
  onSettings: () => void;
}

const CONN: Record<Connection, { icon: string; text: string }> = {
  checking: { icon: "…", text: "Connecting" },
  connected: { icon: "●", text: "Connected" },
  unavailable: { icon: "✕", text: "Server unavailable" },
  unauthorized: { icon: "!", text: "Not authorized" },
};

export function TopBar({ projectName, modelName, connection, health, onSettings }: Props) {
  const c = CONN[connection];
  return (
    <header className="topbar">
      <span className="brand">
        <span aria-hidden="true" className="brand-mark">
          ▮▮▮
        </span>{" "}
        WeightLab
      </span>
      <dl className="topbar-ctx">
        <div>
          <dt>Project</dt>
          <dd>{projectName ?? "none"}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{modelName ?? "none"}</dd>
        </div>
      </dl>
      <div className="topbar-right">
        <span className="pill" title="All data stays on this computer; WeightLab makes no network requests">
          LOCAL{health?.offline ? " · OFFLINE" : ""}
        </span>
        <span className={`pill conn conn-${connection}`} role="status" aria-label={`Backend: ${c.text}`}>
          <span aria-hidden="true">{c.icon}</span> {c.text}
        </span>
        <button type="button" className="btn btn-small" onClick={onSettings}>
          Settings
        </button>
      </div>
    </header>
  );
}
