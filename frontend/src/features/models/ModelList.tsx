import { useState } from "react";
import { StatusBadge } from "../../components/StatusBadge";
import type { Model } from "../../lib/api/types";
import { formatInt } from "../../lib/format";
import { formatLabel, Warnings } from "./ModelSummary";
import { RelinkForm } from "./RelinkForm";

interface Props {
  models: Model[];
  selectedId: string | null;
  onExplore: (id: string) => void;
  onRelinked: (m: Model) => void;
}

export function ModelList({ models, selectedId, onExplore, onRelinked }: Props) {
  const [relinking, setRelinking] = useState<string | null>(null);
  if (models.length === 0) {
    return (
      <p className="muted" role="status">
        No models in this project yet. Import a checkpoint below; WeightLab reads it where it is.
      </p>
    );
  }
  return (
    <ul className="model-list" aria-label="Models in this project">
      {models.map((m) => (
        <li key={m.id} className="model-item" aria-current={m.id === selectedId ? "true" : undefined}>
          <div className="model-head">
            <span className="model-name">{m.display_name}</span>
            <StatusBadge status={m.status} />
          </div>
          <p className="list-item-sub">
            {formatLabel(m.format)} · {formatInt(m.tensor_count)} tensors
          </p>
          {m.status === "changed" ? (
            <p className="help">Files changed since import: import the model again to analyze the new contents.</p>
          ) : null}
          {m.status === "disconnected" ? (
            <p className="help">Files not found at the recorded location. If you moved them, relink.</p>
          ) : null}
          <Warnings warnings={m.warnings} />
          <div className="row">
            <button type="button" className="btn btn-small btn-primary" onClick={() => onExplore(m.id)}>
              Explore tensors
            </button>
            <button
              type="button"
              className="btn btn-small"
              aria-expanded={relinking === m.id}
              onClick={() => setRelinking(relinking === m.id ? null : m.id)}
            >
              Relink…
            </button>
          </div>
          {relinking === m.id ? (
            <RelinkForm
              model={m}
              onCancel={() => setRelinking(null)}
              onRelinked={(r) => {
                setRelinking(null);
                onRelinked(r);
              }}
            />
          ) : null}
        </li>
      ))}
    </ul>
  );
}
