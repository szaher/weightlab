import { useState } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import type { Model, Tensor } from "../../lib/api/types";
import type { Resource } from "../../lib/useResource";
import { ModelFacts, ModelStatusNotice, Warnings } from "../models/ModelSummary";
import { TensorCatalog } from "./TensorCatalog";
import { TensorInspector } from "./TensorInspector";

interface Props {
  projectName: string | null;
  models: Resource<Model[]>;
  modelId: string | null;
  onSelectModel: (id: string) => void;
  onGoWorkspace: () => void;
}

export function ExploreView({ projectName, models, modelId, onSelectModel, onGoWorkspace }: Props) {
  const [tensor, setTensor] = useState<{ model: string; id: string } | null>(null);
  if (!projectName) {
    return (
      <div className="empty">
        <h1 className="pane-title">No project selected</h1>
        <p className="muted">Choose or create a project in the Workspace, then import a checkpoint.</p>
        <button type="button" className="btn btn-primary" onClick={onGoWorkspace}>
          Go to Workspace
        </button>
      </div>
    );
  }
  if (models.error) return <ErrorPanel error={models.error} onRetry={models.reload} />;
  if (!models.data) {
    return (
      <p className="muted" role="status">
        Loading models…
      </p>
    );
  }
  const list = models.data;
  const model = list.find((m) => m.id === modelId) ?? null;
  if (list.length === 0) {
    return (
      <div className="empty">
        <h1 className="pane-title">No models in “{projectName}”</h1>
        <p className="muted">Import a SafeTensors checkpoint in the Workspace to browse its tensors.</p>
        <button type="button" className="btn btn-primary" onClick={onGoWorkspace}>
          Import a model
        </button>
      </div>
    );
  }
  const selectedTensor = model && tensor?.model === model.id ? tensor.id : null;
  return (
    <div className="explore">
      <div className="explore-head">
        <div className="field">
          <label htmlFor="explore-model" className="label">
            Model
          </label>
          <select id="explore-model" className="input" value={model?.id ?? ""} onChange={(e) => onSelectModel(e.target.value)}>
            {model ? null : <option value="">Choose a model…</option>}
            {list.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name}
              </option>
            ))}
          </select>
        </div>
        {model ? <ModelFacts model={model} /> : null}
      </div>
      {model ? (
        <>
          <ModelStatusNotice model={model} />
          <Warnings warnings={model.warnings} />
          <div className={selectedTensor ? "explore-body has-tensor" : "explore-body"}>
            <TensorCatalog key={model.id} modelId={model.id} selectedId={selectedTensor} onSelect={(t: Tensor) => setTensor({ model: model.id, id: t.id })} />
            <TensorInspector modelId={model.id} modelName={model.display_name} tensorId={selectedTensor} />
          </div>
        </>
      ) : (
        <p className="muted">Choose a model to browse its tensors.</p>
      )}
    </div>
  );
}
