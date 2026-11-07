import { useState } from "react";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { Model, Project } from "../../lib/api/types";
import type { Resource } from "../../lib/useResource";
import { ImportForm } from "../models/ImportForm";
import { ModelList } from "../models/ModelList";
import { CreateProjectForm } from "../projects/CreateProjectForm";
import { ProjectList } from "../projects/ProjectList";
import { Welcome } from "./Welcome";

interface Props {
  projects: Resource<Project[]>;
  models: Resource<Model[]>;
  projectId: string | null;
  modelId: string | null;
  onSelectProject: (id: string | null) => void;
  onExplore: (modelId: string) => void;
}

export function WorkspaceView({ projects, models, projectId, modelId, onSelectProject, onExplore }: Props) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<unknown>(null);
  const [lastImport, setLastImport] = useState<Model | null>(null);
  const list = projects.data;
  const project = list?.find((p) => p.id === projectId) ?? null;

  if (list === undefined) {
    if (projects.error) return <ErrorPanel error={projects.error} onRetry={projects.reload} />;
    return (
      <p className="muted" role="status">
        Loading projects…
      </p>
    );
  }

  const created = (p: Project) => {
    projects.reload();
    onSelectProject(p.id);
  };

  if (list.length === 0) return <Welcome onCreated={created} />;

  async function doDelete() {
    if (!project) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteProject(project.id);
      setConfirming(false);
      onSelectProject(null);
      projects.reload();
    } catch (e) {
      setDeleteError(e);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="workspace">
      <section className="pane pane-side" aria-labelledby="projects-h">
        <h2 id="projects-h" className="pane-title">
          Projects
        </h2>
        <ProjectList projects={list} selectedId={projectId} onSelect={(id) => onSelectProject(id)} />
        <CreateProjectForm onCreated={created} />
      </section>
      <section className="pane pane-main" aria-labelledby="project-h">
        {project ? (
          <div className="stack">
            <div className="pane-head">
              <h2 id="project-h" className="pane-title">
                {project.name}
              </h2>
              <button type="button" className="btn btn-small btn-danger-outline" onClick={() => setConfirming(true)}>
                Delete project…
              </button>
            </div>
            {models.error ? <ErrorPanel error={models.error} onRetry={models.reload} /> : null}
            {models.data === undefined && !models.error ? (
              <p className="muted" role="status">
                Loading models…
              </p>
            ) : null}
            {lastImport && lastImport.project_id === project.id ? (
              <div className="notice notice-ok" role="status">
                <p className="notice-title">
                  <span aria-hidden="true">✓ </span>Imported “{lastImport.display_name}”
                </p>
                <p>
                  {lastImport.format === "sharded_safetensors" ? "Sharded SafeTensors" : "SafeTensors"},{" "}
                  {lastImport.tensor_count.toLocaleString("en-US")} tensors, status {lastImport.status}
                  {lastImport.warnings?.length ? `, ${lastImport.warnings.length} warning(s)` : ""}.
                </p>
              </div>
            ) : null}
            {models.data ? (
              <ModelList
                models={models.data}
                selectedId={modelId}
                onExplore={onExplore}
                onRelinked={() => models.reload()}
              />
            ) : null}
            <ImportForm
              key={project.id}
              projectId={project.id}
              onImported={(m) => {
                setLastImport(m);
                models.reload();
              }}
            />
            <ConfirmDialog
              open={confirming}
              title={`Delete project “${project.name}”?`}
              confirmLabel="Delete project"
              danger
              busy={deleting}
              onCancel={() => {
                setConfirming(false);
                setDeleteError(null);
              }}
              onConfirm={() => void doDelete()}
            >
              <p>
                This removes the project and its records of imported models from WeightLab’s database on this computer.
              </p>
              <p>
                <strong>Your checkpoint files are not deleted or modified.</strong> You can import them again at any time.
              </p>
              {deleteError ? <ErrorPanel error={deleteError} /> : null}
            </ConfirmDialog>
          </div>
        ) : (
          <div className="empty">
            <h2 id="project-h" className="pane-title">
              Select a project
            </h2>
            <p className="muted">Choose a project on the left, or create a new one.</p>
          </div>
        )}
      </section>
    </div>
  );
}
