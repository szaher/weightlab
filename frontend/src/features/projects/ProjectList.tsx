import type { Project } from "../../lib/api/types";

interface Props {
  projects: Project[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function ProjectList({ projects, selectedId, onSelect }: Props) {
  if (projects.length === 0) {
    return <p className="muted">No projects yet. Create one to import a checkpoint.</p>;
  }
  return (
    <ul className="list" aria-label="Projects">
      {projects.map((p) => (
        <li key={p.id}>
          <button
            type="button"
            className="list-item"
            aria-current={p.id === selectedId ? "true" : undefined}
            onClick={() => onSelect(p.id)}
          >
            <span className="list-item-title">{p.name}</span>
            <span className="list-item-sub">created {new Date(p.created_at).toLocaleString()}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
