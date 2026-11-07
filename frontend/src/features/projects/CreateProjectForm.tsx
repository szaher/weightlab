import { useId, useState, type FormEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { Project } from "../../lib/api/types";

export const PROJECT_NAME_MAX = 120;

export function validateProjectName(name: string): string | null {
  const t = name.trim();
  if (!t) return "Enter a project name.";
  if (t.length > PROJECT_NAME_MAX) return `At most ${PROJECT_NAME_MAX} characters.`;
  return null;
}

export function CreateProjectForm({ onCreated, autoFocus }: { onCreated: (p: Project) => void; autoFocus?: boolean }) {
  const [name, setName] = useState("");
  const [touched, setTouched] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const inputId = useId();
  const errId = useId();
  const invalid = validateProjectName(name);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setTouched(true);
    if (invalid) return;
    setBusy(true);
    setError(null);
    try {
      const p = await api.createProject(name.trim());
      setName("");
      setTouched(false);
      onCreated(p);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  const showInvalid = touched && invalid !== null;
  return (
    <form className="stack-sm" onSubmit={(e) => void submit(e)} noValidate aria-label="Create project">
      <label htmlFor={inputId} className="label">
        New project name
      </label>
      <div className="row">
        <input
          id={inputId}
          className="input"
          value={name}
          maxLength={PROJECT_NAME_MAX + 1}
          autoFocus={autoFocus}
          onChange={(e) => setName(e.target.value)}
          aria-invalid={showInvalid}
          aria-describedby={showInvalid ? errId : undefined}
          placeholder="e.g. Qwen fine-tune review"
        />
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? "Creating…" : "Create project"}
        </button>
      </div>
      {showInvalid ? (
        <p id={errId} className="field-error">
          {invalid}
        </p>
      ) : null}
      {error ? <ErrorPanel error={error} /> : null}
    </form>
  );
}
