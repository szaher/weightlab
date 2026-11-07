import { useId, useState, type FormEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { Model } from "../../lib/api/types";
import { parentDir } from "../../lib/format";
import { PathField } from "./PathField";
import { PATH_MAX, validatePathFields, type FieldErrors } from "./pathForm";

/** Reconnect a model whose files moved (POST /models/{id}/relink). Accepted only when every file matches the
 * import exactly (size, modification time, content of the index and the whole tensor catalog). */
export function RelinkForm({ model, onRelinked, onCancel }: { model: Model; onRelinked: (m: Model) => void; onCancel: () => void }) {
  const id = useId();
  const [entry, setEntry] = useState("");
  const [root, setRoot] = useState("");
  const [errors, setErrors] = useState<FieldErrors>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const fields = { entry_path: entry, selected_root: root };
    const errs = validatePathFields(fields);
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;
    setBusy(true);
    setError(null);
    try {
      onRelinked(await api.relinkModel(model.id, fields));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack-sm relink" onSubmit={(e) => void submit(e)} noValidate aria-label={`Relink ${model.display_name}`} aria-busy={busy}>
      <p className="help">
        Moved the files? Enter their new location. Relink succeeds only if the files are byte-for-byte the imported ones
        (same sizes, modification times and tensor catalog); a modified model must be imported again.
      </p>
      <PathField
        id={`${id}-entry`}
        label="New model path"
        value={entry}
        onChange={setEntry}
        required
        maxLength={PATH_MAX}
        error={errors.entry_path}
        help="The checkpoint file, folder or index file at its new location."
      />
      <PathField
        id={`${id}-root`}
        label="New selected root"
        value={root}
        onChange={setRoot}
        required
        maxLength={PATH_MAX}
        error={errors.selected_root}
        help="The folder that now contains the model and all of its shards."
      />
      <div className="row">
        <button type="button" className="btn btn-small" disabled={!entry.trim()} onClick={() => setRoot(parentDir(entry.trim()))}>
          Use the folder containing the model path
        </button>
      </div>
      <div className="row">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? "Relinking… checking files" : "Relink"}
        </button>
        <button type="button" className="btn" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
      {error ? <ErrorPanel error={error} context="relink" /> : null}
    </form>
  );
}
