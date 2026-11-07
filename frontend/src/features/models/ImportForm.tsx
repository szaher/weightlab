import { useId, useState, type FormEvent } from "react";
import { ErrorPanel } from "../../components/ErrorPanel";
import { api } from "../../lib/api/endpoints";
import type { Model } from "../../lib/api/types";
import { parentDir } from "../../lib/format";
import { PathField } from "./PathField";
import { DISPLAY_NAME_MAX, PATH_MAX, validatePathFields, type FieldErrors } from "./pathForm";

interface Props {
  projectId: string;
  onImported: (m: Model) => void;
}

/** Import in place by typed absolute path (doc 09 section 1.1): no upload, nothing is copied, the browser
 * never reads the file. The server reads only SafeTensors headers and records quick fingerprints. */
export function ImportForm({ projectId, onImported }: Props) {
  const id = useId();
  const [entry, setEntry] = useState("");
  const [root, setRoot] = useState("");
  const [name, setName] = useState("");
  const [errors, setErrors] = useState<FieldErrors>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const fields = { entry_path: entry, selected_root: root, ...(name ? { display_name: name } : {}) };
    const errs = validatePathFields(fields);
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;
    setBusy(true);
    setError(null);
    try {
      const m = await api.importModel(projectId, fields);
      setEntry("");
      setRoot("");
      setName("");
      onImported(m);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card stack" onSubmit={(e) => void submit(e)} noValidate aria-labelledby={`${id}-h`} aria-busy={busy}>
      <h3 id={`${id}-h`} className="card-title">
        Import a SafeTensors checkpoint
      </h3>
      <p className="help">
        Paste the absolute path of a <code>.safetensors</code> file, a folder of sharded SafeTensors, or its{" "}
        <code>.safetensors.index.json</code>. The model is read in place: nothing is uploaded or copied, and WeightLab never
        modifies it. Copy a path with Option-⌘-C in Finder, Shift + right-click → “Copy as path” in Explorer, or{" "}
        <code>realpath</code> in a terminal.
      </p>
      <PathField
        id={`${id}-entry`}
        label="Model path (entry_path)"
        value={entry}
        onChange={setEntry}
        required
        maxLength={PATH_MAX}
        error={errors.entry_path}
        placeholder="/Users/me/models/my-model/model.safetensors"
        help="The checkpoint file, the sharded model folder, or its index file."
      />
      <PathField
        id={`${id}-root`}
        label="Selected root (selected_root)"
        value={root}
        onChange={setRoot}
        required
        maxLength={PATH_MAX}
        error={errors.selected_root}
        placeholder="/Users/me/models/my-model"
        help={
          <>
            The folder WeightLab is allowed to read for this model. The model and every shard it references must be inside
            it; anything that resolves outside (<code>../</code>, absolute or symlinked shard names) is refused with{" "}
            <code>PATH_OUTSIDE_ROOT</code>. Usually the folder that contains the model.
          </>
        }
      />
      <div>
        <button
          type="button"
          className="btn btn-small"
          disabled={!entry.trim()}
          onClick={() => setRoot(parentDir(entry.trim()))}
        >
          Use the folder containing the model path
        </button>
      </div>
      <PathField
        id={`${id}-name`}
        label="Display name"
        value={name}
        onChange={setName}
        maxLength={DISPLAY_NAME_MAX}
        error={errors.display_name}
        mono={false}
        help="Shown in WeightLab only. Defaults to the file or folder name."
      />
      <div className="row">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? "Importing… reading headers" : "Import model"}
        </button>
        {busy ? (
          <span role="status" className="muted">
            Validating the files and reading tensor headers. No tensor values are read.
          </span>
        ) : null}
      </div>
      {error ? <ErrorPanel error={error} context="import" /> : null}
    </form>
  );
}
