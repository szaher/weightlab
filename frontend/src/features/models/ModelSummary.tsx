import { StatusBadge } from "../../components/StatusBadge";
import type { Model } from "../../lib/api/types";
import { formatInt } from "../../lib/format";

const FORMAT: Record<Model["format"], string> = {
  safetensors: "SafeTensors (single file)",
  sharded_safetensors: "SafeTensors (sharded)",
};

export function formatLabel(f: Model["format"]): string {
  return FORMAT[f] ?? f;
}

/** What the import returned, plus what the status means for the user. */
export function ModelStatusNotice({ model }: { model: Model }) {
  if (model.status === "changed") {
    return (
      <div className="notice notice-warn" role="status">
        <p className="notice-title">
          <span aria-hidden="true">▲ </span>Files changed since import
        </p>
        <p>
          The catalog below is the one recorded at import. Reading tensor values from these files is refused (
          <code>FILE_CHANGED</code>) until you import the model again. If the files only moved and are unchanged, use Relink.
        </p>
      </div>
    );
  }
  if (model.status === "disconnected") {
    return (
      <div className="notice notice-warn" role="status">
        <p className="notice-title">
          <span aria-hidden="true">○ </span>Model disconnected
        </p>
        <p>
          The files are not at the recorded location (moved, renamed, deleted or on an unmounted drive). The recorded
          catalog is still shown. Use Relink in the Workspace to point WeightLab at the new location.
        </p>
      </div>
    );
  }
  return null;
}

export function ModelFacts({ model }: { model: Model }) {
  return (
    <dl className="facts">
      <div>
        <dt>Status</dt>
        <dd>
          <StatusBadge status={model.status} />
        </dd>
      </div>
      <div>
        <dt>Format</dt>
        <dd>{formatLabel(model.format)}</dd>
      </div>
      <div>
        <dt>Tensors</dt>
        <dd>{formatInt(model.tensor_count)}</dd>
      </div>
      <div>
        <dt>Fingerprint</dt>
        <dd>{model.fingerprint_method === "size_mtime" ? "size + modification time" : model.fingerprint_method}</dd>
      </div>
    </dl>
  );
}

export function Warnings({ warnings }: { warnings: string[] | undefined }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div className="notice notice-warn">
      <p className="notice-title">Import warnings</p>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}
