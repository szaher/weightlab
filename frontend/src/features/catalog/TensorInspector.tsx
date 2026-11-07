import { CopyButton } from "../../components/CopyButton";
import { ErrorPanel } from "../../components/ErrorPanel";
import { KeyText } from "../../components/KeyText";
import { api } from "../../lib/api/endpoints";
import { formatBytes, formatInt, formatShape } from "../../lib/format";
import { useResource } from "../../lib/useResource";
import { RoiExplorer } from "../roi/RoiExplorer";

interface Props {
  modelId: string;
  modelName: string;
  tensorId: string | null;
}

/** One tensor: its metadata (GET /models/{id}/tensors/{tensor_id}) and the exact-window explorer (POST /tensors/roi). */
export function TensorInspector({ modelId, modelName, tensorId }: Props) {
  const res = useResource(tensorId ? `${modelId}/${tensorId}` : null, (signal) => api.getTensor(modelId, tensorId ?? "", signal));
  const t = res.data;
  return (
    <aside className="inspector" aria-labelledby="inspector-h" aria-busy={res.loading}>
      <h2 id="inspector-h" className="pane-title">
        Tensor details
      </h2>
      {!tensorId ? <p className="muted">Select a tensor name in the table to see its metadata and exact values.</p> : null}
      {res.error ? <ErrorPanel error={res.error} onRetry={res.reload} /> : null}
      {tensorId && !t && !res.error ? (
        <p className="muted" role="status">
          Loading…
        </p>
      ) : null}
      {t ? (
        <div className="stack">
          <div>
            <p className="label">Raw key</p>
            <div className="key-box">
              <code className="key-full" data-testid="inspector-key">
                <KeyText value={t.raw_key} />
              </code>
              <CopyButton value={t.raw_key} label="Copy raw tensor key" />
            </div>
            <p className="tensor-sub mono" data-testid="inspector-summary">
              Shape {formatShape(t.shape)} · {t.dtype}
            </p>
          </div>
          <RoiExplorer key={`${modelId}/${t.id}`} modelId={modelId} tensor={t} />
          <h3 className="h3">Metadata</h3>
          <dl className="facts facts-grid">
            <div>
              <dt>Model</dt>
              <dd>{modelName}</dd>
            </div>
            <div>
              <dt>Shape</dt>
              <dd className="mono" data-testid="inspector-shape">
                {formatShape(t.shape)}
              </dd>
            </div>
            <div>
              <dt>Rank</dt>
              <dd className="mono" data-testid="inspector-rank">
                {t.rank}
              </dd>
            </div>
            <div>
              <dt>Dtype (as stored)</dt>
              <dd className="mono" data-testid="inspector-dtype">
                {t.dtype}
              </dd>
            </div>
            <div>
              <dt>Elements</dt>
              <dd className="mono" data-testid="inspector-numel">
                {formatInt(t.numel)}
              </dd>
            </div>
            <div>
              <dt>Byte length</dt>
              <dd className="mono">
                {formatInt(t.byte_length)} bytes{t.byte_length >= 1024 ? ` (${formatBytes(t.byte_length)})` : ""}
              </dd>
            </div>
            <div>
              <dt>Shard (relative to selected root)</dt>
              <dd className="mono wrap">{t.shard_relpath}</dd>
            </div>
            <div>
              <dt>Byte offsets in shard</dt>
              <dd className="mono">
                [{formatInt(t.file_offset_begin)}, {formatInt(t.file_offset_end)})
              </dd>
            </div>
            <div>
              <dt>Role (heuristic)</dt>
              <dd>{t.role ?? "—"}</dd>
            </div>
            <div>
              <dt>Layer (heuristic)</dt>
              <dd className="mono">{t.layer_index ?? "—"}</dd>
            </div>
          </dl>
          <p className="help">
            Role and layer are guessed from the tensor name. Offsets are absolute positions in the shard file, end exclusive.
          </p>
        </div>
      ) : null}
    </aside>
  );
}
