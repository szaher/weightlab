import type { ModelStatus } from "../lib/api/types";

const LABEL: Record<ModelStatus, { icon: string; text: string; hint: string }> = {
  ready: { icon: "●", text: "Ready", hint: "Files match the import" },
  changed: { icon: "▲", text: "Changed", hint: "Files differ from the import (size, time or identity)" },
  disconnected: { icon: "○", text: "Disconnected", hint: "Files not found at the recorded location" },
  incomplete: { icon: "◐", text: "Incomplete", hint: "Part of the model could not be catalogued" },
  unsupported: { icon: "✕", text: "Unsupported", hint: "Format not supported" },
};

/** Status as icon + word (never colour alone, doc 08 section 8). */
export function StatusBadge({ status }: { status: ModelStatus }) {
  const l = LABEL[status] ?? { icon: "?", text: status, hint: "" };
  return (
    <span className={`badge badge-${status}`} title={l.hint}>
      <span aria-hidden="true">{l.icon}</span> {l.text}
    </span>
  );
}

export function statusHint(status: ModelStatus): string {
  return LABEL[status]?.hint ?? "";
}
