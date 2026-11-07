import { useEffect, useState } from "react";

/** Copies `value` exactly (e.g. a full raw tensor key). Falls back to selecting text if the clipboard API
 * is unavailable. */
export function CopyButton({ value, label, text = "Copy" }: { value: string; label: string; text?: string }) {
  const [done, setDone] = useState<"idle" | "ok" | "fail">("idle");
  useEffect(() => {
    if (done === "idle") return;
    const t = setTimeout(() => setDone("idle"), 2000);
    return () => clearTimeout(t);
  }, [done]);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setDone("ok");
    } catch {
      setDone("fail");
    }
  }
  return (
    <>
      <button type="button" className="btn btn-small" onClick={() => void copy()} aria-label={done === "ok" ? `Copied: ${label}` : label}>
        {done === "ok" ? "Copied" : text}
      </button>
      <span role="status" className="visually-hidden">
        {done === "ok" ? "Copied to clipboard" : done === "fail" ? "Copy failed; select the text instead" : ""}
      </span>
    </>
  );
}
