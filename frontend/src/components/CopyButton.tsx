import { useEffect, useState } from "react";

/** Copies `value` exactly (e.g. a full raw tensor key). Falls back to selecting text if the clipboard API
 * is unavailable. */
export function CopyButton({ value, label }: { value: string; label: string }) {
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
      <button type="button" className="btn btn-small" onClick={() => void copy()} aria-label={label}>
        {done === "ok" ? "Copied" : "Copy"}
      </button>
      <span role="status" className="visually-hidden">
        {done === "ok" ? "Copied to clipboard" : done === "fail" ? "Copy failed; select the text instead" : ""}
      </span>
    </>
  );
}
