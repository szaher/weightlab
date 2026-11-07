const INT = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function formatInt(n: number): string {
  return INT.format(n);
}

const UNITS = ["KiB", "MiB", "GiB", "TiB"];

/** Exact byte count, with a binary-unit approximation for large values. */
export function formatBytes(n: number): string {
  if (n < 1024) return `${formatInt(n)} B`;
  let v = n;
  let unit = "B";
  for (const u of UNITS) {
    if (v < 1024) break;
    v /= 1024;
    unit = u;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${unit}`;
}

export function formatShape(shape: readonly number[]): string {
  return shape.length === 0 ? "scalar" : shape.map(formatInt).join(" × ");
}

/** Plain-text shape for copying ("257x263"). */
export function shapeText(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : shape.join("x");
}

/** The absolute-path check the server applies, so obvious mistakes are caught before a request. */
export function isAbsolutePath(p: string): boolean {
  return p.startsWith("/") || /^[A-Za-z]:[\\/]/.test(p) || p.startsWith("\\\\");
}

/** The folder containing `p` (lexically; the server re-validates everything). */
export function parentDir(p: string): string {
  const trimmed = p.replace(/[\\/]+$/, "");
  const i = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (i < 0) return p;
  if (i === 0) return trimmed.slice(0, 1);
  const head = trimmed.slice(0, i);
  return /^[A-Za-z]:$/.test(head) ? `${head}\\` : head;
}
