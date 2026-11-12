// Layer x role matrix of one metric. Every cell is aggregated from the tensors of exactly that layer and role
// (metrics.ts); the value is printed in the cell, so colour is never the only encoding. Keyboard: arrow keys move,
// Home/End jump within a row, Enter or Space filters the rankings to the cell (again: clears).
import { useMemo, useState, type KeyboardEvent } from "react";
import type { TensorMetric } from "../../lib/api/compare";
import { cellKey, domain, intensity, METRIC_INFO, UNAVAILABLE, valText, valTitle, type Grid } from "./metrics";

import type { Filter } from "./Rankings";

interface Props {
  grid: Grid;
  metric: TensorMetric;
  log: boolean;
  selected: Filter | null;
  onSelect: (c: Filter | null) => void;
}

const layerName = (l: number | null) => (l === null ? "No layer" : `Layer ${l}`);

export function ChangeGrid({ grid, metric, log, selected, onSelect }: Props) {
  const [focus, setFocus] = useState<[number, number]>([0, 0]);
  const vals = useMemo(() => [...grid.cells.values()].map((c) => c.metrics[metric]), [grid, metric]);
  const { lo, hi } = useMemo(() => domain(vals, metric, log), [vals, metric, log]);
  const info = METRIC_INFO[metric];
  const R = grid.layers.length;
  const C = grid.roles.length;

  function move(e: KeyboardEvent<HTMLTableElement>) {
    const [r, c] = focus;
    let next: [number, number] | null = null;
    if (e.key === "ArrowRight") next = [r, Math.min(C - 1, c + 1)];
    else if (e.key === "ArrowLeft") next = [r, Math.max(0, c - 1)];
    else if (e.key === "ArrowDown") next = [Math.min(R - 1, r + 1), c];
    else if (e.key === "ArrowUp") next = [Math.max(0, r - 1), c];
    else if (e.key === "Home") next = [r, 0];
    else if (e.key === "End") next = [r, C - 1];
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const layer = grid.layers[r];
      const role = grid.roles[c];
      if (layer !== undefined && role !== undefined && grid.cells.has(cellKey(layer, role))) toggle(layer, role);
      return;
    }
    if (!next) return;
    e.preventDefault();
    setFocus(next);
    document.getElementById(`gc-${next[0]}-${next[1]}`)?.focus();
  }

  function toggle(layer: number | null, role: string) {
    const same = selected && selected.layer === layer && selected.role === role;
    onSelect(same ? null : { layer, role });
  }

  return (
    <div className="stack-sm">
      <div className="grid-scroll">
        <table className="change-grid" role="grid" aria-label={`${info.label} by layer and role`} aria-rowcount={R + 1} onKeyDown={move}>
          <thead>
            <tr>
              <th scope="col" className="corner">
                Layer \ role
              </th>
              {grid.roles.map((role) => (
                <th key={role} scope="col" className="mono">
                  {role}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.layers.map((layer, r) => (
              <tr key={String(layer)}>
                <th scope="row" className="mono">
                  {layerName(layer)}
                </th>
                {grid.roles.map((role, c) => {
                  const cell = grid.cells.get(cellKey(layer, role));
                  const v = cell?.metrics[metric];
                  const t = v ? intensity(v, metric, lo, hi, log) : null;
                  const isSel = !!selected && selected.layer === layer && selected.role === role;
                  const text = cell && v ? valText(v) : "";
                  const label =
                    cell && v
                      ? `${layerName(layer)}, ${role}: ${info.label} ${valTitle(v)}, ${cell.tensors.length} tensor${cell.tensors.length === 1 ? "" : "s"}${isSel ? ", filtering rankings" : ""}`
                      : `${layerName(layer)}, ${role}: no tensors`;
                  return (
                    <td
                      key={role}
                      id={`gc-${r}-${c}`}
                      role="gridcell"
                      tabIndex={focus[0] === r && focus[1] === c ? 0 : -1}
                      aria-selected={isSel}
                      aria-label={label}
                      title={label}
                      className={`heat-cell${!cell ? " heat-empty" : v && v.kind !== "value" ? ` heat-${v.kind}` : ""}${isSel ? " heat-selected" : ""}`}
                      style={t === null ? undefined : { ["--t" as string]: String(t), color: t > 0.55 ? "var(--heat-text-strong)" : undefined }}
                      onClick={() => {
                        setFocus([r, c]);
                        if (cell) toggle(layer, role);
                      }}
                    >
                      {text}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <GridLegend metric={metric} lo={lo} hi={hi} log={log} />
    </div>
  );
}

function GridLegend({ metric, lo, hi, log }: { metric: TensorMetric; lo: number; hi: number; log: boolean }) {
  const info = METRIC_INFO[metric];
  const what = metric === "cosine" ? "1 − cosine" : info.label;
  return (
    <div className="grid-legend" aria-label="Colour scale">
      <span>
        Colour: {what}, {log ? "log₁₀" : "linear"} scale
      </span>
      <span className="legend-grad" aria-hidden="true" />
      <span className="mono">
        {lo.toExponential(2)} … {hi.toExponential(2)}
      </span>
      {(["withheld", "undefined", "not_aggregable"] as const).map((k) => (
        <span key={k} className="legend-key" title={UNAVAILABLE[k].hint}>
          <span className={`legend-sw heat-${k}`} aria-hidden="true" /> {UNAVAILABLE[k].text}
        </span>
      ))}
      <span className="legend-key">
        <span className="legend-sw heat-empty" aria-hidden="true" /> no tensors
      </span>
    </div>
  );
}
