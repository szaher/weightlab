import { CopyButton } from "../../components/CopyButton";
import type { CellRef } from "./HeatmapCanvas";
import { cellText, copyText, STATE_LABEL, type RoiGrid } from "./values";

interface Props {
  grid: RoiGrid | null;
  selected: CellRef | null;
  tensorIndex: number[] | null;
  rank: number;
}

/** The selected cell: exact value, dtype, state and coordinates, with exact copy actions. */
export function SelectedCell({ grid, selected, tensorIndex, rank }: Props) {
  const inGrid =
    grid &&
    selected &&
    selected.row >= grid.bounds.r0 &&
    selected.row < grid.bounds.r1 &&
    selected.col >= grid.bounds.c0 &&
    selected.col < grid.bounds.c1;
  const cell = inGrid ? grid.cells[(selected.row - grid.bounds.r0) * grid.cols + (selected.col - grid.bounds.c0)] : undefined;
  const idx = tensorIndex ? `[${tensorIndex.join(", ")}]` : "";
  return (
    <section className="selected-cell" aria-labelledby="selected-cell-h" data-testid="selected-cell">
      <h3 id="selected-cell-h" className="h3">
        Selected cell
      </h3>
      <div aria-live="polite" aria-atomic="true">
        {!selected ? (
          <p className="muted">Select a cell in the heatmap or the Values table to see its exact value.</p>
        ) : !cell || !grid ? (
          <p className="muted">Loading the exact value…</p>
        ) : (
          <dl className="facts">
            <div>
              <dt>{grid.precision === "lossless" ? "Value (exact)" : `Value (${grid.precision}, not exact)`}</dt>
              <dd className="mono cell-value" data-testid="selected-value">
                {cellText(cell)}
              </dd>
            </div>
            <div>
              <dt>Dtype</dt>
              <dd className="mono" data-testid="selected-dtype">
                {grid.dtype}
              </dd>
            </div>
            <div>
              <dt>State</dt>
              <dd data-testid="selected-state">
                {cell.state === "finite" ? (cell.color === null ? "finite; colour withheld (beyond ±(2^53 − 1))" : "finite") : STATE_LABEL[cell.state]}
              </dd>
            </div>
            <div>
              <dt>View row, column</dt>
              <dd className="mono" data-testid="selected-coords">
                {rank === 0 ? "scalar" : `${selected.row}, ${selected.col}`}
              </dd>
            </div>
            {rank > 0 ? (
              <div>
                <dt>Tensor index</dt>
                <dd className="mono" data-testid="selected-index">
                  {idx}
                </dd>
              </div>
            ) : null}
          </dl>
        )}
      </div>
      {cell && grid ? (
        <div className="row">
          <CopyButton value={copyText(cell)} label="Copy exact value" text="Copy exact value" />
          {rank > 0 ? <CopyButton value={idx} label="Copy coordinates (tensor index)" text="Copy coordinates" /> : null}
        </div>
      ) : null}
    </section>
  );
}
