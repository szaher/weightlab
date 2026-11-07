import { useState } from "react";
import { CopyButton } from "../../components/CopyButton";
import type { CellRef } from "./HeatmapCanvas";
import { cellText, copyText, STATE_LABEL, type RoiGrid } from "./values";

export const PAGE_CELLS = 256;

interface Props {
  grid: RoiGrid;
  selected: CellRef | null;
  onSelect: (c: CellRef) => void;
}

/** The same exact region as an accessible table, one row per cell (paged; nothing is rounded). */
export function ValuesTable({ grid, selected, onSelect }: Props) {
  const [onlySpecial, setOnlySpecial] = useState(false);
  const { r0, c0 } = grid.bounds;
  const indices: number[] = [];
  for (let k = 0; k < grid.cells.length; k++) {
    if (!onlySpecial || grid.cells[k]?.state !== "finite") indices.push(k);
  }
  const selIndex =
    selected && selected.row >= r0 && selected.row < r0 + grid.rows && selected.col >= c0 && selected.col < c0 + grid.cols
      ? (selected.row - r0) * grid.cols + (selected.col - c0)
      : -1;
  const pages = Math.max(1, Math.ceil(indices.length / PAGE_CELLS));
  const selPos = indices.indexOf(selIndex);
  const [pageState, setPage] = useState<{ page: number; forSel: number }>(() => ({
    page: selPos >= 0 ? Math.floor(selPos / PAGE_CELLS) : 0,
    forSel: selIndex,
  }));
  // Follow the selection to its page when it changes (derived, not an effect).
  const page = Math.min(pages - 1, pageState.forSel !== selIndex && selPos >= 0 ? Math.floor(selPos / PAGE_CELLS) : pageState.page);
  const go = (p: number) => setPage({ page: Math.max(0, Math.min(pages - 1, p)), forSel: selIndex });
  const shown = indices.slice(page * PAGE_CELLS, (page + 1) * PAGE_CELLS);
  const special = grid.cells.length - grid.counts.finite;
  return (
    <div className="stack-sm">
      <div className="row">
        <label className="radio">
          <input type="checkbox" checked={onlySpecial} onChange={(e) => { setOnlySpecial(e.target.checked); setPage({ page: 0, forSel: selIndex }); }} />
          Only non-finite cells ({special})
        </label>
      </div>
      <p className="help">
        Every value is shown exactly as the server read it ({grid.encoding}); 64-bit integers stay decimal strings and −0.0 keeps
        its sign. Copy puts exactly the shown text on the clipboard.
      </p>
      <div className="table-wrap values-scroll">
        <table className="table values-table" data-testid="values-table">
          <caption className="visually-hidden">
            Exact values, rows [{r0}, {r0 + grid.rows}) and columns [{c0}, {c0 + grid.cols}), dtype {grid.dtype}
          </caption>
          <thead>
            <tr>
              <th scope="col" className="num">Row</th>
              <th scope="col" className="num">Column</th>
              <th scope="col">Exact value</th>
              <th scope="col">Dtype</th>
              <th scope="col">State</th>
              <th scope="col">
                <span className="visually-hidden">Copy</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 ? (
              <tr>
                <td colSpan={6} className="muted">
                  No non-finite cells in this region.
                </td>
              </tr>
            ) : null}
            {shown.map((k) => {
              const cell = grid.cells[k];
              if (!cell) return null;
              const row = r0 + Math.floor(k / grid.cols);
              const col = c0 + (k % grid.cols);
              const isSel = k === selIndex;
              return (
                <tr key={k} className={isSel ? "selected" : undefined} data-testid={`value-${row}-${col}`}>
                  <td className="num mono">
                    {row}
                    {isSel ? <span className="visually-hidden"> (selected)</span> : null}
                  </td>
                  <td className="num mono">{col}</td>
                  <td className="mono value-cell">
                    <button type="button" className="value-btn" onClick={() => onSelect({ row, col })} aria-label={`Select row ${row}, column ${col}: ${cellText(cell)}`}>
                      {cellText(cell)}
                    </button>
                  </td>
                  <td className="mono">{grid.dtype}</td>
                  <td>{cell.state === "finite" ? (cell.color === null ? "finite (colour withheld)" : STATE_LABEL.finite) : <strong>{STATE_LABEL[cell.state]}</strong>}</td>
                  <td>
                    <CopyButton value={copyText(cell)} label={`Copy exact value at row ${row}, column ${col}`} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {pages > 1 ? (
        <div className="pager">
          <span className="catalog-status">
            Cells {page * PAGE_CELLS + 1}–{Math.min(indices.length, (page + 1) * PAGE_CELLS)} of {indices.length}
          </span>
          <div className="row">
            <button type="button" className="btn btn-small" disabled={page === 0} onClick={() => go(page - 1)}>
              Previous cells
            </button>
            <button type="button" className="btn btn-small" disabled={page >= pages - 1} onClick={() => go(page + 1)}>
              Next cells
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
