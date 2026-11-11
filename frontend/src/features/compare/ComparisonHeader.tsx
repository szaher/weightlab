// What is compared, how trustworthy the alignment is, and how complete and fresh the report is - shown before
// any ranking (ROADMAP P0.1: compatibility first).
import { StatusBadge } from "../../components/StatusBadge";
import type { CompatClass, Comparison, ComparisonReport } from "../../lib/api/compare";
import type { Model } from "../../lib/api/types";
import { formatInt } from "../../lib/format";

export const CLASS_INFO: Record<CompatClass, { name: string; text: string; warn: boolean }> = {
  V: {
    name: "Class V · base file verified",
    text: "The base file's SHA-256 matches the base the derivative names. This proves file identity, not that B was trained from A.",
    warn: false,
  },
  L: {
    name: "Class L · lineage declared or attested",
    text: "A is declared (or you attested it) to be B's training base, but this is not cryptographically verified.",
    warn: false,
  },
  S: {
    name: "Class S · no lineage evidence",
    text:
      "Nothing shows that B was trained from A. Independently trained models need not share a coordinate system (neurons can be permuted), so element-wise differences may be meaningless. Treat every number below as descriptive inspection of raw coordinates only.",
    warn: true,
  },
  X: {
    name: "Class X · structurally incompatible",
    text:
      "The checkpoints differ in structure (shapes or tensors). Only tensors that match by name and shape are compared, without any alignment guarantee; results describe raw coordinates only.",
    warn: true,
  },
};

export const CAUSAL =
  "Weight changes describe how parameters moved. They do not show that the derived model is more accurate, reasons better or has any new capability; behavioural claims need independent evaluation.";

const fmtTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

const label = (s: string) => s.replace(/_/g, " ");

export function ClassBadge({ cls }: { cls: CompatClass }) {
  return (
    <span className={`class-badge class-${cls}`} title={CLASS_INFO[cls].name}>
      {cls}
    </span>
  );
}

interface Props {
  comparison: Comparison;
  modelA: Model | undefined;
  modelB: Model | undefined;
  report: ComparisonReport | undefined;
}

export function ComparisonHeader({ comparison, modelA, modelB, report }: Props) {
  const lin = comparison.lineage;
  const info = CLASS_INFO[lin.compatibility_class];
  const c = comparison.counts;
  const r = report?.report;
  const stale = report && (report.current_model_status.model_a !== "ready" || report.current_model_status.model_b !== "ready");
  return (
    <section className="cmp-header stack-sm" aria-labelledby="cmp-title">
      <h2 id="cmp-title" className="cmp-title">
        <span className="cmp-side">
          <span className="cmp-tag">A · base</span> {modelA?.display_name ?? comparison.model_a_id}
        </span>
        <span className="cmp-arrow" aria-label="compared with">
          →
        </span>
        <span className="cmp-side">
          <span className="cmp-tag">B · derived</span> {modelB?.display_name ?? comparison.model_b_id}
        </span>
      </h2>
      <dl className="facts">
        <div>
          <dt>Orientation</dt>
          <dd>
            <strong>B − A</strong> (changes are derived minus base)
          </dd>
        </div>
        <div>
          <dt>Compatibility</dt>
          <dd>
            <ClassBadge cls={lin.compatibility_class} /> {info.name.split(" · ")[1]}
          </dd>
        </div>
        <div>
          <dt>Lineage evidence</dt>
          <dd>
            {label(lin.lineage_evidence)} · trust {label(lin.lineage_trust_level)} · claim {label(lin.lineage_claim_source)}
          </dd>
        </div>
        <div>
          <dt>Policy</dt>
          <dd>{comparison.policy === "descriptive_only" ? "descriptive only (acknowledged)" : "strict"}</dd>
        </div>
        <div>
          <dt>Tensors</dt>
          <dd>
            {formatInt(c.matched)} matched
            {c.dtype_mismatch ? `, ${formatInt(c.dtype_mismatch)} dtype-mismatched` : ""}
            {c.tied ? `, ${formatInt(c.tied)} tied` : ""}
            {c.missing_in_a + c.missing_in_b ? `, ${formatInt(c.missing_in_a)} only in B, ${formatInt(c.missing_in_b)} only in A` : ""}
            {c.shape_mismatch ? `, ${formatInt(c.shape_mismatch)} shape-mismatched` : ""}
          </dd>
        </div>
        {report ? (
          <>
            <div>
              <dt>Report</dt>
              <dd>
                {report.complete ? (
                  <span className="badge badge-ready">
                    <span aria-hidden="true">●</span> Complete
                  </span>
                ) : (
                  <span className="badge badge-incomplete">
                    <span aria-hidden="true">◐</span> Incomplete
                  </span>
                )}
              </dd>
            </div>
            <div>
              <dt>Consistency</dt>
              <dd title={report.consistency === "verified" ? "SHA-256 of every input file matched before and after the scan" : "Quick fingerprints (size, times, file id) checked while reading"}>
                {report.consistency === "verified" ? "verified (SHA-256 before and after)" : "best effort (quick fingerprints)"}
              </dd>
            </div>
            <div>
              <dt>Report created</dt>
              <dd>
                <time dateTime={report.created_at}>{fmtTime(report.created_at)}</time>
              </dd>
            </div>
          </>
        ) : null}
      </dl>

      {info.warn ? (
        <div className="notice notice-warn" role="note" aria-label={`${info.name} warning`}>
          <p className="notice-title">
            <span aria-hidden="true">⚠ </span>
            {info.name}
          </p>
          <p>{info.text}</p>
        </div>
      ) : (
        <p className="muted small">{info.text}</p>
      )}

      {r && !report.complete ? (
        <div className="notice notice-warn" role="note" aria-label="Incomplete report">
          <p className="notice-title">
            <span aria-hidden="true">◐ </span>Incomplete report: not a complete comparison
          </p>
          <p>
            {formatInt(r.completeness.tensors_withheld)} tensor(s) and {formatInt(r.completeness.rollups_withheld)} rollup(s) have
            withheld metrics (float64 overflow or unreliable arithmetic); {formatInt(r.completeness.entries_not_compared)} entr
            {r.completeness.entries_not_compared === 1 ? "y was" : "ies were"} not compared
            {r.completeness.tied_entries ? ` (${formatInt(r.completeness.tied_entries)} tied)` : ""}. Withheld values are labelled
            “withheld” below, never shown as numbers or zeros.
          </p>
        </div>
      ) : null}

      {stale ? (
        <div className="notice notice-warn" role="note" aria-label="Files changed since the report">
          <p className="notice-title">
            <span aria-hidden="true">▲ </span>Files changed since this report
          </p>
          <p>
            This report is a snapshot of the bytes read at <time dateTime={report.created_at}>{fmtTime(report.created_at)}</time>. Now:
            A <StatusBadge status={report.current_model_status.model_a} />, B <StatusBadge status={report.current_model_status.model_b} />.
            Re-import the changed model and create a new comparison to describe the current files.
          </p>
        </div>
      ) : null}

      <p className="causal" role="note">
        <strong>Interpretation:</strong> {CAUSAL}
      </p>
      {r && r.caveats.length ? (
        <details className="caveats">
          <summary>Report caveats ({r.caveats.length})</summary>
          <ul>
            {r.caveats.map((cv) => (
              <li key={cv}>{cv}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {lin.reasons.length ? (
        <details className="caveats">
          <summary>Lineage assessment reasons ({lin.reasons.length})</summary>
          <ul>
            {lin.reasons.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
