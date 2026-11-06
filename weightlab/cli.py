"""`weightlab` command line.

    weightlab diff BASE DERIVED [--out DIR] [--json F] [--csv F] [--md F]
    weightlab diff BASE --adapter ADAPTER_DIR [...]
    weightlab start [--port 0] [--host 127.0.0.1] [--data-dir DIR]

Exit codes: 0 complete report; 3 report written but incomplete (unsupported/unmatched parts);
2 input or validation error; 130 cancelled (nothing written).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from weightlab import __version__
from weightlab.analysis.change_report import SORT_KEYS, Options, run
from weightlab.analysis.diff import DEFAULT_CHUNK_ELEMENTS, DEFAULT_EPS_NORM, Cancelled, RunControl
from weightlab.analysis.lineage import LineageInputs, load_manifest
from weightlab.analysis.lora import DEFAULT_MAX_GMAC
from weightlab.analysis.report_io import rollup_csv, tensors_csv, to_json, to_markdown, write_report_set
from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.formats.base import Consistency

EXIT_OK, EXIT_ERROR, EXIT_INCOMPLETE, EXIT_CANCELLED = 0, 2, 3, 130


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="weightlab", description="Local, CPU-only, read-only checkpoint change reports.")
    p.add_argument("--version", action="version", version=f"weightlab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    st = sub.add_parser("start", help="serve the local API on loopback (prints the URL with its access token)")
    st.add_argument("--port", type=int, default=0, help="TCP port; 0 picks a free one (default)")
    st.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (default) or ::1; nothing else is accepted")
    st.add_argument("--data-dir", type=Path, help="application data directory (default: platform user data dir)")
    d = sub.add_parser("diff", help="ranked change report: BASE vs DERIVED, or BASE + --adapter")
    d.add_argument("base", help="base checkpoint: .safetensors, *.safetensors.index.json or a model directory")
    d.add_argument("derived", nargs="?", help="derived checkpoint (omit with --adapter)")
    d.add_argument("--adapter", help="PEFT LoRA adapter directory (adapter_config.json + adapter_model.safetensors)")
    d.add_argument("--selected-root", help="root no input may escape (default: each input's own directory)")
    d.add_argument("--out", help="write report.json, tensors.csv, layers.csv, roles.csv and report.md here")
    d.add_argument("--json", help="write the JSON report to this file")
    d.add_argument("--csv", help="write the tensor/module table as CSV to this file")
    d.add_argument("--md", help="write the Markdown report to this file")
    d.add_argument(
        "--sort",
        choices=SORT_KEYS,
        default="relative_l2",
        help="ranking metric (default relative_l2; cosine ranks lowest first)",
    )
    d.add_argument("--top", type=int, default=25, help="rows in the Markdown/terminal table (default 25)")
    d.add_argument(
        "--epsilon-norm",
        type=float,
        default=DEFAULT_EPS_NORM,
        help="relative_l2 denominator floor: ||B-A|| / max(||A||, eps) (default 1e-12)",
    )
    d.add_argument(
        "--chunk-elements",
        type=int,
        default=DEFAULT_CHUNK_ELEMENTS,
        help="elements per streamed chunk; bounds memory, never changes results beyond rounding (default 262144)",
    )
    d.add_argument(
        "--consistency",
        choices=[c.value for c in Consistency],
        default=Consistency.BEST_EFFORT.value,
        help="verified also hashes every input before and after (two extra full reads)",
    )
    d.add_argument(
        "--max-lora-gmac",
        type=float,
        default=DEFAULT_MAX_GMAC,
        help="per-module budget for the dense B@A product, in 1e9 multiply-adds; "
        "larger modules get the exact norm only (default 50)",
    )
    d.add_argument("--lineage-manifest", help="JSON manifest recording the base file SHA-256 digests")
    d.add_argument("--trust-manifest", action="store_true", help="mark the manifest as a trusted statement")
    d.add_argument(
        "--attest",
        action="store_true",
        help="record that you attest these exact files are base and derivative (Class L, never V)",
    )
    d.add_argument("--include-paths", action="store_true", help="include absolute paths in exports (off by default)")
    d.add_argument("--progress", action="store_true", help="progress on stderr (default when stderr is a terminal)")
    d.add_argument("--quiet", action="store_true", help="do not print the Markdown summary")
    return p


def _inputs(args: argparse.Namespace) -> list[Path]:
    return [Path(os.path.abspath(x)) for x in (args.base, args.derived, args.adapter) if x]


def _check_outputs(outputs: list[Path], inputs: list[Path]) -> None:
    """Never write into an input checkpoint location or over an input file."""
    roots = [Path(os.path.realpath(i if i.is_dir() else i.parent)) for i in inputs]
    for out in outputs:
        real = Path(os.path.realpath(out))
        for root in roots:
            if real == root or root in real.parents:
                raise WeightLabError(
                    ErrorCode.VALIDATION_ERROR, f"refusing to write {out.name} inside an input model directory"
                )


def _progress(enabled: bool) -> tuple[RunControl, threading.Event]:
    control = RunControl()
    last = [0.0]

    def show(done: int, total: int, key: str) -> None:
        now = time.monotonic()
        if now - last[0] < 0.5 and done < total:
            return
        last[0] = now
        pct = 100.0 * done / total if total else 100.0
        sys.stderr.write(f"\r{pct:5.1f}%  {done / 2**20:,.0f}/{total / 2**20:,.0f} MiB  {key[:60]:<60}")
        sys.stderr.flush()

    if enabled:
        control.progress = show
    return control, control.cancel


def diff(args: argparse.Namespace) -> int:
    if (args.derived is None) == (args.adapter is None):
        raise WeightLabError(ErrorCode.VALIDATION_ERROR, "give either DERIVED or --adapter, not both or neither")
    if args.top < 0:
        raise WeightLabError(ErrorCode.VALIDATION_ERROR, "--top must be >= 0")
    outputs: dict[str, Path] = {}
    if args.out:
        o = Path(args.out)
        outputs |= {
            "json": o / "report.json",
            "csv": o / "tensors.csv",
            "layers": o / "layers.csv",
            "roles": o / "roles.csv",
            "md": o / "report.md",
        }
    for k in ("json", "csv", "md"):
        if getattr(args, k):
            outputs[k] = Path(getattr(args, k))
    _check_outputs(list(outputs.values()), _inputs(args))
    lin = LineageInputs(
        manifest=load_manifest(Path(args.lineage_manifest)) if args.lineage_manifest else None,
        trust_manifest=args.trust_manifest,
        user_attestation=args.attest,
    )
    opts = Options(  # validated (finite, bounded) by Options.validate() before any input is opened
        sort=args.sort,
        eps_norm=args.epsilon_norm,
        chunk_elements=args.chunk_elements,
        consistency=Consistency(args.consistency),
        include_paths=args.include_paths,
        max_lora_gmac=args.max_lora_gmac,
        created_at=os.environ.get("WEIGHTLAB_REPORT_TIMESTAMP"),
    )
    control, _ = _progress(args.progress or sys.stderr.isatty())
    report = run(
        args.base,
        args.derived,
        adapter=args.adapter,
        selected_root=args.selected_root,
        lineage=lin,
        options=opts,
        control=control,
    )
    if control.progress is not None:
        sys.stderr.write("\n")
    md = to_markdown(report, args.top)
    texts = {
        "json": to_json(report),
        "csv": tensors_csv(report),
        "layers": rollup_csv(report, "layers"),
        "roles": rollup_csv(report, "roles"),
        "md": md,
    }
    write_report_set({path: texts[kind] for kind, path in outputs.items()}, Path(args.out) if args.out else None)
    if not args.quiet:
        sys.stdout.write(md)
    return EXIT_OK if report["complete"] else EXIT_INCOMPLETE


def start(args: argparse.Namespace) -> int:
    from weightlab.api.config import check_host
    from weightlab.api.server import serve

    try:
        check_host(args.host)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return EXIT_ERROR
    if not 0 <= args.port < 65536:
        sys.stderr.write("error: --port must be in 0-65535\n")
        return EXIT_ERROR
    try:
        serve(args.host, args.port, args.data_dir)
    except OSError as e:
        sys.stderr.write(f"error: cannot listen on {args.host}:{args.port} ({e.strerror})\n")
        return EXIT_ERROR
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "diff":
            return diff(args)
        if args.command == "start":
            return start(args)
    except (KeyboardInterrupt, Cancelled):
        sys.stderr.write("\ncancelled: no report written\n")
        return EXIT_CANCELLED
    except WeightLabError as e:
        sys.stderr.write(f"error: {e.code.value}: {e.detail}\n")
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
