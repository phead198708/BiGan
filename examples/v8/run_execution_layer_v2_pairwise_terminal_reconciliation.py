#!/usr/bin/env python3
"""Reconcile terminal async finalization metadata without target access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_terminal_reconciliation import (  # noqa: E402
    PairwiseTerminalReconciliationConfig,
    run_pairwise_terminal_reconciliation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument(
        "--precollection-freeze-manifest",
        required=True,
    )
    parser.add_argument(
        "--precollection-freeze-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--batch-progress-pin",
        action="append",
        required=True,
        metavar="PATH=SHA256",
    )
    parser.add_argument(
        "--training-corpus-root",
        default="/Volumes/PHILIPS/v8",
    )
    return parser


def _pins(values: list[str]) -> tuple[tuple[Path, str], ...]:
    pins = []
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not digest:
            raise ValueError("batch progress pins must use PATH=SHA256")
        pins.append((Path(path), digest))
    return tuple(pins)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pairwise_terminal_reconciliation(
        PairwiseTerminalReconciliationConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            precollection_freeze_manifest_path=(
                args.precollection_freeze_manifest
            ),
            expected_precollection_freeze_manifest_sha256=(
                args.precollection_freeze_manifest_sha256
            ),
            batch_progress_pins=_pins(args.batch_progress_pin),
            training_corpus_root=args.training_corpus_root,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "terminal_reconciliation_ready": report[
                    "terminal_reconciliation_ready"
                ],
                "source_capture_count": report["source_capture_count"],
                "applied_terminal_finalization_count": report[
                    "applied_terminal_finalization_count"
                ],
                "reconciled_exported_finalization_count": report[
                    "reconciled_exported_finalization_count"
                ],
                "blocking_reason_codes": report[
                    "blocking_reason_codes"
                ],
                "reconciled_batch_progress_pins": [
                    f"{path}={digest}"
                    for path, digest in result[
                        "reconciled_batch_progress_pins"
                    ]
                ],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "labels_or_outcomes_opened_for_reconciliation": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["terminal_reconciliation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
