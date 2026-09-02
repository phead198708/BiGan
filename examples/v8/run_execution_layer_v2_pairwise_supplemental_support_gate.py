#!/usr/bin/env python3
"""Run the #188 outcome-blind supplemental support gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_supplemental_support import (  # noqa: E402
    PairwiseSupplementalSupportGateConfig,
    run_pairwise_supplemental_support_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
    )
    parser.add_argument("--successor-freeze", required=True)
    parser.add_argument("--successor-freeze-sha256", required=True)
    parser.add_argument(
        "--supplemental-batch-progress-pin",
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
    pins: list[tuple[Path, str]] = []
    for value in values:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or not digest:
            raise ValueError(
                "supplemental batch pins must use PATH=SHA256"
            )
        pins.append((Path(path), digest))
    return tuple(pins)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pairwise_supplemental_support_gate(
        PairwiseSupplementalSupportGateConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            successor_freeze_path=args.successor_freeze,
            successor_freeze_sha256=args.successor_freeze_sha256,
            supplemental_batch_progress_pins=_pins(
                args.supplemental_batch_progress_pin
            ),
            training_corpus_root=args.training_corpus_root,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "supplemental_support_target_ready": report[
                    "supplemental_support_target_ready"
                ],
                "combined_capture_attempt_count": report[
                    "combined_capture_attempt_count"
                ],
                "selected_market_count": report[
                    "selected_market_count"
                ],
                "new_selected_market_count": report[
                    "new_selected_market_count"
                ],
                "role_market_counts": report["role_market_counts"],
                "blocking_reason_codes": report[
                    "blocking_reason_codes"
                ],
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "labels_or_outcomes_opened_for_support_gate": False,
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["supplemental_support_target_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
