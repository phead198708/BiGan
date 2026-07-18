"""Bind #190 outcome-blind collection to the terminal #188 source boundary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_future_unseen_holdout import (  # noqa: E402
    PairwiseFutureUnseenCollectionFreezeConfig,
    create_pairwise_future_unseen_collection_freeze,
)


def run_freeze(
    *,
    run_id: str,
    output_dir: Path | str,
    collection_freeze_created_ts: int,
    pre_registration_manifest: Path | str,
    pre_registration_manifest_sha256: str,
    source_support_gate_manifest: Path | str,
    source_support_gate_manifest_sha256: str,
    builder_git_commit: str,
) -> dict:
    result = create_pairwise_future_unseen_collection_freeze(
        PairwiseFutureUnseenCollectionFreezeConfig(
            run_id=run_id,
            output_dir=output_dir,
            collection_freeze_created_ts=collection_freeze_created_ts,
            pre_registration_manifest_path=pre_registration_manifest,
            expected_pre_registration_manifest_sha256=(
                pre_registration_manifest_sha256
            ),
            source_support_gate_manifest_path=source_support_gate_manifest,
            expected_source_support_gate_manifest_sha256=(
                source_support_gate_manifest_sha256
            ),
            builder_git_commit=builder_git_commit,
        )
    )
    manifest = result["manifest"]
    return {
        "run_id": run_id,
        "source_boundary_validation_passed": True,
        "source_selected_market_count": manifest["source_selected_market_count"],
        "source_max_decision_ts": manifest["source_max_decision_ts"],
        "minimum_collection_decision_ts": manifest[
            "minimum_collection_decision_ts"
        ],
        "target_valid_market_count": manifest["target_valid_market_count"],
        "maximum_capture_attempt_count": manifest[
            "maximum_capture_attempt_count"
        ],
        "collection_started": False,
        "labels_or_outcomes_opened": False,
        "manifest_path": str(result["manifest_path"]),
        "manifest_sha256": result["manifest_sha256"],
        "report_path": str(result["report_path"]),
        "report_sha256": result["report_sha256"],
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument(
        "--collection-freeze-created-ts",
        type=int,
        default=int(time.time() * 1000),
    )
    parser.add_argument("--pre-registration-manifest", required=True)
    parser.add_argument("--pre-registration-manifest-sha256", required=True)
    parser.add_argument("--source-support-gate-manifest", required=True)
    parser.add_argument("--source-support-gate-manifest-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    args = parser.parse_args(argv)
    summary = run_freeze(
        run_id=args.run_id,
        output_dir=args.output_dir,
        collection_freeze_created_ts=args.collection_freeze_created_ts,
        pre_registration_manifest=args.pre_registration_manifest,
        pre_registration_manifest_sha256=(
            args.pre_registration_manifest_sha256
        ),
        source_support_gate_manifest=args.source_support_gate_manifest,
        source_support_gate_manifest_sha256=(
            args.source_support_gate_manifest_sha256
        ),
        builder_git_commit=args.builder_git_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
