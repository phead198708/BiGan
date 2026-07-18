"""Freeze #190 future-unseen collection and PnL gates before label access."""

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
    PairwiseFutureUnseenHoldoutPreRegistrationConfig,
    create_pairwise_future_unseen_holdout_pre_registration,
)

DEFAULT_HOLDOUT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_future_unseen_holdout_v1.json"
)
DEFAULT_CANDIDATE_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
DEFAULT_FEATURE_CONTRACT = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def run_freeze(
    *,
    run_id: str,
    output_dir: Path | str,
    pre_registration_created_ts: int,
    holdout_protocol: Path | str,
    holdout_protocol_sha256: str,
    candidate_protocol: Path | str,
    candidate_protocol_sha256: str,
    feature_contract: Path | str,
    feature_contract_sha256: str,
    power_analysis_manifest: Path | str,
    power_analysis_manifest_sha256: str,
    builder_git_commit: str,
) -> dict:
    result = create_pairwise_future_unseen_holdout_pre_registration(
        PairwiseFutureUnseenHoldoutPreRegistrationConfig(
            run_id=run_id,
            output_dir=output_dir,
            pre_registration_created_ts=pre_registration_created_ts,
            holdout_protocol_path=holdout_protocol,
            expected_holdout_protocol_sha256=holdout_protocol_sha256,
            candidate_protocol_path=candidate_protocol,
            expected_candidate_protocol_sha256=candidate_protocol_sha256,
            feature_contract_path=feature_contract,
            expected_feature_contract_sha256=feature_contract_sha256,
            power_analysis_manifest_path=power_analysis_manifest,
            expected_power_analysis_manifest_sha256=(
                power_analysis_manifest_sha256
            ),
            builder_git_commit=builder_git_commit,
        )
    )
    return {
        "run_id": run_id,
        "pre_registration_ready": result["manifest"]["pre_registration_ready"],
        "target_valid_market_count": result["manifest"]["target_valid_market_count"],
        "maximum_capture_attempt_count": result["manifest"][
            "maximum_capture_attempt_count"
        ],
        "minimum_accepted_unique_market_count": result["manifest"][
            "minimum_accepted_unique_market_count"
        ],
        "power_analysis_manifest": result["manifest"]["power_analysis_manifest"],
        "candidate_agnostic_raw_collection": True,
        "collection_may_run_before_issue189_confirmatory_result": True,
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
        "--pre-registration-created-ts",
        type=int,
        default=int(time.time() * 1000),
    )
    parser.add_argument("--holdout-protocol", default=str(DEFAULT_HOLDOUT_PROTOCOL))
    parser.add_argument("--holdout-protocol-sha256", required=True)
    parser.add_argument("--candidate-protocol", default=str(DEFAULT_CANDIDATE_PROTOCOL))
    parser.add_argument("--candidate-protocol-sha256", required=True)
    parser.add_argument("--feature-contract", default=str(DEFAULT_FEATURE_CONTRACT))
    parser.add_argument("--feature-contract-sha256", required=True)
    parser.add_argument(
        "--power-analysis-manifest",
        required=True,
    )
    parser.add_argument("--power-analysis-manifest-sha256", required=True)
    parser.add_argument("--builder-git-commit", required=True)
    args = parser.parse_args(argv)
    summary = run_freeze(
        run_id=args.run_id,
        output_dir=args.output_dir,
        pre_registration_created_ts=args.pre_registration_created_ts,
        holdout_protocol=args.holdout_protocol,
        holdout_protocol_sha256=args.holdout_protocol_sha256,
        candidate_protocol=args.candidate_protocol,
        candidate_protocol_sha256=args.candidate_protocol_sha256,
        feature_contract=args.feature_contract,
        feature_contract_sha256=args.feature_contract_sha256,
        power_analysis_manifest=args.power_analysis_manifest,
        power_analysis_manifest_sha256=args.power_analysis_manifest_sha256,
        builder_git_commit=args.builder_git_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
