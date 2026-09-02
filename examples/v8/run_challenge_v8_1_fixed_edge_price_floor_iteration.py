#!/usr/bin/env python3
"""Run preregistered historical-development iteration 5 exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.challenge_historical_development import (
    SAFE_FALSES,
    HistoricalDevelopmentEvaluationConfig,
    run_historical_development_evaluation,
)
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_price_floor import (
    CANDIDATE_ID,
    build_fixed_edge_price_floor_comparison,
    materialize_fixed_edge_price_floor_decisions,
    validate_fixed_edge_price_floor_profile,
)
from bigan.v8.polymarket.challenge_v8_1_fixed_edge_support_recovery import (
    validate_fixed_edge_support_recovery_profile,
)
from examples.v8.run_challenge_historical_development import (
    CONFIG_DIR,
    DEFAULT_CLOSURE,
    DEFAULT_LEDGER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REGISTRY,
    DEFAULT_STANDARD,
    _require_clean_preregistered_state,
    _sidecar_digest,
)
from examples.v8.run_challenge_v8_1_fixed_edge_support_recovery_iteration import (
    _descriptor,
    _json,
    _jsonl,
    _sha256,
    _verify,
    _write_json,
    _write_jsonl,
)

DEFAULT_PROFILE = (
    CONFIG_DIR
    / "challenge_v8_1_fixed_edge_0_025_price_floor_0_30_profile.json"
)
DEFAULT_FIXED_EDGE_PROFILE = (
    CONFIG_DIR
    / "challenge_v8_1_fixed_edge_support_recovery_0_025_profile.json"
)
DEFAULT_PREREGISTRATION = (
    CONFIG_DIR
    / "challenge_historical_development_iteration_005_preregistration.json"
)
DEFAULT_MARKET_IDS = (
    CONFIG_DIR / "challenge_historical_development_exact_195_market_ids.txt"
)
DEFAULT_PREVIOUS_ENTRY = (
    CONFIG_DIR / "challenge_historical_development_iteration_004_entry.json"
)
PREVIOUS_ENTRY_SHA256 = (
    "3c1ce6fcddf650ff9bd30bd46c3761b5df7fe785c88b06fa0b07ec9084a265b1"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--base-guard-rows", required=True, type=Path)
    parser.add_argument("--five-action-rows", required=True, type=Path)
    parser.add_argument("--base-comparison-rows", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--fixed-edge-profile",
        type=Path,
        default=DEFAULT_FIXED_EDGE_PROFILE,
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument("--market-ids", type=Path, default=DEFAULT_MARKET_IDS)
    parser.add_argument(
        "--previous-entry",
        type=Path,
        default=DEFAULT_PREVIOUS_ENTRY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attempt-closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--development-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--success-standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    git_state = _require_clean_preregistered_state(args.preregistration)
    profile_path = args.profile.resolve()
    fixed_edge_profile_path = args.fixed_edge_profile.resolve()
    preregistration_path = args.preregistration.resolve()
    market_ids_path = args.market_ids.resolve()
    previous_entry_path = args.previous_entry.resolve()
    profile = _json(profile_path)
    fixed_edge_profile = _json(fixed_edge_profile_path)
    validate_fixed_edge_price_floor_profile(profile)
    validate_fixed_edge_support_recovery_profile(fixed_edge_profile)
    preregistration = _json(preregistration_path)
    if (
        preregistration.get("candidate_id") != CANDIDATE_ID
        or preregistration.get("previous_iteration_entry_sha256")
        != PREVIOUS_ENTRY_SHA256
    ):
        raise ValueError("iteration-5 preregistration mismatch")
    for path, label in (
        (profile_path, "candidate profile"),
        (fixed_edge_profile_path, "fixed-edge profile"),
        (preregistration_path, "iteration preregistration"),
    ):
        _verify(path, _sidecar_digest(path), label=label)
    lineage = profile["lineage"]

    # Freeze target-free decisions before reading any outcome comparison.
    _verify(
        args.base_guard_rows.resolve(),
        lineage["base_v8_1_guard_replay_sha256"],
        label="base v8.1 guard replay",
    )
    _verify(
        args.five_action_rows.resolve(),
        lineage["exact_195_five_action_rows_sha256"],
        label="exact-195 five-action rows",
    )
    _verify(
        market_ids_path,
        lineage["exact_195_market_ids_sha256"],
        label="frozen market ids",
    )
    frozen_market_ids = market_ids_path.read_text(encoding="utf-8").splitlines()
    candidate_decisions = materialize_fixed_edge_price_floor_decisions(
        base_guard_rows=_jsonl(args.base_guard_rows.resolve()),
        five_action_rows=_jsonl(args.five_action_rows.resolve()),
        frozen_market_ids=frozen_market_ids,
        fixed_edge_profile=fixed_edge_profile,
        profile=profile,
    )
    run_dir = args.output_dir.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    decisions_path = run_dir / "candidate_target_free_decisions.jsonl"
    _write_jsonl(decisions_path, candidate_decisions)
    accepted_count = sum(
        row["selected_action"] != "NO_TRADE" for row in candidate_decisions
    )
    target_free_manifest = {
        "schema_version": (
            "bigan-v8-challenge-fixed-edge-price-floor-"
            "target-free-manifest-v1"
        ),
        "run_id": args.run_id,
        "candidate_id": CANDIDATE_ID,
        "implementation_base_commit": git_state[
            "implementation_base_commit"
        ],
        "preregistration_commit": git_state["preregistration_commit"],
        "implementation_commit": git_state["implementation_commit"],
        "profile": _descriptor(profile_path),
        "fixed_edge_profile": _descriptor(fixed_edge_profile_path),
        "preregistration": _descriptor(preregistration_path),
        "frozen_market_ids": _descriptor(market_ids_path),
        "base_guard_rows": _descriptor(args.base_guard_rows.resolve()),
        "five_action_rows": _descriptor(args.five_action_rows.resolve()),
        "candidate_decisions": _descriptor(decisions_path),
        "target_free_accepted_market_count": accepted_count,
        "outcomes_labels_settlement_or_pnl_opened": False,
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "collection_started": False,
        "safety": SAFE_FALSES,
    }
    target_free_manifest_path = run_dir / "candidate_target_free_manifest.json"
    _write_json(target_free_manifest_path, target_free_manifest)

    _verify(
        args.base_comparison_rows.resolve(),
        lineage["base_v8_1_market_comparison_sha256"],
        label="base v8.1 market comparison",
    )
    comparison = build_fixed_edge_price_floor_comparison(
        candidate_decisions=candidate_decisions,
        base_comparison_rows=_jsonl(args.base_comparison_rows.resolve()),
        frozen_market_ids=frozen_market_ids,
    )
    comparison_path = run_dir / "candidate_market_comparison.jsonl"
    _write_jsonl(comparison_path, comparison)
    result = run_historical_development_evaluation(
        HistoricalDevelopmentEvaluationConfig(
            run_id="evaluation",
            output_dir=run_dir,
            iteration_number=5,
            candidate_id=CANDIDATE_ID,
            comparison_rows_path=comparison_path,
            expected_comparison_rows_sha256=_sha256(comparison_path),
            preregistration_path=preregistration_path,
            expected_preregistration_sha256=_sha256(preregistration_path),
            success_standard_path=args.success_standard,
            expected_success_standard_sha256=_sidecar_digest(
                args.success_standard.resolve()
            ),
            registry_path=args.development_registry,
            expected_registry_sha256=_sidecar_digest(
                args.development_registry.resolve()
            ),
            ledger_root_path=args.ledger_root,
            expected_ledger_root_sha256=_sidecar_digest(
                args.ledger_root.resolve()
            ),
            attempt_closure_path=args.attempt_closure,
            expected_attempt_closure_sha256=_sidecar_digest(
                args.attempt_closure.resolve()
            ),
            implementation_base_commit=git_state[
                "implementation_base_commit"
            ],
            preregistration_commit=git_state["preregistration_commit"],
            implementation_commit=git_state["implementation_commit"],
            evaluated_at=args.evaluated_at,
            previous_iteration_entry_sha256=PREVIOUS_ENTRY_SHA256,
            previous_iteration_entry_path=previous_entry_path,
        )
    )
    manifest = {
        "schema_version": (
            "bigan-v8-challenge-fixed-edge-price-floor-"
            "development-manifest-v1"
        ),
        "run_id": args.run_id,
        "candidate_id": CANDIDATE_ID,
        "target_free_manifest": _descriptor(target_free_manifest_path),
        "candidate_comparison": _descriptor(comparison_path),
        "historical_evaluation_manifest": _descriptor(
            result["manifest_path"]
        ),
        "historical_evaluation_report": _descriptor(result["report_path"]),
        "iteration_entry": _descriptor(result["iteration_entry_path"]),
        "all_historical_success_criteria_passed": result["report"][
            "all_historical_success_criteria_passed"
        ],
        "replacement_future_attempt_preregistration_allowed": result[
            "report"
        ]["replacement_future_attempt_preregistration_allowed"],
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "collection_started": False,
        "safety": SAFE_FALSES,
    }
    manifest_path = run_dir / "candidate_development_manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "run_dir": str(run_dir),
        "target_free_manifest_sha256": _sha256(target_free_manifest_path),
        "candidate_comparison_sha256": _sha256(comparison_path),
        "evaluation_report_sha256": result["report_sha256"],
        "iteration_entry_sha256": result["iteration_entry_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "accepted_market_count": accepted_count,
        "all_historical_success_criteria_passed": result["report"][
            "all_historical_success_criteria_passed"
        ],
        "replacement_future_attempt_preregistration_allowed": result[
            "report"
        ]["replacement_future_attempt_preregistration_allowed"],
        "promotion_evidence_eligible": False,
        "collection_started": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_historical_success_criteria_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
