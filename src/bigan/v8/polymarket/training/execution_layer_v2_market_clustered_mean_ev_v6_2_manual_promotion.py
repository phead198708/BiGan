"""Manual promotion review for the frozen v6.2 future gate."""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-manual-promotion"
CANDIDATE_NAME = "market_clustered_mean_ev_v6_2"
ISSUE_NUMBER = 218
PRE_REGISTRATION_GIT_COMMIT = "3238a0d8ab03d0760635c894a4c694511bd06628"

FROZEN_HASHES = {
    "candidate_manifest": "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9",
    "candidate_model": "7e292852673fe2072017effc2d40fce000be81734f0c8c3d6950c02e957bcf0c",
    "candidate_calibration": "dc82ddebc51e95e46477894f2a0ba7bd8fa2f6845b22ced43402822b66b68e43",
    "prediction_freeze_manifest": "0833fed7c17b67937911cf4e5bc8b5acda6a8380f0571422592e8395b5c7bd91",
    "decision_freeze": "b8805b03d0cfcebfad64726f97eb2045197a461500c383db908af20774bcf092",
    "settlement_manifest": "73f90f78a3c36170e7e145622043cad4bbd56e0cf70b3653861eed79c42316f6",
    "settled_corpus_index": "ce0e56b05e8c420c4ca84e7a6e6f43c377e90f5eea185b4086f6cb6fcc10b7db",
    "single_use_claim": "92b5cb4a9b39ec1b5a32d6a0388e8cf3953a8d502f388bb5c42e276676d0cea4",
    "evaluation_manifest": "40908fba0ca5658bf9edb92ccf9c9111843c81de9cb7f819a546acea0d072ca9",
    "side_only_gate_report": "af4c33e7e272fe9a12aa37836d2cee82ea9b71e0c6b8be7d7292f6d9ab887a5b",
    "historical_diagnostic_report": "8a52abc44c10b859112eb6018b936963471053b44a4da2f66c5aa7d3ffa963fd",
}

FROZEN_SOURCE_PINS = (
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_market_clustered_mean_ev_v6_2.py",
        "apply_market_clustered_mean_ev_scores",
        "c6254e4d92a0b305e205a7e23fa59f06ec12d2d6376ef974ef252eadc85ce334",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_outcome_blind_acceptance_viability.py",
        "_outcome_blind_acceptance_replay",
        "a59d839f38054ca1ebf955e14cdf78b32400e457baf42957db09033806ba9cc4",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation.py",
        "validate_market_clustered_mean_ev_v6_2_future_profile",
        "f1a7827af755e25290f7dff31b5d2f78929cb0d5ba951ffac4fb74a684b16b2e",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation.py",
        "build_market_clustered_mean_ev_v6_2_side_only_gate",
        "66adbfe6f70e5d21826aa54bcd46c6dfd85b8a0be7e43e70f25d4bdbccdd01a8",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation.py",
        "_accepted_metrics",
        "15e6597d49261367fb6628176640f1448f7b48b8d2555312bd57ae7b0e67c3f8",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_market_clustered_mean_ev_v6_2_future_evaluation.py",
        "_action_family",
        "0090ea5eb5344a0c873b4e98a39bbbdd59aa8fbb92f096c835b41f7b87c7562d",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_conformal_v5_future_settlement.py",
        "_join_frozen_replay_targets",
        "3333018edfd1030bbb2c7b6c2fca299c310a8c7edf09df2a0c234cc1f2e4a11e",
    ),
    (
        "src/bigan/v8/polymarket/training/execution_layer_v2_direct_advantage_estimand_audit.py",
        "_market_bootstrap_interval",
        "49597d702e2ccb271b6dc3038476fd78c515262a70a9da30ff9d12f54b0cfe84",
    ),
)


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62ManualPromotionConfig:
    """Exact frozen inputs for the one-time #218 manual review."""

    run_id: str
    output_dir: Path | str
    repo_root: Path | str
    candidate_manifest_path: Path | str
    prediction_freeze_manifest_path: Path | str
    settlement_manifest_path: Path | str
    settled_corpus_index_path: Path | str
    single_use_claim_path: Path | str
    evaluation_manifest_path: Path | str
    side_only_gate_report_path: Path | str
    historical_diagnostic_report_path: Path | str
    builder_git_commit: str
    review_completed_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("run_id is required")
        _require_git_sha(self.builder_git_commit)
        if self.review_completed_ts <= 0:
            raise ValueError("review_completed_ts must be positive")
        for name in self.__dataclass_fields__:
            if name.endswith("_path") or name in {"output_dir", "repo_root"}:
                object.__setattr__(self, name, Path(getattr(self, name)))


def run_market_clustered_mean_ev_v6_2_manual_promotion_review(
    config: MarketClusteredMeanEVV62ManualPromotionConfig,
) -> dict[str, Any]:
    """Audit exact frozen evidence and promote only the research candidate."""

    paths = {
        "candidate_manifest": config.candidate_manifest_path,
        "prediction_freeze_manifest": config.prediction_freeze_manifest_path,
        "settlement_manifest": config.settlement_manifest_path,
        "settled_corpus_index": config.settled_corpus_index_path,
        "single_use_claim": config.single_use_claim_path,
        "evaluation_manifest": config.evaluation_manifest_path,
        "side_only_gate_report": config.side_only_gate_report_path,
        "historical_diagnostic_report": config.historical_diagnostic_report_path,
    }
    for name, path in paths.items():
        _verify_pin(Path(path), FROZEN_HASHES[name], name)

    candidate = _load_json(config.candidate_manifest_path)
    freeze = _load_json(config.prediction_freeze_manifest_path)
    settlement_manifest = _load_json(config.settlement_manifest_path)
    settlement_index = _load_json(config.settled_corpus_index_path)
    claim = _load_json(config.single_use_claim_path)
    evaluation_manifest = _load_json(config.evaluation_manifest_path)
    gate = _load_json(config.side_only_gate_report_path)
    historical = _load_json(config.historical_diagnostic_report_path)

    freeze_report = _load_json(
        Path(_verified_descriptor(freeze["report"], "prediction freeze report")["path"])
    )
    settlement_report = _load_json(
        Path(_verified_descriptor(settlement_manifest["report"], "settlement report")["path"])
    )
    evaluation_report = _load_json(
        Path(_verified_descriptor(evaluation_manifest["report"], "evaluation report")["path"])
    )
    selected_rows = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window rows")["path"])
    )
    raw_evidence = _verify_selected_raw_evidence(selected_rows)
    settled_evidence = _verify_settled_evidence(settlement_index, evaluation_manifest)
    source_audit = _audit_frozen_source_surface(
        config.repo_root,
        builder_git_commit=config.builder_git_commit,
    )

    review = _build_review_report(
        run_id=config.run_id,
        review_completed_ts=config.review_completed_ts,
        builder_git_commit=config.builder_git_commit,
        candidate=candidate,
        freeze=freeze,
        freeze_report=freeze_report,
        settlement_manifest=settlement_manifest,
        settlement_report=settlement_report,
        settlement_index=settlement_index,
        claim=claim,
        evaluation_manifest=evaluation_manifest,
        evaluation_report=evaluation_report,
        gate=gate,
        historical=historical,
        selected_rows=selected_rows,
        raw_evidence=raw_evidence,
        settled_evidence=settled_evidence,
        source_audit=source_audit,
    )

    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    report_path = run_dir / "v6_2_manual_promotion_readiness_report.json"
    report_md_path = run_dir / "v6_2_manual_promotion_readiness_report.md"
    _write_json(report_path, review)
    _write_text(report_md_path, _review_markdown(review))

    promotion_manifest = _promotion_manifest(
        review=review,
        report_path=report_path,
        candidate=candidate,
        paths=paths,
    )
    promotion_manifest_path = run_dir / "v6_2_promoted_research_candidate_manifest.json"
    _write_json(promotion_manifest_path, promotion_manifest)

    handoff = _paper_handoff_plan(review)
    handoff_path = run_dir / "v6_2_paper_candidate_handoff_plan.json"
    handoff_md_path = run_dir / "v6_2_paper_candidate_handoff_plan.md"
    _write_json(handoff_path, handoff)
    _write_text(handoff_md_path, _handoff_markdown(handoff))

    bundle = {
        "schema_version": f"{SCHEMA_PREFIX}-bundle-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "manual_promotion_review": _descriptor(report_path),
        "manual_promotion_review_markdown": _descriptor(report_md_path),
        "promoted_research_candidate_manifest": _descriptor(promotion_manifest_path),
        "paper_candidate_handoff_plan": _descriptor(handoff_path),
        "paper_candidate_handoff_plan_markdown": _descriptor(handoff_md_path),
        "manual_promotion_review_passed": review["manual_promotion_review_passed"],
        "research_candidate_promoted": review["research_candidate_promoted"],
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        **_execution_safety_fields(),
    }
    bundle["manifest_id"] = canonical_json_sha256(bundle)
    bundle_path = run_dir / "v6_2_manual_promotion_bundle_manifest.json"
    _write_json(bundle_path, bundle)
    return {
        "run_dir": run_dir,
        "report": review,
        "report_path": report_path,
        "promotion_manifest": promotion_manifest,
        "promotion_manifest_path": promotion_manifest_path,
        "paper_handoff_plan": handoff,
        "paper_handoff_plan_path": handoff_path,
        "bundle_manifest": bundle,
        "bundle_manifest_path": bundle_path,
    }


def _build_review_report(
    *,
    run_id: str,
    review_completed_ts: int,
    builder_git_commit: str,
    candidate: dict[str, Any],
    freeze: dict[str, Any],
    freeze_report: dict[str, Any],
    settlement_manifest: dict[str, Any],
    settlement_report: dict[str, Any],
    settlement_index: dict[str, Any],
    claim: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    evaluation_report: dict[str, Any],
    gate: dict[str, Any],
    historical: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    raw_evidence: dict[str, Any],
    settled_evidence: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build the fail-closed promotion decision from already-frozen evidence."""

    source_model = _verified_descriptor(candidate.get("source_model"), "candidate model")
    calibration = _verified_descriptor(
        candidate.get("market_clustered_mean_risk_calibration"),
        "candidate calibration",
    )
    selected_markets = {str(row.get("market_id") or "") for row in selected_rows}
    sides = dict(gate.get("accepted_side_metrics") or {})
    bootstrap = dict(gate.get("candidate_minus_matched_v5_market_bootstrap") or {})
    fallback_entries = [
        row
        for row in settlement_index.get("entries") or []
        if row.get("evaluation_only_settlement_fallback") is True
    ]
    checks = {
        "candidate_lineage_hashes_verified": source_model["sha256"]
        == FROZEN_HASHES["candidate_model"]
        and calibration["sha256"] == FROZEN_HASHES["candidate_calibration"],
        "candidate_frozen_before_future_collection": candidate.get(
            "research_actionability_candidate_frozen"
        )
        is True
        and candidate.get("target_free_actionability_gate_passed") is True
        and candidate.get("target_free_labels_outcomes_settlement_targets_or_pnl_opened")
        is False,
        "exact_200_strictly_later_disjoint_window_verified": len(selected_rows) == 200
        and len(selected_markets) == 200
        and "" not in selected_markets
        and freeze_report.get("selected_market_count") == 200
        and freeze_report.get("future_strictly_later_disjoint_and_exact_window_passed")
        is True
        and all(
            int(row.get("market_start_ts") or 0)
            > int(candidate.get("future_collection_minimum_created_ts_exclusive") or 0)
            for row in selected_rows
        ),
        "prediction_and_guard_frozen_before_target_access": freeze.get(
            "decision_freeze_written_before_target_access"
        )
        is True
        and freeze.get("labels_outcomes_or_pnl_opened") is False
        and freeze.get("settlement_provider_called") is False
        and freeze.get("resolution_artifact_opened") is False,
        "target_free_feature_causality_and_grid_verified": freeze_report.get(
            "feature_causality_violation_count"
        )
        == 0
        and freeze_report.get("complete_five_action_grid_passed") is True
        and freeze_report.get("target_free_support_gate_passed") is True,
        "source_raw_evidence_hashes_and_outcome_seal_verified": raw_evidence["passed"],
        "official_settlement_complete_after_freeze": settlement_report.get(
            "settled_corpus_ready_market_count"
        )
        == 200
        and settlement_report.get("unresolved_or_failed_market_count") == 0
        and settlement_report.get("official_read_only_resolution_only") is True
        and settlement_report.get("source_outcome_blind_rounds_mutated") is False
        and settlement_report.get("future_results_used_for_tuning") is False
        and settled_evidence["passed"],
        "evaluation_only_fallback_narrow_and_training_gate_unrelaxed": settlement_report.get(
            "direct_training_eligibility_relaxed"
        )
        is False
        and len(fallback_entries) == 1
        and all(
            row.get("official_read_only_resolution") is True
            and row.get("direct_training_eligibility_relaxed") is False
            and row.get("evaluation_only_settlement_fallback_reason_codes")
            == ["frozen_feature_equivalent_chainlink_training_gate_block"]
            for row in fallback_entries
        ),
        "single_use_claim_binds_exact_freeze_and_settlement": dict(
            claim.get("prediction_freeze_manifest") or {}
        ).get("sha256")
        == FROZEN_HASHES["prediction_freeze_manifest"]
        and dict(claim.get("settled_corpus_index") or {}).get("sha256")
        == FROZEN_HASHES["settled_corpus_index"]
        and claim.get("future_result_driven_rerun_allowed") is False
        and dict(evaluation_manifest.get("single_use_claim") or {}).get("sha256")
        == FROZEN_HASHES["single_use_claim"]
        and evaluation_report.get("side_only_gate_executed_exactly_once") is True,
        "side_only_gate_passed_without_blockers": gate.get("future_gate_passed") is True
        and gate.get("future_gate_blocking_reason_codes") == []
        and all((gate.get("future_gate_checks") or {}).values())
        and gate.get("pnl_hard_gate_aggregation")
        == "selected_side_buy_up_buy_down_only"
        and gate.get("action_and_action_family_pnl_diagnostic_only") is True,
        "both_buy_sides_supported_and_profitable": set(sides) == {"UP", "DOWN"}
        and all(
            int(sides[side].get("accepted_unique_market_count") or 0) >= 17
            and float(sides[side].get("accepted_bet_net_pnl_sum") or 0.0) > 0.0
            and sides[side].get("diagnostic_only") is False
            for side in ("UP", "DOWN")
        ),
        "candidate_post_cost_pnl_and_baseline_delta_positive": float(
            gate.get("candidate_post_cost_net_pnl") or 0.0
        )
        > 0.0
        and float(gate.get("candidate_minus_matched_v5_post_cost_net_pnl") or 0.0)
        > 0.0,
        "market_bootstrap_lcb_positive": bootstrap.get("bootstrap_unit") == "market_id"
        and int(bootstrap.get("market_count") or 0) == 200
        and float(bootstrap.get("lower_confidence_bound") or 0.0) > 0.0,
        "largest_winner_removed_pnl_positive": float(
            gate.get("largest_winner_removed_candidate_pnl") or 0.0
        )
        > 0.0,
        "future_results_not_used_for_tuning_or_rerun": evaluation_manifest.get(
            "future_results_used_for_tuning"
        )
        is False
        and evaluation_manifest.get("future_result_driven_rerun_allowed") is False
        and evaluation_report.get("future_results_used_for_tuning") is False
        and gate.get("future_result_driven_rerun_allowed") is False,
        "frozen_scoring_guard_cost_and_gate_source_unchanged": source_audit["passed"],
        "historical_v5_replay_remains_diagnostic_only": historical.get(
            "historical_outcome_aware_diagnostic_only"
        )
        is True
        and historical.get("uses_historical_pnl_for_tuning") is False
        and historical.get("no_strictly_unseen_split_in_this_report") is True
        and historical.get("promotion_evidence") is False,
        "source_artifact_safety_flags_remained_blocked_during_evaluation": all(
            _source_artifact_safety_blocked(payload)
            for payload in (
                candidate,
                freeze,
                settlement_manifest,
                settlement_report,
                settlement_index,
                claim,
                evaluation_manifest,
                evaluation_report,
                gate,
                historical,
            )
        ),
    }
    blockers = [f"manual_review_{name}_failed" for name, passed in checks.items() if not passed]
    passed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-readiness-report-v1",
        "run_id": run_id,
        "issue": ISSUE_NUMBER,
        "candidate_name": CANDIDATE_NAME,
        "review_completed_ts": review_completed_ts,
        "builder_git_commit": builder_git_commit,
        "pre_registration_git_commit": PRE_REGISTRATION_GIT_COMMIT,
        "frozen_lineage": {
            "candidate_manifest_sha256": FROZEN_HASHES["candidate_manifest"],
            "model_sha256": source_model["sha256"],
            "calibration_sha256": calibration["sha256"],
            "prediction_freeze_manifest_sha256": FROZEN_HASHES[
                "prediction_freeze_manifest"
            ],
            "decision_freeze_sha256": FROZEN_HASHES["decision_freeze"],
            "settled_corpus_index_sha256": FROZEN_HASHES["settled_corpus_index"],
            "single_use_claim_sha256": FROZEN_HASHES["single_use_claim"],
            "side_only_gate_report_sha256": FROZEN_HASHES["side_only_gate_report"],
        },
        "future_evidence_summary": {
            "evaluation_market_count": 200,
            "guard_accepted_bet_count": gate.get("guard_accepted_bet_count"),
            "guard_accepted_unique_market_count": gate.get(
                "guard_accepted_unique_market_count"
            ),
            "accepted_side_metrics": sides,
            "candidate_post_cost_net_pnl": gate.get("candidate_post_cost_net_pnl"),
            "matched_v5_post_cost_net_pnl": gate.get("matched_v5_post_cost_net_pnl"),
            "candidate_minus_matched_v5_post_cost_net_pnl": gate.get(
                "candidate_minus_matched_v5_post_cost_net_pnl"
            ),
            "market_bootstrap": bootstrap,
            "largest_winner_removed_candidate_pnl": gate.get(
                "largest_winner_removed_candidate_pnl"
            ),
        },
        "raw_evidence_audit": raw_evidence,
        "settled_evidence_audit": settled_evidence,
        "post_preregistration_source_audit": source_audit,
        "manual_promotion_review_checks": checks,
        "manual_promotion_review_blocking_reason_codes": blockers,
        "manual_promotion_review_passed": passed,
        "research_candidate_promoted": passed,
        "source_model_candidate_eligible": passed,
        "freeze_ready": passed,
        "promotion_evidence_eligible": passed,
        "future_result_used_for_promotion_decision_only": True,
        "future_results_used_for_model_threshold_cost_sizing_or_guard_tuning": False,
        "future_result_driven_rerun_allowed": False,
        "paper_candidate_requires_separate_explicit_gate": True,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        **_execution_safety_fields(),
    }
    payload = dict(report)
    report["manual_promotion_review_id"] = canonical_json_sha256(payload)
    return report


def _verify_selected_raw_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    descriptor_count = 0
    invalid_rows: list[str] = []
    for row in rows:
        market_id = str(row.get("market_id") or "")
        try:
            _verified_descriptor(row.get("pending_round_capture_manifest"), "capture manifest")
            _verified_descriptor(row.get("pending_round_capture_report"), "capture report")
            descriptor_count += 2
            for name, descriptor in sorted((row.get("raw_artifacts") or {}).items()):
                _verified_descriptor(descriptor, f"raw artifact {name}")
                descriptor_count += 1
            if (
                row.get("capture_quality_valid") is not True
                or row.get("labels_outcomes_or_pnl_opened") is not False
                or row.get("resolution_provider_called") is not False
                or int(row.get("raw_resolution_row_count") or 0) != 0
                or row.get("settlement_finalizer_started") is not False
                or row.get("training_corpus_export_attempted") is not False
            ):
                invalid_rows.append(market_id)
        except ValueError:
            invalid_rows.append(market_id)
    return {
        "selected_market_count": len(rows),
        "verified_raw_descriptor_count": descriptor_count,
        "invalid_market_ids": sorted(set(invalid_rows)),
        "passed": len(rows) == 200 and not invalid_rows,
    }


def _verify_settled_evidence(
    index: dict[str, Any], evaluation_manifest: dict[str, Any]
) -> dict[str, Any]:
    entries = list(index.get("entries") or [])
    target_sources = list(evaluation_manifest.get("settled_target_sources") or [])
    invalid: list[str] = []
    descriptor_count = 0
    for row in entries:
        market_id = str(row.get("market_id") or "")
        try:
            for name in ("corpus_manifest", "feature_rows", "label_rows", "resolution_events"):
                _verified_descriptor(row.get(name), f"settled {name}")
                descriptor_count += 1
            if (
                row.get("official_read_only_resolution") is not True
                or row.get("corpus_built_after_decision_freeze") is not True
                or row.get("settled_after_market_close") is not True
                or row.get("source_outcome_blind_round_mutated") is not False
                or row.get("direct_training_corpus_exported") is not False
                or row.get("direct_training_eligibility_relaxed") is not False
            ):
                invalid.append(market_id)
        except ValueError:
            invalid.append(market_id)
    for row in target_sources:
        for name in ("corpus_manifest", "feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(row.get(name), f"evaluation target {name}")
            descriptor_count += 1
    index_markets = {str(row.get("market_id") or "") for row in entries}
    target_markets = {str(row.get("market_id") or "") for row in target_sources}
    return {
        "settled_market_count": len(entries),
        "evaluation_target_source_market_count": len(target_sources),
        "verified_settled_descriptor_count": descriptor_count,
        "invalid_market_ids": sorted(set(invalid)),
        "market_identity_reconciliation_passed": index_markets == target_markets,
        "passed": len(entries) == 200
        and len(target_sources) == 200
        and index_markets == target_markets
        and not invalid,
    }


def _audit_frozen_source_surface(
    repo_root: Path, *, builder_git_commit: str
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PRE_REGISTRATION_GIT_COMMIT, builder_git_commit],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    rows = []
    for path, function_name, expected in FROZEN_SOURCE_PINS:
        prereg = _git_function_sha256(repo_root, PRE_REGISTRATION_GIT_COMMIT, path, function_name)
        reviewed = _git_function_sha256(repo_root, builder_git_commit, path, function_name)
        rows.append(
            {
                "path": path,
                "function_name": function_name,
                "expected_frozen_sha256": expected,
                "pre_registration_sha256": prereg,
                "reviewed_commit_sha256": reviewed,
                "unchanged": prereg == expected == reviewed,
            }
        )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{PRE_REGISTRATION_GIT_COMMIT}..{builder_git_commit}"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    commits = []
    for line in subprocess.check_output(
        ["git", "log", "--format=%H%x09%s", f"{PRE_REGISTRATION_GIT_COMMIT}..{builder_git_commit}"],
        cwd=repo_root,
        text=True,
    ).splitlines():
        commit, _, subject = line.partition("\t")
        commits.append({"commit": commit, "subject": subject})
    return {
        "pre_registration_commit_is_ancestor": ancestor,
        "frozen_function_pin_count": len(rows),
        "frozen_function_pins": rows,
        "all_frozen_function_pins_unchanged": all(row["unchanged"] for row in rows),
        "post_preregistration_changed_paths": changed,
        "post_preregistration_commits": commits,
        "passed": ancestor and all(row["unchanged"] for row in rows),
    }


def _git_function_sha256(
    repo_root: Path, commit: str, path: str, function_name: str
) -> str:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        text=True,
    )
    tree = ast.parse(source)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function_name
        ),
        None,
    )
    if node is None:
        raise ValueError(f"frozen function missing: {path}:{function_name}")
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"frozen function source unavailable: {path}:{function_name}")
    return hashlib.sha256(segment.encode()).hexdigest()


def _promotion_manifest(
    *,
    review: dict[str, Any],
    report_path: Path,
    candidate: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    promoted = review["manual_promotion_review_passed"] is True
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-research-candidate-manifest-v1",
        "candidate_name": CANDIDATE_NAME,
        "manual_promotion_review": _descriptor(report_path),
        "original_frozen_candidate_manifest": _descriptor(paths["candidate_manifest"]),
        "prediction_freeze_manifest": _descriptor(paths["prediction_freeze_manifest"]),
        "settlement_manifest": _descriptor(paths["settlement_manifest"]),
        "settled_corpus_index": _descriptor(paths["settled_corpus_index"]),
        "single_use_claim": _descriptor(paths["single_use_claim"]),
        "evaluation_manifest": _descriptor(paths["evaluation_manifest"]),
        "side_only_gate_report": _descriptor(paths["side_only_gate_report"]),
        "source_model": _verified_descriptor(candidate["source_model"], "candidate model"),
        "market_clustered_mean_risk_calibration": _verified_descriptor(
            candidate["market_clustered_mean_risk_calibration"], "candidate calibration"
        ),
        "future_evidence_summary": review["future_evidence_summary"],
        "research_candidate_promoted": promoted,
        "source_model_candidate_eligible": promoted,
        "freeze_ready": promoted,
        "promotion_evidence_eligible": promoted,
        "future_result_used_for_promotion_decision_only": True,
        "future_result_driven_rerun_allowed": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        **_execution_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    return manifest


def _paper_handoff_plan(review: dict[str, Any]) -> dict[str, Any]:
    plan = {
        "schema_version": f"{SCHEMA_PREFIX}-paper-handoff-plan-v1",
        "candidate_name": CANDIDATE_NAME,
        "research_candidate_promoted": review["research_candidate_promoted"],
        "paper_candidate_gate_required": True,
        "paper_candidate_required_checks": [
            "separate_preregistered_issue_and_manual_approval",
            "pin_promoted_manifest_model_calibration_profile_guard_cost_and_sizing_hashes",
            "paper_only_read_only_provider_and_zero_write_capability",
            "target_free_runtime_causality_and_provenance_validation",
            "bounded_forward_paper_canary_with_kill_switch",
            "no_retraining_or_threshold_tuning_from_future_gate_outcomes",
        ],
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **_execution_safety_fields(),
    }
    plan["plan_id"] = canonical_json_sha256(plan)
    return plan


def _source_artifact_safety_blocked(payload: dict[str, Any]) -> bool:
    return (
        payload.get("source_model_candidate_eligible") is False
        and payload.get("freeze_ready") is False
        and payload.get("promotion_evidence_eligible") is False
        and payload.get("paper_candidate_allowed") is False
        and payload.get("v8_execution_handoff_allowed") is False
        and payload.get("paper_only") is True
        and payload.get("capital_at_risk") is False
        and payload.get("polymarket_write_enabled") is False
        and payload.get("wallet_signing_enabled") is False
        and payload.get("#134_resume_allowed") is False
        and payload.get("#146_start_allowed") is False
    )


def _execution_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_trading_enabled": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _review_markdown(report: dict[str, Any]) -> str:
    evidence = report["future_evidence_summary"]
    return "\n".join(
        [
            "# v6.2 manual promotion review",
            "",
            f"- review passed: `{str(report['manual_promotion_review_passed']).lower()}`",
            f"- research candidate promoted: `{str(report['research_candidate_promoted']).lower()}`",
            f"- accepted bets: `{evidence['guard_accepted_bet_count']}`",
            f"- candidate post-cost PnL: `{evidence['candidate_post_cost_net_pnl']}`",
            f"- bootstrap 95% LCB: `{evidence['market_bootstrap']['lower_confidence_bound']}`",
            f"- blockers: `{', '.join(report['manual_promotion_review_blocking_reason_codes']) or 'none'}`",
            "- future result used for promotion decision only: `true`",
            "- paper/live/handoff allowed: `false`",
            "",
        ]
    )


def _handoff_markdown(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 paper candidate handoff plan",
            "",
            f"- research candidate promoted: `{str(plan['research_candidate_promoted']).lower()}`",
            "- separate paper candidate gate required: `true`",
            "- paper candidate allowed: `false`",
            "- execution handoff allowed: `false`",
            "",
        ]
    )
