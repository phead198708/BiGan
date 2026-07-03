"""Paper-candidate unlock and paper-only internal loop for v8 O handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)

O_V8_PAPER_CANDIDATE_UNLOCK_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-candidate-unlock-v1"
)
O_V8_PAPER_INTERNAL_EXECUTION_LOOP_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-internal-execution-loop-v1"
)
O_V8_PAPER_FILL_SIMULATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fill-simulation-v1"
)
O_V8_PAPER_RUNTIME_SAFETY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-runtime-safety-v1"
)
O_V8_PAPER_CANDIDATE_UNLOCK_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-candidate-unlock-manifest-v1"
)

PINNED_ISSUE_159_RUN_ID = (
    "o-v8-future-holdout-diversified7-eval-20260703T061730Z-"
    "20260703T065518Z"
)
PINNED_ISSUE_159_HASHES: dict[str, str] = {
    "source_manifest": (
        "fbb3e563d4dc6fc04452b8379a464c9733931f46c1842a4259c42ff4eede1a44"
    ),
    "raw_collection_manifest": (
        "bf9a2cb06235d70976a8ec827a3a12711b35644ba88f1bf624dea28ec9e223e4"
    ),
    "execution_replay_report": (
        "742ea0bd1abd53ad13a5f4f1373b7d9ef59c19430c65cadfdb216c64d662e107"
    ),
    "policy_readiness_report": (
        "10cd7a8e61d14f3a90109d15d39f5544f910f04bea78bab16ebfd9963cd26865"
    ),
    "handoff_gate_report": (
        "7e36066eb913d8a68fa2081b44bbabd0cae4ccaefe82cec96bff478c0fa59bdd"
    ),
    "paper_candidate_gate_report": (
        "bbbd1c4e09fb67ff4814120ec5e1e94428935e6705283e2a22b9e33c24b6d115"
    ),
}
PINNED_ISSUE_159_ARTIFACT_FILENAMES: dict[str, str] = {
    "source_manifest": "o_replay_aligned_source_ranking_manifest.json",
    "raw_collection_manifest": (
        "o_v8_future_unseen_holdout_raw_collection_manifest.json"
    ),
    "execution_replay_report": "o_v8_future_unseen_holdout_execution_replay_report.json",
    "policy_readiness_report": (
        "o_v8_future_unseen_holdout_policy_readiness_report.json"
    ),
    "handoff_gate_report": "o_v8_future_unseen_holdout_handoff_gate_report.json",
    "paper_candidate_gate_report": (
        "o_v8_future_unseen_holdout_paper_candidate_gate_report.json"
    ),
}

O_V8_PAPER_CANDIDATE_REQUIRED_CHECKS: tuple[str, ...] = (
    "pinned_issue_159_artifact_hashes_match",
    "future_unseen_holdout_raw_collection_ready",
    "future_window_time_validation_passed",
    "no_overlap_with_prior_validation_shadow_replay_source_data",
    "zero_missing_runtime_fields",
    "zero_provenance_violations",
    "min_allowed_order_count_met",
    "future_unseen_holdout_policy_readiness_passed",
    "future_unseen_holdout_handoff_gate_passed",
    "future_unseen_holdout_paper_candidate_gate_passed",
    "no_threshold_tuning",
    "no_forbidden_outcome_usage",
    "no_source_score_mutation",
    "manual_approval_explicit_and_hashable",
    "live_safety_flags_remain_blocked",
)

O_V8_PAPER_INTERNAL_LOOP_REQUIRED_CHECKS: tuple[str, ...] = (
    "paper_candidate_allowed",
    "paper_order_intent_rows_present",
    "accepted_intents_have_safe_runtime_quality",
    "accepted_intents_have_p_up_agreement",
    "accepted_intents_have_safe_origin",
    "paper_fills_are_deterministic",
    "paper_ledger_updates_only_accepted_intents",
    "no_polymarket_write_or_wallet_signing",
    "source_freeze_promotion_remain_blocked",
)

_FALSE_SAFETY_FIELDS = (
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
)


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperCandidateUnlockConfig:
    """Configuration for the #160 paper-only unlock evidence."""

    run_id: str
    output_dir: Path | str
    issue_159_eval_dir: Path | str
    manual_approval_approved: bool
    manual_approval_id: str
    manual_approval_operator: str
    manual_approval_scope: str = "v8_o_paper_candidate_internal_loop_only"
    expected_issue_159_hashes: dict[str, str] | None = None
    paper_order_notional: float = 0.2
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.manual_approval_id.strip():
            raise ValueError("manual_approval_id is required")
        if not self.manual_approval_operator.strip():
            raise ValueError("manual_approval_operator is required")
        if self.paper_order_notional <= 0.0:
            raise ValueError("paper_order_notional must be positive")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        if self.polymarket_write_enabled is not False:
            raise ValueError("polymarket_write_enabled must be false")
        if self.wallet_signing_enabled is not False:
            raise ValueError("wallet_signing_enabled must be false")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "issue_159_eval_dir", Path(self.issue_159_eval_dir))


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperCandidateUnlockResult:
    """Generated #160 report bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    paper_candidate_unlock_report: dict[str, Any]
    paper_internal_execution_loop_report: dict[str, Any]
    paper_fill_simulation_report: dict[str, Any]
    paper_runtime_safety_report: dict[str, Any]
    manifest: dict[str, Any]


def run_polymarket_o_v8_paper_candidate_unlock(
    config: PolymarketOV8PaperCandidateUnlockConfig,
) -> PolymarketOV8PaperCandidateUnlockResult:
    """Validate #159 evidence and build the local paper-only internal loop."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"paper candidate unlock output_dir already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    input_artifacts = _load_issue_159_artifacts(config)
    issue_159_reports = {
        name: _read_json(path)
        for name, path in sorted(input_artifacts["paths"].items())
    }
    allowed_rows = _allowed_order_rows(issue_159_reports["execution_replay_report"])

    unlock_report = _paper_candidate_unlock_report(
        config=config,
        input_artifacts=input_artifacts,
        reports=issue_159_reports,
        allowed_rows=allowed_rows,
    )
    paper_candidate_allowed = bool(unlock_report["paper_candidate_allowed"])
    intents = _paper_order_intents(
        config=config,
        allowed_rows=allowed_rows,
        enabled=paper_candidate_allowed,
    )
    fills = _paper_fills_from_intents(intents)
    ledger_rows = _paper_ledger_from_fills(fills)

    loop_report = _paper_internal_execution_loop_report(
        config=config,
        unlock_report=unlock_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    fill_report = _paper_fill_simulation_report(config=config, fills=fills)
    safety_report = _paper_runtime_safety_report(
        config=config,
        unlock_report=unlock_report,
        loop_report=loop_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )

    artifact_paths = {
        "paper_candidate_unlock_report": (
            output_dir / "o_v8_paper_candidate_unlock_report.json"
        ),
        "paper_candidate_unlock_summary": (
            output_dir / "o_v8_paper_candidate_unlock_report.md"
        ),
        "paper_internal_execution_loop_report": (
            output_dir / "o_v8_paper_internal_execution_loop_report.json"
        ),
        "paper_internal_execution_loop_summary": (
            output_dir / "o_v8_paper_internal_execution_loop_report.md"
        ),
        "paper_order_intent_log": output_dir / "o_v8_paper_order_intent_log.jsonl",
        "paper_fill_simulation_report": (
            output_dir / "o_v8_paper_fill_simulation_report.json"
        ),
        "paper_fill_simulation_summary": (
            output_dir / "o_v8_paper_fill_simulation_report.md"
        ),
        "paper_runtime_safety_report": (
            output_dir / "o_v8_paper_runtime_safety_report.json"
        ),
        "paper_runtime_safety_summary": (
            output_dir / "o_v8_paper_runtime_safety_report.md"
        ),
        "manifest": output_dir / "o_v8_paper_candidate_unlock_manifest.json",
    }

    _write_json(artifact_paths["paper_candidate_unlock_report"], unlock_report)
    _write_text(
        artifact_paths["paper_candidate_unlock_summary"],
        _paper_candidate_unlock_markdown(unlock_report),
    )
    _write_json(artifact_paths["paper_internal_execution_loop_report"], loop_report)
    _write_text(
        artifact_paths["paper_internal_execution_loop_summary"],
        _paper_internal_execution_loop_markdown(loop_report),
    )
    _write_jsonl(artifact_paths["paper_order_intent_log"], intents)
    _write_json(artifact_paths["paper_fill_simulation_report"], fill_report)
    _write_text(
        artifact_paths["paper_fill_simulation_summary"],
        _paper_fill_simulation_markdown(fill_report),
    )
    _write_json(artifact_paths["paper_runtime_safety_report"], safety_report)
    _write_text(
        artifact_paths["paper_runtime_safety_summary"],
        _paper_runtime_safety_markdown(safety_report),
    )

    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "manifest"
    }
    manifest = _paper_candidate_unlock_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        input_artifacts=input_artifacts,
        unlock_report=unlock_report,
        loop_report=loop_report,
        fill_report=fill_report,
        safety_report=safety_report,
    )
    _write_json(artifact_paths["manifest"], manifest)
    artifact_hashes["manifest"] = _sha256_file(artifact_paths["manifest"])

    return PolymarketOV8PaperCandidateUnlockResult(
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        paper_candidate_unlock_report=unlock_report,
        paper_internal_execution_loop_report=loop_report,
        paper_fill_simulation_report=fill_report,
        paper_runtime_safety_report=safety_report,
        manifest=manifest,
    )


def _paper_candidate_unlock_report(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    input_artifacts: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    allowed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manual_payload = _manual_approval_payload(config)
    checks = _paper_candidate_unlock_checks(
        config=config,
        input_artifacts=input_artifacts,
        reports=reports,
        allowed_rows=allowed_rows,
        manual_payload=manual_payload,
    )
    blockers = _blocking_reason_codes(checks)
    report = {
        "schema_version": O_V8_PAPER_CANDIDATE_UNLOCK_SCHEMA_VERSION,
        "report_type": "o_v8_paper_candidate_unlock",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "source_lineage": "O_replay_aligned_source_ranking_candidate",
        "pinned_issue": "#159",
        "pinned_issue_159_run_id": PINNED_ISSUE_159_RUN_ID,
        "diagnostic_only": False,
        "simulation_only": True,
        "paper_candidate_required_checks": checks,
        "paper_candidate_blocking_reason_codes": blockers,
        "paper_candidate_allowed": blockers == [],
        "paper_candidate_allowed_scope": "local_paper_internal_loop_only",
        "paper_internal_execution_loop_candidate": blockers == [],
        "paper_order_intent_log_enabled": blockers == [],
        "paper_fill_simulation_enabled": blockers == [],
        "manual_approval_payload": manual_payload,
        "manual_approval_hash": canonical_json_sha256(manual_payload),
        "pinned_artifact_paths": {
            name: str(path) for name, path in sorted(input_artifacts["paths"].items())
        },
        "pinned_artifact_expected_hashes": input_artifacts["expected_hashes"],
        "pinned_artifact_observed_hashes": input_artifacts["observed_hashes"],
        "pinned_artifact_hashes_verified": bool(
            checks["pinned_issue_159_artifact_hashes_match"]["passed"]
        ),
        "future_unseen_holdout_raw_collection_ready": bool(
            reports["raw_collection_manifest"].get(
                "future_unseen_holdout_raw_collection_ready"
            )
        ),
        "future_window_time_validation_passed": bool(
            reports["raw_collection_manifest"].get(
                "future_window_time_validation_passed"
            )
        ),
        "zero_missing_runtime_fields": bool(
            reports["execution_replay_report"].get("zero_missing_runtime_fields")
        ),
        "zero_provenance_violations": bool(
            reports["execution_replay_report"].get("zero_provenance_violations")
        ),
        "simulated_allowed_order_count": len(allowed_rows),
        "min_allowed_order_count": _min_allowed_order_count(reports),
        "future_unseen_holdout_policy_readiness_passed": bool(
            reports["policy_readiness_report"].get(
                "future_unseen_holdout_policy_readiness_passed"
            )
        ),
        "future_unseen_holdout_handoff_gate_passed": bool(
            reports["handoff_gate_report"].get(
                "future_unseen_holdout_handoff_gate_passed"
            )
        ),
        "future_unseen_holdout_paper_candidate_gate_passed": bool(
            reports["paper_candidate_gate_report"].get(
                "future_unseen_holdout_paper_candidate_gate_passed"
            )
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "v8_paper_internal_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_candidate_unlock_report_id")


def _paper_candidate_unlock_checks(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    input_artifacts: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    allowed_rows: list[dict[str, Any]],
    manual_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_checks = reports["raw_collection_manifest"].get(
        "future_unseen_holdout_raw_collection_required_checks", {}
    )
    no_overlap_passed = bool(
        raw_checks.get("no_overlap_with_prior_replay_validation_shadow", {}).get(
            "passed"
        )
    )
    source_freeze_blocked = all(
        _all_false_safety_flags(report)
        for report in (
            reports["source_manifest"],
            reports["raw_collection_manifest"],
            reports["execution_replay_report"],
            reports["policy_readiness_report"],
            reports["handoff_gate_report"],
            reports["paper_candidate_gate_report"],
        )
    )
    no_threshold_tuning = all(
        report.get("thresholds_tuned", False) is False
        and report.get("uses_validation_outcomes_for_tuning", False) is False
        for report in reports.values()
    )
    no_forbidden_outcomes = all(
        report.get("uses_realized_pnl_or_labels_for_analysis", False) is False
        and report.get("uses_oracle_actions_for_analysis", False) is False
        and report.get("forbidden_outcome_fields_used", []) == []
        for report in reports.values()
    )
    no_source_mutation = all(
        report.get("mutates_o_model_predicted_score", False) is False
        and report.get("mutates_source_ranking_scores", False) is False
        for report in reports.values()
    )
    live_safety = (
        config.paper_only is True
        and config.capital_at_risk is False
        and config.polymarket_write_enabled is False
        and config.wallet_signing_enabled is False
        and source_freeze_blocked
    )
    min_allowed = _min_allowed_order_count(reports)
    checks = {
        "pinned_issue_159_artifact_hashes_match": _check(
            passed=input_artifacts["hashes_match"],
            reason_code="pinned_issue_159_artifact_hash_mismatch",
            observed=input_artifacts["observed_hashes"],
            required=input_artifacts["expected_hashes"],
        ),
        "future_unseen_holdout_raw_collection_ready": _check(
            passed=reports["raw_collection_manifest"].get(
                "future_unseen_holdout_raw_collection_ready"
            )
            is True,
            reason_code="future_unseen_holdout_raw_collection_not_ready",
            observed=reports["raw_collection_manifest"].get(
                "future_unseen_holdout_raw_collection_ready"
            ),
            required=True,
        ),
        "future_window_time_validation_passed": _check(
            passed=reports["raw_collection_manifest"].get(
                "future_window_time_validation_passed"
            )
            is True,
            reason_code="future_window_time_validation_failed",
            observed=reports["raw_collection_manifest"].get(
                "future_window_time_validation_passed"
            ),
            required=True,
        ),
        "no_overlap_with_prior_validation_shadow_replay_source_data": _check(
            passed=no_overlap_passed,
            reason_code="future_holdout_overlap_with_prior_data",
            observed=raw_checks.get(
                "no_overlap_with_prior_replay_validation_shadow", {}
            ).get("observed"),
            required="zero overlap",
        ),
        "zero_missing_runtime_fields": _check(
            passed=reports["execution_replay_report"].get(
                "zero_missing_runtime_fields"
            )
            is True,
            reason_code="future_holdout_missing_runtime_fields",
            observed=reports["execution_replay_report"].get(
                "zero_missing_runtime_fields"
            ),
            required=True,
        ),
        "zero_provenance_violations": _check(
            passed=reports["execution_replay_report"].get(
                "zero_provenance_violations"
            )
            is True,
            reason_code="future_holdout_provenance_violations",
            observed=reports["execution_replay_report"].get(
                "zero_provenance_violations"
            ),
            required=True,
        ),
        "min_allowed_order_count_met": _check(
            passed=len(allowed_rows) >= min_allowed,
            reason_code="future_holdout_allowed_order_support_insufficient",
            observed=len(allowed_rows),
            required=min_allowed,
        ),
        "future_unseen_holdout_policy_readiness_passed": _check(
            passed=reports["policy_readiness_report"].get(
                "future_unseen_holdout_policy_readiness_passed"
            )
            is True,
            reason_code="future_holdout_policy_readiness_failed",
            observed=reports["policy_readiness_report"].get(
                "future_unseen_holdout_policy_readiness_passed"
            ),
            required=True,
        ),
        "future_unseen_holdout_handoff_gate_passed": _check(
            passed=reports["handoff_gate_report"].get(
                "future_unseen_holdout_handoff_gate_passed"
            )
            is True,
            reason_code="future_holdout_handoff_gate_failed",
            observed=reports["handoff_gate_report"].get(
                "future_unseen_holdout_handoff_gate_passed"
            ),
            required=True,
        ),
        "future_unseen_holdout_paper_candidate_gate_passed": _check(
            passed=reports["paper_candidate_gate_report"].get(
                "future_unseen_holdout_paper_candidate_gate_passed"
            )
            is True,
            reason_code="future_holdout_paper_candidate_gate_failed",
            observed=reports["paper_candidate_gate_report"].get(
                "future_unseen_holdout_paper_candidate_gate_passed"
            ),
            required=True,
        ),
        "no_threshold_tuning": _check(
            passed=no_threshold_tuning,
            reason_code="paper_candidate_threshold_tuning_detected",
            observed=no_threshold_tuning,
            required=True,
        ),
        "no_forbidden_outcome_usage": _check(
            passed=no_forbidden_outcomes,
            reason_code="paper_candidate_forbidden_outcome_usage_detected",
            observed=no_forbidden_outcomes,
            required=True,
        ),
        "no_source_score_mutation": _check(
            passed=no_source_mutation,
            reason_code="paper_candidate_source_score_mutation_detected",
            observed=no_source_mutation,
            required=True,
        ),
        "manual_approval_explicit_and_hashable": _check(
            passed=bool(
                config.manual_approval_approved
                and config.manual_approval_id.strip()
                and config.manual_approval_operator.strip()
                and canonical_json_sha256(manual_payload)
            ),
            reason_code="manual_approval_required_before_paper_candidate",
            observed=manual_payload,
            required="approved=true with id/operator/scope hash",
        ),
        "live_safety_flags_remain_blocked": _check(
            passed=live_safety,
            reason_code="paper_candidate_live_safety_flags_not_blocked",
            observed={
                "paper_only": config.paper_only,
                "capital_at_risk": config.capital_at_risk,
                "polymarket_write_enabled": config.polymarket_write_enabled,
                "wallet_signing_enabled": config.wallet_signing_enabled,
                "source_freeze_promotion_remain_blocked": source_freeze_blocked,
            },
            required={
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "source_freeze_promotion_remain_blocked": True,
            },
        ),
    }
    return {name: checks[name] for name in O_V8_PAPER_CANDIDATE_REQUIRED_CHECKS}


def _paper_order_intents(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    allowed_rows: list[dict[str, Any]],
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    intents: list[dict[str, Any]] = []
    for index, row in enumerate(allowed_rows, start=1):
        micro = dict(row.get("microstructure_snapshot") or {})
        intent = {
            "paper_order_intent_id": f"{config.run_id}-intent-{index:06d}",
            "simulated_order_id": row.get("simulated_order_id"),
            "decision_group_id": row.get("decision_group_id"),
            "market_id": row.get("market_id"),
            "decision_ts": row.get("decision_ts"),
            "source_selected_action": row.get("source_selected_action"),
            "source_selected_family": row.get("source_selected_family"),
            "source_selected_side": row.get("source_selected_side"),
            "execution_guarded_action": row.get("execution_guarded_action"),
            "execution_guarded_family": row.get("execution_guarded_family"),
            "execution_guarded_side": row.get("execution_guarded_side"),
            "source_model_score": _float(row.get("source_model_score")),
            "execution_guarded_score": _float(row.get("execution_guarded_score")),
            "source_raw_model_score": _float(row.get("source_raw_model_score")),
            "p_up": _float(row.get("p_up")),
            "p_down": _float(row.get("p_down")),
            "p_up_action_disagreement": bool(row.get("p_up_action_disagreement")),
            "order_origin": row.get("order_origin", "original_selected_action"),
            "came_from_original_selected_action": bool(
                row.get("came_from_original_selected_action", True)
            ),
            "proposed_order_size": _float(row.get("proposed_order_size")),
            "paper_order_notional": config.paper_order_notional,
            "paper_order_size": min(
                _float(row.get("proposed_order_size")),
                config.paper_order_notional,
            ),
            "paper_limit_price": _fill_price_from_microstructure(micro),
            "spread_bps": _float(micro.get("spread_bps")),
            "book_staleness_ms": _float(micro.get("book_staleness_ms")),
            "queue_fill_proxy": _float(micro.get("queue_fill_proxy")),
            "time_to_close_seconds": _float(micro.get("time_to_close_seconds")),
            "entry_ask": _float(micro.get("entry_ask")),
            "executable_exit_bid_proxy": _float(
                micro.get("executable_exit_bid_proxy")
            ),
            "pre_decision_exposure_state": row.get("pre_decision_exposure_state"),
            "post_decision_exposure_state": row.get("post_decision_exposure_state"),
            "execution_guard_reason_codes": row.get("execution_guard_reason_codes", []),
            "execution_blocking_reason_codes": row.get(
                "execution_blocking_reason_codes", []
            ),
            "sizing_reason_codes": row.get("sizing_reason_codes", []),
            "paper_order_intent_status": "accepted_for_internal_paper_loop",
            "order_intent_contract": "local_paper_intent_no_exchange_write_v1",
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        intent["paper_order_intent_hash"] = canonical_json_sha256(intent)
        intents.append(intent)
    return intents


def _paper_fills_from_intents(intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    for index, intent in enumerate(intents, start=1):
        fill_size = _float(intent.get("paper_order_size"))
        fill_price = _float(intent.get("paper_limit_price"))
        queue_fill = _float(intent.get("queue_fill_proxy"))
        spread_cost = fill_size * _float(intent.get("spread_bps")) / 10_000.0
        fill = {
            "paper_fill_id": f"paper-fill-{index:06d}",
            "paper_order_intent_id": intent["paper_order_intent_id"],
            "simulated_order_id": intent.get("simulated_order_id"),
            "market_id": intent.get("market_id"),
            "decision_ts": intent.get("decision_ts"),
            "execution_guarded_action": intent.get("execution_guarded_action"),
            "execution_guarded_family": intent.get("execution_guarded_family"),
            "execution_guarded_side": intent.get("execution_guarded_side"),
            "fill_simulation_status": "paper_filled",
            "fill_simulation_rule_id": "deterministic_queue_fill_proxy_v1",
            "requested_size": fill_size,
            "filled_size": fill_size,
            "fill_probability": queue_fill,
            "paper_fill_price": fill_price,
            "spread_cost": spread_cost,
            "fee_cost": 0.0,
            "slippage_cost": 0.0,
            "liquidity_impact_cost": 0.0,
            "total_execution_cost": spread_cost,
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "synthetic_paper_cash_delta": -(fill_size * fill_price + spread_cost),
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        fill["paper_fill_hash"] = canonical_json_sha256(fill)
        fills.append(fill)
    return fills


def _paper_ledger_from_fills(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cash = 10_000.0
    exposure_by_side = {"UP": 0.0, "DOWN": 0.0, "NONE": 0.0}
    exposure_by_market: dict[str, float] = {}
    ledger_rows: list[dict[str, Any]] = []
    for index, fill in enumerate(fills, start=1):
        side = str(fill.get("execution_guarded_side") or "NONE")
        market_id = str(fill.get("market_id") or "")
        size = _float(fill.get("filled_size"))
        cash_before = cash
        cash += _float(fill.get("synthetic_paper_cash_delta"))
        exposure_by_side[side] = exposure_by_side.get(side, 0.0) + size
        exposure_by_market[market_id] = exposure_by_market.get(market_id, 0.0) + size
        row = {
            "paper_ledger_entry_id": f"paper-ledger-{index:06d}",
            "paper_fill_id": fill["paper_fill_id"],
            "paper_order_intent_id": fill["paper_order_intent_id"],
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": side,
            "cash_before": cash_before,
            "cash_after": cash,
            "synthetic_position_after": exposure_by_market[market_id],
            "total_exposure_after": sum(exposure_by_market.values()),
            "side_exposure_after": exposure_by_side[side],
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["paper_ledger_entry_hash"] = canonical_json_sha256(row)
        ledger_rows.append(row)
    return ledger_rows


def _paper_internal_execution_loop_report(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    unlock_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "paper_candidate_allowed": _check(
            passed=unlock_report["paper_candidate_allowed"] is True,
            reason_code="paper_candidate_unlock_not_allowed",
            observed=unlock_report["paper_candidate_allowed"],
            required=True,
        ),
        "paper_order_intent_rows_present": _check(
            passed=len(intents) > 0,
            reason_code="paper_order_intent_rows_missing",
            observed=len(intents),
            required=">0",
        ),
        "accepted_intents_have_safe_runtime_quality": _check(
            passed=all(_intent_runtime_quality_passed(intent) for intent in intents),
            reason_code="paper_intent_runtime_quality_failed",
            observed=_intent_quality_summary(intents),
            required="spread/staleness/queue/time_to_close present and positive",
        ),
        "accepted_intents_have_p_up_agreement": _check(
            passed=all(
                intent.get("p_up_action_disagreement") is False for intent in intents
            ),
            reason_code="paper_intent_p_up_disagreement_detected",
            observed=Counter(
                str(intent.get("p_up_action_disagreement")) for intent in intents
            ),
            required=False,
        ),
        "accepted_intents_have_safe_origin": _check(
            passed=all(
                intent.get("came_from_original_selected_action") is True
                or intent.get("order_origin") == "safe_downgrade"
                for intent in intents
            ),
            reason_code="paper_intent_unsafe_origin_detected",
            observed=Counter(str(intent.get("order_origin")) for intent in intents),
            required="original_selected_action or explicit safe_downgrade",
        ),
        "paper_fills_are_deterministic": _check(
            passed=len(fills) == len(intents)
            and all(fill.get("fill_simulation_rule_id") for fill in fills),
            reason_code="paper_fill_simulation_not_deterministic",
            observed={"intent_count": len(intents), "fill_count": len(fills)},
            required="one deterministic fill per accepted intent",
        ),
        "paper_ledger_updates_only_accepted_intents": _check(
            passed=len(ledger_rows) == len(intents)
            and {row["paper_order_intent_id"] for row in ledger_rows}
            == {intent["paper_order_intent_id"] for intent in intents},
            reason_code="paper_ledger_updates_unaccepted_intents",
            observed={
                "intent_count": len(intents),
                "ledger_entry_count": len(ledger_rows),
            },
            required="ledger ids equal accepted intent ids",
        ),
        "no_polymarket_write_or_wallet_signing": _check(
            passed=config.polymarket_write_enabled is False
            and config.wallet_signing_enabled is False
            and all(
                row.get("polymarket_write_enabled") is False
                and row.get("wallet_signing_enabled") is False
                for row in [*intents, *fills, *ledger_rows]
            ),
            reason_code="paper_internal_loop_write_or_wallet_enabled",
            observed={
                "polymarket_write_enabled": config.polymarket_write_enabled,
                "wallet_signing_enabled": config.wallet_signing_enabled,
            },
            required=False,
        ),
        "source_freeze_promotion_remain_blocked": _check(
            passed=unlock_report["source_model_candidate_eligible"] is False
            and unlock_report["freeze_ready"] is False
            and unlock_report["promotion_evidence_eligible"] is False
            and unlock_report["#146_start_allowed"] is False
            and unlock_report["#134_resume_allowed"] is False,
            reason_code="source_freeze_promotion_unexpectedly_unlocked",
            observed={
                "source_model_candidate_eligible": unlock_report[
                    "source_model_candidate_eligible"
                ],
                "freeze_ready": unlock_report["freeze_ready"],
                "promotion_evidence_eligible": unlock_report[
                    "promotion_evidence_eligible"
                ],
                "#146_start_allowed": unlock_report["#146_start_allowed"],
                "#134_resume_allowed": unlock_report["#134_resume_allowed"],
            },
            required=False,
        ),
    }
    blockers = _blocking_reason_codes(checks)
    report = {
        "schema_version": O_V8_PAPER_INTERNAL_EXECUTION_LOOP_SCHEMA_VERSION,
        "report_type": "o_v8_paper_internal_execution_loop",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_candidate_unlock_report_id": unlock_report[
            "o_v8_paper_candidate_unlock_report_id"
        ],
        "paper_internal_execution_loop_required_checks": checks,
        "paper_internal_execution_loop_blocking_reason_codes": blockers,
        "paper_internal_execution_loop_enabled": blockers == [],
        "v8_paper_internal_handoff_allowed": blockers == [],
        "v8_execution_handoff_allowed": False,
        "paper_order_intent_log_enabled": blockers == [],
        "paper_fill_simulation_enabled": blockers == [],
        "paper_order_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "paper_ledger_entry_count": len(ledger_rows),
        "action_distribution": Counter(
            str(intent.get("execution_guarded_action")) for intent in intents
        ),
        "side_distribution": Counter(
            str(intent.get("execution_guarded_side")) for intent in intents
        ),
        "family_distribution": Counter(
            str(intent.get("execution_guarded_family")) for intent in intents
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_internal_execution_loop_report_id")


def _paper_fill_simulation_report(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    total_size = sum(_float(fill.get("filled_size")) for fill in fills)
    total_cost = sum(_float(fill.get("total_execution_cost")) for fill in fills)
    report = {
        "schema_version": O_V8_PAPER_FILL_SIMULATION_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fill_simulation",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fill_simulation_enabled": bool(fills),
        "fill_count": len(fills),
        "filled_size_sum": total_size,
        "deterministic_fill_rule_ids": sorted(
            {str(fill.get("fill_simulation_rule_id")) for fill in fills}
        ),
        "mean_fill_probability": (
            sum(_float(fill.get("fill_probability")) for fill in fills) / len(fills)
            if fills
            else 0.0
        ),
        "total_synthetic_execution_cost": total_cost,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "outcome_pnl_used": False,
        "realized_pnl_used": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "capital_at_risk": False,
        "paper_only": True,
    }
    return _with_report_id(report, "o_v8_paper_fill_simulation_report_id")


def _paper_runtime_safety_report(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    unlock_report: dict[str, Any],
    loop_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    safety_checks = {
        "paper_only_true": _check(
            passed=config.paper_only is True
            and all(row.get("paper_only") is True for row in [*intents, *fills, *ledger_rows]),
            reason_code="paper_runtime_not_paper_only",
            observed=True,
            required=True,
        ),
        "capital_at_risk_false": _check(
            passed=config.capital_at_risk is False
            and all(
                row.get("capital_at_risk") is False
                for row in [*intents, *fills, *ledger_rows]
            ),
            reason_code="paper_runtime_capital_at_risk",
            observed=False,
            required=False,
        ),
        "polymarket_writes_disabled": _check(
            passed=config.polymarket_write_enabled is False
            and all(
                row.get("polymarket_write_enabled") is False
                for row in [*intents, *fills, *ledger_rows]
            ),
            reason_code="paper_runtime_polymarket_write_enabled",
            observed=False,
            required=False,
        ),
        "wallet_signing_disabled": _check(
            passed=config.wallet_signing_enabled is False
            and all(
                row.get("wallet_signing_enabled") is False
                for row in [*intents, *fills, *ledger_rows]
            ),
            reason_code="paper_runtime_wallet_signing_enabled",
            observed=False,
            required=False,
        ),
        "paper_live_handoff_remains_blocked": _check(
            passed=loop_report["v8_execution_handoff_allowed"] is False
            and unlock_report["#134_resume_allowed"] is False
            and unlock_report["#146_start_allowed"] is False,
            reason_code="paper_runtime_live_handoff_unexpectedly_unlocked",
            observed={
                "v8_execution_handoff_allowed": loop_report[
                    "v8_execution_handoff_allowed"
                ],
                "#134_resume_allowed": unlock_report["#134_resume_allowed"],
                "#146_start_allowed": unlock_report["#146_start_allowed"],
            },
            required=False,
        ),
        "no_forbidden_outcome_usage": _check(
            passed=True,
            reason_code="paper_runtime_forbidden_outcome_usage_detected",
            observed=[],
            required=[],
        ),
    }
    blockers = _blocking_reason_codes(safety_checks)
    report = {
        "schema_version": O_V8_PAPER_RUNTIME_SAFETY_SCHEMA_VERSION,
        "report_type": "o_v8_paper_runtime_safety",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_runtime_safety_checks": safety_checks,
        "paper_runtime_safety_blocking_reason_codes": blockers,
        "paper_runtime_safety_passed": blockers == [],
        "paper_candidate_allowed": unlock_report["paper_candidate_allowed"],
        "paper_internal_execution_loop_enabled": loop_report[
            "paper_internal_execution_loop_enabled"
        ],
        "v8_paper_internal_handoff_allowed": loop_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "paper_order_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "paper_ledger_entry_count": len(ledger_rows),
        "ledger_updates_only_accepted_intents": (
            len(ledger_rows) == len(intents)
            and {row["paper_order_intent_id"] for row in ledger_rows}
            == {intent["paper_order_intent_id"] for intent in intents}
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_runtime_safety_report_id")


def _paper_candidate_unlock_manifest(
    *,
    config: PolymarketOV8PaperCandidateUnlockConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    input_artifacts: dict[str, Any],
    unlock_report: dict[str, Any],
    loop_report: dict[str, Any],
    fill_report: dict[str, Any],
    safety_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": O_V8_PAPER_CANDIDATE_UNLOCK_MANIFEST_SCHEMA_VERSION,
        "report_type": "o_v8_paper_candidate_unlock_manifest",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "source_lineage": "O_replay_aligned_source_ranking_candidate",
        "pinned_issue_159_run_id": PINNED_ISSUE_159_RUN_ID,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "input_artifact_paths": {
            name: str(path) for name, path in sorted(input_artifacts["paths"].items())
        },
        "input_artifact_expected_hashes": input_artifacts["expected_hashes"],
        "input_artifact_observed_hashes": input_artifacts["observed_hashes"],
        "input_artifact_hashes_verified": input_artifacts["hashes_match"],
        "paper_candidate_unlock_report_id": unlock_report[
            "o_v8_paper_candidate_unlock_report_id"
        ],
        "paper_internal_execution_loop_report_id": loop_report[
            "o_v8_paper_internal_execution_loop_report_id"
        ],
        "paper_fill_simulation_report_id": fill_report[
            "o_v8_paper_fill_simulation_report_id"
        ],
        "paper_runtime_safety_report_id": safety_report[
            "o_v8_paper_runtime_safety_report_id"
        ],
        "paper_candidate_allowed": unlock_report["paper_candidate_allowed"],
        "paper_internal_execution_loop_enabled": loop_report[
            "paper_internal_execution_loop_enabled"
        ],
        "v8_paper_internal_handoff_allowed": loop_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "paper_order_intent_log_enabled": loop_report[
            "paper_order_intent_log_enabled"
        ],
        "paper_fill_simulation_enabled": loop_report[
            "paper_fill_simulation_enabled"
        ],
        "paper_order_intent_count": loop_report["paper_order_intent_count"],
        "paper_fill_count": loop_report["paper_fill_count"],
        "paper_ledger_entry_count": loop_report["paper_ledger_entry_count"],
        "manual_approval_hash": unlock_report["manual_approval_hash"],
        "paper_candidate_blocking_reason_codes": unlock_report[
            "paper_candidate_blocking_reason_codes"
        ],
        "paper_internal_execution_loop_blocking_reason_codes": loop_report[
            "paper_internal_execution_loop_blocking_reason_codes"
        ],
        "paper_runtime_safety_passed": safety_report["paper_runtime_safety_passed"],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(manifest, "o_v8_paper_candidate_unlock_manifest_id")


def _load_issue_159_artifacts(
    config: PolymarketOV8PaperCandidateUnlockConfig,
) -> dict[str, Any]:
    eval_dir = Path(config.issue_159_eval_dir)
    paths = {
        name: eval_dir / filename
        for name, filename in PINNED_ISSUE_159_ARTIFACT_FILENAMES.items()
    }
    expected = dict(config.expected_issue_159_hashes or PINNED_ISSUE_159_HASHES)
    observed: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if not path.exists():
            observed[name] = "missing"
        else:
            observed[name] = _sha256_file(path)
    return {
        "paths": paths,
        "expected_hashes": expected,
        "observed_hashes": observed,
        "hashes_match": all(observed.get(name) == expected.get(name) for name in paths),
    }


def _allowed_order_rows(execution_replay_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (
        execution_replay_report.get("derived_reports", {})
        .get("simulated_order_replay", {})
        .get("simulated_decision_rows", [])
    )
    return [
        row
        for row in rows
        if row.get("order_allowed") is True
        and row.get("fail_closed") is False
        and row.get("simulated_order_id")
    ]


def _min_allowed_order_count(reports: dict[str, dict[str, Any]]) -> int:
    return int(
        reports["policy_readiness_report"].get("min_allowed_order_count")
        or reports["source_manifest"].get("future_unseen_holdout_simulated_allowed_order_count")
        or 5
    )


def _manual_approval_payload(
    config: PolymarketOV8PaperCandidateUnlockConfig,
) -> dict[str, Any]:
    return {
        "manual_approval_approved": config.manual_approval_approved,
        "manual_approval_id": config.manual_approval_id,
        "manual_approval_operator": config.manual_approval_operator,
        "manual_approval_scope": config.manual_approval_scope,
        "paper_candidate_allowed_scope": "local_paper_internal_loop_only",
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "capital_at_risk": False,
        "paper_only": True,
    }


def _all_false_safety_flags(report: dict[str, Any]) -> bool:
    return all(report.get(field_name) is False for field_name in _FALSE_SAFETY_FIELDS)


def _check(
    *,
    passed: bool,
    reason_code: str,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reason_code": reason_code,
        "observed": observed,
        "required": required,
    }


def _blocking_reason_codes(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        str(check["reason_code"])
        for check in checks.values()
        if check.get("passed") is not True
    )


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    report = dict(payload)
    report[field_name] = canonical_json_sha256(report)
    return report


def _fill_price_from_microstructure(micro: dict[str, Any]) -> float:
    entry_ask = _float(micro.get("entry_ask"))
    if entry_ask > 0.0:
        return entry_ask
    p = _float(micro.get("executable_exit_bid_proxy"))
    return p if p > 0.0 else 1.0


def _intent_runtime_quality_passed(intent: dict[str, Any]) -> bool:
    return (
        _float(intent.get("spread_bps")) >= 0.0
        and _float(intent.get("book_staleness_ms")) >= 0.0
        and _float(intent.get("queue_fill_proxy")) > 0.0
        and _float(intent.get("time_to_close_seconds")) > 0.0
    )


def _intent_quality_summary(intents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "intent_count": len(intents),
        "missing_runtime_quality_count": sum(
            1 for intent in intents if not _intent_runtime_quality_passed(intent)
        ),
        "min_time_to_close_seconds": min(
            (_float(intent.get("time_to_close_seconds")) for intent in intents),
            default=0.0,
        ),
        "max_book_staleness_ms": max(
            (_float(intent.get("book_staleness_ms")) for intent in intents),
            default=0.0,
        ),
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _paper_candidate_unlock_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Candidate Unlock",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_candidate_allowed: `{str(report['paper_candidate_allowed']).lower()}`",
            f"- simulated_allowed_order_count: `{report['simulated_allowed_order_count']}`",
            f"- min_allowed_order_count: `{report['min_allowed_order_count']}`",
            f"- manual_approval_hash: `{report['manual_approval_hash']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- polymarket_write_enabled: `{str(report['polymarket_write_enabled']).lower()}`",
            f"- wallet_signing_enabled: `{str(report['wallet_signing_enabled']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_candidate_blocking_reason_codes"]),
            "",
            "## Check Summary",
            "",
            *_markdown_check_table(report["paper_candidate_required_checks"]),
            "",
        ]
    )


def _paper_internal_execution_loop_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Internal Execution Loop",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_internal_execution_loop_enabled: `{str(report['paper_internal_execution_loop_enabled']).lower()}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_order_intent_count: `{report['paper_order_intent_count']}`",
            f"- paper_fill_count: `{report['paper_fill_count']}`",
            f"- paper_ledger_entry_count: `{report['paper_ledger_entry_count']}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_internal_execution_loop_blocking_reason_codes"]),
            "",
            "## Check Summary",
            "",
            *_markdown_check_table(report["paper_internal_execution_loop_required_checks"]),
            "",
        ]
    )


def _paper_fill_simulation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fill Simulation",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fill_simulation_enabled: `{str(report['paper_fill_simulation_enabled']).lower()}`",
            f"- fill_count: `{report['fill_count']}`",
            f"- filled_size_sum: `{report['filled_size_sum']}`",
            f"- total_synthetic_execution_cost: `{report['total_synthetic_execution_cost']}`",
            f"- outcome_pnl_used: `{str(report['outcome_pnl_used']).lower()}`",
            f"- realized_pnl_used: `{str(report['realized_pnl_used']).lower()}`",
            "",
        ]
    )


def _paper_runtime_safety_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Runtime Safety",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_runtime_safety_passed: `{str(report['paper_runtime_safety_passed']).lower()}`",
            f"- paper_candidate_allowed: `{str(report['paper_candidate_allowed']).lower()}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_only: `{str(report['paper_only']).lower()}`",
            f"- capital_at_risk: `{str(report['capital_at_risk']).lower()}`",
            f"- polymarket_write_enabled: `{str(report['polymarket_write_enabled']).lower()}`",
            f"- wallet_signing_enabled: `{str(report['wallet_signing_enabled']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_runtime_safety_blocking_reason_codes"]),
            "",
            "## Check Summary",
            "",
            *_markdown_check_table(report["paper_runtime_safety_checks"]),
            "",
        ]
    )


def _markdown_list(rows: list[str]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- `{row}`" for row in rows]


def _markdown_check_table(checks: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| check | passed | reason_code |", "| --- | --- | --- |"]
    for name, check in checks.items():
        lines.append(
            f"| `{name}` | `{str(check['passed']).lower()}` | "
            f"`{check['reason_code']}` |"
        )
    return lines


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
