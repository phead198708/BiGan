"""Bind regime and policy evidence to the winning parallel challenge candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256
from bigan.v8.polymarket.challenge_future_freeze import (
    CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION,
)
from bigan.v8.polymarket.challenge_future_post_freeze import (
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    SAFETY,
    SETTLED_INDEX_SCHEMA_VERSION,
)
from bigan.v8.polymarket.challenge_model_promotion import (
    audit_challenge_model_promotion,
    promotion_readiness_markdown,
)
from bigan.v8.polymarket.execution_policy_framework import (
    build_policy_safety_report,
    build_replay_parity_report,
    run_execution_policy_replay,
    validate_execution_policy_contract,
    validate_source_execution_compatibility,
)
from bigan.v8.polymarket.feature_completeness import (
    build_provider_health_diagnostics,
)
from bigan.v8.polymarket.regime_diagnostics import (
    REGIME_REPORT_SCHEMA_VERSION,
    assign_regime,
    build_regime_stratified_diagnostics,
    regime_diagnostics_markdown,
    validate_regime_definition_contract,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

PROMOTION_EVIDENCE_PROTOCOL_SCHEMA_VERSION = "bigan-v8-challenge-promotion-evidence-protocol-v1"
AGGREGATE_PARITY_SCHEMA_VERSION = "bigan-v8-challenge-execution-policy-replay-parity-v1"
AGGREGATE_SAFETY_SCHEMA_VERSION = "bigan-v8-challenge-execution-policy-safety-v1"
AGGREGATE_RECONCILIATION_SCHEMA_VERSION = "bigan-v8-challenge-execution-policy-reconciliation-v1"
POWERED_PAPER_GATE_SCHEMA_VERSION = "bigan-v8-challenge-powered-paper-gate-v1"
PROMOTION_EVIDENCE_MANIFEST_SCHEMA_VERSION = "bigan-v8-challenge-promotion-evidence-manifest-v1"
ALLOWED_SELECTED_CANDIDATES = {
    "v8_1_primary_no_fallback",
    "v8_3_primary_with_fallback",
}


class ChallengePromotionEvidenceError(ValueError):
    """Raised when downstream evidence does not match the winning freeze."""


@dataclass(frozen=True, slots=True)
class ChallengePromotionEvidenceConfig:
    """Hash-pinned inputs for post-evaluation diagnostics and promotion."""

    run_id: str
    output_dir: Path | str
    repository_root: Path | str
    parallel_evaluation_manifest_path: Path | str
    expected_parallel_evaluation_manifest_sha256: str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    promotion_evidence_protocol_path: Path | str
    expected_promotion_evidence_protocol_sha256: str
    regime_definition_contract_path: Path | str
    expected_regime_definition_contract_sha256: str
    execution_policy_contract_path: Path | str
    expected_execution_policy_contract_sha256: str
    policy_candidate_manifest_path: Path | str
    expected_policy_candidate_manifest_sha256: str
    source_execution_compatibility_manifest_path: Path | str
    expected_source_execution_compatibility_manifest_sha256: str
    implementation_commit: str
    generated_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.implementation_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_commit.lower()
        ):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.generated_ts <= 0:
            raise ValueError("generated_ts must be positive")
        path_fields = (
            "output_dir",
            "repository_root",
            "parallel_evaluation_manifest_path",
            "target_free_freeze_manifest_path",
            "promotion_evidence_protocol_path",
            "regime_definition_contract_path",
            "execution_policy_contract_path",
            "policy_candidate_manifest_path",
            "source_execution_compatibility_manifest_path",
        )
        for name in path_fields:
            object.__setattr__(self, name, Path(getattr(self, name)))
        hash_fields = (
            "expected_parallel_evaluation_manifest_sha256",
            "expected_target_free_freeze_manifest_sha256",
            "expected_promotion_evidence_protocol_sha256",
            "expected_regime_definition_contract_sha256",
            "expected_execution_policy_contract_sha256",
            "expected_policy_candidate_manifest_sha256",
            "expected_source_execution_compatibility_manifest_sha256",
        )
        for name in hash_fields:
            _require_sha256(str(getattr(self, name)), name=name)


def validate_challenge_promotion_evidence_protocol(
    protocol: dict[str, Any],
    *,
    regime_contract_sha256: str,
    execution_policy_contract_sha256: str,
    policy_candidate_manifest_sha256: str,
    compatibility_manifest_sha256: str,
    post_freeze_protocol_sha256: str,
    feature_missingness_contract_sha256: str,
    canonical_payload_contract_sha256: str,
) -> None:
    """Reject any post-result change to diagnostic or policy mappings."""

    expected_lineage = {
        "regime_definition_contract_sha256": regime_contract_sha256,
        "execution_policy_contract_sha256": (execution_policy_contract_sha256),
        "policy_candidate_manifest_sha256": (policy_candidate_manifest_sha256),
        "source_execution_compatibility_manifest_sha256": (compatibility_manifest_sha256),
        "challenge_future_post_freeze_protocol_sha256": (post_freeze_protocol_sha256),
        "feature_missingness_contract_sha256": (feature_missingness_contract_sha256),
        "canonical_payload_contract_sha256": (canonical_payload_contract_sha256),
    }
    expected_selected_rule = {
        "source": ("parallel_evaluation_report.multiplicity_aware_selected_candidate"),
        "allowed_candidates": [
            "v8_1_primary_no_fallback",
            "v8_3_primary_with_fallback",
        ],
        "selected_candidate_all_hard_gates_required": True,
        "candidate_or_threshold_override_allowed": False,
    }
    expected_regime_mapping = {
        "reference_return": "features.btc_return_5m",
        "realized_volatility": "features.btc_volatility_5m",
        "combined_spread_bps": "features.combined_spread_bps",
        "liquidity_depth": ("features.up_liquidity_depth_plus_down_liquidity_depth"),
        "provider_health_score": "features.provider_health_score",
        "provider_coverage_complete": ("all_provider_coverage_complete_flags_and_health_present"),
        "missing_policy_grid_context": ("causal_unknown_context_at_shared_decision_ts"),
        "outcome_or_settlement_fields_allowed": False,
        "diagnostic_only": True,
    }
    expected_policy_mapping = {
        "source_action_scores": ("v8_1_native_predicted_baseline_and_opposite_returns"),
        "uncertainty": ("absolute_predicted_baseline_minus_opposite_return"),
        "opportunity_window_id": "frozen_collector_batch_id",
        "fill_quality_score": ("minimum_up_down_queue_fill_probability_proxy"),
        "provider_health_score": ("features.provider_health_score_or_zero_if_missing"),
        "provider_features_complete": ("all_provider_coverage_complete_flags_and_health_present"),
        "kill_switch_active": False,
        "source_scores_mutated": False,
        "outcome_or_settlement_inputs_allowed": False,
    }
    expected_policy_validation = {
        "evaluate_all_preregistered_policy_fixtures": True,
        "outcome_selected_policy_allowed": False,
        "offline_and_paper_exact_decision_parity_required": True,
        "offline_and_paper_exact_risk_state_parity_required": True,
        "all_policy_safety_reports_required": True,
        "all_policy_reconciliations_required": True,
        "policy_performance_is_not_a_model_promotion_gate": True,
    }
    expected_powered = {
        "source": "exact_selected_candidate_parallel_hard_gate",
        "separate_result_selected_retest_allowed": False,
        "minimum_support_inherited": True,
        "multiplicity_adjusted_lcb_inherited": True,
        "largest_winner_removed_gates_inherited": True,
        "paper_only": True,
        "capital_at_risk": False,
    }
    checks = {
        "schema": (protocol.get("schema_version") == PROMOTION_EVIDENCE_PROTOCOL_SCHEMA_VERSION),
        "issues": protocol.get("issues") == [257, 258, 256],
        "frozen": protocol.get("frozen_before_target_access") is True,
        "lineage": protocol.get("lineage") == expected_lineage,
        "selected_model_rule": (protocol.get("selected_model_rule") == expected_selected_rule),
        "regime_context_mapping": (
            protocol.get("regime_context_mapping") == expected_regime_mapping
        ),
        "provider_health_diagnostics": (
            protocol.get("provider_health_diagnostics")
            == {
                "feature_source": "frozen_feature_rows.features",
                "decision_source": (
                    "multiplicity_aware_selected_candidate_settled_rows"
                ),
                "decision_identity_mapping": (
                    "market_id_plus_policy_grid_decision_ts"
                ),
                "feature_completeness_report_required": True,
                "missing_versus_zero_audit_report_required": True,
                "fallback_provider_health_association_report_required": True,
                "side_and_action_composition_required": True,
                "all_selected_decisions_must_match_feature_rows": True,
                "outcome_settlement_pnl_or_future_information_used": False,
                "diagnostic_only": True,
            }
        ),
        "execution_policy_input_mapping": (
            protocol.get("execution_policy_input_mapping") == expected_policy_mapping
        ),
        "execution_policy_validation": (
            protocol.get("execution_policy_validation") == expected_policy_validation
        ),
        "powered_paper_gate": (protocol.get("powered_paper_gate") == expected_powered),
        "safety": protocol.get("safety") == SAFETY,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengePromotionEvidenceError(
            "challenge promotion evidence protocol invalid: " + ",".join(blockers)
        )


def run_challenge_promotion_evidence(
    config: ChallengePromotionEvidenceConfig,
) -> dict[str, Any]:
    """Build exact-lineage downstream evidence and run the final audit."""

    paths = {
        "evaluation_manifest": (config.parallel_evaluation_manifest_path.resolve()),
        "freeze_manifest": (config.target_free_freeze_manifest_path.resolve()),
        "evidence_protocol": (config.promotion_evidence_protocol_path.resolve()),
        "regime_contract": (config.regime_definition_contract_path.resolve()),
        "execution_policy_contract": (config.execution_policy_contract_path.resolve()),
        "policy_candidate_manifest": (config.policy_candidate_manifest_path.resolve()),
        "compatibility_manifest": (config.source_execution_compatibility_manifest_path.resolve()),
    }
    pins = {
        "evaluation_manifest": (config.expected_parallel_evaluation_manifest_sha256),
        "freeze_manifest": (config.expected_target_free_freeze_manifest_sha256),
        "evidence_protocol": (config.expected_promotion_evidence_protocol_sha256),
        "regime_contract": (config.expected_regime_definition_contract_sha256),
        "execution_policy_contract": (config.expected_execution_policy_contract_sha256),
        "policy_candidate_manifest": (config.expected_policy_candidate_manifest_sha256),
        "compatibility_manifest": (config.expected_source_execution_compatibility_manifest_sha256),
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"challenge promotion {name}")

    evaluation_manifest = _load_json(paths["evaluation_manifest"])
    freeze_manifest = _load_json(paths["freeze_manifest"])
    evidence_protocol = _load_json(paths["evidence_protocol"])
    regime_contract = _load_json(paths["regime_contract"])
    execution_policy_contract = _load_json(paths["execution_policy_contract"])
    policy_manifest = _load_json(paths["policy_candidate_manifest"])
    compatibility = _load_json(paths["compatibility_manifest"])
    validate_regime_definition_contract(regime_contract)
    validate_execution_policy_contract(execution_policy_contract)
    _validate_static_lineage(
        evidence_protocol=evidence_protocol,
        regime_contract_sha256=pins["regime_contract"].lower(),
        execution_policy_contract_sha256=pins["execution_policy_contract"].lower(),
        policy_candidate_manifest_sha256=pins["policy_candidate_manifest"].lower(),
        compatibility_manifest_sha256=pins["compatibility_manifest"].lower(),
        repository_root=config.repository_root.resolve(),
    )

    (
        parallel_report,
        selected_candidate,
        candidate_rows,
        baseline_rows,
        parallel_freeze_sha256,
    ) = _validated_parallel_evaluation(
        evaluation_manifest=evaluation_manifest,
        evaluation_manifest_path=paths["evaluation_manifest"],
        freeze_manifest=freeze_manifest,
        freeze_manifest_path=paths["freeze_manifest"],
    )
    parallel_report_descriptor = evaluation_manifest["parallel_evaluation_report"]
    source_report_sha256 = str(parallel_report_descriptor["sha256"])
    source_rows, feature_rows, native_decisions = _frozen_sources(freeze_manifest)
    assignments = _regime_assignments(
        selected_candidate_rows=candidate_rows,
        source_rows=source_rows,
        feature_rows=feature_rows,
        regime_contract=regime_contract,
    )
    regime_artifacts = build_regime_stratified_diagnostics(
        assignments=assignments,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        contract=regime_contract,
    )
    regime_report = dict(regime_artifacts["regime_stratified_pnl_report"])
    regime_report.pop("report_sha256", None)
    regime_report.update(
        _common_lineage(
            evaluation_manifest=evaluation_manifest,
            selected_candidate=selected_candidate,
            parallel_freeze_sha256=parallel_freeze_sha256,
            source_report_sha256=source_report_sha256,
        )
    )
    regime_report["report_sha256"] = canonical_payload_sha256(
        regime_report,
        payload_schema_version=REGIME_REPORT_SCHEMA_VERSION,
    )
    regime_artifacts["regime_stratified_pnl_report"] = regime_report

    policy_inputs = _execution_policy_inputs(
        source_rows=source_rows,
        feature_rows=feature_rows,
        native_decisions=native_decisions,
        source_model_hash=str(compatibility["source_model_hash"]),
    )
    policy_results = _run_all_execution_policies(
        policy_inputs=policy_inputs,
        policy_manifest=policy_manifest,
        policy_manifest_path=paths["policy_candidate_manifest"],
        compatibility=compatibility,
    )
    common = _common_lineage(
        evaluation_manifest=evaluation_manifest,
        selected_candidate=selected_candidate,
        parallel_freeze_sha256=parallel_freeze_sha256,
        source_report_sha256=source_report_sha256,
    )
    provider_health_report = build_provider_health_diagnostics(
        feature_rows=feature_rows,
        decision_rows=_provider_health_diagnostic_decisions(
            selected_candidate_rows=candidate_rows,
            source_rows=source_rows,
        ),
    )
    provider_health_report.update(common)
    provider_health_report["report_id"] = canonical_payload_sha256(
        provider_health_report,
        payload_schema_version=(
            "bigan-v8-provider-health-diagnostics-v1"
        ),
    )
    if (
        provider_health_report["decision_row_count"] != 120
        or provider_health_report["matched_decision_count"] != 120
        or provider_health_report["unmatched_decision_count"] != 0
    ):
        raise ChallengePromotionEvidenceError(
            "provider-health decision identities do not reconcile"
        )
    parity_report = _aggregate_policy_report(
        schema_version=AGGREGATE_PARITY_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="parity",
    )
    safety_report = _aggregate_policy_report(
        schema_version=AGGREGATE_SAFETY_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="safety",
    )
    reconciliation_report = _aggregate_policy_report(
        schema_version=AGGREGATE_RECONCILIATION_SCHEMA_VERSION,
        common=common,
        policy_results=policy_results,
        report_key="reconciliation",
    )
    powered_paper_report = _powered_paper_gate(
        parallel_report=parallel_report,
        common=common,
    )

    run_dir = _prepare_run_dir(
        Path(config.output_dir),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    assignment_path = run_dir / "challenge_regime_assignments.jsonl"
    regime_path = run_dir / "challenge_regime_stratified_pnl_report.json"
    regime_bootstrap_path = run_dir / "challenge_regime_bootstrap_report.json"
    side_path = run_dir / "challenge_side_action_attribution_report.json"
    regime_md_path = run_dir / "challenge_regime_diagnostics.md"
    policy_inputs_path = run_dir / "challenge_execution_policy_inputs.jsonl"
    provider_health_path = (
        run_dir / "challenge_provider_health_diagnostics.json"
    )
    parity_path = run_dir / "challenge_replay_parity_report.json"
    safety_path = run_dir / "challenge_policy_safety_report.json"
    reconciliation_path = run_dir / "challenge_policy_reconciliation_report.json"
    powered_path = run_dir / "challenge_powered_paper_gate_report.json"
    _write_jsonl(assignment_path, assignments)
    _write_json(regime_path, regime_report)
    _write_json(
        regime_bootstrap_path,
        regime_artifacts["regime_bootstrap_report"],
    )
    _write_json(
        side_path,
        regime_artifacts["side_action_attribution_report"],
    )
    _write_text(regime_md_path, regime_diagnostics_markdown(regime_artifacts))
    _write_jsonl(policy_inputs_path, policy_inputs)
    _write_json(provider_health_path, provider_health_report)
    replay_descriptors = _write_policy_replays(
        run_dir=run_dir,
        policy_results=policy_results,
    )
    _write_json(parity_path, parity_report)
    _write_json(safety_path, safety_report)
    _write_json(reconciliation_path, reconciliation_report)
    _write_json(powered_path, powered_paper_report)

    settled_index = _load_json(
        Path(
            _verified_descriptor(
                evaluation_manifest["settled_index"],
                "challenge promotion settled index",
            )["path"]
        )
    )
    if settled_index.get("schema_version") != SETTLED_INDEX_SCHEMA_VERSION:
        raise ChallengePromotionEvidenceError("promotion evidence settled index schema invalid")
    attempt_descriptor = _verified_descriptor(
        settled_index["attempt_consumption_record"],
        "challenge attempt consumption record",
    )
    runtime_evidence = {
        "parallel_evaluation_report": _descriptor(Path(str(parallel_report_descriptor["path"]))),
        "regime_stratified_pnl_report": _descriptor(regime_path),
        "replay_parity_report": _descriptor(parity_path),
        "policy_safety_report": _descriptor(safety_path),
        "policy_reconciliation_report": _descriptor(reconciliation_path),
        "powered_paper_gate_report": _descriptor(powered_path),
        "provider_health_diagnostics_report": _descriptor(
            provider_health_path
        ),
        "attempt_consumption_record": attempt_descriptor,
    }
    runtime_evidence_path = run_dir / "challenge_runtime_evidence_manifest.json"
    _write_json(runtime_evidence_path, runtime_evidence)
    readiness = audit_challenge_model_promotion(
        repository_root=config.repository_root,
        runtime_evidence=runtime_evidence,
    )
    readiness.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "generated_ts": config.generated_ts,
            "fresh_attempt_id": evaluation_manifest["fresh_attempt_id"],
            "parallel_freeze_sha256": parallel_freeze_sha256,
            "parallel_evaluation_report_sha256": source_report_sha256,
            "promotion_evidence_protocol_sha256": pins["evidence_protocol"].lower(),
        }
    )
    readiness_path = run_dir / "challenge_model_promotion_readiness.json"
    readiness_md_path = run_dir / "challenge_model_promotion_readiness.md"
    _write_json(readiness_path, readiness)
    _write_text(
        readiness_md_path,
        promotion_readiness_markdown(readiness),
    )
    manifest = {
        "schema_version": PROMOTION_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "generated_ts": config.generated_ts,
        "fresh_attempt_id": evaluation_manifest["fresh_attempt_id"],
        "selected_candidate_id": selected_candidate,
        "parallel_freeze_sha256": parallel_freeze_sha256,
        "parallel_evaluation_manifest": _descriptor(paths["evaluation_manifest"]),
        "target_free_freeze_manifest": _descriptor(paths["freeze_manifest"]),
        "promotion_evidence_protocol": _descriptor(paths["evidence_protocol"]),
        "regime_definition_contract": _descriptor(paths["regime_contract"]),
        "execution_policy_contract": _descriptor(paths["execution_policy_contract"]),
        "policy_candidate_manifest": _descriptor(paths["policy_candidate_manifest"]),
        "source_execution_compatibility_manifest": _descriptor(paths["compatibility_manifest"]),
        "regime_assignments": _descriptor(assignment_path),
        "regime_stratified_pnl_report": _descriptor(regime_path),
        "regime_bootstrap_report": _descriptor(regime_bootstrap_path),
        "side_action_attribution_report": _descriptor(side_path),
        "execution_policy_inputs": _descriptor(policy_inputs_path),
        "provider_health_diagnostics_report": _descriptor(
            provider_health_path
        ),
        "policy_replays": replay_descriptors,
        "replay_parity_report": _descriptor(parity_path),
        "policy_safety_report": _descriptor(safety_path),
        "policy_reconciliation_report": _descriptor(reconciliation_path),
        "powered_paper_gate_report": _descriptor(powered_path),
        "runtime_evidence_manifest": _descriptor(runtime_evidence_path),
        "promotion_readiness_report": _descriptor(readiness_path),
        "promotion_readiness_markdown": _descriptor(readiness_md_path),
        "challenge_model_promotion_eligible": readiness["challenge_model_promotion_eligible"],
        "selected_champion_candidate": readiness["selected_champion_candidate"],
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    manifest["manifest_id"] = canonical_payload_sha256(
        manifest,
        payload_schema_version=PROMOTION_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    )
    manifest_path = run_dir / "challenge_promotion_evidence_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        readiness,
        readiness_path,
        manifest,
        manifest_path,
    )


def _validate_static_lineage(
    *,
    evidence_protocol: dict[str, Any],
    regime_contract_sha256: str,
    execution_policy_contract_sha256: str,
    policy_candidate_manifest_sha256: str,
    compatibility_manifest_sha256: str,
    repository_root: Path,
) -> None:
    config_dir = repository_root / "examples/v8/polymarket_configs"
    validate_challenge_promotion_evidence_protocol(
        evidence_protocol,
        regime_contract_sha256=regime_contract_sha256,
        execution_policy_contract_sha256=execution_policy_contract_sha256,
        policy_candidate_manifest_sha256=(policy_candidate_manifest_sha256),
        compatibility_manifest_sha256=compatibility_manifest_sha256,
        post_freeze_protocol_sha256=_sha256_file(
            config_dir / "challenge_future_post_freeze_protocol.json"
        ),
        feature_missingness_contract_sha256=_sha256_file(
            config_dir / "feature_missingness_contract.json"
        ),
        canonical_payload_contract_sha256=_sha256_file(
            config_dir / "canonical_payload_contract.json"
        ),
    )


def _validated_parallel_evaluation(
    *,
    evaluation_manifest: dict[str, Any],
    evaluation_manifest_path: Path,
    freeze_manifest: dict[str, Any],
    freeze_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    if (
        evaluation_manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA_VERSION
        or evaluation_manifest.get("target_free_freeze_manifest")
        != _descriptor(freeze_manifest_path)
        or evaluation_manifest.get("attempt_and_alpha_consumed") is not True
        or evaluation_manifest.get("future_results_used_for_tuning") is not False
        or evaluation_manifest.get("result_selected_rerun_allowed") is not False
        or any(evaluation_manifest.get(field) is not expected for field, expected in SAFETY.items())
    ):
        raise ChallengePromotionEvidenceError(
            "parallel evaluation manifest is not promotion eligible"
        )
    if (
        freeze_manifest.get("schema_version") != CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION
        or freeze_manifest.get("parallel_target_free_freeze_passed") is not True
        or freeze_manifest.get("future_target_access_allowed") is not True
    ):
        raise ChallengePromotionEvidenceError("target-free freeze manifest is invalid")
    report_path = Path(
        _verified_descriptor(
            evaluation_manifest["parallel_evaluation_report"],
            "challenge parallel evaluation report",
        )["path"]
    )
    parallel_report = _load_json(report_path)
    selected = parallel_report.get("multiplicity_aware_selected_candidate")
    selected_gate = dict((parallel_report.get("candidate_gates") or {}).get(selected) or {})
    parallel_freeze_sha256 = str(
        parallel_report.get("single_use_claim", {}).get("freeze_sha256") or ""
    )
    if (
        selected not in ALLOWED_SELECTED_CANDIDATES
        or selected_gate.get("all_hard_gates_passed") is not True
        or parallel_freeze_sha256 != evaluation_manifest.get("parallel_freeze_sha256")
        or evaluation_manifest.get("fresh_attempt_id") != freeze_manifest.get("fresh_attempt_id")
    ):
        raise ChallengePromotionEvidenceError(
            "parallel evaluation did not select an eligible challenge"
        )
    candidate_descriptors = evaluation_manifest["candidate_settled_rows"]
    candidate_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                candidate_descriptors[selected],
                "selected candidate settled rows",
            )["path"]
        )
    )
    baseline_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                candidate_descriptors["matched_frozen_v6_7"],
                "matched baseline settled rows",
            )["path"]
        )
    )
    if len(candidate_rows) != 120 or len(baseline_rows) != 120:
        raise ChallengePromotionEvidenceError("selected candidate settled grid is not exact-120")
    del evaluation_manifest_path
    return (
        parallel_report,
        str(selected),
        candidate_rows,
        baseline_rows,
        parallel_freeze_sha256,
    )


def _frozen_sources(
    freeze_manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze_manifest["shared_source_rows"],
                "challenge shared source rows",
            )["path"]
        )
    )
    feature_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze_manifest["feature_rows"],
                "challenge feature rows",
            )["path"]
        )
    )
    native_decisions = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze_manifest["v8_1_native_decisions"],
                "challenge v8.1 native decisions",
            )["path"]
        )
    )
    if (
        len(source_rows) != 120
        or len(native_decisions) != 120
        or len({str(row["market_id"]) for row in source_rows}) != 120
    ):
        raise ChallengePromotionEvidenceError("frozen promotion sources are incomplete")
    return source_rows, feature_rows, native_decisions


def _regime_assignments(
    *,
    selected_candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    regime_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    source_by_market = {str(row["market_id"]): row for row in source_rows}
    feature_by_key = {(str(row["market_id"]), int(row["decision_ts"])): row for row in feature_rows}
    assignments = []
    for decision in selected_candidate_rows:
        market_id = str(decision["market_id"])
        source = source_by_market[market_id]
        policy_ts = int(source.get("policy_grid_decision_ts") or 0)
        feature = feature_by_key.get((market_id, policy_ts))
        context = _causal_context(
            decision_ts=int(decision["decision_ts"]),
            feature_row=feature,
        )
        assignments.append(
            assign_regime(
                decision=decision,
                causal_context=context,
                contract=regime_contract,
            )
        )
    return assignments


def _provider_health_diagnostic_decisions(
    *,
    selected_candidate_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_market = {
        str(row["market_id"]): row for row in source_rows
    }
    if len(source_by_market) != len(source_rows):
        raise ChallengePromotionEvidenceError(
            "provider-health source market identities are duplicated"
        )
    decisions: list[dict[str, Any]] = []
    for selected in selected_candidate_rows:
        market_id = str(selected["market_id"])
        source = source_by_market.get(market_id)
        if source is None:
            raise ChallengePromotionEvidenceError(
                "provider-health selected decision lacks a source market"
            )
        policy_grid_decision_ts = int(
            source.get("policy_grid_decision_ts") or 0
        )
        if policy_grid_decision_ts <= 0:
            raise ChallengePromotionEvidenceError(
                "provider-health policy-grid decision timestamp is missing"
            )
        decisions.append(
            {
                **selected,
                "market_id": market_id,
                "decision_ts": policy_grid_decision_ts,
            }
        )
    return decisions


def _causal_context(
    *,
    decision_ts: int,
    feature_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if feature_row is None:
        return {
            "available_at_ts": decision_ts,
            "max_input_ts": decision_ts,
            "reference_return": None,
            "realized_volatility": None,
            "combined_spread_bps": None,
            "liquidity_depth": None,
            "provider_health_score": None,
            "provider_coverage_complete": 0,
        }
    features = dict(feature_row.get("features") or {})
    health, complete = _provider_health(features)
    up_depth = _finite_or_none(features.get("up_liquidity_depth"))
    down_depth = _finite_or_none(features.get("down_liquidity_depth"))
    liquidity = up_depth + down_depth if up_depth is not None and down_depth is not None else None
    return {
        "available_at_ts": int(feature_row.get("available_at_ts") or 0),
        "max_input_ts": int(feature_row.get("max_input_ts") or 0),
        "reference_return": _finite_or_none(features.get("btc_return_5m")),
        "realized_volatility": _finite_or_none(features.get("btc_volatility_5m")),
        "combined_spread_bps": _finite_or_none(features.get("combined_spread_bps")),
        "liquidity_depth": liquidity,
        "provider_health_score": health,
        "provider_coverage_complete": int(complete),
        "trade_tape_provider_timeout": int(bool(features.get("trade_tape_provider_timeout"))),
        "trade_tape_truncated": int(bool(features.get("trade_tape_truncated"))),
        "trade_tape_censored": int(bool(features.get("trade_tape_censored"))),
        "trade_tape_historical_backfill": int(
            str(features.get("trade_tape_collection_mode") or "") == "backfill"
        ),
    }


def _execution_policy_inputs(
    *,
    source_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    native_decisions: list[dict[str, Any]],
    source_model_hash: str,
) -> list[dict[str, Any]]:
    features_by_key = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in feature_rows
    }
    native_by_market = {str(row["market_id"]): row for row in native_decisions}
    inputs = []
    for source in source_rows:
        market_id = str(source["market_id"])
        decision_ts = int(source["decision_ts"])
        policy_ts = int(source.get("policy_grid_decision_ts") or 0)
        feature_row = features_by_key.get((market_id, policy_ts))
        features = dict(feature_row.get("features") or {}) if feature_row is not None else {}
        native = native_by_market[market_id]
        scores: dict[str, float] = {"NO_TRADE": 0.0}
        values = []
        for action_field, score_field in (
            ("baseline_action", "predicted_baseline_return"),
            ("opposite_action", "predicted_opposite_return"),
        ):
            action = str(native.get(action_field) or "")
            score = _finite_or_none(native.get(score_field))
            if action and score is not None:
                scores[action] = score
                values.append(score)
        health, provider_complete = _provider_health(features)
        fill_candidates = [
            value
            for value in (
                _finite_or_none(features.get("up_queue_fill_probability_proxy")),
                _finite_or_none(features.get("down_queue_fill_probability_proxy")),
            )
            if value is not None
        ]
        inputs.append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "source_model_hash": source_model_hash,
                "source_action_scores": scores,
                "uncertainty": (abs(values[0] - values[1]) if len(values) == 2 else 1.0),
                "opportunity_window_id": str(source.get("collector_batch_id") or "unknown_batch"),
                "fill_quality_score": (min(fill_candidates) if fill_candidates else 0.0),
                "provider_health_score": (health if health is not None else 0.0),
                "provider_features_complete": provider_complete,
                "kill_switch_active": False,
            }
        )
    return sorted(
        inputs,
        key=lambda row: (int(row["decision_ts"]), str(row["market_id"])),
    )


def _provider_health(
    features: dict[str, Any],
) -> tuple[float | None, bool]:
    health = _finite_or_none(features.get("provider_health_score"))
    coverage = [value for key, value in features.items() if "coverage_complete" in str(key)]
    complete = (
        health is not None
        and bool(coverage)
        and all(value is True or value == 1 for value in coverage)
        and not bool(features.get("trade_tape_provider_timeout"))
        and not bool(features.get("trade_tape_truncated"))
        and not bool(features.get("trade_tape_censored"))
        and str(features.get("trade_tape_collection_mode") or "") != "backfill"
    )
    return health, complete


def _run_all_execution_policies(
    *,
    policy_inputs: list[dict[str, Any]],
    policy_manifest: dict[str, Any],
    policy_manifest_path: Path,
    compatibility: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fixtures = list(policy_manifest.get("candidate_fixtures") or [])
    if (
        policy_manifest.get("candidate_count_cap") != 3
        or len(fixtures) != 3
        or policy_manifest.get("outcome_selected_candidate_enabled") is not False
    ):
        raise ChallengePromotionEvidenceError("execution policy candidate manifest is invalid")
    output = {}
    for descriptor in fixtures:
        path = policy_manifest_path.parent / str(descriptor["path"])
        _verify_pin(
            path,
            str(descriptor["raw_sha256"]),
            "challenge execution policy fixture",
        )
        policy = _load_json(path)
        if policy.get("candidate_id") != descriptor.get("candidate_id"):
            raise ChallengePromotionEvidenceError("execution policy candidate identity mismatch")
        validate_source_execution_compatibility(
            compatibility_manifest=compatibility,
            policy=policy,
            source_model_hash=str(compatibility["source_model_hash"]),
        )
        offline = run_execution_policy_replay(
            inputs=policy_inputs,
            policy=policy,
            compatibility_manifest=compatibility,
            runtime_mode="offline_replay",
        )
        paper = run_execution_policy_replay(
            inputs=policy_inputs,
            policy=policy,
            compatibility_manifest=compatibility,
            runtime_mode="paper_runtime",
        )
        parity = build_replay_parity_report(
            offline_replay=offline,
            paper_runtime=paper,
        )
        safety = build_policy_safety_report(paper)
        reconciliation = dict(paper["reconciliation_report"])
        output[str(policy["candidate_id"])] = {
            "policy_path": path,
            "offline": offline,
            "paper": paper,
            "parity": parity,
            "safety": safety,
            "reconciliation": reconciliation,
        }
    return output


def _aggregate_policy_report(
    *,
    schema_version: str,
    common: dict[str, Any],
    policy_results: dict[str, dict[str, Any]],
    report_key: str,
) -> dict[str, Any]:
    reports = {
        candidate_id: result[report_key] for candidate_id, result in sorted(policy_results.items())
    }
    passed = len(reports) == 3 and all(report.get("passed") is True for report in reports.values())
    report = {
        "schema_version": schema_version,
        **common,
        "policy_candidate_count": len(reports),
        "all_preregistered_policy_candidates_evaluated": len(reports) == 3,
        "outcome_selected_policy_used": False,
        "policy_reports": reports,
        "passed": passed,
        **SAFETY,
    }
    report["report_id"] = canonical_payload_sha256(
        report,
        payload_schema_version=schema_version,
    )
    return report


def _powered_paper_gate(
    *,
    parallel_report: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    selected = common["selected_candidate_id"]
    gate = dict(parallel_report["candidate_gates"][selected])
    checks = {
        "selected_candidate_matches_parallel_winner": (
            selected == parallel_report["multiplicity_aware_selected_candidate"]
        ),
        "selected_candidate_all_hard_gates_passed": (gate.get("all_hard_gates_passed") is True),
        "minimum_support_passed": (
            int(gate.get("accepted_bet_count") or 0) >= int(gate.get("minimum_total_support") or 0)
        ),
        "candidate_total_after_cost_pnl_positive": (
            float(gate.get("total_after_cost_pnl") or 0.0) > 0.0
        ),
        "candidate_minus_baseline_positive": (
            float(gate.get("candidate_minus_baseline_after_cost_pnl") or 0.0) > 0.0
        ),
        "multiplicity_adjusted_lcb_positive": (
            float(gate.get("candidate_minus_baseline_bootstrap_lcb") or 0.0) > 0.0
        ),
        "candidate_largest_winner_removed_positive": (
            float(gate.get("candidate_largest_winner_removed_after_cost_pnl") or 0.0) > 0.0
        ),
        "delta_largest_winner_removed_positive": (
            float(gate.get("candidate_minus_baseline_largest_winner_removed_after_cost_pnl") or 0.0)
            > 0.0
        ),
    }
    report = {
        "schema_version": POWERED_PAPER_GATE_SCHEMA_VERSION,
        **common,
        "selected_candidate_gate": gate,
        "checks": checks,
        "powered_paper_gate_passed": all(checks.values()),
        "separate_result_selected_retest_performed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
    }
    report["report_id"] = canonical_payload_sha256(
        report,
        payload_schema_version=POWERED_PAPER_GATE_SCHEMA_VERSION,
    )
    return report


def _common_lineage(
    *,
    evaluation_manifest: dict[str, Any],
    selected_candidate: str,
    parallel_freeze_sha256: str,
    source_report_sha256: str,
) -> dict[str, Any]:
    return {
        "fresh_attempt_id": evaluation_manifest["fresh_attempt_id"],
        "selected_candidate_id": selected_candidate,
        "parallel_freeze_sha256": parallel_freeze_sha256,
        "source_parallel_evaluation_report_sha256": source_report_sha256,
    }


def _write_policy_replays(
    *,
    run_dir: Path,
    policy_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for candidate_id, result in sorted(policy_results.items()):
        offline_path = run_dir / f"{candidate_id}_offline_replay.json"
        paper_path = run_dir / f"{candidate_id}_paper_runtime.json"
        _write_json(offline_path, result["offline"])
        _write_json(paper_path, result["paper"])
        output[candidate_id] = {
            "policy_fixture": _descriptor(result["policy_path"]),
            "offline_replay": _descriptor(offline_path),
            "paper_runtime": _descriptor(paper_path),
        }
    return output


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


__all__ = [
    "ChallengePromotionEvidenceConfig",
    "ChallengePromotionEvidenceError",
    "run_challenge_promotion_evidence",
    "validate_challenge_promotion_evidence_protocol",
]
