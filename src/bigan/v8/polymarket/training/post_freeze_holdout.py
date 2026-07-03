"""Post-freeze holdout validation for the frozen Polymarket M selector."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.execution_ev import run_polymarket_policy_replay
from bigan.v8.polymarket.training.action_family_eligibility import (
    M_EXECUTION_PNL_AWARE_GAP_PENALTY_WEIGHT,
    M_EXECUTION_PNL_AWARE_IMMEDIATE_EXIT_RETURN_WEIGHT,
    M_EXECUTION_PNL_AWARE_MARGIN_WEIGHT,
    M_EXECUTION_PNL_AWARE_MODEL_SCORE_WEIGHT,
    M_EXECUTION_PNL_AWARE_QUALITY_WEIGHT,
    build_sell_before_close_side_balanced_prediction_set,
)
from bigan.v8.polymarket.training.action_value_calibration import (
    apply_action_value_calibration,
)
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    PolymarketPolicyModel,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.dataset import load_polymarket_policy_dataset
from bigan.v8.polymarket.training.model import predict_polymarket_policy_examples
from bigan.v8.polymarket.training.runner import (
    _m_component_stats,
    _m_promotion_attribution_row,
    _m_promotion_attribution_summary,
    _m_rank_score_component_summary,
    _m_replay_entry_contexts,
    _m_top_negative_selected_entries,
)
from bigan.v8.polymarket.training.sell_before_close_exit_reliability import (
    build_sell_before_close_exit_reliability_guard_decisions,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m-post-freeze-holdout-validation-v1"
)
FROZEN_M_SELECTOR_BASELINE_COMMIT = "cd4559dfed2965f2a90226acdd43323d0a27677c"
FROZEN_M_SELECTOR_PARENT_COMMIT = "f35231014290b88e65970fab10193ec8acad0b49"
FROZEN_M_SELECTOR_METHOD = "position_state_aware_execution_pnl_score_ranked_per_side_quota"
FROZEN_M_RANK_SCORE_COMPONENTS = (
    "0.20*calibrated_action_score + 0.10*best_action_margin + "
    "entry_exit_quality_score + 8.00*immediate_exit_return - "
    "0.05*model_vs_immediate_exit_pnl_gap_estimate"
)


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezeHoldoutConfig:
    frozen_model_dir: Path | str
    frozen_corpus_dir: Path | str
    holdout_corpus_dir: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_m_post_freeze_holdout_validation"
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "frozen_model_dir",
            "frozen_corpus_dir",
            "holdout_corpus_dir",
            "output_dir",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                object.__setattr__(self, field_name, Path(value))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezeHoldoutResult:
    run_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_m_post_freeze_holdout_validation(
    config: PolymarketPostFreezeHoldoutConfig,
) -> PolymarketPostFreezeHoldoutResult:
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "report": run_dir / "m_post_freeze_holdout_validation_report.json",
        "summary": run_dir / "m_post_freeze_holdout_validation_report.md",
        "manifest": run_dir / "m_post_freeze_holdout_validation_manifest.json",
    }
    report = _build_post_freeze_holdout_report(config=config)
    _write_json(artifact_paths["report"], report)
    artifact_paths["summary"].write_text(
        _post_freeze_holdout_markdown(report),
        encoding="utf-8",
    )
    artifact_manifest = {
        "schema_version": "bigan-v8-polymarket-m-post-freeze-holdout-artifacts-v1",
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path.relative_to(run_dir))
            for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": {
            name: _sha256_file(path)
            for name, path in sorted(artifact_paths.items())
            if name != "manifest"
        },
        **compact_safety_fields(),
    }
    artifact_manifest["artifact_hashes"]["manifest"] = canonical_json_sha256(
        artifact_manifest
    )
    _write_json(artifact_paths["manifest"], artifact_manifest)
    return PolymarketPostFreezeHoldoutResult(
        run_dir=run_dir,
        report=report,
        artifact_paths=artifact_paths,
    )


def _build_post_freeze_holdout_report(
    *,
    config: PolymarketPostFreezeHoldoutConfig,
) -> dict[str, Any]:
    frozen_model_dir = config.frozen_model_dir.expanduser().resolve()
    frozen_corpus_dir = config.frozen_corpus_dir.expanduser().resolve()
    holdout_corpus_dir = config.holdout_corpus_dir.expanduser().resolve()
    frozen_manifest_path = frozen_model_dir / "polymarket_policy_model_manifest.json"
    frozen_model_path = frozen_model_dir / "polymarket_policy_model.json"
    frozen_calibration_path = frozen_model_dir / "polymarket_action_value_calibration.json"
    frozen_training_config_path = frozen_model_dir / "polymarket_policy_training_config.json"
    frozen_corpus_manifest_path = frozen_corpus_dir / "polymarket_corpus_manifest.json"
    frozen_split_path = frozen_corpus_dir / "polymarket_train_shadow_split.json"
    frozen_manifest = _read_json(frozen_manifest_path)
    frozen_model_payload = _read_json(frozen_model_path)
    frozen_training_config = _read_json(frozen_training_config_path)
    frozen_split = _read_json(frozen_split_path) if frozen_split_path.exists() else {}
    frozen_dataset = load_polymarket_policy_dataset(
        _dataset_config(
            corpus_dir=frozen_corpus_dir,
            output_dir=config.run_dir / "_frozen_dataset_probe",
            base_config=frozen_training_config,
        )
    )
    holdout_dataset = load_polymarket_policy_dataset(
        _dataset_config(
            corpus_dir=holdout_corpus_dir,
            output_dir=config.run_dir / "_holdout_dataset_probe",
            base_config=frozen_training_config,
        )
    )
    frozen_market_ids = {example.market_id for example in frozen_dataset.examples}
    holdout_market_ids = {example.market_id for example in holdout_dataset.examples}
    frozen_max_decision_ts = max(example.decision_ts for example in frozen_dataset.examples)
    holdout_min_decision_ts = min(example.decision_ts for example in holdout_dataset.examples)
    overlapping_market_ids = sorted(frozen_market_ids & holdout_market_ids)
    frozen_model_sha256 = _sha256_file(frozen_model_path)
    frozen_manifest_model_sha256 = frozen_manifest.get("model_sha256")
    frozen_corpus_manifest_sha256 = _sha256_file(frozen_corpus_manifest_path)
    frozen_model_training_corpus_hash = frozen_model_payload.get("training_corpus_hash")
    frozen_manifest_training_corpus_hash = frozen_manifest.get("training_corpus_hash")
    frozen_manifest_phase2_corpus_manifest_sha256 = frozen_manifest.get(
        "phase2_corpus_manifest_sha256"
    )
    frozen_manifest_policy_dataset_hash = frozen_manifest.get("policy_dataset_hash")
    frozen_manifest_split_hash = frozen_manifest.get("split_hash")
    frozen_split_hash = frozen_split.get("split_hash")
    frozen_model_sha256_matches_manifest = (
        frozen_model_sha256 == frozen_manifest_model_sha256
    )
    frozen_dataset_hash_matches_manifest = (
        frozen_dataset.dataset_hash == frozen_manifest_policy_dataset_hash
    )
    frozen_split_hash_matches_manifest = (
        True
        if frozen_manifest_split_hash is None
        else frozen_split_hash == frozen_manifest_split_hash
    )
    frozen_corpus_lineage_checks = {
        "corpus_manifest_matches_model_training_corpus_hash": (
            frozen_corpus_manifest_sha256 == frozen_model_training_corpus_hash
        ),
        "corpus_manifest_matches_manifest_training_corpus_hash": (
            True
            if frozen_manifest_training_corpus_hash is None
            else frozen_corpus_manifest_sha256 == frozen_manifest_training_corpus_hash
        ),
        "corpus_manifest_matches_manifest_phase2_corpus_manifest_sha256": (
            True
            if frozen_manifest_phase2_corpus_manifest_sha256 is None
            else frozen_corpus_manifest_sha256
            == frozen_manifest_phase2_corpus_manifest_sha256
        ),
        "dataset_hash_matches_manifest_policy_dataset_hash": (
            frozen_dataset_hash_matches_manifest
        ),
        "split_hash_matches_manifest_split_hash": frozen_split_hash_matches_manifest,
    }
    frozen_corpus_dir_matches_frozen_training_lineage = all(
        frozen_corpus_lineage_checks.values()
    )
    provenance = {
        "frozen_model_dir": str(frozen_model_dir),
        "frozen_corpus_dir": str(frozen_corpus_dir),
        "holdout_corpus_dir": str(holdout_corpus_dir),
        "frozen_model_manifest_path": str(frozen_manifest_path),
        "frozen_model_path": str(frozen_model_path),
        "frozen_action_value_calibration_path": str(frozen_calibration_path),
        "frozen_corpus_manifest_path": str(frozen_corpus_manifest_path),
        "frozen_split_path": str(frozen_split_path),
        "frozen_model_manifest_sha256": _sha256_file(frozen_manifest_path),
        "frozen_model_sha256": frozen_model_sha256,
        "frozen_action_value_calibration_sha256": _sha256_file(
            frozen_calibration_path
        ),
        "frozen_corpus_manifest_sha256": frozen_corpus_manifest_sha256,
        "frozen_model_training_corpus_hash": frozen_model_training_corpus_hash,
        "frozen_manifest_training_corpus_hash": frozen_manifest_training_corpus_hash,
        "frozen_manifest_phase2_corpus_manifest_sha256": (
            frozen_manifest_phase2_corpus_manifest_sha256
        ),
        "frozen_manifest_model_sha256": frozen_manifest_model_sha256,
        "frozen_model_sha256_matches_manifest": (
            frozen_model_sha256_matches_manifest
        ),
        "frozen_manifest_policy_dataset_hash": frozen_manifest_policy_dataset_hash,
        "frozen_dataset_hash_matches_manifest": (
            frozen_dataset_hash_matches_manifest
        ),
        "frozen_manifest_split_hash": frozen_manifest_split_hash,
        "frozen_split_hash": frozen_split_hash,
        "frozen_split_hash_available": frozen_split_hash is not None,
        "frozen_split_hash_matches_manifest": frozen_split_hash_matches_manifest,
        "frozen_corpus_lineage_checks": frozen_corpus_lineage_checks,
        "frozen_corpus_dir_matches_frozen_training_lineage": (
            frozen_corpus_dir_matches_frozen_training_lineage
        ),
        "frozen_dataset_hash": frozen_dataset.dataset_hash,
        "holdout_dataset_hash": holdout_dataset.dataset_hash,
        "frozen_training_corpus_hash": frozen_dataset.training_corpus_hash,
        "holdout_training_corpus_hash": holdout_dataset.training_corpus_hash,
        "frozen_row_count": len(frozen_dataset.examples),
        "holdout_row_count": len(holdout_dataset.examples),
        "frozen_market_count": len(frozen_market_ids),
        "holdout_market_count": len(holdout_market_ids),
        "frozen_max_decision_ts": frozen_max_decision_ts,
        "holdout_min_decision_ts": holdout_min_decision_ts,
        "holdout_max_decision_ts": max(
            example.decision_ts for example in holdout_dataset.examples
        ),
        "holdout_strictly_after_frozen": holdout_min_decision_ts
        > frozen_max_decision_ts,
        "market_id_overlap_count": len(overlapping_market_ids),
        "market_id_overlap_sample": overlapping_market_ids[:20],
        "market_id_disjoint": not overlapping_market_ids,
        "dataset_hash_changed": holdout_dataset.dataset_hash
        != frozen_dataset.dataset_hash,
        "training_corpus_hash_changed": holdout_dataset.training_corpus_hash
        != frozen_dataset.training_corpus_hash,
    }
    reason_codes = _provenance_reason_codes(provenance)
    if reason_codes:
        report = _blocked_report(
            config=config,
            frozen_manifest=frozen_manifest,
            provenance=provenance,
            reason_codes=reason_codes,
        )
        report["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
            report
        )
        return report

    model = _load_model(frozen_model_path)
    calibration = _read_json(frozen_calibration_path)
    raw_predictions = predict_polymarket_policy_examples(
        model,
        tuple(holdout_dataset.examples),
        missing_feature_mode="strict",
    )
    predictions = apply_action_value_calibration(
        predictions=raw_predictions,
        calibration_artifact=calibration,
    )
    prediction_set = build_sell_before_close_side_balanced_prediction_set(
        predictions=predictions,
        execution_buffer=float(frozen_training_config.get("ev_threshold", 0.015)),
    )
    replay_config = _dataset_config(
        corpus_dir=holdout_corpus_dir,
        output_dir=config.run_dir / "_holdout_replay",
        base_config=frozen_training_config,
    )
    decisions, guard_summary = build_sell_before_close_exit_reliability_guard_decisions(
        predictions=tuple(prediction_set["predictions"]),
        config=replay_config,
        thresholds=prediction_set.get("entry_filter_thresholds"),
        exit_policy=str(prediction_set["exit_policy"]),
        candidate_name=str(prediction_set["variant"]),
        p_up_side_alignment_filter_enabled=bool(
            prediction_set.get("p_up_side_alignment_filter_enabled", False)
        ),
    )
    replay_dataset = SimpleNamespace(
        shadow_examples=tuple(holdout_dataset.examples),
        market_metadata=holdout_dataset.market_metadata,
        resolution_events=holdout_dataset.resolution_events,
    )
    replay_report = run_polymarket_policy_replay(
        dataset=replay_dataset,
        decisions=decisions,
        config=replay_config,
        calibration_error=float(frozen_manifest.get("calibration_error", 0.0) or 0.0),
        calibration_split="validation",
        replay_split="shadow",
        prediction_count=len(predictions),
    )
    rows = _attribution_rows(
        entries=prediction_set.get("side_balance_candidate_entries", []),
        decisions=[decision.to_dict() for decision in decisions],
        examples=[example.to_dict() for example in holdout_dataset.examples],
        replay_report=replay_report,
        paper_notional=float(frozen_training_config.get("max_paper_notional", 0.20)),
    )
    summary = _m_promotion_attribution_summary(
        rows=rows,
        replay_report=dict(replay_report),
    )
    replay_total_pnl = float(summary["replay_total_pnl_sum"])
    selected_exit_decision_count = int(summary["selected_exit_decision_count"])
    reconciliation = dict(summary["replay_entry_reconciliation"])
    holdout_validation_passed = (
        bool(reconciliation.get("reconciled", False))
        and selected_exit_decision_count == 0
        and replay_total_pnl > 0.0
    )
    report = {
        "schema_version": M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m_post_freeze_holdout_validation",
        "validation_status": "completed",
        "diagnostic_only": True,
        "prediction_attempted": True,
        "baseline_selector_commit": FROZEN_M_SELECTOR_BASELINE_COMMIT,
        "baseline_selector_parent_commit": FROZEN_M_SELECTOR_PARENT_COMMIT,
        "selector_method": FROZEN_M_SELECTOR_METHOD,
        "selector_weights_unchanged_from_baseline": True,
        "rank_weight_tuning_allowed": False,
        "rank_weight_tuning_performed": False,
        "holdout_feedback_used_for_tuning": False,
        "no_shadow_gate_feedback_used_for_tuning": True,
        "true_post_freeze_holdout": True,
        "provenance": provenance,
        "rank_score_components": prediction_set.get("rank_score_components"),
        "rank_score_components_match_frozen_baseline": (
            prediction_set.get("rank_score_components")
            == FROZEN_M_RANK_SCORE_COMPONENTS
        ),
        "frozen_rank_weights": _frozen_rank_weights(),
        "rank_score_component_summary": _m_rank_score_component_summary(
            prediction_set.get("side_balance_candidate_entries", [])
        ),
        "p_up_side_alignment_filter_enabled": bool(
            prediction_set.get("p_up_side_alignment_filter_enabled", False)
        ),
        "p_up_side_alignment_diagnostic_enabled": bool(
            prediction_set.get("p_up_side_alignment_diagnostic_enabled", False)
        ),
        "side_balance_selection_summary": dict(
            prediction_set.get("side_balance_selection_summary", {})
        ),
        "exit_reliability_guard_summary": dict(guard_summary),
        "selected_entry_count": int(summary["selected_entry_count"]),
        "replay_entry_count": int(summary["replay_entry_count"]),
        "selected_exit_decision_count": selected_exit_decision_count,
        "replay_entry_reconciliation": reconciliation,
        "replay_total_pnl_sum": replay_total_pnl,
        "label_vs_replay_pnl_gap": float(summary["label_vs_replay_pnl_gap"]),
        "replay_pnl_by_side": dict(summary["replay_total_pnl_by_side"]),
        "selected_label_pnl_sum_by_side": dict(
            summary["selected_label_pnl_sum_by_side"]
        ),
        "label_vs_replay_pnl_gap_by_side": dict(
            summary["label_vs_replay_pnl_gap_by_side"]
        ),
        "replay_pnl_positive": replay_total_pnl > 0.0,
        "holdout_validation_passed": holdout_validation_passed,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "top_negative_replay_entries": _m_top_negative_selected_entries(rows),
        "side_replay_entry_counts": dict(
            sorted(Counter(row["selected_side"] for row in rows if row["entry_order_opened"]).items())
        ),
        "rows": rows,
        "reason_codes": [],
        "ineligible_reason_codes": (
            []
            if holdout_validation_passed
            else ["post_freeze_holdout_validation_not_passed"]
        )
        + ["diagnostic_only_no_paper_live_unlock"],
        **compact_safety_fields(),
    }
    report["m_post_freeze_holdout_validation_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _blocked_report(
    *,
    config: PolymarketPostFreezeHoldoutConfig,
    frozen_manifest: dict[str, Any],
    provenance: dict[str, Any],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m_post_freeze_holdout_validation",
        "validation_status": "blocked_fail_closed",
        "diagnostic_only": True,
        "prediction_attempted": False,
        "baseline_selector_commit": FROZEN_M_SELECTOR_BASELINE_COMMIT,
        "baseline_selector_parent_commit": FROZEN_M_SELECTOR_PARENT_COMMIT,
        "selector_method": FROZEN_M_SELECTOR_METHOD,
        "selector_weights_unchanged_from_baseline": True,
        "rank_weight_tuning_allowed": False,
        "rank_weight_tuning_performed": False,
        "holdout_feedback_used_for_tuning": False,
        "no_shadow_gate_feedback_used_for_tuning": True,
        "true_post_freeze_holdout": False,
        "provenance": provenance,
        "frozen_manifest_policy_dataset_hash": frozen_manifest.get(
            "policy_dataset_hash",
        ),
        "rank_score_components": FROZEN_M_RANK_SCORE_COMPONENTS,
        "rank_score_components_match_frozen_baseline": True,
        "frozen_rank_weights": _frozen_rank_weights(),
        "rank_score_component_summary": {
            "candidate_row_count": 0,
            "selected_entry_count": 0,
            "fields": {
                "execution_pnl_aware_rank_score": {
                    "all_candidates": _m_component_stats([], "none"),
                    "selected_entries": _m_component_stats([], "none"),
                },
            },
        },
        "p_up_side_alignment_filter_enabled": False,
        "p_up_side_alignment_diagnostic_enabled": True,
        "selected_entry_count": 0,
        "replay_entry_count": 0,
        "selected_exit_decision_count": 0,
        "replay_entry_reconciliation": {
            "selected_entry_count": 0,
            "replay_entry_count": 0,
            "selected_without_replay_entry_count": 0,
            "selected_exit_decision_count": 0,
            "reconciled": False,
        },
        "replay_total_pnl_sum": 0.0,
        "label_vs_replay_pnl_gap": 0.0,
        "replay_pnl_by_side": {"UP": 0.0, "DOWN": 0.0},
        "top_negative_replay_entries": [],
        "holdout_validation_passed": False,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "reason_codes": reason_codes,
        "ineligible_reason_codes": [
            *reason_codes,
            "diagnostic_only_no_paper_live_unlock",
        ],
        **compact_safety_fields(),
    }


def _dataset_config(
    *,
    corpus_dir: Path,
    output_dir: Path,
    base_config: dict[str, Any],
) -> PolymarketPolicyTrainingConfig:
    allowed = {
        "run_id",
        "model_version",
        "created_at",
        "train_fraction",
        "validation_fraction",
        "ev_threshold",
        "min_confidence",
        "max_paper_notional",
        "fee_rate",
        "slippage_rate",
        "liquidity_impact_rate",
        "sell_before_close_exit_buffer_seconds",
        "paper_only",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
    }
    payload = {key: base_config[key] for key in allowed if key in base_config}
    payload.update(
        {
            "corpus_dir": corpus_dir,
            "output_dir": output_dir,
            "run_id": "post_freeze_holdout_probe",
            "overwrite_existing": True,
            **compact_safety_fields(),
        }
    )
    return PolymarketPolicyTrainingConfig(**payload)


def _provenance_reason_codes(provenance: dict[str, Any]) -> list[str]:
    reason_codes = []
    if not bool(provenance["frozen_model_sha256_matches_manifest"]):
        reason_codes.append("frozen_model_sha256_mismatch_manifest")
    if not bool(provenance["frozen_dataset_hash_matches_manifest"]):
        reason_codes.append("frozen_dataset_hash_mismatch_manifest")
    if not bool(provenance["frozen_split_hash_matches_manifest"]):
        reason_codes.append("frozen_split_hash_mismatch_manifest")
    if not bool(provenance["frozen_corpus_dir_matches_frozen_training_lineage"]):
        reason_codes.append("frozen_corpus_dir_not_frozen_training_lineage")
    if not bool(provenance["holdout_strictly_after_frozen"]):
        reason_codes.append("holdout_not_strictly_after_frozen_training_window")
    if not bool(provenance["market_id_disjoint"]):
        reason_codes.append("holdout_market_ids_overlap_frozen_training_corpus")
    if not bool(provenance["dataset_hash_changed"]):
        reason_codes.append("holdout_dataset_hash_matches_frozen_dataset")
    if not bool(provenance["training_corpus_hash_changed"]):
        reason_codes.append("holdout_training_corpus_hash_matches_frozen_corpus")
    return reason_codes


def _load_model(path: Path) -> PolymarketPolicyModel:
    payload = _read_json(path)
    for field_name in ("feature_columns", "action_value_feature_columns"):
        if field_name in payload:
            payload[field_name] = tuple(payload[field_name])
    return PolymarketPolicyModel(**payload)


def _attribution_rows(
    *,
    entries: Any,
    decisions: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    replay_report: dict[str, Any],
    paper_notional: float,
) -> list[dict[str, Any]]:
    examples_by_key = {
        (str(example.get("market_id")), int(example.get("decision_ts", 0))): example
        for example in examples
    }
    decisions_by_key = {
        (str(row.get("market_id")), int(row.get("decision_ts", 0))): row
        for row in decisions
    }
    decisions_by_market: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        decisions_by_market.setdefault(str(decision.get("market_id")), []).append(
            decision
        )
    for market_decisions in decisions_by_market.values():
        market_decisions.sort(key=lambda row: int(row.get("decision_ts", 0)))
    entry_contexts = _m_replay_entry_contexts(
        decisions=decisions,
        decisions_by_market=decisions_by_market,
        examples_by_key=examples_by_key,
        replay_report=dict(replay_report),
    )
    rows = []
    for entry in entries:
        key = (str(entry.get("market_id")), int(entry.get("decision_ts", 0)))
        rows.append(
            _m_promotion_attribution_row(
                candidate_entry=dict(entry),
                decision=decisions_by_key.get(key),
                entry_context=entry_contexts.get(key),
                decisions_by_market=decisions_by_market,
                example=examples_by_key.get(key),
                paper_notional=paper_notional,
            )
        )
    rows.sort(
        key=lambda row: (
            str(row["selected_side"]),
            not bool(row["side_quota_selected"]),
            int(row["side_quota_rank"] or 999_999),
            int(row["decision_ts"]),
            str(row["market_id"]),
        )
    )
    return rows


def _frozen_rank_weights() -> dict[str, float]:
    return {
        "model_score_weight": M_EXECUTION_PNL_AWARE_MODEL_SCORE_WEIGHT,
        "margin_weight": M_EXECUTION_PNL_AWARE_MARGIN_WEIGHT,
        "entry_exit_quality_weight": M_EXECUTION_PNL_AWARE_QUALITY_WEIGHT,
        "immediate_exit_return_weight": (
            M_EXECUTION_PNL_AWARE_IMMEDIATE_EXIT_RETURN_WEIGHT
        ),
        "gap_penalty_weight": M_EXECUTION_PNL_AWARE_GAP_PENALTY_WEIGHT,
    }


def _post_freeze_holdout_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M Post-Freeze Holdout Validation",
        "",
        f"- validation_status: `{report['validation_status']}`",
        f"- true_post_freeze_holdout: `{str(report['true_post_freeze_holdout']).lower()}`",
        f"- prediction_attempted: `{str(report['prediction_attempted']).lower()}`",
        f"- selector_method: `{report['selector_method']}`",
        "- selector_weights_unchanged_from_baseline: "
        f"`{str(report['selector_weights_unchanged_from_baseline']).lower()}`",
        f"- replay_total_pnl_sum: `{report['replay_total_pnl_sum']}`",
        f"- holdout_validation_passed: `{str(report['holdout_validation_passed']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Provenance",
        "",
        "| check | value |",
        "|---|---:|",
        "| frozen_max_decision_ts | {frozen} |".format(
            frozen=report["provenance"]["frozen_max_decision_ts"]
        ),
        "| holdout_min_decision_ts | {holdout} |".format(
            holdout=report["provenance"]["holdout_min_decision_ts"]
        ),
        "| holdout_strictly_after_frozen | {value} |".format(
            value=str(
                report["provenance"]["holdout_strictly_after_frozen"]
            ).lower()
        ),
        "| market_id_overlap_count | {value} |".format(
            value=report["provenance"]["market_id_overlap_count"]
        ),
        "",
        "## Required Metrics",
        "",
        "| selected | replay_entries | selected_exit_decisions | reconciled | replay_pnl | label_vs_replay_gap |",
        "|---:|---:|---:|---|---:|---:|",
        "| {selected} | {entries} | {exits} | {reconciled} | {pnl:.6f} | {gap:.6f} |".format(
            selected=report["selected_entry_count"],
            entries=report["replay_entry_count"],
            exits=report["selected_exit_decision_count"],
            reconciled=str(
                report["replay_entry_reconciliation"]["reconciled"]
            ).lower(),
            pnl=float(report["replay_total_pnl_sum"]),
            gap=float(report["label_vs_replay_pnl_gap"]),
        ),
        "",
        "## Reason Codes",
        "",
        *[f"- `{reason}`" for reason in report.get("reason_codes", [])],
        *[f"- `{reason}`" for reason in report.get("ineligible_reason_codes", [])],
        "",
        "- paper_only: true",
        "- capital_at_risk: false",
        "- polymarket_write_enabled: false",
        "- wallet_signing_enabled: false",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
