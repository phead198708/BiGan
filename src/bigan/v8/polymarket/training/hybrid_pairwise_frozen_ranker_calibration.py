"""Fresh LCB calibration for the frozen historical pairwise ranker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_hierarchical_action_value import (
    _accepted_bet_metrics,
    _market_robustness,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _action_advantage_lcb_artifact,
    _apply_action_advantage_lcb_scores,
    _confirmatory_gate,
    _development_freeze_gate,
    _load_corpus_action_rows,
    _predict_role_rows,
    _run_policy_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.hybrid_pairwise_precollection_readiness import (
    PROTOCOL_SCHEMA_VERSION as HYBRID_PROTOCOL_SCHEMA_VERSION,
)

SCHEMA_PREFIX = "bigan-v8-hybrid-pairwise-frozen-ranker"
CALIBRATION_ROLE = "fresh_development_calibration"
CONFIRMATORY_ROLE = "fresh_confirmatory_validation"
CALIBRATION_MARKET_COUNT = 45
CONFIRMATORY_MARKET_COUNT = 60
TOTAL_FRESH_MARKET_COUNT = CALIBRATION_MARKET_COUNT + CONFIRMATORY_MARKET_COUNT
ROLE_ASSIGNMENT_SCHEMA_VERSION = (
    "bigan-v8-hybrid-pairwise-fresh-role-assignment-v1"
)
OOF_SCORE_FIELDS = (
    "market_id",
    "decision_ts",
    "action",
    "action_family",
    "side",
    "fold_index",
    "oof_raw_prediction",
    "pairwise_action_rank",
    "pairwise_rank_percentile",
    "pairwise_group_normalized_rank_score",
    "pairwise_group_score_range",
    "pairwise_normalized_margin_vs_no_trade",
    "pairwise_normalized_margin_vs_best_alternative",
    "pairwise_rank_normalization_scope",
    "raw_rank_score_cross_model_comparison_allowed",
)
READINESS_FORBIDDEN_FIELDS = {
    "accepted_bet_net_pnl",
    "confirmatory_gate_passed",
    "future_return",
    "net_pnl",
    "oracle_action",
    "realized_pnl",
    "resolved_outcome",
    "settlement_pnl",
    "target_net_return_after_cost",
    "total_net_pnl_per_notional",
}


@dataclass(frozen=True, slots=True)
class HybridPairwiseCalibrationReadinessConfig:
    """Outcome-blind readiness inputs before fresh role assignment exists."""

    run_id: str
    output_dir: Path | str
    hybrid_protocol_path: Path | str
    expected_hybrid_protocol_sha256: str
    historical_ranker_descriptor_path: Path | str
    expected_historical_ranker_descriptor_sha256: str
    historical_ranker_manifest_path: Path | str
    expected_historical_ranker_manifest_sha256: str
    upstream_terminal_freeze_state_path: Path | str
    expected_upstream_terminal_freeze_state_sha256: str
    fresh_role_assignment_manifest_path: Path | str | None = None
    expected_fresh_role_assignment_manifest_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, digest in (
            ("hybrid protocol", self.expected_hybrid_protocol_sha256),
            (
                "historical ranker descriptor",
                self.expected_historical_ranker_descriptor_sha256,
            ),
            (
                "historical ranker manifest",
                self.expected_historical_ranker_manifest_sha256,
            ),
            (
                "upstream terminal freeze state",
                self.expected_upstream_terminal_freeze_state_sha256,
            ),
        ):
            _require_sha256(digest, name=f"{name} SHA-256")
        if (self.fresh_role_assignment_manifest_path is None) != (
            self.expected_fresh_role_assignment_manifest_sha256 is None
        ):
            raise ValueError(
                "fresh role manifest path and SHA-256 must be provided together"
            )
        if self.expected_fresh_role_assignment_manifest_sha256 is not None:
            _require_sha256(
                self.expected_fresh_role_assignment_manifest_sha256,
                name="fresh role assignment manifest SHA-256",
            )
        for name in (
            "output_dir",
            "hybrid_protocol_path",
            "historical_ranker_descriptor_path",
            "historical_ranker_manifest_path",
            "upstream_terminal_freeze_state_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.fresh_role_assignment_manifest_path is not None:
            object.__setattr__(
                self,
                "fresh_role_assignment_manifest_path",
                Path(self.fresh_role_assignment_manifest_path),
            )


@dataclass(frozen=True, slots=True)
class HybridPairwiseFreshCalibrationConfig:
    """Inputs for the 45-market calibration-only stage."""

    run_id: str
    output_dir: Path | str
    hybrid_protocol_path: Path | str
    expected_hybrid_protocol_sha256: str
    source_pairwise_protocol_path: Path | str
    expected_source_pairwise_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    historical_ranker_descriptor_path: Path | str
    expected_historical_ranker_descriptor_sha256: str
    historical_ranker_manifest_path: Path | str
    expected_historical_ranker_manifest_sha256: str
    fresh_role_assignment_manifest_path: Path | str
    expected_fresh_role_assignment_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name, digest in (
            ("hybrid protocol", self.expected_hybrid_protocol_sha256),
            ("source pairwise protocol", self.expected_source_pairwise_protocol_sha256),
            ("feature contract", self.expected_feature_contract_sha256),
            (
                "historical ranker descriptor",
                self.expected_historical_ranker_descriptor_sha256,
            ),
            (
                "historical ranker manifest",
                self.expected_historical_ranker_manifest_sha256,
            ),
            (
                "fresh role assignment manifest",
                self.expected_fresh_role_assignment_manifest_sha256,
            ),
        ):
            _require_sha256(digest, name=f"{name} SHA-256")
        for name in (
            "output_dir",
            "hybrid_protocol_path",
            "source_pairwise_protocol_path",
            "feature_contract_path",
            "historical_ranker_descriptor_path",
            "historical_ranker_manifest_path",
            "fresh_role_assignment_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class HybridPairwiseConfirmatoryConfig:
    """Inputs for the one-shot 60-market confirmatory stage."""

    run_id: str
    output_dir: Path | str
    calibration_freeze_manifest_path: Path | str
    expected_calibration_freeze_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_calibration_freeze_manifest_sha256,
            name="calibration freeze manifest SHA-256",
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "calibration_freeze_manifest_path",
            Path(self.calibration_freeze_manifest_path),
        )


def evaluate_hybrid_pairwise_calibration_readiness(
    config: HybridPairwiseCalibrationReadinessConfig,
) -> dict[str, Any]:
    """Write a blocked artifact until #183 freeze and 45/60 roles exist."""

    hybrid_protocol_path = config.hybrid_protocol_path.resolve()
    ranker_descriptor_path = config.historical_ranker_descriptor_path.resolve()
    ranker_manifest_path = config.historical_ranker_manifest_path.resolve()
    upstream_state_path = config.upstream_terminal_freeze_state_path.resolve()
    for path, digest, name in (
        (
            hybrid_protocol_path,
            config.expected_hybrid_protocol_sha256,
            "hybrid protocol",
        ),
        (
            ranker_descriptor_path,
            config.expected_historical_ranker_descriptor_sha256,
            "historical ranker descriptor",
        ),
        (
            ranker_manifest_path,
            config.expected_historical_ranker_manifest_sha256,
            "historical ranker manifest",
        ),
        (
            upstream_state_path,
            config.expected_upstream_terminal_freeze_state_sha256,
            "upstream terminal freeze state",
        ),
    ):
        _verify_pin(path, digest, name=name)
    hybrid_protocol = _load_json(hybrid_protocol_path)
    ranker_descriptor = _load_json(ranker_descriptor_path)
    ranker_manifest = _load_json(ranker_manifest_path)
    upstream_state = _load_json(upstream_state_path)
    forbidden = sorted(
        _find_fields(upstream_state, READINESS_FORBIDDEN_FIELDS)
    )
    if forbidden:
        raise ValueError(
            "upstream readiness state contains forbidden fields: "
            + ", ".join(forbidden)
        )
    frozen_ranker = dict(
        hybrid_protocol.get("historical_ranker_freeze") or {}
    )
    identity_verified = (
        hybrid_protocol.get("schema_version")
        == HYBRID_PROTOCOL_SCHEMA_VERSION
        and frozen_ranker.get("descriptor_sha256")
        == config.expected_historical_ranker_descriptor_sha256
        and frozen_ranker.get("freeze_id")
        == ranker_descriptor.get("freeze_id")
        == ranker_manifest.get("freeze_id")
        and frozen_ranker.get("model_sha256")
        == ranker_descriptor.get("model_sha256")
        == ranker_manifest.get("model_sha256")
        and frozen_ranker.get("dataset_hash")
        == ranker_descriptor.get("dataset_hash")
        == ranker_manifest.get("dataset_hash")
        and frozen_ranker.get("oof_dataset_hash")
        == ranker_manifest.get("oof_dataset_hash")
        and frozen_ranker.get("split_hash")
        == ranker_descriptor.get("split_hash")
        == ranker_manifest.get("split_hash")
        and frozen_ranker.get("model_config_hash")
        == ranker_descriptor.get("model_config_hash")
        == ranker_manifest.get("model_config_hash")
    )
    if not identity_verified:
        raise ValueError("frozen historical ranker identity mismatch")

    blockers = []
    upstream_complete = (
        upstream_state.get("status") == "completed"
        and upstream_state.get("precollection_readiness_passed") is True
        and upstream_state.get("precollection_freeze_created") is True
        and upstream_state.get("collection_start_allowed") is False
        and upstream_state.get("collection_start_command_generated") is False
    )
    if not upstream_complete:
        blockers.append("issue183_terminal_freeze_incomplete")

    role_manifest_descriptor = None
    role_assignment_ready = False
    role_market_counts = {
        CALIBRATION_ROLE: 0,
        CONFIRMATORY_ROLE: 0,
    }
    if config.fresh_role_assignment_manifest_path is None:
        blockers.append("fresh_45_60_role_assignment_missing")
    else:
        role_path = Path(
            config.fresh_role_assignment_manifest_path
        ).resolve()
        assert config.expected_fresh_role_assignment_manifest_sha256
        _verify_pin(
            role_path,
            config.expected_fresh_role_assignment_manifest_sha256,
            name="fresh role assignment manifest",
        )
        role_manifest = _load_json(role_path)
        role_rows_descriptor = _verified_descriptor(
            role_manifest.get("selected_rows"),
            name="fresh role assignment rows",
        )
        role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
        try:
            _validate_role_assignment(role_manifest, role_rows)
        except ValueError:
            blockers.append("fresh_45_60_role_assignment_invalid")
        else:
            role_assignment_ready = True
            role_market_counts = dict(
                Counter(str(row["role"]) for row in role_rows)
            )
        role_manifest_descriptor = _descriptor(role_path)

    readiness_passed = not blockers
    run_dir = _prepare_run_dir(
        config.output_dir / config.run_id,
        overwrite=config.overwrite_existing,
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-readiness-report-v1",
        "run_id": config.run_id,
        "readiness_status": (
            "ready_for_fresh_calibration"
            if readiness_passed
            else "blocked_fail_closed"
        ),
        "hybrid_protocol": _descriptor(hybrid_protocol_path),
        "historical_ranker_descriptor": _descriptor(
            ranker_descriptor_path
        ),
        "historical_ranker_manifest": _descriptor(ranker_manifest_path),
        "upstream_terminal_freeze_state": _descriptor(
            upstream_state_path
        ),
        "fresh_role_assignment_manifest": role_manifest_descriptor,
        "historical_ranker_identity_verified": identity_verified,
        "upstream_terminal_freeze_complete": upstream_complete,
        "fresh_role_assignment_ready": role_assignment_ready,
        "fresh_role_market_counts": role_market_counts,
        "calibration_readiness_passed": readiness_passed,
        "calibration_start_allowed": readiness_passed,
        "model_prediction_attempted": False,
        "ranker_retraining_attempted": False,
        "ranker_score_mutation_attempted": False,
        "label_or_outcome_artifacts_opened": False,
        "oof_or_validation_pnl_used_for_tuning": False,
        "blocking_reason_codes": sorted(set(blockers)),
        **_blocked_safety_fields(),
    }
    report_path = (
        run_dir / "hybrid_pairwise_calibration_readiness_report.json"
    )
    _write_json(report_path, report)
    _write_text(
        run_dir / "hybrid_pairwise_calibration_readiness_report.md",
        "\n".join(
            [
                "# Hybrid Pairwise Calibration Readiness",
                "",
                f"- status: `{report['readiness_status']}`",
                (
                    "- calibration start allowed: "
                    f"`{str(readiness_passed).lower()}`"
                ),
                f"- blocking reasons: `{report['blocking_reason_codes']}`",
                "- labels/outcomes opened: `false`",
                "- model prediction attempted: `false`",
                "- ranker retraining attempted: `false`",
                "- paper/live/handoff unlock: `false`",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-readiness-manifest-v1",
        "run_id": config.run_id,
        "readiness_report": _descriptor(report_path),
        "calibration_readiness_passed": readiness_passed,
        "calibration_start_allowed": readiness_passed,
        "model_prediction_attempted": False,
        "ranker_retraining_attempted": False,
        "blocking_reason_codes": report["blocking_reason_codes"],
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = (
        run_dir / "hybrid_pairwise_calibration_readiness_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "report": report,
        "manifest": manifest,
    }


def freeze_hybrid_pairwise_fresh_calibration(
    config: HybridPairwiseFreshCalibrationConfig,
) -> dict[str, Any]:
    """Calibrate LCB estimates without retraining or opening confirmatory labels."""

    lineage = _load_and_validate_lineage(config)
    role_rows = lineage["role_rows"]
    calibration_role_rows = [
        row for row in role_rows if row["role"] == CALIBRATION_ROLE
    ]
    run_dir = _prepare_run_dir(
        config.output_dir / config.run_id,
        overwrite=config.overwrite_existing,
    )
    feature_columns = tuple(lineage["feature_contract"]["feature_columns"])
    split_manifest = _fresh_split_manifest(
        run_id=config.run_id,
        role_rows=role_rows,
        role_manifest_descriptor=lineage["role_manifest_descriptor"],
        precollection_freeze_descriptor=lineage[
            "precollection_freeze_descriptor"
        ],
        final_quarantine_descriptor=lineage[
            "final_quarantine_descriptor"
        ],
    )
    split_manifest_path = run_dir / "hybrid_pairwise_fresh_split_manifest.json"
    _write_json(split_manifest_path, split_manifest)
    calibration_rows, corpus_audits = _materialize_fresh_action_rows(
        calibration_role_rows,
        feature_columns=feature_columns,
        expected_market_count=CALIBRATION_MARKET_COUNT,
    )
    calibration_action_rows_path = (
        run_dir / "hybrid_pairwise_fresh_calibration_action_rows.jsonl"
    )
    _write_jsonl(calibration_action_rows_path, calibration_rows)

    model_path = Path(lineage["ranker_descriptor"]["model"]["path"]).resolve()
    model_sha256_before = _sha256_file(model_path)
    booster = _load_frozen_booster(model_path)
    calibration_predictions = _predict_role_rows(
        calibration_rows,
        booster=booster,
        feature_columns=feature_columns,
    )
    model_sha256_after_prediction = _sha256_file(model_path)
    if model_sha256_after_prediction != model_sha256_before:
        raise ValueError("frozen ranker model mutated during prediction")
    calibration_prediction_path = (
        run_dir / "hybrid_pairwise_fresh_calibration_predictions.jsonl"
    )
    _write_jsonl(calibration_prediction_path, calibration_predictions)

    oof_score_rows, oof_score_audit = _score_only_oof_rows(
        Path(lineage["ranker_descriptor"]["train_oof_predictions"]["path"])
    )
    oof_score_path = (
        run_dir / "hybrid_pairwise_historical_oof_score_only_rows.jsonl"
    )
    _write_jsonl(oof_score_path, oof_score_rows)
    lcb_artifact = _action_advantage_lcb_artifact(
        calibration_predictions,
        train_oof_predictions=oof_score_rows,
        protocol=lineage["source_protocol"],
        feature_contract_sha256=lineage["feature_contract_descriptor"][
            "sha256"
        ],
    )
    lcb_artifact.update(
        {
            "schema_version": f"{SCHEMA_PREFIX}-fresh-lcb-artifact-v1",
            "candidate_lineage": lineage["hybrid_protocol"][
                "candidate_lineage"
            ],
            "historical_ranker_freeze_id": lineage["ranker_descriptor"][
                "freeze_id"
            ],
            "historical_ranker_model_sha256": model_sha256_before,
            "historical_oof_score_only_rows": _descriptor(oof_score_path),
            "historical_oof_target_values_used_for_bucket_construction": False,
            "fresh_calibration_market_ids_sha256": canonical_json_sha256(
                sorted({str(row["market_id"]) for row in calibration_rows})
            ),
            "ranker_retrained": False,
            "ranker_score_mutated": False,
            "confirmatory_labels_opened": False,
            "uses_current_oof_or_validation_pnl_for_tuning": False,
        }
    )
    lcb_artifact["calibration_artifact_id"] = canonical_json_sha256(
        {
            key: value
            for key, value in lcb_artifact.items()
            if key != "calibration_artifact_id"
        }
    )
    lcb_path = run_dir / "hybrid_pairwise_fresh_lcb_calibration_artifact.json"
    _write_json(lcb_path, lcb_artifact)
    calibrated_predictions = _apply_action_advantage_lcb_scores(
        calibration_predictions,
        lcb_artifact=lcb_artifact,
    )
    calibrated_prediction_path = (
        run_dir / "hybrid_pairwise_fresh_calibrated_predictions.jsonl"
    )
    _write_jsonl(calibrated_prediction_path, calibrated_predictions)

    entry_threshold = float(
        lineage["source_protocol"]["frozen_execution_contract"][
            "entry_edge_threshold"
        ]
    )
    runner_up_threshold = float(
        lineage["source_protocol"]["frozen_execution_contract"][
            "runner_up_advantage_threshold"
        ]
    )
    candidate_replay = _run_policy_replay(
        calibrated_predictions,
        score_field="action_advantage_lcb_net_return",
        policy_name="hybrid_frozen_ranker_fresh_action_advantage_lcb",
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=runner_up_threshold,
    )
    baseline_replay = _run_policy_replay(
        calibrated_predictions,
        score_field="calibrated_action_expected_net_return",
        policy_name="hybrid_frozen_ranker_uncertainty_unadjusted_baseline",
        entry_threshold=entry_threshold,
        runner_up_advantage_threshold=runner_up_threshold,
    )
    candidate_replay_path = (
        run_dir / "hybrid_pairwise_development_candidate_replay.jsonl"
    )
    baseline_replay_path = (
        run_dir / "hybrid_pairwise_development_baseline_replay.jsonl"
    )
    _write_jsonl(candidate_replay_path, candidate_replay)
    _write_jsonl(baseline_replay_path, baseline_replay)
    candidate_metrics = _accepted_bet_metrics(candidate_replay)
    baseline_metrics = _accepted_bet_metrics(baseline_replay)
    robustness = _market_robustness(candidate_replay, baseline_replay)
    development_gate = _development_freeze_gate(
        protocol=lineage["source_protocol"],
        action_rows=calibration_rows,
        candidate_replay=candidate_replay,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
    )
    calibration_leakage_audit = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-leakage-role-audit-v1",
        "run_id": config.run_id,
        "fresh_split_manifest": _descriptor(split_manifest_path),
        "calibration_market_count": len(
            {str(row["market_id"]) for row in calibration_rows}
        ),
        "calibration_decision_group_count": (
            len(calibration_rows) // len(REQUIRED_ACTIONS)
        ),
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"])
            for audit in corpus_audits
        ),
        "forbidden_inference_field_violation_count": sum(
            int(
                row["target_used_as_decision_input"] is not False
                or row["outcome_fields_used_as_decision_input"] is not False
            )
            for row in calibration_rows
        ),
        "complete_five_action_grid": True,
        "prior_market_overlap_count": 0,
        "role_market_overlap_count": 0,
        "chronology_validation_passed": True,
        "confirmatory_labels_opened": False,
        "historical_oof_target_values_used_for_bucket_construction": False,
        "leakage_and_role_audit_passed": (
            all(
                int(audit["feature_causality_violation_count"]) == 0
                for audit in corpus_audits
            )
            and all(
                row["target_used_as_decision_input"] is False
                and row["outcome_fields_used_as_decision_input"] is False
                for row in calibration_rows
            )
        ),
        **_blocked_safety_fields(),
    }
    calibration_leakage_path = (
        run_dir / "hybrid_pairwise_calibration_leakage_role_audit.json"
    )
    _write_json(calibration_leakage_path, calibration_leakage_audit)

    identity_report = {
        "schema_version": f"{SCHEMA_PREFIX}-identity-report-v1",
        "run_id": config.run_id,
        "historical_ranker_descriptor": lineage[
            "ranker_descriptor_descriptor"
        ],
        "historical_ranker_manifest": lineage[
            "ranker_manifest_descriptor"
        ],
        "historical_ranker_freeze_id": lineage["ranker_descriptor"][
            "freeze_id"
        ],
        "model_sha256": model_sha256_before,
        "dataset_hash": lineage["ranker_descriptor"]["dataset_hash"],
        "oof_dataset_hash": lineage["ranker_manifest"]["oof_dataset_hash"],
        "split_hash": lineage["ranker_descriptor"]["split_hash"],
        "model_config_hash": lineage["ranker_descriptor"][
            "model_config_hash"
        ],
        "model_sha256_unchanged_after_prediction": (
            model_sha256_before == model_sha256_after_prediction
        ),
        "ranker_retrained": False,
        "ranker_score_mutated": False,
        "rank_scores_execution_eligible": False,
        **_blocked_safety_fields(),
    }
    identity_report_path = (
        run_dir / "hybrid_pairwise_frozen_ranker_identity_report.json"
    )
    _write_json(identity_report_path, identity_report)
    _write_text(
        run_dir / "hybrid_pairwise_frozen_ranker_identity_report.md",
        _identity_markdown(identity_report),
    )
    calibration_report = {
        "schema_version": f"{SCHEMA_PREFIX}-fresh-calibration-report-v1",
        "run_id": config.run_id,
        "source_split": CALIBRATION_ROLE,
        "fresh_calibration_market_count": CALIBRATION_MARKET_COUNT,
        "fresh_calibration_action_row_count": len(calibration_rows),
        "fresh_calibration_decision_group_count": (
            len(calibration_rows) // len(REQUIRED_ACTIONS)
        ),
        "corpus_audit_count": len(corpus_audits),
        "feature_causality_violation_count": sum(
            int(audit["feature_causality_violation_count"])
            for audit in corpus_audits
        ),
        "historical_oof_score_only_audit": oof_score_audit,
        "calibration_artifact": _descriptor(lcb_path),
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline_net_pnl": (
            float(candidate_metrics["net_pnl_sum"])
            - float(baseline_metrics["net_pnl_sum"])
        ),
        "market_robustness_diagnostics": robustness,
        "ranker_retrained": False,
        "ranker_score_mutated": False,
        "confirmatory_labels_opened": False,
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_current_oof_or_validation_pnl_for_tuning": False,
        **_blocked_safety_fields(),
    }
    calibration_report_path = (
        run_dir / "hybrid_pairwise_fresh_calibration_report.json"
    )
    _write_json(calibration_report_path, calibration_report)
    _write_text(
        run_dir / "hybrid_pairwise_fresh_calibration_report.md",
        _calibration_markdown(calibration_report),
    )
    development_report = {
        "schema_version": f"{SCHEMA_PREFIX}-development-gate-report-v1",
        "run_id": config.run_id,
        "development_gate_checks": development_gate["checks"],
        "development_gate_passed": development_gate["passed"],
        "development_gate_blocking_reason_codes": development_gate[
            "reason_codes"
        ],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "market_robustness_diagnostics": robustness,
        "calibration_artifact_frozen_before_confirmatory_label_access": True,
        "confirmatory_labels_opened": False,
        "confirmatory_label_access_allowed": development_gate["passed"],
        "uses_confirmatory_validation_labels_for_tuning": False,
        **_blocked_safety_fields(),
    }
    development_report_path = (
        run_dir / "hybrid_pairwise_development_gate_report.json"
    )
    _write_json(development_report_path, development_report)
    _write_text(
        run_dir / "hybrid_pairwise_development_gate_report.md",
        _development_markdown(development_report),
    )

    freeze_manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-freeze-manifest-v1",
        "run_id": config.run_id,
        "hybrid_protocol": lineage["hybrid_protocol_descriptor"],
        "source_pairwise_protocol": lineage["source_protocol_descriptor"],
        "feature_contract": lineage["feature_contract_descriptor"],
        "historical_ranker_descriptor": lineage[
            "ranker_descriptor_descriptor"
        ],
        "historical_ranker_manifest": lineage[
            "ranker_manifest_descriptor"
        ],
        "fresh_role_assignment_manifest": lineage[
            "role_manifest_descriptor"
        ],
        "fresh_role_assignment_rows": lineage["role_rows_descriptor"],
        "hybrid_precollection_freeze": lineage[
            "precollection_freeze_descriptor"
        ],
        "final_prior_lineage_quarantine": lineage[
            "final_quarantine_descriptor"
        ],
        "fresh_split_manifest": _descriptor(split_manifest_path),
        "historical_ranker_model": _descriptor(model_path),
        "historical_oof_score_only_rows": _descriptor(oof_score_path),
        "fresh_calibration_action_rows": _descriptor(
            calibration_action_rows_path
        ),
        "fresh_calibration_predictions": _descriptor(
            calibration_prediction_path
        ),
        "fresh_calibrated_predictions": _descriptor(
            calibrated_prediction_path
        ),
        "action_advantage_lcb_calibration_artifact": _descriptor(lcb_path),
        "frozen_ranker_identity_report": _descriptor(
            identity_report_path
        ),
        "fresh_calibration_report": _descriptor(
            calibration_report_path
        ),
        "development_gate_report": _descriptor(
            development_report_path
        ),
        "calibration_leakage_role_audit": _descriptor(
            calibration_leakage_path
        ),
        "development_candidate_replay": _descriptor(
            candidate_replay_path
        ),
        "development_baseline_replay": _descriptor(
            baseline_replay_path
        ),
        "development_gate_passed": development_gate["passed"],
        "development_gate_blocking_reason_codes": development_gate[
            "reason_codes"
        ],
        "calibration_frozen": True,
        "ranker_retrained": False,
        "ranker_score_mutated": False,
        "confirmatory_evaluation_started": False,
        "confirmatory_labels_opened": False,
        "confirmatory_label_access_allowed": development_gate["passed"],
        "uses_confirmatory_validation_labels_for_tuning": False,
        "uses_current_oof_or_validation_pnl_for_tuning": False,
        "future_unseen_execution_holdout_required": True,
        **_blocked_safety_fields(),
    }
    freeze_manifest["calibration_freeze_id"] = canonical_json_sha256(
        freeze_manifest
    )
    freeze_manifest_path = (
        run_dir / "hybrid_pairwise_calibration_freeze_manifest.json"
    )
    _write_json(freeze_manifest_path, freeze_manifest)
    return {
        "run_dir": run_dir,
        "identity_report_path": identity_report_path,
        "calibration_report_path": calibration_report_path,
        "development_gate_report_path": development_report_path,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest_sha256": _sha256_file(freeze_manifest_path),
        "development_gate_passed": development_gate["passed"],
        "freeze_manifest": freeze_manifest,
    }


def evaluate_hybrid_pairwise_confirmatory_once(
    config: HybridPairwiseConfirmatoryConfig,
) -> dict[str, Any]:
    """Open the fixed 60-market confirmatory labels exactly once."""

    freeze_path = config.calibration_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_calibration_freeze_manifest_sha256,
        name="calibration freeze manifest",
    )
    freeze = _load_json(freeze_path)
    if freeze.get("calibration_frozen") is not True:
        raise ValueError("calibration artifact is not frozen")
    if freeze.get("development_gate_passed") is not True:
        raise ValueError("development gate did not pass")
    if freeze.get("confirmatory_label_access_allowed") is not True:
        raise ValueError("confirmatory label access is not allowed")
    if (
        freeze.get("ranker_retrained") is not False
        or freeze.get("ranker_score_mutated") is not False
        or freeze.get("confirmatory_labels_opened") is not False
    ):
        raise ValueError("calibration freeze lineage is unsafe")
    _require_blocked_safety(freeze, name="calibration freeze manifest")

    role_manifest_descriptor = _verified_descriptor(
        freeze.get("fresh_role_assignment_manifest"),
        name="fresh role assignment manifest",
    )
    role_rows_descriptor = _verified_descriptor(
        freeze.get("fresh_role_assignment_rows"),
        name="fresh role assignment rows",
    )
    role_manifest = _load_json(Path(role_manifest_descriptor["path"]))
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
    _validate_role_assignment(role_manifest, role_rows)
    confirmatory_role_rows = [
        row for row in role_rows if row["role"] == CONFIRMATORY_ROLE
    ]

    evaluation_id = canonical_json_sha256(
        {
            "calibration_freeze_manifest_sha256": (
                config.expected_calibration_freeze_manifest_sha256
            ),
            "role_assignment_manifest_sha256": role_manifest_descriptor[
                "sha256"
            ],
            "confirmatory_market_ids_sha256": canonical_json_sha256(
                [str(row["market_id"]) for row in confirmatory_role_rows]
            ),
        }
    )
    claim_dir = config.output_dir / ".hybrid_pairwise_confirmatory_claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claim_dir / f"{evaluation_id}.json"
    _claim_confirmatory_evaluation(
        claim_path,
        {
            "schema_version": f"{SCHEMA_PREFIX}-confirmatory-claim-v1",
            "evaluation_id": evaluation_id,
            "calibration_freeze_manifest": _descriptor(freeze_path),
            "confirmatory_labels_opened": False,
            "evaluation_completed": False,
            **_blocked_safety_fields(),
        },
    )
    run_dir = _prepare_run_dir(
        config.output_dir / config.run_id,
        overwrite=config.overwrite_existing,
    )
    try:
        source_protocol_descriptor = _verified_descriptor(
            freeze.get("source_pairwise_protocol"),
            name="source pairwise protocol",
        )
        source_protocol = _load_json(
            Path(source_protocol_descriptor["path"])
        )
        validate_pairwise_action_advantage_lcb_protocol(source_protocol)
        feature_contract_descriptor = _verified_descriptor(
            freeze.get("feature_contract"),
            name="feature contract",
        )
        feature_contract = _load_json(
            Path(feature_contract_descriptor["path"])
        )
        validate_pairwise_action_advantage_lcb_feature_contract(
            feature_contract,
            expected_parent_protocol_sha256=source_protocol_descriptor[
                "sha256"
            ],
        )
        model_descriptor = _verified_descriptor(
            freeze.get("historical_ranker_model"),
            name="historical ranker model",
        )
        model_path = Path(model_descriptor["path"])
        model_sha_before = _sha256_file(model_path)
        booster = _load_frozen_booster(model_path)
        lcb_descriptor = _verified_descriptor(
            freeze.get("action_advantage_lcb_calibration_artifact"),
            name="action advantage LCB artifact",
        )
        lcb_artifact = _load_json(Path(lcb_descriptor["path"]))
        if (
            lcb_artifact.get("confirmatory_labels_opened") is not False
            or lcb_artifact.get("ranker_retrained") is not False
            or lcb_artifact.get("ranker_score_mutated") is not False
        ):
            raise ValueError("LCB artifact lineage is unsafe")

        _update_claim(
            claim_path,
            confirmatory_labels_opened=True,
            evaluation_completed=False,
        )
        confirmatory_rows, corpus_audits = _materialize_fresh_action_rows(
            confirmatory_role_rows,
            feature_columns=tuple(feature_contract["feature_columns"]),
            expected_market_count=CONFIRMATORY_MARKET_COUNT,
        )
        confirmatory_action_rows_path = (
            run_dir / "hybrid_pairwise_confirmatory_action_rows.jsonl"
        )
        _write_jsonl(confirmatory_action_rows_path, confirmatory_rows)
        predictions = _predict_role_rows(
            confirmatory_rows,
            booster=booster,
            feature_columns=tuple(feature_contract["feature_columns"]),
        )
        predictions = _apply_action_advantage_lcb_scores(
            predictions,
            lcb_artifact=lcb_artifact,
        )
        if _sha256_file(model_path) != model_sha_before:
            raise ValueError("frozen ranker model mutated during confirmatory prediction")
        prediction_path = (
            run_dir / "hybrid_pairwise_confirmatory_predictions.jsonl"
        )
        _write_jsonl(prediction_path, predictions)
        entry_threshold = float(
            source_protocol["frozen_execution_contract"][
                "entry_edge_threshold"
            ]
        )
        runner_up_threshold = float(
            source_protocol["frozen_execution_contract"][
                "runner_up_advantage_threshold"
            ]
        )
        candidate_replay = _run_policy_replay(
            predictions,
            score_field="action_advantage_lcb_net_return",
            policy_name="hybrid_frozen_ranker_fresh_action_advantage_lcb",
            entry_threshold=entry_threshold,
            runner_up_advantage_threshold=runner_up_threshold,
        )
        baseline_replay = _run_policy_replay(
            predictions,
            score_field="calibrated_action_expected_net_return",
            policy_name="hybrid_frozen_ranker_uncertainty_unadjusted_baseline",
            entry_threshold=entry_threshold,
            runner_up_advantage_threshold=runner_up_threshold,
        )
        candidate_replay_path = (
            run_dir / "hybrid_pairwise_confirmatory_candidate_replay.jsonl"
        )
        baseline_replay_path = (
            run_dir / "hybrid_pairwise_confirmatory_baseline_replay.jsonl"
        )
        _write_jsonl(candidate_replay_path, candidate_replay)
        _write_jsonl(baseline_replay_path, baseline_replay)
        candidate_metrics = _accepted_bet_metrics(candidate_replay)
        baseline_metrics = _accepted_bet_metrics(baseline_replay)
        robustness = _market_robustness(candidate_replay, baseline_replay)
        gate = _confirmatory_gate(
            protocol=source_protocol,
            action_rows=confirmatory_rows,
            candidate_replay=candidate_replay,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            robustness=robustness,
        )
        confirmatory_leakage_audit = {
            "schema_version": (
                f"{SCHEMA_PREFIX}-confirmatory-leakage-role-audit-v1"
            ),
            "run_id": config.run_id,
            "evaluation_id": evaluation_id,
            "confirmatory_market_count": len(
                {str(row["market_id"]) for row in confirmatory_rows}
            ),
            "confirmatory_decision_group_count": (
                len(confirmatory_rows) // len(REQUIRED_ACTIONS)
            ),
            "feature_causality_violation_count": sum(
                int(audit["feature_causality_violation_count"])
                for audit in corpus_audits
            ),
            "forbidden_inference_field_violation_count": sum(
                int(
                    row["target_used_as_decision_input"] is not False
                    or row["outcome_fields_used_as_decision_input"] is not False
                )
                for row in confirmatory_rows
            ),
            "complete_five_action_grid": True,
            "confirmatory_labels_used_for_report_only": True,
            "confirmatory_labels_used_for_tuning": False,
            "ranker_retrained": False,
            "ranker_score_mutated": False,
            "leakage_and_role_audit_passed": (
                all(
                    int(audit["feature_causality_violation_count"]) == 0
                    for audit in corpus_audits
                )
                and all(
                    row["target_used_as_decision_input"] is False
                    and row["outcome_fields_used_as_decision_input"] is False
                    for row in confirmatory_rows
                )
            ),
            **_blocked_safety_fields(),
        }
        confirmatory_leakage_path = (
            run_dir / "hybrid_pairwise_confirmatory_leakage_role_audit.json"
        )
        _write_json(
            confirmatory_leakage_path,
            confirmatory_leakage_audit,
        )
        report = {
            "schema_version": f"{SCHEMA_PREFIX}-confirmatory-report-v1",
            "run_id": config.run_id,
            "evaluation_id": evaluation_id,
            "calibration_freeze_manifest": _descriptor(freeze_path),
            "confirmatory_market_count": CONFIRMATORY_MARKET_COUNT,
            "confirmatory_action_row_count": len(confirmatory_rows),
            "confirmatory_decision_group_count": (
                len(confirmatory_rows) // len(REQUIRED_ACTIONS)
            ),
            "corpus_audit_count": len(corpus_audits),
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "candidate_minus_baseline_net_pnl": (
                float(candidate_metrics["net_pnl_sum"])
                - float(baseline_metrics["net_pnl_sum"])
            ),
            "market_robustness_diagnostics": robustness,
            "confirmatory_gate_checks": gate["checks"],
            "confirmatory_gate_passed": gate["passed"],
            "confirmatory_gate_blocking_reason_codes": gate[
                "reason_codes"
            ],
            "confirmatory_labels_used_for_report_only": True,
            "confirmatory_labels_used_for_tuning": False,
            "confirmatory_evaluation_one_shot": True,
            "ranker_retrained": False,
            "ranker_score_mutated": False,
            "future_unseen_execution_holdout_required": True,
            **_blocked_safety_fields(),
        }
        report_path = (
            run_dir / "hybrid_pairwise_confirmatory_validation_report.json"
        )
        _write_json(report_path, report)
        _write_text(
            run_dir / "hybrid_pairwise_confirmatory_validation_report.md",
            _confirmatory_markdown(report),
        )
        candidate_freeze = {
            "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
            "run_id": config.run_id,
            "evaluation_id": evaluation_id,
            "calibration_freeze_manifest": _descriptor(freeze_path),
            "confirmatory_action_rows": _descriptor(
                confirmatory_action_rows_path
            ),
            "confirmatory_predictions": _descriptor(prediction_path),
            "confirmatory_candidate_replay": _descriptor(
                candidate_replay_path
            ),
            "confirmatory_baseline_replay": _descriptor(
                baseline_replay_path
            ),
            "confirmatory_validation_report": _descriptor(report_path),
            "confirmatory_leakage_role_audit": _descriptor(
                confirmatory_leakage_path
            ),
            "confirmatory_gate_passed": gate["passed"],
            "confirmatory_gate_blocking_reason_codes": gate[
                "reason_codes"
            ],
            "diagnostic_candidate_evidence_passed": gate["passed"],
            "source_model_candidate_eligible": False,
            "promotion_evidence_eligible": False,
            "future_unseen_execution_holdout_required": True,
            "ranker_retrained": False,
            "ranker_score_mutated": False,
            "confirmatory_labels_used_for_tuning": False,
            **_blocked_safety_fields(),
        }
        candidate_freeze["candidate_freeze_id"] = canonical_json_sha256(
            candidate_freeze
        )
        candidate_freeze_path = (
            run_dir / "hybrid_pairwise_candidate_freeze_manifest.json"
        )
        _write_json(candidate_freeze_path, candidate_freeze)
        _update_claim(
            claim_path,
            confirmatory_labels_opened=True,
            evaluation_completed=True,
            confirmatory_gate_passed=gate["passed"],
            candidate_freeze_manifest=_descriptor(candidate_freeze_path),
        )
        return {
            "run_dir": run_dir,
            "claim_path": claim_path,
            "report_path": report_path,
            "candidate_freeze_path": candidate_freeze_path,
            "candidate_freeze_sha256": _sha256_file(
                candidate_freeze_path
            ),
            "confirmatory_gate_passed": gate["passed"],
            "report": report,
            "candidate_freeze": candidate_freeze,
        }
    except BaseException:
        _update_claim(
            claim_path,
            confirmatory_labels_opened=bool(
                _load_json(claim_path).get("confirmatory_labels_opened")
            ),
            evaluation_completed=False,
            evaluation_failed_closed=True,
        )
        raise


def _load_and_validate_lineage(
    config: HybridPairwiseFreshCalibrationConfig,
) -> dict[str, Any]:
    paths = {
        "hybrid_protocol": config.hybrid_protocol_path.resolve(),
        "source_protocol": config.source_pairwise_protocol_path.resolve(),
        "feature_contract": config.feature_contract_path.resolve(),
        "ranker_descriptor": config.historical_ranker_descriptor_path.resolve(),
        "ranker_manifest": config.historical_ranker_manifest_path.resolve(),
        "role_manifest": config.fresh_role_assignment_manifest_path.resolve(),
    }
    expected = {
        "hybrid_protocol": config.expected_hybrid_protocol_sha256,
        "source_protocol": config.expected_source_pairwise_protocol_sha256,
        "feature_contract": config.expected_feature_contract_sha256,
        "ranker_descriptor": (
            config.expected_historical_ranker_descriptor_sha256
        ),
        "ranker_manifest": config.expected_historical_ranker_manifest_sha256,
        "role_manifest": (
            config.expected_fresh_role_assignment_manifest_sha256
        ),
    }
    for name, path in paths.items():
        _verify_pin(path, expected[name], name=name.replace("_", " "))
    hybrid_protocol = _load_json(paths["hybrid_protocol"])
    source_protocol = _load_json(paths["source_protocol"])
    feature_contract = _load_json(paths["feature_contract"])
    ranker_descriptor = _load_json(paths["ranker_descriptor"])
    ranker_manifest = _load_json(paths["ranker_manifest"])
    role_manifest = _load_json(paths["role_manifest"])
    if hybrid_protocol.get("schema_version") != HYBRID_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("hybrid protocol schema mismatch")
    validate_pairwise_action_advantage_lcb_protocol(source_protocol)
    validate_pairwise_action_advantage_lcb_feature_contract(
        feature_contract,
        expected_parent_protocol_sha256=expected["source_protocol"],
    )
    frozen_ranker = dict(hybrid_protocol.get("historical_ranker_freeze") or {})
    identity_checks = {
        "descriptor": frozen_ranker.get("descriptor_sha256")
        == expected["ranker_descriptor"],
        "freeze_id": frozen_ranker.get("freeze_id")
        == ranker_descriptor.get("freeze_id")
        == ranker_manifest.get("freeze_id"),
        "model": frozen_ranker.get("model_sha256")
        == ranker_descriptor.get("model_sha256")
        == ranker_manifest.get("model_sha256"),
        "dataset": frozen_ranker.get("dataset_hash")
        == ranker_descriptor.get("dataset_hash")
        == ranker_manifest.get("dataset_hash"),
        "oof_dataset": frozen_ranker.get("oof_dataset_hash")
        == ranker_manifest.get("oof_dataset_hash"),
        "split": frozen_ranker.get("split_hash")
        == ranker_descriptor.get("split_hash")
        == ranker_manifest.get("split_hash"),
        "model_config": frozen_ranker.get("model_config_hash")
        == ranker_descriptor.get("model_config_hash")
        == ranker_manifest.get("model_config_hash"),
        "fresh_calibration": frozen_ranker.get("fresh_calibration_required")
        is True,
        "execution_ineligible": frozen_ranker.get(
            "rank_scores_execution_eligible"
        )
        is False
        and ranker_descriptor.get("rank_scores_execution_eligible") is False
        and ranker_manifest.get("rank_scores_execution_eligible") is False,
        "source_protocol": hybrid_protocol.get(
            "source_pairwise_protocol_sha256"
        )
        == expected["source_protocol"],
        "feature_contract": hybrid_protocol.get(
            "source_feature_contract_sha256"
        )
        == expected["feature_contract"],
    }
    failed = sorted(
        name for name, passed in identity_checks.items() if not passed
    )
    if failed:
        raise ValueError("frozen historical ranker identity mismatch: " + ", ".join(failed))
    model_descriptor = _verified_descriptor(
        ranker_descriptor.get("model"),
        name="historical ranker model",
    )
    if model_descriptor["sha256"] != frozen_ranker["model_sha256"]:
        raise ValueError("historical ranker model file hash mismatch")
    oof_descriptor = _verified_descriptor(
        ranker_descriptor.get("train_oof_predictions"),
        name="historical OOF predictions",
    )
    if oof_descriptor["sha256"] != ranker_manifest[
        "train_oof_predictions"
    ]["sha256"]:
        raise ValueError("historical OOF prediction identity mismatch")

    role_rows_descriptor = _verified_descriptor(
        role_manifest.get("selected_rows"),
        name="fresh role assignment rows",
    )
    role_rows = _load_jsonl(Path(role_rows_descriptor["path"]))
    _validate_role_assignment(role_manifest, role_rows)
    precollection_freeze_descriptor = _verified_descriptor(
        role_manifest.get("hybrid_precollection_freeze"),
        name="hybrid precollection freeze",
    )
    final_quarantine_descriptor = _verified_descriptor(
        role_manifest.get("final_prior_lineage_quarantine"),
        name="final prior-lineage quarantine",
    )
    _validate_fresh_role_lineage(
        role_manifest=role_manifest,
        role_rows=role_rows,
        hybrid_protocol=hybrid_protocol,
        precollection_freeze_descriptor=precollection_freeze_descriptor,
        final_quarantine_descriptor=final_quarantine_descriptor,
    )
    return {
        "hybrid_protocol": hybrid_protocol,
        "hybrid_protocol_descriptor": _descriptor(paths["hybrid_protocol"]),
        "source_protocol": source_protocol,
        "source_protocol_descriptor": _descriptor(paths["source_protocol"]),
        "feature_contract": feature_contract,
        "feature_contract_descriptor": _descriptor(paths["feature_contract"]),
        "ranker_descriptor": ranker_descriptor,
        "ranker_descriptor_descriptor": _descriptor(
            paths["ranker_descriptor"]
        ),
        "ranker_manifest": ranker_manifest,
        "ranker_manifest_descriptor": _descriptor(paths["ranker_manifest"]),
        "role_manifest": role_manifest,
        "role_manifest_descriptor": _descriptor(paths["role_manifest"]),
        "role_rows": role_rows,
        "role_rows_descriptor": role_rows_descriptor,
        "precollection_freeze_descriptor": precollection_freeze_descriptor,
        "final_quarantine_descriptor": final_quarantine_descriptor,
    }


def _validate_role_assignment(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if manifest.get("schema_version") != ROLE_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("fresh role assignment schema mismatch")
    if manifest.get("role_assignment_ready") is not True:
        raise ValueError("fresh role assignment is not ready")
    if int(manifest.get("selected_market_count") or 0) != TOTAL_FRESH_MARKET_COUNT:
        raise ValueError("fresh role manifest selected market count is invalid")
    _require_blocked_safety(manifest, name="fresh role assignment manifest")
    if (
        manifest.get("labels_or_outcomes_opened_for_role_assignment")
        is not False
        or manifest.get("prior_market_overlap_count") != 0
        or manifest.get("role_market_overlap_count") != 0
        or manifest.get("chronology_validation_passed") is not True
    ):
        raise ValueError("fresh role assignment leakage boundary failed")
    if len(rows) != TOTAL_FRESH_MARKET_COUNT:
        raise ValueError("fresh role assignment must contain 105 markets")
    expected_ranks = list(range(1, TOTAL_FRESH_MARKET_COUNT + 1))
    ranks = [int(row.get("selection_rank") or 0) for row in rows]
    if ranks != expected_ranks:
        raise ValueError("fresh role assignment ranks are incomplete")
    market_ids = [str(row.get("market_id") or "") for row in rows]
    if any(not market_id for market_id in market_ids):
        raise ValueError("fresh role assignment market identity is missing")
    if len(market_ids) != len(set(market_ids)):
        raise ValueError("fresh role assignment market identity is duplicated")
    counts = Counter(str(row.get("role") or "") for row in rows)
    if counts != Counter(
        {
            CALIBRATION_ROLE: CALIBRATION_MARKET_COUNT,
            CONFIRMATORY_ROLE: CONFIRMATORY_MARKET_COUNT,
        }
    ):
        raise ValueError("fresh role assignment counts do not match 45/60")
    if any(
        row.get("execution_compatibility_validated_before_label_access")
        is not True
        or row.get("labels_or_outcomes_opened_for_role_assignment")
        is not False
        for row in rows
    ):
        raise ValueError("fresh role rows violate pre-label validation")
    calibration_max = max(
        int(row.get("maximum_decision_ts") or 0)
        for row in rows
        if row["role"] == CALIBRATION_ROLE
    )
    confirmatory_min = min(
        int(row.get("minimum_decision_ts") or 0)
        for row in rows
        if row["role"] == CONFIRMATORY_ROLE
    )
    if calibration_max <= 0 or confirmatory_min <= calibration_max:
        raise ValueError("fresh role chronology overlaps")


def _validate_fresh_role_lineage(
    *,
    role_manifest: dict[str, Any],
    role_rows: list[dict[str, Any]],
    hybrid_protocol: dict[str, Any],
    precollection_freeze_descriptor: dict[str, str],
    final_quarantine_descriptor: dict[str, str],
) -> None:
    precollection_freeze = _load_json(
        Path(precollection_freeze_descriptor["path"])
    )
    final_quarantine = _load_json(
        Path(final_quarantine_descriptor["path"])
    )
    if (
        precollection_freeze.get("schema_version")
        != "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
        or precollection_freeze.get("candidate_lineage")
        != hybrid_protocol.get("candidate_lineage")
        or precollection_freeze.get("fresh_role_plan")
        != hybrid_protocol.get("fresh_role_plan")
        or precollection_freeze.get("collection_plan")
        != hybrid_protocol.get("collection_plan")
        or precollection_freeze.get("ranker_retraining_allowed") is not False
        or precollection_freeze.get("ranker_score_mutation_allowed") is not False
    ):
        raise ValueError("hybrid precollection freeze lineage mismatch")
    if (
        final_quarantine.get("status") != "prior_lineage_complete"
        or final_quarantine.get("final") is not True
        or final_quarantine.get("active_prior_lineage_complete") is not True
        or final_quarantine.get("includes_issue175_through_issue179")
        is not True
    ):
        raise ValueError("final prior-lineage quarantine is incomplete")
    _require_blocked_safety(
        precollection_freeze,
        name="hybrid precollection freeze",
    )
    _require_blocked_safety(
        final_quarantine,
        name="final prior-lineage quarantine",
    )
    if (
        role_manifest.get("hybrid_precollection_freeze")
        != precollection_freeze_descriptor
        or role_manifest.get("final_prior_lineage_quarantine")
        != final_quarantine_descriptor
    ):
        raise ValueError("fresh role manifest lineage descriptors mismatch")
    prior_market_ids = {
        str(value)
        for value in final_quarantine.get("prior_market_ids") or []
    }
    selected_market_ids = {str(row["market_id"]) for row in role_rows}
    if selected_market_ids & prior_market_ids:
        raise ValueError("fresh role assignment overlaps final quarantine")
    minimum_collection_decision_ts = int(
        precollection_freeze.get("minimum_collection_decision_ts") or 0
    )
    if minimum_collection_decision_ts <= int(
        final_quarantine.get("maximum_prior_decision_ts") or 0
    ):
        raise ValueError("fresh collection boundary does not follow quarantine")
    if any(
        int(row.get("minimum_decision_ts") or 0)
        < minimum_collection_decision_ts
        for row in role_rows
    ):
        raise ValueError("fresh role row predates precollection freeze boundary")
    if any(
        row.get("source_precollection_freeze_sha256")
        != precollection_freeze_descriptor["sha256"]
        or row.get("source_final_quarantine_sha256")
        != final_quarantine_descriptor["sha256"]
        for row in role_rows
    ):
        raise ValueError("fresh role row lineage hash mismatch")


def _fresh_split_manifest(
    *,
    run_id: str,
    role_rows: list[dict[str, Any]],
    role_manifest_descriptor: dict[str, str],
    precollection_freeze_descriptor: dict[str, str],
    final_quarantine_descriptor: dict[str, str],
) -> dict[str, Any]:
    roles = {}
    previous_max: int | None = None
    role_market_sets = {}
    for role in (CALIBRATION_ROLE, CONFIRMATORY_ROLE):
        selected = [row for row in role_rows if row["role"] == role]
        market_ids = [str(row["market_id"]) for row in selected]
        minimum = min(int(row["minimum_decision_ts"]) for row in selected)
        maximum = max(int(row["maximum_decision_ts"]) for row in selected)
        if previous_max is not None and minimum <= previous_max:
            raise ValueError("fresh split chronology overlaps")
        previous_max = maximum
        role_market_sets[role] = set(market_ids)
        roles[role] = {
            "market_count": len(market_ids),
            "market_ids": market_ids,
            "market_ids_sha256": canonical_json_sha256(market_ids),
            "minimum_decision_ts": minimum,
            "maximum_decision_ts": maximum,
        }
    overlap = role_market_sets[CALIBRATION_ROLE] & role_market_sets[
        CONFIRMATORY_ROLE
    ]
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-fresh-split-manifest-v1",
        "run_id": run_id,
        "fresh_role_assignment_manifest": role_manifest_descriptor,
        "hybrid_precollection_freeze": precollection_freeze_descriptor,
        "final_prior_lineage_quarantine": final_quarantine_descriptor,
        "roles": roles,
        "role_market_overlap_count": len(overlap),
        "chronology_validation_passed": not overlap,
        "labels_or_outcomes_used_for_split_assignment": False,
        "confirmatory_labels_opened": False,
        **_blocked_safety_fields(),
    }
    manifest["split_id"] = canonical_json_sha256(manifest)
    return manifest


def _materialize_fresh_action_rows(
    role_rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    expected_market_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    audits = []
    for role_row in role_rows:
        corpus_dir = Path(str(role_row["source_corpus_dir"])).resolve()
        rows, audit = _load_corpus_action_rows(
            corpus_dir,
            role_row=role_row,
            feature_columns=feature_columns,
        )
        if audit["blocking_reason_codes"]:
            raise ValueError(
                f"fresh corpus materialization failed for {corpus_dir}: "
                + ", ".join(audit["blocking_reason_codes"])
            )
        output.extend(rows)
        audits.append(audit)
    output.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["action"]),
        )
    )
    markets = {str(row["market_id"]) for row in output}
    if len(markets) != expected_market_count:
        raise ValueError("fresh action rows have incomplete market support")
    _validate_complete_action_grid(output)
    return output, audits


def _score_only_oof_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(path)
    score_rows = []
    target_field_present_count = 0
    for row in rows:
        target_field_present_count += int(
            "target_net_pnl_per_contract" in row
        )
        missing = sorted(field for field in OOF_SCORE_FIELDS if field not in row)
        if missing:
            raise ValueError(f"historical OOF score fields are missing: {missing}")
        score_row = {field: row[field] for field in OOF_SCORE_FIELDS}
        if not math.isfinite(
            float(score_row["pairwise_group_normalized_rank_score"])
        ):
            raise ValueError("historical OOF normalized rank score is not finite")
        score_rows.append(score_row)
    _validate_complete_action_grid(score_rows)
    market_ids = {str(row["market_id"]) for row in score_rows}
    if len(market_ids) != 75:
        raise ValueError("historical OOF score-only rows require 75 markets")
    score_rows.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["action"]),
        )
    )
    return score_rows, {
        "source_path": str(path.resolve()),
        "source_sha256": _sha256_file(path),
        "source_row_count": len(rows),
        "score_only_row_count": len(score_rows),
        "oof_market_count": len(market_ids),
        "target_field_present_count": target_field_present_count,
        "target_field_values_used_for_bucket_construction": False,
        "score_bucket_boundaries_source": (
            "historical_train_oof_group_normalized_rank_scores_only"
        ),
    }


def _validate_complete_action_grid(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        grouped[
            (str(row["market_id"]), int(row["decision_ts"]))
        ].add(str(row["action"]))
    if not grouped or any(actions != set(REQUIRED_ACTIONS) for actions in grouped.values()):
        raise ValueError("complete five-action decision grid is required")


def _load_frozen_booster(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _claim_confirmatory_evaluation(
    path: Path,
    payload: dict[str, Any],
) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise ValueError(
            "confirmatory evaluation was already claimed"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _update_claim(path: Path, **updates: Any) -> None:
    payload = _load_json(path)
    payload.update(updates)
    payload["claim_id"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "claim_id"}
    )
    _write_json(path, payload)


def _prepare_run_dir(path: Path, *, overwrite: bool) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        if not overwrite:
            raise ValueError(f"run directory already exists: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _identity_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Frozen Ranker Identity",
            "",
            f"- freeze id: `{report['historical_ranker_freeze_id']}`",
            f"- model SHA-256: `{report['model_sha256']}`",
            "- model unchanged after prediction: `true`",
            "- ranker retrained: `false`",
            "- ranker score mutated: `false`",
            "- execution eligible: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Fresh Calibration",
            "",
            (
                "- calibration markets: "
                f"`{report['fresh_calibration_market_count']}`"
            ),
            (
                "- calibration action rows: "
                f"`{report['fresh_calibration_action_row_count']}`"
            ),
            "- historical OOF target values used for buckets: `false`",
            "- confirmatory labels opened: `false`",
            "- ranker retrained: `false`",
            "",
        ]
    )


def _development_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Development Gate",
            "",
            (
                "- gate passed: "
                f"`{str(report['development_gate_passed']).lower()}`"
            ),
            (
                "- blocking reasons: "
                f"`{report['development_gate_blocking_reason_codes']}`"
            ),
            "- calibration frozen before confirmatory access: `true`",
            (
                "- confirmatory label access allowed: "
                f"`{str(report['confirmatory_label_access_allowed']).lower()}`"
            ),
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _confirmatory_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hybrid Pairwise Confirmatory Validation",
            "",
            f"- evaluation id: `{report['evaluation_id']}`",
            (
                "- confirmatory gate passed: "
                f"`{str(report['confirmatory_gate_passed']).lower()}`"
            ),
            (
                "- blocking reasons: "
                f"`{report['confirmatory_gate_blocking_reason_codes']}`"
            ),
            "- confirmatory labels used for tuning: `false`",
            "- future unseen execution holdout required: `true`",
            "- paper/live/handoff unlock: `false`",
            "",
        ]
    )


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _require_blocked_safety(payload: dict[str, Any], *, name: str) -> None:
    expected = _blocked_safety_fields()
    if any(payload.get(key) is not value for key, value in expected.items()):
        raise ValueError(f"{name} safety contract failed")


def _verified_descriptor(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} descriptor is missing")
    path = Path(str(payload.get("path") or "")).resolve()
    digest = str(payload.get("sha256") or "")
    _verify_pin(path, digest, name=name)
    return {"path": str(path), "sha256": digest.lower()}


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _verify_pin(path: Path, expected_sha256: str, *, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    _require_sha256(expected_sha256, name=f"{name} SHA-256")
    if _sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"{name} SHA-256 mismatch")


def _require_sha256(value: str, *, name: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _find_fields(
    payload: Any,
    forbidden: set[str],
    prefix: str = "",
) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.add(path)
            found.update(_find_fields(value, forbidden, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(
                _find_fields(value, forbidden, f"{prefix}[{index}]")
            )
    return found
