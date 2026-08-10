"""Frozen regime-conditioned EV contract and outcome-free shadow consumer.

This module deliberately stops at diagnostic shadow evidence.  It validates a
separately calibrated, immutable artifact and applies it only to decision-time
features before intersecting candidates with the existing execution guard.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2 import (
    EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    DEFAULT_FORWARD_SHADOW_EXECUTION_COST,
    _execution_guard_status,
    _forbidden_fields_by_row,
    _load_forward_shadow_rows,
    _lookup_value,
    _normalize_forward_shadow_row,
    _recursive_forbidden_field_paths,
    _safety_report_fields,
    _sha256_file,
    _validate_safety_flags,
    _write_json,
    _write_text,
)

FROZEN_REGIME_CONDITIONED_EV_ARTIFACT_NAME = (
    "execution_layer_v2_frozen_regime_conditioned_ev_v1"
)
FROZEN_REGIME_CONDITIONED_EV_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-frozen-regime-conditioned-ev-v1"
)
FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME = (
    "execution_layer_v2_frozen_regime_conditioned_ev_v2"
)
FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-frozen-regime-conditioned-ev-v2"
)
REGIME_CONDITIONED_EV_FORWARD_SHADOW_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-forward-shadow-v1"
)
REGIME_CONDITIONED_EV_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-execution-layer-v2-regime-conditioned-ev-forward-shadow-manifest-v1"
)
CURRENT_75_ROW_REPLAY_RUN_ID = (
    "execution-layer-v2-regime-entry-edge-replay-20260710T123338Z"
)
LATEST_ONE_HOUR_RECONCILED_RUN_ID = (
    "execution-layer-v2-one-hour-remap-paper-goal-20260710T042608Z-"
    "clob-settlement-reconciled"
)
V2_CALIBRATION_ROW_SCHEMA_PATH = (
    Path(__file__).resolve().parents[5]
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_regime_conditioned_ev_calibration_row_v2.schema.json"
)

REGIME_CONDITIONED_EV_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "canonical_o_score_and_action_margin": (
        "canonical_o_action_score",
        "action_score_margin",
    ),
    "btc_anchor_direction": (
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
    ),
    "market_price_value": (
        "p_up",
        "p_down",
        "execution_price",
    ),
    "execution_quality": (
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "time_to_close_seconds",
    ),
    "pre_entry_exposure_state": (
        "entry_index_within_market",
        "cumulative_market_exposure_before_entry",
        "same_side_reentry",
        "side_flip",
    ),
}
REGIME_CONDITIONED_EV_REQUIRED_FEATURES = tuple(
    feature
    for group_features in REGIME_CONDITIONED_EV_FEATURE_GROUPS.values()
    for feature in group_features
)
REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "canonical_o_score_and_action_margin": (
        "canonical_o_action_score",
        "action_score_margin",
    ),
    "btc_anchor_direction": (
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
    ),
    "market_price_value": (
        "selected_side_probability",
        "execution_price",
        "selected_side_probability_minus_execution_price",
    ),
    "execution_quality": REGIME_CONDITIONED_EV_FEATURE_GROUPS[
        "execution_quality"
    ],
    "pre_entry_exposure_state": REGIME_CONDITIONED_EV_FEATURE_GROUPS[
        "pre_entry_exposure_state"
    ],
}
_HEX_DIGEST_LENGTH = 64
_ARTIFACT_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_name",
    "diagnostic_only",
    "frozen",
    "decision_time_safe",
    "uses_validation_labels_for_tuning",
    "market_implied_probability_used_as_direct_fair_value_ev",
    "market_implied_probability_used_as_conditioning_feature",
    "market_implied_probability_used_as_regime_direction_vote",
    "no_outcome_field_usage",
    "no_oracle_field_usage",
    "no_future_return_field_usage",
    "source_score_mutation_enabled",
    "o_score_mutation_enabled",
    "paper_only",
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
    "fit_provenance",
    "independence_constraints",
    "feature_groups",
    "coefficients",
}
_V2_ARTIFACT_TOP_LEVEL_FIELDS = _ARTIFACT_TOP_LEVEL_FIELDS | {
    "calibration_protocol"
}


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVForwardShadowConfig:
    """Configuration for diagnostic-only regime-conditioned EV shadowing."""

    run_id: str
    input_path: Path | str
    output_dir: Path | str
    frozen_regime_conditioned_ev_artifact: Path | str | None = None
    entry_ev_threshold: float = 0.02
    default_execution_cost: float = DEFAULT_FORWARD_SHADOW_EXECUTION_COST
    max_rows: int | None = None
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        if not math.isfinite(self.entry_ev_threshold) or self.entry_ev_threshold < 0:
            raise ValueError("entry_ev_threshold must be finite and non-negative")
        if not math.isfinite(self.default_execution_cost) or self.default_execution_cost < 0:
            raise ValueError("default_execution_cost must be finite and non-negative")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.frozen_regime_conditioned_ev_artifact is not None:
            object.__setattr__(
                self,
                "frozen_regime_conditioned_ev_artifact",
                Path(self.frozen_regime_conditioned_ev_artifact),
            )
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        if self.frozen_regime_conditioned_ev_artifact is not None:
            payload["frozen_regime_conditioned_ev_artifact"] = str(
                self.frozen_regime_conditioned_ev_artifact
            )
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2RegimeConditionedEVForwardShadowResult:
    """Written regime-conditioned EV forward-shadow bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    artifact_validation_report: dict[str, Any]
    forward_shadow_report: dict[str, Any]
    manifest: dict[str, Any]


def frozen_regime_conditioned_ev_artifact_contract(
    *, schema_version: str = FROZEN_REGIME_CONDITIONED_EV_SCHEMA_VERSION
) -> dict[str, Any]:
    """Return the immutable semantic contract enforced by the validator."""

    is_v2 = schema_version == FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION
    feature_groups = (
        REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS
        if is_v2
        else REGIME_CONDITIONED_EV_FEATURE_GROUPS
    )
    return {
        "artifact_name": (
            FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME
            if is_v2
            else FROZEN_REGIME_CONDITIONED_EV_ARTIFACT_NAME
        ),
        "schema_version": schema_version,
        "required_feature_groups": {
            name: list(features)
            for name, features in feature_groups.items()
        },
        "independence_constraints": (
            {
                "selected_side_probability_single_group": "market_price_value",
                "p_up_p_down_used_only_to_derive_selected_side_probability": True,
                "btc_anchor_fields_single_group": "btc_anchor_direction",
                "btc_anchor_maximum_signal_vote_weight": 1.0,
                "correlated_momentum_reference_counted_as_independent_votes": False,
            }
            if is_v2
            else {
                "p_up_p_down_single_group": "market_price_value",
                "btc_anchor_fields_single_group": "btc_anchor_direction",
                "btc_anchor_maximum_signal_vote_weight": 1.0,
                "correlated_momentum_reference_counted_as_independent_votes": False,
            }
        ),
        "market_implied_probability_semantics": {
            "market_implied_probability_used_as_direct_fair_value_ev": False,
            "market_implied_probability_used_as_conditioning_feature": True,
            "market_implied_probability_used_as_regime_direction_vote": False,
        },
        "excluded_fit_run_ids": (
            [CURRENT_75_ROW_REPLAY_RUN_ID, LATEST_ONE_HOUR_RECONCILED_RUN_ID]
            if is_v2
            else [CURRENT_75_ROW_REPLAY_RUN_ID]
        ),
        "coefficients_must_be_fitted_separately": True,
        "allowed_top_level_fields": sorted(
            _V2_ARTIFACT_TOP_LEVEL_FIELDS if is_v2 else _ARTIFACT_TOP_LEVEL_FIELDS
        ),
        "outcome_fields_allowed_as_inputs": False,
        "settled_outcomes_allowed_as_historical_fit_targets": is_v2,
        "subtract_execution_cost": not is_v2,
        "production_gate_implemented": False,
    }


def validate_frozen_regime_conditioned_ev_artifact(
    path: Path | str | None,
) -> dict[str, Any]:
    """Load and strictly validate a frozen regime-conditioned EV artifact."""

    contract = frozen_regime_conditioned_ev_artifact_contract()
    if path is None:
        return _invalid_artifact_result(
            path=None,
            status="missing_frozen_regime_conditioned_ev_artifact",
            reason_codes=["missing_frozen_regime_conditioned_ev_artifact"],
            contract=contract,
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return _invalid_artifact_result(
            path=str(resolved),
            status="frozen_regime_conditioned_ev_artifact_not_found",
            reason_codes=["frozen_regime_conditioned_ev_artifact_not_found"],
            contract=contract,
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _invalid_artifact_result(
            path=str(resolved),
            status="frozen_regime_conditioned_ev_artifact_invalid_json",
            reason_codes=["frozen_regime_conditioned_ev_artifact_invalid_json"],
            contract=contract,
            sha256=_sha256_file(resolved),
        )
    if not isinstance(payload, dict):
        return _invalid_artifact_result(
            path=str(resolved),
            status="frozen_regime_conditioned_ev_artifact_invalid_payload",
            reason_codes=["frozen_regime_conditioned_ev_artifact_not_object"],
            contract=contract,
            sha256=_sha256_file(resolved),
        )

    schema_version = str(payload.get("schema_version") or "")
    supported_schema_versions = {
        FROZEN_REGIME_CONDITIONED_EV_SCHEMA_VERSION,
        FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION,
    }
    if schema_version not in supported_schema_versions:
        return _invalid_artifact_result(
            path=str(resolved),
            status="frozen_regime_conditioned_ev_artifact_unsupported_schema",
            reason_codes=["regime_conditioned_ev_schema_version_unsupported"],
            contract=contract,
            sha256=_sha256_file(resolved),
        )
    contract = frozen_regime_conditioned_ev_artifact_contract(
        schema_version=schema_version
    )
    is_v2 = schema_version == FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION
    allowed_top_level_fields = (
        _V2_ARTIFACT_TOP_LEVEL_FIELDS if is_v2 else _ARTIFACT_TOP_LEVEL_FIELDS
    )
    feature_groups = (
        REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS
        if is_v2
        else REGIME_CONDITIONED_EV_FEATURE_GROUPS
    )

    reasons: list[str] = []
    if "market_implied_probability_used_for_ev" in payload:
        reasons.append(
            "legacy_ambiguous_market_implied_probability_used_for_ev_present"
        )
    unknown_top_level = sorted(set(payload) - allowed_top_level_fields)
    missing_top_level = sorted(allowed_top_level_fields - set(payload))
    if unknown_top_level:
        reasons.extend(
            f"regime_conditioned_ev_unknown_top_level_field:{field}"
            for field in unknown_top_level
        )
    if missing_top_level:
        reasons.extend(
            f"regime_conditioned_ev_missing_top_level_field:{field}"
            for field in missing_top_level
        )
    _require_equal(
        payload,
        "artifact_name",
        contract["artifact_name"],
        "regime_conditioned_ev_artifact_name_mismatch",
        reasons,
    )
    _require_equal(
        payload,
        "schema_version",
        schema_version,
        "regime_conditioned_ev_schema_version_mismatch",
        reasons,
    )
    for field_name in (
        "frozen",
        "decision_time_safe",
        "diagnostic_only",
        "paper_only",
        "no_outcome_field_usage",
        "no_oracle_field_usage",
        "no_future_return_field_usage",
        "market_implied_probability_used_as_conditioning_feature",
    ):
        if payload.get(field_name) is not True:
            reasons.append(f"regime_conditioned_ev_artifact_{field_name}_not_true")
    false_flags = (
        "uses_validation_labels_for_tuning",
        "market_implied_probability_used_as_direct_fair_value_ev",
        "market_implied_probability_used_as_regime_direction_vote",
        "source_score_mutation_enabled",
        "o_score_mutation_enabled",
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
    for field_name in false_flags:
        if payload.get(field_name) is not False:
            reasons.append(f"regime_conditioned_ev_artifact_{field_name}_not_false")

    forbidden_paths = _recursive_forbidden_field_paths(
        payload,
        set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS),
    )
    if forbidden_paths:
        reasons.append("regime_conditioned_ev_artifact_forbidden_fields_present")
    _validate_fit_provenance(payload.get("fit_provenance"), reasons, is_v2=is_v2)
    _validate_independence_constraints(
        payload.get("independence_constraints"), reasons, contract=contract
    )
    _validate_feature_groups(
        payload.get("feature_groups"), reasons, feature_groups=feature_groups
    )
    _validate_coefficient_contract(
        payload.get("coefficients"),
        reasons,
        feature_groups=feature_groups,
        subtract_execution_cost=contract["subtract_execution_cost"],
    )
    if is_v2:
        _validate_v2_calibration_protocol(payload.get("calibration_protocol"), reasons)

    reasons = sorted(set(reasons))
    valid = not reasons
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "payload": payload,
        "valid": valid,
        "status": (
            "frozen_regime_conditioned_ev_artifact_valid"
            if valid
            else "frozen_regime_conditioned_ev_artifact_invalid"
        ),
        "blocking_reason_codes": reasons,
        "forbidden_field_paths": forbidden_paths,
        "contract": contract,
        "artifact_schema_version": schema_version,
    }


def run_execution_layer_v2_regime_conditioned_ev_forward_shadow(
    config: ExecutionLayerV2RegimeConditionedEVForwardShadowConfig,
) -> ExecutionLayerV2RegimeConditionedEVForwardShadowResult:
    """Apply a valid frozen artifact to fresh outcome-free trace rows."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"forward shadow input not found: {config.input_path}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"forward shadow output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    raw_rows = _load_forward_shadow_rows(config.input_path)
    if config.max_rows is not None:
        raw_rows = raw_rows[: config.max_rows]
    forbidden = _forbidden_fields_by_row(raw_rows)
    artifact_validation = validate_frozen_regime_conditioned_ev_artifact(
        config.frozen_regime_conditioned_ev_artifact
    )
    decision_rows = (
        []
        if forbidden
        else _build_regime_conditioned_shadow_rows(
            raw_rows,
            artifact_validation=artifact_validation,
            entry_ev_threshold=config.entry_ev_threshold,
            default_execution_cost=config.default_execution_cost,
        )
    )
    report = _build_regime_conditioned_forward_shadow_report(
        decision_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
        artifact_validation=artifact_validation,
        entry_ev_threshold=config.entry_ev_threshold,
    )
    artifact_report = _build_artifact_validation_report(
        artifact_validation,
        run_id=config.run_id,
    )
    artifact_paths = {
        "execution_layer_v2_frozen_regime_conditioned_ev_validation_report": (
            run_dir
            / "execution_layer_v2_frozen_regime_conditioned_ev_validation_report.json"
        ),
        "execution_layer_v2_frozen_regime_conditioned_ev_validation_summary": (
            run_dir
            / "execution_layer_v2_frozen_regime_conditioned_ev_validation_report.md"
        ),
        "execution_layer_v2_regime_conditioned_ev_forward_shadow_report": (
            run_dir / "execution_layer_v2_regime_conditioned_ev_forward_shadow_report.json"
        ),
        "execution_layer_v2_regime_conditioned_ev_forward_shadow_summary": (
            run_dir / "execution_layer_v2_regime_conditioned_ev_forward_shadow_report.md"
        ),
        "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest": (
            run_dir / "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest.json"
        ),
    }
    _write_json(
        artifact_paths[
            "execution_layer_v2_frozen_regime_conditioned_ev_validation_report"
        ],
        artifact_report,
    )
    _write_text(
        artifact_paths[
            "execution_layer_v2_frozen_regime_conditioned_ev_validation_summary"
        ],
        _artifact_validation_report_to_markdown(artifact_report),
    )
    _write_json(
        artifact_paths["execution_layer_v2_regime_conditioned_ev_forward_shadow_report"],
        report,
    )
    _write_text(
        artifact_paths[
            "execution_layer_v2_regime_conditioned_ev_forward_shadow_summary"
        ],
        execution_layer_v2_regime_conditioned_ev_forward_shadow_report_to_markdown(
            report
        ),
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in artifact_paths.items()
        if name
        != "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest"
    }
    manifest = {
        "schema_version": (
            REGIME_CONDITIONED_EV_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION
        ),
        "run_id": config.run_id,
        "input_path": str(config.input_path),
        "frozen_regime_conditioned_ev_artifact": (
            str(config.frozen_regime_conditioned_ev_artifact)
            if config.frozen_regime_conditioned_ev_artifact is not None
            else None
        ),
        "frozen_regime_conditioned_ev_artifact_hash": artifact_validation["sha256"],
        "frozen_regime_conditioned_ev_contract_hash": canonical_json_sha256(
            artifact_validation["contract"]
        ),
        "frozen_regime_conditioned_ev_artifact_valid": artifact_validation["valid"],
        "artifact_validation_report_id": artifact_report["report_id"],
        "forward_shadow_report_id": report["report_id"],
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "raw_row_count": len(raw_rows),
        "accepted_signal_row_count": len(decision_rows),
        "regime_conditioned_ev_produced_count": report[
            "regime_conditioned_ev_produced_count"
        ],
        "regime_conditioned_ev_missing_count": report[
            "regime_conditioned_ev_missing_count"
        ],
        "candidate_count": report["candidate_count"],
        "full_guard_passed_count": report["full_guard_passed_count"],
        "executable_shadow_count": report["executable_shadow_count"],
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "future_v2_probability_value_contract_recommendation": (
            _future_v2_probability_value_contract_recommendation()
        ),
        "legacy_ambiguous_probability_flag_present": artifact_report[
            "legacy_ambiguous_probability_flag_present"
        ],
        "diagnostic_only": True,
        "production_gate_implemented": False,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(
        artifact_paths[
            "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest"
        ],
        manifest,
    )
    artifact_hashes[
        "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest"
    ] = _sha256_file(
        artifact_paths[
            "execution_layer_v2_regime_conditioned_ev_forward_shadow_manifest"
        ]
    )
    return ExecutionLayerV2RegimeConditionedEVForwardShadowResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_validation_report=artifact_report,
        forward_shadow_report=report,
        manifest=manifest,
    )


def execution_layer_v2_regime_conditioned_ev_forward_shadow_report_to_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render the outcome-free shadow summary."""

    lines = [
        "# Execution Layer v2 Regime-Conditioned EV Forward Shadow",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- status: `{report['forward_shadow_status']}`",
        f"- artifact_valid: `{report['artifact_valid']}`",
        f"- raw_row_count: `{report['raw_row_count']}`",
        f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
        "- regime_conditioned_ev_produced_count: "
        f"`{report['regime_conditioned_ev_produced_count']}`",
        "- regime_conditioned_ev_missing_count: "
        f"`{report['regime_conditioned_ev_missing_count']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- full_guard_passed_count: `{report['full_guard_passed_count']}`",
        f"- executable_shadow_count: `{report['executable_shadow_count']}`",
        "- market_implied_probability_used_as_direct_fair_value_ev: `false`",
        "- market_implied_probability_used_as_conditioning_feature: `true`",
        "- market_implied_probability_used_as_regime_direction_vote: `false`",
        "- outcome_or_pnl_fields_used: `false`",
        "- production_gate_implemented: `false`",
        "- v8_execution_handoff_allowed: `false`",
        "",
        "## Rejections",
        "",
    ]
    rejection_counts = report["rejection_reason_distribution"]
    lines.extend(
        [f"- `{reason}`: `{count}`" for reason, count in rejection_counts.items()]
        or ["- none"]
    )
    lines.extend(["", "## Feature Coverage", ""])
    for feature, metrics in report["feature_coverage"].items():
        lines.append(
            f"- `{feature}`: `{metrics['available_count']}/{metrics['row_count']}`"
        )
    recommendation = report["future_v2_probability_value_contract_recommendation"]
    lines.extend(
        [
            "",
            "## Future v2 Probability/Value Contract",
            "",
            f"- status: `{recommendation['status']}`",
            "- fields: "
            + ", ".join(f"`{field}`" for field in recommendation["fields"]),
            "- real_coefficients_created: `false`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_regime_conditioned_shadow_rows(
    raw_rows: list[dict[str, Any]],
    *,
    artifact_validation: Mapping[str, Any],
    entry_ev_threshold: float,
    default_execution_cost: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    market_state: dict[str, dict[str, Any]] = {}
    missing_legacy_ev = {
        "path": None,
        "sha256": None,
        "payload": {},
        "valid": False,
        "source": "regime_conditioned_ev_consumer_does_not_use_legacy_ev",
        "blocking_reason_codes": ["legacy_ev_mapping_disabled_for_consumer"],
    }
    for index, raw in enumerate(raw_rows):
        normalized = _normalize_forward_shadow_row(
            raw,
            index,
            default_execution_cost=default_execution_cost,
            ev_calibration_artifact=missing_legacy_ev,
        )
        market_id = normalized["market_id"]
        state = market_state.setdefault(
            market_id,
            {
                "entry_count": 0,
                "cumulative_exposure": 0.0,
                "exposure_notional_complete": True,
                "last_side": None,
            },
        )
        features, sources, provenance = _extract_decision_time_features(
            raw,
            normalized,
            state,
        )
        value, components, ev_reasons = _score_regime_conditioned_ev(
            features,
            artifact_validation=artifact_validation,
            execution_cost=normalized["execution_cost"],
            side=normalized["side"],
            family=normalized["family"],
            provenance=provenance,
        )
        guard_status, guard_reasons = _execution_guard_status(normalized)
        candidate_reasons: list[str] = []
        if normalized["action"] == "NO_TRADE":
            candidate_reasons.append("selected_action_no_trade")
        if value is None:
            candidate_reasons.extend(ev_reasons or ["regime_conditioned_ev_missing"])
        elif value < entry_ev_threshold:
            candidate_reasons.append("regime_conditioned_ev_below_threshold")
        candidate = not candidate_reasons
        full_guard_passed = candidate and guard_status == "guard_passed"
        rejection_reasons = list(candidate_reasons)
        if candidate and not full_guard_passed:
            rejection_reasons.extend(guard_reasons)
        regime = _regime_bucket(components)
        exposure_state = _exposure_state_bucket(features)
        row = {
            "row_index": index,
            "market_id": market_id,
            "decision_ts": normalized["decision_ts"],
            "selected_action": normalized["action"],
            "selected_side": normalized["side"],
            "action_family": normalized["family"],
            "regime": regime,
            "exposure_state": exposure_state,
            "decision_time_features": features,
            "decision_time_feature_sources": sources,
            "decision_time_feature_provenance": provenance,
            "regime_conditioned_ev": value,
            "regime_conditioned_ev_available": value is not None,
            "regime_conditioned_ev_source": (
                str(artifact_validation["contract"]["artifact_name"])
                if artifact_validation["valid"]
                else artifact_validation["status"]
            ),
            "regime_conditioned_ev_artifact_hash": artifact_validation["sha256"],
            "regime_conditioned_ev_components": components,
            "regime_conditioned_ev_blocking_reason_codes": ev_reasons,
            "entry_ev_threshold": entry_ev_threshold,
            "candidate": candidate,
            "full_guard_status": guard_status,
            "full_guard_reason_codes": guard_reasons,
            "full_guard_passed": full_guard_passed,
            "executable_shadow": full_guard_passed,
            "rejection_reason_codes": sorted(set(rejection_reasons)),
            "market_implied_probability_used_as_regime_direction_vote": False,
            "market_implied_probability_used_as_direct_fair_value_ev": False,
            "market_implied_probability_used_as_conditioning_feature": True,
            "correlated_momentum_reference_counted_as_independent_votes": False,
            "outcome_fields_used": False,
            "source_scores_mutated": False,
            "o_score_mutated": False,
        }
        rows.append(row)

        if guard_status == "guard_passed" and normalized["action"] != "NO_TRADE":
            state["entry_count"] += 1
            state["last_side"] = normalized["side"]
            notional = _first_number(
                raw,
                (
                    "paper_notional",
                    "order_notional",
                    "requested_notional",
                    "execution_notional",
                ),
            )
            if notional is not None and notional >= 0:
                state["cumulative_exposure"] += notional
            else:
                state["exposure_notional_complete"] = False
    return rows


def _extract_decision_time_features(
    raw: Mapping[str, Any],
    normalized: Mapping[str, Any],
    market_state: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, str | None], dict[str, Any]]:
    p_up, p_up_source = _first_number_with_field(
        raw,
        ("p_up", "p_market_implied_up", "market_implied_probability_up"),
    )
    p_down, p_down_source = _first_number_with_field(
        raw,
        ("p_down", "p_market_implied_down", "market_implied_probability_down"),
    )
    if p_down is None and p_up is not None:
        p_down = 1.0 - p_up
        p_down_source = f"derived_one_minus:{p_up_source}"
    if p_up is None and p_down is not None:
        p_up = 1.0 - p_down
        p_up_source = f"derived_one_minus:{p_down_source}"
    selected_side = normalized["side"]
    selected_side_probability = p_down if selected_side == "DOWN" else p_up
    selected_side_probability_source = (
        p_down_source if selected_side == "DOWN" else p_up_source
    )
    execution_price = _optional_float(normalized.get("execution_price"))
    probability_minus_price = (
        selected_side_probability - execution_price
        if selected_side_probability is not None and execution_price is not None
        else None
    )

    action_margin, action_margin_source = _first_number_with_field(
        raw,
        (
            "action_score_margin",
            "best_action_margin",
            "side_specific_action_score_margin",
        ),
    )
    momentum, momentum_source = _first_number_with_field(
        raw,
        ("btc_momentum", "recent_reference_price_momentum_60s"),
    )
    reference_distance, reference_source = _first_number_with_field(
        raw,
        ("reference_price_to_beat_distance_at_decision",),
    )
    explicit_entry_index, explicit_entry_source = _first_number_with_field(
        raw,
        ("entry_index_within_market",),
    )
    explicit_exposure, explicit_exposure_source = _first_number_with_field(
        raw,
        (
            "cumulative_market_exposure_before_entry",
            "runtime_market_exposure_before_entry",
            "market_exposure_before_entry",
        ),
    )
    explicit_same_side, same_side_source = _first_bool_with_field(
        raw,
        ("same_side_reentry",),
    )
    explicit_side_flip, side_flip_source = _first_bool_with_field(
        raw,
        ("side_flip",),
    )
    derived_entry_index = int(market_state["entry_count"]) + 1
    prior_side = market_state.get("last_side")
    same_side = (
        explicit_same_side
        if explicit_same_side is not None
        else bool(prior_side and prior_side == selected_side)
    )
    side_flip = (
        explicit_side_flip
        if explicit_side_flip is not None
        else bool(prior_side and prior_side != selected_side)
    )
    derived_exposure = None
    if int(market_state["entry_count"]) == 0:
        derived_exposure = 0.0
    elif bool(market_state["exposure_notional_complete"]):
        derived_exposure = float(market_state["cumulative_exposure"])
    features: dict[str, float | None] = {
        "canonical_o_action_score": _optional_float(
            normalized.get("canonical_o_action_score")
        ),
        "action_score_margin": action_margin,
        "btc_momentum": momentum,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "p_up": p_up,
        "p_down": p_down,
        "selected_side_probability": selected_side_probability,
        "execution_price": execution_price,
        "selected_side_probability_minus_execution_price": probability_minus_price,
        "spread_bps": _optional_float(normalized.get("spread_bps")),
        "book_staleness_ms": _optional_float(normalized.get("book_staleness_ms")),
        "queue_fill_proxy": _optional_float(normalized.get("queue_fill_proxy")),
        "time_to_close_seconds": _optional_float(
            normalized.get("time_to_close_seconds")
        ),
        "entry_index_within_market": (
            explicit_entry_index
            if explicit_entry_index is not None
            else float(derived_entry_index)
        ),
        "cumulative_market_exposure_before_entry": (
            explicit_exposure
            if explicit_exposure is not None
            else derived_exposure
        ),
        "same_side_reentry": 1.0 if same_side else 0.0,
        "side_flip": 1.0 if side_flip else 0.0,
    }
    sources = {
        "canonical_o_action_score": normalized.get(
            "canonical_o_action_score_source_field"
        ),
        "action_score_margin": action_margin_source,
        "btc_momentum": momentum_source,
        "reference_price_to_beat_distance_at_decision": reference_source,
        "p_up": p_up_source,
        "p_down": p_down_source,
        "selected_side_probability": selected_side_probability_source,
        "execution_price": normalized.get("execution_price_source_field"),
        "selected_side_probability_minus_execution_price": (
            "derived_selected_side_probability_minus_execution_price"
        ),
        "spread_bps": "spread_bps",
        "book_staleness_ms": "book_staleness_ms",
        "queue_fill_proxy": "queue_fill_proxy",
        "time_to_close_seconds": "time_to_close_seconds",
        "entry_index_within_market": (
            explicit_entry_source or "derived_prior_guard_passed_market_entries"
        ),
        "cumulative_market_exposure_before_entry": (
            explicit_exposure_source
            or (
                "derived_prior_guard_passed_paper_notional"
                if derived_exposure is not None
                else "missing_prior_guard_passed_paper_notional"
            )
        ),
        "same_side_reentry": same_side_source or "derived_prior_guard_passed_side",
        "side_flip": side_flip_source or "derived_prior_guard_passed_side",
    }
    decision_ts = _timestamp_number(normalized.get("decision_ts"))
    feature_provenance: dict[str, Any] = {}
    violation_count = 0
    for feature, value in features.items():
        max_input_ts, provenance_source = _feature_max_input_ts(
            raw,
            feature,
            decision_ts,
        )
        valid = (
            value is not None
            and decision_ts is not None
            and max_input_ts is not None
            and max_input_ts <= decision_ts
        )
        if value is not None and not valid:
            violation_count += 1
        feature_provenance[feature] = {
            "source_field": sources.get(feature),
            "max_input_ts": max_input_ts,
            "decision_ts": decision_ts,
            "provenance_source": provenance_source,
            "valid": valid,
        }
    return features, sources, {
        "features": feature_provenance,
        "violation_count": violation_count,
        "all_available_features_valid": violation_count == 0,
    }


def _score_regime_conditioned_ev(
    features: Mapping[str, float | None],
    *,
    artifact_validation: Mapping[str, Any],
    execution_cost: float | None,
    side: str,
    family: str,
    provenance: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any], list[str]]:
    if not artifact_validation["valid"]:
        return (
            None,
            {},
            list(artifact_validation["blocking_reason_codes"]),
        )
    feature_groups = {
        str(name): tuple(str(feature) for feature in group_features)
        for name, group_features in artifact_validation["contract"][
            "required_feature_groups"
        ].items()
    }
    required_features = tuple(
        feature
        for group_features in feature_groups.values()
        for feature in group_features
    )
    missing = sorted(
        feature
        for feature in required_features
        if features.get(feature) is None
    )
    invalid_provenance = sorted(
        feature
        for feature in required_features
        if features.get(feature) is not None
        and not provenance["features"][feature]["valid"]
    )
    reasons = [f"missing_decision_time_feature:{feature}" for feature in missing]
    reasons.extend(
        f"decision_time_feature_provenance_invalid:{feature}"
        for feature in invalid_provenance
    )
    if reasons:
        return None, {}, reasons

    payload = artifact_validation["payload"]
    coefficients = payload["coefficients"]
    total = float(coefficients["intercept"])
    group_components: dict[str, Any] = {}
    for group_name, expected_features in feature_groups.items():
        group_config = coefficients["groups"][group_name]
        feature_values: dict[str, float] = {}
        raw_group_score = 0.0
        side_sign = -1.0 if side == "DOWN" else 1.0
        for feature in expected_features:
            raw_value = float(features[feature])
            if group_name == "btc_anchor_direction":
                raw_value *= side_sign
            transform = group_config["feature_transforms"][feature]
            normalized = (raw_value - float(transform["center"])) / float(
                transform["scale"]
            )
            normalized = max(
                float(transform["clip_min"]),
                min(float(transform["clip_max"]), normalized),
            )
            feature_values[feature] = normalized
            raw_group_score += (
                float(group_config["feature_weights"][feature]) * normalized
            )
        raw_group_score = max(-1.0, min(1.0, raw_group_score))
        contribution = float(group_config["group_coefficient"]) * raw_group_score
        max_contribution = float(group_config["maximum_absolute_contribution"])
        contribution = max(-max_contribution, min(max_contribution, contribution))
        total += contribution
        group_components[group_name] = {
            "normalized_feature_values": feature_values,
            "raw_group_score": raw_group_score,
            "group_coefficient": float(group_config["group_coefficient"]),
            "contribution": contribution,
        }
    total += _mapping_offset(coefficients.get("side_offsets"), side)
    total += _mapping_offset(coefficients.get("family_offsets"), family)
    if bool(coefficients.get("subtract_execution_cost", True)):
        total -= 0.0 if execution_cost is None else float(execution_cost)
    return (
        total,
        {
            "intercept": float(coefficients["intercept"]),
            "groups": group_components,
            "side_offset": _mapping_offset(coefficients.get("side_offsets"), side),
            "family_offset": _mapping_offset(
                coefficients.get("family_offsets"), family
            ),
            "execution_cost_subtracted": bool(
                coefficients.get("subtract_execution_cost", True)
            ),
            "execution_cost": execution_cost,
            "formula": (
                "intercept_plus_independent_group_contributions_minus_cost"
                if bool(coefficients.get("subtract_execution_cost", True))
                else "intercept_plus_independent_group_contributions_net_return_target"
            ),
        },
        [],
    )


def _build_regime_conditioned_forward_shadow_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int,
    forbidden_outcome_fields_by_row: list[dict[str, Any]],
    artifact_validation: Mapping[str, Any],
    entry_ev_threshold: float,
) -> dict[str, Any]:
    produced = [row for row in rows if row["regime_conditioned_ev_available"]]
    candidates = [row for row in rows if row["candidate"]]
    guard_passed = [row for row in candidates if row["full_guard_passed"]]
    rejection_reasons: Counter[str] = Counter()
    for row in rows:
        rejection_reasons.update(row["rejection_reason_codes"])
    if forbidden_outcome_fields_by_row:
        rejection_reasons["forbidden_outcome_fields_present"] += len(
            forbidden_outcome_fields_by_row
        )
    status = "diagnostic_only_fail_closed"
    blocking_reasons: list[str] = []
    if forbidden_outcome_fields_by_row:
        status = "blocked_fail_closed"
        blocking_reasons.append("forbidden_outcome_fields_present")
    if not artifact_validation["valid"]:
        status = "blocked_fail_closed"
        blocking_reasons.extend(artifact_validation["blocking_reason_codes"])
    if not rows and not forbidden_outcome_fields_by_row:
        status = "blocked_fail_closed"
        blocking_reasons.append("no_forward_shadow_rows")
    required_features = tuple(
        feature
        for group_features in artifact_validation["contract"][
            "required_feature_groups"
        ].values()
        for feature in group_features
    )
    report = {
        "schema_version": REGIME_CONDITIONED_EV_FORWARD_SHADOW_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count,
        "accepted_signal_row_count": len(rows),
        "forward_shadow_status": status,
        "forward_shadow_blocking_reason_codes": sorted(set(blocking_reasons)),
        "artifact_valid": artifact_validation["valid"],
        "artifact_status": artifact_validation["status"],
        "artifact_hash": artifact_validation["sha256"],
        "entry_ev_threshold": entry_ev_threshold,
        "regime_conditioned_ev_produced_count": len(produced),
        "regime_conditioned_ev_missing_count": len(rows) - len(produced),
        "candidate_count": len(candidates),
        "full_guard_passed_count": len(guard_passed),
        "executable_shadow_count": len(guard_passed),
        "counts_by_stage": {
            "produced": _multi_dimension_counts(produced),
            "candidate": _multi_dimension_counts(candidates),
            "full_guard_passed": _multi_dimension_counts(guard_passed),
            "executable_shadow": _multi_dimension_counts(guard_passed),
        },
        "rejection_reason_distribution": dict(sorted(rejection_reasons.items())),
        "feature_coverage": _feature_coverage(rows, required_features),
        "provenance_coverage": _provenance_coverage(rows, required_features),
        "decision_rows": rows,
        "independent_signal_group_contract": artifact_validation["contract"],
        "p_up_p_down_used_only_in_market_price_value_group": True,
        "correlated_momentum_reference_counted_as_independent_votes": False,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "future_v2_probability_value_contract_recommendation": (
            _future_v2_probability_value_contract_recommendation()
        ),
        "forbidden_outcome_fields_present": bool(forbidden_outcome_fields_by_row),
        "forbidden_outcome_fields_by_row": forbidden_outcome_fields_by_row,
        "diagnostic_only": True,
        "paper_only": True,
        "production_gate_implemented": False,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _build_artifact_validation_report(
    artifact_validation: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    report = {
        "schema_version": artifact_validation["contract"]["schema_version"],
        "run_id": run_id,
        "artifact_path": artifact_validation["path"],
        "artifact_hash": artifact_validation["sha256"],
        "artifact_valid": artifact_validation["valid"],
        "artifact_status": artifact_validation["status"],
        "artifact_blocking_reason_codes": artifact_validation[
            "blocking_reason_codes"
        ],
        "forbidden_field_paths": artifact_validation["forbidden_field_paths"],
        "contract": artifact_validation["contract"],
        "contract_hash": canonical_json_sha256(artifact_validation["contract"]),
        "market_implied_probability_used_as_direct_fair_value_ev": (
            _artifact_probability_semantic(
                artifact_validation.get("payload"),
                "market_implied_probability_used_as_direct_fair_value_ev",
            )
        ),
        "market_implied_probability_used_as_conditioning_feature": (
            _artifact_probability_semantic(
                artifact_validation.get("payload"),
                "market_implied_probability_used_as_conditioning_feature",
            )
        ),
        "market_implied_probability_used_as_regime_direction_vote": (
            _artifact_probability_semantic(
                artifact_validation.get("payload"),
                "market_implied_probability_used_as_regime_direction_vote",
            )
        ),
        "legacy_ambiguous_probability_flag_present": bool(
            isinstance(artifact_validation.get("payload"), Mapping)
            and "market_implied_probability_used_for_ev"
            in artifact_validation["payload"]
        ),
        "future_v2_probability_value_contract_recommendation": (
            _future_v2_probability_value_contract_recommendation()
        ),
        "current_75_row_replay_used_for_fitting": (
            _current_replay_used_for_fitting(artifact_validation.get("payload"))
        ),
        "latest_one_hour_reconciled_run_used_for_fitting": _fit_run_used(
            artifact_validation.get("payload"), LATEST_ONE_HOUR_RECONCILED_RUN_ID
        ),
        "future_unseen_forward_shadow_run_used_for_fitting": (
            _future_shadow_run_used_for_fitting(
                artifact_validation.get("payload")
            )
        ),
        "diagnostic_only": True,
        "production_gate_implemented": False,
        **_safety_report_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _artifact_validation_report_to_markdown(report: Mapping[str, Any]) -> str:
    reasons = report["artifact_blocking_reason_codes"]
    lines = [
        "# Frozen Regime-Conditioned EV Artifact Validation",
        "",
        f"- artifact_valid: `{report['artifact_valid']}`",
        f"- artifact_status: `{report['artifact_status']}`",
        f"- artifact_hash: `{report['artifact_hash']}`",
        "- market_implied_probability_used_as_direct_fair_value_ev: "
        f"`{report['market_implied_probability_used_as_direct_fair_value_ev']}`",
        "- market_implied_probability_used_as_conditioning_feature: "
        f"`{report['market_implied_probability_used_as_conditioning_feature']}`",
        "- market_implied_probability_used_as_regime_direction_vote: "
        f"`{report['market_implied_probability_used_as_regime_direction_vote']}`",
        "- legacy_ambiguous_probability_flag_present: "
        f"`{str(report['legacy_ambiguous_probability_flag_present']).lower()}`",
        "- current_75_row_replay_used_for_fitting: "
        f"`{str(report['current_75_row_replay_used_for_fitting']).lower()}`",
        "- latest_one_hour_reconciled_run_used_for_fitting: "
        f"`{str(report['latest_one_hour_reconciled_run_used_for_fitting']).lower()}`",
        "- future_unseen_forward_shadow_run_used_for_fitting: "
        f"`{str(report['future_unseen_forward_shadow_run_used_for_fitting']).lower()}`",
        "- production_gate_implemented: `false`",
        "- v8_execution_handoff_allowed: `false`",
        "",
        "## Blocking Reasons",
        "",
    ]
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- none"])
    return "\n".join(lines) + "\n"


def _validate_fit_provenance(
    payload: Any, reasons: list[str], *, is_v2: bool
) -> None:
    if not isinstance(payload, Mapping):
        reasons.append("regime_conditioned_ev_fit_provenance_missing")
        return
    if is_v2:
        _validate_v2_fit_provenance(payload, reasons)
        return
    expected_fields = {
        "coefficients_source",
        "coefficients_fitted_from_current_75_row_replay",
        "uses_settlement_pnl_for_fitting",
        "uses_outcomes_for_fitting",
        "uses_oracle_actions_for_fitting",
        "uses_future_returns_for_fitting",
        "fitted_from_run_ids",
        "excluded_run_ids",
        "fit_dataset_hash",
        "fit_config_hash",
    }
    if set(payload) != expected_fields:
        reasons.append("regime_conditioned_ev_fit_provenance_fields_mismatch")
    if payload.get("coefficients_source") not in {
        "separate_calibration_training_split",
        "shadow_split_only",
    }:
        reasons.append("regime_conditioned_ev_coefficients_source_not_separate_split")
    for field_name in (
        "coefficients_fitted_from_current_75_row_replay",
        "uses_settlement_pnl_for_fitting",
        "uses_outcomes_for_fitting",
        "uses_oracle_actions_for_fitting",
        "uses_future_returns_for_fitting",
    ):
        if payload.get(field_name) is not False:
            reasons.append(f"regime_conditioned_ev_fit_provenance_{field_name}_not_false")
    fitted_run_ids = payload.get("fitted_from_run_ids")
    if not isinstance(fitted_run_ids, list):
        reasons.append("regime_conditioned_ev_fitted_from_run_ids_missing")
    elif CURRENT_75_ROW_REPLAY_RUN_ID in {str(value) for value in fitted_run_ids}:
        reasons.append("current_75_row_replay_present_in_fit_lineage")
    excluded_run_ids = payload.get("excluded_run_ids")
    if not isinstance(excluded_run_ids, list):
        reasons.append("regime_conditioned_ev_excluded_run_ids_missing")
    elif CURRENT_75_ROW_REPLAY_RUN_ID not in {
        str(value) for value in excluded_run_ids
    }:
        reasons.append("current_75_row_replay_not_explicitly_excluded_from_fit")
    for field_name in ("fit_dataset_hash", "fit_config_hash"):
        if not _is_sha256(payload.get(field_name)):
            reasons.append(f"regime_conditioned_ev_{field_name}_invalid")
def _validate_v2_fit_provenance(payload: Mapping[str, Any], reasons: list[str]) -> None:
    expected_fields = {
        "coefficients_source",
        "settled_outcomes_or_pnl_used_as_training_targets",
        "settled_outcomes_or_pnl_used_as_decision_time_inputs",
        "uses_validation_labels_for_fitting",
        "uses_validation_labels_for_threshold_selection",
        "uses_holdout_labels_for_fitting",
        "uses_holdout_labels_for_threshold_selection",
        "uses_oracle_actions_for_fitting",
        "uses_future_returns_for_fitting",
        "future_unseen_run_pattern_excluded",
        "fitted_from_run_ids",
        "excluded_run_ids",
        "fit_dataset_hash",
        "validation_dataset_hash",
        "split_hash",
        "calibration_config_hash",
        "fit_coefficients_hash",
        "calibration_row_schema_sha256",
        "statistical_eligibility_config_hash",
        "statistical_eligibility_summary_hash",
        "statistical_eligibility_passed",
    }
    if set(payload) != expected_fields:
        reasons.append("regime_conditioned_ev_v2_fit_provenance_fields_mismatch")
    if payload.get("coefficients_source") != "historical_fit_split_only":
        reasons.append("regime_conditioned_ev_v2_coefficients_source_invalid")
    if payload.get("settled_outcomes_or_pnl_used_as_training_targets") is not True:
        reasons.append("regime_conditioned_ev_v2_historical_target_usage_not_true")
    false_fields = (
        "settled_outcomes_or_pnl_used_as_decision_time_inputs",
        "uses_validation_labels_for_fitting",
        "uses_validation_labels_for_threshold_selection",
        "uses_holdout_labels_for_fitting",
        "uses_holdout_labels_for_threshold_selection",
        "uses_oracle_actions_for_fitting",
        "uses_future_returns_for_fitting",
    )
    for field_name in false_fields:
        if payload.get(field_name) is not False:
            reasons.append(f"regime_conditioned_ev_v2_{field_name}_not_false")
    if payload.get("future_unseen_run_pattern_excluded") is not True:
        reasons.append("future_unseen_forward_shadow_runs_not_excluded_from_fit")
    if payload.get("statistical_eligibility_passed") is not True:
        reasons.append("regime_conditioned_ev_v2_statistical_eligibility_not_passed")
    fitted_run_ids = payload.get("fitted_from_run_ids")
    fitted = (
        {str(value) for value in fitted_run_ids}
        if isinstance(fitted_run_ids, list)
        else set()
    )
    if not fitted:
        reasons.append("regime_conditioned_ev_fitted_from_run_ids_missing")
    excluded_run_ids = payload.get("excluded_run_ids")
    excluded = (
        {str(value) for value in excluded_run_ids}
        if isinstance(excluded_run_ids, list)
        else set()
    )
    for run_id in (CURRENT_75_ROW_REPLAY_RUN_ID, LATEST_ONE_HOUR_RECONCILED_RUN_ID):
        if run_id in fitted:
            reasons.append(f"excluded_run_present_in_fit_lineage:{run_id}")
        if run_id not in excluded:
            reasons.append(f"required_excluded_run_missing:{run_id}")
    if any("forward-shadow" in run_id or "forward_shadow" in run_id for run_id in fitted):
        reasons.append("future_unseen_forward_shadow_run_present_in_fit_lineage")
    for field_name in (
        "fit_dataset_hash",
        "validation_dataset_hash",
        "split_hash",
        "calibration_config_hash",
        "fit_coefficients_hash",
        "calibration_row_schema_sha256",
        "statistical_eligibility_config_hash",
        "statistical_eligibility_summary_hash",
    ):
        if not _is_sha256(payload.get(field_name)):
            reasons.append(f"regime_conditioned_ev_{field_name}_invalid")
    if not V2_CALIBRATION_ROW_SCHEMA_PATH.is_file():
        reasons.append("regime_conditioned_ev_v2_calibration_row_schema_missing")
    elif payload.get("calibration_row_schema_sha256") != _sha256_file(
        V2_CALIBRATION_ROW_SCHEMA_PATH
    ):
        reasons.append("regime_conditioned_ev_v2_calibration_row_schema_hash_mismatch")


def _validate_v2_calibration_protocol(payload: Any, reasons: list[str]) -> None:
    expected = {
        "split_order": [
            "historical_fit",
            "validation",
            "future_unseen_shadow_holdout",
        ],
        "chronological_split_required": True,
        "market_id_disjointness_required_where_possible": True,
        "feature_max_input_ts_must_not_exceed_decision_ts": True,
        "historical_fit_target": "settled_net_return_after_cost",
        "validation_labels_used_for_evaluation_only": True,
        "future_shadow_outcome_free_at_inference": True,
        "threshold_selection_source": "fixed_pre_validation_config",
        "refit_from_future_shadow_result_allowed": False,
        "statistical_eligibility_required": True,
        "market_level_evaluation_required": True,
        "market_bootstrap_confidence_required": True,
        "coefficient_stability_required": True,
        "validation_coverage_required": True,
    }
    if not isinstance(payload, Mapping):
        reasons.append("regime_conditioned_ev_v2_calibration_protocol_missing")
        return
    if set(payload) != set(expected):
        reasons.append("regime_conditioned_ev_v2_calibration_protocol_fields_mismatch")
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            reasons.append(
                f"regime_conditioned_ev_v2_calibration_protocol_invalid:{field_name}"
            )


def _validate_independence_constraints(
    payload: Any,
    reasons: list[str],
    *,
    contract: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        reasons.append("regime_conditioned_ev_independence_constraints_missing")
        return
    expected = contract["independence_constraints"]
    if set(payload) != set(expected):
        reasons.append("regime_conditioned_ev_independence_constraint_fields_mismatch")
    for key, value in expected.items():
        if payload.get(key) != value:
            reasons.append(f"regime_conditioned_ev_independence_constraint_invalid:{key}")


def _validate_feature_groups(
    payload: Any,
    reasons: list[str],
    *,
    feature_groups: Mapping[str, tuple[str, ...]],
) -> None:
    if not isinstance(payload, Mapping):
        reasons.append("regime_conditioned_ev_feature_groups_missing")
        return
    if set(payload) != set(feature_groups):
        reasons.append("regime_conditioned_ev_feature_group_names_mismatch")
    memberships: dict[str, list[str]] = {}
    for group_name, expected_features in feature_groups.items():
        group = payload.get(group_name)
        if not isinstance(group, Mapping):
            reasons.append(f"regime_conditioned_ev_feature_group_missing:{group_name}")
            continue
        if set(group) != {"features"}:
            reasons.append(
                f"regime_conditioned_ev_feature_group_schema_mismatch:{group_name}"
            )
        features = group.get("features")
        if not isinstance(features, list) or tuple(features) != expected_features:
            reasons.append(f"regime_conditioned_ev_feature_group_fields_mismatch:{group_name}")
            continue
        for feature in features:
            memberships.setdefault(str(feature), []).append(group_name)
    for feature, groups in memberships.items():
        if len(groups) != 1:
            reasons.append(f"regime_conditioned_ev_feature_double_counted:{feature}")
    probability_features = (
        ("selected_side_probability",)
        if "selected_side_probability" in memberships
        else ("p_up", "p_down")
    )
    for feature in probability_features:
        if memberships.get(feature) != ["market_price_value"]:
            reasons.append(f"regime_conditioned_ev_probability_group_invalid:{feature}")
    for feature in (
        "btc_momentum",
        "reference_price_to_beat_distance_at_decision",
    ):
        if memberships.get(feature) != ["btc_anchor_direction"]:
            reasons.append(f"regime_conditioned_ev_btc_anchor_group_invalid:{feature}")


def _validate_coefficient_contract(
    payload: Any,
    reasons: list[str],
    *,
    feature_groups: Mapping[str, tuple[str, ...]],
    subtract_execution_cost: bool,
) -> None:
    if not isinstance(payload, Mapping):
        reasons.append("regime_conditioned_ev_coefficients_missing")
        return
    expected_coefficient_fields = {
        "intercept",
        "groups",
        "side_offsets",
        "family_offsets",
        "subtract_execution_cost",
    }
    if set(payload) != expected_coefficient_fields:
        reasons.append("regime_conditioned_ev_coefficient_fields_mismatch")
    if not _is_finite_number(payload.get("intercept")):
        reasons.append("regime_conditioned_ev_intercept_invalid")
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        reasons.append("regime_conditioned_ev_group_coefficients_missing")
        return
    if set(groups) != set(feature_groups):
        reasons.append("regime_conditioned_ev_group_coefficient_names_mismatch")
    for group_name, expected_features in feature_groups.items():
        group = groups.get(group_name)
        if not isinstance(group, Mapping):
            reasons.append(f"regime_conditioned_ev_group_coefficient_missing:{group_name}")
            continue
        expected_group_fields = {
            "group_coefficient",
            "maximum_absolute_contribution",
            "feature_weights",
            "feature_transforms",
        }
        if set(group) != expected_group_fields:
            reasons.append(
                f"regime_conditioned_ev_group_fields_mismatch:{group_name}"
            )
        coefficient = group.get("group_coefficient")
        maximum = group.get("maximum_absolute_contribution")
        if not _is_finite_number(coefficient):
            reasons.append(f"regime_conditioned_ev_group_coefficient_invalid:{group_name}")
        if not _is_finite_number(maximum) or float(maximum) < 0:
            reasons.append(f"regime_conditioned_ev_group_maximum_invalid:{group_name}")
        elif _is_finite_number(coefficient) and abs(float(coefficient)) > float(maximum):
            reasons.append(f"regime_conditioned_ev_group_coefficient_exceeds_max:{group_name}")
        weights = group.get("feature_weights")
        transforms = group.get("feature_transforms")
        if not isinstance(weights, Mapping) or set(weights) != set(expected_features):
            reasons.append(f"regime_conditioned_ev_feature_weights_mismatch:{group_name}")
        elif any(not _is_finite_number(value) for value in weights.values()):
            reasons.append(f"regime_conditioned_ev_feature_weight_invalid:{group_name}")
        elif sum(abs(float(value)) for value in weights.values()) > 1.0 + 1e-12:
            reasons.append(f"regime_conditioned_ev_feature_vote_weight_exceeds_one:{group_name}")
        if not isinstance(transforms, Mapping) or set(transforms) != set(expected_features):
            reasons.append(f"regime_conditioned_ev_feature_transforms_mismatch:{group_name}")
            continue
        for feature in expected_features:
            transform = transforms.get(feature)
            if not isinstance(transform, Mapping):
                reasons.append(f"regime_conditioned_ev_feature_transform_invalid:{feature}")
                continue
            required = ("center", "scale", "clip_min", "clip_max")
            if set(transform) != set(required):
                reasons.append(
                    f"regime_conditioned_ev_feature_transform_fields_mismatch:{feature}"
                )
            if any(not _is_finite_number(transform.get(name)) for name in required):
                reasons.append(f"regime_conditioned_ev_feature_transform_invalid:{feature}")
            elif float(transform["scale"]) <= 0:
                reasons.append(f"regime_conditioned_ev_feature_scale_not_positive:{feature}")
            elif float(transform["clip_min"]) > float(transform["clip_max"]):
                reasons.append(f"regime_conditioned_ev_feature_clip_invalid:{feature}")
    for offset_name in ("side_offsets", "family_offsets"):
        offsets = payload.get(offset_name, {})
        if not isinstance(offsets, Mapping) or any(
            not _is_finite_number(value) for value in offsets.values()
        ):
            reasons.append(f"regime_conditioned_ev_{offset_name}_invalid")
    if payload.get("subtract_execution_cost") is not subtract_execution_cost:
        reasons.append("regime_conditioned_ev_execution_cost_semantics_invalid")


def _invalid_artifact_result(
    *,
    path: str | None,
    status: str,
    reason_codes: list[str],
    contract: dict[str, Any],
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "payload": {},
        "valid": False,
        "status": status,
        "blocking_reason_codes": reason_codes,
        "forbidden_field_paths": [],
        "contract": contract,
    }


def _multi_dimension_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_side": _count(rows, "selected_side"),
        "by_action": _count(rows, "selected_action"),
        "by_regime": _count(rows, "regime"),
        "by_exposure_state": _count(rows, "exposure_state"),
    }


def _count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "UNKNOWN") for row in rows).items()))


def _feature_coverage(
    rows: list[dict[str, Any]], required_features: tuple[str, ...]
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for feature in required_features:
        available = [
            row
            for row in rows
            if row["decision_time_features"].get(feature) is not None
        ]
        source_counts = Counter(
            str(row["decision_time_feature_sources"].get(feature) or "missing")
            for row in available
        )
        coverage[feature] = {
            "row_count": len(rows),
            "available_count": len(available),
            "missing_count": len(rows) - len(available),
            "source_distribution": dict(sorted(source_counts.items())),
        }
    return coverage


def _provenance_coverage(
    rows: list[dict[str, Any]], required_features: tuple[str, ...]
) -> dict[str, Any]:
    total = len(rows) * len(required_features)
    valid = 0
    available = 0
    violation = 0
    by_feature: dict[str, Any] = {}
    for feature in required_features:
        feature_available = 0
        feature_valid = 0
        feature_violations = 0
        for row in rows:
            if row["decision_time_features"].get(feature) is None:
                continue
            feature_available += 1
            provenance = row["decision_time_feature_provenance"]["features"][feature]
            if provenance["valid"]:
                feature_valid += 1
            else:
                feature_violations += 1
        available += feature_available
        valid += feature_valid
        violation += feature_violations
        by_feature[feature] = {
            "available_count": feature_available,
            "valid_count": feature_valid,
            "violation_count": feature_violations,
        }
    return {
        "expected_feature_provenance_count": total,
        "available_feature_provenance_count": available,
        "valid_feature_provenance_count": valid,
        "violation_count": violation,
        "by_feature": by_feature,
    }


def _regime_bucket(components: Mapping[str, Any]) -> str:
    anchor = components.get("groups", {}).get("btc_anchor_direction", {})
    score = _optional_float(anchor.get("raw_group_score"))
    if score is None or abs(score) <= 1e-12:
        return "neutral_or_conflicted"
    return "selected_side_anchor_confirmed" if score > 0 else "selected_side_anchor_opposed"


def _exposure_state_bucket(features: Mapping[str, float | None]) -> str:
    if bool(features.get("side_flip")):
        return "side_flip"
    if bool(features.get("same_side_reentry")):
        return "same_side_reentry"
    if (features.get("entry_index_within_market") or 1.0) <= 1.0:
        return "first_entry"
    return "repeated_entry_unknown_side"


def _feature_max_input_ts(
    raw: Mapping[str, Any],
    feature: str,
    decision_ts: float | None,
) -> tuple[float | None, str]:
    direct = _lookup_value(raw, f"{feature}_provenance")
    if isinstance(direct, Mapping):
        for key in ("max_input_ts", "source_ts", "timestamp"):
            parsed = _timestamp_number(direct.get(key))
            if parsed is not None:
                return parsed, f"{feature}_provenance.{key}"
    shared = _lookup_value(raw, "decision_time_regime_feature_max_input_ts")
    parsed_shared = _timestamp_number(shared)
    if parsed_shared is not None:
        return parsed_shared, "decision_time_regime_feature_max_input_ts"
    return decision_ts, "inline_decision_row_field_at_decision_ts"


def _first_number_with_field(
    row: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[float | None, str | None]:
    for field in fields:
        value = _optional_float(_lookup_value(row, field))
        if value is not None:
            return value, field
    return None, None


def _first_number(row: Mapping[str, Any], fields: tuple[str, ...]) -> float | None:
    return _first_number_with_field(row, fields)[0]


def _first_bool_with_field(
    row: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[bool | None, str | None]:
    for field in fields:
        value = _lookup_value(row, field)
        if isinstance(value, bool):
            return value, field
    return None, None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _timestamp_number(value: Any) -> float | None:
    return _optional_float(value)


def _mapping_offset(payload: Any, key: str) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    return _optional_float(payload.get(key)) or 0.0


def _is_finite_number(value: Any) -> bool:
    return _optional_float(value) is not None


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == _HEX_DIGEST_LENGTH and all(
        char in "0123456789abcdef" for char in text
    )


def _require_equal(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
    reason: str,
    reasons: list[str],
) -> None:
    if payload.get(key) != expected:
        reasons.append(reason)


def _current_replay_used_for_fitting(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    fit = payload.get("fit_provenance")
    if not isinstance(fit, Mapping):
        return False
    fitted_run_ids = fit.get("fitted_from_run_ids")
    return bool(
        fit.get("coefficients_fitted_from_current_75_row_replay")
        or (
            isinstance(fitted_run_ids, list)
            and CURRENT_75_ROW_REPLAY_RUN_ID
            in {str(value) for value in fitted_run_ids}
        )
    )


def _fit_run_used(payload: Any, run_id: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    fit = payload.get("fit_provenance")
    if not isinstance(fit, Mapping):
        return False
    fitted_run_ids = fit.get("fitted_from_run_ids")
    return bool(
        isinstance(fitted_run_ids, list)
        and run_id in {str(value) for value in fitted_run_ids}
    )


def _future_shadow_run_used_for_fitting(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    fit = payload.get("fit_provenance")
    if not isinstance(fit, Mapping):
        return False
    fitted_run_ids = fit.get("fitted_from_run_ids")
    return bool(
        isinstance(fitted_run_ids, list)
        and any(
            "forward-shadow" in str(value).lower()
            or "forward_shadow" in str(value).lower()
            for value in fitted_run_ids
        )
    )


def _artifact_probability_semantic(payload: Any, field_name: str) -> bool | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field_name)
    return value if isinstance(value, bool) else None


def _future_v2_probability_value_contract_recommendation() -> dict[str, Any]:
    return {
        "contract_name": FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME,
        "status": "calibration_protocol_implemented_real_coefficients_pending",
        "fields": [
            "selected_side_probability",
            "execution_price",
            "selected_side_probability_minus_execution_price",
        ],
        "selected_side_probability_definition": (
            "p_up_for_up_action_and_p_down_for_down_action"
        ),
        "derived_value_formula": (
            "selected_side_probability - execution_price"
        ),
        "derived_value_semantics": (
            "market_relative_value_conditioner_not_calibrated_fair_value_ev"
        ),
        "calibrated_ev_formula_defined": True,
        "direct_fair_value_ev_fallback_allowed": False,
        "regime_direction_vote_allowed": False,
        "conditioning_feature_only": True,
        "real_coefficients_created": False,
        "future_schema_version_required": False,
    }


__all__ = [
    "CURRENT_75_ROW_REPLAY_RUN_ID",
    "LATEST_ONE_HOUR_RECONCILED_RUN_ID",
    "FROZEN_REGIME_CONDITIONED_EV_ARTIFACT_NAME",
    "FROZEN_REGIME_CONDITIONED_EV_SCHEMA_VERSION",
    "FROZEN_REGIME_CONDITIONED_EV_V2_ARTIFACT_NAME",
    "FROZEN_REGIME_CONDITIONED_EV_V2_SCHEMA_VERSION",
    "REGIME_CONDITIONED_EV_FEATURE_GROUPS",
    "REGIME_CONDITIONED_EV_V2_FEATURE_GROUPS",
    "REGIME_CONDITIONED_EV_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION",
    "REGIME_CONDITIONED_EV_FORWARD_SHADOW_SCHEMA_VERSION",
    "ExecutionLayerV2RegimeConditionedEVForwardShadowConfig",
    "ExecutionLayerV2RegimeConditionedEVForwardShadowResult",
    "execution_layer_v2_regime_conditioned_ev_forward_shadow_report_to_markdown",
    "frozen_regime_conditioned_ev_artifact_contract",
    "run_execution_layer_v2_regime_conditioned_ev_forward_shadow",
    "validate_frozen_regime_conditioned_ev_artifact",
]
