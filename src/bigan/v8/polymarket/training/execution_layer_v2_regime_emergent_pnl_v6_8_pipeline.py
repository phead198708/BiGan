"""Target-free future-confirmatory freeze for #229 v6.8."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2 import (
    apply_market_clustered_mean_ev_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
    validate_v6_7_collection_plan_correction,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation import (
    validate_v6_7_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    _validate_outcome_blind_index_row,
    _validate_target_free_grid,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    apply_v6_8_pooled_residual_calibration,
    build_regime_emergent_target_free_support,
    validate_regime_emergent_pnl_v6_8_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_adoption import (
    SCHEMA_PREFIX as ADOPTION_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-regime-emergent-pnl-v6-8-confirmatory-freeze"
FROZEN_EVALUATION_PROFILE_SHA256 = (
    "d885b5a81fc217175eefac8a27c53eadd8044fd7731148624396709db5167dfe"
)
FROZEN_V6_7_EVALUATION_PROFILE_SHA256 = (
    "900dba0b3d1e280271ff2489e0d0320f1eca150787bf2be30b8b751a3a993c3e"
)
FROZEN_COLLECTION_PLAN_CORRECTION_SHA256 = (
    "c3162eaa39917ae099c05a8aaf24ca37bc11ba51a3c3afdf60ecfa66f381daba"
)
CONFIRMATORY_WINDOW_MARKET_COUNT = 120
CONFIRMATORY_SCAN_CAP = 180


@dataclass(frozen=True, slots=True)
class V68ConfirmatoryFreezeConfig:
    """Pinned inputs for the single target-free v6.8 confirmatory freeze."""

    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    v6_7_evaluation_profile_path: Path | str
    expected_v6_7_evaluation_profile_sha256: str
    candidate_freeze_manifest_path: Path | str
    expected_candidate_freeze_manifest_sha256: str
    collection_plan_path: Path | str
    expected_collection_plan_sha256: str
    collection_plan_correction_path: Path | str
    expected_collection_plan_correction_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    calibration_adoption_manifest_path: Path | str
    expected_calibration_adoption_manifest_sha256: str
    calibration_manifest_path: Path | str
    expected_calibration_manifest_sha256: str
    calibration_artifact_path: Path | str
    expected_calibration_artifact_sha256: str
    implementation_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.decision_freeze_created_ts <= 0:
            raise ValueError("#229 confirmatory run id and freeze timestamp are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_evaluation_profile_sha256",
            "expected_v6_7_evaluation_profile_sha256",
            "expected_candidate_freeze_manifest_sha256",
            "expected_collection_plan_sha256",
            "expected_collection_plan_correction_sha256",
            "expected_collector_index_sha256",
            "expected_calibration_adoption_manifest_sha256",
            "expected_calibration_manifest_sha256",
            "expected_calibration_artifact_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "evaluation_profile_path",
            "v6_7_evaluation_profile_path",
            "candidate_freeze_manifest_path",
            "collection_plan_path",
            "collection_plan_correction_path",
            "collector_index_path",
            "calibration_adoption_manifest_path",
            "calibration_manifest_path",
            "calibration_artifact_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def freeze_v6_8_confirmatory_window(
    config: V68ConfirmatoryFreezeConfig,
) -> dict[str, Any]:
    """Freeze and score the earliest exact-120 strictly-later window."""

    inputs = _verified_inputs(config)
    profile = _load_json(inputs["evaluation_profile"])
    v6_7_profile = _load_json(inputs["v6_7_evaluation_profile"])
    candidate_freeze = _load_json(inputs["candidate_freeze"])
    calibration_adoption = _load_json(inputs["calibration_adoption"])
    calibration_artifact = _load_json(inputs["calibration_artifact"])
    index_rows = load_and_validate_persistent_outcome_blind_index(inputs["collector_index"])
    selected_index_rows, attempted_rows = select_v6_8_confirmatory_index_rows(
        index_rows,
        calibration_adoption_manifest=calibration_adoption,
    )
    if config.decision_freeze_created_ts <= max(
        int(row["market_end_ts"]) for row in selected_index_rows
    ):
        raise ValueError("#229 confirmatory freeze attempted before all markets closed")

    candidate_profile_path, model_descriptor, calibration_descriptor, feature_columns = (
        _frozen_model_inputs(
            candidate_freeze,
            v6_7_profile=v6_7_profile,
        )
    )
    validate_v6_7_collection_plan_correction(
        _load_json(inputs["collection_plan_correction"]),
        original_plan_path=inputs["collection_plan"],
        candidate_freeze_path=inputs["candidate_freeze"],
        profile_path=candidate_profile_path,
    )
    feature_rows, raw_lineage = _materialize_selected_window_features(selected_index_rows)
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_index_rows,
        feature_columns=feature_columns,
    )
    _validate_target_free_grid(
        feature_rows,
        action_rows,
        selected_index_rows=selected_index_rows,
        minimum_created_ts_exclusive=max(
            int(row["market_end_ts"])
            for row in _load_jsonl_descriptor(
                calibration_adoption["selected_window_rows"],
                "calibration selected rows",
            )
        ),
    )
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw_predictions = _raw_target_stripped_predictions(
        booster, action_rows, feature_columns=feature_columns
    )
    v6_2_predictions = apply_market_clustered_mean_ev_scores(
        attach_frozen_execution_compatibility(raw_predictions),
        calibration_artifact=_load_json(Path(calibration_descriptor["path"])),
    )
    candidate_profile = _load_json(candidate_profile_path)
    candidate_rows, candidate_summary = build_v6_7_target_free_candidate_rows(
        v6_2_predictions,
        action_rows=action_rows,
        profile=candidate_profile,
    )
    base_selected = [
        {**row, "v6_7_base_score": float(row["v6_7_selection_score"])}
        for row in select_v6_7_target_free_rows(candidate_rows, profile=candidate_profile)
    ]
    selected_decisions = apply_v6_8_pooled_residual_calibration(
        base_selected,
        calibration_artifact=calibration_artifact,
    )
    support = build_regime_emergent_target_free_support(
        selected_decisions,
        exact_window_market_count=len(selected_index_rows),
        expected_window_market_count=CONFIRMATORY_WINDOW_MARKET_COUNT,
        required_total_market_count=int(
            profile["future_confirmatory"]["minimum_guard_accepted_unique_market_count_total"]
        ),
        score_field="v6_8_calibrated_runtime_pnl_lcb",
    )
    legacy_replay = _outcome_blind_acceptance_replay(
        v6_2_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run path exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    paths = {
        "selected_window_rows": run_dir / "v6_8_confirmatory_selected_index_rows.jsonl",
        "attempted_window_rows": run_dir / "v6_8_confirmatory_attempted_index_rows.jsonl",
        "target_free_feature_rows": run_dir / "v6_8_confirmatory_feature_rows.jsonl",
        "target_free_five_action_rows": run_dir / "v6_8_confirmatory_five_action_rows.jsonl",
        "v6_2_target_free_predictions": run_dir / "v6_2_confirmatory_predictions.jsonl",
        "v6_7_candidate_rows": run_dir / "v6_7_confirmatory_candidate_rows.jsonl",
        "v6_7_base_selected_rows": run_dir / "v6_7_confirmatory_base_selected_rows.jsonl",
        "v6_8_selected_decisions": run_dir / "v6_8_confirmatory_selected_decisions.jsonl",
        "matched_legacy_guard_replay": run_dir / "v6_2_confirmatory_legacy_guard_replay.jsonl",
    }
    for name, rows in (
        ("selected_window_rows", selected_index_rows),
        ("attempted_window_rows", attempted_rows),
        ("target_free_feature_rows", feature_rows),
        ("target_free_five_action_rows", action_rows),
        ("v6_2_target_free_predictions", v6_2_predictions),
        ("v6_7_candidate_rows", candidate_rows),
        ("v6_7_base_selected_rows", base_selected),
        ("v6_8_selected_decisions", selected_decisions),
        ("matched_legacy_guard_replay", legacy_replay),
    ):
        _write_jsonl(paths[name], rows)
    selected_ids = [str(row["market_id"]) for row in selected_index_rows]
    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "selected_window_market_count": len(selected_index_rows),
        "selected_window_market_ids": selected_ids,
        "attempted_index_row_count": len(attempted_rows),
        "attempted_sequence_start": int(attempted_rows[0]["sequence"]),
        "attempted_sequence_end": int(attempted_rows[-1]["sequence"]),
        "regime_emergent_target_free_support": support,
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "all_selected_markets_closed_before_freeze": True,
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "threshold_or_guard_tuning_performed": False,
        "manual_approval_does_not_bypass_execution_pnl_gate": True,
        **_blocked_safety_fields(),
    }
    decision["decision_freeze_id"] = canonical_json_sha256(decision)
    decision_path = run_dir / "v6_8_confirmatory_accepted_bet_decision_freeze.json"
    _write_json(decision_path, decision)
    side_count = Counter(str(row["side"]) for row in selected_decisions)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "selected_window_market_count": len(selected_index_rows),
        "attempted_index_row_count": len(attempted_rows),
        "candidate_summary": candidate_summary,
        "base_selected_market_count": len(base_selected),
        "guard_accepted_market_count": len(selected_decisions),
        "guard_accepted_side_count_diagnostic": dict(sorted(side_count.items())),
        "side_count_hard_gate_enabled": False,
        "target_free_support_gate_passed": support["target_free_support_gate_passed"],
        "target_free_support_blocking_reason_codes": support["blocking_reason_codes"],
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "feature_causality_violation_count": 0,
        "complete_five_action_grid_passed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_8_confirmatory_target_free_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "v6_7_evaluation_profile": _descriptor(inputs["v6_7_evaluation_profile"]),
        "candidate_freeze_manifest": _descriptor(inputs["candidate_freeze"]),
        "collection_plan": _descriptor(inputs["collection_plan"]),
        "collection_plan_correction": _descriptor(inputs["collection_plan_correction"]),
        "collector_index": _descriptor(inputs["collector_index"]),
        "calibration_adoption_manifest": _descriptor(inputs["calibration_adoption"]),
        "calibration_manifest": _descriptor(inputs["calibration_manifest"]),
        "calibration_artifact": _descriptor(inputs["calibration_artifact"]),
        "opened_raw_feature_artifacts": raw_lineage,
        **{name: _descriptor(path) for name, path in paths.items()},
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_8_confirmatory_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "decision_freeze_path": decision_path,
        "decision_freeze_sha256": _sha256_file(decision_path),
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def select_v6_8_confirmatory_index_rows(
    index_rows: list[dict[str, Any]],
    *,
    calibration_adoption_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the earliest exact-120 rows after the sealed calibration boundary."""

    _validate_calibration_adoption(calibration_adoption_manifest)
    selected_calibration = _load_jsonl_descriptor(
        calibration_adoption_manifest["selected_window_rows"],
        "calibration selected rows",
    )
    attempted_calibration = _load_jsonl_descriptor(
        calibration_adoption_manifest["attempted_window_rows"],
        "calibration attempted rows",
    )
    prior_ids = {str(row["market_id"]) for row in selected_calibration}
    minimum_sequence = max(int(row["sequence"]) for row in attempted_calibration) + 1
    minimum_start_ts = max(int(row["market_end_ts"]) for row in selected_calibration)
    eligible = [
        row
        for row in index_rows
        if int(row["sequence"]) >= minimum_sequence
        and int(row["scheduled_round_start_ts"]) > minimum_start_ts
    ][:CONFIRMATORY_SCAN_CAP]
    selected: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    seen = set(prior_ids)
    for row in eligible:
        attempted.append(row)
        _validate_outcome_blind_index_row(row)
        if row.get("capture_quality_valid") is False:
            continue
        if row.get("capture_quality_valid") is not True:
            raise ValueError("#229 capture quality status is not explicit")
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in seen:
            raise ValueError("#229 confirmatory market identity is missing or overlapping")
        selected.append(row)
        seen.add(market_id)
        if len(selected) == CONFIRMATORY_WINDOW_MARKET_COUNT:
            break
    if len(selected) != CONFIRMATORY_WINDOW_MARKET_COUNT:
        raise ValueError(
            "#229 future_confirmatory window has insufficient quality-valid rows before scan cap"
        )
    if not attempted:
        raise ValueError("#229 confirmatory window has no attempted rows")
    return selected, attempted


def _verified_inputs(config: V68ConfirmatoryFreezeConfig) -> dict[str, Path]:
    inputs = {
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
        "v6_7_evaluation_profile": Path(config.v6_7_evaluation_profile_path).resolve(),
        "candidate_freeze": Path(config.candidate_freeze_manifest_path).resolve(),
        "collection_plan": Path(config.collection_plan_path).resolve(),
        "collection_plan_correction": Path(config.collection_plan_correction_path).resolve(),
        "collector_index": Path(config.collector_index_path).resolve(),
        "calibration_adoption": Path(config.calibration_adoption_manifest_path).resolve(),
        "calibration_manifest": Path(config.calibration_manifest_path).resolve(),
        "calibration_artifact": Path(config.calibration_artifact_path).resolve(),
    }
    expected = {
        "evaluation_profile": config.expected_evaluation_profile_sha256,
        "v6_7_evaluation_profile": config.expected_v6_7_evaluation_profile_sha256,
        "candidate_freeze": config.expected_candidate_freeze_manifest_sha256,
        "collection_plan": config.expected_collection_plan_sha256,
        "collection_plan_correction": config.expected_collection_plan_correction_sha256,
        "collector_index": config.expected_collector_index_sha256,
        "calibration_adoption": config.expected_calibration_adoption_manifest_sha256,
        "calibration_manifest": config.expected_calibration_manifest_sha256,
        "calibration_artifact": config.expected_calibration_artifact_sha256,
    }
    for name, path in inputs.items():
        _verify_pin(path, expected[name], f"#229 {name}")
    if (
        config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256
        or config.expected_v6_7_evaluation_profile_sha256 != FROZEN_V6_7_EVALUATION_PROFILE_SHA256
        or config.expected_collection_plan_correction_sha256
        != FROZEN_COLLECTION_PLAN_CORRECTION_SHA256
    ):
        raise ValueError("#229 confirmatory frozen contract hash mismatch")
    profile = _load_json(inputs["evaluation_profile"])
    validate_regime_emergent_pnl_v6_8_profile(profile)
    v6_7_profile = _load_json(inputs["v6_7_evaluation_profile"])
    validate_v6_7_evaluation_profile(v6_7_profile)
    if (
        expected["v6_7_evaluation_profile"] != profile["lineage"]["v6_7_evaluation_profile_sha256"]
        or expected["candidate_freeze"] != profile["lineage"]["candidate_freeze_manifest_sha256"]
        or expected["candidate_freeze"]
        != v6_7_profile["lineage"]["candidate_freeze_manifest_sha256"]
        or expected["collection_plan"] != v6_7_profile["lineage"]["collection_plan_sha256"]
    ):
        raise ValueError("#229 confirmatory source lineage mismatch")
    calibration_manifest = _load_json(inputs["calibration_manifest"])
    if (
        calibration_manifest.get("calibration_gate_passed") is not True
        or calibration_manifest.get("future_confirmatory_freeze_allowed") is not True
        or calibration_manifest.get("calibration_artifact")
        != _descriptor(inputs["calibration_artifact"])
    ):
        raise ValueError("#229 pooled calibration manifest is not confirmatory eligible")
    calibration_artifact = _load_json(inputs["calibration_artifact"])
    if (
        calibration_artifact.get("schema_version") != CALIBRATION_ARTIFACT_SCHEMA_VERSION
        or calibration_artifact.get("calibration_gate_passed") is not True
        or calibration_artifact.get("calibration_gate_blocking_reason_codes") != []
    ):
        raise ValueError("#229 pooled calibration artifact is not eligible")
    return inputs


def _frozen_model_inputs(
    candidate_freeze: dict[str, Any],
    *,
    v6_7_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    candidate_profile_path = Path(
        _verified_descriptor(candidate_freeze["profile"], "v6.7 candidate profile")["path"]
    )
    validate_p_up_semantic_compatibility_v6_7_profile(_load_json(candidate_profile_path))
    source_manifest = _load_json(
        Path(
            _verified_descriptor(candidate_freeze["source_freeze_manifest"], "v6.7 source freeze")[
                "path"
            ]
        )
    )
    v6_2_descriptor = _verified_descriptor(
        source_manifest["v6_2_candidate_manifest"], "v6.2 candidate"
    )
    if v6_2_descriptor["sha256"] != v6_7_profile["lineage"]["v6_2_candidate_manifest_sha256"]:
        raise ValueError("#229 v6.2 candidate lineage mismatch")
    v6_2_candidate = _load_json(Path(v6_2_descriptor["path"]))
    model = _verified_descriptor(v6_2_candidate["source_model"], "v6.2 source model")
    calibration = _verified_descriptor(
        v6_2_candidate["market_clustered_mean_risk_calibration"],
        "v6.2 mean-risk calibration",
    )
    if (
        model["sha256"] != v6_7_profile["lineage"]["v6_2_source_model_sha256"]
        or calibration["sha256"] != v6_7_profile["lineage"]["v6_2_calibration_sha256"]
    ):
        raise ValueError("#229 frozen v6.2 model/calibration lineage mismatch")
    pre_audit = _load_json(
        Path(
            _verified_descriptor(
                v6_2_candidate["pre_target_access_audit"], "v6.2 pre-target audit"
            )["path"]
        )
    )
    feature_contract = _verified_descriptor(pre_audit["feature_contract"], "v6.2 feature contract")
    if feature_contract["sha256"] != v6_7_profile["lineage"]["feature_contract_sha256"]:
        raise ValueError("#229 feature contract lineage mismatch")
    columns = tuple(
        str(value) for value in _load_json(Path(feature_contract["path"]))["feature_columns"]
    )
    return candidate_profile_path, model, calibration, columns


def _validate_calibration_adoption(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != f"{ADOPTION_SCHEMA_PREFIX}-manifest-v1"
        or manifest.get("role") != "fresh_calibration"
        or manifest.get("future_target_access_allowed") is not True
        or manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or manifest.get("side_count_hard_gate_enabled") is not False
    ):
        raise ValueError("#229 calibration adoption is not a valid future boundary")


def _load_jsonl_descriptor(descriptor: dict[str, Any], name: str) -> list[dict[str, Any]]:
    from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
        _load_jsonl,
    )

    return _load_jsonl(Path(_verified_descriptor(descriptor, name)["path"]))


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.8 Future Confirmatory Target-Free Freeze",
            "",
            f"- window markets: `{report['selected_window_market_count']}`",
            f"- guard-accepted markets: `{report['guard_accepted_market_count']}`",
            f"- side composition diagnostic: `{report['guard_accepted_side_count_diagnostic']}`",
            "- side-count hard gate: `false`",
            f"- target-free support passed: `{str(report['target_free_support_gate_passed']).lower()}`",
            f"- blockers: `{report['target_free_support_blocking_reason_codes']}`",
            "- labels/outcomes/PnL opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "CONFIRMATORY_SCAN_CAP",
    "CONFIRMATORY_WINDOW_MARKET_COUNT",
    "V68ConfirmatoryFreezeConfig",
    "freeze_v6_8_confirmatory_window",
    "select_v6_8_confirmatory_index_rows",
]
