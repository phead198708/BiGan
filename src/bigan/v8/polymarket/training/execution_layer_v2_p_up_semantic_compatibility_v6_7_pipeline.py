"""Target-free window freeze pipeline for #227 v6.7 evaluation."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    apply_v6_7_side_residual_calibration,
    validate_v6_7_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

SCHEMA_PREFIX = "bigan-v8-p-up-semantic-execution-compatibility-v6-7-window-freeze"
FROZEN_EVALUATION_PROFILE_SHA256 = (
    "900dba0b3d1e280271ff2489e0d0320f1eca150787bf2be30b8b751a3a993c3e"
)
FROZEN_COLLECTION_PLAN_CORRECTION_SHA256 = (
    "c3162eaa39917ae099c05a8aaf24ca37bc11ba51a3c3afdf60ecfa66f381daba"
)
WINDOW_COUNTS = {
    "fresh_calibration": (60, 90),
    "future_confirmatory": (120, 180),
}


@dataclass(frozen=True, slots=True)
class V67TargetFreeWindowFreezeConfig:
    """Pinned inputs for one target-free calibration or confirmatory freeze."""

    role: Literal["fresh_calibration", "future_confirmatory"]
    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    candidate_freeze_manifest_path: Path | str
    expected_candidate_freeze_manifest_sha256: str
    collection_plan_path: Path | str
    expected_collection_plan_sha256: str
    collection_plan_correction_path: Path | str
    expected_collection_plan_correction_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    implementation_commit: str
    decision_freeze_created_ts: int
    calibration_artifact_path: Path | str | None = None
    expected_calibration_artifact_sha256: str | None = None
    calibration_prediction_freeze_manifest_path: Path | str | None = None
    expected_calibration_prediction_freeze_manifest_sha256: str | None = None
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.role not in WINDOW_COUNTS or not self.run_id.strip():
            raise ValueError("#227 role and run_id are required")
        _require_git_sha(self.implementation_commit)
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("#227 decision freeze timestamp must be positive")
        required_hashes = (
            "expected_evaluation_profile_sha256",
            "expected_candidate_freeze_manifest_sha256",
            "expected_collection_plan_sha256",
            "expected_collection_plan_correction_sha256",
            "expected_collector_index_sha256",
        )
        for name in required_hashes:
            _require_sha256(str(getattr(self, name)), name=name)
        if self.role == "future_confirmatory":
            for name in (
                "calibration_artifact_path",
                "expected_calibration_artifact_sha256",
                "calibration_prediction_freeze_manifest_path",
                "expected_calibration_prediction_freeze_manifest_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#227 confirmatory input missing: {name}")
            for name in (
                "expected_calibration_artifact_sha256",
                "expected_calibration_prediction_freeze_manifest_sha256",
            ):
                _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "evaluation_profile_path",
            "candidate_freeze_manifest_path",
            "collection_plan_path",
            "collection_plan_correction_path",
            "collector_index_path",
            "calibration_artifact_path",
            "calibration_prediction_freeze_manifest_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


def freeze_v6_7_target_free_window(
    config: V67TargetFreeWindowFreezeConfig,
) -> dict[str, Any]:
    """Freeze one exact target-free window and its accepted-bet decisions."""

    inputs = _verified_inputs(config)
    profile = _load_json(inputs["evaluation_profile"])
    candidate_freeze = _load_json(inputs["candidate_freeze"])
    candidate_profile_path = Path(
        _verified_descriptor(candidate_freeze["profile"], "v6.7 candidate profile")[
            "path"
        ]
    )
    validate_p_up_semantic_compatibility_v6_7_profile(
        _load_json(candidate_profile_path)
    )
    correction = _load_json(inputs["collection_plan_correction"])
    validate_v6_7_collection_plan_correction(
        correction,
        original_plan_path=inputs["collection_plan"],
        candidate_freeze_path=inputs["candidate_freeze"],
        profile_path=candidate_profile_path,
    )
    index_rows = load_and_validate_persistent_outcome_blind_index(
        inputs["collector_index"]
    )
    calibration_freeze = (
        _load_json(inputs["calibration_prediction_freeze"])
        if config.role == "future_confirmatory"
        else None
    )
    selected_index_rows, attempted_rows = select_v6_7_window_index_rows(
        index_rows,
        profile=profile,
        role=config.role,
        calibration_prediction_freeze=calibration_freeze,
    )
    if config.decision_freeze_created_ts <= max(
        int(row["market_end_ts"]) for row in selected_index_rows
    ):
        raise ValueError("#227 decision freeze attempted before all markets closed")

    source_manifest = _load_json(
        Path(
            _verified_descriptor(
                candidate_freeze["source_freeze_manifest"], "v6.7 source freeze"
            )["path"]
        )
    )
    candidate_descriptor = _verified_descriptor(
        source_manifest["v6_2_candidate_manifest"], "v6.2 candidate"
    )
    if candidate_descriptor["sha256"] != profile["lineage"][
        "v6_2_candidate_manifest_sha256"
    ]:
        raise ValueError("#227 v6.2 candidate lineage mismatch")
    v6_2_candidate = _load_json(Path(candidate_descriptor["path"]))
    model_descriptor = _verified_descriptor(
        v6_2_candidate["source_model"], "v6.2 source model"
    )
    calibration_descriptor = _verified_descriptor(
        v6_2_candidate["market_clustered_mean_risk_calibration"],
        "v6.2 mean-risk calibration",
    )
    if (
        model_descriptor["sha256"]
        != profile["lineage"]["v6_2_source_model_sha256"]
        or calibration_descriptor["sha256"]
        != profile["lineage"]["v6_2_calibration_sha256"]
    ):
        raise ValueError("#227 frozen v6.2 model/calibration lineage mismatch")
    pre_audit = _load_json(
        Path(
            _verified_descriptor(
                v6_2_candidate["pre_target_access_audit"], "v6.2 pre-target audit"
            )["path"]
        )
    )
    feature_contract = _verified_descriptor(
        pre_audit["feature_contract"], "v6.2 feature contract"
    )
    if feature_contract["sha256"] != profile["lineage"]["feature_contract_sha256"]:
        raise ValueError("#227 feature contract lineage mismatch")
    feature_columns = tuple(
        str(value)
        for value in _load_json(Path(feature_contract["path"]))["feature_columns"]
    )
    feature_rows, raw_lineage = _materialize_selected_window_features(
        selected_index_rows
    )
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_index_rows,
        feature_columns=feature_columns,
    )
    _validate_target_free_grid(
        feature_rows,
        action_rows,
        selected_index_rows=selected_index_rows,
        minimum_created_ts_exclusive=int(
            profile["collection_windows"][
                "future_collection_minimum_created_ts_exclusive"
            ]
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
    candidate_rows, candidate_summary = build_v6_7_target_free_candidate_rows(
        v6_2_predictions,
        action_rows=action_rows,
        profile=_load_json(candidate_profile_path),
    )
    base_selected = [
        {**row, "v6_7_base_score": float(row["v6_7_selection_score"])}
        for row in select_v6_7_target_free_rows(
            candidate_rows, profile=_load_json(candidate_profile_path)
        )
    ]
    legacy_replay = _outcome_blind_acceptance_replay(
        v6_2_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    if config.role == "future_confirmatory":
        selected_decisions = apply_v6_7_side_residual_calibration(
            base_selected,
            calibration_artifact=_load_json(inputs["calibration_artifact"]),
        )
    else:
        selected_decisions = base_selected
    support = _target_free_support(
        selected_decisions,
        profile=profile,
        role=config.role,
        exact_window_market_count=len(selected_index_rows),
    )

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run path exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    paths = {
        "selected_window_rows": run_dir / "v6_7_selected_index_rows.jsonl",
        "attempted_window_rows": run_dir / "v6_7_attempted_index_rows.jsonl",
        "target_free_feature_rows": run_dir / "v6_7_target_free_feature_rows.jsonl",
        "target_free_five_action_rows": run_dir / "v6_7_target_free_five_action_rows.jsonl",
        "v6_2_target_free_predictions": run_dir / "v6_2_target_free_predictions.jsonl",
        "v6_7_candidate_rows": run_dir / "v6_7_candidate_rows.jsonl",
        "v6_7_base_selected_rows": run_dir / "v6_7_base_selected_rows.jsonl",
        "v6_7_selected_decisions": run_dir / "v6_7_selected_decisions.jsonl",
        "matched_legacy_guard_replay": run_dir / "v6_2_matched_legacy_guard_replay.jsonl",
    }
    for name, rows in (
        ("selected_window_rows", selected_index_rows),
        ("attempted_window_rows", attempted_rows),
        ("target_free_feature_rows", feature_rows),
        ("target_free_five_action_rows", action_rows),
        ("v6_2_target_free_predictions", v6_2_predictions),
        ("v6_7_candidate_rows", candidate_rows),
        ("v6_7_base_selected_rows", base_selected),
        ("v6_7_selected_decisions", selected_decisions),
        ("matched_legacy_guard_replay", legacy_replay),
    ):
        _write_jsonl(paths[name], rows)
    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-v1",
        "run_id": config.run_id,
        "role": config.role,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "selected_window_market_count": len(selected_index_rows),
        "selected_window_market_ids": [
            str(row["market_id"]) for row in selected_index_rows
        ],
        "attempted_index_row_count": len(attempted_rows),
        "attempted_sequence_start": int(attempted_rows[0]["sequence"]),
        "attempted_sequence_end": int(attempted_rows[-1]["sequence"]),
        "target_free_support": support,
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "all_selected_markets_closed_before_freeze": True,
        "manual_approval_scope": "offline_v6_7_calibration_and_confirmatory_only",
        "manual_approval_does_not_bypass_statistical_or_side_only_pnl_gate": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "threshold_or_guard_tuning_performed": False,
        **_blocked_safety_fields(),
    }
    decision["decision_freeze_id"] = canonical_json_sha256(decision)
    decision_path = run_dir / "v6_7_accepted_bet_decision_freeze.json"
    _write_json(decision_path, decision)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "role": config.role,
        "selected_window_market_count": len(selected_index_rows),
        "attempted_index_row_count": len(attempted_rows),
        "candidate_summary": candidate_summary,
        "base_selected_side_count": dict(
            sorted(Counter(str(row["side"]) for row in base_selected).items())
        ),
        "final_selected_side_count": support["count_by_side"],
        "target_free_support_gate_passed": support["target_free_support_gate_passed"],
        "target_free_support_blocking_reason_codes": support["blocking_reason_codes"],
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "feature_causality_violation_count": 0,
        "complete_five_action_grid_passed": True,
        "collection_plan_correction_verified": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_7_target_free_window_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _freeze_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "role": config.role,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "candidate_freeze_manifest": _descriptor(inputs["candidate_freeze"]),
        "collection_plan": _descriptor(inputs["collection_plan"]),
        "collection_plan_correction": _descriptor(inputs["collection_plan_correction"]),
        "collector_index": _descriptor(inputs["collector_index"]),
        "opened_raw_feature_artifacts": raw_lineage,
        **{name: _descriptor(path) for name, path in paths.items()},
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "calibration_artifact": (
            _descriptor(inputs["calibration_artifact"])
            if config.role == "future_confirmatory"
            else None
        ),
        "calibration_prediction_freeze_manifest": (
            _descriptor(inputs["calibration_prediction_freeze"])
            if config.role == "future_confirmatory"
            else None
        ),
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_7_target_free_window_freeze_manifest.json"
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


def select_v6_7_window_index_rows(
    index_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    role: str,
    calibration_prediction_freeze: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the earliest exact preregistered quality-valid window."""

    validate_v6_7_evaluation_profile(profile)
    if role not in WINDOW_COUNTS:
        raise ValueError("#227 unsupported window role")
    target, scan_cap = WINDOW_COUNTS[role]
    minimum_ts = int(
        profile["collection_windows"][
            "future_collection_minimum_created_ts_exclusive"
        ]
    )
    prior_market_ids: set[str] = set()
    minimum_sequence = 1
    minimum_confirmatory_ts = minimum_ts
    if role == "future_confirmatory":
        if calibration_prediction_freeze is None:
            raise ValueError("#227 confirmatory requires calibration prediction freeze")
        if (
            calibration_prediction_freeze.get("schema_version")
            != f"{SCHEMA_PREFIX}-manifest-v1"
            or calibration_prediction_freeze.get("role") != "fresh_calibration"
            or calibration_prediction_freeze.get("future_target_access_allowed") is not True
            or calibration_prediction_freeze.get(
                "labels_outcomes_resolution_or_pnl_opened"
            )
            is not False
        ):
            raise ValueError("#227 calibration prediction freeze is not eligible")
        decision = _load_json(
            Path(
                _verified_descriptor(
                    calibration_prediction_freeze["accepted_bet_decision_freeze"],
                    "calibration decision freeze",
                )["path"]
            )
        )
        selected = _load_jsonl(
            Path(
                _verified_descriptor(
                    calibration_prediction_freeze["selected_window_rows"],
                    "calibration selected rows",
                )["path"]
            )
        )
        prior_market_ids = {str(row["market_id"]) for row in selected}
        minimum_sequence = int(decision["attempted_sequence_end"]) + 1
        minimum_confirmatory_ts = max(int(row["market_end_ts"]) for row in selected)

    eligible = [
        row
        for row in index_rows
        if int(row["sequence"]) >= minimum_sequence
        and int(row.get("scheduled_round_start_ts") or 0) > minimum_ts
    ][:scan_cap]
    selected_rows: list[dict[str, Any]] = []
    attempted_rows: list[dict[str, Any]] = []
    seen: set[str] = set(prior_market_ids)
    for row in eligible:
        attempted_rows.append(row)
        _validate_outcome_blind_index_row(row)
        if row.get("capture_quality_valid") is False:
            continue
        if row.get("capture_quality_valid") is not True:
            raise ValueError("#227 capture quality status is not explicit")
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in seen:
            raise ValueError("#227 window market identity is missing or overlapping")
        if role == "future_confirmatory" and int(
            row["scheduled_round_start_ts"]
        ) <= minimum_confirmatory_ts:
            raise ValueError("#227 confirmatory row is not strictly after calibration")
        selected_rows.append(row)
        seen.add(market_id)
        if len(selected_rows) == target:
            break
    if len(selected_rows) != target:
        raise ValueError(
            f"#227 {role} window has insufficient quality-valid rows before scan cap"
        )
    if not attempted_rows:
        raise ValueError("#227 selected window has no attempted rows")
    return selected_rows, attempted_rows


def _verified_inputs(config: V67TargetFreeWindowFreezeConfig) -> dict[str, Path]:
    inputs = {
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
        "candidate_freeze": Path(config.candidate_freeze_manifest_path).resolve(),
        "collection_plan": Path(config.collection_plan_path).resolve(),
        "collection_plan_correction": Path(
            config.collection_plan_correction_path
        ).resolve(),
        "collector_index": Path(config.collector_index_path).resolve(),
    }
    expected = {
        "evaluation_profile": config.expected_evaluation_profile_sha256,
        "candidate_freeze": config.expected_candidate_freeze_manifest_sha256,
        "collection_plan": config.expected_collection_plan_sha256,
        "collection_plan_correction": (
            config.expected_collection_plan_correction_sha256
        ),
        "collector_index": config.expected_collector_index_sha256,
    }
    if config.role == "future_confirmatory":
        inputs["calibration_artifact"] = Path(config.calibration_artifact_path).resolve()
        inputs["calibration_prediction_freeze"] = Path(
            config.calibration_prediction_freeze_manifest_path
        ).resolve()
        expected["calibration_artifact"] = str(
            config.expected_calibration_artifact_sha256
        )
        expected["calibration_prediction_freeze"] = str(
            config.expected_calibration_prediction_freeze_manifest_sha256
        )
    for name, path in inputs.items():
        _verify_pin(path, str(expected[name]), f"#227 {name}")
    if config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256:
        raise ValueError("#227 evaluation profile is not the frozen contract")
    if (
        config.expected_collection_plan_correction_sha256
        != FROZEN_COLLECTION_PLAN_CORRECTION_SHA256
    ):
        raise ValueError("#227 collection-plan correction is not authoritative")
    profile = _load_json(inputs["evaluation_profile"])
    validate_v6_7_evaluation_profile(profile)
    if config.expected_candidate_freeze_manifest_sha256 != profile["lineage"][
        "candidate_freeze_manifest_sha256"
    ]:
        raise ValueError("#227 candidate freeze hash mismatches evaluation profile")
    if config.expected_collection_plan_sha256 != profile["lineage"][
        "collection_plan_sha256"
    ]:
        raise ValueError("#227 collection plan hash mismatches evaluation profile")
    return inputs


def _validate_outcome_blind_index_row(row: dict[str, Any]) -> None:
    if row.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("#227 collector index opened targets")
    if int(row.get("raw_resolution_row_count") or 0) != 0:
        raise ValueError("#227 collector index contains resolution rows")
    for field, expected in _blocked_safety_fields().items():
        if row.get(field) != expected:
            raise ValueError(f"#227 collector safety mismatch: {field}")


def _validate_target_free_grid(
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    *,
    selected_index_rows: list[dict[str, Any]],
    minimum_created_ts_exclusive: int,
) -> None:
    selected_by_market = {
        str(row["market_id"]): row for row in selected_index_rows
    }
    feature_keys = {
        (str(row["market_id"]), int(row["decision_ts"])) for row in feature_rows
    }
    if {market_id for market_id, _ in feature_keys} != set(selected_by_market):
        raise ValueError("#227 feature market coverage mismatch")
    expected_actions = {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    }
    grouped: dict[tuple[str, int], set[str]] = {}
    for row in action_rows:
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#227 target-free feature causality violation")
        market_id = str(row["market_id"])
        source = selected_by_market[market_id]
        if (
            int(row["decision_ts"]) <= int(source["market_start_ts"])
            or int(row["decision_ts"]) >= int(source["market_end_ts"])
            or int(source["scheduled_round_start_ts"])
            <= minimum_created_ts_exclusive
        ):
            raise ValueError("#227 target-free decision is outside frozen window")
        grouped.setdefault((market_id, int(row["decision_ts"])), set()).add(
            str(row["action"])
        )
    if set(grouped) != feature_keys or any(
        actions != expected_actions for actions in grouped.values()
    ):
        raise ValueError("#227 target-free five-action grid incomplete")


def _target_free_support(
    rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    role: str,
    exact_window_market_count: int,
) -> dict[str, Any]:
    count = Counter(str(row["side"]) for row in rows)
    if role == "fresh_calibration":
        minimum_side = int(
            profile["fresh_calibration"][
                "minimum_selected_unique_market_count_per_side"
            ]
        )
        minimum_total = 60
    else:
        minimum_side = int(
            profile["confirmatory_side_only_pnl_gate"][
                "minimum_supported_side_unique_market_count"
            ]
        )
        minimum_total = int(
            profile["confirmatory_side_only_pnl_gate"][
                "minimum_guard_accepted_unique_market_count"
            ]
        )
    checks = {
        "exact_window_market_count": exact_window_market_count
        == WINDOW_COUNTS[role][0],
        "one_selected_row_per_market": len({str(row["market_id"]) for row in rows})
        == len(rows),
        "minimum_selected_market_support": len(rows) >= minimum_total,
        "buy_up_support": count["UP"] >= minimum_side,
        "buy_down_support": count["DOWN"] >= minimum_side,
        "all_scores_positive": all(
            float(
                row.get("v6_7_calibrated_runtime_pnl_lcb")
                if role == "future_confirmatory"
                else row["v6_7_base_score"]
            )
            > 0.0
            for row in rows
        ),
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in rows
        ),
        "targets_sealed": all(
            row.get("labels_outcomes_resolution_or_pnl_opened") is False for row in rows
        ),
        "source_scores_unchanged": all(
            row.get("source_score_mutated") is False for row in rows
        ),
    }
    blockers = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    return {
        "selected_market_count": len(rows),
        "count_by_side": {side: count[side] for side in ("UP", "DOWN")},
        "minimum_total_required": minimum_total,
        "minimum_per_side_required": minimum_side,
        "checks": checks,
        "target_free_support_gate_passed": not blockers,
        "blocking_reason_codes": blockers,
    }


def _freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Target-Free Window Freeze",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- role: `{report['role']}`",
            f"- selected window markets: `{report['selected_window_market_count']}`",
            f"- attempted index rows: `{report['attempted_index_row_count']}`",
            f"- base side count: `{report['base_selected_side_count']}`",
            f"- final side count: `{report['final_selected_side_count']}`",
            "- target-free support passed: "
            f"`{str(report['target_free_support_gate_passed']).lower()}`",
            f"- blockers: `{report['target_free_support_blocking_reason_codes']}`",
            "- labels/outcomes/resolution/PnL opened: `false`",
            "- source score mutated: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "V67TargetFreeWindowFreezeConfig",
    "freeze_v6_7_target_free_window",
    "select_v6_7_window_index_rows",
]
