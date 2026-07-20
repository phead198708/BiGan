"""Freeze, settle, and calibrate the preregistered #226 v6.6 candidate."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
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
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_runtime_pnl_v6_6 import (
    CANDIDATE_NAME,
    score_policy_selected_runtime_pnl_rows,
    validate_policy_selected_runtime_pnl_v6_6_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _market_runtime_target_rows,
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)

SCHEMA_PREFIX = "bigan-v8-policy-selected-runtime-pnl-v6-6-fresh-calibration"
TARGET_MARKET_COUNT = 60
MAXIMUM_ATTEMPT_COUNT = 90
SIDES = ("UP", "DOWN")
SBC_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}
SINGLE_USE_CLAIM_FILENAME = "v6_6_fresh_calibration_single_use_claim.json"
FROZEN_PROFILE_SHA256 = "89623c0feafcdfb5a2f491a7ca22c0daa8a5503f5feb876ed7a5c47f790f3f7f"
FROZEN_POINT_FREEZE_MANIFEST_SHA256 = (
    "f4be1044fcd0934a0d93a5ce04307e064c2437bb62f60f28321e21b3bef3d469"
)
FROZEN_V6_2_CANDIDATE_MANIFEST_SHA256 = (
    "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9"
)
FROZEN_V6_2_CALIBRATION_SHA256 = (
    "dc82ddebc51e95e46477894f2a0ba7bd8fa2f6845b22ced43402822b66b68e43"
)
FROZEN_RUNTIME_POLICY_PROFILE_SHA256 = (
    "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
)


@dataclass(frozen=True, slots=True)
class PolicySelectedRuntimePNLV66CalibrationConfig:
    """Pinned inputs for one #226 calibration stage."""

    stage: Literal["freeze_predictions", "settle", "calibrate"]
    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    point_freeze_manifest_path: Path | str
    expected_point_freeze_manifest_sha256: str
    implementation_commit: str
    v6_2_candidate_manifest_path: Path | str | None = None
    expected_v6_2_candidate_manifest_sha256: str | None = None
    collector_index_path: Path | str | None = None
    expected_collector_index_sha256: str | None = None
    runtime_policy_profile_path: Path | str | None = None
    expected_runtime_policy_profile_sha256: str | None = None
    prediction_freeze_manifest_path: Path | str | None = None
    expected_prediction_freeze_manifest_sha256: str | None = None
    settled_corpus_index_path: Path | str | None = None
    expected_settled_corpus_index_sha256: str | None = None
    stage_started_ts: int = 0
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"freeze_predictions", "settle", "calibrate"}:
            raise ValueError("unsupported #226 calibration stage")
        if not self.run_id.strip() or self.stage_started_ts <= 0:
            raise ValueError("run_id and stage_started_ts are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_profile_sha256",
            "expected_point_freeze_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        for name in (
            "output_dir",
            "profile_path",
            "point_freeze_manifest_path",
            "v6_2_candidate_manifest_path",
            "collector_index_path",
            "runtime_policy_profile_path",
            "prediction_freeze_manifest_path",
            "settled_corpus_index_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        required = {
            "freeze_predictions": (
                "v6_2_candidate_manifest_path",
                "expected_v6_2_candidate_manifest_sha256",
                "collector_index_path",
                "expected_collector_index_sha256",
            ),
            "settle": (
                "prediction_freeze_manifest_path",
                "expected_prediction_freeze_manifest_sha256",
            ),
            "calibrate": (
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
                "prediction_freeze_manifest_path",
                "expected_prediction_freeze_manifest_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
            ),
        }[self.stage]
        missing = [name for name in required if getattr(self, name) in (None, "")]
        if missing:
            raise ValueError("missing #226 stage inputs: " + ",".join(missing))
        for name in required:
            if name.startswith("expected_"):
                _require_sha256(str(getattr(self, name)), name)


def run_policy_selected_runtime_pnl_v6_6_calibration(
    config: PolicySelectedRuntimePNLV66CalibrationConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Run one explicitly requested #226 calibration stage."""

    if config.stage == "freeze_predictions":
        return _freeze_predictions(config)
    if config.stage == "settle":
        return _settle(
            config,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    return _calibrate(config)


def _freeze_predictions(
    config: PolicySelectedRuntimePNLV66CalibrationConfig,
) -> dict[str, Any]:
    profile_path, point_path = _verified_common_inputs(config)
    candidate_path = Path(config.v6_2_candidate_manifest_path).resolve()
    index_path = Path(config.collector_index_path).resolve()
    if (
        config.expected_v6_2_candidate_manifest_sha256
        != FROZEN_V6_2_CANDIDATE_MANIFEST_SHA256
    ):
        raise ValueError("v6.2 candidate is not the frozen #226 source candidate")
    _verify_pin(
        candidate_path,
        str(config.expected_v6_2_candidate_manifest_sha256),
        "frozen v6.2 candidate",
    )
    _verify_pin(
        index_path,
        str(config.expected_collector_index_sha256),
        "collector index",
    )
    profile = _load_json(profile_path)
    point_manifest = _validated_point_freeze(point_path, profile=profile)
    point_model_descriptor = _verified_descriptor(point_manifest["point_model"], "point model")
    point_model = _load_json(Path(point_model_descriptor["path"]))
    candidate = _load_json(candidate_path)
    source_model_descriptor = _verified_descriptor(candidate["source_model"], "v6.2 model")
    if source_model_descriptor["sha256"] != profile["source_lineage"]["v6_2_source_model_sha256"]:
        raise ValueError("v6.2 source model lineage mismatch")
    calibration_descriptor = _verified_descriptor(
        candidate["market_clustered_mean_risk_calibration"], "v6.2 calibration"
    )
    if calibration_descriptor["sha256"] != FROZEN_V6_2_CALIBRATION_SHA256:
        raise ValueError("v6.2 calibration lineage mismatch")
    pre_audit = _load_json(
        Path(_verified_descriptor(candidate["pre_target_access_audit"], "v6.2 audit")["path"])
    )
    feature_contract_descriptor = _verified_descriptor(
        pre_audit["feature_contract"], "feature contract"
    )
    feature_columns = tuple(
        str(value)
        for value in _load_json(Path(feature_contract_descriptor["path"]))["feature_columns"]
    )
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    _verify_index_boundary(index_path, index_rows=index_rows, profile=profile)
    train_rows = _load_jsonl(
        Path(_verified_descriptor(point_manifest["selected_train_rows"], "selected train rows")["path"])
    )
    selected_rows, attempted_rows = select_exact_v6_6_calibration_index_rows(
        index_rows,
        profile=profile,
        prior_market_ids={str(row["market_id"]) for row in train_rows},
    )
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("prediction freeze attempted before all exact-60 markets closed")
    feature_rows, raw_lineage = _materialize_selected_window_features(selected_rows)
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_rows,
        feature_columns=feature_columns,
    )
    _validate_target_free_grid(
        feature_rows,
        action_rows,
        selected_rows=selected_rows,
        minimum_market_start_ts_exclusive=int(
            profile["fresh_calibration_collection"]["minimum_market_start_ts_exclusive"]
        ),
    )
    booster = xgb.Booster()
    booster.load_model(source_model_descriptor["path"])
    raw_predictions = _raw_target_stripped_predictions(
        booster, action_rows, feature_columns=feature_columns
    )
    v6_2_predictions = apply_market_clustered_mean_ev_scores(
        attach_frozen_execution_compatibility(raw_predictions),
        calibration_artifact=_load_json(Path(calibration_descriptor["path"])),
    )
    replay = _outcome_blind_acceptance_replay(
        v6_2_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    accepted_rows = build_v6_6_policy_selected_target_free_rows(
        replay, predictions=v6_2_predictions
    )
    point_predictions = score_policy_selected_runtime_pnl_rows(
        accepted_rows, model=point_model
    )
    support = _target_free_support(point_predictions, profile=profile)
    run_dir = _prepare_run_dir(config)
    paths = {
        "selected_window_rows": run_dir / "v6_6_exact_60_selected_index_rows.jsonl",
        "target_free_feature_rows": run_dir / "v6_6_target_free_feature_rows.jsonl",
        "target_free_five_action_rows": run_dir / "v6_6_target_free_five_action_rows.jsonl",
        "v6_2_target_free_predictions": run_dir / "v6_2_target_free_predictions.jsonl",
        "v6_2_outcome_blind_guard_replay": run_dir / "v6_2_outcome_blind_guard_replay.jsonl",
        "policy_selected_target_free_rows": run_dir / "v6_6_policy_selected_target_free_rows.jsonl",
        "point_predictions": run_dir / "v6_6_runtime_pnl_point_predictions.jsonl",
    }
    for name, rows in (
        ("selected_window_rows", selected_rows),
        ("target_free_feature_rows", feature_rows),
        ("target_free_five_action_rows", action_rows),
        ("v6_2_target_free_predictions", v6_2_predictions),
        ("v6_2_outcome_blind_guard_replay", replay),
        ("policy_selected_target_free_rows", accepted_rows),
        ("point_predictions", point_predictions),
    ):
        _write_jsonl(paths[name], rows)
    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-freeze-v1",
        "run_id": config.run_id,
        "decision_freeze_created_ts": config.stage_started_ts,
        "selected_market_count": len(selected_rows),
        "selected_market_ids": [str(row["market_id"]) for row in selected_rows],
        "attempted_index_row_count": len(attempted_rows),
        "target_free_support": support,
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "decision_freeze_written_before_target_access": True,
        "all_selected_markets_closed_before_freeze": True,
        "manual_approval_scope": "offline_point_freeze_and_fresh_calibration_only",
        "manual_approval_does_not_bypass_statistical_gate": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    decision["decision_freeze_id"] = canonical_json_sha256(decision)
    decision_path = run_dir / "v6_6_fresh_calibration_decision_freeze.json"
    _write_json(decision_path, decision)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-report-v1",
        "run_id": config.run_id,
        "selected_market_count": len(selected_rows),
        "attempted_index_row_count": len(attempted_rows),
        "selected_sequence_start": int(selected_rows[0]["sequence"]),
        "selected_sequence_end": int(selected_rows[-1]["sequence"]),
        "policy_selected_guard_accepted_sbc_count": len(point_predictions),
        "policy_selected_guard_accepted_sbc_count_by_side": support["count_by_side"],
        "target_free_support_gate_passed": support["target_free_support_gate_passed"],
        "target_free_support_blocking_reason_codes": support["blocking_reason_codes"],
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "feature_causality_violation_count": 0,
        "complete_five_action_grid_passed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_6_fresh_calibration_prediction_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _freeze_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "point_freeze_manifest": _descriptor(point_path),
        "v6_2_candidate_manifest": _descriptor(candidate_path),
        "collector_index": _descriptor(index_path),
        "opened_raw_feature_artifacts": raw_lineage,
        **{name: _descriptor(path) for name, path in paths.items()},
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "labels_outcomes_resolution_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_6_fresh_calibration_prediction_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest)


def _settle(
    config: PolicySelectedRuntimePNLV66CalibrationConfig,
    *,
    provider_factory: Callable[[], Any] | None,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    clock_ms_fn: Callable[[], int],
) -> dict[str, Any]:
    from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider

    _verified_common_inputs(config)
    freeze_path = Path(config.prediction_freeze_manifest_path).resolve()
    _verify_pin(
        freeze_path,
        str(config.expected_prediction_freeze_manifest_sha256),
        "#226 prediction freeze",
    )
    freeze = _load_json(freeze_path)
    _validate_prediction_freeze(
        freeze,
        profile_path=Path(config.profile_path).resolve(),
        point_path=Path(config.point_freeze_manifest_path).resolve(),
    )
    selected_descriptor = _verified_descriptor(freeze["selected_window_rows"], "selected rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    if len(selected_rows) != TARGET_MARKET_COUNT:
        raise ValueError("#226 settlement requires exact frozen 60-market window")
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    decision = _load_json(Path(decision_descriptor["path"]))
    if config.stage_started_ts <= int(decision["decision_freeze_created_ts"]):
        raise ValueError("settlement attempted before prediction freeze")
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("settlement attempted before all calibration markets closed")
    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_descriptor = _verified_descriptor(
        freeze["target_free_feature_rows"], "target-free feature rows"
    )
    for row in _load_jsonl(Path(feature_descriptor["path"])):
        frozen_features[str(row["market_id"])].append(row)
    run_dir = _prepare_run_dir(config)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "target_access_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "v6_6_fresh_calibration_settlement_started.json"
    _write_json(marker_path, marker)
    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    pending = list(selected_rows)
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    retried: set[str] = set()
    attempt = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending:
        attempt += 1
        for result in _finalize_selected_rounds(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt,
            evaluation_only_frozen_features_by_market=frozen_features,
        ):
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"]:
                successes[market_id] = result["index_entry"]
                failures.pop(market_id, None)
            else:
                failures[market_id] = result["failure"]
        retry_ids = {
            market_id
            for market_id, failure in failures.items()
            if market_id not in successes and _is_retryable_settlement_failure(failure)
        }
        if not retry_ids:
            break
        retried.update(retry_ids)
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            for market_id in retry_ids:
                failures[market_id]["reason_codes"] = sorted(
                    {*failures[market_id].get("reason_codes", []), "settlement_max_wait_elapsed"}
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [selected_by_market[market_id] for market_id in sorted(retry_ids)]
    entries = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    unresolved = sorted(
        (failure for key, failure in failures.items() if key not in successes),
        key=lambda row: str(row["market_id"]),
    )
    complete = len(entries) == TARGET_MARKET_COUNT and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.stage_started_ts:
        raise ValueError("settlement finalization timestamp precedes access start")
    index_path = run_dir / "v6_6_fresh_calibration_settled_corpus_index.json"
    index_payload = None
    if complete:
        index_payload = {
            "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
            "run_id": config.run_id,
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "prediction_freeze_manifest": _descriptor(freeze_path),
            "decision_freeze_sha256": decision_descriptor["sha256"],
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_or_selection": False,
            "source_outcome_blind_rounds_mutated": False,
            **_blocked_safety_fields(),
        }
        index_payload["settled_corpus_index_id"] = canonical_json_sha256(index_payload)
        _write_json(index_path, index_payload)
    reasons = Counter(
        str(reason) for failure in unresolved for reason in failure.get("reason_codes", [])
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": TARGET_MARKET_COUNT,
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_6_fresh_calibration_settlement_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_6_fresh_calibration_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(run_dir, report_path, report, manifest_path, manifest)
    result["index_path"] = index_path if complete else None
    return result


def _calibrate(config: PolicySelectedRuntimePNLV66CalibrationConfig) -> dict[str, Any]:
    profile_path, point_path = _verified_common_inputs(config)
    runtime_path = Path(config.runtime_policy_profile_path).resolve()
    freeze_path = Path(config.prediction_freeze_manifest_path).resolve()
    index_path = Path(config.settled_corpus_index_path).resolve()
    for path, expected, name in (
        (runtime_path, config.expected_runtime_policy_profile_sha256, "runtime policy profile"),
        (freeze_path, config.expected_prediction_freeze_manifest_sha256, "prediction freeze"),
        (index_path, config.expected_settled_corpus_index_sha256, "settled corpus index"),
    ):
        _verify_pin(path, str(expected), name)
    profile = _load_json(profile_path)
    point_manifest = _validated_point_freeze(point_path, profile=profile)
    runtime_profile = _load_json(runtime_path)
    if config.expected_runtime_policy_profile_sha256 != FROZEN_RUNTIME_POLICY_PROFILE_SHA256:
        raise ValueError("runtime policy profile is not the frozen #226 target contract")
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if runtime_policy_source_hashes() != runtime_profile["runtime_policy_contract"][
        "source_function_sha256"
    ]:
        raise ValueError("runtime policy source hashes drifted before calibration")
    freeze = _load_json(freeze_path)
    _validate_prediction_freeze(
        freeze,
        profile_path=profile_path,
        point_path=point_path,
    )
    index = _load_json(index_path)
    _validate_settled_index(
        index,
        freeze_path=freeze_path,
        evaluation_started_ts=config.stage_started_ts,
    )
    run_dir = _prepare_run_dir(config)
    claim_path = _single_use_claim_path(freeze_path)
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-single-use-claim-v1",
        "run_id": config.run_id,
        "calibration_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "settled_corpus_index": _descriptor(index_path),
        "result_selected_rerun_allowed": False,
        **_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    replay = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_2_outcome_blind_guard_replay"], "guard replay")["path"])
    )
    selected_replay = [
        row
        for row in replay
        if row.get("execution_guard_order_allowed") is True
        and row.get("selected_action_family") == "SELL_BEFORE_CLOSE"
        and row.get("executed_action") in SBC_ACTIONS
    ]
    point_rows = _load_jsonl(
        Path(_verified_descriptor(freeze["point_predictions"], "point predictions")["path"])
    )
    point_by_key = {_decision_action_key(row): row for row in point_rows}
    target_rows = _runtime_targets_for_selected_replay(
        selected_replay,
        selected_point_rows=point_rows,
        settled_entries=index["entries"],
        runtime_profile=runtime_profile,
        run_id=config.run_id,
    )
    target_by_key = {_decision_action_key(row): row for row in target_rows}
    if set(point_by_key) != set(target_by_key):
        raise ValueError("fresh calibration prediction/target identity mismatch")
    joined = []
    for key in sorted(point_by_key, key=lambda value: (value[1], value[0], value[2])):
        point = point_by_key[key]
        target = target_by_key[key]
        updated = {
            **point,
            "runtime_policy_after_cost_net_pnl_per_contract": float(
                target["runtime_policy_after_cost_net_pnl_per_contract"]
            ),
            "position_lifecycle_class": target["position_lifecycle_class"],
            "resolved_outcome": target["resolved_outcome"],
            "target_available_only_post_exit_or_official_resolution": True,
            "target_used_as_decision_time_input": False,
        }
        updated["point_residual"] = float(updated["runtime_expected_net_pnl_point"]) - float(
            updated["runtime_policy_after_cost_net_pnl_per_contract"]
        )
        joined.append(updated)
    calibration = build_v6_6_fresh_calibration_artifact(
        joined,
        train_rows=_load_jsonl(
            Path(_verified_descriptor(point_manifest["selected_train_rows"], "train rows")["path"])
        ),
        profile=profile,
        point_model_descriptor=_verified_descriptor(point_manifest["point_model"], "point model"),
        decision_freeze_descriptor=_verified_descriptor(
            freeze["accepted_bet_decision_freeze"], "decision freeze"
        ),
        settled_index_descriptor=_descriptor(index_path),
        runtime_policy_profile_descriptor=_descriptor(runtime_path),
    )
    target_path = run_dir / "v6_6_fresh_calibration_runtime_targets.jsonl"
    joined_path = run_dir / "v6_6_fresh_calibration_joined_rows.jsonl"
    artifact_path = run_dir / "v6_6_policy_selected_runtime_pnl_calibration.json"
    report_path = run_dir / "v6_6_fresh_calibration_gate_report.json"
    _write_jsonl(target_path, target_rows)
    _write_jsonl(joined_path, joined)
    _write_json(artifact_path, calibration)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-report-v1",
        "run_id": config.run_id,
        "selected_target_count": len(joined),
        "selected_target_count_by_side": calibration["selected_market_count_by_side"],
        "side_calibration": calibration["side_calibration"],
        "error_metrics": calibration["error_metrics"],
        "positive_lcb_selected_market_count_by_side": calibration[
            "positive_lcb_selected_market_count_by_side"
        ],
        "fresh_calibration_gate_checks": calibration["fresh_calibration_gate_checks"],
        "fresh_calibration_gate_passed": calibration["fresh_calibration_gate_passed"],
        "fresh_calibration_gate_blocking_reason_codes": calibration[
            "fresh_calibration_gate_blocking_reason_codes"
        ],
        "candidate_scoring_frozen": calibration["fresh_calibration_gate_passed"],
        "strictly_later_future_side_only_pnl_gate_required": True,
        "manual_approval_does_not_bypass_statistical_gate": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _calibration_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-candidate-freeze-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "point_freeze_manifest": _descriptor(point_path),
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "settled_corpus_index": _descriptor(index_path),
        "runtime_policy_profile": _descriptor(runtime_path),
        "single_use_claim": _descriptor(claim_path),
        "runtime_targets": _descriptor(target_path),
        "joined_calibration_rows": _descriptor(joined_path),
        "fresh_calibration_artifact": _descriptor(artifact_path),
        "gate_report": _descriptor(report_path),
        "gate_report_markdown": _descriptor(report_path.with_suffix(".md")),
        "candidate_scoring_frozen": calibration["fresh_calibration_gate_passed"],
        "fresh_calibration_gate_passed": calibration["fresh_calibration_gate_passed"],
        "strictly_later_future_side_only_pnl_gate_required": True,
        "future_collection_allowed": calibration["fresh_calibration_gate_passed"],
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_6_policy_selected_runtime_pnl_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest)


def select_exact_v6_6_calibration_index_rows(
    index_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    prior_market_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select earliest exact-60 quality rows after the immutable sequence-528 boundary."""

    collection = profile["fresh_calibration_collection"]
    boundary = int(collection["collector_index_boundary_sequence"])
    target = int(collection["target_quality_valid_market_count"])
    scan_cap = int(collection["maximum_attempted_market_count"])
    minimum_start = int(collection["minimum_market_start_ts_exclusive"])
    if target != TARGET_MARKET_COUNT or scan_cap != MAXIMUM_ATTEMPT_COUNT:
        raise ValueError("#226 exact calibration window contract drifted")
    eligible = [row for row in index_rows if int(row["sequence"]) > boundary][:scan_cap]
    if not eligible or int(eligible[0]["sequence"]) != boundary + 1:
        raise ValueError("#226 calibration index does not continue from sequence 528")
    selected: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in eligible:
        attempted.append(row)
        if row.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("#226 calibration index opened targets")
        if int(row.get("raw_resolution_row_count") or 0) != 0:
            raise ValueError("#226 calibration index contains resolution rows")
        if int(row.get("scheduled_round_start_ts") or 0) <= minimum_start:
            raise ValueError("#226 calibration attempt is not strictly later")
        if row.get("capture_quality_valid") is False:
            continue
        if row.get("capture_quality_valid") is not True:
            raise ValueError("#226 calibration quality status is not explicit")
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in prior_market_ids:
            raise ValueError("#226 calibration market identity missing or overlaps train")
        if int(row.get("market_start_ts") or 0) <= minimum_start:
            raise ValueError("#226 calibration market start is not strictly later")
        if market_id in seen:
            raise ValueError("#226 calibration market identity repeated")
        seen.add(market_id)
        selected.append(row)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError("#226 exact 60 quality-valid calibration markets not reached")
    sequences = [int(row["sequence"]) for row in attempted]
    if sequences != list(range(boundary + 1, boundary + 1 + len(sequences))):
        raise ValueError("#226 calibration attempted index is not contiguous")
    return selected, attempted


def build_v6_6_policy_selected_target_free_rows(
    replay_rows: list[dict[str, Any]],
    *,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map frozen v6.2 selected/accepted SBC bets into the compact v6.6 inputs."""

    by_key = {_decision_action_key(row): row for row in predictions}
    output = []
    for replay in replay_rows:
        action = str(replay.get("executed_action") or "")
        if (
            replay.get("execution_guard_order_allowed") is not True
            or replay.get("selected_action_family") != "SELL_BEFORE_CLOSE"
            or action not in SBC_ACTIONS
        ):
            continue
        key = (str(replay["market_id"]), int(replay["decision_ts"]), action)
        source = by_key.get(key)
        if source is None:
            raise ValueError("v6.6 selected replay prediction identity missing")
        side = str(replay["selected_side"])
        micro = source["microstructure_snapshot"]
        values = source["decision_time_features"]
        row = {
            "schema_version": f"{SCHEMA_PREFIX}-target-free-row-v1",
            "market_id": key[0],
            "decision_ts": key[1],
            "market_close_ts": int(source["market_close_ts"]),
            "max_input_ts": int(source["max_input_ts"]),
            "side": side,
            "action": action,
            "features": {
                "side_is_up": float(side == "UP"),
                "execution_price": float(values["execution_price"]),
                "current_bid": float(micro["entry_bid"]),
                "spread_bps": float(micro["spread_bps"]),
                "queue_fill_probability_proxy": float(micro["queue_fill_proxy"]),
                "time_to_close_seconds": float(micro["time_to_close_seconds"]),
                "selected_side_probability": float(values["selected_side_probability"]),
                "canonical_v6_2_score": float(replay["decision_score"]),
            },
            "v6_2_replay_row_sha256": str(replay["viability_row_sha256"]),
            "target_fields_used_for_selection": False,
            "target_fields_used_as_model_inputs": False,
            **_blocked_safety_fields(),
        }
        if row["max_input_ts"] > row["decision_ts"]:
            raise ValueError("v6.6 selected row feature causality violation")
        row["target_free_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    output.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    if len({str(row["market_id"]) for row in output}) != len(output):
        raise ValueError("v6.6 selected population contains repeated market")
    return output


def build_v6_6_fresh_calibration_artifact(
    joined_rows: list[dict[str, Any]],
    *,
    train_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    point_model_descriptor: dict[str, Any],
    decision_freeze_descriptor: dict[str, Any],
    settled_index_descriptor: dict[str, Any],
    runtime_policy_profile_descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Build the fixed side-specific residual UCB and its preregistered gate."""

    config = profile["fresh_calibration_gate"]
    count_by_side = Counter(str(row["side"]) for row in joined_rows)
    side_calibration = {}
    for side in SIDES:
        residuals = [
            float(row["point_residual"]) for row in joined_rows if row["side"] == side
        ]
        if not residuals:
            raise ValueError(f"#226 fresh calibration has no {side} support")
        side_calibration[side] = _market_bootstrap_interval(
            residuals,
            resample_count=int(config["bootstrap_resample_count"]),
            confidence_level=float(config["confidence_level"]),
            seed=int(config["bootstrap_seed"]),
        )
    calibrated = []
    for row in joined_rows:
        correction = float(side_calibration[str(row["side"])]["upper_confidence_bound"])
        calibrated.append(
            {
                **row,
                "side_residual_upper_confidence_bound": correction,
                "runtime_expected_net_pnl_lcb": float(row["runtime_expected_net_pnl_point"])
                - correction,
            }
        )
    positive_by_side = {
        side: sum(
            row["side"] == side and float(row["runtime_expected_net_pnl_lcb"]) > 0.0
            for row in calibrated
        )
        for side in SIDES
    }
    targets = np.asarray(
        [float(row["runtime_policy_after_cost_net_pnl_per_contract"]) for row in joined_rows]
    )
    point = np.asarray([float(row["runtime_expected_net_pnl_point"]) for row in joined_rows])
    lcb = np.asarray([float(row["runtime_expected_net_pnl_lcb"]) for row in calibrated])
    train_constant = float(
        np.mean(
            [float(row[profile["model"]["target"]]) for row in train_rows]
        )
    )
    constant = np.full(len(joined_rows), train_constant)
    v6_2 = np.asarray([float(row["features"]["canonical_v6_2_score"]) for row in joined_rows])
    metrics = {
        "point_model": _metrics(targets, point),
        "calibrated_lcb_report_only": _metrics(targets, lcb),
        "train_constant": _metrics(targets, constant),
        "matched_v6_2_canonical_score": _metrics(targets, v6_2),
        "train_constant_value": train_constant,
    }
    minimum = int(config["minimum_positive_lcb_unique_market_count_per_side"])
    checks = {
        "selected_identity_unique_and_within_exact_60_window": len(joined_rows)
        <= TARGET_MARKET_COUNT
        and len({str(row["market_id"]) for row in joined_rows}) == len(joined_rows),
        "side_calibration_support": all(count_by_side[side] >= minimum for side in SIDES),
        "positive_lcb_support": all(positive_by_side[side] >= minimum for side in SIDES),
        "point_mae_improves_train_constant": metrics["point_model"]["mae"]
        < metrics["train_constant"]["mae"],
        "point_mse_improves_train_constant": metrics["point_model"]["mse"]
        < metrics["train_constant"]["mse"],
        "point_mae_improves_matched_v6_2": metrics["point_model"]["mae"]
        < metrics["matched_v6_2_canonical_score"]["mae"],
        "point_mse_improves_matched_v6_2": metrics["point_model"]["mse"]
        < metrics["matched_v6_2_canonical_score"]["mse"],
        "feature_causality": all(int(row["max_input_ts"]) <= int(row["decision_ts"]) for row in joined_rows),
        "no_result_selected_rerun": config["result_selected_rerun_allowed"] is False,
        "no_threshold_search": config["threshold_search_enabled"] is False,
    }
    reasons = [f"{name}_gate_failed" for name, passed in checks.items() if not passed]
    artifact = {
        "schema_version": f"{SCHEMA_PREFIX}-artifact-v1",
        "candidate_name": CANDIDATE_NAME,
        "method": config["method"],
        "point_model": point_model_descriptor,
        "decision_freeze": decision_freeze_descriptor,
        "settled_corpus_index": settled_index_descriptor,
        "runtime_policy_profile": runtime_policy_profile_descriptor,
        "selected_market_count": len(joined_rows),
        "selected_market_count_by_side": {side: count_by_side[side] for side in SIDES},
        "side_calibration": side_calibration,
        "positive_lcb_selected_market_count_by_side": positive_by_side,
        "decision_boundary": config["decision_boundary"],
        "error_metrics": metrics,
        "fresh_calibration_gate_checks": checks,
        "fresh_calibration_gate_passed": not reasons,
        "fresh_calibration_gate_blocking_reason_codes": reasons,
        "fresh_calibration_outcomes_used_for_residual_calibration_only": True,
        "fresh_calibration_outcomes_used_as_model_inputs": False,
        "fresh_calibration_outcomes_used_for_threshold_search": False,
        "result_selected_rerun_allowed": False,
        "future_unseen_side_only_pnl_gate_required": True,
        **_blocked_safety_fields(),
    }
    artifact["calibration_artifact_id"] = canonical_json_sha256(artifact)
    return artifact


def _runtime_targets_for_selected_replay(
    replay: list[dict[str, Any]],
    *,
    selected_point_rows: list[dict[str, Any]],
    settled_entries: list[dict[str, Any]],
    runtime_profile: dict[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    by_market = {str(row["market_id"]): row for row in replay}
    point_by_key = {_decision_action_key(row): row for row in selected_point_rows}
    if len(by_market) != len(replay) or len(point_by_key) != len(selected_point_rows):
        raise ValueError("selected calibration rows contain duplicate market/decision identity")
    entries = {str(row["market_id"]): row for row in settled_entries}
    if not set(by_market).issubset(entries):
        raise ValueError("settled corpus does not cover selected calibration replay")
    output = []
    for market_id in sorted(by_market):
        replay_row = by_market[market_id]
        entry = entries[market_id]
        feature_rows = _load_jsonl(
            Path(_verified_descriptor(entry["feature_rows"], "settled feature rows")["path"])
        )
        label_rows = _load_jsonl(
            Path(_verified_descriptor(entry["label_rows"], "settled label rows")["path"])
        )
        action = str(replay_row["executed_action"])
        side = str(replay_row["selected_side"])
        decision_ts = int(replay_row["decision_ts"])
        point_row = point_by_key.get((market_id, decision_ts, action))
        if point_row is None:
            raise ValueError("selected calibration point row missing for runtime target")
        decision_rows = [
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "max_input_ts": int(point_row["max_input_ts"]),
                "side": side,
                "action": action,
                "features": {
                    "time_to_close_seconds": float(
                        replay_row["microstructure_snapshot"]["time_to_close_seconds"]
                    )
                },
            }
        ]
        source = {
            "market_id": market_id,
            "slug": str(entry.get("run_id") or market_id),
            "role": "fresh_calibration",
        }
        rows, _ = _market_runtime_target_rows(
            source=source,
            feature_rows=feature_rows,
            label_rows=[
                row
                for row in label_rows
                if int(row.get("decision_ts") or -1) == decision_ts
                and str(row.get("action") or "") == action
            ],
            decision_rows=decision_rows,
            profile=runtime_profile,
            run_id=run_id,
        )
        if len(rows) != 1:
            raise ValueError("selected calibration row did not produce exactly one runtime target")
        output.extend(rows)
    return output


def _verified_common_inputs(
    config: PolicySelectedRuntimePNLV66CalibrationConfig,
) -> tuple[Path, Path]:
    profile_path = Path(config.profile_path).resolve()
    point_path = Path(config.point_freeze_manifest_path).resolve()
    if config.expected_profile_sha256 != FROZEN_PROFILE_SHA256:
        raise ValueError("profile is not the frozen #226 profile")
    if config.expected_point_freeze_manifest_sha256 != FROZEN_POINT_FREEZE_MANIFEST_SHA256:
        raise ValueError("point freeze is not the authoritative #226 freeze")
    _verify_pin(profile_path, config.expected_profile_sha256, "#226 profile")
    _verify_pin(
        point_path,
        config.expected_point_freeze_manifest_sha256,
        "#226 point freeze",
    )
    validate_policy_selected_runtime_pnl_v6_6_profile(_load_json(profile_path))
    return profile_path, point_path


def _validated_point_freeze(point_path: Path, *, profile: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json(point_path)
    point_model = _verified_descriptor(manifest["point_model"], "point model")
    selected_rows = _verified_descriptor(manifest["selected_train_rows"], "selected train rows")
    if (
        manifest.get("point_model_frozen") is not True
        or manifest.get("fresh_calibration_collection_allowed") is not True
        or manifest.get("candidate_scoring_frozen") is not False
        or manifest.get("fresh_calibration_outcomes_opened") is not False
        or manifest.get("profile") != _descriptor(Path(manifest["profile"]["path"]))
    ):
        raise ValueError("#226 point freeze is not calibration eligible")
    profile_file = Path(manifest["profile"]["path"])
    if _descriptor(profile_file) != manifest["profile"]:
        raise ValueError("#226 point freeze profile lineage mismatch")
    if manifest["profile"]["sha256"] != FROZEN_PROFILE_SHA256:
        raise ValueError("#226 point freeze profile hash mismatch")
    if point_model["sha256"] != manifest.get("model_sha256"):
        raise ValueError("#226 point model hash mismatch")
    if selected_rows["sha256"] != manifest.get("policy_dataset_hash"):
        raise ValueError("#226 policy dataset hash mismatch")
    return manifest


def _verify_index_boundary(
    index_path: Path,
    *,
    index_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    collection = profile["fresh_calibration_collection"]
    boundary = int(collection["collector_index_boundary_sequence"])
    if len(index_rows) < boundary:
        raise ValueError("collector index is shorter than the frozen boundary")
    boundary_row = index_rows[boundary - 1]
    if int(boundary_row["sequence"]) != boundary:
        raise ValueError("collector boundary sequence mismatch")
    if boundary_row["entry_sha256"] != collection["collector_last_entry_sha256"]:
        raise ValueError("collector boundary last-entry hash mismatch")
    with index_path.open("rb") as handle:
        lines = [handle.readline() for _ in range(boundary)]
    import hashlib

    if hashlib.sha256(b"".join(lines)).hexdigest() != collection["collector_index_boundary_sha256"]:
        raise ValueError("collector append-only prefix hash mismatch")


def _validate_target_free_grid(
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    minimum_market_start_ts_exclusive: int,
) -> None:
    selected_markets = {str(row["market_id"]) for row in selected_rows}
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    feature_keys = {(str(row["market_id"]), int(row["decision_ts"])) for row in feature_rows}
    if any(
        int(row["decision_ts"])
        <= int(selected_by_market[str(row["market_id"])]["market_start_ts"])
        or int(row["decision_ts"])
        >= int(selected_by_market[str(row["market_id"])]["market_end_ts"])
        or int(row["decision_ts"]) <= minimum_market_start_ts_exclusive
        for row in feature_rows
    ):
        raise ValueError("#226 calibration feature decision is outside its frozen market window")
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in action_rows:
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("#226 calibration action feature causality violation")
        selected = selected_by_market[str(row["market_id"])]
        if int(row["market_close_ts"]) != int(selected["market_end_ts"]):
            raise ValueError("#226 calibration action market-close lineage mismatch")
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if {market for market, _ in feature_keys} != selected_markets:
        raise ValueError("#226 calibration feature market coverage mismatch")
    expected = {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    }
    if set(grouped) != feature_keys or any(actions != expected for actions in grouped.values()):
        raise ValueError("#226 calibration five-action grid incomplete")


def _target_free_support(
    rows: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    count = Counter(str(row["side"]) for row in rows)
    minimum = int(
        profile["fresh_calibration_gate"]["minimum_positive_lcb_unique_market_count_per_side"]
    )
    blockers = [
        f"insufficient_{side.lower()}_selected_guard_accepted_sbc_support"
        for side in SIDES
        if count[side] < minimum
    ]
    return {
        "selected_guard_accepted_sbc_count": len(rows),
        "count_by_side": {side: count[side] for side in SIDES},
        "minimum_required_per_side": minimum,
        "target_free_support_gate_passed": not blockers,
        "blocking_reason_codes": blockers,
    }


def _validate_prediction_freeze(
    manifest: dict[str, Any], *, profile_path: Path, point_path: Path
) -> None:
    if (
        manifest.get("schema_version") != f"{SCHEMA_PREFIX}-prediction-freeze-manifest-v1"
        or manifest.get("future_target_access_allowed") is not True
        or manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or manifest.get("resolution_artifact_opened") is not False
        or manifest.get("settlement_provider_called") is not False
    ):
        raise ValueError("#226 prediction freeze is not eligible for target access")
    if manifest.get("profile") != _descriptor(profile_path):
        raise ValueError("#226 prediction freeze profile lineage mismatch")
    if manifest.get("point_freeze_manifest") != _descriptor(point_path):
        raise ValueError("#226 prediction freeze point-model lineage mismatch")
    for field, expected in _blocked_safety_fields().items():
        if manifest.get(field) != expected:
            raise ValueError("#226 prediction freeze safety mismatch")

    profile = _load_json(profile_path)
    decision = _load_json(
        Path(
            _verified_descriptor(
                manifest["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
            )["path"]
        )
    )
    report = _load_json(
        Path(
            _verified_descriptor(
                manifest["report"], "prediction-freeze report"
            )["path"]
        )
    )
    selected_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                manifest["selected_window_rows"], "selected calibration window"
            )["path"]
        )
    )
    point_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                manifest["point_predictions"], "runtime-PnL point predictions"
            )["path"]
        )
    )
    selected_market_ids = [str(row.get("market_id") or "") for row in selected_rows]
    point_market_ids = [str(row.get("market_id") or "") for row in point_rows]
    support = _target_free_support(point_rows, profile=profile)
    nested_safety = _blocked_safety_fields()
    if (
        len(selected_rows) != TARGET_MARKET_COUNT
        or "" in selected_market_ids
        or len(set(selected_market_ids)) != TARGET_MARKET_COUNT
        or "" in point_market_ids
        or len(set(point_market_ids)) != len(point_market_ids)
        or not set(point_market_ids).issubset(selected_market_ids)
        or any(str(row.get("side") or "") not in SIDES for row in point_rows)
        or any(str(row.get("action") or "") not in SBC_ACTIONS for row in point_rows)
    ):
        raise ValueError("#226 prediction freeze selected-market evidence mismatch")
    if (
        decision.get("schema_version") != f"{SCHEMA_PREFIX}-decision-freeze-v1"
        or decision.get("future_target_access_allowed") is not True
        or decision.get("target_free_support") != support
        or decision.get("selected_market_count") != TARGET_MARKET_COUNT
        or decision.get("selected_market_ids") != selected_market_ids
        or decision.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or decision.get("settlement_provider_called") is not False
        or any(decision.get(field) != expected for field, expected in nested_safety.items())
    ):
        raise ValueError("#226 accepted-bet decision freeze target-access mismatch")
    if (
        report.get("schema_version") != f"{SCHEMA_PREFIX}-prediction-freeze-report-v1"
        or report.get("target_free_support_gate_passed") is not True
        or report.get("future_target_access_allowed") is not True
        or report.get("selected_market_count") != TARGET_MARKET_COUNT
        or report.get("policy_selected_guard_accepted_sbc_count") != len(point_rows)
        or report.get("policy_selected_guard_accepted_sbc_count_by_side")
        != support["count_by_side"]
        or report.get("target_free_support_blocking_reason_codes") != []
        or report.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or any(report.get(field) != expected for field, expected in nested_safety.items())
    ):
        raise ValueError("#226 prediction-freeze report target-access mismatch")
    if support["target_free_support_gate_passed"] is not True:
        raise ValueError("#226 target-free support gate did not pass")


def _validate_settled_index(
    index: dict[str, Any], *, freeze_path: Path, evaluation_started_ts: int
) -> None:
    freeze = _load_json(freeze_path)
    selected_descriptor = _verified_descriptor(freeze["selected_window_rows"], "selected rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    selected_markets = {str(row["market_id"]) for row in selected_rows}
    entries = list(index.get("entries") or [])
    entry_markets = {str(row.get("market_id") or "") for row in entries}
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    if (
        index.get("schema_version") != f"{SCHEMA_PREFIX}-settled-corpus-index-v1"
        or int(index.get("entry_count") or 0) != TARGET_MARKET_COUNT
        or index.get("prediction_freeze_manifest") != _descriptor(freeze_path)
        or index.get("decision_freeze_sha256") != decision_descriptor["sha256"]
        or len(entries) != TARGET_MARKET_COUNT
        or entry_markets != selected_markets
        or "" in entry_markets
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
    ):
        raise ValueError("#226 settled corpus index is not calibration eligible")
    for field, expected in _blocked_safety_fields().items():
        if index.get(field) != expected:
            raise ValueError("#226 settled corpus index safety mismatch")


def _single_use_claim_path(freeze_path: Path) -> Path:
    return freeze_path.parent / SINGLE_USE_CLAIM_FILENAME


def _write_single_use_claim(path: Path, claim: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("#226 fresh calibration has already been consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_run_dir(config: PolicySelectedRuntimePNLV66CalibrationConfig) -> Path:
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _decision_action_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["market_id"]), int(row["decision_ts"]), str(row.get("action") or row["executed_action"])


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("calibration metrics require non-empty aligned rows")
    errors = predictions - targets
    if not np.all(np.isfinite(errors)):
        raise ValueError("calibration metrics must be finite")
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "mean_error": float(np.mean(errors)),
    }


def _result(
    run_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "report": report,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def _freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #226 v6.6 fresh calibration prediction freeze",
            "",
            f"- exact selected markets: `{report['selected_market_count']}`",
            f"- accepted SBC rows by side: `{report['policy_selected_guard_accepted_sbc_count_by_side']}`",
            f"- target-free support passed: `{report['target_free_support_gate_passed']}`",
            f"- target access allowed: `{report['future_target_access_allowed']}`",
            "- outcome/settlement/PnL access: `false`",
            "- manual approval bypassed statistical gates: `false`",
        ]
    ) + "\n"


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #226 v6.6 one-shot fresh calibration gate",
            "",
            f"- selected targets: `{report['selected_target_count']}`",
            f"- side support: `{report['selected_target_count_by_side']}`",
            f"- positive LCB support: `{report['positive_lcb_selected_market_count_by_side']}`",
            f"- calibration passed: `{report['fresh_calibration_gate_passed']}`",
            f"- blockers: `{report['fresh_calibration_gate_blocking_reason_codes']}`",
            "- future unseen side-only PnL gate still required: `true`",
            "- paper/live/promotion unlock: `false`",
        ]
    ) + "\n"
