"""Frozen #212 v6.2 future-window prediction, settlement, and PnL gate."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_prediction_freeze import (
    _materialize_future_action_rows,
    _materialize_selected_window_features,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
    _join_frozen_replay_targets,
    _load_and_validate_targets,
)
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
    apply_conformal_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2 import (
    CANDIDATE_NAME,
    apply_market_clustered_mean_ev_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary import (
    TARGET_FIELDS,
    _find_nonempty_fields,
    _prior_market_reference,
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

PROFILE_SCHEMA_VERSION = (
    "bigan-v8-market-clustered-mean-ev-v6-2-future-evaluation-profile-v1"
)
SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-future"
EXPECTED_ACTIONS = frozenset(
    {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    }
)
SIDES = ("UP", "DOWN")
SINGLE_USE_CLAIM_FILENAME = "v6_2_future_side_only_gate_single_use_claim.json"


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62FutureFreezeConfig:
    """Pinned target-free inputs for the exact earliest-200 freeze."""

    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    collection_profile_path: Path | str
    expected_collection_profile_sha256: str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    cumulative_canary_manifest_path: Path | str
    expected_cumulative_canary_manifest_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    builder_git_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common_config(self)
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("decision_freeze_created_ts must be positive")


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62FutureSettlementConfig:
    """Pinned post-freeze official settlement on quarantine copies."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    builder_git_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common_config(self)
        if self.target_access_started_ts <= 0:
            raise ValueError("target_access_started_ts must be positive")
        if self.provider_timeout_seconds <= 0 or self.provider_http_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.settlement_max_wait_seconds < 0:
            raise ValueError("settlement_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0 or self.max_workers <= 0:
            raise ValueError("settlement polling and worker settings must be positive")


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62FutureGateConfig:
    """Pinned single-use post-settlement side-only PnL gate."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    settled_corpus_index_path: Path | str
    expected_settled_corpus_index_sha256: str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    single_use_claim_path: Path | str
    builder_git_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common_config(self)
        object.__setattr__(self, "single_use_claim_path", Path(self.single_use_claim_path))
        if self.evaluation_started_ts <= 0:
            raise ValueError("evaluation_started_ts must be positive")


def validate_market_clustered_mean_ev_v6_2_future_profile(
    profile: dict[str, Any],
) -> None:
    """Validate the frozen #212 outcome-access and PnL-gate contract."""

    window = dict(profile.get("window") or {})
    gates = dict(profile.get("support_and_pnl_gates") or {})
    access = dict(profile.get("access_sequence") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "frozen": profile.get("frozen") is True,
        "collection_profile": _is_sha256(profile.get("collection_profile_sha256")),
        "candidate": _is_sha256(profile.get("candidate_manifest_sha256")),
        "model": _is_sha256(profile.get("candidate_model_sha256")),
        "calibration": _is_sha256(profile.get("candidate_calibration_sha256")),
        "baseline": _is_sha256(profile.get("matched_v5_manifest_sha256")),
        "feature_contract": _is_sha256(profile.get("feature_contract_sha256")),
        "first_sequence": int(window.get("first_eligible_index_sequence") or 0) == 313,
        "exact_200": int(window.get("quality_valid_market_count") or 0) == 200,
        "scan_240": int(window.get("maximum_index_scan_count") or 0) == 240,
        "selection": window.get("selection_rule")
        == "chronological_earliest_quality_valid_strictly_later_disjoint",
        "single_use": window.get("single_use_holdout") is True,
        "no_extension": window.get("result_dependent_extension_allowed") is False,
        "support": int(gates.get("minimum_guard_accepted_unique_market_count") or 0)
        == 120,
        "side_support": int(gates.get("minimum_supported_side_market_count") or 0) == 17,
        "sides": list(gates.get("required_supported_sides") or []) == ["UP", "DOWN"],
        "side_only": gates.get("pnl_hard_gate_aggregation")
        == "selected_side_buy_up_buy_down_only",
        "action_diagnostic": gates.get("action_and_action_family_pnl_diagnostic_only")
        is True,
        "market_bootstrap": gates.get("bootstrap_unit") == "market_id",
        "target_free_first": access.get("target_free_prediction_and_full_guard_freeze_first")
        is True,
        "quarantine_second": access.get(
            "official_read_only_settlement_on_quarantine_copies_second"
        )
        is True,
        "gate_last": access.get("single_side_only_pnl_gate_last") is True,
        "no_tuning": access.get(
            "outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning"
        )
        is False,
        "no_rerun": access.get("future_result_driven_rerun_allowed") is False,
        "safety": dict(profile.get("safety") or {}) == _expected_safety(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6.2 future evaluation profile invalid:" + ",".join(failed))


def freeze_market_clustered_mean_ev_v6_2_future_predictions(
    config: MarketClusteredMeanEVV62FutureFreezeConfig,
) -> dict[str, Any]:
    """Freeze exact-earliest-200 v6.2 and matched-v5 decisions before targets."""

    profile_path = Path(config.evaluation_profile_path).resolve()
    collection_profile_path = Path(config.collection_profile_path).resolve()
    candidate_path = Path(config.candidate_manifest_path).resolve()
    cumulative_path = Path(config.cumulative_canary_manifest_path).resolve()
    index_path = Path(config.collector_index_path).resolve()
    _verify_all_pins(
        (
            (profile_path, config.expected_evaluation_profile_sha256, "evaluation profile"),
            (
                collection_profile_path,
                config.expected_collection_profile_sha256,
                "collection profile",
            ),
            (candidate_path, config.expected_candidate_manifest_sha256, "candidate manifest"),
            (
                cumulative_path,
                config.expected_cumulative_canary_manifest_sha256,
                "cumulative canary manifest",
            ),
            (index_path, config.expected_collector_index_sha256, "collector index"),
        )
    )
    profile = _load_json(profile_path)
    validate_market_clustered_mean_ev_v6_2_future_profile(profile)
    if profile["collection_profile_sha256"] != config.expected_collection_profile_sha256:
        raise ValueError("collection profile hash does not match evaluation profile")
    _validate_collection_profile(
        _load_json(collection_profile_path),
        candidate_sha256=config.expected_candidate_manifest_sha256,
    )
    if profile["candidate_manifest_sha256"] != (
        config.expected_candidate_manifest_sha256
    ):
        raise ValueError("candidate manifest hash does not match evaluation profile")
    candidate = _load_json(candidate_path)
    _validate_candidate(candidate, profile=profile)
    cumulative = _load_json(cumulative_path)
    cumulative_report = _validate_complete_cumulative_canary(
        cumulative,
        candidate_sha256=config.expected_candidate_manifest_sha256,
    )
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    selected_rows, attempted_rows = _select_exact_future_index_rows(
        index_rows,
        profile=profile,
        candidate=candidate,
    )
    pre_audit = _load_json(
        Path(_verified_descriptor(candidate["pre_target_access_audit"], "candidate audit")["path"])
    )
    feature_contract_descriptor = _verified_descriptor(
        pre_audit["feature_contract"], "feature contract"
    )
    if feature_contract_descriptor["sha256"] != profile["feature_contract_sha256"]:
        raise ValueError("candidate feature contract does not match evaluation profile")
    feature_contract = _load_json(Path(feature_contract_descriptor["path"]))
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    selected_market_ids = [str(row["market_id"]) for row in selected_rows]
    feature_rows, raw_feature_lineage = _materialize_selected_window_features(selected_rows)
    action_rows = _materialize_future_action_rows(
        feature_rows,
        selected_rows=selected_rows,
        feature_columns=feature_columns,
    )
    _validate_exact_feature_action_grid(
        feature_rows,
        action_rows,
        selected_rows=selected_rows,
        candidate=candidate,
    )
    if config.decision_freeze_created_ts <= max(
        int(row["market_end_ts"]) for row in selected_rows
    ):
        raise ValueError("decision freeze attempted before all selected markets closed")
    model_descriptor = _verified_descriptor(candidate["source_model"], "source model")
    calibration_descriptor = _verified_descriptor(
        candidate["market_clustered_mean_risk_calibration"], "v6.2 calibration"
    )
    if model_descriptor["sha256"] != profile["candidate_model_sha256"]:
        raise ValueError("candidate model hash does not match evaluation profile")
    if calibration_descriptor["sha256"] != profile["candidate_calibration_sha256"]:
        raise ValueError("candidate calibration hash does not match evaluation profile")
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])
    raw_predictions = _raw_target_stripped_predictions(
        booster,
        action_rows,
        feature_columns=feature_columns,
    )
    candidate_predictions = apply_market_clustered_mean_ev_scores(
        attach_frozen_execution_compatibility(raw_predictions),
        calibration_artifact=_load_json(Path(calibration_descriptor["path"])),
    )
    candidate_replay = _outcome_blind_acceptance_replay(
        candidate_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )

    v5_manifest_descriptor = _verified_descriptor(pre_audit["v5_freeze_manifest"], "v5 manifest")
    if v5_manifest_descriptor["sha256"] != profile["matched_v5_manifest_sha256"]:
        raise ValueError("matched v5 manifest does not match evaluation profile")
    v5_manifest = _load_json(Path(v5_manifest_descriptor["path"]))
    v5_calibration_descriptor = _verified_descriptor(
        v5_manifest["calibration_artifact"], "v5 calibration"
    )
    v5_profile_descriptor = _verified_descriptor(v5_manifest["fit_profile"], "v5 profile")
    baseline_predictions = apply_conformal_scores(
        raw_predictions,
        calibration_artifact=_load_json(Path(v5_calibration_descriptor["path"])),
        profile=_load_json(Path(v5_profile_descriptor["path"])),
    )
    baseline_replay = _outcome_blind_acceptance_replay(
        baseline_predictions,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    support = _target_free_support(candidate_replay, profile=profile)

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    paths = {
        "selected_window_rows": run_dir / "v6_2_future_exact_200_selected_index_rows.jsonl",
        "target_free_feature_rows": run_dir / "v6_2_future_target_free_feature_rows.jsonl",
        "target_free_five_action_rows": run_dir
        / "v6_2_future_target_free_five_action_rows.jsonl",
        "candidate_target_free_predictions": run_dir
        / "v6_2_future_target_free_predictions.jsonl",
        "candidate_outcome_blind_guard_replay": run_dir
        / "v6_2_future_outcome_blind_guard_replay.jsonl",
        "matched_v5_target_free_predictions": run_dir
        / "matched_v5_future_target_free_predictions.jsonl",
        "matched_v5_outcome_blind_guard_replay": run_dir
        / "matched_v5_future_outcome_blind_guard_replay.jsonl",
    }
    for key, rows in (
        ("selected_window_rows", selected_rows),
        ("target_free_feature_rows", feature_rows),
        ("target_free_five_action_rows", action_rows),
        ("candidate_target_free_predictions", candidate_predictions),
        ("candidate_outcome_blind_guard_replay", candidate_replay),
        ("matched_v5_target_free_predictions", baseline_predictions),
        ("matched_v5_outcome_blind_guard_replay", baseline_replay),
    ):
        _write_jsonl(paths[key], rows)
    freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-freeze-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "candidate_name": CANDIDATE_NAME,
        "candidate_manifest": _descriptor(candidate_path),
        "matched_v5_manifest": v5_manifest_descriptor,
        "evaluation_profile": _descriptor(profile_path),
        "collection_profile": _descriptor(collection_profile_path),
        "cumulative_canary_manifest": _descriptor(cumulative_path),
        "collector_index": _descriptor(index_path),
        "collector_index_attempted_row_count": len(attempted_rows),
        "selected_market_count": len(selected_rows),
        "selected_market_ids": selected_market_ids,
        "selected_market_ids_sha256": canonical_json_sha256(selected_market_ids),
        "selected_sequence_start": int(selected_rows[0]["sequence"]),
        "selected_sequence_end": int(selected_rows[-1]["sequence"]),
        "target_free_support": support,
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "decision_freeze_written_before_target_access": True,
        "all_selected_markets_closed_before_freeze": True,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    freeze["decision_freeze_id"] = canonical_json_sha256(freeze)
    freeze_path = run_dir / "v6_2_future_accepted_bet_decision_freeze.json"
    _write_json(freeze_path, freeze)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "selected_market_count": len(selected_rows),
        "attempted_index_row_count": len(attempted_rows),
        "selected_sequence_start": int(selected_rows[0]["sequence"]),
        "selected_sequence_end": int(selected_rows[-1]["sequence"]),
        "complete_five_action_grid_passed": True,
        "feature_causality_violation_count": 0,
        "candidate_guard_accepted_unique_market_count": support[
            "guard_accepted_unique_market_count"
        ],
        "candidate_guard_accepted_unique_market_count_by_side": support[
            "guard_accepted_unique_market_count_by_side"
        ],
        "matched_v5_guard_accepted_unique_market_count": len(
            {
                str(row["market_id"])
                for row in baseline_replay
                if row["execution_guard_order_allowed"]
            }
        ),
        "target_free_support_gate_passed": support["target_free_support_gate_passed"],
        "target_free_support_blocking_reason_codes": support[
            "target_free_support_blocking_reason_codes"
        ],
        "future_target_access_allowed": freeze["future_target_access_allowed"],
        "future_strictly_later_disjoint_and_exact_window_passed": True,
        "cumulative_canary_collection_complete": cumulative_report[
            "future_holdout_collection_complete"
        ],
        "labels_outcomes_or_pnl_opened": False,
        "promotion_evidence": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_2_future_prediction_freeze_report.json"
    report_md_path = run_dir / "v6_2_future_prediction_freeze_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _prediction_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "evaluation_profile": _descriptor(profile_path),
        "collection_profile": _descriptor(collection_profile_path),
        "candidate_manifest": _descriptor(candidate_path),
        "matched_v5_manifest": v5_manifest_descriptor,
        "cumulative_canary_manifest": _descriptor(cumulative_path),
        "collector_index": _descriptor(index_path),
        "cumulative_batch_report_lineage": [
            _verified_descriptor(value, "batch report")
            for value in cumulative["batch_reports"]
        ],
        "opened_raw_feature_artifacts": raw_feature_lineage,
        **{name: _descriptor(path) for name, path in paths.items()},
        "accepted_bet_decision_freeze": _descriptor(freeze_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "decision_freeze_written_before_target_access": True,
        "future_target_access_allowed": freeze["future_target_access_allowed"],
        "labels_outcomes_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_future_prediction_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def build_market_clustered_mean_ev_v6_2_future_settled_corpus(
    config: MarketClusteredMeanEVV62FutureSettlementConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Copy and settle the frozen exact window without mutating source rounds."""

    from bigan.v8.polymarket.recorder import (
        PolymarketPublicHTTPRealCorpusProvider,
    )

    freeze_path = Path(config.prediction_freeze_manifest_path).resolve()
    _verify_pin(
        freeze_path,
        config.expected_prediction_freeze_manifest_sha256,
        "v6.2 prediction freeze manifest",
    )
    freeze_manifest = _load_json(freeze_path)
    _validate_freeze_manifest_for_target_access(freeze_manifest)
    decision_descriptor = _verified_descriptor(
        freeze_manifest["accepted_bet_decision_freeze"], "decision freeze"
    )
    decision = _load_json(Path(decision_descriptor["path"]))
    selected_descriptor = _verified_descriptor(
        freeze_manifest["selected_window_rows"], "selected rows"
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    if len(selected_rows) != 200:
        raise ValueError("settlement requires exact frozen 200-market window")
    max_close = max(int(row["market_end_ts"]) for row in selected_rows)
    if config.target_access_started_ts <= int(decision["decision_freeze_created_ts"]):
        raise ValueError("target access attempted before decision freeze")
    if config.target_access_started_ts <= max_close:
        raise ValueError("target access attempted before all markets closed")
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "target_access_started_ts": config.target_access_started_ts,
        "decision_freeze": decision_descriptor,
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "v6_2_future_settlement_started.json"
    _write_json(marker_path, marker)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    selected_by_market = {str(row["market_id"]): row for row in selected_rows}
    pending_rows = list(selected_rows)
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    retried: set[str] = set()
    attempt = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending_rows:
        attempt += 1
        for result in _finalize_selected_rounds(
            pending_rows,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt,
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
                    {
                        *failures[market_id].get("reason_codes", []),
                        "settlement_resolution_max_wait_elapsed",
                    }
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending_rows = [selected_by_market[market_id] for market_id in sorted(retry_ids)]
    entries = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    unresolved = sorted(
        (failure for market_id, failure in failures.items() if market_id not in successes),
        key=lambda row: str(row["market_id"]),
    )
    complete = len(entries) == len(selected_rows) and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.target_access_started_ts:
        raise ValueError("settlement finalization timestamp precedes target access")
    index_path = run_dir / "v6_2_future_settled_corpus_index.json"
    index_payload: dict[str, Any] | None = None
    if complete:
        index_payload = {
            "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
            "run_id": config.run_id,
            "builder_git_commit": config.builder_git_commit,
            "target_access_started_ts": config.target_access_started_ts,
            "index_finalized_ts": finalized_ts,
            "decision_freeze_sha256": decision_descriptor["sha256"],
            "prediction_freeze_manifest": _descriptor(freeze_path),
            "selected_window_rows": selected_descriptor,
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_or_selection": False,
            "outcomes_used_for_threshold_or_model_tuning": False,
            "source_outcome_blind_rounds_mutated": False,
            **_blocked_safety_fields(),
        }
        index_payload["settled_corpus_index_id"] = canonical_json_sha256(index_payload)
        _write_json(index_path, index_payload)
    reasons = Counter(
        str(reason)
        for failure in unresolved
        for reason in failure.get("reason_codes", [])
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": len(selected_rows),
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "blocking_reason_codes": [] if complete else ["settled_corpus_window_incomplete"],
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_2_future_settlement_report.json"
    report_md_path = run_dir / "v6_2_future_settlement_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "selected_window_rows": selected_descriptor,
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "settled_corpus_index_ready": complete,
        "future_results_used_for_tuning": False,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(run_dir, report, report_path, manifest, manifest_path)
    result.update(
        {
            "index": index_payload,
            "index_path": index_path if complete else None,
            "index_sha256": _sha256(index_path) if complete else None,
        }
    )
    return result


def run_market_clustered_mean_ev_v6_2_future_gate(
    config: MarketClusteredMeanEVV62FutureGateConfig,
) -> dict[str, Any]:
    """Consume the holdout once and evaluate the frozen side-only PnL gate."""

    freeze_path = Path(config.prediction_freeze_manifest_path).resolve()
    index_path = Path(config.settled_corpus_index_path).resolve()
    profile_path = Path(config.evaluation_profile_path).resolve()
    _verify_all_pins(
        (
            (
                freeze_path,
                config.expected_prediction_freeze_manifest_sha256,
                "prediction freeze manifest",
            ),
            (index_path, config.expected_settled_corpus_index_sha256, "settled index"),
            (profile_path, config.expected_evaluation_profile_sha256, "evaluation profile"),
        )
    )
    profile = _load_json(profile_path)
    validate_market_clustered_mean_ev_v6_2_future_profile(profile)
    freeze_manifest = _load_json(freeze_path)
    _validate_freeze_manifest_for_target_access(freeze_manifest)
    if freeze_manifest["evaluation_profile"] != _descriptor(profile_path):
        raise ValueError("freeze/evaluation profile lineage mismatch")
    claim_path = Path(config.single_use_claim_path).resolve()
    expected_claim_path = _bound_single_use_claim_path(freeze_path)
    if claim_path != expected_claim_path:
        raise ValueError("single-use claim path is not bound to prediction freeze")
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-single-use-gate-claim-v1",
        "run_id": config.run_id,
        "evaluation_started_ts": config.evaluation_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "settled_corpus_index": _descriptor(index_path),
        "evaluation_profile": _descriptor(profile_path),
        "future_result_driven_rerun_allowed": False,
        **_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    _claim_single_use(claim_path, claim)

    decision_descriptor = _verified_descriptor(
        freeze_manifest["accepted_bet_decision_freeze"], "decision freeze"
    )
    selected_descriptor = _verified_descriptor(
        freeze_manifest["selected_window_rows"], "selected rows"
    )
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    feature_descriptor = _verified_descriptor(
        freeze_manifest["target_free_feature_rows"], "frozen feature rows"
    )
    feature_rows = _load_jsonl(Path(feature_descriptor["path"]))
    candidate_replay_descriptor = _verified_descriptor(
        freeze_manifest["candidate_outcome_blind_guard_replay"], "candidate replay"
    )
    baseline_replay_descriptor = _verified_descriptor(
        freeze_manifest["matched_v5_outcome_blind_guard_replay"], "matched v5 replay"
    )
    candidate_replay = _load_jsonl(Path(candidate_replay_descriptor["path"]))
    baseline_replay = _load_jsonl(Path(baseline_replay_descriptor["path"]))
    settled_index = _load_json(index_path)
    entries = _validate_v6_2_settled_index(
        settled_index,
        decision_freeze_sha256=decision_descriptor["sha256"],
        selected_rows=selected_rows,
        evaluation_started_ts=config.evaluation_started_ts,
    )
    targets, target_sources = _load_and_validate_targets(
        entries,
        selected_rows=selected_rows,
        frozen_features=feature_rows,
    )
    targets_by_decision = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in targets
    }
    candidate_eval = _join_frozen_replay_targets(
        candidate_replay,
        targets_by_decision=targets_by_decision,
        policy_name=CANDIDATE_NAME,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    baseline_eval = _join_frozen_replay_targets(
        baseline_replay,
        targets_by_decision=targets_by_decision,
        policy_name="matched_v5_individual_outcome_conformal",
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    gate = build_market_clustered_mean_ev_v6_2_side_only_gate(
        candidate_eval,
        matched_v5_rows=baseline_eval,
        evaluation_market_ids=[str(row["market_id"]) for row in selected_rows],
        profile=profile,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    target_path = run_dir / "v6_2_future_post_freeze_targets.jsonl"
    candidate_path = run_dir / "v6_2_future_settled_evaluation_rows.jsonl"
    baseline_path = run_dir / "matched_v5_future_settled_evaluation_rows.jsonl"
    gate_path = run_dir / "v6_2_future_side_only_pnl_gate_report.json"
    gate_md_path = run_dir / "v6_2_future_side_only_pnl_gate_report.md"
    _write_jsonl(target_path, targets)
    _write_jsonl(candidate_path, candidate_eval)
    _write_jsonl(baseline_path, baseline_eval)
    _write_json(gate_path, gate)
    _write_text(gate_md_path, _gate_markdown(gate))
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "evaluation_market_count": len(selected_rows),
        "target_row_count": len(targets),
        "future_gate_passed": gate["future_gate_passed"],
        "future_gate_blocking_reason_codes": gate["future_gate_blocking_reason_codes"],
        "candidate_post_cost_net_pnl": gate["candidate_post_cost_net_pnl"],
        "matched_v5_post_cost_net_pnl": gate["matched_v5_post_cost_net_pnl"],
        "candidate_minus_matched_v5_post_cost_net_pnl": gate[
            "candidate_minus_matched_v5_post_cost_net_pnl"
        ],
        "side_only_gate_executed_exactly_once": True,
        "future_results_used_for_tuning": False,
        "promotion_requires_separate_manual_review": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_2_future_evaluation_report.json"
    report_md_path = run_dir / "v6_2_future_evaluation_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _evaluation_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "single_use_claim": _descriptor(claim_path),
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "settled_corpus_index": _descriptor(index_path),
        "evaluation_profile": _descriptor(profile_path),
        "settled_target_sources": target_sources,
        "post_freeze_targets": _descriptor(target_path),
        "candidate_settled_evaluation_rows": _descriptor(candidate_path),
        "matched_v5_settled_evaluation_rows": _descriptor(baseline_path),
        "side_only_pnl_gate_report": _descriptor(gate_path),
        "side_only_pnl_gate_report_markdown": _descriptor(gate_md_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "future_results_used_for_tuning": False,
        "future_result_driven_rerun_allowed": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_future_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def build_market_clustered_mean_ev_v6_2_side_only_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    matched_v5_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    profile: dict[str, Any],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Compute the preregistered market-grouped side-only post-cost gate."""

    validate_market_clustered_mean_ev_v6_2_future_profile(profile)
    _require_sha256(decision_freeze_sha256, name="decision_freeze_sha256")
    gates = dict(profile["support_and_pnl_gates"])
    accepted = [row for row in candidate_rows if row["execution_guard_order_allowed"]]
    baseline = [row for row in matched_v5_rows if row["execution_guard_order_allowed"]]
    markets = sorted(set(evaluation_market_ids))
    if len(markets) != 200 or "" in markets:
        raise ValueError("side-only gate requires exact unique 200-market universe")
    candidate_by_market = dict.fromkeys(markets, 0.0)
    baseline_by_market = dict.fromkeys(markets, 0.0)
    for row in accepted:
        candidate_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    for row in baseline:
        baseline_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    if not set(candidate_by_market).issuperset(
        {str(row["market_id"]) for row in accepted + baseline}
    ):
        raise ValueError("evaluation row market outside exact frozen window")
    delta = [candidate_by_market[market] - baseline_by_market[market] for market in markets]
    bootstrap = _market_bootstrap_interval(
        delta,
        resample_count=int(gates["bootstrap_resample_count"]),
        confidence_level=float(gates["bootstrap_confidence_level"]),
        seed=int(gates["bootstrap_seed"]),
    )
    side_rows = {
        side: [row for row in accepted if str(row.get("selected_side") or "") == side]
        for side in SIDES
    }
    side_metrics = {
        side: _accepted_metrics(rows, diagnostic_only=False)
        for side, rows in side_rows.items()
    }
    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        action = str(row["executed_action"])
        action_groups[action].append(row)
        family_groups[_action_family(action)].append(row)
    candidate_pnl = sum(candidate_by_market.values())
    baseline_pnl = sum(baseline_by_market.values())
    largest_winner = max(candidate_by_market.values(), default=0.0)
    accepted_market_count = len({str(row["market_id"]) for row in accepted})
    checks = {
        "minimum_guard_accepted_bet_support": len(accepted)
        >= int(gates["minimum_guard_accepted_bet_count"]),
        "minimum_guard_accepted_unique_market_support": accepted_market_count
        >= int(gates["minimum_guard_accepted_unique_market_count"]),
        "supported_side_post_cost_pnl_gate": all(
            side_metrics[side]["accepted_unique_market_count"]
            >= int(gates["minimum_supported_side_market_count"])
            and side_metrics[side]["accepted_bet_net_pnl_sum"]
            > float(gates["supported_side_post_cost_pnl_minimum_exclusive"])
            for side in SIDES
        ),
        "accepted_bet_total_post_cost_pnl_positive": candidate_pnl
        > float(gates["accepted_bet_total_post_cost_pnl_minimum_exclusive"]),
        "candidate_exceeds_matched_v5": candidate_pnl - baseline_pnl
        > float(gates["candidate_minus_matched_v5_pnl_minimum_exclusive"]),
        "candidate_minus_matched_v5_bootstrap_lcb_positive": bootstrap[
            "lower_confidence_bound"
        ]
        > float(gates["candidate_minus_matched_v5_bootstrap_lcb_minimum_exclusive"]),
        "largest_winner_removed_pnl_positive": candidate_pnl - max(largest_winner, 0.0)
        > float(gates["largest_winner_removed_pnl_minimum_exclusive"]),
        "settlement_causality_provenance_and_runtime_safety": all(
            row.get("settlement_resolved") is True
            and row.get("target_joined_after_decision_freeze") is True
            and row.get("target_used_as_decision_input") is False
            and row.get("forbidden_outcome_field_used_for_decision") is False
            and row.get("feature_causality_violation") is False
            and row.get("provenance_violation") is False
            and row.get("runtime_state_violation") is False
            for row in accepted + baseline
        ),
    }
    reason_map = {
        "minimum_guard_accepted_bet_support": "insufficient_guard_accepted_bet_support",
        "minimum_guard_accepted_unique_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "supported_side_post_cost_pnl_gate": "supported_side_post_cost_pnl_gate_failed",
        "accepted_bet_total_post_cost_pnl_positive": (
            "accepted_bet_total_post_cost_pnl_not_positive"
        ),
        "candidate_exceeds_matched_v5": "candidate_does_not_exceed_matched_v5",
        "candidate_minus_matched_v5_bootstrap_lcb_positive": (
            "candidate_minus_matched_v5_bootstrap_lcb_not_positive"
        ),
        "largest_winner_removed_pnl_positive": "largest_winner_removed_pnl_not_positive",
        "settlement_causality_provenance_and_runtime_safety": (
            "settlement_causality_provenance_or_runtime_safety_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-side-only-pnl-gate-v1",
        "candidate_name": CANDIDATE_NAME,
        "decision_freeze_sha256": decision_freeze_sha256,
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": accepted_market_count,
        "accepted_side_distribution": dict(
            sorted(Counter(str(row["selected_side"]) for row in accepted).items())
        ),
        "accepted_side_metrics": side_metrics,
        "accepted_action_metrics": {
            action: _accepted_metrics(rows, diagnostic_only=True)
            for action, rows in sorted(action_groups.items())
        },
        "accepted_action_family_metrics": {
            family: _accepted_metrics(rows, diagnostic_only=True)
            for family, rows in sorted(family_groups.items())
        },
        "candidate_post_cost_net_pnl": candidate_pnl,
        "matched_v5_post_cost_net_pnl": baseline_pnl,
        "candidate_minus_matched_v5_post_cost_net_pnl": candidate_pnl - baseline_pnl,
        "candidate_minus_matched_v5_market_bootstrap": bootstrap,
        "largest_winning_market_pnl": largest_winner,
        "largest_winner_removed_candidate_pnl": candidate_pnl - max(largest_winner, 0.0),
        "future_gate_checks": checks,
        "future_gate_passed": not blockers,
        "future_gate_blocking_reason_codes": blockers,
        "manual_promotion_review_required": True,
        "future_result_driven_rerun_allowed": False,
        **_blocked_safety_fields(),
    }


def _select_exact_future_index_rows(
    index_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window = dict(profile["window"])
    first = int(window["first_eligible_index_sequence"])
    scan_cap = int(window["maximum_index_scan_count"])
    target = int(window["quality_valid_market_count"])
    freeze_ts = int(window["candidate_freeze_created_ts_exclusive"])
    prior_market_ids, _ = _prior_market_reference(candidate)
    eligible = [row for row in index_rows if int(row["sequence"]) >= first][:scan_cap]
    if not eligible or int(eligible[0]["sequence"]) != first:
        raise ValueError("future index does not begin at the preregistered sequence")
    selected: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    for row in eligible:
        attempted.append(row)
        if row.get("labels_outcomes_or_pnl_opened") is not False:
            raise ValueError("future index row opened target fields")
        if row.get("raw_resolution_row_count") != 0:
            raise ValueError("future index row contains resolution before freeze")
        if int(row.get("market_start_ts") or 0) <= freeze_ts:
            raise ValueError("future index row is not strictly later than candidate freeze")
        if str(row.get("market_id") or "") in prior_market_ids:
            raise ValueError("future index row overlaps candidate lineage")
        if row.get("capture_quality_valid") is True:
            selected.append(row)
            if len(selected) == target:
                break
    if len(selected) != target:
        raise ValueError("exact future quality-valid market target not reached")
    market_ids = [str(row["market_id"]) for row in selected]
    if len(set(market_ids)) != target:
        raise ValueError("exact future window contains duplicate market identity")
    if [int(row["sequence"]) for row in attempted] != sorted(
        int(row["sequence"]) for row in attempted
    ):
        raise ValueError("future index scan is not chronological")
    return selected, attempted


def _validate_exact_feature_action_grid(
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    selected_markets = {str(row["market_id"]) for row in selected_rows}
    if _find_nonempty_fields(action_rows, TARGET_FIELDS):
        raise ValueError("future exact action grid contains target fields")
    feature_keys = {(str(row["market_id"]), int(row["decision_ts"])) for row in feature_rows}
    if len(feature_keys) != len(feature_rows):
        raise ValueError("future exact feature rows contain duplicate decision")
    if {market for market, _ in feature_keys} != selected_markets:
        raise ValueError("future exact feature market set mismatch")
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    freeze_ts = int(candidate["future_collection_minimum_created_ts_exclusive"])
    for row in action_rows:
        if int(row["decision_ts"]) <= freeze_ts:
            raise ValueError("future exact action decision is not strictly later")
        if int(row["max_input_ts"]) > int(row["decision_ts"]):
            raise ValueError("future exact action feature causality violation")
        grouped[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    if set(grouped) != feature_keys or any(actions != EXPECTED_ACTIONS for actions in grouped.values()):
        raise ValueError("future exact five-action grid is incomplete")


def _target_free_support(
    replay: list[dict[str, Any]], *, profile: dict[str, Any]
) -> dict[str, Any]:
    gates = dict(profile["support_and_pnl_gates"])
    accepted = [row for row in replay if row["execution_guard_order_allowed"]]
    markets = {str(row["market_id"]) for row in accepted}
    by_side = {
        side: {
            str(row["market_id"])
            for row in accepted
            if str(row.get("selected_side") or "") == side
        }
        for side in SIDES
    }
    blockers = []
    if len(markets) < int(gates["minimum_guard_accepted_unique_market_count"]):
        blockers.append("insufficient_guard_accepted_unique_market_support")
    for side in SIDES:
        if len(by_side[side]) < int(gates["minimum_supported_side_market_count"]):
            blockers.append(f"insufficient_{side.lower()}_guard_accepted_market_support")
    return {
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(markets),
        "guard_accepted_unique_market_count_by_side": {
            side: len(by_side[side]) for side in SIDES
        },
        "target_free_support_gate_passed": not blockers,
        "target_free_support_blocking_reason_codes": blockers,
    }


def _validate_complete_cumulative_canary(
    manifest: dict[str, Any], *, candidate_sha256: str
) -> dict[str, Any]:
    if manifest.get("future_holdout_collection_complete") is not True:
        raise ValueError("v6.2 cumulative collection is not complete")
    if manifest.get("target_free_terminal_blocked") is not False:
        raise ValueError("v6.2 cumulative canary is terminally blocked")
    if manifest.get("labels_outcomes_or_pnl_opened") is not False:
        raise ValueError("v6.2 cumulative canary opened target fields")
    report = _load_json(Path(_verified_descriptor(manifest["report"], "cumulative report")["path"]))
    if (
        report.get("future_holdout_collection_complete") is not True
        or report.get("quality_valid_market_count", 0) < 200
        or report.get("attempted_market_count", 0) > 240
        or report.get("target_free_terminal_blocked") is not False
    ):
        raise ValueError("v6.2 cumulative report is not eligible for exact freeze")
    for descriptor in manifest.get("batch_reports") or []:
        batch = _load_json(Path(_verified_descriptor(descriptor, "batch report")["path"]))
        if batch.get("candidate_manifest_sha256") != candidate_sha256:
            raise ValueError("cumulative batch candidate hash mismatch")
    return report


def _validate_collection_profile(
    profile: dict[str, Any], *, candidate_sha256: str
) -> None:
    candidate = dict(profile.get("candidate") or {})
    collection = dict(profile.get("collection") or {})
    evaluation = dict(profile.get("future_evaluation") or {})
    checks = {
        "schema": profile.get("schema_version")
        == "bigan-v8-market-clustered-mean-ev-v6-2-future-holdout-profile-v1",
        "candidate": candidate.get("candidate_name") == CANDIDATE_NAME,
        "candidate_hash": candidate.get("candidate_manifest_sha256")
        == candidate_sha256,
        "first_sequence": int(candidate.get("first_eligible_index_sequence") or 0)
        == 313,
        "quality_target": int(collection.get("quality_valid_market_target") or 0)
        == 200,
        "scan_cap": int(collection.get("attempt_scan_cap") or 0) == 240,
        "exact_freeze": int(collection.get("exact_freeze_market_count") or 0)
        == 200,
        "side_only": evaluation.get("side_only_buy_up_buy_down_after_cost_pnl_gate")
        is True,
        "action_diagnostic": evaluation.get("action_and_family_pnl_diagnostic_only")
        is True,
        "single_use": evaluation.get("single_evaluation_allowed") is True,
        "safety": all(
            profile.get("safety", {}).get(field) == expected
            for field, expected in _blocked_safety_fields().items()
            if field != "paper_candidate_allowed"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6.2 future collection profile invalid:" + ",".join(failed))


def _validate_candidate(candidate: dict[str, Any], *, profile: dict[str, Any]) -> None:
    checks = {
        "candidate": candidate.get("candidate_name") == CANDIDATE_NAME,
        "frozen": candidate.get("research_actionability_candidate_frozen") is True,
        "actionability": candidate.get("target_free_actionability_gate_passed") is True,
        "future": candidate.get("new_strictly_later_future_holdout_required") is True,
        "targets": candidate.get(
            "target_free_labels_outcomes_settlement_targets_or_pnl_opened"
        )
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6.2 candidate invalid:" + ",".join(failed))


def _validate_freeze_manifest_for_target_access(manifest: dict[str, Any]) -> None:
    checks = {
        "schema": manifest.get("schema_version")
        == f"{SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "frozen": manifest.get("decision_freeze_written_before_target_access") is True,
        "support": manifest.get("future_target_access_allowed") is True,
        "target_sealed": manifest.get("labels_outcomes_or_pnl_opened") is False,
        "resolution_sealed": manifest.get("resolution_artifact_opened") is False,
        "provider_sealed": manifest.get("settlement_provider_called") is False,
        "safety": all(
            manifest.get(field) == expected
            for field, expected in _blocked_safety_fields().items()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6.2 prediction freeze not eligible:" + ",".join(failed))


def _validate_v6_2_settled_index(
    index: dict[str, Any],
    *,
    decision_freeze_sha256: str,
    selected_rows: list[dict[str, Any]],
    evaluation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    expected_markets = {str(row["market_id"]) for row in selected_rows}
    checks = {
        "schema": index.get("schema_version") == f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
        "freeze": index.get("decision_freeze_sha256") == decision_freeze_sha256,
        "market_set": len(entries) == 200
        and {str(row.get("market_id") or "") for row in entries} == expected_markets,
        "official": all(row.get("official_read_only_resolution") is True for row in entries),
        "post_freeze": all(row.get("corpus_built_after_decision_freeze") is True for row in entries),
        "post_close": all(row.get("settled_after_market_close") is True for row in entries),
        "before_gate": int(index.get("index_finalized_ts") or 0) <= evaluation_started_ts,
        "no_tuning": index.get("outcomes_used_for_decision_or_selection") is False
        and index.get("outcomes_used_for_threshold_or_model_tuning") is False,
        "safety": all(
            index.get(field) == expected
            for field, expected in _blocked_safety_fields().items()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v6.2 settled index invalid:" + ",".join(failed))
    return sorted(entries, key=lambda row: str(row["market_id"]))


def _accepted_metrics(rows: list[dict[str, Any]], *, diagnostic_only: bool) -> dict[str, Any]:
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "accepted_bet_net_pnl_sum": sum(float(row["accepted_bet_net_pnl"]) for row in rows),
        "win_rate": (
            sum(float(row["accepted_bet_net_pnl"]) > 0.0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "diagnostic_only": diagnostic_only,
    }


def _action_family(action: str) -> str:
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    return "NO_TRADE"


def _claim_single_use(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError("future holdout side-only gate already consumed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _bound_single_use_claim_path(prediction_freeze_manifest_path: Path) -> Path:
    return prediction_freeze_manifest_path.resolve().parent / SINGLE_USE_CLAIM_FILENAME


def _validate_common_config(config: Any) -> None:
    if not str(config.run_id).strip():
        raise ValueError("run_id is required")
    _require_git_sha(str(config.builder_git_commit))
    for name in config.__dataclass_fields__:
        if name.startswith("expected_") and name.endswith("_sha256"):
            _require_sha256(str(getattr(config, name)), name=name)
        if name.endswith("_path") and name != "single_use_claim_path":
            object.__setattr__(config, name, Path(getattr(config, name)))
    object.__setattr__(config, "output_dir", Path(config.output_dir))


def _verify_all_pins(values: tuple[tuple[Path, str, str], ...]) -> None:
    for path, expected, name in values:
        _verify_pin(path, expected, name)


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _result(
    run_dir: Path,
    report: dict[str, Any],
    report_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _expected_safety() -> dict[str, Any]:
    return {**_blocked_safety_fields(), "paper_candidate_allowed": False}


def _prediction_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 future prediction freeze",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- guard accepted: `{report['candidate_guard_accepted_unique_market_count']}`",
            f"- support gate passed: `{str(report['target_free_support_gate_passed']).lower()}`",
            "- outcomes opened: `false`",
        ]
    ) + "\n"


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 future settlement",
            "",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            f"- index ready: `{str(report['settled_corpus_index_ready']).lower()}`",
        ]
    ) + "\n"


def _gate_markdown(gate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 future side-only PnL gate",
            "",
            f"- gate passed: `{str(gate['future_gate_passed']).lower()}`",
            f"- candidate PnL: `{gate['candidate_post_cost_net_pnl']}`",
            f"- matched v5 PnL: `{gate['matched_v5_post_cost_net_pnl']}`",
            f"- blockers: `{', '.join(gate['future_gate_blocking_reason_codes']) or 'none'}`",
            "- action/family metrics: diagnostic only",
            "- automatic promotion: `false`",
        ]
    ) + "\n"


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.2 future evaluation",
            "",
            f"- gate passed: `{str(report['future_gate_passed']).lower()}`",
            f"- candidate PnL: `{report['candidate_post_cost_net_pnl']}`",
            f"- candidate minus matched v5: `{report['candidate_minus_matched_v5_post_cost_net_pnl']}`",
            "- paper/live/promotion unlock: `false`",
        ]
    ) + "\n"
