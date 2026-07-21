"""Settle, calibrate, and evaluate the frozen #227 v6.7 windows."""

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

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    FORBIDDEN_TARGET_FIELDS,
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation import (
    build_v6_7_side_only_confirmatory_gate,
    build_v6_7_side_residual_calibration,
    validate_v6_7_evaluation_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    SCHEMA_PREFIX as WINDOW_FREEZE_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    _target_free_support,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _market_runtime_target_rows,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)

SCHEMA_PREFIX = "bigan-v8-p-up-semantic-execution-compatibility-v6-7-post-freeze"
FROZEN_EVALUATION_PROFILE_SHA256 = (
    "900dba0b3d1e280271ff2489e0d0320f1eca150787bf2be30b8b751a3a993c3e"
)
FROZEN_RUNTIME_POLICY_PROFILE_SHA256 = (
    "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
)
ROLE_MARKET_COUNTS = {"fresh_calibration": 60, "future_confirmatory": 120}
STAGES = {"settle", "calibrate", "evaluate_confirmatory"}


@dataclass(frozen=True, slots=True)
class V67PostFreezeConfig:
    """Pinned inputs for one post-decision #227 stage."""

    stage: Literal["settle", "calibrate", "evaluate_confirmatory"]
    role: Literal["fresh_calibration", "future_confirmatory"]
    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    implementation_commit: str
    stage_started_ts: int
    runtime_policy_profile_path: Path | str | None = None
    expected_runtime_policy_profile_sha256: str | None = None
    settled_corpus_index_path: Path | str | None = None
    expected_settled_corpus_index_sha256: str | None = None
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES or self.role not in ROLE_MARKET_COUNTS:
            raise ValueError("#227 post-freeze stage or role is invalid")
        if self.stage == "calibrate" and self.role != "fresh_calibration":
            raise ValueError("#227 calibration requires the fresh-calibration role")
        if self.stage == "evaluate_confirmatory" and self.role != "future_confirmatory":
            raise ValueError("#227 confirmatory evaluation requires the future role")
        if not self.run_id.strip() or self.stage_started_ts <= 0:
            raise ValueError("#227 run_id and stage_started_ts are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_evaluation_profile_sha256",
            "expected_prediction_freeze_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if self.stage in {"calibrate", "evaluate_confirmatory"}:
            for name in (
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#227 post-freeze input missing: {name}")
            for name in (
                "expected_runtime_policy_profile_sha256",
                "expected_settled_corpus_index_sha256",
            ):
                _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "evaluation_profile_path",
            "prediction_freeze_manifest_path",
            "runtime_policy_profile_path",
            "settled_corpus_index_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


def run_v6_7_post_freeze(
    config: V67PostFreezeConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Run one explicit post-freeze stage without changing frozen policy inputs."""

    inputs, profile, freeze = _verified_common_inputs(config)
    if config.stage == "settle":
        return _settle(
            config,
            inputs=inputs,
            profile=profile,
            freeze=freeze,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    runtime_profile, settled_index = _verified_target_inputs(
        config, inputs=inputs, profile=profile, freeze=freeze
    )
    if config.stage == "calibrate":
        return _calibrate(
            config,
            inputs=inputs,
            profile=profile,
            freeze=freeze,
            runtime_profile=runtime_profile,
            settled_index=settled_index,
        )
    return _evaluate_confirmatory(
        config,
        inputs=inputs,
        profile=profile,
        freeze=freeze,
        runtime_profile=runtime_profile,
        settled_index=settled_index,
    )


def _settle(
    config: V67PostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
    provider_factory: Callable[[], Any] | None,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    clock_ms_fn: Callable[[], int],
) -> dict[str, Any]:
    from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider

    selected_rows = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision = _load_json(Path(decision_descriptor["path"]))
    if config.stage_started_ts <= int(decision["decision_freeze_created_ts"]):
        raise ValueError("#227 settlement attempted before the decision freeze")
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("#227 settlement attempted before all selected markets closed")

    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_path = Path(
        _verified_descriptor(freeze["target_free_feature_rows"], "target-free feature rows")["path"]
    )
    for row in _load_jsonl(feature_path):
        frozen_features[str(row["market_id"])].append(row)

    run_dir = _prepare_run_dir(config)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "role": config.role,
        "target_access_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "manual_approval_scope": "offline_v6_7_calibration_and_confirmatory_only",
        "manual_approval_does_not_bypass_any_gate": True,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / f"v6_7_{config.role}_settlement_started.json"
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
        results = _finalize_selected_rounds(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt,
            evaluation_only_frozen_features_by_market=frozen_features,
        )
        for result in results:
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
                        "settlement_max_wait_elapsed",
                    }
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [selected_by_market[market_id] for market_id in sorted(retry_ids)]

    entries = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    unresolved = sorted(
        (failure for key, failure in failures.items() if key not in successes),
        key=lambda row: str(row["market_id"]),
    )
    expected_count = ROLE_MARKET_COUNTS[config.role]
    complete = len(entries) == expected_count and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.stage_started_ts:
        raise ValueError("#227 settlement finalization timestamp precedes access start")
    index_path = run_dir / f"v6_7_{config.role}_settled_corpus_index.json"
    index_payload = None
    if complete:
        index_payload = {
            "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
            "run_id": config.run_id,
            "role": config.role,
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
            "decision_freeze_sha256": decision_descriptor["sha256"],
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_selection_or_tuning": False,
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
        "role": config.role,
        "selected_market_count": expected_count,
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "outcomes_used_for_decision_selection_or_tuning": False,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / f"v6_7_{config.role}_settlement_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "role": config.role,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / f"v6_7_{config.role}_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest, index_path)


def _calibrate(
    config: V67PostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
    runtime_profile: dict[str, Any],
    settled_index: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(config)
    claim_path = _single_use_claim_path(inputs["prediction_freeze"], config.role)
    claim = _claim(config, inputs=inputs)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_7_selected_decisions"], "v6.7 decisions")["path"])
    )
    targets = _runtime_targets_for_decisions(
        decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role=config.role,
    )
    target_by_key = {_decision_action_key(row): row for row in targets}
    joined = []
    for decision in decisions:
        target = target_by_key.get(_decision_action_key(decision))
        if target is None:
            raise ValueError("#227 calibration decision/target identity mismatch")
        joined.append(
            {
                **decision,
                "runtime_policy_after_cost_net_pnl_per_contract": float(
                    target["runtime_policy_after_cost_net_pnl_per_contract"]
                ),
                "runtime_policy_after_cost_net_pnl_at_frozen_size": float(
                    target["runtime_policy_after_cost_net_pnl_at_frozen_size"]
                ),
                "position_lifecycle_class": target["position_lifecycle_class"],
                "resolved_outcome": target["resolved_outcome"],
                "target_available_only_post_exit_or_official_resolution": True,
                "target_used_as_decision_time_input": False,
            }
        )
    artifact, calibrated_rows = build_v6_7_side_residual_calibration(
        joined,
        profile=profile,
        decision_freeze_descriptor=_verified_descriptor(
            freeze["accepted_bet_decision_freeze"], "decision freeze"
        ),
        settled_index_descriptor=_descriptor(inputs["settled_index"]),
        runtime_policy_profile_descriptor=_descriptor(inputs["runtime_profile"]),
    )
    target_path = run_dir / "v6_7_fresh_calibration_runtime_targets.jsonl"
    joined_path = run_dir / "v6_7_fresh_calibration_joined_rows.jsonl"
    calibrated_path = run_dir / "v6_7_fresh_calibration_scored_rows.jsonl"
    artifact_path = run_dir / "v6_7_side_residual_calibration_artifact.json"
    report_path = run_dir / "v6_7_side_residual_calibration_gate_report.json"
    _write_jsonl(target_path, targets)
    _write_jsonl(joined_path, joined)
    _write_jsonl(calibrated_path, calibrated_rows)
    _write_json(artifact_path, artifact)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-gate-report-v1",
        "run_id": config.run_id,
        "selected_market_count": len(joined),
        "selected_market_count_by_side": artifact["selected_market_count_by_side"],
        "side_calibration": artifact["side_calibration"],
        "positive_calibrated_lcb_unique_market_count_by_side": artifact[
            "positive_calibrated_lcb_unique_market_count_by_side"
        ],
        "error_metrics": artifact["error_metrics"],
        "calibration_gate_checks": artifact["calibration_gate_checks"],
        "calibration_gate_passed": artifact["calibration_gate_passed"],
        "calibration_gate_blocking_reason_codes": artifact[
            "calibration_gate_blocking_reason_codes"
        ],
        "candidate_scoring_frozen": artifact["calibration_gate_passed"],
        "manual_approval_does_not_bypass_statistical_gate": True,
        "strictly_later_single_use_confirmatory_required": True,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _calibration_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-calibration-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "runtime_policy_profile": _descriptor(inputs["runtime_profile"]),
        "single_use_claim": _descriptor(claim_path),
        "runtime_targets": _descriptor(target_path),
        "joined_calibration_rows": _descriptor(joined_path),
        "calibrated_rows": _descriptor(calibrated_path),
        "calibration_artifact": _descriptor(artifact_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "calibration_gate_passed": artifact["calibration_gate_passed"],
        "future_confirmatory_freeze_allowed": artifact["calibration_gate_passed"],
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_7_side_residual_calibration_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest)


def _evaluate_confirmatory(
    config: V67PostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
    runtime_profile: dict[str, Any],
    settled_index: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(config)
    claim_path = _single_use_claim_path(inputs["prediction_freeze"], config.role)
    claim = _claim(config, inputs=inputs)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    candidate_decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_7_selected_decisions"], "v6.7 decisions")["path"])
    )
    candidate_targets = _runtime_targets_for_decisions(
        candidate_decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-candidate",
        role=config.role,
    )
    legacy_replay = _load_jsonl(
        Path(_verified_descriptor(freeze["matched_legacy_guard_replay"], "legacy replay")["path"])
    )
    legacy_decisions = _legacy_guard_accepted_sbc_decisions(
        legacy_replay,
        predictions=_load_jsonl(
            Path(
                _verified_descriptor(freeze["v6_2_target_free_predictions"], "v6.2 predictions")[
                    "path"
                ]
            )
        ),
    )
    legacy_targets = _runtime_targets_for_decisions(
        legacy_decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-legacy",
        role=config.role,
    )
    selected_window = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    evaluation_market_ids = [str(row["market_id"]) for row in selected_window]
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    gate = build_v6_7_side_only_confirmatory_gate(
        candidate_targets,
        matched_legacy_rows=legacy_targets,
        evaluation_market_ids=evaluation_market_ids,
        profile=profile,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    candidate_path = run_dir / "v6_7_confirmatory_candidate_runtime_targets.jsonl"
    legacy_path = run_dir / "v6_7_confirmatory_matched_legacy_runtime_targets.jsonl"
    report_path = run_dir / "v6_7_side_only_confirmatory_gate_report.json"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(legacy_path, legacy_targets)
    _write_json(report_path, gate)
    _write_text(report_path.with_suffix(".md"), _confirmatory_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-confirmatory-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "runtime_policy_profile": _descriptor(inputs["runtime_profile"]),
        "single_use_claim": _descriptor(claim_path),
        "candidate_runtime_targets": _descriptor(candidate_path),
        "matched_legacy_runtime_targets": _descriptor(legacy_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "confirmatory_side_only_pnl_gate_passed": gate["confirmatory_side_only_pnl_gate_passed"],
        "confirmatory_side_only_pnl_gate_blocking_reason_codes": gate[
            "confirmatory_side_only_pnl_gate_blocking_reason_codes"
        ],
        "manual_approval_does_not_bypass_side_only_pnl_gate": True,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_7_side_only_confirmatory_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, gate, manifest_path, manifest)


def _verified_common_inputs(
    config: V67PostFreezeConfig,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    inputs = {
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
        "prediction_freeze": Path(config.prediction_freeze_manifest_path).resolve(),
    }
    if config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256:
        raise ValueError("#227 evaluation profile is not the frozen contract")
    _verify_pin(
        inputs["evaluation_profile"],
        config.expected_evaluation_profile_sha256,
        "#227 evaluation profile",
    )
    _verify_pin(
        inputs["prediction_freeze"],
        config.expected_prediction_freeze_manifest_sha256,
        "#227 prediction freeze",
    )
    profile = _load_json(inputs["evaluation_profile"])
    validate_v6_7_evaluation_profile(profile)
    freeze = _load_json(inputs["prediction_freeze"])
    _validate_window_freeze(
        freeze,
        role=config.role,
        profile=profile,
        profile_path=inputs["evaluation_profile"],
    )
    return inputs, profile, freeze


def _verified_target_inputs(
    config: V67PostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_path = Path(config.runtime_policy_profile_path).resolve()
    index_path = Path(config.settled_corpus_index_path).resolve()
    if (
        config.expected_runtime_policy_profile_sha256 != FROZEN_RUNTIME_POLICY_PROFILE_SHA256
        or config.expected_runtime_policy_profile_sha256
        != profile["lineage"]["runtime_policy_profile_sha256"]
    ):
        raise ValueError("#227 runtime target contract is not frozen")
    _verify_pin(
        runtime_path,
        str(config.expected_runtime_policy_profile_sha256),
        "#227 runtime policy profile",
    )
    _verify_pin(
        index_path,
        str(config.expected_settled_corpus_index_sha256),
        "#227 settled corpus index",
    )
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if (
        runtime_policy_source_hashes()
        != runtime_profile["runtime_policy_contract"]["source_function_sha256"]
    ):
        raise ValueError("#227 runtime policy source hashes drifted")
    settled_index = _load_json(index_path)
    _validate_settled_index(
        settled_index,
        freeze=freeze,
        freeze_path=inputs["prediction_freeze"],
        role=config.role,
        evaluation_started_ts=config.stage_started_ts,
    )
    inputs["runtime_profile"] = runtime_path
    inputs["settled_index"] = index_path
    return runtime_profile, settled_index


def _validate_window_freeze(
    manifest: dict[str, Any],
    *,
    role: str,
    profile: dict[str, Any],
    profile_path: Path,
) -> None:
    if (
        manifest.get("schema_version") != f"{WINDOW_FREEZE_SCHEMA_PREFIX}-manifest-v1"
        or manifest.get("role") != role
        or manifest.get("evaluation_profile") != _descriptor(profile_path)
        or manifest.get("future_target_access_allowed") is not True
        or manifest.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or manifest.get("settlement_provider_called") is not False
        or manifest.get("source_score_mutated") is not False
    ):
        raise ValueError("#227 target-free prediction freeze is not eligible")
    for field, expected in _blocked_safety_fields().items():
        if manifest.get(field) != expected:
            raise ValueError(f"#227 prediction-freeze safety mismatch: {field}")
    selected = _load_jsonl(
        Path(_verified_descriptor(manifest["selected_window_rows"], "selected window")["path"])
    )
    decisions = _load_jsonl(
        Path(_verified_descriptor(manifest["v6_7_selected_decisions"], "v6.7 decisions")["path"])
    )
    decision = _load_json(
        Path(
            _verified_descriptor(manifest["accepted_bet_decision_freeze"], "decision freeze")[
                "path"
            ]
        )
    )
    expected_count = ROLE_MARKET_COUNTS[role]
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    support = _target_free_support(
        decisions,
        profile=profile,
        role=role,
        exact_window_market_count=len(selected),
    )
    if (
        len(selected) != expected_count
        or "" in selected_ids
        or len(set(selected_ids)) != expected_count
        or decision.get("role") != role
        or decision.get("selected_window_market_ids") != selected_ids
        or decision.get("target_free_support") != support
        or decision.get("future_target_access_allowed") is not True
        or decision.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or decision.get("settlement_provider_called") is not False
        or support["target_free_support_gate_passed"] is not True
    ):
        raise ValueError("#227 decision-freeze target-access evidence mismatch")
    allowed_markets = set(selected_ids)
    for row in decisions:
        if (
            str(row.get("market_id") or "") not in allowed_markets
            or FORBIDDEN_TARGET_FIELDS.intersection(row)
            or row.get("labels_outcomes_resolution_or_pnl_opened") is not False
            or row.get("source_score_mutated") is not False
        ):
            raise ValueError("#227 selected decision contains invalid target evidence")
    if role == "future_confirmatory":
        calibration_descriptor = _verified_descriptor(
            manifest["calibration_artifact"], "v6.7 calibration artifact"
        )
        calibration = _load_json(Path(calibration_descriptor["path"]))
        if (
            calibration.get("calibration_gate_passed") is not True
            or calibration.get("calibration_gate_blocking_reason_codes") != []
        ):
            raise ValueError("#227 confirmatory freeze lacks eligible calibration")


def _validate_settled_index(
    index: dict[str, Any],
    *,
    freeze: dict[str, Any],
    freeze_path: Path,
    role: str,
    evaluation_started_ts: int,
) -> None:
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    selected_ids = {str(row["market_id"]) for row in selected}
    entries = list(index.get("entries") or [])
    entry_ids = {str(row.get("market_id") or "") for row in entries}
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    if (
        index.get("schema_version") != f"{SCHEMA_PREFIX}-settled-corpus-index-v1"
        or index.get("role") != role
        or index.get("prediction_freeze_manifest") != _descriptor(freeze_path)
        or index.get("decision_freeze_sha256") != decision_descriptor["sha256"]
        or int(index.get("entry_count") or 0) != ROLE_MARKET_COUNTS[role]
        or len(entries) != ROLE_MARKET_COUNTS[role]
        or entry_ids != selected_ids
        or "" in entry_ids
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
    ):
        raise ValueError("#227 settled corpus index is not evaluation eligible")
    for field, expected in _blocked_safety_fields().items():
        if index.get(field) != expected:
            raise ValueError(f"#227 settled-index safety mismatch: {field}")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#227 settled entry violates quarantine contract")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"settled {name}")


def _runtime_targets_for_decisions(
    decisions: list[dict[str, Any]],
    *,
    settled_entries: list[dict[str, Any]],
    runtime_profile: dict[str, Any],
    run_id: str,
    role: str,
) -> list[dict[str, Any]]:
    entries = {str(row["market_id"]): row for row in settled_entries}
    if len(entries) != len(settled_entries):
        raise ValueError("#227 settled entries contain duplicate markets")
    seen: set[str] = set()
    output = []
    for decision in sorted(
        decisions, key=lambda row: (int(row["decision_ts"]), str(row["market_id"]))
    ):
        market_id, decision_ts, action = _decision_action_key(decision)
        side = str(decision.get("side") or decision.get("selected_side") or "")
        if (
            market_id in seen
            or side not in SIDES
            or action not in SBC_ACTIONS
            or market_id not in entries
        ):
            raise ValueError("#227 runtime target decision identity invalid")
        seen.add(market_id)
        max_input_ts = int(decision["max_input_ts"])
        if max_input_ts > decision_ts:
            raise ValueError("#227 runtime target feature causality violation")
        micro = dict(decision.get("microstructure_snapshot") or {})
        time_to_close = float(micro["time_to_close_seconds"])
        entry = entries[market_id]
        feature_rows = _load_jsonl(
            Path(_verified_descriptor(entry["feature_rows"], "settled feature rows")["path"])
        )
        label_rows = _load_jsonl(
            Path(_verified_descriptor(entry["label_rows"], "settled label rows")["path"])
        )
        matching_labels = [
            row
            for row in label_rows
            if int(row.get("decision_ts") or -1) == decision_ts
            and str(row.get("action") or "") == action
        ]
        rows, _ = _market_runtime_target_rows(
            source={
                "market_id": market_id,
                "slug": str(entry.get("run_id") or market_id),
                "role": role,
            },
            feature_rows=feature_rows,
            label_rows=matching_labels,
            decision_rows=[
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "max_input_ts": max_input_ts,
                    "side": side,
                    "action": action,
                    "features": {"time_to_close_seconds": time_to_close},
                }
            ],
            profile=runtime_profile,
            run_id=run_id,
        )
        if len(rows) != 1:
            raise ValueError("#227 selected decision did not produce one runtime target")
        output.extend(rows)
    return output


def _legacy_guard_accepted_sbc_decisions(
    replay_rows: list[dict[str, Any]],
    *,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_by_key = {_decision_action_key(row): row for row in predictions}
    output = []
    seen: set[str] = set()
    for replay in replay_rows:
        action = str(replay.get("executed_action") or "")
        if (
            replay.get("execution_guard_order_allowed") is not True
            or replay.get("selected_action_family") != "SELL_BEFORE_CLOSE"
            or action not in SBC_ACTIONS
        ):
            continue
        key = (str(replay["market_id"]), int(replay["decision_ts"]), action)
        source = prediction_by_key.get(key)
        if source is None:
            raise ValueError("#227 matched legacy prediction identity missing")
        if key[0] in seen:
            raise ValueError("#227 matched legacy replay repeats a market")
        seen.add(key[0])
        output.append(
            {
                "market_id": key[0],
                "decision_ts": key[1],
                "max_input_ts": int(source["max_input_ts"]),
                "side": str(replay["selected_side"]),
                "action": action,
                "microstructure_snapshot": dict(source["microstructure_snapshot"]),
            }
        )
    return sorted(output, key=lambda row: (int(row["decision_ts"]), row["market_id"]))


def _decision_action_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["market_id"]),
        int(row["decision_ts"]),
        str(row.get("action") or row["executed_action"]),
    )


def _single_use_claim_path(freeze_path: Path, role: str) -> Path:
    return freeze_path.parent / f"v6_7_{role}_single_use_target_claim.json"


def _claim(config: V67PostFreezeConfig, *, inputs: dict[str, Path]) -> dict[str, Any]:
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-single-use-claim-v1",
        "run_id": config.run_id,
        "stage": config.stage,
        "role": config.role,
        "target_evaluation_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "result_selected_rerun_allowed": False,
        "manual_approval_does_not_bypass_any_gate": True,
        **_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    return claim


def _write_single_use_claim(path: Path, claim: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("#227 frozen window targets have already been consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_run_dir(config: V67PostFreezeConfig) -> Path:
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _result(
    run_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    index_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "index_path": index_path if index_path and index_path.is_file() else None,
        "index_sha256": (_sha256_file(index_path) if index_path and index_path.is_file() else None),
    }


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Read-Only Settlement",
            "",
            f"- role: `{report['role']}`",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            f"- index ready: `{str(report['settled_corpus_index_ready']).lower()}`",
            "- source outcome-blind rounds mutated: `false`",
            "- outcomes used for decision/selection/tuning: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _calibration_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Side Residual Calibration",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- side support: `{report['selected_market_count_by_side']}`",
            f"- positive calibrated support: `{report['positive_calibrated_lcb_unique_market_count_by_side']}`",
            f"- calibration gate passed: `{str(report['calibration_gate_passed']).lower()}`",
            f"- blockers: `{report['calibration_gate_blocking_reason_codes']}`",
            "- manual approval bypassed statistical gates: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _confirmatory_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.7 Side-Only Confirmatory Gate",
            "",
            f"- evaluation markets: `{report['evaluation_market_count']}`",
            f"- accepted markets: `{report['accepted_unique_market_count']}`",
            f"- accepted side metrics: `{report['accepted_side_metrics']}`",
            f"- candidate after-cost PnL: `{report['candidate_after_cost_pnl']}`",
            f"- matched legacy after-cost PnL: `{report['matched_legacy_after_cost_pnl']}`",
            f"- side-only gate passed: `{str(report['confirmatory_side_only_pnl_gate_passed']).lower()}`",
            f"- blockers: `{report['confirmatory_side_only_pnl_gate_blocking_reason_codes']}`",
            "- action/family PnL: `diagnostic_only`",
            "- manual approval bypassed PnL gate: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = ["V67PostFreezeConfig", "run_v6_7_post_freeze"]
