"""Read-only settlement and single-use confirmatory PnL gate for #229."""

from __future__ import annotations

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
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _legacy_guard_accepted_sbc_decisions,
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    build_regime_emergent_target_free_support,
    build_v6_8_regime_emergent_confirmatory_gate,
    validate_regime_emergent_pnl_v6_8_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_pipeline import (
    CONFIRMATORY_WINDOW_MARKET_COUNT,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_pipeline import (
    SCHEMA_PREFIX as FREEZE_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_post_freeze import (
    FROZEN_EVALUATION_PROFILE_SHA256,
    FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
    _prepare_run_dir,
    _result,
    _write_single_use_claim,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
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
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)

SCHEMA_PREFIX = "bigan-v8-regime-emergent-pnl-v6-8-confirmatory-post-freeze"
STAGES = {"settle", "evaluate_confirmatory"}


@dataclass(frozen=True, slots=True)
class V68ConfirmatoryPostFreezeConfig:
    """Pinned inputs for one future-confirmatory target-access stage."""

    stage: Literal["settle", "evaluate_confirmatory"]
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
        if self.stage not in STAGES or not self.run_id.strip():
            raise ValueError("#229 confirmatory stage and run_id are required")
        if self.stage_started_ts <= 0:
            raise ValueError("#229 confirmatory stage_started_ts must be positive")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_evaluation_profile_sha256",
            "expected_prediction_freeze_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if self.stage == "evaluate_confirmatory":
            for name in (
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#229 confirmatory evaluation input missing: {name}")
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


def run_v6_8_confirmatory_post_freeze(
    config: V68ConfirmatoryPostFreezeConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Settle or evaluate the frozen confirmatory window exactly once."""

    inputs, profile, freeze = _verified_common_inputs(config)
    if config.stage == "settle":
        return _settle(
            config,
            inputs=inputs,
            freeze=freeze,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    runtime_profile, settled_index = _verified_target_inputs(
        config,
        inputs=inputs,
        profile=profile,
        freeze=freeze,
    )
    return _evaluate(
        config,
        inputs=inputs,
        profile=profile,
        freeze=freeze,
        runtime_profile=runtime_profile,
        settled_index=settled_index,
    )


def _settle(
    config: V68ConfirmatoryPostFreezeConfig,
    *,
    inputs: dict[str, Path],
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
        raise ValueError("#229 confirmatory settlement attempted before decision freeze")
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("#229 confirmatory settlement attempted before markets closed")
    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_path = Path(
        _verified_descriptor(freeze["target_free_feature_rows"], "target-free features")["path"]
    )
    for row in _load_jsonl(feature_path):
        frozen_features[str(row["market_id"])].append(row)

    run_dir = _prepare_run_dir(config)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "target_access_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "v6_8_confirmatory_settlement_started.json"
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
    complete = len(entries) == CONFIRMATORY_WINDOW_MARKET_COUNT and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.stage_started_ts:
        raise ValueError("#229 confirmatory settlement finalization precedes access")
    index_path = run_dir / "v6_8_confirmatory_settled_corpus_index.json"
    if complete:
        index_payload = {
            "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
            "run_id": config.run_id,
            "role": "future_confirmatory",
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
            "decision_freeze_sha256": decision_descriptor["sha256"],
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "source_outcome_blind_rounds_mutated": False,
            "side_quota_applied": False,
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
        "role": "future_confirmatory",
        "selected_market_count": CONFIRMATORY_WINDOW_MARKET_COUNT,
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "outcomes_used_for_decision_selection_or_tuning": False,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_8_confirmatory_settlement_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_8_confirmatory_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest, index_path)


def _evaluate(
    config: V68ConfirmatoryPostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
    runtime_profile: dict[str, Any],
    settled_index: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(config)
    claim_path = (
        inputs["prediction_freeze"].parent / "v6_8_confirmatory_single_use_target_claim.json"
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-single-use-claim-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "stage": "evaluate_confirmatory",
        "target_evaluation_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "result_selected_rerun_allowed": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    candidate_decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_8_selected_decisions"], "v6.8 decisions")["path"])
    )
    candidate_targets = _runtime_targets_for_decisions(
        candidate_decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-candidate",
        role="future_confirmatory",
    )
    legacy_replay = _load_jsonl(
        Path(_verified_descriptor(freeze["matched_legacy_guard_replay"], "legacy replay")["path"])
    )
    predictions = _load_jsonl(
        Path(
            _verified_descriptor(freeze["v6_2_target_free_predictions"], "v6.2 predictions")["path"]
        )
    )
    legacy_decisions = _legacy_guard_accepted_sbc_decisions(
        legacy_replay,
        predictions=predictions,
    )
    legacy_targets = _runtime_targets_for_decisions(
        legacy_decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-legacy",
        role="future_confirmatory",
    )
    selected_window = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    evaluation_market_ids = [str(row["market_id"]) for row in selected_window]
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    gate = build_v6_8_regime_emergent_confirmatory_gate(
        candidate_targets,
        matched_legacy_rows=legacy_targets,
        evaluation_market_ids=evaluation_market_ids,
        profile=profile,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    candidate_path = run_dir / "v6_8_confirmatory_candidate_runtime_targets.jsonl"
    legacy_path = run_dir / "v6_8_confirmatory_matched_legacy_runtime_targets.jsonl"
    report_path = run_dir / "v6_8_confirmatory_execution_pnl_gate_report.json"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(legacy_path, legacy_targets)
    _write_json(report_path, gate)
    _write_text(report_path.with_suffix(".md"), _evaluation_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-manifest-v1",
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
        "confirmatory_execution_pnl_gate_passed": gate["confirmatory_execution_pnl_gate_passed"],
        "confirmatory_execution_pnl_gate_blocking_reason_codes": gate[
            "confirmatory_execution_pnl_gate_blocking_reason_codes"
        ],
        "side_count_and_side_pnl_diagnostic_only": True,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_8_confirmatory_execution_pnl_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, gate, manifest_path, manifest)


def _verified_common_inputs(
    config: V68ConfirmatoryPostFreezeConfig,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    inputs = {
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
        "prediction_freeze": Path(config.prediction_freeze_manifest_path).resolve(),
    }
    if config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256:
        raise ValueError("#229 evaluation profile is not frozen")
    _verify_pin(
        inputs["evaluation_profile"],
        config.expected_evaluation_profile_sha256,
        "#229 evaluation profile",
    )
    _verify_pin(
        inputs["prediction_freeze"],
        config.expected_prediction_freeze_manifest_sha256,
        "#229 confirmatory freeze",
    )
    profile = _load_json(inputs["evaluation_profile"])
    validate_regime_emergent_pnl_v6_8_profile(profile)
    freeze = _load_json(inputs["prediction_freeze"])
    _validate_freeze(freeze, profile=profile, profile_path=inputs["evaluation_profile"])
    return inputs, profile, freeze


def _verified_target_inputs(
    config: V68ConfirmatoryPostFreezeConfig,
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
        raise ValueError("#229 runtime target contract is not frozen")
    _verify_pin(
        runtime_path, str(config.expected_runtime_policy_profile_sha256), "#229 runtime profile"
    )
    _verify_pin(index_path, str(config.expected_settled_corpus_index_sha256), "#229 settled index")
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if (
        runtime_policy_source_hashes()
        != runtime_profile["runtime_policy_contract"]["source_function_sha256"]
    ):
        raise ValueError("#229 runtime policy source hashes drifted")
    settled_index = _load_json(index_path)
    _validate_settled_index(
        settled_index,
        freeze=freeze,
        freeze_path=inputs["prediction_freeze"],
        evaluation_started_ts=config.stage_started_ts,
    )
    inputs["runtime_profile"] = runtime_path
    inputs["settled_index"] = index_path
    return runtime_profile, settled_index


def _validate_freeze(
    freeze: dict[str, Any],
    *,
    profile: dict[str, Any],
    profile_path: Path,
) -> None:
    if (
        freeze.get("schema_version") != f"{FREEZE_SCHEMA_PREFIX}-manifest-v1"
        or freeze.get("role") != "future_confirmatory"
        or freeze.get("evaluation_profile") != _descriptor(profile_path)
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("side_count_hard_gate_enabled") is not False
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_score_mutated") is not False
    ):
        raise ValueError("#229 confirmatory target-free freeze is not eligible")
    for field, expected in _blocked_safety_fields().items():
        if freeze.get(field) != expected:
            raise ValueError(f"#229 confirmatory freeze safety mismatch: {field}")
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_8_selected_decisions"], "v6.8 decisions")["path"])
    )
    decision = _load_json(
        Path(
            _verified_descriptor(freeze["accepted_bet_decision_freeze"], "decision freeze")["path"]
        )
    )
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=len(selected),
        expected_window_market_count=CONFIRMATORY_WINDOW_MARKET_COUNT,
        required_total_market_count=int(
            profile["future_confirmatory"]["minimum_guard_accepted_unique_market_count_total"]
        ),
        score_field="v6_8_calibrated_runtime_pnl_lcb",
    )
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    decision_ids = {str(row.get("market_id") or "") for row in decisions}
    if (
        len(selected) != CONFIRMATORY_WINDOW_MARKET_COUNT
        or "" in selected_ids
        or len(set(selected_ids)) != CONFIRMATORY_WINDOW_MARKET_COUNT
        or "" in decision_ids
        or not decision_ids.issubset(set(selected_ids))
        or decision.get("selected_window_market_ids") != selected_ids
        or decision.get("regime_emergent_target_free_support") != support
        or support["target_free_support_gate_passed"] is not True
        or decision.get("future_target_access_allowed") is not True
    ):
        raise ValueError("#229 confirmatory decision-freeze evidence mismatch")
    calibration_selected = _load_jsonl(
        Path(
            _verified_descriptor(
                _load_json(
                    Path(
                        _verified_descriptor(
                            freeze["calibration_adoption_manifest"], "calibration adoption"
                        )["path"]
                    )
                )["selected_window_rows"],
                "calibration selected rows",
            )["path"]
        )
    )
    calibration_ids = {str(row["market_id"]) for row in calibration_selected}
    boundary = max(int(row["market_end_ts"]) for row in calibration_selected)
    if calibration_ids.intersection(selected_ids) or any(
        int(row["scheduled_round_start_ts"]) <= boundary for row in selected
    ):
        raise ValueError("#229 confirmatory window is not strictly later/disjoint")
    calibration = _load_json(
        Path(
            _verified_descriptor(freeze["calibration_artifact"], "pooled calibration artifact")[
                "path"
            ]
        )
    )
    if (
        calibration.get("schema_version") != CALIBRATION_ARTIFACT_SCHEMA_VERSION
        or calibration.get("calibration_gate_passed") is not True
        or calibration.get("calibration_gate_blocking_reason_codes") != []
        or calibration.get("side_count_hard_gate_enabled") is not False
    ):
        raise ValueError("#229 confirmatory freeze lacks eligible pooled calibration")


def _validate_settled_index(
    index: dict[str, Any],
    *,
    freeze: dict[str, Any],
    freeze_path: Path,
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
        or index.get("role") != "future_confirmatory"
        or index.get("prediction_freeze_manifest") != _descriptor(freeze_path)
        or index.get("decision_freeze_sha256") != decision_descriptor["sha256"]
        or int(index.get("entry_count") or 0) != CONFIRMATORY_WINDOW_MARKET_COUNT
        or len(entries) != CONFIRMATORY_WINDOW_MARKET_COUNT
        or entry_ids != selected_ids
        or "" in entry_ids
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
        or index.get("side_quota_applied") is not False
    ):
        raise ValueError("#229 confirmatory settled index is not eligible")
    for field, expected in _blocked_safety_fields().items():
        if index.get(field) != expected:
            raise ValueError(f"#229 confirmatory settled-index safety mismatch: {field}")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#229 confirmatory settled entry violates quarantine")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"settled {name}")


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.8 Confirmatory Read-Only Settlement",
            "",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            f"- index ready: `{str(report['settled_corpus_index_ready']).lower()}`",
            "- source outcome-blind rounds mutated: `false`",
            "- side quota applied: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.8 Confirmatory Execution-PnL Gate",
            "",
            f"- gate passed: `{str(report['confirmatory_execution_pnl_gate_passed']).lower()}`",
            f"- blockers: `{report['confirmatory_execution_pnl_gate_blocking_reason_codes']}`",
            f"- candidate total PnL: `{report['candidate_after_cost_pnl']}`",
            f"- legacy total PnL: `{report['matched_legacy_after_cost_pnl']}`",
            f"- candidate-minus-legacy PnL: `{report['candidate_minus_matched_legacy_after_cost_pnl']}`",
            "- side/action metrics: `diagnostic_only`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "V68ConfirmatoryPostFreezeConfig",
    "run_v6_8_confirmatory_post_freeze",
]
