"""Official settlement and single-use evaluation for the challenge window."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256
from bigan.v8.polymarket.challenge_future_freeze import (
    CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION,
)
from bigan.v8.polymarket.parallel_future_gate import (
    PARALLEL_FREEZE_SCHEMA_VERSION,
    evaluate_parallel_future_gate,
)
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    FORBIDDEN_TARGET_FIELDS,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _find_nonempty_fields,
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
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)

POST_FREEZE_PROTOCOL_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-post-freeze-protocol-v1"
)
SETTLED_INDEX_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-settled-index-v1"
)
SETTLEMENT_REPORT_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-settlement-report-v1"
)
SETTLEMENT_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-settlement-manifest-v1"
)
EVALUATION_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-evaluation-manifest-v1"
)
ATTEMPT_CONSUMPTION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-consumption-v1"
)
TRADE_ACTIONS = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
)
SAFETY = {
    "paper_candidate_unlocked": False,
    "promotion_unlocked": False,
    "live_unlocked": False,
    "write_enabled": False,
    "wallet_enabled": False,
    "capital_at_risk": False,
}


class ChallengeFuturePostFreezeError(ValueError):
    """Raised when post-freeze evidence violates the preregistration."""


@dataclass(frozen=True, slots=True)
class ChallengeFutureSettlementConfig:
    """Pinned inputs for official read-only settlement of the frozen window."""

    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    post_freeze_protocol_path: Path | str
    expected_post_freeze_protocol_sha256: str
    implementation_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 21_600.0
    settlement_poll_interval_seconds: float = 30.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_run_identity(
            self.run_id,
            self.implementation_commit,
            self.target_access_started_ts,
        )
        for name in (
            "expected_target_free_freeze_manifest_sha256",
            "expected_post_freeze_protocol_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if (
            self.provider_timeout_seconds <= 0
            or self.provider_http_timeout_seconds <= 0
            or self.settlement_max_wait_seconds < 0
            or self.settlement_poll_interval_seconds <= 0
            or self.max_workers <= 0
        ):
            raise ValueError("settlement timeout and worker values are invalid")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "target_free_freeze_manifest_path",
            Path(self.target_free_freeze_manifest_path),
        )
        object.__setattr__(
            self,
            "post_freeze_protocol_path",
            Path(self.post_freeze_protocol_path),
        )


@dataclass(frozen=True, slots=True)
class ChallengeFutureEvaluationConfig:
    """Pinned inputs for the one parallel future evaluation."""

    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    settled_index_path: Path | str
    expected_settled_index_sha256: str
    post_freeze_protocol_path: Path | str
    expected_post_freeze_protocol_sha256: str
    runtime_policy_profile_path: Path | str
    expected_runtime_policy_profile_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_run_identity(
            self.run_id,
            self.implementation_commit,
            self.evaluation_started_ts,
        )
        for name in (
            "expected_target_free_freeze_manifest_sha256",
            "expected_settled_index_sha256",
            "expected_post_freeze_protocol_sha256",
            "expected_runtime_policy_profile_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "target_free_freeze_manifest_path",
            "settled_index_path",
            "post_freeze_protocol_path",
            "runtime_policy_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_challenge_future_post_freeze_protocol(
    protocol: dict[str, Any],
    *,
    parallel_protocol_sha256: str,
    collection_plan_sha256: str,
    frozen_model_binding_sha256: str,
    runtime_policy_profile_sha256: str,
) -> None:
    """Reject operational drift from the pre-target-access protocol."""

    expected_lineage = {
        "parallel_candidate_protocol_sha256": parallel_protocol_sha256,
        "parallel_future_collection_plan_sha256": collection_plan_sha256,
        "frozen_model_binding_sha256": frozen_model_binding_sha256,
        "runtime_policy_profile_sha256": runtime_policy_profile_sha256,
    }
    expected_settlement = {
        "exact_market_count": 120,
        "official_read_only_resolution_only": True,
        "quarantine_copies_required": True,
        "source_outcome_blind_rounds_mutated": False,
        "frozen_feature_fallback_requires_exact_payload_match": True,
        "target_access_must_follow_decision_freeze_and_market_close": True,
        "single_target_access_claim": True,
        "retry_only_unresolved_or_transient_provider_failures": True,
        "result_dependent_window_extension_allowed": False,
    }
    expected_target_mapping = {
        "actions": [
            "NO_TRADE",
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
        ],
        "no_trade_after_cost_pnl_per_notional": 0.0,
        "trade_value_field": (
            "runtime_policy_after_cost_net_pnl_per_contract"
        ),
        "paper_position_size": 0.2,
        "costs_subtracted_exactly_once": True,
        "same_runtime_profile_for_all_candidates": True,
        "same_settled_entry_for_all_candidates": True,
        "target_available_after_decision_freeze_required": True,
        "target_used_as_decision_input": False,
    }
    expected_evaluation = {
        "single_use": True,
        "attempt_and_alpha_consumed_at_first_target_access": True,
        "parallel_candidates_evaluated_without_mutation": True,
        "statistical_gate_owned_by_parallel_candidate_protocol": True,
        "regime_policy_and_replay_reports_are_downstream_diagnostics": True,
        "result_selected_rerun_allowed": False,
        "threshold_model_cost_sizing_guard_or_candidate_change_allowed": False,
    }
    checks = {
        "schema": (
            protocol.get("schema_version")
            == POST_FREEZE_PROTOCOL_SCHEMA_VERSION
        ),
        "issues": protocol.get("issues") == [254, 258, 256],
        "goal": (
            protocol.get("goal")
            == "challenge_model_promote_to_champion_model"
        ),
        "frozen": protocol.get("frozen_before_target_access") is True,
        "lineage": protocol.get("lineage") == expected_lineage,
        "settlement": protocol.get("settlement") == expected_settlement,
        "target_mapping": (
            protocol.get("target_mapping") == expected_target_mapping
        ),
        "evaluation": protocol.get("evaluation") == expected_evaluation,
        "safety": protocol.get("safety") == SAFETY,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeFuturePostFreezeError(
            "challenge post-freeze protocol invalid: "
            + ",".join(blockers)
        )


def build_challenge_future_settled_index(
    config: ChallengeFutureSettlementConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Settle exact frozen rows on quarantine copies after one access claim."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    protocol_path = config.post_freeze_protocol_path.resolve()
    freeze, parallel_freeze, selected, feature_rows, _ = (
        _validated_challenge_freeze(
            freeze_path,
            expected_sha256=(
                config.expected_target_free_freeze_manifest_sha256
            ),
        )
    )
    protocol = _validated_post_freeze_protocol(
        protocol_path,
        expected_sha256=config.expected_post_freeze_protocol_sha256,
        freeze_manifest=freeze,
    )
    del protocol
    freeze_created_ts = int(freeze["decision_freeze_created_ts"])
    max_market_end_ts = max(int(row["market_end_ts"]) for row in selected)
    if config.target_access_started_ts <= max(
        freeze_created_ts,
        max_market_end_ts,
    ):
        raise ChallengeFuturePostFreezeError(
            "target access attempted before freeze or market close"
        )

    run_dir = _prepare_run_dir(
        Path(config.output_dir),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    claim = {
        "schema_version": (
            "bigan-v8-challenge-future-target-access-claim-v1"
        ),
        "run_id": config.run_id,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "implementation_commit": config.implementation_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
        "post_freeze_protocol": _descriptor(protocol_path),
        "attempt_and_alpha_consumed": True,
        "official_read_only_settlement_on_quarantine_copies": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    claim["claim_id"] = canonical_payload_sha256(
        claim,
        payload_schema_version=str(claim["schema_version"]),
    )
    claim_path = (
        freeze_path.parent
        / "challenge_future_single_use_target_access_claim.json"
    )
    _write_single_use_claim(claim_path, claim)
    marker_path = run_dir / "challenge_settlement_start_marker.json"
    _write_json(marker_path, claim)
    attempt_record = _attempt_consumption_record(
        freeze=freeze,
        parallel_freeze=parallel_freeze,
        claim_path=claim_path,
        target_access_started_ts=config.target_access_started_ts,
    )
    attempt_record_path = (
        run_dir / "challenge_attempt_consumption_record.json"
    )
    _write_json(attempt_record_path, attempt_record)

    frozen_features = _frozen_features_by_market(
        feature_rows,
        selected_market_ids=[str(row["market_id"]) for row in selected],
    )
    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    selected_by_market = {str(row["market_id"]): row for row in selected}
    pending = list(selected)
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    retry_market_ids: set[str] = set()
    attempt_count = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending:
        attempt_count += 1
        results = _finalize_selected_rounds(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt_count,
            evaluation_only_frozen_features_by_market=frozen_features,
        )
        retryable: set[str] = set()
        for result in results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"] is True:
                successes[market_id] = result["index_entry"]
                failures.pop(market_id, None)
            else:
                failure = result["failure"]
                failures[market_id] = failure
                if _is_retryable_settlement_failure(failure):
                    retryable.add(market_id)
        if not retryable:
            break
        retry_market_ids.update(retryable)
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            for market_id in retryable:
                failures[market_id]["reason_codes"] = sorted(
                    {
                        *failures[market_id].get("reason_codes", []),
                        "settlement_resolution_max_wait_elapsed",
                    }
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [
            selected_by_market[market_id]
            for market_id in sorted(retryable)
        ]

    success_rows = sorted(
        successes.values(),
        key=lambda row: str(row["market_id"]),
    )
    failure_rows = sorted(
        failures.values(),
        key=lambda row: str(row["market_id"]),
    )
    complete = len(success_rows) == 120 and not failure_rows
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.target_access_started_ts:
        raise ChallengeFuturePostFreezeError(
            "settlement finalized before target access start"
        )
    index_path = run_dir / "challenge_future_settled_index.json"
    index: dict[str, Any] | None = None
    if complete:
        index = {
            "schema_version": SETTLED_INDEX_SCHEMA_VERSION,
            "run_id": config.run_id,
            "fresh_attempt_id": freeze["fresh_attempt_id"],
            "implementation_commit": config.implementation_commit,
            "target_free_freeze_manifest": _descriptor(freeze_path),
            "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
            "post_freeze_protocol": _descriptor(protocol_path),
            "target_access_claim": _descriptor(claim_path),
            "attempt_consumption_record": _descriptor(
                attempt_record_path
            ),
            "target_access_started_ts": config.target_access_started_ts,
            "index_finalized_ts": finalized_ts,
            "entry_count": len(success_rows),
            "entries": success_rows,
            "official_read_only_resolution": True,
            "source_outcome_blind_rounds_mutated": False,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "future_results_used_for_tuning": False,
            "result_selected_rerun_allowed": False,
            **SAFETY,
        }
        index["settled_index_id"] = canonical_payload_sha256(
            index,
            payload_schema_version=SETTLED_INDEX_SCHEMA_VERSION,
        )
        _write_json(index_path, index)
    reasons = Counter(
        reason
        for row in failure_rows
        for reason in row.get("reason_codes", [])
    )
    report = {
        "schema_version": SETTLEMENT_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "selected_market_count": len(selected),
        "settled_market_count": len(success_rows),
        "unresolved_or_failed_market_count": len(failure_rows),
        "settlement_attempt_count": attempt_count,
        "settlement_retry_market_count": len(retry_market_ids),
        "unresolved_or_failed_reason_distribution": dict(
            sorted(reasons.items())
        ),
        "unresolved_or_failed_markets": failure_rows,
        "settled_index_ready": complete,
        "attempt_and_alpha_consumed": True,
        "target_access_started_ts": config.target_access_started_ts,
        "index_finalized_ts": finalized_ts,
        "source_outcome_blind_rounds_mutated": False,
        "official_read_only_resolution_only": True,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "blocking_reason_codes": (
            [] if complete else ["settled_window_incomplete"]
        ),
        **SAFETY,
    }
    report["report_id"] = canonical_payload_sha256(
        report,
        payload_schema_version=SETTLEMENT_REPORT_SCHEMA_VERSION,
    )
    report_path = run_dir / "challenge_future_settlement_report.json"
    report_md_path = run_dir / "challenge_future_settlement_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _settlement_markdown(report))
    manifest = {
        "schema_version": SETTLEMENT_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "post_freeze_protocol": _descriptor(protocol_path),
        "target_access_claim": _descriptor(claim_path),
        "attempt_consumption_record": _descriptor(
            attempt_record_path
        ),
        "settlement_start_marker": _descriptor(marker_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "settled_index": _descriptor(index_path) if complete else None,
        "settled_index_ready": complete,
        "attempt_and_alpha_consumed": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_payload_sha256(
        manifest,
        payload_schema_version=SETTLEMENT_MANIFEST_SCHEMA_VERSION,
    )
    manifest_path = run_dir / "challenge_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(
        run_dir,
        report,
        report_path,
        manifest,
        manifest_path,
    )
    result.update(
        {
            "index": index,
            "index_path": index_path if complete else None,
            "index_sha256": (
                _sha256_file(index_path) if complete else None
            ),
            "attempt_consumption_record_path": attempt_record_path,
            "attempt_consumption_record_sha256": _sha256_file(
                attempt_record_path
            ),
        }
    )
    return result


def evaluate_challenge_parallel_future_gate(
    config: ChallengeFutureEvaluationConfig,
) -> dict[str, Any]:
    """Consume the settled window exactly once for every frozen candidate."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    settled_path = config.settled_index_path.resolve()
    protocol_path = config.post_freeze_protocol_path.resolve()
    runtime_path = config.runtime_policy_profile_path.resolve()
    freeze, parallel_freeze, selected, _, action_rows = (
        _validated_challenge_freeze(
            freeze_path,
            expected_sha256=(
                config.expected_target_free_freeze_manifest_sha256
            ),
        )
    )
    _validated_post_freeze_protocol(
        protocol_path,
        expected_sha256=config.expected_post_freeze_protocol_sha256,
        freeze_manifest=freeze,
        runtime_profile_path=runtime_path,
        expected_runtime_profile_sha256=(
            config.expected_runtime_policy_profile_sha256
        ),
    )
    _verify_pin(
        settled_path,
        config.expected_settled_index_sha256,
        "challenge settled index",
    )
    _verify_pin(
        runtime_path,
        config.expected_runtime_policy_profile_sha256,
        "challenge runtime policy profile",
    )
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    settled = _load_json(settled_path)
    entries = _validate_settled_index(
        settled,
        freeze_path=freeze_path,
        freeze_sha256=(
            config.expected_target_free_freeze_manifest_sha256
        ),
        parallel_freeze_sha256=parallel_freeze["freeze_sha256"],
        selected_market_ids=[str(row["market_id"]) for row in selected],
        evaluation_started_ts=config.evaluation_started_ts,
    )
    claim = {
        "schema_version": (
            "bigan-v8-challenge-future-parallel-evaluation-claim-v1"
        ),
        "run_id": config.run_id,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "implementation_commit": config.implementation_commit,
        "evaluation_started_ts": config.evaluation_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
        "settled_index": _descriptor(settled_path),
        "post_freeze_protocol": _descriptor(protocol_path),
        "single_use": True,
        "attempt_and_alpha_already_consumed_at_target_access": True,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    claim["claim_id"] = canonical_payload_sha256(
        claim,
        payload_schema_version=str(claim["schema_version"]),
    )
    claim_path = (
        freeze_path.parent
        / "challenge_future_single_use_parallel_evaluation_claim.json"
    )
    _write_single_use_claim(claim_path, claim)
    runtime_decisions_by_action = _runtime_decisions_by_action(
        parallel_freeze=parallel_freeze,
        action_rows=action_rows,
    )
    action_targets: list[dict[str, Any]] = []
    for action in TRADE_ACTIONS:
        decisions = runtime_decisions_by_action[action]
        if decisions:
            action_targets.extend(
                _runtime_targets_for_decisions(
                    decisions,
                    settled_entries=entries,
                    runtime_profile=runtime_profile,
                    run_id=config.run_id,
                    role=f"challenge_parallel_{action.lower()}",
                )
            )
    settled_targets = build_parallel_settled_targets(
        parallel_freeze=parallel_freeze,
        action_runtime_targets=action_targets,
    )
    result = evaluate_parallel_future_gate(
        protocol=_load_json(
            Path(
                _verified_descriptor(
                    freeze["parallel_protocol"],
                    "challenge parallel protocol",
                )["path"]
            )
        ),
        freeze=parallel_freeze,
        settled_targets=settled_targets,
        evaluation_started_ts=config.evaluation_started_ts,
        consumed_freeze_sha256s=set(),
    )
    run_dir = _prepare_run_dir(
        Path(config.output_dir),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    targets_path = run_dir / "challenge_parallel_settled_targets.jsonl"
    action_targets_path = (
        run_dir / "challenge_parallel_action_runtime_targets.jsonl"
    )
    claim_copy_path = run_dir / "challenge_parallel_evaluation_claim.json"
    _write_jsonl(targets_path, settled_targets)
    _write_jsonl(action_targets_path, action_targets)
    _write_json(claim_copy_path, claim)
    candidate_paths: dict[str, Path] = {}
    for candidate_id, rows in result["candidate_rows"].items():
        path = run_dir / f"challenge_{candidate_id}_settled_rows.jsonl"
        _write_jsonl(path, rows)
        candidate_paths[candidate_id] = path
    parallel_claim_path = (
        run_dir / "challenge_parallel_gate_single_use_claim.json"
    )
    report_path = run_dir / "challenge_parallel_evaluation_report.json"
    final_path = run_dir / "challenge_parallel_final_manifest.json"
    report_md_path = run_dir / "challenge_parallel_evaluation_report.md"
    _write_json(parallel_claim_path, result["claim"])
    _write_json(report_path, result["report"])
    _write_json(final_path, result["final_manifest"])
    _write_text(
        report_md_path,
        _evaluation_markdown(result["report"]),
    )
    manifest = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "implementation_commit": config.implementation_commit,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
        "settled_index": _descriptor(settled_path),
        "post_freeze_protocol": _descriptor(protocol_path),
        "runtime_policy_profile": _descriptor(runtime_path),
        "evaluation_claim": _descriptor(claim_path),
        "evaluation_claim_copy": _descriptor(claim_copy_path),
        "parallel_gate_claim": _descriptor(parallel_claim_path),
        "settled_targets": _descriptor(targets_path),
        "action_runtime_targets": _descriptor(action_targets_path),
        "candidate_settled_rows": {
            candidate_id: _descriptor(path)
            for candidate_id, path in candidate_paths.items()
        },
        "parallel_evaluation_report": _descriptor(report_path),
        "parallel_evaluation_report_markdown": _descriptor(
            report_md_path
        ),
        "parallel_final_manifest": _descriptor(final_path),
        "multiplicity_aware_selected_candidate": result["report"][
            "multiplicity_aware_selected_candidate"
        ],
        "attempt_and_alpha_consumed": True,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_payload_sha256(
        manifest,
        payload_schema_version=EVALUATION_MANIFEST_SCHEMA_VERSION,
    )
    manifest_path = run_dir / "challenge_parallel_evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        result["report"],
        report_path,
        manifest,
        manifest_path,
    )


def build_parallel_settled_targets(
    *,
    parallel_freeze: dict[str, Any],
    action_runtime_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map official runtime PnL to the shared freeze decision grid."""

    target_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in action_runtime_targets:
        key = (
            str(row.get("market_id") or ""),
            int(row.get("decision_ts") or 0),
            str(row.get("action") or ""),
        )
        if (
            not key[0]
            or key[1] <= 0
            or key[2] not in TRADE_ACTIONS
            or key in target_by_key
        ):
            raise ChallengeFuturePostFreezeError(
                "runtime action target identity missing or duplicated"
            )
        if (
            row.get("cost_fields_subtracted_exactly_once") is not True
            or row.get("target_used_as_decision_time_input") is not False
        ):
            raise ChallengeFuturePostFreezeError(
                "runtime action target cost or causality contract invalid"
            )
        target_by_key[key] = row
    frozen_actions_by_key: dict[tuple[str, int], set[str]] = {}
    for stream in parallel_freeze["candidate_decision_streams"].values():
        for decision in stream["decisions"]:
            key = (
                str(decision["market_id"]),
                int(decision["decision_ts"]),
            )
            frozen_actions_by_key.setdefault(key, set()).add(
                str(decision["executed_action"])
            )
    output = []
    for source in parallel_freeze["shared_source_rows"]:
        market_id = str(source["market_id"])
        shared_ts = int(source["decision_ts"])
        policy_ts = int(
            source.get("policy_grid_decision_ts") or shared_ts
        )
        pnl_by_action = {"NO_TRADE": 0.0}
        resolved_outcomes: set[str] = set()
        for action in TRADE_ACTIONS:
            target = target_by_key.get((market_id, policy_ts, action))
            if target is not None:
                pnl_by_action[action] = float(
                    target[
                        "runtime_policy_after_cost_net_pnl_per_contract"
                    ]
                )
                if target.get("resolved_outcome"):
                    resolved_outcomes.add(str(target["resolved_outcome"]))
        required_actions = frozen_actions_by_key[(market_id, shared_ts)]
        missing = sorted(required_actions - set(pnl_by_action))
        if missing:
            raise ChallengeFuturePostFreezeError(
                f"settled target missing frozen action for {market_id}: "
                + ",".join(missing)
            )
        if len(resolved_outcomes) > 1:
            raise ChallengeFuturePostFreezeError(
                f"runtime target outcome mismatch for {market_id}"
            )
        output.append(
            {
                "market_id": market_id,
                "decision_ts": shared_ts,
                "policy_grid_decision_ts": policy_ts,
                "after_cost_pnl_per_notional_by_action": pnl_by_action,
                "resolved_outcome": (
                    next(iter(resolved_outcomes))
                    if resolved_outcomes
                    else None
                ),
                "official_read_only_resolution": True,
                "costs_subtracted_exactly_once": True,
                "target_available_after_decision_freeze": True,
                "target_used_as_decision_input": False,
            }
        )
    return output


def _validated_challenge_freeze(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _verify_pin(path, expected_sha256, "challenge target-free freeze")
    manifest = _load_json(path)
    if (
        manifest.get("schema_version")
        != CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION
        or int(manifest.get("exact_market_count") or 0) != 120
        or manifest.get("parallel_target_free_freeze_passed") is not True
        or manifest.get("future_target_access_allowed") is not True
        or manifest.get("decision_freeze_written_before_target_access")
        is not True
        or manifest.get(
            "outcomes_labels_settlement_returns_or_pnl_opened"
        )
        is not False
        or manifest.get("settlement_provider_called") is not False
        or manifest.get("result_selected_extension_or_rerun_allowed")
        is not False
        or any(manifest.get(field) is not expected for field, expected in SAFETY.items())
    ):
        raise ChallengeFuturePostFreezeError(
            "challenge freeze is not target-access eligible"
        )
    parallel_path = Path(
        _verified_descriptor(
            manifest["parallel_target_free_freeze"],
            "challenge parallel freeze",
        )["path"]
    )
    parallel_freeze = _load_json(parallel_path)
    expected_parallel_hash = canonical_payload_sha256(
        {
            key: value
            for key, value in parallel_freeze.items()
            if key != "freeze_sha256"
        },
        payload_schema_version=PARALLEL_FREEZE_SCHEMA_VERSION,
    )
    if (
        parallel_freeze.get("schema_version")
        != PARALLEL_FREEZE_SCHEMA_VERSION
        or parallel_freeze.get("freeze_sha256")
        != expected_parallel_hash
        or parallel_freeze.get("freeze_sha256")
        != manifest.get("parallel_freeze_sha256")
        or parallel_freeze.get(
            "all_candidate_decisions_frozen_before_target_access"
        )
        is not True
        or parallel_freeze.get(
            "outcomes_labels_settlement_returns_or_pnl_opened"
        )
        is not False
    ):
        raise ChallengeFuturePostFreezeError(
            "challenge parallel freeze hash or sealing invalid"
        )
    selected = _load_jsonl(
        Path(
            _verified_descriptor(
                manifest["selected_index_rows"],
                "challenge selected rows",
            )["path"]
        )
    )
    features = _load_jsonl(
        Path(
            _verified_descriptor(
                manifest["feature_rows"],
                "challenge frozen feature rows",
            )["path"]
        )
    )
    actions = _load_jsonl(
        Path(
            _verified_descriptor(
                manifest["action_rows"],
                "challenge frozen action rows",
            )["path"]
        )
    )
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    source_ids = [
        str(row.get("market_id") or "")
        for row in parallel_freeze["shared_source_rows"]
    ]
    if (
        len(selected) != 120
        or "" in selected_ids
        or len(set(selected_ids)) != 120
        or selected_ids != source_ids
        or int(manifest.get("decision_freeze_created_ts") or 0)
        != int(parallel_freeze["decision_freeze_created_ts"])
    ):
        raise ChallengeFuturePostFreezeError(
            "challenge frozen market grid is invalid"
        )
    return manifest, parallel_freeze, selected, features, actions


def _validated_post_freeze_protocol(
    path: Path,
    *,
    expected_sha256: str,
    freeze_manifest: dict[str, Any],
    runtime_profile_path: Path | None = None,
    expected_runtime_profile_sha256: str | None = None,
) -> dict[str, Any]:
    _verify_pin(path, expected_sha256, "challenge post-freeze protocol")
    protocol = _load_json(path)
    fit_manifest = _load_json(
        Path(
            _verified_descriptor(
                freeze_manifest["historical_fit_manifest"],
                "challenge historical fit manifest",
            )["path"]
        )
    )
    runtime_descriptor = _verified_descriptor(
        fit_manifest["runtime_policy_profile"],
        "challenge frozen runtime policy profile",
    )
    if runtime_profile_path is not None:
        _verify_pin(
            runtime_profile_path,
            str(expected_runtime_profile_sha256),
            "challenge runtime policy profile",
        )
        if _descriptor(runtime_profile_path) != runtime_descriptor:
            raise ChallengeFuturePostFreezeError(
                "runtime policy profile differs from the exact model lineage"
            )
    validate_challenge_future_post_freeze_protocol(
        protocol,
        parallel_protocol_sha256=freeze_manifest[
            "parallel_protocol"
        ]["sha256"],
        collection_plan_sha256=freeze_manifest[
            "collection_plan"
        ]["sha256"],
        frozen_model_binding_sha256=freeze_manifest[
            "frozen_model_binding"
        ]["sha256"],
        runtime_policy_profile_sha256=runtime_descriptor["sha256"],
    )
    return protocol


def _frozen_features_by_market(
    feature_rows: list[dict[str, Any]],
    *,
    selected_market_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    forbidden = _find_nonempty_fields(
        feature_rows,
        FORBIDDEN_TARGET_FIELDS,
    )
    selected = set(selected_market_ids)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        market_id = str(row.get("market_id") or "")
        decision_ts = int(row.get("decision_ts") or 0)
        max_input_ts = int(row.get("max_input_ts") or 0)
        if (
            not market_id
            or market_id not in selected
            or decision_ts <= 0
            or max_input_ts > decision_ts
        ):
            raise ChallengeFuturePostFreezeError(
                "frozen feature identity or causality invalid"
            )
        grouped.setdefault(market_id, []).append(row)
    if forbidden:
        raise ChallengeFuturePostFreezeError(
            "frozen features contain target fields: "
            + ",".join(sorted(forbidden))
        )
    if set(grouped) != selected or any(not rows for rows in grouped.values()):
        raise ChallengeFuturePostFreezeError(
            "frozen feature market coverage incomplete"
        )
    return grouped


def _runtime_decisions_by_action(
    *,
    parallel_freeze: dict[str, Any],
    action_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    source_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in action_rows:
        key = (
            str(row.get("market_id") or ""),
            int(row.get("decision_ts") or 0),
            str(row.get("action") or ""),
        )
        source_by_key.setdefault(key, []).append(row)
    output = {action: [] for action in TRADE_ACTIONS}
    for source in parallel_freeze["shared_source_rows"]:
        market_id = str(source["market_id"])
        policy_ts = int(source.get("policy_grid_decision_ts") or 0)
        if policy_ts <= 0:
            continue
        for action in TRADE_ACTIONS:
            matches = source_by_key.get((market_id, policy_ts, action), [])
            if len(matches) != 1:
                raise ChallengeFuturePostFreezeError(
                    f"frozen action source missing or duplicated: "
                    f"{market_id}/{policy_ts}/{action}"
                )
            row = matches[0]
            output[action].append(
                {
                    "market_id": market_id,
                    "decision_ts": policy_ts,
                    "max_input_ts": int(row.get("max_input_ts") or 0),
                    "market_close_ts": int(
                        row.get("market_close_ts") or 0
                    ),
                    "side": (
                        "UP"
                        if action.startswith("BUY_UP_")
                        else "DOWN"
                    ),
                    "action": action,
                    "microstructure_snapshot": dict(
                        row.get("microstructure_snapshot") or {}
                    ),
                }
            )
    return output


def _validate_settled_index(
    index: dict[str, Any],
    *,
    freeze_path: Path,
    freeze_sha256: str,
    parallel_freeze_sha256: str,
    selected_market_ids: list[str],
    evaluation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    entry_ids = [str(row.get("market_id") or "") for row in entries]
    freeze_descriptor = _verified_descriptor(
        index["target_free_freeze_manifest"],
        "challenge settled freeze",
    )
    if (
        index.get("schema_version") != SETTLED_INDEX_SCHEMA_VERSION
        or freeze_descriptor != _descriptor(freeze_path)
        or freeze_descriptor["sha256"] != freeze_sha256.lower()
        or index.get("parallel_freeze_sha256")
        != parallel_freeze_sha256
        or int(index.get("entry_count") or 0) != 120
        or len(entries) != 120
        or set(entry_ids) != set(selected_market_ids)
        or "" in entry_ids
        or evaluation_started_ts
        <= int(index.get("index_finalized_ts") or 0)
        or index.get("official_read_only_resolution") is not True
        or index.get("source_outcome_blind_rounds_mutated") is not False
        or index.get("outcomes_used_for_decision_selection_or_tuning")
        is not False
        or any(index.get(field) is not expected for field, expected in SAFETY.items())
    ):
        raise ChallengeFuturePostFreezeError(
            "challenge settled index is not evaluation eligible"
        )
    _verified_descriptor(
        index["target_access_claim"],
        "challenge target access claim",
    )
    _verified_descriptor(
        index["attempt_consumption_record"],
        "challenge attempt consumption record",
    )
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ChallengeFuturePostFreezeError(
                "settled entry violates quarantine contract"
            )
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"challenge settled {name}")
    return entries


def _attempt_consumption_record(
    *,
    freeze: dict[str, Any],
    parallel_freeze: dict[str, Any],
    claim_path: Path,
    target_access_started_ts: int,
) -> dict[str, Any]:
    record = {
        "schema_version": ATTEMPT_CONSUMPTION_SCHEMA_VERSION,
        "fresh_attempt_id": freeze["fresh_attempt_id"],
        "fresh_attempt_number": 1,
        "familywise_window_alpha": 0.025,
        "parallel_tested_candidate_count": 2,
        "per_candidate_alpha": 0.0125,
        "attempt_consumed": True,
        "alpha_consumed": True,
        "consumes_attempt": True,
        "consumes_alpha": True,
        "evidence_permanently_consumed": True,
        "consumption_event": "single_use_target_access_claim",
        "target_access_started_ts": target_access_started_ts,
        "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
        "target_access_claim": _descriptor(claim_path),
        "result_selected_extension_or_rerun_allowed": False,
    }
    record["record_id"] = canonical_payload_sha256(
        record,
        payload_schema_version=ATTEMPT_CONSUMPTION_SCHEMA_VERSION,
    )
    return record


def _write_single_use_claim(path: Path, claim: dict[str, Any]) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as error:
        raise ChallengeFuturePostFreezeError(
            "challenge frozen future targets have already been consumed"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_run_identity(
    run_id: str,
    implementation_commit: str,
    stage_started_ts: int,
) -> None:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if (
        len(implementation_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in implementation_commit.lower()
        )
    ):
        raise ValueError("implementation_commit must be a Git SHA-1")
    if stage_started_ts <= 0:
        raise ValueError("stage timestamp must be positive")


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Challenge Future Read-Only Settlement",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- settled markets: `{report['settled_market_count']}`",
            "- unresolved/failed markets: "
            f"`{report['unresolved_or_failed_market_count']}`",
            "- attempt and alpha consumed: "
            f"`{str(report['attempt_and_alpha_consumed']).lower()}`",
            "- settled index ready: "
            f"`{str(report['settled_index_ready']).lower()}`",
            f"- blockers: `{report['blocking_reason_codes']}`",
            "- source outcome-blind rounds mutated: `false`",
            "- official read-only resolution only: `true`",
            "- paper/live/write/wallet/capital remain blocked.",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Challenge Parallel Future Evaluation",
        "",
        "- selected candidate: "
        f"`{report['multiplicity_aware_selected_candidate']}`",
    ]
    for candidate_id, gate in report["candidate_gates"].items():
        lines.extend(
            [
                f"- {candidate_id} support: `{gate['accepted_bet_count']}`",
                f"- {candidate_id} total PnL: `{gate['total_after_cost_pnl']}`",
                "- "
                f"{candidate_id} candidate-minus-baseline LCB: "
                f"`{gate['candidate_minus_baseline_bootstrap_lcb']}`",
                f"- {candidate_id} hard gates: "
                f"`{str(gate['all_hard_gates_passed']).lower()}`",
            ]
        )
    lines.extend(
        [
            "- result-selected rerun allowed: `false`",
            "- paper/live/write/wallet/capital remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ATTEMPT_CONSUMPTION_SCHEMA_VERSION",
    "ChallengeFutureEvaluationConfig",
    "ChallengeFuturePostFreezeError",
    "ChallengeFutureSettlementConfig",
    "build_challenge_future_settled_index",
    "build_parallel_settled_targets",
    "evaluate_challenge_parallel_future_gate",
    "validate_challenge_future_post_freeze_protocol",
]
