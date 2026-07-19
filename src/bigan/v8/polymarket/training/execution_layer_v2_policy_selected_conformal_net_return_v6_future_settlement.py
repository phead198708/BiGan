"""Reconcile #207 v6 future targets once and execute the side-only PnL gate."""

from __future__ import annotations

import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
    _join_frozen_replay_targets,
    _load_and_validate_targets,
)
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    CANDIDATE_NAME,
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
    validate_policy_selected_conformal_v6_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future_prediction import (
    SCHEMA_PREFIX as PREDICTION_SCHEMA_PREFIX,
)

SCHEMA_PREFIX = "bigan-v8-policy-selected-conformal-net-return-v6-future-settlement"
SETTLED_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-corpus-index-v1"
MATCHED_BASELINE_NAME = "guard_compatible_direct_net_return_v4"


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6FutureSettlementIndexConfig:
    """Pinned post-freeze official resolution collection for all 300 markets."""

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
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_prediction_freeze_manifest_sha256,
            name="expected_prediction_freeze_manifest_sha256",
        )
        _require_git_sha(self.builder_git_commit)
        if self.target_access_started_ts <= 0:
            raise ValueError("target_access_started_ts must be positive")
        if self.provider_timeout_seconds <= 0 or self.provider_http_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.settlement_max_wait_seconds < 0:
            raise ValueError("settlement_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0 or self.max_workers <= 0:
            raise ValueError("settlement polling and worker values must be positive")
        for field in ("output_dir", "prediction_freeze_manifest_path"):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True, slots=True)
class PolicySelectedConformalV6FutureGateConfig:
    """Pinned one-shot target join and side-only PnL gate inputs."""

    run_id: str
    output_dir: Path | str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    settled_corpus_index_path: Path | str
    expected_settled_corpus_index_sha256: str
    builder_git_commit: str
    reconciliation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field in (
            "expected_prediction_freeze_manifest_sha256",
            "expected_settled_corpus_index_sha256",
        ):
            _require_sha256(str(getattr(self, field)), name=field)
        _require_git_sha(self.builder_git_commit)
        if self.reconciliation_started_ts <= 0:
            raise ValueError("reconciliation_started_ts must be positive")
        for field in (
            "output_dir",
            "prediction_freeze_manifest_path",
            "settled_corpus_index_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def build_policy_selected_conformal_v6_future_settled_corpus_index(
    config: PolicySelectedConformalV6FutureSettlementIndexConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    round_finalizer: Callable[..., list[dict[str, Any]]] = _finalize_selected_rounds,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Finalize quarantine copies only after the immutable supported decision freeze."""

    freeze_path = config.prediction_freeze_manifest_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_prediction_freeze_manifest_sha256,
        "v6 future prediction freeze",
    )
    freeze = _load_json(freeze_path)
    _validate_prediction_freeze_for_target_access(freeze)
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision_freeze = _load_json(Path(decision_descriptor["path"]))
    if decision_freeze.get("future_target_free_support_gate_passed") is not True:
        raise ValueError("future target-free support gate did not pass")
    decision_freeze_ts = int(decision_freeze["decision_freeze_created_ts"])
    if config.target_access_started_ts <= decision_freeze_ts:
        raise ValueError("future target access attempted before decision freeze")
    selected_descriptor = _verified_descriptor(freeze["selected_window_rows"], "future rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    if len(selected_rows) != 300 or len({str(row["market_id"]) for row in selected_rows}) != 300:
        raise ValueError("future settlement requires exact frozen 300-market window")
    if config.target_access_started_ts <= max(int(row["market_end_ts"]) for row in selected_rows):
        raise ValueError("future target access attempted before all markets closed")

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-target-access-marker-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "decision_freeze_created_ts": decision_freeze_ts,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "selected_window_rows": selected_descriptor,
        "target_access_started_after_decision_freeze": True,
        "all_markets_closed_before_target_access": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "direct_training_corpus_exported": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "conformal_v6_future_target_access_started.json"
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
    pending = list(selected_rows)
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    retry_ids: set[str] = set()
    attempt_count = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending:
        attempt_count += 1
        results = round_finalizer(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt_count,
        )
        retryable = set()
        for result in results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"]:
                successes[market_id] = dict(result["index_entry"])
                failures.pop(market_id, None)
            else:
                failure = dict(result["failure"])
                failures[market_id] = failure
                if _is_retryable_settlement_failure(failure):
                    retryable.add(market_id)
        if not retryable:
            break
        retry_ids.update(retryable)
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
        pending = [selected_by_market[market_id] for market_id in sorted(retryable)]

    entries = [successes[market_id] for market_id in sorted(successes)]
    unresolved = [failures[market_id] for market_id in sorted(set(selected_by_market) - set(successes))]
    ready = len(entries) == 300 and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.target_access_started_ts:
        raise ValueError("future settlement index finalized before target access marker")
    index_payload = None
    index_path = run_dir / "conformal_v6_future_settled_corpus_index.json"
    if ready:
        index_payload = {
            "schema_version": SETTLED_INDEX_SCHEMA_VERSION,
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
            "direct_training_corpus_exported": False,
            **_blocked_safety_fields(),
        }
        index_payload["settled_corpus_index_id"] = canonical_json_sha256(index_payload)
        _write_json(index_path, index_payload)
    reason_distribution = Counter(
        str(reason) for row in unresolved for reason in row.get("reason_codes", [])
    )
    blockers = [] if ready else ["future_settled_corpus_window_incomplete"]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-corpus-index-report-v1",
        "report_id": None,
        "run_id": config.run_id,
        "selected_market_count": len(selected_rows),
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "unresolved_or_failed_reason_distribution": dict(sorted(reason_distribution.items())),
        "settlement_attempt_count": attempt_count,
        "settlement_retry_market_count": len(retry_ids),
        "settled_corpus_index_ready": ready,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "direct_training_corpus_exported": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "conformal_v6_future_settled_corpus_index_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-corpus-index-manifest-v1",
        "run_id": config.run_id,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "selected_window_rows": selected_descriptor,
        "target_access_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if ready else None,
        "report": _descriptor(report_path),
        "settled_corpus_index_ready": ready,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "source_outcome_blind_rounds_mutated": False,
        "direct_training_corpus_exported": False,
        "blocking_reason_codes": blockers,
        **_blocked_safety_fields(),
    }
    manifest["settlement_index_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_future_settled_corpus_index_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "index": index_payload,
        "index_path": index_path if ready else None,
        "index_sha256": _sha256_file(index_path) if ready else None,
    }


def reconcile_policy_selected_conformal_v6_future_gate(
    config: PolicySelectedConformalV6FutureGateConfig,
) -> dict[str, Any]:
    """Join official targets after freeze and run the preregistered gate exactly once."""

    freeze_path = config.prediction_freeze_manifest_path.resolve()
    index_path = config.settled_corpus_index_path.resolve()
    _verify_pin(
        freeze_path,
        config.expected_prediction_freeze_manifest_sha256,
        "v6 future prediction freeze",
    )
    _verify_pin(
        index_path,
        config.expected_settled_corpus_index_sha256,
        "v6 future settled corpus index",
    )
    freeze = _load_json(freeze_path)
    _validate_prediction_freeze_for_target_access(freeze)
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision_freeze = _load_json(Path(decision_descriptor["path"]))
    if decision_freeze.get("future_target_free_support_gate_passed") is not True:
        raise ValueError("future target-free support gate did not pass")
    if config.reconciliation_started_ts <= int(decision_freeze["decision_freeze_created_ts"]):
        raise ValueError("future gate reconciliation attempted before decision freeze")
    selected_descriptor = _verified_descriptor(freeze["selected_window_rows"], "future rows")
    selected_rows = _load_jsonl(Path(selected_descriptor["path"]))
    feature_descriptor = _verified_descriptor(
        freeze["target_free_feature_rows"], "target-free features"
    )
    frozen_features = _load_jsonl(Path(feature_descriptor["path"]))
    settled_index = _load_json(index_path)
    entries = _validate_settled_index(
        settled_index,
        expected_decision_freeze_sha256=decision_descriptor["sha256"],
        decision_freeze_created_ts=int(decision_freeze["decision_freeze_created_ts"]),
        selected_rows=selected_rows,
        reconciliation_started_ts=config.reconciliation_started_ts,
    )

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-target-access-marker-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "reconciliation_started_ts": config.reconciliation_started_ts,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "settled_corpus_index": _descriptor(index_path),
        "target_opened_after_decision_freeze": True,
        "future_outcomes_opened_before_decision_freeze": False,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        **_blocked_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "conformal_v6_future_gate_target_access_started.json"
    _write_json(marker_path, marker)
    targets, source_descriptors = _load_and_validate_targets(
        entries,
        selected_rows=selected_rows,
        frozen_features=frozen_features,
    )
    targets = [_v6_target_schema(row) for row in targets]
    target_path = run_dir / "conformal_v6_future_settled_five_action_targets.jsonl"
    _write_jsonl(target_path, targets)
    targets_by_decision = {
        (str(row["market_id"]), int(row["decision_ts"])): row for row in targets
    }
    candidate_replay_descriptor = _verified_descriptor(
        freeze["candidate_outcome_blind_guard_replay"], "candidate guard replay"
    )
    baseline_replay_descriptor = _verified_descriptor(
        freeze["matched_baseline_outcome_blind_guard_replay"], "baseline guard replay"
    )
    candidate_evaluation = _join_frozen_replay_targets(
        _load_jsonl(Path(candidate_replay_descriptor["path"])),
        targets_by_decision=targets_by_decision,
        policy_name=CANDIDATE_NAME,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    baseline_evaluation = _join_frozen_replay_targets(
        _load_jsonl(Path(baseline_replay_descriptor["path"])),
        targets_by_decision=targets_by_decision,
        policy_name=MATCHED_BASELINE_NAME,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    candidate_path = run_dir / "conformal_v6_future_settled_evaluation_rows.jsonl"
    baseline_path = run_dir / "matched_v4_future_settled_evaluation_rows.jsonl"
    _write_jsonl(candidate_path, candidate_evaluation)
    _write_jsonl(baseline_path, baseline_evaluation)
    prereg_descriptor = _verified_descriptor(
        freeze["future_preregistration_manifest"], "future preregistration"
    )
    prereg = _load_json(Path(prereg_descriptor["path"]))
    profile_descriptor = _verified_descriptor(prereg["candidate_profile"], "v6 profile")
    profile = _load_json(Path(profile_descriptor["path"]))
    validate_policy_selected_conformal_v6_profile(profile)
    gate = build_policy_selected_conformal_v6_side_only_future_pnl_gate(
        candidate_evaluation,
        matched_baseline_evaluation_rows=baseline_evaluation,
        evaluation_market_ids=[str(row["market_id"]) for row in selected_rows],
        profile=profile,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    gate.update(
        {
            "run_id": config.run_id,
            "builder_git_commit": config.builder_git_commit,
            "settled_corpus_index": _descriptor(index_path),
            "settled_target_rows": _descriptor(target_path),
            "candidate_evaluation_rows": _descriptor(candidate_path),
            "matched_baseline_evaluation_rows": _descriptor(baseline_path),
            "target_access_marker": _descriptor(marker_path),
            "target_opened_after_decision_freeze": True,
            "future_results_used_for_tuning": False,
            "future_results_used_for_rerun": False,
            "future_results_used_for_automatic_unlock": False,
            **_blocked_safety_fields(),
        }
    )
    gate["report_id"] = canonical_json_sha256(gate)
    gate_path = run_dir / "conformal_v6_future_side_only_pnl_gate_report.json"
    _write_json(gate_path, gate)
    _write_text(gate_path.with_suffix(".md"), _gate_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-gate-manifest-v1",
        "run_id": config.run_id,
        "builder_git_commit": config.builder_git_commit,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "accepted_bet_decision_freeze": decision_descriptor,
        "settled_corpus_index": _descriptor(index_path),
        "source_settled_corpora": source_descriptors,
        "target_access_marker": _descriptor(marker_path),
        "settled_five_action_targets": _descriptor(target_path),
        "candidate_settled_evaluation_rows": _descriptor(candidate_path),
        "matched_baseline_settled_evaluation_rows": _descriptor(baseline_path),
        "side_only_pnl_gate_report": _descriptor(gate_path),
        "future_gate_passed": gate["future_gate_passed"],
        "future_gate_blocking_reason_codes": gate["future_gate_blocking_reason_codes"],
        "target_opened_after_decision_freeze": True,
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "future_results_used_for_automatic_unlock": False,
        "manual_promotion_review_required": True,
        **_blocked_safety_fields(),
    }
    manifest["future_gate_manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "conformal_v6_future_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "gate": gate,
        "gate_path": gate_path,
        "gate_sha256": _sha256_file(gate_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_policy_selected_conformal_v6_side_only_future_pnl_gate(
    evaluation_rows: list[dict[str, Any]],
    *,
    matched_baseline_evaluation_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    profile: dict[str, Any],
    decision_freeze_sha256: str,
) -> dict[str, Any]:
    """Apply only the preregistered BUY_UP/BUY_DOWN market-level PnL hard gates."""

    validate_policy_selected_conformal_v6_profile(profile)
    _require_sha256(decision_freeze_sha256, name="decision_freeze_sha256")
    future = dict(profile["future_evaluation"])
    accepted = [row for row in evaluation_rows if row.get("execution_guard_order_allowed") is True]
    baseline_accepted = [
        row
        for row in matched_baseline_evaluation_rows
        if row.get("execution_guard_order_allowed") is True
    ]
    market_ids = sorted(set(evaluation_market_ids))
    if len(evaluation_market_ids) != len(market_ids):
        raise ValueError("future gate evaluation market identities are duplicated")
    if len(market_ids) != int(future["target_quality_valid_market_count"]):
        raise ValueError("future gate evaluation market count is not the frozen target")
    if any(not str(value) for value in market_ids):
        raise ValueError("future gate market identity is empty")
    candidate_by_market = dict.fromkeys(market_ids, 0.0)
    baseline_by_market = dict.fromkeys(market_ids, 0.0)
    if any(str(row.get("market_id") or "") not in candidate_by_market for row in accepted):
        raise ValueError("candidate accepted row is outside the frozen future market window")
    if any(str(row.get("market_id") or "") not in baseline_by_market for row in baseline_accepted):
        raise ValueError("baseline accepted row is outside the frozen future market window")
    for row in accepted:
        candidate_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    for row in baseline_accepted:
        baseline_by_market[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    delta_by_market = {
        market_id: candidate_by_market[market_id] - baseline_by_market[market_id]
        for market_id in market_ids
    }
    bootstrap = _market_bootstrap_interval(
        list(delta_by_market.values()),
        resample_count=int(future["bootstrap_resample_count"]),
        confidence_level=float(future["bootstrap_confidence_level"]),
        seed=int(future["bootstrap_seed"]),
    )
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_side[str(row["selected_side"])].append(row)
        by_action[str(row["executed_action"])].append(row)
        by_family[_action_family(str(row["executed_action"]))].append(row)
    side_metrics = {
        side: _group_metrics(rows, diagnostic_only=False)
        for side, rows in sorted(by_side.items())
        if side in {"UP", "DOWN"}
    }
    action_metrics = {
        action: _group_metrics(rows, diagnostic_only=True)
        for action, rows in sorted(by_action.items())
    }
    family_metrics = {
        family: _group_metrics(rows, diagnostic_only=True)
        for family, rows in sorted(by_family.items())
    }
    candidate_pnl = float(sum(candidate_by_market.values()))
    baseline_pnl = float(sum(baseline_by_market.values()))
    delta_pnl = candidate_pnl - baseline_pnl
    largest_winner = max(candidate_by_market.values(), default=0.0)
    largest_removed = candidate_pnl - max(largest_winner, 0.0)
    required_sides = list(future["required_supported_sides"])
    side_gate = all(
        side in side_metrics
        and side_metrics[side]["accepted_unique_market_count"]
        >= int(future["minimum_supported_side_market_count"])
        and side_metrics[side]["accepted_bet_net_pnl_sum"]
        > float(future["supported_side_post_cost_pnl_minimum_exclusive"])
        for side in required_sides
    )
    accepted_markets = {str(row["market_id"]) for row in accepted}
    baseline_accepted_markets = {str(row["market_id"]) for row in baseline_accepted}
    safety = (
        len(accepted_markets) == len(accepted)
        and len(baseline_accepted_markets) == len(baseline_accepted)
        and all(str(row.get("selected_side") or "") in {"UP", "DOWN"} for row in accepted)
        and all(_settled_evaluation_row_safe(row) for row in accepted + baseline_accepted)
    )
    checks = {
        "minimum_guard_accepted_unique_market_support": len(accepted_markets)
        >= int(future["minimum_guard_accepted_unique_market_count"]),
        "supported_side_post_cost_pnl_gate": side_gate,
        "accepted_bet_total_post_cost_pnl_positive": candidate_pnl
        > float(future["accepted_bet_total_post_cost_pnl_minimum_exclusive"]),
        "candidate_exceeds_matched_baseline": delta_pnl
        > float(future["candidate_minus_matched_baseline_pnl_minimum_exclusive"]),
        "candidate_minus_baseline_bootstrap_lcb_positive": bootstrap[
            "lower_confidence_bound"
        ]
        > float(future["candidate_minus_baseline_bootstrap_lcb_minimum_exclusive"]),
        "largest_winner_removed_pnl_positive": largest_removed
        > float(future["largest_winner_removed_pnl_minimum_exclusive"]),
        "settlement_causality_provenance_and_runtime_safety": safety,
    }
    reason_map = {
        "minimum_guard_accepted_unique_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "supported_side_post_cost_pnl_gate": "supported_side_post_cost_pnl_gate_failed",
        "accepted_bet_total_post_cost_pnl_positive": (
            "accepted_bet_total_post_cost_pnl_not_positive"
        ),
        "candidate_exceeds_matched_baseline": "candidate_does_not_exceed_matched_baseline",
        "candidate_minus_baseline_bootstrap_lcb_positive": (
            "candidate_minus_baseline_bootstrap_lcb_not_positive"
        ),
        "largest_winner_removed_pnl_positive": "largest_winner_removed_pnl_not_positive",
        "settlement_causality_provenance_and_runtime_safety": (
            "settlement_causality_provenance_or_runtime_safety_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    return {
        "schema_version": f"{SCHEMA_PREFIX}-side-only-pnl-gate-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "decision_freeze_sha256": decision_freeze_sha256,
        "pnl_hard_gate_aggregation": "selected_side_buy_up_buy_down_only",
        "action_and_action_family_pnl_diagnostic_only": True,
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted_markets),
        "accepted_side_distribution": dict(
            sorted(Counter(str(row["selected_side"]) for row in accepted).items())
        ),
        "accepted_action_distribution": dict(
            sorted(Counter(str(row["executed_action"]) for row in accepted).items())
        ),
        "accepted_side_metrics": side_metrics,
        "accepted_action_metrics": action_metrics,
        "accepted_action_family_metrics": family_metrics,
        "candidate_post_cost_net_pnl": candidate_pnl,
        "matched_baseline_post_cost_net_pnl": baseline_pnl,
        "candidate_minus_matched_baseline_post_cost_net_pnl": delta_pnl,
        "matched_baseline_guard_accepted_bet_count": len(baseline_accepted),
        "comparison_market_count": len(market_ids),
        "candidate_minus_baseline_market_bootstrap": bootstrap,
        "largest_winning_market_pnl": largest_winner,
        "largest_winner_removed_candidate_pnl": largest_removed,
        "future_gate_checks": checks,
        "future_gate_passed": not blockers,
        "future_gate_blocking_reason_codes": blockers,
        "manual_promotion_review_required": True,
        **_blocked_safety_fields(),
    }


def _validate_prediction_freeze_for_target_access(freeze: dict[str, Any]) -> None:
    blockers = []
    if freeze.get("schema_version") != f"{PREDICTION_SCHEMA_PREFIX}-manifest-v1":
        blockers.append("prediction_freeze_schema_invalid")
    if freeze.get("decision_freeze_written_before_target_access") is not True:
        blockers.append("decision_freeze_not_written")
    if freeze.get("future_target_free_support_gate_passed") is not True:
        blockers.append("target_free_support_gate_failed")
    if freeze.get("future_target_access_allowed_after_decision_freeze") is not True:
        blockers.append("future_target_access_not_allowed")
    if freeze.get("future_labels_outcomes_or_pnl_opened") is not False:
        blockers.append("prediction_freeze_target_sealing_invalid")
    for key, expected in _blocked_safety_fields().items():
        if freeze.get(key) != expected:
            blockers.append(f"prediction_freeze_safety_invalid:{key}")
    if blockers:
        raise ValueError("future prediction freeze not eligible for target access: " + ", ".join(blockers))


def _validate_settled_index(
    index: dict[str, Any],
    *,
    expected_decision_freeze_sha256: str,
    decision_freeze_created_ts: int,
    selected_rows: list[dict[str, Any]],
    reconciliation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    expected_markets = {str(row["market_id"]) for row in selected_rows}
    entry_markets = {str(row.get("market_id") or "") for row in entries}
    checks = {
        "schema": index.get("schema_version") == SETTLED_INDEX_SCHEMA_VERSION,
        "freeze_hash": index.get("decision_freeze_sha256") == expected_decision_freeze_sha256,
        "complete_market_set": len(entries) == 300 and entry_markets == expected_markets,
        "official_read_only": all(row.get("official_read_only_resolution") is True for row in entries),
        "post_freeze": all(row.get("corpus_built_after_decision_freeze") is True for row in entries),
        "post_close": all(row.get("settled_after_market_close") is True for row in entries),
        "target_access_after_freeze": int(index.get("target_access_started_ts") or 0)
        > decision_freeze_created_ts,
        "finalized_after_target_access": int(index.get("index_finalized_ts") or 0)
        >= int(index.get("target_access_started_ts") or 0),
        "finalized_before_reconciliation": int(index.get("index_finalized_ts") or 0)
        <= reconciliation_started_ts,
        "no_result_tuning": index.get("outcomes_used_for_decision_or_selection") is False
        and index.get("outcomes_used_for_threshold_or_model_tuning") is False,
        "safety": all(
            index.get(key) == expected for key, expected in _blocked_safety_fields().items()
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("v6 future settled index invalid: " + ", ".join(blockers))
    return sorted(entries, key=lambda row: str(row["market_id"]))


def _v6_target_schema(row: dict[str, Any]) -> dict[str, Any]:
    updated = {
        **row,
        "schema_version": f"{SCHEMA_PREFIX}-five-action-target-v1",
    }
    updated.pop("target_row_sha256", None)
    updated["target_row_sha256"] = canonical_json_sha256(updated)
    return updated


def _settled_evaluation_row_safe(row: dict[str, Any]) -> bool:
    return bool(
        row.get("settlement_resolved") is True
        and row.get("target_joined_after_decision_freeze") is True
        and row.get("target_used_as_decision_input") is False
        and row.get("forbidden_outcome_field_used_for_decision") is False
        and row.get("feature_causality_violation") is False
        and row.get("provenance_violation") is False
        and row.get("runtime_state_violation") is False
        and row.get("future_results_used_for_tuning") is False
    )


def _group_metrics(rows: list[dict[str, Any]], *, diagnostic_only: bool) -> dict[str, Any]:
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "accepted_bet_net_pnl_sum": float(sum(float(row["accepted_bet_net_pnl"]) for row in rows)),
        "diagnostic_only": diagnostic_only,
    }


def _action_family(action: str) -> str:
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    return "NO_TRADE" if action == "NO_TRADE" else "UNKNOWN"


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# #207 v6 future settled corpus",
            "",
            f"- ready: `{report['settled_corpus_index_ready']}`",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            "- source rounds mutated: `false`",
            "- results used for tuning/rerun: `false/false`",
            "- paper/live/promotion unlock: `false`",
            "",
        ]
    )


def _gate_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# #207 v6 future side-only PnL gate",
        "",
        f"- passed: `{gate['future_gate_passed']}`",
        f"- candidate PnL: `{gate['candidate_post_cost_net_pnl']:.8f}`",
        f"- matched baseline PnL: `{gate['matched_baseline_post_cost_net_pnl']:.8f}`",
        "- action/family PnL: `diagnostic_only`",
        "- results used for tuning/rerun: `false/false`",
        "- automatic promotion/paper/live unlock: `false`",
        "",
        "| Side | Markets | Post-cost PnL |",
        "|---|---:|---:|",
    ]
    for side, row in gate["accepted_side_metrics"].items():
        lines.append(
            f"| {side} | {row['accepted_unique_market_count']} | "
            f"{row['accepted_bet_net_pnl_sum']:.8f} |"
        )
    lines.extend(["", f"Blocking reasons: `{gate['future_gate_blocking_reason_codes']}`", ""])
    return "\n".join(lines)
