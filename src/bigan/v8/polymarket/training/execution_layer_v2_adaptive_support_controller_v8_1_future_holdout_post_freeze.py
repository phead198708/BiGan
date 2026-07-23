"""Single-use read-only settlement and PnL gate for issue #246 v8.1."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    SCHEMA_PREFIX,
    _v7_0_blocked_safety_fields,
    build_adaptive_support_controller_v8_1_future_pnl_gate,
    validate_adaptive_support_controller_v8_1_future_holdout_plan,
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

SETTLED_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-settled-index-v1"


@dataclass(frozen=True, slots=True)
class AdaptiveSupportControllerV81FutureSettlementConfig:
    """Pinned inputs for one bounded official read-only settlement attempt."""

    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    implementation_commit: str
    target_access_started_ts: int
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common_config(
            run_id=self.run_id,
            implementation_commit=self.implementation_commit,
            stage_started_ts=self.target_access_started_ts,
        )
        _require_sha256(
            self.expected_target_free_freeze_manifest_sha256,
            name="expected_target_free_freeze_manifest_sha256",
        )
        if self.provider_timeout_seconds <= 0 or self.provider_http_timeout_seconds <= 0:
            raise ValueError("provider timeouts must be positive")
        if self.settlement_max_wait_seconds < 0:
            raise ValueError("settlement_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0 or self.max_workers <= 0:
            raise ValueError("settlement polling and workers must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "target_free_freeze_manifest_path",
            Path(self.target_free_freeze_manifest_path),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveSupportControllerV81FutureEvaluationConfig:
    """Pinned inputs for the one authoritative future PnL comparison."""

    run_id: str
    output_dir: Path | str
    target_free_freeze_manifest_path: Path | str
    expected_target_free_freeze_manifest_sha256: str
    settled_index_path: Path | str
    expected_settled_index_sha256: str
    runtime_policy_profile_path: Path | str
    expected_runtime_policy_profile_sha256: str
    implementation_commit: str
    evaluation_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        _validate_common_config(
            run_id=self.run_id,
            implementation_commit=self.implementation_commit,
            stage_started_ts=self.evaluation_started_ts,
        )
        for name in (
            "expected_target_free_freeze_manifest_sha256",
            "expected_settled_index_sha256",
            "expected_runtime_policy_profile_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "target_free_freeze_manifest_path",
            "settled_index_path",
            "runtime_policy_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def build_adaptive_support_controller_v8_1_future_settled_index(
    config: AdaptiveSupportControllerV81FutureSettlementConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Settle quarantine copies after an exclusive target-access claim."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    freeze, selected, _, _ = _validated_freeze(
        freeze_path,
        expected_sha256=config.expected_target_free_freeze_manifest_sha256,
    )
    freeze_created_ts = int(freeze["decision_freeze_created_ts"])
    max_market_end_ts = max(int(row["market_end_ts"]) for row in selected)
    if config.target_access_started_ts <= max(freeze_created_ts, max_market_end_ts):
        raise ValueError("#246 target access attempted before freeze or market close")
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-target-access-claim-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "target_access_started_ts": config.target_access_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "official_read_only_settlement_on_quarantine_copies": True,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    claim_path = (
        freeze_path.parent / "v8_1_future_single_use_target_access_claim.json"
    )
    _write_single_use_claim(claim_path, claim)
    marker_path = run_dir / "v8_1_future_settlement_start_marker.json"
    _write_json(marker_path, claim)

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
                failure = failures[market_id]
                failure["reason_codes"] = sorted(
                    {
                        *failure.get("reason_codes", []),
                        "settlement_resolution_max_wait_elapsed",
                    }
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [selected_by_market[market_id] for market_id in sorted(retryable)]

    success_rows = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    failure_rows = sorted(failures.values(), key=lambda row: str(row["market_id"]))
    complete = len(success_rows) == EXACT_MARKET_COUNT and not failure_rows
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.target_access_started_ts:
        raise ValueError("#246 settlement finalized before target access start")
    index_path = run_dir / "v8_1_future_settled_corpus_index.json"
    index: dict[str, Any] | None = None
    if complete:
        index = {
            "schema_version": SETTLED_INDEX_SCHEMA_VERSION,
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "target_free_freeze_manifest": _descriptor(freeze_path),
            "target_access_claim": _descriptor(claim_path),
            "target_access_started_ts": config.target_access_started_ts,
            "index_finalized_ts": finalized_ts,
            "entry_count": len(success_rows),
            "entries": success_rows,
            "official_read_only_resolution": True,
            "source_outcome_blind_rounds_mutated": False,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "future_results_used_for_tuning": False,
            "result_selected_rerun_allowed": False,
            **_v7_0_blocked_safety_fields(),
        }
        index["settled_index_id"] = canonical_json_sha256(index)
        _write_json(index_path, index)
    reason_distribution = Counter(
        reason for row in failure_rows for reason in row.get("reason_codes", [])
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": len(selected),
        "settled_market_count": len(success_rows),
        "unresolved_or_failed_market_count": len(failure_rows),
        "settlement_attempt_count": attempt_count,
        "settlement_retry_market_count": len(retry_market_ids),
        "unresolved_or_failed_reason_distribution": dict(
            sorted(reason_distribution.items())
        ),
        "unresolved_or_failed_markets": failure_rows,
        "settled_index_ready": complete,
        "target_access_started_ts": config.target_access_started_ts,
        "index_finalized_ts": finalized_ts,
        "source_outcome_blind_rounds_mutated": False,
        "official_read_only_resolution_only": True,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "blocking_reason_codes": [] if complete else ["settled_window_incomplete"],
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v8_1_future_settlement_report.json"
    report_md_path = run_dir / "v8_1_future_settlement_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "target_access_claim": _descriptor(claim_path),
        "settlement_start_marker": _descriptor(marker_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "settled_index": _descriptor(index_path) if complete else None,
        "settled_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_1_future_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(run_dir, report, report_path, manifest, manifest_path)
    result.update(
        {
            "index": index,
            "index_path": index_path if complete else None,
            "index_sha256": _sha256_file(index_path) if complete else None,
        }
    )
    return result


def evaluate_adaptive_support_controller_v8_1_future_pnl_gate(
    config: AdaptiveSupportControllerV81FutureEvaluationConfig,
) -> dict[str, Any]:
    """Consume complete official targets once and compare v8.1 with v6.7."""

    freeze_path = config.target_free_freeze_manifest_path.resolve()
    freeze, selected, candidate_decisions, baseline_decisions = _validated_freeze(
        freeze_path,
        expected_sha256=config.expected_target_free_freeze_manifest_sha256,
    )
    settled_path = config.settled_index_path.resolve()
    runtime_path = config.runtime_policy_profile_path.resolve()
    _verify_pin(settled_path, config.expected_settled_index_sha256, "#246 settled index")
    _verify_pin(
        runtime_path,
        config.expected_runtime_policy_profile_sha256,
        "#246 runtime policy profile",
    )
    plan_descriptor = _verified_descriptor(freeze["plan"], "#246 frozen plan")
    if plan_descriptor["sha256"] != FROZEN_PLAN_SHA256:
        raise ValueError("#246 freeze references a different plan")
    plan = _load_json(Path(plan_descriptor["path"]))
    validate_adaptive_support_controller_v8_1_future_holdout_plan(plan)
    if (
        config.expected_runtime_policy_profile_sha256.lower()
        != plan["lineage"]["runtime_policy_profile_sha256"]
    ):
        raise ValueError("#246 runtime policy profile pin drifted")
    settled = _load_json(settled_path)
    entries = _validate_settled_index(
        settled,
        freeze_path=freeze_path,
        freeze_sha256=config.expected_target_free_freeze_manifest_sha256,
        selected_market_ids=[str(row["market_id"]) for row in selected],
        evaluation_started_ts=config.evaluation_started_ts,
    )
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-gate-claim-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_started_ts": config.evaluation_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_index": _descriptor(settled_path),
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    claim_path = freeze_path.parent / "v8_1_future_single_use_pnl_gate_claim.json"
    _write_single_use_claim(claim_path, claim)
    runtime_profile = _load_json(runtime_path)
    candidate_targets = _runtime_targets_for_decisions(
        candidate_decisions,
        settled_entries=entries,
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role="future_unseen_holdout_v8_1_candidate",
    )
    baseline_targets = _runtime_targets_for_decisions(
        baseline_decisions,
        settled_entries=entries,
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role="future_unseen_holdout_v6_7_baseline",
    )
    market_ids = [str(row["market_id"]) for row in selected]
    gate = build_adaptive_support_controller_v8_1_future_pnl_gate(
        candidate_targets,
        baseline_rows=baseline_targets,
        evaluation_market_ids=market_ids,
        settled_market_ids=[str(row["market_id"]) for row in entries],
        plan=plan,
        target_free_freeze_sha256=(
            config.expected_target_free_freeze_manifest_sha256
        ),
    )
    candidate_path = run_dir / "v8_1_future_candidate_runtime_targets.jsonl"
    baseline_path = run_dir / "v8_1_future_v6_7_runtime_targets.jsonl"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(baseline_path, baseline_targets)
    report = {
        **gate,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_started_ts": config.evaluation_started_ts,
        "target_access_claim": _descriptor(
            Path(
                _verified_descriptor(
                    settled["target_access_claim"],
                    "#246 target claim",
                )["path"]
            )
        ),
        "pnl_gate_claim": _descriptor(claim_path),
        "future_results_used_for_tuning": False,
        "future_results_used_for_rerun": False,
        "automatic_paper_or_live_unlock_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    report_path = run_dir / "v8_1_future_pnl_noninferiority_gate_report.json"
    report_md_path = run_dir / "v8_1_future_pnl_noninferiority_gate_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _evaluation_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-gate-manifest-v1",
        "run_id": config.run_id,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_index": _descriptor(settled_path),
        "runtime_policy_profile": _descriptor(runtime_path),
        "pnl_gate_claim": _descriptor(claim_path),
        "candidate_runtime_targets": _descriptor(candidate_path),
        "v6_7_runtime_targets": _descriptor(baseline_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "future_pnl_gate_passed": report["future_pnl_gate_passed"],
        "future_pnl_gate_blocking_reason_codes": report[
            "future_pnl_gate_blocking_reason_codes"
        ],
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "automatic_paper_or_live_unlock_allowed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_1_future_pnl_noninferiority_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validated_freeze(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _verify_pin(path, expected_sha256, "#246 target-free freeze")
    freeze = _load_json(path)
    if (
        freeze.get("schema_version")
        != f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1"
        or freeze.get("exact_market_count") != EXACT_MARKET_COUNT
        or freeze.get("target_free_freeze_passed") is not True
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("decision_freeze_written_before_target_access") is not True
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_scores_mutated") is not False
        or freeze.get("threshold_model_or_controller_tuning_performed") is not False
    ):
        raise ValueError("#246 target-free freeze is not target-access eligible")
    for field, expected in _v7_0_blocked_safety_fields().items():
        if freeze.get(field) != expected:
            raise ValueError(f"#246 target-free freeze safety mismatch: {field}")
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_rows"], "#246 selected rows")["path"])
    )
    candidate = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["candidate_runtime"],
                "#246 v8.1 decisions",
            )["path"]
        )
    )
    baseline = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["v6_7_runtime"],
                "#246 v6.7 decisions",
            )["path"]
        )
    )
    market_ids = [str(row.get("market_id") or "") for row in selected]
    if (
        len(selected) != EXACT_MARKET_COUNT
        or "" in market_ids
        or len(set(market_ids)) != EXACT_MARKET_COUNT
        or len(candidate) < 40
        or any(
            str(row.get("market_id") or "") not in set(market_ids)
            for row in candidate + baseline
        )
    ):
        raise ValueError("#246 frozen market or accepted-decision support invalid")
    return freeze, selected, candidate, baseline


def _validate_settled_index(
    index: dict[str, Any],
    *,
    freeze_path: Path,
    freeze_sha256: str,
    selected_market_ids: list[str],
    evaluation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    entry_ids = [str(row.get("market_id") or "") for row in entries]
    freeze_descriptor = _verified_descriptor(
        index["target_free_freeze_manifest"],
        "#246 settled freeze",
    )
    if (
        index.get("schema_version") != SETTLED_INDEX_SCHEMA_VERSION
        or freeze_descriptor != _descriptor(freeze_path)
        or freeze_descriptor["sha256"] != freeze_sha256.lower()
        or index.get("entry_count") != EXACT_MARKET_COUNT
        or len(entries) != EXACT_MARKET_COUNT
        or set(entry_ids) != set(selected_market_ids)
        or "" in entry_ids
        or int(index.get("target_access_started_ts") or 0) <= 0
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("official_read_only_resolution") is not True
        or index.get("source_outcome_blind_rounds_mutated") is not False
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
    ):
        raise ValueError("#246 settled index is not evaluation eligible")
    _verified_descriptor(index["target_access_claim"], "#246 target access claim")
    for field, expected in _v7_0_blocked_safety_fields().items():
        if index.get(field) != expected:
            raise ValueError(f"#246 settled index safety mismatch: {field}")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#246 settled entry violates quarantine contract")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"#246 settled {name}")
    return entries


def _write_single_use_claim(path: Path, claim: dict[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("#246 frozen future targets have already been consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_common_config(
    *,
    run_id: str,
    implementation_commit: str,
    stage_started_ts: int,
) -> None:
    if not run_id.strip():
        raise ValueError("run_id is required")
    if len(implementation_commit) != 40:
        raise ValueError("implementation_commit must be a Git SHA-1")
    if stage_started_ts <= 0:
        raise ValueError("stage timestamp must be positive")


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v8.1 Future Read-Only Settlement",
            "",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- settled markets: `{report['settled_market_count']}`",
            "- unresolved/failed markets: "
            f"`{report['unresolved_or_failed_market_count']}`",
            f"- settled index ready: `{str(report['settled_index_ready']).lower()}`",
            f"- blockers: `{report['blocking_reason_codes']}`",
            "- source outcome-blind rounds mutated: `false`",
            "- official read-only resolution only: `true`",
            "- paper/live/write/wallet/capital/handoff remain blocked.",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v8.1 Future PnL Non-Inferiority Gate",
            "",
            f"- candidate after-cost PnL: `{report['candidate_after_cost_pnl']}`",
            f"- v6.7 after-cost PnL: `{report['v6_7_after_cost_pnl']}`",
            f"- total delta: `{report['candidate_minus_v6_7_after_cost_pnl']}`",
            "- comparison operator: `greater_than_or_equal`",
            "- equality passes non-inferiority: `true`",
            f"- future gate passed: `{str(report['future_pnl_gate_passed']).lower()}`",
            f"- blockers: `{report['future_pnl_gate_blocking_reason_codes']}`",
            "- automatic paper/live unlock: `false`",
            "- future results used for tuning/rerun: `false`",
            "",
        ]
    )
