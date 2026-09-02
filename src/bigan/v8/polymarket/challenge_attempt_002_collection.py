"""Authorization-gated outcome-blind collection supervisor for attempt-002."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002 import (
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    validate_attempt_002_operator_authorization,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
    validate_persistent_outcome_blind_collector_protocol,
)

COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-collection-supervisor-state-v1"
)
COLLECTION_BATCH_REPORT_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-collection-batch-report-v1"
)
COLLECTOR_INDEX_FILENAME = "persistent_outcome_blind_round_index.jsonl"
SUPERVISOR_STATE_FILENAME = "attempt_002_collection_supervisor_state.json"
TARGET_QUALITY_VALID_COUNT = 120
MAXIMUM_ATTEMPTED_COUNT = 180
BATCH_ROUND_COUNT = 12
MAXIMUM_BATCH_COUNT = 15
ATTEMPT_002_PROTOCOL_SHA256 = (
    "0fa091610966a3a3470872a7e1b5832c8a32985fc312235366ad41aa891f249f"
)
COLLECTOR_PROTOCOL_SHA256 = (
    "9a4020f173c3ee7f1f396ceb5387672f67939c64906d2f263cabe623a1d7c083"
)
FEATURE_CONTRACT_SHA256 = (
    "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
)


class ChallengeAttempt002CollectionError(ValueError):
    """Raised when collection cannot remain within attempt-002's freeze."""


@dataclass(frozen=True, slots=True)
class Attempt002CollectionConfig:
    """Hash-pinned collection inputs; authorization is always mandatory."""

    repository_root: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    operator_authorization_path: Path | str
    expected_operator_authorization_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    service_root: Path | str
    implementation_commit: str
    run_id: str
    max_consecutive_failures: int = 3
    failure_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not _is_git_commit(self.implementation_commit):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        if self.failure_backoff_seconds < 0.0:
            raise ValueError("failure_backoff_seconds must be nonnegative")
        for name in (
            "repository_root",
            "protocol_path",
            "operator_authorization_path",
            "collector_protocol_path",
            "feature_contract_path",
            "service_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "expected_protocol_sha256",
            "expected_operator_authorization_sha256",
            "expected_collector_protocol_sha256",
            "expected_feature_contract_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")


def preflight_attempt_002_collection(
    config: Attempt002CollectionConfig,
    *,
    allow_existing_service_root: bool = True,
) -> dict[str, Any]:
    """Validate launch inputs without creating a directory or using network."""

    root = config.repository_root.resolve()
    protocol_path = config.protocol_path.resolve()
    authorization_path = config.operator_authorization_path.resolve()
    collector_protocol_path = config.collector_protocol_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    service_root = config.service_root.resolve()
    for path, expected, label in (
        (
            protocol_path,
            config.expected_protocol_sha256,
            "attempt-002 protocol",
        ),
        (
            authorization_path,
            config.expected_operator_authorization_sha256,
            "operator authorization",
        ),
        (
            collector_protocol_path,
            config.expected_collector_protocol_sha256,
            "collector protocol",
        ),
        (
            feature_contract_path,
            config.expected_feature_contract_sha256,
            "feature contract",
        ),
    ):
        _verify(path, expected, label=label)
    protocol = _load_json(protocol_path)
    authorization = _load_json(authorization_path)
    collector_protocol = _load_json(collector_protocol_path)
    validate_attempt_002_preregistration(protocol)
    validate_attempt_002_operator_authorization(
        authorization,
        protocol=protocol,
        protocol_sha256=config.expected_protocol_sha256,
    )
    validate_persistent_outcome_blind_collector_protocol(
        collector_protocol
    )
    if (
        config.expected_protocol_sha256.lower()
        != ATTEMPT_002_PROTOCOL_SHA256
        or config.expected_collector_protocol_sha256.lower()
        != COLLECTOR_PROTOCOL_SHA256
        or config.expected_feature_contract_sha256.lower()
        != FEATURE_CONTRACT_SHA256
    ):
        raise ChallengeAttempt002CollectionError(
            "collection inputs do not match the frozen attempt-002 launch pins"
        )
    expected_service_root = (
        root / str(protocol["future_window"]["service_root"])
    ).resolve()
    if (
        service_root != expected_service_root
        or root not in service_root.parents
    ):
        raise ChallengeAttempt002CollectionError(
            "service root does not match the frozen attempt-002 path"
        )
    if service_root.exists() and not allow_existing_service_root:
        raise ChallengeAttempt002CollectionError(
            "attempt-002 service root already exists"
        )
    if service_root.exists():
        _validate_resumable_service_root(
            service_root=service_root,
            config=config,
            protocol=protocol,
        )
    window = protocol["future_window"]
    if (
        window["exact_quality_valid_market_count"]
        != TARGET_QUALITY_VALID_COUNT
        or window["maximum_attempted_market_count"]
        != MAXIMUM_ATTEMPTED_COUNT
        or window["bounded_batch_market_count"] != BATCH_ROUND_COUNT
        or window["maximum_batch_count"] != MAXIMUM_BATCH_COUNT
        or window["candidate_scoring_during_raw_capture_allowed"] is not False
        or window["settlement_finalizer_enabled_during_collection"] is not False
        or window["resolution_provider_enabled_during_collection"] is not False
        or window["outcomes_resolution_labels_or_pnl_opened"] is not False
    ):
        raise ChallengeAttempt002CollectionError(
            "attempt-002 collection limits or outcome-blind controls changed"
        )
    return {
        "attempt_id": protocol["attempt_id"],
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "protocol_sha256": config.expected_protocol_sha256.lower(),
        "operator_authorization_sha256": (
            config.expected_operator_authorization_sha256.lower()
        ),
        "collector_protocol_sha256": (
            config.expected_collector_protocol_sha256.lower()
        ),
        "feature_contract_sha256": (
            config.expected_feature_contract_sha256.lower()
        ),
        "service_root": str(service_root),
        "service_root_exists": service_root.exists(),
        "exact_quality_valid_market_target": TARGET_QUALITY_VALID_COUNT,
        "maximum_attempted_market_count": MAXIMUM_ATTEMPTED_COUNT,
        "bounded_batch_market_count": BATCH_ROUND_COUNT,
        "maximum_batch_count": MAXIMUM_BATCH_COUNT,
        "network_or_collection_invoked": False,
        "service_root_created": False,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "safety": SAFE_FALSES,
    }


def _single_supervisor_instance(function):
    """Require authorization before creating and locking the service root."""

    def wrapped(
        config: Attempt002CollectionConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        preflight_attempt_002_collection(config)
        service_root = config.service_root.resolve()
        service_root.mkdir(parents=True, exist_ok=True)
        lock_path = service_root / "attempt_002_collection_supervisor.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                raise ChallengeAttempt002CollectionError(
                    "attempt-002 collection supervisor already running"
                ) from error
            try:
                return function(config, **kwargs)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return wrapped


@_single_supervisor_instance
def run_attempt_002_collection(
    config: Attempt002CollectionConfig,
    *,
    run_batch: Callable[..., Mapping[str, Any]] | None = None,
    load_index: Callable[[Path], Sequence[Mapping[str, Any]]] | None = None,
    process_id: int | None = None,
) -> dict[str, Any]:
    """Collect one bounded batch at a time and stop at target or attempt cap."""

    preflight = preflight_attempt_002_collection(config)
    protocol = _load_json(config.protocol_path.resolve())
    service_root = config.service_root.resolve()
    service_root.mkdir(parents=True, exist_ok=True)
    state_path = service_root / SUPERVISOR_STATE_FILENAME
    current_state = (
        _load_json(state_path) if state_path.is_file() else {}
    )
    pid = process_id if process_id is not None else os.getpid()
    state = {
        "schema_version": COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION,
        **preflight,
        "collector_pid": pid,
        "status": "authorized_collection_starting",
        "batch_count": int(current_state.get("batch_count") or 0),
        "attempted_market_count": int(
            current_state.get("attempted_market_count") or 0
        ),
        "quality_valid_market_count": int(
            current_state.get("quality_valid_market_count") or 0
        ),
        "collection_started": True,
        "network_or_collection_invoked": False,
        "service_root_created": True,
        "batch_reports": list(current_state.get("batch_reports") or []),
        "outcomes_resolution_labels_or_pnl_opened": False,
        "safety": SAFE_FALSES,
    }
    _write_json_atomic(state_path, state)
    batch_runner = run_batch or _default_run_batch
    index_loader = load_index or (
        lambda path: load_and_validate_persistent_outcome_blind_index(path)
    )
    boundary = int(
        protocol["future_window"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    while True:
        rows_before = list(
            index_loader(service_root / COLLECTOR_INDEX_FILENAME)
        )
        progress_before = summarize_attempt_002_collection(
            rows_before,
            boundary_exclusive=boundary,
        )
        if progress_before["quality_valid_market_count"] >= (
            TARGET_QUALITY_VALID_COUNT
        ):
            return _finish_state(
                state_path=state_path,
                state=state,
                progress=progress_before,
                status="quality_valid_target_reached",
                collector_pid=None,
            )
        if (
            progress_before["attempted_market_count"]
            >= MAXIMUM_ATTEMPTED_COUNT
            or state["batch_count"] >= MAXIMUM_BATCH_COUNT
        ):
            return _finish_state(
                state_path=state_path,
                state=state,
                progress=progress_before,
                status="attempt_cap_exhausted_fail_closed",
                collector_pid=None,
            )

        state["status"] = "collecting_outcome_blind_batch"
        state["network_or_collection_invoked"] = True
        _write_json_atomic(state_path, state)
        try:
            collector_state = dict(
                batch_runner(
                    service_root=service_root,
                    collector_protocol_path=(
                        config.collector_protocol_path.resolve()
                    ),
                    collector_protocol_sha256=(
                        config.expected_collector_protocol_sha256
                    ),
                    feature_contract_path=(
                        config.feature_contract_path.resolve()
                    ),
                    feature_contract_sha256=(
                        config.expected_feature_contract_sha256
                    ),
                    batch_round_count=BATCH_ROUND_COUNT,
                    max_consecutive_failures=(
                        config.max_consecutive_failures
                    ),
                    failure_backoff_seconds=config.failure_backoff_seconds,
                )
            )
        except Exception as error:
            state.update(
                {
                    "status": "collection_batch_failed_fail_closed",
                    "collector_pid": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fail_closed": True,
                    "outcomes_resolution_labels_or_pnl_opened": False,
                    "safety": SAFE_FALSES,
                }
            )
            _write_json_atomic(state_path, state)
            raise
        rows_after = list(
            index_loader(service_root / COLLECTOR_INDEX_FILENAME)
        )
        progress_after = summarize_attempt_002_collection(
            rows_after,
            boundary_exclusive=boundary,
        )
        attempted_delta = (
            progress_after["attempted_market_count"]
            - progress_before["attempted_market_count"]
        )
        if attempted_delta <= 0 or attempted_delta > BATCH_ROUND_COUNT:
            raise ChallengeAttempt002CollectionError(
                "bounded collector batch did not add 1..12 attempted rows"
            )
        state["batch_count"] += 1
        report = _build_batch_report(
            state=state,
            progress_before=progress_before,
            progress_after=progress_after,
            collector_state=collector_state,
            service_root=service_root,
        )
        report_path = service_root / (
            f"attempt_002_collection_batch_{state['batch_count']:03d}.json"
        )
        _write_json_exclusive(report_path, report)
        state.update(
            {
                "status": "outcome_blind_batch_complete",
                "attempted_market_count": progress_after[
                    "attempted_market_count"
                ],
                "quality_valid_market_count": progress_after[
                    "quality_valid_market_count"
                ],
                "remaining_quality_valid_market_count": progress_after[
                    "remaining_quality_valid_market_count"
                ],
                "exclusion_reason_distribution": progress_after[
                    "exclusion_reason_distribution"
                ],
                "provider_health": report["provider_health"],
                "batch_reports": [
                    *state["batch_reports"],
                    _descriptor(report_path),
                ],
                "outcomes_resolution_labels_or_pnl_opened": False,
                "safety": SAFE_FALSES,
            }
        )
        _write_json_atomic(state_path, state)
        if progress_after["quality_valid_market_count"] >= (
            TARGET_QUALITY_VALID_COUNT
        ):
            return _finish_state(
                state_path=state_path,
                state=state,
                progress=progress_after,
                status="quality_valid_target_reached",
                collector_pid=None,
            )
        if (
            progress_after["attempted_market_count"]
            >= MAXIMUM_ATTEMPTED_COUNT
            or state["batch_count"] >= MAXIMUM_BATCH_COUNT
        ):
            return _finish_state(
                state_path=state_path,
                state=state,
                progress=progress_after,
                status="attempt_cap_exhausted_fail_closed",
                collector_pid=None,
            )


def summarize_attempt_002_collection(
    rows: Sequence[Mapping[str, Any]],
    *,
    boundary_exclusive: int,
) -> dict[str, Any]:
    """Apply the frozen chronological window rule without opening targets."""

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("sequence") or 0),
        ),
    )
    attempted = ordered[:MAXIMUM_ATTEMPTED_COUNT]
    selected = []
    reasons: Counter[str] = Counter()
    market_ids: set[str] = set()
    slugs: set[str] = set()
    decision_ids: set[str] = set()
    source_hashes: set[str] = set()
    for row in attempted:
        if row.get("labels_outcomes_or_pnl_opened") is not False:
            raise ChallengeAttempt002CollectionError(
                "collector index contains opened outcomes or labels"
            )
        row_reasons = set()
        market_id = str(row.get("market_id") or "")
        slug = str(row.get("slug") or "")
        decision_id = str(row.get("decision_id") or "")
        source_hash = str(row.get("source_row_hash") or "")
        if row.get("capture_quality_valid") is not True:
            row_reasons.add("capture_quality_invalid")
            row_reasons.update(
                str(value)
                for value in (
                    row.get("capture_quality_reason_codes") or []
                )
            )
        if (
            int(row.get("scheduled_round_start_ts") or 0)
            <= boundary_exclusive
        ):
            row_reasons.add("scheduled_round_not_strictly_later")
        if int(row.get("market_start_ts") or 0) <= boundary_exclusive:
            row_reasons.add("market_start_not_strictly_later")
        for value, label, seen in (
            (market_id, "market_id", market_ids),
            (slug, "slug", slugs),
            (decision_id, "decision_id", decision_ids),
            (source_hash, "source_row_hash", source_hashes),
        ):
            if not value:
                row_reasons.add(f"{label}_missing")
            elif value in seen:
                row_reasons.add(f"{label}_duplicate")
        row_reasons.update(
            str(value)
            for value in (row.get("duplicate_identity_reason_codes") or [])
        )
        if row_reasons:
            reasons.update(row_reasons)
            continue
        selected.append(row)
        market_ids.add(market_id)
        slugs.add(slug)
        decision_ids.add(decision_id)
        source_hashes.add(source_hash)
        if len(selected) == TARGET_QUALITY_VALID_COUNT:
            break
    return {
        "attempted_market_count": len(attempted),
        "quality_valid_market_count": len(selected),
        "remaining_quality_valid_market_count": max(
            0,
            TARGET_QUALITY_VALID_COUNT - len(selected),
        ),
        "target_quality_valid_market_count": TARGET_QUALITY_VALID_COUNT,
        "maximum_attempted_market_count": MAXIMUM_ATTEMPTED_COUNT,
        "exclusion_reason_distribution": dict(sorted(reasons.items())),
        "target_reached": len(selected) == TARGET_QUALITY_VALID_COUNT,
        "attempt_cap_exhausted": (
            len(attempted) >= MAXIMUM_ATTEMPTED_COUNT
            and len(selected) < TARGET_QUALITY_VALID_COUNT
        ),
        "selected_market_ids": [
            str(row["market_id"]) for row in selected
        ],
        "outcomes_resolution_labels_or_pnl_opened": False,
    }


def _build_batch_report(
    *,
    state: Mapping[str, Any],
    progress_before: Mapping[str, Any],
    progress_after: Mapping[str, Any],
    collector_state: Mapping[str, Any],
    service_root: Path,
) -> dict[str, Any]:
    provider_health = _provider_health_from_collector_state(
        collector_state,
    )
    before_reasons = Counter(
        progress_before["exclusion_reason_distribution"]
    )
    after_reasons = Counter(
        progress_after["exclusion_reason_distribution"]
    )
    batch_reasons = {
        name: count
        for name, count in sorted((after_reasons - before_reasons).items())
        if count
    }
    report = {
        "schema_version": COLLECTION_BATCH_REPORT_SCHEMA_VERSION,
        "attempt_id": state["attempt_id"],
        "run_id": state["run_id"],
        "batch_number": state["batch_count"],
        "collector_pid": state["collector_pid"],
        "implementation_commit": state["implementation_commit"],
        "protocol_sha256": state["protocol_sha256"],
        "operator_authorization_sha256": state[
            "operator_authorization_sha256"
        ],
        "service_root": str(service_root),
        "batch_attempted_market_count": (
            progress_after["attempted_market_count"]
            - progress_before["attempted_market_count"]
        ),
        "batch_quality_valid_market_count": (
            progress_after["quality_valid_market_count"]
            - progress_before["quality_valid_market_count"]
        ),
        "cumulative_attempted_market_count": progress_after[
            "attempted_market_count"
        ],
        "cumulative_quality_valid_market_count": progress_after[
            "quality_valid_market_count"
        ],
        "remaining_quality_valid_market_count": progress_after[
            "remaining_quality_valid_market_count"
        ],
        "batch_exclusion_reason_distribution": batch_reasons,
        "cumulative_exclusion_reason_distribution": progress_after[
            "exclusion_reason_distribution"
        ],
        "provider_health": provider_health,
        "collector_state_sha256": canonical_json_sha256(
            dict(collector_state)
        ),
        "github_issue": 260,
        "github_comment_policy": "one_summary_comment_per_completed_batch",
        "candidate_scoring_during_collection": False,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "safety": SAFE_FALSES,
    }
    report["github_comment_markdown"] = format_attempt_002_batch_comment(
        report
    )
    return report


def format_attempt_002_batch_comment(
    report: Mapping[str, Any],
) -> str:
    """Render the single issue #260 comment allowed for one completed batch."""

    provider = dict(report.get("provider_health") or {})
    completeness = dict(
        provider.get("feature_completeness_report") or {}
    )
    return "\n".join(
        [
            (
                "attempt-002 outcome-blind batch "
                f"{report['batch_number']} complete"
            ),
            "",
            f"- collector PID: `{report['collector_pid']}`",
            f"- commit: `{report['implementation_commit']}`",
            f"- frozen plan SHA-256: `{report['protocol_sha256']}`",
            (
                "- batch attempted / quality-valid: "
                f"`{report['batch_attempted_market_count']} / "
                f"{report['batch_quality_valid_market_count']}`"
            ),
            (
                "- cumulative attempted / quality-valid / remaining: "
                f"`{report['cumulative_attempted_market_count']} / "
                f"{report['cumulative_quality_valid_market_count']} / "
                f"{report['remaining_quality_valid_market_count']}`"
            ),
            (
                "- batch exclusion reasons: `"
                + json.dumps(
                    report["batch_exclusion_reason_distribution"],
                    sort_keys=True,
                )
                + "`"
            ),
            (
                "- provider completeness complete / incomplete: "
                f"`{completeness.get('complete_feature_row_count')} / "
                f"{completeness.get('incomplete_feature_row_count')}`"
            ),
            (
                "- provider health buckets: `"
                + json.dumps(
                    provider.get("provider_health_bucket_counts"),
                    sort_keys=True,
                )
                + "`"
            ),
            "- outcomes, resolution labels, and PnL opened: `false`",
        ]
    )


def _provider_health_from_collector_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    report_path_value = state.get("last_batch_canary_report_path")
    expected = str(
        state.get("last_batch_canary_report_sha256") or ""
    ).lower()
    if not isinstance(report_path_value, str) or not _is_sha256(expected):
        raise ChallengeAttempt002CollectionError(
            "collector state lacks the completed batch canary report"
        )
    report_path = Path(report_path_value).resolve()
    _verify(report_path, expected, label="batch canary report")
    report = _load_json(report_path)
    diagnostics = dict(report.get("provider_health_diagnostics") or {})
    if (
        report.get("development_data_canary_passed") is not True
        or int(report.get("provider_health_validation_error_count") or 0)
        != 0
        or not diagnostics
    ):
        raise ChallengeAttempt002CollectionError(
            "batch provider-health canary did not pass"
        )
    return {
        "development_data_canary_passed": True,
        "provider_health_validation_error_count": 0,
        "feature_completeness_report": diagnostics.get(
            "feature_completeness_report"
        ),
        "missing_versus_zero_audit_report": diagnostics.get(
            "missing_versus_zero_audit_report"
        ),
        "provider_health_bucket_counts": (
            diagnostics.get("feature_completeness_report") or {}
        ).get("provider_health_bucket_counts"),
        "diagnostic_only": True,
    }


def _default_run_batch(**kwargs: Any) -> Mapping[str, Any]:
    from examples.v8.run_execution_layer_v2_persistent_outcome_blind_collector import (
        run_service,
    )

    return run_service(
        service_root=kwargs["service_root"],
        protocol_path=kwargs["collector_protocol_path"],
        protocol_sha256=kwargs["collector_protocol_sha256"],
        batch_round_count=kwargs["batch_round_count"],
        max_batches=1,
        max_consecutive_failures=kwargs["max_consecutive_failures"],
        failure_backoff_seconds=kwargs["failure_backoff_seconds"],
        batch_canary_feature_contract_path=kwargs[
            "feature_contract_path"
        ],
        batch_canary_feature_contract_sha256=kwargs[
            "feature_contract_sha256"
        ],
    )


def _validate_resumable_service_root(
    *,
    service_root: Path,
    config: Attempt002CollectionConfig,
    protocol: Mapping[str, Any],
) -> None:
    state_path = service_root / SUPERVISOR_STATE_FILENAME
    if not state_path.is_file():
        entry_names = {path.name for path in service_root.iterdir()}
        if entry_names - {"attempt_002_collection_supervisor.lock"}:
            raise ChallengeAttempt002CollectionError(
                "existing service root lacks attempt-002 supervisor state"
            )
        return
    state = _load_json(state_path)
    checks = {
        "schema": state.get("schema_version")
        == COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION,
        "attempt": state.get("attempt_id") == protocol["attempt_id"],
        "commit": state.get("implementation_commit")
        == config.implementation_commit,
        "protocol": state.get("protocol_sha256")
        == config.expected_protocol_sha256.lower(),
        "authorization": state.get("operator_authorization_sha256")
        == config.expected_operator_authorization_sha256.lower(),
        "collector": state.get("collector_protocol_sha256")
        == config.expected_collector_protocol_sha256.lower(),
        "feature": state.get("feature_contract_sha256")
        == config.expected_feature_contract_sha256.lower(),
        "target": state.get("service_root") == str(service_root),
        "outcomes": state.get(
            "outcomes_resolution_labels_or_pnl_opened"
        )
        is False,
        "safety": state.get("safety") == SAFE_FALSES,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeAttempt002CollectionError(
            "existing attempt-002 service root cannot resume: "
            + ",".join(blockers)
        )


def _finish_state(
    *,
    state_path: Path,
    state: dict[str, Any],
    progress: Mapping[str, Any],
    status: str,
    collector_pid: int | None,
) -> dict[str, Any]:
    state.update(
        {
            "status": status,
            "collector_pid": collector_pid,
            "attempted_market_count": progress["attempted_market_count"],
            "quality_valid_market_count": progress[
                "quality_valid_market_count"
            ],
            "remaining_quality_valid_market_count": progress[
                "remaining_quality_valid_market_count"
            ],
            "exclusion_reason_distribution": progress[
                "exclusion_reason_distribution"
            ],
            "collection_complete": status == "quality_valid_target_reached",
            "fail_closed": status == "attempt_cap_exhausted_fail_closed",
            "outcomes_resolution_labels_or_pnl_opened": False,
            "safety": SAFE_FALSES,
        }
    )
    _write_json_atomic(state_path, state)
    return state


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise ChallengeAttempt002CollectionError(f"{label} is missing")
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ChallengeAttempt002CollectionError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeAttempt002CollectionError(
            f"JSON object required: {path}"
        )
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


__all__ = [
    "Attempt002CollectionConfig",
    "ATTEMPT_002_PROTOCOL_SHA256",
    "BATCH_ROUND_COUNT",
    "ChallengeAttempt002CollectionError",
    "COLLECTOR_PROTOCOL_SHA256",
    "COLLECTION_BATCH_REPORT_SCHEMA_VERSION",
    "COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION",
    "FEATURE_CONTRACT_SHA256",
    "format_attempt_002_batch_comment",
    "MAXIMUM_ATTEMPTED_COUNT",
    "MAXIMUM_BATCH_COUNT",
    "TARGET_QUALITY_VALID_COUNT",
    "preflight_attempt_002_collection",
    "run_attempt_002_collection",
    "summarize_attempt_002_collection",
]
