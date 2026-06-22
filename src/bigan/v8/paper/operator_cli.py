"""One-command operator workflow for v8 read-only paper runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.paper.alert_delivery import (
    DEFAULT_ALERT_DELIVERY_CREATED_AT,
    GitHubCommentDeliveryConfig,
    GitHubPaperCommentDeliveryResult,
    deliver_github_paper_comment,
)
from bigan.v8.paper.contracts import json_ready
from bigan.v8.paper.feed import ReadOnlyMarketFeed
from bigan.v8.paper.live_feed import (
    DEFAULT_PUBLIC_INSTRUMENT_ID,
    DEFAULT_PUBLIC_PROVIDER_ENDPOINT,
    DEFAULT_PUBLIC_PROVIDER_NAME,
    DETERMINISTIC_REPLAY_FEED_MODE,
    LIVE_READONLY_FEED_MODE,
    FeedMode,
    LiveReadOnlyFeedConfig,
)
from bigan.v8.paper.live_feed_adapters import create_public_live_readonly_feed
from bigan.v8.paper.observability import (
    DEFAULT_OBSERVABILITY_CREATED_AT,
    PaperObservabilityResult,
    summarize_paper_run,
)
from bigan.v8.paper.soak import (
    DEFAULT_READONLY_SHADOW_CREATED_AT,
    ReadOnlyShadowSoakConfig,
    run_readonly_shadow_soak,
)

PAPER_OPERATOR_CLI_PHASE = "paper_operator_cli"
PAPER_OPERATOR_SCHEMA_VERSION = "bigan-v8-paper-operator-cli-v1"
DEFAULT_OPERATOR_CREATED_AT = "2026-06-22T06:00:00Z"

OperatorPostMode = Literal["dry_run", "gh_command", "direct_comment"]
OperatorStatus = Literal[
    "completed_continue_paper",
    "completed_warning",
    "completed_blocked_fail_closed",
    "failed_fail_closed",
    "operator_stopped",
]


class PaperOperatorCLIError(RuntimeError):
    """Raised when the operator CLI fails closed."""


@dataclass(frozen=True, slots=True)
class PaperOperatorRunConfig:
    """Configuration for one bounded 24h-capable paper operator run."""

    run_id: str
    output_dir: Path | str
    repo_full_name: str
    issue_number: int
    post_mode: OperatorPostMode = "dry_run"
    duration_seconds: int = 24 * 60 * 60
    feed_event_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 60
    summary_interval_seconds: int = 300
    feed_mode: FeedMode = DETERMINISTIC_REPLAY_FEED_MODE
    provider_name: str = DEFAULT_PUBLIC_PROVIDER_NAME
    provider_endpoint: str = DEFAULT_PUBLIC_PROVIDER_ENDPOINT
    instrument_id: str = DEFAULT_PUBLIC_INSTRUMENT_ID
    request_timeout_seconds: float = 10.0
    max_reconnect_attempts: int = 3
    max_stale_seconds: float = 120.0
    created_at: str = DEFAULT_OPERATOR_CREATED_AT
    soak_created_at: str = DEFAULT_READONLY_SHADOW_CREATED_AT
    observability_created_at: str = DEFAULT_OBSERVABILITY_CREATED_AT
    alert_delivery_created_at: str = DEFAULT_ALERT_DELIVERY_CREATED_AT
    overwrite_existing: bool = False
    stop_after_events: int | None = None
    inject_degradation: bool = False
    max_feed_gap_seconds: float = 120.0
    max_event_lag_seconds: float = 10.0
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.repo_full_name.strip() or "/" not in self.repo_full_name:
            raise ValueError("repo_full_name must be owner/repo")
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive")
        if self.post_mode not in ("dry_run", "gh_command", "direct_comment"):
            raise ValueError("post_mode must be dry_run, gh_command, or direct_comment")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.feed_event_interval_seconds <= 0:
            raise ValueError("feed_event_interval_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.summary_interval_seconds <= 0:
            raise ValueError("summary_interval_seconds must be positive")
        if self.feed_mode not in (
            DETERMINISTIC_REPLAY_FEED_MODE,
            LIVE_READONLY_FEED_MODE,
        ):
            raise ValueError("feed_mode must be deterministic-replay or live-readonly")
        if not self.provider_name.strip():
            raise ValueError("provider_name is required")
        if not self.provider_endpoint.strip():
            raise ValueError("provider_endpoint is required")
        if not self.instrument_id.strip():
            raise ValueError("instrument_id is required")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be non-negative")
        if self.max_stale_seconds <= 0.0:
            raise ValueError("max_stale_seconds must be positive")
        if self.stop_after_events is not None and self.stop_after_events <= 0:
            raise ValueError("stop_after_events must be positive when provided")
        if self.max_feed_gap_seconds <= 0.0:
            raise ValueError("max_feed_gap_seconds must be positive")
        if self.max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        if self.broker_exchange_write_enabled:
            raise PaperOperatorCLIError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise PaperOperatorCLIError("live exchange writes are forbidden")
        if self.paper_only is not True:
            raise PaperOperatorCLIError("operator run must be paper-only")
        if self.capital_at_risk is not False:
            raise PaperOperatorCLIError("operator run cannot put capital at risk")

    @property
    def operator_run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    @property
    def paper_run_dir(self) -> Path:
        return self.operator_run_dir / "paper_run"

    @property
    def observability_dir(self) -> Path:
        return self.operator_run_dir / "observability"

    @property
    def github_comment_dir(self) -> Path:
        return self.operator_run_dir / "github_comment"

    @property
    def manifest_path(self) -> Path:
        return self.operator_run_dir / "operator_run_manifest.json"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class PaperOperatorRunResult:
    """Result and output handles for one paper operator run."""

    run_id: str
    operator_run_dir: Path
    paper_run_dir: Path
    observability_dir: Path
    github_comment_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    console_summary: dict[str, Any]
    observability_result: PaperObservabilityResult
    comment_result: GitHubPaperCommentDeliveryResult


def run_24h_paper_operator(
    *,
    config: PaperOperatorRunConfig,
    feed: ReadOnlyMarketFeed | None = None,
) -> PaperOperatorRunResult:
    """Run paper soak, observability, comment delivery, and manifest writing."""

    return _run_24h_paper_operator_impl(
        config=config,
        feed=feed,
        _after_paper_run_fault_injection_hook_for_tests=None,
    )


def _run_24h_paper_operator_with_fault_injection_for_tests(
    *,
    config: PaperOperatorRunConfig,
    feed: ReadOnlyMarketFeed | None = None,
    _after_paper_run_fault_injection_hook_for_tests: Callable[[Path], None],
) -> PaperOperatorRunResult:
    """Test-only fault injection path; not part of the operator API."""

    return _run_24h_paper_operator_impl(
        config=config,
        feed=feed,
        _after_paper_run_fault_injection_hook_for_tests=(
            _after_paper_run_fault_injection_hook_for_tests
        ),
    )


def _run_24h_paper_operator_impl(
    *,
    config: PaperOperatorRunConfig,
    feed: ReadOnlyMarketFeed | None,
    _after_paper_run_fault_injection_hook_for_tests: (
        Callable[[Path], None] | None
    ),
) -> PaperOperatorRunResult:
    """Run the workflow; optional mutation hook is private and test-only."""

    operator_run_dir = config.operator_run_dir
    _prepare_operator_run_dir(config)
    stage = "paper_run"
    try:
        resolved_feed = _resolve_operator_feed(config=config, feed=feed)
        run_readonly_shadow_soak(
            config=ReadOnlyShadowSoakConfig(
                run_id=config.run_id,
                output_dir=operator_run_dir,
                run_dir_override=config.paper_run_dir,
                duration_seconds=config.duration_seconds,
                feed_event_interval_seconds=config.feed_event_interval_seconds,
                heartbeat_interval_seconds=config.heartbeat_interval_seconds,
                summary_interval_seconds=config.summary_interval_seconds,
                created_at=config.soak_created_at,
                overwrite_existing=False,
                stop_after_events=config.stop_after_events,
                inject_degradation=config.inject_degradation,
                max_feed_gap_seconds=config.max_feed_gap_seconds,
                max_event_lag_seconds=config.max_event_lag_seconds,
                broker_exchange_write_enabled=config.broker_exchange_write_enabled,
                live_exchange_write_enabled=config.live_exchange_write_enabled,
                paper_only=config.paper_only,
                capital_at_risk=config.capital_at_risk,
            ),
            feed=resolved_feed,
        )

        if _after_paper_run_fault_injection_hook_for_tests is not None:
            _after_paper_run_fault_injection_hook_for_tests(config.paper_run_dir)

        stage = "observability"
        observability_result = summarize_paper_run(
            run_dir=config.paper_run_dir,
            output_dir=config.observability_dir,
            created_at=config.observability_created_at,
            overwrite_existing=False,
        )

        stage = "github_comment"
        comment_result = deliver_github_paper_comment(
            observability_dir=config.observability_dir,
            config=GitHubCommentDeliveryConfig(
                repo_full_name=config.repo_full_name,
                issue_number=config.issue_number,
                output_dir=config.github_comment_dir,
                post_mode=config.post_mode,
                created_at=config.alert_delivery_created_at,
                overwrite_existing=False,
            ),
        )

        stage = "manifest"
        manifest = _operator_manifest(
            config=config,
            observability_result=observability_result,
            comment_result=comment_result,
        )
        _write_json(config.manifest_path, manifest)
        console_summary = _console_summary(
            manifest=manifest,
            observability_result=observability_result,
            comment_result=comment_result,
        )
        return PaperOperatorRunResult(
            run_id=config.run_id,
            operator_run_dir=operator_run_dir,
            paper_run_dir=config.paper_run_dir,
            observability_dir=config.observability_dir,
            github_comment_dir=config.github_comment_dir,
            manifest_path=config.manifest_path,
            manifest=manifest,
            console_summary=console_summary,
            observability_result=observability_result,
            comment_result=comment_result,
        )
    except Exception as exc:
        _write_failure_manifest(
            config=config,
            stage=stage,
            error=exc,
        )
        raise PaperOperatorCLIError(
            f"paper operator run failed closed during {stage}: {exc}"
        ) from exc


def _resolve_operator_feed(
    *,
    config: PaperOperatorRunConfig,
    feed: ReadOnlyMarketFeed | None,
) -> ReadOnlyMarketFeed | None:
    if config.feed_mode == DETERMINISTIC_REPLAY_FEED_MODE:
        return feed
    if config.feed_mode != LIVE_READONLY_FEED_MODE:
        raise PaperOperatorCLIError(f"unsupported feed_mode: {config.feed_mode}")
    if feed is not None:
        if getattr(feed, "feed_mode", None) != LIVE_READONLY_FEED_MODE:
            raise PaperOperatorCLIError(
                "live-readonly mode requires a live-readonly feed adapter; "
                "refusing deterministic replay fallback"
            )
        if getattr(feed, "read_only", False) is not True:
            raise PaperOperatorCLIError("live feed must be read-only")
        if getattr(feed, "write_capable", True) is not False:
            raise PaperOperatorCLIError("write-capable live feed is forbidden")
        return feed
    return create_public_live_readonly_feed(
        LiveReadOnlyFeedConfig(
            provider_name=config.provider_name,
            provider_endpoint=config.provider_endpoint,
            instrument_id=config.instrument_id,
            poll_interval_seconds=float(config.feed_event_interval_seconds),
            request_timeout_seconds=config.request_timeout_seconds,
            max_reconnect_attempts=config.max_reconnect_attempts,
            max_allowed_gap_seconds=config.max_feed_gap_seconds,
            max_event_lag_seconds=config.max_event_lag_seconds,
            max_stale_seconds=config.max_stale_seconds,
            expected_wall_clock_duration_seconds=config.duration_seconds,
            started_at=config.soak_created_at,
        )
    )


def _prepare_operator_run_dir(config: PaperOperatorRunConfig) -> None:
    run_dir = config.operator_run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"operator run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def _operator_manifest(
    *,
    config: PaperOperatorRunConfig,
    observability_result: PaperObservabilityResult,
    comment_result: GitHubPaperCommentDeliveryResult,
) -> dict[str, Any]:
    summary = _read_json(config.paper_run_dir / "paper_run_summary.json")
    phase5 = _read_json(config.paper_run_dir / "phase5_safety_layer_report.json")
    phase6_path = _phase6_report_path(config.paper_run_dir)
    live_metadata_path = config.paper_run_dir / "live_feed_metadata.json"
    live_health_path = config.paper_run_dir / "live_feed_health_report.json"
    live_metadata = _read_json_if_exists(live_metadata_path)
    live_health = _read_json_if_exists(live_health_path)
    report = observability_result.report
    status = _operator_status(summary, report)
    reason_codes = _reason_codes(summary, report, status)
    feed_mode = str(summary.get("feed_mode", config.feed_mode))
    return {
        "schema_version": PAPER_OPERATOR_SCHEMA_VERSION,
        "run_id": config.run_id,
        "phase": PAPER_OPERATOR_CLI_PHASE,
        "created_at": config.created_at,
        "started_at": summary["started_at"],
        "ended_at": summary["ended_at"],
        "duration_seconds": summary["duration_seconds"],
        "configured_duration_seconds": config.duration_seconds,
        "operator_run_dir": str(config.operator_run_dir),
        "paper_run_dir": str(config.paper_run_dir),
        "observability_dir": str(config.observability_dir),
        "github_comment_dir": str(config.github_comment_dir),
        "feed_mode": feed_mode,
        "real_live_data": feed_mode == LIVE_READONLY_FEED_MODE,
        "deterministic_replay": feed_mode == DETERMINISTIC_REPLAY_FEED_MODE,
        "provider_name": summary.get(
            "provider_name",
            live_metadata.get("provider_name", config.provider_name),
        ),
        "provider_endpoint_or_endpoint_type": live_metadata.get(
            "provider_endpoint_or_endpoint_type",
            config.provider_endpoint,
        ),
        "instrument_id": summary.get(
            "instrument_id",
            live_metadata.get("instrument_id", config.instrument_id),
        ),
        "paper_summary_path": str(config.paper_run_dir / "paper_run_summary.json"),
        "observability_report_path": str(
            observability_result.artifact_paths["observability_report"]
        ),
        "operator_summary_path": str(
            observability_result.artifact_paths["operator_summary"]
        ),
        "github_comment_payload_path": str(
            comment_result.artifact_paths["payload"]
        ),
        "github_comment_body_path": str(
            comment_result.artifact_paths["comment_body"]
        ),
        "github_comment_gh_command_path": (
            None
            if "gh_command" not in comment_result.artifact_paths
            else str(comment_result.artifact_paths["gh_command"])
        ),
        "paper_summary_sha256": _sha256_file(
            config.paper_run_dir / "paper_run_summary.json"
        ),
        "paper_bundle_sha256": _sha256_file(
            config.paper_run_dir / "paper_bundle_manifest.json"
        ),
        "observability_report_sha256": _sha256_file(
            observability_result.artifact_paths["observability_report"]
        ),
        "operator_summary_sha256": _sha256_file(
            observability_result.artifact_paths["operator_summary"]
        ),
        "github_comment_payload_sha256": _sha256_file(
            comment_result.artifact_paths["payload"]
        ),
        "phase5_passed": bool(summary["phase5_passed"]),
        "phase5_kill_switch_triggered": bool(
            summary["phase5_kill_switch_triggered"]
        ),
        "phase5_status": "passed"
        if bool(summary["phase5_passed"])
        else "blocked_fail_closed",
        "phase6_deployment_status": str(summary["phase6_deployment_status"]),
        "phase6_report_sha256": _sha256_file(phase6_path),
        "live_feed_metadata_path": str(live_metadata_path)
        if live_metadata_path.exists()
        else None,
        "live_feed_metadata_sha256": _optional_sha256_file(live_metadata_path),
        "live_feed_health_path": str(live_health_path)
        if live_health_path.exists()
        else None,
        "live_feed_health_sha256": _optional_sha256_file(live_health_path),
        "provider_disconnect_count": summary.get(
            "provider_disconnect_count",
            live_health.get("provider_disconnect_count", 0),
        ),
        "provider_reconnect_count": summary.get(
            "provider_reconnect_count",
            live_health.get("provider_reconnect_count", 0),
        ),
        "provider_error_count": summary.get(
            "provider_error_count",
            live_health.get("provider_error_count", 0),
        ),
        "stale_event_count": summary.get(
            "stale_event_count",
            live_health.get("stale_event_count", 0),
        ),
        "empty_response_count": summary.get(
            "empty_response_count",
            live_health.get("empty_response_count", 0),
        ),
        "rate_limit_count": summary.get(
            "rate_limit_count",
            live_health.get("rate_limit_count", 0),
        ),
        "feed_health_passed": bool(summary["feed_health_passed"]),
        "feed_health_status": report.feed_health_status,
        "alert_count": report.alert_count,
        "critical_alert_count": report.alert_severity_counts["critical"],
        "warning_alert_count": report.alert_severity_counts["warning"],
        "operator_recommendation": report.operator_recommendation,
        "stop_reason": summary["stop_reason"],
        "paper_only": report.paper_only,
        "capital_at_risk": report.capital_at_risk,
        "broker_exchange_write_enabled": report.broker_exchange_write_enabled,
        "live_exchange_write_enabled": report.live_exchange_write_enabled,
        "status": status,
        "reason_codes": reason_codes,
        "direct_pnl_optimization": False,
        "shadow_return_used_for_training": False,
        "capital_deployment_allowed": False,
        "live_deployment_allowed": False,
        "broker_exchange_write_allowed": False,
        "config": config.to_dict(),
        "phase5_report_sha256": _sha256_file(
            config.paper_run_dir / "phase5_safety_layer_report.json"
        ),
        "phase5_kill_switch_reason_codes": list(
            phase5.get("safety_action", {}).get("reason_codes", [])
        ),
    }


def _console_summary(
    *,
    manifest: dict[str, Any],
    observability_result: PaperObservabilityResult,
    comment_result: GitHubPaperCommentDeliveryResult,
) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "operator_run_dir": manifest["operator_run_dir"],
        "run_dir": manifest["paper_run_dir"],
        "feed_mode": manifest["feed_mode"],
        "real_live_data": manifest["real_live_data"],
        "deterministic_replay": manifest["deterministic_replay"],
        "provider_name": manifest["provider_name"],
        "instrument_id": manifest["instrument_id"],
        "paper_summary_path": manifest["paper_summary_path"],
        "phase5_status": manifest["phase5_status"],
        "phase6_deployment_status": manifest["phase6_deployment_status"],
        "feed_health_status": manifest["feed_health_status"],
        "alert_count": manifest["alert_count"],
        "critical_alert_count": manifest["critical_alert_count"],
        "operator_recommendation": manifest["operator_recommendation"],
        "observability_report_path": str(
            observability_result.artifact_paths["observability_report"]
        ),
        "operator_summary_path": str(
            observability_result.artifact_paths["operator_summary"]
        ),
        "comment_body_path": str(comment_result.artifact_paths["comment_body"]),
        "gh_command_path": (
            None
            if "gh_command" not in comment_result.artifact_paths
            else str(comment_result.artifact_paths["gh_command"])
        ),
        "paper_only": manifest["paper_only"],
        "capital_at_risk": manifest["capital_at_risk"],
        "status": manifest["status"],
    }


def _write_failure_manifest(
    *,
    config: PaperOperatorRunConfig,
    stage: str,
    error: Exception,
) -> None:
    if not config.operator_run_dir.exists():
        return
    manifest = {
        "schema_version": PAPER_OPERATOR_SCHEMA_VERSION,
        "run_id": config.run_id,
        "phase": PAPER_OPERATOR_CLI_PHASE,
        "created_at": config.created_at,
        "started_at": config.soak_created_at,
        "ended_at": config.soak_created_at,
        "duration_seconds": 0,
        "configured_duration_seconds": config.duration_seconds,
        "paper_run_dir": str(config.paper_run_dir),
        "observability_dir": str(config.observability_dir),
        "github_comment_dir": str(config.github_comment_dir),
        "feed_mode": config.feed_mode,
        "real_live_data": config.feed_mode == LIVE_READONLY_FEED_MODE,
        "deterministic_replay": config.feed_mode == DETERMINISTIC_REPLAY_FEED_MODE,
        "provider_name": config.provider_name,
        "provider_endpoint_or_endpoint_type": config.provider_endpoint,
        "instrument_id": config.instrument_id,
        "paper_only": config.paper_only,
        "capital_at_risk": config.capital_at_risk,
        "broker_exchange_write_enabled": config.broker_exchange_write_enabled,
        "live_exchange_write_enabled": config.live_exchange_write_enabled,
        "status": "failed_fail_closed",
        "reason_codes": _failure_reason_codes(stage=stage, error=error),
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "capital_deployment_allowed": False,
        "live_deployment_allowed": False,
        "broker_exchange_write_allowed": False,
    }
    _write_json(config.manifest_path, manifest)


def _failure_reason_codes(*, stage: str, error: Exception) -> list[str]:
    codes = [f"{stage}_failed"]
    codes.extend(str(code) for code in getattr(error, "reason_codes", ()) or ())
    return list(dict.fromkeys(codes))


def _operator_status(
    summary: dict[str, Any],
    report: Any,
) -> OperatorStatus:
    if summary.get("stop_reason") == "operator_stop":
        return "operator_stopped"
    if (
        report.operator_recommendation == "continue_paper_run"
        and report.alert_severity_counts["critical"] == 0
        and summary.get("phase6_deployment_status") == "approved_for_staged_live"
    ):
        return "completed_continue_paper"
    if report.operator_recommendation == "investigate_warning":
        return "completed_warning"
    return "completed_blocked_fail_closed"


def _reason_codes(
    summary: dict[str, Any],
    report: Any,
    status: OperatorStatus,
) -> list[str]:
    codes = set(summary.get("phase5_reason_codes", []) or [])
    codes.update(summary.get("feed_health_reason_codes", []) or [])
    codes.update(alert["code"] for alert in report.alerts)
    if summary.get("stop_reason") == "operator_stop":
        codes.add("operator_stop")
    if status == "completed_blocked_fail_closed":
        codes.add("operator_run_blocked_fail_closed")
    return sorted(str(code) for code in codes)


def _phase6_report_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("phase6_cicd_pipeline_report_*.json"))
    if len(matches) != 1:
        raise PaperOperatorCLIError(
            "expected exactly one Phase 6 report in paper run directory, "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_run_id(run_id: str) -> None:
    if not run_id.strip():
        raise ValueError("run_id is required")
    path = Path(run_id)
    if path.name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single path segment")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_file(path)
