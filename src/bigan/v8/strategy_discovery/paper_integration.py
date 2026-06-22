"""Paper-only integration runner for Strategy Discovery candidates."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.paper import (
    DEFAULT_ALERT_DELIVERY_CREATED_AT,
    DEFAULT_OBSERVABILITY_CREATED_AT,
    DEFAULT_READONLY_SHADOW_CREATED_AT,
    DeterministicReplayFeed,
    GitHubCommentDeliveryConfig,
    ReadOnlyShadowSoakConfig,
    deliver_github_paper_comment,
    run_readonly_shadow_soak,
    summarize_paper_run,
    synthetic_readonly_feed_events,
)
from bigan.v8.paper.contracts import canonical_payload_sha256, json_ready
from bigan.v8.paper.feed import ReadOnlyMarketFeed
from bigan.v8.strategy_discovery.contracts import (
    DEFAULT_STRATEGY_DISCOVERY_CREATED_AT,
    STRATEGY_DISCOVERY_SCHEMA_VERSION,
    CandidateStatus,
    StrategyCandidate,
    StrategyDiscoveryError,
    build_strategy_candidate_manifest,
    raw_candidate_id,
    strategy_candidate_from_mapping,
)
from bigan.v8.strategy_discovery.registry import (
    StrategyCandidateRegistryEntry,
    build_strategy_candidate_registry,
)

STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE = (
    "strategy_discovery_paper_integration"
)

StrategyReplayPostMode = Literal["dry_run", "gh_command", "direct_comment"]


@dataclass(frozen=True, slots=True)
class StrategyCandidateReplayConfig:
    """Configuration for one strategy candidate replay batch."""

    batch_id: str
    output_dir: Path | str
    repo_full_name: str
    issue_number: int
    post_mode: StrategyReplayPostMode = "dry_run"
    duration_seconds: int = 300
    feed_event_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 30
    summary_interval_seconds: int = 120
    created_at: str = DEFAULT_STRATEGY_DISCOVERY_CREATED_AT
    soak_created_at: str = DEFAULT_READONLY_SHADOW_CREATED_AT
    observability_created_at: str = DEFAULT_OBSERVABILITY_CREATED_AT
    alert_delivery_created_at: str = DEFAULT_ALERT_DELIVERY_CREATED_AT
    overwrite_existing: bool = False
    max_feed_gap_seconds: float = 120.0
    max_event_lag_seconds: float = 10.0
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("batch_id is required")
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
        if self.max_feed_gap_seconds <= 0.0:
            raise ValueError("max_feed_gap_seconds must be positive")
        if self.max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        if self.broker_exchange_write_enabled:
            raise StrategyDiscoveryError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise StrategyDiscoveryError("live exchange writes are forbidden")
        if self.paper_only is not True:
            raise StrategyDiscoveryError("strategy replay must be paper-only")
        if self.capital_at_risk is not False:
            raise StrategyDiscoveryError("strategy replay cannot put capital at risk")

    @property
    def batch_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class StrategyCandidateReplayBatchResult:
    """Result handles from a strategy candidate replay batch."""

    batch_id: str
    output_dir: Path
    candidate_summaries: list[dict[str, Any]]
    batch_manifest: dict[str, Any]
    ranking: dict[str, Any]
    console_summary: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_strategy_candidate_replay_batch(
    *,
    candidates: Iterable[StrategyCandidate | Mapping[str, Any]],
    config: StrategyCandidateReplayConfig,
) -> StrategyCandidateReplayBatchResult:
    """Replay strategy candidates through the paper-only safety pipeline."""

    output_path = config.batch_dir
    if output_path.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"strategy candidate output_dir already exists: {output_path}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    registry_entries: list[StrategyCandidateRegistryEntry] = []
    for index, raw_candidate in enumerate(candidates, start=1):
        summary, entry = _run_one_candidate(
            raw_candidate=raw_candidate,
            index=index,
            config=config,
        )
        summaries.append(summary)
        registry_entries.append(entry)

    registry = build_strategy_candidate_registry(
        batch_id=config.batch_id,
        entries=registry_entries,
        created_at=config.created_at,
    )
    ranking = _ranking_report(
        batch_id=config.batch_id,
        summaries=summaries,
        created_at=config.created_at,
    )
    artifact_paths = {
        "registry": output_path / "strategy_candidate_registry.json",
        "ranking_json": output_path / "strategy_candidate_ranking.json",
        "ranking_md": output_path / "strategy_candidate_ranking.md",
        "batch_summary_md": output_path / "strategy_candidate_batch_summary.md",
        "batch_manifest": output_path / "strategy_candidate_batch_manifest.json",
    }
    _write_json(artifact_paths["registry"], registry.to_dict())
    _write_json(artifact_paths["ranking_json"], ranking)
    _write_text(artifact_paths["ranking_md"], _ranking_markdown(ranking))
    _write_text(
        artifact_paths["batch_summary_md"],
        _batch_summary_markdown(config.batch_id, summaries),
    )
    batch_manifest = _batch_manifest(
        config=config,
        summaries=summaries,
        artifact_paths=artifact_paths,
    )
    _write_json(artifact_paths["batch_manifest"], batch_manifest)
    console_summary = _console_summary(batch_manifest, artifact_paths)
    return StrategyCandidateReplayBatchResult(
        batch_id=config.batch_id,
        output_dir=output_path,
        candidate_summaries=summaries,
        batch_manifest=batch_manifest,
        ranking=ranking,
        console_summary=console_summary,
        artifact_paths=artifact_paths,
    )


def load_strategy_candidates_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load raw strategy candidates from a deterministic JSONL file."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_one_candidate(
    *,
    raw_candidate: StrategyCandidate | Mapping[str, Any],
    index: int,
    config: StrategyCandidateReplayConfig,
) -> tuple[dict[str, Any], StrategyCandidateRegistryEntry]:
    raw_payload = (
        raw_candidate.to_dict()
        if isinstance(raw_candidate, StrategyCandidate)
        else dict(raw_candidate)
    )
    candidate_id = raw_candidate_id(raw_payload, index)
    candidate_dir = config.batch_dir / candidate_id
    candidate_dir.mkdir(parents=True)
    try:
        candidate = (
            raw_candidate
            if isinstance(raw_candidate, StrategyCandidate)
            else strategy_candidate_from_mapping(raw_payload)
        )
        return _run_valid_candidate(
            candidate=candidate,
            candidate_dir=candidate_dir,
            config=config,
        )
    except Exception as exc:
        summary = _invalid_candidate_summary(
            raw_payload=raw_payload,
            candidate_id=candidate_id,
            candidate_dir=candidate_dir,
            error=exc,
            created_at=config.created_at,
        )
        summary_path = candidate_dir / "candidate_replay_summary.json"
        _write_json(summary_path, summary)
        entry = StrategyCandidateRegistryEntry(
            candidate_id=candidate_id,
            status="candidate_invalid",
            candidate_sha256=None,
            candidate_manifest_sha256=None,
            candidate_manifest_path=None,
            candidate_replay_summary_path=str(summary_path),
            operator_recommendation=None,
            phase6_deployment_status=None,
            critical_alert_count=1,
            warning_alert_count=0,
            artifact_hashes={"candidate_replay_summary": _sha256_file(summary_path)},
            reason_codes=list(summary["reason_codes"]),
        )
        return summary, entry


def _run_valid_candidate(
    *,
    candidate: StrategyCandidate,
    candidate_dir: Path,
    config: StrategyCandidateReplayConfig,
) -> tuple[dict[str, Any], StrategyCandidateRegistryEntry]:
    paper_run_dir = candidate_dir / "paper_run"
    observability_dir = candidate_dir / "observability"
    github_comment_dir = candidate_dir / "github_comment"
    pipeline_config = _candidate_pipeline_config(candidate, config)
    manifest = build_strategy_candidate_manifest(
        candidate=candidate,
        paper_pipeline_config=pipeline_config,
        created_at=config.created_at,
    )
    manifest_path = candidate_dir / "candidate_manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    run_readonly_shadow_soak(
        config=ReadOnlyShadowSoakConfig(
            run_id=candidate.candidate_id,
            output_dir=candidate_dir,
            run_dir_override=paper_run_dir,
            duration_seconds=pipeline_config["duration_seconds"],
            feed_event_interval_seconds=pipeline_config[
                "feed_event_interval_seconds"
            ],
            heartbeat_interval_seconds=pipeline_config[
                "heartbeat_interval_seconds"
            ],
            summary_interval_seconds=pipeline_config["summary_interval_seconds"],
            created_at=config.soak_created_at,
            overwrite_existing=False,
            inject_degradation=bool(pipeline_config["inject_degradation"]),
            max_feed_gap_seconds=config.max_feed_gap_seconds,
            max_event_lag_seconds=config.max_event_lag_seconds,
            broker_exchange_write_enabled=config.broker_exchange_write_enabled,
            live_exchange_write_enabled=config.live_exchange_write_enabled,
            paper_only=config.paper_only,
            capital_at_risk=config.capital_at_risk,
        ),
        feed=_candidate_feed(candidate, pipeline_config),
    )
    observability_result = summarize_paper_run(
        run_dir=paper_run_dir,
        output_dir=observability_dir,
        created_at=config.observability_created_at,
        overwrite_existing=False,
    )
    comment_result = deliver_github_paper_comment(
        observability_dir=observability_dir,
        config=GitHubCommentDeliveryConfig(
            repo_full_name=config.repo_full_name,
            issue_number=config.issue_number,
            output_dir=github_comment_dir,
            post_mode=config.post_mode,
            created_at=config.alert_delivery_created_at,
            overwrite_existing=False,
        ),
    )
    summary = _candidate_replay_summary(
        candidate=candidate,
        candidate_dir=candidate_dir,
        paper_run_dir=paper_run_dir,
        observability_dir=observability_dir,
        github_comment_dir=github_comment_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        observability_report=observability_result.report.to_dict(),
        comment_payload=comment_result.payload.to_dict(),
        config=config,
    )
    summary_path = candidate_dir / "candidate_replay_summary.json"
    _write_json(summary_path, summary)
    summary["candidate_replay_summary_path"] = str(summary_path)
    _write_json(summary_path, summary)
    entry = StrategyCandidateRegistryEntry(
        candidate_id=candidate.candidate_id,
        status=summary["status"],
        candidate_sha256=candidate.candidate_sha256,
        candidate_manifest_sha256=manifest.manifest_sha256,
        candidate_manifest_path=str(manifest_path),
        candidate_replay_summary_path=str(summary_path),
        operator_recommendation=summary["operator_recommendation"],
        phase6_deployment_status=summary["phase6_deployment_status"],
        critical_alert_count=summary["critical_alert_count"],
        warning_alert_count=summary["warning_alert_count"],
        artifact_hashes=dict(summary["artifact_hashes"]),
        reason_codes=list(summary["reason_codes"]),
    )
    return summary, entry


def _candidate_pipeline_config(
    candidate: StrategyCandidate,
    config: StrategyCandidateReplayConfig,
) -> dict[str, Any]:
    policy = dict(candidate.policy_config)
    execution = dict(candidate.execution_config)
    risk = dict(candidate.risk_config)
    return {
        "duration_seconds": int(policy.get("duration_seconds", config.duration_seconds)),
        "feed_event_interval_seconds": int(
            execution.get(
                "feed_event_interval_seconds",
                config.feed_event_interval_seconds,
            )
        ),
        "heartbeat_interval_seconds": int(
            execution.get(
                "heartbeat_interval_seconds",
                config.heartbeat_interval_seconds,
            )
        ),
        "summary_interval_seconds": int(
            execution.get("summary_interval_seconds", config.summary_interval_seconds)
        ),
        "inject_degradation": bool(
            policy.get("inject_degradation", risk.get("inject_degradation", False))
        ),
        "feed_anomaly": str(execution.get("feed_anomaly", "none")),
        "candidate_family": candidate.candidate_family,
        "strategy_name": candidate.strategy_name,
    }


def _candidate_feed(
    candidate: StrategyCandidate,
    pipeline_config: Mapping[str, Any],
) -> ReadOnlyMarketFeed | None:
    if pipeline_config.get("feed_anomaly") != "gap":
        return None
    events = list(
        synthetic_readonly_feed_events(
            row_count=5,
            interval_ms=int(pipeline_config["feed_event_interval_seconds"]) * 1000,
            instrument_id=candidate.expected_instruments[0],
        )
    )
    events[2] = _replace_event_gap(events[1], events[2], gap_ms=180_000)
    return DeterministicReplayFeed(events=tuple(events))


def _replace_event_gap(previous: Any, current: Any, *, gap_ms: int) -> Any:
    from dataclasses import replace

    return replace(
        current,
        event_ts=previous.event_ts + gap_ms,
        received_ts=previous.event_ts + gap_ms + 250,
    )


def _candidate_replay_summary(
    *,
    candidate: StrategyCandidate,
    candidate_dir: Path,
    paper_run_dir: Path,
    observability_dir: Path,
    github_comment_dir: Path,
    manifest: Any,
    manifest_path: Path,
    observability_report: dict[str, Any],
    comment_payload: dict[str, Any],
    config: StrategyCandidateReplayConfig,
) -> dict[str, Any]:
    paper_summary_path = paper_run_dir / "paper_run_summary.json"
    phase6_path = _phase6_report_path(paper_run_dir)
    paper_summary = _read_json(paper_summary_path)
    status = _candidate_status(paper_summary, observability_report)
    reason_codes = _candidate_reason_codes(status, paper_summary, observability_report)
    artifact_hashes = {
        "candidate_manifest": _sha256_file(manifest_path),
        "paper_run_summary": _sha256_file(paper_summary_path),
        "paper_bundle_manifest": _sha256_file(
            paper_run_dir / "paper_bundle_manifest.json"
        ),
        "phase5_report": _sha256_file(
            paper_run_dir / "phase5_safety_layer_report.json"
        ),
        "phase6_report": _sha256_file(phase6_path),
        "observability_report": _sha256_file(
            observability_dir / "paper_observability_report.json"
        ),
        "operator_summary": _sha256_file(
            observability_dir / "paper_operator_summary.md"
        ),
        "github_comment_payload": _sha256_file(
            github_comment_dir / "github_paper_comment_payload.json"
        ),
    }
    return {
        "schema_version": STRATEGY_DISCOVERY_SCHEMA_VERSION,
        "phase": STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE,
        "batch_id": config.batch_id,
        "candidate_id": candidate.candidate_id,
        "candidate_dir": str(candidate_dir),
        "candidate_sha256": candidate.candidate_sha256,
        "candidate_manifest_sha256": manifest.manifest_sha256,
        "candidate_manifest_path": str(manifest_path),
        "paper_run_dir": str(paper_run_dir),
        "observability_dir": str(observability_dir),
        "github_comment_dir": str(github_comment_dir),
        "status": status,
        "reason_codes": reason_codes,
        "operator_recommendation": observability_report["operator_recommendation"],
        "phase5_passed": paper_summary["phase5_passed"],
        "phase5_kill_switch_triggered": paper_summary[
            "phase5_kill_switch_triggered"
        ],
        "phase6_deployment_status": paper_summary["phase6_deployment_status"],
        "feed_health_status": observability_report["feed_health_status"],
        "critical_alert_count": observability_report["alert_severity_counts"][
            "critical"
        ],
        "warning_alert_count": observability_report["alert_severity_counts"][
            "warning"
        ],
        "mean_net_return": observability_report["performance_metrics"][
            "mean_net_return"
        ],
        "cumulative_net_return": observability_report["performance_metrics"][
            "cumulative_net_return"
        ],
        "max_drawdown": observability_report["risk_metrics"]["max_drawdown"],
        "total_execution_cost": observability_report["performance_metrics"][
            "total_execution_cost"
        ],
        "cost_drift_ratio": observability_report["risk_metrics"]["cost_drift_ratio"],
        "pnl_drift": observability_report["risk_metrics"]["pnl_drift"],
        "regime_mismatch_rate": observability_report["risk_metrics"][
            "regime_mismatch_rate"
        ],
        "paper_only": observability_report["paper_only"],
        "capital_at_risk": observability_report["capital_at_risk"],
        "broker_exchange_write_enabled": observability_report[
            "broker_exchange_write_enabled"
        ],
        "live_exchange_write_enabled": observability_report[
            "live_exchange_write_enabled"
        ],
        "github_comment_payload_sha256": artifact_hashes["github_comment_payload"],
        "github_comment_body_path": str(
            github_comment_dir / "github_paper_comment.md"
        ),
        "github_comment_gh_command_path": (
            None
            if comment_payload["post_mode"] == "dry_run"
            else str(github_comment_dir / "github_paper_comment_gh_command.sh")
        ),
        "artifact_hashes": artifact_hashes,
        "created_at": config.created_at,
    }


def _invalid_candidate_summary(
    *,
    raw_payload: Mapping[str, Any],
    candidate_id: str,
    candidate_dir: Path,
    error: Exception,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": STRATEGY_DISCOVERY_SCHEMA_VERSION,
        "phase": STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "status": "candidate_invalid",
        "reason_codes": ["candidate_invalid"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_candidate_sha256": canonical_payload_sha256(dict(raw_payload)),
        "operator_recommendation": None,
        "phase6_deployment_status": None,
        "critical_alert_count": 1,
        "warning_alert_count": 0,
        "mean_net_return": None,
        "cumulative_net_return": None,
        "max_drawdown": None,
        "total_execution_cost": None,
        "cost_drift_ratio": None,
        "pnl_drift": None,
        "regime_mismatch_rate": None,
        "paper_only": raw_payload.get("paper_only", False) is True,
        "capital_at_risk": raw_payload.get("capital_at_risk", True) is True,
        "broker_exchange_write_enabled": (
            raw_payload.get("broker_exchange_write_enabled", False) is True
        ),
        "live_exchange_write_enabled": (
            raw_payload.get("live_exchange_write_enabled", False) is True
        ),
        "artifact_hashes": {},
        "created_at": created_at,
    }


def _candidate_status(
    paper_summary: dict[str, Any],
    observability_report: dict[str, Any],
) -> CandidateStatus:
    if paper_summary["phase5_kill_switch_triggered"]:
        return "phase5_blocked"
    if paper_summary["phase6_deployment_status"] == "blocked_fail_closed":
        return "phase6_blocked_fail_closed"
    if observability_report["alert_severity_counts"]["critical"] > 0:
        return "observability_critical"
    if observability_report["alert_severity_counts"]["warning"] > 0:
        return "observability_warning"
    if (
        observability_report["operator_recommendation"] == "continue_paper_run"
        and paper_summary["phase6_deployment_status"] == "approved_for_staged_live"
    ):
        return "ready_for_manual_review"
    return "paper_replay_failed"


def _candidate_reason_codes(
    status: CandidateStatus,
    paper_summary: dict[str, Any],
    observability_report: dict[str, Any],
) -> list[str]:
    codes = set(paper_summary.get("phase5_reason_codes", []) or [])
    codes.update(paper_summary.get("feed_health_reason_codes", []) or [])
    codes.update(alert["code"] for alert in observability_report["alerts"])
    if status != "ready_for_manual_review":
        codes.add(status)
    return sorted(str(code) for code in codes)


def _ranking_report(
    *,
    batch_id: str,
    summaries: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    rows = [_ranking_row(summary) for summary in summaries]
    ranked = sorted(rows, key=_ranking_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "schema_version": STRATEGY_DISCOVERY_SCHEMA_VERSION,
        "batch_id": batch_id,
        "ranking_rule": (
            "ready candidates first; demote critical/blocked/invalid; "
            "then lower drawdown/cost drift; then higher paper return"
        ),
        "candidates": ranked,
        "created_at": created_at,
    }


def _ranking_row(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id",
        "status",
        "operator_recommendation",
        "phase6_deployment_status",
        "critical_alert_count",
        "warning_alert_count",
        "mean_net_return",
        "cumulative_net_return",
        "max_drawdown",
        "total_execution_cost",
        "cost_drift_ratio",
        "pnl_drift",
        "regime_mismatch_rate",
        "artifact_hashes",
    )
    return {field: summary.get(field) for field in fields}


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    status = row["status"]
    blocked_or_invalid = status in {
        "candidate_invalid",
        "phase5_blocked",
        "phase6_blocked_fail_closed",
        "observability_critical",
        "paper_replay_failed",
    }
    return (
        1 if blocked_or_invalid else 0,
        int(row.get("critical_alert_count") or 0),
        1 if row.get("phase6_deployment_status") == "blocked_fail_closed" else 0,
        _number(row.get("max_drawdown"), default=1.0),
        _number(row.get("cost_drift_ratio"), default=1.0),
        -_number(row.get("cumulative_net_return"), default=-1.0),
        str(row["candidate_id"]),
    )


def _batch_manifest(
    *,
    config: StrategyCandidateReplayConfig,
    summaries: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    ready_count = sum(
        summary["status"] == "ready_for_manual_review" for summary in summaries
    )
    blocked_count = sum(
        summary["status"]
        in {
            "candidate_invalid",
            "phase5_blocked",
            "phase6_blocked_fail_closed",
            "observability_critical",
            "paper_replay_failed",
        }
        for summary in summaries
    )
    return {
        "schema_version": STRATEGY_DISCOVERY_SCHEMA_VERSION,
        "phase": STRATEGY_DISCOVERY_PAPER_INTEGRATION_PHASE,
        "batch_id": config.batch_id,
        "candidate_count": len(summaries),
        "ready_for_manual_review_count": ready_count,
        "blocked_count": blocked_count,
        "invalid_count": sum(
            summary["status"] == "candidate_invalid" for summary in summaries
        ),
        "critical_alert_candidate_count": sum(
            int(summary.get("critical_alert_count") or 0) > 0
            for summary in summaries
        ),
        "candidate_summaries": summaries,
        "ranking_path": str(artifact_paths["ranking_json"]),
        "ranking_sha256": _sha256_file(artifact_paths["ranking_json"]),
        "ranking_markdown_path": str(artifact_paths["ranking_md"]),
        "ranking_markdown_sha256": _sha256_file(artifact_paths["ranking_md"]),
        "batch_summary_path": str(artifact_paths["batch_summary_md"]),
        "batch_summary_sha256": _sha256_file(artifact_paths["batch_summary_md"]),
        "registry_path": str(artifact_paths["registry"]),
        "registry_sha256": _sha256_file(artifact_paths["registry"]),
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "created_at": config.created_at,
        "config": config.to_dict(),
    }


def _console_summary(
    batch_manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "batch_id": batch_manifest["batch_id"],
        "candidate_count": batch_manifest["candidate_count"],
        "ready_for_manual_review_count": batch_manifest[
            "ready_for_manual_review_count"
        ],
        "blocked_count": batch_manifest["blocked_count"],
        "critical_alert_candidate_count": batch_manifest[
            "critical_alert_candidate_count"
        ],
        "ranking_path": str(artifact_paths["ranking_json"]),
        "batch_summary_path": str(artifact_paths["batch_summary_md"]),
    }


def _ranking_markdown(ranking: dict[str, Any]) -> str:
    lines = [
        f"# Strategy Candidate Ranking: {ranking['batch_id']}",
        "",
        "| Rank | Candidate | Status | Phase 6 | Critical | Max DD | Return |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in ranking["candidates"]:
        lines.append(
            "| {rank} | `{candidate_id}` | `{status}` | `{phase6}` | "
            "{critical} | `{drawdown}` | `{ret}` |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                status=row["status"],
                phase6=row["phase6_deployment_status"],
                critical=row["critical_alert_count"],
                drawdown=row["max_drawdown"],
                ret=row["cumulative_net_return"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _batch_summary_markdown(batch_id: str, summaries: list[dict[str, Any]]) -> str:
    lines = [f"# Strategy Candidate Batch Summary: {batch_id}", ""]
    for summary in sorted(summaries, key=lambda item: item["candidate_id"]):
        lines.extend(
            [
                f"## {summary['candidate_id']}",
                "",
                f"- status: `{summary['status']}`",
                f"- operator_recommendation: `{summary['operator_recommendation']}`",
                f"- phase6_deployment_status: `{summary['phase6_deployment_status']}`",
                f"- critical_alert_count: `{summary['critical_alert_count']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _phase6_report_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("phase6_cicd_pipeline_report_*.json"))
    if len(matches) != 1:
        raise StrategyDiscoveryError(
            "expected exactly one Phase 6 report in paper run directory, "
            f"found {len(matches)}"
        )
    return matches[0]


def _number(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
