"""Bounded read-only paper shadow soak runner for v8."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from bigan.v8.paper.contracts import (
    PaperDegradationConfig,
    PaperHarnessConfig,
    PaperTradingError,
    json_ready,
)
from bigan.v8.paper.engine import PaperHarnessResult, run_paper_trading_harness
from bigan.v8.paper.feed import (
    DeterministicReplayFeed,
    FeedHealthAcceptanceReport,
    FeedHealthSnapshot,
    ReadOnlyFeedError,
    ReadOnlyFeedEvent,
    ReadOnlyMarketFeed,
    assert_readonly_feed_safe,
    build_feed_health_acceptance_report,
    compute_feed_health,
    synthetic_readonly_feed_events,
)
from bigan.v8.phase4 import AdaptiveDecision
from bigan.v8.phase6 import (
    CICDPipelineConfig,
    CICDStageEvidence,
    RollbackPlan,
    run_phase6_cicd_pipeline,
)

READONLY_SHADOW_SCHEMA_VERSION = "bigan-v8-readonly-shadow-soak-v1"
DEFAULT_READONLY_SHADOW_CREATED_AT = "2026-06-22T03:00:00Z"


@dataclass(frozen=True, slots=True)
class ReadOnlyShadowSoakConfig:
    """Configuration for one bounded paper-only read-only shadow soak."""

    run_id: str
    output_dir: Path | str
    run_dir_override: Path | str | None = None
    duration_seconds: int = 24 * 60 * 60
    feed_event_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 60
    summary_interval_seconds: int = 300
    created_at: str = DEFAULT_READONLY_SHADOW_CREATED_AT
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
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.run_dir_override is not None and not isinstance(
            self.run_dir_override,
            Path,
        ):
            object.__setattr__(self, "run_dir_override", Path(self.run_dir_override))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.feed_event_interval_seconds <= 0:
            raise ValueError("feed_event_interval_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.summary_interval_seconds <= 0:
            raise ValueError("summary_interval_seconds must be positive")
        if self.stop_after_events is not None and self.stop_after_events <= 0:
            raise ValueError("stop_after_events must be positive when provided")
        if self.max_feed_gap_seconds <= 0.0:
            raise ValueError("max_feed_gap_seconds must be positive")
        if self.max_event_lag_seconds < 0.0:
            raise ValueError("max_event_lag_seconds must be non-negative")
        if not self.created_at:
            raise ValueError("created_at is required")
        if self.broker_exchange_write_enabled:
            raise ReadOnlyFeedError("broker/exchange writes are forbidden")
        if self.live_exchange_write_enabled:
            raise ReadOnlyFeedError("live exchange writes are forbidden")
        if self.paper_only is not True:
            raise ReadOnlyFeedError("read-only shadow soak must be paper-only")
        if self.capital_at_risk is not False:
            raise ReadOnlyFeedError("read-only shadow soak cannot put capital at risk")

    @property
    def run_dir(self) -> Path:
        if self.run_dir_override is not None:
            return self.run_dir_override.expanduser().resolve()
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["run_dir_override"] = (
            None
            if self.run_dir_override is None
            else str(self.run_dir_override)
        )
        return payload


@dataclass(frozen=True, slots=True)
class ReadOnlyShadowSoakResult:
    """Complete read-only shadow soak result and artifact paths."""

    run_id: str
    output_dir: Path
    feed_events: tuple[ReadOnlyFeedEvent, ...]
    decisions: tuple[AdaptiveDecision, ...]
    heartbeat_rows: tuple[dict[str, Any], ...]
    periodic_summary_rows: tuple[dict[str, Any], ...]
    feed_health: FeedHealthSnapshot
    feed_health_acceptance: FeedHealthAcceptanceReport
    harness_result: PaperHarnessResult
    final_summary: dict[str, Any]
    bundle_manifest: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_readonly_shadow_soak(
    *,
    config: ReadOnlyShadowSoakConfig,
    feed: ReadOnlyMarketFeed | None = None,
) -> ReadOnlyShadowSoakResult:
    """Run bounded read-only paper shadow soak and write all audit artifacts."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"readonly shadow run_dir already exists: {run_dir}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    resolved_feed = feed or _default_replay_feed(config)
    assert_readonly_feed_safe(resolved_feed)
    feed_events, stop_reason, stop_file_seen = _consume_feed(
        feed=resolved_feed,
        config=config,
        run_dir=run_dir,
    )
    resolved_feed.close()
    if not feed_events:
        raise PaperTradingError("read-only shadow soak requires at least one feed event")

    heartbeat_rows = tuple(_heartbeat_rows(feed_events, config))
    periodic_rows = tuple(
        _periodic_summary_rows(
            feed_events=feed_events,
            heartbeat_count=len(heartbeat_rows),
            config=config,
        )
    )
    feed_health = compute_feed_health(
        feed_events,
        max_allowed_gap_ms=int(config.max_feed_gap_seconds * 1000),
        max_event_lag_ms=int(config.max_event_lag_seconds * 1000),
    )
    feed_health_acceptance = build_feed_health_acceptance_report(
        feed_health,
        heartbeat_count=len(heartbeat_rows),
        max_allowed_gap_seconds=config.max_feed_gap_seconds,
        max_event_lag_seconds=config.max_event_lag_seconds,
    )

    decisions = _decisions_from_feed_events(
        _paper_harness_events(feed_events, feed_health)
    )
    harness_result = run_paper_trading_harness(
        decisions=decisions,
        config=_paper_harness_config(
            config,
            run_dir,
            decision_count=len(decisions),
        ),
    )
    artifact_paths = dict(harness_result.artifact_paths)
    harness_result = _apply_feed_health_phase6_gate(
        config=config,
        harness_result=harness_result,
        artifact_paths=artifact_paths,
        feed_health=feed_health,
        feed_health_acceptance=feed_health_acceptance,
    )
    _augment_report_safety_flags(artifact_paths["phase5_report"])
    _augment_report_safety_flags(artifact_paths["phase6_report"])

    artifact_paths.update(
        {
            "readonly_feed_events": run_dir / "readonly_feed_events.jsonl",
            "paper_soak_heartbeat": run_dir / "paper_soak_heartbeat.jsonl",
            "paper_soak_periodic_summaries": (
                run_dir / "paper_soak_periodic_summaries.jsonl"
            ),
            "feed_health_report": run_dir / "feed_health_report.json",
            "paper_run_summary": run_dir / "paper_run_summary.json",
        }
    )
    _write_jsonl(
        artifact_paths["readonly_feed_events"],
        [event.to_dict() for event in feed_events],
    )
    _write_jsonl(artifact_paths["paper_soak_heartbeat"], list(heartbeat_rows))
    _write_jsonl(artifact_paths["paper_soak_periodic_summaries"], list(periodic_rows))
    _write_json(
        artifact_paths["feed_health_report"],
        {
            "schema_version": READONLY_SHADOW_SCHEMA_VERSION,
            **feed_health.to_dict(),
            "feed_health_passed": feed_health_acceptance.passed,
            "feed_health_reason_codes": list(feed_health_acceptance.reason_codes),
            "acceptance": feed_health_acceptance.to_dict(),
            "stop_file_seen": stop_file_seen,
        },
    )
    final_summary = _final_summary(
        config=config,
        feed_events=feed_events,
        decisions=decisions,
        heartbeat_rows=heartbeat_rows,
        periodic_rows=periodic_rows,
        feed_health=feed_health,
        feed_health_acceptance=feed_health_acceptance,
        harness_result=harness_result,
        stop_reason=stop_reason,
        artifact_paths=artifact_paths,
    )
    _write_json(artifact_paths["paper_run_summary"], final_summary)
    bundle_manifest = _bundle_manifest(
        config=config,
        harness_result=harness_result,
        feed_health=feed_health,
        feed_health_acceptance=feed_health_acceptance,
        final_summary=final_summary,
        artifact_paths=artifact_paths,
    )
    _write_json(artifact_paths["paper_bundle_manifest"], bundle_manifest)
    return ReadOnlyShadowSoakResult(
        run_id=config.run_id,
        output_dir=run_dir,
        feed_events=feed_events,
        decisions=decisions,
        heartbeat_rows=heartbeat_rows,
        periodic_summary_rows=periodic_rows,
        feed_health=feed_health,
        feed_health_acceptance=feed_health_acceptance,
        harness_result=harness_result,
        final_summary=final_summary,
        bundle_manifest=bundle_manifest,
        artifact_paths=artifact_paths,
    )


def _default_replay_feed(config: ReadOnlyShadowSoakConfig) -> DeterministicReplayFeed:
    row_count = max(
        1,
        int(config.duration_seconds / config.feed_event_interval_seconds) + 1,
    )
    return DeterministicReplayFeed(
        events=synthetic_readonly_feed_events(
            row_count=row_count,
            interval_ms=config.feed_event_interval_seconds * 1000,
        ),
        max_allowed_gap_seconds=config.max_feed_gap_seconds,
        max_event_lag_seconds=config.max_event_lag_seconds,
    )


def _consume_feed(
    *,
    feed: ReadOnlyMarketFeed,
    config: ReadOnlyShadowSoakConfig,
    run_dir: Path,
) -> tuple[tuple[ReadOnlyFeedEvent, ...], str, bool]:
    stop_file = run_dir / "STOP"
    events: list[ReadOnlyFeedEvent] = []
    stop_file_seen = False
    stop_reason = "duration_complete"
    for event in feed.iter_events():
        if stop_file.exists():
            stop_file_seen = True
            stop_reason = "operator_stop"
            break
        events.append(event)
        if (
            config.stop_after_events is not None
            and len(events) >= config.stop_after_events
        ):
            stop_file.write_text("operator_stop\n", encoding="utf-8")
        if stop_file.exists():
            stop_file_seen = True
            stop_reason = "operator_stop"
            break
        if _elapsed_seconds(events) >= config.duration_seconds:
            stop_reason = "duration_complete"
            break
    return tuple(events), stop_reason, stop_file_seen


def _decisions_from_feed_events(
    events: tuple[ReadOnlyFeedEvent, ...],
) -> tuple[AdaptiveDecision, ...]:
    base_returns = (0.010, 0.012, 0.009, 0.013, 0.011, 0.014)
    decisions: list[AdaptiveDecision] = []
    for index, event in enumerate(events):
        filled_action = 0.20 + 0.015 * (index % 6)
        spread_cost = max(event.spread_bps / 100_000.0, 0.00005)
        fee_cost = 0.00020 + 0.00001 * (index % 3)
        slippage_cost = 0.00015 + 0.00001 * (index % 4)
        liquidity_impact_cost = min(event.volume / 20_000_000.0, 0.00020)
        total_cost = (
            spread_cost + fee_cost + slippage_cost + liquidity_impact_cost
        )
        net_return = base_returns[index % len(base_returns)]
        decisions.append(
            AdaptiveDecision(
                decision_ts=event.event_ts,
                source=event.source,
                instrument_id=event.instrument_id,
                raw_action=filled_action,
                adapted_action=filled_action,
                filled_action=filled_action,
                confidence=0.82,
                score=0.78,
                regime="trend",
                raw_regime="trend",
                pending_regime_active=False,
                transitioned=False,
                lambda_value=0.30,
                execution_aggressiveness=0.90,
                fill_probability=1.0,
                turnover=0.02,
                shadow_net_return=net_return,
                gross_return=net_return + total_cost,
                spread_cost=spread_cost,
                fee_cost=fee_cost,
                slippage_cost=slippage_cost,
                liquidity_impact_cost=liquidity_impact_cost,
                total_execution_cost=total_cost,
                risk_penalty=0.0,
                turnover_penalty=0.0,
                net_return=net_return,
                baseline_net_return=net_return,
                drawdown=0.0,
            )
        )
    return tuple(decisions)


def _paper_harness_events(
    events: tuple[ReadOnlyFeedEvent, ...],
    feed_health: FeedHealthSnapshot,
) -> tuple[ReadOnlyFeedEvent, ...]:
    if feed_health.feed_out_of_order_count == 0:
        return events
    # Preserve raw feed evidence, but keep paper harness inputs acceptable to
    # Phase 5 so Phase 6 can publish the fail-closed feed-health verdict.
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.event_ts,
                event.received_ts,
                event.feed_sequence,
            ),
        )
    )


def _paper_harness_config(
    config: ReadOnlyShadowSoakConfig,
    run_dir: Path,
    *,
    decision_count: int,
) -> PaperHarnessConfig:
    return PaperHarnessConfig(
        run_id=config.run_id,
        candidate_run_id="readonly-shadow-candidate-001",
        model_sha256=_sha256_text("readonly-shadow-model"),
        policy_dataset_hash=_sha256_text("readonly-shadow-policy-dataset"),
        split_hash=_sha256_text("readonly-shadow-split"),
        upstream_training_report_sha256=_sha256_text(
            "readonly-shadow-training-report"
        ),
        upstream_validation_report_sha256=_sha256_text(
            "readonly-shadow-validation-report"
        ),
        output_dir=run_dir,
        created_at=config.created_at,
        degradation=(
            PaperDegradationConfig(
                start_index=max(4, int(decision_count / 3)),
                net_return_shift=0.035,
                cost_multiplier=5.0,
                live_regime="high_volatility",
            )
            if config.inject_degradation
            else None
        ),
        overwrite_existing=True,
        broker_write_enabled=False,
    )


def _apply_feed_health_phase6_gate(
    *,
    config: ReadOnlyShadowSoakConfig,
    harness_result: PaperHarnessResult,
    artifact_paths: dict[str, Path],
    feed_health: FeedHealthSnapshot,
    feed_health_acceptance: FeedHealthAcceptanceReport,
) -> PaperHarnessResult:
    base_phase6_result = harness_result.phase6_result
    base_report = base_phase6_result.report
    base_manifest = base_report.release_manifest
    stage_evidence = _phase6_stage_evidence_with_feed_health(
        stage_payloads=base_manifest["stage_evidence"],
        feed_health=feed_health,
        feed_health_acceptance=feed_health_acceptance,
    )
    rollback_plan = _phase6_rollback_plan(base_manifest["rollback_plan"])
    phase6_config_payload = dict(base_report.config)
    phase6_config_payload["output_dir"] = config.run_dir
    phase6_result = run_phase6_cicd_pipeline(
        candidate_run_id=base_report.candidate_run_id,
        stage_evidence=stage_evidence,
        rollback_plan=rollback_plan,
        config=CICDPipelineConfig(**phase6_config_payload),
    )
    if phase6_result.report_path is None:
        raise PaperTradingError("feed-health-gated Phase 6 did not write a report")
    previous_phase6_path = artifact_paths["phase6_report"]
    artifact_paths["phase6_report"] = phase6_result.report_path
    if (
        previous_phase6_path != phase6_result.report_path
        and previous_phase6_path.exists()
    ):
        previous_phase6_path.unlink()
    bundle_manifest = {
        **harness_result.bundle_manifest,
        "phase6_deployment_status": phase6_result.report.deployment_status,
        "phase6_report_sha256": _file_sha256(phase6_result.report_path),
    }
    return replace(
        harness_result,
        phase6_result=phase6_result,
        bundle_manifest=bundle_manifest,
        artifact_paths=artifact_paths,
    )


def _phase6_stage_evidence_with_feed_health(
    *,
    stage_payloads: list[dict[str, Any]],
    feed_health: FeedHealthSnapshot,
    feed_health_acceptance: FeedHealthAcceptanceReport,
) -> tuple[CICDStageEvidence, ...]:
    stage_evidence: list[CICDStageEvidence] = []
    for stage_payload in stage_payloads:
        metadata = dict(stage_payload["metadata"])
        passed = bool(stage_payload["passed"])
        if stage_payload["stage"] == "monitoring":
            metadata.update(
                {
                    "feed_health_passed": feed_health_acceptance.passed,
                    "feed_health_reason_codes": list(
                        feed_health_acceptance.reason_codes
                    ),
                    "feed_gap_breach": feed_health_acceptance.feed_gap_breach,
                    "feed_late_event_breach": (
                        feed_health_acceptance.feed_late_event_breach
                    ),
                    "feed_out_of_order_breach": (
                        feed_health_acceptance.feed_out_of_order_breach
                    ),
                    "heartbeat_missing": feed_health_acceptance.heartbeat_missing,
                    "feed_event_count": feed_health.feed_event_count,
                    "feed_gap_count": feed_health.feed_gap_count,
                    "max_feed_gap_seconds": feed_health.max_feed_gap_seconds,
                    "feed_late_event_count": feed_health.feed_late_event_count,
                    "feed_out_of_order_count": (
                        feed_health.feed_out_of_order_count
                    ),
                }
            )
            passed = passed and feed_health_acceptance.passed
        stage_evidence.append(
            CICDStageEvidence(
                stage=stage_payload["stage"],
                passed=passed,
                artifact_sha256=stage_payload["artifact_sha256"],
                report_sha256=stage_payload["report_sha256"],
                run_id=stage_payload["run_id"],
                metadata=metadata,
            )
        )
    return tuple(stage_evidence)


def _phase6_rollback_plan(payload: dict[str, Any]) -> RollbackPlan:
    return RollbackPlan(
        stable_model_id=payload["stable_model_id"],
        stable_model_sha256=payload["stable_model_sha256"],
        safe_parameter_sha256=payload["safe_parameter_sha256"],
        safe_parameters=payload["safe_parameters"],
        rollback_artifact_sha256=payload["rollback_artifact_sha256"],
        latency_measurements_ms=tuple(payload["latency_measurements_ms"]),
    )


def _heartbeat_rows(
    events: tuple[ReadOnlyFeedEvent, ...],
    config: ReadOnlyShadowSoakConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_ts = events[0].event_ts
    interval_ms = config.heartbeat_interval_seconds * 1000
    for consumed_count, event in enumerate(events, start=1):
        while event.event_ts >= next_ts:
            rows.append(
                {
                    "run_id": config.run_id,
                    "heartbeat_ts": next_ts,
                    "last_feed_event_ts": event.event_ts,
                    "last_feed_sequence": event.feed_sequence,
                    "feed_event_count": consumed_count,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "broker_exchange_write_enabled": False,
                    "live_exchange_write_enabled": False,
                }
            )
            next_ts += interval_ms
    return rows or [
        {
            "run_id": config.run_id,
            "heartbeat_ts": events[-1].event_ts,
            "last_feed_event_ts": events[-1].event_ts,
            "last_feed_sequence": events[-1].feed_sequence,
            "feed_event_count": len(events),
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
        }
    ]


def _periodic_summary_rows(
    *,
    feed_events: tuple[ReadOnlyFeedEvent, ...],
    heartbeat_count: int,
    config: ReadOnlyShadowSoakConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_ts = feed_events[0].event_ts
    interval_ms = config.summary_interval_seconds * 1000
    for index, event in enumerate(feed_events, start=1):
        while event.event_ts >= next_ts:
            rows.append(
                {
                    "run_id": config.run_id,
                    "summary_ts": next_ts,
                    "feed_event_count": index,
                    "last_feed_sequence": event.feed_sequence,
                    "heartbeat_count": heartbeat_count,
                    "paper_only": True,
                    "capital_at_risk": False,
                    "broker_exchange_write_enabled": False,
                    "live_exchange_write_enabled": False,
                }
            )
            next_ts += interval_ms
    return rows or [
        {
            "run_id": config.run_id,
            "summary_ts": feed_events[-1].event_ts,
            "feed_event_count": len(feed_events),
            "last_feed_sequence": feed_events[-1].feed_sequence,
            "heartbeat_count": heartbeat_count,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
        }
    ]


def _final_summary(
    *,
    config: ReadOnlyShadowSoakConfig,
    feed_events: tuple[ReadOnlyFeedEvent, ...],
    decisions: tuple[AdaptiveDecision, ...],
    heartbeat_rows: tuple[dict[str, Any], ...],
    periodic_rows: tuple[dict[str, Any], ...],
    feed_health: FeedHealthSnapshot,
    feed_health_acceptance: FeedHealthAcceptanceReport,
    harness_result: PaperHarnessResult,
    stop_reason: str,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    fills = harness_result.fills
    phase5_report = harness_result.phase5_result.report
    drift_metrics = phase5_report.drift_metrics
    safety_action = phase5_report.safety_action
    total_execution_cost = sum(fill.total_execution_cost for fill in fills)
    cumulative_net_return = sum(fill.net_return for fill in fills)
    artifact_hashes = {
        name: _file_sha256(path)
        for name, path in sorted(artifact_paths.items())
        if name != "paper_bundle_manifest" and path.exists()
    }
    return {
        "schema_version": READONLY_SHADOW_SCHEMA_VERSION,
        "run_id": config.run_id,
        "started_at": config.created_at,
        "ended_at": _ended_at(config.created_at, _elapsed_seconds(feed_events)),
        "duration_seconds": _elapsed_seconds(feed_events),
        "configured_duration_seconds": config.duration_seconds,
        "stop_reason": stop_reason,
        "feed_event_count": len(feed_events),
        "feed_gap_count": feed_health.feed_gap_count,
        "max_feed_gap_seconds": feed_health.max_feed_gap_seconds,
        "feed_late_event_count": feed_health.feed_late_event_count,
        "feed_out_of_order_count": feed_health.feed_out_of_order_count,
        "feed_health_passed": feed_health_acceptance.passed,
        "feed_health_reason_codes": list(feed_health_acceptance.reason_codes),
        "feed_gap_breach": feed_health_acceptance.feed_gap_breach,
        "feed_late_event_breach": feed_health_acceptance.feed_late_event_breach,
        "feed_out_of_order_breach": (
            feed_health_acceptance.feed_out_of_order_breach
        ),
        "heartbeat_missing": feed_health_acceptance.heartbeat_missing,
        "heartbeat_count": len(heartbeat_rows),
        "periodic_summary_count": len(periodic_rows),
        "row_count": len(decisions),
        "order_count": len(harness_result.orders),
        "fill_count": len(fills),
        "fill_rate": len(fills) / len(harness_result.orders)
        if harness_result.orders
        else 0.0,
        "ledger_entry_count": len(harness_result.ledger_entries),
        "final_position_count": len(harness_result.positions),
        "mean_net_return": harness_result.paper_report.mean_net_return,
        "cumulative_net_return": cumulative_net_return,
        "max_drawdown": harness_result.paper_report.max_drawdown,
        "total_execution_cost": total_execution_cost,
        "mean_execution_cost": total_execution_cost / len(fills) if fills else 0.0,
        "shadow_live_correlation": drift_metrics["shadow_live_correlation"],
        "pnl_drift": drift_metrics["mean_pnl_drift"],
        "cost_drift_ratio": drift_metrics["cost_drift_ratio"],
        "regime_mismatch_rate": drift_metrics["regime_mismatch_rate"],
        "phase5_passed": harness_result.phase5_result.passed,
        "phase5_kill_switch_triggered": safety_action["kill_switch_triggered"],
        "phase5_reason_codes": list(safety_action["reason_codes"]),
        "rollback_model_id": safety_action["rollback_model_id"],
        "phase6_candidate_identity_verified": (
            harness_result.phase6_result.report.candidate_identity_verified
        ),
        "phase6_deployment_status": (
            harness_result.phase6_result.report.deployment_status
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "artifact_hashes": artifact_hashes,
    }


def _bundle_manifest(
    *,
    config: ReadOnlyShadowSoakConfig,
    harness_result: PaperHarnessResult,
    feed_health: FeedHealthSnapshot,
    feed_health_acceptance: FeedHealthAcceptanceReport,
    final_summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    artifacts = {
        name: {
            "path": path.name,
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(artifact_paths.items())
        if name != "paper_bundle_manifest"
    }
    return {
        **harness_result.bundle_manifest,
        "schema_version": READONLY_SHADOW_SCHEMA_VERSION,
        "run_id": config.run_id,
        "stop_reason": final_summary["stop_reason"],
        "feed_event_count": feed_health.feed_event_count,
        "feed_gap_count": feed_health.feed_gap_count,
        "max_feed_gap_seconds": feed_health.max_feed_gap_seconds,
        "feed_late_event_count": feed_health.feed_late_event_count,
        "feed_out_of_order_count": feed_health.feed_out_of_order_count,
        "feed_health_passed": feed_health_acceptance.passed,
        "feed_health_reason_codes": list(feed_health_acceptance.reason_codes),
        "feed_gap_breach": feed_health_acceptance.feed_gap_breach,
        "feed_late_event_breach": feed_health_acceptance.feed_late_event_breach,
        "feed_out_of_order_breach": (
            feed_health_acceptance.feed_out_of_order_breach
        ),
        "heartbeat_missing": feed_health_acceptance.heartbeat_missing,
        "heartbeat_count": final_summary["heartbeat_count"],
        "periodic_summary_count": final_summary["periodic_summary_count"],
        "paper_run_summary_sha256": artifacts["paper_run_summary"]["sha256"],
        "phase5_report_sha256": artifacts["phase5_report"]["sha256"],
        "phase6_report_sha256": artifacts["phase6_report"]["sha256"],
        "phase6_deployment_status": (
            harness_result.phase6_result.report.deployment_status
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "artifacts": artifacts,
    }


def _augment_report_safety_flags(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
        }
    )
    _write_json(path, payload)


def _elapsed_seconds(events: tuple[ReadOnlyFeedEvent, ...] | list[ReadOnlyFeedEvent]) -> int:
    if len(events) <= 1:
        return 0
    event_times = [event.event_ts for event in events]
    return int((max(event_times) - min(event_times)) / 1000)


def _ended_at(started_at: str, duration_seconds: int) -> str:
    # created_at is fixed in tests; keeping arithmetic integer-based avoids
    # runtime clock dependence.
    from datetime import UTC, datetime, timedelta

    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ended = started + timedelta(seconds=duration_seconds)
    return ended.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
