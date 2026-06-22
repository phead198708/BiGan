"""Operator-facing observability for v8 paper-only runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.paper.contracts import json_ready

PAPER_OBSERVABILITY_PHASE = "paper_observability"
DEFAULT_OBSERVABILITY_CREATED_AT = "2026-06-22T04:00:00Z"

AlertSeverity = Literal["info", "warning", "critical"]
AlertCategory = Literal[
    "feed",
    "safety",
    "pnl",
    "execution",
    "phase6",
    "artifact",
    "paper_boundary",
]
OperatorRecommendation = Literal[
    "continue_paper_run",
    "investigate_warning",
    "stop_paper_run",
    "blocked_fail_closed",
]

OBSERVABILITY_OUTPUT_FILENAMES: tuple[str, ...] = (
    "paper_observability_report.json",
    "paper_operator_summary.md",
    "paper_alerts.jsonl",
    "paper_dashboard_summary.json",
    "paper_periodic_metrics.csv",
)

_REQUIRED_STATIC_ARTIFACTS: dict[str, str] = {
    "paper_run_summary": "paper_run_summary.json",
    "paper_bundle_manifest": "paper_bundle_manifest.json",
    "feed_health_report": "feed_health_report.json",
    "phase5_report": "phase5_safety_layer_report.json",
    "paper_orders": "paper_orders.jsonl",
    "paper_fills": "paper_fills.jsonl",
    "paper_ledger": "paper_ledger.jsonl",
    "paper_positions": "paper_positions.json",
    "paper_pnl_report": "paper_pnl_report.json",
    "paper_soak_heartbeat": "paper_soak_heartbeat.jsonl",
    "paper_soak_periodic_summaries": "paper_soak_periodic_summaries.jsonl",
}
_PHASE6_REPORT_GLOB = "phase6_cicd_pipeline_report_*.json"
_ALL_BOUNDARY_FIELDS: dict[str, bool] = {
    "paper_only": True,
    "capital_at_risk": False,
    "broker_exchange_write_enabled": False,
    "live_exchange_write_enabled": False,
}
_PAPER_ONLY_BOUNDARY_FIELDS: dict[str, bool] = {
    "paper_only": True,
    "capital_at_risk": False,
}
_JSON_BOUNDARY_FIELDS: dict[str, dict[str, bool]] = {
    "paper_run_summary": _ALL_BOUNDARY_FIELDS,
    "paper_bundle_manifest": _ALL_BOUNDARY_FIELDS,
    "feed_health_report": _ALL_BOUNDARY_FIELDS,
    "phase5_report": _ALL_BOUNDARY_FIELDS,
    "phase6_report": _ALL_BOUNDARY_FIELDS,
    "paper_positions": _PAPER_ONLY_BOUNDARY_FIELDS,
    "paper_pnl_report": _PAPER_ONLY_BOUNDARY_FIELDS,
}
_JSONL_BOUNDARY_FIELDS: dict[str, dict[str, bool]] = {
    "paper_orders": _PAPER_ONLY_BOUNDARY_FIELDS,
    "paper_fills": _PAPER_ONLY_BOUNDARY_FIELDS,
    "paper_ledger": _PAPER_ONLY_BOUNDARY_FIELDS,
    "paper_soak_heartbeat": _ALL_BOUNDARY_FIELDS,
    "paper_soak_periodic_summaries": _ALL_BOUNDARY_FIELDS,
}
_BOUNDARY_MISSING_CODES: dict[str, str] = {
    "paper_only": "paper_only_missing",
    "capital_at_risk": "capital_at_risk_missing",
    "broker_exchange_write_enabled": "broker_write_flag_missing",
    "live_exchange_write_enabled": "live_write_flag_missing",
}
_BOUNDARY_VIOLATION_CODES: dict[str, str] = {
    "paper_only": "paper_only_violation",
    "capital_at_risk": "capital_risk_violation",
    "broker_exchange_write_enabled": "broker_write_enabled",
    "live_exchange_write_enabled": "live_write_enabled",
}


class PaperObservabilityError(RuntimeError):
    """Raised when paper observability cannot safely summarize a run."""


@dataclass(frozen=True, slots=True)
class PaperObservabilityThresholds:
    """Deterministic alert thresholds for one paper observability report."""

    max_drawdown_warning: float = 0.05
    max_cost_drift_ratio_warning: float = 0.50
    max_abs_pnl_drift_warning: float = 0.006
    max_regime_mismatch_rate_warning: float = 0.25

    def __post_init__(self) -> None:
        for field_name in (
            "max_drawdown_warning",
            "max_cost_drift_ratio_warning",
            "max_abs_pnl_drift_warning",
            "max_regime_mismatch_rate_warning",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperAlert:
    """One deterministic operator alert derived from paper artifacts."""

    alert_id: str
    severity: AlertSeverity
    category: AlertCategory
    code: str
    message: str
    metric_name: str | None = None
    metric_value: Any = None
    threshold: Any = None
    recommendation: str | None = None

    def __post_init__(self) -> None:
        if not self.alert_id:
            raise ValueError("alert_id is required")
        if self.severity not in ("info", "warning", "critical"):
            raise ValueError("severity must be info, warning, or critical")
        if self.category not in (
            "feed",
            "safety",
            "pnl",
            "execution",
            "phase6",
            "artifact",
            "paper_boundary",
        ):
            raise ValueError("category is invalid")
        if not self.code:
            raise ValueError("code is required")
        if not self.message:
            raise ValueError("message is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperRunObservabilityReport:
    """Complete operator observability report for one paper run."""

    phase: str
    run_id: str
    source_run_dir: str
    summary_sha256: str
    bundle_sha256: str
    phase5_report_sha256: str
    phase6_report_sha256: str
    source_artifact_hashes: dict[str, str]
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    feed_health_status: str
    safety_status: str
    phase6_status: str
    performance_metrics: dict[str, Any]
    risk_metrics: dict[str, Any]
    feed_metrics: dict[str, Any]
    execution_metrics: dict[str, Any]
    alert_count: int
    alert_severity_counts: dict[str, int]
    alerts: list[dict[str, Any]]
    operator_recommendation: OperatorRecommendation
    thresholds: dict[str, Any]
    created_at: str

    @property
    def passed(self) -> bool:
        return self.operator_recommendation == "continue_paper_run"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True, slots=True)
class PaperRunComparisonReport:
    """Deterministic comparison between two paper observability reports."""

    left_run_id: str
    right_run_id: str
    metric_deltas: dict[str, Any]
    alert_deltas: dict[str, Any]
    status_change: str
    phase6_status_change: str
    recommendation: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperObservabilityResult:
    """Observability run result and output artifact paths."""

    report: PaperRunObservabilityReport
    dashboard_summary: dict[str, Any]
    comparison_report: PaperRunComparisonReport | None
    output_dir: Path
    artifact_paths: dict[str, Path]


def summarize_paper_run(
    *,
    run_dir: Path | str,
    output_dir: Path | str,
    thresholds: PaperObservabilityThresholds | None = None,
    created_at: str = DEFAULT_OBSERVABILITY_CREATED_AT,
    overwrite_existing: bool = False,
    compare_run_dir: Path | str | None = None,
) -> PaperObservabilityResult:
    """Read paper artifacts and write deterministic operator observability files."""

    resolved_thresholds = thresholds or PaperObservabilityThresholds()
    source_path = Path(run_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    _assert_output_dir_safe(
        source_path=source_path,
        output_path=output_path,
        source_label="run_dir",
    )
    compare_path = (
        None if compare_run_dir is None else Path(compare_run_dir).expanduser().resolve()
    )
    if compare_path is not None:
        _assert_output_dir_safe(
            source_path=compare_path,
            output_path=output_path,
            source_label="compare_run_dir",
        )

    loaded = _load_paper_artifacts(source_path)
    comparison_loaded = (
        None if compare_path is None else _load_paper_artifacts(compare_path)
    )

    if output_path.exists():
        if not overwrite_existing:
            raise FileExistsError(
                f"observability output_dir already exists: {output_path}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    report = _build_observability_report(
        loaded,
        thresholds=resolved_thresholds,
        created_at=created_at,
    )
    dashboard = _dashboard_summary(report)
    artifact_paths = {
        "observability_report": output_path / "paper_observability_report.json",
        "operator_summary": output_path / "paper_operator_summary.md",
        "alerts": output_path / "paper_alerts.jsonl",
        "dashboard_summary": output_path / "paper_dashboard_summary.json",
        "periodic_metrics_csv": output_path / "paper_periodic_metrics.csv",
    }
    _write_json(artifact_paths["observability_report"], report.to_dict())
    _write_json(artifact_paths["dashboard_summary"], dashboard)
    _write_jsonl(artifact_paths["alerts"], report.alerts)
    _write_text(artifact_paths["operator_summary"], _operator_markdown(report))
    _write_periodic_metrics_csv(
        artifact_paths["periodic_metrics_csv"],
        loaded["jsonl"]["paper_soak_periodic_summaries"],
    )

    comparison_report = None
    if comparison_loaded is not None:
        comparison_report = compare_paper_runs(
            report,
            _build_observability_report(
                comparison_loaded,
                thresholds=resolved_thresholds,
                created_at=created_at,
            ),
            created_at=created_at,
        )
        artifact_paths["run_comparison_json"] = output_path / "paper_run_comparison.json"
        artifact_paths["run_comparison_md"] = output_path / "paper_run_comparison.md"
        _write_json(
            artifact_paths["run_comparison_json"],
            comparison_report.to_dict(),
        )
        _write_text(
            artifact_paths["run_comparison_md"],
            _comparison_markdown(comparison_report),
        )

    return PaperObservabilityResult(
        report=report,
        dashboard_summary=dashboard,
        comparison_report=comparison_report,
        output_dir=output_path,
        artifact_paths=artifact_paths,
    )


def build_paper_observability_report(
    *,
    run_dir: Path | str,
    thresholds: PaperObservabilityThresholds | None = None,
    created_at: str = DEFAULT_OBSERVABILITY_CREATED_AT,
) -> PaperRunObservabilityReport:
    """Build an observability report without writing output files."""

    return _build_observability_report(
        _load_paper_artifacts(Path(run_dir)),
        thresholds=thresholds or PaperObservabilityThresholds(),
        created_at=created_at,
    )


def compare_paper_runs(
    left: PaperRunObservabilityReport,
    right: PaperRunObservabilityReport,
    *,
    created_at: str = DEFAULT_OBSERVABILITY_CREATED_AT,
) -> PaperRunComparisonReport:
    """Compare two already-built paper observability reports."""

    metric_names = (
        "cumulative_net_return",
        "mean_net_return",
        "max_drawdown",
        "total_execution_cost",
        "cost_drift_ratio",
        "pnl_drift",
        "regime_mismatch_rate",
    )
    metric_deltas: dict[str, Any] = {}
    left_metrics = {
        **left.performance_metrics,
        **left.risk_metrics,
        **left.execution_metrics,
    }
    right_metrics = {
        **right.performance_metrics,
        **right.risk_metrics,
        **right.execution_metrics,
    }
    for metric_name in metric_names:
        left_value = _number(left_metrics.get(metric_name), default=0.0)
        right_value = _number(right_metrics.get(metric_name), default=0.0)
        metric_deltas[metric_name] = right_value - left_value
    alert_deltas = {
        "alert_count_delta": right.alert_count - left.alert_count,
        "critical_alert_count_delta": (
            right.alert_severity_counts["critical"]
            - left.alert_severity_counts["critical"]
        ),
        "warning_alert_count_delta": (
            right.alert_severity_counts["warning"]
            - left.alert_severity_counts["warning"]
        ),
    }
    status_change = f"{left.operator_recommendation}->{right.operator_recommendation}"
    phase6_status_change = f"{left.phase6_status}->{right.phase6_status}"
    recommendation = (
        "right_run_risk_increased"
        if alert_deltas["critical_alert_count_delta"] > 0
        or right.phase6_status == "blocked_fail_closed"
        else "right_run_not_worse"
    )
    return PaperRunComparisonReport(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        metric_deltas=metric_deltas,
        alert_deltas=alert_deltas,
        status_change=status_change,
        phase6_status_change=phase6_status_change,
        recommendation=recommendation,
        created_at=created_at,
    )


def _assert_output_dir_safe(
    *,
    source_path: Path,
    output_path: Path,
    source_label: str,
) -> None:
    if output_path == source_path:
        raise PaperObservabilityError(f"output_dir must not equal {source_label}")
    if source_path in output_path.parents:
        raise PaperObservabilityError(f"output_dir must not be inside {source_label}")
    if output_path in source_path.parents:
        raise PaperObservabilityError(f"output_dir must not contain {source_label}")


def _load_paper_artifacts(run_dir: Path) -> dict[str, Any]:
    source_dir = run_dir.expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise PaperObservabilityError(f"run_dir does not exist: {source_dir}")
    paths = _required_artifact_paths(source_dir)
    json_payloads = {
        key: _read_json(path)
        for key, path in paths.items()
        if path.suffix == ".json"
    }
    jsonl_payloads = {
        key: _read_jsonl(path)
        for key, path in paths.items()
        if path.suffix == ".jsonl"
    }
    artifact_hashes = {key: _sha256_file(path) for key, path in paths.items()}
    return {
        "source_dir": source_dir,
        "paths": paths,
        "json": json_payloads,
        "jsonl": jsonl_payloads,
        "artifact_hashes": artifact_hashes,
    }


def _required_artifact_paths(source_dir: Path) -> dict[str, Path]:
    paths = {
        key: source_dir / filename
        for key, filename in _REQUIRED_STATIC_ARTIFACTS.items()
    }
    phase6_paths = sorted(source_dir.glob(_PHASE6_REPORT_GLOB))
    if len(phase6_paths) != 1:
        raise PaperObservabilityError(
            "expected exactly one Phase 6 report matching "
            f"{_PHASE6_REPORT_GLOB}, found {len(phase6_paths)}"
        )
    paths["phase6_report"] = phase6_paths[0]
    missing = [str(path.name) for path in paths.values() if not path.exists()]
    if missing:
        raise PaperObservabilityError(
            "missing required paper artifacts: " + ", ".join(sorted(missing))
        )
    return paths


def _build_observability_report(
    loaded: dict[str, Any],
    *,
    thresholds: PaperObservabilityThresholds,
    created_at: str,
) -> PaperRunObservabilityReport:
    payloads = loaded["json"]
    jsonl = loaded["jsonl"]
    hashes = loaded["artifact_hashes"]
    summary = payloads["paper_run_summary"]
    feed = payloads["feed_health_report"]
    phase5 = payloads["phase5_report"]
    phase6 = payloads["phase6_report"]
    pnl = payloads["paper_pnl_report"]
    positions = payloads["paper_positions"]
    metrics = _extract_metrics(summary, feed, phase5, phase6, pnl, positions, jsonl)
    alerts: list[PaperAlert] = []
    _add_artifact_integrity_alerts(alerts, loaded)
    _add_feed_alerts(alerts, metrics)
    _add_safety_alerts(alerts, metrics, phase5)
    _add_phase6_alerts(alerts, metrics)
    _add_paper_boundary_alerts(alerts, payloads, jsonl)
    _add_performance_alerts(alerts, metrics, thresholds)

    alert_dicts = [alert.to_dict() for alert in alerts]
    severity_counts = _alert_severity_counts(alerts)
    recommendation = _operator_recommendation(alerts, metrics["phase6_status"])
    paper_only_clean = not any(
        alert.code in {"paper_only_missing", "paper_only_violation"}
        for alert in alerts
    )
    capital_clean = not any(
        alert.code in {"capital_at_risk_missing", "capital_risk_violation"}
        for alert in alerts
    )
    broker_write_clean = not any(
        alert.code in {"broker_write_flag_missing", "broker_write_enabled"}
        for alert in alerts
    )
    live_write_clean = not any(
        alert.code in {"live_write_flag_missing", "live_write_enabled"}
        for alert in alerts
    )
    return PaperRunObservabilityReport(
        phase=PAPER_OBSERVABILITY_PHASE,
        run_id=str(summary["run_id"]),
        source_run_dir=str(loaded["source_dir"]),
        summary_sha256=hashes["paper_run_summary"],
        bundle_sha256=hashes["paper_bundle_manifest"],
        phase5_report_sha256=hashes["phase5_report"],
        phase6_report_sha256=hashes["phase6_report"],
        source_artifact_hashes=dict(sorted(hashes.items())),
        paper_only=paper_only_clean and summary.get("paper_only") is True,
        capital_at_risk=not capital_clean
        or summary.get("capital_at_risk") is not False,
        broker_exchange_write_enabled=not broker_write_clean
        or summary.get("broker_exchange_write_enabled") is not False,
        live_exchange_write_enabled=not live_write_clean
        or summary.get("live_exchange_write_enabled") is not False,
        feed_health_status=metrics["feed_health_status"],
        safety_status=metrics["safety_status"],
        phase6_status=metrics["phase6_status"],
        performance_metrics=metrics["performance_metrics"],
        risk_metrics=metrics["risk_metrics"],
        feed_metrics=metrics["feed_metrics"],
        execution_metrics=metrics["execution_metrics"],
        alert_count=len(alerts),
        alert_severity_counts=severity_counts,
        alerts=alert_dicts,
        operator_recommendation=recommendation,
        thresholds=thresholds.to_dict(),
        created_at=created_at,
    )


def _extract_metrics(
    summary: dict[str, Any],
    feed: dict[str, Any],
    phase5: dict[str, Any],
    phase6: dict[str, Any],
    pnl: dict[str, Any],
    positions: dict[str, Any],
    jsonl: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    safety_action = phase5.get("safety_action", {})
    drift_metrics = phase5.get("drift_metrics", {})
    phase6_status = str(
        summary.get("phase6_deployment_status")
        or phase6.get("deployment_status")
        or "unknown"
    )
    feed_health_passed = summary.get(
        "feed_health_passed",
        feed.get("feed_health_passed"),
    )
    feed_metrics = {
        "feed_event_count": _int(summary.get("feed_event_count")),
        "feed_gap_count": _int(summary.get("feed_gap_count", feed.get("feed_gap_count"))),
        "max_feed_gap_seconds": _number(
            summary.get("max_feed_gap_seconds", feed.get("max_feed_gap_seconds"))
        ),
        "feed_late_event_count": _int(
            summary.get("feed_late_event_count", feed.get("feed_late_event_count"))
        ),
        "feed_out_of_order_count": _int(
            summary.get(
                "feed_out_of_order_count",
                feed.get("feed_out_of_order_count"),
            )
        ),
        "heartbeat_count": _int(
            summary.get(
                "heartbeat_count",
                len(jsonl.get("paper_soak_heartbeat", ())),
            )
        ),
        "periodic_summary_count": _int(
            summary.get(
                "periodic_summary_count",
                len(jsonl.get("paper_soak_periodic_summaries", ())),
            )
        ),
        "feed_health_passed": feed_health_passed is True,
        "feed_health_reason_codes": list(
            summary.get(
                "feed_health_reason_codes",
                feed.get("feed_health_reason_codes", ()),
            )
            or []
        ),
    }
    performance_metrics = {
        "mean_net_return": _number(
            summary.get("mean_net_return", pnl.get("mean_net_return"))
        ),
        "cumulative_net_return": _number(summary.get("cumulative_net_return")),
        "total_execution_cost": _number(
            summary.get("total_execution_cost", pnl.get("total_execution_cost"))
        ),
        "row_count": _int(summary.get("row_count", pnl.get("row_count"))),
    }
    risk_metrics = {
        "max_drawdown": _number(summary.get("max_drawdown", pnl.get("max_drawdown"))),
        "pnl_drift": _number(
            summary.get("pnl_drift", drift_metrics.get("mean_pnl_drift"))
        ),
        "cost_drift_ratio": _number(
            summary.get("cost_drift_ratio", drift_metrics.get("cost_drift_ratio"))
        ),
        "regime_mismatch_rate": _number(
            summary.get(
                "regime_mismatch_rate",
                drift_metrics.get("regime_mismatch_rate"),
            )
        ),
        "final_position_count": _int(
            summary.get("final_position_count", len(positions.get("positions", ())))
        ),
    }
    execution_metrics = {
        "order_count": _int(summary.get("order_count")),
        "fill_count": _int(summary.get("fill_count")),
        "fill_rate": _number(summary.get("fill_rate")),
        "ledger_entry_count": _int(summary.get("ledger_entry_count")),
    }
    return {
        "feed_health_status": "passed" if feed_health_passed is True else "failed",
        "safety_status": (
            "kill_switch_triggered"
            if safety_action.get("kill_switch_triggered") is True
            else "passed"
            if phase5.get("passed") is True
            else "failed"
        ),
        "phase6_status": phase6_status,
        "phase5_kill_switch_triggered": (
            safety_action.get("kill_switch_triggered") is True
            or summary.get("phase5_kill_switch_triggered") is True
        ),
        "phase5_reason_codes": list(
            safety_action.get("reason_codes", summary.get("phase5_reason_codes", ()))
            or []
        ),
        "rollback_executes_reliably": phase5.get("acceptance_criteria", {}).get(
            "rollback_executes_reliably"
        ),
        "phase6_candidate_identity_verified": (
            summary.get("phase6_candidate_identity_verified")
            if "phase6_candidate_identity_verified" in summary
            else phase6.get("candidate_identity_verified")
        ),
        "feed_metrics": feed_metrics,
        "performance_metrics": performance_metrics,
        "risk_metrics": risk_metrics,
        "execution_metrics": execution_metrics,
    }


def _add_artifact_integrity_alerts(
    alerts: list[PaperAlert],
    loaded: dict[str, Any],
) -> None:
    paths = loaded["paths"]
    hashes = loaded["artifact_hashes"]
    bundle = loaded["json"]["paper_bundle_manifest"]
    summary = loaded["json"]["paper_run_summary"]
    for artifact_name, artifact in bundle.get("artifacts", {}).items():
        path = loaded["source_dir"] / artifact.get("path", "")
        if not path.exists():
            _append_alert(
                alerts,
                severity="critical",
                category="artifact",
                code="artifact_missing_from_bundle",
                message=f"Bundle references missing artifact {artifact_name}.",
                metric_name=artifact_name,
                recommendation="Regenerate the paper run bundle before review.",
            )
            continue
        observed = _sha256_file(path)
        if observed != artifact.get("sha256"):
            _append_alert(
                alerts,
                severity="critical",
                category="artifact",
                code="artifact_hash_mismatch",
                message=f"Artifact hash mismatch for {artifact_name}.",
                metric_name=artifact_name,
                metric_value=observed,
                threshold=artifact.get("sha256"),
                recommendation="Treat source artifacts as tampered or stale.",
            )
    for artifact_name, expected_hash in summary.get("artifact_hashes", {}).items():
        if artifact_name not in paths:
            continue
        observed = hashes[artifact_name]
        if observed != expected_hash:
            _append_alert(
                alerts,
                severity="critical",
                category="artifact",
                code="summary_artifact_hash_mismatch",
                message=f"Summary hash mismatch for {artifact_name}.",
                metric_name=artifact_name,
                metric_value=observed,
                threshold=expected_hash,
                recommendation="Regenerate the paper summary before review.",
            )


def _add_feed_alerts(alerts: list[PaperAlert], metrics: dict[str, Any]) -> None:
    feed = metrics["feed_metrics"]
    if feed["feed_gap_count"] > 0:
        _append_alert(
            alerts,
            severity="critical",
            category="feed",
            code="feed_gap_breach",
            message="Read-only feed had gaps above the configured threshold.",
            metric_name="feed_gap_count",
            metric_value=feed["feed_gap_count"],
            threshold=0,
            recommendation="Stop or block promotion until feed continuity is proven.",
        )
    if feed["feed_late_event_count"] > 0:
        _append_alert(
            alerts,
            severity="critical",
            category="feed",
            code="feed_late_event_breach",
            message="Read-only feed had late events.",
            metric_name="feed_late_event_count",
            metric_value=feed["feed_late_event_count"],
            threshold=0,
            recommendation="Investigate feed latency before continuing the run.",
        )
    if feed["feed_out_of_order_count"] > 0:
        _append_alert(
            alerts,
            severity="critical",
            category="feed",
            code="feed_out_of_order_breach",
            message="Read-only feed had out-of-order events.",
            metric_name="feed_out_of_order_count",
            metric_value=feed["feed_out_of_order_count"],
            threshold=0,
            recommendation="Block staged-live approval until ordering is restored.",
        )
    if feed["heartbeat_count"] == 0:
        _append_alert(
            alerts,
            severity="critical",
            category="feed",
            code="heartbeat_missing",
            message="No paper soak heartbeat rows were written.",
            metric_name="heartbeat_count",
            metric_value=feed["heartbeat_count"],
            threshold=1,
            recommendation="Treat monitoring as unavailable.",
        )
    if feed["periodic_summary_count"] == 0:
        _append_alert(
            alerts,
            severity="warning",
            category="feed",
            code="periodic_summary_missing",
            message="No periodic paper summary rows were written.",
            metric_name="periodic_summary_count",
            metric_value=feed["periodic_summary_count"],
            threshold=1,
            recommendation="Check periodic summary writer configuration.",
        )


def _add_safety_alerts(
    alerts: list[PaperAlert],
    metrics: dict[str, Any],
    phase5: dict[str, Any],
) -> None:
    if metrics["phase5_kill_switch_triggered"]:
        _append_alert(
            alerts,
            severity="critical",
            category="safety",
            code="kill_switch_triggered",
            message="Phase 5 triggered the paper kill-switch.",
            metric_name="phase5_kill_switch_triggered",
            metric_value=True,
            threshold=False,
            recommendation="Stop paper run and inspect safety reason codes.",
        )
    reason_codes = metrics["phase5_reason_codes"]
    if reason_codes:
        _append_alert(
            alerts,
            severity="critical" if metrics["phase5_kill_switch_triggered"] else "warning",
            category="safety",
            code="safety_reason_codes_present",
            message="Phase 5 emitted safety reason codes.",
            metric_name="phase5_reason_codes",
            metric_value=reason_codes,
            threshold=[],
            recommendation="Review safety-layer drift and rollback evidence.",
        )
    if metrics["rollback_executes_reliably"] is not True:
        _append_alert(
            alerts,
            severity="critical",
            category="safety",
            code="rollback_not_reliable",
            message="Phase 5 rollback reliability criterion is not true.",
            metric_name="rollback_executes_reliably",
            metric_value=metrics["rollback_executes_reliably"],
            threshold=True,
            recommendation="Do not continue until rollback evidence is reliable.",
        )
    if phase5.get("passed") is not True:
        _append_alert(
            alerts,
            severity="critical",
            category="safety",
            code="phase5_not_passed",
            message="Phase 5 report did not pass acceptance criteria.",
            metric_name="phase5_passed",
            metric_value=phase5.get("passed"),
            threshold=True,
            recommendation="Block operator approval until Phase 5 passes.",
        )


def _add_phase6_alerts(alerts: list[PaperAlert], metrics: dict[str, Any]) -> None:
    if metrics["phase6_candidate_identity_verified"] is not True:
        _append_alert(
            alerts,
            severity="critical",
            category="phase6",
            code="candidate_identity_not_verified",
            message="Phase 6 candidate identity was not verified.",
            metric_name="phase6_candidate_identity_verified",
            metric_value=metrics["phase6_candidate_identity_verified"],
            threshold=True,
            recommendation="Block approval until candidate identity is consistent.",
        )
    status = metrics["phase6_status"]
    if status == "blocked_fail_closed":
        _append_alert(
            alerts,
            severity="critical",
            category="phase6",
            code="phase6_blocked",
            message="Phase 6 blocked the run fail-closed.",
            metric_name="phase6_deployment_status",
            metric_value=status,
            threshold="approved_for_staged_live",
            recommendation="Keep the run blocked and inspect upstream gates.",
        )
    elif status != "approved_for_staged_live":
        _append_alert(
            alerts,
            severity="critical",
            category="phase6",
            code="phase6_unknown_status",
            message="Phase 6 deployment status is not recognized.",
            metric_name="phase6_deployment_status",
            metric_value=status,
            threshold="approved_for_staged_live|blocked_fail_closed",
            recommendation="Treat the run as not approved until status is clear.",
        )


def _add_paper_boundary_alerts(
    alerts: list[PaperAlert],
    payloads: dict[str, dict[str, Any]],
    jsonl: dict[str, list[dict[str, Any]]],
) -> None:
    for source_name, payload in payloads.items():
        for field_name, expected in _JSON_BOUNDARY_FIELDS.get(source_name, {}).items():
            _check_boundary_field(
                alerts,
                source_name=source_name,
                field_name=field_name,
                payload=payload,
                expected=expected,
            )
    for source_name, rows in jsonl.items():
        required_fields = _JSONL_BOUNDARY_FIELDS.get(source_name, {})
        for field_name, expected in required_fields.items():
            for row in rows:
                if _check_boundary_field(
                    alerts,
                    source_name=f"{source_name} row",
                    field_name=field_name,
                    payload=row,
                    expected=expected,
                ):
                    break
    positions = payloads["paper_positions"]
    for position in positions.get("positions", ()):
        if _check_boundary_field(
            alerts,
            source_name="paper_positions position",
            field_name="paper_only",
            payload=position,
            expected=True,
        ):
            break
        if _check_boundary_field(
            alerts,
            source_name="paper_positions position",
            field_name="capital_at_risk",
            payload=position,
            expected=False,
        ):
            break


def _check_boundary_field(
    alerts: list[PaperAlert],
    *,
    source_name: str,
    field_name: str,
    payload: dict[str, Any],
    expected: Any,
) -> bool:
    if field_name not in payload:
        _append_alert(
            alerts,
            severity="critical",
            category="paper_boundary",
            code=_BOUNDARY_MISSING_CODES[field_name],
            message=(
                f"Paper boundary field {field_name} is missing. "
                f"Source: {source_name}."
            ),
            metric_name=f"{source_name}.{field_name}",
            metric_value=None,
            threshold=expected,
            recommendation="Treat missing paper-boundary evidence as fail-closed.",
        )
        return True
    observed = payload[field_name]
    if observed != expected:
        _append_alert(
            alerts,
            severity="critical",
            category="paper_boundary",
            code=_BOUNDARY_VIOLATION_CODES[field_name],
            message=(
                f"Paper boundary field {field_name} is unsafe. "
                f"Source: {source_name}."
            ),
            metric_name=f"{source_name}.{field_name}",
            metric_value=observed,
            threshold=expected,
            recommendation="Treat this as a paper boundary violation.",
        )
        return True
    return False


def _add_performance_alerts(
    alerts: list[PaperAlert],
    metrics: dict[str, Any],
    thresholds: PaperObservabilityThresholds,
) -> None:
    risk = metrics["risk_metrics"]
    if risk["max_drawdown"] > thresholds.max_drawdown_warning:
        _append_alert(
            alerts,
            severity="warning",
            category="pnl",
            code="drawdown_threshold_breach",
            message="Paper max drawdown exceeded warning threshold.",
            metric_name="max_drawdown",
            metric_value=risk["max_drawdown"],
            threshold=thresholds.max_drawdown_warning,
            recommendation="Inspect paper PnL path before continuing.",
        )
    if risk["cost_drift_ratio"] > thresholds.max_cost_drift_ratio_warning:
        _append_alert(
            alerts,
            severity="warning",
            category="execution",
            code="cost_drift_breach",
            message="Cost drift ratio exceeded warning threshold.",
            metric_name="cost_drift_ratio",
            metric_value=risk["cost_drift_ratio"],
            threshold=thresholds.max_cost_drift_ratio_warning,
            recommendation="Inspect execution cost assumptions.",
        )
    if abs(risk["pnl_drift"]) > thresholds.max_abs_pnl_drift_warning:
        _append_alert(
            alerts,
            severity="warning",
            category="pnl",
            code="pnl_drift_breach",
            message="PnL drift exceeded warning threshold.",
            metric_name="pnl_drift",
            metric_value=risk["pnl_drift"],
            threshold=thresholds.max_abs_pnl_drift_warning,
            recommendation="Review shadow/live paper return divergence.",
        )
    if risk["regime_mismatch_rate"] > thresholds.max_regime_mismatch_rate_warning:
        _append_alert(
            alerts,
            severity="warning",
            category="execution",
            code="regime_mismatch_breach",
            message="Regime mismatch exceeded warning threshold.",
            metric_name="regime_mismatch_rate",
            metric_value=risk["regime_mismatch_rate"],
            threshold=thresholds.max_regime_mismatch_rate_warning,
            recommendation="Inspect adaptive regime handoff evidence.",
        )


def _operator_recommendation(
    alerts: list[PaperAlert],
    phase6_status: str,
) -> OperatorRecommendation:
    if phase6_status == "blocked_fail_closed":
        return "blocked_fail_closed"
    if any(alert.severity == "critical" for alert in alerts):
        return "stop_paper_run"
    if any(alert.severity == "warning" for alert in alerts):
        return "investigate_warning"
    return "continue_paper_run"


def _dashboard_summary(report: PaperRunObservabilityReport) -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-paper-observability-dashboard-v1",
        "run_id": report.run_id,
        "phase": report.phase,
        "phase6_deployment_status": report.phase6_status,
        "feed_health_status": report.feed_health_status,
        "safety_status": report.safety_status,
        "alert_count": report.alert_count,
        "critical_alert_count": report.alert_severity_counts["critical"],
        "warning_alert_count": report.alert_severity_counts["warning"],
        "operator_recommendation": report.operator_recommendation,
        "paper_only": report.paper_only,
        "capital_at_risk": report.capital_at_risk,
        "broker_exchange_write_enabled": report.broker_exchange_write_enabled,
        "live_exchange_write_enabled": report.live_exchange_write_enabled,
        "performance_metrics": report.performance_metrics,
        "risk_metrics": report.risk_metrics,
        "feed_metrics": report.feed_metrics,
        "execution_metrics": report.execution_metrics,
        "source_artifact_hashes": report.source_artifact_hashes,
        "created_at": report.created_at,
    }


def _operator_markdown(report: PaperRunObservabilityReport) -> str:
    critical = report.alert_severity_counts["critical"]
    warning = report.alert_severity_counts["warning"]
    lines = [
        "# v8 Paper Operator Summary",
        "",
        f"- run_id: `{report.run_id}`",
        f"- phase6_deployment_status: `{report.phase6_status}`",
        f"- feed_health_status: `{report.feed_health_status}`",
        f"- safety_status: `{report.safety_status}`",
        f"- alert_count: `{report.alert_count}`",
        f"- critical_alert_count: `{critical}`",
        f"- warning_alert_count: `{warning}`",
        f"- operator_recommendation: `{report.operator_recommendation}`",
        f"- paper_only: `{str(report.paper_only).lower()}`",
        f"- capital_at_risk: `{str(report.capital_at_risk).lower()}`",
        "",
        "## Key Metrics",
        "",
        f"- cumulative_net_return: `{report.performance_metrics['cumulative_net_return']}`",
        f"- max_drawdown: `{report.risk_metrics['max_drawdown']}`",
        f"- cost_drift_ratio: `{report.risk_metrics['cost_drift_ratio']}`",
        f"- feed_gap_count: `{report.feed_metrics['feed_gap_count']}`",
        f"- feed_late_event_count: `{report.feed_metrics['feed_late_event_count']}`",
        f"- feed_out_of_order_count: `{report.feed_metrics['feed_out_of_order_count']}`",
        "",
        "## Alerts",
        "",
    ]
    if not report.alerts:
        lines.append("- none")
    else:
        for alert in report.alerts:
            lines.append(
                f"- `{alert['severity']}` `{alert['category']}` "
                f"`{alert['code']}`: {alert['message']}"
            )
    lines.append("")
    return "\n".join(lines)


def _comparison_markdown(report: PaperRunComparisonReport) -> str:
    return "\n".join(
        [
            "# v8 Paper Run Comparison",
            "",
            f"- left_run_id: `{report.left_run_id}`",
            f"- right_run_id: `{report.right_run_id}`",
            f"- status_change: `{report.status_change}`",
            f"- phase6_status_change: `{report.phase6_status_change}`",
            f"- recommendation: `{report.recommendation}`",
            "",
            "## Alert Deltas",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in sorted(report.alert_deltas.items())
            ],
            "",
            "## Metric Deltas",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in sorted(report.metric_deltas.items())
            ],
            "",
        ]
    )


def _append_alert(
    alerts: list[PaperAlert],
    *,
    severity: AlertSeverity,
    category: AlertCategory,
    code: str,
    message: str,
    metric_name: str | None = None,
    metric_value: Any = None,
    threshold: Any = None,
    recommendation: str | None = None,
) -> None:
    alerts.append(
        PaperAlert(
            alert_id=f"paper_alert_{len(alerts) + 1:04d}",
            severity=severity,
            category=category,
            code=code,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            recommendation=recommendation,
        )
    )


def _alert_severity_counts(alerts: list[PaperAlert]) -> dict[str, int]:
    return {
        "info": sum(alert.severity == "info" for alert in alerts),
        "warning": sum(alert.severity == "warning" for alert in alerts),
        "critical": sum(alert.severity == "critical" for alert in alerts),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(json_ready(row), sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _write_periodic_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = (
        "run_id",
        "summary_ts",
        "feed_event_count",
        "heartbeat_count",
        "paper_only",
        "capital_at_risk",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return default
    return observed if math.isfinite(observed) else default


def _int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
