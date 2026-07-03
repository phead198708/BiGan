"""Model promotion rules and reports (issue #19)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

REQUIRED_MARKET_FAMILIES: tuple[str, ...] = ("BTC-15M", "ETH-15M", "BTC-5M", "ETH-5M")
NEW_MARKET_SIGNAL_FAMILIES: tuple[str, ...] = ("ETH-15M", "ETH-5M")
SPLITS_FOR_DETAIL: tuple[str, ...] = ("train", "val", "test")
STATUS_ARTIFACT_MAX_AGE_SECONDS = 30 * 60
EXPECTED_XGBOOST_V4_LIVE_ROOT = Path(
    "data/live/xgboost-v4-multimarket-7d-atomic-20260523T125657Z"
)
EXPECTED_XGBOOST_V4_SCREEN_SESSION = "xgbv4_7d_atomic_20260523T125657Z"
DEFAULT_CHAMPION_PROMOTION_PROCESS_PATH = Path("/Users/tcscoder/Downloads/champion-promotion.md")
DEFAULT_CHAMPION_PROMOTION_REPO_RUNBOOK_PATH = Path("docs/runbooks/champion_promotion.md")
REQUIRED_BOOTSTRAP_CHECKLIST_KEYS: tuple[str, ...] = (
    "beats_baseline",
    "calibration_acceptable",
    "backtest_acceptable",
    "serving_readiness_acceptable",
    "rollback_fallback_available",
    "schema_stable",
    "simple_enough",
)
REQUIRED_CUTOVER_GITHUB_ISSUES: tuple[int, ...] = (52, 53)
EXPECTED_CUTOVER_GITHUB_REPO = "phead198708/BiGan"
REQUIRED_CUTOVER_REGISTRY_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("auc", ("auc", "roc_auc")),
    ("brier", ("brier", "brier_score")),
    ("net_pnl", ("net_pnl", "best_net_pnl")),
    ("delta_vs_baseline", ("delta_vs_baseline", "net_pnl_delta")),
    ("max_dd", ("max_dd", "max_drawdown", "max_drawdown_pct")),
    ("sharpe", ("sharpe", "sharpe_ratio")),
    ("edge_trigger_rate", ("edge_trigger_rate", "challenger_edge_trigger_rate")),
    ("shadow_p95_ms", ("shadow_p95_ms", "p95_ms", "p95_latency_ms", "serving_latency_ms")),
    ("schema_error_rate", ("schema_error_rate",)),
)


@dataclass(frozen=True, slots=True)
class PromotionRules:
    """Thresholds for promoting a candidate model over the baseline."""

    min_roc_auc_delta: float = 0.0
    max_brier_delta: float = 0.0
    min_backtest_net_pnl: float = 0.0
    require_calibration_improved: bool = True

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    """One pass/fail promotion checklist item."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, bool | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Serializable model-promotion decision report."""

    passed: bool
    decision: str
    baseline_model_version: str
    candidate_model_version: str
    dataset_version: str | None
    checks: tuple[PromotionCheck, ...]
    baseline_test_metrics: dict[str, float | int | None]
    candidate_test_metrics: dict[str, float | int | None]
    calibration: dict[str, Any] | None
    backtest: dict[str, Any] | None
    rules: PromotionRules
    artifact_paths: dict[str, str]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "baseline_model_version": self.baseline_model_version,
            "candidate_model_version": self.candidate_model_version,
            "dataset_version": self.dataset_version,
            "checks": [check.to_dict() for check in self.checks],
            "baseline_test_metrics": self.baseline_test_metrics,
            "candidate_test_metrics": self.candidate_test_metrics,
            "calibration": self.calibration,
            "backtest": self.backtest,
            "rules": self.rules.to_dict(),
            "artifact_paths": self.artifact_paths,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True, slots=True)
class ChampionPromotionAuditRules:
    """Conservative gates from the champion-promotion runbook."""

    max_candidate_ece: float = 0.05
    max_drawdown_pct: float = 0.20
    max_turnover_rate: float = 0.50
    allow_lower_sharpe_if_brier_gap: float = 0.05
    min_shadow_session_seconds: int = 86_400
    max_prediction_latency_ms: float = 50.0
    max_shadow_probability_mean_abs_diff: float = 0.05
    max_shadow_probability_std_relative_diff: float = 0.20
    min_new_market_roc_auc: float = 0.50

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChampionPromotionGateCheck:
    """One concrete champion-promotion evidence check."""

    name: str
    passed: bool
    detail: str
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, bool | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChampionPromotionGate:
    """A sequential stage from ``champion-promotion.md``."""

    name: str
    passed: bool
    checks: tuple[ChampionPromotionGateCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class ChampionPromotionAuditReport:
    """Full prompt-to-artifact champion-promotion audit result."""

    passed: bool
    decision: str
    expected_candidate_model_version: str
    expected_fallback_model_version: str
    stages: tuple[ChampionPromotionGate, ...]
    rules: ChampionPromotionAuditRules
    artifact_paths: dict[str, str | None]
    promotion_process: dict[str, Any]
    output_dir: str

    @property
    def earliest_failed_stage(self) -> str | None:
        for stage in self.stages:
            if not stage.passed:
                return stage.name
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "earliest_failed_stage": self.earliest_failed_stage,
            "expected_candidate_model_version": self.expected_candidate_model_version,
            "expected_fallback_model_version": self.expected_fallback_model_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "rules": self.rules.to_dict(),
            "artifact_paths": self.artifact_paths,
            "promotion_process": self.promotion_process,
            "output_dir": self.output_dir,
        }

    def to_markdown(self) -> str:
        source = self.promotion_process.get("source_path")
        source_sha = self.promotion_process.get("source_sha256")
        lines = [
            "# Champion Promotion Audit",
            "",
            f"Decision: **{self.decision}**",
            f"Earliest failed stage: **{self.earliest_failed_stage or 'None'}**",
            "",
            f"- Expected candidate: `{self.expected_candidate_model_version}`",
            f"- Expected fallback: `{self.expected_fallback_model_version}`",
            f"- Promotion process source: `{source or 'not supplied'}`",
            f"- Promotion process SHA256: `{source_sha or 'not available'}`",
            "",
            "## Stage Checklist",
            "| Stage | Status | Check | Detail |",
            "|---|---|---|---|",
        ]
        for stage in self.stages:
            stage_status = "PASS" if stage.passed else "FAIL"
            for check in stage.checks:
                check_status = "PASS" if check.passed else "FAIL"
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _escape_markdown_table(stage.name),
                            stage_status,
                            f"{check_status}: `{check.name}`",
                            _escape_markdown_table(check.detail),
                        )
                    )
                    + " |"
                )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class OfflineRerunReport:
    """Stage 1 same-dataset offline comparison report."""

    passed: bool
    decision: str
    baseline_model_version: str | None
    candidate_model_version: str | None
    dataset_dir: str | None
    dataset_version: str | None
    checks: tuple[ChampionPromotionGateCheck, ...]
    baseline_test_metrics: dict[str, Any]
    candidate_test_metrics: dict[str, Any]
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "baseline_model_version": self.baseline_model_version,
            "candidate_model_version": self.candidate_model_version,
            "dataset_dir": self.dataset_dir,
            "dataset_version": self.dataset_version,
            "checks": [check.to_dict() for check in self.checks],
            "baseline_test_metrics": self.baseline_test_metrics,
            "candidate_test_metrics": self.candidate_test_metrics,
            "output_path": self.output_path,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Rerun Report",
            "",
            "Stage 1 offline evaluation for `champion-promotion.md`.",
            "",
            f"Decision: **{self.decision}**",
            "",
            f"- Baseline: `{self.baseline_model_version}`",
            f"- Candidate: `{self.candidate_model_version}`",
            f"- Dataset: `{self.dataset_dir}`",
            f"- Dataset version: `{self.dataset_version}`",
            "",
            "## Test Metrics",
            "| Metric | Baseline | Candidate |",
            "|---|---:|---:|",
        ]
        for metric in ("roc_auc", "brier_score", "ece", "sample_count"):
            lines.append(
                "| "
                + " | ".join(
                    (
                        metric,
                        _format_metric_cell(self.baseline_test_metrics.get(metric)),
                        _format_metric_cell(self.candidate_test_metrics.get(metric)),
                    )
                )
                + " |"
            )
        lines.extend(["", "## Checks"])
        for check in self.checks:
            mark = "x" if check.passed else " "
            lines.append(f"- [{mark}] `{check.name}`: {check.detail}")
        return "\n".join(lines) + "\n"


def generate_offline_rerun_report(
    *,
    baseline_eval_dir: Path | str,
    candidate_eval_dir: Path | str,
    output_path: Path | str,
    expected_candidate_model_version: str = "xgboost-v4",
    rules: ChampionPromotionAuditRules | None = None,
) -> OfflineRerunReport:
    """Write the Stage 1 ``rerun_report.md`` required by champion-promotion.md."""

    active_rules = rules or ChampionPromotionAuditRules()
    baseline_manifest = _read_eval_json(baseline_eval_dir, "manifest.json")
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    baseline_metrics = _read_eval_json(baseline_eval_dir, "metrics.json")
    candidate_metrics = _read_eval_json(candidate_eval_dir, "metrics.json")
    baseline_family_metrics = _read_eval_json(baseline_eval_dir, "family_metrics.json")
    candidate_family_metrics = _read_eval_json(candidate_eval_dir, "family_metrics.json")
    baseline_test = _test_metrics_or_empty(baseline_metrics)
    candidate_test = _test_metrics_or_empty(candidate_metrics)
    checks = tuple(
        _offline_eval_checks(
            baseline_manifest=baseline_manifest,
            candidate_manifest=candidate_manifest,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_family_metrics=baseline_family_metrics,
            candidate_family_metrics=candidate_family_metrics,
            baseline_test=baseline_test,
            candidate_test=candidate_test,
            expected_candidate_model_version=expected_candidate_model_version,
            rules=active_rules,
            artifact_path=candidate_eval_dir,
        )
    )
    report = OfflineRerunReport(
        passed=all(check.passed for check in checks),
        decision="PASS" if all(check.passed for check in checks) else "FAIL",
        baseline_model_version=_dict_str(baseline_manifest, "model_version"),
        candidate_model_version=_dict_str(candidate_manifest, "model_version"),
        dataset_dir=_dict_str(candidate_manifest, "dataset_dir"),
        dataset_version=_dict_str(candidate_manifest, "dataset_version"),
        checks=checks,
        baseline_test_metrics=baseline_test,
        candidate_test_metrics=candidate_test,
        output_path=str(output_path),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(target, report.to_markdown())
    _write_json_atomic(target.with_suffix(".json"), report.to_dict())
    return report


def audit_champion_promotion_process(
    *,
    output_dir: Path | str,
    promotion_process_path: Path | str | None = None,
    repo_promotion_runbook_path: Path | str | None = None,
    live_status_path: Path | str | None = None,
    offline_rerun_report_path: Path | str | None = None,
    baseline_eval_dir: Path | str | None = None,
    candidate_eval_dir: Path | str | None = None,
    baseline_backtest_summary_path: Path | str | None = None,
    candidate_backtest_summary_path: Path | str | None = None,
    shadow_evaluation_path: Path | str | None = None,
    serving_readiness_path: Path | str | None = None,
    bootstrap_decision_path: Path | str | None = None,
    cutover_report_path: Path | str | None = None,
    rollback_runbook_path: Path | str | None = None,
    expected_candidate_model_version: str = "xgboost-v4",
    expected_fallback_model_version: str = "xgboost-v3",
    rules: ChampionPromotionAuditRules | None = None,
) -> ChampionPromotionAuditReport:
    """Audit every sequential gate from ``champion-promotion.md``.

    The audit is intentionally fail-closed: missing artifacts, stale data
    readiness, mismatched datasets, zero-cost backtests, skipped shadow
    duration evidence, or stale cutover evidence all block promotion.
    """

    active_rules = rules or ChampionPromotionAuditRules()
    paths = {
        "promotion_process_path": _path_str(promotion_process_path),
        "repo_promotion_runbook_path": _path_str(repo_promotion_runbook_path),
        "live_status_path": _path_str(live_status_path),
        "offline_rerun_report_path": _path_str(offline_rerun_report_path),
        "baseline_eval_dir": _path_str(baseline_eval_dir),
        "candidate_eval_dir": _path_str(candidate_eval_dir),
        "baseline_backtest_summary_path": _path_str(baseline_backtest_summary_path),
        "candidate_backtest_summary_path": _path_str(candidate_backtest_summary_path),
        "shadow_evaluation_path": _path_str(shadow_evaluation_path),
        "serving_readiness_path": _path_str(serving_readiness_path),
        "bootstrap_decision_path": _path_str(bootstrap_decision_path),
        "cutover_report_path": _path_str(cutover_report_path),
        "rollback_runbook_path": _path_str(rollback_runbook_path),
    }
    promotion_process = _promotion_process_source_evidence(
        promotion_process_path=promotion_process_path,
        repo_promotion_runbook_path=repo_promotion_runbook_path,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stages = (
        _data_readiness_gate(live_status_path, promotion_process=promotion_process),
        _offline_eval_gate(
            offline_rerun_report_path=offline_rerun_report_path,
            baseline_eval_dir=baseline_eval_dir,
            candidate_eval_dir=candidate_eval_dir,
            expected_candidate_model_version=expected_candidate_model_version,
            rules=active_rules,
        ),
        _backtest_gate(
            baseline_backtest_summary_path=baseline_backtest_summary_path,
            candidate_backtest_summary_path=candidate_backtest_summary_path,
            baseline_eval_dir=baseline_eval_dir,
            candidate_eval_dir=candidate_eval_dir,
            rules=active_rules,
        ),
        _shadow_gate(
            shadow_evaluation_path,
            baseline_eval_dir=baseline_eval_dir,
            candidate_eval_dir=candidate_eval_dir,
            expected_candidate_model_version=expected_candidate_model_version,
            rules=active_rules,
        ),
        _bootstrap_gate(
            bootstrap_decision_path=bootstrap_decision_path,
            baseline_eval_dir=baseline_eval_dir,
            candidate_eval_dir=candidate_eval_dir,
            baseline_backtest_summary_path=baseline_backtest_summary_path,
            candidate_backtest_summary_path=candidate_backtest_summary_path,
            shadow_evaluation_path=shadow_evaluation_path,
            serving_readiness_path=serving_readiness_path,
            rollback_runbook_path=rollback_runbook_path,
            expected_candidate_model_version=expected_candidate_model_version,
            rules=active_rules,
        ),
        _cutover_gate(
            cutover_report_path=cutover_report_path,
            candidate_eval_dir=candidate_eval_dir,
            bootstrap_decision_path=bootstrap_decision_path,
            shadow_evaluation_path=shadow_evaluation_path,
            serving_readiness_path=serving_readiness_path,
            expected_candidate_model_version=expected_candidate_model_version,
            expected_fallback_model_version=expected_fallback_model_version,
            rules=active_rules,
        ),
    )
    passed = all(stage.passed for stage in stages)
    report = ChampionPromotionAuditReport(
        passed=passed,
        decision="PROMOTION_COMPLETE" if passed else "BLOCKED",
        expected_candidate_model_version=expected_candidate_model_version,
        expected_fallback_model_version=expected_fallback_model_version,
        stages=stages,
        rules=active_rules,
        artifact_paths=paths,
        promotion_process=promotion_process,
        output_dir=str(target),
    )
    _write_json_atomic(target / "champion_promotion_audit.json", report.to_dict())
    _write_text_atomic(target / "champion_promotion_audit.md", report.to_markdown())
    return report


def evaluate_model_promotion(
    baseline_dir: Path | str,
    candidate_dir: Path | str,
    calibration_dir: Path | str,
    backtest_summary_path: Path | str,
    output_dir: Path | str,
    *,
    rules: PromotionRules | None = None,
) -> PromotionReport:
    """Compare baseline/candidate metrics and write a promotion decision."""

    active_rules = rules or PromotionRules()
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    calibration_root = Path(calibration_dir)
    backtest_path = Path(backtest_summary_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    baseline_manifest = _read_json(baseline_root / "manifest.json")
    candidate_manifest = _read_json(candidate_root / "manifest.json")
    baseline_metrics = _read_json(baseline_root / "metrics.json")
    candidate_metrics = _read_json(candidate_root / "metrics.json")
    calibration = _read_optional_json(calibration_root / "calibration_report.json")
    backtest = _summarize_backtest(_read_optional_json(backtest_path))

    baseline_test = _split_metrics(baseline_metrics, "test")
    candidate_test = _split_metrics(candidate_metrics, "test")
    baseline_model_version = str(baseline_manifest.get("model_version") or "unknown")
    candidate_model_version = str(candidate_manifest.get("model_version") or "unknown")
    dataset_version = _dataset_version(baseline_manifest, candidate_manifest)

    checks = tuple(
        _build_checks(
            baseline_test=baseline_test,
            candidate_test=candidate_test,
            calibration=calibration,
            backtest=backtest,
            dataset_version=dataset_version,
            baseline_model_version=baseline_model_version,
            candidate_model_version=candidate_model_version,
            rules=active_rules,
        )
    )
    passed = all(check.passed for check in checks)
    report = PromotionReport(
        passed=passed,
        decision="promote" if passed else "reject",
        baseline_model_version=baseline_model_version,
        candidate_model_version=candidate_model_version,
        dataset_version=dataset_version,
        checks=checks,
        baseline_test_metrics=baseline_test,
        candidate_test_metrics=candidate_test,
        calibration=calibration,
        backtest=backtest,
        rules=active_rules,
        artifact_paths={
            "baseline_dir": str(baseline_root),
            "candidate_dir": str(candidate_root),
            "calibration_dir": str(calibration_root),
            "backtest_summary_path": str(backtest_path),
        },
        output_dir=str(target),
    )
    _write_json_atomic(target / "promotion_report.json", report.to_dict())
    _write_text_atomic(target / "promotion_checklist.md", _checklist_markdown(report))
    return report


def _data_readiness_gate(
    live_status_path: Path | str | None,
    *,
    promotion_process: dict[str, Any] | None = None,
) -> ChampionPromotionGate:
    status = _read_optional_json(None if live_status_path is None else Path(live_status_path))
    checks = [
        _audit_check(
            "status_artifact_exists",
            isinstance(status, dict),
            "live collection status JSON is available"
            if isinstance(status, dict)
            else "live collection status JSON is missing or unreadable",
            live_status_path,
        )
    ]
    if isinstance(promotion_process, dict) and promotion_process.get("checked"):
        checks.append(
            _audit_check(
                "promotion_process_source",
                _is_true(promotion_process.get("passed")),
                _promotion_process_source_detail(promotion_process),
                _path_str(promotion_process.get("source_path")),
            )
        )
    if not isinstance(status, dict):
        return _audit_gate("Stage 0: 7-day Data Readiness", checks)

    readiness = status.get("collection_readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    feature = readiness.get("features_15m_v1")
    feature = feature if isinstance(feature, dict) else {}
    labels = readiness.get("labels_15m_v1")
    labels = labels if isinstance(labels, dict) else {}
    invalid_gzip = _metric(
        status.get("raw_segment_integrity") if isinstance(status.get("raw_segment_integrity"), dict) else {},
        "invalid_count",
    )
    raw_quarantine = status.get("raw_segment_quarantine")
    raw_quarantine = raw_quarantine if isinstance(raw_quarantine, dict) else {}
    quarantined_raw_segments = _metric(raw_quarantine, "quarantined_count")
    quarantine_clean_window = readiness.get("quarantine_clean_window")
    quarantine_clean_window = (
        quarantine_clean_window if isinstance(quarantine_clean_window, dict) else {}
    )
    quarantine_clean_window_ready = _is_true(quarantine_clean_window.get("meets_target"))
    unrecovered_errors = _metric(
        status.get("health_evidence") if isinstance(status.get("health_evidence"), dict) else {},
        "unrecovered_error_match_count",
    )
    disk_headroom = status.get("disk_headroom_evidence")
    disk_headroom = disk_headroom if isinstance(disk_headroom, dict) else {}
    disk_headroom_ok = _is_true(disk_headroom.get("headroom_ok"))
    current_disk_headroom = _current_filesystem_headroom_evidence(status, disk_headroom)
    current_disk_headroom_ok = (
        current_disk_headroom is None or _is_true(current_disk_headroom.get("headroom_ok"))
    )
    raw_manifest_coverage = status.get("raw_manifest_coverage_evidence")
    raw_manifest_coverage = (
        raw_manifest_coverage if isinstance(raw_manifest_coverage, dict) else {}
    )
    stale_missing_processed = _metric(raw_manifest_coverage, "stale_missing_processed_count")
    extra_processed = _metric(raw_manifest_coverage, "extra_processed_count")
    raw_manifest_coverage_ok = stale_missing_processed == 0.0 and extra_processed == 0.0
    live_root_lock = status.get("live_root_lock_evidence")
    live_root_lock = live_root_lock if isinstance(live_root_lock, dict) else {}
    live_root_lock_ok = (
        _is_true(live_root_lock.get("lock_dir_exists"))
        and _is_true(live_root_lock.get("pid_file_exists"))
        and _is_true(live_root_lock.get("owner_running"))
        and not live_root_lock.get("pid_parse_error")
    )
    collector_process_liveness_ok = status.get("screen_state") == "running" and live_root_lock_ok
    liveness = status.get("liveness_evidence")
    liveness = liveness if isinstance(liveness, dict) else {}
    warehouse = status.get("warehouse_freshness_evidence")
    warehouse_tables = (
        warehouse.get("tables")
        if isinstance(warehouse, dict) and isinstance(warehouse.get("tables"), dict)
        else {}
    )
    warehouse_fresh = bool(warehouse_tables) and all(
        isinstance(row, dict) and _is_true(row.get("fresh")) for row in warehouse_tables.values()
    )
    label_freshness = status.get("label_freshness_evidence")
    label_freshness = label_freshness if isinstance(label_freshness, dict) else {}
    label_fresh = _is_true(label_freshness.get("fresh"))
    clean_corpus_root = _path_matches(status.get("live_root"), EXPECTED_XGBOOST_V4_LIVE_ROOT)
    clean_corpus_warehouse = _path_matches(
        status.get("warehouse"),
        EXPECTED_XGBOOST_V4_LIVE_ROOT / "warehouse",
    )
    clean_corpus_screen = status.get("screen_session") == EXPECTED_XGBOOST_V4_SCREEN_SESSION
    status_age_seconds = _status_artifact_age_seconds(status)
    status_artifact_fresh = (
        status_age_seconds is not None
        and status_age_seconds <= STATUS_ARTIFACT_MAX_AGE_SECONDS
    )
    checks.extend(
        [
            _audit_check(
                "clean_atomic_live_root",
                clean_corpus_root and clean_corpus_warehouse and clean_corpus_screen,
                (
                    f"live_root={status.get('live_root')}, "
                    f"warehouse={status.get('warehouse')}, "
                    f"screen_session={status.get('screen_session')}, "
                    f"expected_live_root={EXPECTED_XGBOOST_V4_LIVE_ROOT}, "
                    f"expected_screen_session={EXPECTED_XGBOOST_V4_SCREEN_SESSION}"
                ),
                live_status_path,
            ),
            _audit_check(
                "status_artifact_fresh",
                status_artifact_fresh,
                (
                    f"generated_at={status.get('generated_at')}, "
                    f"age_seconds={_format_float(status_age_seconds)}, "
                    f"max_age_seconds={STATUS_ARTIFACT_MAX_AGE_SECONDS}"
                ),
                live_status_path,
            ),
            _audit_check(
                "ready_for_training",
                _is_true(readiness.get("ready_for_training")),
                _data_readiness_detail(readiness, feature, labels),
                live_status_path,
            ),
            _audit_check(
                "raw_segment_integrity",
                invalid_gzip == 0.0,
                f"invalid gzip segments={_format_float(invalid_gzip)}",
                live_status_path,
            ),
            _audit_check(
                "raw_segment_quarantine",
                quarantined_raw_segments == 0.0 or quarantine_clean_window_ready,
                _raw_segment_quarantine_detail(
                    raw_quarantine,
                    quarantine_clean_window,
                    quarantined_raw_segments=quarantined_raw_segments,
                    quarantine_clean_window_ready=quarantine_clean_window_ready,
                ),
                live_status_path,
            ),
            _audit_check(
                "unrecovered_log_errors",
                unrecovered_errors == 0.0,
                f"unrecovered fatal-pattern matches={_format_float(unrecovered_errors)}",
                live_status_path,
            ),
            _audit_check(
                "disk_headroom",
                disk_headroom_ok and current_disk_headroom_ok,
                (
                    f"headroom_ok={disk_headroom_ok}, "
                    f"free_bytes={disk_headroom.get('free_bytes')}, "
                    f"required_free_bytes={disk_headroom.get('required_free_bytes')}, "
                    f"projected_remaining_bytes={disk_headroom.get('projected_remaining_bytes')}, "
                    f"headroom_margin_bytes={disk_headroom.get('headroom_margin_bytes')}, "
                    f"headroom_low_margin={disk_headroom.get('headroom_low_margin')}, "
                    f"current_filesystem={_current_filesystem_headroom_detail(current_disk_headroom)}"
                ),
                live_status_path,
            ),
            _audit_check(
                "collector_process_liveness",
                collector_process_liveness_ok,
                (
                    f"screen_state={status.get('screen_state')}, "
                    f"lock_dir_exists={_is_true(live_root_lock.get('lock_dir_exists'))}, "
                    f"pid_file_exists={_is_true(live_root_lock.get('pid_file_exists'))}, "
                    f"pid={live_root_lock.get('pid')}, "
                    f"owner_running={_is_true(live_root_lock.get('owner_running'))}, "
                    f"pid_parse_error={live_root_lock.get('pid_parse_error')}"
                ),
                live_status_path,
            ),
            _audit_check(
                "raw_manifest_coverage",
                raw_manifest_coverage_ok,
                (
                    f"stale_missing_processed_count={_format_float(stale_missing_processed)}, "
                    f"extra_processed_count={_format_float(extra_processed)}"
                ),
                live_status_path,
            ),
            _audit_check(
                "raw_and_manifest_fresh",
                _is_true(liveness.get("raw_segments_fresh"))
                and _is_true(liveness.get("processed_manifest_fresh")),
                (
                    f"raw_fresh={_is_true(liveness.get('raw_segments_fresh'))}, "
                    f"manifest_fresh={_is_true(liveness.get('processed_manifest_fresh'))}"
                ),
                live_status_path,
            ),
            _audit_check(
                "warehouse_fresh",
                warehouse_fresh,
                f"fresh warehouse tables={warehouse_fresh}",
                live_status_path,
            ),
            _audit_check(
                "label_freshness",
                label_fresh,
                (
                    f"fresh={label_fresh}, "
                    f"stale_families={label_freshness.get('stale_families') or []}, "
                    f"missing_label_families="
                    f"{label_freshness.get('missing_label_families') or []}"
                ),
                live_status_path,
            ),
        ]
    )
    return _audit_gate("Stage 0: 7-day Data Readiness", checks)


def _offline_eval_gate(
    *,
    offline_rerun_report_path: Path | str | None,
    baseline_eval_dir: Path | str | None,
    candidate_eval_dir: Path | str | None,
    expected_candidate_model_version: str,
    rules: ChampionPromotionAuditRules,
) -> ChampionPromotionGate:
    baseline_manifest = _read_eval_json(baseline_eval_dir, "manifest.json")
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    baseline_metrics = _read_eval_json(baseline_eval_dir, "metrics.json")
    candidate_metrics = _read_eval_json(candidate_eval_dir, "metrics.json")
    baseline_family_metrics = _read_eval_json(baseline_eval_dir, "family_metrics.json")
    candidate_family_metrics = _read_eval_json(candidate_eval_dir, "family_metrics.json")
    baseline_test = _test_metrics_or_empty(baseline_metrics)
    candidate_test = _test_metrics_or_empty(candidate_metrics)
    checks = [
        _audit_check(
            "rerun_report_exists",
            _valid_rerun_report(
                offline_rerun_report_path,
                expected_candidate_model_version=expected_candidate_model_version,
                baseline_manifest=baseline_manifest,
                candidate_manifest=candidate_manifest,
            ),
            (
                "`rerun_report.md` exists with a matching passing JSON sidecar for "
                f"{expected_candidate_model_version}"
            ),
            offline_rerun_report_path,
        ),
        *_offline_eval_checks(
            baseline_manifest=baseline_manifest,
            candidate_manifest=candidate_manifest,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_family_metrics=baseline_family_metrics,
            candidate_family_metrics=candidate_family_metrics,
            baseline_test=baseline_test,
            candidate_test=candidate_test,
            expected_candidate_model_version=expected_candidate_model_version,
            rules=rules,
            artifact_path=candidate_eval_dir,
        ),
    ]
    return _audit_gate("Stage 1: Offline Evaluation", checks)


def _offline_eval_checks(
    *,
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    candidate_metrics: dict[str, Any] | None,
    baseline_family_metrics: dict[str, Any] | None,
    candidate_family_metrics: dict[str, Any] | None,
    baseline_test: dict[str, Any],
    candidate_test: dict[str, Any],
    expected_candidate_model_version: str,
    rules: ChampionPromotionAuditRules,
    artifact_path: Path | str | None,
) -> list[ChampionPromotionGateCheck]:
    baseline_auc = _metric(baseline_test, "roc_auc")
    candidate_auc = _metric(candidate_test, "roc_auc")
    baseline_brier = _metric(baseline_test, "brier_score")
    candidate_brier = _metric(candidate_test, "brier_score")
    candidate_ece = _metric(candidate_test, "ece")
    dataset_same = _same_eval_dataset(baseline_manifest, candidate_manifest)
    return [
        _audit_check(
            "baseline_eval_exists",
            isinstance(baseline_manifest, dict) and isinstance(baseline_metrics, dict),
            "baseline same-dataset evaluation artifacts exist",
            artifact_path,
        ),
        _audit_check(
            "candidate_eval_exists",
            isinstance(candidate_manifest, dict) and isinstance(candidate_metrics, dict),
            "candidate same-dataset evaluation artifacts exist",
            artifact_path,
        ),
        _audit_check(
            "same_dataset_split",
            dataset_same,
            _dataset_detail(baseline_manifest, candidate_manifest),
            artifact_path,
        ),
        _audit_check(
            "dataset_time_split_5_1_1",
            _valid_dataset_time_split(candidate_manifest),
            _dataset_time_split_detail(candidate_manifest),
            _dataset_dir_from_manifest(candidate_manifest),
        ),
        _audit_check(
            "dataset_required_families_present",
            _dataset_required_families_present(candidate_manifest),
            _dataset_required_families_detail(candidate_manifest),
            _dataset_dir_from_manifest(candidate_manifest),
        ),
        _audit_check(
            "required_family_metrics_present",
            _required_family_metrics_present(
                baseline_family_metrics,
                candidate_family_metrics,
            ),
            _required_family_metrics_detail(
                baseline_family_metrics,
                candidate_family_metrics,
            ),
            artifact_path,
        ),
        _audit_check(
            "new_market_signal_present",
            _new_market_signal_present(candidate_family_metrics, rules),
            _new_market_signal_detail(candidate_family_metrics, rules),
            artifact_path,
        ),
        _audit_check(
            "candidate_model_version",
            _dict_str(candidate_manifest, "model_version") == expected_candidate_model_version,
            (
                f"candidate={_dict_str(candidate_manifest, 'model_version')}, "
                f"expected={expected_candidate_model_version}"
            ),
            artifact_path,
        ),
        _audit_check(
            "test_auc_beats_champion",
            (
                baseline_auc is not None
                and candidate_auc is not None
                and candidate_auc > baseline_auc
            ),
            f"candidate AUC={_format_float(candidate_auc)}, baseline AUC={_format_float(baseline_auc)}",
            artifact_path,
        ),
        _audit_check(
            "test_brier_beats_champion",
            (
                baseline_brier is not None
                and candidate_brier is not None
                and candidate_brier < baseline_brier
            ),
            (
                f"candidate Brier={_format_float(candidate_brier)}, "
                f"baseline Brier={_format_float(baseline_brier)}"
            ),
            artifact_path,
        ),
        _audit_check(
            "calibrated_ece",
            candidate_ece is not None and candidate_ece < rules.max_candidate_ece,
            (
                f"candidate ECE={_format_float(candidate_ece)}, "
                f"required < {rules.max_candidate_ece:.4f}"
            ),
            artifact_path,
        ),
        _audit_check(
            "candidate_calibration_applied",
            bool(_dict_str(candidate_manifest, "calibration_path"))
            or bool(_dict_str(candidate_manifest, "calibration_method")),
            "candidate eval manifest includes calibration evidence",
            artifact_path,
        ),
    ]


def _backtest_gate(
    *,
    baseline_backtest_summary_path: Path | str | None,
    candidate_backtest_summary_path: Path | str | None,
    baseline_eval_dir: Path | str | None,
    candidate_eval_dir: Path | str | None,
    rules: ChampionPromotionAuditRules,
) -> ChampionPromotionGate:
    baseline_row = _best_backtest_row(baseline_backtest_summary_path)
    candidate_row = _best_backtest_row(candidate_backtest_summary_path)
    baseline_diagnostics = _backtest_diagnostics(baseline_backtest_summary_path)
    candidate_diagnostics = _backtest_diagnostics(candidate_backtest_summary_path)
    baseline_manifest = _read_eval_json(baseline_eval_dir, "manifest.json")
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    baseline_net_pnl = _metric(baseline_row or {}, "net_pnl")
    candidate_net_pnl = _metric(candidate_row or {}, "net_pnl")
    candidate_drawdown = _first_metric(candidate_row or {}, ("max_drawdown_pct", "max_drawdown"))
    baseline_sharpe = _first_metric(baseline_row or {}, ("sharpe_ratio", "sharpe"))
    candidate_sharpe = _first_metric(candidate_row or {}, ("sharpe_ratio", "sharpe"))
    turnover = _turnover_rate(candidate_row or {})
    brier_improvement = _eval_brier_improvement(baseline_eval_dir, candidate_eval_dir)
    net_pnl_delta = (
        None
        if baseline_net_pnl is None or candidate_net_pnl is None
        else candidate_net_pnl - baseline_net_pnl
    )
    lower_sharpe_allowed = _audit_lower_sharpe_allowed(
        candidate_sharpe=candidate_sharpe,
        baseline_sharpe=baseline_sharpe,
        net_pnl_delta=net_pnl_delta,
        brier_improvement=brier_improvement,
        rules=rules,
    )
    baseline_cost_settings = _backtest_cost_settings(baseline_row)
    candidate_cost_settings = _backtest_cost_settings(candidate_row)
    checks = [
        _audit_check(
            "baseline_backtest_exists",
            baseline_row is not None,
            "baseline/champion cost-adjusted backtest summary exists",
            baseline_backtest_summary_path,
        ),
        _audit_check(
            "candidate_backtest_exists",
            candidate_row is not None,
            "candidate cost-adjusted backtest summary exists",
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "baseline_backtest_summary_matches_diagnostics",
            _backtest_summary_matches_diagnostics(baseline_backtest_summary_path),
            _backtest_summary_diagnostics_detail(baseline_backtest_summary_path),
            baseline_backtest_summary_path,
        ),
        _audit_check(
            "candidate_backtest_summary_matches_diagnostics",
            _backtest_summary_matches_diagnostics(candidate_backtest_summary_path),
            _backtest_summary_diagnostics_detail(candidate_backtest_summary_path),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "baseline_backtest_matches_eval",
            _backtest_matches_eval(baseline_diagnostics, baseline_manifest),
            _backtest_eval_match_detail(baseline_diagnostics, baseline_manifest),
            baseline_backtest_summary_path,
        ),
        _audit_check(
            "candidate_backtest_matches_eval",
            _backtest_matches_eval(candidate_diagnostics, candidate_manifest),
            _backtest_eval_match_detail(candidate_diagnostics, candidate_manifest),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "matched_backtest_holdout_period",
            _backtest_holdout_config_matches(baseline_diagnostics, candidate_diagnostics),
            _backtest_holdout_config_detail(baseline_diagnostics, candidate_diagnostics),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "net_pnl_beats_champion",
            (
                baseline_net_pnl is not None
                and candidate_net_pnl is not None
                and candidate_net_pnl > baseline_net_pnl
            ),
            (
                f"candidate net_pnl={_format_float(candidate_net_pnl)}, "
                f"baseline net_pnl={_format_float(baseline_net_pnl)}"
            ),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "max_drawdown",
            candidate_drawdown is not None and candidate_drawdown < rules.max_drawdown_pct,
            (
                f"candidate drawdown={_format_float(candidate_drawdown)}, "
                f"required < {rules.max_drawdown_pct:.4f}"
            ),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "sharpe_or_brier_justification",
            (
                candidate_sharpe is not None
                and baseline_sharpe is not None
                and (candidate_sharpe >= baseline_sharpe or lower_sharpe_allowed)
            ),
            (
                f"candidate sharpe={_format_float(candidate_sharpe)}, "
                f"baseline sharpe={_format_float(baseline_sharpe)}, "
                f"brier_improvement={_format_float(brier_improvement)}"
            ),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "turnover_reasonable",
            turnover is not None and 0.0 < turnover <= rules.max_turnover_rate,
            (
                f"candidate turnover={_format_float(turnover)}, "
                f"required in (0, {rules.max_turnover_rate:.4f}]"
            ),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "realistic_nonzero_costs",
            _backtest_costs_nonzero(baseline_cost_settings)
            and _backtest_costs_nonzero(candidate_cost_settings),
            _backtest_cost_settings_detail(
                baseline_cost_settings,
                candidate_cost_settings,
            ),
            candidate_backtest_summary_path,
        ),
        _audit_check(
            "matched_backtest_cost_assumptions",
            _backtest_cost_settings_match(baseline_cost_settings, candidate_cost_settings),
            _backtest_cost_settings_detail(
                baseline_cost_settings,
                candidate_cost_settings,
            ),
            candidate_backtest_summary_path,
        ),
    ]
    return _audit_gate("Stage 2: Cost-Adjusted Backtest", checks)


def _shadow_gate(
    shadow_evaluation_path: Path | str | None,
    *,
    baseline_eval_dir: Path | str | None,
    candidate_eval_dir: Path | str | None,
    expected_candidate_model_version: str,
    rules: ChampionPromotionAuditRules,
) -> ChampionPromotionGate:
    report = _read_optional_json(None if shadow_evaluation_path is None else Path(shadow_evaluation_path))
    checks = [
        _audit_check(
            "shadow_evaluation_exists",
            isinstance(report, dict),
            "shadow evaluation JSON exists",
            shadow_evaluation_path,
        )
    ]
    if not isinstance(report, dict):
        return _audit_gate("Stage 3: Shadow Evaluation", checks)

    shadow_checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    required_names = (
        "prediction_distribution_stability",
        "edge_trigger_rate",
        "simulated_pnl",
        "prediction_latency",
        "schema_error_rate",
        "scoring_error_rate",
    )
    failed_required = [
        name
        for name in required_names
        if not (
            isinstance(shadow_checks.get(name), dict)
            and _is_true(shadow_checks[name].get("passed"))
        )
    ]
    overall_shadow_passed = _is_true(report.get("overall_passed")) or _is_true(report.get("passed"))
    shadow_window = _shadow_window_evidence(report)
    duration_seconds = _safe_float(shadow_window.get("duration_seconds"))
    sample_count = _metric(report, "sample_count")
    scored_count = _metric(report, "scored_count")
    challenger_distribution = report.get("challenger_probability_distribution")
    challenger_distribution = (
        challenger_distribution if isinstance(challenger_distribution, dict) else {}
    )
    challenger_distribution_count = _metric(challenger_distribution, "count")
    shadow_rows_present = (
        sample_count is not None
        and scored_count is not None
        and challenger_distribution_count is not None
        and sample_count > 0.0
        and scored_count > 0.0
        and challenger_distribution_count > 0.0
        and sample_count >= scored_count
        and scored_count >= challenger_distribution_count
    )
    trigger_rate = _metric(report, "challenger_edge_trigger_rate")
    schema_error_rate = _metric(report, "schema_error_rate")
    scoring_error_rate = _metric(report, "scoring_error_rate")
    latency_p95 = _shadow_challenger_p95(report)
    distribution_evidence = _shadow_distribution_stability_evidence(
        report,
        candidate_eval_dir=candidate_eval_dir,
        rules=rules,
    )
    baseline_manifest = _read_eval_json(baseline_eval_dir, "manifest.json")
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    simulated_pnl = report.get("simulated_pnl")
    simulated_pnl = simulated_pnl if isinstance(simulated_pnl, dict) else {}
    pnl_delta = _metric(simulated_pnl, "net_pnl_delta")
    champion_pnl = _metric(simulated_pnl, "champion_net_pnl")
    challenger_pnl = _metric(simulated_pnl, "challenger_net_pnl")
    champion_trade_count = _metric(simulated_pnl, "champion_trade_count")
    challenger_trade_count = _metric(simulated_pnl, "challenger_trade_count")
    checks.extend(
        [
            _audit_check(
                "shadow_scored_rows_present",
                shadow_rows_present,
                (
                    f"sample_count={_format_float(sample_count)}, "
                    f"scored_count={_format_float(scored_count)}, "
                    "challenger_distribution_count="
                    f"{_format_float(challenger_distribution_count)}"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "full_shadow_session_evidence",
                (
                    bool(shadow_window["bounds_present"])
                    and bool(shadow_window["duration_consistent"])
                    and duration_seconds is not None
                    and duration_seconds >= rules.min_shadow_session_seconds
                ),
                (
                    f"window_start_ts={_format_float(shadow_window['start_ms'])}, "
                    f"window_end_ts={_format_float(shadow_window['end_ms'])}, "
                    f"duration_seconds={_format_float(duration_seconds)}, "
                    "reported_session_duration_seconds="
                    f"{_format_float(shadow_window['reported_duration_seconds'])}, "
                    f"duration_consistent={shadow_window['duration_consistent']}, "
                    f"required >= {rules.min_shadow_session_seconds}"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "overall_shadow_passed",
                overall_shadow_passed,
                (
                    f"overall_passed={report.get('overall_passed')}, "
                    f"passed={report.get('passed')}"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "shadow_models_match_eval",
                _shadow_models_match_eval(
                    report,
                    baseline_manifest=baseline_manifest,
                    candidate_manifest=candidate_manifest,
                    expected_candidate_model_version=expected_candidate_model_version,
                ),
                _shadow_model_match_detail(
                    report,
                    baseline_manifest=baseline_manifest,
                    candidate_manifest=candidate_manifest,
                    expected_candidate_model_version=expected_candidate_model_version,
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "shadow_offline_reference_matches_eval",
                _shadow_offline_reference_matches_eval(
                    report,
                    candidate_eval_dir=candidate_eval_dir,
                    candidate_manifest=candidate_manifest,
                ),
                _shadow_offline_reference_detail(
                    report,
                    candidate_eval_dir=candidate_eval_dir,
                    candidate_manifest=candidate_manifest,
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "required_shadow_checks_passed",
                not failed_required,
                "failed required checks=" + (", ".join(failed_required) if failed_required else "none"),
                shadow_evaluation_path,
            ),
            _audit_check(
                "prediction_distribution_drift_within_bounds",
                bool(distribution_evidence["passed"]),
                (
                    f"mean_abs_diff={_format_float(distribution_evidence['mean_abs_diff'])}, "
                    f"required < {rules.max_shadow_probability_mean_abs_diff:.4f}; "
                    f"std_relative_diff={_format_float(distribution_evidence['std_relative_diff'])}, "
                    f"required < {rules.max_shadow_probability_std_relative_diff:.4f}; "
                    f"reference_path={distribution_evidence['reference_path']}"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "edge_trigger_rate_reasonable",
                trigger_rate is not None and 0.0 < trigger_rate <= 0.50,
                f"challenger_edge_trigger_rate={_format_float(trigger_rate)}",
                shadow_evaluation_path,
            ),
            _audit_check(
                "simulated_pnl_beats_champion",
                (
                    pnl_delta is not None
                    and pnl_delta > 0.0
                    and champion_pnl is not None
                    and challenger_pnl is not None
                    and champion_trade_count is not None
                    and challenger_trade_count is not None
                ),
                (
                    f"champion_net_pnl={_format_float(champion_pnl)}, "
                    f"challenger_net_pnl={_format_float(challenger_pnl)}, "
                    f"net_pnl_delta={_format_float(pnl_delta)}, "
                    f"champion_trade_count={_format_float(champion_trade_count)}, "
                    f"challenger_trade_count={_format_float(challenger_trade_count)}"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "prediction_latency",
                latency_p95 is not None and latency_p95 < rules.max_prediction_latency_ms,
                (
                    f"challenger p95 latency={_format_float(latency_p95)}ms, "
                    f"required < {rules.max_prediction_latency_ms:.4f}ms"
                ),
                shadow_evaluation_path,
            ),
            _audit_check(
                "schema_error_rate_zero",
                schema_error_rate == 0.0,
                f"schema_error_rate={_format_float(schema_error_rate)}",
                shadow_evaluation_path,
            ),
            _audit_check(
                "scoring_error_rate_zero",
                scoring_error_rate == 0.0,
                f"scoring_error_rate={_format_float(scoring_error_rate)}",
                shadow_evaluation_path,
            ),
        ]
    )
    return _audit_gate("Stage 3: Shadow Evaluation", checks)


def _bootstrap_gate(
    *,
    bootstrap_decision_path: Path | str | None,
    baseline_eval_dir: Path | str | None,
    candidate_eval_dir: Path | str | None,
    baseline_backtest_summary_path: Path | str | None,
    candidate_backtest_summary_path: Path | str | None,
    shadow_evaluation_path: Path | str | None,
    serving_readiness_path: Path | str | None,
    rollback_runbook_path: Path | str | None,
    expected_candidate_model_version: str,
    rules: ChampionPromotionAuditRules,
) -> ChampionPromotionGate:
    bootstrap = _read_optional_json(
        None if bootstrap_decision_path is None else Path(bootstrap_decision_path)
    )
    serving = _read_optional_json(
        None if serving_readiness_path is None else Path(serving_readiness_path)
    )
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    checklist = (
        bootstrap.get("bootstrap_promotion_checklist")
        if isinstance(bootstrap, dict) and isinstance(bootstrap.get("bootstrap_promotion_checklist"), dict)
        else {}
    )
    hard_gate_results = (
        bootstrap.get("hard_gate_results")
        if isinstance(bootstrap, dict) and isinstance(bootstrap.get("hard_gate_results"), list)
        else []
    )
    missing_evidence_present = isinstance(bootstrap, dict) and isinstance(
        bootstrap.get("missing_or_weak_evidence"), list
    )
    missing = (
        bootstrap.get("missing_or_weak_evidence")
        if missing_evidence_present
        else []
    )
    missing_checklist_items = sorted(set(REQUIRED_BOOTSTRAP_CHECKLIST_KEYS) - set(checklist))
    failed_checklist_items = sorted(
        key for key in REQUIRED_BOOTSTRAP_CHECKLIST_KEYS if not _is_true(checklist.get(key))
    )
    passed_hard_gate_versions = [
        str(row.get("model_version"))
        for row in hard_gate_results
        if isinstance(row, dict) and _is_true(row.get("passed")) and row.get("model_version")
    ]
    expected_candidate_hard_gate_passed = expected_candidate_model_version in passed_hard_gate_versions
    bootstrap_candidate_version = _bootstrap_candidate_model_version(bootstrap)
    serving_error_rate = _metric(serving if isinstance(serving, dict) else {}, "error_rate")
    serving_p95 = _first_metric(
        serving if isinstance(serving, dict) else {},
        ("p95_latency_ms", "latency_p95_ms"),
    )
    checks = [
        _audit_check(
            "bootstrap_decision_exists",
            isinstance(bootstrap, dict),
            "bootstrap decision JSON exists",
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_promotes",
            isinstance(bootstrap, dict)
            and bootstrap.get("recommended_action") == "PROMOTE_CHAMPION",
            (
                "recommended_action="
                + (str(bootstrap.get("recommended_action")) if isinstance(bootstrap, dict) else "missing")
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_candidate_matches_expected",
            bootstrap_candidate_version == expected_candidate_model_version,
            (
                f"bootstrap candidate={bootstrap_candidate_version}, "
                f"expected={expected_candidate_model_version}"
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_checklist_passed",
            bool(checklist) and not missing_checklist_items and not failed_checklist_items,
            _bootstrap_checklist_detail(
                checklist,
                missing_items=missing_checklist_items,
                failed_items=failed_checklist_items,
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_hard_gates_passed",
            bool(hard_gate_results)
            and all(isinstance(row, dict) and _is_true(row.get("passed")) for row in hard_gate_results)
            and expected_candidate_hard_gate_passed,
            (
                f"hard_gate_count={len(hard_gate_results)}, "
                f"passed_versions={', '.join(passed_hard_gate_versions) if passed_hard_gate_versions else 'none'}, "
                f"expected_candidate_gate_passed={expected_candidate_hard_gate_passed}"
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "no_missing_or_weak_evidence",
            missing_evidence_present and not missing,
            _missing_or_weak_evidence_detail(
                present=missing_evidence_present,
                missing=missing,
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_candidate_backtest",
            _bootstrap_artifact_matches(
                bootstrap,
                "candidate_backtest_summary_path",
                candidate_backtest_summary_path,
            ),
            _bootstrap_artifact_detail(
                bootstrap,
                "candidate_backtest_summary_path",
                candidate_backtest_summary_path,
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_candidate_eval",
            _bootstrap_artifact_matches(bootstrap, "candidate_eval_dir", candidate_eval_dir),
            _bootstrap_artifact_detail(bootstrap, "candidate_eval_dir", candidate_eval_dir),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_shadow",
            _bootstrap_artifact_matches(bootstrap, "shadow_evaluation_path", shadow_evaluation_path),
            _bootstrap_artifact_detail(bootstrap, "shadow_evaluation_path", shadow_evaluation_path),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_serving_readiness",
            _bootstrap_artifact_matches(bootstrap, "serving_readiness_path", serving_readiness_path),
            _bootstrap_artifact_detail(bootstrap, "serving_readiness_path", serving_readiness_path),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_rollback_runbook",
            _bootstrap_artifact_matches(bootstrap, "rollback_runbook_path", rollback_runbook_path),
            _bootstrap_artifact_detail(bootstrap, "rollback_runbook_path", rollback_runbook_path),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_baseline_backtest",
            _bootstrap_artifact_matches(
                bootstrap,
                "baseline_backtest_summary_path",
                baseline_backtest_summary_path,
            ),
            _bootstrap_artifact_detail(
                bootstrap,
                "baseline_backtest_summary_path",
                baseline_backtest_summary_path,
            ),
            bootstrap_decision_path,
        ),
        _audit_check(
            "bootstrap_uses_current_baseline_eval",
            _bootstrap_artifact_matches(bootstrap, "baseline_eval_dir", baseline_eval_dir),
            _bootstrap_artifact_detail(bootstrap, "baseline_eval_dir", baseline_eval_dir),
            bootstrap_decision_path,
        ),
        _audit_check(
            "serving_readiness",
            isinstance(serving, dict)
            and (_is_true(serving.get("ready")) or _is_true(serving.get("serving_ready")))
            and serving_error_rate == 0.0
            and serving_p95 is not None
            and serving_p95 < rules.max_prediction_latency_ms
            and str(serving.get("model_version")) == expected_candidate_model_version
            and _serving_readiness_matches_candidate(
                serving,
                candidate_manifest=candidate_manifest,
                rollback_runbook_path=rollback_runbook_path,
            ),
            _serving_readiness_detail(
                serving if isinstance(serving, dict) else {},
                candidate_manifest=candidate_manifest,
                rollback_runbook_path=rollback_runbook_path,
                serving_error_rate=serving_error_rate,
                serving_p95=serving_p95,
            ),
            serving_readiness_path,
        ),
        _audit_check(
            "rollback_runbook_available",
            rollback_runbook_path is not None and Path(rollback_runbook_path).exists(),
            "rollback/fallback runbook exists",
            rollback_runbook_path,
        ),
    ]
    return _audit_gate("Stage 4: Bootstrap Decision", checks)


def _cutover_gate(
    *,
    cutover_report_path: Path | str | None,
    candidate_eval_dir: Path | str | None,
    bootstrap_decision_path: Path | str | None,
    shadow_evaluation_path: Path | str | None,
    serving_readiness_path: Path | str | None,
    expected_candidate_model_version: str,
    expected_fallback_model_version: str,
    rules: ChampionPromotionAuditRules,
) -> ChampionPromotionGate:
    cutover = _read_optional_json(None if cutover_report_path is None else Path(cutover_report_path))
    checks = [
        _audit_check(
            "cutover_report_exists",
            isinstance(cutover, dict),
            "cutover JSON evidence exists",
            cutover_report_path,
        )
    ]
    if not isinstance(cutover, dict):
        return _audit_gate("Stage 5: Champion Cutover", checks)

    champion = cutover.get("current_champion")
    champion = champion if isinstance(champion, dict) else {}
    candidate_manifest = _read_eval_json(candidate_eval_dir, "manifest.json")
    online = cutover.get("current_online_model")
    online = online if isinstance(online, dict) else {}
    fallback = cutover.get("fallback_registry_model")
    fallback = fallback if isinstance(fallback, dict) else {}
    smoke = cutover.get("smoke")
    smoke = smoke if isinstance(smoke, dict) else {}
    evidence = cutover.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    drift_baseline_path = cutover.get("drift_baseline_path")
    drift_baseline = (
        _read_optional_json(Path(drift_baseline_path))
        if isinstance(drift_baseline_path, str)
        else None
    )
    shadow = _read_optional_json(
        None if shadow_evaluation_path is None else Path(shadow_evaluation_path)
    )
    smoke_latency = _first_metric(smoke, ("serving_latency_ms", "p95_latency_ms", "latency_p95_ms"))
    smoke_error_rate = _metric(smoke, "error_rate")
    online_traffic_percent = _safe_float(online.get("traffic_percent"))
    checks.extend(
        [
            _audit_check(
                "registry_champion_switched",
                champion.get("model_version") == expected_candidate_model_version,
                (
                    f"champion={champion.get('model_version')}, "
                    f"expected={expected_candidate_model_version}"
                ),
                cutover_report_path,
            ),
            _audit_check(
                "registry_champion_artifacts_match_candidate",
                _registry_champion_artifacts_match_candidate(champion, candidate_manifest),
                _registry_champion_artifact_detail(champion, candidate_manifest),
                cutover_report_path,
            ),
            _audit_check(
                "registry_champion_promotion_metrics_recorded",
                _registry_champion_promotion_metrics_passed(champion, rules=rules),
                _registry_champion_promotion_metrics_detail(champion, rules=rules),
                cutover_report_path,
            ),
            _audit_check(
                "online_deployment_succeeded",
                online.get("model_version") == expected_candidate_model_version
                and online.get("deployment_status") == "succeeded"
                and online_traffic_percent is not None
                and online_traffic_percent >= 100.0,
                (
                    f"online={online.get('model_version')}, "
                    f"status={online.get('deployment_status')}, "
                    f"traffic={_format_float(online_traffic_percent)}"
                ),
                cutover_report_path,
            ),
            _audit_check(
                "fallback_recorded",
                online.get("rollback_to_version") == expected_fallback_model_version,
                (
                    f"rollback_to={online.get('rollback_to_version')}, "
                    f"expected={expected_fallback_model_version}"
                ),
                cutover_report_path,
            ),
            _audit_check(
                "fallback_registry_model_available",
                _fallback_registry_model_available(
                    fallback,
                    expected_fallback_model_version=expected_fallback_model_version,
                ),
                _fallback_registry_model_detail(fallback, expected_fallback_model_version),
                cutover_report_path,
            ),
            _audit_check(
                "inference_smoke_passed",
                _is_true(smoke.get("passed"))
                and smoke.get("model_version") == expected_candidate_model_version
                and smoke_error_rate == 0.0
                and smoke_latency is not None
                and smoke_latency < rules.max_prediction_latency_ms,
                (
                    f"smoke_passed={smoke.get('passed')}, "
                    f"model={smoke.get('model_version')}, "
                    f"error_rate={_format_float(smoke_error_rate)}, "
                    f"latency={_format_float(smoke_latency)}ms"
                ),
                cutover_report_path,
            ),
            _audit_check(
                "cutover_uses_current_smoke",
                _embedded_smoke_matches_evidence(
                    smoke,
                    evidence.get("smoke"),
                    expected_candidate_model_version=expected_candidate_model_version,
                    rules=rules,
                ),
                _embedded_smoke_evidence_detail(smoke, evidence.get("smoke")),
                cutover_report_path,
            ),
            _audit_check(
                "smoke_artifacts_match_candidate",
                _smoke_artifacts_match_candidate(smoke, candidate_manifest),
                _smoke_artifact_candidate_detail(smoke, candidate_manifest),
                cutover_report_path,
            ),
            _audit_check(
                "drift_baseline_registered",
                isinstance(drift_baseline, dict),
                f"drift_baseline_path={drift_baseline_path}",
                cutover_report_path,
            ),
            _audit_check(
                "drift_baseline_matches_shadow_reference",
                _drift_baseline_matches_shadow_reference(
                    drift_baseline,
                    shadow,
                    expected_candidate_model_version=expected_candidate_model_version,
                ),
                _drift_baseline_reference_detail(drift_baseline, shadow),
                cutover_report_path,
            ),
            _audit_check(
                "cutover_uses_current_bootstrap",
                _path_matches(evidence.get("bootstrap"), bootstrap_decision_path),
                f"cutover bootstrap={evidence.get('bootstrap')}",
                cutover_report_path,
            ),
            _audit_check(
                "cutover_uses_current_shadow",
                _path_matches(evidence.get("shadow"), shadow_evaluation_path),
                f"cutover shadow={evidence.get('shadow')}",
                cutover_report_path,
            ),
            _audit_check(
                "cutover_uses_current_serving_readiness",
                _path_matches(evidence.get("serving_readiness"), serving_readiness_path),
                f"cutover serving_readiness={evidence.get('serving_readiness')}",
                cutover_report_path,
            ),
            _audit_check(
                "github_issue_closures_recorded",
                _github_issue_closures_passed(
                    cutover,
                    expected_candidate_model_version=expected_candidate_model_version,
                ),
                _github_issue_closures_detail(
                    cutover,
                    expected_candidate_model_version=expected_candidate_model_version,
                ),
                cutover_report_path,
            ),
        ]
    )
    return _audit_gate("Stage 5: Champion Cutover", checks)


def _build_checks(
    *,
    baseline_test: dict[str, float | int | None],
    candidate_test: dict[str, float | int | None],
    calibration: dict[str, Any] | None,
    backtest: dict[str, Any] | None,
    dataset_version: str | None,
    baseline_model_version: str,
    candidate_model_version: str,
    rules: PromotionRules,
) -> list[PromotionCheck]:
    baseline_auc = _metric(baseline_test, "roc_auc")
    candidate_auc = _metric(candidate_test, "roc_auc")
    baseline_brier = _metric(baseline_test, "brier_score")
    candidate_brier = _metric(candidate_test, "brier_score")
    backtest_net_pnl = None if backtest is None else _metric(backtest, "net_pnl")
    calibration_improved = bool(calibration and calibration.get("improved"))
    checks = [
        PromotionCheck(
            name="traceable_versions",
            passed=bool(dataset_version and baseline_model_version and candidate_model_version),
            detail=(
                f"dataset={dataset_version}, baseline={baseline_model_version}, "
                f"candidate={candidate_model_version}"
            ),
        ),
        PromotionCheck(
            name="test_roc_auc_vs_baseline",
            passed=(
                baseline_auc is not None
                and candidate_auc is not None
                and candidate_auc >= baseline_auc + rules.min_roc_auc_delta
            ),
            detail=(
                f"candidate ROC AUC {candidate_auc} vs baseline {baseline_auc}; "
                f"required delta >= {rules.min_roc_auc_delta}"
            ),
        ),
        PromotionCheck(
            name="test_brier_vs_baseline",
            passed=(
                baseline_brier is not None
                and candidate_brier is not None
                and candidate_brier <= baseline_brier + rules.max_brier_delta
            ),
            detail=(
                f"candidate Brier {candidate_brier} vs baseline {baseline_brier}; "
                f"allowed delta <= {rules.max_brier_delta}"
            ),
        ),
        PromotionCheck(
            name="calibration_improved",
            passed=(calibration_improved or not rules.require_calibration_improved),
            detail=f"calibration improved={calibration_improved}",
        ),
        PromotionCheck(
            name="backtest_net_pnl",
            passed=(
                backtest_net_pnl is not None
                and backtest_net_pnl >= rules.min_backtest_net_pnl
            ),
            detail=(
                f"best/backtest net_pnl={backtest_net_pnl}; "
                f"required >= {rules.min_backtest_net_pnl}"
            ),
        ),
    ]
    return checks


def _checklist_markdown(report: PromotionReport) -> str:
    lines = [
        "# Model Promotion Checklist",
        "",
        f"Decision: **{report.decision}**",
        "",
        f"- Baseline: `{report.baseline_model_version}`",
        f"- Candidate: `{report.candidate_model_version}`",
        f"- Dataset: `{report.dataset_version}`",
        "",
        "## Checks",
    ]
    for check in report.checks:
        mark = "x" if check.passed else " "
        lines.append(f"- [{mark}] {check.name}: {check.detail}")
    lines.extend(
        [
            "",
            "## Metrics",
            f"- Baseline test ROC AUC: {report.baseline_test_metrics.get('roc_auc')}",
            f"- Candidate test ROC AUC: {report.candidate_test_metrics.get('roc_auc')}",
            f"- Baseline test Brier: {report.baseline_test_metrics.get('brier_score')}",
            f"- Candidate test Brier: {report.candidate_test_metrics.get('brier_score')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _promotion_process_source_evidence(
    *,
    promotion_process_path: Path | str | None,
    repo_promotion_runbook_path: Path | str | None,
) -> dict[str, Any]:
    checked = promotion_process_path is not None
    source_path = Path(promotion_process_path) if promotion_process_path is not None else None
    repo_path = (
        Path(repo_promotion_runbook_path) if repo_promotion_runbook_path is not None else None
    )
    source_text = _read_optional_text(source_path)
    repo_text = _read_optional_text(repo_path)
    required_markers = {
        "stage_1_offline_eval": "Stage 1: Offline Evaluation",
        "stage_2_backtest": "Stage 2: Cost-Adjusted Backtest",
        "stage_3_shadow": "Stage 3: Shadow Evaluation",
        "stage_4_bootstrap": "Stage 4: Bootstrap Decision",
        "stage_5_cutover": "Stage 5: Champion Cutover",
        "time_split_5_1_1": "Train: past 5 days",
        "rerun_report": "rerun_report.md",
        "promote_champion": "PROMOTE_CHAMPION",
        "full_trading_session": "one full trading session",
    }
    marker_presence = {
        key: bool(source_text and marker in source_text)
        for key, marker in required_markers.items()
    }
    repo_declares_source = bool(
        repo_text
        and source_path is not None
        and str(source_path) in repo_text
        and "Local repo copy" in repo_text
    )
    source_exists = source_path is not None and source_path.exists() and source_path.is_file()
    passed = bool(checked and source_exists and all(marker_presence.values()))
    return {
        "checked": checked,
        "passed": passed if checked else True,
        "source_path": str(source_path) if source_path is not None else None,
        "source_exists": source_exists,
        "source_sha256": _file_sha256(source_path),
        "required_markers": marker_presence,
        "missing_required_markers": [
            key for key, present in marker_presence.items() if not present
        ],
        "repo_mirror_path": str(repo_path) if repo_path is not None else None,
        "repo_mirror_exists": repo_path is not None and repo_path.exists() and repo_path.is_file(),
        "repo_mirror_sha256": _file_sha256(repo_path),
        "repo_mirror_declares_source": repo_declares_source,
    }


def _promotion_process_source_detail(evidence: dict[str, Any]) -> str:
    return (
        f"source={evidence.get('source_path')}, "
        f"source_exists={_is_true(evidence.get('source_exists'))}, "
        f"source_sha256={evidence.get('source_sha256')}, "
        f"missing_required_markers={evidence.get('missing_required_markers') or []}, "
        f"repo_mirror={evidence.get('repo_mirror_path')}, "
        f"repo_mirror_sha256={evidence.get('repo_mirror_sha256')}, "
        f"repo_mirror_declares_source={_is_true(evidence.get('repo_mirror_declares_source'))}"
    )


def _read_optional_text(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _split_metrics(metrics: dict[str, Any], split: str) -> dict[str, float | int | None]:
    row = metrics.get(split)
    if not isinstance(row, dict):
        raise ValueError(f"metrics missing {split!r} split")
    return row


def _dataset_version(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> str | None:
    baseline_version = baseline_manifest.get("dataset_version")
    candidate_version = candidate_manifest.get("dataset_version")
    if baseline_version and candidate_version and baseline_version != candidate_version:
        return None
    return None if candidate_version is None else str(candidate_version)


def _summarize_backtest(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]
        if not rows:
            return None
        return max(rows, key=_backtest_net_pnl_sort_key)
    if isinstance(raw, dict):
        return raw
    return None


def _metric(row: dict[str, Any], name: str) -> float | None:
    return _safe_float(row.get(name))


def _current_filesystem_headroom_evidence(
    status: dict[str, Any], disk_headroom: dict[str, Any]
) -> dict[str, Any] | None:
    live_root = status.get("live_root")
    if not live_root:
        return None
    required_free_bytes = _safe_int(disk_headroom.get("required_free_bytes"))
    if required_free_bytes is None:
        return None
    root = Path(str(live_root))
    if not root.exists():
        return None
    try:
        stat = os.statvfs(root)
    except OSError as exc:
        return {
            "available": False,
            "path": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "headroom_ok": False,
        }
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    margin_bytes = free_bytes - required_free_bytes
    low_margin_threshold_bytes = _safe_int(disk_headroom.get("low_margin_threshold_bytes"))
    if low_margin_threshold_bytes is None:
        low_margin_threshold_bytes = max(
            1 * 1024 * 1024 * 1024,
            int(required_free_bytes * 0.10),
        )
    headroom_ok = free_bytes >= required_free_bytes
    return {
        "available": True,
        "path": str(root),
        "free_bytes": free_bytes,
        "status_free_bytes": disk_headroom.get("free_bytes"),
        "required_free_bytes": required_free_bytes,
        "headroom_margin_bytes": margin_bytes,
        "low_margin_threshold_bytes": low_margin_threshold_bytes,
        "headroom_low_margin": headroom_ok and margin_bytes < low_margin_threshold_bytes,
        "headroom_ok": headroom_ok,
    }


def _current_filesystem_headroom_detail(evidence: dict[str, Any] | None) -> str:
    if evidence is None:
        return "not_checked"
    if evidence.get("available") is False:
        return (
            f"available=False, path={evidence.get('path')}, "
            f"error={evidence.get('error')}"
        )
    return (
        f"available={evidence.get('available')}, "
        f"path={evidence.get('path')}, "
        f"headroom_ok={_is_true(evidence.get('headroom_ok'))}, "
        f"free_bytes={evidence.get('free_bytes')}, "
        f"required_free_bytes={evidence.get('required_free_bytes')}, "
        f"headroom_margin_bytes={evidence.get('headroom_margin_bytes')}, "
        f"headroom_low_margin={evidence.get('headroom_low_margin')}"
    )


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _backtest_net_pnl_sort_key(row: dict[str, Any]) -> float:
    net_pnl = _safe_float(row.get("net_pnl"))
    return net_pnl if net_pnl is not None else float("-inf")


def _audit_check(
    name: str,
    passed: bool,
    detail: str,
    artifact_path: Path | str | None = None,
) -> ChampionPromotionGateCheck:
    return ChampionPromotionGateCheck(
        name=name,
        passed=passed is True,
        detail=detail,
        artifact_path=_path_str(artifact_path),
    )


def _audit_gate(
    name: str,
    checks: list[ChampionPromotionGateCheck],
) -> ChampionPromotionGate:
    return ChampionPromotionGate(
        name=name,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _path_str(path: Path | str | None) -> str | None:
    return None if path is None else str(path)


def _read_eval_json(eval_dir: Path | str | None, filename: str) -> dict[str, Any] | None:
    if eval_dir is None:
        return None
    payload = _read_optional_json(Path(eval_dir) / filename)
    return payload if isinstance(payload, dict) else None


def _test_metrics_or_empty(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    row = metrics.get("test")
    return row if isinstance(row, dict) else {}


def _same_eval_dataset(
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
) -> bool:
    if not isinstance(baseline_manifest, dict) or not isinstance(candidate_manifest, dict):
        return False
    baseline_dir = baseline_manifest.get("dataset_dir")
    candidate_dir = candidate_manifest.get("dataset_dir")
    baseline_version = baseline_manifest.get("dataset_version")
    candidate_version = candidate_manifest.get("dataset_version")
    return bool(
        baseline_dir
        and candidate_dir
        and str(baseline_dir) == str(candidate_dir)
        and baseline_version
        and candidate_version
        and str(baseline_version) == str(candidate_version)
    )


def _dataset_dir_from_manifest(manifest: dict[str, Any] | None) -> str | None:
    return _dict_str(manifest, "dataset_dir")


def _dataset_manifest_from_eval(eval_manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    dataset_dir = _dataset_dir_from_manifest(eval_manifest)
    if dataset_dir is None:
        return None
    payload = _read_optional_json(Path(dataset_dir) / "manifest.json")
    return payload if isinstance(payload, dict) else None


def _valid_dataset_time_split(eval_manifest: dict[str, Any] | None) -> bool:
    manifest = _dataset_manifest_from_eval(eval_manifest)
    if not isinstance(manifest, dict):
        return False
    split_config = manifest.get("split_config")
    splits = manifest.get("splits")
    if not isinstance(split_config, dict) or not isinstance(splits, dict):
        return False
    train_fraction = _metric(split_config, "train_fraction")
    val_fraction = _metric(split_config, "val_fraction")
    if train_fraction is None or val_fraction is None:
        return False
    expected_train = 5.0 / 7.0
    expected_val = 1.0 / 7.0
    if abs(train_fraction - expected_train) > 1e-6:
        return False
    if abs(val_fraction - expected_val) > 1e-6:
        return False
    split_rows = {name: splits.get(name) for name in ("train", "val", "test")}
    if not all(isinstance(row, dict) for row in split_rows.values()):
        return False
    if any(int(row.get("row_count") or 0) <= 0 for row in split_rows.values() if isinstance(row, dict)):
        return False
    return _chronological_splits(split_rows)


def _dataset_required_families_present(eval_manifest: dict[str, Any] | None) -> bool:
    manifest = _dataset_manifest_from_eval(eval_manifest)
    if not isinstance(manifest, dict):
        return False
    family_splits = manifest.get("family_splits")
    if not isinstance(family_splits, dict):
        return False
    return not _missing_dataset_families(family_splits)


def _missing_dataset_families(family_splits: dict[str, Any]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for family in REQUIRED_MARKET_FAMILIES:
        split_rows = family_splits.get(family)
        if not isinstance(split_rows, dict):
            missing[family] = list(SPLITS_FOR_DETAIL)
            continue
        missing_splits = [
            split
            for split in SPLITS_FOR_DETAIL
            if not isinstance(split_rows.get(split), dict)
            or int(split_rows[split].get("row_count") or 0) <= 0
        ]
        if missing_splits:
            missing[family] = missing_splits
    return missing


def _dataset_required_families_detail(eval_manifest: dict[str, Any] | None) -> str:
    manifest = _dataset_manifest_from_eval(eval_manifest)
    if not isinstance(manifest, dict):
        return "dataset manifest missing"
    family_splits = manifest.get("family_splits")
    if not isinstance(family_splits, dict):
        return "dataset manifest missing family_splits"
    missing = _missing_dataset_families(family_splits)
    if not missing:
        return f"required families present: {', '.join(REQUIRED_MARKET_FAMILIES)}"
    return "missing required family split rows: " + ", ".join(
        f"{family}({','.join(splits)})" for family, splits in sorted(missing.items())
    )


def _required_family_metrics_present(
    baseline_family_metrics: dict[str, Any] | None,
    candidate_family_metrics: dict[str, Any] | None,
) -> bool:
    return not _missing_eval_families(baseline_family_metrics, "baseline") and not _missing_eval_families(
        candidate_family_metrics,
        "candidate",
    )


def _missing_eval_families(metrics: dict[str, Any] | None, prefix: str) -> list[str]:
    if not isinstance(metrics, dict):
        return [f"{prefix}:{family}" for family in REQUIRED_MARKET_FAMILIES]
    test = metrics.get("test")
    if not isinstance(test, dict):
        return [f"{prefix}:{family}" for family in REQUIRED_MARKET_FAMILIES]
    missing = []
    for family in REQUIRED_MARKET_FAMILIES:
        row = test.get(family)
        if not isinstance(row, dict) or int(row.get("sample_count") or 0) <= 0:
            missing.append(f"{prefix}:{family}")
    return missing


def _required_family_metrics_detail(
    baseline_family_metrics: dict[str, Any] | None,
    candidate_family_metrics: dict[str, Any] | None,
) -> str:
    missing = [
        *_missing_eval_families(baseline_family_metrics, "baseline"),
        *_missing_eval_families(candidate_family_metrics, "candidate"),
    ]
    if not missing:
        return f"test family metrics present for {', '.join(REQUIRED_MARKET_FAMILIES)}"
    return "missing test family metrics: " + ", ".join(missing)


def _new_market_signal_present(
    candidate_family_metrics: dict[str, Any] | None,
    rules: ChampionPromotionAuditRules,
) -> bool:
    return any(
        _family_has_usable_signal(candidate_family_metrics, family, rules)
        for family in NEW_MARKET_SIGNAL_FAMILIES
    )


def _family_has_usable_signal(
    family_metrics: dict[str, Any] | None,
    family: str,
    rules: ChampionPromotionAuditRules,
) -> bool:
    if not isinstance(family_metrics, dict):
        return False
    test = family_metrics.get("test")
    if not isinstance(test, dict):
        return False
    row = test.get(family)
    if not isinstance(row, dict) or int(row.get("sample_count") or 0) <= 0:
        return False
    roc_auc = _metric(row, "roc_auc")
    brier_score = _metric(row, "brier_score")
    return (
        roc_auc is not None
        and roc_auc > rules.min_new_market_roc_auc
        and brier_score is not None
        and 0.0 <= brier_score <= 1.0
    )


def _new_market_signal_detail(
    candidate_family_metrics: dict[str, Any] | None,
    rules: ChampionPromotionAuditRules,
) -> str:
    if not isinstance(candidate_family_metrics, dict):
        return "candidate family metrics missing"
    test = candidate_family_metrics.get("test")
    if not isinstance(test, dict):
        return "candidate test family metrics missing"
    parts = []
    for family in NEW_MARKET_SIGNAL_FAMILIES:
        row = test.get(family)
        if not isinstance(row, dict):
            parts.append(f"{family}: missing")
            continue
        sample_count = int(row.get("sample_count") or 0)
        roc_auc = _metric(row, "roc_auc")
        brier_score = _metric(row, "brier_score")
        parts.append(
            f"{family}: samples={sample_count}, "
            f"roc_auc={_format_float(roc_auc)}, "
            f"brier={_format_float(brier_score)}"
        )
    return (
        "requires at least one newly added ETH market family with "
        f"roc_auc > {rules.min_new_market_roc_auc:.4f} and finite Brier; "
        + "; ".join(parts)
    )


def _dataset_time_split_detail(eval_manifest: dict[str, Any] | None) -> str:
    manifest = _dataset_manifest_from_eval(eval_manifest)
    if not isinstance(manifest, dict):
        return "dataset manifest missing"
    split_config = manifest.get("split_config") if isinstance(manifest.get("split_config"), dict) else {}
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    split_parts = []
    for name in ("train", "val", "test"):
        row = splits.get(name) if isinstance(splits.get(name), dict) else {}
        split_parts.append(
            f"{name}: rows={row.get('row_count')}, start={row.get('start_ts')}, end={row.get('end_ts')}"
        )
    return (
        f"split_config={split_config}; expected train=5/7 val=1/7; "
        + "; ".join(split_parts)
    )


def _chronological_splits(split_rows: dict[str, Any]) -> bool:
    rows = []
    for name in ("train", "val", "test"):
        row = split_rows.get(name)
        if not isinstance(row, dict):
            return False
        start_ts = row.get("start_ts")
        end_ts = row.get("end_ts")
        if start_ts is None or end_ts is None:
            return False
        if int(end_ts) < int(start_ts):
            return False
        rows.append((int(start_ts), int(end_ts)))
    return rows[0][1] <= rows[1][0] and rows[1][1] <= rows[2][0]


def _dataset_detail(
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
) -> str:
    return (
        f"baseline dataset={_dict_str(baseline_manifest, 'dataset_dir')} "
        f"({ _dict_str(baseline_manifest, 'dataset_version') }), "
        f"candidate dataset={_dict_str(candidate_manifest, 'dataset_dir')} "
        f"({ _dict_str(candidate_manifest, 'dataset_version') })"
    )


def _dict_str(row: dict[str, Any] | None, name: str) -> str | None:
    if not isinstance(row, dict):
        return None
    value = row.get(name)
    return None if value is None else str(value)


def _data_readiness_detail(
    readiness: dict[str, Any],
    feature: dict[str, Any],
    labels: dict[str, Any],
) -> str:
    parts = [
        "feature span "
        f"{_format_float(_metric(feature, 'min_family_span_days'))}d",
        "label span "
        f"{_format_float(_metric(labels, 'min_family_span_days'))}d",
        f"target {_format_float(_metric(readiness, 'target_days'))}d",
    ]
    if readiness.get("estimated_ready_at"):
        parts.append(f"estimated_ready_at {readiness['estimated_ready_at']}")
    if feature.get("limiting_family"):
        parts.append(f"feature limiting_family {feature['limiting_family']}")
    if labels.get("limiting_family"):
        parts.append(f"label limiting_family {labels['limiting_family']}")
    return ", ".join(parts)


def _raw_segment_quarantine_detail(
    raw_quarantine: dict[str, Any],
    quarantine_clean_window: dict[str, Any],
    *,
    quarantined_raw_segments: float | None,
    quarantine_clean_window_ready: bool,
) -> str:
    parts = [
        f"quarantined raw segments={_format_float(quarantined_raw_segments)}",
        f"clean_window_ready={quarantine_clean_window_ready}",
        f"estimated_ready_at={quarantine_clean_window.get('estimated_ready_at')}",
    ]
    latest = raw_quarantine.get("latest_quarantined_segment")
    if not isinstance(latest, dict):
        latest = quarantine_clean_window.get("latest_quarantined_segment")
    if isinstance(latest, dict):
        parts.append(f"latest_path={latest.get('path')}")
        parts.append(f"latest_segment_ts={latest.get('segment_ts')}")
        probe = latest.get("gzip_probe")
        if isinstance(probe, dict):
            parts.append(f"gzip_valid={probe.get('gzip_valid')}")
            parts.append(f"readable_prefix_lines={probe.get('readable_prefix_lines')}")
            parts.append(f"readable_prefix_bytes={probe.get('readable_prefix_bytes')}")
            if probe.get("error"):
                parts.append(f"gzip_error={probe.get('error')}")
    return ", ".join(parts)


def _valid_rerun_report(
    path: Path | str | None,
    *,
    expected_candidate_model_version: str,
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
) -> bool:
    if path is None:
        return False
    report_path = Path(path)
    if not report_path.exists() or not report_path.is_file():
        return False
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError:
        return False
    sidecar = _read_optional_json(report_path.with_suffix(".json"))
    if not isinstance(sidecar, dict):
        return False
    return (
        "Rerun Report" in text
        and "Decision: **PASS**" in text
        and expected_candidate_model_version in text
        and sidecar.get("passed") is True
        and sidecar.get("decision") == "PASS"
        and sidecar.get("candidate_model_version") == expected_candidate_model_version
        and _sidecar_output_path_matches(sidecar.get("output_path"), report_path)
        and sidecar.get("baseline_model_version") == _dict_str(baseline_manifest, "model_version")
        and sidecar.get("candidate_model_version") == _dict_str(candidate_manifest, "model_version")
        and sidecar.get("dataset_dir") == _dict_str(candidate_manifest, "dataset_dir")
        and sidecar.get("dataset_version") == _dict_str(candidate_manifest, "dataset_version")
    )


def _sidecar_output_path_matches(value: Any, report_path: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == report_path.resolve()
    except OSError:
        return str(value) == str(report_path)


def _best_backtest_row(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _summarize_backtest(_read_optional_json(Path(path)))


def _backtest_diagnostics(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_optional_json(Path(path).with_name("diagnostics.json"))
    return payload if isinstance(payload, dict) else None


def _backtest_summary_matches_diagnostics(path: Path | str | None) -> bool:
    if path is None:
        return False
    summary = _read_optional_json(Path(path))
    diagnostics = _backtest_diagnostics(path)
    if not isinstance(summary, list) or not isinstance(diagnostics, dict):
        return False
    diagnostic_summary = diagnostics.get("summary")
    return isinstance(diagnostic_summary, list) and diagnostic_summary == summary


def _backtest_summary_diagnostics_detail(path: Path | str | None) -> str:
    summary = _read_optional_json(Path(path)) if path is not None else None
    diagnostics = _backtest_diagnostics(path)
    diagnostic_summary = diagnostics.get("summary") if isinstance(diagnostics, dict) else None
    return (
        f"summary_rows={len(summary) if isinstance(summary, list) else 'missing'}, "
        f"diagnostic_summary_rows={len(diagnostic_summary) if isinstance(diagnostic_summary, list) else 'missing'}"
    )


def _backtest_matches_eval(
    diagnostics: dict[str, Any] | None,
    eval_manifest: dict[str, Any] | None,
) -> bool:
    if not isinstance(diagnostics, dict) or not isinstance(eval_manifest, dict):
        return False
    metadata = diagnostics.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    warehouse_dir = metadata.get("warehouse_dir")
    return (
        diagnostics.get("model_version") == _dict_str(eval_manifest, "model_version")
        and metadata.get("backtest_kind") == "direct_model"
        and _path_value_matches(metadata.get("model_path"), _dict_str(eval_manifest, "model_path"))
        and _path_value_matches(metadata.get("dataset_dir"), _dict_str(eval_manifest, "dataset_dir"))
        and metadata.get("dataset_version") == _dict_str(eval_manifest, "dataset_version")
        and isinstance(warehouse_dir, str)
        and bool(warehouse_dir.strip())
    )


def _backtest_eval_match_detail(
    diagnostics: dict[str, Any] | None,
    eval_manifest: dict[str, Any] | None,
) -> str:
    metadata = diagnostics.get("metadata") if isinstance(diagnostics, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        f"backtest model={diagnostics.get('model_version') if isinstance(diagnostics, dict) else None}, "
        f"eval model={_dict_str(eval_manifest, 'model_version')}, "
        f"backtest_kind={metadata.get('backtest_kind')}, "
        f"backtest model_path={metadata.get('model_path')}, "
        f"eval model_path={_dict_str(eval_manifest, 'model_path')}, "
        f"backtest dataset={metadata.get('dataset_dir')}, "
        f"eval dataset={_dict_str(eval_manifest, 'dataset_dir')}, "
        f"backtest dataset_version={metadata.get('dataset_version')}, "
        f"eval dataset_version={_dict_str(eval_manifest, 'dataset_version')}, "
        f"warehouse_dir={metadata.get('warehouse_dir')}"
    )


def _backtest_holdout_config_matches(
    baseline_diagnostics: dict[str, Any] | None,
    candidate_diagnostics: dict[str, Any] | None,
) -> bool:
    if not isinstance(baseline_diagnostics, dict) or not isinstance(candidate_diagnostics, dict):
        return False
    baseline_metadata = _backtest_metadata(baseline_diagnostics)
    candidate_metadata = _backtest_metadata(candidate_diagnostics)
    return (
        _path_values_match(
            baseline_metadata.get("dataset_dir"),
            candidate_metadata.get("dataset_dir"),
        )
        and _nonblank_values_match(
            baseline_metadata.get("dataset_version"),
            candidate_metadata.get("dataset_version"),
        )
        and _path_values_match(
            baseline_metadata.get("warehouse_dir"),
            candidate_metadata.get("warehouse_dir"),
        )
        and _backtest_required_outcome_side_matches(
            baseline_diagnostics,
            candidate_diagnostics,
        )
        and _tuple_values_match(
            _backtest_threshold_grid(baseline_diagnostics),
            _backtest_threshold_grid(candidate_diagnostics),
        )
        and _tuple_values_match(
            _backtest_hold_ms_grid(baseline_diagnostics),
            _backtest_hold_ms_grid(candidate_diagnostics),
        )
    )


def _backtest_holdout_config_detail(
    baseline_diagnostics: dict[str, Any] | None,
    candidate_diagnostics: dict[str, Any] | None,
) -> str:
    baseline_metadata = _backtest_metadata(baseline_diagnostics)
    candidate_metadata = _backtest_metadata(candidate_diagnostics)
    baseline_side = _backtest_required_outcome_side(baseline_diagnostics)
    candidate_side = _backtest_required_outcome_side(candidate_diagnostics)
    return (
        f"baseline dataset={baseline_metadata.get('dataset_dir')}, "
        f"candidate dataset={candidate_metadata.get('dataset_dir')}, "
        f"baseline dataset_version={baseline_metadata.get('dataset_version')}, "
        f"candidate dataset_version={candidate_metadata.get('dataset_version')}, "
        f"baseline warehouse_dir={baseline_metadata.get('warehouse_dir')}, "
        f"candidate warehouse_dir={candidate_metadata.get('warehouse_dir')}, "
        f"baseline required_outcome_side={_format_optional_side(baseline_side)}, "
        f"candidate required_outcome_side={_format_optional_side(candidate_side)}, "
        f"baseline thresholds={_format_tuple(_backtest_threshold_grid(baseline_diagnostics))}, "
        f"candidate thresholds={_format_tuple(_backtest_threshold_grid(candidate_diagnostics))}, "
        f"baseline hold_ms={_format_tuple(_backtest_hold_ms_grid(baseline_diagnostics))}, "
        f"candidate hold_ms={_format_tuple(_backtest_hold_ms_grid(candidate_diagnostics))}"
    )


def _backtest_metadata(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    metadata = diagnostics.get("metadata") if isinstance(diagnostics, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _path_values_match(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return _path_value_matches(left, right)


def _nonblank_values_match(left: Any, right: Any) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and bool(left.strip())
        and left == right
    )


def _backtest_required_outcome_side_matches(
    baseline_diagnostics: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
) -> bool:
    baseline_side = _backtest_required_outcome_side(baseline_diagnostics)
    candidate_side = _backtest_required_outcome_side(candidate_diagnostics)
    return baseline_side[0] and candidate_side[0] and baseline_side[1] == candidate_side[1]


def _backtest_required_outcome_side(diagnostics: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(diagnostics, dict) or "required_outcome_side" not in diagnostics:
        return (False, None)
    value = diagnostics.get("required_outcome_side")
    if value is None:
        return (True, None)
    if isinstance(value, str) and value.strip():
        return (True, value.strip().upper())
    return (False, None)


def _backtest_threshold_grid(diagnostics: dict[str, Any] | None) -> tuple[float, ...] | None:
    return _backtest_summary_float_grid(diagnostics, ("threshold", "edge_threshold"))


def _backtest_hold_ms_grid(diagnostics: dict[str, Any] | None) -> tuple[float, ...] | None:
    return _backtest_summary_float_grid(diagnostics, ("hold_ms",))


def _backtest_summary_float_grid(
    diagnostics: dict[str, Any] | None,
    names: tuple[str, ...],
) -> tuple[float, ...] | None:
    summary = diagnostics.get("summary") if isinstance(diagnostics, dict) else None
    if not isinstance(summary, list) or not summary:
        return None
    values: list[float] = []
    for row in summary:
        if not isinstance(row, dict):
            return None
        value = _first_metric(row, names)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _tuple_values_match(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> bool:
    return (
        left is not None
        and right is not None
        and len(left) == len(right)
        and all(
            _float_values_match(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def _format_optional_side(value: tuple[bool, str | None]) -> str:
    if not value[0]:
        return "missing"
    return "all" if value[1] is None else value[1]


def _format_tuple(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "missing"
    return "[" + ", ".join(_format_float(value) for value in values) + "]"


def _backtest_cost_settings(row: dict[str, Any] | None) -> dict[str, float | None]:
    settings = row.get("settings") if isinstance(row, dict) else None
    settings = settings if isinstance(settings, dict) else {}
    return {
        "fee_bps": _metric(settings, "fee_bps"),
        "slippage_bps": _metric(settings, "slippage_bps"),
        "latency_ms": _metric(settings, "latency_ms"),
    }


def _backtest_costs_nonzero(settings: dict[str, float | None]) -> bool:
    fee_bps = settings.get("fee_bps")
    slippage_bps = settings.get("slippage_bps")
    latency_ms = settings.get("latency_ms")
    return (
        fee_bps is not None
        and fee_bps > 0.0
        and slippage_bps is not None
        and slippage_bps > 0.0
        and latency_ms is not None
        and latency_ms >= 0.0
    )


def _backtest_cost_settings_match(
    baseline_settings: dict[str, float | None],
    candidate_settings: dict[str, float | None],
) -> bool:
    return all(
        _float_values_match(baseline_settings.get(name), candidate_settings.get(name))
        for name in ("fee_bps", "slippage_bps", "latency_ms")
    )


def _backtest_cost_settings_detail(
    baseline_settings: dict[str, float | None],
    candidate_settings: dict[str, float | None],
) -> str:
    return (
        f"baseline fee_bps={_format_float(baseline_settings.get('fee_bps'))}, "
        f"candidate fee_bps={_format_float(candidate_settings.get('fee_bps'))}, "
        f"baseline slippage_bps={_format_float(baseline_settings.get('slippage_bps'))}, "
        f"candidate slippage_bps={_format_float(candidate_settings.get('slippage_bps'))}, "
        f"baseline latency_ms={_format_float(baseline_settings.get('latency_ms'))}, "
        f"candidate latency_ms={_format_float(candidate_settings.get('latency_ms'))}"
    )


def _float_values_match(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and abs(left - right) <= 1e-9


def _first_metric(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _metric(row, name)
        if value is not None:
            return value
    return None


def _turnover_rate(row: dict[str, Any]) -> float | None:
    value = _first_metric(row, ("turnover", "turnover_trades_per_signal"))
    if value is not None:
        return value
    per_1000 = _first_metric(row, ("turnover_trades_per_1000_signals", "trades_per_1000_signals"))
    if per_1000 is not None:
        return per_1000 / 1_000.0
    return None


def _eval_brier_improvement(
    baseline_eval_dir: Path | str | None,
    candidate_eval_dir: Path | str | None,
) -> float | None:
    baseline_metrics = _test_metrics_or_empty(_read_eval_json(baseline_eval_dir, "metrics.json"))
    candidate_metrics = _test_metrics_or_empty(_read_eval_json(candidate_eval_dir, "metrics.json"))
    baseline_brier = _metric(baseline_metrics, "brier_score")
    candidate_brier = _metric(candidate_metrics, "brier_score")
    if baseline_brier is None or candidate_brier is None:
        return None
    return baseline_brier - candidate_brier


def _audit_lower_sharpe_allowed(
    *,
    candidate_sharpe: float | None,
    baseline_sharpe: float | None,
    net_pnl_delta: float | None,
    brier_improvement: float | None,
    rules: ChampionPromotionAuditRules,
) -> bool:
    if candidate_sharpe is None or baseline_sharpe is None:
        return False
    if candidate_sharpe >= baseline_sharpe:
        return True
    return (
        brier_improvement is not None
        and brier_improvement >= rules.allow_lower_sharpe_if_brier_gap
        and candidate_sharpe > 0.0
        and net_pnl_delta is not None
        and net_pnl_delta > 0.0
    )


def _shadow_window_evidence(report: dict[str, Any]) -> dict[str, Any]:
    reported_duration = _first_metric(
        report,
        ("session_duration_seconds", "duration_seconds", "shadow_session_seconds"),
    )
    start_ms = _first_metric(
        report,
        ("window_start_ts", "started_at_ms", "start_ms", "session_started_at_ms"),
    )
    end_ms = _first_metric(
        report,
        ("window_end_ts", "ended_at_ms", "end_ms", "session_ended_at_ms"),
    )
    computed_duration = (
        (end_ms - start_ms) / 1_000.0
        if start_ms is not None and end_ms is not None and end_ms >= start_ms
        else None
    )
    duration_consistent = (
        computed_duration is not None
        and (
            reported_duration is None
            or abs(computed_duration - reported_duration) <= 1.0
        )
    )
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "bounds_present": computed_duration is not None,
        "computed_duration_seconds": computed_duration,
        "reported_duration_seconds": reported_duration,
        "duration_seconds": computed_duration if computed_duration is not None else reported_duration,
        "duration_consistent": duration_consistent,
    }


def _shadow_models_match_eval(
    report: dict[str, Any],
    *,
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
    expected_candidate_model_version: str,
) -> bool:
    champion_version = _dict_str(report, "champion_model_version")
    challenger_version = _dict_str(report, "challenger_model_version")
    return (
        challenger_version == expected_candidate_model_version
        and challenger_version == _dict_str(candidate_manifest, "model_version")
        and champion_version == _dict_str(baseline_manifest, "model_version")
    )


def _shadow_model_match_detail(
    report: dict[str, Any],
    *,
    baseline_manifest: dict[str, Any] | None,
    candidate_manifest: dict[str, Any] | None,
    expected_candidate_model_version: str,
) -> str:
    return (
        f"shadow champion={_dict_str(report, 'champion_model_version')}, "
        f"baseline eval={_dict_str(baseline_manifest, 'model_version')}, "
        f"shadow challenger={_dict_str(report, 'challenger_model_version')}, "
        f"candidate eval={_dict_str(candidate_manifest, 'model_version')}, "
        f"expected candidate={expected_candidate_model_version}"
    )


def _shadow_offline_reference_matches_eval(
    report: dict[str, Any],
    *,
    candidate_eval_dir: Path | str | None,
    candidate_manifest: dict[str, Any] | None,
) -> bool:
    expected_path = _candidate_offline_reference_path(candidate_eval_dir)
    reference = report.get("offline_reference")
    reference = reference if isinstance(reference, dict) else {}
    return (
        _path_value_matches(_dict_str(report, "offline_reference_path"), expected_path)
        and _dict_str(reference, "model_version") == _dict_str(candidate_manifest, "model_version")
        and _dict_str(reference, "dataset_dir") == _dict_str(candidate_manifest, "dataset_dir")
        and _dict_str(reference, "dataset_version") == _dict_str(candidate_manifest, "dataset_version")
        and _dict_str(reference, "split") == "val"
    )


def _shadow_offline_reference_detail(
    report: dict[str, Any],
    *,
    candidate_eval_dir: Path | str | None,
    candidate_manifest: dict[str, Any] | None,
) -> str:
    expected_path = _candidate_offline_reference_path(candidate_eval_dir)
    reference = report.get("offline_reference")
    reference = reference if isinstance(reference, dict) else {}
    return (
        f"shadow offline_reference_path={_dict_str(report, 'offline_reference_path')}, "
        f"expected={expected_path}, "
        f"reference model={_dict_str(reference, 'model_version')}, "
        f"candidate eval model={_dict_str(candidate_manifest, 'model_version')}, "
        f"reference dataset={_dict_str(reference, 'dataset_dir')} "
        f"({_dict_str(reference, 'dataset_version')}), "
        f"candidate dataset={_dict_str(candidate_manifest, 'dataset_dir')} "
        f"({_dict_str(candidate_manifest, 'dataset_version')}), "
        f"reference split={_dict_str(reference, 'split')}"
    )


def _shadow_distribution_stability_evidence(
    report: dict[str, Any],
    *,
    candidate_eval_dir: Path | str | None,
    rules: ChampionPromotionAuditRules,
) -> dict[str, Any]:
    reference_path = _candidate_offline_reference_path(candidate_eval_dir)
    reference_payload = _read_optional_json(None if reference_path is None else Path(reference_path))
    reference_distribution = (
        reference_payload.get("probability_distribution")
        if isinstance(reference_payload, dict)
        else None
    )
    current_distribution = report.get("challenger_probability_distribution")
    current_distribution = current_distribution if isinstance(current_distribution, dict) else {}
    reference_distribution = reference_distribution if isinstance(reference_distribution, dict) else {}
    current_mean = _safe_float(current_distribution.get("mean"))
    current_std = _safe_float(current_distribution.get("std"))
    reference_mean = _safe_float(reference_distribution.get("mean"))
    reference_std = _safe_float(reference_distribution.get("std"))
    mean_abs_diff = (
        None if current_mean is None or reference_mean is None else abs(current_mean - reference_mean)
    )
    std_relative_diff = _relative_diff_optional(current_std, reference_std)
    passed = (
        mean_abs_diff is not None
        and std_relative_diff is not None
        and mean_abs_diff < rules.max_shadow_probability_mean_abs_diff
        and std_relative_diff < rules.max_shadow_probability_std_relative_diff
    )
    return {
        "passed": passed,
        "reference_path": reference_path,
        "current_mean": current_mean,
        "reference_mean": reference_mean,
        "mean_abs_diff": mean_abs_diff,
        "current_std": current_std,
        "reference_std": reference_std,
        "std_relative_diff": std_relative_diff,
    }


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _relative_diff_optional(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    if reference == 0.0:
        return 0.0 if current == 0.0 else None
    return abs(current - reference) / abs(reference)


def _candidate_offline_reference_path(candidate_eval_dir: Path | str | None) -> str | None:
    if candidate_eval_dir is None:
        return None
    return str(Path(candidate_eval_dir) / "offline_reference.json")


def _drift_baseline_matches_shadow_reference(
    drift_baseline: Any,
    shadow: Any,
    *,
    expected_candidate_model_version: str,
) -> bool:
    if not isinstance(drift_baseline, dict) or not isinstance(shadow, dict):
        return False
    shadow_reference = shadow.get("offline_reference")
    shadow_reference = shadow_reference if isinstance(shadow_reference, dict) else {}
    return (
        _dict_str(drift_baseline, "model_version") == expected_candidate_model_version
        and _path_value_matches(
            _dict_str(drift_baseline, "source_offline_reference_path"),
            _dict_str(shadow, "offline_reference_path"),
        )
        and _dict_str(drift_baseline, "dataset_dir") == _dict_str(shadow_reference, "dataset_dir")
        and _dict_str(drift_baseline, "dataset_version")
        == _dict_str(shadow_reference, "dataset_version")
        and _dict_str(drift_baseline, "split") == "val"
        and isinstance(drift_baseline.get("probability_distribution"), dict)
        and isinstance(drift_baseline.get("thresholds"), dict)
    )


def _drift_baseline_reference_detail(drift_baseline: Any, shadow: Any) -> str:
    drift = drift_baseline if isinstance(drift_baseline, dict) else {}
    shadow_report = shadow if isinstance(shadow, dict) else {}
    shadow_reference = shadow_report.get("offline_reference")
    shadow_reference = shadow_reference if isinstance(shadow_reference, dict) else {}
    return (
        f"drift model={_dict_str(drift, 'model_version')}, "
        f"drift source={_dict_str(drift, 'source_offline_reference_path')}, "
        f"shadow source={_dict_str(shadow_report, 'offline_reference_path')}, "
        f"drift dataset={_dict_str(drift, 'dataset_dir')} "
        f"({_dict_str(drift, 'dataset_version')}), "
        f"shadow dataset={_dict_str(shadow_reference, 'dataset_dir')} "
        f"({_dict_str(shadow_reference, 'dataset_version')}), "
        f"drift split={_dict_str(drift, 'split')}, "
        f"thresholds_present={isinstance(drift.get('thresholds'), dict)}"
    )


def _registry_champion_artifacts_match_candidate(
    champion: dict[str, Any],
    candidate_manifest: dict[str, Any] | None,
) -> bool:
    if not isinstance(candidate_manifest, dict):
        return False
    model_matches = _path_value_matches(
        _dict_str(champion, "artifact_uri"),
        _dict_str(candidate_manifest, "model_path"),
    )
    candidate_calibration_path = _dict_str(candidate_manifest, "calibration_path")
    calibration_matches = bool(candidate_calibration_path) and _path_value_matches(
        _dict_str(champion, "calibration_artifact_uri"),
        candidate_calibration_path,
    )
    return model_matches and calibration_matches


def _registry_champion_artifact_detail(
    champion: dict[str, Any],
    candidate_manifest: dict[str, Any] | None,
) -> str:
    candidate = candidate_manifest if isinstance(candidate_manifest, dict) else {}
    return (
        f"champion artifact={_dict_str(champion, 'artifact_uri')}, "
        f"candidate model={_dict_str(candidate, 'model_path')}, "
        f"champion calibration={_dict_str(champion, 'calibration_artifact_uri')}, "
        f"candidate calibration={_dict_str(candidate, 'calibration_path')}"
    )


def _registry_champion_promotion_metrics_passed(
    champion: dict[str, Any],
    *,
    rules: ChampionPromotionAuditRules,
) -> bool:
    metrics = _registry_champion_promotion_metrics(champion)
    return (
        all(metrics.get(name) is not None for name, _ in REQUIRED_CUTOVER_REGISTRY_METRICS)
        and (metrics.get("max_dd") or 0.0) < rules.max_drawdown_pct
        and (metrics.get("shadow_p95_ms") or 0.0) < rules.max_prediction_latency_ms
        and metrics.get("schema_error_rate") == 0.0
    )


def _registry_champion_promotion_metrics_detail(
    champion: dict[str, Any],
    *,
    rules: ChampionPromotionAuditRules,
) -> str:
    metrics = _registry_champion_promotion_metrics(champion)
    missing = [name for name, _ in REQUIRED_CUTOVER_REGISTRY_METRICS if metrics.get(name) is None]
    invalid: list[str] = []
    if metrics.get("max_dd") is not None and metrics["max_dd"] >= rules.max_drawdown_pct:
        invalid.append(f"max_dd={_format_float(metrics['max_dd'])}")
    if (
        metrics.get("shadow_p95_ms") is not None
        and metrics["shadow_p95_ms"] >= rules.max_prediction_latency_ms
    ):
        invalid.append(f"shadow_p95_ms={_format_float(metrics['shadow_p95_ms'])}")
    if metrics.get("schema_error_rate") is not None and metrics["schema_error_rate"] != 0.0:
        invalid.append(f"schema_error_rate={_format_float(metrics['schema_error_rate'])}")
    values = ", ".join(
        f"{name}={_format_float(metrics.get(name))}"
        for name, _ in REQUIRED_CUTOVER_REGISTRY_METRICS
    )
    return (
        f"{values}; missing={', '.join(missing) if missing else 'none'}; "
        f"invalid={', '.join(invalid) if invalid else 'none'}"
    )


def _registry_champion_promotion_metrics(champion: dict[str, Any]) -> dict[str, float | None]:
    metrics_payload = _parse_json_object(champion.get("metrics_json"))
    backtest_payload = _summarize_backtest(_parse_json_payload(champion.get("backtest_json")))
    sources = (
        *_metric_sources(metrics_payload),
        *_metric_sources(backtest_payload if isinstance(backtest_payload, dict) else {}),
    )
    return {
        name: _first_metric_from_sources(sources, aliases)
        for name, aliases in REQUIRED_CUTOVER_REGISTRY_METRICS
    }


def _metric_sources(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, dict):
        return ()
    sources = [payload]
    for key in ("promotion_metrics", "metrics", "test", "summary", "best", "backtest"):
        value = payload.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return tuple(sources)


def _first_metric_from_sources(
    sources: tuple[dict[str, Any], ...],
    aliases: tuple[str, ...],
) -> float | None:
    for source in sources:
        for alias in aliases:
            value = _metric(source, alias)
            if value is not None:
                return value
    return None


def _parse_json_object(value: Any) -> dict[str, Any]:
    payload = _parse_json_payload(value)
    return payload if isinstance(payload, dict) else {}


def _parse_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _fallback_registry_model_available(
    fallback: dict[str, Any],
    *,
    expected_fallback_model_version: str,
) -> bool:
    return (
        _dict_str(fallback, "model_version") == expected_fallback_model_version
        and bool(_dict_str(fallback, "artifact_uri"))
        and _dict_str(fallback, "status") in {"champion", "retired", "challenger", "candidate"}
    )


def _fallback_registry_model_detail(
    fallback: dict[str, Any],
    expected_fallback_model_version: str,
) -> str:
    return (
        f"fallback={_dict_str(fallback, 'model_version')}, "
        f"expected={expected_fallback_model_version}, "
        f"artifact={_dict_str(fallback, 'artifact_uri')}, "
        f"status={_dict_str(fallback, 'status')}"
    )


def _embedded_smoke_matches_evidence(
    smoke: dict[str, Any],
    smoke_path: Any,
    *,
    expected_candidate_model_version: str,
    rules: ChampionPromotionAuditRules,
) -> bool:
    if not isinstance(smoke_path, str):
        return False
    smoke_artifact = _read_optional_json(Path(smoke_path))
    if not isinstance(smoke_artifact, dict) or smoke_artifact != smoke:
        return False
    smoke_latency = _first_metric(
        smoke_artifact,
        ("serving_latency_ms", "p95_latency_ms", "latency_p95_ms"),
    )
    smoke_error_rate = _metric(smoke_artifact, "error_rate")
    return (
        _is_true(smoke_artifact.get("passed"))
        and smoke_artifact.get("model_version") == expected_candidate_model_version
        and smoke_error_rate == 0.0
        and smoke_latency is not None
        and smoke_latency < rules.max_prediction_latency_ms
    )


def _embedded_smoke_evidence_detail(smoke: dict[str, Any], smoke_path: Any) -> str:
    smoke_artifact = _read_optional_json(Path(smoke_path)) if isinstance(smoke_path, str) else None
    return (
        f"cutover smoke_path={smoke_path}, "
        f"path_exists={isinstance(smoke_artifact, dict)}, "
        f"embedded_model={smoke.get('model_version')}, "
        f"artifact_model={smoke_artifact.get('model_version') if isinstance(smoke_artifact, dict) else None}, "
        f"embedded_error_rate={_format_float(_metric(smoke, 'error_rate'))}, "
        f"artifact_error_rate={_format_float(_metric(smoke_artifact, 'error_rate') if isinstance(smoke_artifact, dict) else None)}"
    )


def _smoke_artifacts_match_candidate(
    smoke: dict[str, Any],
    candidate_manifest: dict[str, Any] | None,
) -> bool:
    if not isinstance(candidate_manifest, dict):
        return False
    smoke_model_path = _dict_str(smoke, "model_path") or _dict_str(smoke, "artifact_uri")
    candidate_model_path = _dict_str(candidate_manifest, "model_path")
    candidate_calibration_path = _dict_str(candidate_manifest, "calibration_path")
    smoke_calibration_path = _dict_str(smoke, "calibration_path") or _dict_str(
        smoke,
        "calibration_artifact_uri",
    )
    model_matches = _path_value_matches(smoke_model_path, candidate_model_path)
    calibration_matches = (
        True
        if not candidate_calibration_path
        else _path_value_matches(smoke_calibration_path, candidate_calibration_path)
    )
    return model_matches and calibration_matches


def _smoke_artifact_candidate_detail(
    smoke: dict[str, Any],
    candidate_manifest: dict[str, Any] | None,
) -> str:
    candidate = candidate_manifest if isinstance(candidate_manifest, dict) else {}
    return (
        f"smoke model_path={_dict_str(smoke, 'model_path') or _dict_str(smoke, 'artifact_uri')}, "
        f"candidate model_path={_dict_str(candidate, 'model_path')}, "
        f"smoke calibration_path={_dict_str(smoke, 'calibration_path') or _dict_str(smoke, 'calibration_artifact_uri')}, "
        f"candidate calibration_path={_dict_str(candidate, 'calibration_path')}"
    )


def _shadow_challenger_p95(report: dict[str, Any]) -> float | None:
    latency = report.get("latency_ms")
    if not isinstance(latency, dict):
        return None
    challenger = _dict_str(report, "challenger_model_version")
    rows: list[dict[str, Any]] = []
    if challenger and isinstance(latency.get(challenger), dict):
        rows.append(latency[challenger])
    rows.extend(row for row in latency.values() if isinstance(row, dict))
    for row in rows:
        value = _first_metric(row, ("p95", "p95_latency_ms", "latency_p95_ms"))
        if value is not None:
            return value
    return None


def _bootstrap_candidate_model_version(report: Any) -> str | None:
    if not isinstance(report, dict):
        return None
    for key in (
        "candidate_model_version",
        "selected_model_version",
        "promoted_model_version",
        "model_version",
    ):
        value = report.get(key)
        if value:
            return str(value)
    recommended_action = report.get("recommended_action")
    if isinstance(recommended_action, str) and recommended_action.startswith("PROMOTE_FIRST_CHAMPION:"):
        return recommended_action.split(":", 1)[1] or None
    hard_gate_results = report.get("hard_gate_results")
    if isinstance(hard_gate_results, list):
        passed_versions = [
            str(row.get("model_version"))
            for row in hard_gate_results
            if isinstance(row, dict) and _is_true(row.get("passed")) and row.get("model_version")
        ]
        if len(passed_versions) == 1:
            return passed_versions[0]
    return None


def _bootstrap_checklist_detail(
    checklist: dict[str, Any],
    *,
    missing_items: list[str],
    failed_items: list[str],
) -> str:
    if not checklist:
        return "checklist missing"
    parts: list[str] = []
    if missing_items:
        parts.append("missing checklist items=" + ", ".join(missing_items))
    if failed_items:
        parts.append("failed checklist items=" + ", ".join(failed_items))
    return "; ".join(parts) if parts else "all required checklist items passed"


def _missing_or_weak_evidence_detail(*, present: bool, missing: list[Any]) -> str:
    if not present:
        return "missing_or_weak_evidence=missing"
    return "missing_or_weak_evidence=" + (", ".join(map(str, missing)) if missing else "none")


def _serving_readiness_matches_candidate(
    serving: Any,
    *,
    candidate_manifest: dict[str, Any] | None,
    rollback_runbook_path: Path | str | None,
) -> bool:
    if not isinstance(serving, dict) or not isinstance(candidate_manifest, dict):
        return False
    schema = serving.get("schema_validation")
    schema = schema if isinstance(schema, dict) else {}
    fallback = serving.get("fallback")
    fallback = fallback if isinstance(fallback, dict) else {}
    return (
        serving.get("schema_version") == "serving_readiness_v1"
        and _path_value_matches(_dict_str(serving, "model_path"), _dict_str(candidate_manifest, "model_path"))
        and _path_value_matches(_dict_str(serving, "dataset_dir"), _dict_str(candidate_manifest, "dataset_dir"))
        and _dict_str(serving, "split") == "test"
        and _is_true(schema.get("valid_input_accepted"))
        and _is_true(schema.get("invalid_input_rejected"))
        and schema.get("silent_failure") is False
        and _is_true(fallback.get("fallback_model_available"))
        and _is_true(fallback.get("rollback_runbook_available"))
        and _path_value_matches(fallback.get("rollback_runbook_path"), _path_str(rollback_runbook_path))
    )


def _serving_readiness_detail(
    serving: dict[str, Any],
    *,
    candidate_manifest: dict[str, Any] | None,
    rollback_runbook_path: Path | str | None,
    serving_error_rate: float | None,
    serving_p95: float | None,
) -> str:
    schema = serving.get("schema_validation")
    schema = schema if isinstance(schema, dict) else {}
    fallback = serving.get("fallback")
    fallback = fallback if isinstance(fallback, dict) else {}
    return (
        f"model={serving.get('model_version')}, "
        f"ready={_is_true(serving.get('ready')) or _is_true(serving.get('serving_ready'))}, "
        f"error_rate={_format_float(serving_error_rate)}, "
        f"p95={_format_float(serving_p95)}ms, "
        f"schema_version={serving.get('schema_version')}, "
        f"serving model_path={serving.get('model_path')}, "
        f"candidate model_path={_dict_str(candidate_manifest, 'model_path')}, "
        f"serving dataset={serving.get('dataset_dir')}, "
        f"candidate dataset={_dict_str(candidate_manifest, 'dataset_dir')}, "
        f"split={serving.get('split')}, "
        f"schema_valid={_is_true(schema.get('valid_input_accepted'))}, "
        f"schema_rejects_invalid={_is_true(schema.get('invalid_input_rejected'))}, "
        f"schema_silent_failure={schema.get('silent_failure') is True}, "
        f"fallback_model_available={_is_true(fallback.get('fallback_model_available'))}, "
        f"rollback_runbook_available={_is_true(fallback.get('rollback_runbook_available'))}, "
        f"rollback_runbook={fallback.get('rollback_runbook_path')}, "
        f"expected_rollback={rollback_runbook_path}"
    )


def _is_true(value: Any) -> bool:
    return value is True


def _bootstrap_artifact_matches(
    report: Any,
    name: str,
    expected: Path | str | None,
) -> bool:
    if expected is None:
        return False
    artifact_paths = report.get("artifact_paths") if isinstance(report, dict) else None
    artifact_paths = artifact_paths if isinstance(artifact_paths, dict) else {}
    return _path_value_matches(artifact_paths.get(name), str(expected))


def _bootstrap_artifact_detail(
    report: Any,
    name: str,
    expected: Path | str | None,
) -> str:
    artifact_paths = report.get("artifact_paths") if isinstance(report, dict) else None
    artifact_paths = artifact_paths if isinstance(artifact_paths, dict) else {}
    return f"{name}: bootstrap={artifact_paths.get(name)}, expected={expected}"


def _path_matches(actual: Any, expected: Path | str | None) -> bool:
    if expected is None:
        return False
    return _path_value_matches(actual, str(expected))


def _github_issue_closures_passed(
    cutover: dict[str, Any],
    *,
    expected_candidate_model_version: str,
) -> bool:
    by_issue = _github_issue_closures_by_issue(cutover)
    return all(
        _github_issue_closure_passed(
            by_issue.get(issue),
            issue=issue,
            expected_candidate_model_version=expected_candidate_model_version,
        )
        for issue in REQUIRED_CUTOVER_GITHUB_ISSUES
    )


def _github_issue_closures_detail(
    cutover: dict[str, Any],
    *,
    expected_candidate_model_version: str,
) -> str:
    by_issue = _github_issue_closures_by_issue(cutover)
    missing = [issue for issue in REQUIRED_CUTOVER_GITHUB_ISSUES if issue not in by_issue]
    invalid = [
        (
            f"#{issue}: state={_github_issue_closure_state(by_issue.get(issue))}, "
            f"repo={_github_issue_closure_repo(by_issue.get(issue))}, "
            f"comment={_github_issue_closure_comment(by_issue.get(issue))}"
        )
        for issue in REQUIRED_CUTOVER_GITHUB_ISSUES
        if issue in by_issue
        and not _github_issue_closure_passed(
            by_issue.get(issue),
            issue=issue,
            expected_candidate_model_version=expected_candidate_model_version,
        )
    ]
    return (
        "required="
        + ", ".join(f"#{issue}" for issue in REQUIRED_CUTOVER_GITHUB_ISSUES)
        + f"; missing={', '.join(f'#{issue}' for issue in missing) if missing else 'none'}"
        + f"; invalid={'; '.join(invalid) if invalid else 'none'}"
    )


def _github_issue_closures_by_issue(cutover: dict[str, Any]) -> dict[int, dict[str, Any]]:
    payload = cutover.get("github_issue_closures")
    if isinstance(payload, dict):
        rows = payload.get("closures")
        if not isinstance(rows, list):
            rows = payload.get("issues")
    else:
        rows = payload
    if not isinstance(rows, list):
        return {}
    by_issue: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        issue = _github_issue_number(row)
        if issue is not None:
            by_issue[issue] = row
    return by_issue


def _github_issue_closure_passed(
    row: dict[str, Any] | None,
    *,
    issue: int,
    expected_candidate_model_version: str,
) -> bool:
    if not isinstance(row, dict):
        return False
    state = _github_issue_closure_state(row)
    repo = _github_issue_closure_repo(row)
    comment = _github_issue_closure_comment(row)
    normalized_comment = comment.lower()
    comment_passed = (
        ("shadow pass" in normalized_comment and "promote_champion" in normalized_comment)
        if issue == 52
        else (
            "cutover complete" in normalized_comment
            and expected_candidate_model_version.lower() in normalized_comment
        )
    )
    return (
        state == "closed"
        and repo == EXPECTED_CUTOVER_GITHUB_REPO
        and comment_passed
    )


def _github_issue_number(row: dict[str, Any]) -> int | None:
    for key in ("issue", "issue_number", "number"):
        raw = row.get(key)
        try:
            return int(str(raw).lstrip("#"))
        except (TypeError, ValueError):
            continue
    return None


def _github_issue_closure_state(row: dict[str, Any] | None) -> str | None:
    state = row.get("state") if isinstance(row, dict) else None
    return state.strip().lower() if isinstance(state, str) else None


def _github_issue_closure_repo(row: dict[str, Any] | None) -> str | None:
    repo = row.get("repo") if isinstance(row, dict) else None
    return repo.strip() if isinstance(repo, str) else None


def _github_issue_closure_comment(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("comment", "closure_comment", "comment_body", "close_comment"):
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _path_value_matches(actual: Any, expected: str | None) -> bool:
    if not isinstance(actual, str) or expected is None:
        return False
    try:
        return Path(actual).resolve() == Path(expected).resolve()
    except OSError:
        return actual == expected


def _status_artifact_age_seconds(status: dict[str, Any]) -> float | None:
    raw_generated_at = status.get("generated_at")
    if not isinstance(raw_generated_at, str) or not raw_generated_at.strip():
        return None
    try:
        generated_at = datetime.fromisoformat(raw_generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - generated_at.astimezone(UTC)).total_seconds())


def _format_float(value: float | None) -> str:
    return "missing" if value is None else f"{value:.4f}"


def _format_metric_cell(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
