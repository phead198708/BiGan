"""Diagnostic UP SELL_BEFORE_CLOSE full candidate-pool reports."""

from __future__ import annotations

import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.post_freeze_m2_replay_parity import (
    M2_REPLAY_PARITY_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.post_freeze_n2_up_feature_proxy import (
    N2_ALLOWED_DECISION_INPUT_FIELDS,
    N2_FORBIDDEN_SELECTION_FIELDS,
    PolymarketN2UpFeatureProxyConfig,
    _assert_non_leaky_inputs,
    _decision_time_proxy,
)
from bigan.v8.polymarket.training.post_freeze_n_up_replay_aligned import (
    _overlay_pnl_sum,
)
from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (
    _compact_row,
    _label,
    _pearson,
    _pnl,
    _read_json,
    _score,
    _sha256_file,
    _write_json,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

UP_FULL_CANDIDATE_POOL_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-up-full-candidate-pool-diagnostic-v1"
)
UP_FULL_CANDIDATE_POOL_FEATURE_PROXY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-up-full-candidate-pool-feature-proxy-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketUpFullCandidatePoolConfig:
    """Configuration for diagnostic-only UP full-pool reports."""

    m2_candidate_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_up_full_candidate_pool_diagnostic"
    overwrite_existing: bool = False
    min_feature_proxy_score: float = 0.03
    min_immediate_exit_pnl_proxy: float = 0.0
    max_spread_bps: float = 900.0
    min_queue_fill: float = 0.65
    max_book_staleness_ms: float = 10_000.0
    min_time_to_close_seconds: float = 90.0
    min_recent_book_update_count_1m: float = 1.0
    high_score_risk_threshold: float = 0.75
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in ("m2_candidate_report_path", "output_dir"):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                object.__setattr__(self, field_name, Path(value))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def n2_proxy_config(self) -> PolymarketN2UpFeatureProxyConfig:
        return PolymarketN2UpFeatureProxyConfig(
            m2_candidate_report_path=self.m2_candidate_report_path,
            output_dir=self.output_dir,
            run_id=f"{self.run_id}_proxy_config",
            min_feature_proxy_score=self.min_feature_proxy_score,
            min_immediate_exit_pnl_proxy=self.min_immediate_exit_pnl_proxy,
            max_spread_bps=self.max_spread_bps,
            min_queue_fill=self.min_queue_fill,
            max_book_staleness_ms=self.max_book_staleness_ms,
            min_time_to_close_seconds=self.min_time_to_close_seconds,
            min_recent_book_update_count_1m=self.min_recent_book_update_count_1m,
            high_score_risk_threshold=self.high_score_risk_threshold,
        )


@dataclass(frozen=True, slots=True)
class PolymarketUpFullCandidatePoolResult:
    run_dir: Path
    candidate_pool_report: dict[str, Any]
    feature_proxy_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_up_full_candidate_pool_diagnostics(
    config: PolymarketUpFullCandidatePoolConfig,
) -> PolymarketUpFullCandidatePoolResult:
    """Build diagnostic-only full-pool UP SELL_BEFORE_CLOSE reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "candidate_pool_report": run_dir
        / "up_full_candidate_pool_diagnostic_report.json",
        "candidate_pool_summary": run_dir
        / "up_full_candidate_pool_diagnostic_report.md",
        "feature_proxy_report": run_dir
        / "up_full_candidate_pool_feature_proxy_report.json",
        "feature_proxy_summary": run_dir
        / "up_full_candidate_pool_feature_proxy_report.md",
        "manifest": run_dir / "up_full_candidate_pool_diagnostic_manifest.json",
    }
    candidate_pool_report, feature_proxy_report = _build_reports(config=config)
    _write_json(artifact_paths["candidate_pool_report"], candidate_pool_report)
    artifact_paths["candidate_pool_summary"].write_text(
        _candidate_pool_markdown(candidate_pool_report),
        encoding="utf-8",
    )
    _write_json(artifact_paths["feature_proxy_report"], feature_proxy_report)
    artifact_paths["feature_proxy_summary"].write_text(
        _feature_proxy_markdown(feature_proxy_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-up-full-candidate-pool-artifacts-v1",
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path.relative_to(run_dir))
            for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": {
            name: _sha256_file(path)
            for name, path in sorted(artifact_paths.items())
            if name != "manifest"
        },
        **compact_safety_fields(),
    }
    manifest["artifact_hashes"]["manifest"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["manifest"], manifest)
    return PolymarketUpFullCandidatePoolResult(
        run_dir=run_dir,
        candidate_pool_report=candidate_pool_report,
        feature_proxy_report=feature_proxy_report,
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketUpFullCandidatePoolConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m2_report_path = config.m2_candidate_report_path.expanduser().resolve()
    m2_report = _read_json(m2_report_path)
    if m2_report.get("schema_version") != M2_REPLAY_PARITY_SCHEMA_VERSION:
        raise ValueError("not an M2 replay-parity candidate report")

    source_rows, source_reports = _load_source_rows(m2_report)
    selected_keys = {_row_key(row) for row in m2_report.get("m2_selected_rows", [])}
    total_up_rows = [
        _pool_row(row, selected_keys, config)
        for row in source_rows
        if str(row.get("action") or "") == "BUY_UP_SELL_BEFORE_CLOSE"
        or (
            str(row.get("selected_side") or "") == "UP"
            and str(row.get("action") or "").endswith("SELL_BEFORE_CLOSE")
        )
    ]
    guard_compatible_rows = [
        row for row in total_up_rows if bool(row.get("guard_compatible_candidate"))
    ]
    selected_rows = [row for row in guard_compatible_rows if row["m2_selected"]]
    non_selected_rows = [
        row for row in guard_compatible_rows if not row["m2_selected"]
    ]

    candidate_pool_report = _candidate_pool_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        source_reports=source_reports,
        total_up_rows=total_up_rows,
        guard_compatible_rows=guard_compatible_rows,
        selected_rows=selected_rows,
        non_selected_rows=non_selected_rows,
    )
    feature_proxy_report = _feature_proxy_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        guard_compatible_rows=guard_compatible_rows,
        selected_rows=selected_rows,
        non_selected_rows=non_selected_rows,
    )
    return candidate_pool_report, feature_proxy_report


def _load_source_rows(
    m2_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        {
            str(row.get("source_report_path") or "")
            for row in [
                *m2_report.get("m2_selected_rows", []),
                *m2_report.get("m2_blocked_rows", []),
            ]
            if row.get("source_report_path")
        }
    )
    rows: list[dict[str, Any]] = []
    reports = []
    seen: set[tuple[str, str, int, str]] = set()
    for path_text in paths:
        path = Path(path_text).expanduser().resolve()
        report = _read_json(path)
        reports.append(
            {
                "source_report_path": str(path),
                "source_report_sha256": _sha256_file(path),
                "run_id": report.get("run_id"),
                "row_count": len(report.get("rows", [])),
            }
        )
        for row in report.get("rows", []):
            payload = dict(row)
            payload["source_report_path"] = str(path)
            key = _row_key(payload)
            if key in seen:
                continue
            seen.add(key)
            rows.append(payload)
    return rows, reports


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("source_report_path") or ""),
        str(row.get("market_id") or ""),
        int(row.get("decision_ts") or 0),
        str(row.get("action") or ""),
    )


def _pool_row(
    row: dict[str, Any],
    selected_keys: set[tuple[str, str, int, str]],
    config: PolymarketUpFullCandidatePoolConfig,
) -> dict[str, Any]:
    n2_decision = _decision_time_proxy(row, config.n2_proxy_config())
    _assert_non_leaky_inputs(n2_decision["n2_decision_input_fields_used"])
    label = _label(row)
    pnl = _pnl(row)
    result = {
        **_compact_row(row),
        **n2_decision,
        "guard_compatible_candidate": bool(row.get("guard_compatible_candidate")),
        "pre_guard_candidate": bool(row.get("pre_guard_candidate")),
        "entry_order_opened": bool(row.get("entry_order_opened")),
        "m2_selected": _row_key(row) in selected_keys,
        "segment": "m2_selected" if _row_key(row) in selected_keys else "non_selected",
        "original_label_target": label,
        "realized_replay_pnl": pnl,
        "label_vs_replay_gap_before": label - pnl,
        "feature_proxy_label_vs_replay_gap_after": (
            float(n2_decision["n2_feature_exit_label_proxy"]) - pnl
        ),
        "positive_label_replay_negative_flagged_for_evaluation": (
            label > 0.0 and pnl < 0.0
        ),
        "negative_label_selected_flagged_for_evaluation": label < 0.0,
        "n2_forbidden_fields_present_for_evaluation_only": sorted(
            field for field in N2_FORBIDDEN_SELECTION_FIELDS if field in row
        ),
        "n2_forbidden_fields_used_for_selection": [],
    }
    return result


def _candidate_pool_report(
    *,
    config: PolymarketUpFullCandidatePoolConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    source_reports: list[dict[str, Any]],
    total_up_rows: list[dict[str, Any]],
    guard_compatible_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    non_selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del config
    viable_non_selected = [
        row for row in non_selected_rows if bool(row.get("n2_would_select"))
    ]
    report = {
        "schema_version": UP_FULL_CANDIDATE_POOL_DIAGNOSTIC_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        ],
        "report_type": "up_full_candidate_pool_diagnostic",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "source_reports": source_reports,
        "total_up_candidate_pool_size": len(total_up_rows),
        "guard_compatible_up_pool_size": len(guard_compatible_rows),
        "m2_selected_up_count": len(selected_rows),
        "m2_non_selected_guard_compatible_up_count": len(non_selected_rows),
        "pool_segment_metrics": {
            "guard_compatible_all": _segment_metrics(guard_compatible_rows),
            "m2_selected": _segment_metrics(selected_rows),
            "m2_non_selected": _segment_metrics(non_selected_rows),
        },
        "replay_pnl_distribution_by_pool_segment": {
            "guard_compatible_all": _pnl_distribution(guard_compatible_rows),
            "m2_selected": _pnl_distribution(selected_rows),
            "m2_non_selected": _pnl_distribution(non_selected_rows),
        },
        "label_vs_replay_gap_by_pool_segment": {
            "guard_compatible_all": _label_gap(guard_compatible_rows),
            "m2_selected": _label_gap(selected_rows),
            "m2_non_selected": _label_gap(non_selected_rows),
        },
        "decision_time_field_availability": _field_availability(
            guard_compatible_rows
        ),
        "evaluation_only_fields": list(N2_FORBIDDEN_SELECTION_FIELDS),
        "allowed_decision_time_input_fields": list(N2_ALLOWED_DECISION_INPUT_FIELDS),
        "non_selected_up_rows_viable_under_non_leaky_proxy_count": len(
            viable_non_selected
        ),
        "non_selected_up_rows_viable_under_non_leaky_proxy": [
            _compact_pool_row(row) for row in viable_non_selected[:25]
        ],
        "up_path_should_remain_fully_blocked": len(viable_non_selected) == 0,
        "up_path_block_recommendation_reason_codes": _block_recommendation_reasons(
            viable_non_selected
        ),
        "rows": [_compact_pool_row(row) for row in guard_compatible_rows],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["up_full_candidate_pool_diagnostic_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _feature_proxy_report(
    *,
    config: PolymarketUpFullCandidatePoolConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    guard_compatible_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    non_selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del config
    n2_selected = [row for row in guard_compatible_rows if row["n2_would_select"]]
    n2_non_selected_viable = [
        row for row in non_selected_rows if row["n2_would_select"]
    ]
    score_before = [_score(row) for row in guard_compatible_rows]
    score_after = [
        float(row["n2_replay_aligned_feature_score_proxy"])
        for row in guard_compatible_rows
    ]
    pnl = [_pnl(row) for row in guard_compatible_rows]
    report = {
        "schema_version": UP_FULL_CANDIDATE_POOL_FEATURE_PROXY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        ],
        "report_type": "up_full_candidate_pool_feature_proxy",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "allowed_decision_time_input_fields": list(N2_ALLOWED_DECISION_INPUT_FIELDS),
        "evaluation_only_fields": list(N2_FORBIDDEN_SELECTION_FIELDS),
        "selection_uses_only_allowed_fields": _all_rows_non_leaky(
            guard_compatible_rows
        ),
        "guard_compatible_up_pool_size": len(guard_compatible_rows),
        "m2_selected_up_count": len(selected_rows),
        "m2_non_selected_guard_compatible_up_count": len(non_selected_rows),
        "n2_would_selected_full_pool_count": len(n2_selected),
        "n2_would_selected_non_selected_pool_count": len(n2_non_selected_viable),
        "n2_would_selected_full_pool_replay_pnl_sum": _overlay_pnl_sum(n2_selected),
        "n2_would_selected_non_selected_replay_pnl_sum": _overlay_pnl_sum(
            n2_non_selected_viable
        ),
        "original_score_vs_replay_correlation": _pearson(score_before, pnl),
        "feature_proxy_score_vs_replay_correlation": _pearson(score_after, pnl),
        "score_correlation_delta": _correlation_delta(score_before, score_after, pnl),
        "label_vs_replay_gap_before_overlay": _label_gap(guard_compatible_rows),
        "label_vs_replay_gap_after_feature_proxy_overlay": _proxy_gap(
            guard_compatible_rows
        ),
        "decision_time_proxy_buckets": _bucket_metrics(
            guard_compatible_rows,
            "n2_calibrated_score_bucket",
        ),
        "immediate_exit_proxy_buckets": _bucket_metrics(
            guard_compatible_rows,
            "immediate_exit_proxy_bucket",
            bucket_fn=_immediate_exit_bucket,
        ),
        "spread_buckets": _bucket_metrics(
            guard_compatible_rows,
            "spread_bucket",
            bucket_fn=_spread_bucket,
        ),
        "queue_buckets": _bucket_metrics(
            guard_compatible_rows,
            "queue_bucket",
            bucket_fn=_queue_bucket,
        ),
        "staleness_buckets": _bucket_metrics(
            guard_compatible_rows,
            "staleness_bucket",
            bucket_fn=_staleness_bucket,
        ),
        "time_to_close_buckets": _bucket_metrics(
            guard_compatible_rows,
            "time_to_close_bucket",
            bucket_fn=_time_to_close_bucket,
        ),
        "m2_selected_vs_non_selected_proxy_comparison": {
            "m2_selected": _segment_metrics(selected_rows),
            "m2_non_selected": _segment_metrics(non_selected_rows),
        },
        "non_selected_up_rows_viable_under_non_leaky_proxy": [
            _compact_pool_row(row) for row in n2_non_selected_viable[:25]
        ],
        "up_path_should_remain_fully_blocked": len(n2_non_selected_viable) == 0,
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["up_full_candidate_pool_feature_proxy_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _segment_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if _pnl(row) > 0.0]
    negatives = [row for row in rows if _pnl(row) < 0.0]
    n2_viable = [row for row in rows if row.get("n2_would_select")]
    return {
        "row_count": len(rows),
        "replay_entry_count": sum(1 for row in rows if row.get("entry_order_opened")),
        "replay_pnl_sum": _overlay_pnl_sum(rows),
        "positive_replay_count": len(positives),
        "negative_replay_count": len(negatives),
        "label_target_sum": sum(_label(row) for row in rows),
        "label_vs_replay_gap": _label_gap(rows),
        "n2_would_selected_count": len(n2_viable),
        "n2_would_selected_replay_pnl_sum": _overlay_pnl_sum(n2_viable),
        "score_vs_replay_correlation": _pearson(
            [_score(row) for row in rows],
            [_pnl(row) for row in rows],
        ),
        "feature_proxy_score_vs_replay_correlation": _pearson(
            [float(row["n2_replay_aligned_feature_score_proxy"]) for row in rows],
            [_pnl(row) for row in rows],
        ),
    }


def _pnl_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_pnl(row) for row in rows]
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
            "sum": 0.0,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
        "sum": sum(values),
    }


def _label_gap(rows: list[dict[str, Any]]) -> float:
    return sum(_label(row) - _pnl(row) for row in rows)


def _proxy_gap(rows: list[dict[str, Any]]) -> float:
    return sum(
        float(row["n2_feature_exit_label_proxy"]) - _pnl(row) for row in rows
    )


def _field_availability(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if row.get(field) is not None)
        for field in N2_ALLOWED_DECISION_INPUT_FIELDS
    }


def _all_rows_non_leaky(rows: list[dict[str, Any]]) -> bool:
    return all(
        not set(row["n2_decision_input_fields_used"]).intersection(
            N2_FORBIDDEN_SELECTION_FIELDS
        )
        and not row["n2_forbidden_fields_used_for_selection"]
        for row in rows
    )


def _block_recommendation_reasons(viable_non_selected: list[dict[str, Any]]) -> list[str]:
    if not viable_non_selected:
        return ["no_non_selected_up_rows_viable_under_non_leaky_proxy"]
    if _overlay_pnl_sum(viable_non_selected) <= 0.0:
        return ["non_selected_up_proxy_viable_rows_do_not_show_positive_replay_pnl"]
    return ["non_selected_up_proxy_viable_rows_require_future_holdout_validation"]


def _bucket_metrics(
    rows: list[dict[str, Any]],
    key: str,
    bucket_fn: Any | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = str(bucket_fn(row) if bucket_fn else row.get(key, "unknown"))
        groups.setdefault(bucket, []).append(row)
    return [
        {
            key: bucket,
            **_segment_metrics(bucket_rows),
        }
        for bucket, bucket_rows in sorted(groups.items())
    ]


def _immediate_exit_bucket(row: dict[str, Any]) -> str:
    value = row.get("n2_immediate_exit_pnl_proxy")
    if value is None:
        return "unknown"
    pnl = float(value)
    if pnl <= 0.0:
        return "<=0"
    if pnl < 0.02:
        return "0-0.02"
    if pnl < 0.05:
        return "0.02-0.05"
    return ">=0.05"


def _spread_bucket(row: dict[str, Any]) -> str:
    value = row.get("n2_spread_bps")
    if value is None:
        return "unknown"
    spread = float(value)
    if spread < 300:
        return "<300bps"
    if spread < 600:
        return "300-600bps"
    if spread < 900:
        return "600-900bps"
    return ">=900bps"


def _queue_bucket(row: dict[str, Any]) -> str:
    value = row.get("n2_queue_fill_proxy")
    if value is None:
        return "unknown"
    queue = float(value)
    if queue < 0.50:
        return "<0.50"
    if queue < 0.65:
        return "0.50-0.65"
    if queue < 0.80:
        return "0.65-0.80"
    return ">=0.80"


def _staleness_bucket(row: dict[str, Any]) -> str:
    value = row.get("n2_book_staleness_ms")
    if value is None:
        return "unknown"
    stale = float(value)
    if stale < 1000:
        return "<1s"
    if stale < 5000:
        return "1-5s"
    if stale < 10000:
        return "5-10s"
    return ">=10s"


def _time_to_close_bucket(row: dict[str, Any]) -> str:
    value = row.get("n2_time_to_close_seconds")
    if value is None:
        return "unknown"
    seconds = float(value)
    if seconds < 90:
        return "<90s"
    if seconds < 180:
        return "90-180s"
    if seconds < 300:
        return "180-300s"
    return ">=300s"


def _correlation_delta(
    score_before: list[float],
    score_after: list[float],
    pnl: list[float],
) -> float | None:
    before = _pearson(score_before, pnl)
    after = _pearson(score_after, pnl)
    if before is None or after is None:
        return None
    return after - before


def _compact_pool_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_row(row)
    extra_fields = (
        "guard_compatible_candidate",
        "pre_guard_candidate",
        "entry_order_opened",
        "m2_selected",
        "segment",
        "original_label_target",
        "realized_replay_pnl",
        "label_vs_replay_gap_before",
        "feature_proxy_label_vs_replay_gap_after",
        "n2_would_select",
        "n2_decision_reason_codes",
        "n2_replay_aligned_feature_score_proxy",
        "n2_immediate_exit_pnl_proxy",
        "n2_calibrated_score_bucket",
        "n2_side_specific_up_risk_bucket",
        "n2_spread_bps",
        "n2_queue_fill_proxy",
        "n2_book_staleness_ms",
        "n2_time_to_close_seconds",
        "n2_recent_book_update_count_1m",
        "n2_decision_input_fields_used",
        "n2_forbidden_fields_used_for_selection",
    )
    compact.update({field: row.get(field) for field in extra_fields if field in row})
    return compact


def _candidate_pool_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UP Full Candidate-Pool Diagnostic",
        "",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- total_up_candidate_pool_size: `{report['total_up_candidate_pool_size']}`",
        f"- guard_compatible_up_pool_size: `{report['guard_compatible_up_pool_size']}`",
        f"- m2_selected_up_count: `{report['m2_selected_up_count']}`",
        "- m2_non_selected_guard_compatible_up_count: "
        f"`{report['m2_non_selected_guard_compatible_up_count']}`",
        "- non_selected_up_rows_viable_under_non_leaky_proxy_count: "
        f"`{report['non_selected_up_rows_viable_under_non_leaky_proxy_count']}`",
        "- up_path_should_remain_fully_blocked: "
        f"`{str(report['up_path_should_remain_fully_blocked']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Segment Metrics",
        "",
        "| segment | rows | pnl_sum | label_gap | n2_selected |",
        "|---|---:|---:|---:|---:|",
    ]
    for segment, metrics in report["pool_segment_metrics"].items():
        lines.append(
            "| {segment} | {rows} | {pnl:.6f} | {gap:.6f} | {n2} |".format(
                segment=segment,
                rows=metrics["row_count"],
                pnl=float(metrics["replay_pnl_sum"]),
                gap=float(metrics["label_vs_replay_gap"]),
                n2=metrics["n2_would_selected_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- diagnostic-only; no selector/rank/gate changes",
            "- replay/PnL fields are evaluation-only",
            "- paper_only: true",
            "- capital_at_risk: false",
            "",
        ]
    )
    return "\n".join(lines)


def _feature_proxy_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UP Full Candidate-Pool Feature Proxy",
        "",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        "- selection_uses_only_allowed_fields: "
        f"`{str(report['selection_uses_only_allowed_fields']).lower()}`",
        f"- guard_compatible_up_pool_size: `{report['guard_compatible_up_pool_size']}`",
        "- n2_would_selected_full_pool_count: "
        f"`{report['n2_would_selected_full_pool_count']}`",
        "- n2_would_selected_non_selected_pool_count: "
        f"`{report['n2_would_selected_non_selected_pool_count']}`",
        "- original_score_vs_replay_correlation: "
        f"`{report['original_score_vs_replay_correlation']}`",
        "- feature_proxy_score_vs_replay_correlation: "
        f"`{report['feature_proxy_score_vs_replay_correlation']}`",
        "- up_path_should_remain_fully_blocked: "
        f"`{str(report['up_path_should_remain_fully_blocked']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Immediate Exit Buckets",
        "",
        "| bucket | rows | pnl_sum | n2_selected |",
        "|---|---:|---:|---:|",
    ]
    for row in report["immediate_exit_proxy_buckets"]:
        lines.append(
            "| {bucket} | {rows} | {pnl:.6f} | {n2} |".format(
                bucket=row["immediate_exit_proxy_bucket"],
                rows=row["row_count"],
                pnl=float(row["replay_pnl_sum"]),
                n2=row["n2_would_selected_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- diagnostic-only; no paper/live unlock",
            "- paper_only: true",
            "- capital_at_risk: false",
            "",
        ]
    )
    return "\n".join(lines)
