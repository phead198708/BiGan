"""UP SELL_BEFORE_CLOSE label/replay and calibration diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
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
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

UP_LABEL_REPLAY_ALIGNMENT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-up-sell-before-close-label-replay-alignment-v1"
)
UP_ACTION_VALUE_CALIBRATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-up-sell-before-close-action-value-calibration-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketUpSellBeforeCloseDiagnosticsConfig:
    """Configuration for diagnostic-only UP SELL_BEFORE_CLOSE reports."""

    m2_candidate_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_up_sell_before_close_diagnostics"
    overwrite_existing: bool = False
    high_score_threshold: float = 0.80
    low_score_threshold: float = 0.50
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
        if self.high_score_threshold <= self.low_score_threshold:
            raise ValueError("high_score_threshold must exceed low_score_threshold")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id


@dataclass(frozen=True, slots=True)
class PolymarketUpSellBeforeCloseDiagnosticsResult:
    run_dir: Path
    label_replay_report: dict[str, Any]
    calibration_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_up_sell_before_close_diagnostics(
    config: PolymarketUpSellBeforeCloseDiagnosticsConfig,
) -> PolymarketUpSellBeforeCloseDiagnosticsResult:
    """Create UP SELL_BEFORE_CLOSE diagnostic reports from M2 evidence."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "label_replay_report": run_dir
        / "up_sell_before_close_label_replay_alignment_report.json",
        "label_replay_summary": run_dir
        / "up_sell_before_close_label_replay_alignment_report.md",
        "calibration_report": run_dir
        / "up_sell_before_close_action_value_calibration_diagnostic.json",
        "calibration_summary": run_dir
        / "up_sell_before_close_action_value_calibration_diagnostic.md",
        "manifest": run_dir / "up_sell_before_close_diagnostics_manifest.json",
    }
    label_replay_report, calibration_report = _build_reports(config=config)
    _write_json(artifact_paths["label_replay_report"], label_replay_report)
    artifact_paths["label_replay_summary"].write_text(
        _label_replay_markdown(label_replay_report),
        encoding="utf-8",
    )
    _write_json(artifact_paths["calibration_report"], calibration_report)
    artifact_paths["calibration_summary"].write_text(
        _calibration_markdown(calibration_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-up-sell-before-close-diagnostics-artifacts-v1",
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
    return PolymarketUpSellBeforeCloseDiagnosticsResult(
        run_dir=run_dir,
        label_replay_report=label_replay_report,
        calibration_report=calibration_report,
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketUpSellBeforeCloseDiagnosticsConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m2_report_path = config.m2_candidate_report_path.expanduser().resolve()
    m2_report = _read_json(m2_report_path)
    if m2_report.get("schema_version") != M2_REPLAY_PARITY_SCHEMA_VERSION:
        raise ValueError("not an M2 replay-parity candidate report")
    enriched_rows = _enriched_selected_rows(m2_report)
    up_rows = [row for row in enriched_rows if row.get("selected_side") == "UP"]
    down_rows = [row for row in enriched_rows if row.get("selected_side") == "DOWN"]
    up_replay_rows = [row for row in up_rows if bool(row.get("entry_order_opened"))]
    down_replay_rows = [row for row in down_rows if bool(row.get("entry_order_opened"))]
    label_report = _build_label_replay_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        up_rows=up_rows,
        up_replay_rows=up_replay_rows,
        down_replay_rows=down_replay_rows,
    )
    calibration_report = _build_calibration_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        up_rows=up_rows,
        up_replay_rows=up_replay_rows,
        down_replay_rows=down_replay_rows,
    )
    return label_report, calibration_report


def _enriched_selected_rows(m2_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for row in m2_report.get("m2_selected_rows", []):
        source_path = str(row.get("source_report_path") or "")
        key = (str(row.get("market_id")), int(row.get("decision_ts") or 0))
        source_row = {}
        if source_path:
            if source_path not in source_cache:
                source_cache[source_path] = _source_rows_by_key(Path(source_path))
            source_row = source_cache[source_path].get(key, {})
        rows.append({**source_row, **row})
    return rows


def _source_rows_by_key(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    report = _read_json(path)
    return {
        (str(row.get("market_id")), int(row.get("decision_ts") or 0)): row
        for row in report.get("rows", [])
    }


def _build_label_replay_report(
    *,
    config: PolymarketUpSellBeforeCloseDiagnosticsConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    up_rows: list[dict[str, Any]],
    up_replay_rows: list[dict[str, Any]],
    down_replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_label_negative_replay = [
        row
        for row in up_replay_rows
        if _label(row) > 0.0 and _pnl(row) < 0.0
    ]
    negative_label_selected = [row for row in up_rows if _label(row) < 0.0]
    first_executable_negative = [
        row
        for row in up_replay_rows
        if _pnl(row) < 0.0
        and _has_reason(row, "closed_before_settlement_with_negative_replay_pnl")
    ]
    closed_before_settlement_negative = [
        row
        for row in up_replay_rows
        if _pnl(row) < 0.0
        and (
            bool(row.get("closed_before_settlement", False))
            or _has_reason(row, "closed_before_settlement_with_negative_replay_pnl")
        )
    ]
    microstructure = _microstructure_summary(up_replay_rows)
    root_causes = _root_cause_codes(
        up_replay_rows=up_replay_rows,
        positive_label_negative_replay=positive_label_negative_replay,
        negative_label_selected=negative_label_selected,
        first_executable_negative=first_executable_negative,
        score_correlation=_pearson(
            [_score(row) for row in up_replay_rows],
            [_pnl(row) for row in up_replay_rows],
        ),
    )
    report = {
        "schema_version": UP_LABEL_REPLAY_ALIGNMENT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "up_sell_before_close_label_replay_alignment",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "current_frozen_m_promotion_status": m2_report.get(
            "current_frozen_m_promotion_status",
            "reject_promotion_for_now",
        ),
        "current_frozen_m_evidence_status": m2_report.get(
            "current_frozen_m_evidence_status",
            "weak_mixed_structural",
        ),
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_promotion_ready": False,
        "up_selected_rows": [_compact_row(row) for row in up_rows],
        "up_known_replay_rows": [_compact_row(row) for row in up_replay_rows],
        "up_selected_entry_count": len(up_rows),
        "up_known_replay_entry_count": len(up_replay_rows),
        "up_label_target_sum": _sum_labels(up_rows),
        "up_replay_pnl_sum": _sum_pnl(up_replay_rows),
        "up_label_vs_replay_gap": _sum_labels(up_replay_rows) - _sum_pnl(up_replay_rows),
        "up_label_vs_replay_gap_by_market": _group_gap(up_replay_rows, ("market_id",)),
        "up_label_vs_replay_gap_by_time_to_close_bucket": _group_gap(
            up_replay_rows,
            ("time_to_close_bucket",),
            bucket_overrides={"time_to_close_bucket": _time_to_close_bucket},
        ),
        "up_label_vs_replay_gap_by_spread_bucket": _group_gap(
            up_replay_rows,
            ("spread_bucket",),
            bucket_overrides={"spread_bucket": _spread_bucket},
        ),
        "up_label_vs_replay_gap_by_liquidity_bucket": _group_gap(
            up_replay_rows,
            ("liquidity_bucket",),
            bucket_overrides={"liquidity_bucket": _liquidity_bucket},
        ),
        "up_label_vs_replay_gap_by_staleness_bucket": _group_gap(
            up_replay_rows,
            ("staleness_bucket",),
            bucket_overrides={"staleness_bucket": _staleness_bucket},
        ),
        "up_positive_label_replay_negative_count": len(positive_label_negative_replay),
        "up_positive_label_replay_negative_rows": [
            _compact_row(row) for row in positive_label_negative_replay
        ],
        "up_negative_label_selected_count": len(negative_label_selected),
        "up_negative_label_selected_rows": [
            _compact_row(row) for row in negative_label_selected
        ],
        "up_top_false_positives": _top_false_positives(up_replay_rows),
        "up_top_false_negatives": _top_false_negatives(up_replay_rows),
        "first_executable_exit_negative_count": len(first_executable_negative),
        "first_executable_exit_negative_rows": [
            _compact_row(row) for row in first_executable_negative
        ],
        "closed_before_settlement_negative_count": len(
            closed_before_settlement_negative
        ),
        "closed_before_settlement_negative_rows": [
            _compact_row(row) for row in closed_before_settlement_negative
        ],
        "entry_exit_microstructure_summary": microstructure,
        "label_exit_target_vs_actual_replay_exit_path": {
            "label_path_available": bool(up_replay_rows),
            "observed_replay_exit_policy": "first_executable_exit_after_entry",
            "label_target_field": "action_return_target",
            "actual_replay_pnl_field": "total_polymarket_pnl",
            "gap": _sum_labels(up_replay_rows) - _sum_pnl(up_replay_rows),
        },
        "side_comparison": _side_comparison(up_replay_rows, down_replay_rows),
        "root_cause_classification": _classification(root_causes),
        "root_cause_codes": root_causes,
        "recommended_next_actions": _recommended_actions(root_causes),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["up_sell_before_close_label_replay_alignment_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _build_calibration_report(
    *,
    config: PolymarketUpSellBeforeCloseDiagnosticsConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    up_rows: list[dict[str, Any]],
    up_replay_rows: list[dict[str, Any]],
    down_replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    high_score_negative = [
        row
        for row in up_replay_rows
        if _score(row) >= config.high_score_threshold and _pnl(row) < 0.0
    ]
    low_score_positive = [
        row
        for row in up_replay_rows
        if _score(row) <= config.low_score_threshold and _pnl(row) > 0.0
    ]
    negative_label_selected = [row for row in up_rows if _label(row) < 0.0]
    positive_label_negative_replay = [
        row for row in up_replay_rows if _label(row) > 0.0 and _pnl(row) < 0.0
    ]
    score_corr = _pearson([_score(row) for row in up_replay_rows], [_pnl(row) for row in up_replay_rows])
    rank_corr = _pearson([_rank(row) for row in up_replay_rows], [_pnl(row) for row in up_replay_rows])
    target_score_corr = _pearson([_label(row) for row in up_rows], [_score(row) for row in up_rows])
    root_causes = _root_cause_codes(
        up_replay_rows=up_replay_rows,
        positive_label_negative_replay=positive_label_negative_replay,
        negative_label_selected=negative_label_selected,
        first_executable_negative=[
            row
            for row in up_replay_rows
            if _pnl(row) < 0.0
            and _has_reason(row, "closed_before_settlement_with_negative_replay_pnl")
        ],
        score_correlation=score_corr,
    )
    report = {
        "schema_version": UP_ACTION_VALUE_CALIBRATION_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "up_sell_before_close_action_value_calibration_diagnostic",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "current_frozen_m_promotion_status": m2_report.get(
            "current_frozen_m_promotion_status",
            "reject_promotion_for_now",
        ),
        "current_frozen_m_evidence_status": m2_report.get(
            "current_frozen_m_evidence_status",
            "weak_mixed_structural",
        ),
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_promotion_ready": False,
        "up_selected_entry_count": len(up_rows),
        "up_known_replay_entry_count": len(up_replay_rows),
        "calibrated_action_score_vs_realized_up_replay_pnl_correlation": score_corr,
        "rank_score_vs_realized_up_replay_pnl_correlation": rank_corr,
        "action_target_vs_calibrated_score_correlation": target_score_corr,
        "score_buckets": _score_bucket_summaries(up_replay_rows),
        "score_calibration_error_by_bucket": _score_calibration_error_by_bucket(
            up_replay_rows
        ),
        "high_score_threshold": config.high_score_threshold,
        "high_score_negative_replay_up_count": len(high_score_negative),
        "high_score_negative_replay_up_rows": [
            _compact_row(row) for row in high_score_negative
        ],
        "low_score_threshold": config.low_score_threshold,
        "low_score_positive_replay_up_count": len(low_score_positive),
        "low_score_positive_replay_up_rows": [
            _compact_row(row) for row in low_score_positive
        ],
        "up_negative_label_selected_count": len(negative_label_selected),
        "up_positive_label_replay_negative_count": len(
            positive_label_negative_replay
        ),
        "side_comparison": _side_comparison(up_replay_rows, down_replay_rows),
        "root_cause_classification": _classification(root_causes),
        "root_cause_codes": root_causes,
        "recommended_next_actions": _recommended_actions(root_causes),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["up_sell_before_close_action_value_calibration_diagnostic_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "market_id",
        "slug",
        "decision_ts",
        "selected_side",
        "action",
        "action_return_target",
        "label_pnl_target",
        "total_polymarket_pnl",
        "realized_trade_pnl",
        "settlement_pnl",
        "raw_calibrated_action_score",
        "candidate_rank_score",
        "best_action_margin",
        "entry_quality_ask",
        "exit_quality_bid",
        "entry_exit_quality_spread_bps",
        "entry_exit_quality_queue_fill",
        "entry_exit_quality_book_staleness_ms",
        "entry_exit_quality_time_to_close_seconds",
        "execution_pnl_immediate_exit_pnl",
        "execution_pnl_model_vs_immediate_exit_pnl_gap_estimate",
        "closed_before_settlement",
        "exit_reason_codes",
        "replay_reason_codes",
        "attrition_reason_codes",
        "source_report_path",
    )
    return {field: row.get(field) for field in fields if field in row}


def _sum_labels(rows: list[dict[str, Any]]) -> float:
    return sum(_label(row) for row in rows)


def _sum_pnl(rows: list[dict[str, Any]]) -> float:
    return sum(_pnl(row) for row in rows)


def _label(row: dict[str, Any]) -> float:
    return float(row.get("action_return_target") or row.get("label_pnl_target") or 0.0)


def _pnl(row: dict[str, Any]) -> float:
    return float(row.get("total_polymarket_pnl") or 0.0)


def _score(row: dict[str, Any]) -> float:
    return float(row.get("raw_calibrated_action_score") or 0.0)


def _rank(row: dict[str, Any]) -> float:
    return float(row.get("candidate_rank_score") or 0.0)


def _group_gap(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    bucket_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parts = []
        for key in keys:
            if bucket_overrides and key in bucket_overrides:
                parts.append(str(bucket_overrides[key](row)))
            else:
                parts.append(str(row.get(key, "unknown")))
        groups[tuple(parts)].append(row)
    result = []
    for parts, group_rows in sorted(groups.items()):
        label_sum = _sum_labels(group_rows)
        replay_sum = _sum_pnl(group_rows)
        payload = {
            "row_count": len(group_rows),
            "label_target_sum": label_sum,
            "replay_pnl_sum": replay_sum,
            "label_vs_replay_gap": label_sum - replay_sum,
        }
        for index, key in enumerate(keys):
            payload[key] = parts[index]
        result.append(payload)
    return result


def _time_to_close_bucket(row: dict[str, Any]) -> str:
    value = row.get("entry_exit_quality_time_to_close_seconds")
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


def _spread_bucket(row: dict[str, Any]) -> str:
    value = row.get("entry_exit_quality_spread_bps")
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


def _liquidity_bucket(row: dict[str, Any]) -> str:
    value = row.get("entry_exit_quality_liquidity_ratio")
    if value is None:
        value = row.get("entry_exit_quality_queue_fill")
    if value is None:
        return "unknown"
    ratio = float(value)
    if ratio < 0.50:
        return "<0.50"
    if ratio < 0.65:
        return "0.50-0.65"
    if ratio < 0.80:
        return "0.65-0.80"
    return ">=0.80"


def _staleness_bucket(row: dict[str, Any]) -> str:
    value = row.get("entry_exit_quality_book_staleness_ms")
    if value is None:
        return "unknown"
    staleness = float(value)
    if staleness < 1000:
        return "<1s"
    if staleness < 5000:
        return "1-5s"
    if staleness < 10000:
        return "5-10s"
    return ">=10s"


def _microstructure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entry_ask": _numeric_summary(rows, "entry_quality_ask"),
        "exit_bid": _numeric_summary(rows, "exit_quality_bid"),
        "spread_bps": _numeric_summary(rows, "entry_exit_quality_spread_bps"),
        "queue_fill": _numeric_summary(rows, "entry_exit_quality_queue_fill"),
        "book_staleness_ms": _numeric_summary(
            rows,
            "entry_exit_quality_book_staleness_ms",
        ),
        "time_to_close_seconds": _numeric_summary(
            rows,
            "entry_exit_quality_time_to_close_seconds",
        ),
        "field_availability": {
            field: sum(1 for row in rows if row.get(field) is not None)
            for field in (
                "entry_quality_ask",
                "exit_quality_bid",
                "entry_exit_quality_spread_bps",
                "entry_exit_quality_queue_fill",
                "entry_exit_quality_book_staleness_ms",
                "entry_exit_quality_time_to_close_seconds",
            )
        },
    }


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"available_count": 0, "minimum": None, "median": None, "mean": None}
    return {
        "available_count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
    }


def _side_comparison(
    up_rows: list[dict[str, Any]],
    down_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "UP": _side_metrics(up_rows),
        "DOWN": _side_metrics(down_rows),
    }


def _side_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_label = [row for row in rows if _label(row) > 0.0]
    positive_label_negative = [
        row for row in rows if _label(row) > 0.0 and _pnl(row) < 0.0
    ]
    negative_label = [row for row in rows if _label(row) < 0.0]
    first_exit_negative = [
        row
        for row in rows
        if _pnl(row) < 0.0
        and _has_reason(row, "closed_before_settlement_with_negative_replay_pnl")
    ]
    return {
        "known_replay_entry_count": len(rows),
        "label_target_sum": _sum_labels(rows),
        "replay_pnl_sum": _sum_pnl(rows),
        "label_vs_replay_gap": _sum_labels(rows) - _sum_pnl(rows),
        "score_vs_replay_pnl_correlation": _pearson(
            [_score(row) for row in rows],
            [_pnl(row) for row in rows],
        ),
        "positive_label_replay_negative_count": len(positive_label_negative),
        "positive_label_replay_negative_rate": (
            len(positive_label_negative) / len(positive_label)
            if positive_label
            else 0.0
        ),
        "negative_label_selected_count": len(negative_label),
        "negative_label_selected_rate": len(negative_label) / len(rows) if rows else 0.0,
        "first_executable_exit_negative_count": len(first_exit_negative),
        "first_executable_exit_negative_rate": (
            len(first_exit_negative) / len(rows) if rows else 0.0
        ),
        "microstructure_summary": _microstructure_summary(rows),
    }


def _score_bucket_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_score_bucket(_score(row))].append(row)
    summaries = []
    for bucket, bucket_rows in sorted(groups.items()):
        positives = [row for row in bucket_rows if _pnl(row) > 0.0]
        negatives = [row for row in bucket_rows if _pnl(row) < 0.0]
        summaries.append(
            {
                "score_bucket": bucket,
                "row_count": len(bucket_rows),
                "positive_replay_count": len(positives),
                "negative_replay_count": len(negatives),
                "label_target_mean": statistics.mean(_label(row) for row in bucket_rows),
                "realized_replay_pnl_mean": statistics.mean(
                    _pnl(row) for row in bucket_rows
                ),
                "realized_replay_pnl_sum": _sum_pnl(bucket_rows),
            }
        )
    return summaries


def _score_calibration_error_by_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for bucket in _score_bucket_summaries(rows):
        result.append(
            {
                "score_bucket": bucket["score_bucket"],
                "row_count": bucket["row_count"],
                "label_target_mean": bucket["label_target_mean"],
                "realized_replay_pnl_mean": bucket["realized_replay_pnl_mean"],
                "calibration_error": bucket["label_target_mean"]
                - bucket["realized_replay_pnl_mean"],
            }
        )
    return result


def _score_bucket(score: float) -> str:
    if score < 0.0:
        return "<0"
    if score < 0.25:
        return "0.00-0.25"
    if score < 0.50:
        return "0.25-0.50"
    if score < 0.75:
        return "0.50-0.75"
    if score < 1.00:
        return "0.75-1.00"
    return ">=1.00"


def _top_false_positives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if _pnl(row) < 0.0 and (_label(row) > 0.0 or _score(row) > 0.0)
    ]
    candidates.sort(key=lambda row: (_pnl(row), -_score(row), int(row.get("decision_ts") or 0)))
    return [_compact_row(row) for row in candidates[:10]]


def _top_false_negatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if _pnl(row) > 0.0 and (_label(row) <= 0.0 or _score(row) <= 0.0)
    ]
    candidates.sort(key=lambda row: (-_pnl(row), _score(row), int(row.get("decision_ts") or 0)))
    return [_compact_row(row) for row in candidates[:10]]


def _root_cause_codes(
    *,
    up_replay_rows: list[dict[str, Any]],
    positive_label_negative_replay: list[dict[str, Any]],
    negative_label_selected: list[dict[str, Any]],
    first_executable_negative: list[dict[str, Any]],
    score_correlation: float | None,
) -> list[str]:
    codes = []
    if _sum_labels(up_replay_rows) > 0.0 and _sum_pnl(up_replay_rows) < 0.0:
        codes.append("up_label_target_optimistic")
    if score_correlation is not None and score_correlation < 0.0:
        codes.append("up_action_value_overcalibrated")
    if positive_label_negative_replay or negative_label_selected:
        codes.append("up_rank_score_false_positive_bias")
    if first_executable_negative:
        codes.append("up_executable_exit_path_mismatch")
    if _microstructure_weakness_detected(up_replay_rows):
        codes.append("up_liquidity_spread_staleness_regime_weakness")
    if len(codes) >= 2:
        codes.append("mixed")
    return list(dict.fromkeys(codes or ["up_market_regime_specific_weakness"]))


def _microstructure_weakness_detected(rows: list[dict[str, Any]]) -> bool:
    negative_rows = [row for row in rows if _pnl(row) < 0.0]
    if not negative_rows:
        return False
    spread_values = [
        float(row["entry_exit_quality_spread_bps"])
        for row in negative_rows
        if row.get("entry_exit_quality_spread_bps") is not None
    ]
    queue_values = [
        float(row["entry_exit_quality_queue_fill"])
        for row in negative_rows
        if row.get("entry_exit_quality_queue_fill") is not None
    ]
    return bool(
        spread_values
        and statistics.mean(spread_values) >= 600.0
        or queue_values
        and statistics.mean(queue_values) < 0.65
    )


def _classification(codes: list[str]) -> str:
    return "mixed" if "mixed" in codes else codes[0]


def _recommended_actions(codes: list[str]) -> list[str]:
    actions = ["continue_diagnostics", "reject_up_sell_before_close_promotion_for_now"]
    if "up_action_value_overcalibrated" in codes:
        actions.append("introduce_side_specific_calibration")
    if "up_rank_score_false_positive_bias" in codes:
        actions.append("introduce_up_replay_gap_penalty")
        actions.append("introduce_up_high_score_negative_replay_guard")
    if "up_executable_exit_path_mismatch" in codes or "up_label_target_optimistic" in codes:
        actions.append("introduce_up_executable_exit_label_correction")
    return list(dict.fromkeys(actions))


def _has_reason(row: dict[str, Any], reason: str) -> bool:
    return reason in _reason_tokens(row)


def _reason_tokens(row: dict[str, Any]) -> list[str]:
    tokens = []
    for field in ("exit_reason_codes", "replay_reason_codes", "attrition_reason_codes"):
        value = row.get(field) or []
        if isinstance(value, list):
            tokens.extend(str(item) for item in value)
        else:
            tokens.append(str(value))
    return tokens


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _label_replay_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UP SELL_BEFORE_CLOSE Label-Replay Alignment",
        "",
        f"- root_cause_classification: `{report['root_cause_classification']}`",
        f"- root_cause_codes: `{', '.join(report['root_cause_codes'])}`",
        f"- up_selected_entry_count: `{report['up_selected_entry_count']}`",
        f"- up_known_replay_entry_count: `{report['up_known_replay_entry_count']}`",
        f"- up_label_target_sum: `{report['up_label_target_sum']}`",
        f"- up_replay_pnl_sum: `{report['up_replay_pnl_sum']}`",
        f"- up_label_vs_replay_gap: `{report['up_label_vs_replay_gap']}`",
        "- up_positive_label_replay_negative_count: "
        f"`{report['up_positive_label_replay_negative_count']}`",
        f"- up_negative_label_selected_count: `{report['up_negative_label_selected_count']}`",
        f"- first_executable_exit_negative_count: `{report['first_executable_exit_negative_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Recommended Actions",
        "",
        *[f"- `{action}`" for action in report["recommended_next_actions"]],
        "",
        "## Safety",
        "",
        "- diagnostic-only; no paper/live unlock",
        "- paper_only: true",
        "- capital_at_risk: false",
        "",
    ]
    return "\n".join(lines)


def _calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UP SELL_BEFORE_CLOSE Action-Value Calibration Diagnostic",
        "",
        f"- root_cause_classification: `{report['root_cause_classification']}`",
        "- calibrated_action_score_vs_realized_up_replay_pnl_correlation: "
        f"`{report['calibrated_action_score_vs_realized_up_replay_pnl_correlation']}`",
        "- rank_score_vs_realized_up_replay_pnl_correlation: "
        f"`{report['rank_score_vs_realized_up_replay_pnl_correlation']}`",
        "- action_target_vs_calibrated_score_correlation: "
        f"`{report['action_target_vs_calibrated_score_correlation']}`",
        f"- high_score_negative_replay_up_count: `{report['high_score_negative_replay_up_count']}`",
        f"- low_score_positive_replay_up_count: `{report['low_score_positive_replay_up_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Score Buckets",
        "",
        "| bucket | rows | pos | neg | pnl_sum |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["score_buckets"]:
        lines.append(
            "| {bucket} | {rows} | {pos} | {neg} | {pnl:.6f} |".format(
                bucket=row["score_bucket"],
                rows=row["row_count"],
                pos=row["positive_replay_count"],
                neg=row["negative_replay_count"],
                pnl=float(row["realized_replay_pnl_sum"]),
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
