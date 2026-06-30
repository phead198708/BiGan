"""Diagnostic N2 non-leaky UP feature-proxy candidate reports."""

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
from bigan.v8.polymarket.training.post_freeze_n_up_replay_aligned import (
    N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION,
    _overlay_pnl_sum,
    _score_bucket,
)
from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (
    _compact_row,
    _enriched_selected_rows,
    _label,
    _pearson,
    _pnl,
    _read_json,
    _sha256_file,
    _sum_labels,
    _sum_pnl,
    _write_json,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

N2_NON_LEAKY_UP_FEATURE_PROXY_CANDIDATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-n2-non-leaky-up-feature-proxy-candidate-v1"
)
N2_NON_LEAKY_UP_FEATURE_PROXY_SCORE_OVERLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-n2-non-leaky-up-feature-proxy-score-overlay-v1"
)
N2_ALLOWED_DECISION_INPUT_FIELDS = (
    "raw_calibrated_action_score",
    "best_action_margin",
    "execution_pnl_immediate_exit_pnl",
    "execution_pnl_immediate_exit_return",
    "entry_quality_ask",
    "exit_quality_bid",
    "entry_exit_quality_score",
    "entry_exit_quality_edge",
    "entry_exit_quality_liquidity_ratio",
    "entry_exit_quality_queue_fill",
    "entry_exit_quality_spread_bps",
    "entry_exit_quality_book_staleness_ms",
    "entry_exit_quality_time_to_close_seconds",
    "entry_exit_quality_recent_book_update_count_1m",
    "recent_book_update_count_1m",
    "up_recent_book_update_count_1m",
)
N2_FORBIDDEN_SELECTION_FIELDS = (
    "action_return_target",
    "label_pnl_target",
    "realized_trade_pnl",
    "settlement_pnl",
    "total_polymarket_pnl",
    "first_executable_exit_replay_pnl_proxy",
    "closed_before_settlement",
    "exit_reason_codes",
    "replay_reason_codes",
    "attrition_reason_codes",
    "attrition_stage",
)


@dataclass(frozen=True, slots=True)
class PolymarketN2UpFeatureProxyConfig:
    """Configuration for diagnostic-only N2 non-leaky UP proxy reports."""

    m2_candidate_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_n2_non_leaky_up_feature_proxy"
    n_candidate_report_path: Path | str | None = None
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
        if self.n_candidate_report_path is not None and not isinstance(
            self.n_candidate_report_path,
            Path,
        ):
            object.__setattr__(
                self,
                "n_candidate_report_path",
                Path(self.n_candidate_report_path),
            )
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id


@dataclass(frozen=True, slots=True)
class PolymarketN2UpFeatureProxyResult:
    run_dir: Path
    candidate_report: dict[str, Any]
    score_overlay_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_n2_up_feature_proxy_candidate(
    config: PolymarketN2UpFeatureProxyConfig,
) -> PolymarketN2UpFeatureProxyResult:
    """Build diagnostic-only N2 non-leaky UP feature-proxy reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "candidate_report": run_dir
        / "n2_non_leaky_up_feature_proxy_candidate_report.json",
        "candidate_summary": run_dir
        / "n2_non_leaky_up_feature_proxy_candidate_report.md",
        "score_overlay_report": run_dir
        / "n2_non_leaky_up_feature_proxy_score_overlay_report.json",
        "score_overlay_summary": run_dir
        / "n2_non_leaky_up_feature_proxy_score_overlay_report.md",
        "manifest": run_dir / "n2_non_leaky_up_feature_proxy_manifest.json",
    }
    candidate_report, score_overlay_report = _build_reports(config=config)
    _write_json(artifact_paths["candidate_report"], candidate_report)
    artifact_paths["candidate_summary"].write_text(
        _candidate_markdown(candidate_report),
        encoding="utf-8",
    )
    _write_json(artifact_paths["score_overlay_report"], score_overlay_report)
    artifact_paths["score_overlay_summary"].write_text(
        _score_overlay_markdown(score_overlay_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-n2-non-leaky-up-feature-proxy-artifacts-v1",
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
    return PolymarketN2UpFeatureProxyResult(
        run_dir=run_dir,
        candidate_report=candidate_report,
        score_overlay_report=score_overlay_report,
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketN2UpFeatureProxyConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m2_report_path = config.m2_candidate_report_path.expanduser().resolve()
    m2_report = _read_json(m2_report_path)
    if m2_report.get("schema_version") != M2_REPLAY_PARITY_SCHEMA_VERSION:
        raise ValueError("not an M2 replay-parity candidate report")
    n_report = _load_n_report(config.n_candidate_report_path)
    enriched_rows = _enriched_selected_rows(m2_report)
    up_rows = [row for row in enriched_rows if row.get("selected_side") == "UP"]
    down_rows = [row for row in enriched_rows if row.get("selected_side") == "DOWN"]
    overlay_rows = [_overlay_row(row, config) for row in up_rows]
    n2_selected = [row for row in overlay_rows if row["n2_would_select"]]
    n2_blocked = [row for row in overlay_rows if not row["n2_would_select"]]
    candidate_report = _candidate_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        n_report=n_report,
        overlay_rows=overlay_rows,
        n2_selected=n2_selected,
        n2_blocked=n2_blocked,
        down_rows=down_rows,
    )
    score_overlay_report = _score_overlay_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        n_report=n_report,
        overlay_rows=overlay_rows,
        n2_selected=n2_selected,
    )
    return candidate_report, score_overlay_report


def _load_n_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    report = _read_json(resolved)
    if report.get("schema_version") != N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("not an N replay-aligned candidate report")
    return {**report, "_report_path": str(resolved), "_report_sha256": _sha256_file(resolved)}


def _overlay_row(
    row: dict[str, Any],
    config: PolymarketN2UpFeatureProxyConfig,
) -> dict[str, Any]:
    decision = _decision_time_proxy(row, config)
    used_fields = decision["n2_decision_input_fields_used"]
    _assert_non_leaky_inputs(used_fields)

    label_target = _label(row)
    replay_pnl = _pnl(row)
    compact = _compact_row(row)
    positive_label_negative = label_target > 0.0 and replay_pnl < 0.0
    negative_label = label_target < 0.0
    return {
        **compact,
        **decision,
        "original_label_target": label_target,
        "realized_replay_pnl": replay_pnl,
        "label_vs_replay_gap_before": label_target - replay_pnl,
        "feature_proxy_label_vs_replay_gap_after": (
            decision["n2_feature_exit_label_proxy"] - replay_pnl
        ),
        "positive_label_replay_negative_flagged_for_evaluation": (
            positive_label_negative
        ),
        "negative_label_selected_flagged_for_evaluation": negative_label,
        "n2_forbidden_fields_present_for_evaluation_only": sorted(
            field for field in N2_FORBIDDEN_SELECTION_FIELDS if field in row
        ),
        "n2_forbidden_fields_used_for_selection": [],
    }


def _decision_time_proxy(
    row: dict[str, Any],
    config: PolymarketN2UpFeatureProxyConfig,
) -> dict[str, Any]:
    used_fields: list[str] = []

    score = _decision_float(row, "raw_calibrated_action_score", used_fields, 0.0)
    best_action_margin = _decision_float(row, "best_action_margin", used_fields, 0.0)
    entry_ask = _decision_float(row, "entry_quality_ask", used_fields, None)
    exit_bid = _decision_float(row, "exit_quality_bid", used_fields, None)
    immediate_exit_pnl = _decision_float(
        row,
        "execution_pnl_immediate_exit_pnl",
        used_fields,
        None,
    )
    if immediate_exit_pnl is None and entry_ask is not None and exit_bid is not None:
        immediate_exit_pnl = exit_bid - entry_ask
    if immediate_exit_pnl is None:
        immediate_exit_pnl = 0.0

    spread_bps = _decision_float(
        row,
        "entry_exit_quality_spread_bps",
        used_fields,
        None,
    )
    queue_fill = _decision_float(
        row,
        "entry_exit_quality_queue_fill",
        used_fields,
        None,
    )
    staleness_ms = _decision_float(
        row,
        "entry_exit_quality_book_staleness_ms",
        used_fields,
        None,
    )
    time_to_close = _decision_float(
        row,
        "entry_exit_quality_time_to_close_seconds",
        used_fields,
        None,
    )
    recent_updates = _decision_recent_book_updates(row, used_fields)

    score_bucket = _score_bucket(score)
    risk_bucket, risk_score = _up_risk_bucket(
        score=score,
        immediate_exit_pnl=immediate_exit_pnl,
        spread_bps=spread_bps,
        queue_fill=queue_fill,
        staleness_ms=staleness_ms,
        time_to_close=time_to_close,
        config=config,
    )
    spread_penalty = (
        max(0.0, (spread_bps - config.max_spread_bps) / config.max_spread_bps)
        if spread_bps is not None
        else 0.0
    )
    queue_penalty = (
        max(0.0, config.min_queue_fill - queue_fill)
        if queue_fill is not None
        else 0.0
    )
    staleness_penalty = (
        max(0.0, (staleness_ms - config.max_book_staleness_ms) / config.max_book_staleness_ms)
        if staleness_ms is not None
        else 0.0
    )
    time_penalty = (
        max(0.0, (config.min_time_to_close_seconds - time_to_close) / config.min_time_to_close_seconds)
        if time_to_close is not None
        else 0.0
    )
    recent_update_penalty = (
        max(
            0.0,
            (config.min_recent_book_update_count_1m - recent_updates)
            / config.min_recent_book_update_count_1m,
        )
        if recent_updates is not None
        else 0.0
    )
    feature_proxy_score = (
        score
        + best_action_margin
        + (2.0 * immediate_exit_pnl)
        - risk_score
        - spread_penalty
        - queue_penalty
        - staleness_penalty
        - time_penalty
        - recent_update_penalty
    )
    block_reasons = []
    if immediate_exit_pnl <= config.min_immediate_exit_pnl_proxy:
        block_reasons.append("n2_blocked_nonpositive_immediate_exit_proxy")
    if spread_bps is not None and spread_bps > config.max_spread_bps:
        block_reasons.append("n2_blocked_spread_too_wide")
    if queue_fill is not None and queue_fill < config.min_queue_fill:
        block_reasons.append("n2_blocked_queue_fill_too_low")
    if staleness_ms is not None and staleness_ms > config.max_book_staleness_ms:
        block_reasons.append("n2_blocked_book_staleness_too_high")
    if time_to_close is not None and time_to_close < config.min_time_to_close_seconds:
        block_reasons.append("n2_blocked_time_to_close_too_short")
    if (
        recent_updates is not None
        and recent_updates < config.min_recent_book_update_count_1m
    ):
        block_reasons.append("n2_blocked_recent_book_update_count_too_low")
    if risk_bucket == "up_high_score_adverse_microstructure":
        block_reasons.append("n2_blocked_up_high_score_adverse_microstructure")
    if feature_proxy_score <= config.min_feature_proxy_score:
        block_reasons.append("n2_blocked_feature_proxy_score_below_threshold")

    n2_would_select = not block_reasons
    return {
        "original_calibrated_action_score": score,
        "n2_calibrated_score_bucket": score_bucket,
        "n2_side_specific_up_risk_bucket": risk_bucket,
        "n2_side_specific_up_risk_score": risk_score,
        "n2_immediate_exit_pnl_proxy": immediate_exit_pnl,
        "n2_feature_exit_label_proxy": immediate_exit_pnl,
        "n2_entry_ask": entry_ask,
        "n2_executable_exit_bid": exit_bid,
        "n2_spread_bps": spread_bps,
        "n2_queue_fill_proxy": queue_fill,
        "n2_book_staleness_ms": staleness_ms,
        "n2_time_to_close_seconds": time_to_close,
        "n2_recent_book_update_count_1m": recent_updates,
        "n2_feature_proxy_penalty_components": {
            "up_side_risk_score": risk_score,
            "spread_penalty": spread_penalty,
            "queue_penalty": queue_penalty,
            "staleness_penalty": staleness_penalty,
            "time_to_close_penalty": time_penalty,
            "recent_book_update_penalty": recent_update_penalty,
        },
        "n2_replay_aligned_feature_score_proxy": feature_proxy_score,
        "n2_score_delta": feature_proxy_score - score,
        "n2_would_select": n2_would_select,
        "n2_would_block": not n2_would_select,
        "n2_decision_reason_codes": block_reasons
        if block_reasons
        else ["n2_non_leaky_feature_proxy_would_select"],
        "n2_decision_input_fields_used": sorted(set(used_fields)),
        "n2_selection_uses_only_allowed_fields": True,
    }


def _decision_float(
    row: dict[str, Any],
    field: str,
    used_fields: list[str],
    default: float | None,
) -> float | None:
    if field in row and row.get(field) is not None:
        used_fields.append(field)
        return float(row[field])
    return default


def _decision_recent_book_updates(
    row: dict[str, Any],
    used_fields: list[str],
) -> float | None:
    for field in (
        "entry_exit_quality_recent_book_update_count_1m",
        "recent_book_update_count_1m",
        "up_recent_book_update_count_1m",
    ):
        value = _decision_float(row, field, used_fields, None)
        if value is not None:
            return value
    return None


def _up_risk_bucket(
    *,
    score: float,
    immediate_exit_pnl: float,
    spread_bps: float | None,
    queue_fill: float | None,
    staleness_ms: float | None,
    time_to_close: float | None,
    config: PolymarketN2UpFeatureProxyConfig,
) -> tuple[str, float]:
    adverse = 0
    if immediate_exit_pnl <= config.min_immediate_exit_pnl_proxy:
        adverse += 1
    if spread_bps is not None and spread_bps > config.max_spread_bps:
        adverse += 1
    if queue_fill is not None and queue_fill < config.min_queue_fill:
        adverse += 1
    if staleness_ms is not None and staleness_ms > config.max_book_staleness_ms:
        adverse += 1
    if time_to_close is not None and time_to_close < config.min_time_to_close_seconds:
        adverse += 1
    if score >= config.high_score_risk_threshold and adverse:
        return "up_high_score_adverse_microstructure", 0.35 + 0.10 * adverse
    if adverse >= 2:
        return "up_adverse_microstructure", 0.20 + 0.05 * adverse
    if score >= config.high_score_risk_threshold:
        return "up_high_score_clean_microstructure", 0.05
    return "up_baseline_feature_proxy_risk", 0.0


def _assert_non_leaky_inputs(fields: list[str]) -> None:
    forbidden = sorted(set(fields).intersection(N2_FORBIDDEN_SELECTION_FIELDS))
    if forbidden:
        raise ValueError(f"N2 selection used forbidden fields: {forbidden}")
    unexpected = sorted(set(fields).difference(N2_ALLOWED_DECISION_INPUT_FIELDS))
    if unexpected:
        raise ValueError(f"N2 selection used non-allowlisted fields: {unexpected}")


def _candidate_report(
    *,
    config: PolymarketN2UpFeatureProxyConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    n_report: dict[str, Any] | None,
    overlay_rows: list[dict[str, Any]],
    n2_selected: list[dict[str, Any]],
    n2_blocked: list[dict[str, Any]],
    down_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [row for row in n2_selected if row["realized_replay_pnl"] > 0.0]
    negative = [row for row in n2_selected if row["realized_replay_pnl"] < 0.0]
    m2_positive = [row for row in overlay_rows if row["realized_replay_pnl"] > 0.0]
    m2_negative = [row for row in overlay_rows if row["realized_replay_pnl"] < 0.0]
    report = {
        "schema_version": N2_NON_LEAKY_UP_FEATURE_PROXY_CANDIDATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
        ],
        "report_type": "n2_non_leaky_up_feature_proxy_candidate",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "n_reference_report": _n_reference_summary(n_report),
        "current_m_m2_n_status": {
            "current_frozen_m_promotion_status": m2_report.get(
                "current_frozen_m_promotion_status",
                "reject_promotion_for_now",
            ),
            "m2_promotion_ready": False,
            "n_promotion_ready": False,
            "n2_current_m_m2_n_evidence_reused_for_promotion": False,
        },
        "n2_allowed_decision_input_fields": list(N2_ALLOWED_DECISION_INPUT_FIELDS),
        "n2_forbidden_selection_fields": list(N2_FORBIDDEN_SELECTION_FIELDS),
        "n2_selection_uses_only_allowed_fields": _all_rows_non_leaky(overlay_rows),
        "n2_overlay_parameters": {
            "min_feature_proxy_score": config.min_feature_proxy_score,
            "min_immediate_exit_pnl_proxy": config.min_immediate_exit_pnl_proxy,
            "max_spread_bps": config.max_spread_bps,
            "min_queue_fill": config.min_queue_fill,
            "max_book_staleness_ms": config.max_book_staleness_ms,
            "min_time_to_close_seconds": config.min_time_to_close_seconds,
            "min_recent_book_update_count_1m": (
                config.min_recent_book_update_count_1m
            ),
            "high_score_risk_threshold": config.high_score_risk_threshold,
        },
        "original_up_selected_rows": overlay_rows,
        "n2_would_selected_rows": n2_selected,
        "n2_would_blocked_rows": n2_blocked,
        "m2_up_selected_count": len(overlay_rows),
        "m2_up_replay_pnl_sum": _overlay_pnl_sum(overlay_rows),
        "m2_up_positive_replay_count": len(m2_positive),
        "m2_up_negative_replay_count": len(m2_negative),
        "m2_up_label_vs_replay_gap": _overlay_gap_before(overlay_rows),
        "n2_would_selected_up_count": len(n2_selected),
        "n2_would_blocked_up_count": len(n2_blocked),
        "n2_would_selected_up_replay_pnl_sum": _overlay_pnl_sum(n2_selected),
        "n2_would_selected_up_positive_replay_count": len(positive),
        "n2_would_selected_up_negative_replay_count": len(negative),
        "n2_selected_label_vs_replay_gap_after_feature_proxy": _overlay_gap_after(
            n2_selected
        ),
        "n2_all_rows_label_vs_replay_gap_after_feature_proxy": _overlay_gap_after(
            overlay_rows
        ),
        "n2_false_positive_reduction_count": _false_positive_reduction_count(
            n2_blocked
        ),
        "n2_blocked_up_false_positive_count": _false_positive_reduction_count(
            n2_blocked
        ),
        "n2_block_reason_counts": _reason_counts(n2_blocked),
        "top_rows_changed_by_n2_overlay": _top_changed_rows(overlay_rows),
        "n2_vs_m2_n_comparison": {
            "m2": {
                "up_selected_count": len(overlay_rows),
                "up_replay_pnl_sum": _overlay_pnl_sum(overlay_rows),
                "up_positive_replay_count": len(m2_positive),
                "up_negative_replay_count": len(m2_negative),
                "up_label_vs_replay_gap": _overlay_gap_before(overlay_rows),
            },
            "n": _n_comparison(n_report),
            "n2": {
                "would_selected_up_count": len(n2_selected),
                "would_blocked_up_count": len(n2_blocked),
                "would_selected_up_replay_pnl_sum": _overlay_pnl_sum(n2_selected),
                "would_selected_up_positive_replay_count": len(positive),
                "would_selected_up_negative_replay_count": len(negative),
                "label_vs_replay_gap_after_feature_proxy": _overlay_gap_after(
                    n2_selected
                ),
                "false_positive_reduction_count": _false_positive_reduction_count(
                    n2_blocked
                ),
            },
        },
        "down_side_reference": {
            "m2_down_selected_count": len(down_rows),
            "m2_down_replay_pnl_sum": _sum_pnl(down_rows),
            "m2_down_label_vs_replay_gap": _sum_labels(down_rows)
            - _sum_pnl(down_rows),
        },
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["n2_non_leaky_up_feature_proxy_candidate_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _score_overlay_report(
    *,
    config: PolymarketN2UpFeatureProxyConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    n_report: dict[str, Any] | None,
    overlay_rows: list[dict[str, Any]],
    n2_selected: list[dict[str, Any]],
) -> dict[str, Any]:
    del config
    score_before = [row["original_calibrated_action_score"] for row in overlay_rows]
    score_after = [row["n2_replay_aligned_feature_score_proxy"] for row in overlay_rows]
    pnl = [row["realized_replay_pnl"] for row in overlay_rows]
    report = {
        "schema_version": N2_NON_LEAKY_UP_FEATURE_PROXY_SCORE_OVERLAY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
        ],
        "report_type": "n2_non_leaky_up_feature_proxy_score_overlay",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "n_reference_report": _n_reference_summary(n_report),
        "n2_allowed_decision_input_fields": list(N2_ALLOWED_DECISION_INPUT_FIELDS),
        "n2_forbidden_selection_fields": list(N2_FORBIDDEN_SELECTION_FIELDS),
        "n2_selection_uses_only_allowed_fields": _all_rows_non_leaky(overlay_rows),
        "original_score_vs_replay_correlation": _pearson(score_before, pnl),
        "n2_feature_proxy_score_vs_replay_correlation": _pearson(score_after, pnl),
        "score_correlation_delta": _correlation_delta(score_before, score_after, pnl),
        "label_vs_replay_gap_before_overlay": _overlay_gap_before(overlay_rows),
        "label_vs_replay_gap_after_feature_proxy_overlay": _overlay_gap_after(
            overlay_rows
        ),
        "label_vs_replay_gap_delta": _overlay_gap_before(overlay_rows)
        - _overlay_gap_after(overlay_rows),
        "score_overlay_rows": overlay_rows,
        "score_overlay_bucket_comparison": _score_bucket_comparison(overlay_rows),
        "top_rows_changed_by_n2_overlay": _top_changed_rows(overlay_rows),
        "n2_selected_score_summary": _score_summary(n2_selected),
        "n2_blocked_score_summary": _score_summary(
            [row for row in overlay_rows if not row["n2_would_select"]]
        ),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["n2_non_leaky_up_feature_proxy_score_overlay_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _n_reference_summary(n_report: dict[str, Any] | None) -> dict[str, Any]:
    if n_report is None:
        return {
            "provided": False,
            "candidate_name": SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
        }
    return {
        "provided": True,
        "candidate_name": n_report.get("candidate_name"),
        "report_path": n_report.get("_report_path"),
        "report_sha256": n_report.get("_report_sha256"),
        "n_would_selected_up_count": n_report.get("n_would_selected_up_count"),
        "n_would_blocked_up_count": n_report.get("n_would_blocked_up_count"),
        "n_would_selected_up_replay_pnl_sum": n_report.get(
            "n_would_selected_up_replay_pnl_sum"
        ),
        "n_blocked_up_false_positive_count": n_report.get(
            "n_blocked_up_false_positive_count"
        ),
    }


def _n_comparison(n_report: dict[str, Any] | None) -> dict[str, Any]:
    summary = _n_reference_summary(n_report)
    return {
        key: value
        for key, value in summary.items()
        if key
        in {
            "provided",
            "n_would_selected_up_count",
            "n_would_blocked_up_count",
            "n_would_selected_up_replay_pnl_sum",
            "n_blocked_up_false_positive_count",
        }
    }


def _all_rows_non_leaky(rows: list[dict[str, Any]]) -> bool:
    return all(
        not set(row["n2_decision_input_fields_used"]).intersection(
            N2_FORBIDDEN_SELECTION_FIELDS
        )
        and not row["n2_forbidden_fields_used_for_selection"]
        for row in rows
    )


def _overlay_gap_before(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["label_vs_replay_gap_before"]) for row in rows)


def _overlay_gap_after(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["feature_proxy_label_vs_replay_gap_after"]) for row in rows)


def _false_positive_reduction_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row["original_label_target"] > 0.0 and row["realized_replay_pnl"] < 0.0
    )


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["n2_decision_reason_codes"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _top_changed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row["n2_score_delta"])),
            abs(float(row["label_vs_replay_gap_before"]))
            - abs(float(row["feature_proxy_label_vs_replay_gap_after"])),
        ),
        reverse=True,
    )
    return [_compact_overlay_row(row) for row in ranked[:10]]


def _compact_overlay_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "market_id",
        "slug",
        "decision_ts",
        "selected_side",
        "action",
        "original_label_target",
        "n2_feature_exit_label_proxy",
        "realized_replay_pnl",
        "label_vs_replay_gap_before",
        "feature_proxy_label_vs_replay_gap_after",
        "original_calibrated_action_score",
        "n2_replay_aligned_feature_score_proxy",
        "n2_immediate_exit_pnl_proxy",
        "n2_side_specific_up_risk_bucket",
        "n2_would_select",
        "n2_decision_reason_codes",
        "n2_decision_input_fields_used",
    )
    return {field: row.get(field) for field in fields}


def _score_bucket_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = str(row["n2_calibrated_score_bucket"])
        groups.setdefault(bucket, []).append(row)
    result = []
    for bucket, bucket_rows in sorted(groups.items()):
        selected = [row for row in bucket_rows if row["n2_would_select"]]
        blocked = [row for row in bucket_rows if not row["n2_would_select"]]
        result.append(
            {
                "original_score_bucket": bucket,
                "row_count": len(bucket_rows),
                "n2_would_selected_count": len(selected),
                "n2_would_blocked_count": len(blocked),
                "original_replay_pnl_sum": _overlay_pnl_sum(bucket_rows),
                "n2_selected_replay_pnl_sum": _overlay_pnl_sum(selected),
                "label_vs_replay_gap_before": _overlay_gap_before(bucket_rows),
                "label_vs_replay_gap_after_feature_proxy": _overlay_gap_after(
                    bucket_rows
                ),
            }
        )
    return result


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "score_mean": None,
            "feature_proxy_score_mean": None,
            "replay_pnl_sum": 0.0,
        }
    return {
        "row_count": len(rows),
        "score_mean": statistics.mean(
            float(row["original_calibrated_action_score"]) for row in rows
        ),
        "feature_proxy_score_mean": statistics.mean(
            float(row["n2_replay_aligned_feature_score_proxy"]) for row in rows
        ),
        "replay_pnl_sum": _overlay_pnl_sum(rows),
    }


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


def _candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# N2 Non-Leaky UP Feature-Proxy Candidate Report",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        "- n2_selection_uses_only_allowed_fields: "
        f"`{str(report['n2_selection_uses_only_allowed_fields']).lower()}`",
        f"- m2_up_selected_count: `{report['m2_up_selected_count']}`",
        f"- m2_up_replay_pnl_sum: `{report['m2_up_replay_pnl_sum']}`",
        f"- m2_up_label_vs_replay_gap: `{report['m2_up_label_vs_replay_gap']}`",
        f"- n2_would_selected_up_count: `{report['n2_would_selected_up_count']}`",
        f"- n2_would_blocked_up_count: `{report['n2_would_blocked_up_count']}`",
        "- n2_would_selected_up_replay_pnl_sum: "
        f"`{report['n2_would_selected_up_replay_pnl_sum']}`",
        "- n2_selected_label_vs_replay_gap_after_feature_proxy: "
        f"`{report['n2_selected_label_vs_replay_gap_after_feature_proxy']}`",
        f"- n2_blocked_up_false_positive_count: `{report['n2_blocked_up_false_positive_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Block Reasons",
        "",
    ]
    for reason, count in report["n2_block_reason_counts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Non-Leaky Contract",
            "",
            "- Selection fields are restricted to the allowlist in JSON.",
            "- Forbidden replay/label fields are reported only for evaluation.",
            "",
            "## Safety",
            "",
            "- diagnostic-only; current M/M2/N evidence is not N2 promotion evidence",
            "- paper_only: true",
            "- capital_at_risk: false",
            "",
        ]
    )
    return "\n".join(lines)

def _score_overlay_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# N2 Non-Leaky UP Feature-Proxy Score Overlay",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        "- n2_selection_uses_only_allowed_fields: "
        f"`{str(report['n2_selection_uses_only_allowed_fields']).lower()}`",
        "- original_score_vs_replay_correlation: "
        f"`{report['original_score_vs_replay_correlation']}`",
        "- n2_feature_proxy_score_vs_replay_correlation: "
        f"`{report['n2_feature_proxy_score_vs_replay_correlation']}`",
        f"- score_correlation_delta: `{report['score_correlation_delta']}`",
        "- label_vs_replay_gap_before_overlay: "
        f"`{report['label_vs_replay_gap_before_overlay']}`",
        "- label_vs_replay_gap_after_feature_proxy_overlay: "
        f"`{report['label_vs_replay_gap_after_feature_proxy_overlay']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Score Buckets",
        "",
        "| bucket | rows | selected | blocked | original_pnl | selected_pnl |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["score_overlay_bucket_comparison"]:
        lines.append(
            "| {bucket} | {rows} | {selected} | {blocked} | {original:.6f} | {selected_pnl:.6f} |".format(
                bucket=row["original_score_bucket"],
                rows=row["row_count"],
                selected=row["n2_would_selected_count"],
                blocked=row["n2_would_blocked_count"],
                original=float(row["original_replay_pnl_sum"]),
                selected_pnl=float(row["n2_selected_replay_pnl_sum"]),
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
