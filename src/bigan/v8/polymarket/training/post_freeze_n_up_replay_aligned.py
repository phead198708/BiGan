"""Diagnostic N UP replay-aligned action-value candidate reports."""

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
from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (
    _compact_row,
    _enriched_selected_rows,
    _label,
    _pearson,
    _pnl,
    _rank,
    _read_json,
    _score,
    _sha256_file,
    _sum_labels,
    _sum_pnl,
    _write_json,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-n-up-replay-aligned-candidate-v1"
)
N_UP_REPLAY_ALIGNED_SCORE_OVERLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-n-up-replay-aligned-score-overlay-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketNUpReplayAlignedConfig:
    """Configuration for diagnostic-only N UP replay-aligned reports."""

    m2_candidate_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_n_up_replay_aligned_candidate"
    overwrite_existing: bool = False
    high_score_threshold: float = 0.80
    n_min_replay_aligned_score_proxy: float = 0.0
    n_min_corrected_label_proxy: float = 0.0
    high_score_negative_guard_penalty: float = 1.0
    negative_label_penalty: float = 0.25
    positive_label_negative_replay_penalty: float = 0.25
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


@dataclass(frozen=True, slots=True)
class PolymarketNUpReplayAlignedResult:
    run_dir: Path
    candidate_report: dict[str, Any]
    score_overlay_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_n_up_replay_aligned_candidate(
    config: PolymarketNUpReplayAlignedConfig,
) -> PolymarketNUpReplayAlignedResult:
    """Build diagnostic-only N UP replay-aligned candidate reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "candidate_report": run_dir / "n_up_replay_aligned_candidate_report.json",
        "candidate_summary": run_dir / "n_up_replay_aligned_candidate_report.md",
        "score_overlay_report": run_dir
        / "n_up_replay_aligned_score_overlay_report.json",
        "score_overlay_summary": run_dir
        / "n_up_replay_aligned_score_overlay_report.md",
        "manifest": run_dir / "n_up_replay_aligned_manifest.json",
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
        "schema_version": "bigan-v8-polymarket-n-up-replay-aligned-artifacts-v1",
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
    return PolymarketNUpReplayAlignedResult(
        run_dir=run_dir,
        candidate_report=candidate_report,
        score_overlay_report=score_overlay_report,
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketNUpReplayAlignedConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m2_report_path = config.m2_candidate_report_path.expanduser().resolve()
    m2_report = _read_json(m2_report_path)
    if m2_report.get("schema_version") != M2_REPLAY_PARITY_SCHEMA_VERSION:
        raise ValueError("not an M2 replay-parity candidate report")

    enriched_rows = _enriched_selected_rows(m2_report)
    up_rows = [row for row in enriched_rows if row.get("selected_side") == "UP"]
    down_rows = [row for row in enriched_rows if row.get("selected_side") == "DOWN"]
    overlay_rows = [_overlay_row(row, config) for row in up_rows]
    n_selected = [row for row in overlay_rows if row["n_would_select"]]
    n_blocked = [row for row in overlay_rows if not row["n_would_select"]]

    candidate_report = _candidate_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        overlay_rows=overlay_rows,
        n_selected=n_selected,
        n_blocked=n_blocked,
        down_rows=down_rows,
    )
    score_overlay_report = _score_overlay_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        overlay_rows=overlay_rows,
        n_selected=n_selected,
    )
    return candidate_report, score_overlay_report


def _overlay_row(
    row: dict[str, Any],
    config: PolymarketNUpReplayAlignedConfig,
) -> dict[str, Any]:
    label_target = _label(row)
    replay_pnl = _pnl(row)
    score = _score(row)
    executable_exit_proxy = _first_executable_exit_proxy(row)
    replay_gap_penalty = max(0.0, label_target - executable_exit_proxy)
    high_score_negative = (
        score >= config.high_score_threshold and replay_pnl < 0.0
    )
    positive_label_negative = label_target > 0.0 and replay_pnl < 0.0
    negative_label = label_target < 0.0
    penalty_components = {
        "replay_gap_penalty": replay_gap_penalty,
        "high_score_negative_guard_penalty": (
            config.high_score_negative_guard_penalty
            if high_score_negative
            else 0.0
        ),
        "negative_label_penalty": config.negative_label_penalty
        if negative_label
        else 0.0,
        "positive_label_negative_replay_penalty": (
            config.positive_label_negative_replay_penalty
            if positive_label_negative
            else 0.0
        ),
    }
    total_penalty = sum(penalty_components.values())
    score_proxy = score - total_penalty
    corrected_label_proxy = executable_exit_proxy
    block_reasons = []
    if high_score_negative:
        block_reasons.append("n_blocked_high_score_negative_replay")
    if positive_label_negative:
        block_reasons.append("n_flagged_positive_label_replay_negative")
    if negative_label:
        block_reasons.append("n_flagged_negative_label_selected")
    if corrected_label_proxy <= config.n_min_corrected_label_proxy:
        block_reasons.append("n_blocked_nonpositive_executable_exit_label_proxy")
    if score_proxy <= config.n_min_replay_aligned_score_proxy:
        block_reasons.append("n_blocked_nonpositive_replay_aligned_score_proxy")

    n_would_select = (
        not block_reasons
        and corrected_label_proxy > config.n_min_corrected_label_proxy
        and score_proxy > config.n_min_replay_aligned_score_proxy
    )
    compact = _compact_row(row)
    return {
        **compact,
        "original_label_target": label_target,
        "original_calibrated_action_score": score,
        "original_candidate_rank_score": _rank(row),
        "first_executable_exit_replay_pnl_proxy": executable_exit_proxy,
        "executable_exit_label_corrected_proxy": corrected_label_proxy,
        "realized_replay_pnl": replay_pnl,
        "label_vs_replay_gap_before": label_target - replay_pnl,
        "label_vs_replay_gap_after": corrected_label_proxy - replay_pnl,
        "up_replay_gap_penalty": replay_gap_penalty,
        "penalty_components": penalty_components,
        "n_replay_aligned_score_proxy": score_proxy,
        "n_score_delta": score_proxy - score,
        "high_score_negative_replay_guard_triggered": high_score_negative,
        "positive_label_replay_negative_flagged": positive_label_negative,
        "negative_label_selected_flagged": negative_label,
        "n_would_select": n_would_select,
        "n_would_block": not n_would_select,
        "n_decision_reason_codes": block_reasons
        if block_reasons
        else ["n_replay_aligned_overlay_would_select"],
        "n_overlay_uses_current_replay_evidence_for_diagnostics_only": True,
    }


def _candidate_report(
    *,
    config: PolymarketNUpReplayAlignedConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    overlay_rows: list[dict[str, Any]],
    n_selected: list[dict[str, Any]],
    n_blocked: list[dict[str, Any]],
    down_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [row for row in n_selected if row["realized_replay_pnl"] > 0.0]
    negative = [row for row in n_selected if row["realized_replay_pnl"] < 0.0]
    m2_positive = [row for row in overlay_rows if row["realized_replay_pnl"] > 0.0]
    m2_negative = [row for row in overlay_rows if row["realized_replay_pnl"] < 0.0]
    report = {
        "schema_version": N_UP_REPLAY_ALIGNED_CANDIDATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        ],
        "report_type": "n_up_replay_aligned_candidate",
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
        "m2_promotion_ready": False,
        "n_current_m_m2_evidence_reused_for_promotion": False,
        "n_overlay_uses_current_replay_evidence_for_diagnostics_only": True,
        "n_overlay_parameters": {
            "high_score_threshold": config.high_score_threshold,
            "n_min_replay_aligned_score_proxy": (
                config.n_min_replay_aligned_score_proxy
            ),
            "n_min_corrected_label_proxy": config.n_min_corrected_label_proxy,
            "high_score_negative_guard_penalty": (
                config.high_score_negative_guard_penalty
            ),
            "negative_label_penalty": config.negative_label_penalty,
            "positive_label_negative_replay_penalty": (
                config.positive_label_negative_replay_penalty
            ),
        },
        "original_up_selected_rows": overlay_rows,
        "n_would_selected_rows": n_selected,
        "n_would_blocked_rows": n_blocked,
        "m2_up_selected_count": len(overlay_rows),
        "m2_up_replay_pnl_sum": _overlay_pnl_sum(overlay_rows),
        "m2_up_positive_replay_count": len(m2_positive),
        "m2_up_negative_replay_count": len(m2_negative),
        "m2_up_label_vs_replay_gap": _overlay_gap_before(overlay_rows),
        "n_would_selected_up_count": len(n_selected),
        "n_would_blocked_up_count": len(n_blocked),
        "n_would_selected_up_replay_pnl_sum": _overlay_pnl_sum(n_selected),
        "n_would_selected_up_positive_replay_count": len(positive),
        "n_would_selected_up_negative_replay_count": len(negative),
        "n_label_vs_replay_gap_after_correction": _overlay_gap_after(n_selected),
        "n_false_positive_reduction_count": _false_positive_reduction_count(
            overlay_rows,
            n_blocked,
        ),
        "n_blocked_up_false_positive_count": _false_positive_reduction_count(
            overlay_rows,
            n_blocked,
        ),
        "n_block_reason_counts": _reason_counts(n_blocked),
        "top_rows_changed_by_n_overlay": _top_changed_rows(overlay_rows),
        "n_vs_m2_comparison": {
            "m2_up_selected_count": len(overlay_rows),
            "m2_up_replay_pnl_sum": _overlay_pnl_sum(overlay_rows),
            "m2_up_positive_replay_count": len(m2_positive),
            "m2_up_negative_replay_count": len(m2_negative),
            "m2_up_label_vs_replay_gap": _overlay_gap_before(overlay_rows),
            "n_would_selected_up_count": len(n_selected),
            "n_would_selected_up_replay_pnl_sum": _overlay_pnl_sum(n_selected),
            "n_would_selected_up_positive_replay_count": len(positive),
            "n_would_selected_up_negative_replay_count": len(negative),
            "n_label_vs_replay_gap_after_correction": _overlay_gap_after(
                n_selected
            ),
            "n_false_positive_reduction_count": _false_positive_reduction_count(
                overlay_rows,
                n_blocked,
            ),
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
    report["n_up_replay_aligned_candidate_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _score_overlay_report(
    *,
    config: PolymarketNUpReplayAlignedConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    overlay_rows: list[dict[str, Any]],
    n_selected: list[dict[str, Any]],
) -> dict[str, Any]:
    score_before = [row["original_calibrated_action_score"] for row in overlay_rows]
    score_after = [row["n_replay_aligned_score_proxy"] for row in overlay_rows]
    pnl = [row["realized_replay_pnl"] for row in overlay_rows]
    report = {
        "schema_version": N_UP_REPLAY_ALIGNED_SCORE_OVERLAY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": (
            SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME
        ),
        "baseline_candidate_names": [
            SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
            SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        ],
        "report_type": "n_up_replay_aligned_score_overlay",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "n_overlay_parameters": {
            "high_score_threshold": config.high_score_threshold,
            "n_min_replay_aligned_score_proxy": (
                config.n_min_replay_aligned_score_proxy
            ),
            "n_min_corrected_label_proxy": config.n_min_corrected_label_proxy,
        },
        "original_score_vs_replay_correlation": _pearson(score_before, pnl),
        "replay_aligned_score_proxy_vs_replay_correlation": _pearson(
            score_after,
            pnl,
        ),
        "score_correlation_delta": _correlation_delta(score_before, score_after, pnl),
        "label_vs_replay_gap_before_overlay": _overlay_gap_before(overlay_rows),
        "label_vs_replay_gap_after_overlay": _overlay_gap_after(overlay_rows),
        "label_vs_replay_gap_delta": _overlay_gap_before(overlay_rows)
        - _overlay_gap_after(overlay_rows),
        "score_overlay_rows": overlay_rows,
        "score_overlay_bucket_comparison": _score_bucket_comparison(overlay_rows),
        "top_rows_changed_by_n_overlay": _top_changed_rows(overlay_rows),
        "n_selected_score_summary": _score_summary(n_selected),
        "n_blocked_score_summary": _score_summary(
            [row for row in overlay_rows if not row["n_would_select"]]
        ),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["n_up_replay_aligned_score_overlay_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _first_executable_exit_proxy(row: dict[str, Any]) -> float:
    for field in (
        "first_executable_exit_replay_pnl_proxy",
        "execution_pnl_immediate_exit_pnl",
        "realized_trade_pnl",
        "total_polymarket_pnl",
    ):
        value = row.get(field)
        if value is not None:
            return float(value)
    return _label(row)


def _overlay_pnl_sum(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["realized_replay_pnl"]) for row in rows)


def _overlay_gap_before(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["label_vs_replay_gap_before"]) for row in rows)


def _overlay_gap_after(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["label_vs_replay_gap_after"]) for row in rows)


def _false_positive_reduction_count(
    all_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
) -> int:
    del all_rows
    return sum(
        1
        for row in blocked_rows
        if row["original_label_target"] > 0.0 and row["realized_replay_pnl"] < 0.0
    )


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["n_decision_reason_codes"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _top_changed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row["n_score_delta"])),
            abs(float(row["label_vs_replay_gap_before"]))
            - abs(float(row["label_vs_replay_gap_after"])),
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
        "executable_exit_label_corrected_proxy",
        "realized_replay_pnl",
        "label_vs_replay_gap_before",
        "label_vs_replay_gap_after",
        "original_calibrated_action_score",
        "n_replay_aligned_score_proxy",
        "up_replay_gap_penalty",
        "high_score_negative_replay_guard_triggered",
        "positive_label_replay_negative_flagged",
        "negative_label_selected_flagged",
        "n_would_select",
        "n_decision_reason_codes",
    )
    return {field: row.get(field) for field in fields}


def _score_bucket_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = _score_bucket(float(row["original_calibrated_action_score"]))
        groups.setdefault(bucket, []).append(row)
    result = []
    for bucket, bucket_rows in sorted(groups.items()):
        selected = [row for row in bucket_rows if row["n_would_select"]]
        blocked = [row for row in bucket_rows if not row["n_would_select"]]
        result.append(
            {
                "original_score_bucket": bucket,
                "row_count": len(bucket_rows),
                "n_would_selected_count": len(selected),
                "n_would_blocked_count": len(blocked),
                "original_replay_pnl_sum": _overlay_pnl_sum(bucket_rows),
                "n_selected_replay_pnl_sum": _overlay_pnl_sum(selected),
                "label_vs_replay_gap_before": _overlay_gap_before(bucket_rows),
                "label_vs_replay_gap_after": _overlay_gap_after(bucket_rows),
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


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "score_mean": None,
            "score_proxy_mean": None,
            "replay_pnl_sum": 0.0,
        }
    return {
        "row_count": len(rows),
        "score_mean": statistics.mean(
            float(row["original_calibrated_action_score"]) for row in rows
        ),
        "score_proxy_mean": statistics.mean(
            float(row["n_replay_aligned_score_proxy"]) for row in rows
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
        "# N UP Replay-Aligned Candidate Report",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
        f"- m2_up_selected_count: `{report['m2_up_selected_count']}`",
        f"- m2_up_replay_pnl_sum: `{report['m2_up_replay_pnl_sum']}`",
        f"- m2_up_label_vs_replay_gap: `{report['m2_up_label_vs_replay_gap']}`",
        f"- n_would_selected_up_count: `{report['n_would_selected_up_count']}`",
        f"- n_would_blocked_up_count: `{report['n_would_blocked_up_count']}`",
        "- n_would_selected_up_replay_pnl_sum: "
        f"`{report['n_would_selected_up_replay_pnl_sum']}`",
        "- n_label_vs_replay_gap_after_correction: "
        f"`{report['n_label_vs_replay_gap_after_correction']}`",
        f"- n_blocked_up_false_positive_count: `{report['n_blocked_up_false_positive_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Block Reasons",
        "",
    ]
    for reason, count in report["n_block_reason_counts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- diagnostic-only; current M/M2 evidence is not N promotion evidence",
            "- paper_only: true",
            "- capital_at_risk: false",
            "",
        ]
    )
    return "\n".join(lines)


def _score_overlay_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# N UP Replay-Aligned Score Overlay",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        "- original_score_vs_replay_correlation: "
        f"`{report['original_score_vs_replay_correlation']}`",
        "- replay_aligned_score_proxy_vs_replay_correlation: "
        f"`{report['replay_aligned_score_proxy_vs_replay_correlation']}`",
        f"- score_correlation_delta: `{report['score_correlation_delta']}`",
        "- label_vs_replay_gap_before_overlay: "
        f"`{report['label_vs_replay_gap_before_overlay']}`",
        "- label_vs_replay_gap_after_overlay: "
        f"`{report['label_vs_replay_gap_after_overlay']}`",
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
                selected=row["n_would_selected_count"],
                blocked=row["n_would_blocked_count"],
                original=float(row["original_replay_pnl_sum"]),
                selected_pnl=float(row["n_selected_replay_pnl_sum"]),
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
