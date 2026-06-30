"""Diagnostic M2 replay-parity selection for post-freeze Polymarket evidence."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.post_freeze_holdout import (
    M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.post_freeze_holdout_accumulation import (
    M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.post_freeze_weak_evidence_drilldown import (
    M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCE_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

M2_REPLAY_PARITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m2-stateful-replay-parity-candidate-v1"
)
M2_UP_ALIGNMENT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m2-up-label-replay-alignment-diagnostic-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketM2ReplayParityConfig:
    """Configuration for diagnostic M2 replay-parity reports."""

    weak_evidence_drilldown_report_path: Path | str
    accumulation_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_m2_stateful_replay_parity"
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "weak_evidence_drilldown_report_path",
            "accumulation_report_path",
            "output_dir",
        ):
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
class PolymarketM2ReplayParityResult:
    run_dir: Path
    candidate_report: dict[str, Any]
    up_alignment_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_m2_replay_parity_diagnostics(
    config: PolymarketM2ReplayParityConfig,
) -> PolymarketM2ReplayParityResult:
    """Build diagnostic-only M2 replay-parity reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "candidate_report": run_dir / "m2_stateful_replay_parity_candidate_report.json",
        "candidate_summary": run_dir / "m2_stateful_replay_parity_candidate_report.md",
        "up_alignment_report": run_dir / "m2_up_label_replay_alignment_diagnostic.json",
        "up_alignment_summary": run_dir
        / "m2_up_label_replay_alignment_diagnostic.md",
        "manifest": run_dir / "m2_stateful_replay_parity_manifest.json",
    }
    candidate_report, up_alignment_report = _build_reports(config=config)
    _write_json(artifact_paths["candidate_report"], candidate_report)
    artifact_paths["candidate_summary"].write_text(
        _candidate_markdown(candidate_report),
        encoding="utf-8",
    )
    _write_json(artifact_paths["up_alignment_report"], up_alignment_report)
    artifact_paths["up_alignment_summary"].write_text(
        _up_alignment_markdown(up_alignment_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-m2-stateful-replay-parity-artifacts-v1",
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
    return PolymarketM2ReplayParityResult(
        run_dir=run_dir,
        candidate_report=candidate_report,
        up_alignment_report=up_alignment_report,
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketM2ReplayParityConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    drilldown_path = config.weak_evidence_drilldown_report_path.expanduser().resolve()
    accumulation_path = config.accumulation_report_path.expanduser().resolve()
    drilldown = _read_json(drilldown_path)
    accumulation = _read_json(accumulation_path)
    if (
        drilldown.get("schema_version")
        != M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION
    ):
        raise ValueError("not a post-freeze weak-evidence drilldown report")
    if (
        accumulation.get("schema_version")
        != M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION
    ):
        raise ValueError("not a post-freeze holdout accumulation report")

    loaded_holdouts = _load_included_holdout_reports(
        accumulation=accumulation,
        base_dir=accumulation_path.parent,
    )
    run_results = [
        _m2_select_run(source)
        for source in loaded_holdouts
    ]
    selected_rows = [
        row for run in run_results for row in run["m2_selected_rows"]
    ]
    blocked_rows = [
        row for run in run_results for row in run["m2_blocked_rows"]
    ]
    m2_turnover_or_max_attrition = [
        row
        for row in selected_rows
        if not bool(row.get("entry_order_opened", False))
        and _has_turnover_or_max_entry_reason(row)
    ]
    m2_missing_replay_rows = [
        row for row in selected_rows if not bool(row.get("entry_order_opened", False))
    ]
    current_m_turnover_or_max_attrition_count = int(
        drilldown.get("turnover_or_max_entry_blocked_selected_row_count", 0)
    )
    known_replay_rows = [
        row for row in selected_rows if bool(row.get("entry_order_opened", False))
    ]
    replay_reconciliation = {
        "m2_selected_entry_count": len(selected_rows),
        "m2_known_replay_entry_count": len(known_replay_rows),
        "m2_selected_without_replay_count": len(m2_missing_replay_rows),
        "m2_turnover_or_max_entry_selected_without_replay_count": len(
            m2_turnover_or_max_attrition
        ),
        "reconciled": not m2_missing_replay_rows,
        "failure_reason_codes": (
            []
            if not m2_missing_replay_rows
            else ["m2_selected_rows_missing_replay_evidence"]
        ),
    }
    candidate_report = {
        "schema_version": M2_REPLAY_PARITY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m2_stateful_replay_parity_candidate",
        "diagnostic_only": True,
        "weak_evidence_drilldown_report_path": str(drilldown_path),
        "weak_evidence_drilldown_report_sha256": _sha256_file(drilldown_path),
        "weak_evidence_drilldown_report_id": drilldown.get(
            "m_post_freeze_weak_evidence_drilldown_report_id"
        ),
        "accumulation_report_path": str(accumulation_path),
        "accumulation_report_sha256": _sha256_file(accumulation_path),
        "accumulation_report_id": accumulation.get(
            "m_post_freeze_holdout_accumulation_report_id"
        ),
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "current_frozen_m_remains_blocked": True,
        "m2_selection_method": "stateful_replay_parity_side_balanced_selection",
        "m2_rank_weight_tuning_performed": False,
        "m2_holdout_feedback_used_for_rank_tuning": False,
        "m2_replay_guards_relaxed": False,
        "max_entries_per_market": int(
            float(
                SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS[
                    "max_entries_per_market"
                ]
            )
        ),
        "min_reentry_cooldown_seconds": float(
            SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS[
                "min_reentry_cooldown_seconds"
            ]
        ),
        "side_quota_per_side": int(
            float(SELL_BEFORE_CLOSE_SIDE_BALANCE_THRESHOLDS["side_quota_per_side"])
        ),
        "included_holdout_run_count": len(run_results),
        "run_results": run_results,
        "m2_selected_entry_count": len(selected_rows),
        "m2_known_replay_entry_count": len(known_replay_rows),
        "m2_selected_without_replay_count": len(m2_missing_replay_rows),
        "current_m_turnover_or_max_entry_attrition_count": (
            current_m_turnover_or_max_attrition_count
        ),
        "m2_turnover_or_max_entry_attrition_count": len(m2_turnover_or_max_attrition),
        "turnover_or_max_entry_attrition_reduced_to_zero": not m2_turnover_or_max_attrition,
        "turnover_or_max_entry_attrition_delta": (
            current_m_turnover_or_max_attrition_count
            - len(m2_turnover_or_max_attrition)
        ),
        "m2_replay_entry_reconciliation": replay_reconciliation,
        "m2_selected_rows": selected_rows,
        "m2_blocked_rows": blocked_rows,
        "m2_selection_reason_counts": _reason_counts(
            row.get("m2_reason_codes", []) for row in [*selected_rows, *blocked_rows]
        ),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    candidate_report["m2_stateful_replay_parity_candidate_report_id"] = (
        canonical_json_sha256(candidate_report)
    )
    up_alignment_report = _build_up_alignment_report(
        candidate_report=candidate_report,
        drilldown=drilldown,
    )
    return candidate_report, up_alignment_report


def _load_included_holdout_reports(
    *,
    accumulation: dict[str, Any],
    base_dir: Path,
) -> list[dict[str, Any]]:
    loaded = []
    for run in accumulation.get("included_runs", []):
        path = Path(str(run.get("report_path", ""))).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"included holdout report does not exist: {path}")
        report = _read_json(path)
        if report.get("schema_version") != M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION:
            raise ValueError(f"not a post-freeze holdout report: {path}")
        loaded.append(
            {
                "report_path": str(path),
                "report_sha256": _sha256_file(path),
                "run_summary": run,
                "report": report,
            }
        )
    return loaded


def _m2_select_run(source: dict[str, Any]) -> dict[str, Any]:
    report = source["report"]
    rows = [_candidate_row(row, source) for row in report.get("rows", [])]
    candidates = [
        row for row in rows if _eligible_for_m2_selection_pool(row)
    ]
    quota = int(float(SELL_BEFORE_CLOSE_SIDE_BALANCE_THRESHOLDS["side_quota_per_side"]))
    max_entries = int(
        float(
            SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS[
                "max_entries_per_market"
            ]
        )
    )
    cooldown_seconds = float(
        SELL_BEFORE_CLOSE_SIDE_BALANCED_ENTRY_GUARD_THRESHOLDS[
            "min_reentry_cooldown_seconds"
        ]
    )
    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    entries_per_market: Counter[str] = Counter()
    selected_count_by_side: Counter[str] = Counter()
    last_selected_ts_by_market: dict[str, int] = {}
    for row in sorted(
        candidates,
        key=lambda item: (
            int(item.get("decision_ts") or 0),
            -float(item.get("candidate_rank_score") or 0.0),
            str(item.get("market_id") or ""),
            str(item.get("selected_side") or ""),
        ),
    ):
        side = str(row.get("selected_side"))
        market_id = str(row.get("market_id"))
        reasons = []
        if side not in {"UP", "DOWN"}:
            reasons.append("m2_entry_blocked_invalid_side")
        if selected_count_by_side[side] >= quota:
            reasons.append("m2_entry_blocked_side_quota_full")
        if entries_per_market[market_id] >= max_entries:
            reasons.append("m2_entry_blocked_max_entries_per_market")
        last_selected_ts = last_selected_ts_by_market.get(market_id)
        if last_selected_ts is not None and cooldown_seconds > 0.0:
            elapsed = max(
                0.0,
                (int(row.get("decision_ts") or 0) - int(last_selected_ts)) / 1000.0,
            )
            if elapsed < cooldown_seconds:
                reasons.append("m2_entry_blocked_reentry_cooldown")
        if reasons:
            blocked.append(
                {
                    **row,
                    "m2_side_quota_selected": False,
                    "m2_reason_codes": list(
                        dict.fromkeys(("m2_entry_blocked_replay_parity_guard", *reasons))
                    ),
                }
            )
            continue
        selected_count_by_side[side] += 1
        entries_per_market[market_id] += 1
        last_selected_ts_by_market[market_id] = int(row.get("decision_ts") or 0)
        selected.append(
            {
                **row,
                "m2_side_quota_rank": selected_count_by_side[side],
                "m2_side_quota_selected": True,
                "m2_reason_codes": [
                    "m2_stateful_replay_parity_selected",
                    "m2_max_entries_per_market_guard_passed",
                    "m2_reentry_cooldown_guard_passed",
                ],
            }
        )
    known_replay_selected = [
        row for row in selected if bool(row.get("entry_order_opened", False))
    ]
    missing_replay_selected = [
        row for row in selected if not bool(row.get("entry_order_opened", False))
    ]
    turnover_or_max_blocked = [
        row
        for row in blocked
        if "m2_entry_blocked_max_entries_per_market" in row["m2_reason_codes"]
        or "m2_entry_blocked_reentry_cooldown" in row["m2_reason_codes"]
    ]
    return {
        "report_path": source["report_path"],
        "report_sha256": source["report_sha256"],
        "holdout_validation_passed": report.get("holdout_validation_passed"),
        "current_m_selected_entry_count": int(report.get("selected_entry_count", 0)),
        "current_m_replay_entry_count": int(report.get("replay_entry_count", 0)),
        "m2_candidate_count": len(candidates),
        "m2_selected_entry_count": len(selected),
        "m2_known_replay_entry_count": len(known_replay_selected),
        "m2_missing_replay_evidence_count": len(missing_replay_selected),
        "m2_turnover_or_max_entry_blocked_at_selection_count": len(
            turnover_or_max_blocked
        ),
        "m2_selected_without_replay_turnover_or_max_entry_count": len(
            [
                row
                for row in selected
                if not bool(row.get("entry_order_opened", False))
                and _has_turnover_or_max_entry_reason(row)
            ]
        ),
        "m2_replay_entry_reconciliation": {
            "reconciled": not missing_replay_selected,
            "selected_entry_count": len(selected),
            "known_replay_entry_count": len(known_replay_selected),
            "missing_replay_evidence_count": len(missing_replay_selected),
        },
        "m2_selected_rows": selected,
        "m2_blocked_rows": blocked,
    }


def _candidate_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "market_id",
        "decision_ts",
        "selected_side",
        "action",
        "side_quota_rank",
        "side_quota_selected",
        "entry_order_opened",
        "raw_calibrated_action_score",
        "best_action_margin",
        "candidate_rank_score",
        "action_return_target",
        "realized_trade_pnl",
        "settlement_pnl",
        "total_polymarket_pnl",
        "exit_reason_codes",
        "replay_reason_codes",
        "attrition_stage",
        "attrition_reason_codes",
    )
    payload = {field: row.get(field) for field in fields}
    payload["source_report_path"] = source["report_path"]
    payload["total_polymarket_pnl"] = float(payload.get("total_polymarket_pnl") or 0.0)
    payload["action_return_target"] = float(payload.get("action_return_target") or 0.0)
    payload["candidate_rank_score"] = float(payload.get("candidate_rank_score") or 0.0)
    payload["raw_calibrated_action_score"] = float(
        payload.get("raw_calibrated_action_score") or 0.0
    )
    payload["decision_ts"] = int(payload.get("decision_ts") or 0)
    return payload


def _eligible_for_m2_selection_pool(row: dict[str, Any]) -> bool:
    reasons = _all_reason_tokens(row)
    if any("position_state" in reason for reason in reasons):
        return False
    if any("existing_position" in reason for reason in reasons):
        return False
    quality_blockers = {
        "entry_blocked_exit_reliability_guard",
        "entry_blocked_insufficient_executable_bid_notional",
        "entry_blocked_low_queue_fill_probability",
        "entry_blocked_spread_too_wide",
        "entry_blocked_stale_book",
    }
    if any(reason in quality_blockers for reason in reasons):
        return False
    action = str(row.get("action") or "")
    return action in {"BUY_UP_SELL_BEFORE_CLOSE", "BUY_DOWN_SELL_BEFORE_CLOSE"}


def _build_up_alignment_report(
    *,
    candidate_report: dict[str, Any],
    drilldown: dict[str, Any],
) -> dict[str, Any]:
    up_rows = [
        row
        for row in candidate_report["m2_selected_rows"]
        if row.get("selected_side") == "UP"
    ]
    known_up_rows = [row for row in up_rows if bool(row.get("entry_order_opened"))]
    up_positive_replay = [
        row for row in known_up_rows if float(row.get("total_polymarket_pnl", 0.0)) > 0.0
    ]
    up_negative_replay = [
        row for row in known_up_rows if float(row.get("total_polymarket_pnl", 0.0)) < 0.0
    ]
    up_negative_label_selected = [
        row for row in up_rows if float(row.get("action_return_target", 0.0)) < 0.0
    ]
    up_positive_label_replay_negative = [
        row
        for row in known_up_rows
        if float(row.get("action_return_target", 0.0)) > 0.0
        and float(row.get("total_polymarket_pnl", 0.0)) < 0.0
    ]
    up_closed_negative = [
        row
        for row in up_negative_replay
        if any(
            reason == "closed_before_settlement_with_negative_replay_pnl"
            for reason in _all_reason_tokens(row)
        )
    ]
    report = {
        "schema_version": M2_UP_ALIGNMENT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        "baseline_candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m2_up_label_replay_alignment_diagnostic",
        "diagnostic_only": True,
        "current_frozen_m_promotion_status": "reject_promotion_for_now",
        "current_frozen_m_evidence_status": "weak_mixed_structural",
        "current_frozen_m_evidence_reused_for_m2_promotion": False,
        "m2_candidate_report_id": candidate_report[
            "m2_stateful_replay_parity_candidate_report_id"
        ],
        "m2_up_selected_entry_count": len(up_rows),
        "m2_up_known_replay_entry_count": len(known_up_rows),
        "m2_up_positive_replay_pnl_count": len(up_positive_replay),
        "m2_up_negative_replay_pnl_count": len(up_negative_replay),
        "m2_up_replay_pnl_sum": sum(
            float(row.get("total_polymarket_pnl", 0.0)) for row in known_up_rows
        ),
        "m2_up_label_target_sum": sum(
            float(row.get("action_return_target", 0.0)) for row in up_rows
        ),
        "m2_up_label_vs_replay_pnl_gap": sum(
            float(row.get("action_return_target", 0.0))
            - float(row.get("total_polymarket_pnl", 0.0))
            for row in known_up_rows
        ),
        "m2_up_negative_label_selected_count": len(up_negative_label_selected),
        "m2_up_negative_label_selected_rows": up_negative_label_selected,
        "m2_up_positive_label_replay_negative_count": len(
            up_positive_label_replay_negative
        ),
        "m2_up_positive_label_replay_negative_rows": (
            up_positive_label_replay_negative
        ),
        "m2_up_calibrated_action_score_vs_replay_pnl_correlation": _pearson(
            [
                float(row.get("raw_calibrated_action_score", 0.0))
                for row in known_up_rows
            ],
            [float(row.get("total_polymarket_pnl", 0.0)) for row in known_up_rows],
        ),
        "m2_up_first_executable_exit_negative_count": len(up_closed_negative),
        "m2_up_first_executable_exit_negative_rows": up_closed_negative,
        "m2_top_up_false_positives": _top_up_false_positives(known_up_rows),
        "baseline_m_up_loss_entry_count": int(drilldown.get("up_loss_entry_count", 0)),
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    report["m2_up_label_replay_alignment_diagnostic_id"] = canonical_json_sha256(
        report
    )
    return report


def _has_turnover_or_max_entry_reason(row: dict[str, Any]) -> bool:
    reasons = _all_reason_tokens(row)
    return any(
        "turnover" in reason
        or "max_entry" in reason
        or "max_entries" in reason
        or "max-position" in reason
        or "max_position" in reason
        for reason in reasons
    )


def _all_reason_tokens(row: dict[str, Any]) -> list[str]:
    tokens = []
    for field in ("exit_reason_codes", "replay_reason_codes", "attrition_reason_codes"):
        value = row.get(field) or []
        if isinstance(value, list):
            tokens.extend(str(item) for item in value)
        else:
            tokens.append(str(value))
    return [token.lower() for token in tokens]


def _reason_counts(reason_groups: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for reasons in reason_groups:
        counter.update(str(reason) for reason in reasons)
    return dict(sorted(counter.items()))


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


def _top_up_false_positives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_positives = [
        row
        for row in rows
        if float(row.get("total_polymarket_pnl", 0.0)) < 0.0
        and (
            float(row.get("action_return_target", 0.0)) > 0.0
            or float(row.get("raw_calibrated_action_score", 0.0)) > 0.0
        )
    ]
    false_positives.sort(
        key=lambda row: (
            float(row.get("total_polymarket_pnl", 0.0)),
            -float(row.get("raw_calibrated_action_score", 0.0)),
            int(row.get("decision_ts") or 0),
        )
    )
    return false_positives[:10]


def _candidate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M2 Stateful Replay-Parity Candidate Report",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- current_frozen_m_promotion_status: `{report['current_frozen_m_promotion_status']}`",
        f"- current_frozen_m_evidence_status: `{report['current_frozen_m_evidence_status']}`",
        f"- m2_selected_entry_count: `{report['m2_selected_entry_count']}`",
        f"- m2_known_replay_entry_count: `{report['m2_known_replay_entry_count']}`",
        f"- m2_selected_without_replay_count: `{report['m2_selected_without_replay_count']}`",
        "- current_m_turnover_or_max_entry_attrition_count: "
        f"`{report['current_m_turnover_or_max_entry_attrition_count']}`",
        "- m2_turnover_or_max_entry_attrition_count: "
        f"`{report['m2_turnover_or_max_entry_attrition_count']}`",
        "- replay_reconciliation: "
        f"`{str(report['m2_replay_entry_reconciliation']['reconciled']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Selection Reason Counts",
        "",
    ]
    for reason, count in report["m2_selection_reason_counts"].items():
        lines.append(f"- `{reason}`: `{count}`")
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


def _up_alignment_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M2 UP Label-Replay Alignment Diagnostic",
        "",
        f"- candidate_name: `{report['candidate_name']}`",
        f"- m2_up_selected_entry_count: `{report['m2_up_selected_entry_count']}`",
        f"- m2_up_known_replay_entry_count: `{report['m2_up_known_replay_entry_count']}`",
        f"- m2_up_positive_replay_pnl_count: `{report['m2_up_positive_replay_pnl_count']}`",
        f"- m2_up_negative_replay_pnl_count: `{report['m2_up_negative_replay_pnl_count']}`",
        f"- m2_up_replay_pnl_sum: `{report['m2_up_replay_pnl_sum']}`",
        "- m2_up_negative_label_selected_count: "
        f"`{report['m2_up_negative_label_selected_count']}`",
        "- m2_up_positive_label_replay_negative_count: "
        f"`{report['m2_up_positive_label_replay_negative_count']}`",
        "- m2_up_first_executable_exit_negative_count: "
        f"`{report['m2_up_first_executable_exit_negative_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Top UP False Positives",
        "",
        "| pnl | target | score | market |",
        "|---:|---:|---:|---|",
    ]
    for row in report["m2_top_up_false_positives"]:
        lines.append(
            "| {pnl:.6f} | {target:.6f} | {score:.6f} | {market} |".format(
                pnl=float(row.get("total_polymarket_pnl", 0.0)),
                target=float(row.get("action_return_target", 0.0)),
                score=float(row.get("raw_calibrated_action_score", 0.0)),
                market=row.get("market_id"),
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
