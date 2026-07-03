"""Root-cause drilldown for weak post-freeze M promotion evidence."""

from __future__ import annotations

import hashlib
import json
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
from bigan.v8.polymarket.training.post_freeze_promotion_readiness_audit import (
    M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m-post-freeze-weak-evidence-drilldown-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezeWeakEvidenceDrilldownConfig:
    """Configuration for a diagnostic weak-evidence root-cause drilldown."""

    promotion_readiness_audit_path: Path | str
    accumulation_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_m_post_freeze_weak_evidence_drilldown"
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "promotion_readiness_audit_path",
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
class PolymarketPostFreezeWeakEvidenceDrilldownResult:
    run_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_m_post_freeze_weak_evidence_drilldown(
    config: PolymarketPostFreezeWeakEvidenceDrilldownConfig,
) -> PolymarketPostFreezeWeakEvidenceDrilldownResult:
    """Create a diagnostic root-cause report for weak post-freeze M evidence."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "report": run_dir / "m_post_freeze_weak_evidence_drilldown_report.json",
        "summary": run_dir / "m_post_freeze_weak_evidence_drilldown_report.md",
        "manifest": run_dir / "m_post_freeze_weak_evidence_drilldown_manifest.json",
    }
    report = _build_drilldown_report(config=config)
    _write_json(artifact_paths["report"], report)
    artifact_paths["summary"].write_text(
        _drilldown_markdown(report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "bigan-v8-polymarket-m-post-freeze-weak-evidence-drilldown-artifacts-v1"
        ),
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
    return PolymarketPostFreezeWeakEvidenceDrilldownResult(
        run_dir=run_dir,
        report=report,
        artifact_paths=artifact_paths,
    )


def _build_drilldown_report(
    *,
    config: PolymarketPostFreezeWeakEvidenceDrilldownConfig,
) -> dict[str, Any]:
    audit_path = config.promotion_readiness_audit_path.expanduser().resolve()
    accumulation_path = config.accumulation_report_path.expanduser().resolve()
    audit = _read_json(audit_path)
    accumulation = _read_json(accumulation_path)
    if (
        audit.get("schema_version")
        != M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION
    ):
        raise ValueError("not a post-freeze promotion-readiness audit report")
    if (
        accumulation.get("schema_version")
        != M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION
    ):
        raise ValueError("not a post-freeze holdout accumulation report")

    loaded_holdouts = _load_included_holdout_reports(
        accumulation=accumulation,
        base_dir=accumulation_path.parent,
    )
    replay_entries = [
        _row_payload(row=row, source=source)
        for source in loaded_holdouts
        for row in source["report"].get("rows", [])
        if bool(row.get("entry_order_opened", False))
    ]
    selected_rows = [
        _row_payload(row=row, source=source)
        for source in loaded_holdouts
        for row in source["report"].get("rows", [])
        if bool(row.get("side_quota_selected", False))
    ]
    selected_without_replay_rows = [
        row for row in selected_rows if not bool(row.get("entry_order_opened", False))
    ]
    turnover_or_max_entry_blocked_rows = [
        row
        for row in selected_without_replay_rows
        if _has_turnover_or_max_entry_reason(row)
    ]
    run_rows = [_run_payload(source) for source in loaded_holdouts]
    passed_runs = [
        row for row in run_rows if row.get("holdout_validation_passed") is True
    ]
    failed_runs = [
        row for row in run_rows if row.get("holdout_validation_passed") is not True
    ]
    selected_zero_runs = [
        row for row in run_rows if int(row.get("selected_entry_count", 0)) == 0
    ]
    selected_without_replay_runs = _selected_without_replay_runs(
        run_rows=run_rows,
        selected_without_replay_rows=selected_without_replay_rows,
    )
    up_loss_entries = _loss_entries(replay_entries, "UP")
    down_loss_entries = _loss_entries(replay_entries, "DOWN")
    up_loss_runs = _loss_runs(run_rows, "UP")
    down_loss_runs = _loss_runs(run_rows, "DOWN")
    largest_winner = _largest_winner_dependency(
        audit=audit,
        replay_entries=replay_entries,
    )
    median_weakness = _median_entry_pnl_weakness(
        audit=audit,
        replay_entries=replay_entries,
    )
    top_positive_entries = _top_entries(replay_entries, reverse=True)
    top_negative_entries = _top_entries(replay_entries, reverse=False)
    failed_reason_summary = _failed_reason_summary(failed_runs)
    failed_pnl_sign_summary = _failed_runs_by_pnl_sign(failed_runs)
    root_cause_indicators = _root_cause_indicators(
        accumulation=accumulation,
        failed_runs=failed_runs,
        selected_without_replay_rows=selected_without_replay_rows,
        turnover_or_max_entry_blocked_rows=turnover_or_max_entry_blocked_rows,
        up_loss_entries=up_loss_entries,
        down_loss_entries=down_loss_entries,
        largest_winner=largest_winner,
        median_weakness=median_weakness,
    )
    root_cause_classification = _root_cause_classification(root_cause_indicators)
    recommended_next_actions = _recommended_next_actions(root_cause_indicators)
    report = {
        "schema_version": M_POST_FREEZE_WEAK_EVIDENCE_DRILLDOWN_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m_post_freeze_weak_evidence_drilldown",
        "diagnostic_only": True,
        "promotion_readiness_audit_path": str(audit_path),
        "promotion_readiness_audit_sha256": _sha256_file(audit_path),
        "promotion_readiness_audit_id": audit.get(
            "m_post_freeze_promotion_readiness_audit_id"
        ),
        "accumulation_report_path": str(accumulation_path),
        "accumulation_report_sha256": _sha256_file(accumulation_path),
        "accumulation_report_id": accumulation.get(
            "m_post_freeze_holdout_accumulation_report_id"
        ),
        "promotion_readiness": audit.get("promotion_readiness"),
        "support_gate_passed": bool(accumulation.get("support_gate_passed", False)),
        "promotion_evidence_eligible": bool(
            accumulation.get("promotion_evidence_eligible", False)
        ),
        "source_model_candidate_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_run_resume_allowed": False,
        "promotion_gate_reason_codes": list(
            accumulation.get("promotion_gate_reason_codes", [])
        ),
        "source_model_candidate_ineligible_reason_codes": list(
            accumulation.get("source_model_candidate_ineligible_reason_codes", [])
        ),
        "included_holdout_run_count": len(run_rows),
        "passed_included_holdout_run_count": len(passed_runs),
        "failed_included_holdout_run_count": len(failed_runs),
        "passed_vs_failed_included_holdout_runs": {
            "passed": len(passed_runs),
            "failed": len(failed_runs),
            "failed_ratio": len(failed_runs) / len(run_rows) if run_rows else 0.0,
            "passed_runs": passed_runs,
            "failed_runs": failed_runs,
        },
        "failed_run_reason_code_counts": failed_reason_summary["reason_code_counts"],
        "failed_run_ineligible_reason_code_counts": (
            failed_reason_summary["ineligible_reason_code_counts"]
        ),
        "failed_run_reason_codes_by_run": failed_reason_summary["by_run"],
        "failed_runs_by_replay_pnl_sign": failed_pnl_sign_summary,
        "selected_zero_run_count": len(selected_zero_runs),
        "selected_zero_runs": selected_zero_runs,
        "selected_without_replay_run_count": len(selected_without_replay_runs),
        "selected_without_replay_runs": selected_without_replay_runs,
        "selected_without_replay_row_count": len(selected_without_replay_rows),
        "selected_without_replay_rows": selected_without_replay_rows,
        "turnover_or_max_entry_blocked_selected_row_count": (
            len(turnover_or_max_entry_blocked_rows)
        ),
        "selected_rows_blocked_by_replay_turnover_or_max_entry_guards": (
            turnover_or_max_entry_blocked_rows
        ),
        "up_loss_run_count": len(up_loss_runs),
        "up_loss_runs": up_loss_runs,
        "up_loss_entry_count": len(up_loss_entries),
        "up_loss_entries": up_loss_entries,
        "down_loss_run_count": len(down_loss_runs),
        "down_loss_runs": down_loss_runs,
        "down_loss_entry_count": len(down_loss_entries),
        "down_loss_entries": down_loss_entries,
        "largest_winner_dependency": largest_winner,
        "median_entry_pnl_weakness": median_weakness,
        "top_positive_replay_entries": top_positive_entries,
        "top_negative_replay_entries": top_negative_entries,
        "weakness_type": _weakness_type(root_cause_indicators),
        "root_cause_indicators": root_cause_indicators,
        "root_cause_classification": root_cause_classification,
        "recommended_next_action": recommended_next_actions[0],
        "recommended_next_actions": recommended_next_actions,
        "selector_logic_changed": False,
        "rank_weights_changed": False,
        "validator_logic_changed": False,
        "accumulator_logic_changed": False,
        "gates_relaxed": False,
        **compact_safety_fields(),
    }
    report["m_post_freeze_weak_evidence_drilldown_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


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
            loaded.append(
                {
                    "report_path": str(path),
                    "run_summary": run,
                    "missing": True,
                    "report": {"rows": []},
                }
            )
            continue
        report = _read_json(path)
        if report.get("schema_version") != M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION:
            raise ValueError(f"not a post-freeze holdout report: {path}")
        loaded.append(
            {
                "report_path": str(path),
                "report_sha256": _sha256_file(path),
                "run_summary": run,
                "missing": False,
                "report": report,
            }
        )
    return loaded


def _row_payload(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "market_id",
        "decision_ts",
        "selected_side",
        "action",
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
    payload["source_holdout_validation_passed"] = source["report"].get(
        "holdout_validation_passed"
    )
    payload["total_polymarket_pnl"] = float(payload.get("total_polymarket_pnl") or 0.0)
    payload["action_return_target"] = float(payload.get("action_return_target") or 0.0)
    payload["decision_ts"] = int(payload.get("decision_ts") or 0)
    return payload


def _run_payload(source: dict[str, Any]) -> dict[str, Any]:
    report = source["report"]
    summary = dict(source.get("run_summary", {}))
    reconciliation = dict(report.get("replay_entry_reconciliation", {}))
    rows = list(report.get("rows", []))
    selected_without_replay_count = len(
        [
            row
            for row in rows
            if bool(row.get("side_quota_selected", False))
            and not bool(row.get("entry_order_opened", False))
        ]
    )
    replay_pnl_by_side = dict(report.get("replay_pnl_by_side", {}))
    return {
        "report_path": source["report_path"],
        "report_sha256": source.get("report_sha256"),
        "validation_status": report.get("validation_status"),
        "holdout_validation_passed": report.get("holdout_validation_passed"),
        "selected_entry_count": int(report.get("selected_entry_count", 0)),
        "replay_entry_count": int(report.get("replay_entry_count", 0)),
        "selected_without_replay_entry_count": int(
            reconciliation.get(
                "selected_without_replay_entry_count",
                selected_without_replay_count,
            )
        ),
        "row_level_selected_without_replay_count": selected_without_replay_count,
        "selected_exit_decision_count": int(
            report.get("selected_exit_decision_count", 0)
        ),
        "replay_total_pnl_sum": float(report.get("replay_total_pnl_sum", 0.0)),
        "replay_pnl_by_side": replay_pnl_by_side,
        "dominant_replay_side": _dominant_side(replay_pnl_by_side),
        "reason_codes": list(report.get("reason_codes", [])),
        "ineligible_reason_codes": list(report.get("ineligible_reason_codes", [])),
        "holdout_min_decision_ts": summary.get("holdout_min_decision_ts"),
        "holdout_max_decision_ts": summary.get("holdout_max_decision_ts"),
    }


def _selected_without_replay_runs(
    *,
    run_rows: list[dict[str, Any]],
    selected_without_replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_counts = Counter(
        str(row.get("source_report_path")) for row in selected_without_replay_rows
    )
    results = []
    for run in run_rows:
        row_count = int(row_counts.get(str(run.get("report_path")), 0))
        summary_count = int(run.get("selected_without_replay_entry_count", 0))
        if row_count <= 0 and summary_count <= 0:
            continue
        payload = dict(run)
        payload["selected_without_replay_row_count"] = row_count
        results.append(payload)
    return results


def _has_turnover_or_max_entry_reason(row: dict[str, Any]) -> bool:
    reasons = _all_reason_tokens(row)
    return any(
        "turnover" in reason
        or "max_entry" in reason
        or "max-entry" in reason
        or "max entry" in reason
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


def _loss_entries(entries: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in entries
        if row.get("selected_side") == side
        and float(row.get("total_polymarket_pnl", 0.0)) < 0.0
    ]
    rows.sort(
        key=lambda row: (
            float(row["total_polymarket_pnl"]),
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
        )
    )
    return rows


def _loss_runs(run_rows: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in run_rows
        if float(dict(row.get("replay_pnl_by_side", {})).get(side, 0.0)) < 0.0
    ]
    rows.sort(
        key=lambda row: (
            float(dict(row.get("replay_pnl_by_side", {})).get(side, 0.0)),
            str(row.get("report_path") or ""),
        )
    )
    return rows


def _largest_winner_dependency(
    *,
    audit: dict[str, Any],
    replay_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_leave_one_out = dict(audit.get("leave_one_out_replay_pnl_sensitivity", {}))
    largest_positive = audit_leave_one_out.get("largest_positive_entry")
    if largest_positive is None:
        positive_entries = [
            row for row in replay_entries if float(row["total_polymarket_pnl"]) > 0.0
        ]
        largest_positive = max(
            positive_entries,
            key=lambda row: float(row["total_polymarket_pnl"]),
            default=None,
        )
    total_after = float(
        audit.get(
            "largest_positive_entry_removed_total_pnl",
            audit_leave_one_out.get("total_pnl_after_largest_positive_entry_removed", 0.0),
        )
    )
    return {
        "largest_positive_entry": largest_positive,
        "total_pnl_after_largest_positive_entry_removed": total_after,
        "total_pnl_remains_positive_if_largest_positive_entry_removed": total_after
        > 0.0,
        "winner_concentration_detected": total_after <= 0.0,
    }


def _median_entry_pnl_weakness(
    *,
    audit: dict[str, Any],
    replay_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    stats = dict(audit.get("pnl_per_replay_entry_stats", {}))
    pnls = [float(row["total_polymarket_pnl"]) for row in replay_entries]
    if "median" not in stats:
        stats["median"] = statistics.median(pnls) if pnls else 0.0
    if "mean" not in stats:
        stats["mean"] = statistics.mean(pnls) if pnls else 0.0
    if "minimum" not in stats:
        stats["minimum"] = min(pnls) if pnls else 0.0
    if "count" not in stats:
        stats["count"] = len(pnls)
    median = float(stats.get("median", 0.0))
    return {
        "pnl_per_replay_entry_stats": stats,
        "median_entry_pnl_non_positive": median <= 0.0,
        "median_entry_pnl": median,
    }


def _top_entries(
    entries: list[dict[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    rows = list(entries)
    rows.sort(
        key=lambda row: (
            float(row["total_polymarket_pnl"]),
            -int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
        ),
        reverse=reverse,
    )
    return rows[:10]


def _failed_reason_summary(failed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    ineligible_counts: Counter[str] = Counter()
    by_run = []
    for run in failed_runs:
        reason_codes = list(run.get("reason_codes", []))
        ineligible_codes = list(run.get("ineligible_reason_codes", []))
        reason_counts.update(reason_codes)
        ineligible_counts.update(ineligible_codes)
        by_run.append(
            {
                "report_path": run.get("report_path"),
                "replay_total_pnl_sum": run.get("replay_total_pnl_sum"),
                "reason_codes": reason_codes,
                "ineligible_reason_codes": ineligible_codes,
            }
        )
    return {
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "ineligible_reason_code_counts": dict(sorted(ineligible_counts.items())),
        "by_run": by_run,
    }


def _failed_runs_by_pnl_sign(failed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "negative": [],
        "zero": [],
        "positive": [],
    }
    for run in failed_runs:
        pnl = float(run.get("replay_total_pnl_sum", 0.0))
        if pnl < 0.0:
            buckets["negative"].append(run)
        elif pnl > 0.0:
            buckets["positive"].append(run)
        else:
            buckets["zero"].append(run)
    return {
        "negative_count": len(buckets["negative"]),
        "zero_count": len(buckets["zero"]),
        "positive_count": len(buckets["positive"]),
        "negative_runs": buckets["negative"],
        "zero_runs": buckets["zero"],
        "positive_runs": buckets["positive"],
    }


def _root_cause_indicators(
    *,
    accumulation: dict[str, Any],
    failed_runs: list[dict[str, Any]],
    selected_without_replay_rows: list[dict[str, Any]],
    turnover_or_max_entry_blocked_rows: list[dict[str, Any]],
    up_loss_entries: list[dict[str, Any]],
    down_loss_entries: list[dict[str, Any]],
    largest_winner: dict[str, Any],
    median_weakness: dict[str, Any],
) -> dict[str, bool]:
    support_gate_passed = bool(accumulation.get("support_gate_passed", False))
    replay_pnl_by_side = dict(accumulation.get("replay_pnl_by_side", {}))
    up_pnl = float(replay_pnl_by_side.get("UP", 0.0))
    down_pnl = float(replay_pnl_by_side.get("DOWN", 0.0))
    failed_ratio = (
        len(failed_runs) / int(accumulation.get("holdout_run_count", 0))
        if int(accumulation.get("holdout_run_count", 0)) > 0
        else 0.0
    )
    return {
        "sample_size_insufficient": not support_gate_passed,
        "side_imbalance": up_pnl < 0.0
        or down_pnl < 0.0
        or bool(up_loss_entries and not down_loss_entries)
        or bool(down_loss_entries and not up_loss_entries),
        "winner_concentration": bool(
            largest_winner.get("winner_concentration_detected", False)
        ),
        "execution_attrition": bool(
            selected_without_replay_rows or turnover_or_max_entry_blocked_rows
        ),
        "structural_weakness": bool(
            failed_runs
            and (
                failed_ratio >= 0.25
                or median_weakness["median_entry_pnl_non_positive"]
                or up_pnl < 0.0
                or down_pnl < 0.0
            )
        ),
    }


def _root_cause_classification(indicators: dict[str, bool]) -> str:
    if indicators["sample_size_insufficient"]:
        return "sample_size_insufficient"
    active = [
        name
        for name in (
            "side_imbalance",
            "winner_concentration",
            "execution_attrition",
            "structural_weakness",
        )
        if indicators.get(name)
    ]
    if len(active) == 1:
        return active[0]
    if active:
        return "mixed"
    return "structural_weakness"


def _weakness_type(indicators: dict[str, bool]) -> str:
    if indicators["sample_size_insufficient"]:
        return "sample_size_driven"
    if indicators["structural_weakness"] or indicators["side_imbalance"]:
        return "structural_or_mixed_not_sample_size_driven"
    return "inconclusive"


def _recommended_next_actions(indicators: dict[str, bool]) -> list[str]:
    actions = ["keep_blocked"]
    if indicators["sample_size_insufficient"]:
        actions.append("continue_collecting_data")
    if indicators["side_imbalance"]:
        actions.append("investigate_side_specific_weakness")
    if indicators["execution_attrition"]:
        actions.append("investigate_execution_attrition")
    if indicators["winner_concentration"] or indicators["structural_weakness"]:
        actions.append("reject_promotion_for_now")
    if "continue_collecting_data" not in actions:
        actions.append("continue_collecting_data")
    return list(dict.fromkeys(actions))


def _dominant_side(pnl_by_side: dict[str, Any]) -> str:
    nonzero = [
        side for side in ("UP", "DOWN") if abs(float(pnl_by_side.get(side, 0.0))) > 0.0
    ]
    if len(nonzero) == 1:
        return nonzero[0]
    if len(nonzero) > 1:
        return "MIXED"
    return "NONE"


def _drilldown_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M Post-Freeze Weak Evidence Drilldown",
        "",
        f"- root_cause_classification: `{report['root_cause_classification']}`",
        f"- weakness_type: `{report['weakness_type']}`",
        f"- recommended_next_action: `{report['recommended_next_action']}`",
        f"- promotion_readiness: `{report['promotion_readiness']}`",
        f"- support_gate_passed: `{str(report['support_gate_passed']).lower()}`",
        "- promotion_evidence_eligible: "
        f"`{str(report['promotion_evidence_eligible']).lower()}`",
        "- source_model_candidate_eligible: "
        f"`{str(report['source_model_candidate_eligible']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Included Runs",
        "",
        f"- passed: `{report['passed_included_holdout_run_count']}`",
        f"- failed: `{report['failed_included_holdout_run_count']}`",
        f"- selected_zero_run_count: `{report['selected_zero_run_count']}`",
        "- selected_without_replay_run_count: "
        f"`{report['selected_without_replay_run_count']}`",
        "- turnover_or_max_entry_blocked_selected_row_count: "
        f"`{report['turnover_or_max_entry_blocked_selected_row_count']}`",
        "",
        "## Failed Runs By PnL Sign",
        "",
        "| sign | count |",
        "|---|---:|",
        "| negative | {negative_count} |".format(
            **report["failed_runs_by_replay_pnl_sign"]
        ),
        "| zero | {zero_count} |".format(**report["failed_runs_by_replay_pnl_sign"]),
        "| positive | {positive_count} |".format(
            **report["failed_runs_by_replay_pnl_sign"]
        ),
        "",
        "## Side Losses",
        "",
        f"- UP loss runs: `{report['up_loss_run_count']}`",
        f"- UP loss entries: `{report['up_loss_entry_count']}`",
        f"- DOWN loss runs: `{report['down_loss_run_count']}`",
        f"- DOWN loss entries: `{report['down_loss_entry_count']}`",
        "",
        "## Winner And Median Weakness",
        "",
        "- winner_concentration_detected: "
        f"`{str(report['largest_winner_dependency']['winner_concentration_detected']).lower()}`",
        "- total_pnl_after_largest_positive_entry_removed: "
        f"`{report['largest_winner_dependency']['total_pnl_after_largest_positive_entry_removed']}`",
        "- median_entry_pnl_non_positive: "
        f"`{str(report['median_entry_pnl_weakness']['median_entry_pnl_non_positive']).lower()}`",
        "- median_entry_pnl: "
        f"`{report['median_entry_pnl_weakness']['median_entry_pnl']}`",
        "",
        "## Top Negative Entries",
        "",
        "| side | pnl | market |",
        "|---|---:|---|",
    ]
    for row in report["top_negative_replay_entries"][:10]:
        lines.append(
            "| {side} | {pnl:.6f} | {market} |".format(
                side=row.get("selected_side"),
                pnl=float(row.get("total_polymarket_pnl", 0.0)),
                market=row.get("market_id"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Positive Entries",
            "",
            "| side | pnl | market |",
            "|---|---:|---|",
        ]
    )
    for row in report["top_positive_replay_entries"][:10]:
        lines.append(
            "| {side} | {pnl:.6f} | {market} |".format(
                side=row.get("selected_side"),
                pnl=float(row.get("total_polymarket_pnl", 0.0)),
                market=row.get("market_id"),
            )
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Evidence stays fail-closed. This report does not change selector, rank "
            "weights, validator, accumulator, or gates.",
            "- paper_only: true",
            "- capital_at_risk: false",
            "- polymarket_write_enabled: false",
            "- wallet_signing_enabled: false",
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
