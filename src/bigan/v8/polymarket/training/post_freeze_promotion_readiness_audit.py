"""Promotion-readiness audit for post-freeze M holdout accumulation evidence."""

from __future__ import annotations

import hashlib
import json
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
from bigan.v8.polymarket.training.post_freeze_holdout_accumulation import (
    M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m-post-freeze-promotion-readiness-audit-v1"
)


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezePromotionReadinessAuditConfig:
    """Configuration for a diagnostic promotion-readiness audit."""

    accumulation_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_m_post_freeze_promotion_readiness_audit"
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        for field_name in ("accumulation_report_path", "output_dir"):
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
class PolymarketPostFreezePromotionReadinessAuditResult:
    run_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_m_post_freeze_promotion_readiness_audit(
    config: PolymarketPostFreezePromotionReadinessAuditConfig,
) -> PolymarketPostFreezePromotionReadinessAuditResult:
    """Create a diagnostic audit from a holdout accumulation report."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "report": run_dir / "m_post_freeze_promotion_readiness_audit.json",
        "summary": run_dir / "m_post_freeze_promotion_readiness_audit.md",
        "manifest": run_dir / "m_post_freeze_promotion_readiness_audit_manifest.json",
    }
    report = _build_audit_report(config=config)
    _write_json(artifact_paths["report"], report)
    artifact_paths["summary"].write_text(
        _audit_markdown(report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "bigan-v8-polymarket-m-post-freeze-promotion-readiness-audit-artifacts-v1"
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
    return PolymarketPostFreezePromotionReadinessAuditResult(
        run_dir=run_dir,
        report=report,
        artifact_paths=artifact_paths,
    )


def _build_audit_report(
    *,
    config: PolymarketPostFreezePromotionReadinessAuditConfig,
) -> dict[str, Any]:
    accumulation_report_path = config.accumulation_report_path.expanduser().resolve()
    accumulation = _read_json(accumulation_report_path)
    if (
        accumulation.get("schema_version")
        != M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION
    ):
        raise ValueError("not a post-freeze holdout accumulation report")
    loaded_holdout_reports = _load_included_holdout_reports(
        accumulation=accumulation,
        base_dir=accumulation_report_path.parent,
    )
    replay_entries = [
        _replay_entry_payload(row=row, source=source)
        for source in loaded_holdout_reports
        for row in source["report"].get("rows", [])
        if bool(row.get("entry_order_opened", False))
    ]
    per_run_replay = _per_run_replay(accumulation.get("included_runs", []))
    failed_runs = [
        row
        for row in per_run_replay
        if row.get("holdout_validation_passed") is not True
    ]
    pnls = [float(row["total_polymarket_pnl"]) for row in replay_entries]
    pnl_stats = _pnl_stats(pnls)
    replay_total_pnl = float(accumulation.get("replay_total_pnl_sum", 0.0))
    leave_one_out = _leave_one_out(entries=replay_entries, total_pnl=replay_total_pnl)
    side_imbalance = _side_imbalance(accumulation)
    promotion_gate_reason_codes = list(
        accumulation.get("promotion_gate_reason_codes", [])
    )
    source_ineligible_codes = list(
        accumulation.get("source_model_candidate_ineligible_reason_codes", [])
    )
    audit_findings = _audit_findings(
        accumulation=accumulation,
        failed_runs=failed_runs,
        side_imbalance=side_imbalance,
        leave_one_out=leave_one_out,
    )
    evidence_strength = _evidence_strength(
        accumulation=accumulation,
        failed_runs=failed_runs,
        side_imbalance=side_imbalance,
        leave_one_out=leave_one_out,
    )
    report = {
        "schema_version": M_POST_FREEZE_PROMOTION_READINESS_AUDIT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m_post_freeze_promotion_readiness_audit",
        "diagnostic_only": True,
        "accumulation_report_path": str(accumulation_report_path),
        "accumulation_report_sha256": _sha256_file(accumulation_report_path),
        "accumulation_report_id": accumulation.get(
            "m_post_freeze_holdout_accumulation_report_id"
        ),
        "promotion_gate_reason_codes": promotion_gate_reason_codes,
        "source_model_candidate_ineligible_reason_codes": source_ineligible_codes,
        "support_gate_passed": bool(accumulation.get("support_gate_passed", False)),
        "promotion_evidence_eligible": bool(
            accumulation.get("promotion_evidence_eligible", False)
        ),
        "source_model_candidate_eligible": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_run_resume_allowed": False,
        "holdout_run_count": int(accumulation.get("holdout_run_count", 0)),
        "replay_entry_count": int(accumulation.get("replay_entry_count", 0)),
        "replay_unique_market_count": int(
            accumulation.get("replay_unique_market_count", 0)
        ),
        "replay_entry_count_by_side": dict(
            accumulation.get("replay_entry_count_by_side", {})
        ),
        "replay_total_pnl_sum": replay_total_pnl,
        "replay_pnl_by_side": dict(accumulation.get("replay_pnl_by_side", {})),
        "included_holdout_validation_failed_count": len(failed_runs),
        "included_runs_with_holdout_validation_passed_false": failed_runs,
        "per_run_replay_pnl": per_run_replay,
        "top_negative_replay_entries": _top_negative_entries(replay_entries),
        "up_vs_down_pnl_imbalance": side_imbalance,
        "leave_one_out_replay_pnl_sensitivity": leave_one_out,
        "pnl_per_replay_entry_stats": pnl_stats,
        "total_pnl_remains_positive_if_largest_positive_entry_removed": bool(
            leave_one_out["total_pnl_after_largest_positive_entry_removed"] > 0.0
        ),
        "largest_positive_entry_removed_total_pnl": leave_one_out[
            "total_pnl_after_largest_positive_entry_removed"
        ],
        "up_side_negative_pnl_should_block_promotion_discussion": bool(
            side_imbalance["up_side_negative_pnl"]
        ),
        "up_side_negative_pnl_block_reason": _up_side_block_reason(side_imbalance),
        "promotion_readiness": evidence_strength,
        "audit_findings": audit_findings,
        "missing_included_run_report_paths": [
            row for row in loaded_holdout_reports if row.get("missing")
        ],
        "selector_logic_changed": False,
        "rank_weights_changed": False,
        "validator_logic_changed": False,
        "accumulator_logic_changed": False,
        "gates_relaxed": False,
        **compact_safety_fields(),
    }
    report["m_post_freeze_promotion_readiness_audit_id"] = canonical_json_sha256(
        report
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
        loaded.append(
            {
                "report_path": str(path),
                "report_sha256": _sha256_file(path),
                "run_summary": run,
                "missing": False,
                "report": _read_json(path),
            }
        )
    return loaded


def _replay_entry_payload(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "market_id",
        "decision_ts",
        "selected_side",
        "action",
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
    payload["decision_ts"] = int(payload.get("decision_ts") or 0)
    return payload


def _per_run_replay(included_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in included_runs:
        pnl_by_side = dict(run.get("replay_pnl_by_side", {}))
        rows.append(
            {
                "report_path": run.get("report_path"),
                "holdout_corpus_dir": run.get("holdout_corpus_dir"),
                "holdout_min_decision_ts": run.get("holdout_min_decision_ts"),
                "holdout_max_decision_ts": run.get("holdout_max_decision_ts"),
                "holdout_validation_passed": run.get("holdout_validation_passed"),
                "replay_entry_count": int(run.get("replay_entry_count", 0)),
                "replay_unique_market_count": int(
                    run.get("replay_unique_market_count", 0)
                ),
                "replay_total_pnl_sum": float(run.get("replay_total_pnl_sum", 0.0)),
                "replay_pnl_by_side": pnl_by_side,
                "dominant_replay_side": _dominant_side(pnl_by_side),
                "ineligible_reason_codes": list(run.get("ineligible_reason_codes", [])),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["replay_total_pnl_sum"]),
            int(row.get("holdout_min_decision_ts") or 0),
            str(row.get("report_path") or ""),
        )
    )
    return rows


def _dominant_side(pnl_by_side: dict[str, Any]) -> str:
    nonzero = [
        side for side in ("UP", "DOWN") if abs(float(pnl_by_side.get(side, 0.0))) > 0.0
    ]
    if len(nonzero) == 1:
        return nonzero[0]
    if len(nonzero) > 1:
        return "MIXED"
    return "NONE"


def _pnl_stats(pnls: list[float]) -> dict[str, float | int]:
    if not pnls:
        return {
            "count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(pnls),
        "minimum": min(pnls),
        "median": statistics.median(pnls),
        "mean": statistics.mean(pnls),
    }


def _leave_one_out(
    *,
    entries: list[dict[str, Any]],
    total_pnl: float,
) -> dict[str, Any]:
    rows = []
    for entry in entries:
        pnl = float(entry["total_polymarket_pnl"])
        rows.append(
            {
                "market_id": entry.get("market_id"),
                "decision_ts": entry.get("decision_ts"),
                "selected_side": entry.get("selected_side"),
                "entry_pnl": pnl,
                "total_pnl_without_entry": total_pnl - pnl,
                "source_report_path": entry.get("source_report_path"),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["total_pnl_without_entry"]),
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
        )
    )
    positive_entries = [
        entry for entry in entries if float(entry["total_polymarket_pnl"]) > 0.0
    ]
    largest_positive = max(
        positive_entries,
        key=lambda row: float(row["total_polymarket_pnl"]),
        default=None,
    )
    total_after_largest_positive_removed = (
        total_pnl
        if largest_positive is None
        else total_pnl - float(largest_positive["total_polymarket_pnl"])
    )
    return {
        "entry_count": len(entries),
        "minimum_leave_one_out_total_pnl": (
            min((float(row["total_pnl_without_entry"]) for row in rows), default=0.0)
        ),
        "median_leave_one_out_total_pnl": (
            statistics.median([float(row["total_pnl_without_entry"]) for row in rows])
            if rows
            else 0.0
        ),
        "maximum_leave_one_out_total_pnl": (
            max((float(row["total_pnl_without_entry"]) for row in rows), default=0.0)
        ),
        "total_pnl_positive_after_removing_any_single_entry": all(
            float(row["total_pnl_without_entry"]) > 0.0 for row in rows
        )
        if rows
        else False,
        "largest_positive_entry": largest_positive,
        "total_pnl_after_largest_positive_entry_removed": (
            total_after_largest_positive_removed
        ),
        "worst_case_entries": rows[:10],
    }


def _side_imbalance(accumulation: dict[str, Any]) -> dict[str, Any]:
    pnl_by_side = dict(accumulation.get("replay_pnl_by_side", {}))
    up_pnl = float(pnl_by_side.get("UP", 0.0))
    down_pnl = float(pnl_by_side.get("DOWN", 0.0))
    total = up_pnl + down_pnl
    return {
        "up_pnl": up_pnl,
        "down_pnl": down_pnl,
        "net_pnl": total,
        "up_minus_down_pnl": up_pnl - down_pnl,
        "absolute_side_pnl_gap": abs(up_pnl - down_pnl),
        "up_side_negative_pnl": up_pnl < 0.0,
        "down_side_negative_pnl": down_pnl < 0.0,
        "side_gap_to_abs_net_ratio": (
            abs(up_pnl - down_pnl) / abs(total) if total else 0.0
        ),
    }


def _up_side_block_reason(side_imbalance: dict[str, Any]) -> str:
    if not bool(side_imbalance["up_side_negative_pnl"]):
        return "up_side_replay_pnl_non_negative"
    return (
        "UP-side replay PnL is negative while aggregate PnL is carried by DOWN; "
        "promotion discussion should remain blocked until two-sided profitability "
        "is demonstrated."
    )


def _audit_findings(
    *,
    accumulation: dict[str, Any],
    failed_runs: list[dict[str, Any]],
    side_imbalance: dict[str, Any],
    leave_one_out: dict[str, Any],
) -> list[str]:
    findings = []
    if accumulation.get("support_gate_passed") is True:
        findings.append("support_gate_passed")
    else:
        findings.append("support_gate_not_passed")
    if accumulation.get("promotion_evidence_eligible") is not True:
        findings.append("promotion_evidence_not_eligible")
    if failed_runs:
        findings.append("one_or_more_included_holdout_runs_failed_validation")
    if bool(side_imbalance["up_side_negative_pnl"]):
        findings.append("up_side_replay_pnl_negative")
    if bool(side_imbalance["down_side_negative_pnl"]):
        findings.append("down_side_replay_pnl_negative")
    if not bool(leave_one_out["total_pnl_positive_after_removing_any_single_entry"]):
        findings.append("leave_one_out_total_pnl_not_robust_to_largest_winner")
    return findings


def _evidence_strength(
    *,
    accumulation: dict[str, Any],
    failed_runs: list[dict[str, Any]],
    side_imbalance: dict[str, Any],
    leave_one_out: dict[str, Any],
) -> str:
    if accumulation.get("support_gate_passed") is not True:
        return "insufficient"
    if (
        accumulation.get("promotion_evidence_eligible") is True
        and not failed_runs
        and not bool(side_imbalance["up_side_negative_pnl"])
        and not bool(side_imbalance["down_side_negative_pnl"])
        and bool(leave_one_out["total_pnl_positive_after_removing_any_single_entry"])
    ):
        return "strong"
    return "weak"


def _top_negative_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        entry for entry in entries if float(entry.get("total_polymarket_pnl", 0.0)) < 0.0
    ]
    rows.sort(
        key=lambda row: (
            float(row["total_polymarket_pnl"]),
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
        )
    )
    return rows[:10]


def _audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M Post-Freeze Promotion Readiness Audit",
        "",
        f"- promotion_readiness: `{report['promotion_readiness']}`",
        f"- support_gate_passed: `{str(report['support_gate_passed']).lower()}`",
        "- promotion_evidence_eligible: "
        f"`{str(report['promotion_evidence_eligible']).lower()}`",
        "- source_model_candidate_eligible: "
        f"`{str(report['source_model_candidate_eligible']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        f"- holdout_run_count: `{report['holdout_run_count']}`",
        f"- replay_entry_count: `{report['replay_entry_count']}`",
        f"- replay_unique_market_count: `{report['replay_unique_market_count']}`",
        f"- replay_total_pnl_sum: `{report['replay_total_pnl_sum']}`",
        "",
        "## Blockers",
        "",
        *[
            f"- promotion gate: `{reason}`"
            for reason in report["promotion_gate_reason_codes"]
        ],
        *[
            f"- source model: `{reason}`"
            for reason in report["source_model_candidate_ineligible_reason_codes"]
        ],
        "",
        "## Side PnL",
        "",
        "| side | entries | pnl |",
        "|---|---:|---:|",
    ]
    side_counts = report["replay_entry_count_by_side"]
    pnl_by_side = report["replay_pnl_by_side"]
    for side in ("UP", "DOWN"):
        entries = int(side_counts.get(side, 0))
        pnl = float(pnl_by_side.get(side, 0.0))
        lines.append(
            f"| {side} | {entries} | {pnl:.6f} |"
        )
    stats = report["pnl_per_replay_entry_stats"]
    lines.extend(
        [
            "",
            "## Entry PnL Stats",
            "",
            f"- minimum: `{stats['minimum']}`",
            f"- median: `{stats['median']}`",
            f"- mean: `{stats['mean']}`",
            "- total_pnl_after_largest_positive_entry_removed: "
            f"`{report['largest_positive_entry_removed_total_pnl']}`",
            "- total_pnl_remains_positive_if_largest_positive_entry_removed: "
            f"`{str(report['total_pnl_remains_positive_if_largest_positive_entry_removed']).lower()}`",
            "",
            "## Failed Included Runs",
            "",
        ]
    )
    failed = report["included_runs_with_holdout_validation_passed_false"]
    if failed:
        lines.extend(
            [
                "| report | replay_entries | pnl | side |",
                "|---|---:|---:|---|",
                *[
                    "| {report_path} | {entries} | {pnl:.6f} | {side} |".format(
                        report_path=Path(str(row.get("report_path"))).name,
                        entries=int(row.get("replay_entry_count", 0)),
                        pnl=float(row.get("replay_total_pnl_sum", 0.0)),
                        side=row.get("dominant_replay_side"),
                    )
                    for row in failed[:20]
                ],
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- {report['up_side_negative_pnl_block_reason']}",
            "- This audit is diagnostic-only and does not enable paper/live.",
            "",
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
