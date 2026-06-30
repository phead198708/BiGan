"""Accumulation report for post-freeze M holdout validation evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
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
    FROZEN_M_SELECTOR_BASELINE_COMMIT,
    FROZEN_M_SELECTOR_METHOD,
    M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-m-post-freeze-holdout-accumulation-v1"
)
DEFAULT_MIN_REPLAY_ENTRY_SUPPORT = 20
DEFAULT_MIN_UNIQUE_MARKET_SUPPORT = 10


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezeHoldoutAccumulationConfig:
    """Configuration for aggregating frozen M post-freeze holdout reports."""

    holdout_report_paths: tuple[Path | str, ...]
    output_dir: Path | str
    run_id: str = "polymarket_m_post_freeze_holdout_accumulation"
    min_replay_entry_support: int = DEFAULT_MIN_REPLAY_ENTRY_SUPPORT
    min_unique_market_support: int = DEFAULT_MIN_UNIQUE_MARKET_SUPPORT
    require_both_side_replay_entries: bool = True
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "holdout_report_paths",
            tuple(Path(path) for path in self.holdout_report_paths),
        )
        if not self.holdout_report_paths:
            raise ValueError("holdout_report_paths must not be empty")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.min_replay_entry_support <= 0:
            raise ValueError("min_replay_entry_support must be positive")
        if self.min_unique_market_support <= 0:
            raise ValueError("min_unique_market_support must be positive")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id


@dataclass(frozen=True, slots=True)
class PolymarketPostFreezeHoldoutAccumulationResult:
    run_dir: Path
    report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_m_post_freeze_holdout_accumulation(
    config: PolymarketPostFreezeHoldoutAccumulationConfig,
) -> PolymarketPostFreezeHoldoutAccumulationResult:
    """Aggregate existing post-freeze holdout validation reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "report": run_dir / "m_post_freeze_holdout_accumulation_report.json",
        "summary": run_dir / "m_post_freeze_holdout_accumulation_report.md",
        "manifest": run_dir / "m_post_freeze_holdout_accumulation_manifest.json",
    }
    report = _build_accumulation_report(config=config)
    _write_json(artifact_paths["report"], report)
    artifact_paths["summary"].write_text(
        _accumulation_markdown(report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "bigan-v8-polymarket-m-post-freeze-holdout-accumulation-artifacts-v1"
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
    return PolymarketPostFreezeHoldoutAccumulationResult(
        run_dir=run_dir,
        report=report,
        artifact_paths=artifact_paths,
    )


def _build_accumulation_report(
    *,
    config: PolymarketPostFreezeHoldoutAccumulationConfig,
) -> dict[str, Any]:
    loaded_reports = [
        _load_report(path.expanduser().resolve())
        for path in config.holdout_report_paths
    ]
    included = [
        item for item in loaded_reports if _counts_toward_promotion_evidence(item["report"])
    ]
    included, duplicate_excluded = _dedupe_included_reports(included)
    excluded = [
        _excluded_run_summary(item)
        for item in loaded_reports
        if not _counts_toward_promotion_evidence(item["report"])
    ]
    blocked = [
        _blocked_run_summary(item)
        for item in loaded_reports
        if item["report"].get("validation_status") == "blocked_fail_closed"
    ]
    included_reports = [item["report"] for item in included]
    candidate_rows = [
        row
        for report in included_reports
        for row in report.get("rows", [])
    ]
    selected_rows = [
        row
        for row in candidate_rows
        if bool(row.get("side_quota_selected", False))
    ]
    replay_entry_rows = [
        row
        for row in candidate_rows
        if bool(row.get("entry_order_opened", False))
    ]
    candidate_market_ids = _row_market_ids(candidate_rows)
    selected_market_ids = _row_market_ids(selected_rows)
    replay_market_ids = _row_market_ids(replay_entry_rows)
    side_counts = _side_counts(replay_entry_rows)
    replay_pnl_by_side = _sum_side_maps(
        report.get("replay_pnl_by_side", {}) for report in included_reports
    )
    replay_total_pnl_sum = sum(
        float(report.get("replay_total_pnl_sum", 0.0))
        for report in included_reports
    )
    replay_entry_count = sum(
        int(report.get("replay_entry_count", 0))
        for report in included_reports
    )
    selected_entry_count = sum(
        int(report.get("selected_entry_count", 0))
        for report in included_reports
    )
    label_vs_replay_pnl_gap = sum(
        float(report.get("label_vs_replay_pnl_gap", 0.0))
        for report in included_reports
    )
    support_gate_reason_codes = _support_gate_reason_codes(
        holdout_run_count=len(included_reports),
        replay_unique_market_count=len(replay_market_ids),
        replay_entry_count=replay_entry_count,
        replay_entry_count_by_side=side_counts,
        config=config,
    )
    promotion_gate_reason_codes = list(support_gate_reason_codes)
    if replay_total_pnl_sum <= 0.0 and included_reports:
        promotion_gate_reason_codes.append("non_positive_accumulated_replay_pnl")
    if any(not bool(report.get("holdout_validation_passed", False)) for report in included_reports):
        promotion_gate_reason_codes.append("included_holdout_validation_not_passed")
    if not included_reports:
        promotion_gate_reason_codes.append("no_true_post_freeze_holdout_runs")
    promotion_gate_reason_codes = _dedupe(promotion_gate_reason_codes)
    promotion_evidence_eligible = not promotion_gate_reason_codes
    report = {
        "schema_version": M_POST_FREEZE_HOLDOUT_ACCUMULATION_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        "report_type": "m_post_freeze_holdout_accumulation",
        "diagnostic_only": True,
        "loaded_report_count": len(loaded_reports),
        "holdout_run_count": len(included_reports),
        "excluded_run_count": len(excluded),
        "duplicate_excluded_run_count": len(duplicate_excluded),
        "failed_provenance_run_count": len(blocked),
        "candidate_market_count": len(candidate_market_ids),
        "selected_market_count": len(selected_market_ids),
        "replay_unique_market_count": len(replay_market_ids),
        "unique_market_count": len(replay_market_ids),
        "selected_entry_count": selected_entry_count,
        "replay_entry_count": replay_entry_count,
        "replay_entry_count_by_side": side_counts,
        "replay_total_pnl_sum": replay_total_pnl_sum,
        "replay_pnl_by_side": replay_pnl_by_side,
        "mean_pnl_per_entry": (
            replay_total_pnl_sum / replay_entry_count
            if replay_entry_count > 0
            else 0.0
        ),
        "label_vs_replay_pnl_gap": label_vs_replay_pnl_gap,
        "support_gate_thresholds": {
            "min_replay_entry_support": config.min_replay_entry_support,
            "min_unique_market_support": config.min_unique_market_support,
            "require_both_side_replay_entries": (
                config.require_both_side_replay_entries
            ),
        },
        "support_gate_passed": not support_gate_reason_codes,
        "support_gate_reason_codes": support_gate_reason_codes,
        "promotion_evidence_eligible": promotion_evidence_eligible,
        "promotion_gate_reason_codes": promotion_gate_reason_codes,
        "source_model_candidate_eligible": False,
        "source_model_candidate_ineligible_reason_codes": (
            []
            if promotion_evidence_eligible
            else list(promotion_gate_reason_codes)
        )
        + ["accumulation_report_diagnostic_only_no_source_eligibility_unlock"],
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "paper_run_resume_allowed": False,
        "selector_method": FROZEN_M_SELECTOR_METHOD,
        "baseline_selector_commit": FROZEN_M_SELECTOR_BASELINE_COMMIT,
        "selector_weights_changed": False,
        "rank_weight_tuning_performed": False,
        "holdout_feedback_used_for_tuning": False,
        "only_true_post_freeze_holdout_runs_counted": True,
        "duplicate_holdout_runs_excluded_from_pnl": True,
        "blocked_provenance_runs_excluded_from_pnl": True,
        "included_runs": [_included_run_summary(item) for item in included],
        "excluded_runs": excluded,
        "duplicate_excluded_runs": duplicate_excluded,
        "blocked_provenance_runs": blocked,
        "top_negative_replay_entries": _top_negative_entries(replay_entry_rows),
        **compact_safety_fields(),
    }
    report["m_post_freeze_holdout_accumulation_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _load_report(path: Path) -> dict[str, Any]:
    report_path = _resolve_report_path(path)
    report = _read_json(report_path)
    return {
        "path": str(report_path),
        "path_sha256": _sha256_file(report_path),
        "report": report,
    }


def _resolve_report_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "m_post_freeze_holdout_validation_report.json"
    if not path.exists():
        raise FileNotFoundError(f"holdout report does not exist: {path}")
    return path


def _counts_toward_promotion_evidence(report: dict[str, Any]) -> bool:
    return (
        report.get("schema_version") == M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION
        and report.get("validation_status") == "completed"
        and report.get("true_post_freeze_holdout") is True
        and report.get("prediction_attempted") is True
    )


def _dedupe_included_reports(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for item in items:
        duplicate_of = _find_duplicate(item, unique)
        if duplicate_of is None:
            unique.append(item)
            continue
        duplicates.append(_duplicate_run_summary(item, duplicate_of))
    return unique, duplicates


def _find_duplicate(
    item: dict[str, Any],
    unique_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for existing in unique_items:
        if _duplicate_reason_codes(item, existing):
            return existing
    return None


def _duplicate_reason_codes(
    item: dict[str, Any],
    existing: dict[str, Any],
) -> list[str]:
    identity = _dedupe_identity(item)
    existing_identity = _dedupe_identity(existing)
    reason_codes = []
    if (
        identity["report_sha256"]
        and identity["report_sha256"] == existing_identity["report_sha256"]
    ):
        reason_codes.append("duplicate_report_sha256")
    if (
        identity["report_id"]
        and identity["report_id"] == existing_identity["report_id"]
    ):
        reason_codes.append("duplicate_report_id")
    if identity["run_id"] and identity["run_id"] == existing_identity["run_id"]:
        reason_codes.append("duplicate_run_id")
    if (
        identity["holdout_corpus_manifest_sha256"]
        and identity["holdout_corpus_manifest_sha256"]
        == existing_identity["holdout_corpus_manifest_sha256"]
        and identity["holdout_window"] == existing_identity["holdout_window"]
        and identity["market_ids"] == existing_identity["market_ids"]
    ):
        reason_codes.append("duplicate_holdout_corpus_window_market_ids")
    return reason_codes


def _dedupe_identity(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    provenance = dict(report.get("provenance", {}))
    return {
        "report_sha256": item["path_sha256"],
        "report_id": str(
            report.get("m_post_freeze_holdout_validation_report_id") or ""
        ),
        "run_id": str(report.get("run_id") or ""),
        "holdout_corpus_manifest_sha256": str(
            provenance.get("holdout_corpus_manifest_sha256")
            or provenance.get("holdout_phase2_corpus_manifest_sha256")
            or provenance.get("holdout_training_corpus_hash")
            or ""
        ),
        "holdout_window": (
            provenance.get("holdout_min_decision_ts"),
            provenance.get("holdout_max_decision_ts"),
        ),
        "market_ids": _holdout_market_ids(report),
    }


def _holdout_market_ids(report: dict[str, Any]) -> tuple[str, ...]:
    provenance = dict(report.get("provenance", {}))
    market_ids = provenance.get("holdout_market_ids")
    if isinstance(market_ids, list):
        return tuple(sorted(str(market_id) for market_id in market_ids))
    row_market_ids = {
        str(row.get("market_id"))
        for row in report.get("rows", [])
        if row.get("market_id")
    }
    return tuple(sorted(row_market_ids))


def _duplicate_run_summary(
    item: dict[str, Any],
    duplicate_of: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_path": item["path"],
        "report_sha256": item["path_sha256"],
        "duplicate_of_report_path": duplicate_of["path"],
        "duplicate_of_report_sha256": duplicate_of["path_sha256"],
        "duplicate_reason_codes": _duplicate_reason_codes(item, duplicate_of),
        "dedupe_identity": _dedupe_identity(item),
        "duplicate_of_dedupe_identity": _dedupe_identity(duplicate_of),
    }


def _included_run_summary(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    provenance = dict(report.get("provenance", {}))
    rows = list(report.get("rows", []))
    replay_rows = [row for row in rows if bool(row.get("entry_order_opened", False))]
    selected_rows = [
        row for row in rows if bool(row.get("side_quota_selected", False))
    ]
    return {
        "report_path": item["path"],
        "report_sha256": item["path_sha256"],
        "dedupe_identity": _dedupe_identity(item),
        "validation_status": report.get("validation_status"),
        "true_post_freeze_holdout": report.get("true_post_freeze_holdout"),
        "prediction_attempted": report.get("prediction_attempted"),
        "holdout_validation_passed": report.get("holdout_validation_passed"),
        "holdout_corpus_dir": provenance.get("holdout_corpus_dir"),
        "holdout_dataset_hash": provenance.get("holdout_dataset_hash"),
        "holdout_training_corpus_hash": provenance.get("holdout_training_corpus_hash"),
        "holdout_min_decision_ts": provenance.get("holdout_min_decision_ts"),
        "holdout_max_decision_ts": provenance.get("holdout_max_decision_ts"),
        "market_id_overlap_count": provenance.get("market_id_overlap_count"),
        "candidate_market_count": len(_row_market_ids(rows)),
        "selected_market_count": len(_row_market_ids(selected_rows)),
        "replay_unique_market_count": len(_row_market_ids(replay_rows)),
        "selected_entry_count": int(report.get("selected_entry_count", 0)),
        "replay_entry_count": int(report.get("replay_entry_count", 0)),
        "selected_exit_decision_count": int(
            report.get("selected_exit_decision_count", 0)
        ),
        "replay_total_pnl_sum": float(report.get("replay_total_pnl_sum", 0.0)),
        "replay_pnl_by_side": dict(report.get("replay_pnl_by_side", {})),
        "label_vs_replay_pnl_gap": float(
            report.get("label_vs_replay_pnl_gap", 0.0)
        ),
        "reason_codes": list(report.get("reason_codes", [])),
        "ineligible_reason_codes": list(report.get("ineligible_reason_codes", [])),
    }


def _excluded_run_summary(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    return {
        "report_path": item["path"],
        "report_sha256": item["path_sha256"],
        "validation_status": report.get("validation_status"),
        "true_post_freeze_holdout": report.get("true_post_freeze_holdout"),
        "prediction_attempted": report.get("prediction_attempted"),
        "excluded_reason_codes": _excluded_reason_codes(report),
        "reason_codes": list(report.get("reason_codes", [])),
        "ineligible_reason_codes": list(report.get("ineligible_reason_codes", [])),
    }


def _blocked_run_summary(item: dict[str, Any]) -> dict[str, Any]:
    report = item["report"]
    return {
        "report_path": item["path"],
        "report_sha256": item["path_sha256"],
        "validation_status": report.get("validation_status"),
        "prediction_attempted": report.get("prediction_attempted"),
        "reason_codes": list(report.get("reason_codes", [])),
        "ineligible_reason_codes": list(report.get("ineligible_reason_codes", [])),
    }


def _excluded_reason_codes(report: dict[str, Any]) -> list[str]:
    reason_codes = []
    if report.get("schema_version") != M_POST_FREEZE_HOLDOUT_SCHEMA_VERSION:
        reason_codes.append("invalid_holdout_report_schema")
    if report.get("validation_status") != "completed":
        reason_codes.append("holdout_validation_not_completed")
    if report.get("true_post_freeze_holdout") is not True:
        reason_codes.append("not_true_post_freeze_holdout")
    if report.get("prediction_attempted") is not True:
        reason_codes.append("prediction_not_attempted")
    return reason_codes


def _support_gate_reason_codes(
    *,
    holdout_run_count: int,
    replay_unique_market_count: int,
    replay_entry_count: int,
    replay_entry_count_by_side: dict[str, int],
    config: PolymarketPostFreezeHoldoutAccumulationConfig,
) -> list[str]:
    reason_codes = []
    if holdout_run_count <= 0:
        reason_codes.append("no_true_post_freeze_holdout_runs")
    if replay_entry_count < config.min_replay_entry_support:
        reason_codes.append("insufficient_replay_entry_support")
    if replay_unique_market_count < config.min_unique_market_support:
        reason_codes.append("insufficient_unique_market_support")
    if config.require_both_side_replay_entries:
        if replay_entry_count_by_side.get("UP", 0) <= 0:
            reason_codes.append("missing_up_replay_entry_support")
        if replay_entry_count_by_side.get("DOWN", 0) <= 0:
            reason_codes.append("missing_down_replay_entry_support")
    return reason_codes


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _side_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("selected_side")) for row in rows)
    return {side: int(counts.get(side, 0)) for side in ("UP", "DOWN")}


def _row_market_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("market_id")) for row in rows if row.get("market_id")}


def _sum_side_maps(maps: Any) -> dict[str, float]:
    totals = {"UP": 0.0, "DOWN": 0.0}
    for payload in maps:
        for side in totals:
            totals[side] += float(payload.get(side, 0.0))
    return totals


def _top_negative_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    negative_rows = [
        row for row in rows if float(row.get("total_polymarket_pnl", 0.0)) < 0.0
    ]
    negative_rows.sort(
        key=lambda row: (
            float(row.get("total_polymarket_pnl", 0.0)),
            int(row.get("decision_ts", 0)),
            str(row.get("market_id", "")),
        )
    )
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
    return [{field: row.get(field) for field in fields} for row in negative_rows[:10]]


def _accumulation_markdown(report: dict[str, Any]) -> str:
    support_reasons = report.get("support_gate_reason_codes", [])
    lines = [
        "# M Post-Freeze Holdout Accumulation",
        "",
        f"- loaded_report_count: `{report['loaded_report_count']}`",
        f"- holdout_run_count: `{report['holdout_run_count']}`",
        f"- duplicate_excluded_run_count: `{report['duplicate_excluded_run_count']}`",
        f"- candidate_market_count: `{report['candidate_market_count']}`",
        f"- selected_market_count: `{report['selected_market_count']}`",
        f"- replay_unique_market_count: `{report['replay_unique_market_count']}`",
        f"- selected_entry_count: `{report['selected_entry_count']}`",
        f"- replay_entry_count: `{report['replay_entry_count']}`",
        f"- replay_total_pnl_sum: `{report['replay_total_pnl_sum']}`",
        f"- mean_pnl_per_entry: `{report['mean_pnl_per_entry']}`",
        f"- label_vs_replay_pnl_gap: `{report['label_vs_replay_pnl_gap']}`",
        f"- failed_provenance_run_count: `{report['failed_provenance_run_count']}`",
        f"- support_gate_passed: `{str(report['support_gate_passed']).lower()}`",
        "- source_model_candidate_eligible: "
        f"`{str(report['source_model_candidate_eligible']).lower()}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "## Replay By Side",
        "",
        "| side | entries | pnl |",
        "|---|---:|---:|",
    ]
    for side in ("UP", "DOWN"):
        lines.append(
            "| {side} | {entries} | {pnl:.6f} |".format(
                side=side,
                entries=report["replay_entry_count_by_side"].get(side, 0),
                pnl=float(report["replay_pnl_by_side"].get(side, 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Support Gate Reason Codes",
            "",
            *[f"- `{reason}`" for reason in support_reasons],
            "",
            "## Duplicate Excluded Runs",
            "",
        ]
    )
    if report["duplicate_excluded_runs"]:
        lines.extend(
            [
                "| report | duplicate_of | reason_codes |",
                "|---|---|---|",
                *[
                    "| {report_path} | {duplicate_of} | {reasons} |".format(
                        report_path=Path(row["report_path"]).name,
                        duplicate_of=Path(row["duplicate_of_report_path"]).name,
                        reasons=", ".join(row.get("duplicate_reason_codes", [])),
                    )
                    for row in report["duplicate_excluded_runs"]
                ],
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Excluded Runs",
            "",
        ]
    )
    if report["excluded_runs"]:
        lines.extend(
            [
                "| report | status | reason_codes |",
                "|---|---|---|",
                *[
                    "| {report_path} | {status} | {reasons} |".format(
                        report_path=Path(row["report_path"]).name,
                        status=row.get("validation_status"),
                        reasons=", ".join(row.get("excluded_reason_codes", [])),
                    )
                    for row in report["excluded_runs"]
                ],
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
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
