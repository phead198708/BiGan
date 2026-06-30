"""Diagnostic O replay-aligned source-ranking reports."""

from __future__ import annotations

import shutil
import statistics
from collections import Counter, defaultdict
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
    N2_FORBIDDEN_SELECTION_FIELDS,
)
from bigan.v8.polymarket.training.post_freeze_up_diagnostics import (
    _label,
    _pnl,
    _read_json,
    _score,
    _sha256_file,
    _write_json,
)
from bigan.v8.polymarket.training.sell_before_close_source_candidates import (
    REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
    SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
)

O_LABEL_CONSTRUCTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-replay-aligned-label-construction-v1"
)
O_SOURCE_RANKING_OBJECTIVE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-source-ranking-objective-v1"
)
O_FEATURE_AND_LABEL_LEAKAGE_AUDIT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-feature-and-label-leakage-audit-v1"
)
O_SOURCE_CANDIDATE_COMPARISON_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-source-candidate-comparison-v1"
)
O_MODEL_INPUT_FIELDS = (
    "raw_calibrated_action_score",
    "best_action_margin",
    "entry_quality_ask",
    "exit_quality_bid",
    "execution_pnl_immediate_exit_pnl",
    "execution_pnl_immediate_exit_return",
    "entry_exit_quality_spread_bps",
    "entry_exit_quality_queue_fill",
    "entry_exit_quality_book_staleness_ms",
    "entry_exit_quality_time_to_close_seconds",
    "up_recent_book_update_count_1m",
    "down_recent_book_update_count_1m",
)
O_TRAINING_LABEL_FIELDS = (
    "action_return_target",
    "label_pnl_target",
    "exit_quality_bid",
    "execution_pnl_immediate_exit_pnl",
    "execution_pnl_immediate_exit_return",
)
O_REPORT_ONLY_EVALUATION_FIELDS = (
    "realized_trade_pnl",
    "settlement_pnl",
    "total_polymarket_pnl",
)
O_FORBIDDEN_MODEL_INPUT_FIELDS = (
    *N2_FORBIDDEN_SELECTION_FIELDS,
    "future_exit_reason_codes",
    "post_entry_close_state",
    "post_settlement_values",
)
O_VARIANTS = (
    "current_source_baseline",
    "o_replay_aligned_labels_only",
    "o_replay_aligned_labels_family_priors",
    "o_replay_aligned_pairwise_listwise_correction",
    "o_replay_aligned_stronger_no_trade_prior",
)
O_REQUIRED_DECISION_ACTION_FAMILIES = (
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "NO_TRADE",
)
O_FULL_DECISION_GROUP_SCOPE = "full_decision_group"
O_PARTIAL_DECISION_GROUP_SCOPE = "partial_decision_group_diagnostic"


@dataclass(frozen=True, slots=True)
class PolymarketOReplayAlignedSourceRankingConfig:
    """Configuration for diagnostic-only O replay-aligned source ranking."""

    m2_candidate_report_path: Path | str
    output_dir: Path | str
    run_id: str = "polymarket_o_replay_aligned_source_ranking"
    overwrite_existing: bool = False
    high_score_threshold: float = 0.75
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
class PolymarketOReplayAlignedSourceRankingResult:
    run_dir: Path
    label_construction_report: dict[str, Any]
    ranking_objective_report: dict[str, Any]
    leakage_audit_report: dict[str, Any]
    candidate_comparison_report: dict[str, Any]
    artifact_paths: dict[str, Path]


def run_polymarket_o_replay_aligned_source_ranking(
    config: PolymarketOReplayAlignedSourceRankingConfig,
) -> PolymarketOReplayAlignedSourceRankingResult:
    """Build diagnostic-only O source-ranking reports."""

    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run_dir already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "label_construction_report": run_dir
        / "o_replay_aligned_label_construction_report.json",
        "label_construction_summary": run_dir
        / "o_replay_aligned_label_construction_report.md",
        "ranking_objective_report": run_dir
        / "o_source_ranking_objective_report.json",
        "ranking_objective_summary": run_dir
        / "o_source_ranking_objective_report.md",
        "leakage_audit_report": run_dir / "o_feature_and_label_leakage_audit.json",
        "leakage_audit_summary": run_dir / "o_feature_and_label_leakage_audit.md",
        "candidate_comparison_report": run_dir
        / "o_source_candidate_comparison_report.json",
        "candidate_comparison_summary": run_dir
        / "o_source_candidate_comparison_report.md",
        "manifest": run_dir / "o_replay_aligned_source_ranking_manifest.json",
    }
    reports = _build_reports(config=config)
    _write_json(artifact_paths["label_construction_report"], reports[0])
    artifact_paths["label_construction_summary"].write_text(
        _label_markdown(reports[0]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["ranking_objective_report"], reports[1])
    artifact_paths["ranking_objective_summary"].write_text(
        _ranking_markdown(reports[1]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["leakage_audit_report"], reports[2])
    artifact_paths["leakage_audit_summary"].write_text(
        _leakage_markdown(reports[2]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["candidate_comparison_report"], reports[3])
    artifact_paths["candidate_comparison_summary"].write_text(
        _comparison_markdown(reports[3]),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "bigan-v8-polymarket-o-replay-aligned-source-ranking-artifacts-v1",
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
    return PolymarketOReplayAlignedSourceRankingResult(
        run_dir=run_dir,
        label_construction_report=reports[0],
        ranking_objective_report=reports[1],
        leakage_audit_report=reports[2],
        candidate_comparison_report=reports[3],
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    m2_report_path = config.m2_candidate_report_path.expanduser().resolve()
    m2_report = _read_json(m2_report_path)
    if m2_report.get("schema_version") != M2_REPLAY_PARITY_SCHEMA_VERSION:
        raise ValueError("not an M2 replay-parity candidate report")
    rows, source_reports, label_lookup = _load_source_rows(m2_report)
    action_rows = _build_complete_decision_action_rows(
        rows=rows,
        label_lookup=label_lookup,
    )
    grouped = _groups_with_required_actions(action_rows)
    labeled_rows = _construct_replay_aligned_labels(grouped)
    ranking_rows = _ranking_rows(labeled_rows)
    label_report = _label_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        source_reports=source_reports,
        rows=labeled_rows,
    )
    ranking_report = _ranking_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
    )
    leakage_report = _leakage_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=labeled_rows,
    )
    comparison_report = _comparison_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
    )
    return label_report, ranking_report, leakage_report, comparison_report


def _load_source_rows(
    m2_report: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, int, str], dict[str, Any]],
]:
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
    source_reports = []
    label_lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    seen: set[tuple[str, str, int, str]] = set()
    for path_text in paths:
        path = Path(path_text).expanduser().resolve()
        report = _read_json(path)
        source_label_rows, source_label_path = _load_source_label_rows(report)
        for label_row in source_label_rows:
            label_lookup[
                (
                    str(path),
                    str(label_row.get("market_id") or ""),
                    int(label_row.get("decision_ts") or 0),
                    str(label_row.get("action") or ""),
                )
            ] = label_row
        source_reports.append(
            {
                "source_report_path": str(path),
                "source_report_sha256": _sha256_file(path),
                "run_id": report.get("run_id"),
                "row_count": len(report.get("rows", [])),
                "holdout_corpus_dir": report.get("provenance", {}).get(
                    "holdout_corpus_dir"
                ),
                "label_rows_path": str(source_label_path) if source_label_path else None,
                "label_row_count": len(source_label_rows),
                "full_action_label_rows_available": bool(source_label_rows),
            }
        )
        for row in report.get("rows", []):
            payload = dict(row)
            payload["source_report_path"] = str(path)
            key = (
                str(payload.get("source_report_path") or ""),
                str(payload.get("market_id") or ""),
                int(payload.get("decision_ts") or 0),
                str(payload.get("action") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(payload)
    return rows, source_reports, label_lookup


def _load_source_label_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    corpus_dir_text = str(report.get("provenance", {}).get("holdout_corpus_dir") or "")
    if not corpus_dir_text:
        return [], None
    label_path = Path(corpus_dir_text).expanduser().resolve() / "polymarket_label_rows.jsonl"
    if not label_path.exists():
        return [], label_path
    return _read_jsonl(label_path), label_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        _read_json_line(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def _build_complete_decision_action_rows(
    *,
    rows: list[dict[str, Any]],
    label_lookup: dict[tuple[str, str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        contexts[
            (
                str(row.get("source_report_path") or ""),
                str(row.get("market_id") or ""),
                int(row.get("decision_ts") or 0),
            )
        ].append(row)
    action_rows = []
    for context_key, context_rows in sorted(contexts.items()):
        source_report_path, market_id, decision_ts = context_key
        template = _decision_template(context_rows)
        observed_by_action = {
            str(row.get("action") or ""): row
            for row in context_rows
            if row.get("action")
        }
        for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
            label_row = label_lookup.get(
                (source_report_path, market_id, decision_ts, action)
            )
            observed_row = observed_by_action.get(action)
            action_rows.append(
                _normalize_action_row(
                    _candidate_row_from_label_or_template(
                        template=template,
                        action=action,
                        label_row=label_row,
                        observed_row=observed_row,
                    )
                )
            )
    return action_rows


def _decision_template(context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        context_rows,
        key=lambda row: (
            bool(row.get("side_quota_selected")),
            bool(row.get("entry_order_opened")),
            float(row.get("raw_calibrated_action_score") or 0.0),
        ),
    )


def _candidate_row_from_label_or_template(
    *,
    template: dict[str, Any],
    action: str,
    label_row: dict[str, Any] | None,
    observed_row: dict[str, Any] | None,
) -> dict[str, Any]:
    base = dict(template)
    base["action"] = action
    base["selected_side"] = _side_from_action(action)
    base["candidate_observation_type"] = (
        "observed_replay_action" if observed_row is not None else "counterfactual_action"
    )
    base["observed_replay_action"] = observed_row is not None
    if observed_row is not None:
        base["observed_total_polymarket_pnl"] = _pnl(observed_row)
        base["observed_replay_reason_codes"] = observed_row.get("replay_reason_codes", [])
    if label_row is None:
        base["candidate_label_source"] = "synthetic_missing_label_fallback"
        base["label_candidate_available"] = False
        base["source_score_available"] = observed_row is not None
        if observed_row is not None:
            base.update(observed_row)
            base["candidate_observation_type"] = "observed_replay_action"
            base["observed_replay_action"] = True
        else:
            base["action_return_target"] = 0.0
            base["label_pnl_target"] = 0.0
            base["total_polymarket_pnl"] = 0.0
            base["raw_calibrated_action_score"] = 0.0
            base["best_action_margin"] = 0.0
        return base
    label_target = _label_target_from_corpus_label(label_row)
    base.update(
        {
            "candidate_label_source": "holdout_corpus_label_rows",
            "label_candidate_available": True,
            "source_score_available": observed_row is not None,
            "action_return_target": label_target,
            "label_pnl_target": label_target,
            "total_polymarket_pnl": label_target,
            "realized_trade_pnl": float(label_row.get("realized_trade_return") or 0.0),
            "settlement_pnl": float(label_row.get("settlement_return") or 0.0),
            "entry_quality_ask": float(label_row.get("entry_ask") or 0.0),
            "exit_quality_bid": float(label_row.get("exit_bid") or 0.0),
            "execution_pnl_immediate_exit_pnl": label_target,
            "execution_pnl_immediate_exit_return": float(
                label_row.get("total_net_return") or label_target
            ),
            "sell_before_close_execution_class": label_row.get(
                "sell_before_close_execution_class"
            ),
            "label_uses_executable_exit_path": bool(
                label_row.get("label_uses_executable_exit_path")
            ),
            "queue_fill_probability_estimate": float(
                label_row.get("queue_fill_probability_estimate") or 0.0
            ),
            "executable_liquidity_notional": float(
                label_row.get("executable_liquidity_notional") or 0.0
            ),
            "theoretical_terminal_bid_return": float(
                label_row.get("theoretical_terminal_bid_return") or 0.0
            ),
            "realized_executable_sell_before_close_return": float(
                label_row.get("realized_executable_sell_before_close_return") or 0.0
            ),
            "execution_gap_return": float(label_row.get("execution_gap_return") or 0.0),
        }
    )
    if observed_row is not None:
        base["raw_calibrated_action_score"] = _score(observed_row)
        base["best_action_margin"] = float(observed_row.get("best_action_margin") or 0.0)
    else:
        base["raw_calibrated_action_score"] = _counterfactual_source_score(
            template=template,
            action=action,
        )
        base["best_action_margin"] = 0.0
    return base


def _label_target_from_corpus_label(label_row: dict[str, Any]) -> float:
    if label_row.get("total_net_pnl_per_notional") is not None:
        return float(label_row["total_net_pnl_per_notional"])
    if label_row.get("total_net_return") is not None:
        return float(label_row["total_net_return"])
    return 0.0


def _counterfactual_source_score(
    *,
    template: dict[str, Any],
    action: str,
) -> float:
    if action == str(template.get("action") or ""):
        return _score(template)
    return 0.0


def _normalize_action_row(row: dict[str, Any]) -> dict[str, Any]:
    action = str(row.get("action") or "")
    side = str(row.get("selected_side") or _side_from_action(action))
    return {
        **row,
        "action": action,
        "action_family": _action_family(action),
        "selected_side": side,
        "decision_group_id": "|".join(
            (
                str(row.get("source_report_path") or ""),
                str(row.get("market_id") or ""),
                str(int(row.get("decision_ts") or 0)),
            )
        ),
        "original_label_target": _label(row),
        "realized_replay_return": _pnl(row),
        "baseline_source_score": _score(row),
    }


def _groups_with_required_actions(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["decision_group_id"]].append(row)
    for group_rows in groups.values():
        _annotate_decision_group_completeness(group_rows)
    return groups


def _annotate_decision_group_completeness(group_rows: list[dict[str, Any]]) -> None:
    available = sorted(
        {
            _decision_action_family(row)
            for row in group_rows
            if bool(row.get("label_candidate_available", True))
        }
    )
    missing = sorted(set(O_REQUIRED_DECISION_ACTION_FAMILIES).difference(available))
    complete = not missing
    scope = (
        O_FULL_DECISION_GROUP_SCOPE
        if complete
        else O_PARTIAL_DECISION_GROUP_SCOPE
    )
    for row in group_rows:
        row["decision_group_completeness"] = complete
        row["available_action_families"] = available
        row["missing_action_families"] = missing
        row["ranking_metric_scope"] = scope


def _construct_replay_aligned_labels(
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    labeled = []
    for group_id, rows in sorted(grouped.items()):
        base_labels = [_base_replay_label(row) for row in rows]
        best_base = max(base_labels) if base_labels else 0.0
        for row, base_label in zip(rows, base_labels, strict=True):
            opportunity_cost = max(0.0, best_base - base_label) * 0.10
            components = _label_components(row, base_label, opportunity_cost)
            replay_label = base_label - opportunity_cost
            labeled.append(
                {
                    **row,
                    "decision_group_id": group_id,
                    "replay_aligned_executable_label_target": replay_label,
                    "label_delta": replay_label - float(row["original_label_target"]),
                    "label_vs_realized_replay_gap_before": (
                        float(row["original_label_target"])
                        - float(row["realized_replay_return"])
                    ),
                    "label_vs_realized_replay_gap_after": (
                        replay_label - float(row["realized_replay_return"])
                    ),
                    "label_components": components,
                    "label_component_field_classes": _label_component_field_classes(),
                    "split": _split_for_group(group_id),
                    "time_to_close_bucket": _time_to_close_bucket(row),
                    "spread_bucket": _spread_bucket(row),
                    "queue_bucket": _queue_bucket(row),
                    "staleness_bucket": _staleness_bucket(row),
                }
            )
    return labeled


def _base_replay_label(row: dict[str, Any]) -> float:
    if row.get("candidate_label_source") == "holdout_corpus_label_rows":
        return float(row.get("original_label_target") or 0.0)
    if row.get("action") == "NO_TRADE":
        return 0.0
    original = float(row.get("original_label_target") or 0.0)
    immediate = _immediate_exit_pnl(row)
    spread_penalty = _spread_penalty(row)
    queue_penalty = _queue_penalty(row)
    staleness_penalty = _staleness_penalty(row)
    time_penalty = _time_penalty(row)
    no_trade_baseline = 0.0
    executable_exit_label = min(original, immediate) if immediate is not None else original
    return (
        executable_exit_label
        - spread_penalty
        - queue_penalty
        - staleness_penalty
        - time_penalty
        - no_trade_baseline
    )


def _label_components(
    row: dict[str, Any],
    base_label: float,
    opportunity_cost: float,
) -> dict[str, Any]:
    return {
        "entry_ask": _optional_float(row.get("entry_quality_ask")),
        "executable_entry_cost": _optional_float(row.get("entry_quality_ask")),
        "first_executable_exit_bid_after_entry": _optional_float(
            row.get("exit_quality_bid")
        ),
        "immediate_exit_downside_proxy": _immediate_exit_pnl(row),
        "spread_penalty": _spread_penalty(row),
        "queue_fill_penalty": _queue_penalty(row),
        "book_staleness_penalty": _staleness_penalty(row),
        "time_to_close_penalty": _time_penalty(row),
        "no_trade_baseline": 0.0,
        "action_family_opportunity_cost": opportunity_cost,
        "base_replay_aligned_label": base_label,
    }


def _label_component_field_classes() -> dict[str, str]:
    return {
        "entry_ask": "decision_time_available",
        "executable_entry_cost": "decision_time_available",
        "first_executable_exit_bid_after_entry": "replay_derived_label_only",
        "immediate_exit_downside_proxy": "decision_time_available",
        "spread_penalty": "decision_time_available",
        "queue_fill_penalty": "decision_time_available",
        "book_staleness_penalty": "decision_time_available",
        "time_to_close_penalty": "decision_time_available",
        "no_trade_baseline": "decision_time_available",
        "action_family_opportunity_cost": "replay_derived_label_only",
    }


def _ranking_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    ranking = []
    for group_rows in grouped.values():
        oracle = max(group_rows, key=lambda row: float(row["realized_replay_return"]))
        for row in group_rows:
            scores = _variant_scores(row, group_rows)
            ranking.append(
                {
                    **row,
                    "oracle_executable_best_action": oracle["action"],
                    "oracle_executable_best_action_family": oracle["action_family"],
                    "oracle_executable_best_action_return": float(
                        oracle["realized_replay_return"]
                    ),
                    "no_trade_opportunity_cost": max(
                        0.0,
                        float(oracle["realized_replay_return"]),
                    ),
                    "variant_scores": scores,
                }
            )
    return ranking


def _variant_scores(row: dict[str, Any], group_rows: list[dict[str, Any]]) -> dict[str, float]:
    label = float(row["replay_aligned_executable_label_target"])
    group_mean = statistics.mean(
        float(item["replay_aligned_executable_label_target"]) for item in group_rows
    )
    family_prior = 0.02 if row["action_family"] == "NO_TRADE" else -0.01
    return {
        "current_source_baseline": float(row["baseline_source_score"]),
        "o_replay_aligned_labels_only": label,
        "o_replay_aligned_labels_family_priors": label + family_prior,
        "o_replay_aligned_pairwise_listwise_correction": label - group_mean,
        "o_replay_aligned_stronger_no_trade_prior": (
            label + 0.08 if row["action_family"] == "NO_TRADE" else label - 0.03
        ),
    }


def _label_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    source_reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_LABEL_CONSTRUCTION_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "baseline_candidate_names": _baseline_names(),
        "report_type": "o_replay_aligned_label_construction",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "source_reports": source_reports,
        "row_count": len(rows),
        "decision_group_count": len({row["decision_group_id"] for row in rows}),
        "decision_group_completeness_summary": _decision_group_completeness_summary(
            rows
        ),
        "action_candidate_construction_summary": _action_candidate_construction_summary(
            rows
        ),
        "label_rows": [_compact_label_row(row) for row in rows],
        "label_component_field_classes": _label_component_field_classes(),
        "label_gap_before": sum(
            float(row["label_vs_realized_replay_gap_before"]) for row in rows
        ),
        "label_gap_after": sum(
            float(row["label_vs_realized_replay_gap_after"]) for row in rows
        ),
        "label_gap_delta": sum(
            float(row["label_vs_realized_replay_gap_before"])
            - float(row["label_vs_realized_replay_gap_after"])
            for row in rows
        ),
        "breakdown_by_action_family": _group_label_breakdown(rows, "action_family"),
        "breakdown_by_side": _group_label_breakdown(rows, "selected_side"),
        "breakdown_by_time_to_close_bucket": _group_label_breakdown(
            rows,
            "time_to_close_bucket",
        ),
        "breakdown_by_spread_bucket": _group_label_breakdown(rows, "spread_bucket"),
        "breakdown_by_queue_bucket": _group_label_breakdown(rows, "queue_bucket"),
        "breakdown_by_staleness_bucket": _group_label_breakdown(
            rows,
            "staleness_bucket",
        ),
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    del config
    report["o_replay_aligned_label_construction_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _ranking_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    variant_metrics = {
        variant: _ranking_metrics(rows, variant, config.high_score_threshold)
        for variant in O_VARIANTS
    }
    report = {
        "schema_version": O_SOURCE_RANKING_OBJECTIVE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "baseline_candidate_names": _baseline_names(),
        "report_type": "o_source_ranking_objective",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "ranking_metric_by_variant": variant_metrics,
        "primary_variant_name": "o_replay_aligned_labels_family_priors",
        "ranking_metric_scope": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["ranking_metric_scope"],
        "decision_group_completeness_summary": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["decision_group_completeness_summary"],
        "action_candidate_construction_summary": _action_candidate_construction_summary(
            rows
        ),
        "full_source_model_ranking_quality_claimed": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["full_source_model_ranking_quality_claimed"],
        "top1_realized_best_action_hit_rate": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["top1_realized_best_action_hit_rate"],
        "top2_realized_best_action_hit_rate": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["top2_realized_best_action_hit_rate"],
        "top3_realized_best_action_hit_rate": variant_metrics[
            "o_replay_aligned_labels_family_priors"
        ]["top3_realized_best_action_hit_rate"],
        "mean_regret": variant_metrics["o_replay_aligned_labels_family_priors"][
            "mean_regret"
        ],
        "ranking_rows": [
            _compact_ranking_row(row, "o_replay_aligned_labels_family_priors")
            for row in rows
        ],
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    report["o_source_ranking_objective_report_id"] = canonical_json_sha256(report)
    return report


def _leakage_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del config
    model_overlap = sorted(
        set(O_MODEL_INPUT_FIELDS).intersection(O_FORBIDDEN_MODEL_INPUT_FIELDS)
    )
    label_overlap = sorted(
        set(O_TRAINING_LABEL_FIELDS).intersection(O_FORBIDDEN_MODEL_INPUT_FIELDS)
    )
    report = {
        "schema_version": O_FEATURE_AND_LABEL_LEAKAGE_AUDIT_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "baseline_candidate_names": _baseline_names(),
        "report_type": "o_feature_and_label_leakage_audit",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "model_input_fields_decision_time_only": list(O_MODEL_INPUT_FIELDS),
        "training_label_fields_may_use_future_replay_or_settlement": list(
            O_TRAINING_LABEL_FIELDS
        ),
        "report_only_evaluation_fields": list(O_REPORT_ONLY_EVALUATION_FIELDS),
        "forbidden_model_input_fields": list(O_FORBIDDEN_MODEL_INPUT_FIELDS),
        "model_input_forbidden_field_overlap": model_overlap,
        "training_label_forbidden_field_overlap": label_overlap,
        "leakage_audit_passed": not model_overlap,
        "future_replay_outcomes_used_as_model_inputs": False,
        "future_replay_outcomes_used_as_training_labels": True,
        "future_replay_outcomes_used_as_report_only_evaluation": True,
        "row_count": len(rows),
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    report["o_feature_and_label_leakage_audit_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _comparison_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_rows = []
    for variant in O_VARIANTS:
        metrics = _ranking_metrics(rows, variant, config.high_score_threshold)
        reasons = [
            "diagnostic_only_no_paper_live_unlock",
            "current_m_m2_n_n2_evidence_not_o_promotion_evidence",
            "future_unseen_o_holdout_required",
        ]
        if metrics["high_score_realized_return_mean"] <= 0.0:
            reasons.append("high_score_realized_return_mean_not_positive")
        candidate_rows.append(
            {
                "candidate_name": variant,
                "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME
                if variant != "current_source_baseline"
                else "current_source_baseline",
                "shadow_raw_mae": metrics["split_metrics"]["shadow"]["raw_mae"],
                "shadow_calibrated_mae": metrics["split_metrics"]["shadow"][
                    "calibrated_mae"
                ],
                "top1_realized_best_action_hit_rate": metrics[
                    "top1_realized_best_action_hit_rate"
                ],
                "top2_realized_best_action_hit_rate": metrics[
                    "top2_realized_best_action_hit_rate"
                ],
                "top3_realized_best_action_hit_rate": metrics[
                    "top3_realized_best_action_hit_rate"
                ],
                "mean_regret": metrics["mean_regret"],
                "ranking_metric_scope": metrics["ranking_metric_scope"],
                "decision_group_completeness_summary": metrics[
                    "decision_group_completeness_summary"
                ],
                "source_score_completeness_summary": metrics[
                    "source_score_completeness_summary"
                ],
                "full_source_model_ranking_quality_claimed": metrics[
                    "full_source_model_ranking_quality_claimed"
                ],
                "high_score_support_count": metrics["high_score_support_count"],
                "high_score_realized_return_mean": metrics[
                    "high_score_realized_return_mean"
                ],
                "high_score_realized_return_sum": metrics[
                    "high_score_realized_return_sum"
                ],
                "action_family_eligibility_gates": {
                    "SELL_BEFORE_CLOSE": False,
                    "NO_TRADE": False,
                },
                "source_model_candidate_eligible": False,
                "ineligible_reason_codes": reasons,
            }
        )
    report = {
        "schema_version": O_SOURCE_CANDIDATE_COMPARISON_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "baseline_candidate_names": _baseline_names(),
        "report_type": "o_source_candidate_comparison",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "candidate_rows": candidate_rows,
        "eligible_candidate_count": 0,
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    report["o_source_candidate_comparison_report_id"] = canonical_json_sha256(report)
    return report


def _ranking_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    high_score_threshold: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["decision_group_id"]].append(row)
    top_hits = Counter()
    regrets = []
    selected_returns = []
    oracle_returns = []
    family_regret: dict[str, list[float]] = defaultdict(list)
    side_regret: dict[str, list[float]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    high_score_returns = []
    split_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    completeness_summary = _decision_group_completeness_summary(rows)
    source_score_summary = _source_score_completeness_summary(rows)
    source_scores_complete_for_variant = (
        variant != "current_source_baseline"
        or source_score_summary["source_score_complete"]
    )
    for group_rows in groups.values():
        predicted = sorted(
            group_rows,
            key=lambda row: float(row["variant_scores"][variant]),
            reverse=True,
        )
        oracle = max(group_rows, key=lambda row: float(row["realized_replay_return"]))
        selected = predicted[0]
        oracle_return = float(oracle["realized_replay_return"])
        selected_return = float(selected["realized_replay_return"])
        regret = oracle_return - selected_return
        regrets.append(regret)
        selected_returns.append(selected_return)
        oracle_returns.append(oracle_return)
        confusion[(selected["action_family"], oracle["action_family"])] += 1
        family_regret[selected["action_family"]].append(regret)
        side_regret[str(selected.get("selected_side") or "NONE")].append(regret)
        for k in (1, 2, 3):
            if oracle["action"] in {row["action"] for row in predicted[:k]}:
                top_hits[k] += 1
        if float(selected["variant_scores"][variant]) >= high_score_threshold:
            high_score_returns.append(selected_return)
        split_rows[selected["split"]].append(
            (
                float(selected["baseline_source_score"]),
                float(selected["variant_scores"][variant]),
                selected_return,
            )
        )
    group_count = len(groups)
    return {
        "decision_group_count": group_count,
        "ranking_metric_scope": completeness_summary["ranking_metric_scope"],
        "decision_group_completeness_summary": completeness_summary,
        "source_score_completeness_summary": source_score_summary,
        "full_source_model_ranking_quality_claimed": completeness_summary[
            "all_decision_groups_complete"
        ]
        and source_scores_complete_for_variant,
        "top1_realized_best_action_hit_rate": top_hits[1] / group_count
        if group_count
        else 0.0,
        "top2_realized_best_action_hit_rate": top_hits[2] / group_count
        if group_count
        else 0.0,
        "top3_realized_best_action_hit_rate": top_hits[3] / group_count
        if group_count
        else 0.0,
        "selected_action_realized_replay_return_sum": sum(selected_returns),
        "oracle_executable_best_action_return_sum": sum(oracle_returns),
        "mean_regret": statistics.mean(regrets) if regrets else 0.0,
        "no_trade_opportunity_cost_mean": statistics.mean(
            max(0.0, item) for item in oracle_returns
        )
        if oracle_returns
        else 0.0,
        "action_family_level_regret": {
            family: statistics.mean(values) for family, values in family_regret.items()
        },
        "side_level_regret": {
            side: statistics.mean(values) for side, values in side_regret.items()
        },
        "ranking_confusion_matrix": [
            {
                "predicted_top_action_family": predicted,
                "realized_best_action_family": realized,
                "count": count,
            }
            for (predicted, realized), count in sorted(confusion.items())
        ],
        "high_score_support_count": len(high_score_returns),
        "high_score_realized_return_mean": statistics.mean(high_score_returns)
        if high_score_returns
        else 0.0,
        "high_score_realized_return_sum": sum(high_score_returns),
        "split_metrics": {
            split: _split_metrics(split_rows.get(split, []))
            for split in ("shadow", "validation")
        },
    }


def _decision_group_completeness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["decision_group_id"]].append(row)
    partial_groups = []
    complete_count = 0
    for group_id, group_rows in sorted(groups.items()):
        first = group_rows[0]
        complete = bool(first.get("decision_group_completeness"))
        if complete:
            complete_count += 1
            continue
        partial_groups.append(
            {
                "decision_group_id": group_id,
                "source_report_path": first.get("source_report_path"),
                "market_id": first.get("market_id"),
                "decision_ts": first.get("decision_ts"),
                "available_action_families": first.get(
                    "available_action_families",
                    [],
                ),
                "missing_action_families": first.get(
                    "missing_action_families",
                    [],
                ),
            }
        )
    group_count = len(groups)
    all_complete = group_count > 0 and complete_count == group_count
    return {
        "required_action_families": list(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "decision_group_count": group_count,
        "complete_decision_group_count": complete_count,
        "partial_decision_group_count": group_count - complete_count,
        "all_decision_groups_complete": all_complete,
        "ranking_metric_scope": O_FULL_DECISION_GROUP_SCOPE
        if all_complete
        else O_PARTIAL_DECISION_GROUP_SCOPE,
        "partial_decision_groups": partial_groups[:50],
        "partial_decision_group_overflow_count": max(0, len(partial_groups) - 50),
    }


def _action_candidate_construction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_groups = {row["decision_group_id"] for row in rows}
    source_counts = Counter(str(row.get("candidate_label_source") or "unknown") for row in rows)
    observation_counts = Counter(
        str(row.get("candidate_observation_type") or "unknown") for row in rows
    )
    action_counts = Counter(str(row.get("action") or "UNKNOWN") for row in rows)
    label_available_count = sum(1 for row in rows if row.get("label_candidate_available"))
    source_score_available_count = sum(
        1 for row in rows if row.get("source_score_available")
    )
    return {
        "required_actions": list(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "decision_group_count": len(decision_groups),
        "candidate_row_count": len(rows),
        "expected_candidate_row_count": len(decision_groups)
        * len(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "complete_action_candidate_grid": len(rows)
        == len(decision_groups) * len(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "label_candidate_available_count": label_available_count,
        "missing_label_candidate_count": len(rows) - label_available_count,
        "source_score_available_count": source_score_available_count,
        "missing_source_score_count": len(rows) - source_score_available_count,
        "candidate_label_source_counts": dict(sorted(source_counts.items())),
        "candidate_observation_type_counts": dict(sorted(observation_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _source_score_completeness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available_count = sum(1 for row in rows if row.get("source_score_available"))
    return {
        "source_score_available_count": available_count,
        "missing_source_score_count": len(rows) - available_count,
        "source_score_complete": available_count == len(rows),
        "source_score_scope": "observed_replay_actions_only"
        if available_count != len(rows)
        else "all_action_candidates",
    }


def _split_metrics(values: list[tuple[float, float, float]]) -> dict[str, float]:
    if not values:
        return {"raw_mae": 0.0, "calibrated_mae": 0.0}
    return {
        "raw_mae": statistics.mean(abs(raw - realized) for raw, _, realized in values),
        "calibrated_mae": statistics.mean(
            abs(calibrated - realized) for _, calibrated, realized in values
        ),
    }


def _group_label_breakdown(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return [
        {
            field: key,
            "row_count": len(group_rows),
            "original_label_target_sum": sum(
                float(row["original_label_target"]) for row in group_rows
            ),
            "replay_aligned_label_target_sum": sum(
                float(row["replay_aligned_executable_label_target"])
                for row in group_rows
            ),
            "label_delta_sum": sum(float(row["label_delta"]) for row in group_rows),
            "label_vs_realized_replay_gap_before": sum(
                float(row["label_vs_realized_replay_gap_before"])
                for row in group_rows
            ),
            "label_vs_realized_replay_gap_after": sum(
                float(row["label_vs_realized_replay_gap_after"])
                for row in group_rows
            ),
        }
        for key, group_rows in sorted(groups.items())
    ]


def _fail_closed_fields() -> dict[str, Any]:
    return {
        "source_model_candidate_eligible": False,
        "calibration_support_passed": False,
        "calibration_quality_passed": False,
        "action_family_paper_decision_eligible": False,
        "best_action_concentration_passed": False,
        "p_up_action_disagreement_within_limit": False,
        "action_value_paper_decision_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "promotion_evidence_eligible": False,
    }


def _baseline_names() -> list[str]:
    return [
        SELL_BEFORE_CLOSE_SIDE_BALANCED_RANKING_CANDIDATE_NAME,
        SELL_BEFORE_CLOSE_M2_REPLAY_PARITY_CANDIDATE_NAME,
        SELL_BEFORE_CLOSE_N_UP_REPLAY_ALIGNED_ACTION_VALUE_CANDIDATE_NAME,
        SELL_BEFORE_CLOSE_N2_NON_LEAKY_UP_REPLAY_ALIGNED_FEATURE_PROXY_CANDIDATE_NAME,
    ]


def _compact_label_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_group_id": row["decision_group_id"],
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "action": row.get("action"),
        "action_family": row.get("action_family"),
        "candidate_observation_type": row.get("candidate_observation_type"),
        "candidate_label_source": row.get("candidate_label_source"),
        "label_candidate_available": row.get("label_candidate_available"),
        "source_score_available": row.get("source_score_available"),
        "decision_group_completeness": row["decision_group_completeness"],
        "available_action_families": row["available_action_families"],
        "missing_action_families": row["missing_action_families"],
        "ranking_metric_scope": row["ranking_metric_scope"],
        "selected_side": row.get("selected_side"),
        "original_label_target": row["original_label_target"],
        "replay_aligned_executable_label_target": row[
            "replay_aligned_executable_label_target"
        ],
        "label_delta": row["label_delta"],
        "realized_replay_return": row["realized_replay_return"],
        "label_vs_realized_replay_gap_before": row[
            "label_vs_realized_replay_gap_before"
        ],
        "label_vs_realized_replay_gap_after": row[
            "label_vs_realized_replay_gap_after"
        ],
        "label_components": row["label_components"],
        "split": row["split"],
    }


def _compact_ranking_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    return {
        "decision_group_id": row["decision_group_id"],
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "action": row.get("action"),
        "action_family": row.get("action_family"),
        "candidate_observation_type": row.get("candidate_observation_type"),
        "candidate_label_source": row.get("candidate_label_source"),
        "label_candidate_available": row.get("label_candidate_available"),
        "source_score_available": row.get("source_score_available"),
        "decision_group_completeness": row["decision_group_completeness"],
        "available_action_families": row["available_action_families"],
        "missing_action_families": row["missing_action_families"],
        "ranking_metric_scope": row["ranking_metric_scope"],
        "selected_side": row.get("selected_side"),
        "variant_score": row["variant_scores"][variant],
        "realized_replay_return": row["realized_replay_return"],
        "oracle_executable_best_action": row["oracle_executable_best_action"],
        "oracle_executable_best_action_family": row[
            "oracle_executable_best_action_family"
        ],
        "oracle_executable_best_action_return": row[
            "oracle_executable_best_action_return"
        ],
        "regret": row["oracle_executable_best_action_return"]
        - row["realized_replay_return"],
        "no_trade_opportunity_cost": row["no_trade_opportunity_cost"],
        "split": row["split"],
    }


def _side_from_action(action: str) -> str:
    if "BUY_UP" in action:
        return "UP"
    if "BUY_DOWN" in action:
        return "DOWN"
    return "NONE"


def _action_family(action: str) -> str:
    if action == "NO_TRADE":
        return "NO_TRADE"
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    return action or "UNKNOWN"


def _decision_action_family(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "")
    if action in O_REQUIRED_DECISION_ACTION_FAMILIES:
        return action
    return str(row.get("action_family") or _action_family(action))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _immediate_exit_pnl(row: dict[str, Any]) -> float | None:
    for field in ("execution_pnl_immediate_exit_pnl", "realized_trade_pnl"):
        if row.get(field) is not None:
            return float(row[field])
    ask = row.get("entry_quality_ask")
    bid = row.get("exit_quality_bid")
    if ask is not None and bid is not None:
        return float(bid) - float(ask)
    return None


def _spread_penalty(row: dict[str, Any]) -> float:
    spread = row.get("entry_exit_quality_spread_bps")
    if spread is None:
        return 0.0
    return max(0.0, float(spread) - 300.0) / 10_000.0


def _queue_penalty(row: dict[str, Any]) -> float:
    queue = row.get("entry_exit_quality_queue_fill")
    if queue is None:
        return 0.0
    return max(0.0, 0.80 - float(queue)) * 0.05


def _staleness_penalty(row: dict[str, Any]) -> float:
    staleness = row.get("entry_exit_quality_book_staleness_ms")
    if staleness is None:
        return 0.0
    return max(0.0, float(staleness) - 10_000.0) / 1_000_000.0


def _time_penalty(row: dict[str, Any]) -> float:
    seconds = row.get("entry_exit_quality_time_to_close_seconds")
    if seconds is None:
        return 0.0
    return max(0.0, 90.0 - float(seconds)) / 10_000.0


def _split_for_group(group_id: str) -> str:
    return "shadow" if canonical_json_sha256({"group_id": group_id})[-1] in "02468ace" else "validation"


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


def _queue_bucket(row: dict[str, Any]) -> str:
    value = row.get("entry_exit_quality_queue_fill")
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


def _label_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O Replay-Aligned Label Construction",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            f"- row_count: `{report['row_count']}`",
            f"- decision_group_count: `{report['decision_group_count']}`",
            "- partial_decision_group_count: "
            f"`{report['decision_group_completeness_summary']['partial_decision_group_count']}`",
            "- ranking_metric_scope: "
            f"`{report['decision_group_completeness_summary']['ranking_metric_scope']}`",
            f"- label_gap_before: `{report['label_gap_before']}`",
            f"- label_gap_after: `{report['label_gap_after']}`",
            f"- label_gap_delta: `{report['label_gap_delta']}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _ranking_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O Source Ranking Objective",
            "",
            f"- primary_variant_name: `{report['primary_variant_name']}`",
            f"- ranking_metric_scope: `{report['ranking_metric_scope']}`",
            "- full_source_model_ranking_quality_claimed: "
            f"`{str(report['full_source_model_ranking_quality_claimed']).lower()}`",
            "- partial_decision_group_count: "
            f"`{report['decision_group_completeness_summary']['partial_decision_group_count']}`",
            f"- top1_hit_rate: `{report['top1_realized_best_action_hit_rate']}`",
            f"- top2_hit_rate: `{report['top2_realized_best_action_hit_rate']}`",
            f"- top3_hit_rate: `{report['top3_realized_best_action_hit_rate']}`",
            f"- mean_regret: `{report['mean_regret']}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _leakage_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O Feature And Label Leakage Audit",
            "",
            f"- leakage_audit_passed: `{str(report['leakage_audit_passed']).lower()}`",
            "- model_input_forbidden_field_overlap: "
            f"`{report['model_input_forbidden_field_overlap']}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# O Source Candidate Comparison",
        "",
        f"- eligible_candidate_count: `{report['eligible_candidate_count']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "| candidate | scope | top1 | mean_regret | eligible |",
        "|---|---|---:|---:|---|",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            "| {name} | {scope} | {top1:.4f} | {regret:.6f} | {eligible} |".format(
                name=row["candidate_name"],
                scope=row["ranking_metric_scope"],
                top1=float(row["top1_realized_best_action_hit_rate"]),
                regret=float(row["mean_regret"]),
                eligible=str(row["source_model_candidate_eligible"]).lower(),
            )
        )
    lines.append("")
    return "\n".join(lines)
