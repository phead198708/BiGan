"""Diagnostic O replay-aligned source-ranking reports."""

from __future__ import annotations

import json
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
O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-source-model-eligibility-gate-v1"
)
O_FREEZE_READINESS_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-freeze-readiness-v1"
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
O_MODEL_PREDICTED_VARIANT = "o_model_predicted_decision_time_source_model"
O_VARIANTS = (
    "current_source_baseline",
    O_MODEL_PREDICTED_VARIANT,
    "o_replay_aligned_labels_only",
    "o_replay_aligned_labels_family_priors",
    "o_replay_aligned_pairwise_listwise_correction",
    "o_replay_aligned_stronger_no_trade_prior",
)
O_LABEL_DIAGNOSTIC_VARIANTS = (
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
O_ACTION_FEATURE_SLUGS = (
    ("BUY_UP_SELL_BEFORE_CLOSE", "buy_up_sell_before_close"),
    ("BUY_DOWN_SELL_BEFORE_CLOSE", "buy_down_sell_before_close"),
    ("BUY_UP_HOLD_TO_SETTLEMENT", "buy_up_hold_to_settlement"),
    ("BUY_DOWN_HOLD_TO_SETTLEMENT", "buy_down_hold_to_settlement"),
    ("NO_TRADE", "no_trade"),
)
O_ACTION_INTERACTION_SIGNAL_NAMES = (
    "p_up",
    "p_down",
    "time_to_close",
    "spread",
    "queue",
    "staleness",
    "entry_ask",
    "exit_bid_proxy",
)
O_DEPLOYABLE_MODEL_FEATURE_NAMES = (
    "bias",
    "action_buy_up_sell_before_close",
    "action_buy_down_sell_before_close",
    "action_buy_up_hold_to_settlement",
    "action_buy_down_hold_to_settlement",
    "action_no_trade",
    "side_up",
    "side_down",
    "side_none",
    "family_sell_before_close",
    "family_hold_to_settlement",
    "family_no_trade",
    "p_up",
    "p_down_proxy",
    "entry_ask",
    "spread_bps_scaled",
    "queue_fill",
    "book_staleness_seconds",
    "time_to_close_minutes",
    "p_up_edge",
    "weak_opportunity_proxy",
    *tuple(
        f"{action_slug}_x_{signal_name}"
        for _, action_slug in O_ACTION_FEATURE_SLUGS
        for signal_name in O_ACTION_INTERACTION_SIGNAL_NAMES
    ),
)
O_MIN_VALIDATION_DECISION_GROUPS = 20
O_MIN_HIGH_SCORE_SUPPORT_COUNT = 10
O_MIN_TOP1_HIT_RATE = 0.35
O_MAX_MEAN_REGRET = 0.15
O_MAX_NO_TRADE_SELECTION_RATE = 0.80
O_MAX_P_UP_ACTION_DISAGREEMENT_RATE = 0.35


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
    source_model_eligibility_gate_report: dict[str, Any]
    freeze_readiness_report: dict[str, Any]
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
        "source_model_eligibility_gate_report": run_dir
        / "o_source_model_eligibility_gate_report.json",
        "source_model_eligibility_gate_summary": run_dir
        / "o_source_model_eligibility_gate_report.md",
        "freeze_readiness_report": run_dir / "o_freeze_readiness_report.json",
        "freeze_readiness_summary": run_dir / "o_freeze_readiness_report.md",
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
    _write_json(artifact_paths["source_model_eligibility_gate_report"], reports[4])
    artifact_paths["source_model_eligibility_gate_summary"].write_text(
        _eligibility_gate_markdown(reports[4]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["freeze_readiness_report"], reports[5])
    artifact_paths["freeze_readiness_summary"].write_text(
        _freeze_readiness_markdown(reports[5]),
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
        source_model_eligibility_gate_report=reports[4],
        freeze_readiness_report=reports[5],
        artifact_paths=artifact_paths,
    )


def _build_reports(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
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
    scored_rows, model_training_summary = _train_o_model_predicted_scores(
        labeled_rows
    )
    ranking_rows = _ranking_rows(scored_rows)
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
        model_training_summary=model_training_summary,
    )
    leakage_report = _leakage_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=scored_rows,
        model_training_summary=model_training_summary,
    )
    comparison_report = _comparison_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
        model_training_summary=model_training_summary,
    )
    eligibility_gate_report = _source_model_eligibility_gate_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
        model_training_summary=model_training_summary,
        leakage_report=leakage_report,
    )
    freeze_readiness_report = _freeze_readiness_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
        model_training_summary=model_training_summary,
        eligibility_gate_report=eligibility_gate_report,
    )
    return (
        label_report,
        ranking_report,
        leakage_report,
        comparison_report,
        eligibility_gate_report,
        freeze_readiness_report,
    )


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


def _train_o_model_predicted_scores(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = [
        row
        for row in rows
        if row.get("split") == "shadow"
        and bool(row.get("label_candidate_available", True))
    ]
    if len({row["decision_group_id"] for row in train_rows}) < 2:
        train_rows = [
            row for row in rows if bool(row.get("label_candidate_available", True))
        ]
        training_split_source = "all_rows_fallback_insufficient_shadow_groups"
    else:
        training_split_source = "shadow_split_only"
    deployable_available = bool(train_rows) and _full_grid_available(rows)
    if deployable_available:
        model = _fit_ridge_regression(
            [_deployable_model_features(row) for row in train_rows],
            [
                float(row["replay_aligned_executable_label_target"])
                for row in train_rows
            ],
        )
        predictions = [
            _dot(model["coefficients"], _deployable_model_features(row))
            for row in rows
        ]
        fit_reason_codes: list[str] = []
    else:
        model = {
            "coefficients": [0.0 for _ in O_DEPLOYABLE_MODEL_FEATURE_NAMES],
            "ridge_lambda": 1.0e-6,
        }
        predictions = [0.0 for _ in rows]
        fit_reason_codes = ["insufficient_complete_action_grid_for_model_training"]
    raw_scored_rows = [
        {
            **row,
            "o_raw_ridge_model_score": prediction,
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    raw_train_rows = [
        row
        for row in raw_scored_rows
        if row.get("split") == "shadow"
        and bool(row.get("label_candidate_available", True))
    ]
    ranking_correction = _learn_o_shadow_ranking_correction(raw_train_rows)
    scored_rows = _apply_o_shadow_ranking_correction(
        rows=raw_scored_rows,
        deployable_available=deployable_available,
        ranking_correction=ranking_correction,
    )
    summary = {
        "model_candidate_name": O_MODEL_PREDICTED_VARIANT,
        "ranking_score_source": "model_predicted_score",
        "deployable_model_score_available": deployable_available,
        "model_family": (
            "deterministic_ridge_action_value_regressor_with_shadow_only_ranking_correction"
        ),
        "raw_model_family": "deterministic_ridge_action_value_regressor",
        "post_model_ranking_correction_enabled": True,
        "ranking_correction_source": "shadow_split_only",
        "ranking_correction_config": ranking_correction,
        "correction_constants_source": ranking_correction[
            "correction_constants_source"
        ],
        "correction_config_hash": ranking_correction["correction_config_hash"],
        "probe_constants_source": ranking_correction["probe_constants_source"],
        "probe_config_hash": ranking_correction["probe_config_hash"],
        "feature_names": list(O_DEPLOYABLE_MODEL_FEATURE_NAMES),
        "model_input_fields_decision_time_only": list(
            O_DEPLOYABLE_MODEL_FEATURE_NAMES
        ),
        "training_target": "replay_aligned_executable_label_target",
        "training_label_fields_may_use_future_replay_or_settlement": list(
            O_TRAINING_LABEL_FIELDS
        ),
        "training_row_count": len(train_rows),
        "training_decision_group_count": len(
            {row["decision_group_id"] for row in train_rows}
        ),
        "scored_row_count": len(scored_rows),
        "scored_decision_group_count": len(
            {row["decision_group_id"] for row in scored_rows}
        ),
        "training_split_source": training_split_source,
        "ridge_lambda": model["ridge_lambda"],
        "coefficients_by_feature": dict(
            zip(
                O_DEPLOYABLE_MODEL_FEATURE_NAMES,
                model["coefficients"],
                strict=True,
            )
        ),
        "fit_reason_codes": fit_reason_codes,
        "label_diagnostic_variants": list(O_LABEL_DIAGNOSTIC_VARIANTS),
        "label_diagnostic_variants_deployable": False,
        "current_source_baseline_counterfactual_scores_complete": False,
        "paper_only": True,
        "capital_at_risk": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
    }
    return scored_rows, summary


def _full_grid_available(rows: list[dict[str, Any]]) -> bool:
    summary = _decision_group_completeness_summary(rows)
    construction = _action_candidate_construction_summary(rows)
    return bool(summary["all_decision_groups_complete"]) and bool(
        construction["complete_action_candidate_grid"]
    )


def _deployable_model_features(row: dict[str, Any]) -> list[float]:
    action = str(row.get("action") or "")
    family = _action_family(action)
    side = _side_from_action(action)
    p_up = _bounded(float(row.get("p_up") or 0.5), 0.0, 1.0)
    p_down = 1.0 - p_up
    spread = _normalized_spread(row)
    queue = _bounded(float(row.get("entry_exit_quality_queue_fill") or 0.0), 0.0, 1.0)
    staleness = _normalized_staleness(row)
    time_to_close = _normalized_time_to_close(row)
    entry_ask = _bounded(float(row.get("entry_quality_ask") or 0.0), 0.0, 1.0)
    exit_bid_proxy = _decision_time_exit_bid_proxy(row)
    p_up_edge = abs(p_up - 0.5)
    weak_opportunity = max(0.0, 0.10 - p_up_edge)
    base_features = [
        1.0,
        _flag(action == "BUY_UP_SELL_BEFORE_CLOSE"),
        _flag(action == "BUY_DOWN_SELL_BEFORE_CLOSE"),
        _flag(action == "BUY_UP_HOLD_TO_SETTLEMENT"),
        _flag(action == "BUY_DOWN_HOLD_TO_SETTLEMENT"),
        _flag(action == "NO_TRADE"),
        _flag(side == "UP"),
        _flag(side == "DOWN"),
        _flag(side == "NONE"),
        _flag(family == "SELL_BEFORE_CLOSE"),
        _flag(family == "HOLD_TO_SETTLEMENT"),
        _flag(family == "NO_TRADE"),
        p_up,
        p_down,
        entry_ask,
        spread,
        queue,
        staleness,
        time_to_close,
        p_up_edge,
        weak_opportunity,
    ]
    signals = {
        "p_up": p_up,
        "p_down": p_down,
        "time_to_close": time_to_close,
        "spread": spread,
        "queue": queue,
        "staleness": staleness,
        "entry_ask": entry_ask,
        "exit_bid_proxy": exit_bid_proxy,
    }
    interactions = [
        _flag(action == action_name) * signals[signal_name]
        for action_name, _ in O_ACTION_FEATURE_SLUGS
        for signal_name in O_ACTION_INTERACTION_SIGNAL_NAMES
    ]
    return [*base_features, *interactions]


def _normalized_spread(row: dict[str, Any]) -> float:
    return _bounded(
        float(row.get("entry_exit_quality_spread_bps") or 0.0) / 10_000.0,
        0.0,
        1.0,
    )


def _normalized_staleness(row: dict[str, Any]) -> float:
    return _bounded(
        float(row.get("entry_exit_quality_book_staleness_ms") or 0.0) / 1000.0,
        0.0,
        60.0,
    )


def _normalized_time_to_close(row: dict[str, Any]) -> float:
    return _bounded(
        float(row.get("entry_exit_quality_time_to_close_seconds") or 0.0) / 60.0,
        0.0,
        15.0,
    )


def _decision_time_exit_bid_proxy(row: dict[str, Any]) -> float:
    entry_ask = _bounded(float(row.get("entry_quality_ask") or 0.0), 0.0, 1.0)
    spread = _normalized_spread(row)
    return _bounded(entry_ask - spread, 0.0, 1.0)


def _learn_o_shadow_ranking_correction(
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    global_mean = statistics.mean(
        float(row["replay_aligned_executable_label_target"]) for row in train_rows
    ) if train_rows else 0.0
    action_priors = {
        action_name: _shadow_shrunk_mean(
            [
                float(row["replay_aligned_executable_label_target"])
                for row in train_rows
                if row["action"] == action_name
            ],
            global_mean=global_mean,
        )
        for action_name in O_REQUIRED_DECISION_ACTION_FAMILIES
    }
    family_priors = {
        family: _shadow_shrunk_mean(
            [
                float(row["replay_aligned_executable_label_target"])
                for row in train_rows
                if row["action_family"] == family
            ],
            global_mean=global_mean,
        )
        for family in ("SELL_BEFORE_CLOSE", "HOLD_TO_SETTLEMENT", "NO_TRADE")
    }
    p_edges_by_group = []
    seen_groups = set()
    for row in train_rows:
        if row["decision_group_id"] in seen_groups:
            continue
        seen_groups.add(row["decision_group_id"])
        p_edges_by_group.append(abs(float(row.get("p_up") or 0.5) - 0.5))
    p_edge_quantiles = _p_edge_quantiles(p_edges_by_group)
    weak_cutoff = p_edge_quantiles["q25"]
    large_regret_reversal_priors = _derive_shadow_large_regret_reversal_priors(
        train_rows
    )
    hts_reliability_thresholds = _derive_shadow_hts_p_up_reliability_thresholds(
        train_rows,
        p_edge_quantiles,
    )
    hts_reliability_priors = _derive_shadow_hts_p_up_reliability_priors(
        train_rows,
        hts_reliability_thresholds,
    )
    probe_score_config = _derive_shadow_probe_score_config(
        p_edge_quantiles=p_edge_quantiles,
        global_mean=global_mean,
        family_priors=family_priors,
    )
    prior_weight, prior_diagnostics = _derive_shadow_prior_weight(
        train_rows,
        action_priors,
    )
    micro_weight, micro_diagnostics = _derive_shadow_microstructure_weight(
        train_rows,
        p_edge_quantiles,
    )
    high_score_threshold = 0.75 + p_edge_quantiles["q25"]
    config = {
        "correction_name": "shadow_only_p_up_aligned_weak_opportunity_ranker",
        "ranking_objective_proxy": "pairwise_group_margin_and_regret_aware_proxy",
        "uses_validation_labels_for_tuning": False,
        "correction_constants_source": "shadow_split_only",
        "correction_constants_are_shadow_derived": True,
        "probe_constants_source": "shadow_split_only",
        "probe_score_config": probe_score_config,
        "probe_config_hash": probe_score_config["probe_config_hash"],
        "p_up_edge_quantiles": p_edge_quantiles,
        "weak_opportunity_p_edge_cutoff": weak_cutoff,
        "weak_opportunity_cutoff_source": "shadow_p_up_edge_lower_quartile",
        "trade_base_score": 0.5 + p_edge_quantiles["q75"],
        "trade_base_score_source": "0.5 + shadow_p_up_edge_q75",
        "sell_before_close_base_score": 0.5 + p_edge_quantiles["q25"] / 2.0,
        "sell_before_close_base_score_source": "0.5 + shadow_p_up_edge_q25 / 2",
        "no_trade_base_score": 0.5 + p_edge_quantiles["median"],
        "no_trade_base_score_source": "0.5 + shadow_p_up_edge_median",
        "confidence_bonus": p_edge_quantiles["median"],
        "confidence_bonus_source": "shadow_p_up_edge_median",
        "weak_opportunity_trade_penalty": -p_edge_quantiles["q25"],
        "weak_opportunity_trade_penalty_source": "-shadow_p_up_edge_q25",
        "sell_before_close_confidence_bonus": p_edge_quantiles["q25"] / 3.0,
        "sell_before_close_confidence_bonus_source": "shadow_p_up_edge_q25 / 3",
        "sell_before_close_weak_penalty": -p_edge_quantiles["q25"] * 0.6,
        "sell_before_close_weak_penalty_source": "-shadow_p_up_edge_q25 * 0.6",
        "group_normalized_raw_model_weight": 0.0,
        "group_normalized_raw_model_weight_source": (
            "shadow_candidate_search_high_score_return_with_strict_p_up_safety"
        ),
        "p_up_misalignment_raw_positive_penalty": 0.0,
        "p_up_misalignment_raw_positive_penalty_source": (
            "shadow_candidate_search_p_up_edge_quantile_grid"
        ),
        "p_up_misalignment_penalty_applies_to": (
            "buy_actions_with_negative_p_up_alignment_and_positive_raw_component"
        ),
        "large_regret_reversal_guard_enabled": True,
        "large_regret_reversal_guard_source": (
            "shadow_split_only_hold_to_settlement_action_pair_regret_priors"
        ),
        "large_regret_reversal_guard_modes": (
            "raw_p_up_opposition_confidence_veto",
            "hold_to_settlement_high_reversal_exposure_veto",
        ),
        "large_regret_reversal_guard_applies_to": (
            "hold_to_settlement_buy_actions_with_positive_raw_component_"
            "and_opposite_p_up_alignment_or_high_reversal_exposure"
        ),
        "large_regret_reversal_confidence_edge_ceiling": min(
            0.5,
            p_edge_quantiles["q25"] + p_edge_quantiles["q75"],
        ),
        "large_regret_reversal_confidence_edge_ceiling_source": (
            "shadow_p_up_edge_q25_plus_q75"
        ),
        "large_regret_reversal_pair_regret_priors": large_regret_reversal_priors[
            "action_pair_priors"
        ],
        "large_regret_reversal_pair_regret_threshold": (
            large_regret_reversal_priors["pair_regret_threshold"]
        ),
        "large_regret_reversal_pair_regret_threshold_source": (
            large_regret_reversal_priors["pair_regret_threshold_source"]
        ),
        "large_regret_reversal_alignment_threshold": 0.0,
        "large_regret_reversal_alignment_threshold_source": (
            "shadow_candidate_search_p_up_edge_quantile_grid"
        ),
        "large_regret_reversal_penalty": 0.0,
        "large_regret_reversal_penalty_source": (
            "shadow_candidate_search_largest_regret_reversal_grid"
        ),
        "hts_p_up_reliability_guard_enabled": True,
        "hts_p_up_reliability_guard_source": (
            "shadow_split_only_hts_p_up_side_bucket_regret"
        ),
        "hts_p_up_reliability_guard_applies_to": (
            "hold_to_settlement_actions_matching_p_up_implied_side_in_"
            "high_shadow_regret_side_confidence_or_microstructure_bucket"
        ),
        "hts_p_up_reliability_bucket_thresholds": hts_reliability_thresholds,
        "hts_p_up_reliability_regime_priors": hts_reliability_priors[
            "regime_priors"
        ],
        "hts_p_up_reliability_bucket_diagnostics": hts_reliability_priors[
            "bucket_diagnostics"
        ],
        "hts_p_up_reliability_regret_threshold": hts_reliability_priors[
            "regret_threshold"
        ],
        "hts_p_up_reliability_regret_threshold_source": hts_reliability_priors[
            "regret_threshold_source"
        ],
        "hts_p_up_reliability_min_support": hts_reliability_priors[
            "min_support"
        ],
        "hts_p_up_reliability_min_support_source": hts_reliability_priors[
            "min_support_source"
        ],
        "hts_p_up_reliability_penalty": p_edge_quantiles["q75"],
        "hts_p_up_reliability_penalty_source": (
            "shadow_p_up_edge_q75"
        ),
        "hts_p_up_reliability_no_trade_buffer_enabled": True,
        "hts_p_up_reliability_no_trade_buffer": p_edge_quantiles["q25"],
        "hts_p_up_reliability_no_trade_buffer_source": "shadow_p_up_edge_q25",
        "hts_p_up_reliability_no_trade_buffer_applies_to": (
            "NO_TRADE weak-opportunity fallback for HTS side-confidence risk"
        ),
        "p_up_safety_target_disagreement_rate": 0.25,
        "p_up_safety_target_source": "config_hashed_stricter_than_hard_gate_target",
        "shadow_p_up_selection_max_disagreement_rate": max(
            0.0,
            O_MAX_P_UP_ACTION_DISAGREEMENT_RATE - p_edge_quantiles["q75"],
        ),
        "shadow_p_up_selection_max_disagreement_rate_source": (
            "max_p_up_action_disagreement_rate_minus_shadow_p_up_edge_q75"
        ),
        "shadow_action_family_prior_weight": prior_weight,
        "shadow_action_family_prior_weight_source": (
            "shadow_prior_concentration_guard"
        ),
        "microstructure_quality_weight": micro_weight,
        "microstructure_quality_weight_source": (
            "shadow_microstructure_target_correlation_scaled_by_p_edge_q25"
        ),
        "action_shadow_priors": action_priors,
        "action_family_shadow_priors": family_priors,
        "shadow_global_target_mean": global_mean,
        "shadow_component_diagnostics": {
            "prior": prior_diagnostics,
            "microstructure": micro_diagnostics,
        },
        "high_score_calibration": {
            "method": "shadow_buffered_threshold_rank_score_calibration",
            "high_score_threshold": high_score_threshold,
            "high_score_threshold_source": "0.75 + shadow_p_up_edge_q25",
            "previous_high_score_threshold": 0.75,
            "high_score_requires_corrected_model_score_gte_threshold": True,
        },
        "NO_TRADE_prior": {
            "enabled": True,
            "weak_opportunity_feature": "max(0, weak_opportunity_p_edge_cutoff - abs(p_up - 0.5))",
        },
    }
    raw_weight, raw_diagnostics = _derive_shadow_raw_model_weight(
        train_rows=train_rows,
        base_ranking_correction=config,
    )
    config["group_normalized_raw_model_weight"] = raw_weight
    config["p_up_misalignment_raw_positive_penalty"] = raw_diagnostics[
        "selected_raw_weight_candidate"
    ]["candidate_p_up_misalignment_penalty"]
    config["large_regret_reversal_alignment_threshold"] = raw_diagnostics[
        "selected_raw_weight_candidate"
    ]["candidate_large_regret_reversal_alignment_threshold"]
    config["large_regret_reversal_penalty"] = raw_diagnostics[
        "selected_raw_weight_candidate"
    ]["candidate_large_regret_reversal_penalty"]
    config["hts_p_up_reliability_penalty"] = raw_diagnostics[
        "selected_raw_weight_candidate"
    ]["candidate_hts_p_up_reliability_penalty"]
    config["hts_p_up_reliability_penalty_source"] = raw_diagnostics[
        "selected_raw_weight_candidate"
    ]["candidate_hts_p_up_reliability_penalty_source"]
    _apply_large_regret_adjusted_high_score_calibration(config)
    config["shadow_component_diagnostics"]["raw_model"] = raw_diagnostics
    config["correction_config_hash"] = canonical_json_sha256(config)
    return config


def _shadow_shrunk_mean(
    values: list[float],
    *,
    global_mean: float,
    shrinkage: float = 20.0,
) -> float:
    if not values:
        return global_mean
    return (sum(values) + shrinkage * global_mean) / (len(values) + shrinkage)


def _p_edge_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"q25": 0.0, "median": 0.0, "q75": 0.0}
    if len(values) < 4:
        value = statistics.median(values)
        return {"q25": min(values), "median": value, "q75": max(values)}
    quartiles = statistics.quantiles(values, n=4)
    return {
        "q25": quartiles[0],
        "median": statistics.median(values),
        "q75": quartiles[2],
    }


def _derive_shadow_probe_score_config(
    *,
    p_edge_quantiles: dict[str, float],
    global_mean: float,
    family_priors: dict[str, float],
) -> dict[str, Any]:
    positive_prior_anchor = max(0.0, global_mean)
    config = {
        "probe_name": "shadow_only_probe_ranker",
        "probe_constants_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "probe_no_trade_base_score": 0.5 + max(0.0, -global_mean),
        "probe_no_trade_base_score_source": "0.5 + max(0, -shadow_global_target_mean)",
        "probe_no_trade_weak_edge_cutoff": p_edge_quantiles["median"],
        "probe_no_trade_weak_edge_cutoff_source": "shadow_p_up_edge_median",
        "probe_hold_to_settlement_base_score": (
            0.5
            + p_edge_quantiles["median"]
            + max(0.0, family_priors.get("HOLD_TO_SETTLEMENT", 0.0))
        ),
        "probe_hold_to_settlement_base_score_source": (
            "0.5 + shadow_p_up_edge_median + positive_shadow_hold_to_settlement_prior"
        ),
        "probe_sell_before_close_base_score": (
            0.5
            - p_edge_quantiles["q25"]
            + max(0.0, family_priors.get("SELL_BEFORE_CLOSE", 0.0))
        ),
        "probe_sell_before_close_base_score_source": (
            "0.5 - shadow_p_up_edge_q25 + positive_shadow_sell_before_close_prior"
        ),
        "probe_sell_before_close_alignment_weight": (
            0.5 + p_edge_quantiles["q25"] + positive_prior_anchor
        ),
        "probe_sell_before_close_alignment_weight_source": (
            "0.5 + shadow_p_up_edge_q25 + positive_shadow_global_target_mean"
        ),
    }
    config["probe_config_hash"] = canonical_json_sha256(config)
    return config


def _derive_shadow_raw_model_weight(
    train_rows: list[dict[str, Any]],
    base_ranking_correction: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    raw_metrics = _shadow_ranker_metrics(
        train_rows,
        lambda row: float(row.get("o_raw_ridge_model_score") or 0.0),
    )
    probe_score_config = base_ranking_correction["probe_score_config"]
    derived_metrics = _shadow_ranker_metrics(
        train_rows,
        lambda row: _shadow_derived_ranker_probe_score(row, probe_score_config),
    )
    candidate_weights = _shadow_raw_weight_candidates(
        base_ranking_correction["p_up_edge_quantiles"]
    )
    candidate_penalties = _shadow_p_up_misalignment_penalty_candidates(
        base_ranking_correction["p_up_edge_quantiles"]
    )
    candidate_reversal_thresholds = _shadow_large_regret_reversal_threshold_candidates(
        base_ranking_correction["p_up_edge_quantiles"]
    )
    candidate_reversal_penalties = _shadow_large_regret_reversal_penalty_candidates(
        base_ranking_correction["p_up_edge_quantiles"]
    )
    candidate_hts_reliability_penalties = _shadow_hts_p_up_reliability_penalty_candidates(
        base_ranking_correction["p_up_edge_quantiles"]
    )
    max_shadow_p_up_disagreement_rate = float(
        base_ranking_correction["shadow_p_up_selection_max_disagreement_rate"]
    )
    candidate_rows = []
    for weight in candidate_weights:
        for penalty in candidate_penalties:
            for reversal_threshold in candidate_reversal_thresholds:
                for reversal_penalty in candidate_reversal_penalties:
                    for hts_reliability_penalty in candidate_hts_reliability_penalties:
                        candidate_config = {
                            **base_ranking_correction,
                            "group_normalized_raw_model_weight": weight,
                            "p_up_misalignment_raw_positive_penalty": penalty,
                            "large_regret_reversal_alignment_threshold": (
                                reversal_threshold
                            ),
                            "large_regret_reversal_penalty": reversal_penalty,
                            "hts_p_up_reliability_penalty": (
                                hts_reliability_penalty
                            ),
                            "hts_p_up_reliability_no_trade_buffer_enabled": False,
                        }
                        _apply_large_regret_adjusted_high_score_calibration(
                            candidate_config
                        )
                        ranking_rows = _ranking_rows(
                            _apply_o_shadow_ranking_correction(
                                rows=train_rows,
                                deployable_available=True,
                                ranking_correction=candidate_config,
                            )
                        )
                        metrics = _split_metric_views(
                            ranking_rows,
                            O_MODEL_PREDICTED_VARIANT,
                            float(
                                candidate_config["high_score_calibration"][
                                    "high_score_threshold"
                                ]
                            ),
                        )["train_shadow"]
                        p_up_summary = _p_up_action_disagreement_summary(
                            rows=ranking_rows,
                            variant=O_MODEL_PREDICTED_VARIANT,
                            split="shadow",
                        )
                        high_score_profitable = (
                            int(metrics["high_score_support_count"])
                            >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
                            and float(metrics["high_score_realized_return_mean"]) > 0.0
                            and float(metrics["high_score_realized_return_sum"]) > 0.0
                        )
                        p_up_safety_passed = (
                            int(
                                p_up_summary[
                                    "candidate_scoped_p_up_action_comparable_count"
                                ]
                            )
                            > 0
                            and float(
                                p_up_summary[
                                    "candidate_scoped_p_up_action_disagreement_rate"
                                ]
                            )
                            <= max_shadow_p_up_disagreement_rate
                        )
                        selected_return_sum = float(
                            metrics["selected_action_realized_replay_return_sum"]
                        )
                        largest_regret_case = metrics["largest_regret_case"]
                        largest_regret = float(
                            largest_regret_case.get("regret") or 0.0
                        )
                        eligible = (
                            high_score_profitable
                            and p_up_safety_passed
                            and selected_return_sum > 0.0
                        )
                        candidate_rows.append(
                            {
                                "candidate_weight": weight,
                                "candidate_weight_source": (
                                    "shadow_p_up_edge_quantile_grid"
                                ),
                                "candidate_p_up_misalignment_penalty": penalty,
                                "candidate_p_up_misalignment_penalty_source": (
                                    "shadow_p_up_edge_quantile_grid"
                                ),
                                "candidate_large_regret_reversal_alignment_threshold": (
                                    reversal_threshold
                                ),
                                "candidate_large_regret_reversal_alignment_threshold_source": (
                                    "shadow_p_up_edge_quantile_grid"
                                ),
                                "candidate_large_regret_reversal_penalty": (
                                    reversal_penalty
                                ),
                                "candidate_large_regret_reversal_penalty_source": (
                                    "shadow_largest_regret_reversal_grid"
                                ),
                                "candidate_hts_p_up_reliability_penalty": (
                                    hts_reliability_penalty
                                ),
                                "candidate_hts_p_up_reliability_penalty_source": (
                                    "shadow_p_up_edge_quantile_grid"
                                ),
                                "shadow_candidate_eligible": eligible,
                                "shadow_high_score_profitable": high_score_profitable,
                                "shadow_p_up_safety_passed": p_up_safety_passed,
                                "shadow_selected_return_sum": selected_return_sum,
                                "shadow_mean_regret": metrics["mean_regret"],
                                "shadow_largest_regret_value": largest_regret,
                                "shadow_largest_regret_case": largest_regret_case,
                                "shadow_action_family_level_regret": metrics[
                                    "action_family_level_regret"
                                ],
                                "shadow_action_pair_regret_summary": metrics[
                                    "action_pair_regret_summary"
                                ],
                                "shadow_hts_p_up_reliability_regret_summary": metrics[
                                    "hts_p_up_reliability_regret_summary"
                                ],
                                "shadow_hold_to_settlement_up_down_reversal_regret": (
                                    metrics[
                                        "hold_to_settlement_up_down_reversal_regret"
                                    ]
                                ),
                                "shadow_no_trade_missed_opportunity": metrics[
                                    "no_trade_missed_opportunity"
                                ],
                                "shadow_high_score_support_count": metrics[
                                    "high_score_support_count"
                                ],
                                "shadow_high_score_realized_return_mean": metrics[
                                    "high_score_realized_return_mean"
                                ],
                                "shadow_high_score_realized_return_sum": metrics[
                                    "high_score_realized_return_sum"
                                ],
                                "shadow_p_up_action_disagreement_rate": p_up_summary[
                                    "candidate_scoped_p_up_action_disagreement_rate"
                                ],
                            }
                        )
    eligible_candidates = [
        row for row in candidate_rows if bool(row["shadow_candidate_eligible"])
    ]
    if eligible_candidates:
        selected = max(
            eligible_candidates,
            key=lambda row: (
                float(row["shadow_selected_return_sum"]),
                -float(row["shadow_mean_regret"]),
                -float(row["shadow_largest_regret_value"]),
                float(row["candidate_hts_p_up_reliability_penalty"]),
                float(row["candidate_large_regret_reversal_penalty"]),
                float(row["shadow_high_score_realized_return_sum"]),
                int(row["shadow_high_score_support_count"]),
                -float(row["shadow_p_up_action_disagreement_rate"]),
                -float(row["candidate_p_up_misalignment_penalty"]),
            ),
        )
    else:
        selected = max(
            candidate_rows,
            key=lambda row: (
                float(row["shadow_selected_return_sum"]),
                -float(row["shadow_mean_regret"]),
            ),
        )
    weight = float(selected["candidate_weight"])
    return weight, {
        "raw_model_shadow_metrics": raw_metrics,
        "probe_ranker_shadow_metrics": derived_metrics,
        "raw_weight_candidate_source": "shadow_p_up_edge_quantile_grid",
        "p_up_misalignment_penalty_candidate_source": (
            "shadow_p_up_edge_quantile_grid"
        ),
        "large_regret_reversal_guard_candidate_source": (
            "shadow_largest_regret_reversal_grid"
        ),
        "large_regret_reversal_guard_selection_metric_source": "shadow_split_only",
        "hts_p_up_reliability_guard_candidate_source": (
            "shadow_p_up_edge_quantile_grid"
        ),
        "hts_p_up_reliability_guard_selection_metric_source": "shadow_split_only",
        "hts_p_up_reliability_no_trade_buffer_excluded_from_raw_weight_search": True,
        "hts_p_up_reliability_no_trade_buffer_application_stage": (
            "post_shadow_raw_weight_selection_safety_buffer"
        ),
        "hts_p_up_reliability_bucket_thresholds": base_ranking_correction[
            "hts_p_up_reliability_bucket_thresholds"
        ],
        "hts_p_up_reliability_regime_priors": base_ranking_correction[
            "hts_p_up_reliability_regime_priors"
        ],
        "hts_p_up_reliability_bucket_diagnostics": base_ranking_correction[
            "hts_p_up_reliability_bucket_diagnostics"
        ],
        "hts_p_up_reliability_regret_threshold": base_ranking_correction[
            "hts_p_up_reliability_regret_threshold"
        ],
        "hts_p_up_reliability_regret_threshold_source": base_ranking_correction[
            "hts_p_up_reliability_regret_threshold_source"
        ],
        "large_regret_reversal_pair_regret_priors": base_ranking_correction[
            "large_regret_reversal_pair_regret_priors"
        ],
        "large_regret_reversal_pair_regret_threshold": base_ranking_correction[
            "large_regret_reversal_pair_regret_threshold"
        ],
        "large_regret_reversal_pair_regret_threshold_source": (
            base_ranking_correction[
                "large_regret_reversal_pair_regret_threshold_source"
            ]
        ),
        "raw_weight_selection_metric_source": "shadow_split_only",
        "raw_weight_selection_objective": (
            "eligible candidates pass stricter p_up safety and positive high-score "
            "return, then maximize shadow selected return while minimizing mean "
            "regret and largest-regret action reversals"
        ),
        "raw_weight_max_shadow_p_up_disagreement_rate": (
            max_shadow_p_up_disagreement_rate
        ),
        "raw_weight_p_up_safety_buffer": (
            O_MAX_P_UP_ACTION_DISAGREEMENT_RATE - max_shadow_p_up_disagreement_rate
        ),
        "raw_weight_candidate_rows": candidate_rows,
        "selected_raw_weight_candidate": selected,
        "weight_enabled": weight > 0.0,
    }


def _shadow_raw_weight_candidates(
    p_edge_quantiles: dict[str, float],
) -> list[float]:
    q25 = float(p_edge_quantiles["q25"])
    median = float(p_edge_quantiles["median"])
    q75 = float(p_edge_quantiles["q75"])
    candidates = {
        0.0,
        q25,
        median,
        q75,
        q25 + median,
        q25 + q75,
        median + q75,
        q25 + median + q75,
        2.0 * q75,
    }
    return sorted(_bounded(value, 0.0, 1.0) for value in candidates)


def _apply_large_regret_adjusted_high_score_calibration(
    ranking_correction: dict[str, Any],
) -> None:
    p_edge_q25 = float(ranking_correction["p_up_edge_quantiles"]["q25"])
    reversal_penalty = float(
        ranking_correction.get("large_regret_reversal_penalty") or 0.0
    )
    threshold = 0.75 + max(0.0, p_edge_q25 - reversal_penalty)
    ranking_correction["high_score_calibration"]["high_score_threshold"] = threshold
    ranking_correction["high_score_calibration"]["high_score_threshold_source"] = (
        "0.75 + max(0, shadow_p_up_edge_q25 - selected_large_regret_reversal_penalty)"
    )
    ranking_correction["high_score_calibration"][
        "large_regret_reversal_penalty_adjusted"
    ] = reversal_penalty > 0.0


def _shadow_p_up_misalignment_penalty_candidates(
    p_edge_quantiles: dict[str, float],
) -> list[float]:
    q25 = float(p_edge_quantiles["q25"])
    median = float(p_edge_quantiles["median"])
    q75 = float(p_edge_quantiles["q75"])
    candidates = {
        0.0,
        q25,
        median,
        q75,
        q25 + median,
        q25 + q75,
        median + q75,
        2.0 * q75,
    }
    return sorted(_bounded(value, 0.0, 1.0) for value in candidates)


def _shadow_large_regret_reversal_threshold_candidates(
    p_edge_quantiles: dict[str, float],
) -> list[float]:
    candidates = {
        0.0,
        float(p_edge_quantiles["q25"]),
        float(p_edge_quantiles["median"]),
        float(p_edge_quantiles["q75"]),
    }
    return sorted(_bounded(value, 0.0, 0.5) for value in candidates)


def _shadow_large_regret_reversal_penalty_candidates(
    p_edge_quantiles: dict[str, float],
) -> list[float]:
    q25 = float(p_edge_quantiles["q25"])
    median = float(p_edge_quantiles["median"])
    q75 = float(p_edge_quantiles["q75"])
    candidates = {
        0.0,
        q75,
        median + q75,
        q25 + median + q75,
        1.0,
    }
    return sorted(_bounded(value, 0.0, 1.5) for value in candidates)


def _shadow_hts_p_up_reliability_penalty_candidates(
    p_edge_quantiles: dict[str, float],
) -> list[float]:
    q75 = float(p_edge_quantiles["q75"])
    candidates = {
        0.0,
        q75,
        2.0 * q75,
    }
    return sorted(_bounded(value, 0.0, 1.0) for value in candidates)


def _derive_shadow_large_regret_reversal_priors(
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        grouped[row["decision_group_id"]][str(row.get("action") or "")] = row
    priors = {
        "BUY_DOWN_HOLD_TO_SETTLEMENT->BUY_UP_HOLD_TO_SETTLEMENT": [],
        "BUY_UP_HOLD_TO_SETTLEMENT->BUY_DOWN_HOLD_TO_SETTLEMENT": [],
    }
    for group_rows in grouped.values():
        up = group_rows.get("BUY_UP_HOLD_TO_SETTLEMENT")
        down = group_rows.get("BUY_DOWN_HOLD_TO_SETTLEMENT")
        if up is None or down is None:
            continue
        up_return = float(up["replay_aligned_executable_label_target"])
        down_return = float(down["replay_aligned_executable_label_target"])
        if up_return > down_return:
            priors[
                "BUY_DOWN_HOLD_TO_SETTLEMENT->BUY_UP_HOLD_TO_SETTLEMENT"
            ].append(up_return - down_return)
        if down_return > up_return:
            priors[
                "BUY_UP_HOLD_TO_SETTLEMENT->BUY_DOWN_HOLD_TO_SETTLEMENT"
            ].append(down_return - up_return)
    all_positive_regrets = [
        value for values in priors.values() for value in values if value > 0.0
    ]
    threshold = (
        statistics.median(all_positive_regrets) if all_positive_regrets else 0.0
    )
    return {
        "pair_regret_threshold": threshold,
        "pair_regret_threshold_source": (
            "shadow_hold_to_settlement_up_down_positive_reversal_regret_median"
        ),
        "action_pair_priors": {
            key: {
                "positive_regret_support_count": len(values),
                "positive_regret_mean": statistics.mean(values) if values else 0.0,
                "positive_regret_sum": sum(values),
                "positive_regret_max": max(values, default=0.0),
                "positive_regret_median": statistics.median(values)
                if values
                else 0.0,
            }
            for key, values in sorted(priors.items())
        },
    }


def _derive_shadow_hts_p_up_reliability_thresholds(
    train_rows: list[dict[str, Any]],
    p_edge_quantiles: dict[str, float],
) -> dict[str, Any]:
    representative_rows = _representative_decision_rows(train_rows)
    return {
        "threshold_source": "shadow_split_decision_group_feature_quantiles",
        "p_up_confidence": p_edge_quantiles,
        "time_to_close": _p_edge_quantiles(
            [_normalized_time_to_close(row) for row in representative_rows]
        ),
        "spread": _p_edge_quantiles(
            [_normalized_spread(row) for row in representative_rows]
        ),
        "queue": _p_edge_quantiles(
            [
                _bounded(
                    float(row.get("entry_exit_quality_queue_fill") or 0.0),
                    0.0,
                    1.0,
                )
                for row in representative_rows
            ]
        ),
        "staleness": _p_edge_quantiles(
            [_normalized_staleness(row) for row in representative_rows]
        ),
    }


def _representative_decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representatives = []
    seen_groups = set()
    for row in rows:
        group_id = row["decision_group_id"]
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        representatives.append(row)
    return representatives


def _derive_shadow_hts_p_up_reliability_priors(
    train_rows: list[dict[str, Any]],
    bucket_thresholds: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        grouped[row["decision_group_id"]][str(row.get("action") or "")] = row
    cases = []
    for group_rows in grouped.values():
        up = group_rows.get("BUY_UP_HOLD_TO_SETTLEMENT")
        down = group_rows.get("BUY_DOWN_HOLD_TO_SETTLEMENT")
        if up is None or down is None:
            continue
        reference = up
        implied_side = _p_up_implied_side(reference)
        selected = up if implied_side == "UP" else down
        oracle = max(
            (up, down),
            key=lambda row: float(row["replay_aligned_executable_label_target"]),
        )
        selected_return = float(selected["replay_aligned_executable_label_target"])
        oracle_return = float(oracle["replay_aligned_executable_label_target"])
        context = _hts_p_up_reliability_bucket_context(
            selected,
            bucket_thresholds,
        )
        cases.append(
            {
                **context,
                "decision_group_id": selected["decision_group_id"],
                "market_id": selected.get("market_id"),
                "decision_ts": selected.get("decision_ts"),
                "selected_action": selected.get("action"),
                "oracle_action": oracle.get("action"),
                "selected_side": implied_side,
                "oracle_side": _side_from_action(str(oracle.get("action") or "")),
                "selected_return": selected_return,
                "oracle_return": oracle_return,
                "regret": oracle_return - selected_return,
                "p_up_confidently_wrong": implied_side
                != _side_from_action(str(oracle.get("action") or "")),
            }
        )
    positive_regrets = [float(case["regret"]) for case in cases if case["regret"] > 0.0]
    regret_quantiles = _p_edge_quantiles(positive_regrets)
    regret_threshold = regret_quantiles["q25"] if positive_regrets else 0.0
    min_support = max(1, int((len(cases) ** 0.5) / 2.0))
    regime_priors = _hts_reliability_case_summary(
        cases,
        key_fn=lambda case: case["hts_p_up_reliability_regime_key"],
        regret_threshold=regret_threshold,
        min_support=min_support,
    )
    return {
        "decision_group_count": len(cases),
        "regret_threshold": regret_threshold,
        "regret_threshold_source": "shadow_hts_p_up_positive_regret_q25",
        "min_support": min_support,
        "min_support_source": "max(1, floor(sqrt(shadow_hts_group_count) / 2))",
        "regime_priors": regime_priors,
        "bucket_diagnostics": {
            "by_p_up_confidence_bucket": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: case["p_up_confidence_bucket"],
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
            "by_time_to_close_bucket": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: case["time_to_close_bucket"],
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
            "by_spread_bucket": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: case["spread_bucket"],
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
            "by_queue_bucket": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: case["queue_bucket"],
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
            "by_staleness_bucket": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: case["staleness_bucket"],
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
            "by_selected_vs_oracle_side": _hts_reliability_case_summary(
                cases,
                key_fn=lambda case: (
                    f"{case['selected_side']}->{case['oracle_side']}"
                ),
                regret_threshold=regret_threshold,
                min_support=min_support,
            ),
        },
        "top_confidently_wrong_shadow_cases": sorted(
            (case for case in cases if bool(case["p_up_confidently_wrong"])),
            key=lambda case: (-float(case["regret"]), str(case["decision_group_id"])),
        )[:10],
    }


def _hts_reliability_case_summary(
    cases: list[dict[str, Any]],
    *,
    key_fn: Any,
    regret_threshold: float,
    min_support: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(key_fn(case))].append(case)
    summary = {}
    for key, bucket_cases in sorted(grouped.items()):
        regrets = [float(case["regret"]) for case in bucket_cases]
        positive_regrets = [value for value in regrets if value > 0.0]
        wrong_count = sum(1 for case in bucket_cases if case["p_up_confidently_wrong"])
        support = len(bucket_cases)
        positive_regret_mean = (
            statistics.mean(positive_regrets) if positive_regrets else 0.0
        )
        high_shadow_regret = (
            support >= min_support
            and positive_regret_mean >= regret_threshold
            and wrong_count > 0
        )
        summary[key] = {
            "support_count": support,
            "p_up_confidently_wrong_count": wrong_count,
            "p_up_confidently_wrong_rate": wrong_count / support if support else 0.0,
            "regret_mean": statistics.mean(regrets) if regrets else 0.0,
            "regret_sum": sum(regrets),
            "regret_max": max(regrets, default=0.0),
            "positive_regret_count": len(positive_regrets),
            "positive_regret_mean": positive_regret_mean,
            "high_shadow_regret": high_shadow_regret,
            "guard_reason_codes": ["shadow_hts_p_up_confidently_wrong_regime"]
            if high_shadow_regret
            else [],
        }
    return summary


def _derive_shadow_prior_weight(
    train_rows: list[dict[str, Any]],
    action_priors: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    prior_metrics = _shadow_ranker_metrics(
        train_rows,
        lambda row: action_priors.get(str(row.get("action") or ""), 0.0),
    )
    max_action_share = max(
        (
            count / prior_metrics["decision_group_count"]
            for count in prior_metrics["selected_action_counts"].values()
        ),
        default=1.0,
    )
    weight = 0.0 if max_action_share > 0.80 else 0.02 * (1.0 - max_action_share)
    return weight, {
        "prior_shadow_metrics": prior_metrics,
        "max_selected_action_share": max_action_share,
        "concentration_guard_triggered": max_action_share > 0.80,
        "weight_enabled": weight > 0.0,
    }


def _derive_shadow_microstructure_weight(
    train_rows: list[dict[str, Any]],
    p_edge_quantiles: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    correlation = _shadow_feature_target_correlation(
        [_microstructure_quality_proxy(row) for row in train_rows],
        [
            float(row["replay_aligned_executable_label_target"])
            for row in train_rows
        ],
    )
    weight = max(0.0, correlation) * p_edge_quantiles["q25"]
    return weight, {
        "shadow_microstructure_target_correlation": correlation,
        "weight_enabled": weight > 0.0,
    }


def _shadow_ranker_metrics(
    rows: list[dict[str, Any]],
    score_fn: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    selected_action_counts: Counter[str] = Counter()
    top_hits = 0
    selected_returns = []
    regrets = []
    for group_rows in grouped.values():
        selected = max(group_rows, key=score_fn)
        oracle = max(
            group_rows,
            key=lambda row: float(row["realized_replay_return"]),
        )
        selected_action_counts[str(selected.get("action") or "")] += 1
        if selected["action"] == oracle["action"]:
            top_hits += 1
        selected_return = float(selected["realized_replay_return"])
        selected_returns.append(selected_return)
        regrets.append(float(oracle["realized_replay_return"]) - selected_return)
    group_count = len(grouped)
    return {
        "decision_group_count": group_count,
        "top1_realized_best_action_hit_rate": top_hits / group_count
        if group_count
        else 0.0,
        "selected_action_realized_replay_return_sum": sum(selected_returns),
        "mean_regret": statistics.mean(regrets) if regrets else 0.0,
        "selected_action_counts": dict(sorted(selected_action_counts.items())),
    }


def _shadow_derived_ranker_probe_score(
    row: dict[str, Any],
    probe_score_config: dict[str, Any],
) -> float:
    p_edge = abs(float(row.get("p_up") or 0.5) - 0.5)
    action_family = _action_family(str(row.get("action") or ""))
    alignment = _p_up_side_alignment_score(row)
    if action_family == "NO_TRADE":
        return float(probe_score_config["probe_no_trade_base_score"]) + max(
            0.0,
            float(probe_score_config["probe_no_trade_weak_edge_cutoff"]) - p_edge,
        )
    if action_family == "HOLD_TO_SETTLEMENT":
        return float(probe_score_config["probe_hold_to_settlement_base_score"]) + alignment
    return float(probe_score_config["probe_sell_before_close_base_score"]) + (
        float(probe_score_config["probe_sell_before_close_alignment_weight"])
        * alignment
    )


def _shadow_feature_target_correlation(
    feature_values: list[float],
    target_values: list[float],
) -> float:
    if not feature_values or len(feature_values) != len(target_values):
        return 0.0
    feature_std = statistics.pstdev(feature_values)
    target_std = statistics.pstdev(target_values)
    if feature_std == 0.0 or target_std == 0.0:
        return 0.0
    feature_mean = statistics.mean(feature_values)
    target_mean = statistics.mean(target_values)
    covariance = statistics.mean(
        (feature - feature_mean) * (target - target_mean)
        for feature, target in zip(feature_values, target_values, strict=True)
    )
    return covariance / (feature_std * target_std)


def _without_key(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _apply_o_shadow_ranking_correction(
    *,
    rows: list[dict[str, Any]],
    deployable_available: bool,
    ranking_correction: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    raw_z_by_key = {}
    for group_rows in grouped.values():
        raw_scores = [float(row["o_raw_ridge_model_score"]) for row in group_rows]
        raw_mean = statistics.mean(raw_scores) if raw_scores else 0.0
        raw_std = statistics.pstdev(raw_scores) or 1.0
        for row in group_rows:
            raw_z_by_key[(row["decision_group_id"], row["action"])] = (
                float(row["o_raw_ridge_model_score"]) - raw_mean
            ) / raw_std
    scored_rows = []
    for row in rows:
        raw_z = raw_z_by_key[(row["decision_group_id"], row["action"])]
        components = _o_model_score_components(row, raw_z, ranking_correction)
        score = sum(float(value) for value in components.values())
        scored_rows.append(
            {
                **row,
                "o_model_predicted_score": score,
                "o_model_score_source": "model_predicted_score",
                "o_group_normalized_raw_model_score": raw_z,
                "o_model_score_components": components,
                "hts_p_up_reliability_buckets": (
                    _hts_p_up_reliability_bucket_context(
                        row,
                        ranking_correction[
                            "hts_p_up_reliability_bucket_thresholds"
                        ],
                    )
                    if _action_family(str(row.get("action") or ""))
                    == "HOLD_TO_SETTLEMENT"
                    else {}
                ),
                "deployable_model_score_available": deployable_available,
                "ranking_score_source_by_variant": {
                    "current_source_baseline": "observed_source_score",
                    O_MODEL_PREDICTED_VARIANT: "model_predicted_score",
                    **dict.fromkeys(
                        O_LABEL_DIAGNOSTIC_VARIANTS,
                        "label_diagnostic_score",
                    ),
                },
            }
        )
    return scored_rows


def _o_model_score_components(
    row: dict[str, Any],
    raw_z: float,
    ranking_correction: dict[str, Any],
) -> dict[str, float]:
    action = str(row.get("action") or "")
    family = _action_family(action)
    p_edge = abs(float(row.get("p_up") or 0.5) - 0.5)
    weak_cutoff = float(ranking_correction["weak_opportunity_p_edge_cutoff"])
    if family == "NO_TRADE":
        base_score = float(ranking_correction["no_trade_base_score"])
        side_alignment_component = 0.0
        confidence_component = max(0.0, weak_cutoff - p_edge)
    elif family == "HOLD_TO_SETTLEMENT":
        base_score = float(ranking_correction["trade_base_score"])
        side_alignment_component = _p_up_side_alignment_score(row)
        confidence_component = (
            float(ranking_correction["confidence_bonus"])
            if p_edge >= weak_cutoff
            else float(ranking_correction["weak_opportunity_trade_penalty"])
        )
    else:
        base_score = float(ranking_correction["sell_before_close_base_score"])
        side_alignment_component = 0.5 * _p_up_side_alignment_score(row)
        confidence_component = (
            float(ranking_correction["sell_before_close_confidence_bonus"])
            if p_edge >= weak_cutoff
            else float(ranking_correction["sell_before_close_weak_penalty"])
        )
    prior = (
        float(ranking_correction["action_shadow_priors"].get(action, 0.0))
        + float(ranking_correction["action_family_shadow_priors"].get(family, 0.0))
    )
    raw_component = (
        float(ranking_correction["group_normalized_raw_model_weight"])
        * _bounded(raw_z, -2.0, 2.0)
    )
    p_up_misalignment_penalty = 0.0
    if (
        ("BUY_UP" in action or "BUY_DOWN" in action)
        and side_alignment_component < 0.0
        and raw_component > 0.0
    ):
        p_up_misalignment_penalty = -float(
            ranking_correction.get("p_up_misalignment_raw_positive_penalty", 0.0)
        )
    large_regret_reversal_guard = 0.0
    if _large_regret_reversal_guard_applies(
        action=action,
        raw_component=raw_component,
        side_alignment_component=side_alignment_component,
        ranking_correction=ranking_correction,
    ):
        large_regret_reversal_guard = -float(
            ranking_correction.get("large_regret_reversal_penalty", 0.0)
        )
    hts_p_up_reliability_guard = 0.0
    if _hts_p_up_reliability_guard_applies(row, ranking_correction):
        hts_p_up_reliability_guard = -float(
            ranking_correction.get("hts_p_up_reliability_penalty", 0.0)
        )
    hts_p_up_reliability_no_trade_buffer = 0.0
    if family == "NO_TRADE" and bool(
        ranking_correction.get("hts_p_up_reliability_no_trade_buffer_enabled", False)
    ):
        hts_p_up_reliability_no_trade_buffer = float(
            ranking_correction.get("hts_p_up_reliability_no_trade_buffer", 0.0)
        )
    return {
        "base_score": base_score,
        "p_up_side_alignment_component": side_alignment_component,
        "confidence_or_weak_opportunity_component": confidence_component,
        "group_normalized_raw_model_component": raw_component,
        "p_up_misalignment_penalty_component": p_up_misalignment_penalty,
        "large_regret_reversal_guard_component": large_regret_reversal_guard,
        "hts_p_up_reliability_guard_component": hts_p_up_reliability_guard,
        "hts_p_up_reliability_no_trade_buffer_component": (
            hts_p_up_reliability_no_trade_buffer
        ),
        "shadow_action_family_prior_component": (
            float(ranking_correction["shadow_action_family_prior_weight"]) * prior
        ),
        "microstructure_quality_component": (
            float(ranking_correction["microstructure_quality_weight"])
            * _microstructure_quality_proxy(row)
        ),
    }


def _hts_p_up_reliability_guard_applies(
    row: dict[str, Any],
    ranking_correction: dict[str, Any],
) -> bool:
    if not bool(ranking_correction.get("hts_p_up_reliability_guard_enabled", False)):
        return False
    action = str(row.get("action") or "")
    if _action_family(action) != "HOLD_TO_SETTLEMENT":
        return False
    if _side_from_action(action) != _p_up_implied_side(row):
        return False
    context = _hts_p_up_reliability_bucket_context(
        row,
        ranking_correction["hts_p_up_reliability_bucket_thresholds"],
    )
    return _hts_p_up_reliability_context_has_high_shadow_regret(
        context,
        ranking_correction,
    )


def _hts_p_up_reliability_context_has_high_shadow_regret(
    context: dict[str, Any],
    ranking_correction: dict[str, Any],
) -> bool:
    regime = ranking_correction.get("hts_p_up_reliability_regime_priors", {}).get(
        context["hts_p_up_reliability_regime_key"],
        {},
    )
    if bool(regime.get("high_shadow_regret", False)):
        return True
    bucket_diagnostics = ranking_correction.get(
        "hts_p_up_reliability_bucket_diagnostics",
        {},
    )
    bucket_lookup = {
        "by_p_up_confidence_bucket": "p_up_confidence_bucket",
        "by_time_to_close_bucket": "time_to_close_bucket",
        "by_spread_bucket": "spread_bucket",
        "by_queue_bucket": "queue_bucket",
        "by_staleness_bucket": "staleness_bucket",
    }
    for diagnostic_key, context_key in bucket_lookup.items():
        bucket = bucket_diagnostics.get(diagnostic_key, {}).get(
            context.get(context_key),
            {},
        )
        if bool(bucket.get("high_shadow_regret", False)):
            return True
    return False


def _large_regret_reversal_guard_applies(
    *,
    action: str,
    raw_component: float,
    side_alignment_component: float,
    ranking_correction: dict[str, Any],
) -> bool:
    if not bool(ranking_correction.get("large_regret_reversal_guard_enabled", False)):
        return False
    if action not in {
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
    }:
        return False
    if raw_component <= 0.0:
        return False
    alignment_threshold = float(
        ranking_correction.get("large_regret_reversal_alignment_threshold", 0.0)
    )
    p_up_edge = abs(side_alignment_component)
    opposite_alignment_conflict = side_alignment_component <= -alignment_threshold
    high_reversal_exposure = p_up_edge <= float(
        ranking_correction.get("large_regret_reversal_confidence_edge_ceiling", 0.0)
    )
    if not opposite_alignment_conflict and not high_reversal_exposure:
        return False
    opposite_action = _opposite_hold_to_settlement_action(action)
    if opposite_action is None:
        return False
    pair_key = f"{action}->{opposite_action}"
    pair_prior = (
        ranking_correction.get("large_regret_reversal_pair_regret_priors", {}).get(
            pair_key,
            {},
        )
    )
    if int(pair_prior.get("positive_regret_support_count") or 0) <= 0:
        return False
    return float(pair_prior.get("positive_regret_mean") or 0.0) >= float(
        ranking_correction.get("large_regret_reversal_pair_regret_threshold", 0.0)
    )


def _hts_p_up_reliability_bucket_context(
    row: dict[str, Any],
    bucket_thresholds: dict[str, Any],
) -> dict[str, Any]:
    p_up = _bounded(float(row.get("p_up") or 0.5), 0.0, 1.0)
    p_down = 1.0 - p_up
    p_edge = abs(p_up - 0.5)
    implied_side = _p_up_implied_side(row)
    p_up_confidence_bucket = _quantile_bucket(
        p_edge,
        bucket_thresholds["p_up_confidence"],
        labels=("weak", "moderate", "strong", "very_strong"),
    )
    context = {
        "p_up": p_up,
        "p_down": p_down,
        "p_up_edge": p_edge,
        "p_up_implied_side": implied_side,
        "p_up_confidence_bucket": p_up_confidence_bucket,
        "time_to_close_bucket": _quantile_bucket(
            _normalized_time_to_close(row),
            bucket_thresholds["time_to_close"],
            labels=("near", "mid", "far", "very_far"),
        ),
        "spread_bucket": _quantile_bucket(
            _normalized_spread(row),
            bucket_thresholds["spread"],
            labels=("tight", "moderate", "wide", "very_wide"),
        ),
        "queue_bucket": _quantile_bucket(
            _bounded(float(row.get("entry_exit_quality_queue_fill") or 0.0), 0.0, 1.0),
            bucket_thresholds["queue"],
            labels=("low", "medium", "high", "very_high"),
        ),
        "staleness_bucket": _quantile_bucket(
            _normalized_staleness(row),
            bucket_thresholds["staleness"],
            labels=("fresh", "normal", "stale", "very_stale"),
        ),
    }
    context["hts_p_up_reliability_regime_key"] = (
        f"side={implied_side}|p_up_confidence={p_up_confidence_bucket}"
    )
    return context


def _quantile_bucket(
    value: float,
    quantiles: dict[str, float],
    *,
    labels: tuple[str, str, str, str],
) -> str:
    if value <= float(quantiles["q25"]):
        return labels[0]
    if value <= float(quantiles["median"]):
        return labels[1]
    if value <= float(quantiles["q75"]):
        return labels[2]
    return labels[3]


def _p_up_implied_side(row: dict[str, Any]) -> str:
    p_up = _bounded(float(row.get("p_up") or 0.5), 0.0, 1.0)
    return "UP" if p_up >= 0.5 else "DOWN"


def _opposite_hold_to_settlement_action(action: str) -> str | None:
    if action == "BUY_UP_HOLD_TO_SETTLEMENT":
        return "BUY_DOWN_HOLD_TO_SETTLEMENT"
    if action == "BUY_DOWN_HOLD_TO_SETTLEMENT":
        return "BUY_UP_HOLD_TO_SETTLEMENT"
    return None


def _p_up_side_alignment_score(row: dict[str, Any]) -> float:
    p_up = _bounded(float(row.get("p_up") or 0.5), 0.0, 1.0)
    action = str(row.get("action") or "")
    if "BUY_UP" in action:
        return p_up - 0.5
    if "BUY_DOWN" in action:
        return 0.5 - p_up
    return 0.0


def _microstructure_quality_proxy(row: dict[str, Any]) -> float:
    queue = _bounded(float(row.get("entry_exit_quality_queue_fill") or 0.0), 0.0, 1.0)
    spread = _normalized_spread(row)
    staleness = _normalized_staleness(row)
    time_to_close = _normalized_time_to_close(row)
    entry_ask = _bounded(float(row.get("entry_quality_ask") or 0.0), 0.0, 1.0)
    exit_bid_proxy = _decision_time_exit_bid_proxy(row)
    return (
        0.01 * (queue - 0.80)
        - 0.02 * spread
        - 0.0005 * staleness
        + 0.0005 * time_to_close
        - 0.005 * entry_ask
        + 0.005 * exit_bid_proxy
    )


def _fit_ridge_regression(
    features: list[list[float]],
    targets: list[float],
    *,
    ridge_lambda: float = 1.0e-6,
) -> dict[str, Any]:
    width = len(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for vector, target in zip(features, targets, strict=True):
        for i in range(width):
            xty[i] += vector[i] * target
            for j in range(width):
                xtx[i][j] += vector[i] * vector[j]
    for i in range(1, width):
        xtx[i][i] += ridge_lambda
    return {
        "coefficients": _solve_linear_system(xtx, xty),
        "ridge_lambda": ridge_lambda,
    }


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector, strict=True)]
    for pivot_index in range(size):
        best = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[best][pivot_index]) < 1.0e-12:
            augmented[best][pivot_index] = 1.0e-12
        if best != pivot_index:
            augmented[pivot_index], augmented[best] = (
                augmented[best],
                augmented[pivot_index],
            )
        pivot = augmented[pivot_index][pivot_index]
        for col in range(pivot_index, size + 1):
            augmented[pivot_index][col] /= pivot
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for col in range(pivot_index, size + 1):
                augmented[row_index][col] -= factor * augmented[pivot_index][col]
    return [augmented[row_index][size] for row_index in range(size)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
        O_MODEL_PREDICTED_VARIANT: float(row.get("o_model_predicted_score") or 0.0),
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
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    high_score_threshold = _model_high_score_threshold(model_training_summary)
    variant_metrics = {
        variant: _ranking_metrics(rows, variant, high_score_threshold)
        for variant in O_VARIANTS
    }
    split_metrics = _split_metric_views(
        rows,
        O_MODEL_PREDICTED_VARIANT,
        high_score_threshold,
    )
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
        "train_shadow_metrics": split_metrics["train_shadow"],
        "validation_metrics": split_metrics["validation"],
        "all_metrics": split_metrics["all"],
        "eligibility_metric_source": "validation_metrics_only",
        "primary_variant_name": O_MODEL_PREDICTED_VARIANT,
        "primary_ranking_score_source": "model_predicted_score",
        "model_predicted_candidate_name": O_MODEL_PREDICTED_VARIANT,
        "deployable_model_score_available": model_training_summary[
            "deployable_model_score_available"
        ],
        "o_model_training_summary": model_training_summary,
        "correction_constants_source": model_training_summary[
            "correction_constants_source"
        ],
        "correction_config_hash": model_training_summary["correction_config_hash"],
        "probe_constants_source": model_training_summary["probe_constants_source"],
        "probe_config_hash": model_training_summary["probe_config_hash"],
        "high_score_threshold": high_score_threshold,
        "high_score_threshold_source": model_training_summary[
            "ranking_correction_config"
        ]["high_score_calibration"]["high_score_threshold_source"],
        "label_diagnostic_variants": list(O_LABEL_DIAGNOSTIC_VARIANTS),
        "label_diagnostic_variants_deployable": False,
        "ranking_metric_scope": variant_metrics[O_MODEL_PREDICTED_VARIANT][
            "ranking_metric_scope"
        ],
        "decision_group_completeness_summary": variant_metrics[
            O_MODEL_PREDICTED_VARIANT
        ]["decision_group_completeness_summary"],
        "action_candidate_construction_summary": _action_candidate_construction_summary(
            rows
        ),
        "full_source_model_ranking_quality_claimed": variant_metrics[
            O_MODEL_PREDICTED_VARIANT
        ]["full_source_model_ranking_quality_claimed"],
        "top1_realized_best_action_hit_rate": variant_metrics[
            O_MODEL_PREDICTED_VARIANT
        ]["top1_realized_best_action_hit_rate"],
        "top2_realized_best_action_hit_rate": variant_metrics[
            O_MODEL_PREDICTED_VARIANT
        ]["top2_realized_best_action_hit_rate"],
        "top3_realized_best_action_hit_rate": variant_metrics[
            O_MODEL_PREDICTED_VARIANT
        ]["top3_realized_best_action_hit_rate"],
        "mean_regret": variant_metrics[O_MODEL_PREDICTED_VARIANT]["mean_regret"],
        "ranking_rows": [
            _compact_ranking_row(row, O_MODEL_PREDICTED_VARIANT)
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
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    del config
    model_overlap = sorted(
        set(O_DEPLOYABLE_MODEL_FEATURE_NAMES).intersection(O_FORBIDDEN_MODEL_INPUT_FIELDS)
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
        "ranking_score_source": "model_predicted_score",
        "deployable_model_score_available": model_training_summary[
            "deployable_model_score_available"
        ],
        "model_input_fields_decision_time_only": list(
            O_DEPLOYABLE_MODEL_FEATURE_NAMES
        ),
        "model_training_summary": model_training_summary,
        "label_diagnostic_score_fields": [
            "replay_aligned_executable_label_target",
            "label_family_prior",
            "label_group_mean",
        ],
        "label_diagnostic_variants": list(O_LABEL_DIAGNOSTIC_VARIANTS),
        "label_diagnostic_variants_deployable": False,
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
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = []
    high_score_threshold = _model_high_score_threshold(model_training_summary)
    for variant in O_VARIANTS:
        metrics = _ranking_metrics(rows, variant, high_score_threshold)
        split_metrics = _split_metric_views(rows, variant, high_score_threshold)
        reasons = [
            "diagnostic_only_no_paper_live_unlock",
            "current_m_m2_n_n2_evidence_not_o_promotion_evidence",
            "future_unseen_o_holdout_required",
        ]
        if metrics["high_score_realized_return_mean"] <= 0.0:
            reasons.append("high_score_realized_return_mean_not_positive")
        eligible_for_source_model_gate = variant == O_MODEL_PREDICTED_VARIANT
        excluded_reason = None
        if variant in O_LABEL_DIAGNOSTIC_VARIANTS:
            excluded_reason = "label_diagnostic_score_not_model_predicted"
        elif variant == "current_source_baseline":
            excluded_reason = "observed_source_score_incomplete_counterfactuals"
        candidate_rows.append(
            {
                "candidate_name": variant,
                "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME
                if variant != "current_source_baseline"
                else "current_source_baseline",
                "ranking_score_source": metrics["ranking_score_source"],
                "deployable_model_score_available": metrics[
                    "deployable_model_score_available"
                ],
                "label_diagnostic_score": metrics["ranking_score_source"]
                == "label_diagnostic_score",
                "eligible_for_source_model_gate": eligible_for_source_model_gate,
                "excluded_from_eligibility_reason": excluded_reason,
                "model_training_summary": model_training_summary
                if variant == O_MODEL_PREDICTED_VARIANT
                else None,
                "correction_constants_source": model_training_summary[
                    "correction_constants_source"
                ]
                if variant == O_MODEL_PREDICTED_VARIANT
                else None,
                "correction_config_hash": model_training_summary[
                    "correction_config_hash"
                ]
                if variant == O_MODEL_PREDICTED_VARIANT
                else None,
                "probe_constants_source": model_training_summary[
                    "probe_constants_source"
                ]
                if variant == O_MODEL_PREDICTED_VARIANT
                else None,
                "probe_config_hash": model_training_summary["probe_config_hash"]
                if variant == O_MODEL_PREDICTED_VARIANT
                else None,
                "train_shadow_metrics": split_metrics["train_shadow"],
                "validation_metrics": split_metrics["validation"],
                "all_metrics": split_metrics["all"],
                "eligibility_metric_source": "validation_metrics_only"
                if eligible_for_source_model_gate
                else "excluded_from_source_model_gate",
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
                    "HOLD_TO_SETTLEMENT": False,
                    "NO_TRADE": False,
                },
                "action_family_gate_metrics": metrics["action_family_gate_metrics"],
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
        "model_predicted_candidate_name": O_MODEL_PREDICTED_VARIANT,
        "model_training_summary": model_training_summary,
        "label_diagnostic_variants": list(O_LABEL_DIAGNOSTIC_VARIANTS),
        "candidate_rows": candidate_rows,
        "eligible_candidate_count": 0,
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    report["o_source_candidate_comparison_report_id"] = canonical_json_sha256(report)
    return report


def _source_model_eligibility_gate_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    leakage_report: dict[str, Any],
) -> dict[str, Any]:
    high_score_threshold = _model_high_score_threshold(model_training_summary)
    validation_metrics = _split_metric_views(
        rows,
        O_MODEL_PREDICTED_VARIANT,
        high_score_threshold,
    )["validation"]
    p_up_summary = _p_up_action_disagreement_summary(
        rows=rows,
        variant=O_MODEL_PREDICTED_VARIANT,
        split="validation",
    )
    deployable = bool(model_training_summary["deployable_model_score_available"])
    leakage_passed = bool(leakage_report["leakage_audit_passed"])
    calibration_support_passed = (
        int(validation_metrics["decision_group_count"])
        >= O_MIN_VALIDATION_DECISION_GROUPS
        and int(validation_metrics["high_score_support_count"])
        >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
    )
    calibration_quality_passed = (
        float(validation_metrics["top1_realized_best_action_hit_rate"])
        >= O_MIN_TOP1_HIT_RATE
        and float(validation_metrics["mean_regret"]) <= O_MAX_MEAN_REGRET
    )
    action_family_paper_decision_eligible = _validation_action_family_gate_passed(
        validation_metrics
    )
    largest_winner_dependency = validation_metrics["largest_winner_dependency"]
    best_action_concentration_passed = (
        float(validation_metrics["NO_TRADE_selection_rate"])
        <= O_MAX_NO_TRADE_SELECTION_RATE
        and not bool(
            largest_winner_dependency[
                "total_return_positive_only_because_of_largest_winner"
            ]
        )
    )
    p_up_action_disagreement_within_limit = bool(
        p_up_summary["candidate_scoped_p_up_action_disagreement_within_limit"]
    )
    high_score_return_positive = (
        int(validation_metrics["high_score_support_count"])
        >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
        and float(validation_metrics["high_score_realized_return_mean"]) > 0.0
        and float(validation_metrics["high_score_realized_return_sum"]) > 0.0
    )
    action_value_paper_decision_eligible = all(
        (
            deployable,
            leakage_passed,
            calibration_support_passed,
            calibration_quality_passed,
            action_family_paper_decision_eligible,
            best_action_concentration_passed,
            p_up_action_disagreement_within_limit,
            high_score_return_positive,
        )
    )
    source_model_candidate_eligible = action_value_paper_decision_eligible
    reason_codes = _o_gate_reason_codes(
        deployable=deployable,
        leakage_passed=leakage_passed,
        calibration_support_passed=calibration_support_passed,
        calibration_quality_passed=calibration_quality_passed,
        action_family_paper_decision_eligible=action_family_paper_decision_eligible,
        best_action_concentration_passed=best_action_concentration_passed,
        p_up_action_disagreement_within_limit=p_up_action_disagreement_within_limit,
        high_score_return_positive=high_score_return_positive,
    )
    report = {
        "schema_version": O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_source_model_eligibility_gate",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "ranking_score_source": "model_predicted_score",
        "deployable_model_score_available": deployable,
        "correction_constants_source": model_training_summary[
            "correction_constants_source"
        ],
        "correction_config_hash": model_training_summary["correction_config_hash"],
        "probe_constants_source": model_training_summary["probe_constants_source"],
        "probe_config_hash": model_training_summary["probe_config_hash"],
        "high_score_threshold": high_score_threshold,
        "high_score_threshold_source": model_training_summary[
            "ranking_correction_config"
        ]["high_score_calibration"]["high_score_threshold_source"],
        "correction_config_hash_verified": (
            canonical_json_sha256(
                _without_key(
                    model_training_summary["ranking_correction_config"],
                    "correction_config_hash",
                )
            )
            == model_training_summary["correction_config_hash"]
        ),
        "probe_config_hash_verified": (
            canonical_json_sha256(
                _without_key(
                    model_training_summary["ranking_correction_config"][
                        "probe_score_config"
                    ],
                    "probe_config_hash",
                )
            )
            == model_training_summary["probe_config_hash"]
        ),
        "eligible_for_source_model_gate": True,
        "validation_metrics_only_for_eligibility": True,
        "train_shadow_metrics": _split_metric_views(
            rows,
            O_MODEL_PREDICTED_VARIANT,
            high_score_threshold,
        )["train_shadow"],
        "validation_metrics": validation_metrics,
        "all_metrics": _split_metric_views(
            rows,
            O_MODEL_PREDICTED_VARIANT,
            high_score_threshold,
        )["all"],
        "gate_thresholds": {
            "min_validation_decision_group_count": O_MIN_VALIDATION_DECISION_GROUPS,
            "min_high_score_support_count": O_MIN_HIGH_SCORE_SUPPORT_COUNT,
            "min_top1_realized_best_action_hit_rate": O_MIN_TOP1_HIT_RATE,
            "max_mean_regret": O_MAX_MEAN_REGRET,
            "max_NO_TRADE_selection_rate": O_MAX_NO_TRADE_SELECTION_RATE,
            "max_p_up_action_disagreement_rate": (
                O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
            ),
            "high_score_realized_return_mean_must_be_positive": True,
            "high_score_realized_return_sum_must_be_positive": True,
            "high_score_threshold": high_score_threshold,
            "high_score_threshold_source": model_training_summary[
                "ranking_correction_config"
            ]["high_score_calibration"]["high_score_threshold_source"],
            "p_up_safety_target_disagreement_rate": model_training_summary[
                "ranking_correction_config"
            ]["p_up_safety_target_disagreement_rate"],
            "p_up_safety_target_is_hard_gate": False,
        },
        "source_model_candidate_eligible": source_model_candidate_eligible,
        "calibration_support_passed": calibration_support_passed,
        "calibration_quality_passed": calibration_quality_passed,
        "action_family_paper_decision_eligible": (
            action_family_paper_decision_eligible
        ),
        "best_action_concentration_passed": best_action_concentration_passed,
        "p_up_action_disagreement_within_limit": (
            p_up_action_disagreement_within_limit
        ),
        "action_value_paper_decision_eligible": (
            action_value_paper_decision_eligible
        ),
        "high_score_support_count": validation_metrics["high_score_support_count"],
        "high_score_realized_return_mean": validation_metrics[
            "high_score_realized_return_mean"
        ],
        "high_score_realized_return_sum": validation_metrics[
            "high_score_realized_return_sum"
        ],
        "mean_regret": validation_metrics["mean_regret"],
        "top1_realized_best_action_hit_rate": validation_metrics[
            "top1_realized_best_action_hit_rate"
        ],
        "top2_realized_best_action_hit_rate": validation_metrics[
            "top2_realized_best_action_hit_rate"
        ],
        "top3_realized_best_action_hit_rate": validation_metrics[
            "top3_realized_best_action_hit_rate"
        ],
        "largest_winner_dependency": largest_winner_dependency,
        "NO_TRADE_selection_rate": validation_metrics["NO_TRADE_selection_rate"],
        "action_family_selected_return_breakdown": validation_metrics[
            "action_family_selected_return_breakdown"
        ],
        "side_selected_return_breakdown": validation_metrics[
            "side_selected_return_breakdown"
        ],
        "p_up_action_disagreement_summary": p_up_summary,
        "p_up_safety_target_met": (
            float(
                p_up_summary[
                    "candidate_scoped_p_up_action_disagreement_rate"
                ]
            )
            < float(
                model_training_summary["ranking_correction_config"][
                    "p_up_safety_target_disagreement_rate"
                ]
            )
        ),
        "p_up_safety_target_note": (
            "diagnostic_target_not_hard_gate_shadow_selection_prioritizes_buffer"
        ),
        "leakage_audit_passed": leakage_passed,
        "ineligible_reason_codes": reason_codes,
        "future_unseen_holdout_required": True,
        "promotion_evidence_eligible": False,
        "promotion_blocking_reason_codes": ["future_unseen_holdout_required"],
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["o_source_model_eligibility_gate_report_id"] = canonical_json_sha256(
        report
    )
    return report


def _freeze_readiness_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    eligibility_gate_report: dict[str, Any],
) -> dict[str, Any]:
    label_grid_payload = [
        {
            "decision_group_id": row["decision_group_id"],
            "market_id": row.get("market_id"),
            "decision_ts": row.get("decision_ts"),
            "action": row.get("action"),
            "label_target": row.get("replay_aligned_executable_label_target"),
            "split": row.get("split"),
        }
        for row in rows
    ]
    model_manifest = {
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "ranking_score_source": "model_predicted_score",
        "model_training_summary": model_training_summary,
    }
    source_model_candidate_eligible = bool(
        eligibility_gate_report["source_model_candidate_eligible"]
    )
    freeze_ready = source_model_candidate_eligible
    blocking_reasons = []
    if not source_model_candidate_eligible:
        blocking_reasons.append("source_model_validation_gates_not_passed")
    if not bool(model_training_summary["deployable_model_score_available"]):
        blocking_reasons.append("deployable_model_score_unavailable")
    report = {
        "schema_version": O_FREEZE_READINESS_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_freeze_readiness",
        "diagnostic_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "freeze_ready": freeze_ready,
        "ranking_score_source": "model_predicted_score",
        "correction_constants_source": model_training_summary[
            "correction_constants_source"
        ],
        "correction_config_hash": model_training_summary["correction_config_hash"],
        "probe_constants_source": model_training_summary["probe_constants_source"],
        "probe_config_hash": model_training_summary["probe_config_hash"],
        "model_sha256": canonical_json_sha256(model_training_summary),
        "model_manifest_sha256": canonical_json_sha256(model_manifest),
        "training_data_hash": canonical_json_sha256(
            [
                {
                    "decision_group_id": row["decision_group_id"],
                    "action": row["action"],
                    "split": row["split"],
                    "features": _deployable_model_features(row),
                    "target": row["replay_aligned_executable_label_target"],
                }
                for row in rows
                if row.get("split") == "shadow"
            ]
        ),
        "label_grid_hash": canonical_json_sha256(label_grid_payload),
        "feature_schema_hash": canonical_json_sha256(
            list(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
        ),
        "split_hash": canonical_json_sha256(
            sorted(
                {
                    row["decision_group_id"]: row["split"]
                    for row in rows
                }.items()
            )
        ),
        "candidate_config_hash": canonical_json_sha256(
            {
                "candidate_name": O_MODEL_PREDICTED_VARIANT,
                "high_score_threshold": _model_high_score_threshold(
                    model_training_summary
                ),
                "gate_thresholds": eligibility_gate_report["gate_thresholds"],
                "feature_names": list(O_DEPLOYABLE_MODEL_FEATURE_NAMES),
                "ranking_correction_config": model_training_summary[
                    "ranking_correction_config"
                ],
            }
        ),
        "freeze_blocking_reason_codes": blocking_reasons,
        "source_model_candidate_eligible": source_model_candidate_eligible,
        "future_unseen_holdout_required": True,
        "promotion_evidence_eligible": False,
        "promotion_blocking_reason_codes": ["future_unseen_holdout_required"],
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["o_freeze_readiness_report_id"] = canonical_json_sha256(report)
    return report


def _o_gate_reason_codes(
    *,
    deployable: bool,
    leakage_passed: bool,
    calibration_support_passed: bool,
    calibration_quality_passed: bool,
    action_family_paper_decision_eligible: bool,
    best_action_concentration_passed: bool,
    p_up_action_disagreement_within_limit: bool,
    high_score_return_positive: bool,
) -> list[str]:
    reasons = [
        "diagnostic_only_no_paper_live_unlock",
        "future_unseen_holdout_required",
    ]
    if not deployable:
        reasons.append("deployable_model_score_unavailable")
    if not leakage_passed:
        reasons.append("model_input_leakage_audit_failed")
    if not calibration_support_passed:
        reasons.append("validation_calibration_support_gate_failed")
    if not calibration_quality_passed:
        reasons.append("validation_calibration_quality_gate_failed")
    if not action_family_paper_decision_eligible:
        reasons.append("validation_action_family_return_gate_failed")
    if not best_action_concentration_passed:
        reasons.append("validation_best_action_concentration_gate_failed")
    if not p_up_action_disagreement_within_limit:
        reasons.append("validation_p_up_action_disagreement_gate_failed")
    if not high_score_return_positive:
        reasons.append("validation_high_score_return_gate_failed")
    return reasons


def _validation_action_family_gate_passed(metrics: dict[str, Any]) -> bool:
    breakdown = metrics["action_family_selected_return_breakdown"]
    if not breakdown:
        return False
    traded_families = {
        family: values
        for family, values in breakdown.items()
        if family != "NO_TRADE" and int(values["support_count"]) > 0
    }
    if not traded_families:
        return False
    return all(
        float(values["selected_return_sum"]) > 0.0
        for values in traded_families.values()
    )


def _p_up_action_disagreement_summary(
    *,
    rows: list[dict[str, Any]],
    variant: str,
    split: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split:
            grouped[row["decision_group_id"]].append(row)
    selected_rows = []
    for group_rows in grouped.values():
        if not group_rows:
            continue
        selected_rows.append(
            max(
                group_rows,
                key=lambda row: float(row["variant_scores"][variant]),
            )
        )
    comparable = []
    missing = 0
    disagreement = 0
    for row in selected_rows:
        action = str(row.get("action") or "")
        if "BUY_UP" not in action and "BUY_DOWN" not in action:
            continue
        if row.get("p_up") is None:
            missing += 1
            continue
        p_up = float(row["p_up"])
        comparable.append(row)
        if ("BUY_UP" in action and p_up < 0.50) or (
            "BUY_DOWN" in action and p_up > 0.50
        ):
            disagreement += 1
    comparable_count = len(comparable)
    disagreement_rate = disagreement / comparable_count if comparable_count else 0.0
    within_limit = (
        comparable_count > 0
        and disagreement_rate <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
    )
    return {
        "split": split,
        "selected_decision_count": len(selected_rows),
        "candidate_scoped_p_up_action_comparable_count": comparable_count,
        "candidate_scoped_p_up_action_missing_count": missing,
        "candidate_scoped_p_up_action_disagreement_count": disagreement,
        "candidate_scoped_p_up_action_disagreement_rate": disagreement_rate,
        "candidate_scoped_p_up_action_disagreement_within_limit": within_limit,
        "diagnostic_only": False,
        "max_allowed_disagreement_rate": O_MAX_P_UP_ACTION_DISAGREEMENT_RATE,
    }


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
    selected_returns_by_family: dict[str, list[float]] = defaultdict(list)
    selected_returns_by_side: dict[str, list[float]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    action_pair_regret_cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    hts_p_up_reliability_cases = []
    high_score_returns = []
    split_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    no_trade_selection_count = 0
    no_trade_missed_opportunities = []
    largest_regret_case: dict[str, Any] | None = None
    completeness_summary = _decision_group_completeness_summary(rows)
    source_score_summary = _source_score_completeness_summary(rows)
    source_scores_complete_for_variant = (
        (
            variant == O_MODEL_PREDICTED_VARIANT
            and _deployable_model_score_available(rows, variant)
        )
        or (
            variant == "current_source_baseline"
            and source_score_summary["source_score_complete"]
        )
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
        selected_components = selected.get("o_model_score_components") or {}
        selected_p_up = _bounded(float(selected.get("p_up") or 0.5), 0.0, 1.0)
        selected_p_down = 1.0 - selected_p_up
        selected_action = str(selected.get("action") or "")
        oracle_action = str(oracle.get("action") or "")
        action_pair_case = {
            "decision_group_id": selected["decision_group_id"],
            "market_id": selected.get("market_id"),
            "decision_ts": selected.get("decision_ts"),
            "selected_action": selected_action,
            "oracle_action": oracle_action,
            "selected_action_family": selected.get("action_family"),
            "oracle_action_family": oracle.get("action_family"),
            "selected_side": selected.get("selected_side"),
            "oracle_side": oracle.get("selected_side"),
            "selected_return": selected_return,
            "oracle_return": oracle_return,
            "regret": regret,
            "p_up": selected_p_up,
            "p_down": selected_p_down,
            "raw_score_component": float(
                selected_components.get("group_normalized_raw_model_component")
                or 0.0
            ),
            "p_up_alignment_component": float(
                selected_components.get("p_up_side_alignment_component") or 0.0
            ),
            "large_regret_reversal_guard_component": float(
                selected_components.get("large_regret_reversal_guard_component")
                or 0.0
            ),
            "hts_p_up_reliability_guard_component": float(
                selected_components.get("hts_p_up_reliability_guard_component")
                or 0.0
            ),
        }
        action_pair_regret_cases[(selected_action, oracle_action)].append(
            action_pair_case
        )
        if selected.get("action_family") == "HOLD_TO_SETTLEMENT":
            hts_p_up_reliability_cases.append(
                {
                    **action_pair_case,
                    **(selected.get("hts_p_up_reliability_buckets") or {}),
                    "selected_side_matches_oracle_side": (
                        selected.get("selected_side") == oracle.get("selected_side")
                    ),
                }
            )
        regrets.append(regret)
        if largest_regret_case is None or regret > float(largest_regret_case["regret"]):
            largest_regret_case = {
                "decision_group_id": selected["decision_group_id"],
                "market_id": selected.get("market_id"),
                "decision_ts": selected.get("decision_ts"),
                "selected_action": selected_action,
                "selected_action_family": selected.get("action_family"),
                "selected_side": selected.get("selected_side"),
                "selected_return": selected_return,
                "selected_score": float(selected["variant_scores"][variant]),
                "oracle_action": oracle_action,
                "oracle_action_family": oracle.get("action_family"),
                "oracle_side": oracle.get("selected_side"),
                "oracle_return": oracle_return,
                "regret": regret,
                "p_up": selected_p_up,
                "p_down": selected_p_down,
                "raw_score_component": action_pair_case["raw_score_component"],
                "p_up_alignment_component": action_pair_case[
                    "p_up_alignment_component"
                ],
                "large_regret_reversal_guard_component": action_pair_case[
                    "large_regret_reversal_guard_component"
                ],
                "hts_p_up_reliability_guard_component": action_pair_case[
                    "hts_p_up_reliability_guard_component"
                ],
                "action_pair_key": f"{selected_action}->{oracle_action}",
            }
        selected_returns.append(selected_return)
        oracle_returns.append(oracle_return)
        confusion[(selected["action_family"], oracle["action_family"])] += 1
        family_regret[selected["action_family"]].append(regret)
        selected_returns_by_family[selected["action_family"]].append(selected_return)
        selected_side = str(selected.get("selected_side") or "NONE")
        selected_returns_by_side[selected_side].append(selected_return)
        side_regret[selected_side].append(regret)
        if selected["action_family"] == "NO_TRADE":
            no_trade_selection_count += 1
            no_trade_missed_opportunities.append(max(0.0, oracle_return))
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
        "ranking_score_source": _ranking_score_source(variant),
        "deployable_model_score_available": _deployable_model_score_available(
            rows,
            variant,
        ),
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
        "no_trade_selection_rate": no_trade_selection_count / group_count
        if group_count
        else 0.0,
        "action_family_selected_return_breakdown": _selected_return_breakdown(
            selected_returns_by_family
        ),
        "side_selected_return_breakdown": _selected_return_breakdown(
            selected_returns_by_side
        ),
        "largest_winner_dependency": _largest_winner_dependency(selected_returns),
        "largest_regret_case": largest_regret_case or {},
        "no_trade_opportunity_cost_mean": statistics.mean(
            max(0.0, item) for item in oracle_returns
        )
        if oracle_returns
        else 0.0,
        "no_trade_missed_opportunity": {
            "selected_no_trade_count": no_trade_selection_count,
            "missed_positive_opportunity_count": sum(
                1 for value in no_trade_missed_opportunities if value > 0.0
            ),
            "missed_positive_opportunity_sum": sum(no_trade_missed_opportunities),
            "missed_positive_opportunity_mean": statistics.mean(
                no_trade_missed_opportunities
            )
            if no_trade_missed_opportunities
            else 0.0,
            "max_missed_positive_opportunity": max(
                no_trade_missed_opportunities,
                default=0.0,
            ),
        },
        "action_family_level_regret": {
            family: statistics.mean(values) for family, values in family_regret.items()
        },
        "action_family_gate_metrics": _action_family_gate_metrics(
            selected_returns_by_family
        ),
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
        "action_pair_regret_summary": _action_pair_regret_summary(
            action_pair_regret_cases
        ),
        "hold_to_settlement_up_down_reversal_regret": (
            _hold_to_settlement_reversal_regret_summary(action_pair_regret_cases)
        ),
        "hts_p_up_reliability_regret_summary": (
            _hts_p_up_reliability_regret_summary(hts_p_up_reliability_cases)
        ),
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


def _split_metric_views(
    rows: list[dict[str, Any]],
    variant: str,
    high_score_threshold: float,
) -> dict[str, dict[str, Any]]:
    return {
        "train_shadow": _eligibility_metric_view(
            _ranking_metrics(
                [row for row in rows if row.get("split") == "shadow"],
                variant,
                high_score_threshold,
            )
        ),
        "validation": _eligibility_metric_view(
            _ranking_metrics(
                [row for row in rows if row.get("split") == "validation"],
                variant,
                high_score_threshold,
            )
        ),
        "all": _eligibility_metric_view(
            _ranking_metrics(rows, variant, high_score_threshold)
        ),
    }


def _eligibility_metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_group_count": metrics["decision_group_count"],
        "top1_realized_best_action_hit_rate": metrics[
            "top1_realized_best_action_hit_rate"
        ],
        "top2_realized_best_action_hit_rate": metrics[
            "top2_realized_best_action_hit_rate"
        ],
        "top3_realized_best_action_hit_rate": metrics[
            "top3_realized_best_action_hit_rate"
        ],
        "selected_action_realized_replay_return_sum": metrics[
            "selected_action_realized_replay_return_sum"
        ],
        "oracle_executable_best_action_return_sum": metrics[
            "oracle_executable_best_action_return_sum"
        ],
        "mean_regret": metrics["mean_regret"],
        "high_score_support_count": metrics["high_score_support_count"],
        "high_score_realized_return_mean": metrics[
            "high_score_realized_return_mean"
        ],
        "high_score_realized_return_sum": metrics["high_score_realized_return_sum"],
        "NO_TRADE_selection_rate": metrics["no_trade_selection_rate"],
        "action_family_selected_return_breakdown": metrics[
            "action_family_selected_return_breakdown"
        ],
        "side_selected_return_breakdown": metrics["side_selected_return_breakdown"],
        "largest_winner_dependency": metrics["largest_winner_dependency"],
        "largest_regret_case": metrics["largest_regret_case"],
        "action_family_level_regret": metrics["action_family_level_regret"],
        "side_level_regret": metrics["side_level_regret"],
        "no_trade_missed_opportunity": metrics["no_trade_missed_opportunity"],
        "no_trade_opportunity_cost_mean": metrics["no_trade_opportunity_cost_mean"],
        "ranking_confusion_matrix": metrics["ranking_confusion_matrix"],
        "action_pair_regret_summary": metrics["action_pair_regret_summary"],
        "hold_to_settlement_up_down_reversal_regret": metrics[
            "hold_to_settlement_up_down_reversal_regret"
        ],
        "hts_p_up_reliability_regret_summary": metrics[
            "hts_p_up_reliability_regret_summary"
        ],
    }


def _action_pair_regret_summary(
    cases_by_pair: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for (selected_action, oracle_action), cases in cases_by_pair.items():
        regrets = [float(case["regret"]) for case in cases]
        p_up_values = [float(case["p_up"]) for case in cases]
        raw_components = [float(case["raw_score_component"]) for case in cases]
        alignment_components = [
            float(case["p_up_alignment_component"]) for case in cases
        ]
        largest_case = max(cases, key=lambda case: float(case["regret"]))
        rows.append(
            {
                "selected_action": selected_action,
                "oracle_action": oracle_action,
                "selected_action_family": largest_case[
                    "selected_action_family"
                ],
                "oracle_action_family": largest_case["oracle_action_family"],
                "count": len(cases),
                "regret_mean": statistics.mean(regrets) if regrets else 0.0,
                "regret_sum": sum(regrets),
                "regret_max": max(regrets, default=0.0),
                "p_up_mean": statistics.mean(p_up_values) if p_up_values else 0.5,
                "p_down_mean": 1.0
                - (statistics.mean(p_up_values) if p_up_values else 0.5),
                "raw_score_component_mean": statistics.mean(raw_components)
                if raw_components
                else 0.0,
                "p_up_alignment_component_mean": statistics.mean(
                    alignment_components
                )
                if alignment_components
                else 0.0,
                "largest_case": largest_case,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["regret_max"]),
            -float(row["regret_sum"]),
            str(row["selected_action"]),
            str(row["oracle_action"]),
        ),
    )


def _hold_to_settlement_reversal_regret_summary(
    cases_by_pair: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    reversal_pairs = {
        ("BUY_UP_HOLD_TO_SETTLEMENT", "BUY_DOWN_HOLD_TO_SETTLEMENT"),
        ("BUY_DOWN_HOLD_TO_SETTLEMENT", "BUY_UP_HOLD_TO_SETTLEMENT"),
    }
    cases = [
        case
        for pair in reversal_pairs
        for case in cases_by_pair.get(pair, [])
    ]
    regrets = [float(case["regret"]) for case in cases]
    positive_regrets = [value for value in regrets if value > 0.0]
    return {
        "reversal_case_count": len(cases),
        "positive_reversal_regret_count": len(positive_regrets),
        "regret_mean": statistics.mean(regrets) if regrets else 0.0,
        "positive_regret_mean": statistics.mean(positive_regrets)
        if positive_regrets
        else 0.0,
        "regret_sum": sum(regrets),
        "regret_max": max(regrets, default=0.0),
        "largest_reversal_case": max(
            cases,
            key=lambda case: float(case["regret"]),
            default={},
        ),
    }


def _hts_p_up_reliability_regret_summary(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cases:
        return {
            "selected_hts_case_count": 0,
            "p_up_confidently_wrong_count": 0,
            "p_up_confidently_wrong_rate": 0.0,
            "by_p_up_confidence_bucket": {},
            "by_time_to_close_bucket": {},
            "by_spread_bucket": {},
            "by_queue_bucket": {},
            "by_staleness_bucket": {},
            "by_selected_vs_oracle_side": {},
            "top_confidently_wrong_cases": [],
        }
    wrong_cases = [
        case
        for case in cases
        if str(case.get("selected_side")) != str(case.get("oracle_side"))
    ]
    return {
        "selected_hts_case_count": len(cases),
        "p_up_confidently_wrong_count": len(wrong_cases),
        "p_up_confidently_wrong_rate": len(wrong_cases) / len(cases),
        "by_p_up_confidence_bucket": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: case["p_up_confidence_bucket"],
            regret_threshold=0.0,
            min_support=1,
        ),
        "by_time_to_close_bucket": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: case["time_to_close_bucket"],
            regret_threshold=0.0,
            min_support=1,
        ),
        "by_spread_bucket": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: case["spread_bucket"],
            regret_threshold=0.0,
            min_support=1,
        ),
        "by_queue_bucket": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: case["queue_bucket"],
            regret_threshold=0.0,
            min_support=1,
        ),
        "by_staleness_bucket": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: case["staleness_bucket"],
            regret_threshold=0.0,
            min_support=1,
        ),
        "by_selected_vs_oracle_side": _hts_reliability_case_summary(
            _metric_hts_cases(cases),
            key_fn=lambda case: f"{case['selected_side']}->{case['oracle_side']}",
            regret_threshold=0.0,
            min_support=1,
        ),
        "top_confidently_wrong_cases": sorted(
            wrong_cases,
            key=lambda case: (-float(case["regret"]), str(case["decision_group_id"])),
        )[:10],
    }


def _metric_hts_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **case,
            "p_up_confidently_wrong": str(case.get("selected_side"))
            != str(case.get("oracle_side")),
        }
        for case in cases
    ]


def _selected_return_breakdown(
    returns_by_key: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "support_count": len(values),
            "selected_return_sum": sum(values),
            "selected_return_mean": statistics.mean(values) if values else 0.0,
        }
        for key, values in sorted(returns_by_key.items())
    }


def _largest_winner_dependency(returns: list[float]) -> dict[str, Any]:
    total = sum(returns)
    positive_returns = [value for value in returns if value > 0.0]
    largest_winner = max(positive_returns) if positive_returns else 0.0
    without_largest = total - largest_winner
    return {
        "largest_winner_return": largest_winner,
        "selected_return_sum_without_largest_winner": without_largest,
        "total_return_positive_only_because_of_largest_winner": (
            total > 0.0 and without_largest <= 0.0
        ),
        "largest_winner_share_of_positive_return": largest_winner
        / sum(positive_returns)
        if positive_returns
        else 0.0,
    }


def _ranking_score_source(variant: str) -> str:
    if variant == O_MODEL_PREDICTED_VARIANT:
        return "model_predicted_score"
    if variant == "current_source_baseline":
        return "observed_source_score"
    return "label_diagnostic_score"


def _model_high_score_threshold(model_training_summary: dict[str, Any]) -> float:
    return float(
        model_training_summary["ranking_correction_config"][
            "high_score_calibration"
        ]["high_score_threshold"]
    )


def _deployable_model_score_available(rows: list[dict[str, Any]], variant: str) -> bool:
    return variant == O_MODEL_PREDICTED_VARIANT and all(
        bool(row.get("deployable_model_score_available")) for row in rows
    )


def _action_family_gate_metrics(
    returns_by_family: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    families = ("SELL_BEFORE_CLOSE", "HOLD_TO_SETTLEMENT", "NO_TRADE")
    return {
        family: {
            "support_count": len(values),
            "realized_return_mean": statistics.mean(values) if values else 0.0,
            "realized_return_sum": sum(values),
            "paper_decision_eligible": False,
            "reason_codes": [
                "diagnostic_only_no_paper_live_unlock",
                "future_unseen_o_holdout_required",
            ],
        }
        for family in families
        for values in (returns_by_family.get(family, []),)
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
        "o_raw_ridge_model_score": row.get("o_raw_ridge_model_score"),
        "o_group_normalized_raw_model_score": row.get(
            "o_group_normalized_raw_model_score"
        ),
        "o_model_predicted_score": row.get("o_model_predicted_score"),
        "o_model_score_components": row.get("o_model_score_components"),
        "hts_p_up_reliability_buckets": row.get("hts_p_up_reliability_buckets"),
        "deployable_model_score_available": row.get("deployable_model_score_available"),
        "ranking_score_source": _ranking_score_source(variant),
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
        "| candidate | score_source | scope | top1 | mean_regret | eligible | excluded_reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            "| {name} | {source} | {scope} | {top1:.4f} | {regret:.6f} | {eligible} | {reason} |".format(
                name=row["candidate_name"],
                source=row["ranking_score_source"],
                scope=row["ranking_metric_scope"],
                top1=float(row["top1_realized_best_action_hit_rate"]),
                regret=float(row["mean_regret"]),
                eligible=str(row["source_model_candidate_eligible"]).lower(),
                reason=row["excluded_from_eligibility_reason"] or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _eligibility_gate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O Source Model Eligibility Gate",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            f"- ranking_score_source: `{report['ranking_score_source']}`",
            "- deployable_model_score_available: "
            f"`{str(report['deployable_model_score_available']).lower()}`",
            "- validation_metrics_only_for_eligibility: "
            f"`{str(report['validation_metrics_only_for_eligibility']).lower()}`",
            "- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            "- calibration_support_passed: "
            f"`{str(report['calibration_support_passed']).lower()}`",
            "- calibration_quality_passed: "
            f"`{str(report['calibration_quality_passed']).lower()}`",
            "- action_family_paper_decision_eligible: "
            f"`{str(report['action_family_paper_decision_eligible']).lower()}`",
            "- best_action_concentration_passed: "
            f"`{str(report['best_action_concentration_passed']).lower()}`",
            "- p_up_action_disagreement_within_limit: "
            f"`{str(report['p_up_action_disagreement_within_limit']).lower()}`",
            "- action_value_paper_decision_eligible: "
            f"`{str(report['action_value_paper_decision_eligible']).lower()}`",
            f"- high_score_support_count: `{report['high_score_support_count']}`",
            "- high_score_realized_return_mean: "
            f"`{report['high_score_realized_return_mean']}`",
            f"- mean_regret: `{report['mean_regret']}`",
            "- top1_realized_best_action_hit_rate: "
            f"`{report['top1_realized_best_action_hit_rate']}`",
            "- NO_TRADE_selection_rate: "
            f"`{report['NO_TRADE_selection_rate']}`",
            f"- ineligible_reason_codes: `{report['ineligible_reason_codes']}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _freeze_readiness_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O Freeze Readiness",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            f"- ranking_score_source: `{report['ranking_score_source']}`",
            f"- freeze_ready: `{str(report['freeze_ready']).lower()}`",
            "- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            f"- model_sha256: `{report['model_sha256']}`",
            f"- model_manifest_sha256: `{report['model_manifest_sha256']}`",
            f"- training_data_hash: `{report['training_data_hash']}`",
            f"- label_grid_hash: `{report['label_grid_hash']}`",
            f"- feature_schema_hash: `{report['feature_schema_hash']}`",
            f"- split_hash: `{report['split_hash']}`",
            f"- candidate_config_hash: `{report['candidate_config_hash']}`",
            "- freeze_blocking_reason_codes: "
            f"`{report['freeze_blocking_reason_codes']}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )
