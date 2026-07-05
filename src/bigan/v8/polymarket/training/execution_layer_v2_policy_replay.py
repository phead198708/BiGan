"""Settlement-CSV policy replay diagnostics for v8 Execution Layer v2.

The replay in this module is outcome-aware by construction because it reads a
settlement PnL CSV.  It is therefore diagnostic-only and never promotion,
paper/live, or execution-handoff evidence.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields

EXECUTION_LAYER_V2_POLICY_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-v1"
)
EXECUTION_LAYER_V2_POLICY_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-manifest-v1"
)
EXECUTION_LAYER_V2_RECOMMENDED_EXECUTION_POLICY_NAME = (
    "bucket_aware_execution_policy_v1_diagnostic"
)

POLICY_REPLAY_VARIANTS: tuple[str, ...] = (
    "all_executed_baseline",
    "price_070_090_only",
    "exclude_buy_up_hts",
    "sell_before_close_only",
    "buy_down_hts_only",
    "five_min_only",
    "fifteen_min_only",
    "bucket_aware_v1_conservative",
    "bucket_aware_v1_plus_sbc",
)

PRICE_BUCKET_EDGES: tuple[tuple[str, float, float | None], ...] = (
    ("lt_0_60", -math.inf, 0.60),
    ("0_60_0_70", 0.60, 0.70),
    ("0_70_0_90", 0.70, 0.90),
    ("gt_0_90", 0.90, None),
)
MISSING_SORT_NUMBER = 10**30


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PolicyReplayConfig:
    """Configuration for a settlement-CSV policy replay bundle."""

    run_id: str
    input_csv: Path | str
    output_dir: Path | str
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False
    v8_execution_handoff_allowed: bool = False
    source_model_candidate_eligible: bool = False
    freeze_ready: bool = False
    promotion_evidence_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "input_csv", Path(self.input_csv))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_csv"] = str(self.input_csv)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2PolicyReplayResult:
    """Written settlement-CSV replay bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    report: dict[str, Any]
    manifest: dict[str, Any]


def run_execution_layer_v2_policy_replay_from_settlement_csv(
    config: ExecutionLayerV2PolicyReplayConfig,
) -> ExecutionLayerV2PolicyReplayResult:
    """Run and write diagnostic policy replay artifacts."""

    if not config.input_csv.exists():
        raise FileNotFoundError(f"settlement CSV not found: {config.input_csv}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"policy replay output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    raw_rows = _load_csv_rows(config.input_csv)
    normalized_rows = [_normalize_settlement_row(row, index) for index, row in enumerate(raw_rows)]
    report = build_execution_layer_v2_policy_replay_report(
        normalized_rows,
        run_id=config.run_id,
        input_csv=str(config.input_csv),
    )

    artifact_paths = {
        "execution_layer_v2_policy_replay_report": run_dir
        / "execution_layer_v2_policy_replay_report.json",
        "execution_layer_v2_policy_replay_summary": run_dir
        / "execution_layer_v2_policy_replay_report.md",
        "execution_layer_v2_policy_replay_manifest": run_dir
        / "execution_layer_v2_policy_replay_manifest.json",
    }
    _write_json(artifact_paths["execution_layer_v2_policy_replay_report"], report)
    _write_text(
        artifact_paths["execution_layer_v2_policy_replay_summary"],
        execution_layer_v2_policy_replay_report_to_markdown(report),
    )
    artifact_hashes = {
        "execution_layer_v2_policy_replay_report": _sha256_file(
            artifact_paths["execution_layer_v2_policy_replay_report"]
        ),
        "execution_layer_v2_policy_replay_summary": _sha256_file(
            artifact_paths["execution_layer_v2_policy_replay_summary"]
        ),
    }
    manifest = {
        "schema_version": EXECUTION_LAYER_V2_POLICY_REPLAY_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_csv": str(config.input_csv),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "report_id": report["execution_layer_v2_policy_replay_report_id"],
        "row_count": report["row_count"],
        "policy_variant_names": list(POLICY_REPLAY_VARIANTS),
        "max_drawdown_ordering": report["max_drawdown_ordering"],
        "chronological_sort_fields": list(report["chronological_sort_fields"]),
        "recommended_execution_policy": report["recommended_execution_policy_v1"][
            "policy_name"
        ],
        "ev_mapping_status": report["signal_to_ev_diagnostic"]["ev_mapping_status"],
        "diagnostic_only": True,
        "uses_settlement_pnl_csv_for_evaluation": True,
        "uses_settlement_pnl_csv_for_tuning": False,
        "source_scores_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["execution_layer_v2_policy_replay_manifest"], manifest)
    artifact_hashes["execution_layer_v2_policy_replay_manifest"] = _sha256_file(
        artifact_paths["execution_layer_v2_policy_replay_manifest"]
    )
    return ExecutionLayerV2PolicyReplayResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        report=report,
        manifest=manifest,
    )


def build_execution_layer_v2_policy_replay_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_csv: str,
) -> dict[str, Any]:
    """Build a diagnostic replay report from normalized settlement rows."""

    variant_reports = {
        name: _policy_variant_metrics(rows, name) for name in POLICY_REPLAY_VARIANTS
    }
    report = {
        "schema_version": EXECUTION_LAYER_V2_POLICY_REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "input_csv": input_csv,
        "row_count": len(rows),
        "diagnostic_only": True,
        "uses_settlement_pnl_csv_for_evaluation": True,
        "uses_settlement_pnl_csv_for_tuning": False,
        "thresholds_tuned": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "source_ranking_score_mutated": False,
        "paper_live_unlock_changed": False,
        "max_drawdown_ordering": "chronological",
        "chronological_sort_fields": [
            "numeric_iteration",
            "decision_ts_numeric",
            "intent_id",
            "row_index",
        ],
        "policy_variant_names": list(POLICY_REPLAY_VARIANTS),
        "policy_variant_definitions": _policy_variant_definitions(),
        "policy_variants": variant_reports,
        "price_bucket_summary": _price_bucket_summary(rows),
        "action_family_summary": _family_summary(rows),
        "signal_to_ev_diagnostic": _signal_to_ev_diagnostic(rows),
        "recommended_execution_policy_v1": _recommended_execution_policy(rows, variant_reports),
        "small_sample_warnings": _small_sample_warnings(variant_reports),
        **_safety_report_fields(),
    }
    report["execution_layer_v2_policy_replay_report_id"] = canonical_json_sha256(report)
    return report


def execution_layer_v2_policy_replay_report_to_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for #166 review."""

    ev = report["signal_to_ev_diagnostic"]
    policy = report["recommended_execution_policy_v1"]
    lines = [
        "# v8 Execution Layer v2 Policy Replay",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- input_csv: `{report['input_csv']}`",
        f"- row_count: `{report['row_count']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- ev_mapping_status: `{ev['ev_mapping_status']}`",
        f"- recommended_ev_source: `{ev['recommended_ev_source']}`",
        f"- recommended_policy: `{policy['policy_name']}`",
        f"- max_drawdown_ordering: `{report['max_drawdown_ordering']}`",
        f"- small_sample_warnings: `{report['small_sample_warnings']}`",
        f"- sell_before_close_positive_in_csv: `{policy['sell_before_close_positive_in_csv']}`",
        f"- paper_only: `{report['paper_only']}`",
        f"- capital_at_risk: `{report['capital_at_risk']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Variants",
        "",
        "| variant | rows | cost_basis | settlement_pnl | roi | win_rate | max_drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in POLICY_REPLAY_VARIANTS:
        metrics = report["policy_variants"][name]
        lines.append(
            f"| `{name}` | {metrics['row_count']} | "
            f"{metrics['cost_basis']:.6f} | {metrics['settlement_pnl']:.6f} | "
            f"{metrics['roi']:.6f} | {metrics['win_rate']:.6f} | "
            f"{metrics['max_drawdown']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## EV Mapping Diagnostic",
            "",
            f"- p_market_implied_source_fields: `{ev['p_market_implied_source_fields']}`",
            f"- p_model_fair_value_source_fields_present: `{ev['p_model_fair_value_source_fields_present']}`",
            f"- ev_mapping_blocking_reason_codes: `{ev['ev_mapping_blocking_reason_codes']}`",
            "",
            "## Recommended Policy Rules",
            "",
        ]
    )
    for rule in policy["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


def _policy_variant_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    for row in rows:
        allowed, reasons = _variant_allows_row(row, variant)
        if allowed:
            selected.append(row)
        else:
            rejected_reasons.update(reasons)
    pnl_values = [float(row["settlement_pnl"]) for row in selected]
    chronological_rows = sorted(
        selected,
        key=lambda row: tuple(row["chronological_sort_key"]),
    )
    chronological_pnl_values = [float(row["settlement_pnl"]) for row in chronological_rows]
    cost_basis = sum(float(row["cost_basis"]) for row in selected)
    settlement_pnl = sum(pnl_values)
    roi = settlement_pnl / cost_basis if cost_basis else 0.0
    win_rate = (
        sum(1 for value in pnl_values if value > 0.0) / len(pnl_values)
        if pnl_values
        else 0.0
    )
    return {
        "variant_name": variant,
        "row_count": len(selected),
        "cost_basis": cost_basis,
        "settlement_pnl": settlement_pnl,
        "roi": roi,
        "win_rate": win_rate,
        "action_distribution": _count_distribution(selected, "action"),
        "family_distribution": _count_distribution(selected, "family"),
        "horizon_distribution": _count_distribution(selected, "horizon"),
        "price_bucket_distribution": _count_distribution(selected, "price_bucket"),
        "max_drawdown": _max_drawdown(chronological_pnl_values),
        "max_drawdown_ordering": "chronological",
        "chronological_sort_fields": [
            "numeric_iteration",
            "decision_ts_numeric",
            "intent_id",
            "row_index",
        ],
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "diagnostic_only": True,
    }


def _policy_variant_definitions() -> dict[str, str]:
    return {
        "all_executed_baseline": "all rows from the settlement CSV",
        "price_070_090_only": "only rows with entry price in [0.70, 0.90]",
        "exclude_buy_up_hts": "all rows except BUY_UP_HOLD_TO_SETTLEMENT",
        "sell_before_close_only": "only SELL_BEFORE_CLOSE family rows",
        "buy_down_hts_only": "only BUY_DOWN_HOLD_TO_SETTLEMENT rows",
        "five_min_only": "only 5m horizon rows",
        "fifteen_min_only": "only 15m horizon rows",
        "bucket_aware_v1_conservative": (
            "price 0.70-0.90, exclude BUY_UP_HOLD_TO_SETTLEMENT, "
            "allow BUY_DOWN_HOLD_TO_SETTLEMENT, allow SELL_BEFORE_CLOSE only "
            "if it also passes the 0.70-0.90 price bucket"
        ),
        "bucket_aware_v1_plus_sbc": (
            "allow BUY_DOWN_HOLD_TO_SETTLEMENT only in price 0.70-0.90, "
            "allow SELL_BEFORE_CLOSE regardless of price bucket, exclude "
            "BUY_UP_HOLD_TO_SETTLEMENT"
        ),
    }


def _variant_allows_row(row: dict[str, Any], variant: str) -> tuple[bool, list[str]]:
    action = str(row["action"])
    family = str(row["family"])
    horizon = str(row["horizon"])
    price = row["entry_price"]
    if variant == "all_executed_baseline":
        return True, []
    if variant == "price_070_090_only":
        if _price_in_070_090(price):
            return True, []
        return False, [_price_rejection_reason(price)]
    if variant == "exclude_buy_up_hts":
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            return False, ["excluded_buy_up_hold_to_settlement"]
        return True, []
    if variant == "sell_before_close_only":
        if family == "SELL_BEFORE_CLOSE":
            return True, []
        return False, ["not_sell_before_close"]
    if variant == "buy_down_hts_only":
        if action == "BUY_DOWN_HOLD_TO_SETTLEMENT":
            return True, []
        return False, ["not_buy_down_hold_to_settlement"]
    if variant == "five_min_only":
        if horizon == "5m":
            return True, []
        return False, ["not_5m_horizon"]
    if variant == "fifteen_min_only":
        if horizon == "15m":
            return True, []
        return False, ["not_15m_horizon"]
    if variant == "bucket_aware_v1_conservative":
        reasons = []
        if not _price_in_070_090(price):
            reasons.append("bucket_aware_conservative_price_not_070_090")
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_conservative_excluded_buy_up_hts")
        if family != "SELL_BEFORE_CLOSE" and action != "BUY_DOWN_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_conservative_action_not_candidate")
        return not reasons, reasons
    if variant == "bucket_aware_v1_plus_sbc":
        reasons = []
        if action == "BUY_UP_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_plus_sbc_excluded_buy_up_hts")
        if family == "SELL_BEFORE_CLOSE":
            return not reasons, reasons
        if action != "BUY_DOWN_HOLD_TO_SETTLEMENT":
            reasons.append("bucket_aware_plus_sbc_action_not_candidate")
        if not _price_in_070_090(price):
            reasons.append("bucket_aware_plus_sbc_price_not_070_090_for_hts")
        return not reasons, reasons
    raise ValueError(f"unsupported policy replay variant: {variant}")


def _signal_to_ev_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_fields = sorted(
        {
            field
            for row in rows
            for field in row["raw_fields"]
            if field.lower()
            in {
                "p_model_fair_value",
                "model_p_up",
                "calibrated_p_up",
                "model_probability",
                "predicted_probability",
                "fair_value_probability",
                "action_expected_net_return",
                "calibrated_action_score",
            }
        }
    )
    implied_fields = sorted(
        {
            row["entry_price_source_field"]
            for row in rows
            if row.get("entry_price_source_field")
        }
    )
    reason_codes = [
        "p_up_probability_provenance_not_confirmed_calibrated_model_fair_value",
        "market_implied_probability_collapses_ev_to_spread_minus_cost",
        "settlement_csv_is_outcome_evaluation_not_decision_time_model_training_input",
    ]
    return {
        "current_ev_formula": "entry_ev = p_side - ask - cost",
        "p_market_implied_source_fields": implied_fields,
        "p_model_fair_value_source_fields_present": bool(model_fields),
        "p_model_fair_value_candidate_fields": model_fields,
        "current_p_up_should_not_be_used_as_ev_fair_value_without_provenance": True,
        "ev_mapping_status": "blocked_requires_calibrated_model_fair_value",
        "ev_mapping_blocking_reason_codes": reason_codes,
        "recommended_ev_source": (
            "calibrated_model_fair_value_probability_or_action_expected_net_return"
        ),
        "code_bug_indicated": False,
        "design_issue_indicated": True,
        "diagnosis": (
            "The no-action behavior is expected when p_side is market-implied "
            "rather than calibrated fair value; EV becomes approximately "
            "negative spread plus cost."
        ),
    }


def _recommended_execution_policy(
    rows: list[dict[str, Any]],
    variant_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sbc_metrics = variant_reports["sell_before_close_only"]
    sbc_count = sbc_metrics["row_count"]
    sbc_positive = float(sbc_metrics["settlement_pnl"]) > 0.0
    return {
        "policy_name": EXECUTION_LAYER_V2_RECOMMENDED_EXECUTION_POLICY_NAME,
        "derived_from_settlement_csv_diagnostics_only": True,
        "uses_validation_labels_for_threshold_tuning": False,
        "do_not_relax_execution_guard_thresholds": True,
        "candidate_variant_name": "bucket_aware_v1_plus_sbc",
        "candidate_variant_metrics": variant_reports["bucket_aware_v1_plus_sbc"],
        "comparison_variant_metrics": {
            "bucket_aware_v1_conservative": variant_reports[
                "bucket_aware_v1_conservative"
            ],
            "bucket_aware_v1_plus_sbc": variant_reports["bucket_aware_v1_plus_sbc"],
        },
        "small_sample_warnings": (
            ["sell_before_close_small_sample"] if 0 < sbc_count < 30 else []
        ),
        "sell_before_close_summary": sbc_metrics,
        "sell_before_close_positive_in_csv": sbc_positive,
        "sell_before_close_diagnostic_interpretation": (
            "SELL_BEFORE_CLOSE is positive in this CSV but remains small-sample."
            if sbc_positive and 0 < sbc_count < 30
            else "SELL_BEFORE_CLOSE support is not sufficient for promotion evidence."
        ),
        "rules": [
            "Do not use market-implied p_up as EV fair value without calibrated provenance.",
            "Avoid BUY_UP_HOLD_TO_SETTLEMENT unless strong calibrated edge exists.",
            "Prefer entry price bucket 0.70-0.90 for BUY_DOWN_HOLD_TO_SETTLEMENT when calibrated edge exists.",
            "Keep SELL_BEFORE_CLOSE as a candidate even below 0.70, but mark small-sample until support grows.",
            "Keep BUY_DOWN_HOLD_TO_SETTLEMENT as a candidate.",
            "Avoid price >0.90 unless calibrated edge is strong.",
            "Avoid price 0.60-0.70 by default.",
            "Do not relax execution guard thresholds.",
        ],
        "expected_follow_up": (
            "Use calibrated O action expected return or calibrated fair-value "
            "probability for EV mapping, then replay against future holdout."
        ),
        "row_count": len(rows),
    }


def _small_sample_warnings(
    variant_reports: dict[str, dict[str, Any]],
) -> list[str]:
    sbc_count = variant_reports["sell_before_close_only"]["row_count"]
    if 0 < sbc_count < 30:
        return ["sell_before_close_small_sample"]
    return []


def _normalize_settlement_row(row: dict[str, str], index: int) -> dict[str, Any]:
    action = _first_text(
        row,
        (
            "action",
            "selected_action",
            "policy_action",
            "entry_policy_action",
            "signal_action",
        ),
        default="UNKNOWN",
    )
    action = _canonical_action(action)
    entry_price, entry_price_field = _first_float_with_field(
        row,
        (
            "entry_price",
            "execution_price",
            "fill_price",
            "avg_price",
            "price",
            "entry_ask",
            "ask",
        ),
    )
    cost_basis = _first_float(
        row,
        (
            "cost_basis",
            "cost_basis_usdc",
            "entry_cost",
            "paper_notional",
            "notional",
            "size_usdc",
            "fill_notional",
            "entry_notional",
        ),
        default=0.0,
    )
    if cost_basis <= 0.0:
        shares = _first_float(row, ("shares", "quantity", "size"), default=0.0)
        if shares > 0.0 and entry_price is not None:
            cost_basis = shares * entry_price
    settlement_pnl = _first_float(
        row,
        (
            "settlement_pnl",
            "settlement_pnl_usdc",
            "total_polymarket_pnl",
            "pnl",
            "realized_pnl",
            "net_pnl",
        ),
        default=0.0,
    )
    horizon = _infer_horizon(row)
    family = _infer_family(row, action)
    decision_ts_raw = _first_text(row, ("decision_ts", "ts", "timestamp"), default=str(index))
    decision_ts_numeric = _parse_sort_number(decision_ts_raw)
    numeric_iteration = _first_sort_number(
        row,
        ("iteration", "round_iteration", "loop_iteration", "cycle_index"),
        default=MISSING_SORT_NUMBER,
    )
    intent_id = _first_text(
        row,
        ("intent_id", "paper_intent_id", "order_intent_id", "signal_id"),
        default="",
    )
    return {
        "row_index": index,
        "market_id": _first_text(row, ("market_id", "condition_id", "slug"), default=""),
        "decision_ts": decision_ts_raw,
        "decision_ts_numeric": decision_ts_numeric,
        "numeric_iteration": numeric_iteration,
        "intent_id": intent_id,
        "chronological_sort_key": [
            numeric_iteration,
            decision_ts_numeric,
            intent_id,
            index,
        ],
        "action": action,
        "family": family,
        "horizon": horizon,
        "entry_price": entry_price,
        "price_bucket": _price_bucket(entry_price),
        "cost_basis": cost_basis,
        "settlement_pnl": settlement_pnl,
        "entry_price_source_field": entry_price_field,
        "raw_fields": sorted(row.keys()),
    }


def _infer_family(row: dict[str, str], action: str) -> str:
    family = _first_text(
        row,
        ("family", "action_family", "selected_action_family", "exit_policy"),
        default="",
    ).upper()
    if "SELL_BEFORE_CLOSE" in family or "SELL_BEFORE_CLOSE" in action:
        return "SELL_BEFORE_CLOSE"
    if "HOLD_TO_SETTLEMENT" in family or "HOLD_TO_SETTLEMENT" in action:
        return "HOLD_TO_SETTLEMENT"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return family or "UNKNOWN"


def _infer_horizon(row: dict[str, str]) -> str:
    explicit = _first_text(
        row,
        ("horizon", "market_horizon", "market_family", "slug"),
        default="",
    ).lower()
    horizon_ms = _first_float(row, ("horizon_ms",), default=None)
    if horizon_ms is not None:
        if int(horizon_ms) == 300_000:
            return "5m"
        if int(horizon_ms) == 900_000:
            return "15m"
    if "15m" in explicit or "15-min" in explicit or "15_min" in explicit:
        return "15m"
    if "5m" in explicit or "5-min" in explicit or "5_min" in explicit:
        return "5m"
    return explicit or "unknown"


def _price_in_070_090(price: float | None) -> bool:
    return price is not None and 0.70 <= price <= 0.90


def _price_rejection_reason(price: float | None) -> str:
    if price is None:
        return "missing_entry_price"
    if price < 0.70:
        return "price_below_070"
    return "price_above_090"


def _price_bucket(price: float | None) -> str:
    if price is None:
        return "missing"
    for name, low, high in PRICE_BUCKET_EDGES:
        if price >= low and (high is None or price < high):
            return name
    return "unknown"


def _price_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for bucket in sorted({row["price_bucket"] for row in rows}):
        bucket_rows = [row for row in rows if row["price_bucket"] == bucket]
        summary[bucket] = _metric_subset(bucket_rows)
    return summary


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        summary[family] = _metric_subset(family_rows)
    return summary


def _metric_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost = sum(float(row["cost_basis"]) for row in rows)
    pnl = sum(float(row["settlement_pnl"]) for row in rows)
    return {
        "row_count": len(rows),
        "cost_basis": cost,
        "settlement_pnl": pnl,
        "roi": pnl / cost if cost else 0.0,
        "win_rate": (
            sum(1 for row in rows if float(row["settlement_pnl"]) > 0.0) / len(rows)
            if rows
            else 0.0
        ),
    }


def _max_drawdown(pnl_values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _count_distribution(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field_name, "unknown")) for row in rows).items()))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _canonical_action(action: str) -> str:
    return action.strip().upper().replace(" ", "_")


def _first_text(
    row: dict[str, str],
    field_names: tuple[str, ...],
    *,
    default: str,
) -> str:
    for field_name in field_names:
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _first_float(
    row: dict[str, str],
    field_names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    value, _ = _first_float_with_field(row, field_names)
    return default if value is None else value


def _first_float_with_field(
    row: dict[str, str],
    field_names: tuple[str, ...],
) -> tuple[float | None, str | None]:
    for field_name in field_names:
        raw_value = row.get(field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            return value, field_name
    return None, None


def _first_sort_number(
    row: dict[str, str],
    field_names: tuple[str, ...],
    *,
    default: float,
) -> float:
    for field_name in field_names:
        raw_value = row.get(field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        parsed = _parse_sort_number(str(raw_value))
        if math.isfinite(parsed):
            return parsed
    return float(default)


def _parse_sort_number(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError:
        return float(MISSING_SORT_NUMBER)
    return value if math.isfinite(value) else float(MISSING_SORT_NUMBER)


def _safety_report_fields() -> dict[str, Any]:
    return {
        **compact_safety_fields(),
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _validate_safety_flags(obj: Any) -> None:
    for field_name, expected in _safety_report_fields().items():
        if field_name.startswith("#"):
            continue
        if hasattr(obj, field_name) and getattr(obj, field_name) is not expected:
            raise ValueError(f"{field_name} must be {expected}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
