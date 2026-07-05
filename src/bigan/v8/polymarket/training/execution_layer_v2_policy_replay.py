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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.execution_layer_v2 import (
    EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS,
)

EXECUTION_LAYER_V2_POLICY_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-v1"
)
EXECUTION_LAYER_V2_POLICY_REPLAY_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-policy-replay-manifest-v1"
)
EXECUTION_LAYER_V2_CALIBRATED_EV_MAPPING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-calibrated-ev-mapping-v1"
)
EXECUTION_LAYER_V2_FORWARD_SHADOW_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-forward-shadow-policy-v1"
)
EXECUTION_LAYER_V2_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-forward-shadow-manifest-v1"
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

FORWARD_SHADOW_POLICY_VARIANTS: tuple[str, ...] = (
    "baseline_current_guard",
    "bucket_aware_v1_conservative",
    "bucket_aware_v1_plus_sbc",
    "calibrated_ev_v2",
    "calibrated_ev_plus_bucket_v2",
)

PRICE_BUCKET_EDGES: tuple[tuple[str, float, float | None], ...] = (
    ("lt_0_60", -math.inf, 0.60),
    ("0_60_0_70", 0.60, 0.70),
    ("0_70_0_90", 0.70, 0.90),
    ("gt_0_90", 0.90, None),
)
MISSING_SORT_NUMBER = 10**30
DEFAULT_FORWARD_SHADOW_EXECUTION_COST = 0.001


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


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2ForwardShadowConfig:
    """Configuration for outcome-free execution-layer v2 forward shadow replay."""

    run_id: str
    input_path: Path | str
    output_dir: Path | str
    overwrite_existing: bool = False
    max_rows: int | None = None
    entry_ev_threshold: float = 0.02
    default_execution_cost: float = DEFAULT_FORWARD_SHADOW_EXECUTION_COST
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
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        if not math.isfinite(self.entry_ev_threshold) or self.entry_ev_threshold < 0.0:
            raise ValueError("entry_ev_threshold must be finite and non-negative")
        if not math.isfinite(self.default_execution_cost) or self.default_execution_cost < 0.0:
            raise ValueError("default_execution_cost must be finite and non-negative")
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        _validate_safety_flags(self)

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2ForwardShadowResult:
    """Written calibrated-EV mapping and forward-shadow artifact bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    ev_mapping_report: dict[str, Any]
    forward_shadow_report: dict[str, Any]
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


def run_execution_layer_v2_forward_shadow_policy(
    config: ExecutionLayerV2ForwardShadowConfig,
) -> ExecutionLayerV2ForwardShadowResult:
    """Run outcome-free calibrated EV mapping and policy shadow diagnostics."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"forward shadow input not found: {config.input_path}")
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"forward shadow output exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    raw_rows = _load_forward_shadow_rows(config.input_path)
    if config.max_rows is not None:
        raw_rows = raw_rows[: config.max_rows]
    forbidden = _forbidden_fields_by_row(raw_rows)
    normalized_rows = (
        []
        if forbidden
        else [
            _normalize_forward_shadow_row(
                row,
                index,
                default_execution_cost=config.default_execution_cost,
            )
            for index, row in enumerate(raw_rows)
        ]
    )
    ev_mapping_report = build_execution_layer_v2_calibrated_ev_mapping_report(
        normalized_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
    )
    forward_shadow_report = build_execution_layer_v2_forward_shadow_policy_report(
        normalized_rows,
        run_id=config.run_id,
        input_path=str(config.input_path),
        raw_row_count=len(raw_rows),
        forbidden_outcome_fields_by_row=forbidden,
        entry_ev_threshold=config.entry_ev_threshold,
    )
    artifact_paths = {
        "execution_layer_v2_calibrated_ev_mapping_report": run_dir
        / "execution_layer_v2_calibrated_ev_mapping_report.json",
        "execution_layer_v2_calibrated_ev_mapping_summary": run_dir
        / "execution_layer_v2_calibrated_ev_mapping_report.md",
        "execution_layer_v2_forward_shadow_policy_report": run_dir
        / "execution_layer_v2_forward_shadow_policy_report.json",
        "execution_layer_v2_forward_shadow_policy_summary": run_dir
        / "execution_layer_v2_forward_shadow_policy_report.md",
        "execution_layer_v2_forward_shadow_manifest": run_dir
        / "execution_layer_v2_forward_shadow_manifest.json",
    }
    _write_json(
        artifact_paths["execution_layer_v2_calibrated_ev_mapping_report"],
        ev_mapping_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_calibrated_ev_mapping_summary"],
        execution_layer_v2_calibrated_ev_mapping_report_to_markdown(ev_mapping_report),
    )
    _write_json(
        artifact_paths["execution_layer_v2_forward_shadow_policy_report"],
        forward_shadow_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_forward_shadow_policy_summary"],
        execution_layer_v2_forward_shadow_policy_report_to_markdown(
            forward_shadow_report
        ),
    )
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in artifact_paths.items()
        if name != "execution_layer_v2_forward_shadow_manifest"
    }
    manifest = {
        "schema_version": EXECUTION_LAYER_V2_FORWARD_SHADOW_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "input_path": str(config.input_path),
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": dict(artifact_hashes),
        "ev_mapping_report_id": ev_mapping_report[
            "execution_layer_v2_calibrated_ev_mapping_report_id"
        ],
        "forward_shadow_report_id": forward_shadow_report[
            "execution_layer_v2_forward_shadow_policy_report_id"
        ],
        "raw_row_count": len(raw_rows),
        "accepted_signal_row_count": len(normalized_rows),
        "forbidden_outcome_fields_present": bool(forbidden),
        "ev_mapping_status": ev_mapping_report["ev_mapping_status"],
        "calibrated_ev_available": ev_mapping_report["calibrated_ev_available"],
        "market_implied_probability_used_for_ev": ev_mapping_report[
            "market_implied_probability_used_for_ev"
        ],
        "forward_shadow_policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    _write_json(artifact_paths["execution_layer_v2_forward_shadow_manifest"], manifest)
    artifact_hashes["execution_layer_v2_forward_shadow_manifest"] = _sha256_file(
        artifact_paths["execution_layer_v2_forward_shadow_manifest"]
    )
    return ExecutionLayerV2ForwardShadowResult(
        output_dir=run_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        ev_mapping_report=ev_mapping_report,
        forward_shadow_report=forward_shadow_report,
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


def build_execution_layer_v2_calibrated_ev_mapping_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the decision-time EV source contract report."""

    forbidden = forbidden_outcome_fields_by_row or []
    row_count = len(rows)
    missing_rows = [row for row in rows if not row["calibrated_ev_available"]]
    blocking_reasons = sorted(
        {
            reason
            for row in missing_rows
            for reason in row["ev_mapping_blocking_reason_codes"]
        }
    )
    if forbidden:
        ev_status = "blocked_forbidden_outcome_fields_present"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    elif row_count == 0:
        ev_status = "blocked_no_forward_shadow_rows"
        blocking_reasons = ["no_forward_shadow_rows"]
    elif missing_rows:
        ev_status = "blocked_missing_calibrated_ev_source"
    else:
        ev_status = "calibrated_ev_available"

    report = {
        "schema_version": EXECUTION_LAYER_V2_CALIBRATED_EV_MAPPING_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else row_count,
        "accepted_signal_row_count": row_count,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "ev_mapping_contract": {
            "required_sources": [
                "p_market_implied",
                "p_model_fair_value",
                "calibrated_action_expected_net_return",
                "canonical_o_action_score",
                "execution_price",
            ],
            "allowed_ev_sources": [
                "calibrated_action_expected_net_return",
                "p_model_fair_value_minus_execution_price_minus_cost",
            ],
            "forbidden_ev_sources": [
                "p_market_implied",
                "execution_price_as_probability",
                "settlement_pnl",
                "oracle_action",
                "future_return",
            ],
            "market_implied_probability_used_for_ev": False,
        },
        "ev_mapping_status": ev_status,
        "ev_mapping_blocking_reason_codes": blocking_reasons,
        "calibrated_ev_available": bool(row_count and not missing_rows and not forbidden),
        "calibrated_ev_available_count": row_count - len(missing_rows),
        "calibrated_ev_missing_count": len(missing_rows),
        "market_implied_probability_used_for_ev": False,
        "probability_source_summary": {
            "p_market_implied_count": sum(
                1 for row in rows if row["p_market_implied"] is not None
            ),
            "p_model_fair_value_count": sum(
                1 for row in rows if row["p_model_fair_value"] is not None
            ),
            "calibrated_action_expected_net_return_count": sum(
                1
                for row in rows
                if row["calibrated_action_expected_net_return"] is not None
            ),
            "canonical_o_action_score_count": sum(
                1 for row in rows if row["canonical_o_action_score"] is not None
            ),
            "execution_price_count": sum(
                1 for row in rows if row["execution_price"] is not None
            ),
        },
        "row_ev_mapping_contracts": [
            _ev_mapping_contract_row(row) for row in rows
        ],
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_calibrated_ev_mapping_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def build_execution_layer_v2_forward_shadow_policy_report(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    input_path: str,
    raw_row_count: int | None = None,
    forbidden_outcome_fields_by_row: list[dict[str, Any]] | None = None,
    entry_ev_threshold: float = 0.02,
) -> dict[str, Any]:
    """Build an outcome-free forward shadow comparison of v2 policy variants."""

    forbidden = forbidden_outcome_fields_by_row or []
    if forbidden:
        variant_reports = {
            name: _blocked_forward_shadow_variant_metrics(
                name,
                rows,
                ["forbidden_outcome_fields_present"],
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        policy_status = "blocked_fail_closed"
        blocking_reasons = ["forbidden_outcome_fields_present"]
    else:
        variant_reports = {
            name: _forward_shadow_variant_metrics(
                rows,
                name,
                entry_ev_threshold=entry_ev_threshold,
            )
            for name in FORWARD_SHADOW_POLICY_VARIANTS
        }
        policy_status = "diagnostic_only_fail_closed"
        blocking_reasons = []
        if not rows:
            policy_status = "blocked_fail_closed"
            blocking_reasons.append("no_forward_shadow_rows")

    report = {
        "schema_version": EXECUTION_LAYER_V2_FORWARD_SHADOW_SCHEMA_VERSION,
        "run_id": run_id,
        "input_path": input_path,
        "raw_row_count": raw_row_count if raw_row_count is not None else len(rows),
        "accepted_signal_row_count": len(rows),
        "forward_shadow_policy_status": policy_status,
        "forward_shadow_policy_blocking_reason_codes": blocking_reasons,
        "diagnostic_only": True,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "uses_settlement_labels_for_threshold_tuning": False,
        "entry_ev_threshold": entry_ev_threshold,
        "policy_variant_names": list(FORWARD_SHADOW_POLICY_VARIANTS),
        "policy_variant_definitions": _forward_shadow_variant_definitions(),
        "policy_variants": variant_reports,
        "forbidden_outcome_fields_present": bool(forbidden),
        "forbidden_outcome_fields_by_row": forbidden,
        "time_window_behavior": _forward_shadow_time_window_behavior(rows),
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "market_implied_probability_diagnostic_only": True,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "paper_live_unlock_changed": False,
        **_safety_report_fields(),
    }
    report["execution_layer_v2_forward_shadow_policy_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def execution_layer_v2_calibrated_ev_mapping_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the calibrated EV mapping contract report."""

    return "\n".join(
        [
            "# v8 Execution Layer v2 Calibrated EV Mapping",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- raw_row_count: `{report['raw_row_count']}`",
            f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
            f"- ev_mapping_status: `{report['ev_mapping_status']}`",
            f"- calibrated_ev_available: `{report['calibrated_ev_available']}`",
            f"- calibrated_ev_available_count: `{report['calibrated_ev_available_count']}`",
            f"- calibrated_ev_missing_count: `{report['calibrated_ev_missing_count']}`",
            f"- market_implied_probability_used_for_ev: `{report['market_implied_probability_used_for_ev']}`",
            f"- forbidden_outcome_fields_present: `{report['forbidden_outcome_fields_present']}`",
            f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["ev_mapping_blocking_reason_codes"]),
            "",
            "## Probability Source Summary",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in report["probability_source_summary"].items()
            ],
            "",
        ]
    )


def execution_layer_v2_forward_shadow_policy_report_to_markdown(
    report: dict[str, Any],
) -> str:
    """Render the forward shadow policy comparison report."""

    lines = [
        "# v8 Execution Layer v2 Forward Shadow Policy",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- raw_row_count: `{report['raw_row_count']}`",
        f"- accepted_signal_row_count: `{report['accepted_signal_row_count']}`",
        f"- forward_shadow_policy_status: `{report['forward_shadow_policy_status']}`",
        f"- diagnostic_only: `{report['diagnostic_only']}`",
        f"- uses_settlement_pnl_or_outcome_labels: `{report['uses_settlement_pnl_or_outcome_labels']}`",
        f"- market_implied_probability_used_as_calibrated_ev_source: `{report['market_implied_probability_used_as_calibrated_ev_source']}`",
        f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
        "",
        "## Policy Variants",
        "",
        "| variant | allowed | entries | exits | holds | no_trade | rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FORWARD_SHADOW_POLICY_VARIANTS:
        metrics = report["policy_variants"][name]
        lines.append(
            f"| `{name}` | {metrics['allowed_decision_count']} | "
            f"{metrics['entry_count']} | {metrics['exit_count']} | "
            f"{metrics['hold_count']} | {metrics['no_trade_count']} | "
            f"{metrics['rejected_decision_count']} |"
        )
    lines.extend(
        [
            "",
            "## Blocking Reasons",
            "",
            *_markdown_list(report["forward_shadow_policy_blocking_reason_codes"]),
            "",
        ]
    )
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


def _forward_shadow_variant_metrics(
    rows: list[dict[str, Any]],
    variant: str,
    *,
    entry_ev_threshold: float,
) -> dict[str, Any]:
    allowed_rows: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        allowed, reasons = _forward_shadow_variant_allows_row(
            row,
            variant,
            entry_ev_threshold=entry_ev_threshold,
        )
        if allowed:
            allowed_rows.append(row)
        else:
            rejected_reasons.update(reasons)
        decision_rows.append(_forward_shadow_decision_row(row, variant, allowed, reasons))

    entry_count = sum(1 for row in allowed_rows if _is_entry_action(row["action"]))
    exit_count = sum(1 for row in allowed_rows if _is_exit_action(row["action"]))
    hold_count = sum(1 for row in allowed_rows if _is_hold_action(row["action"]))
    no_trade_count = sum(1 for row in allowed_rows if row["action"] == "NO_TRADE")
    return {
        "variant_name": variant,
        "allowed_decision_count": len(allowed_rows),
        "rejected_decision_count": len(rows) - len(allowed_rows),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "hold_count": hold_count,
        "no_trade_count": no_trade_count,
        "action_distribution": _count_distribution(allowed_rows, "action"),
        "family_distribution": _count_distribution(allowed_rows, "family"),
        "side_distribution": _count_distribution(allowed_rows, "side"),
        "time_window_distribution": _count_distribution(rows, "time_window_bucket"),
        "allowed_time_window_distribution": _count_distribution(
            allowed_rows,
            "time_window_bucket",
        ),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "decision_rows": decision_rows,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _blocked_forward_shadow_variant_metrics(
    variant: str,
    rows: list[dict[str, Any]],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "allowed_decision_count": 0,
        "rejected_decision_count": len(rows),
        "entry_count": 0,
        "exit_count": 0,
        "hold_count": 0,
        "no_trade_count": 0,
        "action_distribution": {},
        "family_distribution": {},
        "side_distribution": {},
        "time_window_distribution": {},
        "allowed_time_window_distribution": {},
        "rejected_reason_counts": dict(Counter(reason_codes)),
        "decision_rows": [],
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _forward_shadow_variant_allows_row(
    row: dict[str, Any],
    variant: str,
    *,
    entry_ev_threshold: float,
) -> tuple[bool, list[str]]:
    if row["action"] == "NO_TRADE":
        return False, ["selected_action_no_trade"]
    if variant == "baseline_current_guard":
        return _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="baseline_current_guard_missing_calibrated_ev_source",
            threshold_reason="baseline_current_guard_ev_threshold_not_met",
        )
    if variant == "bucket_aware_v1_conservative":
        return _variant_allows_row(row, "bucket_aware_v1_conservative")
    if variant == "bucket_aware_v1_plus_sbc":
        return _variant_allows_row(row, "bucket_aware_v1_plus_sbc")
    if variant == "calibrated_ev_v2":
        return _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="calibrated_ev_source_missing",
            threshold_reason="calibrated_ev_threshold_not_met",
        )
    if variant == "calibrated_ev_plus_bucket_v2":
        allowed, reasons = _calibrated_ev_threshold_decision(
            row,
            entry_ev_threshold=entry_ev_threshold,
            missing_reason="calibrated_ev_source_missing",
            threshold_reason="calibrated_ev_threshold_not_met",
        )
        bucket_allowed, bucket_reasons = _variant_allows_row(
            row,
            "bucket_aware_v1_plus_sbc",
        )
        reasons.extend(bucket_reasons)
        return allowed and bucket_allowed, reasons
    raise ValueError(f"unsupported forward shadow variant: {variant}")


def _calibrated_ev_threshold_decision(
    row: dict[str, Any],
    *,
    entry_ev_threshold: float,
    missing_reason: str,
    threshold_reason: str,
) -> tuple[bool, list[str]]:
    if not row["calibrated_ev_available"]:
        return False, [missing_reason, *row["ev_mapping_blocking_reason_codes"]]
    if float(row["calibrated_ev"]) < entry_ev_threshold:
        return False, [threshold_reason]
    return True, []


def _forward_shadow_decision_row(
    row: dict[str, Any],
    variant: str,
    allowed: bool,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "family": row["family"],
        "time_window_bucket": row["time_window_bucket"],
        "execution_price": row["execution_price"],
        "p_market_implied": row["p_market_implied"],
        "p_model_fair_value": row["p_model_fair_value"],
        "calibrated_action_expected_net_return": row[
            "calibrated_action_expected_net_return"
        ],
        "canonical_o_action_score": row["canonical_o_action_score"],
        "calibrated_ev": row["calibrated_ev"],
        "ev_source": row["ev_source"],
        "ev_mapping_status": row["ev_mapping_status"],
        "shadow_decision_allowed": allowed,
        "shadow_decision_type": _decision_type(row["action"]) if allowed else "REJECTED",
        "rejection_reason_codes": reason_codes,
        "market_implied_probability_used_as_calibrated_ev_source": False,
        "diagnostic_only": True,
    }


def _forward_shadow_variant_definitions() -> dict[str, str]:
    return {
        "baseline_current_guard": (
            "current v2 entry EV threshold contract, fail-closed unless a calibrated "
            "EV source is available"
        ),
        "bucket_aware_v1_conservative": (
            "diagnostic bucket policy from #166: price 0.70-0.90, exclude "
            "BUY_UP_HOLD_TO_SETTLEMENT, and require SELL_BEFORE_CLOSE to pass "
            "the same price bucket"
        ),
        "bucket_aware_v1_plus_sbc": (
            "diagnostic bucket policy from #166: allow SELL_BEFORE_CLOSE regardless "
            "of price bucket, allow BUY_DOWN_HOLD_TO_SETTLEMENT only in 0.70-0.90, "
            "exclude BUY_UP_HOLD_TO_SETTLEMENT"
        ),
        "calibrated_ev_v2": (
            "allow rows only when calibrated_action_expected_net_return or "
            "calibrated fair value minus executable price minus cost clears the "
            "entry EV threshold"
        ),
        "calibrated_ev_plus_bucket_v2": (
            "calibrated_ev_v2 plus bucket_aware_v1_plus_sbc action/price filters"
        ),
    }


def _forward_shadow_time_window_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "time_window_distribution": _count_distribution(rows, "time_window_bucket"),
        "entry_time_window_distribution": _count_distribution(
            [row for row in rows if _is_entry_action(row["action"])],
            "time_window_bucket",
        ),
        "exit_time_window_distribution": _count_distribution(
            [row for row in rows if _is_exit_action(row["action"])],
            "time_window_bucket",
        ),
        "hold_time_window_distribution": _count_distribution(
            [row for row in rows if _is_hold_action(row["action"])],
            "time_window_bucket",
        ),
    }


def _ev_mapping_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row["row_index"],
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "selected_action": row["action"],
        "selected_side": row["side"],
        "p_market_implied": row["p_market_implied"],
        "p_model_fair_value": row["p_model_fair_value"],
        "calibrated_action_expected_net_return": row[
            "calibrated_action_expected_net_return"
        ],
        "canonical_o_action_score": row["canonical_o_action_score"],
        "execution_price": row["execution_price"],
        "execution_cost": row["execution_cost"],
        "ev_source": row["ev_source"],
        "ev_source_provenance": row["ev_source_provenance"],
        "ev_mapping_status": row["ev_mapping_status"],
        "ev_mapping_blocking_reason_codes": row["ev_mapping_blocking_reason_codes"],
        "calibrated_ev_available": row["calibrated_ev_available"],
        "calibrated_ev": row["calibrated_ev"],
        "market_implied_probability_used_for_ev": False,
    }


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


def _load_forward_shadow_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for trace_path in sorted(path.rglob("o_v8_paper_fresh_signal_trace.json")):
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            for row in _extract_forward_shadow_rows_from_object(payload):
                copied = dict(row)
                copied.setdefault("source_signal_trace_path", str(trace_path))
                rows.append(copied)
        if not rows:
            raise ValueError(
                "forward shadow input directory contains no "
                "o_v8_paper_fresh_signal_trace.json trace rows"
            )
        return rows
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("forward shadow JSONL rows must be objects")
                    rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = _extract_forward_shadow_rows_from_object(payload)
    else:
        raise ValueError("forward shadow input must be a JSON object, array, or JSONL")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("forward shadow input rows must be objects")
    return list(rows)


def _extract_forward_shadow_rows_from_object(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for field_name in (
        "signal_trace_rows",
        "trace_rows",
        "holdout_decision_rows",
        "decision_rows",
        "rows",
        "feature_rows",
    ):
        rows = payload.get(field_name)
        if isinstance(rows, list):
            return rows
    raise ValueError(
        "forward shadow JSON object must include one of: signal_trace_rows, "
        "trace_rows, holdout_decision_rows, decision_rows, rows, feature_rows"
    )


def _normalize_forward_shadow_row(
    row: dict[str, Any],
    index: int,
    *,
    default_execution_cost: float,
) -> dict[str, Any]:
    action = _canonical_action(
        _first_text(
            row,
            (
                "selected_action",
                "canonical_selected_action",
                "execution_guarded_action",
                "canonical_o_selected_action",
                "canonical_action",
                "action",
                "policy_action",
                "signal_action",
            ),
            default="NO_TRADE",
        )
    )
    side = _infer_side(row, action)
    execution_price, execution_price_field = _execution_price_for_side(row, side)
    p_market, p_market_field = _probability_source_for_side(
        row,
        side,
        (
            "p_market_implied",
            "market_implied_probability",
            "p_market_implied_{side}",
            "market_implied_probability_{side}",
            "p_{side}",
        ),
    )
    if p_market is None and execution_price is not None:
        p_market = execution_price
        p_market_field = execution_price_field
    p_model, p_model_field = _probability_source_for_side(
        row,
        side,
        (
            "p_model_fair_value",
            "model_fair_value_probability",
            "fair_value_probability",
            "p_model_fair_value_{side}",
            "model_fair_value_{side}",
            "calibrated_p_{side}",
            "model_p_{side}",
            "predicted_probability_{side}",
        ),
    )
    expected_return, expected_return_field = _first_float_with_field(
        row,
        (
            "calibrated_action_expected_net_return",
            "action_expected_net_return",
            "expected_net_return",
            "selected_action_expected_net_return",
            "calibrated_ev",
        ),
    )
    canonical_score, canonical_score_field = _first_float_with_field(
        row,
        (
            "canonical_o_action_score",
            "canonical_corrected_score",
            "execution_guarded_score",
            "canonical_action_score",
            "raw_calibrated_action_score",
            "selected_score",
            "action_score",
            "model_score",
        ),
    )
    execution_cost = _first_float(
        row,
        ("execution_cost", "cost", "estimated_cost", "fee_plus_spread_cost"),
        default=default_execution_cost,
    )
    decision_ts = _first_text(row, ("decision_ts", "ts", "timestamp"), default=str(index))
    decision_ts_numeric = _parse_sort_number(decision_ts)
    market_id = _first_text(row, ("market_id", "condition_id", "slug"), default="")
    family = _infer_family(row, action)
    time_to_close = _first_float(
        row,
        (
            "time_to_close_seconds",
            "time_to_expiry_seconds",
            "microstructure_snapshot.time_to_close_seconds",
            "features.time_to_close_seconds",
        ),
        default=None,
    )
    calibrated_ev, ev_source, ev_provenance, ev_status, ev_reasons = (
        _derive_calibrated_ev_contract(
            expected_return=expected_return,
            expected_return_field=expected_return_field,
            p_model=p_model,
            p_model_field=p_model_field,
            execution_price=execution_price,
            execution_price_field=execution_price_field,
            execution_cost=execution_cost,
        )
    )
    return {
        "row_index": index,
        "market_id": market_id,
        "decision_ts": decision_ts,
        "decision_ts_numeric": decision_ts_numeric,
        "action": action,
        "family": family,
        "side": side,
        "horizon": _infer_horizon(row),
        "entry_price": execution_price,
        "execution_price": execution_price,
        "execution_price_source_field": execution_price_field,
        "price_bucket": _price_bucket(execution_price),
        "p_market_implied": p_market,
        "p_market_implied_source_field": p_market_field,
        "p_model_fair_value": p_model,
        "p_model_fair_value_source_field": p_model_field,
        "calibrated_action_expected_net_return": expected_return,
        "calibrated_action_expected_net_return_source_field": expected_return_field,
        "canonical_o_action_score": canonical_score,
        "canonical_o_action_score_source_field": canonical_score_field,
        "execution_cost": execution_cost,
        "time_to_close_seconds": time_to_close,
        "time_window_bucket": _time_window_bucket(time_to_close),
        "calibrated_ev": calibrated_ev,
        "ev_source": ev_source,
        "ev_source_provenance": ev_provenance,
        "ev_mapping_status": ev_status,
        "ev_mapping_blocking_reason_codes": ev_reasons,
        "calibrated_ev_available": calibrated_ev is not None,
        "market_implied_probability_used_for_ev": False,
        "raw_fields": sorted(row.keys()),
    }


def _derive_calibrated_ev_contract(
    *,
    expected_return: float | None,
    expected_return_field: str | None,
    p_model: float | None,
    p_model_field: str | None,
    execution_price: float | None,
    execution_price_field: str | None,
    execution_cost: float | None,
) -> tuple[float | None, str, dict[str, Any], str, list[str]]:
    if expected_return is not None:
        return (
            expected_return,
            "calibrated_action_expected_net_return",
            {
                "source_fields_used": [expected_return_field],
                "formula": "calibrated_action_expected_net_return",
                "market_implied_probability_used_for_ev": False,
            },
            "calibrated_ev_available",
            [],
        )
    if p_model is not None and execution_price is not None:
        cost = 0.0 if execution_cost is None else execution_cost
        return (
            p_model - execution_price - cost,
            "p_model_fair_value_minus_execution_price_minus_cost",
            {
                "source_fields_used": [p_model_field, execution_price_field, "execution_cost"],
                "formula": "p_model_fair_value - execution_price - cost",
                "market_implied_probability_used_for_ev": False,
            },
            "calibrated_ev_available",
            [],
        )
    reasons = []
    if p_model is None and expected_return is None:
        reasons.append("missing_calibrated_model_fair_value_or_action_expected_return")
    if execution_price is None and expected_return is None:
        reasons.append("missing_execution_price_for_fair_value_ev")
    return (
        None,
        "missing_calibrated_ev_source",
        {
            "source_fields_used": [],
            "formula": None,
            "market_implied_probability_used_for_ev": False,
        },
        "blocked_missing_calibrated_ev_source",
        reasons,
    )


def _execution_price_for_side(
    row: dict[str, Any],
    side: str,
) -> tuple[float | None, str | None]:
    side_lower = side.lower()
    side_fields = (
        f"execution_price_{side_lower}",
        f"entry_ask_{side_lower}",
        f"ask_{side_lower}",
        f"{side_lower}_ask",
        f"microstructure_snapshot.{side_lower}_ask",
        f"features.{side_lower}_ask",
    )
    generic_fields = (
        "execution_price",
        "entry_price",
        "entry_ask",
        "ask",
        "fill_price",
        "microstructure_snapshot.entry_ask",
        "features.entry_ask",
    )
    return _first_float_with_field(row, (*side_fields, *generic_fields))


def _probability_source_for_side(
    row: dict[str, Any],
    side: str,
    field_templates: tuple[str, ...],
) -> tuple[float | None, str | None]:
    side_lower = side.lower()
    side_upper = side.upper()
    fields = tuple(
        template.format(side=side_lower, SIDE=side_upper)
        for template in field_templates
    )
    value, field = _first_float_with_field(row, fields)
    if value is not None:
        return value, field
    return None, None


def _infer_side(row: dict[str, Any], action: str) -> str:
    explicit = _first_text(
        row,
        (
            "selected_side",
            "canonical_selected_side",
            "execution_guarded_side",
            "side",
            "target_side",
            "outcome",
        ),
        default="",
    ).upper()
    if explicit in {"UP", "DOWN", "NONE"}:
        return explicit
    if "_UP_" in f"_{action}_" or action.startswith("BUY_UP"):
        return "UP"
    if "_DOWN_" in f"_{action}_" or action.startswith("BUY_DOWN"):
        return "DOWN"
    return "NONE"


def _infer_family(row: dict[str, Any], action: str) -> str:
    family = _first_text(
        row,
        (
            "family",
            "action_family",
            "selected_action_family",
            "canonical_selected_family",
            "execution_guarded_family",
            "exit_policy",
        ),
        default="",
    ).upper()
    if "SELL_BEFORE_CLOSE" in family or "SELL_BEFORE_CLOSE" in action:
        return "SELL_BEFORE_CLOSE"
    if "HOLD_TO_SETTLEMENT" in family or "HOLD_TO_SETTLEMENT" in action:
        return "HOLD_TO_SETTLEMENT"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return family or "UNKNOWN"


def _infer_horizon(row: dict[str, Any]) -> str:
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
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: str,
) -> str:
    for field_name in field_names:
        value = _lookup_value(row, field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _first_float(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    value, _ = _first_float_with_field(row, field_names)
    return default if value is None else value


def _first_float_with_field(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> tuple[float | None, str | None]:
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value, field_name
    return None, None


def _first_sort_number(
    row: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: float,
) -> float:
    for field_name in field_names:
        raw_value = _lookup_value(row, field_name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        parsed = _parse_sort_number(str(raw_value))
        if math.isfinite(parsed):
            return parsed
    return float(default)


def _lookup_value(row: Mapping[str, Any], field_name: str) -> Any:
    if field_name in row:
        return row.get(field_name)
    current: Any = row
    for part in field_name.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _parse_sort_number(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError:
        return float(MISSING_SORT_NUMBER)
    return value if math.isfinite(value) else float(MISSING_SORT_NUMBER)


def _time_window_bucket(time_to_close: float | None) -> str:
    if time_to_close is None:
        return "missing"
    if time_to_close <= 60.0:
        return "final_no_trade_window"
    if time_to_close <= 120.0:
        return "sell_before_close_only_window"
    if time_to_close <= 300.0:
        return "hts_allowed_window"
    return "early_window"


def _is_entry_action(action: str) -> bool:
    return action.startswith("BUY_") or action in {"ENTER_POSITION", "ROTATE_POSITION"}


def _is_exit_action(action: str) -> bool:
    return action in {"EXIT_POSITION", "SELL_POSITION"} or action.startswith("SELL_")


def _is_hold_action(action: str) -> bool:
    return action in {"HOLD", "HOLD_POSITION"}


def _decision_type(action: str) -> str:
    if _is_entry_action(action):
        return "ENTRY"
    if _is_exit_action(action):
        return "EXIT"
    if _is_hold_action(action):
        return "HOLD"
    if action == "NO_TRADE":
        return "NO_TRADE"
    return "OTHER"


def _forbidden_fields_by_row(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forbidden_set = set(EXECUTION_LAYER_V2_FORBIDDEN_OUTCOME_FIELDS)
    violations = []
    for index, row in enumerate(rows):
        present = sorted(forbidden_set.intersection(row))
        if present:
            violations.append(
                {
                    "row_index": index,
                    "market_id": str(row.get("market_id", "")),
                    "decision_ts": str(row.get("decision_ts", "")),
                    "forbidden_fields": present,
                }
            )
    return violations


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- `{item}`" for item in items]


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
