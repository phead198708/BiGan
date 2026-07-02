"""Diagnostic O replay-aligned source-ranking reports."""

from __future__ import annotations

import json
import shutil
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
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
O_HTS_P_UP_CONFIDENTLY_WRONG_FEATURE_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-hts-p-up-confidently-wrong-feature-diagnostic-v1"
)
O_FEATURE_SET_SELECTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-feature-set-selection-v1"
)
O_JOINT_FEATURE_CORRECTION_SELECTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-joint-feature-correction-selection-v1"
)
O_V8_ACTION_RANK_HANDOFF_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-action-rank-handoff-v1"
)
O_V8_EXECUTION_RISK_GUARD_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-risk-guard-v1"
)
O_V8_EXECUTION_RUNTIME_STATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-runtime-state-v1"
)
O_V8_EXECUTION_SIMULATED_ORDER_REPLAY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-simulated-order-replay-v1"
)
O_V8_EXECUTION_ALLOWED_ORDER_QUALITY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-allowed-order-quality-v1"
)
O_V8_EXECUTION_POLICY_READINESS_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-policy-readiness-v1"
)
O_V8_EXECUTION_HANDOFF_GATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-handoff-gate-v1"
)
O_V8_FUTURE_UNSEEN_HOLDOUT_PLAN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-future-unseen-holdout-plan-v1"
)
O_V8_PAPER_CANDIDATE_GATE_DESIGN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-candidate-gate-design-v1"
)
O_V8_EXECUTION_GUARD_BLOCK_ANALYSIS_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-guard-block-analysis-v1"
)
O_V8_EXECUTION_RUNTIME_FIELD_COVERAGE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-execution-runtime-field-coverage-v1"
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
    "reference_price_to_beat_distance",
    "reference_momentum_30s",
    "reference_momentum_60s",
    "side_book_depth_imbalance",
    "side_book_update_velocity",
    "hts_vs_sell_before_close_exit_value_gap_proxy",
    "p_up_bucket_calibration_residual",
)
O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES = (
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
        for signal_name in (
            "p_up",
            "p_down",
            "time_to_close",
            "spread",
            "queue",
            "staleness",
            "entry_ask",
            "exit_bid_proxy",
        )
    ),
)
O_REFERENCE_PRICE_MODEL_FEATURE_NAMES = (
    "reference_price_to_beat_distance_scaled",
    "recent_reference_price_momentum_30s_scaled",
    "recent_reference_price_momentum_60s_scaled",
    "recent_reference_price_momentum_120s_scaled",
    "reference_price_feature_available",
)
O_BOOK_PRESSURE_MODEL_FEATURE_NAMES = (
    "side_book_depth_imbalance",
    "side_book_update_velocity_scaled",
    "side_book_staleness_seconds",
    "opposite_book_staleness_seconds",
    "side_spread_bps_scaled",
    "side_queue_fill_proxy",
    "hts_vs_sell_before_close_exit_value_gap_proxy",
    "p_up_bucket_calibration_residual",
    "book_pressure_feature_available",
)
O_EXPANDED_DECISION_TIME_FEATURE_FIELDS = (
    "reference_price_to_beat_distance_at_decision",
    "recent_reference_price_momentum_30s",
    "recent_reference_price_momentum_60s",
    "recent_reference_price_momentum_120s",
    "side_book_depth_imbalance",
    "side_book_update_velocity",
    "side_book_staleness_ms",
    "opposite_book_staleness_ms",
    "side_spread_bps",
    "side_queue_fill_proxy",
    "hts_vs_sell_before_close_exit_value_gap_proxy",
    "p_up_calibration_residual_by_time_spread_queue_bucket",
)
O_DEPLOYABLE_MODEL_FEATURE_NAMES = (
    *O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES,
    *O_REFERENCE_PRICE_MODEL_FEATURE_NAMES,
    *O_BOOK_PRESSURE_MODEL_FEATURE_NAMES,
    *tuple(
        f"{action_slug}_x_{signal_name}"
        for _, action_slug in O_ACTION_FEATURE_SLUGS
        for signal_name in O_ACTION_INTERACTION_SIGNAL_NAMES
        if f"{action_slug}_x_{signal_name}"
        not in O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES
    ),
)
O_MIN_VALIDATION_DECISION_GROUPS = 20
O_MIN_HIGH_SCORE_SUPPORT_COUNT = 10
O_FEATURE_SET_SELECTION_MIN_HIGH_SCORE_SUPPORT_COUNT = 5
O_MIN_TOP1_HIT_RATE = 0.35
O_MAX_MEAN_REGRET = 0.15
O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET = 0.25
O_MAX_NO_TRADE_SELECTION_RATE = 0.80
O_MAX_P_UP_ACTION_DISAGREEMENT_RATE = 0.35
O_SHADOW_P_UP_SELECTION_BUFFER_TARGET = 0.25
O_V8_EXECUTION_MAX_SPREAD_BPS = 1_000.0
O_V8_EXECUTION_MAX_BOOK_STALENESS_MS = 2_000.0
O_V8_EXECUTION_MIN_QUEUE_FILL = 0.50
O_V8_EXECUTION_MIN_TIME_TO_CLOSE_SECONDS = 60.0
O_V8_EXECUTION_MIN_HTS_TIME_TO_CLOSE_SECONDS = 120.0
O_V8_EXECUTION_MIN_SCORE_MARGIN = 0.02
O_V8_EXECUTION_BASE_ORDER_SIZE = 0.20
O_V8_EXECUTION_INITIAL_CASH_BUDGET = 2.00
O_V8_EXECUTION_MAX_TOTAL_EXPOSURE = 1.00
O_V8_EXECUTION_MAX_MARKET_EXPOSURE = 0.20
O_V8_EXECUTION_MAX_SIDE_EXPOSURE = 0.70


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
    hts_p_up_confidently_wrong_feature_diagnostic_report: dict[str, Any]
    feature_set_selection_report: dict[str, Any]
    joint_feature_correction_selection_report: dict[str, Any]
    v8_action_rank_handoff_report: dict[str, Any]
    v8_execution_risk_guard_report: dict[str, Any]
    v8_execution_runtime_state_report: dict[str, Any]
    v8_execution_simulated_order_replay_report: dict[str, Any]
    v8_execution_allowed_order_quality_report: dict[str, Any]
    v8_execution_policy_readiness_report: dict[str, Any]
    v8_execution_guard_block_analysis_report: dict[str, Any]
    v8_execution_runtime_field_coverage_report: dict[str, Any]
    v8_execution_handoff_gate_report: dict[str, Any]
    v8_future_unseen_holdout_plan_report: dict[str, Any]
    v8_paper_candidate_gate_design_report: dict[str, Any]
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
        "hts_p_up_confidently_wrong_feature_diagnostic_report": run_dir
        / "o_hts_p_up_confidently_wrong_feature_diagnostic_report.json",
        "hts_p_up_confidently_wrong_feature_diagnostic_summary": run_dir
        / "o_hts_p_up_confidently_wrong_feature_diagnostic_report.md",
        "feature_set_selection_report": run_dir
        / "o_feature_set_selection_report.json",
        "feature_set_selection_summary": run_dir
        / "o_feature_set_selection_report.md",
        "joint_feature_correction_selection_report": run_dir
        / "o_joint_feature_correction_selection_report.json",
        "joint_feature_correction_selection_summary": run_dir
        / "o_joint_feature_correction_selection_report.md",
        "v8_action_rank_handoff_report": run_dir
        / "o_v8_action_rank_handoff_report.json",
        "v8_action_rank_handoff_summary": run_dir
        / "o_v8_action_rank_handoff_report.md",
        "v8_execution_risk_guard_report": run_dir
        / "o_v8_execution_risk_guard_report.json",
        "v8_execution_risk_guard_summary": run_dir
        / "o_v8_execution_risk_guard_report.md",
        "v8_execution_runtime_state_report": run_dir
        / "o_v8_execution_runtime_state_report.json",
        "v8_execution_runtime_state_summary": run_dir
        / "o_v8_execution_runtime_state_report.md",
        "v8_execution_simulated_order_replay_report": run_dir
        / "o_v8_execution_simulated_order_replay_report.json",
        "v8_execution_simulated_order_replay_summary": run_dir
        / "o_v8_execution_simulated_order_replay_report.md",
        "v8_execution_allowed_order_quality_report": run_dir
        / "o_v8_execution_allowed_order_quality_report.json",
        "v8_execution_allowed_order_quality_summary": run_dir
        / "o_v8_execution_allowed_order_quality_report.md",
        "v8_execution_policy_readiness_report": run_dir
        / "o_v8_execution_policy_readiness_report.json",
        "v8_execution_policy_readiness_summary": run_dir
        / "o_v8_execution_policy_readiness_report.md",
        "v8_execution_guard_block_analysis_report": run_dir
        / "o_v8_execution_guard_block_analysis_report.json",
        "v8_execution_guard_block_analysis_summary": run_dir
        / "o_v8_execution_guard_block_analysis_report.md",
        "v8_execution_runtime_field_coverage_report": run_dir
        / "o_v8_execution_runtime_field_coverage_report.json",
        "v8_execution_runtime_field_coverage_summary": run_dir
        / "o_v8_execution_runtime_field_coverage_report.md",
        "v8_execution_handoff_gate_report": run_dir
        / "o_v8_execution_handoff_gate_report.json",
        "v8_execution_handoff_gate_summary": run_dir
        / "o_v8_execution_handoff_gate_report.md",
        "v8_future_unseen_holdout_plan_report": run_dir
        / "o_v8_future_unseen_holdout_plan_report.json",
        "v8_future_unseen_holdout_plan_summary": run_dir
        / "o_v8_future_unseen_holdout_plan_report.md",
        "v8_paper_candidate_gate_design_report": run_dir
        / "o_v8_paper_candidate_gate_design_report.json",
        "v8_paper_candidate_gate_design_summary": run_dir
        / "o_v8_paper_candidate_gate_design_report.md",
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
    _write_json(
        artifact_paths["hts_p_up_confidently_wrong_feature_diagnostic_report"],
        reports[6],
    )
    artifact_paths[
        "hts_p_up_confidently_wrong_feature_diagnostic_summary"
    ].write_text(
        _hts_p_up_confidently_wrong_feature_diagnostic_markdown(reports[6]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["feature_set_selection_report"], reports[7])
    artifact_paths["feature_set_selection_summary"].write_text(
        _feature_set_selection_markdown(reports[7]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["joint_feature_correction_selection_report"], reports[8])
    artifact_paths["joint_feature_correction_selection_summary"].write_text(
        _joint_feature_correction_selection_markdown(reports[8]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["v8_action_rank_handoff_report"], reports[9])
    artifact_paths["v8_action_rank_handoff_summary"].write_text(
        _v8_action_rank_handoff_markdown(reports[9]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["v8_execution_risk_guard_report"], reports[10])
    artifact_paths["v8_execution_risk_guard_summary"].write_text(
        _v8_execution_risk_guard_markdown(reports[10]),
        encoding="utf-8",
    )
    _write_json(artifact_paths["v8_execution_runtime_state_report"], reports[11])
    artifact_paths["v8_execution_runtime_state_summary"].write_text(
        _v8_execution_runtime_state_markdown(reports[11]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_simulated_order_replay_report"],
        reports[12],
    )
    artifact_paths["v8_execution_simulated_order_replay_summary"].write_text(
        _v8_execution_simulated_order_replay_markdown(reports[12]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_allowed_order_quality_report"],
        reports[13],
    )
    artifact_paths["v8_execution_allowed_order_quality_summary"].write_text(
        _v8_execution_allowed_order_quality_markdown(reports[13]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_policy_readiness_report"],
        reports[14],
    )
    artifact_paths["v8_execution_policy_readiness_summary"].write_text(
        _v8_execution_policy_readiness_markdown(reports[14]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_guard_block_analysis_report"],
        reports[15],
    )
    artifact_paths["v8_execution_guard_block_analysis_summary"].write_text(
        _v8_execution_guard_block_analysis_markdown(reports[15]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_runtime_field_coverage_report"],
        reports[16],
    )
    artifact_paths["v8_execution_runtime_field_coverage_summary"].write_text(
        _v8_execution_runtime_field_coverage_markdown(reports[16]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_execution_handoff_gate_report"],
        reports[17],
    )
    artifact_paths["v8_execution_handoff_gate_summary"].write_text(
        _v8_execution_handoff_gate_markdown(reports[17]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_future_unseen_holdout_plan_report"],
        reports[18],
    )
    artifact_paths["v8_future_unseen_holdout_plan_summary"].write_text(
        _v8_future_unseen_holdout_plan_markdown(reports[18]),
        encoding="utf-8",
    )
    _write_json(
        artifact_paths["v8_paper_candidate_gate_design_report"],
        reports[19],
    )
    artifact_paths["v8_paper_candidate_gate_design_summary"].write_text(
        _v8_paper_candidate_gate_design_markdown(reports[19]),
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
        "strict_vs_relaxed_gate_summary": reports[4][
            "strict_vs_relaxed_gate_summary"
        ],
        "strict_calibration_quality_passed": reports[4][
            "strict_calibration_quality_passed"
        ],
        "relaxed_diagnostic_calibration_quality_passed": reports[4][
            "relaxed_diagnostic_calibration_quality_passed"
        ],
        "relaxed_diagnostic_source_candidate": reports[4][
            "relaxed_diagnostic_source_candidate"
        ],
        "relaxed_diagnostic_no_paper_live_unlock": True,
        "v8_action_rank_handoff_report_available": True,
        "v8_execution_risk_guard_report_available": True,
        "v8_execution_risk_guard_report_id": reports[10][
            "o_v8_execution_risk_guard_report_id"
        ],
        "v8_execution_guard_runtime_validation_passed": reports[10][
            "runtime_risk_control_validation_passed"
        ],
        "v8_execution_runtime_state_report_available": True,
        "v8_execution_runtime_state_report_id": reports[11][
            "o_v8_execution_runtime_state_report_id"
        ],
        "v8_execution_runtime_state_validation_passed": reports[11][
            "runtime_state_validation_passed"
        ],
        "v8_execution_simulated_order_replay_report_available": True,
        "v8_execution_simulated_order_replay_report_id": reports[12][
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "v8_execution_simulated_allowed_order_count": reports[12][
            "simulated_allowed_order_count"
        ],
        "v8_execution_simulated_blocked_decision_count": reports[12][
            "blocked_decision_count"
        ],
        "v8_execution_simulated_runtime_risk_control_validation_passed": reports[12][
            "runtime_risk_control_validation_passed"
        ],
        "v8_execution_allowed_order_quality_report_available": True,
        "v8_execution_allowed_order_quality_report_id": reports[13][
            "o_v8_execution_allowed_order_quality_report_id"
        ],
        "v8_execution_allowed_order_quality_allowed_order_count": reports[13][
            "allowed_order_count"
        ],
        "v8_execution_allowed_order_quality_blocked_decision_count": reports[13][
            "blocked_decision_count"
        ],
        "v8_execution_allowed_order_quality_recommendation_counts": reports[13][
            "deterministic_recommendation_counts"
        ],
        "v8_execution_policy_readiness_report_available": True,
        "v8_execution_policy_readiness_report_id": reports[14][
            "o_v8_execution_policy_readiness_report_id"
        ],
        "v8_execution_policy_readiness_diagnostic_passed": reports[14][
            "execution_policy_readiness_diagnostic_passed"
        ],
        "v8_execution_policy_readiness_required_checks": reports[14][
            "execution_policy_readiness_required_checks"
        ],
        "v8_execution_policy_readiness_blocking_reason_codes": reports[14][
            "execution_policy_readiness_blocking_reason_codes"
        ],
        "future_explicit_execution_handoff_gate_required": reports[14][
            "future_explicit_execution_handoff_gate_required"
        ],
        "v8_execution_guard_block_analysis_report_available": True,
        "v8_execution_guard_block_analysis_report_id": reports[15][
            "o_v8_execution_guard_block_analysis_report_id"
        ],
        "v8_execution_guard_block_analysis_safe_order_candidate_count": reports[15][
            "safe_order_discovery_summary"
        ]["safe_order_candidate_count"],
        "v8_execution_guard_block_analysis_fundamentally_unsafe_count": reports[15][
            "safe_order_discovery_summary"
        ]["fundamentally_unsafe_count"],
        "v8_execution_guard_block_analysis_primary_blocker_categories": reports[15][
            "primary_blocker_categories"
        ],
        "v8_execution_runtime_field_coverage_report_available": True,
        "v8_execution_runtime_field_coverage_report_id": reports[16][
            "o_v8_execution_runtime_field_coverage_report_id"
        ],
        "v8_execution_runtime_field_missing_decision_count": reports[16][
            "missing_runtime_field_decision_count"
        ],
        "v8_execution_runtime_field_true_data_gap_count": reports[16][
            "classification_counts"
        ]["true_data_coverage_gap"],
        "v8_execution_runtime_field_safe_backfill_candidate_count": reports[16][
            "safe_backfill_candidate_count"
        ],
        "v8_execution_runtime_field_existing_handoff_backfill_candidate_count": (
            reports[16]["existing_handoff_backfill_candidate_count"]
        ),
        "v8_execution_runtime_field_decision_time_data_join_backfill_candidate_count": (
            reports[16]["decision_time_data_join_backfill_candidate_count"]
        ),
        "v8_execution_runtime_field_optional_for_no_trade_count": reports[16][
            "classification_counts"
        ]["optional_for_no_trade"],
        "v8_execution_runtime_field_simulation_policy_too_strict_count": reports[16][
            "classification_counts"
        ]["too_strict_for_simulation_only_mode"],
        "v8_execution_runtime_field_primary_missing_fields": reports[16][
            "primary_missing_runtime_fields"
        ],
        "v8_execution_runtime_field_backfill_rules_applied": reports[16][
            "runtime_field_backfill_rules_applied"
        ],
        "v8_execution_runtime_field_applied_backfill_count": reports[16][
            "applied_runtime_field_backfill_count"
        ],
        "v8_execution_runtime_field_applied_backfill_rule_counts": reports[16][
            "applied_runtime_field_backfill_rule_counts"
        ],
        "v8_execution_runtime_field_backfill_provenance_validity_summary": reports[16][
            "runtime_field_backfill_provenance_validity_summary"
        ],
        "v8_execution_handoff_gate_report_available": True,
        "v8_execution_handoff_gate_report_id": reports[17][
            "o_v8_execution_handoff_gate_report_id"
        ],
        "explicit_execution_handoff_gate_passed": reports[17][
            "explicit_execution_handoff_gate_passed"
        ],
        "explicit_execution_handoff_blocking_reason_codes": reports[17][
            "explicit_execution_handoff_blocking_reason_codes"
        ],
        "explicit_execution_handoff_gate_mode": reports[17][
            "explicit_execution_handoff_gate_mode"
        ],
        "explicit_execution_handoff_allowed": reports[17][
            "v8_execution_handoff_allowed"
        ],
        "future_unseen_holdout_required": reports[17][
            "future_unseen_holdout_required"
        ],
        "future_paper_candidate_gate_required": reports[17][
            "future_paper_candidate_gate_required"
        ],
        "v8_future_unseen_holdout_plan_report_available": True,
        "v8_future_unseen_holdout_plan_report_id": reports[18][
            "o_v8_future_unseen_holdout_plan_report_id"
        ],
        "future_unseen_holdout_plan_ready": reports[18][
            "future_unseen_holdout_plan_ready"
        ],
        "future_unseen_holdout_blocking_reason_codes": reports[18][
            "future_unseen_holdout_blocking_reason_codes"
        ],
        "v8_paper_candidate_gate_design_report_available": True,
        "v8_paper_candidate_gate_design_report_id": reports[19][
            "o_v8_paper_candidate_gate_design_report_id"
        ],
        "paper_candidate_gate_design_ready": reports[19][
            "paper_candidate_gate_design_ready"
        ],
        "paper_candidate_gate_blocking_reason_codes": reports[19][
            "paper_candidate_gate_blocking_reason_codes"
        ],
        "paper_candidate_allowed": reports[19]["paper_candidate_allowed"],
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
        "large_regret_risk_model_report_available": False,
        "selective_action_guard_report_available": False,
        "large_regret_risk_model_enabled": False,
        "selective_action_guard_enabled": False,
        "v8_action_rank_quality_passed": reports[9][
            "v8_action_rank_quality_passed"
        ],
        "v8_action_rank_candidate_eligible": reports[9][
            "v8_action_rank_candidate_eligible"
        ],
        "v8_action_rank_reason_codes": reports[9][
            "v8_action_rank_reason_codes"
        ],
        "v8_execution_risk_control_required": reports[9][
            "v8_execution_risk_control_required"
        ],
        "v8_execution_handoff_allowed": reports[9][
            "v8_execution_handoff_allowed"
        ],
        "v8_execution_handoff_blocking_reason_codes": reports[9][
            "v8_execution_handoff_blocking_reason_codes"
        ],
        "strict_source_gate_remains_failed": reports[9][
            "strict_source_gate_remains_failed"
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "promotion_evidence_eligible": False,
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
        hts_p_up_confidently_wrong_feature_diagnostic_report=reports[6],
        feature_set_selection_report=reports[7],
        joint_feature_correction_selection_report=reports[8],
        v8_action_rank_handoff_report=reports[9],
        v8_execution_risk_guard_report=reports[10],
        v8_execution_runtime_state_report=reports[11],
        v8_execution_simulated_order_replay_report=reports[12],
        v8_execution_allowed_order_quality_report=reports[13],
        v8_execution_policy_readiness_report=reports[14],
        v8_execution_guard_block_analysis_report=reports[15],
        v8_execution_runtime_field_coverage_report=reports[16],
        v8_execution_handoff_gate_report=reports[17],
        v8_future_unseen_holdout_plan_report=reports[18],
        v8_paper_candidate_gate_design_report=reports[19],
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
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
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
    rows, source_reports, label_lookup, feature_lookup = _load_source_rows(m2_report)
    action_rows = _build_complete_decision_action_rows(
        rows=rows,
        label_lookup=label_lookup,
        feature_lookup=feature_lookup,
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
        leakage_report=leakage_report,
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
    hts_p_up_confidently_wrong_feature_diagnostic_report = (
        _hts_p_up_confidently_wrong_feature_diagnostic_report(
            config=config,
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            rows=ranking_rows,
            model_training_summary=model_training_summary,
            eligibility_gate_report=eligibility_gate_report,
        )
    )
    feature_set_selection_report = _feature_set_selection_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        model_training_summary=model_training_summary,
    )
    joint_feature_correction_selection_report = (
        _joint_feature_correction_selection_report(
            config=config,
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            model_training_summary=model_training_summary,
        )
    )
    v8_action_rank_handoff_report = _v8_action_rank_handoff_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        rows=ranking_rows,
        model_training_summary=model_training_summary,
        leakage_report=leakage_report,
        eligibility_gate_report=eligibility_gate_report,
    )
    v8_execution_risk_guard_report = _v8_execution_risk_guard_report(
        config=config,
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report=v8_action_rank_handoff_report,
    )
    (
        v8_execution_runtime_state_report,
        v8_execution_simulated_order_replay_report,
    ) = _v8_execution_simulated_runtime_reports(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report=v8_action_rank_handoff_report,
        execution_guard_report=v8_execution_risk_guard_report,
    )
    v8_execution_allowed_order_quality_report = (
        _v8_execution_allowed_order_quality_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
        )
    )
    v8_execution_policy_readiness_report = (
        _v8_execution_policy_readiness_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
            allowed_order_quality_report=v8_execution_allowed_order_quality_report,
        )
    )
    v8_execution_guard_block_analysis_report = (
        _v8_execution_guard_block_analysis_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            handoff_report=v8_action_rank_handoff_report,
            execution_guard_report=v8_execution_risk_guard_report,
            runtime_state_report=v8_execution_runtime_state_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
        )
    )
    v8_execution_runtime_field_coverage_report = (
        _v8_execution_runtime_field_coverage_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            handoff_report=v8_action_rank_handoff_report,
            execution_guard_report=v8_execution_risk_guard_report,
            runtime_state_report=v8_execution_runtime_state_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
            block_analysis_report=v8_execution_guard_block_analysis_report,
        )
    )
    v8_execution_handoff_gate_report = _v8_execution_handoff_gate_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        policy_readiness_report=v8_execution_policy_readiness_report,
        allowed_order_quality_report=v8_execution_allowed_order_quality_report,
        simulated_order_replay_report=v8_execution_simulated_order_replay_report,
        runtime_field_coverage_report=v8_execution_runtime_field_coverage_report,
        guard_block_analysis_report=v8_execution_guard_block_analysis_report,
    )
    v8_future_unseen_holdout_plan_report = (
        _v8_future_unseen_holdout_plan_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            action_rank_handoff_report=v8_action_rank_handoff_report,
            execution_guard_report=v8_execution_risk_guard_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
            allowed_order_quality_report=v8_execution_allowed_order_quality_report,
            policy_readiness_report=v8_execution_policy_readiness_report,
            handoff_gate_report=v8_execution_handoff_gate_report,
            runtime_field_coverage_report=v8_execution_runtime_field_coverage_report,
            guard_block_analysis_report=v8_execution_guard_block_analysis_report,
        )
    )
    v8_paper_candidate_gate_design_report = (
        _v8_paper_candidate_gate_design_report(
            m2_report_path=m2_report_path,
            m2_report=m2_report,
            action_rank_handoff_report=v8_action_rank_handoff_report,
            execution_guard_report=v8_execution_risk_guard_report,
            simulated_order_replay_report=v8_execution_simulated_order_replay_report,
            allowed_order_quality_report=v8_execution_allowed_order_quality_report,
            policy_readiness_report=v8_execution_policy_readiness_report,
            handoff_gate_report=v8_execution_handoff_gate_report,
            runtime_field_coverage_report=v8_execution_runtime_field_coverage_report,
            guard_block_analysis_report=v8_execution_guard_block_analysis_report,
            holdout_plan_report=v8_future_unseen_holdout_plan_report,
        )
    )
    return (
        label_report,
        ranking_report,
        leakage_report,
        comparison_report,
        eligibility_gate_report,
        freeze_readiness_report,
        hts_p_up_confidently_wrong_feature_diagnostic_report,
        feature_set_selection_report,
        joint_feature_correction_selection_report,
        v8_action_rank_handoff_report,
        v8_execution_risk_guard_report,
        v8_execution_runtime_state_report,
        v8_execution_simulated_order_replay_report,
        v8_execution_allowed_order_quality_report,
        v8_execution_policy_readiness_report,
        v8_execution_guard_block_analysis_report,
        v8_execution_runtime_field_coverage_report,
        v8_execution_handoff_gate_report,
        v8_future_unseen_holdout_plan_report,
        v8_paper_candidate_gate_design_report,
    )


def _load_source_rows(
    m2_report: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, int, str], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
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
    feature_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    seen: set[tuple[str, str, int, str]] = set()
    for path_text in paths:
        path = Path(path_text).expanduser().resolve()
        report = _read_json(path)
        source_label_rows, source_label_path = _load_source_label_rows(report)
        source_feature_rows, source_feature_path = _load_source_feature_rows(report)
        for label_row in source_label_rows:
            label_lookup[
                (
                    str(path),
                    str(label_row.get("market_id") or ""),
                    int(label_row.get("decision_ts") or 0),
                    str(label_row.get("action") or ""),
                )
            ] = label_row
        for feature_row in source_feature_rows:
            feature_lookup[
                (
                    str(path),
                    str(feature_row.get("market_id") or ""),
                    int(feature_row.get("decision_ts") or 0),
                )
            ] = feature_row
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
                "feature_rows_path": str(source_feature_path)
                if source_feature_path
                else None,
                "feature_row_count": len(source_feature_rows),
                "decision_time_feature_rows_available": bool(source_feature_rows),
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
    return rows, source_reports, label_lookup, feature_lookup


def _load_source_label_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    corpus_dir_text = str(report.get("provenance", {}).get("holdout_corpus_dir") or "")
    if not corpus_dir_text:
        return [], None
    label_path = Path(corpus_dir_text).expanduser().resolve() / "polymarket_label_rows.jsonl"
    if not label_path.exists():
        return [], label_path
    return _read_jsonl(label_path), label_path


def _load_source_feature_rows(
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path | None]:
    corpus_dir_text = str(report.get("provenance", {}).get("holdout_corpus_dir") or "")
    if not corpus_dir_text:
        return [], None
    corpus_dir = Path(corpus_dir_text).expanduser().resolve()
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    if not feature_path.exists():
        return [], feature_path
    metadata_by_market = _load_market_metadata_by_market_id(corpus_dir)
    btc_reference_rows = _load_btc_reference_rows(corpus_dir)
    feature_rows = []
    for row in _read_jsonl(feature_path):
        market_id = str(row.get("market_id") or "")
        metadata = metadata_by_market.get(market_id, {})
        enriched_row = _enrich_feature_row_with_reference_price_distance(
            row=row,
            metadata=metadata,
            btc_reference_rows=btc_reference_rows,
        )
        feature_rows.append(
            {
                **enriched_row,
                "market_metadata": metadata,
                "source_feature_rows_path": str(feature_path),
            }
        )
    return feature_rows, feature_path


def _load_market_metadata_by_market_id(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = corpus_dir / "polymarket_market_metadata.jsonl"
    if not metadata_path.exists():
        return {}
    return {
        str(row.get("market_id") or ""): row
        for row in _read_jsonl(metadata_path)
        if row.get("market_id")
    }


def _load_btc_reference_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    btc_path = corpus_dir / "polymarket_btc_reference_candles.jsonl"
    if not btc_path.exists():
        return []
    return sorted(_read_jsonl(btc_path), key=lambda row: int(row.get("ts") or 0))


def _enrich_feature_row_with_reference_price_distance(
    *,
    row: dict[str, Any],
    metadata: dict[str, Any],
    btc_reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    features = dict(row.get("features") or {})
    provenance = dict(row.get("feature_provenance") or {})
    if features.get("reference_price_to_beat_distance_at_decision") is not None:
        return row
    decision_ts = _optional_int(row.get("decision_ts"))
    btc_mid = _optional_float(features.get("btc_mid_price"))
    if decision_ts is None or btc_mid is None:
        return row
    reference_context = _reference_price_to_beat_context_from_feature_sources(
        metadata=metadata,
        btc_reference_rows=btc_reference_rows,
        decision_ts=decision_ts,
    )
    if reference_context is None:
        return row
    reference_price = float(reference_context["reference_price_to_beat"])
    if reference_price <= 0.0:
        return row
    btc_mid_provenance = dict(provenance.get("btc_mid_price") or {})
    btc_mid_input_ts = _optional_int(btc_mid_provenance.get("input_end_ts"))
    btc_mid_available_at_ts = _optional_int(btc_mid_provenance.get("available_at_ts"))
    current_input_ts = btc_mid_input_ts if btc_mid_input_ts is not None else decision_ts
    current_available_at_ts = (
        btc_mid_available_at_ts if btc_mid_available_at_ts is not None else decision_ts
    )
    max_input_ts = max(int(reference_context["max_input_ts"]), current_input_ts)
    available_at_ts = max(
        int(reference_context["available_at_ts"]),
        current_available_at_ts,
    )
    enriched = dict(row)
    features["reference_price_to_beat"] = reference_price
    features["reference_price_to_beat_distance_at_decision"] = (
        (btc_mid - reference_price) / reference_price
    )
    reference_source_fields = str(reference_context["source_fields_used"])
    provenance["reference_price_to_beat"] = {
        "source": "polymarket_feature_row_reference_distance_backfill",
        "input_start_ts": int(reference_context["input_start_ts"]),
        "input_end_ts": int(reference_context["input_end_ts"]),
        "available_at_ts": int(reference_context["available_at_ts"]),
        "lookback_ms": max(0, decision_ts - int(reference_context["input_start_ts"])),
        "source_fields_used": reference_source_fields,
        "max_input_ts": int(reference_context["max_input_ts"]),
        "decision_ts": decision_ts,
        "provenance_valid": int(reference_context["available_at_ts"]) <= decision_ts,
        "reference_price_to_beat_source": str(reference_context["source_type"]),
    }
    provenance["reference_price_to_beat_distance_at_decision"] = {
        "source": "polymarket_feature_row_reference_distance_backfill",
        "input_start_ts": int(reference_context["input_start_ts"]),
        "input_end_ts": max_input_ts,
        "available_at_ts": available_at_ts,
        "lookback_ms": max(0, decision_ts - int(reference_context["input_start_ts"])),
        "source_fields_used": (
            f"{reference_source_fields}|polymarket_feature_rows.features.btc_mid_price"
        ),
        "max_input_ts": max_input_ts,
        "decision_ts": decision_ts,
        "provenance_valid": available_at_ts <= decision_ts,
        "reference_price_to_beat_source": str(reference_context["source_type"]),
    }
    enriched["features"] = features
    enriched["feature_provenance"] = provenance
    enriched["max_input_ts"] = max(int(row.get("max_input_ts") or 0), max_input_ts)
    enriched["available_at_ts"] = max(
        int(row.get("available_at_ts") or 0),
        available_at_ts,
    )
    return enriched


def _reference_price_to_beat_context_from_feature_sources(
    *,
    metadata: dict[str, Any],
    btc_reference_rows: list[dict[str, Any]],
    decision_ts: int,
) -> dict[str, float | int | str] | None:
    market_start_ts = _optional_int(metadata.get("market_start_ts"))
    explicit_reference = _optional_float(
        metadata.get("reference_price_start")
        if metadata.get("reference_price_start") is not None
        else metadata.get("reference_price_at_start")
    )
    if explicit_reference is not None and explicit_reference > 0.0:
        input_ts = market_start_ts if market_start_ts is not None else decision_ts
        return {
            "reference_price_to_beat": explicit_reference,
            "input_start_ts": input_ts,
            "input_end_ts": input_ts,
            "available_at_ts": input_ts,
            "max_input_ts": input_ts,
            "source_fields_used": "polymarket_market_metadata.reference_price_start",
            "source_type": "market_metadata_reference_price_start",
        }
    if market_start_ts is None:
        return None
    for candle in btc_reference_rows:
        candle_ts = _optional_int(candle.get("ts"))
        open_price = _optional_float(candle.get("open_price"))
        if candle_ts == market_start_ts and open_price is not None and open_price > 0.0:
            return {
                "reference_price_to_beat": open_price,
                "input_start_ts": candle_ts,
                "input_end_ts": candle_ts,
                "available_at_ts": candle_ts,
                "max_input_ts": candle_ts,
                "source_fields_used": (
                    "polymarket_btc_reference_candles.open_price_at_market_start"
                ),
                "source_type": "market_start_reference_candle_open_price",
            }
        if candle_ts is not None and candle_ts > market_start_ts:
            break
    eligible_prior = [
        candle
        for candle in btc_reference_rows
        if _optional_int(candle.get("ts")) is not None
        and int(candle["ts"]) <= market_start_ts
        and _optional_int(candle.get("available_at_ts")) is not None
        and int(candle["available_at_ts"]) <= decision_ts
    ]
    if not eligible_prior:
        return None
    prior = eligible_prior[-1]
    prior_ts = int(prior["ts"])
    prior_available_at = int(prior["available_at_ts"])
    prior_close = _optional_float(prior.get("close_price"))
    if prior_close is None or prior_close <= 0.0:
        return None
    return {
        "reference_price_to_beat": prior_close,
        "input_start_ts": prior_ts,
        "input_end_ts": prior_ts,
        "available_at_ts": prior_available_at,
        "max_input_ts": prior_ts,
        "source_fields_used": (
            "polymarket_btc_reference_candles.close_price_before_market_start"
        ),
        "source_type": "prior_available_reference_candle_close_price",
    }


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
    feature_lookup: dict[tuple[str, str, int], dict[str, Any]],
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
            feature_row = feature_lookup.get(
                (source_report_path, market_id, decision_ts)
            )
            observed_row = observed_by_action.get(action)
            action_rows.append(
                _normalize_action_row(
                    _candidate_row_from_label_or_template(
                        template=template,
                        action=action,
                        label_row=label_row,
                        feature_row=feature_row,
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
    feature_row: dict[str, Any] | None,
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
        return _attach_decision_time_feature_fields(base, feature_row)
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
    return _attach_decision_time_feature_fields(base, feature_row)


def _label_target_from_corpus_label(label_row: dict[str, Any]) -> float:
    if label_row.get("total_net_pnl_per_notional") is not None:
        return float(label_row["total_net_pnl_per_notional"])
    if label_row.get("total_net_return") is not None:
        return float(label_row["total_net_return"])
    return 0.0


def _attach_decision_time_feature_fields(
    row: dict[str, Any],
    feature_row: dict[str, Any] | None,
) -> dict[str, Any]:
    action = str(row.get("action") or "")
    side = _side_from_action(action)
    enriched = dict(row)
    features = dict(feature_row.get("features") or {}) if feature_row else {}
    metadata = dict(feature_row.get("market_metadata") or {}) if feature_row else {}
    provenance = dict(feature_row.get("feature_provenance") or {}) if feature_row else {}
    feature_max_input_ts = _optional_float(
        feature_row.get("max_input_ts") if feature_row else None
    )
    decision_ts = _optional_float(enriched.get("decision_ts"))
    provenance_valid = (
        feature_max_input_ts is None
        or decision_ts is None
        or feature_max_input_ts <= decision_ts
    )
    missing_reasons: list[str] = []
    if feature_row is None:
        missing_reasons.append("missing_polymarket_feature_row")
    if not provenance_valid:
        missing_reasons.append("feature_row_max_input_ts_after_decision_ts")

    reference_start = _optional_float(metadata.get("reference_price_start"))
    btc_mid = _optional_float(features.get("btc_mid_price"))
    reference_distance = _optional_float(
        features.get("reference_price_to_beat_distance_at_decision")
    )
    if reference_distance is not None:
        reference_start = _optional_float(
            features.get("reference_price_to_beat", reference_start)
        )
    elif reference_start and btc_mid is not None:
        reference_distance = (btc_mid - reference_start) / reference_start
    else:
        missing_reasons.append("reference_price_to_beat_distance_unavailable")

    momentum_30s = _optional_float(features.get("btc_return_30s"))
    momentum_60s = _optional_float(features.get("btc_return_1m"))
    momentum_120s = _optional_float(
        features.get("btc_return_120s", features.get("btc_return_2m"))
    )
    if momentum_30s is None:
        missing_reasons.append("recent_reference_price_momentum_30s_unavailable")
    if momentum_60s is None:
        missing_reasons.append("recent_reference_price_momentum_60s_unavailable")
    if momentum_120s is None:
        missing_reasons.append("recent_reference_price_momentum_120s_unavailable")

    side_prefix = side.lower() if side in {"UP", "DOWN"} else ""
    opposite_prefix = "down" if side == "UP" else "up" if side == "DOWN" else ""
    side_depth = _optional_float(features.get(f"{side_prefix}_liquidity_depth"))
    opposite_depth = _optional_float(
        features.get(f"{opposite_prefix}_liquidity_depth")
    )
    depth_total = (side_depth or 0.0) + (opposite_depth or 0.0)
    depth_imbalance = (
        ((side_depth or 0.0) - (opposite_depth or 0.0)) / depth_total
        if side_prefix and depth_total > 0.0
        else None
    )
    side_update_count = _optional_float(
        features.get(f"{side_prefix}_recent_book_update_count_1m")
    )
    side_update_velocity = (
        side_update_count / 60.0 if side_update_count is not None else None
    )
    side_staleness = _optional_float(features.get(f"{side_prefix}_book_staleness_ms"))
    opposite_staleness = _optional_float(
        features.get(f"{opposite_prefix}_book_staleness_ms")
    )
    side_spread = _optional_float(
        features.get(f"{side_prefix}_spread_bps", features.get("combined_spread_bps"))
    )
    side_queue = _optional_float(
        features.get(f"{side_prefix}_queue_fill_probability_proxy")
    )
    side_ask = _optional_float(features.get(f"{side_prefix}_ask"))
    side_bid = _optional_float(features.get(f"{side_prefix}_bid"))
    hts_vs_sbc_gap = None
    if side == "UP":
        hts_vs_sbc_gap = _bounded(float(enriched.get("p_up") or 0.5), 0.0, 1.0) - (
            side_bid or 0.0
        )
    elif side == "DOWN":
        hts_vs_sbc_gap = (
            1.0 - _bounded(float(enriched.get("p_up") or 0.5), 0.0, 1.0)
        ) - (side_bid or 0.0)

    if depth_imbalance is None:
        missing_reasons.append("side_specific_book_depth_imbalance_unavailable")
    if side_update_velocity is None:
        missing_reasons.append("side_specific_book_update_velocity_unavailable")
    if side_staleness is None:
        missing_reasons.append("side_specific_book_staleness_unavailable")
    if side_spread is None:
        missing_reasons.append("side_specific_spread_unavailable")
    if side_queue is None:
        missing_reasons.append("side_specific_queue_fill_unavailable")
    if hts_vs_sbc_gap is None:
        missing_reasons.append("hts_vs_sell_before_close_exit_gap_unavailable")

    reference_available = reference_distance is not None and momentum_30s is not None
    book_available = (
        depth_imbalance is not None
        and side_update_velocity is not None
        and side_staleness is not None
        and side_spread is not None
        and side_queue is not None
    )
    enriched.update(
        {
            "decision_time_feature_row_available": feature_row is not None,
            "decision_time_feature_source_path": feature_row.get(
                "source_feature_rows_path"
            )
            if feature_row
            else None,
            "decision_time_feature_max_input_ts": feature_max_input_ts,
            "decision_time_feature_provenance_valid": provenance_valid,
            "decision_time_feature_provenance": provenance,
            "decision_time_feature_missing_reason_codes": sorted(
                set(missing_reasons)
            ),
            "reference_price_to_beat_distance_at_decision": reference_distance,
            "recent_reference_price_momentum_30s": momentum_30s,
            "recent_reference_price_momentum_60s": momentum_60s,
            "recent_reference_price_momentum_120s": momentum_120s,
            "reference_price_feature_available": reference_available,
            "side_book_depth_imbalance": depth_imbalance,
            "side_book_update_velocity": side_update_velocity,
            "side_book_staleness_ms": side_staleness,
            "opposite_book_staleness_ms": opposite_staleness,
            "side_spread_bps": side_spread,
            "side_queue_fill_proxy": side_queue,
            "hts_vs_sell_before_close_exit_value_gap_proxy": hts_vs_sbc_gap,
            "book_pressure_feature_available": book_available,
        }
    )
    if enriched.get("entry_quality_ask") is None and side_ask is not None:
        enriched["entry_quality_ask"] = side_ask
    if enriched.get("exit_quality_bid") is None and side_bid is not None:
        enriched["exit_quality_bid"] = side_bid
    if enriched.get("entry_exit_quality_spread_bps") is None and side_spread is not None:
        enriched["entry_exit_quality_spread_bps"] = side_spread
    if enriched.get("entry_exit_quality_queue_fill") is None and side_queue is not None:
        enriched["entry_exit_quality_queue_fill"] = side_queue
    if (
        enriched.get("entry_exit_quality_book_staleness_ms") is None
        and side_staleness is not None
    ):
        enriched["entry_exit_quality_book_staleness_ms"] = side_staleness
    if enriched.get("entry_exit_quality_time_to_close_seconds") is None:
        time_to_close = _optional_float(features.get("time_to_close_seconds"))
        if time_to_close is not None:
            enriched["entry_exit_quality_time_to_close_seconds"] = time_to_close
        elif decision_ts is not None:
            enriched["entry_exit_quality_time_to_close_seconds"] = (
                _derive_time_to_close_seconds_from_slug(
                    slug=str(enriched.get("slug") or ""),
                    decision_ts=int(decision_ts),
                )
            )
    return enriched


def _derive_time_to_close_seconds_from_slug(*, slug: str, decision_ts: int) -> float | None:
    parts = slug.rsplit("-", 1)
    if len(parts) != 2:
        return None
    try:
        start_seconds = int(parts[-1])
    except ValueError:
        return None
    horizon_seconds = 300 if "5m" in slug else 900 if "15m" in slug else None
    if horizon_seconds is None:
        return None
    end_ms = (start_seconds + horizon_seconds) * 1000
    return max(0.0, (end_ms - decision_ts) / 1000.0)


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
    rows, p_up_residual_summary = _attach_shadow_p_up_bucket_residuals(rows)
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
    feature_set_selection, selected_scoring = _select_o_shadow_feature_set(
        rows=rows,
        train_rows=train_rows,
        deployable_available=deployable_available,
    )
    selected_feature_names = tuple(selected_scoring["feature_names"])
    selected_feature_set_name = str(feature_set_selection["selected_feature_set_name"])
    selected_correction_policy_name = str(
        feature_set_selection["selected_correction_policy_name"]
    )
    selected_high_score_threshold_profile_name = str(
        feature_set_selection["selected_high_score_threshold_profile_name"]
    )
    model = selected_scoring["model"]
    ranking_correction = selected_scoring["ranking_correction"]
    scored_rows = selected_scoring["scored_rows"]
    fit_reason_codes = list(selected_scoring["fit_reason_codes"])
    feature_coverage = _decision_time_feature_coverage(
        rows,
        model_input_fields=selected_feature_names,
    )
    feature_ablation = _o_feature_ablation_diagnostics(rows)
    summary = {
        "model_candidate_name": O_MODEL_PREDICTED_VARIANT,
        "ranking_score_source": "model_predicted_score",
        "final_scoring_source": feature_set_selection[
            "selected_effective_scoring_source"
        ],
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
        "feature_names": list(selected_feature_names),
        "selected_feature_set_name": selected_feature_set_name,
        "selected_correction_policy_name": selected_correction_policy_name,
        "selected_high_score_threshold_profile_name": (
            selected_high_score_threshold_profile_name
        ),
        "selected_high_score_threshold_profile": feature_set_selection[
            "selected_high_score_threshold_profile"
        ],
        "selected_joint_candidate_name": feature_set_selection[
            "selected_joint_candidate_name"
        ],
        "selected_feature_set_config_hash": feature_set_selection[
            "feature_set_selection_config_hash"
        ],
        "joint_feature_correction_selection_config_hash": feature_set_selection[
            "joint_feature_correction_selection_config_hash"
        ],
        "feature_set_selection": feature_set_selection,
        "joint_feature_correction_selection": feature_set_selection[
            "joint_feature_correction_selection"
        ],
        "candidate_feature_set_names": list(_o_feature_set_selection_feature_sets()),
        "candidate_correction_policy_names": [
            profile["correction_policy_name"]
            for profile in _o_correction_policy_profiles()
        ],
        "candidate_high_score_threshold_profile_names": list(
            _o_high_score_threshold_profile_names()
        ),
        "all_candidate_feature_names": list(O_DEPLOYABLE_MODEL_FEATURE_NAMES),
        "model_input_fields_decision_time_only": list(
            selected_feature_names
        ),
        "legacy_feature_names": list(O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES),
        "reference_price_feature_names": list(O_REFERENCE_PRICE_MODEL_FEATURE_NAMES),
        "book_pressure_feature_names": list(O_BOOK_PRESSURE_MODEL_FEATURE_NAMES),
        "expanded_decision_time_feature_fields": list(
            O_EXPANDED_DECISION_TIME_FEATURE_FIELDS
        ),
        "decision_time_feature_coverage": feature_coverage,
        "feature_ablation_diagnostics": feature_ablation,
        "reference_price_feature_effect_summary": (
            _reference_price_feature_effect_summary(
                feature_coverage=feature_coverage,
                feature_ablation=feature_ablation,
            )
        ),
        "p_up_bucket_calibration_residual_summary": p_up_residual_summary,
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
                selected_feature_names,
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


def _attach_shadow_p_up_bucket_residuals(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shadow_rows = [
        row
        for row in rows
        if row.get("split") == "shadow"
        and bool(row.get("label_candidate_available", True))
    ]
    p_edges = []
    seen_groups = set()
    for row in shadow_rows:
        group_id = row["decision_group_id"]
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        p_edges.append(abs(float(row.get("p_up") or 0.5) - 0.5))
    thresholds = _derive_shadow_hts_p_up_reliability_thresholds(
        shadow_rows,
        _p_edge_quantiles(p_edges),
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in shadow_rows:
        grouped[row["decision_group_id"]][str(row.get("action") or "")] = row
    residuals_by_key: dict[str, list[float]] = defaultdict(list)
    for group_rows in grouped.values():
        up = group_rows.get("BUY_UP_HOLD_TO_SETTLEMENT")
        down = group_rows.get("BUY_DOWN_HOLD_TO_SETTLEMENT")
        if up is None or down is None:
            continue
        up_target = float(up["replay_aligned_executable_label_target"])
        down_target = float(down["replay_aligned_executable_label_target"])
        realized_up_probability = 0.5
        if up_target > down_target:
            realized_up_probability = 1.0
        elif down_target > up_target:
            realized_up_probability = 0.0
        key = _p_up_residual_bucket_key(up, thresholds)
        residuals_by_key[key].append(
            realized_up_probability - _bounded(float(up.get("p_up") or 0.5), 0.0, 1.0)
        )
    residual_map = {
        key: statistics.mean(values)
        for key, values in sorted(residuals_by_key.items())
        if values
    }
    enriched_rows = []
    for row in rows:
        key = _p_up_residual_bucket_key(row, thresholds)
        residual = residual_map.get(key)
        enriched_rows.append(
            {
                **row,
                "p_up_calibration_residual_bucket_key": key,
                "p_up_calibration_residual_by_time_spread_queue_bucket": (
                    residual if residual is not None else 0.0
                ),
                "p_up_calibration_residual_available": residual is not None,
                "p_up_calibration_residual_source": "shadow_split_only"
                if residual is not None
                else "unavailable_bucket_default_zero",
            }
        )
    return enriched_rows, {
        "residual_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "bucket_key_fields": ["time_to_close_bucket", "spread_bucket", "queue_bucket"],
        "bucket_thresholds": thresholds,
        "shadow_bucket_count": len(residual_map),
        "shadow_group_count": len(grouped),
        "residual_map": residual_map,
        "rows_with_residual_count": sum(
            1
            for row in enriched_rows
            if bool(row.get("p_up_calibration_residual_available"))
        ),
        "rows_without_residual_count": sum(
            1
            for row in enriched_rows
            if not bool(row.get("p_up_calibration_residual_available"))
        ),
    }


def _p_up_residual_bucket_key(
    row: dict[str, Any],
    thresholds: dict[str, Any],
) -> str:
    context = _hts_p_up_reliability_bucket_context(row, thresholds)
    return "|".join(
        (
            f"time={context['time_to_close_bucket']}",
            f"spread={context['spread_bucket']}",
            f"queue={context['queue_bucket']}",
        )
    )


def _decision_time_feature_coverage(
    rows: list[dict[str, Any]],
    *,
    model_input_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    row_count = len(rows)
    selected_model_inputs = (
        set(model_input_fields) if model_input_fields is not None else None
    )
    field_coverage = {}
    for field in O_EXPANDED_DECISION_TIME_FEATURE_FIELDS:
        available_count = sum(row.get(field) is not None for row in rows)
        field_coverage[field] = {
            "available_count": available_count,
            "missing_count": row_count - available_count,
            "availability_rate": available_count / row_count if row_count else 0.0,
            "decision_time_only": True,
            "used_as_model_input": _expanded_field_used_as_model_input(
                field,
                model_input_fields=selected_model_inputs,
            ),
        }
    provenance_violations = [
        {
            "decision_group_id": row.get("decision_group_id"),
            "market_id": row.get("market_id"),
            "decision_ts": row.get("decision_ts"),
            "decision_time_feature_max_input_ts": row.get(
                "decision_time_feature_max_input_ts"
            ),
        }
        for row in rows
        if row.get("decision_time_feature_provenance_valid") is False
    ]
    return {
        "row_count": row_count,
        "decision_group_count": len({row["decision_group_id"] for row in rows}),
        "feature_row_available_count": sum(
            bool(row.get("decision_time_feature_row_available")) for row in rows
        ),
        "feature_row_missing_count": sum(
            not bool(row.get("decision_time_feature_row_available")) for row in rows
        ),
        "reference_price_feature_available_count": sum(
            bool(row.get("reference_price_feature_available")) for row in rows
        ),
        "book_pressure_feature_available_count": sum(
            bool(row.get("book_pressure_feature_available")) for row in rows
        ),
        "p_up_residual_available_count": sum(
            bool(row.get("p_up_calibration_residual_available")) for row in rows
        ),
        "field_coverage": field_coverage,
        "missing_reason_code_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in row.get(
                        "decision_time_feature_missing_reason_codes",
                        [],
                    )
                ).items()
            )
        ),
        "feature_provenance_violation_count": len(provenance_violations),
        "feature_provenance_violations": provenance_violations[:20],
        "fail_closed_on_provenance_violation": True,
    }


def _expanded_field_used_as_model_input(
    field: str,
    *,
    model_input_fields: set[str] | None = None,
) -> bool:
    model_input_fragments = {
        "reference_price_to_beat_distance_at_decision": (
            "reference_price_to_beat_distance_scaled"
        ),
        "recent_reference_price_momentum_30s": (
            "recent_reference_price_momentum_30s_scaled"
        ),
        "recent_reference_price_momentum_60s": (
            "recent_reference_price_momentum_60s_scaled"
        ),
        "recent_reference_price_momentum_120s": (
            "recent_reference_price_momentum_120s_scaled"
        ),
        "side_book_depth_imbalance": "side_book_depth_imbalance",
        "side_book_update_velocity": "side_book_update_velocity_scaled",
        "side_book_staleness_ms": "side_book_staleness_seconds",
        "opposite_book_staleness_ms": "opposite_book_staleness_seconds",
        "side_spread_bps": "side_spread_bps_scaled",
        "side_queue_fill_proxy": "side_queue_fill_proxy",
        "hts_vs_sell_before_close_exit_value_gap_proxy": (
            "hts_vs_sell_before_close_exit_value_gap_proxy"
        ),
        "p_up_calibration_residual_by_time_spread_queue_bucket": (
            "p_up_bucket_calibration_residual"
        ),
    }
    if model_input_fields is None:
        model_input_fields = set(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
    return model_input_fragments[field] in model_input_fields


def _select_o_shadow_feature_set(
    *,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    deployable_available: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_sets = _o_feature_set_selection_feature_sets()
    correction_policies = _o_correction_policy_profiles()
    high_score_threshold_profiles = _o_high_score_threshold_profile_names()
    baseline_scoring = _score_o_feature_set_candidate(
        rows=rows,
        train_rows=train_rows,
        deployable_available=deployable_available,
        feature_names=feature_sets["old_features_only"],
        full_correction_search=False,
        correction_policy_profile=_o_correction_policy_profile_by_name(
            "balanced_hts_sbc"
        ),
        high_score_threshold_profile_name="current_threshold",
    )
    baseline_largest_regret = _largest_regret_value(
        baseline_scoring["split_metrics"]["train_shadow"]
    )
    scorings: dict[str, dict[str, Any]] = {}
    joint_candidate_rows: list[dict[str, Any]] = []
    for feature_set_name, feature_names in feature_sets.items():
        for policy in correction_policies:
            for threshold_profile_name in high_score_threshold_profiles:
                scoring = _score_o_feature_set_candidate(
                    rows=rows,
                    train_rows=train_rows,
                    deployable_available=deployable_available,
                    feature_names=feature_names,
                    full_correction_search=False,
                    correction_policy_profile=policy,
                    high_score_threshold_profile_name=threshold_profile_name,
                )
                joint_name = _joint_candidate_name(
                    feature_set_name,
                    policy,
                    threshold_profile_name,
                )
                scorings[joint_name] = scoring
                joint_candidate_rows.append(
                    _joint_feature_correction_candidate_row(
                        feature_set_name=feature_set_name,
                        correction_policy_profile=policy,
                        high_score_threshold_profile_name=threshold_profile_name,
                        scoring=scoring,
                        baseline_largest_regret=baseline_largest_regret,
                    )
                )
    eligible_rows = [
        row
        for row in joint_candidate_rows
        if bool(row["shadow_selection_gate_passed"])
    ]
    if eligible_rows:
        selected_row = min(
            eligible_rows,
            key=_joint_selection_sort_key,
        )
        fallback_reason_codes: list[str] = []
    else:
        selected_row = min(
            joint_candidate_rows,
            key=lambda row: (
                not bool(row["shadow_selection_metrics"]["p_up_hard_gate_passed"]),
                int(row["shadow_high_score_support_deficit_to_source_gate"]),
                float(row["shadow_selection_metrics"]["p_up_disagreement_rate"]),
                float(row["shadow_selection_metrics"]["mean_regret"]),
                float(row["shadow_selection_metrics"]["largest_regret_value"]),
                -float(row["shadow_selection_metrics"]["selected_return_sum"]),
            ),
        )
        fallback_reason_codes = ["no_joint_feature_correction_candidate_passed_shadow_gates"]
    selected_name = str(selected_row["feature_set_name"])
    selected_joint_name = str(selected_row["joint_candidate_name"])
    selected_threshold_profile_name = str(
        selected_row["high_score_threshold_profile_name"]
    )
    selected_policy = _o_correction_policy_profile_by_name(
        str(selected_row["correction_policy_name"])
    )
    selected_lightweight_scoring = scorings[selected_joint_name]
    selected_full_scoring = _score_o_feature_set_candidate(
        rows=rows,
        train_rows=train_rows,
        deployable_available=deployable_available,
        feature_names=feature_sets[selected_name],
        full_correction_search=True,
        correction_policy_profile=selected_policy,
        high_score_threshold_profile_name=selected_threshold_profile_name,
    )
    selected_final_row = _joint_feature_correction_candidate_row(
        feature_set_name=selected_name,
        correction_policy_profile=selected_policy,
        high_score_threshold_profile_name=selected_threshold_profile_name,
        scoring=selected_full_scoring,
        baseline_largest_regret=baseline_largest_regret,
    )
    selected_final_row["joint_candidate_name"] = selected_joint_name
    selected_full_correction_acceptance = _selected_full_correction_acceptance(
        lightweight_row=selected_row,
        final_row=selected_final_row,
    )
    selected_full_correction_diagnostics = _selected_full_correction_diagnostics(
        lightweight_row=selected_row,
        final_row=selected_final_row,
        acceptance=selected_full_correction_acceptance,
    )
    selected_scoring = (
        selected_full_scoring
        if bool(selected_full_correction_acceptance["accepted_for_final_scoring"])
        else selected_lightweight_scoring
    )
    selected_effective_row = (
        selected_final_row
        if bool(selected_full_correction_acceptance["accepted_for_final_scoring"])
        else selected_row
    )
    rejected_rows = []
    for row in joint_candidate_rows:
        payload = dict(row)
        payload["selected_joint_candidate"] = (
            row["joint_candidate_name"] == selected_joint_name
        )
        payload["selected_feature_set"] = row["feature_set_name"] == selected_name
        reason_codes = list(payload["shadow_selection_reason_codes"])
        if row["joint_candidate_name"] != selected_joint_name:
            reason_codes.append("not_selected_by_shadow_joint_selection")
        payload["shadow_selection_reason_codes"] = sorted(set(reason_codes))
        if not payload["selected_joint_candidate"]:
            rejected_rows.append(payload)
    feature_candidate_rows = _feature_set_candidate_summary_rows(
        joint_candidate_rows=joint_candidate_rows,
        selected_feature_set_name=selected_name,
    )
    selected_feature_names = list(selected_scoring["feature_names"])
    selected_includes_reference_distance = any(
        _is_reference_distance_feature_name(name) for name in selected_feature_names
    )
    selected_shadow_full_metrics = selected_scoring["split_metrics"]["train_shadow"]
    selected_validation_full_metrics = selected_scoring["split_metrics"]["validation"]
    regret_reduction_selection_config = {
        "selection_metric_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "p_up_safety_target_rate": O_SHADOW_P_UP_SELECTION_BUFFER_TARGET,
        "min_top1_hit_rate": O_MIN_TOP1_HIT_RATE,
        "min_high_score_support_count": O_MIN_HIGH_SCORE_SUPPORT_COUNT,
        "selection_terms": [
            "shadow_mean_regret",
            "shadow_top1_miss_regret_sum",
            "shadow_largest_regret_value",
            "shadow_positive_regret_sum",
            "shadow_no_trade_missed_positive_opportunity_sum",
        ],
    }
    regret_reduction_selection_config_hash = canonical_json_sha256(
        regret_reduction_selection_config
    )
    gate_preservation_diagnostics = {
        "shadow": _gate_preservation_diagnostics_from_summary(
            selected_effective_row["shadow_selection_metrics"]
        ),
        "validation_report_only": _gate_preservation_diagnostics_from_summary(
            selected_effective_row["validation_metrics_report_only"]
        ),
        "selection_metric_source": "shadow_split_only",
        "validation_metrics_report_only": True,
    }
    mean_regret_gate_tradeoff_diagnostics = (
        _mean_regret_gate_tradeoff_diagnostics(
            candidate_rows=joint_candidate_rows,
            selected_effective_row=selected_effective_row,
        )
    )
    joint_selection = {
        "schema_version": O_JOINT_FEATURE_CORRECTION_SELECTION_SCHEMA_VERSION,
        "report_type": "o_joint_feature_correction_selection",
        "diagnostic_only": True,
        "ranking_score_source": "model_predicted_score",
        "uses_validation_labels_for_tuning": False,
        "selection_metric_source": "shadow_split_only",
        "selection_criteria": [
            "shadow_high_score_support_reaches_source_gate_when_possible",
            "shadow_p_up_safety_target_before_regret",
            "shadow_top1_hit_rate_before_regret",
            "shadow_top1_miss_regret_reduction",
            "positive_shadow_selected_return",
            "profitable_shadow_high_score_support",
            "shadow_largest_regret_not_worsened_vs_old_features",
            "shadow_action_family_selected_returns_not_negative",
            "lower_shadow_mean_regret_after_support_and_safety_gates",
        ],
        "shadow_p_up_safety_constrained_selection_enabled": True,
        "shadow_p_up_safety_target_rate": O_SHADOW_P_UP_SELECTION_BUFFER_TARGET,
        "shadow_top1_aware_selection_enabled": True,
        "regret_reduction_selection_config_hash": (
            regret_reduction_selection_config_hash
        ),
        "regret_reduction_selection_config": regret_reduction_selection_config,
        "candidate_feature_set_names": list(feature_sets),
        "candidate_correction_policy_names": [
            profile["correction_policy_name"] for profile in correction_policies
        ],
        "candidate_high_score_threshold_profile_names": list(
            high_score_threshold_profiles
        ),
        "selected_joint_candidate_name": selected_joint_name,
        "selected_feature_set_name": selected_name,
        "selected_correction_policy_name": selected_row["correction_policy_name"],
        "selected_correction_policy_profile": selected_row[
            "correction_policy_profile"
        ],
        "selected_high_score_threshold_profile_name": (
            selected_threshold_profile_name
        ),
        "selected_high_score_threshold_profile": selected_final_row[
            "high_score_threshold_profile"
        ],
        "selected_feature_names": selected_feature_names,
        "selected_feature_count": len(selected_feature_names),
        "selected_effective_scoring_source": (
            "full_shadow_correction_search"
            if bool(selected_full_correction_acceptance["accepted_for_final_scoring"])
            else "lightweight_preselection_shadow_ranker"
        ),
        "selected_shadow_metrics": selected_effective_row["shadow_selection_metrics"],
        "selected_validation_metrics_report_only": selected_effective_row[
            "validation_metrics_report_only"
        ],
        "selected_lightweight_preselection_shadow_metrics": selected_row[
            "shadow_selection_metrics"
        ],
        "selected_lightweight_preselection_validation_metrics_report_only": selected_row[
            "validation_metrics_report_only"
        ],
        "selected_final_full_correction_shadow_metrics": selected_final_row[
            "shadow_selection_metrics"
        ],
        "selected_final_full_correction_validation_metrics_report_only": selected_final_row[
            "validation_metrics_report_only"
        ],
        "selected_lightweight_preselection_candidate_row": selected_row,
        "selected_final_full_correction_candidate_row": selected_final_row,
        "selected_full_correction_rerun_diagnostics": (
            selected_full_correction_diagnostics
        ),
        "mean_regret_reduction_diagnostics": {
            "shadow": selected_shadow_full_metrics[
                "mean_regret_reduction_diagnostics"
            ],
            "validation_report_only": selected_validation_full_metrics[
                "mean_regret_reduction_diagnostics"
            ],
        },
        "largest_regret_case_diagnostics": {
            "shadow": selected_shadow_full_metrics["largest_regret_case"],
            "validation_report_only": selected_validation_full_metrics[
                "largest_regret_case"
            ],
        },
        "top1_miss_regret_diagnostics": {
            "shadow": selected_shadow_full_metrics["top1_miss_diagnostics"],
            "validation_report_only": selected_validation_full_metrics[
                "top1_miss_diagnostics"
            ],
        },
        "action_pair_regret_reduction_diagnostics": {
            "shadow": selected_shadow_full_metrics["action_pair_regret_summary"],
            "validation_report_only": selected_validation_full_metrics[
                "action_pair_regret_summary"
            ],
        },
        "no_trade_missed_opportunity_diagnostics": {
            "shadow": selected_shadow_full_metrics[
                "no_trade_missed_opportunity"
            ],
            "validation_report_only": selected_validation_full_metrics[
                "no_trade_missed_opportunity"
            ],
        },
        "gate_preservation_diagnostics": gate_preservation_diagnostics,
        "mean_regret_gate_tradeoff_diagnostics": (
            mean_regret_gate_tradeoff_diagnostics
        ),
        "selected_joint_shadow_gate_passed": selected_effective_row[
            "shadow_selection_gate_passed"
        ],
        "selection_fallback_reason_codes": fallback_reason_codes,
        "feature_set_selection_min_high_score_support_count": (
            O_FEATURE_SET_SELECTION_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "source_model_gate_min_high_score_support_count": (
            O_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "candidate_count": len(joint_candidate_rows),
        "eligible_candidate_count": len(eligible_rows),
        "eligible_source_support_candidate_count": sum(
            1
            for row in joint_candidate_rows
            if bool(row["shadow_source_gate_high_score_support_passed"])
        ),
        "candidate_rows": joint_candidate_rows,
        "rejected_candidates": rejected_rows,
        "joint_feature_correction_selection_config_hash": "",
        **compact_safety_fields(),
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    joint_selection["joint_feature_correction_selection_config_hash"] = (
        canonical_json_sha256(
            _without_key(
                joint_selection,
                "joint_feature_correction_selection_config_hash",
            )
        )
    )
    report = {
        "schema_version": O_FEATURE_SET_SELECTION_SCHEMA_VERSION,
        "report_type": "o_feature_set_selection",
        "diagnostic_only": True,
        "ranking_score_source": "model_predicted_score",
        "uses_validation_labels_for_tuning": False,
        "selection_metric_source": "shadow_split_only",
        "selection_criteria": [
            "joint_feature_set_and_correction_policy_shadow_selection",
            "shadow_high_score_support_reaches_source_gate_when_possible",
            "shadow_p_up_safety_target_before_regret",
            "shadow_top1_hit_rate_before_regret",
            "shadow_top1_miss_regret_reduction",
            "lower_shadow_mean_regret_after_support_and_safety_gates",
            "positive_shadow_selected_return",
            "profitable_shadow_high_score_support",
            "shadow_action_family_selected_returns_not_negative",
        ],
        "shadow_p_up_safety_constrained_selection_enabled": True,
        "shadow_p_up_safety_target_rate": O_SHADOW_P_UP_SELECTION_BUFFER_TARGET,
        "shadow_top1_aware_selection_enabled": True,
        "joint_selection_report_available": True,
        "joint_selection_schema_version": O_JOINT_FEATURE_CORRECTION_SELECTION_SCHEMA_VERSION,
        "selected_joint_candidate_name": selected_joint_name,
        "selected_correction_policy_name": selected_row["correction_policy_name"],
        "selected_correction_policy_profile": selected_row[
            "correction_policy_profile"
        ],
        "selected_high_score_threshold_profile_name": (
            selected_threshold_profile_name
        ),
        "selected_high_score_threshold_profile": selected_final_row[
            "high_score_threshold_profile"
        ],
        "candidate_correction_policy_names": [
            profile["correction_policy_name"] for profile in correction_policies
        ],
        "candidate_high_score_threshold_profile_names": list(
            high_score_threshold_profiles
        ),
        "feature_set_selection_derived_from_joint_selection": True,
        "feature_set_selection_aggregation": "best_shadow_gate_candidate_per_feature_set",
        "joint_feature_correction_selection": joint_selection,
        "selection_safety_order": [
            "source_gate_high_score_support",
            "support_deficit_to_source_gate",
            "shadow_p_up_safety_target",
            "shadow_top1_hit_rate",
            "shadow_top1_miss_regret_sum",
            "feature_set_complexity_priority",
            "positive_shadow_selected_return",
            "profitable_shadow_high_score_support",
            "shadow_p_up_hard_gate_safety",
            "shadow_largest_regret_not_worsened_vs_old_features",
            "shadow_action_family_selected_returns_not_negative",
            "lower_shadow_mean_regret",
        ],
        "feature_set_selection_min_high_score_support_count": (
            O_FEATURE_SET_SELECTION_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "source_model_gate_min_high_score_support_count": (
            O_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "selected_feature_set_name": selected_name,
        "selected_feature_names": selected_feature_names,
        "selected_feature_count": len(selected_feature_names),
        "selected_effective_scoring_source": (
            "full_shadow_correction_search"
            if bool(selected_full_correction_acceptance["accepted_for_final_scoring"])
            else "lightweight_preselection_shadow_ranker"
        ),
        "selected_shadow_metrics": selected_effective_row["shadow_selection_metrics"],
        "selected_validation_metrics_report_only": selected_effective_row[
            "validation_metrics_report_only"
        ],
        "selected_lightweight_preselection_shadow_metrics": selected_row[
            "shadow_selection_metrics"
        ],
        "selected_final_full_correction_shadow_metrics": selected_final_row[
            "shadow_selection_metrics"
        ],
        "selected_full_correction_rerun_diagnostics": (
            selected_full_correction_diagnostics
        ),
        "mean_regret_reduction_diagnostics": {
            "shadow": selected_shadow_full_metrics[
                "mean_regret_reduction_diagnostics"
            ],
            "validation_report_only": selected_validation_full_metrics[
                "mean_regret_reduction_diagnostics"
            ],
        },
        "largest_regret_case_diagnostics": {
            "shadow": selected_shadow_full_metrics["largest_regret_case"],
            "validation_report_only": selected_validation_full_metrics[
                "largest_regret_case"
            ],
        },
        "top1_miss_regret_diagnostics": {
            "shadow": selected_shadow_full_metrics["top1_miss_diagnostics"],
            "validation_report_only": selected_validation_full_metrics[
                "top1_miss_diagnostics"
            ],
        },
        "action_pair_regret_reduction_diagnostics": {
            "shadow": selected_shadow_full_metrics["action_pair_regret_summary"],
            "validation_report_only": selected_validation_full_metrics[
                "action_pair_regret_summary"
            ],
        },
        "no_trade_missed_opportunity_diagnostics": {
            "shadow": selected_shadow_full_metrics[
                "no_trade_missed_opportunity"
            ],
            "validation_report_only": selected_validation_full_metrics[
                "no_trade_missed_opportunity"
            ],
        },
        "gate_preservation_diagnostics": gate_preservation_diagnostics,
        "mean_regret_gate_tradeoff_diagnostics": (
            mean_regret_gate_tradeoff_diagnostics
        ),
        "regret_reduction_selection_config_hash": (
            regret_reduction_selection_config_hash
        ),
        "selected_feature_set_shadow_gate_passed": selected_effective_row[
            "shadow_selection_gate_passed"
        ],
        "candidate_feature_set_scoring_mode": "joint_shadow_only_prefit_ranker",
        "full_shadow_correction_search_evaluated": True,
        "final_selected_feature_set_uses_full_shadow_correction_search": bool(
            selected_full_correction_acceptance["accepted_for_final_scoring"]
        ),
        "final_selected_correction_policy_uses_shadow_only_profile": True,
        "final_selected_high_score_threshold_profile_uses_shadow_only": True,
        "selection_fallback_reason_codes": fallback_reason_codes,
        "reference_distance_in_selected_feature_set": (
            selected_includes_reference_distance
        ),
        "reference_distance_covered_but_excluded_from_final_model": (
            not selected_includes_reference_distance
        ),
        "candidate_feature_sets": feature_candidate_rows,
        "candidate_joint_feature_correction_rows": joint_candidate_rows,
        "rejected_feature_sets": [
            row for row in feature_candidate_rows if row["feature_set_name"] != selected_name
        ],
        "rejected_joint_feature_correction_candidates": rejected_rows,
        "feature_set_selection_config_hash": "",
        "joint_feature_correction_selection_config_hash": joint_selection[
            "joint_feature_correction_selection_config_hash"
        ],
        **compact_safety_fields(),
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    report["feature_set_selection_config_hash"] = canonical_json_sha256(
        _without_key(report, "feature_set_selection_config_hash")
    )
    return report, selected_scoring


def _score_o_feature_set_candidate(
    *,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    deployable_available: bool,
    feature_names: tuple[str, ...],
    full_correction_search: bool,
    correction_policy_profile: dict[str, Any] | None = None,
    high_score_threshold_profile_name: str = "current_threshold",
) -> dict[str, Any]:
    if deployable_available:
        model = _fit_ridge_regression(
            [_deployable_model_features(row, feature_names) for row in train_rows],
            [
                float(row["replay_aligned_executable_label_target"])
                for row in train_rows
            ],
        )
        predictions = [
            _dot(model["coefficients"], _deployable_model_features(row, feature_names))
            for row in rows
        ]
        fit_reason_codes: list[str] = []
    else:
        model = {
            "coefficients": [0.0 for _ in feature_names],
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
    ranking_correction = _learn_o_shadow_ranking_correction(
        raw_train_rows,
        run_raw_weight_search=full_correction_search,
    )
    if correction_policy_profile is not None:
        ranking_correction = _apply_o_correction_policy_profile(
            ranking_correction,
            correction_policy_profile,
        )
    scored_rows = _apply_o_shadow_ranking_correction(
        rows=raw_scored_rows,
        deployable_available=deployable_available,
        ranking_correction=ranking_correction,
    )
    ranking_rows = _ranking_rows(scored_rows)
    high_score_threshold_profile = _apply_o_high_score_threshold_profile(
        ranking_correction=ranking_correction,
        ranking_rows=ranking_rows,
        profile_name=high_score_threshold_profile_name,
    )
    high_score_threshold = float(
        ranking_correction["high_score_calibration"]["high_score_threshold"]
    )
    return {
        "feature_names": feature_names,
        "correction_policy_profile": correction_policy_profile,
        "high_score_threshold_profile_name": high_score_threshold_profile_name,
        "high_score_threshold_profile": high_score_threshold_profile,
        "model": model,
        "ranking_correction": ranking_correction,
        "scored_rows": scored_rows,
        "ranking_rows": ranking_rows,
        "fit_reason_codes": fit_reason_codes,
        "split_metrics": _split_metric_views(
            ranking_rows,
            O_MODEL_PREDICTED_VARIANT,
            high_score_threshold,
        ),
        "p_up_shadow_summary": _p_up_action_disagreement_summary(
            rows=ranking_rows,
            variant=O_MODEL_PREDICTED_VARIANT,
            split="shadow",
        ),
        "p_up_validation_summary": _p_up_action_disagreement_summary(
            rows=ranking_rows,
            variant=O_MODEL_PREDICTED_VARIANT,
            split="validation",
        ),
    }


def _feature_set_selection_candidate_row(
    *,
    name: str,
    scoring: dict[str, Any],
    baseline_largest_regret: float,
) -> dict[str, Any]:
    shadow_metrics = scoring["split_metrics"]["train_shadow"]
    validation_metrics = scoring["split_metrics"]["validation"]
    shadow_p_up = scoring["p_up_shadow_summary"]
    shadow_summary = _feature_set_selection_metric_summary(
        metrics=shadow_metrics,
        p_up_summary=shadow_p_up,
    )
    validation_summary = _feature_set_selection_metric_summary(
        metrics=validation_metrics,
        p_up_summary=scoring["p_up_validation_summary"],
    )
    reason_codes = []
    if float(shadow_summary["selected_return_sum"]) <= 0.0:
        reason_codes.append("shadow_selected_return_not_positive")
    if (
        int(shadow_summary["high_score_support_count"])
        < O_FEATURE_SET_SELECTION_MIN_HIGH_SCORE_SUPPORT_COUNT
    ):
        reason_codes.append("shadow_high_score_support_insufficient")
    if float(shadow_summary["high_score_return_mean"]) <= 0.0:
        reason_codes.append("shadow_high_score_return_mean_not_positive")
    if float(shadow_summary["high_score_return_sum"]) <= 0.0:
        reason_codes.append("shadow_high_score_return_sum_not_positive")
    if not bool(shadow_summary["p_up_hard_gate_passed"]):
        reason_codes.append("shadow_p_up_hard_gate_failed")
    if float(shadow_summary["largest_regret_value"]) > baseline_largest_regret + 1.0e-12:
        reason_codes.append("shadow_largest_regret_worsened_vs_old_features")
    return {
        "feature_set_name": name,
        "feature_names": list(scoring["feature_names"]),
        "feature_count": len(scoring["feature_names"]),
        "feature_schema_hash": canonical_json_sha256(list(scoring["feature_names"])),
        "excluded_reference_distance_features": [
            feature
            for feature in O_DEPLOYABLE_MODEL_FEATURE_NAMES
            if _is_reference_distance_feature_name(feature)
            and feature not in scoring["feature_names"]
        ],
        "uses_validation_labels_for_tuning": False,
        "shadow_selection_metrics": shadow_summary,
        "validation_metrics_report_only": validation_summary,
        "ranking_correction_config_hash": scoring["ranking_correction"][
            "correction_config_hash"
        ],
        "probe_config_hash": scoring["ranking_correction"]["probe_config_hash"],
        "shadow_selection_gate_passed": not reason_codes,
        "shadow_selection_reason_codes": reason_codes,
    }


def _o_correction_policy_profiles() -> list[dict[str, Any]]:
    return [
        {
            "correction_policy_name": "conservative_hts",
            "correction_policy_family": "conservative_hts",
            "description": "Prefer HTS only when p_up and reliability are clean; otherwise damp HTS.",
            "trade_base_delta": -0.08,
            "sell_before_close_base_delta": 0.02,
            "no_trade_base_delta": 0.0,
            "raw_weight_multiplier": 0.60,
            "p_up_misalignment_penalty_multiplier": 3.0,
            "large_regret_reversal_penalty_multiplier": 3.0,
            "hts_reliability_penalty_multiplier": 3.0,
            "no_trade_tail_buffer_multiplier": 2.0,
            "hts_vs_sell_before_close_priority_profile": "hts_damped_when_reliability_weak",
        },
        {
            "correction_policy_name": "balanced_hts_sbc",
            "correction_policy_family": "balanced_hts_sbc",
            "description": "Balanced HTS/SBC profile using the shadow-derived base correction.",
            "trade_base_delta": 0.0,
            "sell_before_close_base_delta": 0.0,
            "no_trade_base_delta": 0.0,
            "raw_weight_multiplier": 1.0,
            "p_up_misalignment_penalty_multiplier": 1.0,
            "large_regret_reversal_penalty_multiplier": 1.0,
            "hts_reliability_penalty_multiplier": 1.0,
            "no_trade_tail_buffer_multiplier": 1.0,
            "hts_vs_sell_before_close_priority_profile": "balanced",
        },
        {
            "correction_policy_name": "sbc_preferred_when_hts_reliability_weak",
            "correction_policy_family": "sbc_preferred_when_hts_reliability_weak",
            "description": "Prefer SBC when HTS reliability is weak, with strict p_up safety.",
            "trade_base_delta": -0.10,
            "sell_before_close_base_delta": 0.05,
            "no_trade_base_delta": 0.0,
            "raw_weight_multiplier": 0.80,
            "p_up_misalignment_penalty_multiplier": 4.0,
            "large_regret_reversal_penalty_multiplier": 2.5,
            "hts_reliability_penalty_multiplier": 2.5,
            "no_trade_tail_buffer_multiplier": 1.5,
            "hts_vs_sell_before_close_priority_profile": "sell_before_close_preferred",
        },
        {
            "correction_policy_name": "no_trade_tail_risk_buffer",
            "correction_policy_family": "no_trade_tail_risk_buffer",
            "description": "Add a NO_TRADE buffer for weak or risky opportunity tails.",
            "trade_base_delta": -0.12,
            "sell_before_close_base_delta": 0.06,
            "no_trade_base_delta": 0.08,
            "raw_weight_multiplier": 0.70,
            "p_up_misalignment_penalty_multiplier": 3.0,
            "large_regret_reversal_penalty_multiplier": 2.5,
            "hts_reliability_penalty_multiplier": 3.0,
            "no_trade_tail_buffer_multiplier": 2.0,
            "hts_vs_sell_before_close_priority_profile": "tail_risk_buffered",
        },
        {
            "correction_policy_name": "high_score_profitability_preserving",
            "correction_policy_family": "high_score_profitability_preserving",
            "description": "Preserve high-score profitability with stronger raw and safety penalties.",
            "trade_base_delta": -0.04,
            "sell_before_close_base_delta": 0.02,
            "no_trade_base_delta": 0.0,
            "raw_weight_multiplier": 1.20,
            "p_up_misalignment_penalty_multiplier": 4.0,
            "large_regret_reversal_penalty_multiplier": 3.0,
            "hts_reliability_penalty_multiplier": 2.0,
            "no_trade_tail_buffer_multiplier": 1.0,
            "hts_vs_sell_before_close_priority_profile": "high_score_profitability",
        },
        {
            "correction_policy_name": "largest_regret_dampening",
            "correction_policy_family": "regret_reduction",
            "description": "Dampen historically large-regret HTS reversals while preserving p_up safety.",
            "trade_base_delta": -0.06,
            "sell_before_close_base_delta": 0.03,
            "no_trade_base_delta": 0.02,
            "raw_weight_multiplier": 0.90,
            "p_up_misalignment_penalty_multiplier": 3.0,
            "large_regret_reversal_penalty_multiplier": 5.0,
            "hts_reliability_penalty_multiplier": 3.0,
            "no_trade_tail_buffer_multiplier": 1.5,
            "hts_vs_sell_before_close_priority_profile": "largest_regret_dampening",
        },
        {
            "correction_policy_name": "no_trade_missed_opportunity_recovery",
            "correction_policy_family": "regret_reduction",
            "description": "Recover shadow NO_TRADE missed opportunities without using validation labels.",
            "trade_base_delta": 0.02,
            "sell_before_close_base_delta": 0.04,
            "no_trade_base_delta": -0.06,
            "raw_weight_multiplier": 1.0,
            "p_up_misalignment_penalty_multiplier": 2.0,
            "large_regret_reversal_penalty_multiplier": 2.0,
            "hts_reliability_penalty_multiplier": 2.0,
            "no_trade_tail_buffer_multiplier": 0.5,
            "hts_vs_sell_before_close_priority_profile": "no_trade_missed_opportunity_recovery",
        },
        {
            "correction_policy_name": "hts_sbc_regret_balancing",
            "correction_policy_family": "regret_reduction",
            "description": "Balance HTS/SBC action-family regret using shadow-only family priors.",
            "trade_base_delta": -0.03,
            "sell_before_close_base_delta": 0.04,
            "no_trade_base_delta": 0.0,
            "raw_weight_multiplier": 1.05,
            "p_up_misalignment_penalty_multiplier": 2.5,
            "large_regret_reversal_penalty_multiplier": 3.5,
            "hts_reliability_penalty_multiplier": 3.0,
            "no_trade_tail_buffer_multiplier": 1.0,
            "hts_vs_sell_before_close_priority_profile": "hts_sbc_regret_balancing",
        },
        {
            "correction_policy_name": "top1_miss_regret_minimizing",
            "correction_policy_family": "regret_reduction",
            "description": "Reduce shadow top1 miss regret concentration while preserving support.",
            "trade_base_delta": -0.02,
            "sell_before_close_base_delta": 0.02,
            "no_trade_base_delta": 0.02,
            "raw_weight_multiplier": 0.75,
            "p_up_misalignment_penalty_multiplier": 3.5,
            "large_regret_reversal_penalty_multiplier": 4.5,
            "hts_reliability_penalty_multiplier": 4.0,
            "no_trade_tail_buffer_multiplier": 1.2,
            "hts_vs_sell_before_close_priority_profile": "top1_miss_regret_minimizing",
        },
        {
            "correction_policy_name": "p_up_safe_regret_reduction",
            "correction_policy_family": "regret_reduction",
            "description": "Favor lower shadow regret only inside a strict p_up-safe selection envelope.",
            "trade_base_delta": -0.04,
            "sell_before_close_base_delta": 0.03,
            "no_trade_base_delta": 0.01,
            "raw_weight_multiplier": 0.85,
            "p_up_misalignment_penalty_multiplier": 5.0,
            "large_regret_reversal_penalty_multiplier": 4.0,
            "hts_reliability_penalty_multiplier": 4.0,
            "no_trade_tail_buffer_multiplier": 1.5,
            "hts_vs_sell_before_close_priority_profile": "p_up_safe_regret_reduction",
        },
    ]


def _o_correction_policy_profile_by_name(name: str) -> dict[str, Any]:
    for profile in _o_correction_policy_profiles():
        if profile["correction_policy_name"] == name:
            return profile
    raise KeyError(name)


def _o_high_score_threshold_profile_names() -> tuple[str, ...]:
    return (
        "current_threshold",
        "slightly_lower_shadow_derived_threshold",
        "support_preserving_threshold",
        "high_score_profitability_threshold",
    )


def _apply_o_high_score_threshold_profile(
    *,
    ranking_correction: dict[str, Any],
    ranking_rows: list[dict[str, Any]],
    profile_name: str,
) -> dict[str, Any]:
    if profile_name not in _o_high_score_threshold_profile_names():
        raise KeyError(profile_name)
    current_threshold = float(
        ranking_correction["high_score_calibration"]["high_score_threshold"]
    )
    shadow_selected_rows = _selected_score_return_rows(
        ranking_rows,
        variant=O_MODEL_PREDICTED_VARIANT,
        split="shadow",
    )
    p_edge_q25 = float(ranking_correction["p_up_edge_quantiles"]["q25"])
    slightly_lower_threshold = _shadow_slightly_lower_high_score_threshold(
        current_threshold=current_threshold,
        shadow_selected_rows=shadow_selected_rows,
        p_edge_q25=p_edge_q25,
    )
    support_threshold = _shadow_support_preserving_high_score_threshold(
        current_threshold=current_threshold,
        shadow_selected_rows=shadow_selected_rows,
    )
    profitability_threshold = _shadow_high_score_profitability_threshold(
        ranking_rows=ranking_rows,
        current_threshold=current_threshold,
        fallback_threshold=support_threshold,
        extra_thresholds=(slightly_lower_threshold, support_threshold),
    )
    threshold_by_name = {
        "current_threshold": current_threshold,
        "slightly_lower_shadow_derived_threshold": slightly_lower_threshold,
        "support_preserving_threshold": support_threshold,
        "high_score_profitability_threshold": profitability_threshold,
    }
    candidate_rows = [
        _high_score_threshold_profile_candidate_row(
            ranking_rows=ranking_rows,
            profile_name=name,
            threshold=threshold_by_name[name],
        )
        for name in _o_high_score_threshold_profile_names()
    ]
    selected = next(row for row in candidate_rows if row["profile_name"] == profile_name)
    calibration = ranking_correction["high_score_calibration"]
    calibration["selected_high_score_threshold_profile_name"] = profile_name
    calibration["selected_high_score_threshold_profile"] = selected
    calibration["high_score_threshold_profile_source"] = "shadow_split_only"
    calibration["uses_validation_labels_for_threshold_tuning"] = False
    calibration["source_model_gate_min_high_score_support_count"] = (
        O_MIN_HIGH_SCORE_SUPPORT_COUNT
    )
    calibration["high_score_threshold_profile_candidates"] = candidate_rows
    calibration["previous_high_score_threshold_before_profile"] = current_threshold
    calibration["high_score_threshold"] = float(selected["threshold"])
    calibration["high_score_threshold_source"] = selected["threshold_source"]
    calibration["high_score_threshold_profile_config_hash"] = canonical_json_sha256(
        {
            "profile_name": profile_name,
            "current_threshold": current_threshold,
            "candidate_rows": candidate_rows,
            "source_model_gate_min_high_score_support_count": (
                O_MIN_HIGH_SCORE_SUPPORT_COUNT
            ),
            "uses_validation_labels_for_threshold_tuning": False,
        }
    )
    ranking_correction["correction_config_hash"] = canonical_json_sha256(
        _without_key(ranking_correction, "correction_config_hash")
    )
    return selected


def _selected_score_return_rows(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    split: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split:
            groups[str(row["decision_group_id"])].append(row)
    selected_rows = []
    for group_rows in groups.values():
        selected = max(
            group_rows,
            key=lambda row: float(row["variant_scores"][variant]),
        )
        selected_rows.append(
            {
                "decision_group_id": selected["decision_group_id"],
                "market_id": selected.get("market_id"),
                "decision_ts": selected.get("decision_ts"),
                "action": selected.get("action"),
                "score": float(selected["variant_scores"][variant]),
                "realized_return": float(selected["realized_replay_return"]),
            }
        )
    return selected_rows


def _shadow_slightly_lower_high_score_threshold(
    *,
    current_threshold: float,
    shadow_selected_rows: list[dict[str, Any]],
    p_edge_q25: float,
) -> float:
    if not shadow_selected_rows:
        return current_threshold
    score_values = sorted(float(row["score"]) for row in shadow_selected_rows)
    score_span = max(score_values) - min(score_values)
    decrement = max(0.01, min(abs(p_edge_q25), score_span / 10.0))
    return current_threshold - decrement


def _shadow_support_preserving_high_score_threshold(
    *,
    current_threshold: float,
    shadow_selected_rows: list[dict[str, Any]],
) -> float:
    if not shadow_selected_rows:
        return current_threshold
    current_support = sum(
        1 for row in shadow_selected_rows if float(row["score"]) >= current_threshold
    )
    if current_support >= O_MIN_HIGH_SCORE_SUPPORT_COUNT:
        return current_threshold
    sorted_scores = sorted(
        (float(row["score"]) for row in shadow_selected_rows),
        reverse=True,
    )
    if len(sorted_scores) < O_MIN_HIGH_SCORE_SUPPORT_COUNT:
        return min(sorted_scores)
    return sorted_scores[O_MIN_HIGH_SCORE_SUPPORT_COUNT - 1]


def _shadow_high_score_profitability_threshold(
    *,
    ranking_rows: list[dict[str, Any]],
    current_threshold: float,
    fallback_threshold: float,
    extra_thresholds: tuple[float, ...],
) -> float:
    shadow_selected_rows = _selected_score_return_rows(
        ranking_rows,
        variant=O_MODEL_PREDICTED_VARIANT,
        split="shadow",
    )
    candidate_thresholds = {
        current_threshold,
        fallback_threshold,
        *extra_thresholds,
        *(float(row["score"]) for row in shadow_selected_rows),
    }
    candidate_rows = [
        _high_score_threshold_profile_candidate_row(
            ranking_rows=ranking_rows,
            profile_name="high_score_profitability_threshold_candidate",
            threshold=threshold,
        )
        for threshold in candidate_thresholds
    ]
    eligible = [
        row
        for row in candidate_rows
        if int(row["shadow_high_score_support_count"])
        >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
        and float(row["shadow_high_score_return_mean"]) > 0.0
        and float(row["shadow_high_score_return_sum"]) > 0.0
    ]
    if not eligible:
        return fallback_threshold
    selected = max(
        eligible,
        key=lambda row: (
            float(row["shadow_high_score_return_mean"]),
            float(row["shadow_high_score_return_sum"]),
            int(row["shadow_high_score_support_count"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"])


def _high_score_threshold_profile_candidate_row(
    *,
    ranking_rows: list[dict[str, Any]],
    profile_name: str,
    threshold: float,
) -> dict[str, Any]:
    metrics = _split_metric_views(
        ranking_rows,
        O_MODEL_PREDICTED_VARIANT,
        float(threshold),
    )["train_shadow"]
    return {
        "profile_name": profile_name,
        "threshold": float(threshold),
        "threshold_source": f"{profile_name}_shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "shadow_high_score_support_count": metrics["high_score_support_count"],
        "shadow_high_score_support_deficit_to_source_gate": max(
            0,
            O_MIN_HIGH_SCORE_SUPPORT_COUNT
            - int(metrics["high_score_support_count"]),
        ),
        "shadow_high_score_return_mean": metrics[
            "high_score_realized_return_mean"
        ],
        "shadow_high_score_return_sum": metrics[
            "high_score_realized_return_sum"
        ],
        "shadow_selected_return_sum": metrics[
            "selected_action_realized_replay_return_sum"
        ],
        "shadow_mean_regret": metrics["mean_regret"],
        "shadow_support_gate_passed": (
            int(metrics["high_score_support_count"])
            >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "shadow_high_score_return_positive": (
            float(metrics["high_score_realized_return_mean"]) > 0.0
            and float(metrics["high_score_realized_return_sum"]) > 0.0
        ),
    }


def _apply_o_correction_policy_profile(
    ranking_correction: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    config = dict(ranking_correction)
    config["trade_base_score"] += float(profile["trade_base_delta"])
    config["sell_before_close_base_score"] += float(
        profile["sell_before_close_base_delta"]
    )
    config["no_trade_base_score"] += float(profile["no_trade_base_delta"])
    config["group_normalized_raw_model_weight"] = _bounded(
        float(config["group_normalized_raw_model_weight"])
        * float(profile["raw_weight_multiplier"]),
        0.0,
        1.0,
    )
    config["p_up_misalignment_raw_positive_penalty"] = _bounded(
        float(config["p_up_misalignment_raw_positive_penalty"])
        * float(profile["p_up_misalignment_penalty_multiplier"]),
        0.0,
        1.0,
    )
    config["large_regret_reversal_penalty"] = _bounded(
        float(config["large_regret_reversal_penalty"])
        * float(profile["large_regret_reversal_penalty_multiplier"]),
        0.0,
        1.5,
    )
    config["hts_p_up_reliability_penalty"] = _bounded(
        float(config["hts_p_up_reliability_penalty"])
        * float(profile["hts_reliability_penalty_multiplier"]),
        0.0,
        1.0,
    )
    if float(profile["no_trade_tail_buffer_multiplier"]) > 0.0:
        config["hts_p_up_reliability_no_trade_buffer_enabled"] = True
        config["hts_p_up_reliability_no_trade_buffer"] = (
            float(config["hts_p_up_reliability_no_trade_buffer"])
            * float(profile["no_trade_tail_buffer_multiplier"])
        )
        config["hts_p_up_reliability_no_trade_buffer_multiplier"] = float(
            profile["no_trade_tail_buffer_multiplier"]
        )
        config["hts_p_up_reliability_no_trade_buffer_multiplier_source"] = (
            "shadow_split_only_config_hashed_correction_policy_profile"
        )
    config["joint_correction_policy_name"] = profile["correction_policy_name"]
    config["joint_correction_policy_family"] = profile["correction_policy_family"]
    config["joint_correction_policy_profile"] = dict(profile)
    config["joint_correction_policy_source"] = "shadow_split_only_config_hashed_profile"
    config["joint_correction_policy_uses_validation_labels_for_tuning"] = False
    _apply_large_regret_adjusted_high_score_calibration(config)
    config["correction_config_hash"] = canonical_json_sha256(
        _without_key(config, "correction_config_hash")
    )
    return config


def _joint_candidate_name(
    feature_set_name: str,
    correction_policy_profile: dict[str, Any],
    high_score_threshold_profile_name: str,
) -> str:
    return (
        f"{feature_set_name}__"
        f"{correction_policy_profile['correction_policy_name']}__"
        f"{high_score_threshold_profile_name}"
    )


def _joint_feature_correction_candidate_row(
    *,
    feature_set_name: str,
    correction_policy_profile: dict[str, Any],
    high_score_threshold_profile_name: str,
    scoring: dict[str, Any],
    baseline_largest_regret: float,
) -> dict[str, Any]:
    row = _feature_set_selection_candidate_row(
        name=feature_set_name,
        scoring=scoring,
        baseline_largest_regret=baseline_largest_regret,
    )
    row["joint_candidate_name"] = _joint_candidate_name(
        feature_set_name,
        correction_policy_profile,
        high_score_threshold_profile_name,
    )
    row["correction_policy_name"] = correction_policy_profile[
        "correction_policy_name"
    ]
    row["correction_policy_family"] = correction_policy_profile[
        "correction_policy_family"
    ]
    row["correction_policy_profile"] = dict(correction_policy_profile)
    row["correction_policy_config_hash"] = scoring["ranking_correction"][
        "correction_config_hash"
    ]
    row["high_score_threshold_profile_name"] = high_score_threshold_profile_name
    row["high_score_threshold_profile"] = scoring["high_score_threshold_profile"]
    row["high_score_threshold"] = scoring["high_score_threshold_profile"][
        "threshold"
    ]
    row["joint_candidate_config_hash"] = canonical_json_sha256(
        {
            "feature_set_name": feature_set_name,
            "feature_names": row["feature_names"],
            "correction_policy_profile": correction_policy_profile,
            "high_score_threshold_profile_name": high_score_threshold_profile_name,
            "high_score_threshold_profile": row["high_score_threshold_profile"],
            "correction_policy_config_hash": row["correction_policy_config_hash"],
        }
    )
    reason_codes = list(row["shadow_selection_reason_codes"])
    shadow_metrics = row["shadow_selection_metrics"]
    support_deficit = max(
        0,
        O_MIN_HIGH_SCORE_SUPPORT_COUNT
        - int(shadow_metrics["high_score_support_count"]),
    )
    row["shadow_source_gate_high_score_support_passed"] = support_deficit == 0
    row["shadow_high_score_support_deficit_to_source_gate"] = support_deficit
    if support_deficit:
        reason_codes.append("shadow_high_score_support_below_source_gate_threshold")
    if not _action_family_selected_returns_not_negative(shadow_metrics):
        reason_codes.append("shadow_action_family_selected_return_negative")
    row["shadow_action_family_selected_returns_not_negative"] = (
        _action_family_selected_returns_not_negative(shadow_metrics)
    )
    row["shadow_p_up_safety_buffer"] = max(
        0.0,
        O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
        - float(shadow_metrics["p_up_disagreement_rate"]),
    )
    row["shadow_p_up_safety_target_rate"] = O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    row["shadow_p_up_safety_target_passed"] = (
        float(shadow_metrics["p_up_disagreement_rate"])
        <= O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    row["shadow_p_up_near_hard_gate"] = (
        float(shadow_metrics["p_up_disagreement_rate"])
        > O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    row["shadow_top1_quality_target_passed"] = (
        float(shadow_metrics["top1_hit_rate"]) >= O_MIN_TOP1_HIT_RATE
    )
    row["shadow_top1_miss_regret_sum"] = float(
        shadow_metrics.get("top1_miss_regret_sum") or 0.0
    )
    row["shadow_positive_regret_sum"] = float(
        shadow_metrics.get("positive_regret_sum") or 0.0
    )
    row["shadow_no_trade_missed_positive_opportunity_sum"] = float(
        shadow_metrics.get("no_trade_missed_positive_opportunity_sum") or 0.0
    )
    if not bool(row["shadow_p_up_safety_target_passed"]):
        reason_codes.append("shadow_p_up_safety_buffer_below_target")
    row["shadow_selection_gate_passed"] = not reason_codes
    row["shadow_selection_reason_codes"] = sorted(set(reason_codes))
    return row


def _action_family_selected_returns_not_negative(
    metrics: dict[str, Any],
) -> bool:
    family_returns = metrics.get("action_family_selected_return_sum", {})
    return all(float(value) >= 0.0 for value in family_returns.values())


def _joint_selection_sort_key(
    row: dict[str, Any],
) -> tuple[float, ...]:
    metrics = row["shadow_selection_metrics"]
    support_deficit = int(row["shadow_high_score_support_deficit_to_source_gate"])
    return (
        float(support_deficit),
        float(not bool(row["shadow_p_up_safety_target_passed"])),
        -float(metrics["top1_hit_rate"]),
        float(metrics["mean_regret"]),
        float(metrics["top1_miss_regret_sum"]),
        float(metrics["largest_regret_value"]),
        float(metrics["positive_regret_sum"]),
        float(metrics["no_trade_missed_positive_opportunity_sum"]),
        float(_joint_feature_set_priority(row)),
        -float(metrics["selected_return_sum"]),
        -float(metrics["high_score_return_mean"]),
        -float(metrics["high_score_return_sum"]),
        -float(metrics["high_score_support_count"]),
        float(metrics["p_up_disagreement_rate"]),
        float(_joint_correction_policy_priority(row)),
    )


def _joint_feature_set_priority(row: dict[str, Any]) -> int:
    feature_set_name = str(row.get("feature_set_name") or "")
    priority = {
        "combined_features": 0,
        "book_pressure_features": 1,
        "combined_minus_reference_distance": 2,
        "reference_price_features": 3,
        "old_features_only": 4,
    }
    return priority.get(feature_set_name, 99)


def _joint_correction_policy_priority(row: dict[str, Any]) -> int:
    policy_name = str(row.get("correction_policy_name") or "")
    priority = {
        "sbc_preferred_when_hts_reliability_weak": 0,
        "high_score_profitability_preserving": 1,
        "balanced_hts_sbc": 2,
        "no_trade_tail_risk_buffer": 3,
        "conservative_hts": 4,
    }
    return priority.get(policy_name, 99)


def _selected_full_correction_diagnostics(
    *,
    lightweight_row: dict[str, Any],
    final_row: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    lightweight_shadow = lightweight_row["shadow_selection_metrics"]
    final_shadow = final_row["shadow_selection_metrics"]
    lightweight_validation = lightweight_row["validation_metrics_report_only"]
    final_validation = final_row["validation_metrics_report_only"]
    return {
        "full_correction_rerun_enabled": True,
        "full_correction_search_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "accepted_for_final_scoring": acceptance["accepted_for_final_scoring"],
        "acceptance_reason_codes": acceptance["reason_codes"],
        "shadow_top1_quality_acceptance_path": acceptance[
            "shadow_top1_quality_acceptance_path"
        ],
        "shadow_p_up_safety_target_rate": O_SHADOW_P_UP_SELECTION_BUFFER_TARGET,
        "final_scoring_source": (
            "full_shadow_correction_search"
            if bool(acceptance["accepted_for_final_scoring"])
            else "lightweight_preselection_shadow_ranker"
        ),
        "lightweight_preselection_joint_candidate_name": lightweight_row[
            "joint_candidate_name"
        ],
        "final_full_correction_joint_candidate_name": final_row[
            "joint_candidate_name"
        ],
        "shadow_mean_regret_delta_final_minus_lightweight": (
            float(final_shadow["mean_regret"])
            - float(lightweight_shadow["mean_regret"])
        ),
        "shadow_high_score_support_delta_final_minus_lightweight": (
            int(final_shadow["high_score_support_count"])
            - int(lightweight_shadow["high_score_support_count"])
        ),
        "shadow_high_score_return_sum_delta_final_minus_lightweight": (
            float(final_shadow["high_score_return_sum"])
            - float(lightweight_shadow["high_score_return_sum"])
        ),
        "validation_mean_regret_delta_final_minus_lightweight_report_only": (
            float(final_validation["mean_regret"])
            - float(lightweight_validation["mean_regret"])
        ),
        "validation_high_score_support_delta_final_minus_lightweight_report_only": (
            int(final_validation["high_score_support_count"])
            - int(lightweight_validation["high_score_support_count"])
        ),
        "validation_high_score_return_sum_delta_final_minus_lightweight_report_only": (
            float(final_validation["high_score_return_sum"])
            - float(lightweight_validation["high_score_return_sum"])
        ),
    }


def _selected_full_correction_acceptance(
    *,
    lightweight_row: dict[str, Any],
    final_row: dict[str, Any],
) -> dict[str, Any]:
    lightweight_shadow = lightweight_row["shadow_selection_metrics"]
    final_shadow = final_row["shadow_selection_metrics"]
    reason_codes = []
    final_shadow_high_score_support_ok = (
        int(final_shadow["high_score_support_count"])
        >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
    )
    final_shadow_high_score_positive = (
        float(final_shadow["high_score_return_mean"]) > 0.0
        and float(final_shadow["high_score_return_sum"]) > 0.0
    )
    final_shadow_p_up_target_passed = (
        float(final_shadow["p_up_disagreement_rate"])
        <= O_SHADOW_P_UP_SELECTION_BUFFER_TARGET
    )
    final_shadow_top1_quality_passed = (
        float(final_shadow["top1_hit_rate"]) >= O_MIN_TOP1_HIT_RATE
    )
    shadow_top1_quality_acceptance_path = all(
        (
            final_shadow_high_score_support_ok,
            final_shadow_high_score_positive,
            final_shadow_p_up_target_passed,
            final_shadow_top1_quality_passed,
        )
    )
    if (
        int(final_shadow["high_score_support_count"])
        < int(lightweight_shadow["high_score_support_count"])
        and not final_shadow_high_score_support_ok
    ):
        reason_codes.append("full_correction_reduced_shadow_high_score_support")
    if float(final_shadow["high_score_return_mean"]) <= 0.0:
        reason_codes.append("full_correction_shadow_high_score_mean_not_positive")
    if float(final_shadow["high_score_return_sum"]) <= 0.0:
        reason_codes.append("full_correction_shadow_high_score_sum_not_positive")
    if not bool(final_shadow["p_up_hard_gate_passed"]):
        reason_codes.append("full_correction_shadow_p_up_gate_failed")
    if not final_shadow_p_up_target_passed:
        reason_codes.append("full_correction_shadow_p_up_safety_target_failed")
    if (
        not _action_family_selected_returns_not_negative(final_shadow)
        and not shadow_top1_quality_acceptance_path
    ):
        reason_codes.append("full_correction_shadow_action_family_return_negative")
    return {
        "accepted_for_final_scoring": not reason_codes,
        "reason_codes": sorted(reason_codes),
        "uses_validation_labels_for_tuning": False,
        "acceptance_metric_source": "shadow_split_only",
        "shadow_top1_quality_acceptance_path": (
            shadow_top1_quality_acceptance_path
        ),
        "requires_shadow_high_score_support_not_reduced": True,
        "requires_shadow_high_score_profitability_positive": True,
        "requires_shadow_p_up_hard_gate_passed": True,
        "requires_shadow_p_up_safety_target_passed": True,
        "requires_shadow_top1_quality_or_action_family_returns": True,
        "requires_shadow_action_family_returns_non_negative": True,
    }


def _feature_set_candidate_summary_rows(
    *,
    joint_candidate_rows: list[dict[str, Any]],
    selected_feature_set_name: str,
) -> list[dict[str, Any]]:
    rows_by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joint_candidate_rows:
        rows_by_feature[row["feature_set_name"]].append(row)
    summary_rows = []
    for feature_set_name, rows in sorted(rows_by_feature.items()):
        eligible = [row for row in rows if bool(row["shadow_selection_gate_passed"])]
        best = min(eligible or rows, key=_joint_selection_sort_key)
        payload = {
            key: value
            for key, value in best.items()
            if key
            not in {
                "joint_candidate_name",
                "correction_policy_name",
                "correction_policy_family",
                "correction_policy_profile",
                "correction_policy_config_hash",
            }
        }
        payload["feature_set_name"] = feature_set_name
        payload["best_joint_candidate_name"] = best["joint_candidate_name"]
        payload["best_correction_policy_name"] = best["correction_policy_name"]
        payload["best_correction_policy_family"] = best["correction_policy_family"]
        payload["eligible_joint_candidate_count"] = len(eligible)
        payload["joint_candidate_count"] = len(rows)
        payload["selected_feature_set"] = feature_set_name == selected_feature_set_name
        if not payload["selected_feature_set"]:
            payload["shadow_selection_reason_codes"] = sorted(
                set(
                    {
                        *payload["shadow_selection_reason_codes"],
                        "not_selected_by_shadow_joint_feature_set_selection",
                    }
                )
            )
        return_row = dict(payload)
        summary_rows.append(return_row)
    return summary_rows


def _feature_set_selection_metric_summary(
    *,
    metrics: dict[str, Any],
    p_up_summary: dict[str, Any],
) -> dict[str, Any]:
    largest_regret_value = _largest_regret_value(metrics)
    top1_miss = metrics.get("top1_miss_diagnostics") or {}
    regret_diagnostics = metrics.get("mean_regret_reduction_diagnostics") or {}
    no_trade_contribution = regret_diagnostics.get(
        "no_trade_missed_opportunity_contribution",
        {},
    )
    return {
        "decision_group_count": metrics["decision_group_count"],
        "top1_hit_rate": metrics["top1_realized_best_action_hit_rate"],
        "top2_hit_rate": metrics["top2_realized_best_action_hit_rate"],
        "top3_hit_rate": metrics["top3_realized_best_action_hit_rate"],
        "mean_regret": metrics["mean_regret"],
        "selected_return_sum": metrics[
            "selected_action_realized_replay_return_sum"
        ],
        "high_score_support_count": metrics["high_score_support_count"],
        "high_score_return_mean": metrics["high_score_realized_return_mean"],
        "high_score_return_sum": metrics["high_score_realized_return_sum"],
        "largest_regret_value": largest_regret_value,
        "top1_miss_rate": top1_miss.get("top1_miss_rate", 0.0),
        "top1_miss_regret_sum": top1_miss.get("top1_miss_regret_sum", 0.0),
        "top2_contained_miss_rate": top1_miss.get(
            "top2_contained_miss_rate",
            0.0,
        ),
        "top3_contained_miss_rate": top1_miss.get(
            "top3_contained_miss_rate",
            0.0,
        ),
        "p_up_conflicting_top1_miss_rate": top1_miss.get(
            "p_up_conflicting_top1_miss_rate",
            0.0,
        ),
        "positive_regret_sum": regret_diagnostics.get(
            "positive_regret_sum",
            0.0,
        ),
        "no_trade_missed_positive_opportunity_sum": no_trade_contribution.get(
            "missed_positive_opportunity_sum",
            0.0,
        ),
        "action_family_selected_return_sum": {
            family: family_metrics["selected_return_sum"]
            for family, family_metrics in metrics[
                "action_family_selected_return_breakdown"
            ].items()
        },
        "p_up_comparable_count": p_up_summary[
            "candidate_scoped_p_up_action_comparable_count"
        ],
        "p_up_disagreement_rate": p_up_summary[
            "candidate_scoped_p_up_action_disagreement_rate"
        ],
        "p_up_hard_gate_passed": p_up_summary[
            "candidate_scoped_p_up_action_disagreement_within_limit"
        ],
    }


def _gate_preservation_diagnostics_from_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    action_family_returns = summary.get("action_family_selected_return_sum", {})
    return {
        "p_up_disagreement_within_hard_gate": (
            float(summary["p_up_disagreement_rate"])
            <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
        ),
        "top1_hit_rate_gate_passed": (
            float(summary["top1_hit_rate"]) >= O_MIN_TOP1_HIT_RATE
        ),
        "high_score_support_gate_passed": (
            int(summary["high_score_support_count"])
            >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "high_score_return_mean_positive": (
            float(summary["high_score_return_mean"]) > 0.0
        ),
        "high_score_return_sum_positive": (
            float(summary["high_score_return_sum"]) > 0.0
        ),
        "selected_return_sum_positive": (
            float(summary["selected_return_sum"]) > 0.0
        ),
        "action_family_returns_non_negative": all(
            float(value) >= 0.0 for value in action_family_returns.values()
        ),
        "all_currently_passing_gates_preserved": all(
            (
                float(summary["p_up_disagreement_rate"])
                <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE,
                float(summary["top1_hit_rate"]) >= O_MIN_TOP1_HIT_RATE,
                int(summary["high_score_support_count"])
                >= O_MIN_HIGH_SCORE_SUPPORT_COUNT,
                float(summary["high_score_return_mean"]) > 0.0,
                float(summary["high_score_return_sum"]) > 0.0,
                float(summary["selected_return_sum"]) > 0.0,
                all(float(value) >= 0.0 for value in action_family_returns.values()),
            )
        ),
        "diagnostic_only": True,
    }


def _mean_regret_gate_tradeoff_diagnostics(
    *,
    candidate_rows: list[dict[str, Any]],
    selected_effective_row: dict[str, Any],
) -> dict[str, Any]:
    selected_shadow = selected_effective_row["shadow_selection_metrics"]
    selected_validation = selected_effective_row["validation_metrics_report_only"]
    selected_shadow_mean_regret = float(selected_shadow["mean_regret"])
    selected_validation_mean_regret = float(selected_validation["mean_regret"])
    lower_shadow_rows = [
        row
        for row in candidate_rows
        if float(row["shadow_selection_metrics"]["mean_regret"])
        < selected_shadow_mean_regret
    ]
    lower_shadow_gate_passing_rows = [
        row for row in lower_shadow_rows if bool(row["shadow_selection_gate_passed"])
    ]
    lower_shadow_blocker_counts: Counter[str] = Counter()
    for row in lower_shadow_rows:
        if bool(row["shadow_selection_gate_passed"]):
            continue
        reason_codes = row.get("shadow_selection_reason_codes") or []
        if not reason_codes:
            lower_shadow_blocker_counts["shadow_gate_failed_without_reason"] += 1
        else:
            lower_shadow_blocker_counts.update(str(reason) for reason in reason_codes)

    lower_validation_rows = [
        row
        for row in candidate_rows
        if float(row["validation_metrics_report_only"]["mean_regret"])
        < selected_validation_mean_regret
    ]
    lower_validation_gate_preserving_rows = [
        row
        for row in lower_validation_rows
        if _validation_summary_preserves_current_o_gates(
            row["validation_metrics_report_only"]
        )
    ]
    if lower_shadow_gate_passing_rows:
        conclusion = "shadow_gate_passing_lower_mean_regret_candidates_exist"
    elif lower_shadow_rows:
        conclusion = "lower_shadow_mean_regret_candidates_break_shadow_gates"
    elif lower_validation_gate_preserving_rows:
        conclusion = (
            "validation_report_only_lower_mean_regret_candidates_exist_but_not_"
            "shadow_selected"
        )
    else:
        conclusion = "no_gate_preserving_lower_mean_regret_candidate_found"
    return {
        "diagnostic_only": True,
        "selection_metric_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "validation_metrics_report_only": True,
        "selected_joint_candidate_name": selected_effective_row["joint_candidate_name"],
        "selected_shadow_mean_regret": selected_shadow_mean_regret,
        "selected_validation_mean_regret_report_only": selected_validation_mean_regret,
        "lower_shadow_mean_regret_candidate_count": len(lower_shadow_rows),
        "lower_shadow_gate_passing_candidate_count": len(
            lower_shadow_gate_passing_rows
        ),
        "lower_shadow_blocker_reason_counts": dict(
            sorted(lower_shadow_blocker_counts.items())
        ),
        "best_lower_shadow_mean_regret_candidates": [
            _mean_regret_tradeoff_candidate_summary(row)
            for row in sorted(
                lower_shadow_rows,
                key=lambda row: float(row["shadow_selection_metrics"]["mean_regret"]),
            )[:5]
        ],
        "lower_validation_mean_regret_report_only_candidate_count": len(
            lower_validation_rows
        ),
        "lower_validation_gate_preserving_report_only_candidate_count": len(
            lower_validation_gate_preserving_rows
        ),
        "best_lower_validation_mean_regret_report_only_candidates": [
            _mean_regret_tradeoff_candidate_summary(row)
            for row in sorted(
                lower_validation_rows,
                key=lambda row: float(
                    row["validation_metrics_report_only"]["mean_regret"]
                ),
            )[:5]
        ],
        "mean_regret_not_improved_without_breaking_shadow_gate": (
            len(lower_shadow_gate_passing_rows) == 0
        ),
        "tradeoff_conclusion": conclusion,
    }


def _validation_summary_preserves_current_o_gates(summary: dict[str, Any]) -> bool:
    action_family_returns = summary.get("action_family_selected_return_sum", {})
    return all(
        (
            float(summary["p_up_disagreement_rate"])
            <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE,
            float(summary["top1_hit_rate"]) >= O_MIN_TOP1_HIT_RATE,
            int(summary["high_score_support_count"])
            >= O_MIN_HIGH_SCORE_SUPPORT_COUNT,
            float(summary["high_score_return_mean"]) > 0.0,
            float(summary["high_score_return_sum"]) > 0.0,
            float(summary["selected_return_sum"]) > 0.0,
            all(float(value) >= 0.0 for value in action_family_returns.values()),
        )
    )


def _mean_regret_tradeoff_candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    shadow = row["shadow_selection_metrics"]
    validation = row["validation_metrics_report_only"]
    return {
        "joint_candidate_name": row["joint_candidate_name"],
        "feature_set_name": row["feature_set_name"],
        "correction_policy_name": row["correction_policy_name"],
        "high_score_threshold_profile_name": row[
            "high_score_threshold_profile_name"
        ],
        "shadow_mean_regret": shadow["mean_regret"],
        "shadow_top1_hit_rate": shadow["top1_hit_rate"],
        "shadow_p_up_disagreement_rate": shadow["p_up_disagreement_rate"],
        "shadow_high_score_support_count": shadow["high_score_support_count"],
        "shadow_high_score_return_sum": shadow["high_score_return_sum"],
        "shadow_selected_return_sum": shadow["selected_return_sum"],
        "validation_mean_regret_report_only": validation["mean_regret"],
        "validation_top1_hit_rate_report_only": validation["top1_hit_rate"],
        "validation_p_up_disagreement_rate_report_only": validation[
            "p_up_disagreement_rate"
        ],
        "validation_high_score_support_count_report_only": validation[
            "high_score_support_count"
        ],
        "validation_high_score_return_sum_report_only": validation[
            "high_score_return_sum"
        ],
        "validation_selected_return_sum_report_only": validation[
            "selected_return_sum"
        ],
        "shadow_selection_gate_passed": row["shadow_selection_gate_passed"],
        "shadow_selection_reason_codes": row["shadow_selection_reason_codes"],
    }


def _largest_regret_value(metrics: dict[str, Any]) -> float:
    return float((metrics.get("largest_regret_case") or {}).get("regret") or 0.0)


def _o_feature_set_selection_feature_sets() -> dict[str, tuple[str, ...]]:
    reference_names = _o_reference_feature_names()
    book_names = _o_book_pressure_feature_names()
    return {
        "old_features_only": O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES,
        "book_pressure_features": tuple(
            dict.fromkeys((*O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES, *book_names))
        ),
        "reference_price_features": tuple(
            dict.fromkeys((*O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES, *reference_names))
        ),
        "combined_features": O_DEPLOYABLE_MODEL_FEATURE_NAMES,
        "combined_minus_reference_distance": tuple(
            feature
            for feature in O_DEPLOYABLE_MODEL_FEATURE_NAMES
            if not _is_reference_distance_feature_name(feature)
        ),
    }


def _o_reference_feature_names() -> tuple[str, ...]:
    return tuple(
        name
        for name in O_DEPLOYABLE_MODEL_FEATURE_NAMES
        if name in O_REFERENCE_PRICE_MODEL_FEATURE_NAMES
        or "reference_" in name
    )


def _o_book_pressure_feature_names() -> tuple[str, ...]:
    return tuple(
        name
        for name in O_DEPLOYABLE_MODEL_FEATURE_NAMES
        if name in O_BOOK_PRESSURE_MODEL_FEATURE_NAMES
        or "book_" in name
        or "side_book_" in name
        or "side_queue" in name
        or "side_spread" in name
        or "hts_vs_sell_before_close" in name
        or "p_up_bucket_calibration_residual" in name
    )


def _is_reference_distance_feature_name(name: str) -> bool:
    return (
        name == "reference_price_to_beat_distance_scaled"
        or name.endswith("_x_reference_price_to_beat_distance")
    )


def _o_feature_ablation_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_sets = _o_feature_ablation_feature_sets()
    diagnostics = {}
    for name, feature_names in feature_sets.items():
        diagnostics[name] = _raw_model_feature_set_diagnostic(
            rows=rows,
            feature_names=feature_names,
        )
    combined = diagnostics["combined_feature_set"]["validation_metrics"]
    old = diagnostics["old_features_only"]["validation_metrics"]
    return {
        "diagnostic_only": True,
        "uses_validation_labels_for_tuning": False,
        "ablation_score_source": "raw_ridge_model_score_by_feature_subset",
        "feature_sets": diagnostics,
        "validation_mean_regret_delta_combined_vs_old": (
            float(old["mean_regret"]) - float(combined["mean_regret"])
        ),
        "validation_selected_return_delta_combined_vs_old": (
            float(combined["selected_action_realized_replay_return_sum"])
            - float(old["selected_action_realized_replay_return_sum"])
        ),
    }


def _reference_price_feature_effect_summary(
    *,
    feature_coverage: dict[str, Any],
    feature_ablation: dict[str, Any],
) -> dict[str, Any]:
    field = feature_coverage["field_coverage"][
        "reference_price_to_beat_distance_at_decision"
    ]
    feature_sets = feature_ablation["feature_sets"]
    old_validation = feature_sets["old_features_only"]["validation_metrics"]
    reference_validation = feature_sets["new_reference_price_features"][
        "validation_metrics"
    ]
    combined_validation = feature_sets["combined_feature_set"]["validation_metrics"]
    reference_mean_regret_delta = float(old_validation["mean_regret"]) - float(
        reference_validation["mean_regret"]
    )
    combined_mean_regret_delta = float(old_validation["mean_regret"]) - float(
        combined_validation["mean_regret"]
    )
    return {
        "diagnostic_only": True,
        "uses_validation_labels_for_tuning": False,
        "reference_price_to_beat_distance_available_count": field["available_count"],
        "reference_price_to_beat_distance_missing_count": field["missing_count"],
        "reference_price_to_beat_distance_availability_rate": field[
            "availability_rate"
        ],
        "reference_price_to_beat_distance_used_as_model_input": field[
            "used_as_model_input"
        ],
        "reference_feature_validation_mean_regret_delta_vs_old_features": (
            reference_mean_regret_delta
        ),
        "combined_feature_validation_mean_regret_delta_vs_old_features": (
            combined_mean_regret_delta
        ),
        "reference_feature_validation_selected_return_delta_vs_old_features": (
            float(reference_validation["selected_action_realized_replay_return_sum"])
            - float(old_validation["selected_action_realized_replay_return_sum"])
        ),
        "combined_feature_validation_selected_return_delta_vs_old_features": (
            float(combined_validation["selected_action_realized_replay_return_sum"])
            - float(old_validation["selected_action_realized_replay_return_sum"])
        ),
        "reference_distance_has_raw_model_signal": (
            reference_mean_regret_delta > 0.0
            or float(
                reference_validation["selected_action_realized_replay_return_sum"]
            )
            > float(old_validation["selected_action_realized_replay_return_sum"])
        ),
        "combined_expanded_features_have_raw_model_signal": (
            combined_mean_regret_delta > 0.0
            or float(combined_validation["selected_action_realized_replay_return_sum"])
            > float(old_validation["selected_action_realized_replay_return_sum"])
        ),
        "final_shadow_corrected_gate_remains_fail_closed": True,
        "final_shadow_corrected_blocker_interpretation": (
            "reference distance is now decision-time covered; remaining O blocker is "
            "validation calibration quality under the shadow-derived corrected ranker, "
            "not missing reference-distance coverage"
        ),
    }


def _o_feature_ablation_feature_sets() -> dict[str, tuple[str, ...]]:
    reference_names = _o_reference_feature_names()
    book_names = _o_book_pressure_feature_names()
    return {
        "old_features_only": O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES,
        "new_reference_price_features": tuple(
            dict.fromkeys((*O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES, *reference_names))
        ),
        "new_book_pressure_features": tuple(
            dict.fromkeys((*O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES, *book_names))
        ),
        "combined_feature_set": O_DEPLOYABLE_MODEL_FEATURE_NAMES,
    }


def _raw_model_feature_set_diagnostic(
    *,
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    train_rows = [
        row
        for row in rows
        if row.get("split") == "shadow"
        and bool(row.get("label_candidate_available", True))
    ]
    deployable = bool(train_rows) and _full_grid_available(rows)
    if deployable:
        model = _fit_ridge_regression(
            [_deployable_model_features(row, feature_names) for row in train_rows],
            [
                float(row["replay_aligned_executable_label_target"])
                for row in train_rows
            ],
        )
        predictions = [
            _dot(model["coefficients"], _deployable_model_features(row, feature_names))
            for row in rows
        ]
    else:
        predictions = [0.0 for _ in rows]
    scored_rows = [
        {**row, "o_model_predicted_score": score}
        for row, score in zip(rows, predictions, strict=True)
    ]
    ranking_rows = _ranking_rows(scored_rows)
    split_metrics = _split_metric_views(
        ranking_rows,
        O_MODEL_PREDICTED_VARIANT,
        high_score_threshold=0.75,
    )
    return {
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "deployable_model_score_available": deployable,
        "training_split_source": "shadow_split_only",
        "uses_validation_labels_for_tuning": False,
        "ranking_correction_applied": False,
        "train_shadow_metrics": split_metrics["train_shadow"],
        "validation_metrics": split_metrics["validation"],
        "all_metrics": split_metrics["all"],
    }


def _deployable_model_features(
    row: dict[str, Any],
    feature_names: tuple[str, ...] = O_DEPLOYABLE_MODEL_FEATURE_NAMES,
) -> list[float]:
    feature_map = _deployable_model_feature_map(row)
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def _deployable_model_feature_map(row: dict[str, Any]) -> dict[str, float]:
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
    reference_distance = _scaled_reference_distance(row)
    momentum_30s = _scaled_return(row.get("recent_reference_price_momentum_30s"))
    momentum_60s = _scaled_return(row.get("recent_reference_price_momentum_60s"))
    momentum_120s = _scaled_return(row.get("recent_reference_price_momentum_120s"))
    depth_imbalance = _bounded(
        float(row.get("side_book_depth_imbalance") or 0.0),
        -1.0,
        1.0,
    )
    update_velocity = _bounded(
        float(row.get("side_book_update_velocity") or 0.0),
        0.0,
        2.0,
    )
    side_staleness = _bounded(
        float(row.get("side_book_staleness_ms") or 0.0) / 1000.0,
        0.0,
        60.0,
    )
    opposite_staleness = _bounded(
        float(row.get("opposite_book_staleness_ms") or 0.0) / 1000.0,
        0.0,
        60.0,
    )
    side_spread = _bounded(
        float(row.get("side_spread_bps") or 0.0) / 10_000.0,
        0.0,
        1.0,
    )
    side_queue = _bounded(
        float(row.get("side_queue_fill_proxy") or 0.0),
        0.0,
        1.0,
    )
    hts_sbc_gap = _bounded(
        float(row.get("hts_vs_sell_before_close_exit_value_gap_proxy") or 0.0),
        -1.0,
        1.0,
    )
    p_up_bucket_residual = _bounded(
        float(row.get("p_up_calibration_residual_by_time_spread_queue_bucket") or 0.0),
        -1.0,
        1.0,
    )
    feature_map = {
        "bias": 1.0,
        "action_buy_up_sell_before_close": _flag(
            action == "BUY_UP_SELL_BEFORE_CLOSE"
        ),
        "action_buy_down_sell_before_close": _flag(
            action == "BUY_DOWN_SELL_BEFORE_CLOSE"
        ),
        "action_buy_up_hold_to_settlement": _flag(
            action == "BUY_UP_HOLD_TO_SETTLEMENT"
        ),
        "action_buy_down_hold_to_settlement": _flag(
            action == "BUY_DOWN_HOLD_TO_SETTLEMENT"
        ),
        "action_no_trade": _flag(action == "NO_TRADE"),
        "side_up": _flag(side == "UP"),
        "side_down": _flag(side == "DOWN"),
        "side_none": _flag(side == "NONE"),
        "family_sell_before_close": _flag(family == "SELL_BEFORE_CLOSE"),
        "family_hold_to_settlement": _flag(family == "HOLD_TO_SETTLEMENT"),
        "family_no_trade": _flag(family == "NO_TRADE"),
        "p_up": p_up,
        "p_down_proxy": p_down,
        "entry_ask": entry_ask,
        "spread_bps_scaled": spread,
        "queue_fill": queue,
        "book_staleness_seconds": staleness,
        "time_to_close_minutes": time_to_close,
        "p_up_edge": p_up_edge,
        "weak_opportunity_proxy": weak_opportunity,
        "reference_price_to_beat_distance_scaled": reference_distance,
        "recent_reference_price_momentum_30s_scaled": momentum_30s,
        "recent_reference_price_momentum_60s_scaled": momentum_60s,
        "recent_reference_price_momentum_120s_scaled": momentum_120s,
        "reference_price_feature_available": _flag(
            bool(row.get("reference_price_feature_available"))
        ),
        "side_book_depth_imbalance": depth_imbalance,
        "side_book_update_velocity_scaled": update_velocity,
        "side_book_staleness_seconds": side_staleness,
        "opposite_book_staleness_seconds": opposite_staleness,
        "side_spread_bps_scaled": side_spread,
        "side_queue_fill_proxy": side_queue,
        "hts_vs_sell_before_close_exit_value_gap_proxy": hts_sbc_gap,
        "p_up_bucket_calibration_residual": p_up_bucket_residual,
        "book_pressure_feature_available": _flag(
            bool(row.get("book_pressure_feature_available"))
        ),
    }
    signals = {
        "p_up": p_up,
        "p_down": p_down,
        "time_to_close": time_to_close,
        "spread": spread,
        "queue": queue,
        "staleness": staleness,
        "entry_ask": entry_ask,
        "exit_bid_proxy": exit_bid_proxy,
        "reference_price_to_beat_distance": reference_distance,
        "reference_momentum_30s": momentum_30s,
        "reference_momentum_60s": momentum_60s,
        "side_book_depth_imbalance": depth_imbalance,
        "side_book_update_velocity": update_velocity,
        "hts_vs_sell_before_close_exit_value_gap_proxy": hts_sbc_gap,
        "p_up_bucket_calibration_residual": p_up_bucket_residual,
    }
    feature_map.update(
        {
            f"{action_slug}_x_{signal_name}": (
                _flag(action == action_name) * signals[signal_name]
            )
            for action_name, action_slug in O_ACTION_FEATURE_SLUGS
            for signal_name in O_ACTION_INTERACTION_SIGNAL_NAMES
        }
    )
    return feature_map


def _scaled_reference_distance(row: dict[str, Any]) -> float:
    value = _optional_float(row.get("reference_price_to_beat_distance_at_decision"))
    if value is None:
        return 0.0
    return _bounded(value * 100.0, -1.0, 1.0)


def _scaled_return(value: Any) -> float:
    numeric = _optional_float(value)
    if numeric is None:
        return 0.0
    return _bounded(numeric * 100.0, -1.0, 1.0)


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
    *,
    run_raw_weight_search: bool = True,
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
    if run_raw_weight_search:
        raw_weight, raw_diagnostics = _derive_shadow_raw_model_weight(
            train_rows=train_rows,
            base_ranking_correction=config,
        )
    else:
        raw_weight, raw_diagnostics = _lightweight_shadow_raw_model_weight(
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


def _lightweight_shadow_raw_model_weight(
    train_rows: list[dict[str, Any]],
    base_ranking_correction: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    p_edge_quantiles = base_ranking_correction["p_up_edge_quantiles"]
    selected = {
        "candidate_weight": _bounded(float(p_edge_quantiles["q75"]), 0.0, 1.0),
        "candidate_weight_source": "shadow_p_up_edge_q75_lightweight_feature_set_selection",
        "candidate_p_up_misalignment_penalty": _bounded(
            float(p_edge_quantiles["q25"]),
            0.0,
            1.0,
        ),
        "candidate_p_up_misalignment_penalty_source": (
            "shadow_p_up_edge_q25_lightweight_feature_set_selection"
        ),
        "candidate_large_regret_reversal_alignment_threshold": _bounded(
            float(p_edge_quantiles["q25"]),
            0.0,
            0.5,
        ),
        "candidate_large_regret_reversal_alignment_threshold_source": (
            "shadow_p_up_edge_q25_lightweight_feature_set_selection"
        ),
        "candidate_large_regret_reversal_penalty": _bounded(
            float(p_edge_quantiles["q75"]),
            0.0,
            1.5,
        ),
        "candidate_large_regret_reversal_penalty_source": (
            "shadow_p_up_edge_q75_lightweight_feature_set_selection"
        ),
        "candidate_hts_p_up_reliability_penalty": _bounded(
            float(p_edge_quantiles["q75"]),
            0.0,
            1.0,
        ),
        "candidate_hts_p_up_reliability_penalty_source": (
            "shadow_p_up_edge_q75_lightweight_feature_set_selection"
        ),
    }
    candidate_config = {
        **base_ranking_correction,
        "group_normalized_raw_model_weight": selected["candidate_weight"],
        "p_up_misalignment_raw_positive_penalty": selected[
            "candidate_p_up_misalignment_penalty"
        ],
        "large_regret_reversal_alignment_threshold": selected[
            "candidate_large_regret_reversal_alignment_threshold"
        ],
        "large_regret_reversal_penalty": selected[
            "candidate_large_regret_reversal_penalty"
        ],
        "hts_p_up_reliability_penalty": selected[
            "candidate_hts_p_up_reliability_penalty"
        ],
    }
    _apply_large_regret_adjusted_high_score_calibration(candidate_config)
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
        float(candidate_config["high_score_calibration"]["high_score_threshold"]),
    )["train_shadow"]
    p_up_summary = _p_up_action_disagreement_summary(
        rows=ranking_rows,
        variant=O_MODEL_PREDICTED_VARIANT,
        split="shadow",
    )
    selected.update(
        {
            "shadow_candidate_eligible": (
                float(metrics["selected_action_realized_replay_return_sum"]) > 0.0
                and int(metrics["high_score_support_count"])
                >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
                and float(metrics["high_score_realized_return_mean"]) > 0.0
                and float(metrics["high_score_realized_return_sum"]) > 0.0
                and bool(
                    p_up_summary[
                        "candidate_scoped_p_up_action_disagreement_within_limit"
                    ]
                )
            ),
            "shadow_high_score_profitable": (
                float(metrics["high_score_realized_return_mean"]) > 0.0
                and float(metrics["high_score_realized_return_sum"]) > 0.0
            ),
            "shadow_p_up_safety_passed": p_up_summary[
                "candidate_scoped_p_up_action_disagreement_within_limit"
            ],
            "shadow_selected_return_sum": metrics[
                "selected_action_realized_replay_return_sum"
            ],
            "shadow_mean_regret": metrics["mean_regret"],
            "shadow_largest_regret_value": _largest_regret_value(metrics),
            "shadow_largest_regret_case": metrics["largest_regret_case"],
            "shadow_action_family_level_regret": metrics[
                "action_family_level_regret"
            ],
            "shadow_action_pair_regret_summary": metrics[
                "action_pair_regret_summary"
            ],
            "shadow_hts_p_up_reliability_regret_summary": metrics[
                "hts_p_up_reliability_regret_summary"
            ],
            "shadow_hold_to_settlement_up_down_reversal_regret": metrics[
                "hold_to_settlement_up_down_reversal_regret"
            ],
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
    return float(selected["candidate_weight"]), {
        "raw_weight_selection_metric_source": "shadow_split_only_lightweight_feature_set_selection",
        "selected_raw_weight_candidate": selected,
        "raw_weight_candidate_rows": [selected],
        "p_up_misalignment_penalty_candidate_source": (
            "shadow_p_up_edge_q25_lightweight_feature_set_selection"
        ),
        "large_regret_reversal_guard_candidate_source": (
            "shadow_p_up_edge_q75_lightweight_feature_set_selection"
        ),
        "large_regret_reversal_guard_selection_metric_source": (
            "shadow_split_only_lightweight_feature_set_selection"
        ),
        "hts_p_up_reliability_guard_candidate_source": (
            "shadow_p_up_edge_q75_lightweight_feature_set_selection"
        ),
        "hts_p_up_reliability_guard_selection_metric_source": (
            "shadow_split_only_lightweight_feature_set_selection"
        ),
        "hts_p_up_reliability_no_trade_buffer_excluded_from_raw_weight_search": True,
        "hts_p_up_reliability_no_trade_buffer_application_stage": (
            "post_lightweight_feature_set_selection_safety_buffer"
        ),
        "hts_p_up_reliability_bucket_thresholds": base_ranking_correction[
            "hts_p_up_reliability_bucket_thresholds"
        ],
        "hts_p_up_reliability_bucket_diagnostics": base_ranking_correction[
            "hts_p_up_reliability_bucket_diagnostics"
        ],
        "large_regret_reversal_pair_regret_priors": base_ranking_correction[
            "large_regret_reversal_pair_regret_priors"
        ],
        "raw_weight_max_shadow_p_up_disagreement_rate": (
            base_ranking_correction["shadow_p_up_selection_max_disagreement_rate"]
        ),
        "raw_weight_p_up_safety_buffer": max(
            0.0,
            float(
                base_ranking_correction["shadow_p_up_selection_max_disagreement_rate"]
            )
            - float(selected["shadow_p_up_action_disagreement_rate"]),
        ),
    }


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
    p_up_safety_target_disagreement_rate = float(
        base_ranking_correction["p_up_safety_target_disagreement_rate"]
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
                            <= O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
                        )
                        p_up_safety_target_passed = (
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
                            <= p_up_safety_target_disagreement_rate
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
                                "shadow_p_up_safety_target_passed": (
                                    p_up_safety_target_passed
                                ),
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
        p_up_target_safe_candidates = [
            row
            for row in eligible_candidates
            if bool(row["shadow_p_up_safety_target_passed"])
        ]
        if p_up_target_safe_candidates:
            selected = max(
                p_up_target_safe_candidates,
                key=lambda row: (
                    int(row["shadow_high_score_support_count"]),
                    float(row["shadow_high_score_realized_return_sum"]),
                    float(row["shadow_high_score_realized_return_mean"]),
                    float(row["shadow_selected_return_sum"]),
                    -float(row["shadow_mean_regret"]),
                    -float(row["shadow_largest_regret_value"]),
                    -float(row["shadow_p_up_action_disagreement_rate"]),
                ),
            )
        else:
            selected = min(
                eligible_candidates,
                key=lambda row: (
                    float(row["shadow_p_up_action_disagreement_rate"]),
                    float(row["shadow_mean_regret"]),
                    -int(row["shadow_high_score_support_count"]),
                    -float(row["shadow_high_score_realized_return_sum"]),
                    -float(row["shadow_high_score_realized_return_mean"]),
                    -float(row["shadow_selected_return_sum"]),
                    float(row["shadow_largest_regret_value"]),
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
            "eligible candidates pass the hard p_up gate and positive high-score "
            "return; when shadow candidates also pass the stricter p_up safety "
            "target, the selector chooses lower shadow mean regret while preserving "
            "source-gate high-score support; otherwise it chooses the lowest shadow "
            "p_up disagreement candidate before regret and support"
        ),
        "raw_weight_source_gate_high_score_support_threshold": (
            O_MIN_HIGH_SCORE_SUPPORT_COUNT
        ),
        "raw_weight_p_up_safety_target_disagreement_rate": (
            p_up_safety_target_disagreement_rate
        ),
        "raw_weight_prefers_p_up_safety_target_before_regret": True,
        "raw_weight_max_shadow_p_up_disagreement_rate": (
            max_shadow_p_up_disagreement_rate
        ),
        "raw_weight_hard_gate_max_p_up_disagreement_rate": (
            O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
        ),
        "raw_weight_uses_hard_p_up_gate_for_candidate_eligibility": True,
        "raw_weight_stricter_p_up_buffer_is_diagnostic": True,
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
    width = len(features[0]) if features else len(O_DEPLOYABLE_MODEL_FEATURE_NAMES)
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
        "selected_feature_set_name": model_training_summary[
            "selected_feature_set_name"
        ],
        "selected_correction_policy_name": model_training_summary[
            "selected_correction_policy_name"
        ],
        "selected_high_score_threshold_profile_name": model_training_summary[
            "selected_high_score_threshold_profile_name"
        ],
        "selected_joint_candidate_name": model_training_summary[
            "selected_joint_candidate_name"
        ],
        "selected_feature_set_config_hash": model_training_summary[
            "selected_feature_set_config_hash"
        ],
        "joint_feature_correction_selection_config_hash": model_training_summary[
            "joint_feature_correction_selection_config_hash"
        ],
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
    selected_model_inputs = tuple(model_training_summary["feature_names"])
    model_overlap = sorted(
        set(selected_model_inputs).intersection(O_FORBIDDEN_MODEL_INPUT_FIELDS)
    )
    label_overlap = sorted(
        set(O_TRAINING_LABEL_FIELDS).intersection(O_FORBIDDEN_MODEL_INPUT_FIELDS)
    )
    feature_coverage = _decision_time_feature_coverage(
        rows,
        model_input_fields=selected_model_inputs,
    )
    feature_provenance_passed = (
        int(feature_coverage["feature_provenance_violation_count"]) == 0
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
        "model_input_fields_decision_time_only": list(selected_model_inputs),
        "all_candidate_model_input_fields_decision_time_only": list(
            O_DEPLOYABLE_MODEL_FEATURE_NAMES
        ),
        "selected_feature_set_name": model_training_summary[
            "selected_feature_set_name"
        ],
        "selected_correction_policy_name": model_training_summary[
            "selected_correction_policy_name"
        ],
        "selected_high_score_threshold_profile_name": model_training_summary[
            "selected_high_score_threshold_profile_name"
        ],
        "selected_joint_candidate_name": model_training_summary[
            "selected_joint_candidate_name"
        ],
        "selected_feature_set_config_hash": model_training_summary[
            "selected_feature_set_config_hash"
        ],
        "joint_feature_correction_selection_config_hash": model_training_summary[
            "joint_feature_correction_selection_config_hash"
        ],
        "legacy_model_input_fields": list(O_LEGACY_DEPLOYABLE_MODEL_FEATURE_NAMES),
        "expanded_decision_time_feature_fields": list(
            O_EXPANDED_DECISION_TIME_FEATURE_FIELDS
        ),
        "reference_price_model_input_fields": list(
            O_REFERENCE_PRICE_MODEL_FEATURE_NAMES
        ),
        "book_pressure_model_input_fields": list(
            O_BOOK_PRESSURE_MODEL_FEATURE_NAMES
        ),
        "expanded_feature_coverage": feature_coverage,
        "model_training_summary": _compact_o_model_training_summary(
            model_training_summary
        ),
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
        "expanded_decision_time_feature_provenance_passed": (
            feature_provenance_passed
        ),
        "leakage_audit_passed": not model_overlap and feature_provenance_passed,
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


def _feature_set_selection_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    del config
    selection = dict(model_training_summary["feature_set_selection"])
    report = {
        **selection,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "final_model_feature_names": model_training_summary["feature_names"],
        "final_model_feature_schema_hash": canonical_json_sha256(
            model_training_summary["feature_names"]
        ),
        "final_model_correction_config_hash": model_training_summary[
            "correction_config_hash"
        ],
        "final_model_probe_config_hash": model_training_summary["probe_config_hash"],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["o_feature_set_selection_report_id"] = canonical_json_sha256(report)
    return report


def _joint_feature_correction_selection_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    del config
    selection = dict(model_training_summary["joint_feature_correction_selection"])
    report = {
        **selection,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "final_model_feature_names": model_training_summary["feature_names"],
        "final_model_feature_schema_hash": canonical_json_sha256(
            model_training_summary["feature_names"]
        ),
        "final_model_correction_policy_name": model_training_summary[
            "selected_correction_policy_name"
        ],
        "final_model_correction_config_hash": model_training_summary[
            "correction_config_hash"
        ],
        "final_model_probe_config_hash": model_training_summary["probe_config_hash"],
        "source_model_candidate_eligible": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["o_joint_feature_correction_selection_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_action_rank_handoff_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    leakage_report: dict[str, Any],
    eligibility_gate_report: dict[str, Any],
) -> dict[str, Any]:
    del config
    selected_rows = _v8_selected_action_rank_handoff_rows(
        rows=rows,
        model_training_summary=model_training_summary,
        split="validation",
    )
    handoff_contract = {
        "ranking_score_source": "model_predicted_score",
        "final_scoring_source": model_training_summary["final_scoring_source"],
        "required_fields": [
            "decision_group_id",
            "market_id",
            "decision_ts",
            "selected_action",
            "selected_side",
            "selected_action_family",
            "full_5_action_ranking",
            "corrected_model_score",
            "raw_model_score",
            "score_components",
            "high_score_flag",
            "p_up",
            "p_down",
            "p_up_action_disagreement",
            "microstructure_snapshot",
            "reference_price_to_beat_distance_at_decision",
            "reference_price_feature_provenance",
            "model_sha256",
            "feature_schema_hash",
            "split_hash",
        ],
        "execution_layer_required_runtime_constraints": [
            "max_position_exposure",
            "max_order_notional",
            "spread_limit",
            "book_staleness_limit",
            "queue_fill_limit",
            "time_to_close_limit",
            "kill_switch",
            "paper_only_until_separate_execution_gate_passes",
        ],
        "fail_closed_missing_field_policy": (
            "missing_required_handoff_field_blocks_execution_layer_handoff"
        ),
    }
    report = {
        "schema_version": O_V8_ACTION_RANK_HANDOFF_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_action_rank_handoff",
        "diagnostic_only": True,
        "v8_scope": "action_rank_signal_and_execution_layer_handoff_only",
        "model_layer_regret_risk_selection_enabled": False,
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "ranking_score_source": "model_predicted_score",
        "deployable_model_score_available": model_training_summary[
            "deployable_model_score_available"
        ],
        "strict_calibration_quality_passed": eligibility_gate_report[
            "strict_calibration_quality_passed"
        ],
        "calibration_quality_passed": eligibility_gate_report[
            "calibration_quality_passed"
        ],
        "relaxed_diagnostic_calibration_quality_passed": eligibility_gate_report[
            "relaxed_diagnostic_calibration_quality_passed"
        ],
        "relaxed_diagnostic_source_candidate": eligibility_gate_report[
            "relaxed_diagnostic_source_candidate"
        ],
        "v8_action_rank_quality_passed": eligibility_gate_report[
            "v8_action_rank_quality_passed"
        ],
        "v8_action_rank_candidate_eligible": eligibility_gate_report[
            "v8_action_rank_candidate_eligible"
        ],
        "v8_action_rank_reason_codes": eligibility_gate_report[
            "v8_action_rank_reason_codes"
        ],
        "v8_action_rank_gate_summary": eligibility_gate_report[
            "v8_action_rank_gate_summary"
        ],
        "v8_execution_risk_control_required": eligibility_gate_report[
            "v8_execution_risk_control_required"
        ],
        "v8_execution_risk_guard_report_available": True,
        "v8_execution_handoff_allowed": eligibility_gate_report[
            "v8_execution_handoff_allowed"
        ],
        "v8_execution_handoff_blocking_reason_codes": eligibility_gate_report[
            "v8_execution_handoff_blocking_reason_codes"
        ],
        "strict_source_gate_remains_failed": eligibility_gate_report[
            "strict_source_gate_remains_failed"
        ],
        "strict_source_gate_failure_reason_codes": eligibility_gate_report[
            "ineligible_reason_codes"
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "future_unseen_holdout_required": True,
        "execution_handoff_contract": handoff_contract,
        "handoff_contract_hash": canonical_json_sha256(handoff_contract),
        "selected_action_handoff_split": "validation",
        "selected_action_handoff_row_count": len(selected_rows),
        "selected_action_handoff_rows": selected_rows,
        "selected_action_handoff_summary": _v8_action_rank_handoff_summary(
            selected_rows
        ),
        "model_sha256": canonical_json_sha256(model_training_summary),
        "feature_schema_hash": canonical_json_sha256(
            model_training_summary["feature_names"]
        ),
        "split_hash": canonical_json_sha256(
            sorted(
                {
                    row["decision_group_id"]: row["split"]
                    for row in rows
                }.items()
            )
        ),
        "leakage_audit_passed": leakage_report["leakage_audit_passed"],
        "expanded_decision_time_feature_provenance_passed": leakage_report[
            "expanded_decision_time_feature_provenance_passed"
        ],
        "no_paper_live_unlock_from_v8_action_rank_gate": True,
        "no_source_freeze_unlock_from_v8_action_rank_gate": True,
        **compact_safety_fields(),
    }
    report["o_v8_action_rank_handoff_report_id"] = canonical_json_sha256(report)
    return report


def _v8_execution_risk_guard_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
) -> dict[str, Any]:
    del config
    guard_config = _v8_execution_guard_config()
    handoff_rows = list(handoff_report["selected_action_handoff_rows"])
    guard_rows = [
        _v8_execution_guard_decision(row, guard_config=guard_config)
        for row in handoff_rows
    ]
    summary = _v8_execution_guard_summary(guard_rows)
    backfill_summary = _v8_runtime_field_backfill_application_summary(guard_rows)
    blocking_reasons = sorted(
        {
            *handoff_report["v8_execution_handoff_blocking_reason_codes"],
            "runtime_execution_guard_diagnostic_only",
        }
    )
    report = {
        "schema_version": O_V8_EXECUTION_RISK_GUARD_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_risk_guard",
        "diagnostic_only": True,
        "v8_scope": "execution_layer_risk_guarded_action_selection_only",
        "source_action_rank_signal_report_type": handoff_report["report_type"],
        "source_action_rank_signal_report_id": handoff_report[
            "o_v8_action_rank_handoff_report_id"
        ],
        "source_action_rank_signal_available": True,
        "model_layer_regret_risk_selection_enabled": False,
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "ranking_score_source": "model_predicted_score",
        "final_scoring_source": handoff_report["execution_handoff_contract"][
            "final_scoring_source"
        ],
        "execution_guard_score_source": "execution_layer_score_adjustment_only",
        "guard_tuning_source": "fixed_runtime_config_no_outcome_tuning",
        "uses_replay_regret_labels_for_guard_tuning": False,
        "uses_validation_realized_outcomes_for_guard_tuning": False,
        "trains_regret_model": False,
        "trains_risk_head": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "source_score_fields_preserved": [
            "corrected_model_score",
            "raw_model_score",
            "score_components",
            "full_5_action_ranking",
        ],
        "allowed_runtime_input_fields": _v8_execution_allowed_input_fields(),
        "forbidden_guard_input_fields": [
            "realized_trade_pnl",
            "settlement_pnl",
            "total_polymarket_pnl",
            "regret_report_only",
            "realized_replay_return_report_only",
            "oracle_executable_best_action",
        ],
        "required_runtime_fields": _v8_execution_required_runtime_fields(),
        "execution_guard_config": guard_config,
        "execution_guard_config_hash": canonical_json_sha256(guard_config),
        "runtime_exposure_state_available": False,
        "runtime_risk_control_validation_passed": False,
        "v8_action_rank_quality_passed": handoff_report[
            "v8_action_rank_quality_passed"
        ],
        "v8_action_rank_candidate_eligible": handoff_report[
            "v8_action_rank_candidate_eligible"
        ],
        "v8_execution_risk_control_required": True,
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": blocking_reasons,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "future_unseen_holdout_required": True,
        "execution_guard_decision_count": len(guard_rows),
        "execution_guard_decision_rows": guard_rows,
        "execution_guard_summary": summary,
        "runtime_field_backfill_rules_applied": backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "applied_runtime_field_backfill_count": backfill_summary[
            "applied_runtime_field_backfill_count"
        ],
        "applied_runtime_field_backfill_rule_counts": backfill_summary[
            "applied_runtime_field_backfill_rule_counts"
        ],
        "runtime_field_backfill_provenance_validity_summary": backfill_summary[
            "runtime_field_backfill_provenance_validity_summary"
        ],
        "order_allowed_count": summary["order_allowed_count"],
        "blocked_decision_count": summary["blocked_decision_count"],
        "fail_closed_decision_count": summary["fail_closed_decision_count"],
        "proposed_order_size_total": summary["proposed_order_size_total"],
        "no_paper_live_unlock_from_execution_guard": True,
        "no_source_freeze_unlock_from_execution_guard": True,
        **compact_safety_fields(),
    }
    report["o_v8_execution_risk_guard_report_id"] = canonical_json_sha256(report)
    return report


def _v8_execution_guard_config() -> dict[str, Any]:
    return {
        "config_source": "deterministic_v8_execution_layer_runtime_guard",
        "uses_validation_labels_for_tuning": False,
        "uses_replay_regret_labels_for_tuning": False,
        "max_spread_bps": O_V8_EXECUTION_MAX_SPREAD_BPS,
        "max_book_staleness_ms": O_V8_EXECUTION_MAX_BOOK_STALENESS_MS,
        "min_queue_fill": O_V8_EXECUTION_MIN_QUEUE_FILL,
        "min_time_to_close_seconds": O_V8_EXECUTION_MIN_TIME_TO_CLOSE_SECONDS,
        "min_hts_time_to_close_seconds": O_V8_EXECUTION_MIN_HTS_TIME_TO_CLOSE_SECONDS,
        "min_score_margin": O_V8_EXECUTION_MIN_SCORE_MARGIN,
        "base_order_size": O_V8_EXECUTION_BASE_ORDER_SIZE,
        "max_order_size": O_V8_EXECUTION_BASE_ORDER_SIZE,
        "initial_cash_budget": O_V8_EXECUTION_INITIAL_CASH_BUDGET,
        "max_total_exposure": O_V8_EXECUTION_MAX_TOTAL_EXPOSURE,
        "max_market_exposure": O_V8_EXECUTION_MAX_MARKET_EXPOSURE,
        "max_side_exposure": O_V8_EXECUTION_MAX_SIDE_EXPOSURE,
        "duplicate_market_side_position_policy": "block",
        "opposite_side_conflict_policy": "block",
        "cooldown_after_blocked_decision_count": 0,
        "missing_exposure_state_policy": "block_trade_fail_closed",
        "paper_live_unlock_policy": "always_blocked_until_runtime_validation_passes",
    }


def _v8_execution_allowed_input_fields() -> list[str]:
    return [
        "selected_action",
        "selected_side",
        "selected_action_family",
        "full_5_action_ranking",
        "corrected_model_score",
        "raw_model_score",
        "score_components",
        "high_score_flag",
        "p_up",
        "p_down",
        "p_up_action_disagreement",
        "microstructure_snapshot",
        "reference_price_to_beat_distance_at_decision",
        "reference_price_feature_provenance",
        "decision_time_feature_max_input_ts",
        "runtime_exposure_state",
        "configured_execution_limits",
    ]


def _v8_execution_required_runtime_fields() -> list[str]:
    return [
        "selected_action",
        "selected_side",
        "selected_action_family",
        "full_5_action_ranking",
        "corrected_model_score",
        "high_score_flag",
        "p_up",
        "p_down",
        "p_up_action_disagreement",
        "microstructure_snapshot.spread_bps",
        "microstructure_snapshot.book_staleness_ms",
        "microstructure_snapshot.queue_fill_proxy",
        "microstructure_snapshot.time_to_close_seconds",
        "reference_price_feature_provenance.provenance_valid",
        "runtime_exposure_state",
    ]


def _v8_apply_runtime_field_policy_backfills(
    row: dict[str, Any],
    *,
    runtime_state_available: bool,
    runtime_mode: str,
) -> dict[str, Any]:
    cleaned = dict(row)
    source_action = str(cleaned.get("selected_action") or "")
    trade_action = source_action != "NO_TRADE"
    microstructure = dict(cleaned.get("microstructure_snapshot") or {})
    reference_provenance = dict(cleaned.get("reference_price_feature_provenance") or {})
    applied_rows: list[dict[str, Any]] = []
    provenance_violations: list[dict[str, Any]] = []

    if not trade_action:
        optional_fields = _v8_no_trade_optional_runtime_fields(
            microstructure=microstructure,
            reference_provenance=reference_provenance,
            runtime_state_available=runtime_state_available,
        )
        for field_name in optional_fields:
            applied_rows.append(
                _v8_runtime_field_applied_row(
                    row=cleaned,
                    field_name=field_name,
                    deterministic_rule_id=(
                        "make_non_order_runtime_fields_optional_for_no_trade"
                    ),
                    value=None,
                    source_field_name="runtime_required_field_policy",
                    source_timestamp=cleaned.get("decision_ts"),
                    max_input_ts=cleaned.get("decision_ts"),
                    provenance_valid=True,
                    reason_codes=["no_order_is_attempted_for_no_trade"],
                    application_type="required_field_policy_relaxation",
                )
            )
    else:
        if _optional_float(microstructure.get("time_to_close_seconds")) is None:
            source = dict(
                (
                    cleaned.get("runtime_field_backfill_sources")
                    or {}
                ).get("microstructure_snapshot.time_to_close_seconds")
                or {}
            )
            provenance_valid = _v8_runtime_backfill_source_provenance_valid(source)
            value = _optional_float(source.get("value"))
            if provenance_valid and value is not None:
                microstructure["time_to_close_seconds"] = value
                applied_rows.append(
                    _v8_runtime_field_applied_row(
                        row=cleaned,
                        field_name="microstructure_snapshot.time_to_close_seconds",
                        deterministic_rule_id=str(
                            source.get("deterministic_rule_id")
                            or (
                                "backfill_time_to_close_from_decision_time_feature_"
                                "or_market_schedule"
                            )
                        ),
                        value=value,
                        source_field_name=source.get("source_field_name"),
                        source_timestamp=source.get("source_timestamp"),
                        max_input_ts=source.get("max_input_ts"),
                        provenance_valid=True,
                        reason_codes=list(source.get("reason_codes") or []),
                        application_type="decision_time_data_join_backfill",
                    )
                )
            elif source:
                violation = _v8_runtime_field_applied_row(
                    row=cleaned,
                    field_name="microstructure_snapshot.time_to_close_seconds",
                    deterministic_rule_id=str(
                        source.get("deterministic_rule_id")
                        or (
                            "backfill_time_to_close_from_decision_time_feature_"
                            "or_market_schedule"
                        )
                    ),
                    value=value,
                    source_field_name=source.get("source_field_name"),
                    source_timestamp=source.get("source_timestamp"),
                    max_input_ts=source.get("max_input_ts"),
                    provenance_valid=False,
                    reason_codes=list(source.get("reason_codes") or [])
                    or ["time_to_close_source_provenance_invalid"],
                    application_type="decision_time_data_join_backfill",
                )
                provenance_violations.append(violation)

    cleaned["microstructure_snapshot"] = microstructure
    cleaned["reference_price_feature_provenance"] = reference_provenance
    rule_counts = Counter(row["deterministic_rule_id"] for row in applied_rows)
    return {
        "row": cleaned,
        "runtime_field_backfill_rules_applied": bool(applied_rows),
        "runtime_field_applied_backfill_rows": applied_rows,
        "runtime_field_backfill_rule_counts": dict(sorted(rule_counts.items())),
        "runtime_field_backfill_provenance_valid": not provenance_violations,
        "runtime_field_backfill_provenance_violations": provenance_violations,
        "runtime_field_backfill_runtime_mode": runtime_mode,
    }


def _v8_no_trade_optional_runtime_fields(
    *,
    microstructure: dict[str, Any],
    reference_provenance: dict[str, Any],
    runtime_state_available: bool,
) -> list[str]:
    optional_fields = []
    for field_name in (
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "time_to_close_seconds",
    ):
        if microstructure.get(field_name) is None:
            optional_fields.append(f"microstructure_snapshot.{field_name}")
    if reference_provenance.get("provenance_valid") is not True:
        optional_fields.append("reference_price_feature_provenance.provenance_valid")
    if not runtime_state_available:
        optional_fields.append("runtime_exposure_state")
    return sorted(set(optional_fields))


def _v8_runtime_backfill_source_provenance_valid(source: dict[str, Any]) -> bool:
    decision_ts = _optional_int(source.get("decision_ts"))
    source_timestamp = _optional_int(source.get("source_timestamp"))
    max_input_ts = _optional_int(source.get("max_input_ts"))
    return bool(
        source.get("source_field_name")
        and source.get("deterministic_rule_id")
        and source.get("provenance_valid") is True
        and decision_ts is not None
        and source_timestamp is not None
        and max_input_ts is not None
        and source_timestamp <= decision_ts
        and max_input_ts <= decision_ts
    )


def _v8_runtime_field_applied_row(
    *,
    row: dict[str, Any],
    field_name: str,
    deterministic_rule_id: str,
    value: Any,
    source_field_name: Any,
    source_timestamp: Any,
    max_input_ts: Any,
    provenance_valid: bool,
    reason_codes: list[str],
    application_type: str,
) -> dict[str, Any]:
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": row.get("selected_action"),
        "source_selected_side": row.get("selected_side"),
        "source_selected_family": row.get("selected_action_family"),
        "runtime_field_name": field_name,
        "applied_value": value,
        "source_field_name": source_field_name,
        "source_timestamp": source_timestamp,
        "max_input_ts": max_input_ts,
        "deterministic_rule_id": deterministic_rule_id,
        "provenance_valid": provenance_valid,
        "backfill_rule_applied_now": provenance_valid,
        "application_type": application_type,
        "reason_codes": sorted(set(reason_codes)),
    }


def _v8_execution_guard_decision(
    row: dict[str, Any],
    *,
    guard_config: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
    runtime_mode: str = "fail_closed_no_runtime_state",
) -> dict[str, Any]:
    runtime_state_available = (
        runtime_mode == "simulated_runtime_state"
        and runtime_state is not None
        and runtime_state.get("runtime_state_validation_passed") is True
    )
    cleanup = _v8_apply_runtime_field_policy_backfills(
        row,
        runtime_state_available=runtime_state_available,
        runtime_mode=runtime_mode,
    )
    row = cleanup["row"]
    source_action = str(row.get("selected_action") or "")
    source_side = str(row.get("selected_side") or _side_from_action(source_action))
    source_family = str(
        row.get("selected_action_family") or _action_family(source_action)
    )
    source_score = float(row.get("corrected_model_score") or 0.0)
    full_ranking = list(row.get("full_5_action_ranking") or [])
    microstructure = dict(row.get("microstructure_snapshot") or {})
    reference_provenance = dict(row.get("reference_price_feature_provenance") or {})

    spread = _optional_float(microstructure.get("spread_bps"))
    staleness = _optional_float(microstructure.get("book_staleness_ms"))
    queue = _optional_float(microstructure.get("queue_fill_proxy"))
    time_to_close = _optional_float(microstructure.get("time_to_close_seconds"))
    margin = _top_score_margin(full_ranking)

    guarded_action = source_action
    guarded_reason_codes: list[str] = []
    blocking_reason_codes: list[str] = []
    exposure_reason_codes: list[str] = []
    missing_codes = _v8_execution_missing_runtime_field_codes(
        row=row,
        microstructure=microstructure,
        reference_provenance=reference_provenance,
        trade_action=source_action != "NO_TRADE",
        runtime_state_available=runtime_state_available,
    )
    if missing_codes:
        blocking_reason_codes.append("execution_required_runtime_fields_missing")

    if source_action == "NO_TRADE":
        guarded_reason_codes.append("execution_no_trade_selected")
    else:
        if spread is None or spread > float(guard_config["max_spread_bps"]):
            blocking_reason_codes.append("execution_spread_too_wide")
        if staleness is None or staleness > float(
            guard_config["max_book_staleness_ms"]
        ):
            blocking_reason_codes.append("execution_book_stale")
        if queue is None or queue < float(guard_config["min_queue_fill"]):
            blocking_reason_codes.append("execution_liquidity_too_weak")
        min_time_to_close = (
            float(guard_config["min_hts_time_to_close_seconds"])
            if source_family == "HOLD_TO_SETTLEMENT"
            else float(guard_config["min_time_to_close_seconds"])
        )
        if time_to_close is None or time_to_close < min_time_to_close:
            blocking_reason_codes.append("execution_time_to_close_unsafe")
        if row.get("p_up_action_disagreement") is True:
            blocking_reason_codes.append("execution_p_up_side_disagreement")
        if margin is None or margin < float(guard_config["min_score_margin"]):
            guarded_reason_codes.append("execution_score_margin_too_close")

        hts_failed = source_family == "HOLD_TO_SETTLEMENT" and bool(
            set(blocking_reason_codes).intersection(
                {
                    "execution_spread_too_wide",
                    "execution_book_stale",
                    "execution_liquidity_too_weak",
                    "execution_time_to_close_unsafe",
                    "execution_p_up_side_disagreement",
                    "execution_required_runtime_fields_missing",
                }
            )
        )
        if hts_failed:
            guarded_reason_codes.append("execution_hts_guard_failed")
            same_side_sbc = _v8_same_side_sbc_candidate(
                full_ranking=full_ranking,
                side=source_side,
            )
            if same_side_sbc and _v8_sbc_candidate_runtime_safe(
                same_side_sbc,
                guard_config=guard_config,
            ):
                guarded_action = str(same_side_sbc["selected_action"])
                guarded_reason_codes.append(
                    "execution_hts_downgraded_to_same_side_sbc"
                )
            else:
                guarded_action = "NO_TRADE"
                guarded_reason_codes.append("execution_hts_blocked_to_no_trade")

        if not runtime_state_available:
            blocking_reason_codes.append("execution_exposure_state_missing")

    guarded_family = _action_family(guarded_action)
    guarded_side = _side_from_action(guarded_action)
    initial_proposed_size = (
        _v8_execution_order_size(
            high_score=bool(row.get("high_score_flag")),
            guard_reason_codes=guarded_reason_codes,
            guard_config=guard_config,
        )
        if guarded_action != "NO_TRADE" and source_action != "NO_TRADE"
        else 0.0
    )
    exposure_checked_size = initial_proposed_size
    if runtime_state_available and source_action != "NO_TRADE":
        exposure_check = _v8_execution_exposure_check(
            market_id=str(row.get("market_id") or ""),
            side=guarded_side,
            guarded_action=guarded_action,
            proposed_order_size=initial_proposed_size,
            runtime_state=runtime_state or {},
            guard_config=guard_config,
        )
        exposure_checked_size = float(exposure_check["capped_order_size"])
        blocking_reason_codes.extend(exposure_check["blocking_reason_codes"])
        exposure_reason_codes.extend(exposure_check["exposure_reason_codes"])

    penalties = _v8_execution_penalties(
        source_score=source_score,
        spread=spread,
        staleness=staleness,
        queue=queue,
        margin=margin,
        source_family=source_family,
        blocking_reason_codes=blocking_reason_codes,
        guard_config=guard_config,
    )
    execution_score = source_score - sum(penalties.values())
    unique_blocking = sorted(set(blocking_reason_codes))
    unique_guard = sorted(set(guarded_reason_codes))
    unique_exposure_reasons = sorted(set(exposure_reason_codes))
    trade_intent = guarded_action != "NO_TRADE" and source_action != "NO_TRADE"
    order_allowed = trade_intent and not unique_blocking
    if runtime_state_available and source_action != "NO_TRADE":
        final_exposure_reasons = [
            reason
            for reason in unique_exposure_reasons
            if reason
            not in {
                "execution_simulated_order_allowed",
                "execution_simulated_order_blocked",
            }
        ]
        final_exposure_reasons.append(
            "execution_simulated_order_allowed"
            if order_allowed
            else "execution_simulated_order_blocked"
        )
        unique_exposure_reasons = sorted(set(final_exposure_reasons))
    fail_closed = bool(unique_blocking) and source_action != "NO_TRADE"
    proposed_size = exposure_checked_size if order_allowed else 0.0
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": source_action,
        "source_selected_side": source_side,
        "source_selected_family": source_family,
        "source_model_score": source_score,
        "source_raw_model_score": row.get("raw_model_score"),
        "source_high_score_flag": bool(row.get("high_score_flag")),
        "p_up": row.get("p_up"),
        "p_down": row.get("p_down"),
        "p_up_action_disagreement": row.get("p_up_action_disagreement"),
        "microstructure_snapshot": microstructure,
        "reference_price_feature_provenance": reference_provenance,
        "runtime_field_backfill_rules_applied": cleanup[
            "runtime_field_backfill_rules_applied"
        ],
        "runtime_field_applied_backfill_rows": cleanup[
            "runtime_field_applied_backfill_rows"
        ],
        "runtime_field_backfill_rule_counts": cleanup[
            "runtime_field_backfill_rule_counts"
        ],
        "runtime_field_backfill_provenance_valid": cleanup[
            "runtime_field_backfill_provenance_valid"
        ],
        "runtime_field_backfill_provenance_violations": cleanup[
            "runtime_field_backfill_provenance_violations"
        ],
        "runtime_field_backfill_runtime_mode": cleanup[
            "runtime_field_backfill_runtime_mode"
        ],
        "decision_time_feature_max_input_ts": row.get(
            "decision_time_feature_max_input_ts"
        ),
        "top_k_action_ranking": [
            _v8_execution_top_k_action_entry(candidate)
            for candidate in full_ranking[:5]
        ],
        "execution_guarded_action": guarded_action,
        "execution_guarded_side": guarded_side,
        "execution_guarded_family": guarded_family,
        "execution_guarded_score": execution_score,
        "execution_score_penalties": penalties,
        "execution_score_formula": (
            "source_model_score - spread_penalty - staleness_penalty - "
            "liquidity_penalty - exposure_penalty - score_margin_penalty - "
            "family_specific_penalty"
        ),
        "order_allowed": order_allowed,
        "proposed_order_size": proposed_size,
        "uncapped_proposed_order_size": initial_proposed_size,
        "sizing_reason_codes": _v8_execution_sizing_reason_codes(
            proposed_order_size=proposed_size,
            guard_reason_codes=unique_guard,
            blocking_reason_codes=unique_blocking,
            source_action=source_action,
        ),
        "exposure_reason_codes": unique_exposure_reasons,
        "execution_guard_reason_codes": unique_guard,
        "execution_blocking_reason_codes": unique_blocking,
        "required_runtime_fields_present": not missing_codes,
        "missing_runtime_field_codes": missing_codes,
        "fail_closed": fail_closed,
        "simulation_only": True,
        "runtime_mode": runtime_mode,
        "runtime_exposure_state_available": runtime_state_available,
        "runtime_exposure_state": _v8_compact_runtime_state(runtime_state)
        if runtime_state_available
        else None,
        "runtime_limits": {
            "max_order_size": guard_config["max_order_size"],
            "max_total_exposure": guard_config["max_total_exposure"],
            "max_market_exposure": guard_config["max_market_exposure"],
            "max_side_exposure": guard_config["max_side_exposure"],
            "max_spread_bps": guard_config["max_spread_bps"],
            "max_book_staleness_ms": guard_config["max_book_staleness_ms"],
            "min_queue_fill": guard_config["min_queue_fill"],
            "min_time_to_close_seconds": guard_config[
                "min_time_to_close_seconds"
            ],
            "min_hts_time_to_close_seconds": guard_config[
                "min_hts_time_to_close_seconds"
            ],
        },
        "source_score_mutated": False,
        "o_model_predicted_score_mutated": False,
    }


def _v8_execution_exposure_check(
    *,
    market_id: str,
    side: str,
    guarded_action: str,
    proposed_order_size: float,
    runtime_state: dict[str, Any],
    guard_config: dict[str, Any],
) -> dict[str, Any]:
    if guarded_action == "NO_TRADE" or proposed_order_size <= 0.0:
        return {
            "capped_order_size": 0.0,
            "blocking_reason_codes": [],
            "exposure_reason_codes": ["execution_no_exposure_delta_for_no_trade"],
        }
    blocking_reasons: list[str] = []
    exposure_reasons = ["execution_simulated_runtime_state_valid"]
    market_exposure = dict(
        runtime_state.get("current_market_exposure_by_market_id") or {}
    )
    side_exposure = dict(runtime_state.get("current_side_exposure_by_side") or {})
    open_by_market = dict(runtime_state.get("open_position_by_market_id") or {})
    open_by_market_side = dict(
        runtime_state.get("open_position_by_market_side") or {}
    )
    cooldown_state = dict(runtime_state.get("cooldown_state") or {})

    if cooldown_state.get(market_id):
        blocking_reasons.append("execution_cooldown_active")
    market_side_key = f"{market_id}|{side}"
    if market_side_key in open_by_market_side:
        blocking_reasons.append("execution_duplicate_market_side_position")
    existing_market_position = open_by_market.get(market_id)
    if (
        isinstance(existing_market_position, dict)
        and existing_market_position.get("side") not in (None, side)
    ):
        blocking_reasons.append("execution_opposite_side_conflict")

    max_order_size = float(guard_config["max_order_size"])
    capped_size = min(float(proposed_order_size), max_order_size)
    if capped_size < float(proposed_order_size):
        exposure_reasons.append("execution_order_size_capped_by_max_order_size")

    total_exposure = float(runtime_state.get("current_total_exposure") or 0.0)
    current_market_exposure = float(market_exposure.get(market_id) or 0.0)
    current_side_exposure = float(side_exposure.get(side) or 0.0)
    total_remaining = max(
        0.0,
        float(guard_config["max_total_exposure"]) - total_exposure,
    )
    market_remaining = max(
        0.0,
        float(guard_config["max_market_exposure"]) - current_market_exposure,
    )
    side_remaining = max(
        0.0,
        float(guard_config["max_side_exposure"]) - current_side_exposure,
    )
    capped_size = min(capped_size, total_remaining, market_remaining, side_remaining)
    if capped_size <= 0.0:
        if total_remaining <= 0.0:
            blocking_reasons.append("execution_total_exposure_limit_reached")
        if market_remaining <= 0.0:
            blocking_reasons.append("execution_market_exposure_limit_reached")
        if side_remaining <= 0.0:
            blocking_reasons.append("execution_side_exposure_limit_reached")
    elif capped_size < float(proposed_order_size):
        exposure_reasons.append("execution_order_size_capped_by_exposure_limit")

    if blocking_reasons:
        exposure_reasons.append("execution_simulated_order_blocked")
    else:
        exposure_reasons.append("execution_simulated_order_allowed")
    return {
        "capped_order_size": capped_size if not blocking_reasons else 0.0,
        "blocking_reason_codes": sorted(set(blocking_reasons)),
        "exposure_reason_codes": sorted(set(exposure_reasons)),
    }


def _v8_execution_simulated_runtime_reports(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_config = _v8_execution_guard_config()
    initial_state = _v8_initial_runtime_state(runtime_config)
    state = _copy_jsonable(initial_state)
    ordered_rows = sorted(
        handoff_report["selected_action_handoff_rows"],
        key=lambda row: (
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
            str(row.get("decision_group_id") or ""),
        ),
    )
    replay_rows = []
    for index, row in enumerate(ordered_rows, start=1):
        pre_state = _v8_compact_runtime_state(state)
        decision = _v8_execution_guard_decision(
            row,
            guard_config=runtime_config,
            runtime_state=state,
            runtime_mode="simulated_runtime_state",
        )
        simulated_order_id = None
        exposure_delta = 0.0
        exposure_reason_codes = list(decision["exposure_reason_codes"])
        if decision["order_allowed"]:
            simulated_order_id = f"sim-v8-o-{index:06d}"
            exposure_delta = float(decision["proposed_order_size"])
            _v8_apply_simulated_order_to_state(
                state=state,
                decision=decision,
                simulated_order_id=simulated_order_id,
            )
        else:
            _v8_record_blocked_simulated_order(
                state=state,
                decision=decision,
            )
            exposure_reason_codes.append("execution_simulated_order_blocked")
        post_state = _v8_compact_runtime_state(state)
        replay_rows.append(
            {
                **decision,
                "pre_decision_exposure_state": pre_state,
                "post_decision_exposure_state": post_state,
                "exposure_delta": exposure_delta,
                "simulated_order_id": simulated_order_id,
                "exposure_reason_codes": sorted(set(exposure_reason_codes)),
            }
        )
    final_state = _v8_compact_runtime_state(state)
    runtime_report = _v8_execution_runtime_state_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report=handoff_report,
        execution_guard_report=execution_guard_report,
        runtime_config=runtime_config,
        initial_state=_v8_compact_runtime_state(initial_state),
        final_state=final_state,
        replay_rows=replay_rows,
    )
    replay_report = _v8_execution_simulated_order_replay_report(
        m2_report_path=m2_report_path,
        m2_report=m2_report,
        handoff_report=handoff_report,
        execution_guard_report=execution_guard_report,
        runtime_state_report=runtime_report,
        replay_rows=replay_rows,
    )
    return runtime_report, replay_report


def _v8_initial_runtime_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_state_source": "simulated_diagnostic_ledger",
        "runtime_state_validation_passed": True,
        "validation_reason_codes": ["execution_simulated_runtime_state_valid"],
        "initial_cash_budget": float(config["initial_cash_budget"]),
        "max_total_exposure": float(config["max_total_exposure"]),
        "max_market_exposure": float(config["max_market_exposure"]),
        "max_side_exposure": float(config["max_side_exposure"]),
        "max_order_size": float(config["max_order_size"]),
        "current_total_exposure": 0.0,
        "current_market_exposure_by_market_id": {},
        "current_side_exposure_by_side": {"DOWN": 0.0, "NONE": 0.0, "UP": 0.0},
        "open_position_by_market_id": {},
        "open_position_by_market_side": {},
        "executed_simulated_orders": [],
        "blocked_simulated_orders": [],
        "cooldown_state": {},
    }


def _copy_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _v8_compact_runtime_state(
    state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "risk_state_source": state.get("risk_state_source"),
        "runtime_state_validation_passed": bool(
            state.get("runtime_state_validation_passed")
        ),
        "current_total_exposure": float(state.get("current_total_exposure") or 0.0),
        "current_market_exposure_by_market_id": dict(
            sorted(
                (state.get("current_market_exposure_by_market_id") or {}).items()
            )
        ),
        "current_side_exposure_by_side": dict(
            sorted((state.get("current_side_exposure_by_side") or {}).items())
        ),
        "open_position_by_market_id": dict(
            sorted((state.get("open_position_by_market_id") or {}).items())
        ),
        "open_position_by_market_side": dict(
            sorted((state.get("open_position_by_market_side") or {}).items())
        ),
        "executed_simulated_order_count": len(
            state.get("executed_simulated_orders") or []
        ),
        "blocked_simulated_order_count": len(
            state.get("blocked_simulated_orders") or []
        ),
        "cooldown_state": dict(sorted((state.get("cooldown_state") or {}).items())),
    }


def _v8_apply_simulated_order_to_state(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
    simulated_order_id: str,
) -> None:
    market_id = str(decision.get("market_id") or "")
    side = str(decision.get("execution_guarded_side") or "")
    action = str(decision.get("execution_guarded_action") or "")
    size = float(decision.get("proposed_order_size") or 0.0)
    state["current_total_exposure"] = (
        float(state.get("current_total_exposure") or 0.0) + size
    )
    market_exposure = state.setdefault("current_market_exposure_by_market_id", {})
    market_exposure[market_id] = float(market_exposure.get(market_id) or 0.0) + size
    side_exposure = state.setdefault("current_side_exposure_by_side", {})
    side_exposure[side] = float(side_exposure.get(side) or 0.0) + size
    position = {
        "market_id": market_id,
        "side": side,
        "action": action,
        "notional": size,
        "simulated_order_id": simulated_order_id,
    }
    state.setdefault("open_position_by_market_id", {})[market_id] = position
    state.setdefault("open_position_by_market_side", {})[
        f"{market_id}|{side}"
    ] = position
    state.setdefault("executed_simulated_orders", []).append(
        {
            "simulated_order_id": simulated_order_id,
            "decision_group_id": decision.get("decision_group_id"),
            "market_id": market_id,
            "decision_ts": decision.get("decision_ts"),
            "action": action,
            "side": side,
            "notional": size,
            "execution_guarded_score": decision.get("execution_guarded_score"),
        }
    )


def _v8_record_blocked_simulated_order(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    state.setdefault("blocked_simulated_orders", []).append(
        {
            "decision_group_id": decision.get("decision_group_id"),
            "market_id": decision.get("market_id"),
            "decision_ts": decision.get("decision_ts"),
            "source_selected_action": decision.get("source_selected_action"),
            "execution_guarded_action": decision.get("execution_guarded_action"),
            "execution_blocking_reason_codes": decision.get(
                "execution_blocking_reason_codes",
                [],
            ),
        }
    )


def _v8_execution_runtime_state_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    runtime_config: dict[str, Any],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    executed_orders = [
        row
        for row in replay_rows
        if row.get("simulated_order_id") and row.get("order_allowed") is True
    ]
    blocked_orders = [row for row in replay_rows if row.get("order_allowed") is False]
    validation_passed = bool(final_state.get("runtime_state_validation_passed"))
    report = {
        "schema_version": O_V8_EXECUTION_RUNTIME_STATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_runtime_state",
        "diagnostic_only": True,
        "simulation_only": True,
        "risk_state_source": "simulated_diagnostic_ledger",
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "source_action_rank_signal_report_id": handoff_report[
            "o_v8_action_rank_handoff_report_id"
        ],
        "execution_guard_report_id": execution_guard_report[
            "o_v8_execution_risk_guard_report_id"
        ],
        "runtime_state_config": runtime_config,
        "runtime_state_config_hash": canonical_json_sha256(runtime_config),
        "initial_state": initial_state,
        "final_state": final_state,
        "current_total_exposure": final_state["current_total_exposure"],
        "exposure_by_market": final_state["current_market_exposure_by_market_id"],
        "exposure_by_side": final_state["current_side_exposure_by_side"],
        "open_position_by_market_id": final_state["open_position_by_market_id"],
        "open_position_by_market_side": final_state["open_position_by_market_side"],
        "executed_simulated_orders": [
            _v8_runtime_order_summary(row) for row in executed_orders
        ],
        "blocked_simulated_orders": [
            _v8_runtime_blocked_order_summary(row) for row in blocked_orders
        ],
        "cooldown_state": final_state["cooldown_state"],
        "runtime_state_validation_passed": validation_passed,
        "validation_status": "passed_simulated_diagnostic"
        if validation_passed
        else "blocked_fail_closed",
        "validation_reason_codes": ["execution_simulated_runtime_state_valid"]
        if validation_passed
        else ["execution_simulated_runtime_state_invalid"],
        "runtime_risk_control_validation_passed": validation_passed,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    report["o_v8_execution_runtime_state_report_id"] = canonical_json_sha256(report)
    return report


def _v8_execution_simulated_order_replay_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    runtime_state_report: dict[str, Any],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed_rows = [row for row in replay_rows if row["order_allowed"]]
    blocked_rows = [row for row in replay_rows if not row["order_allowed"]]
    block_reason_counts = Counter(
        reason
        for row in replay_rows
        for reason in row["execution_blocking_reason_codes"]
    )
    exposure_reason_counts = Counter(
        reason for row in replay_rows for reason in row["exposure_reason_codes"]
    )
    action_counts = Counter(str(row["execution_guarded_action"]) for row in replay_rows)
    family_counts = Counter(str(row["execution_guarded_family"]) for row in replay_rows)
    side_counts = Counter(str(row["execution_guarded_side"]) for row in replay_rows)
    backfill_summary = _v8_runtime_field_backfill_application_summary(replay_rows)
    replay_hash_payload = [
        {
            "decision_group_id": row["decision_group_id"],
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "execution_guarded_action": row["execution_guarded_action"],
            "order_allowed": row["order_allowed"],
            "proposed_order_size": row["proposed_order_size"],
            "simulated_order_id": row["simulated_order_id"],
            "blocking_reason_codes": row["execution_blocking_reason_codes"],
            "exposure_reason_codes": row["exposure_reason_codes"],
        }
        for row in replay_rows
    ]
    report = {
        "schema_version": O_V8_EXECUTION_SIMULATED_ORDER_REPLAY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_simulated_order_replay",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "replay_source_report_id": handoff_report[
            "o_v8_action_rank_handoff_report_id"
        ],
        "execution_guard_report_id": execution_guard_report[
            "o_v8_execution_risk_guard_report_id"
        ],
        "runtime_state_report_id": runtime_state_report[
            "o_v8_execution_runtime_state_report_id"
        ],
        "decision_count": len(replay_rows),
        "simulated_allowed_order_count": len(allowed_rows),
        "blocked_decision_count": len(blocked_rows),
        "total_proposed_notional": sum(
            float(row["proposed_order_size"]) for row in allowed_rows
        ),
        "final_exposure": runtime_state_report["final_state"],
        "action_distribution": dict(sorted(action_counts.items())),
        "family_distribution": dict(sorted(family_counts.items())),
        "side_distribution": dict(sorted(side_counts.items())),
        "block_reason_distribution": dict(sorted(block_reason_counts.items())),
        "exposure_reason_distribution": dict(sorted(exposure_reason_counts.items())),
        "runtime_field_backfill_rules_applied": backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "applied_runtime_field_backfill_count": backfill_summary[
            "applied_runtime_field_backfill_count"
        ],
        "applied_runtime_field_backfill_rule_counts": backfill_summary[
            "applied_runtime_field_backfill_rule_counts"
        ],
        "runtime_field_backfill_provenance_validity_summary": backfill_summary[
            "runtime_field_backfill_provenance_validity_summary"
        ],
        "deterministic_replay_hash": canonical_json_sha256(replay_hash_payload),
        "runtime_risk_control_validation_passed": runtime_state_report[
            "runtime_risk_control_validation_passed"
        ],
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": [
            "future_explicit_execution_handoff_gate_required",
            "paper_live_unlock_prohibited",
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "simulated_decision_rows": replay_rows,
        **compact_safety_fields(),
    }
    report["o_v8_execution_simulated_order_replay_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_execution_allowed_order_quality_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
) -> dict[str, Any]:
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    allowed_rows = [row for row in replay_rows if row.get("order_allowed") is True]
    blocked_rows = [row for row in replay_rows if row.get("order_allowed") is not True]
    allowed_quality_rows = [
        _v8_allowed_order_quality_row(row)
        for row in sorted(
            allowed_rows,
            key=lambda row: (
                int(row.get("decision_ts") or 0),
                str(row.get("market_id") or ""),
                str(row.get("decision_group_id") or ""),
            ),
        )
    ]
    residual_blocked_rows = [
        _v8_residual_blocked_decision_quality_row(row)
        for row in sorted(
            blocked_rows,
            key=lambda row: (
                int(row.get("decision_ts") or 0),
                str(row.get("market_id") or ""),
                str(row.get("decision_group_id") or ""),
            ),
        )
    ]
    recommendation_counts = Counter(
        code
        for row in residual_blocked_rows
        for code in row["deterministic_recommendation_codes"]
    )
    primary_recommendation_counts = Counter(
        row["primary_deterministic_recommendation"]
        for row in residual_blocked_rows
    )
    residual_reason_counts = Counter(
        reason
        for row in residual_blocked_rows
        for reason in row["execution_blocking_reason_codes"]
    )
    residual_category_counts = Counter(
        category
        for row in residual_blocked_rows
        for category in row["minimal_blocking_set"]["blocker_categories"]
    )
    order_origin_counts = Counter(row["order_origin"] for row in allowed_quality_rows)
    p_up_agreement_counts = Counter(
        row["p_up_agreement_status"] for row in allowed_quality_rows
    )
    report = {
        "schema_version": O_V8_EXECUTION_ALLOWED_ORDER_QUALITY_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_allowed_order_quality",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "simulated_order_replay_report_id": simulated_order_replay_report[
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "analysis_source": "simulated_order_replay_rows_only",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "decision_count": len(replay_rows),
        "allowed_order_count": len(allowed_quality_rows),
        "blocked_decision_count": len(residual_blocked_rows),
        "allowed_order_action_distribution": _v8_distribution(
            row["execution_guarded_action"] for row in allowed_quality_rows
        ),
        "allowed_order_family_distribution": _v8_distribution(
            row["execution_guarded_family"] for row in allowed_quality_rows
        ),
        "allowed_order_side_distribution": _v8_distribution(
            row["execution_guarded_side"] for row in allowed_quality_rows
        ),
        "allowed_order_origin_distribution": dict(sorted(order_origin_counts.items())),
        "allowed_order_p_up_agreement_distribution": dict(
            sorted(p_up_agreement_counts.items())
        ),
        "allowed_order_metric_summary": _v8_allowed_order_metric_summary(
            allowed_quality_rows
        ),
        "allowed_order_quality_rows": allowed_quality_rows,
        "residual_block_reason_distribution": dict(
            sorted(residual_reason_counts.items())
        ),
        "residual_blocker_category_distribution": dict(
            sorted(residual_category_counts.items())
        ),
        "residual_blocker_summary": _v8_residual_blocker_summary(
            residual_blocked_rows
        ),
        "residual_block_reason_summary_by_action": _v8_block_reason_summary_by(
            residual_blocked_rows,
            "source_selected_action",
        ),
        "residual_block_reason_summary_by_family": _v8_block_reason_summary_by(
            residual_blocked_rows,
            "source_selected_family",
        ),
        "residual_block_reason_summary_by_side": _v8_block_reason_summary_by(
            residual_blocked_rows,
            "source_selected_side",
        ),
        "residual_block_reason_summary_by_market": _v8_block_reason_summary_by(
            residual_blocked_rows,
            "market_id",
        ),
        "residual_block_reason_summary_by_time_to_close_bucket": (
            _v8_block_reason_summary_by(
                residual_blocked_rows,
                "time_to_close_bucket",
            )
        ),
        "deterministic_recommendation_counts": dict(
            sorted(recommendation_counts.items())
        ),
        "primary_deterministic_recommendation_counts": dict(
            sorted(primary_recommendation_counts.items())
        ),
        "residual_blocked_decision_rows": residual_blocked_rows,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_allowed_order_quality": True,
        "no_source_freeze_unlock_from_allowed_order_quality": True,
        **compact_safety_fields(),
    }
    report["o_v8_execution_allowed_order_quality_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_allowed_order_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    microstructure = dict(row.get("microstructure_snapshot") or {})
    pre_state = dict(row.get("pre_decision_exposure_state") or {})
    post_state = dict(row.get("post_decision_exposure_state") or {})
    market_id = str(row.get("market_id") or "")
    side = str(row.get("execution_guarded_side") or "")
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "simulated_order_id": row.get("simulated_order_id"),
        "source_selected_action": row.get("source_selected_action"),
        "source_selected_family": row.get("source_selected_family"),
        "source_selected_side": row.get("source_selected_side"),
        "execution_guarded_action": row.get("execution_guarded_action"),
        "execution_guarded_family": row.get("execution_guarded_family"),
        "execution_guarded_side": row.get("execution_guarded_side"),
        "order_origin": _v8_allowed_order_origin(row),
        "came_from_original_selected_action": (
            row.get("source_selected_action") == row.get("execution_guarded_action")
        ),
        "source_model_score": _optional_float(row.get("source_model_score")),
        "source_raw_model_score": _optional_float(row.get("source_raw_model_score")),
        "execution_guarded_score": _optional_float(row.get("execution_guarded_score")),
        "execution_score_penalties": dict(row.get("execution_score_penalties") or {}),
        "spread_bps": _optional_float(microstructure.get("spread_bps")),
        "book_staleness_ms": _optional_float(microstructure.get("book_staleness_ms")),
        "queue_fill_proxy": _optional_float(microstructure.get("queue_fill_proxy")),
        "time_to_close_seconds": _optional_float(
            microstructure.get("time_to_close_seconds")
        ),
        "p_up": _optional_float(row.get("p_up")),
        "p_down": _optional_float(row.get("p_down")),
        "p_up_action_disagreement": bool(row.get("p_up_action_disagreement")),
        "p_up_agreement_status": _v8_p_up_agreement_status(row),
        "proposed_order_size": _optional_float(row.get("proposed_order_size")),
        "uncapped_proposed_order_size": _optional_float(
            row.get("uncapped_proposed_order_size")
        ),
        "exposure_delta": _optional_float(row.get("exposure_delta")),
        "sizing_reason_codes": list(row.get("sizing_reason_codes") or []),
        "exposure_reason_codes": list(row.get("exposure_reason_codes") or []),
        "execution_guard_reason_codes": list(
            row.get("execution_guard_reason_codes") or []
        ),
        "pre_decision_exposure": _v8_quality_exposure_snapshot(
            pre_state,
            market_id=market_id,
            side=side,
        ),
        "post_decision_exposure": _v8_quality_exposure_snapshot(
            post_state,
            market_id=market_id,
            side=side,
        ),
        "source_score_mutated": bool(row.get("source_score_mutated")),
        "o_model_predicted_score_mutated": bool(
            row.get("o_model_predicted_score_mutated")
        ),
    }


def _v8_residual_blocked_decision_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    minimal_blocking_set = _v8_minimal_blocking_set(row)
    recommendations = _v8_residual_blocker_recommendations(
        row,
        minimal_blocking_set,
    )
    microstructure = dict(row.get("microstructure_snapshot") or {})
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": row.get("source_selected_action"),
        "source_selected_family": row.get("source_selected_family"),
        "source_selected_side": row.get("source_selected_side"),
        "execution_guarded_action": row.get("execution_guarded_action"),
        "execution_guarded_family": row.get("execution_guarded_family"),
        "execution_guarded_side": row.get("execution_guarded_side"),
        "source_model_score": _optional_float(row.get("source_model_score")),
        "execution_guarded_score": _optional_float(row.get("execution_guarded_score")),
        "spread_bps": _optional_float(microstructure.get("spread_bps")),
        "book_staleness_ms": _optional_float(microstructure.get("book_staleness_ms")),
        "queue_fill_proxy": _optional_float(microstructure.get("queue_fill_proxy")),
        "time_to_close_seconds": _v8_row_time_to_close_seconds(row),
        "time_to_close_bucket": _v8_time_to_close_bucket_from_decision(row),
        "p_up": _optional_float(row.get("p_up")),
        "p_down": _optional_float(row.get("p_down")),
        "p_up_action_disagreement": bool(row.get("p_up_action_disagreement")),
        "p_up_agreement_status": _v8_p_up_agreement_status(row),
        "proposed_order_size": _optional_float(row.get("proposed_order_size")),
        "uncapped_proposed_order_size": _optional_float(
            row.get("uncapped_proposed_order_size")
        ),
        "sizing_reason_codes": list(row.get("sizing_reason_codes") or []),
        "execution_blocking_reason_codes": list(
            row.get("execution_blocking_reason_codes") or []
        ),
        "execution_guard_reason_codes": list(
            row.get("execution_guard_reason_codes") or []
        ),
        "exposure_reason_codes": list(row.get("exposure_reason_codes") or []),
        "minimal_blocking_set": minimal_blocking_set,
        "deterministic_recommendation_codes": recommendations[
            "recommendation_codes"
        ],
        "primary_deterministic_recommendation": recommendations[
            "primary_recommendation"
        ],
        "recommendation_reason_codes": recommendations["reason_codes"],
        "source_score_mutated": bool(row.get("source_score_mutated")),
        "o_model_predicted_score_mutated": bool(
            row.get("o_model_predicted_score_mutated")
        ),
    }


def _v8_allowed_order_origin(row: dict[str, Any]) -> str:
    if row.get("source_selected_action") == row.get("execution_guarded_action"):
        return "original_selected_action"
    if "execution_hts_downgraded_to_same_side_sbc" in set(
        row.get("execution_guard_reason_codes") or []
    ):
        return "hts_to_sbc_downgrade"
    return "other_execution_guard_adjustment"


def _v8_p_up_agreement_status(row: dict[str, Any]) -> str:
    if row.get("p_up_action_disagreement") is True:
        return "p_up_disagrees"
    if row.get("p_up_action_disagreement") is False:
        return "p_up_agrees"
    return "p_up_not_applicable"


def _v8_quality_exposure_snapshot(
    state: dict[str, Any],
    *,
    market_id: str,
    side: str,
) -> dict[str, Any]:
    market_exposure = dict(state.get("current_market_exposure_by_market_id") or {})
    side_exposure = dict(state.get("current_side_exposure_by_side") or {})
    return {
        "current_total_exposure": _optional_float(
            state.get("current_total_exposure")
        ),
        "current_market_exposure": _optional_float(market_exposure.get(market_id)),
        "current_side_exposure": _optional_float(side_exposure.get(side)),
        "executed_simulated_order_count": int(
            state.get("executed_simulated_order_count") or 0
        ),
        "blocked_simulated_order_count": int(
            state.get("blocked_simulated_order_count") or 0
        ),
    }


def _v8_allowed_order_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_model_score": _v8_numeric_summary(
            row.get("source_model_score") for row in rows
        ),
        "execution_guarded_score": _v8_numeric_summary(
            row.get("execution_guarded_score") for row in rows
        ),
        "spread_bps": _v8_numeric_summary(row.get("spread_bps") for row in rows),
        "book_staleness_ms": _v8_numeric_summary(
            row.get("book_staleness_ms") for row in rows
        ),
        "queue_fill_proxy": _v8_numeric_summary(
            row.get("queue_fill_proxy") for row in rows
        ),
        "time_to_close_seconds": _v8_numeric_summary(
            row.get("time_to_close_seconds") for row in rows
        ),
        "proposed_order_size": _v8_numeric_summary(
            row.get("proposed_order_size") for row in rows
        ),
        "exposure_delta": _v8_numeric_summary(row.get("exposure_delta") for row in rows),
        "pre_total_exposure": _v8_numeric_summary(
            row["pre_decision_exposure"].get("current_total_exposure")
            for row in rows
        ),
        "post_total_exposure": _v8_numeric_summary(
            row["post_decision_exposure"].get("current_total_exposure")
            for row in rows
        ),
    }


def _v8_numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    numeric_values = [
        value
        for value in (_optional_float(value) for value in values)
        if value is not None
    ]
    if not numeric_values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(numeric_values),
        "min": min(numeric_values),
        "median": statistics.median(numeric_values),
        "mean": statistics.mean(numeric_values),
        "max": max(numeric_values),
    }


def _v8_distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _v8_residual_blocker_recommendations(
    row: dict[str, Any],
    minimal_blocking_set: dict[str, Any],
) -> dict[str, Any]:
    categories = set(minimal_blocking_set.get("blocker_categories") or [])
    blocking_reasons = set(row.get("execution_blocking_reason_codes") or [])
    recommendation_codes = []
    reason_codes = []
    if row.get("source_selected_action") == "NO_TRADE":
        recommendation_codes.append("keep_blocked")
        reason_codes.append("no_order_candidate")
    if "exposure_limits" in categories:
        recommendation_codes.append("needs_exposure_policy_review")
        reason_codes.append("exposure_limit_or_position_conflict")
    if "execution_duplicate_market_side_position" in blocking_reasons:
        recommendation_codes.append("keep_blocked")
        reason_codes.append("duplicate_market_side_position_should_remain_blocked")
    if "p_up_disagreement" in categories:
        recommendation_codes.append("needs_p_up_action_rank_review")
        reason_codes.append("p_up_disagrees_with_selected_action_side")
    if "time_to_close" in categories:
        recommendation_codes.append("needs_time_to_close_policy_review")
        reason_codes.append("time_to_close_below_execution_policy")
    if "hts_guard_failure" in categories and not recommendation_codes:
        recommendation_codes.append("keep_blocked")
        reason_codes.append("hts_guard_failed_without_safe_fallback")
    if not recommendation_codes:
        recommendation_codes.append("keep_blocked")
        reason_codes.append("no_safe_order_path_identified")
    recommendation_codes = sorted(set(recommendation_codes))
    return {
        "recommendation_codes": recommendation_codes,
        "primary_recommendation": _v8_primary_residual_recommendation(
            recommendation_codes
        ),
        "reason_codes": sorted(set(reason_codes)),
    }


def _v8_primary_residual_recommendation(recommendation_codes: list[str]) -> str:
    priority = (
        "needs_p_up_action_rank_review",
        "needs_exposure_policy_review",
        "needs_time_to_close_policy_review",
        "keep_blocked",
    )
    for code in priority:
        if code in recommendation_codes:
            return code
    return recommendation_codes[0] if recommendation_codes else "keep_blocked"


def _v8_residual_blocker_summary(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "exposure_limit_blocked_decision_count": sum(
            1
            for row in rows
            if "exposure_limits"
            in set(row["minimal_blocking_set"]["blocker_categories"])
        ),
        "p_up_disagreement_blocked_decision_count": sum(
            1
            for row in rows
            if "p_up_disagreement"
            in set(row["minimal_blocking_set"]["blocker_categories"])
        ),
        "duplicate_market_side_position_count": sum(
            1
            for row in rows
            if "execution_duplicate_market_side_position"
            in set(row["execution_blocking_reason_codes"])
        ),
        "time_to_close_unsafe_count": sum(
            1
            for row in rows
            if "execution_time_to_close_unsafe"
            in set(row["execution_blocking_reason_codes"])
        ),
        "hts_guard_failed_count": sum(
            1
            for row in rows
            if "hts_guard_failure"
            in set(row["minimal_blocking_set"]["blocker_categories"])
        ),
    }


def _v8_execution_policy_readiness_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
    allowed_order_quality_report: dict[str, Any],
) -> dict[str, Any]:
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    allowed_rows = list(allowed_order_quality_report.get("allowed_order_quality_rows") or [])
    config = _v8_execution_guard_config()
    min_allowed_order_count = 5
    microstructure_quality = _v8_allowed_order_microstructure_readiness(
        allowed_rows,
        config=config,
    )
    exposure_quality = _v8_allowed_order_exposure_readiness(
        allowed_rows,
        config=config,
    )
    missing_runtime_field_rows = [
        row
        for row in replay_rows
        if row.get("missing_runtime_field_codes")
    ]
    provenance_violations = [
        violation
        for row in replay_rows
        for violation in row.get("runtime_field_backfill_provenance_violations", [])
    ]
    p_up_disagreement_rows = [
        row
        for row in allowed_rows
        if row.get("p_up_agreement_status") != "p_up_agrees"
    ]
    unsafe_origin_rows = [
        row
        for row in allowed_rows
        if row.get("order_origin")
        not in {"original_selected_action", "hts_to_sbc_downgrade"}
    ]
    safe_downgrade_rows = [
        row
        for row in allowed_rows
        if row.get("order_origin") == "hts_to_sbc_downgrade"
        and "execution_hts_downgraded_to_same_side_sbc"
        in set(row.get("execution_guard_reason_codes") or [])
    ]
    unsafe_downgrade_rows = [
        row
        for row in allowed_rows
        if row.get("order_origin") == "hts_to_sbc_downgrade"
        and "execution_hts_downgraded_to_same_side_sbc"
        not in set(row.get("execution_guard_reason_codes") or [])
    ]
    safety_flags = compact_safety_fields()
    no_paper_live_capital_flags = (
        safety_flags["paper_only"] is True
        and safety_flags["capital_at_risk"] is False
        and safety_flags["polymarket_write_enabled"] is False
        and safety_flags["wallet_signing_enabled"] is False
    )
    required_checks = {
        "min_allowed_order_count": _v8_execution_policy_check(
            passed=len(allowed_rows) >= min_allowed_order_count,
            observed=len(allowed_rows),
            required=f">={min_allowed_order_count}",
            reason_code="execution_policy_min_allowed_order_count_not_met",
        ),
        "zero_missing_runtime_fields": _v8_execution_policy_check(
            passed=not missing_runtime_field_rows,
            observed=len(missing_runtime_field_rows),
            required="0",
            reason_code="execution_policy_runtime_missing_fields_present",
        ),
        "zero_provenance_violations": _v8_execution_policy_check(
            passed=not provenance_violations,
            observed=len(provenance_violations),
            required="0",
            reason_code="execution_policy_provenance_violations_present",
        ),
        "all_allowed_orders_p_up_agreement": _v8_execution_policy_check(
            passed=not p_up_disagreement_rows,
            observed=len(p_up_disagreement_rows),
            required="0 p_up disagreements",
            reason_code="execution_policy_allowed_order_p_up_disagreement_present",
        ),
        "all_allowed_orders_original_or_safe_downgrade": _v8_execution_policy_check(
            passed=not unsafe_origin_rows and not unsafe_downgrade_rows,
            observed={
                "unsafe_origin_count": len(unsafe_origin_rows),
                "safe_downgrade_count": len(safe_downgrade_rows),
                "unsafe_downgrade_count": len(unsafe_downgrade_rows),
            },
            required="original_selected_action or execution_hts_downgraded_to_same_side_sbc",
            reason_code="execution_policy_allowed_order_origin_not_safe",
        ),
        "allowed_order_microstructure_quality_passed": _v8_execution_policy_check(
            passed=microstructure_quality["passed"],
            observed=microstructure_quality["failure_count"],
            required="0 microstructure quality failures",
            reason_code="execution_policy_allowed_order_microstructure_quality_failed",
        ),
        "allowed_order_exposure_within_limits": _v8_execution_policy_check(
            passed=exposure_quality["passed"],
            observed=exposure_quality["failure_count"],
            required="0 exposure limit failures",
            reason_code="execution_policy_allowed_order_exposure_limit_failed",
        ),
        "no_paper_live_write_or_capital_flags": _v8_execution_policy_check(
            passed=no_paper_live_capital_flags,
            observed=safety_flags,
            required={
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            },
            reason_code="execution_policy_paper_live_or_capital_flag_enabled",
        ),
    }
    blocking_reason_codes = sorted(
        check["reason_code"]
        for check in required_checks.values()
        if check["passed"] is not True
    )
    diagnostic_passed = not blocking_reason_codes
    residual_summary = dict(allowed_order_quality_report["residual_blocker_summary"])
    residual_future_work = {
        "exposure_policy_review_required_count": int(
            allowed_order_quality_report["deterministic_recommendation_counts"].get(
                "needs_exposure_policy_review",
                0,
            )
        ),
        "p_up_action_rank_review_required_count": int(
            allowed_order_quality_report["deterministic_recommendation_counts"].get(
                "needs_p_up_action_rank_review",
                0,
            )
        ),
        "time_to_close_policy_review_required_count": int(
            allowed_order_quality_report["deterministic_recommendation_counts"].get(
                "needs_time_to_close_policy_review",
                0,
            )
        ),
        "keep_blocked_count": int(
            allowed_order_quality_report["deterministic_recommendation_counts"].get(
                "keep_blocked",
                0,
            )
        ),
        "residual_blocker_summary": residual_summary,
        "primary_deterministic_recommendation_counts": dict(
            allowed_order_quality_report[
                "primary_deterministic_recommendation_counts"
            ]
        ),
    }
    report = {
        "schema_version": O_V8_EXECUTION_POLICY_READINESS_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_policy_readiness",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "simulated_order_replay_report_id": simulated_order_replay_report[
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "allowed_order_quality_report_id": allowed_order_quality_report[
            "o_v8_execution_allowed_order_quality_report_id"
        ],
        "analysis_source": "simulated_replay_and_allowed_order_quality_reports_only",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "min_allowed_order_count": min_allowed_order_count,
        "allowed_order_count": len(allowed_rows),
        "blocked_decision_count": int(
            allowed_order_quality_report["blocked_decision_count"]
        ),
        "execution_policy_readiness_diagnostic_passed": diagnostic_passed,
        "execution_policy_readiness_required_checks": required_checks,
        "execution_policy_readiness_blocking_reason_codes": blocking_reason_codes,
        "allowed_order_quality_summary": {
            "action_distribution": dict(
                allowed_order_quality_report["allowed_order_action_distribution"]
            ),
            "family_distribution": dict(
                allowed_order_quality_report["allowed_order_family_distribution"]
            ),
            "side_distribution": dict(
                allowed_order_quality_report["allowed_order_side_distribution"]
            ),
            "origin_distribution": dict(
                allowed_order_quality_report["allowed_order_origin_distribution"]
            ),
            "p_up_agreement_distribution": dict(
                allowed_order_quality_report[
                    "allowed_order_p_up_agreement_distribution"
                ]
            ),
            "metric_summary": dict(
                allowed_order_quality_report["allowed_order_metric_summary"]
            ),
        },
        "microstructure_quality_summary": microstructure_quality,
        "exposure_quality_summary": exposure_quality,
        "runtime_integrity_summary": {
            "missing_runtime_field_row_count": len(missing_runtime_field_rows),
            "provenance_violation_count": len(provenance_violations),
            "provenance_valid": not provenance_violations,
        },
        "residual_future_work_summary": residual_future_work,
        "future_explicit_execution_handoff_gate_required": True,
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": [
            "future_explicit_execution_handoff_gate_required",
            "paper_live_unlock_prohibited",
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_execution_policy_readiness": True,
        "no_source_freeze_unlock_from_execution_policy_readiness": True,
        **safety_flags,
    }
    report["o_v8_execution_policy_readiness_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_execution_policy_check(
    *,
    passed: bool,
    observed: Any,
    required: Any,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "reason_code": reason_code,
    }


def _v8_allowed_order_microstructure_readiness(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    for row in rows:
        reason_codes = []
        spread = _optional_float(row.get("spread_bps"))
        staleness = _optional_float(row.get("book_staleness_ms"))
        queue = _optional_float(row.get("queue_fill_proxy"))
        time_to_close = _optional_float(row.get("time_to_close_seconds"))
        if spread is None or spread > float(config["max_spread_bps"]):
            reason_codes.append("allowed_order_spread_not_within_limit")
        if staleness is None or staleness > float(config["max_book_staleness_ms"]):
            reason_codes.append("allowed_order_book_staleness_not_within_limit")
        if queue is None or queue < float(config["min_queue_fill"]):
            reason_codes.append("allowed_order_queue_fill_below_minimum")
        min_time_to_close = (
            float(config["min_hts_time_to_close_seconds"])
            if row.get("execution_guarded_family") == "HOLD_TO_SETTLEMENT"
            else float(config["min_time_to_close_seconds"])
        )
        if time_to_close is None or time_to_close < min_time_to_close:
            reason_codes.append("allowed_order_time_to_close_below_minimum")
        if reason_codes:
            failures.append(
                {
                    "decision_group_id": row.get("decision_group_id"),
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "execution_guarded_action": row.get("execution_guarded_action"),
                    "reason_codes": reason_codes,
                    "spread_bps": spread,
                    "book_staleness_ms": staleness,
                    "queue_fill_proxy": queue,
                    "time_to_close_seconds": time_to_close,
                    "min_time_to_close_seconds": min_time_to_close,
                }
            )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failure_rows": failures,
        "limits": {
            "max_spread_bps": config["max_spread_bps"],
            "max_book_staleness_ms": config["max_book_staleness_ms"],
            "min_queue_fill": config["min_queue_fill"],
            "min_time_to_close_seconds": config["min_time_to_close_seconds"],
            "min_hts_time_to_close_seconds": config[
                "min_hts_time_to_close_seconds"
            ],
        },
    }


def _v8_allowed_order_exposure_readiness(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    for row in rows:
        reason_codes = []
        proposed_size = _optional_float(row.get("proposed_order_size")) or 0.0
        uncapped_size = _optional_float(row.get("uncapped_proposed_order_size")) or 0.0
        pre = dict(row.get("pre_decision_exposure") or {})
        post = dict(row.get("post_decision_exposure") or {})
        post_total = _optional_float(post.get("current_total_exposure")) or 0.0
        post_market = _optional_float(post.get("current_market_exposure")) or 0.0
        post_side = _optional_float(post.get("current_side_exposure")) or 0.0
        pre_total = _optional_float(pre.get("current_total_exposure")) or 0.0
        if proposed_size <= 0.0:
            reason_codes.append("allowed_order_proposed_size_not_positive")
        if proposed_size > float(config["max_order_size"]):
            reason_codes.append("allowed_order_size_above_max_order_size")
        if uncapped_size > float(config["max_order_size"]):
            reason_codes.append("allowed_order_uncapped_size_above_max_order_size")
        if post_total > float(config["max_total_exposure"]):
            reason_codes.append("allowed_order_total_exposure_above_limit")
        if post_market > float(config["max_market_exposure"]):
            reason_codes.append("allowed_order_market_exposure_above_limit")
        if post_side > float(config["max_side_exposure"]):
            reason_codes.append("allowed_order_side_exposure_above_limit")
        if post_total < pre_total:
            reason_codes.append("allowed_order_post_total_exposure_decreased")
        if reason_codes:
            failures.append(
                {
                    "decision_group_id": row.get("decision_group_id"),
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "execution_guarded_action": row.get("execution_guarded_action"),
                    "reason_codes": reason_codes,
                    "proposed_order_size": proposed_size,
                    "uncapped_proposed_order_size": uncapped_size,
                    "pre_decision_exposure": pre,
                    "post_decision_exposure": post,
                }
            )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failure_rows": failures,
        "limits": {
            "max_order_size": config["max_order_size"],
            "max_total_exposure": config["max_total_exposure"],
            "max_market_exposure": config["max_market_exposure"],
            "max_side_exposure": config["max_side_exposure"],
        },
    }


def _v8_execution_handoff_gate_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    policy_readiness_report: dict[str, Any],
    allowed_order_quality_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
    runtime_field_coverage_report: dict[str, Any],
    guard_block_analysis_report: dict[str, Any],
) -> dict[str, Any]:
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    allowed_rows = list(allowed_order_quality_report.get("allowed_order_quality_rows") or [])
    policy_checks = dict(
        policy_readiness_report.get("execution_policy_readiness_required_checks") or {}
    )
    runtime_integrity = dict(
        policy_readiness_report.get("runtime_integrity_summary") or {}
    )
    final_exposure = dict(simulated_order_replay_report.get("final_exposure") or {})
    provenance_summary = dict(
        runtime_field_coverage_report.get(
            "runtime_field_backfill_provenance_validity_summary"
        )
        or {}
    )
    no_source_score_mutation = not any(
        row.get("source_score_mutated") is True
        or row.get("o_model_predicted_score_mutated") is True
        for row in replay_rows
    )
    no_paper_live_capital_flags = _v8_reports_preserve_compact_safety_flags(
        policy_readiness_report,
        allowed_order_quality_report,
        simulated_order_replay_report,
        runtime_field_coverage_report,
        guard_block_analysis_report,
    )
    source_freeze_promotion_blocked = (
        _v8_report_gate_flag(policy_readiness_report, "source_model_candidate_eligible")
        is False
        and _v8_report_gate_flag(policy_readiness_report, "freeze_ready") is False
        and _v8_report_gate_flag(policy_readiness_report, "promotion_evidence_eligible")
        is False
        and _v8_report_gate_flag(policy_readiness_report, "#134_resume_allowed")
        is False
        and _v8_report_gate_flag(policy_readiness_report, "#146_start_allowed")
        is False
        and _v8_report_gate_flag(
            simulated_order_replay_report,
            "source_model_candidate_eligible",
        )
        is False
        and _v8_report_gate_flag(simulated_order_replay_report, "freeze_ready") is False
        and _v8_report_gate_flag(
            simulated_order_replay_report,
            "promotion_evidence_eligible",
        )
        is False
        and _v8_report_gate_flag(simulated_order_replay_report, "#134_resume_allowed")
        is False
        and _v8_report_gate_flag(simulated_order_replay_report, "#146_start_allowed")
        is False
    )
    required_checks = {
        "policy_readiness_diagnostic_passed": _v8_execution_handoff_gate_check(
            passed=bool(
                policy_readiness_report.get(
                    "execution_policy_readiness_diagnostic_passed"
                )
            ),
            observed=policy_readiness_report.get(
                "execution_policy_readiness_diagnostic_passed"
            ),
            required=True,
            reason_code="execution_handoff_policy_readiness_not_passed",
            source_report="o_v8_execution_policy_readiness_report",
        ),
        "min_allowed_order_count_met": _v8_execution_handoff_gate_check(
            passed=_v8_policy_check_passed(policy_checks, "min_allowed_order_count"),
            observed={
                "allowed_order_count": allowed_order_quality_report.get(
                    "allowed_order_count"
                ),
                "min_allowed_order_count": policy_readiness_report.get(
                    "min_allowed_order_count"
                ),
            },
            required="allowed_order_count >= min_allowed_order_count",
            reason_code="execution_handoff_min_allowed_order_count_not_met",
            source_report="o_v8_execution_policy_readiness_report",
        ),
        "zero_missing_runtime_fields": _v8_execution_handoff_gate_check(
            passed=(
                int(
                    runtime_field_coverage_report.get(
                        "missing_runtime_field_decision_count",
                        0,
                    )
                    or 0
                )
                == 0
                and int(runtime_integrity.get("missing_runtime_field_row_count") or 0)
                == 0
            ),
            observed={
                "missing_runtime_field_decision_count": int(
                    runtime_field_coverage_report.get(
                        "missing_runtime_field_decision_count",
                        0,
                    )
                    or 0
                ),
                "missing_runtime_field_row_count": int(
                    runtime_integrity.get("missing_runtime_field_row_count") or 0
                ),
            },
            required="0 missing runtime fields",
            reason_code="execution_handoff_runtime_missing_fields_present",
            source_report="o_v8_execution_runtime_field_coverage_report",
        ),
        "zero_provenance_violations": _v8_execution_handoff_gate_check(
            passed=(
                int(provenance_summary.get("provenance_violation_count") or 0) == 0
                and int(runtime_integrity.get("provenance_violation_count") or 0) == 0
                and provenance_summary.get("provenance_valid") is not False
            ),
            observed={
                "runtime_field_coverage_provenance_violation_count": int(
                    provenance_summary.get("provenance_violation_count") or 0
                ),
                "policy_readiness_provenance_violation_count": int(
                    runtime_integrity.get("provenance_violation_count") or 0
                ),
                "runtime_field_coverage_provenance_valid": provenance_summary.get(
                    "provenance_valid"
                ),
            },
            required="0 provenance violations and provenance_valid=true",
            reason_code="execution_handoff_provenance_violations_present",
            source_report="o_v8_execution_runtime_field_coverage_report",
        ),
        "all_allowed_orders_p_up_agreement": _v8_execution_handoff_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "all_allowed_orders_p_up_agreement",
            ),
            observed=allowed_order_quality_report.get(
                "allowed_order_p_up_agreement_distribution"
            ),
            required="all allowed orders p_up_agrees",
            reason_code="execution_handoff_allowed_order_p_up_disagreement_present",
            source_report="o_v8_execution_policy_readiness_report",
        ),
        "allowed_orders_origin_safe": _v8_execution_handoff_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "all_allowed_orders_original_or_safe_downgrade",
            ),
            observed=allowed_order_quality_report.get(
                "allowed_order_origin_distribution"
            ),
            required="original_selected_action or explicitly safe downgrade",
            reason_code="execution_handoff_allowed_order_origin_not_safe",
            source_report="o_v8_execution_policy_readiness_report",
        ),
        "allowed_order_microstructure_quality_passed": (
            _v8_execution_handoff_gate_check(
                passed=_v8_policy_check_passed(
                    policy_checks,
                    "allowed_order_microstructure_quality_passed",
                ),
                observed=policy_readiness_report.get(
                    "microstructure_quality_summary",
                    {},
                ).get("failure_count"),
                required=0,
                reason_code=(
                    "execution_handoff_allowed_order_microstructure_quality_failed"
                ),
                source_report="o_v8_execution_policy_readiness_report",
            )
        ),
        "allowed_order_exposure_within_limits": _v8_execution_handoff_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "allowed_order_exposure_within_limits",
            ),
            observed=policy_readiness_report.get(
                "exposure_quality_summary",
                {},
            ).get("failure_count"),
            required=0,
            reason_code="execution_handoff_allowed_order_exposure_limit_failed",
            source_report="o_v8_execution_policy_readiness_report",
        ),
        "runtime_state_validation_passed": _v8_execution_handoff_gate_check(
            passed=final_exposure.get("runtime_state_validation_passed") is True,
            observed=final_exposure.get("runtime_state_validation_passed"),
            required=True,
            reason_code="execution_handoff_runtime_state_validation_failed",
            source_report="o_v8_execution_simulated_order_replay_report",
        ),
        "simulated_runtime_risk_control_validation_passed": (
            _v8_execution_handoff_gate_check(
                passed=simulated_order_replay_report.get(
                    "runtime_risk_control_validation_passed"
                )
                is True,
                observed=simulated_order_replay_report.get(
                    "runtime_risk_control_validation_passed"
                ),
                required=True,
                reason_code=(
                    "execution_handoff_runtime_risk_control_validation_failed"
                ),
                source_report="o_v8_execution_simulated_order_replay_report",
            )
        ),
        "no_model_layer_regret_risk_selection_enabled": (
            _v8_execution_handoff_gate_check(
                passed=True,
                observed=False,
                required=False,
                reason_code=(
                    "execution_handoff_model_layer_regret_risk_selection_enabled"
                ),
                source_report="static_v8_execution_layer_design_constraint",
            )
        ),
        "no_source_score_mutation": _v8_execution_handoff_gate_check(
            passed=no_source_score_mutation,
            observed={
                "mutated_row_count": sum(
                    1
                    for row in replay_rows
                    if row.get("source_score_mutated") is True
                    or row.get("o_model_predicted_score_mutated") is True
                ),
                "allowed_order_count": len(allowed_rows),
            },
            required=0,
            reason_code="execution_handoff_source_score_mutation_detected",
            source_report="o_v8_execution_simulated_order_replay_report",
        ),
        "no_paper_live_write_or_capital_flags": _v8_execution_handoff_gate_check(
            passed=no_paper_live_capital_flags,
            observed=_v8_compact_safety_flag_summary(
                policy_readiness_report,
                allowed_order_quality_report,
                simulated_order_replay_report,
                runtime_field_coverage_report,
                guard_block_analysis_report,
            ),
            required={
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            },
            reason_code="execution_handoff_paper_live_or_capital_flag_enabled",
            source_report="all_execution_diagnostic_reports",
        ),
        "source_freeze_promotion_remain_blocked": _v8_execution_handoff_gate_check(
            passed=source_freeze_promotion_blocked,
            observed={
                "source_model_candidate_eligible": policy_readiness_report.get(
                    "source_model_candidate_eligible"
                ),
                "freeze_ready": policy_readiness_report.get("freeze_ready"),
                "promotion_evidence_eligible": policy_readiness_report.get(
                    "promotion_evidence_eligible"
                ),
                "#134_resume_allowed": policy_readiness_report.get(
                    "#134_resume_allowed"
                ),
                "#146_start_allowed": policy_readiness_report.get(
                    "#146_start_allowed"
                ),
            },
            required={
                "source_model_candidate_eligible": False,
                "freeze_ready": False,
                "promotion_evidence_eligible": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            reason_code="execution_handoff_source_freeze_or_promotion_unblocked",
            source_report="all_execution_diagnostic_reports",
        ),
        "future_unseen_holdout_required": _v8_execution_handoff_gate_check(
            passed=True,
            observed=True,
            required=True,
            reason_code="execution_handoff_future_unseen_holdout_not_required",
            source_report="static_v8_execution_layer_design_constraint",
        ),
    }
    blocking_reason_codes = sorted(
        check["reason_code"]
        for check in required_checks.values()
        if check["passed"] is not True
    )
    gate_passed = not blocking_reason_codes
    report = {
        "schema_version": O_V8_EXECUTION_HANDOFF_GATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_handoff_gate",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "policy_readiness_report_id": policy_readiness_report[
            "o_v8_execution_policy_readiness_report_id"
        ],
        "allowed_order_quality_report_id": allowed_order_quality_report[
            "o_v8_execution_allowed_order_quality_report_id"
        ],
        "simulated_order_replay_report_id": simulated_order_replay_report[
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "runtime_field_coverage_report_id": runtime_field_coverage_report[
            "o_v8_execution_runtime_field_coverage_report_id"
        ],
        "guard_block_analysis_report_id": guard_block_analysis_report[
            "o_v8_execution_guard_block_analysis_report_id"
        ],
        "analysis_source": (
            "policy_readiness_allowed_quality_simulated_replay_runtime_coverage_"
            "and_guard_block_analysis_reports_only"
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "explicit_execution_handoff_gate_mode": "diagnostic_only_fail_closed",
        "explicit_execution_handoff_gate_passed": gate_passed,
        "explicit_execution_handoff_required_checks": required_checks,
        "explicit_execution_handoff_blocking_reason_codes": blocking_reason_codes,
        "future_unseen_holdout_required": True,
        "future_paper_candidate_gate_required": True,
        "future_paper_candidate_gate_reason_codes": [
            "future_unseen_holdout_required_before_execution_handoff",
            "paper_candidate_gate_not_implemented",
        ],
        "allowed_order_count": int(
            allowed_order_quality_report.get("allowed_order_count") or 0
        ),
        "blocked_decision_count": int(
            allowed_order_quality_report.get("blocked_decision_count") or 0
        ),
        "residual_blocker_summary": dict(
            allowed_order_quality_report.get("residual_blocker_summary") or {}
        ),
        "safe_order_discovery_summary": dict(
            guard_block_analysis_report.get("safe_order_discovery_summary") or {}
        ),
        "runtime_integrity_summary": {
            "missing_runtime_field_decision_count": int(
                runtime_field_coverage_report.get(
                    "missing_runtime_field_decision_count",
                    0,
                )
                or 0
            ),
            "provenance_violation_count": int(
                provenance_summary.get("provenance_violation_count") or 0
            ),
            "provenance_valid": provenance_summary.get("provenance_valid"),
            "runtime_state_validation_passed": final_exposure.get(
                "runtime_state_validation_passed"
            ),
            "runtime_risk_control_validation_passed": (
                simulated_order_replay_report.get(
                    "runtime_risk_control_validation_passed"
                )
            ),
        },
        "handoff_design_notes": [
            "explicit_gate_is_diagnostic_only_and_cannot_enable_execution_handoff",
            "future_unseen_holdout_required_before_paper_or_live_execution",
            "future_paper_candidate_gate_required_before_any_order_handoff",
        ],
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": [
            "diagnostic_only_fail_closed",
            "future_unseen_holdout_required",
            "future_paper_candidate_gate_required",
            "paper_live_unlock_prohibited",
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_execution_handoff_gate": True,
        "no_source_freeze_unlock_from_execution_handoff_gate": True,
        **compact_safety_fields(),
    }
    report["o_v8_execution_handoff_gate_report_id"] = canonical_json_sha256(report)
    return report


def _v8_execution_handoff_gate_check(
    *,
    passed: bool,
    observed: Any,
    required: Any,
    reason_code: str,
    source_report: str,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "reason_code": reason_code,
        "source_report": source_report,
    }


def _v8_policy_check_passed(
    policy_checks: dict[str, Any],
    check_name: str,
) -> bool:
    check = policy_checks.get(check_name)
    if not isinstance(check, dict):
        return False
    return check.get("passed") is True


def _v8_report_gate_flag(report: dict[str, Any], field_name: str) -> Any:
    return report.get(field_name)


def _v8_compact_safety_flag_summary(
    *reports: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fields = compact_safety_fields()
    return {
        str(report.get("report_type") or f"report_{index}"): {
            field_name: report.get(field_name)
            for field_name in fields
        }
        for index, report in enumerate(reports)
    }


def _v8_reports_preserve_compact_safety_flags(
    *reports: dict[str, Any],
) -> bool:
    expected = compact_safety_fields()
    return all(
        report.get(field_name) is expected_value
        for report in reports
        for field_name, expected_value in expected.items()
    )


def _v8_future_unseen_holdout_plan_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    action_rank_handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
    allowed_order_quality_report: dict[str, Any],
    policy_readiness_report: dict[str, Any],
    handoff_gate_report: dict[str, Any],
    runtime_field_coverage_report: dict[str, Any],
    guard_block_analysis_report: dict[str, Any],
) -> dict[str, Any]:
    reports = {
        "action_rank_handoff": action_rank_handoff_report,
        "execution_risk_guard": execution_guard_report,
        "simulated_order_replay": simulated_order_replay_report,
        "allowed_order_quality": allowed_order_quality_report,
        "policy_readiness": policy_readiness_report,
        "execution_handoff_gate": handoff_gate_report,
        "runtime_field_coverage": runtime_field_coverage_report,
        "guard_block_analysis": guard_block_analysis_report,
    }
    report_ids = _v8_existing_diagnostic_report_ids(reports)
    policy_checks = dict(
        policy_readiness_report.get("execution_policy_readiness_required_checks") or {}
    )
    handoff_checks = dict(
        handoff_gate_report.get("explicit_execution_handoff_required_checks") or {}
    )
    guard_config_hash = execution_guard_report.get("execution_guard_config_hash")
    block_guard_config_hash = guard_block_analysis_report.get(
        "execution_guard_config_hash"
    )
    frozen_hashes_available = all(report_ids.values())
    o_config_fields = {
        "model_sha256": action_rank_handoff_report.get("model_sha256"),
        "split_hash": action_rank_handoff_report.get("split_hash"),
        "feature_schema_hash": action_rank_handoff_report.get("feature_schema_hash"),
        "handoff_contract_hash": action_rank_handoff_report.get(
            "handoff_contract_hash"
        ),
        "ranking_score_source": action_rank_handoff_report.get(
            "ranking_score_source"
        ),
    }
    cleanup_rule_counts = {
        "runtime_field_coverage": runtime_field_coverage_report.get(
            "applied_runtime_field_backfill_rule_counts"
        ),
        "simulated_order_replay": simulated_order_replay_report.get(
            "applied_runtime_field_backfill_rule_counts"
        ),
    }
    residual_blocker_summary = dict(
        allowed_order_quality_report.get("residual_blocker_summary") or {}
    )
    safe_order_discovery = dict(
        guard_block_analysis_report.get("safe_order_discovery_summary") or {}
    )
    input_outcome_safe = _v8_existing_diagnostic_reports_are_outcome_safe(
        *reports.values()
    )
    required_checks = {
        "unseen_date_window_definition": _v8_design_gate_check(
            passed=True,
            observed="future_holdout_run_must_declare_utc_start_end_and_market_family",
            required="strictly_later_unseen_window_declared_before_evaluation",
            reason_code="future_holdout_unseen_window_definition_missing",
            source_report="future_holdout_plan_static_requirement",
            current_evidence_satisfied=False,
        ),
        "no_overlap_with_shadow_validation_decisions": _v8_design_gate_check(
            passed=True,
            observed="future_holdout_must_hash_decision_group_ids_and_market_ids",
            required="zero_overlap_with_shadow_validation_decision_ids_and_markets",
            reason_code="future_holdout_overlap_rule_missing",
            source_report="future_holdout_plan_static_requirement",
            current_evidence_satisfied=False,
        ),
        "deterministic_report_hashes_frozen_before_holdout_evaluation": (
            _v8_design_gate_check(
                passed=frozen_hashes_available,
                observed=report_ids,
                required="all_input_diagnostic_report_ids_present_before_holdout",
                reason_code="future_holdout_frozen_report_hashes_missing",
                source_report="all_existing_v8_diagnostic_reports",
                current_evidence_satisfied=frozen_hashes_available,
            )
        ),
        "same_o_model_action_rank_config": _v8_design_gate_check(
            passed=all(o_config_fields.values()),
            observed=o_config_fields,
            required="model_sha256_split_hash_feature_schema_and_handoff_contract_fixed",
            reason_code="future_holdout_o_action_rank_config_not_frozen",
            source_report="o_v8_action_rank_handoff_report",
            current_evidence_satisfied=all(o_config_fields.values()),
        ),
        "same_execution_guard_config": _v8_design_gate_check(
            passed=bool(guard_config_hash)
            and bool(block_guard_config_hash)
            and guard_config_hash == block_guard_config_hash,
            observed={
                "execution_guard_config_hash": guard_config_hash,
                "block_analysis_execution_guard_config_hash": block_guard_config_hash,
            },
            required="same_execution_guard_config_hash_across_guard_and_block_analysis",
            reason_code="future_holdout_execution_guard_config_not_frozen",
            source_report="o_v8_execution_risk_guard_report",
            current_evidence_satisfied=bool(guard_config_hash)
            and guard_config_hash == block_guard_config_hash,
        ),
        "same_runtime_field_cleanup_backfill_rules": _v8_design_gate_check(
            passed=all(isinstance(value, dict) for value in cleanup_rule_counts.values()),
            observed=cleanup_rule_counts,
            required="runtime_field_cleanup_and_backfill_rule_counts_frozen",
            reason_code="future_holdout_runtime_field_cleanup_rules_not_frozen",
            source_report="o_v8_execution_runtime_field_coverage_report",
            current_evidence_satisfied=all(
                isinstance(value, dict) for value in cleanup_rule_counts.values()
            ),
        ),
        "same_simulated_ledger_rules": _v8_design_gate_check(
            passed=bool(simulated_order_replay_report.get("deterministic_replay_hash"))
            and simulated_order_replay_report.get(
                "runtime_risk_control_validation_passed"
            )
            is True,
            observed={
                "deterministic_replay_hash": simulated_order_replay_report.get(
                    "deterministic_replay_hash"
                ),
                "runtime_risk_control_validation_passed": (
                    simulated_order_replay_report.get(
                        "runtime_risk_control_validation_passed"
                    )
                ),
            },
            required="same_deterministic_replay_hash_inputs_and_runtime_ledger_rules",
            reason_code="future_holdout_simulated_ledger_rules_not_frozen",
            source_report="o_v8_execution_simulated_order_replay_report",
            current_evidence_satisfied=bool(
                simulated_order_replay_report.get("deterministic_replay_hash")
            ),
        ),
        "allowed_order_count_threshold": _v8_design_gate_check(
            passed=policy_readiness_report.get("min_allowed_order_count") is not None,
            observed={
                "current_allowed_order_count": allowed_order_quality_report.get(
                    "allowed_order_count"
                ),
                "min_allowed_order_count": policy_readiness_report.get(
                    "min_allowed_order_count"
                ),
            },
            required="future_holdout_allowed_order_count >= min_allowed_order_count",
            reason_code="future_holdout_allowed_order_count_threshold_not_defined",
            source_report="o_v8_execution_policy_readiness_report",
            current_evidence_satisfied=_v8_policy_check_passed(
                policy_checks,
                "min_allowed_order_count",
            ),
        ),
        "missing_runtime_fields_threshold": _v8_design_gate_check(
            passed=_v8_policy_check_passed(policy_checks, "zero_missing_runtime_fields")
            and int(
                runtime_field_coverage_report.get(
                    "missing_runtime_field_decision_count",
                    0,
                )
                or 0
            )
            == 0,
            observed={
                "missing_runtime_field_decision_count": runtime_field_coverage_report.get(
                    "missing_runtime_field_decision_count"
                ),
            },
            required="0 missing runtime fields",
            reason_code="future_holdout_missing_runtime_field_threshold_not_defined",
            source_report="o_v8_execution_runtime_field_coverage_report",
            current_evidence_satisfied=True,
        ),
        "provenance_violation_threshold": _v8_design_gate_check(
            passed=_v8_policy_check_passed(policy_checks, "zero_provenance_violations"),
            observed=runtime_field_coverage_report.get(
                "runtime_field_backfill_provenance_validity_summary"
            ),
            required="0 provenance violations",
            reason_code="future_holdout_provenance_threshold_not_defined",
            source_report="o_v8_execution_runtime_field_coverage_report",
            current_evidence_satisfied=_v8_policy_check_passed(
                policy_checks,
                "zero_provenance_violations",
            ),
        ),
        "p_up_agreement_requirement": _v8_design_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "all_allowed_orders_p_up_agreement",
            )
            and _v8_handoff_check_passed(
                handoff_checks,
                "all_allowed_orders_p_up_agreement",
            ),
            observed=allowed_order_quality_report.get(
                "allowed_order_p_up_agreement_distribution"
            ),
            required="all future holdout allowed orders must p_up_agree",
            reason_code="future_holdout_p_up_agreement_requirement_not_defined",
            source_report="o_v8_execution_policy_readiness_report",
            current_evidence_satisfied=True,
        ),
        "exposure_microstructure_pass_requirement": _v8_design_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "allowed_order_exposure_within_limits",
            )
            and _v8_policy_check_passed(
                policy_checks,
                "allowed_order_microstructure_quality_passed",
            ),
            observed={
                "exposure_quality_summary": policy_readiness_report.get(
                    "exposure_quality_summary"
                ),
                "microstructure_quality_summary": policy_readiness_report.get(
                    "microstructure_quality_summary"
                ),
            },
            required="future_holdout_allowed_orders_pass_exposure_and_microstructure",
            reason_code=(
                "future_holdout_exposure_microstructure_requirement_not_defined"
            ),
            source_report="o_v8_execution_policy_readiness_report",
            current_evidence_satisfied=True,
        ),
        "allowed_order_origin_safety_requirement": _v8_design_gate_check(
            passed=_v8_policy_check_passed(
                policy_checks,
                "all_allowed_orders_original_or_safe_downgrade",
            ),
            observed=allowed_order_quality_report.get(
                "allowed_order_origin_distribution"
            ),
            required="future_holdout_allowed_orders_original_or_explicit_safe_downgrade",
            reason_code="future_holdout_allowed_order_origin_requirement_not_defined",
            source_report="o_v8_execution_policy_readiness_report",
            current_evidence_satisfied=True,
        ),
        "residual_blocker_classification_requirement": _v8_design_gate_check(
            passed=bool(residual_blocker_summary)
            and "classification_counts" in safe_order_discovery,
            observed={
                "residual_blocker_summary": residual_blocker_summary,
                "safe_order_discovery_summary": safe_order_discovery,
            },
            required="future_holdout_residual_blockers_classified_by_reason",
            reason_code=(
                "future_holdout_residual_blocker_classification_not_defined"
            ),
            source_report="o_v8_execution_guard_block_analysis_report",
            current_evidence_satisfied="classification_counts" in safe_order_discovery,
        ),
        "input_reports_do_not_use_forbidden_outcomes": _v8_design_gate_check(
            passed=input_outcome_safe,
            observed=_v8_outcome_usage_summary(reports.values()),
            required="no_realized_pnl_labels_or_oracle_outcomes_for_tuning",
            reason_code="future_holdout_input_report_forbidden_outcome_usage_detected",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=input_outcome_safe,
        ),
    }
    blocking_reason_codes = sorted(
        check["reason_code"]
        for check in required_checks.values()
        if check["passed"] is not True
    )
    report = {
        "schema_version": O_V8_FUTURE_UNSEEN_HOLDOUT_PLAN_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_future_unseen_holdout_plan",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "input_diagnostic_report_ids": report_ids,
        "analysis_source": "existing_v8_diagnostic_reports_only",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "uses_source_model_training_internals_for_gate_tuning": False,
        "forbidden_outcome_fields_used": [],
        "future_unseen_holdout_plan_ready": not blocking_reason_codes,
        "future_unseen_holdout_required_checks": required_checks,
        "future_unseen_holdout_blocking_reason_codes": blocking_reason_codes,
        "future_unseen_holdout_required": True,
        "future_holdout_current_status": "not_evaluated_future_requirement",
        "future_holdout_current_unlock_blocking_reason_codes": [
            "future_unseen_holdout_not_yet_evaluated",
            "future_paper_candidate_gate_required",
            "manual_approval_not_recorded",
        ],
        "future_holdout_required_evidence_summary": {
            "strictly_later_unseen_window": True,
            "market_and_decision_disjoint_from_shadow_validation": True,
            "frozen_report_hashes_before_evaluation": True,
            "same_o_model_action_rank_config": True,
            "same_execution_guard_config": True,
            "same_runtime_field_cleanup_backfill_rules": True,
            "same_simulated_ledger_rules": True,
            "allowed_order_count_threshold": policy_readiness_report.get(
                "min_allowed_order_count"
            ),
            "missing_runtime_fields_threshold": 0,
            "provenance_violation_threshold": 0,
            "p_up_agreement_required": True,
            "exposure_microstructure_pass_required": True,
            "allowed_order_origin_safety_required": True,
            "residual_blocker_classification_required": True,
        },
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": [
            "future_unseen_holdout_required",
            "future_paper_candidate_gate_required",
            "paper_live_unlock_prohibited",
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_future_holdout_plan": True,
        "no_source_freeze_unlock_from_future_holdout_plan": True,
        **compact_safety_fields(),
    }
    report["o_v8_future_unseen_holdout_plan_report_id"] = canonical_json_sha256(report)
    return report


def _v8_paper_candidate_gate_design_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    action_rank_handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
    allowed_order_quality_report: dict[str, Any],
    policy_readiness_report: dict[str, Any],
    handoff_gate_report: dict[str, Any],
    runtime_field_coverage_report: dict[str, Any],
    guard_block_analysis_report: dict[str, Any],
    holdout_plan_report: dict[str, Any],
) -> dict[str, Any]:
    reports = {
        "action_rank_handoff": action_rank_handoff_report,
        "execution_risk_guard": execution_guard_report,
        "simulated_order_replay": simulated_order_replay_report,
        "allowed_order_quality": allowed_order_quality_report,
        "policy_readiness": policy_readiness_report,
        "execution_handoff_gate": handoff_gate_report,
        "runtime_field_coverage": runtime_field_coverage_report,
        "guard_block_analysis": guard_block_analysis_report,
        "future_unseen_holdout_plan": holdout_plan_report,
    }
    report_ids = _v8_existing_diagnostic_report_ids(reports)
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    provenance_summary = dict(
        runtime_field_coverage_report.get(
            "runtime_field_backfill_provenance_validity_summary"
        )
        or {}
    )
    no_source_score_mutation = not any(
        row.get("source_score_mutated") is True
        or row.get("o_model_predicted_score_mutated") is True
        for row in replay_rows
    )
    no_model_layer_risk = (
        action_rank_handoff_report.get("model_layer_regret_risk_selection_enabled")
        is False
        and execution_guard_report.get("model_layer_regret_risk_selection_enabled")
        is False
        and execution_guard_report.get("trains_regret_model") is False
        and execution_guard_report.get("trains_risk_head") is False
    )
    no_forbidden_outcomes = _v8_existing_diagnostic_reports_are_outcome_safe(
        *reports.values()
    )
    safety_flags_preserved = _v8_reports_preserve_compact_safety_flags(
        action_rank_handoff_report,
        execution_guard_report,
        simulated_order_replay_report,
        allowed_order_quality_report,
        policy_readiness_report,
        handoff_gate_report,
        runtime_field_coverage_report,
        guard_block_analysis_report,
        holdout_plan_report,
    )
    source_freeze_separate = (
        holdout_plan_report.get("source_model_candidate_eligible") is False
        and holdout_plan_report.get("freeze_ready") is False
        and holdout_plan_report.get("promotion_evidence_eligible") is False
        and action_rank_handoff_report.get("strict_source_gate_remains_failed") is True
    )
    required_checks = {
        "future_unseen_holdout_passed": _v8_design_gate_check(
            passed=holdout_plan_report.get("future_unseen_holdout_plan_ready") is True,
            observed={
                "future_unseen_holdout_plan_ready": holdout_plan_report.get(
                    "future_unseen_holdout_plan_ready"
                ),
                "future_holdout_current_status": holdout_plan_report.get(
                    "future_holdout_current_status"
                ),
            },
            required="future_unseen_holdout_must_pass_before_paper_candidate",
            reason_code="paper_candidate_future_unseen_holdout_plan_not_ready",
            source_report="o_v8_future_unseen_holdout_plan_report",
            current_evidence_satisfied=False,
        ),
        "explicit_execution_handoff_gate_passed_on_holdout": _v8_design_gate_check(
            passed=bool(
                handoff_gate_report.get("explicit_execution_handoff_required_checks")
            )
            and handoff_gate_report.get("explicit_execution_handoff_gate_mode")
            == "diagnostic_only_fail_closed",
            observed={
                "current_explicit_execution_handoff_gate_passed": (
                    handoff_gate_report.get("explicit_execution_handoff_gate_passed")
                ),
                "explicit_execution_handoff_gate_mode": handoff_gate_report.get(
                    "explicit_execution_handoff_gate_mode"
                ),
            },
            required="explicit_execution_handoff_gate_passed_on_future_holdout",
            reason_code="paper_candidate_execution_handoff_gate_design_not_ready",
            source_report="o_v8_execution_handoff_gate_report",
            current_evidence_satisfied=handoff_gate_report.get(
                "explicit_execution_handoff_gate_passed"
            )
            is True,
        ),
        "zero_source_score_mutation": _v8_design_gate_check(
            passed=no_source_score_mutation,
            observed={
                "mutated_row_count": sum(
                    1
                    for row in replay_rows
                    if row.get("source_score_mutated") is True
                    or row.get("o_model_predicted_score_mutated") is True
                )
            },
            required=0,
            reason_code="paper_candidate_source_score_mutation_detected",
            source_report="o_v8_execution_simulated_order_replay_report",
            current_evidence_satisfied=no_source_score_mutation,
        ),
        "zero_forbidden_outcome_field_usage": _v8_design_gate_check(
            passed=no_forbidden_outcomes,
            observed=_v8_outcome_usage_summary(reports.values()),
            required="no_realized_pnl_labels_or_oracle_fields_for_gate_tuning",
            reason_code="paper_candidate_forbidden_outcome_usage_detected",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=no_forbidden_outcomes,
        ),
        "zero_provenance_violations": _v8_design_gate_check(
            passed=int(provenance_summary.get("provenance_violation_count") or 0) == 0,
            observed=provenance_summary,
            required="0 provenance violations",
            reason_code="paper_candidate_provenance_violations_present",
            source_report="o_v8_execution_runtime_field_coverage_report",
            current_evidence_satisfied=int(
                provenance_summary.get("provenance_violation_count") or 0
            )
            == 0,
        ),
        "no_model_layer_regret_risk_selection_enabled": _v8_design_gate_check(
            passed=no_model_layer_risk,
            observed={
                "action_rank_model_layer_regret_risk_selection_enabled": (
                    action_rank_handoff_report.get(
                        "model_layer_regret_risk_selection_enabled"
                    )
                ),
                "execution_guard_model_layer_regret_risk_selection_enabled": (
                    execution_guard_report.get(
                        "model_layer_regret_risk_selection_enabled"
                    )
                ),
                "trains_regret_model": execution_guard_report.get(
                    "trains_regret_model"
                ),
                "trains_risk_head": execution_guard_report.get("trains_risk_head"),
            },
            required=False,
            reason_code="paper_candidate_model_layer_regret_risk_selection_enabled",
            source_report="o_v8_action_rank_handoff_report",
            current_evidence_satisfied=no_model_layer_risk,
        ),
        "source_freeze_promotion_gates_remain_separate": _v8_design_gate_check(
            passed=source_freeze_separate,
            observed={
                "strict_source_gate_remains_failed": action_rank_handoff_report.get(
                    "strict_source_gate_remains_failed"
                ),
                "source_model_candidate_eligible": holdout_plan_report.get(
                    "source_model_candidate_eligible"
                ),
                "freeze_ready": holdout_plan_report.get("freeze_ready"),
                "promotion_evidence_eligible": holdout_plan_report.get(
                    "promotion_evidence_eligible"
                ),
            },
            required="paper_candidate_gate_does_not_change_source_freeze_promotion",
            reason_code="paper_candidate_source_freeze_promotion_gate_not_separate",
            source_report="o_v8_action_rank_handoff_report",
            current_evidence_satisfied=source_freeze_separate,
        ),
        "paper_only_flags_enforced": _v8_design_gate_check(
            passed=safety_flags_preserved
            and holdout_plan_report.get("paper_only") is True,
            observed=_v8_compact_safety_flag_summary(*reports.values()),
            required={"paper_only": True},
            reason_code="paper_candidate_paper_only_flag_not_enforced",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=safety_flags_preserved,
        ),
        "wallet_signing_disabled": _v8_design_gate_check(
            passed=safety_flags_preserved
            and all(report.get("wallet_signing_enabled") is False for report in reports.values()),
            observed=_v8_compact_safety_flag_summary(*reports.values()),
            required={"wallet_signing_enabled": False},
            reason_code="paper_candidate_wallet_signing_enabled",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=safety_flags_preserved,
        ),
        "polymarket_writes_disabled": _v8_design_gate_check(
            passed=safety_flags_preserved
            and all(report.get("polymarket_write_enabled") is False for report in reports.values()),
            observed=_v8_compact_safety_flag_summary(*reports.values()),
            required={"polymarket_write_enabled": False},
            reason_code="paper_candidate_polymarket_writes_enabled",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=safety_flags_preserved,
        ),
        "capital_at_risk_false": _v8_design_gate_check(
            passed=safety_flags_preserved
            and all(report.get("capital_at_risk") is False for report in reports.values()),
            observed=_v8_compact_safety_flag_summary(*reports.values()),
            required={"capital_at_risk": False},
            reason_code="paper_candidate_capital_at_risk_enabled",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=safety_flags_preserved,
        ),
        "explicit_manual_approval_required": _v8_design_gate_check(
            passed=True,
            observed="manual_approval_not_recorded_for_current_diagnostic_run",
            required="explicit_manual_approval_record_required_before_paper_candidate",
            reason_code="paper_candidate_manual_approval_requirement_missing",
            source_report="paper_candidate_gate_design_static_requirement",
            current_evidence_satisfied=False,
        ),
        "input_report_hashes_available": _v8_design_gate_check(
            passed=all(report_ids.values()),
            observed=report_ids,
            required="all_input_report_ids_available",
            reason_code="paper_candidate_input_report_hashes_missing",
            source_report="all_existing_v8_diagnostic_reports",
            current_evidence_satisfied=all(report_ids.values()),
        ),
    }
    blocking_reason_codes = sorted(
        check["reason_code"]
        for check in required_checks.values()
        if check["passed"] is not True
    )
    report = {
        "schema_version": O_V8_PAPER_CANDIDATE_GATE_DESIGN_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_paper_candidate_gate_design",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "input_diagnostic_report_ids": report_ids,
        "future_unseen_holdout_plan_report_id": holdout_plan_report[
            "o_v8_future_unseen_holdout_plan_report_id"
        ],
        "analysis_source": "existing_v8_diagnostic_reports_only",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "uses_source_model_training_internals_for_gate_tuning": False,
        "forbidden_outcome_fields_used": [],
        "paper_candidate_gate_design_ready": not blocking_reason_codes,
        "paper_candidate_required_checks": required_checks,
        "paper_candidate_gate_blocking_reason_codes": blocking_reason_codes,
        "future_unseen_holdout_required": True,
        "future_paper_candidate_gate_required": True,
        "paper_candidate_allowed": False,
        "paper_candidate_current_unlock_blocking_reason_codes": [
            "future_unseen_holdout_not_yet_evaluated",
            "future_paper_candidate_gate_design_only",
            "explicit_manual_approval_not_recorded",
            "paper_live_unlock_prohibited",
        ],
        "future_paper_candidate_required_evidence_summary": {
            "future_unseen_holdout_passed": True,
            "explicit_execution_handoff_gate_passed_on_holdout": True,
            "zero_source_score_mutation": True,
            "zero_forbidden_outcome_field_usage": True,
            "zero_provenance_violations": True,
            "no_model_layer_regret_risk_selection_enabled": True,
            "source_freeze_promotion_gates_remain_separate": True,
            "paper_only_flags_enforced": True,
            "wallet_signing_disabled": True,
            "polymarket_writes_disabled": True,
            "capital_at_risk_false": True,
            "explicit_manual_approval_required": True,
        },
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": [
            "future_unseen_holdout_required",
            "future_paper_candidate_gate_required",
            "paper_live_unlock_prohibited",
        ],
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_paper_candidate_gate_design": True,
        "no_source_freeze_unlock_from_paper_candidate_gate_design": True,
        **compact_safety_fields(),
    }
    report["o_v8_paper_candidate_gate_design_report_id"] = canonical_json_sha256(report)
    return report


def _v8_design_gate_check(
    *,
    passed: bool,
    observed: Any,
    required: Any,
    reason_code: str,
    source_report: str,
    current_evidence_satisfied: bool,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "reason_code": reason_code,
        "source_report": source_report,
        "current_evidence_satisfied": bool(current_evidence_satisfied),
    }


def _v8_handoff_check_passed(
    handoff_checks: dict[str, Any],
    check_name: str,
) -> bool:
    check = handoff_checks.get(check_name)
    if not isinstance(check, dict):
        return False
    return check.get("passed") is True


def _v8_existing_diagnostic_report_ids(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report_id_fields = {
        "action_rank_handoff": "o_v8_action_rank_handoff_report_id",
        "execution_risk_guard": "o_v8_execution_risk_guard_report_id",
        "simulated_order_replay": "o_v8_execution_simulated_order_replay_report_id",
        "allowed_order_quality": "o_v8_execution_allowed_order_quality_report_id",
        "policy_readiness": "o_v8_execution_policy_readiness_report_id",
        "execution_handoff_gate": "o_v8_execution_handoff_gate_report_id",
        "runtime_field_coverage": "o_v8_execution_runtime_field_coverage_report_id",
        "guard_block_analysis": "o_v8_execution_guard_block_analysis_report_id",
        "future_unseen_holdout_plan": "o_v8_future_unseen_holdout_plan_report_id",
    }
    return {
        name: report.get(report_id_fields[name])
        for name, report in reports.items()
        if name in report_id_fields
    }


def _v8_existing_diagnostic_reports_are_outcome_safe(
    *reports: dict[str, Any],
) -> bool:
    return all(
        report.get("uses_realized_pnl_or_labels_for_analysis") is not True
        and report.get("uses_oracle_actions_for_analysis") is not True
        and report.get("uses_validation_outcomes_for_tuning") is not True
        and report.get("uses_validation_realized_outcomes_for_guard_tuning") is not True
        and report.get("uses_replay_regret_labels_for_guard_tuning") is not True
        and report.get("safe_order_discovery_uses_realized_pnl") is not True
        and not report.get("forbidden_outcome_fields_used")
        for report in reports
    )


def _v8_outcome_usage_summary(
    reports: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(report.get("report_type") or f"report_{index}"): {
            "uses_realized_pnl_or_labels_for_analysis": report.get(
                "uses_realized_pnl_or_labels_for_analysis"
            ),
            "uses_oracle_actions_for_analysis": report.get(
                "uses_oracle_actions_for_analysis"
            ),
            "uses_validation_outcomes_for_tuning": report.get(
                "uses_validation_outcomes_for_tuning"
            ),
            "uses_validation_realized_outcomes_for_guard_tuning": report.get(
                "uses_validation_realized_outcomes_for_guard_tuning"
            ),
            "uses_replay_regret_labels_for_guard_tuning": report.get(
                "uses_replay_regret_labels_for_guard_tuning"
            ),
            "safe_order_discovery_uses_realized_pnl": report.get(
                "safe_order_discovery_uses_realized_pnl"
            ),
            "forbidden_outcome_fields_used": report.get(
                "forbidden_outcome_fields_used",
                [],
            ),
        }
        for index, report in enumerate(reports)
    }


def _v8_execution_guard_block_analysis_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    runtime_state_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
) -> dict[str, Any]:
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    blocked_rows = [row for row in replay_rows if row.get("order_allowed") is not True]
    analysis_rows = [_v8_execution_block_analysis_row(row) for row in blocked_rows]
    category_counts = Counter(
        category
        for row in analysis_rows
        for category in row["minimal_blocking_set"]["blocker_categories"]
    )
    discovery_counts = Counter(
        row["safe_order_discovery_classification"] for row in analysis_rows
    )
    safe_order_candidate_classifications = {
        "blocked_only_by_missing_runtime_fields",
        "blocked_only_by_configurable_thresholds",
        "would_pass_if_sbc_fallback_selected",
    }
    safe_order_candidate_count = sum(
        count
        for classification, count in discovery_counts.items()
        if classification in safe_order_candidate_classifications
    )
    backfill_summary = _v8_runtime_field_backfill_application_summary(replay_rows)
    fundamentally_unsafe_count = discovery_counts.get("fundamentally_unsafe", 0)
    why_zero = []
    if simulated_order_replay_report.get("simulated_allowed_order_count") == 0:
        why_zero = [
            "all_replay_decisions_blocked_by_execution_guard",
            *[
                f"primary_blocker_category:{category}"
                for category, _ in category_counts.most_common(3)
            ],
        ]
    report = {
        "schema_version": O_V8_EXECUTION_GUARD_BLOCK_ANALYSIS_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_guard_block_analysis",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "source_action_rank_signal_report_id": handoff_report[
            "o_v8_action_rank_handoff_report_id"
        ],
        "execution_guard_report_id": execution_guard_report[
            "o_v8_execution_risk_guard_report_id"
        ],
        "runtime_state_report_id": runtime_state_report[
            "o_v8_execution_runtime_state_report_id"
        ],
        "simulated_order_replay_report_id": simulated_order_replay_report[
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "analysis_source": "simulated_order_replay_rows_only",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "safe_order_discovery_uses_realized_pnl": False,
        "execution_guard_config_hash": canonical_json_sha256(
            _v8_execution_guard_config()
        ),
        "decision_count": len(replay_rows),
        "blocked_decision_count": len(blocked_rows),
        "allowed_decision_count": sum(
            1 for row in replay_rows if row.get("order_allowed") is True
        ),
        "why_simulated_allowed_order_count_zero": why_zero,
        "primary_blocker_categories": [
            category for category, _ in category_counts.most_common(5)
        ],
        "block_reason_category_counts": dict(sorted(category_counts.items())),
        "block_reason_summary_by_action": _v8_block_reason_summary_by(
            analysis_rows,
            "source_selected_action",
        ),
        "block_reason_summary_by_family": _v8_block_reason_summary_by(
            analysis_rows,
            "source_selected_family",
        ),
        "block_reason_summary_by_side": _v8_block_reason_summary_by(
            analysis_rows,
            "source_selected_side",
        ),
        "block_reason_summary_by_market": _v8_block_reason_summary_by(
            analysis_rows,
            "market_id",
        ),
        "block_reason_summary_by_time_to_close_bucket": _v8_block_reason_summary_by(
            analysis_rows,
            "time_to_close_bucket",
        ),
        "safe_order_discovery_summary": {
            "safe_order_candidate_count": safe_order_candidate_count,
            "blocked_only_by_missing_runtime_fields_count": discovery_counts.get(
                "blocked_only_by_missing_runtime_fields",
                0,
            ),
            "blocked_only_by_configurable_thresholds_count": discovery_counts.get(
                "blocked_only_by_configurable_thresholds",
                0,
            ),
            "would_pass_if_sbc_fallback_selected_count": discovery_counts.get(
                "would_pass_if_sbc_fallback_selected",
                0,
            ),
            "fundamentally_unsafe_count": fundamentally_unsafe_count,
            "classification_counts": dict(sorted(discovery_counts.items())),
        },
        "runtime_field_backfill_rules_applied": backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "applied_runtime_field_backfill_count": backfill_summary[
            "applied_runtime_field_backfill_count"
        ],
        "applied_runtime_field_backfill_rule_counts": backfill_summary[
            "applied_runtime_field_backfill_rule_counts"
        ],
        "runtime_field_backfill_provenance_validity_summary": backfill_summary[
            "runtime_field_backfill_provenance_validity_summary"
        ],
        "blocked_decision_analysis_rows": analysis_rows,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_block_analysis": True,
        "no_source_freeze_unlock_from_block_analysis": True,
        **compact_safety_fields(),
    }
    report["o_v8_execution_guard_block_analysis_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_execution_block_analysis_row(row: dict[str, Any]) -> dict[str, Any]:
    minimal_blocking_set = _v8_minimal_blocking_set(row)
    discovery = _v8_safe_order_discovery(row, minimal_blocking_set)
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": row.get("source_selected_action"),
        "source_selected_family": row.get("source_selected_family"),
        "source_selected_side": row.get("source_selected_side"),
        "execution_guarded_action": row.get("execution_guarded_action"),
        "execution_guarded_family": row.get("execution_guarded_family"),
        "execution_guarded_side": row.get("execution_guarded_side"),
        "source_model_score": row.get("source_model_score"),
        "execution_guarded_score": row.get("execution_guarded_score"),
        "order_allowed": bool(row.get("order_allowed")),
        "proposed_order_size": row.get("proposed_order_size"),
        "execution_blocking_reason_codes": list(
            row.get("execution_blocking_reason_codes") or []
        ),
        "execution_guard_reason_codes": list(
            row.get("execution_guard_reason_codes") or []
        ),
        "exposure_reason_codes": list(row.get("exposure_reason_codes") or []),
        "missing_runtime_field_codes": list(
            row.get("missing_runtime_field_codes") or []
        ),
        "runtime_field_backfill_rules_applied": bool(
            row.get("runtime_field_backfill_rules_applied")
        ),
        "runtime_field_applied_backfill_rows": list(
            row.get("runtime_field_applied_backfill_rows") or []
        ),
        "runtime_field_backfill_rule_counts": dict(
            row.get("runtime_field_backfill_rule_counts") or {}
        ),
        "runtime_field_backfill_provenance_valid": bool(
            row.get("runtime_field_backfill_provenance_valid")
        ),
        "runtime_field_backfill_provenance_violations": list(
            row.get("runtime_field_backfill_provenance_violations") or []
        ),
        "time_to_close_seconds": _v8_row_time_to_close_seconds(row),
        "time_to_close_bucket": _v8_time_to_close_bucket_from_decision(row),
        "p_up_action_disagreement": bool(row.get("p_up_action_disagreement")),
        "minimal_blocking_set": minimal_blocking_set,
        "safe_order_discovery_classification": discovery["classification"],
        "safe_order_discovery_reason_codes": discovery["reason_codes"],
        "source_score_mutated": bool(row.get("source_score_mutated")),
        "o_model_predicted_score_mutated": bool(
            row.get("o_model_predicted_score_mutated")
        ),
    }


def _v8_minimal_blocking_set(row: dict[str, Any]) -> dict[str, Any]:
    blocking_reasons = sorted(set(row.get("execution_blocking_reason_codes") or []))
    guard_reasons = sorted(set(row.get("execution_guard_reason_codes") or []))
    exposure_reasons = sorted(set(row.get("exposure_reason_codes") or []))
    missing_fields = sorted(set(row.get("missing_runtime_field_codes") or []))
    categorized: dict[str, list[str]] = {
        "missing_fields": missing_fields,
        "p_up_disagreement": [],
        "time_to_close": [],
        "spread_staleness_queue": [],
        "hts_guard_failure": [],
        "exposure_limits": [],
        "score_margin": [],
        "no_trade": [],
        "other": [],
    }
    for reason in [*blocking_reasons, *guard_reasons, *exposure_reasons]:
        category = _v8_blocking_category(reason)
        if category == "missing_runtime_fields":
            categorized["missing_fields"].append(reason)
        elif category == "p_up_disagreement":
            categorized["p_up_disagreement"].append(reason)
        elif category == "time_to_close":
            categorized["time_to_close"].append(reason)
        elif category in {"spread", "staleness", "liquidity"}:
            categorized["spread_staleness_queue"].append(reason)
        elif category == "hts_guard_failure":
            categorized["hts_guard_failure"].append(reason)
        elif category == "exposure_limits":
            categorized["exposure_limits"].append(reason)
        elif category == "score_margin":
            categorized["score_margin"].append(reason)
        elif category == "no_trade":
            categorized["no_trade"].append(reason)
        elif category not in {"allowed", "simulation"}:
            categorized["other"].append(reason)
    categorized = {
        key: sorted(set(values))
        for key, values in categorized.items()
    }
    blocker_categories = sorted(
        {
            key
            for key, values in categorized.items()
            if values
            and key
            not in {
                "score_margin",
                "no_trade",
            }
        }
    )
    return {
        **categorized,
        "blocker_categories": blocker_categories,
        "blocking_reason_codes": blocking_reasons,
        "guard_reason_codes": guard_reasons,
        "exposure_reason_codes": exposure_reasons,
    }


def _v8_safe_order_discovery(
    row: dict[str, Any],
    minimal_blocking_set: dict[str, Any],
) -> dict[str, Any]:
    if row.get("source_selected_action") == "NO_TRADE":
        return {
            "classification": "no_trade_not_order_candidate",
            "reason_codes": ["source_selected_no_trade_has_no_safe_order_path"],
        }
    categories = set(minimal_blocking_set.get("blocker_categories") or [])
    root_categories = set(categories)
    if "missing_fields" in root_categories:
        root_categories.discard("hts_guard_failure")
    if root_categories.intersection({"time_to_close", "spread_staleness_queue"}):
        root_categories.discard("hts_guard_failure")
    if not categories:
        return {
            "classification": "not_blocked_or_no_trade",
            "reason_codes": ["no_minimal_blocking_categories"],
        }
    if root_categories == {"missing_fields"}:
        return {
            "classification": "blocked_only_by_missing_runtime_fields",
            "reason_codes": ["safe_order_requires_runtime_field_backfill"],
        }
    threshold_categories = {"time_to_close", "spread_staleness_queue", "score_margin"}
    if root_categories.issubset(threshold_categories):
        return {
            "classification": "blocked_only_by_configurable_thresholds",
            "reason_codes": ["safe_order_requires_threshold_policy_review"],
        }
    if (
        "hts_guard_failure" in categories
        and "p_up_disagreement" not in categories
        and "exposure_limits" not in categories
        and "missing_fields" not in categories
    ):
        return {
            "classification": "would_pass_if_sbc_fallback_selected",
            "reason_codes": ["safe_order_requires_same_side_sbc_fallback_available"],
        }
    if categories.intersection({"p_up_disagreement", "exposure_limits"}):
        return {
            "classification": "fundamentally_unsafe",
            "reason_codes": sorted(
                {
                    "unsafe_due_to_p_up_disagreement"
                    if "p_up_disagreement" in categories
                    else "",
                    "unsafe_due_to_exposure_limit_or_position_conflict"
                    if "exposure_limits" in categories
                    else "",
                }
                - {""}
            ),
        }
    return {
        "classification": "mixed_blockers_require_data_and_policy_review",
        "reason_codes": sorted(f"blocker_category:{category}" for category in categories),
    }


def _v8_blocking_category(reason: str) -> str:
    if reason.startswith("missing_") or reason in {
        "execution_required_runtime_fields_missing",
        "execution_exposure_state_missing",
    }:
        return "missing_runtime_fields"
    if reason == "execution_p_up_side_disagreement":
        return "p_up_disagreement"
    if reason == "execution_time_to_close_unsafe":
        return "time_to_close"
    if reason == "execution_spread_too_wide":
        return "spread"
    if reason == "execution_book_stale":
        return "staleness"
    if reason == "execution_liquidity_too_weak":
        return "liquidity"
    if reason.startswith("execution_hts_"):
        return "hts_guard_failure"
    if reason in {
        "execution_duplicate_market_side_position",
        "execution_opposite_side_conflict",
        "execution_total_exposure_limit_reached",
        "execution_market_exposure_limit_reached",
        "execution_side_exposure_limit_reached",
        "execution_cooldown_active",
        "execution_order_size_capped_by_exposure_limit",
        "execution_order_size_capped_by_max_order_size",
    }:
        return "exposure_limits"
    if reason == "execution_score_margin_too_close":
        return "score_margin"
    if reason in {
        "execution_no_trade_selected",
        "execution_no_order_for_no_trade",
        "execution_no_exposure_delta_for_no_trade",
    }:
        return "no_trade"
    if reason in {
        "execution_simulated_order_allowed",
        "execution_simulated_runtime_state_valid",
    }:
        return "allowed"
    if reason == "execution_simulated_order_blocked":
        return "simulation"
    return "other"


def _v8_block_reason_summary_by(
    rows: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field_name) or "UNKNOWN")].append(row)
    summary = {}
    for value, group_rows in sorted(grouped.items()):
        category_counts = Counter(
            category
            for row in group_rows
            for category in row["minimal_blocking_set"]["blocker_categories"]
        )
        reason_counts = Counter(
            reason
            for row in group_rows
            for reason in row["execution_blocking_reason_codes"]
        )
        summary[value] = {
            "blocked_decision_count": len(group_rows),
            "blocker_category_counts": dict(sorted(category_counts.items())),
            "blocking_reason_counts": dict(sorted(reason_counts.items())),
        }
    return summary


def _v8_row_time_to_close_seconds(row: dict[str, Any]) -> float | None:
    runtime_limits = dict(row.get("runtime_limits") or {})
    top_k = list(row.get("top_k_action_ranking") or [])
    for candidate in top_k:
        microstructure = dict(candidate.get("microstructure_snapshot") or {})
        value = _optional_float(microstructure.get("time_to_close_seconds"))
        if value is not None:
            return value
    microstructure = dict(row.get("microstructure_snapshot") or {})
    value = _optional_float(microstructure.get("time_to_close_seconds"))
    if value is not None:
        return value
    return _optional_float(runtime_limits.get("time_to_close_seconds"))


def _v8_time_to_close_bucket_from_decision(row: dict[str, Any]) -> str:
    value = _v8_row_time_to_close_seconds(row)
    if value is None:
        return "missing"
    if value < 30:
        return "lt_30s"
    if value < 60:
        return "30s_to_60s"
    if value < 120:
        return "60s_to_120s"
    if value < 240:
        return "120s_to_240s"
    return "gte_240s"


def _v8_runtime_field_backfill_application_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    applied_rows = [
        applied
        for row in rows
        for applied in row.get("runtime_field_applied_backfill_rows", [])
    ]
    violations = [
        violation
        for row in rows
        for violation in row.get("runtime_field_backfill_provenance_violations", [])
    ]
    rule_counts = Counter(
        str(row.get("deterministic_rule_id") or "UNKNOWN") for row in applied_rows
    )
    application_type_counts = Counter(
        str(row.get("application_type") or "UNKNOWN") for row in applied_rows
    )
    field_counts = Counter(
        str(row.get("runtime_field_name") or "UNKNOWN") for row in applied_rows
    )
    valid_count = sum(1 for row in applied_rows if row.get("provenance_valid") is True)
    invalid_count = sum(
        1 for row in [*applied_rows, *violations] if row.get("provenance_valid") is False
    )
    return {
        "runtime_field_backfill_rules_applied": bool(applied_rows),
        "applied_runtime_field_backfill_count": len(applied_rows),
        "applied_runtime_field_backfill_rule_counts": dict(sorted(rule_counts.items())),
        "applied_runtime_field_backfill_application_type_counts": dict(
            sorted(application_type_counts.items())
        ),
        "applied_runtime_field_backfill_field_counts": dict(
            sorted(field_counts.items())
        ),
        "runtime_field_backfill_provenance_validity_summary": {
            "provenance_checked_count": len(applied_rows) + len(violations),
            "provenance_valid_count": valid_count,
            "provenance_invalid_count": invalid_count,
            "provenance_violation_count": len(violations),
            "provenance_valid": invalid_count == 0,
        },
        "runtime_field_backfill_provenance_violations": violations,
    }


def _v8_execution_runtime_field_coverage_report(
    *,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    handoff_report: dict[str, Any],
    execution_guard_report: dict[str, Any],
    runtime_state_report: dict[str, Any],
    simulated_order_replay_report: dict[str, Any],
    block_analysis_report: dict[str, Any],
) -> dict[str, Any]:
    replay_rows = list(simulated_order_replay_report.get("simulated_decision_rows") or [])
    backfill_summary = _v8_runtime_field_backfill_application_summary(replay_rows)
    rows_with_missing = [
        row for row in replay_rows if row.get("missing_runtime_field_codes")
    ]
    decision_rows = [
        _v8_runtime_field_coverage_decision_row(row)
        for row in rows_with_missing
    ]
    all_candidates = [
        candidate
        for row in decision_rows
        for candidate in row["runtime_field_backfill_candidates"]
    ]
    field_counts = Counter(
        str(candidate["runtime_field_name"]) for candidate in all_candidates
    )
    code_counts = Counter(str(candidate["missing_field_code"]) for candidate in all_candidates)
    classification_counts = Counter(
        str(candidate["field_gap_classification"]) for candidate in all_candidates
    )
    for classification in (
        "true_data_coverage_gap",
        "derived_backfill_from_existing_handoff_fields",
        "optional_for_no_trade",
        "too_strict_for_simulation_only_mode",
    ):
        classification_counts.setdefault(classification, 0)
    report = {
        "schema_version": O_V8_EXECUTION_RUNTIME_FIELD_COVERAGE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_v8_execution_runtime_field_coverage",
        "diagnostic_only": True,
        "simulation_only": True,
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "source_action_rank_signal_report_id": handoff_report[
            "o_v8_action_rank_handoff_report_id"
        ],
        "execution_guard_report_id": execution_guard_report[
            "o_v8_execution_risk_guard_report_id"
        ],
        "runtime_state_report_id": runtime_state_report[
            "o_v8_execution_runtime_state_report_id"
        ],
        "simulated_order_replay_report_id": simulated_order_replay_report[
            "o_v8_execution_simulated_order_replay_report_id"
        ],
        "block_analysis_report_id": block_analysis_report[
            "o_v8_execution_guard_block_analysis_report_id"
        ],
        "analysis_source": "simulated_order_replay_missing_runtime_field_codes",
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "runtime_field_backfill_rules_applied": backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "backfill_rules_applied": backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "proposed_backfill_rules_only": not backfill_summary[
            "runtime_field_backfill_rules_applied"
        ],
        "applied_runtime_field_backfill_count": backfill_summary[
            "applied_runtime_field_backfill_count"
        ],
        "applied_runtime_field_backfill_rule_counts": backfill_summary[
            "applied_runtime_field_backfill_rule_counts"
        ],
        "applied_runtime_field_backfill_application_type_counts": backfill_summary[
            "applied_runtime_field_backfill_application_type_counts"
        ],
        "applied_runtime_field_backfill_field_counts": backfill_summary[
            "applied_runtime_field_backfill_field_counts"
        ],
        "runtime_field_backfill_provenance_validity_summary": backfill_summary[
            "runtime_field_backfill_provenance_validity_summary"
        ],
        "runtime_field_backfill_provenance_violations": backfill_summary[
            "runtime_field_backfill_provenance_violations"
        ],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "required_runtime_fields": _v8_execution_required_runtime_fields(),
        "decision_count": len(replay_rows),
        "missing_runtime_field_decision_count": len(rows_with_missing),
        "missing_runtime_field_occurrence_count": len(all_candidates),
        "missing_runtime_field_code_counts": dict(sorted(code_counts.items())),
        "missing_runtime_field_name_counts": dict(sorted(field_counts.items())),
        "primary_missing_runtime_fields": [
            field for field, _ in field_counts.most_common(10)
        ],
        "classification_counts": dict(sorted(classification_counts.items())),
        "safe_backfill_candidate_count": sum(
            1
            for candidate in all_candidates
            if candidate["can_backfill_in_later_commit"]
        ),
        "existing_handoff_backfill_candidate_count": sum(
            1
            for candidate in all_candidates
            if candidate["backfill_source_class"] == "existing_handoff_fields"
        ),
        "decision_time_data_join_backfill_candidate_count": sum(
            1
            for candidate in all_candidates
            if candidate["backfill_source_class"] == "decision_time_data_join_required"
        ),
        "required_field_policy_relaxation_candidate_count": sum(
            1
            for candidate in all_candidates
            if candidate["backfill_source_class"] == "required_field_policy_relaxation"
        ),
        "missing_field_summary_by_field": _v8_runtime_field_summary_by(
            all_candidates,
            "runtime_field_name",
        ),
        "missing_field_summary_by_action": _v8_runtime_field_summary_by(
            all_candidates,
            "source_selected_action",
        ),
        "missing_field_summary_by_family": _v8_runtime_field_summary_by(
            all_candidates,
            "source_selected_family",
        ),
        "missing_field_summary_by_side": _v8_runtime_field_summary_by(
            all_candidates,
            "source_selected_side",
        ),
        "missing_field_summary_by_market": _v8_runtime_field_summary_by(
            all_candidates,
            "market_id",
        ),
        "missing_field_summary_by_time_to_close_bucket": (
            _v8_runtime_field_summary_by(
                all_candidates,
                "time_to_close_bucket",
            )
        ),
        "runtime_field_coverage_decision_rows": decision_rows,
        "per_decision_applied_runtime_field_backfill_rows": [
            {
                "decision_group_id": row.get("decision_group_id"),
                "market_id": row.get("market_id"),
                "decision_ts": row.get("decision_ts"),
                "source_selected_action": row.get("source_selected_action"),
                "source_selected_family": row.get("source_selected_family"),
                "source_selected_side": row.get("source_selected_side"),
                "runtime_field_applied_backfill_rows": row.get(
                    "runtime_field_applied_backfill_rows",
                    [],
                ),
            }
            for row in replay_rows
            if row.get("runtime_field_applied_backfill_rows")
        ],
        "proposed_deterministic_backfill_rules": (
            _v8_runtime_field_proposed_backfill_rules(
                backfill_summary["applied_runtime_field_backfill_rule_counts"]
            )
        ),
        "runtime_field_policy_findings": (
            _v8_runtime_field_policy_findings(
                decision_rows,
                all_candidates,
                decision_count=len(replay_rows),
            )
        ),
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "no_paper_live_unlock_from_runtime_field_coverage": True,
        "no_source_freeze_unlock_from_runtime_field_coverage": True,
        **compact_safety_fields(),
    }
    report["o_v8_execution_runtime_field_coverage_report_id"] = (
        canonical_json_sha256(report)
    )
    return report


def _v8_runtime_field_coverage_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _v8_runtime_field_backfill_candidate(row, missing_code)
        for missing_code in sorted(set(row.get("missing_runtime_field_codes") or []))
    ]
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": row.get("source_selected_action"),
        "source_selected_family": row.get("source_selected_family"),
        "source_selected_side": row.get("source_selected_side"),
        "execution_guarded_action": row.get("execution_guarded_action"),
        "execution_guarded_family": row.get("execution_guarded_family"),
        "execution_guarded_side": row.get("execution_guarded_side"),
        "runtime_mode": row.get("runtime_mode"),
        "runtime_exposure_state_available": bool(
            row.get("runtime_exposure_state_available")
        ),
        "time_to_close_seconds": _v8_row_time_to_close_seconds(row),
        "time_to_close_bucket": _v8_time_to_close_bucket_from_decision(row),
        "missing_runtime_field_codes": sorted(
            set(row.get("missing_runtime_field_codes") or [])
        ),
        "runtime_field_backfill_candidates": candidates,
        "has_true_data_coverage_gap": any(
            candidate["field_gap_classification"] == "true_data_coverage_gap"
            for candidate in candidates
        ),
        "has_safe_backfill_candidate": any(
            candidate["field_gap_classification"]
            == "derived_backfill_from_existing_handoff_fields"
            for candidate in candidates
        ),
        "has_no_trade_optional_field": any(
            candidate["field_gap_classification"] == "optional_for_no_trade"
            for candidate in candidates
        ),
        "has_simulation_policy_too_strict_field": any(
            candidate["field_gap_classification"]
            == "too_strict_for_simulation_only_mode"
            for candidate in candidates
        ),
    }


def _v8_runtime_field_backfill_candidate(
    row: dict[str, Any],
    missing_code: str,
) -> dict[str, Any]:
    field_name = _v8_runtime_field_name_from_missing_code(missing_code)
    source_action = str(row.get("source_selected_action") or "")
    classification, rule_id, reason_codes = _v8_runtime_field_gap_classification(
        row=row,
        missing_code=missing_code,
        runtime_field_name=field_name,
    )
    backfill_source_class = _v8_runtime_field_backfill_source_class(
        classification,
        rule_id,
    )
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": source_action,
        "source_selected_family": row.get("source_selected_family"),
        "source_selected_side": row.get("source_selected_side"),
        "time_to_close_bucket": _v8_time_to_close_bucket_from_decision(row),
        "missing_field_code": missing_code,
        "runtime_field_name": field_name,
        "field_gap_classification": classification,
        "proposed_rule_id": rule_id,
        "backfill_source_class": backfill_source_class,
        "can_backfill_in_later_commit": backfill_source_class
        in {"existing_handoff_fields", "decision_time_data_join_required"},
        "requires_required_field_policy_change": backfill_source_class
        == "required_field_policy_relaxation",
        "backfill_rule_applied_now": False,
        "reason_codes": reason_codes,
        "existing_handoff_evidence": _v8_runtime_field_existing_evidence(
            row=row,
            runtime_field_name=field_name,
        ),
    }


def _v8_runtime_field_backfill_source_class(
    classification: str,
    rule_id: str,
) -> str:
    if classification == "derived_backfill_from_existing_handoff_fields":
        return "existing_handoff_fields"
    if rule_id.startswith("backfill_"):
        return "decision_time_data_join_required"
    if classification in {
        "optional_for_no_trade",
        "too_strict_for_simulation_only_mode",
    }:
        return "required_field_policy_relaxation"
    return "not_backfillable"


def _v8_runtime_field_name_from_missing_code(missing_code: str) -> str:
    explicit = {
        "execution_exposure_state_missing": "runtime_exposure_state",
        "missing_valid_reference_price_provenance": (
            "reference_price_feature_provenance.provenance_valid"
        ),
    }
    if missing_code in explicit:
        return explicit[missing_code]
    prefix = "missing_microstructure_"
    if missing_code.startswith(prefix):
        return f"microstructure_snapshot.{missing_code.removeprefix(prefix)}"
    if missing_code.startswith("missing_"):
        return missing_code.removeprefix("missing_")
    return missing_code


def _v8_runtime_field_gap_classification(
    *,
    row: dict[str, Any],
    missing_code: str,
    runtime_field_name: str,
) -> tuple[str, str, list[str]]:
    source_action = str(row.get("source_selected_action") or "")
    if source_action == "NO_TRADE" and runtime_field_name in {
        "microstructure_snapshot.spread_bps",
        "microstructure_snapshot.book_staleness_ms",
        "microstructure_snapshot.queue_fill_proxy",
        "microstructure_snapshot.time_to_close_seconds",
        "reference_price_feature_provenance.provenance_valid",
        "runtime_exposure_state",
    }:
        return (
            "optional_for_no_trade",
            "make_non_order_runtime_fields_optional_for_no_trade",
            ["no_order_is_attempted_for_no_trade"],
        )
    if (
        runtime_field_name == "runtime_exposure_state"
        and row.get("runtime_mode") == "simulated_runtime_state"
        and row.get("runtime_exposure_state_available") is True
    ):
        return (
            "too_strict_for_simulation_only_mode",
            "trust_deterministic_simulated_runtime_ledger",
            ["simulated_runtime_state_is_available"],
        )
    derived_rules = {
        "selected_action": (
            row.get("source_selected_action"),
            "copy_selected_action_from_source_handoff",
        ),
        "selected_side": (
            row.get("source_selected_side") or _side_from_action(source_action),
            "derive_selected_side_from_action",
        ),
        "selected_action_family": (
            row.get("source_selected_family") or _action_family(source_action),
            "derive_selected_action_family_from_action",
        ),
        "corrected_model_score": (
            row.get("source_model_score"),
            "copy_corrected_model_score_from_source_handoff",
        ),
        "high_score_flag": (
            row.get("source_high_score_flag"),
            "copy_high_score_flag_from_source_handoff",
        ),
        "full_5_action_ranking": (
            row.get("top_k_action_ranking")
            if len(row.get("top_k_action_ranking") or [])
            >= len(O_REQUIRED_DECISION_ACTION_FAMILIES)
            else None,
            "copy_full_5_action_ranking_from_source_handoff",
        ),
        "p_down": (
            row.get("p_down")
            if row.get("p_down") is not None
            else (
                1.0 - float(row["p_up"])
                if row.get("p_up") is not None
                else None
            ),
            "derive_p_down_from_p_up",
        ),
    }
    if runtime_field_name in derived_rules and derived_rules[runtime_field_name][0] not in (
        None,
        "",
        [],
    ):
        return (
            "derived_backfill_from_existing_handoff_fields",
            derived_rules[runtime_field_name][1],
            ["existing_handoff_field_available"],
        )
    if runtime_field_name == "runtime_exposure_state":
        return (
            "too_strict_for_simulation_only_mode",
            "use_simulated_runtime_ledger_when_runtime_mode_is_simulated",
            ["runtime_exposure_state_should_be_supplied_by_simulation_ledger"],
        )
    if runtime_field_name.startswith("microstructure_snapshot."):
        return (
            "true_data_coverage_gap",
            "backfill_microstructure_snapshot_from_decision_time_book",
            ["decision_time_microstructure_field_missing"],
        )
    if runtime_field_name.startswith("reference_price_feature_provenance."):
        return (
            "true_data_coverage_gap",
            "backfill_reference_price_provenance_from_decision_time_reference_feed",
            ["decision_time_reference_provenance_missing_or_invalid"],
        )
    return (
        "true_data_coverage_gap",
        "collect_missing_runtime_field_before_execution_guard",
        [f"missing_runtime_field:{runtime_field_name}"],
    )


def _v8_runtime_field_existing_evidence(
    *,
    row: dict[str, Any],
    runtime_field_name: str,
) -> dict[str, Any]:
    microstructure = dict(row.get("microstructure_snapshot") or {})
    reference_provenance = dict(row.get("reference_price_feature_provenance") or {})
    if runtime_field_name.startswith("microstructure_snapshot."):
        key = runtime_field_name.split(".", 1)[1]
        return {
            "microstructure_snapshot_present": bool(microstructure),
            "field_value_present": microstructure.get(key) is not None,
            "field_value": microstructure.get(key),
        }
    if runtime_field_name.startswith("reference_price_feature_provenance."):
        key = runtime_field_name.split(".", 1)[1]
        return {
            "reference_price_feature_provenance_present": bool(reference_provenance),
            "field_value_present": reference_provenance.get(key) is not None,
            "field_value": reference_provenance.get(key),
        }
    if runtime_field_name == "runtime_exposure_state":
        return {
            "runtime_mode": row.get("runtime_mode"),
            "runtime_exposure_state_available": bool(
                row.get("runtime_exposure_state_available")
            ),
            "pre_decision_exposure_state_present": bool(
                row.get("pre_decision_exposure_state")
            ),
            "post_decision_exposure_state_present": bool(
                row.get("post_decision_exposure_state")
            ),
        }
    return {
        "source_selected_action_present": bool(row.get("source_selected_action")),
        "source_selected_side_present": bool(row.get("source_selected_side")),
        "source_selected_family_present": bool(row.get("source_selected_family")),
        "source_model_score_present": row.get("source_model_score") is not None,
        "top_k_action_ranking_count": len(row.get("top_k_action_ranking") or []),
        "p_up_present": row.get("p_up") is not None,
        "p_down_present": row.get("p_down") is not None,
    }


def _v8_runtime_field_summary_by(
    candidates: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate.get(field_name) or "UNKNOWN")].append(candidate)
    summary = {}
    for value, rows in sorted(grouped.items()):
        classification_counts = Counter(
            str(row["field_gap_classification"]) for row in rows
        )
        field_counts = Counter(str(row["runtime_field_name"]) for row in rows)
        code_counts = Counter(str(row["missing_field_code"]) for row in rows)
        summary[value] = {
            "missing_field_occurrence_count": len(rows),
            "classification_counts": dict(sorted(classification_counts.items())),
            "runtime_field_name_counts": dict(sorted(field_counts.items())),
            "missing_field_code_counts": dict(sorted(code_counts.items())),
        }
    return summary


def _v8_runtime_field_proposed_backfill_rules(
    applied_rule_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    applied_rule_counts = dict(applied_rule_counts or {})
    rules = [
        {
            "rule_id": "copy_selected_action_from_source_handoff",
            "field": "selected_action",
            "rule": "copy source_selected_action when selected_action is absent",
            "decision_time_only": True,
        },
        {
            "rule_id": "derive_selected_side_from_action",
            "field": "selected_side",
            "rule": "derive UP/DOWN/NONE from selected_action",
            "decision_time_only": True,
        },
        {
            "rule_id": "derive_selected_action_family_from_action",
            "field": "selected_action_family",
            "rule": "derive HOLD_TO_SETTLEMENT/SELL_BEFORE_CLOSE/NO_TRADE from action",
            "decision_time_only": True,
        },
        {
            "rule_id": "copy_full_5_action_ranking_from_source_handoff",
            "field": "full_5_action_ranking",
            "rule": "copy complete five-action handoff ranking into runtime payload",
            "decision_time_only": True,
        },
        {
            "rule_id": "derive_p_down_from_p_up",
            "field": "p_down",
            "rule": "derive p_down as 1 - p_up when p_up is decision-time available",
            "decision_time_only": True,
        },
        {
            "rule_id": "make_non_order_runtime_fields_optional_for_no_trade",
            "field": "NO_TRADE non-order fields",
            "rule": "do not require microstructure/exposure fields when action is NO_TRADE",
            "decision_time_only": True,
        },
        {
            "rule_id": "use_simulated_runtime_ledger_when_runtime_mode_is_simulated",
            "field": "runtime_exposure_state",
            "rule": "use deterministic simulated ledger for simulation-only replay",
            "decision_time_only": True,
        },
        {
            "rule_id": "backfill_microstructure_snapshot_from_decision_time_book",
            "field": "microstructure_snapshot.*",
            "rule": (
                "join spread/staleness/queue/time_to_close from decision-time "
                "orderbook and market schedule only"
            ),
            "decision_time_only": True,
        },
        {
            "rule_id": (
                "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
            ),
            "field": "microstructure_snapshot.time_to_close_seconds",
            "rule": (
                "copy decision-time time-to-close from feature provenance or "
                "market schedule when max_input_ts <= decision_ts"
            ),
            "decision_time_only": True,
        },
        {
            "rule_id": (
                "backfill_reference_price_provenance_from_decision_time_reference_feed"
            ),
            "field": "reference_price_feature_provenance.provenance_valid",
            "rule": "join verified reference provenance with max_input_ts <= decision_ts",
            "decision_time_only": True,
        },
    ]
    for rule in rules:
        rule_id = str(rule["rule_id"])
        rule["applied_now"] = applied_rule_counts.get(rule_id, 0) > 0
        rule["applied_count"] = int(applied_rule_counts.get(rule_id, 0))
    return rules


def _v8_runtime_field_policy_findings(
    decision_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    decision_count: int,
) -> list[str]:
    findings = []
    classification_counts = Counter(
        str(candidate["field_gap_classification"]) for candidate in candidates
    )
    if decision_rows and len(decision_rows) == decision_count:
        findings.append("execution_required_runtime_fields_missing_fires_for_all_rows")
    if classification_counts.get("true_data_coverage_gap", 0):
        findings.append("trade_rows_have_true_decision_time_data_coverage_gaps")
    if classification_counts.get("optional_for_no_trade", 0):
        findings.append("no_trade_rows_require_fields_that_do_not_create_orders")
    if classification_counts.get("derived_backfill_from_existing_handoff_fields", 0):
        findings.append("some_missing_fields_can_be_derived_from_handoff_payload")
    if classification_counts.get("too_strict_for_simulation_only_mode", 0):
        findings.append("some_runtime_requirements_are_too_strict_for_simulation_only")
    if not findings:
        findings.append("no_missing_runtime_fields_detected")
    return sorted(set(findings))


def _v8_runtime_order_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "simulated_order_id": row.get("simulated_order_id"),
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "action": row.get("execution_guarded_action"),
        "side": row.get("execution_guarded_side"),
        "family": row.get("execution_guarded_family"),
        "notional": row.get("proposed_order_size"),
    }


def _v8_runtime_blocked_order_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_group_id": row.get("decision_group_id"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "source_selected_action": row.get("source_selected_action"),
        "execution_guarded_action": row.get("execution_guarded_action"),
        "execution_blocking_reason_codes": row.get(
            "execution_blocking_reason_codes",
            [],
        ),
        "exposure_reason_codes": row.get("exposure_reason_codes", []),
    }


def _v8_execution_missing_runtime_field_codes(
    *,
    row: dict[str, Any],
    microstructure: dict[str, Any],
    reference_provenance: dict[str, Any],
    trade_action: bool,
    runtime_state_available: bool = False,
) -> list[str]:
    missing = []
    for field_name in (
        "selected_action",
        "selected_side",
        "selected_action_family",
        "full_5_action_ranking",
        "corrected_model_score",
    ):
        if row.get(field_name) in (None, "", []):
            missing.append(f"missing_{field_name}")
    if trade_action:
        for field_name in (
            "spread_bps",
            "book_staleness_ms",
            "queue_fill_proxy",
            "time_to_close_seconds",
        ):
            if microstructure.get(field_name) is None:
                missing.append(f"missing_microstructure_{field_name}")
        if reference_provenance.get("provenance_valid") is not True:
            missing.append("missing_valid_reference_price_provenance")
    if trade_action and not runtime_state_available:
        missing.append("execution_exposure_state_missing")
    return sorted(set(missing))


def _top_score_margin(full_ranking: list[dict[str, Any]]) -> float | None:
    if len(full_ranking) < 2:
        return None
    first = _optional_float(full_ranking[0].get("corrected_model_score"))
    second = _optional_float(full_ranking[1].get("corrected_model_score"))
    if first is None or second is None:
        return None
    return first - second


def _v8_same_side_sbc_candidate(
    *,
    full_ranking: list[dict[str, Any]],
    side: str,
) -> dict[str, Any] | None:
    expected = f"BUY_{side}_SELL_BEFORE_CLOSE"
    for row in full_ranking:
        if row.get("selected_action") == expected:
            return row
    return None


def _v8_sbc_candidate_runtime_safe(
    row: dict[str, Any],
    *,
    guard_config: dict[str, Any],
) -> bool:
    microstructure = dict(row.get("microstructure_snapshot") or {})
    spread = _optional_float(microstructure.get("spread_bps"))
    staleness = _optional_float(microstructure.get("book_staleness_ms"))
    queue = _optional_float(microstructure.get("queue_fill_proxy"))
    time_to_close = _optional_float(microstructure.get("time_to_close_seconds"))
    return (
        spread is not None
        and spread <= float(guard_config["max_spread_bps"])
        and staleness is not None
        and staleness <= float(guard_config["max_book_staleness_ms"])
        and queue is not None
        and queue >= float(guard_config["min_queue_fill"])
        and time_to_close is not None
        and time_to_close >= float(guard_config["min_time_to_close_seconds"])
        and row.get("p_up_action_disagreement") is not True
    )


def _v8_execution_penalties(
    *,
    source_score: float,
    spread: float | None,
    staleness: float | None,
    queue: float | None,
    margin: float | None,
    source_family: str,
    blocking_reason_codes: list[str],
    guard_config: dict[str, Any],
) -> dict[str, float]:
    del source_score
    spread_penalty = (
        0.20
        if spread is None
        else max(0.0, spread - float(guard_config["max_spread_bps"]))
        / float(guard_config["max_spread_bps"])
        * 0.10
    )
    staleness_penalty = (
        0.20
        if staleness is None
        else max(0.0, staleness - float(guard_config["max_book_staleness_ms"]))
        / float(guard_config["max_book_staleness_ms"])
        * 0.10
    )
    liquidity_penalty = (
        0.20
        if queue is None
        else max(0.0, float(guard_config["min_queue_fill"]) - queue) * 0.20
    )
    score_margin_penalty = (
        0.05
        if margin is None or margin < float(guard_config["min_score_margin"])
        else 0.0
    )
    family_specific_penalty = (
        0.10
        if source_family == "HOLD_TO_SETTLEMENT"
        and "execution_hts_guard_failed" in blocking_reason_codes
        else 0.0
    )
    return {
        "spread_penalty": spread_penalty,
        "staleness_penalty": staleness_penalty,
        "liquidity_penalty": liquidity_penalty,
        "exposure_penalty": 1.0
        if "execution_exposure_state_missing" in blocking_reason_codes
        else 0.0,
        "score_margin_penalty": score_margin_penalty,
        "family_specific_penalty": family_specific_penalty,
    }


def _v8_execution_order_size(
    *,
    high_score: bool,
    guard_reason_codes: list[str],
    guard_config: dict[str, Any],
) -> float:
    size = float(guard_config["base_order_size"]) if high_score else 0.10
    if "execution_score_margin_too_close" in guard_reason_codes:
        size *= 0.50
    return min(size, float(guard_config["max_order_size"]))


def _v8_execution_sizing_reason_codes(
    *,
    proposed_order_size: float,
    guard_reason_codes: list[str],
    blocking_reason_codes: list[str],
    source_action: str,
) -> list[str]:
    if source_action == "NO_TRADE":
        return ["execution_no_order_for_no_trade"]
    if blocking_reason_codes:
        return ["execution_blocked_size_zero", *blocking_reason_codes]
    if proposed_order_size <= 0.0:
        return ["execution_size_zero"]
    reasons = ["execution_base_size_applied"]
    if "execution_score_margin_too_close" in guard_reason_codes:
        reasons.append("execution_size_reduced_for_close_score_margin")
    return reasons


def _v8_execution_top_k_action_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "selected_action": row.get("selected_action"),
        "selected_side": row.get("selected_side"),
        "selected_action_family": row.get("selected_action_family"),
        "corrected_model_score": row.get("corrected_model_score"),
        "raw_model_score": row.get("raw_model_score"),
        "high_score_flag": row.get("high_score_flag"),
        "p_up_action_disagreement": row.get("p_up_action_disagreement"),
    }


def _v8_execution_guard_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blocking_reasons = Counter(
        reason
        for row in rows
        for reason in row["execution_blocking_reason_codes"]
    )
    guard_reasons = Counter(
        reason for row in rows for reason in row["execution_guard_reason_codes"]
    )
    by_action = Counter(str(row["execution_guarded_action"]) for row in rows)
    by_side = Counter(str(row["execution_guarded_side"]) for row in rows)
    order_allowed_count = sum(1 for row in rows if row["order_allowed"])
    blocked_count = sum(
        1
        for row in rows
        if row["execution_blocking_reason_codes"]
        or row["execution_guarded_action"] == "NO_TRADE"
    )
    fail_closed_count = sum(1 for row in rows if row["fail_closed"])
    return {
        "decision_count": len(rows),
        "order_allowed_count": order_allowed_count,
        "blocked_decision_count": blocked_count,
        "fail_closed_decision_count": fail_closed_count,
        "simulation_only_decision_count": sum(1 for row in rows if row["simulation_only"]),
        "proposed_order_size_total": sum(
            float(row["proposed_order_size"]) for row in rows
        ),
        "execution_guarded_action_count_by_action": dict(sorted(by_action.items())),
        "execution_guarded_action_count_by_side": dict(sorted(by_side.items())),
        "execution_blocking_reason_counts": dict(sorted(blocking_reasons.items())),
        "execution_guard_reason_counts": dict(sorted(guard_reasons.items())),
    }


def _v8_selected_action_rank_handoff_rows(
    *,
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    high_score_threshold = _model_high_score_threshold(model_training_summary)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split:
            grouped[str(row["decision_group_id"])].append(row)
    selected_rows = []
    for decision_group_id, group_rows in sorted(grouped.items()):
        ranked = sorted(
            group_rows,
            key=lambda row: float(row["variant_scores"][O_MODEL_PREDICTED_VARIANT]),
            reverse=True,
        )
        selected = ranked[0]
        selected_rows.append(
            {
                **_v8_action_rank_handoff_action_entry(
                    selected,
                    rank=1,
                    high_score_threshold=high_score_threshold,
                ),
                "decision_group_id": decision_group_id,
                "full_5_action_ranking": [
                    _v8_action_rank_handoff_action_entry(
                        row,
                        rank=rank,
                        high_score_threshold=high_score_threshold,
                    )
                    for rank, row in enumerate(ranked, start=1)
                ],
            }
        )
    return selected_rows


def _v8_action_rank_handoff_action_entry(
    row: dict[str, Any],
    *,
    rank: int,
    high_score_threshold: float,
) -> dict[str, Any]:
    action = str(row.get("action") or "")
    p_up = _optional_float(row.get("p_up"))
    p_down = 1.0 - p_up if p_up is not None else None
    score = float(row["variant_scores"][O_MODEL_PREDICTED_VARIANT])
    feature_provenance = dict(row.get("decision_time_feature_provenance") or {})
    reference_provenance = dict(
        feature_provenance.get("reference_price_to_beat_distance_at_decision") or {}
    )
    time_to_close_backfill_source = _v8_time_to_close_runtime_backfill_source(
        row=row,
        feature_provenance=feature_provenance,
    )
    return {
        "rank": rank,
        "source_report_path": row.get("source_report_path"),
        "market_id": row.get("market_id"),
        "decision_ts": row.get("decision_ts"),
        "split": row.get("split"),
        "selected_action": action,
        "selected_side": _side_from_action(action),
        "selected_action_family": _action_family(action),
        "ranking_score_source": "model_predicted_score",
        "corrected_model_score": score,
        "raw_model_score": row.get("o_raw_ridge_model_score"),
        "group_normalized_raw_model_score": row.get(
            "o_group_normalized_raw_model_score"
        ),
        "base_model_predicted_score": row.get("o_base_model_predicted_score"),
        "score_components": row.get("o_model_score_components") or {},
        "high_score_flag": score >= high_score_threshold,
        "high_score_threshold": high_score_threshold,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_action_disagreement": _p_up_action_disagreement_for_row(row),
        "microstructure_snapshot": {
            "book_staleness_ms": row.get("side_book_staleness_ms"),
            "spread_bps": row.get("side_spread_bps"),
            "queue_fill_proxy": row.get("side_queue_fill_proxy"),
            "time_to_close_seconds": row.get("time_to_close_seconds"),
            "entry_ask": row.get("entry_quality_ask"),
            "executable_exit_bid_proxy": row.get("exit_quality_bid"),
            "side_book_depth_imbalance": row.get("side_book_depth_imbalance"),
            "side_book_update_velocity": row.get("side_book_update_velocity"),
            "hts_vs_sell_before_close_exit_value_gap_proxy": row.get(
                "hts_vs_sell_before_close_exit_value_gap_proxy"
            ),
        },
        "reference_price_to_beat_distance_at_decision": row.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "reference_price_feature_available": row.get(
            "reference_price_feature_available"
        ),
        "reference_price_feature_provenance": {
            "source_fields_used": reference_provenance.get("source_fields"),
            "max_input_ts": reference_provenance.get("max_input_ts"),
            "decision_ts": row.get("decision_ts"),
            "provenance_valid": row.get("decision_time_feature_provenance_valid"),
        },
        "decision_time_feature_max_input_ts": row.get(
            "decision_time_feature_max_input_ts"
        ),
        "decision_time_feature_missing_reason_codes": row.get(
            "decision_time_feature_missing_reason_codes",
            [],
        ),
        "runtime_field_backfill_sources": {
            "microstructure_snapshot.time_to_close_seconds": (
                time_to_close_backfill_source
            ),
        },
        "oracle_executable_best_action": row.get("oracle_executable_best_action"),
        "realized_replay_return_report_only": row.get("realized_replay_return"),
        "regret_report_only": (
            float(row.get("oracle_executable_best_action_return") or 0.0)
            - float(row.get("realized_replay_return") or 0.0)
        ),
    }


def _v8_time_to_close_runtime_backfill_source(
    *,
    row: dict[str, Any],
    feature_provenance: dict[str, Any],
) -> dict[str, Any]:
    decision_ts = _optional_int(row.get("decision_ts"))
    value = _optional_float(row.get("time_to_close_seconds"))
    source_field_name = "polymarket_feature_rows.features.time_to_close_seconds"
    source_provenance = dict(feature_provenance.get("time_to_close_seconds") or {})
    if value is None:
        value = _optional_float(row.get("entry_exit_quality_time_to_close_seconds"))
    if value is None:
        return {
            "field": "microstructure_snapshot.time_to_close_seconds",
            "value": None,
            "source_field_name": None,
            "source_timestamp": None,
            "max_input_ts": None,
            "decision_ts": decision_ts,
            "deterministic_rule_id": (
                "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
            ),
            "provenance_valid": False,
            "reason_codes": ["time_to_close_source_value_missing"],
        }
    if source_provenance:
        source_timestamp = _optional_int(
            source_provenance.get("input_end_ts")
            if source_provenance.get("input_end_ts") is not None
            else source_provenance.get("available_at_ts")
        )
        max_input_ts = _optional_int(
            source_provenance.get("max_input_ts")
            if source_provenance.get("max_input_ts") is not None
            else source_timestamp
        )
        source_field_name = str(
            source_provenance.get("source_fields_used")
            or source_provenance.get("source")
            or source_field_name
        )
    else:
        source_timestamp = decision_ts
        max_input_ts = decision_ts
        source_field_name = (
            "polymarket_market_schedule.slug_or_market_metadata_time_to_close"
        )
    provenance_valid = (
        decision_ts is not None
        and source_timestamp is not None
        and max_input_ts is not None
        and source_timestamp <= decision_ts
        and max_input_ts <= decision_ts
    )
    return {
        "field": "microstructure_snapshot.time_to_close_seconds",
        "value": value,
        "source_field_name": source_field_name,
        "source_timestamp": source_timestamp,
        "max_input_ts": max_input_ts,
        "decision_ts": decision_ts,
        "deterministic_rule_id": (
            "backfill_time_to_close_from_decision_time_feature_or_market_schedule"
        ),
        "provenance_valid": provenance_valid,
        "reason_codes": ["decision_time_time_to_close_source_available"]
        if provenance_valid
        else ["time_to_close_source_provenance_invalid"],
    }


def _p_up_action_disagreement_for_row(row: dict[str, Any]) -> bool | None:
    action = str(row.get("action") or "")
    p_up = _optional_float(row.get("p_up"))
    if p_up is None or ("BUY_UP" not in action and "BUY_DOWN" not in action):
        return None
    return ("BUY_UP" in action and p_up < 0.50) or (
        "BUY_DOWN" in action and p_up > 0.50
    )


def _v8_action_rank_handoff_summary(
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_action = Counter(str(row["selected_action"]) for row in selected_rows)
    by_side = Counter(str(row["selected_side"]) for row in selected_rows)
    high_score_count = sum(1 for row in selected_rows if row["high_score_flag"])
    disagreement_count = sum(
        1 for row in selected_rows if row["p_up_action_disagreement"] is True
    )
    comparable_count = sum(
        1 for row in selected_rows if row["p_up_action_disagreement"] is not None
    )
    return {
        "selected_action_count": len(selected_rows),
        "selected_action_count_by_action": dict(sorted(by_action.items())),
        "selected_action_count_by_side": dict(sorted(by_side.items())),
        "high_score_selected_action_count": high_score_count,
        "p_up_action_comparable_count": comparable_count,
        "p_up_action_disagreement_count": disagreement_count,
        "p_up_action_disagreement_rate": (
            disagreement_count / comparable_count if comparable_count else 0.0
        ),
    }


def _compact_o_model_training_summary(
    model_training_summary: dict[str, Any],
) -> dict[str, Any]:
    """Keep repeated reports small; full training evidence lives in ranking report."""
    correction_config = model_training_summary["ranking_correction_config"]
    return {
        "model_candidate_name": model_training_summary["model_candidate_name"],
        "ranking_score_source": model_training_summary["ranking_score_source"],
        "deployable_model_score_available": model_training_summary[
            "deployable_model_score_available"
        ],
        "selected_feature_set_name": model_training_summary[
            "selected_feature_set_name"
        ],
        "selected_feature_set_config_hash": model_training_summary[
            "selected_feature_set_config_hash"
        ],
        "selected_correction_policy_name": model_training_summary[
            "selected_correction_policy_name"
        ],
        "selected_high_score_threshold_profile_name": model_training_summary[
            "selected_high_score_threshold_profile_name"
        ],
        "selected_joint_candidate_name": model_training_summary[
            "selected_joint_candidate_name"
        ],
        "feature_names": model_training_summary["feature_names"],
        "all_candidate_feature_names": model_training_summary[
            "all_candidate_feature_names"
        ],
        "model_input_fields_decision_time_only": model_training_summary[
            "model_input_fields_decision_time_only"
        ],
        "uses_validation_labels_for_tuning": bool(
            correction_config.get("uses_validation_labels_for_tuning", False)
        ),
        "training_target": model_training_summary["training_target"],
        "correction_constants_source": model_training_summary[
            "correction_constants_source"
        ],
        "correction_config_hash": model_training_summary["correction_config_hash"],
        "probe_constants_source": model_training_summary["probe_constants_source"],
        "probe_config_hash": model_training_summary["probe_config_hash"],
        "high_score_threshold": correction_config["high_score_calibration"][
            "high_score_threshold"
        ],
        "feature_set_selection_report_available": True,
        "joint_feature_correction_selection_report_available": True,
        "full_training_summary_report_path": "o_source_ranking_objective_report.json",
    }


def _comparison_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    leakage_report: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = []
    high_score_threshold = _model_high_score_threshold(model_training_summary)
    model_predicted_relaxed_diagnostic_status: dict[str, Any] | None = None
    model_predicted_v8_action_rank_status: dict[str, Any] | None = None
    full_decision_grid_summary = _v8_decision_grid_summary(
        rows,
        split="validation",
    )
    for variant in O_VARIANTS:
        metrics = _ranking_metrics(rows, variant, high_score_threshold)
        split_metrics = _split_metric_views(rows, variant, high_score_threshold)
        relaxed_diagnostic_status: dict[str, Any] | None = None
        v8_action_rank_status: dict[str, Any] | None = None
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
        if variant == O_MODEL_PREDICTED_VARIANT:
            validation_metrics = split_metrics["validation"]
            p_up_summary = _p_up_action_disagreement_summary(
                rows=rows,
                variant=O_MODEL_PREDICTED_VARIANT,
                split="validation",
            )
            leakage_passed = True
            calibration_support_passed = (
                int(validation_metrics["decision_group_count"])
                >= O_MIN_VALIDATION_DECISION_GROUPS
                and int(validation_metrics["high_score_support_count"])
                >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
            )
            action_family_paper_decision_eligible = (
                _validation_action_family_gate_passed(validation_metrics)
            )
            best_action_concentration_passed = (
                float(validation_metrics["NO_TRADE_selection_rate"])
                <= O_MAX_NO_TRADE_SELECTION_RATE
                and not bool(
                    validation_metrics["largest_winner_dependency"][
                        "total_return_positive_only_because_of_largest_winner"
                    ]
                )
            )
            p_up_action_disagreement_within_limit = bool(
                p_up_summary[
                    "candidate_scoped_p_up_action_disagreement_within_limit"
                ]
            )
            high_score_return_positive = (
                int(validation_metrics["high_score_support_count"])
                >= O_MIN_HIGH_SCORE_SUPPORT_COUNT
                and float(validation_metrics["high_score_realized_return_mean"])
                > 0.0
                and float(validation_metrics["high_score_realized_return_sum"])
                > 0.0
            )
            relaxed_diagnostic_status = _o_relaxed_diagnostic_gate_status(
                validation_metrics=validation_metrics,
                p_up_action_disagreement_within_limit=(
                    p_up_action_disagreement_within_limit
                ),
                calibration_support_passed=calibration_support_passed,
                action_family_paper_decision_eligible=(
                    action_family_paper_decision_eligible
                ),
                best_action_concentration_passed=best_action_concentration_passed,
                high_score_return_positive=high_score_return_positive,
                leakage_passed=leakage_passed,
            )
            model_predicted_relaxed_diagnostic_status = relaxed_diagnostic_status
            strict_calibration_quality_passed = (
                float(validation_metrics["top1_realized_best_action_hit_rate"])
                >= O_MIN_TOP1_HIT_RATE
                and float(validation_metrics["mean_regret"]) <= O_MAX_MEAN_REGRET
            )
            gate_reason_codes = _o_gate_reason_codes(
                deployable=bool(
                    model_training_summary["deployable_model_score_available"]
                ),
                leakage_passed=bool(leakage_report["leakage_audit_passed"]),
                calibration_support_passed=calibration_support_passed,
                calibration_quality_passed=strict_calibration_quality_passed,
                action_family_paper_decision_eligible=(
                    action_family_paper_decision_eligible
                ),
                best_action_concentration_passed=best_action_concentration_passed,
                p_up_action_disagreement_within_limit=(
                    p_up_action_disagreement_within_limit
                ),
                high_score_return_positive=high_score_return_positive,
            )
            gate_reason_code_consistency = _o_gate_reason_code_consistency_diagnostic(
                reason_codes=gate_reason_codes,
                validation_metrics=validation_metrics,
                calibration_support_passed=calibration_support_passed,
                calibration_quality_passed=strict_calibration_quality_passed,
                action_family_paper_decision_eligible=(
                    action_family_paper_decision_eligible
                ),
                best_action_concentration_passed=best_action_concentration_passed,
                p_up_action_disagreement_within_limit=(
                    p_up_action_disagreement_within_limit
                ),
                high_score_return_positive=high_score_return_positive,
            )
            v8_action_rank_status = _o_v8_action_rank_gate_status(
                validation_metrics=validation_metrics,
                deployable_model_score_available=bool(
                    model_training_summary["deployable_model_score_available"]
                ),
                calibration_support_passed=calibration_support_passed,
                action_family_paper_decision_eligible=(
                    action_family_paper_decision_eligible
                ),
                best_action_concentration_passed=best_action_concentration_passed,
                p_up_action_disagreement_within_limit=(
                    p_up_action_disagreement_within_limit
                ),
                high_score_return_positive=high_score_return_positive,
                leakage_audit_passed=bool(leakage_report["leakage_audit_passed"]),
                expanded_feature_provenance_passed=bool(
                    leakage_report[
                        "expanded_decision_time_feature_provenance_passed"
                    ]
                ),
                gate_reason_code_consistency_passed=bool(
                    gate_reason_code_consistency[
                        "gate_reason_code_consistency_passed"
                    ]
                ),
                full_decision_grid_summary=full_decision_grid_summary,
            )
            model_predicted_v8_action_rank_status = v8_action_rank_status
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
                "model_training_summary": (
                    _compact_o_model_training_summary(model_training_summary)
                    if variant == O_MODEL_PREDICTED_VARIANT
                    else None
                ),
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
                "strict_calibration_quality_passed": (
                    (
                        float(split_metrics["validation"][
                            "top1_realized_best_action_hit_rate"
                        ])
                        >= O_MIN_TOP1_HIT_RATE
                    )
                    and (
                        float(split_metrics["validation"]["mean_regret"])
                        <= O_MAX_MEAN_REGRET
                    )
                )
                if variant == O_MODEL_PREDICTED_VARIANT
                else False,
                "relaxed_diagnostic_calibration_quality_passed": (
                    bool(
                        relaxed_diagnostic_status[
                            "relaxed_diagnostic_calibration_quality_passed"
                        ]
                    )
                    if relaxed_diagnostic_status is not None
                    else False
                ),
                "relaxed_diagnostic_source_candidate": (
                    bool(
                        relaxed_diagnostic_status[
                            "relaxed_diagnostic_source_candidate"
                        ]
                    )
                    if relaxed_diagnostic_status is not None
                    else False
                ),
                "relaxed_diagnostic_reason_codes": (
                    relaxed_diagnostic_status["relaxed_diagnostic_reason_codes"]
                    if relaxed_diagnostic_status is not None
                    else ["not_o_model_predicted_candidate"]
                ),
                "strict_vs_relaxed_gate_summary": (
                    relaxed_diagnostic_status["strict_vs_relaxed_gate_summary"]
                    if relaxed_diagnostic_status is not None
                    else None
                ),
                "v8_action_rank_quality_passed": (
                    bool(v8_action_rank_status["v8_action_rank_quality_passed"])
                    if v8_action_rank_status is not None
                    else False
                ),
                "v8_action_rank_candidate_eligible": (
                    bool(v8_action_rank_status["v8_action_rank_candidate_eligible"])
                    if v8_action_rank_status is not None
                    else False
                ),
                "v8_action_rank_reason_codes": (
                    v8_action_rank_status["v8_action_rank_reason_codes"]
                    if v8_action_rank_status is not None
                    else ["not_o_model_predicted_candidate"]
                ),
                "v8_execution_risk_control_required": (
                    bool(
                        v8_action_rank_status[
                            "v8_execution_risk_control_required"
                        ]
                    )
                    if v8_action_rank_status is not None
                    else False
                ),
                "v8_execution_handoff_allowed": (
                    bool(v8_action_rank_status["v8_execution_handoff_allowed"])
                    if v8_action_rank_status is not None
                    else False
                ),
                "v8_execution_handoff_blocking_reason_codes": (
                    v8_action_rank_status[
                        "v8_execution_handoff_blocking_reason_codes"
                    ]
                    if v8_action_rank_status is not None
                    else ["not_o_model_predicted_candidate"]
                ),
                "strict_source_gate_remains_failed": True,
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
        "v8_scope": "action_rank_signal_and_execution_layer_handoff_only",
        "model_layer_regret_risk_selection_enabled": False,
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
        "m2_candidate_report_path": str(m2_report_path),
        "m2_candidate_report_sha256": _sha256_file(m2_report_path),
        "m2_candidate_report_id": m2_report.get(
            "m2_stateful_replay_parity_candidate_report_id"
        ),
        "model_predicted_candidate_name": O_MODEL_PREDICTED_VARIANT,
        "model_training_summary": _compact_o_model_training_summary(
            model_training_summary
        ),
        "label_diagnostic_variants": list(O_LABEL_DIAGNOSTIC_VARIANTS),
        "candidate_rows": candidate_rows,
        "strict_vs_relaxed_gate_summary": (
            model_predicted_relaxed_diagnostic_status[
                "strict_vs_relaxed_gate_summary"
            ]
            if model_predicted_relaxed_diagnostic_status is not None
            else None
        ),
        "relaxed_diagnostic_source_candidate": (
            bool(
                model_predicted_relaxed_diagnostic_status[
                    "relaxed_diagnostic_source_candidate"
                ]
            )
            if model_predicted_relaxed_diagnostic_status is not None
            else False
        ),
        "relaxed_diagnostic_no_paper_live_unlock": True,
        "v8_action_rank_quality_passed": (
            bool(
                model_predicted_v8_action_rank_status[
                    "v8_action_rank_quality_passed"
                ]
            )
            if model_predicted_v8_action_rank_status is not None
            else False
        ),
        "v8_action_rank_candidate_eligible": (
            bool(
                model_predicted_v8_action_rank_status[
                    "v8_action_rank_candidate_eligible"
                ]
            )
            if model_predicted_v8_action_rank_status is not None
            else False
        ),
        "v8_action_rank_reason_codes": (
            model_predicted_v8_action_rank_status["v8_action_rank_reason_codes"]
            if model_predicted_v8_action_rank_status is not None
            else ["o_model_predicted_candidate_missing"]
        ),
        "v8_execution_risk_control_required": (
            bool(
                model_predicted_v8_action_rank_status[
                    "v8_execution_risk_control_required"
                ]
            )
            if model_predicted_v8_action_rank_status is not None
            else True
        ),
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": (
            model_predicted_v8_action_rank_status[
                "v8_execution_handoff_blocking_reason_codes"
            ]
            if model_predicted_v8_action_rank_status is not None
            else ["o_model_predicted_candidate_missing"]
        ),
        "strict_source_gate_remains_failed": True,
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
    strict_calibration_quality_passed = calibration_quality_passed
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
    relaxed_diagnostic_status = _o_relaxed_diagnostic_gate_status(
        validation_metrics=validation_metrics,
        p_up_action_disagreement_within_limit=p_up_action_disagreement_within_limit,
        calibration_support_passed=calibration_support_passed,
        action_family_paper_decision_eligible=(
            action_family_paper_decision_eligible
        ),
        best_action_concentration_passed=best_action_concentration_passed,
        high_score_return_positive=high_score_return_positive,
        leakage_passed=leakage_passed,
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
    gate_reason_code_consistency = _o_gate_reason_code_consistency_diagnostic(
        reason_codes=reason_codes,
        validation_metrics=validation_metrics,
        calibration_support_passed=calibration_support_passed,
        calibration_quality_passed=calibration_quality_passed,
        action_family_paper_decision_eligible=action_family_paper_decision_eligible,
        best_action_concentration_passed=best_action_concentration_passed,
        p_up_action_disagreement_within_limit=p_up_action_disagreement_within_limit,
        high_score_return_positive=high_score_return_positive,
    )
    full_decision_grid_summary = _v8_decision_grid_summary(
        rows,
        split="validation",
    )
    v8_action_rank_status = _o_v8_action_rank_gate_status(
        validation_metrics=validation_metrics,
        deployable_model_score_available=deployable,
        calibration_support_passed=calibration_support_passed,
        action_family_paper_decision_eligible=action_family_paper_decision_eligible,
        best_action_concentration_passed=best_action_concentration_passed,
        p_up_action_disagreement_within_limit=p_up_action_disagreement_within_limit,
        high_score_return_positive=high_score_return_positive,
        leakage_audit_passed=leakage_passed,
        expanded_feature_provenance_passed=bool(
            leakage_report["expanded_decision_time_feature_provenance_passed"]
        ),
        gate_reason_code_consistency_passed=bool(
            gate_reason_code_consistency["gate_reason_code_consistency_passed"]
        ),
        full_decision_grid_summary=full_decision_grid_summary,
    )
    report = {
        "schema_version": O_SOURCE_MODEL_ELIGIBILITY_GATE_SCHEMA_VERSION,
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_source_model_eligibility_gate",
        "diagnostic_only": True,
        "v8_scope": "action_rank_signal_and_execution_layer_handoff_only",
        "model_layer_regret_risk_selection_enabled": False,
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
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
            "relaxed_diagnostic_max_mean_regret": (
                O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
            ),
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
        "strict_calibration_quality_passed": strict_calibration_quality_passed,
        "relaxed_diagnostic_calibration_quality_passed": (
            relaxed_diagnostic_status[
                "relaxed_diagnostic_calibration_quality_passed"
            ]
        ),
        "relaxed_diagnostic_source_candidate": (
            relaxed_diagnostic_status["relaxed_diagnostic_source_candidate"]
        ),
        "relaxed_diagnostic_reason_codes": (
            relaxed_diagnostic_status["relaxed_diagnostic_reason_codes"]
        ),
        "strict_vs_relaxed_gate_summary": (
            relaxed_diagnostic_status["strict_vs_relaxed_gate_summary"]
        ),
        "relaxed_diagnostic_no_paper_live_unlock": True,
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
        "top1_miss_diagnostics": validation_metrics["top1_miss_diagnostics"],
        "mean_regret_reduction_diagnostics": validation_metrics[
            "mean_regret_reduction_diagnostics"
        ],
        "largest_regret_case_diagnostics": validation_metrics[
            "largest_regret_case"
        ],
        "action_pair_regret_reduction_diagnostics": validation_metrics[
            "action_pair_regret_summary"
        ],
        "no_trade_missed_opportunity_diagnostics": validation_metrics[
            "no_trade_missed_opportunity"
        ],
        "gate_reason_code_consistency": gate_reason_code_consistency,
        "gate_reason_code_consistency_passed": gate_reason_code_consistency[
            "gate_reason_code_consistency_passed"
        ],
        "v8_full_decision_grid_summary": full_decision_grid_summary,
        "v8_action_rank_quality_passed": v8_action_rank_status[
            "v8_action_rank_quality_passed"
        ],
        "v8_action_rank_candidate_eligible": v8_action_rank_status[
            "v8_action_rank_candidate_eligible"
        ],
        "v8_action_rank_reason_codes": v8_action_rank_status[
            "v8_action_rank_reason_codes"
        ],
        "v8_action_rank_gate_summary": v8_action_rank_status[
            "v8_action_rank_gate_summary"
        ],
        "v8_execution_risk_control_required": v8_action_rank_status[
            "v8_execution_risk_control_required"
        ],
        "v8_execution_handoff_allowed": v8_action_rank_status[
            "v8_execution_handoff_allowed"
        ],
        "v8_execution_handoff_blocking_reason_codes": v8_action_rank_status[
            "v8_execution_handoff_blocking_reason_codes"
        ],
        "strict_source_gate_remains_failed": not source_model_candidate_eligible,
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
    selected_feature_names = tuple(model_training_summary["feature_names"])
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
        "v8_scope": "action_rank_signal_and_execution_layer_handoff_only",
        "model_layer_regret_risk_selection_enabled": False,
        "model_layer_regret_risk_selection_deferred_to_issue": "#158",
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
                    "features": _deployable_model_features(
                        row,
                        selected_feature_names,
                    ),
                    "target": row["replay_aligned_executable_label_target"],
                }
                for row in rows
                if row.get("split") == "shadow"
            ]
        ),
        "label_grid_hash": canonical_json_sha256(label_grid_payload),
        "feature_schema_hash": canonical_json_sha256(
            list(selected_feature_names)
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
        "strict_calibration_quality_passed": eligibility_gate_report[
            "strict_calibration_quality_passed"
        ],
        "relaxed_diagnostic_calibration_quality_passed": eligibility_gate_report[
            "relaxed_diagnostic_calibration_quality_passed"
        ],
        "relaxed_diagnostic_source_candidate": eligibility_gate_report[
            "relaxed_diagnostic_source_candidate"
        ],
        "relaxed_diagnostic_reason_codes": eligibility_gate_report[
            "relaxed_diagnostic_reason_codes"
        ],
        "strict_vs_relaxed_gate_summary": eligibility_gate_report[
            "strict_vs_relaxed_gate_summary"
        ],
        "relaxed_diagnostic_no_freeze_unlock": True,
        "relaxed_diagnostic_no_paper_live_unlock": True,
        "v8_action_rank_quality_passed": eligibility_gate_report[
            "v8_action_rank_quality_passed"
        ],
        "v8_action_rank_candidate_eligible": eligibility_gate_report[
            "v8_action_rank_candidate_eligible"
        ],
        "v8_action_rank_reason_codes": eligibility_gate_report[
            "v8_action_rank_reason_codes"
        ],
        "v8_action_rank_gate_summary": eligibility_gate_report[
            "v8_action_rank_gate_summary"
        ],
        "v8_execution_risk_control_required": eligibility_gate_report[
            "v8_execution_risk_control_required"
        ],
        "v8_execution_handoff_allowed": eligibility_gate_report[
            "v8_execution_handoff_allowed"
        ],
        "v8_execution_handoff_blocking_reason_codes": eligibility_gate_report[
            "v8_execution_handoff_blocking_reason_codes"
        ],
        "strict_source_gate_remains_failed": eligibility_gate_report[
            "strict_source_gate_remains_failed"
        ],
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


def _hts_p_up_confidently_wrong_feature_diagnostic_report(
    *,
    config: PolymarketOReplayAlignedSourceRankingConfig,
    m2_report_path: Path,
    m2_report: dict[str, Any],
    rows: list[dict[str, Any]],
    model_training_summary: dict[str, Any],
    eligibility_gate_report: dict[str, Any],
) -> dict[str, Any]:
    del config
    cases = _hts_p_up_confidently_wrong_feature_cases(rows)
    split_summaries = {
        split: _hts_p_up_confidently_wrong_case_summary(
            [case for case in cases if case["split"] == split]
        )
        for split in ("shadow", "validation", "all")
    }
    split_summaries["all"] = _hts_p_up_confidently_wrong_case_summary(cases)
    feature_coverage = _hts_p_up_confidently_wrong_feature_coverage(cases)
    alternatives = _hts_p_up_confidently_wrong_alternative_summary(cases)
    recommendations = _hts_p_up_confidently_wrong_recommendations(
        cases=cases,
        feature_coverage=feature_coverage,
        alternative_summary=alternatives,
    )
    report = {
        "schema_version": (
            O_HTS_P_UP_CONFIDENTLY_WRONG_FEATURE_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "candidate_name": O_MODEL_PREDICTED_VARIANT,
        "source_lineage": REPLAY_ALIGNED_SOURCE_RANKING_CANDIDATE_NAME,
        "report_type": "o_hts_p_up_confidently_wrong_feature_diagnostic",
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
        "uses_validation_labels_for_tuning": False,
        "report_uses_replay_outcomes_for_evaluation_only": True,
        "selection_or_gate_mutation": False,
        "eligibility_remains_validation_only": True,
        "model_input_fields_decision_time_only": list(
            O_DEPLOYABLE_MODEL_FEATURE_NAMES
        ),
        "forbidden_model_input_fields": list(O_FORBIDDEN_MODEL_INPUT_FIELDS),
        "forbidden_fields_used_for_selection": [],
        "report_only_evaluation_fields": [
            "oracle_executable_best_action",
            "oracle_executable_best_action_return",
            "realized_replay_return",
            "regret",
        ],
        "existing_decision_time_feature_families": [
            "action identity",
            "side identity",
            "action family identity",
            "p_up / p_down / p_up edge",
            "time_to_close",
            "spread",
            "queue_fill",
            "book_staleness",
            "entry_ask",
            "exit_bid_proxy",
            "action x p_up / p_down interactions",
            "action x microstructure interactions",
            "reference price distance and recent momentum",
            "side-specific book depth and update pressure",
            "HTS vs sell-before-close exit-value gap proxy",
            "shadow-only p_up bucket calibration residual",
        ],
        "missing_or_weak_decision_time_feature_candidates": (
            _hts_missing_or_weak_feature_candidates()
        ),
        "case_count": len(cases),
        "validation_case_count": split_summaries["validation"]["case_count"],
        "shadow_case_count": split_summaries["shadow"]["case_count"],
        "split_summaries": split_summaries,
        "feature_coverage_summary": feature_coverage,
        "alternative_comparison_summary": alternatives,
        "top_confidently_wrong_cases": sorted(
            cases,
            key=lambda case: (
                -float(case["regret"]),
                str(case["decision_group_id"]),
            ),
        )[:20],
        "top_validation_confidently_wrong_cases": sorted(
            [case for case in cases if case["split"] == "validation"],
            key=lambda case: (
                -float(case["regret"]),
                str(case["decision_group_id"]),
            ),
        )[:20],
        "diagnostic_conclusion": recommendations["diagnostic_conclusion"],
        "recommended_next_action": recommendations["recommended_next_action"],
        "recommendation_reason_codes": recommendations["reason_codes"],
        "gate_status_snapshot": {
            "source_model_candidate_eligible": eligibility_gate_report[
                "source_model_candidate_eligible"
            ],
            "promotion_evidence_eligible": eligibility_gate_report[
                "promotion_evidence_eligible"
            ],
            "calibration_quality_passed": eligibility_gate_report[
                "calibration_quality_passed"
            ],
            "p_up_action_disagreement_within_limit": eligibility_gate_report[
                "p_up_action_disagreement_within_limit"
            ],
            "p_up_safety_target_met": eligibility_gate_report[
                "p_up_safety_target_met"
            ],
            "action_family_paper_decision_eligible": eligibility_gate_report[
                "action_family_paper_decision_eligible"
            ],
            "ineligible_reason_codes": eligibility_gate_report[
                "ineligible_reason_codes"
            ],
        },
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **_fail_closed_fields(),
        **compact_safety_fields(),
    }
    report[
        "o_hts_p_up_confidently_wrong_feature_diagnostic_report_id"
    ] = canonical_json_sha256(report)
    return report


def _o_relaxed_diagnostic_gate_status(
    *,
    validation_metrics: dict[str, Any],
    p_up_action_disagreement_within_limit: bool,
    calibration_support_passed: bool,
    action_family_paper_decision_eligible: bool,
    best_action_concentration_passed: bool,
    high_score_return_positive: bool,
    leakage_passed: bool,
) -> dict[str, Any]:
    top1_passed = (
        float(validation_metrics["top1_realized_best_action_hit_rate"])
        >= O_MIN_TOP1_HIT_RATE
    )
    strict_mean_regret_passed = (
        float(validation_metrics["mean_regret"]) <= O_MAX_MEAN_REGRET
    )
    relaxed_mean_regret_passed = (
        float(validation_metrics["mean_regret"])
        <= O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    strict_calibration_quality_passed = top1_passed and strict_mean_regret_passed
    relaxed_diagnostic_calibration_quality_passed = (
        top1_passed and relaxed_mean_regret_passed
    )
    required_checks = {
        "relaxed_diagnostic_calibration_quality_passed": (
            relaxed_diagnostic_calibration_quality_passed
        ),
        "calibration_support_passed": calibration_support_passed,
        "p_up_action_disagreement_within_limit": (
            p_up_action_disagreement_within_limit
        ),
        "high_score_return_positive": high_score_return_positive,
        "action_family_paper_decision_eligible": (
            action_family_paper_decision_eligible
        ),
        "best_action_concentration_passed": best_action_concentration_passed,
        "leakage_audit_passed": leakage_passed,
    }
    reason_codes = [
        "diagnostic_only_no_paper_live_unlock",
        "future_unseen_holdout_required",
        "strict_source_gate_remains_authoritative",
    ]
    if not top1_passed:
        reason_codes.append("relaxed_diagnostic_top1_gate_failed")
    if not relaxed_mean_regret_passed:
        reason_codes.append("relaxed_diagnostic_mean_regret_gate_failed")
    if not calibration_support_passed:
        reason_codes.append("relaxed_diagnostic_calibration_support_failed")
    if not p_up_action_disagreement_within_limit:
        reason_codes.append("relaxed_diagnostic_p_up_gate_failed")
    if not high_score_return_positive:
        reason_codes.append("relaxed_diagnostic_high_score_return_failed")
    if not action_family_paper_decision_eligible:
        reason_codes.append("relaxed_diagnostic_action_family_gate_failed")
    if not best_action_concentration_passed:
        reason_codes.append("relaxed_diagnostic_best_action_concentration_failed")
    if not leakage_passed:
        reason_codes.append("relaxed_diagnostic_leakage_audit_failed")
    relaxed_diagnostic_source_candidate = all(required_checks.values())
    return {
        "strict_calibration_quality_passed": strict_calibration_quality_passed,
        "relaxed_diagnostic_calibration_quality_passed": (
            relaxed_diagnostic_calibration_quality_passed
        ),
        "relaxed_diagnostic_source_candidate": (
            relaxed_diagnostic_source_candidate
        ),
        "relaxed_diagnostic_reason_codes": sorted(set(reason_codes)),
        "strict_vs_relaxed_gate_summary": {
            "strict_source_gate_remains_authoritative": True,
            "relaxed_diagnostic_gate_is_diagnostic_only": True,
            "relaxed_diagnostic_no_paper_live_unlock": True,
            "future_unseen_holdout_required": True,
            "strict_max_mean_regret": O_MAX_MEAN_REGRET,
            "relaxed_diagnostic_max_mean_regret": (
                O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
            ),
            "validation_mean_regret": validation_metrics["mean_regret"],
            "validation_top1_realized_best_action_hit_rate": validation_metrics[
                "top1_realized_best_action_hit_rate"
            ],
            "strict_mean_regret_passed": strict_mean_regret_passed,
            "relaxed_diagnostic_mean_regret_passed": relaxed_mean_regret_passed,
            "top1_gate_passed": top1_passed,
            "strict_calibration_quality_passed": (
                strict_calibration_quality_passed
            ),
            "relaxed_diagnostic_calibration_quality_passed": (
                relaxed_diagnostic_calibration_quality_passed
            ),
            "relaxed_diagnostic_required_checks": required_checks,
            "relaxed_diagnostic_source_candidate": (
                relaxed_diagnostic_source_candidate
            ),
        },
    }


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


def _o_gate_reason_code_consistency_diagnostic(
    *,
    reason_codes: list[str],
    validation_metrics: dict[str, Any],
    calibration_support_passed: bool,
    calibration_quality_passed: bool,
    action_family_paper_decision_eligible: bool,
    best_action_concentration_passed: bool,
    p_up_action_disagreement_within_limit: bool,
    high_score_return_positive: bool,
) -> dict[str, Any]:
    expected_presence = {
        "validation_calibration_support_gate_failed": (
            not calibration_support_passed
        ),
        "validation_calibration_quality_gate_failed": (
            not calibration_quality_passed
        ),
        "validation_action_family_return_gate_failed": (
            not action_family_paper_decision_eligible
        ),
        "validation_best_action_concentration_gate_failed": (
            not best_action_concentration_passed
        ),
        "validation_p_up_action_disagreement_gate_failed": (
            not p_up_action_disagreement_within_limit
        ),
        "validation_high_score_return_gate_failed": (
            not high_score_return_positive
        ),
    }
    unexpected_reason_codes = sorted(
        reason
        for reason, should_be_present in expected_presence.items()
        if reason in reason_codes and not should_be_present
    )
    missing_reason_codes = sorted(
        reason
        for reason, should_be_present in expected_presence.items()
        if should_be_present and reason not in reason_codes
    )
    return {
        "gate_reason_code_consistency_passed": (
            not unexpected_reason_codes and not missing_reason_codes
        ),
        "unexpected_reason_codes": unexpected_reason_codes,
        "missing_reason_codes": missing_reason_codes,
        "high_score_return_gate_should_fail": not high_score_return_positive,
        "high_score_return_gate_reason_present": (
            "validation_high_score_return_gate_failed" in reason_codes
        ),
        "high_score_support_count": validation_metrics[
            "high_score_support_count"
        ],
        "min_high_score_support_count": O_MIN_HIGH_SCORE_SUPPORT_COUNT,
        "high_score_realized_return_mean": validation_metrics[
            "high_score_realized_return_mean"
        ],
        "high_score_realized_return_sum": validation_metrics[
            "high_score_realized_return_sum"
        ],
        "diagnostic_only": True,
    }


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


def _v8_decision_grid_summary(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split:
            grouped[str(row["decision_group_id"])].append(row)
    missing_by_group = []
    for decision_group_id, group_rows in sorted(grouped.items()):
        actions = sorted({str(row.get("action") or "") for row in group_rows})
        missing = sorted(set(O_REQUIRED_DECISION_ACTION_FAMILIES).difference(actions))
        if missing:
            missing_by_group.append(
                {
                    "decision_group_id": decision_group_id,
                    "available_action_families": actions,
                    "missing_action_families": missing,
                    "ranking_metric_scope": O_PARTIAL_DECISION_GROUP_SCOPE,
                }
            )
    return {
        "split": split,
        "decision_group_count": len(grouped),
        "required_action_families": list(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "complete_decision_group_count": len(grouped) - len(missing_by_group),
        "partial_decision_group_count": len(missing_by_group),
        "complete_5_action_decision_grid": bool(grouped) and not missing_by_group,
        "ranking_metric_scope": (
            O_FULL_DECISION_GROUP_SCOPE
            if grouped and not missing_by_group
            else O_PARTIAL_DECISION_GROUP_SCOPE
        ),
        "incomplete_decision_group_examples": missing_by_group[:20],
    }


def _o_v8_action_rank_gate_status(
    *,
    validation_metrics: dict[str, Any],
    deployable_model_score_available: bool,
    calibration_support_passed: bool,
    action_family_paper_decision_eligible: bool,
    best_action_concentration_passed: bool,
    p_up_action_disagreement_within_limit: bool,
    high_score_return_positive: bool,
    leakage_audit_passed: bool,
    expanded_feature_provenance_passed: bool,
    gate_reason_code_consistency_passed: bool,
    full_decision_grid_summary: dict[str, Any],
) -> dict[str, Any]:
    top1_passed = (
        float(validation_metrics["top1_realized_best_action_hit_rate"])
        >= O_MIN_TOP1_HIT_RATE
    )
    strict_mean_regret_passed = (
        float(validation_metrics["mean_regret"]) <= O_MAX_MEAN_REGRET
    )
    relaxed_mean_regret_passed = (
        float(validation_metrics["mean_regret"])
        <= O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
    )
    selected_return_positive = (
        float(validation_metrics["selected_action_realized_replay_return_sum"]) > 0.0
    )
    full_decision_grid_complete = bool(
        full_decision_grid_summary["complete_5_action_decision_grid"]
    )
    required_checks = {
        "deployable_model_score_available": deployable_model_score_available,
        "top1_gate_passed": top1_passed,
        "relaxed_mean_regret_gate_passed": relaxed_mean_regret_passed,
        "calibration_support_passed": calibration_support_passed,
        "p_up_action_disagreement_within_limit": (
            p_up_action_disagreement_within_limit
        ),
        "high_score_return_positive": high_score_return_positive,
        "selected_return_positive": selected_return_positive,
        "action_family_paper_decision_eligible": (
            action_family_paper_decision_eligible
        ),
        "best_action_concentration_passed": best_action_concentration_passed,
        "leakage_audit_passed": leakage_audit_passed,
        "expanded_feature_provenance_passed": expanded_feature_provenance_passed,
        "gate_reason_code_consistency_passed": gate_reason_code_consistency_passed,
        "full_5_action_decision_grid_complete": full_decision_grid_complete,
    }
    reason_codes = [
        "diagnostic_only_no_paper_live_unlock",
        "execution_layer_risk_control_required",
        "future_unseen_holdout_required",
        "strict_source_gate_remains_authoritative",
    ]
    if not deployable_model_score_available:
        reason_codes.append("v8_action_rank_deployable_model_score_unavailable")
    if not top1_passed:
        reason_codes.append("v8_action_rank_top1_gate_failed")
    if not relaxed_mean_regret_passed:
        reason_codes.append("v8_action_rank_relaxed_mean_regret_gate_failed")
    if not calibration_support_passed:
        reason_codes.append("v8_action_rank_calibration_support_failed")
    if not p_up_action_disagreement_within_limit:
        reason_codes.append("v8_action_rank_p_up_gate_failed")
    if not high_score_return_positive:
        reason_codes.append("v8_action_rank_high_score_return_failed")
    if not selected_return_positive:
        reason_codes.append("v8_action_rank_selected_return_not_positive")
    if not action_family_paper_decision_eligible:
        reason_codes.append("v8_action_rank_action_family_gate_failed")
    if not best_action_concentration_passed:
        reason_codes.append("v8_action_rank_best_action_concentration_failed")
    if not leakage_audit_passed:
        reason_codes.append("v8_action_rank_leakage_audit_failed")
    if not expanded_feature_provenance_passed:
        reason_codes.append("v8_action_rank_feature_provenance_failed")
    if not gate_reason_code_consistency_passed:
        reason_codes.append("v8_action_rank_gate_reason_code_consistency_failed")
    if not full_decision_grid_complete:
        reason_codes.append("v8_action_rank_full_decision_grid_incomplete")
    v8_action_rank_quality_passed = all(required_checks.values())
    execution_blocking_reasons = [
        "execution_layer_runtime_risk_control_not_validated",
        "paper_live_unlock_prohibited",
        "future_unseen_holdout_required",
    ]
    if not v8_action_rank_quality_passed:
        execution_blocking_reasons.append("v8_action_rank_quality_not_passed")
    return {
        "v8_action_rank_quality_passed": v8_action_rank_quality_passed,
        "v8_action_rank_candidate_eligible": v8_action_rank_quality_passed,
        "v8_action_rank_reason_codes": sorted(set(reason_codes)),
        "v8_action_rank_gate_summary": {
            "gate_is_diagnostic_only": True,
            "strict_source_gate_remains_authoritative": True,
            "strict_max_mean_regret": O_MAX_MEAN_REGRET,
            "relaxed_diagnostic_max_mean_regret": (
                O_RELAXED_DIAGNOSTIC_MAX_MEAN_REGRET
            ),
            "min_top1_realized_best_action_hit_rate": O_MIN_TOP1_HIT_RATE,
            "min_high_score_support_count": O_MIN_HIGH_SCORE_SUPPORT_COUNT,
            "max_p_up_action_disagreement_rate": (
                O_MAX_P_UP_ACTION_DISAGREEMENT_RATE
            ),
            "validation_top1_realized_best_action_hit_rate": validation_metrics[
                "top1_realized_best_action_hit_rate"
            ],
            "validation_mean_regret": validation_metrics["mean_regret"],
            "validation_selected_return_sum": validation_metrics[
                "selected_action_realized_replay_return_sum"
            ],
            "validation_high_score_support_count": validation_metrics[
                "high_score_support_count"
            ],
            "validation_high_score_realized_return_mean": validation_metrics[
                "high_score_realized_return_mean"
            ],
            "validation_high_score_realized_return_sum": validation_metrics[
                "high_score_realized_return_sum"
            ],
            "strict_mean_regret_passed": strict_mean_regret_passed,
            "relaxed_mean_regret_gate_passed": relaxed_mean_regret_passed,
            "top1_gate_passed": top1_passed,
            "required_checks": required_checks,
        },
        "v8_execution_risk_control_required": True,
        "v8_execution_handoff_allowed": False,
        "v8_execution_handoff_blocking_reason_codes": sorted(
            set(execution_blocking_reasons)
        ),
    }


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
    all_action_pair_cases = []
    top1_miss_cases = []
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
        top2_contains_oracle = oracle["action"] in {
            row["action"] for row in predicted[:2]
        }
        top3_contains_oracle = oracle["action"] in {
            row["action"] for row in predicted[:3]
        }
        selected_components = selected.get("o_model_score_components") or {}
        selected_p_up = _bounded(float(selected.get("p_up") or 0.5), 0.0, 1.0)
        selected_p_down = 1.0 - selected_p_up
        selected_action = str(selected.get("action") or "")
        oracle_action = str(oracle.get("action") or "")
        selected_score = float(selected["variant_scores"][variant])
        oracle_score = float(oracle["variant_scores"][variant])
        p_up_conflicts_with_selected_side = (
            ("BUY_UP" in selected_action and selected_p_up < 0.50)
            or ("BUY_DOWN" in selected_action and selected_p_up > 0.50)
        )
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
            "selected_score": selected_score,
            "oracle_score": oracle_score,
            "score_margin_selected_minus_oracle": selected_score - oracle_score,
            "selected_is_high_score": selected_score >= high_score_threshold,
            "p_up": selected_p_up,
            "p_down": selected_p_down,
            "p_up_conflicts_with_selected_side": p_up_conflicts_with_selected_side,
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
        all_action_pair_cases.append(action_pair_case)
        if selected_action != oracle_action:
            top1_miss_cases.append(
                {
                    **action_pair_case,
                    "top2_contains_oracle_action": top2_contains_oracle,
                    "top3_contains_oracle_action": top3_contains_oracle,
                }
            )
        if selected.get("action_family") == "HOLD_TO_SETTLEMENT":
            hts_p_up_reliability_cases.append(
                {
                    **action_pair_case,
                    **_static_hts_reliability_buckets(selected),
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
        "top1_miss_diagnostics": _top1_miss_diagnostics(
            top1_miss_cases,
            group_count=group_count,
        ),
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
        "mean_regret_reduction_diagnostics": (
            _mean_regret_reduction_diagnostics(
                all_action_pair_cases,
                no_trade_missed_opportunities=no_trade_missed_opportunities,
            )
        ),
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


def _top1_miss_diagnostics(
    cases: list[dict[str, Any]],
    *,
    group_count: int,
) -> dict[str, Any]:
    regret_values = [float(case["regret"]) for case in cases]
    positive_regret_values = [value for value in regret_values if value > 0.0]
    by_action_pair: Counter[tuple[str, str]] = Counter()
    by_family_pair: Counter[tuple[str, str]] = Counter()
    by_side_pair: Counter[tuple[str, str]] = Counter()
    regret_by_miss_type: dict[str, list[float]] = defaultdict(list)
    p_up_conflict_count = 0
    for case in cases:
        by_action_pair[(str(case["selected_action"]), str(case["oracle_action"]))] += 1
        by_family_pair[
            (
                str(case["selected_action_family"]),
                str(case["oracle_action_family"]),
            )
        ] += 1
        by_side_pair[(str(case["selected_side"]), str(case["oracle_side"]))] += 1
        if bool(case["top2_contains_oracle_action"]):
            regret_by_miss_type["top2_contains_oracle"].append(float(case["regret"]))
        elif bool(case["top3_contains_oracle_action"]):
            regret_by_miss_type["top3_contains_oracle"].append(float(case["regret"]))
        else:
            regret_by_miss_type["oracle_outside_top3"].append(float(case["regret"]))
        if bool(case["p_up_conflicts_with_selected_side"]):
            p_up_conflict_count += 1
    return {
        "top1_miss_count": len(cases),
        "decision_group_count": group_count,
        "top1_miss_rate": (len(cases) / group_count) if group_count else 0.0,
        "top2_contained_miss_count": sum(
            1 for case in cases if bool(case["top2_contains_oracle_action"])
        ),
        "top3_contained_miss_count": sum(
            1 for case in cases if bool(case["top3_contains_oracle_action"])
        ),
        "top2_contained_miss_rate": (
            sum(1 for case in cases if bool(case["top2_contains_oracle_action"]))
            / len(cases)
        )
        if cases
        else 0.0,
        "top3_contained_miss_rate": (
            sum(1 for case in cases if bool(case["top3_contains_oracle_action"]))
            / len(cases)
        )
        if cases
        else 0.0,
        "p_up_conflicting_top1_miss_count": p_up_conflict_count,
        "p_up_conflicting_top1_miss_rate": (
            p_up_conflict_count / len(cases)
        )
        if cases
        else 0.0,
        "top1_miss_regret_sum": sum(regret_values),
        "positive_top1_miss_regret_sum": sum(positive_regret_values),
        "top1_miss_regret_mean": statistics.mean(regret_values)
        if regret_values
        else 0.0,
        "positive_top1_miss_regret_mean": statistics.mean(positive_regret_values)
        if positive_regret_values
        else 0.0,
        "action_pair_confusion": [
            {
                "selected_action": selected,
                "oracle_action": oracle,
                "count": count,
            }
            for (selected, oracle), count in sorted(by_action_pair.items())
        ],
        "family_pair_confusion": [
            {
                "selected_family": selected,
                "oracle_family": oracle,
                "count": count,
            }
            for (selected, oracle), count in sorted(by_family_pair.items())
        ],
        "side_pair_confusion": [
            {
                "selected_side": selected,
                "oracle_side": oracle,
                "count": count,
            }
            for (selected, oracle), count in sorted(by_side_pair.items())
        ],
        "regret_contribution_by_miss_type": {
            miss_type: {
                "count": len(values),
                "regret_sum": sum(values),
                "regret_mean": statistics.mean(values) if values else 0.0,
            }
            for miss_type, values in sorted(regret_by_miss_type.items())
        },
        "top_miss_cases": sorted(
            cases,
            key=lambda case: (-float(case["regret"]), str(case["decision_group_id"])),
        )[:10],
    }


def _mean_regret_reduction_diagnostics(
    cases: list[dict[str, Any]],
    *,
    no_trade_missed_opportunities: list[float],
) -> dict[str, Any]:
    regrets = [float(case["regret"]) for case in cases]
    positive_regrets = [value for value in regrets if value > 0.0]
    by_action_pair: dict[str, list[float]] = defaultdict(list)
    by_family_pair: dict[str, list[float]] = defaultdict(list)
    by_side_pair: dict[str, list[float]] = defaultdict(list)
    by_p_up_alignment: dict[str, list[float]] = defaultdict(list)
    by_high_score: dict[str, list[float]] = defaultdict(list)
    by_hts_sbc: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        action_key = f"{case['selected_action']}->{case['oracle_action']}"
        family_key = (
            f"{case['selected_action_family']}->{case['oracle_action_family']}"
        )
        side_key = f"{case['selected_side']}->{case['oracle_side']}"
        by_action_pair[action_key].append(float(case["regret"]))
        by_family_pair[family_key].append(float(case["regret"]))
        by_side_pair[side_key].append(float(case["regret"]))
        alignment_key = (
            "p_up_disagreement"
            if bool(case["p_up_conflicts_with_selected_side"])
            else "p_up_agreement_or_not_applicable"
        )
        by_p_up_alignment[alignment_key].append(float(case["regret"]))
        high_score_key = (
            "high_score_selected"
            if bool(case["selected_is_high_score"])
            else "non_high_score_selected"
        )
        by_high_score[high_score_key].append(float(case["regret"]))
        selected_family = str(case["selected_action_family"])
        oracle_family = str(case["oracle_action_family"])
        if {
            selected_family,
            oracle_family,
        } <= {"HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE"}:
            by_hts_sbc[f"{selected_family}->{oracle_family}"].append(
                float(case["regret"])
            )
    no_trade_positive = [
        float(value) for value in no_trade_missed_opportunities if float(value) > 0.0
    ]
    return {
        "case_count": len(cases),
        "mean_regret": statistics.mean(regrets) if regrets else 0.0,
        "positive_regret_sum": sum(positive_regrets),
        "positive_regret_mean": statistics.mean(positive_regrets)
        if positive_regrets
        else 0.0,
        "largest_regret_cases": sorted(
            cases,
            key=lambda case: (-float(case["regret"]), str(case["decision_group_id"])),
        )[:10],
        "regret_by_selected_action_vs_oracle_action": _regret_bucket_rows(
            by_action_pair
        ),
        "regret_by_selected_family_vs_oracle_family": _regret_bucket_rows(
            by_family_pair
        ),
        "regret_by_selected_side_vs_oracle_side": _regret_bucket_rows(by_side_pair),
        "regret_by_p_up_agreement": _regret_bucket_rows(by_p_up_alignment),
        "regret_by_high_score_selected": _regret_bucket_rows(by_high_score),
        "hts_vs_sbc_regret_contribution": _regret_bucket_rows(by_hts_sbc),
        "no_trade_missed_opportunity_contribution": {
            "missed_positive_opportunity_count": len(no_trade_positive),
            "missed_positive_opportunity_sum": sum(no_trade_positive),
            "missed_positive_opportunity_mean": statistics.mean(no_trade_positive)
            if no_trade_positive
            else 0.0,
            "max_missed_positive_opportunity": max(no_trade_positive, default=0.0),
        },
        "diagnostic_only": True,
    }


def _regret_bucket_rows(values_by_key: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for key, values in values_by_key.items():
        positive_values = [value for value in values if value > 0.0]
        rows.append(
            {
                "bucket": key,
                "count": len(values),
                "regret_sum": sum(values),
                "regret_mean": statistics.mean(values) if values else 0.0,
                "positive_regret_sum": sum(positive_values),
                "positive_regret_mean": statistics.mean(positive_values)
                if positive_values
                else 0.0,
                "regret_max": max(values, default=0.0),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["positive_regret_sum"]),
            -float(row["regret_max"]),
            str(row["bucket"]),
        ),
    )


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
        "top1_miss_diagnostics": metrics["top1_miss_diagnostics"],
        "mean_regret_reduction_diagnostics": metrics[
            "mean_regret_reduction_diagnostics"
        ],
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


def _static_hts_reliability_buckets(row: dict[str, Any]) -> dict[str, Any]:
    p_edge = abs(_bounded(float(row.get("p_up") or 0.5), 0.0, 1.0) - 0.5)
    return {
        "p_up_confidence_bucket": _static_p_up_confidence_bucket(p_edge),
        "time_to_close_bucket": _time_to_close_bucket(row),
        "spread_bucket": _spread_bucket(row),
        "queue_bucket": _queue_bucket(row),
        "staleness_bucket": _staleness_bucket(row),
    }


def _hts_p_up_confidently_wrong_feature_cases(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["decision_group_id"]].append(row)
    cases = []
    for group_rows in grouped.values():
        selected = max(
            group_rows,
            key=lambda row: float(row["variant_scores"][O_MODEL_PREDICTED_VARIANT]),
        )
        if str(selected.get("action_family")) != "HOLD_TO_SETTLEMENT":
            continue
        oracle = max(group_rows, key=lambda row: float(row["realized_replay_return"]))
        selected_side = str(selected.get("selected_side") or "NONE")
        oracle_side = str(oracle.get("selected_side") or "NONE")
        if selected_side == oracle_side:
            continue
        p_up_implied_side = _p_up_implied_side(selected)
        if selected_side != p_up_implied_side:
            continue
        selected_score = float(selected["variant_scores"][O_MODEL_PREDICTED_VARIANT])
        oracle_score = float(oracle["variant_scores"][O_MODEL_PREDICTED_VARIANT])
        selected_return = float(selected["realized_replay_return"])
        oracle_return = float(oracle["realized_replay_return"])
        feature_snapshot = _hts_p_up_feature_snapshot(selected)
        alternatives = _hts_p_up_confidently_wrong_alternatives(
            selected=selected,
            group_rows=group_rows,
        )
        cases.append(
            {
                "decision_group_id": selected["decision_group_id"],
                "market_id": selected.get("market_id"),
                "decision_ts": selected.get("decision_ts"),
                "split": selected.get("split"),
                "selected_action": selected.get("action"),
                "selected_side": selected_side,
                "oracle_action": oracle.get("action"),
                "oracle_side": oracle_side,
                "oracle_action_family": oracle.get("action_family"),
                "p_up_implied_side": p_up_implied_side,
                "p_up_confidently_wrong": True,
                "p_up_confidence_bucket": feature_snapshot[
                    "p_up_confidence_bucket"
                ],
                "selected_score": selected_score,
                "oracle_score": oracle_score,
                "score_gap_selected_minus_oracle": selected_score - oracle_score,
                "selected_return": selected_return,
                "oracle_return": oracle_return,
                "regret": oracle_return - selected_return,
                "feature_snapshot": feature_snapshot,
                "raw_model_score": selected.get("o_raw_ridge_model_score"),
                "group_normalized_raw_model_score": selected.get(
                    "o_group_normalized_raw_model_score"
                ),
                "score_components": selected.get("o_model_score_components") or {},
                "alternative_actions": alternatives,
                "alternative_return_better_than_selected": [
                    name
                    for name, alternative in alternatives.items()
                    if alternative
                    and float(alternative["return_gap_vs_selected"]) > 0.0
                ],
                "alternative_score_close_to_selected": [
                    name
                    for name, alternative in alternatives.items()
                    if alternative
                    and float(alternative["score_gap_selected_minus_alternative"])
                    <= float(
                        feature_snapshot["p_up_edge"]
                    )
                ],
                "decision_time_feature_gaps": (
                    _hts_p_up_decision_time_feature_gap_codes(selected)
                ),
            }
        )
    return sorted(
        cases,
        key=lambda case: (-float(case["regret"]), str(case["decision_group_id"])),
    )


def _hts_p_up_feature_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    p_up = _bounded(float(row.get("p_up") or 0.5), 0.0, 1.0)
    p_down = 1.0 - p_up
    p_edge = abs(p_up - 0.5)
    buckets = row.get("hts_p_up_reliability_buckets") or {}
    return {
        "p_up": p_up,
        "p_down": p_down,
        "p_up_edge": p_edge,
        "p_up_confidence_bucket": buckets.get(
            "p_up_confidence_bucket",
            _static_p_up_confidence_bucket(p_edge),
        ),
        "time_to_close_seconds": row.get(
            "entry_exit_quality_time_to_close_seconds"
        ),
        "time_to_close_bucket": buckets.get(
            "time_to_close_bucket",
            row.get("time_to_close_bucket"),
        ),
        "spread_bps": row.get("entry_exit_quality_spread_bps"),
        "spread_bucket": buckets.get("spread_bucket", row.get("spread_bucket")),
        "queue_fill": row.get("entry_exit_quality_queue_fill"),
        "queue_bucket": buckets.get("queue_bucket", row.get("queue_bucket")),
        "book_staleness_ms": row.get("entry_exit_quality_book_staleness_ms"),
        "staleness_bucket": buckets.get(
            "staleness_bucket",
            row.get("staleness_bucket"),
        ),
        "entry_ask": row.get("entry_quality_ask"),
        "exit_bid_proxy": _decision_time_exit_bid_proxy(row),
        "immediate_exit_pnl_proxy": _immediate_exit_pnl(row),
        "p_up_alignment_score": _p_up_side_alignment_score(row),
        "reference_price_to_beat_distance_at_decision": row.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "recent_reference_price_momentum_30s": row.get(
            "recent_reference_price_momentum_30s"
        ),
        "recent_reference_price_momentum_60s": row.get(
            "recent_reference_price_momentum_60s"
        ),
        "recent_reference_price_momentum_120s": row.get(
            "recent_reference_price_momentum_120s"
        ),
        "side_book_depth_imbalance": row.get("side_book_depth_imbalance"),
        "side_book_update_velocity": row.get("side_book_update_velocity"),
        "side_book_staleness_ms": row.get("side_book_staleness_ms"),
        "opposite_book_staleness_ms": row.get("opposite_book_staleness_ms"),
        "side_spread_bps": row.get("side_spread_bps"),
        "side_queue_fill_proxy": row.get("side_queue_fill_proxy"),
        "hts_vs_sell_before_close_exit_value_gap_proxy": row.get(
            "hts_vs_sell_before_close_exit_value_gap_proxy"
        ),
        "p_up_calibration_residual_by_time_spread_queue_bucket": row.get(
            "p_up_calibration_residual_by_time_spread_queue_bucket"
        ),
        "decision_time_feature_missing_reason_codes": row.get(
            "decision_time_feature_missing_reason_codes",
            [],
        ),
    }


def _static_p_up_confidence_bucket(p_edge: float) -> str:
    if p_edge <= 0.05:
        return "weak"
    if p_edge <= 0.10:
        return "moderate"
    if p_edge <= 0.20:
        return "strong"
    return "very_strong"


def _hts_p_up_confidently_wrong_alternatives(
    *,
    selected: dict[str, Any],
    group_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    selected_side = str(selected.get("selected_side") or "NONE")
    opposite_side = "DOWN" if selected_side == "UP" else "UP"
    return {
        "NO_TRADE": _alternative_row(
            selected,
            _find_action(group_rows, "NO_TRADE"),
        ),
        "same_side_sell_before_close": _alternative_row(
            selected,
            _find_action(group_rows, f"BUY_{selected_side}_SELL_BEFORE_CLOSE"),
        ),
        "opposite_side_sell_before_close": _alternative_row(
            selected,
            _find_action(group_rows, f"BUY_{opposite_side}_SELL_BEFORE_CLOSE"),
        ),
        "opposite_hts_side": _alternative_row(
            selected,
            _find_action(group_rows, f"BUY_{opposite_side}_HOLD_TO_SETTLEMENT"),
        ),
        "best_sell_before_close_by_score": _alternative_row(
            selected,
            max(
                [
                    row
                    for row in group_rows
                    if row.get("action_family") == "SELL_BEFORE_CLOSE"
                ],
                key=lambda row: float(
                    row["variant_scores"][O_MODEL_PREDICTED_VARIANT]
                ),
                default=None,
            ),
        ),
        "best_sell_before_close_by_return": _alternative_row(
            selected,
            max(
                [
                    row
                    for row in group_rows
                    if row.get("action_family") == "SELL_BEFORE_CLOSE"
                ],
                key=lambda row: float(row["realized_replay_return"]),
                default=None,
            ),
        ),
    }


def _find_action(
    group_rows: list[dict[str, Any]],
    action: str,
) -> dict[str, Any] | None:
    return next(
        (row for row in group_rows if str(row.get("action") or "") == action),
        None,
    )


def _alternative_row(
    selected: dict[str, Any],
    alternative: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if alternative is None:
        return None
    selected_score = float(selected["variant_scores"][O_MODEL_PREDICTED_VARIANT])
    alternative_score = float(
        alternative["variant_scores"][O_MODEL_PREDICTED_VARIANT]
    )
    selected_return = float(selected["realized_replay_return"])
    alternative_return = float(alternative["realized_replay_return"])
    return {
        "action": alternative.get("action"),
        "action_family": alternative.get("action_family"),
        "side": alternative.get("selected_side"),
        "score": alternative_score,
        "return": alternative_return,
        "score_gap_selected_minus_alternative": selected_score - alternative_score,
        "return_gap_vs_selected": alternative_return - selected_return,
        "raw_model_score": alternative.get("o_raw_ridge_model_score"),
        "group_normalized_raw_model_score": alternative.get(
            "o_group_normalized_raw_model_score"
        ),
        "score_components": alternative.get("o_model_score_components") or {},
    }


def _hts_p_up_decision_time_feature_gap_codes(row: dict[str, Any]) -> list[str]:
    gaps = []
    if row.get("entry_quality_ask") is None:
        gaps.append("missing_entry_ask")
    if row.get("entry_exit_quality_spread_bps") is None:
        gaps.append("missing_spread_bps")
    if row.get("entry_exit_quality_queue_fill") is None:
        gaps.append("missing_queue_fill")
    if row.get("entry_exit_quality_book_staleness_ms") is None:
        gaps.append("missing_book_staleness")
    if row.get("entry_exit_quality_time_to_close_seconds") is None:
        gaps.append("missing_time_to_close")
    if row.get("reference_price_to_beat_distance_at_decision") is None:
        gaps.append("missing_reference_price_to_beat_distance")
    if row.get("recent_reference_price_momentum_120s") is None:
        gaps.append("missing_reference_price_momentum_120s")
    if row.get("side_book_depth_imbalance") is None:
        gaps.append("missing_side_book_depth_imbalance")
    if row.get("side_book_update_velocity") is None:
        gaps.append("missing_side_book_update_velocity")
    if row.get("hts_vs_sell_before_close_exit_value_gap_proxy") is None:
        gaps.append("missing_hts_vs_sell_before_close_exit_value_gap")
    if float(row.get("entry_exit_quality_spread_bps") or 0.0) >= 600.0:
        gaps.append("wide_spread_regime")
    if float(row.get("entry_exit_quality_queue_fill") or 1.0) < 0.80:
        gaps.append("weak_queue_fill_regime")
    if float(row.get("entry_exit_quality_book_staleness_ms") or 0.0) >= 5000.0:
        gaps.append("stale_book_regime")
    if _normalized_time_to_close(row) <= 3.0:
        gaps.append("near_expiry_regime")
    return gaps


def _hts_p_up_confidently_wrong_case_summary(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    regrets = [float(case["regret"]) for case in cases]
    return {
        "case_count": len(cases),
        "regret_sum": sum(regrets),
        "regret_mean": statistics.mean(regrets) if regrets else 0.0,
        "regret_max": max(regrets, default=0.0),
        "by_p_up_confidence_bucket": _case_count_summary(
            cases,
            lambda case: case["p_up_confidence_bucket"],
        ),
        "by_selected_vs_oracle_side": _case_count_summary(
            cases,
            lambda case: f"{case['selected_side']}->{case['oracle_side']}",
        ),
        "by_time_to_close_bucket": _case_count_summary(
            cases,
            lambda case: case["feature_snapshot"]["time_to_close_bucket"],
        ),
        "by_spread_bucket": _case_count_summary(
            cases,
            lambda case: case["feature_snapshot"]["spread_bucket"],
        ),
        "by_queue_bucket": _case_count_summary(
            cases,
            lambda case: case["feature_snapshot"]["queue_bucket"],
        ),
        "by_staleness_bucket": _case_count_summary(
            cases,
            lambda case: case["feature_snapshot"]["staleness_bucket"],
        ),
    }


def _case_count_summary(
    cases: list[dict[str, Any]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(key_fn(case))].append(case)
    return {
        key: {
            "count": len(group_cases),
            "regret_sum": sum(float(case["regret"]) for case in group_cases),
            "regret_mean": statistics.mean(
                float(case["regret"]) for case in group_cases
            ),
            "regret_max": max(float(case["regret"]) for case in group_cases),
        }
        for key, group_cases in sorted(grouped.items())
    }


def _hts_p_up_confidently_wrong_feature_coverage(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    required_fields = {
        "entry_ask": ("feature_snapshot", "entry_ask"),
        "exit_bid_proxy": ("feature_snapshot", "exit_bid_proxy"),
        "spread_bps": ("feature_snapshot", "spread_bps"),
        "queue_fill": ("feature_snapshot", "queue_fill"),
        "book_staleness_ms": ("feature_snapshot", "book_staleness_ms"),
        "time_to_close_seconds": ("feature_snapshot", "time_to_close_seconds"),
        "raw_model_score": ("raw_model_score",),
        "score_components": ("score_components",),
        "reference_price_to_beat_distance_at_decision": (
            "feature_snapshot",
            "reference_price_to_beat_distance_at_decision",
        ),
        "recent_reference_price_momentum_30s": (
            "feature_snapshot",
            "recent_reference_price_momentum_30s",
        ),
        "recent_reference_price_momentum_60s": (
            "feature_snapshot",
            "recent_reference_price_momentum_60s",
        ),
        "side_book_depth_imbalance": (
            "feature_snapshot",
            "side_book_depth_imbalance",
        ),
        "side_book_update_velocity": (
            "feature_snapshot",
            "side_book_update_velocity",
        ),
        "hts_vs_sell_before_close_exit_value_gap_proxy": (
            "feature_snapshot",
            "hts_vs_sell_before_close_exit_value_gap_proxy",
        ),
        "p_up_calibration_residual_by_time_spread_queue_bucket": (
            "feature_snapshot",
            "p_up_calibration_residual_by_time_spread_queue_bucket",
        ),
    }
    missing_counts = {}
    for field, path in required_fields.items():
        missing_counts[field] = sum(
            1
            for case in cases
            if _nested_case_value(case, path) in (None, {}, [])
        )
    complete_count = sum(
        1
        for case in cases
        if not any(
            _nested_case_value(case, path) in (None, {}, [])
            for path in required_fields.values()
        )
    )
    gap_counts = Counter(
        gap
        for case in cases
        for gap in case.get("decision_time_feature_gaps", [])
    )
    missing_critical_field_names = [
        field for field, count in missing_counts.items() if count > 0
    ]
    missing_critical_field_occurrence_count = sum(
        missing_counts[field] for field in missing_critical_field_names
    )
    return {
        "case_count": len(cases),
        "complete_decision_time_feature_case_count": complete_count,
        "all_cases_have_complete_decision_time_features": complete_count
        == len(cases),
        "missing_critical_field_count": len(missing_critical_field_names),
        "missing_critical_field_names": missing_critical_field_names,
        "missing_critical_field_occurrence_count": (
            missing_critical_field_occurrence_count
        ),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "decision_time_feature_gap_counts": dict(sorted(gap_counts.items())),
        "existing_features_insufficient": bool(cases),
        "existing_feature_coverage_insufficient": bool(cases)
        and complete_count < len(cases),
    }


def _nested_case_value(case: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = case
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _hts_p_up_confidently_wrong_alternative_summary(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    alternative_names = (
        "NO_TRADE",
        "same_side_sell_before_close",
        "opposite_side_sell_before_close",
        "opposite_hts_side",
        "best_sell_before_close_by_score",
        "best_sell_before_close_by_return",
    )
    summary = {}
    for name in alternative_names:
        alternatives = [
            case["alternative_actions"].get(name)
            for case in cases
            if case["alternative_actions"].get(name)
        ]
        return_gaps = [
            float(alternative["return_gap_vs_selected"])
            for alternative in alternatives
        ]
        score_gaps = [
            float(alternative["score_gap_selected_minus_alternative"])
            for alternative in alternatives
        ]
        positive_return_gaps = [gap for gap in return_gaps if gap > 0.0]
        summary[name] = {
            "available_count": len(alternatives),
            "better_return_count": len(positive_return_gaps),
            "better_return_rate": len(positive_return_gaps) / len(alternatives)
            if alternatives
            else 0.0,
            "return_gap_sum": sum(return_gaps),
            "return_gap_mean": statistics.mean(return_gaps)
            if return_gaps
            else 0.0,
            "score_gap_selected_minus_alternative_mean": statistics.mean(score_gaps)
            if score_gaps
            else 0.0,
        }
    return summary


def _hts_missing_or_weak_feature_candidates() -> list[dict[str, str]]:
    return [
        {
            "feature": "reference_price_to_beat_distance_at_decision",
            "reason": "HTS UP/DOWN settlement depends on price-to-beat distance, not just market p_up.",
        },
        {
            "feature": "recent_reference_price_momentum_30s_60s_120s",
            "reason": "p_up can be stale or confidently wrong when reference momentum reverses near expiry.",
        },
        {
            "feature": "outcome_book_depth_imbalance_and_update_velocity",
            "reason": "Current spread/queue/staleness features do not capture side-specific book pressure.",
        },
        {
            "feature": "hts_vs_sell_before_close_exit_value_gap_proxy",
            "reason": "Large regret cases often need a decision-time comparison between waiting for settlement and exiting early.",
        },
        {
            "feature": "p_up_calibration_residual_by_time_spread_queue_bucket",
            "reason": "p_up passes global disagreement gates but can still be unreliable in local HTS regimes.",
        },
    ]


def _hts_p_up_confidently_wrong_recommendations(
    *,
    cases: list[dict[str, Any]],
    feature_coverage: dict[str, Any],
    alternative_summary: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = []
    if cases:
        reason_codes.append("hts_p_up_confidently_wrong_cases_present")
    if bool(feature_coverage["existing_feature_coverage_insufficient"]):
        reason_codes.append("decision_time_feature_coverage_missing_critical_fields")
    if bool(feature_coverage["existing_features_insufficient"]):
        reason_codes.append("existing_decision_time_features_do_not_separate_cases")
    if alternative_summary["opposite_hts_side"]["better_return_count"] > 0:
        reason_codes.append("opposite_hts_side_often_realizes_better_return")
    if alternative_summary["best_sell_before_close_by_return"]["better_return_count"] > 0:
        reason_codes.append("sell_before_close_alternative_can_reduce_regret")
    if alternative_summary["NO_TRADE"]["better_return_count"] > 0:
        reason_codes.append("no_trade_can_reduce_tail_regret")
    if not reason_codes:
        reason_codes.append("no_confidently_wrong_hts_cases_detected")
    missing_fields = set(feature_coverage.get("missing_critical_field_names", []))
    if (
        cases
        and bool(feature_coverage["existing_feature_coverage_insufficient"])
        and missing_fields == {"reference_price_to_beat_distance_at_decision"}
    ):
        conclusion = "reference_price_to_beat_distance_unavailable_for_hts_reliability"
        next_action = (
            "collect_reference_price_to_beat_distance_before_further_hts_priority_"
            "changes"
        )
    elif cases and bool(feature_coverage["existing_feature_coverage_insufficient"]):
        conclusion = "decision_time_feature_coverage_insufficient_for_hts_reliability"
        next_action = (
            "add_new_decision_time_reference_and_book_pressure_features_before_"
            "further_hts_priority_changes"
        )
    elif cases and bool(feature_coverage["existing_features_insufficient"]):
        conclusion = (
            "existing_feature_set_insufficient_for_hts_side_confidence_regimes"
        )
        next_action = (
            "add_new_decision_time_reference_and_book_pressure_features_before_"
            "further_hts_priority_changes"
        )
    elif cases:
        conclusion = "hts_priority_too_high_when_reliability_is_weak"
        next_action = "lower_hts_side_bet_priority_when_reliability_is_weak"
    else:
        conclusion = "no_hts_p_up_confidently_wrong_regime_detected"
        next_action = "continue_monitoring"
    return {
        "diagnostic_conclusion": conclusion,
        "recommended_next_action": next_action,
        "reason_codes": reason_codes,
    }


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
        "decision_time_feature_row_available": row.get(
            "decision_time_feature_row_available"
        ),
        "decision_time_feature_missing_reason_codes": row.get(
            "decision_time_feature_missing_reason_codes",
            [],
        ),
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
        "decision_time_feature_row_available": row.get(
            "decision_time_feature_row_available"
        ),
        "reference_price_to_beat_distance_at_decision": row.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "recent_reference_price_momentum_30s": row.get(
            "recent_reference_price_momentum_30s"
        ),
        "recent_reference_price_momentum_60s": row.get(
            "recent_reference_price_momentum_60s"
        ),
        "recent_reference_price_momentum_120s": row.get(
            "recent_reference_price_momentum_120s"
        ),
        "side_book_depth_imbalance": row.get("side_book_depth_imbalance"),
        "side_book_update_velocity": row.get("side_book_update_velocity"),
        "side_book_staleness_ms": row.get("side_book_staleness_ms"),
        "side_spread_bps": row.get("side_spread_bps"),
        "side_queue_fill_proxy": row.get("side_queue_fill_proxy"),
        "hts_vs_sell_before_close_exit_value_gap_proxy": row.get(
            "hts_vs_sell_before_close_exit_value_gap_proxy"
        ),
        "p_up_calibration_residual_by_time_spread_queue_bucket": row.get(
            "p_up_calibration_residual_by_time_spread_queue_bucket"
        ),
        "decision_time_feature_missing_reason_codes": row.get(
            "decision_time_feature_missing_reason_codes",
            [],
        ),
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
            f"- selected_feature_set_name: `{report['selected_feature_set_name']}`",
            "- selected_correction_policy_name: "
            f"`{report['selected_correction_policy_name']}`",
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


def _feature_set_selection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# O Feature Set Selection",
        "",
        f"- selected_feature_set_name: `{report['selected_feature_set_name']}`",
        "- uses_validation_labels_for_tuning: "
        f"`{str(report['uses_validation_labels_for_tuning']).lower()}`",
        "- selection_metric_source: "
        f"`{report['selection_metric_source']}`",
        "- selected_shadow_mean_regret: "
        f"`{report['selected_shadow_metrics']['mean_regret']}`",
        "- selected_validation_mean_regret_report_only: "
        f"`{report['selected_validation_metrics_report_only']['mean_regret']}`",
        "- selected_validation_top1_report_only: "
        f"`{report['selected_validation_metrics_report_only']['top1_hit_rate']}`",
        "- selected_validation_p_up_disagreement_report_only: "
        f"`{report['selected_validation_metrics_report_only']['p_up_disagreement_rate']}`",
        "- mean_regret_tradeoff_conclusion: "
        f"`{report['mean_regret_gate_tradeoff_diagnostics']['tradeoff_conclusion']}`",
        "- reference_distance_in_selected_feature_set: "
        f"`{str(report['reference_distance_in_selected_feature_set']).lower()}`",
        "- #146_start_allowed: "
        f"`{str(report['#146_start_allowed']).lower()}`",
        "- #134_resume_allowed: "
        f"`{str(report['#134_resume_allowed']).lower()}`",
        "",
        "| feature_set | selected | shadow_top1 | shadow_mean_regret | shadow_return | shadow_high_score_sum | shadow_p_up_disagreement | validation_top1_report_only | validation_mean_regret_report_only | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidate_feature_sets"]:
        shadow = row["shadow_selection_metrics"]
        validation = row["validation_metrics_report_only"]
        lines.append(
            "| {name} | {selected} | {shadow_top1:.6f} | {shadow_regret:.6f} | {shadow_return:.6f} | {shadow_hs:.6f} | {shadow_pup:.6f} | {validation_top1:.6f} | {validation_regret:.6f} | {reasons} |".format(
                name=row["feature_set_name"],
                selected=str(
                    row["feature_set_name"] == report["selected_feature_set_name"]
                ).lower(),
                shadow_top1=float(shadow["top1_hit_rate"]),
                shadow_regret=float(shadow["mean_regret"]),
                shadow_return=float(shadow["selected_return_sum"]),
                shadow_hs=float(shadow["high_score_return_sum"]),
                shadow_pup=float(shadow["p_up_disagreement_rate"]),
                validation_top1=float(validation["top1_hit_rate"]),
                validation_regret=float(validation["mean_regret"]),
                reasons=", ".join(row["shadow_selection_reason_codes"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _joint_feature_correction_selection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# O Joint Feature Correction Selection",
        "",
        f"- selected_joint_candidate_name: `{report['selected_joint_candidate_name']}`",
        f"- selected_feature_set_name: `{report['selected_feature_set_name']}`",
        "- selected_correction_policy_name: "
        f"`{report['selected_correction_policy_name']}`",
        "- uses_validation_labels_for_tuning: "
        f"`{str(report['uses_validation_labels_for_tuning']).lower()}`",
        f"- selection_metric_source: `{report['selection_metric_source']}`",
        "- selected_shadow_mean_regret: "
        f"`{report['selected_shadow_metrics']['mean_regret']}`",
        "- selected_validation_mean_regret_report_only: "
        f"`{report['selected_validation_metrics_report_only']['mean_regret']}`",
        "- selected_validation_top1_report_only: "
        f"`{report['selected_validation_metrics_report_only']['top1_hit_rate']}`",
        "- selected_validation_p_up_disagreement_report_only: "
        f"`{report['selected_validation_metrics_report_only']['p_up_disagreement_rate']}`",
        "- shadow_p_up_safety_target_rate: "
        f"`{report['shadow_p_up_safety_target_rate']}`",
        "- mean_regret_tradeoff_conclusion: "
        f"`{report['mean_regret_gate_tradeoff_diagnostics']['tradeoff_conclusion']}`",
        "- lower_shadow_mean_regret_candidate_count: "
        f"`{report['mean_regret_gate_tradeoff_diagnostics']['lower_shadow_mean_regret_candidate_count']}`",
        "- lower_shadow_gate_passing_candidate_count: "
        f"`{report['mean_regret_gate_tradeoff_diagnostics']['lower_shadow_gate_passing_candidate_count']}`",
        "- #146_start_allowed: "
        f"`{str(report['#146_start_allowed']).lower()}`",
        "- #134_resume_allowed: "
        f"`{str(report['#134_resume_allowed']).lower()}`",
        "",
        "| joint_candidate | selected | shadow_top1 | shadow_mean_regret | shadow_return | shadow_high_score_sum | shadow_p_up_disagreement | validation_top1_report_only | validation_mean_regret_report_only | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidate_rows"][:50]:
        shadow = row["shadow_selection_metrics"]
        validation = row["validation_metrics_report_only"]
        lines.append(
            "| {name} | {selected} | {shadow_top1:.6f} | {shadow_regret:.6f} | {shadow_return:.6f} | {shadow_hs:.6f} | {shadow_pup:.6f} | {validation_top1:.6f} | {validation_regret:.6f} | {reasons} |".format(
                name=row["joint_candidate_name"],
                selected=str(
                    row["joint_candidate_name"]
                    == report["selected_joint_candidate_name"]
                ).lower(),
                shadow_top1=float(shadow["top1_hit_rate"]),
                shadow_regret=float(shadow["mean_regret"]),
                shadow_return=float(shadow["selected_return_sum"]),
                shadow_hs=float(shadow["high_score_return_sum"]),
                shadow_pup=float(shadow["p_up_disagreement_rate"]),
                validation_top1=float(validation["top1_hit_rate"]),
                validation_regret=float(validation["mean_regret"]),
                reasons=", ".join(row["shadow_selection_reason_codes"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


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


def _v8_action_rank_handoff_markdown(report: dict[str, Any]) -> str:
    summary = report["selected_action_handoff_summary"]
    return "\n".join(
        [
            "# O v8 Action-Rank Handoff",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            "- strict_calibration_quality_passed: "
            f"`{str(report['strict_calibration_quality_passed']).lower()}`",
            "- relaxed_diagnostic_source_candidate: "
            f"`{str(report['relaxed_diagnostic_source_candidate']).lower()}`",
            "- v8_action_rank_quality_passed: "
            f"`{str(report['v8_action_rank_quality_passed']).lower()}`",
            "- v8_action_rank_candidate_eligible: "
            f"`{str(report['v8_action_rank_candidate_eligible']).lower()}`",
            "- v8_execution_risk_control_required: "
            f"`{str(report['v8_execution_risk_control_required']).lower()}`",
            "- v8_execution_risk_guard_report_available: "
            f"`{str(report['v8_execution_risk_guard_report_available']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            "- model_layer_regret_risk_selection_deferred_to_issue: "
            f"`{report['model_layer_regret_risk_selection_deferred_to_issue']}`",
            "- strict_source_gate_remains_failed: "
            f"`{str(report['strict_source_gate_remains_failed']).lower()}`",
            "- selected_action_handoff_row_count: "
            f"`{report['selected_action_handoff_row_count']}`",
            "- high_score_selected_action_count: "
            f"`{summary['high_score_selected_action_count']}`",
            "- p_up_action_disagreement_rate: "
            f"`{summary['p_up_action_disagreement_rate']}`",
            f"- v8_action_rank_reason_codes: `{report['v8_action_rank_reason_codes']}`",
            "- v8_execution_handoff_blocking_reason_codes: "
            f"`{report['v8_execution_handoff_blocking_reason_codes']}`",
            "- no_paper_live_unlock_from_v8_action_rank_gate: "
            f"`{str(report['no_paper_live_unlock_from_v8_action_rank_gate']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_risk_guard_markdown(report: dict[str, Any]) -> str:
    summary = report["execution_guard_summary"]
    return "\n".join(
        [
            "# O v8 Execution Risk Guard",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- v8_scope: `{report['v8_scope']}`",
            "- source_action_rank_signal_available: "
            f"`{str(report['source_action_rank_signal_available']).lower()}`",
            "- model_layer_regret_risk_selection_enabled: "
            f"`{str(report['model_layer_regret_risk_selection_enabled']).lower()}`",
            "- model_layer_regret_risk_selection_deferred_to_issue: "
            f"`{report['model_layer_regret_risk_selection_deferred_to_issue']}`",
            "- mutates_o_model_predicted_score: "
            f"`{str(report['mutates_o_model_predicted_score']).lower()}`",
            "- runtime_risk_control_validation_passed: "
            f"`{str(report['runtime_risk_control_validation_passed']).lower()}`",
            "- v8_action_rank_candidate_eligible: "
            f"`{str(report['v8_action_rank_candidate_eligible']).lower()}`",
            "- v8_execution_risk_control_required: "
            f"`{str(report['v8_execution_risk_control_required']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            "- execution_guard_decision_count: "
            f"`{report['execution_guard_decision_count']}`",
            f"- order_allowed_count: `{report['order_allowed_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            f"- fail_closed_decision_count: `{report['fail_closed_decision_count']}`",
            "- proposed_order_size_total: "
            f"`{report['proposed_order_size_total']}`",
            "- execution_blocking_reason_counts: "
            f"`{summary['execution_blocking_reason_counts']}`",
            "- execution_guard_reason_counts: "
            f"`{summary['execution_guard_reason_counts']}`",
            "- v8_execution_handoff_blocking_reason_codes: "
            f"`{report['v8_execution_handoff_blocking_reason_codes']}`",
            "- no_paper_live_unlock_from_execution_guard: "
            f"`{str(report['no_paper_live_unlock_from_execution_guard']).lower()}`",
            "- no_source_freeze_unlock_from_execution_guard: "
            f"`{str(report['no_source_freeze_unlock_from_execution_guard']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_runtime_state_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Execution Runtime State",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            f"- risk_state_source: `{report['risk_state_source']}`",
            "- runtime_state_validation_passed: "
            f"`{str(report['runtime_state_validation_passed']).lower()}`",
            "- runtime_risk_control_validation_passed: "
            f"`{str(report['runtime_risk_control_validation_passed']).lower()}`",
            f"- current_total_exposure: `{report['current_total_exposure']}`",
            f"- exposure_by_side: `{report['exposure_by_side']}`",
            f"- executed_simulated_order_count: `{len(report['executed_simulated_orders'])}`",
            f"- blocked_simulated_order_count: `{len(report['blocked_simulated_orders'])}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_simulated_order_replay_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Execution Simulated Order Replay",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            f"- decision_count: `{report['decision_count']}`",
            "- simulated_allowed_order_count: "
            f"`{report['simulated_allowed_order_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            f"- total_proposed_notional: `{report['total_proposed_notional']}`",
            "- runtime_risk_control_validation_passed: "
            f"`{str(report['runtime_risk_control_validation_passed']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- block_reason_distribution: `{report['block_reason_distribution']}`",
            "- exposure_reason_distribution: "
            f"`{report['exposure_reason_distribution']}`",
            f"- deterministic_replay_hash: `{report['deterministic_replay_hash']}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_allowed_order_quality_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Execution Allowed Order Quality",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- mutates_o_model_predicted_score: "
            f"`{str(report['mutates_o_model_predicted_score']).lower()}`",
            "- uses_realized_pnl_or_labels_for_analysis: "
            f"`{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            f"- decision_count: `{report['decision_count']}`",
            f"- allowed_order_count: `{report['allowed_order_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            "- allowed_order_action_distribution: "
            f"`{report['allowed_order_action_distribution']}`",
            "- allowed_order_family_distribution: "
            f"`{report['allowed_order_family_distribution']}`",
            "- allowed_order_side_distribution: "
            f"`{report['allowed_order_side_distribution']}`",
            "- allowed_order_origin_distribution: "
            f"`{report['allowed_order_origin_distribution']}`",
            "- allowed_order_p_up_agreement_distribution: "
            f"`{report['allowed_order_p_up_agreement_distribution']}`",
            "- residual_blocker_summary: "
            f"`{report['residual_blocker_summary']}`",
            "- deterministic_recommendation_counts: "
            f"`{report['deterministic_recommendation_counts']}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_policy_readiness_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Execution Policy Readiness",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- uses_realized_pnl_or_labels_for_analysis: "
            f"`{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            "- execution_policy_readiness_diagnostic_passed: "
            f"`{str(report['execution_policy_readiness_diagnostic_passed']).lower()}`",
            "- execution_policy_readiness_blocking_reason_codes: "
            f"`{report['execution_policy_readiness_blocking_reason_codes']}`",
            "- future_explicit_execution_handoff_gate_required: "
            f"`{str(report['future_explicit_execution_handoff_gate_required']).lower()}`",
            f"- allowed_order_count: `{report['allowed_order_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            "- residual_future_work_summary: "
            f"`{report['residual_future_work_summary']}`",
            "- microstructure_quality_passed: "
            f"`{str(report['microstructure_quality_summary']['passed']).lower()}`",
            "- exposure_quality_passed: "
            f"`{str(report['exposure_quality_summary']['passed']).lower()}`",
            "- runtime_integrity_summary: "
            f"`{report['runtime_integrity_summary']}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_handoff_gate_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Execution Handoff Gate",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- explicit_execution_handoff_gate_mode: "
            f"`{report['explicit_execution_handoff_gate_mode']}`",
            "- explicit_execution_handoff_gate_passed: "
            f"`{str(report['explicit_execution_handoff_gate_passed']).lower()}`",
            "- explicit_execution_handoff_blocking_reason_codes: "
            f"`{report['explicit_execution_handoff_blocking_reason_codes']}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- uses_realized_pnl_or_labels_for_analysis: "
            f"`{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            "- uses_oracle_actions_for_analysis: "
            f"`{str(report['uses_oracle_actions_for_analysis']).lower()}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            "- future_paper_candidate_gate_required: "
            f"`{str(report['future_paper_candidate_gate_required']).lower()}`",
            f"- allowed_order_count: `{report['allowed_order_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            "- runtime_integrity_summary: "
            f"`{report['runtime_integrity_summary']}`",
            "- residual_blocker_summary: "
            f"`{report['residual_blocker_summary']}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            f"- freeze_ready: `{str(report['freeze_ready']).lower()}`",
            "- promotion_evidence_eligible: "
            f"`{str(report['promotion_evidence_eligible']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_future_unseen_holdout_plan_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Future Unseen Holdout Plan",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- future_unseen_holdout_plan_ready: "
            f"`{str(report['future_unseen_holdout_plan_ready']).lower()}`",
            "- future_unseen_holdout_blocking_reason_codes: "
            f"`{report['future_unseen_holdout_blocking_reason_codes']}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            "- future_holdout_current_status: "
            f"`{report['future_holdout_current_status']}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- uses_realized_pnl_or_labels_for_analysis: "
            f"`{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            "- uses_oracle_actions_for_analysis: "
            f"`{str(report['uses_oracle_actions_for_analysis']).lower()}`",
            "- paper_candidate_allowed: "
            f"`{str(report['paper_candidate_allowed']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            f"- freeze_ready: `{str(report['freeze_ready']).lower()}`",
            "- promotion_evidence_eligible: "
            f"`{str(report['promotion_evidence_eligible']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_paper_candidate_gate_design_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Candidate Gate Design",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- paper_candidate_gate_design_ready: "
            f"`{str(report['paper_candidate_gate_design_ready']).lower()}`",
            "- paper_candidate_gate_blocking_reason_codes: "
            f"`{report['paper_candidate_gate_blocking_reason_codes']}`",
            "- paper_candidate_allowed: "
            f"`{str(report['paper_candidate_allowed']).lower()}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            "- future_paper_candidate_gate_required: "
            f"`{str(report['future_paper_candidate_gate_required']).lower()}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- uses_realized_pnl_or_labels_for_analysis: "
            f"`{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            "- uses_oracle_actions_for_analysis: "
            f"`{str(report['uses_oracle_actions_for_analysis']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            f"- freeze_ready: `{str(report['freeze_ready']).lower()}`",
            "- promotion_evidence_eligible: "
            f"`{str(report['promotion_evidence_eligible']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_guard_block_analysis_markdown(report: dict[str, Any]) -> str:
    discovery = report["safe_order_discovery_summary"]
    return "\n".join(
        [
            "# O v8 Execution Guard Block Analysis",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            "- mutates_o_model_predicted_score: "
            f"`{str(report['mutates_o_model_predicted_score']).lower()}`",
            f"- decision_count: `{report['decision_count']}`",
            f"- blocked_decision_count: `{report['blocked_decision_count']}`",
            f"- allowed_decision_count: `{report['allowed_decision_count']}`",
            "- primary_blocker_categories: "
            f"`{report['primary_blocker_categories']}`",
            "- safe_order_candidate_count: "
            f"`{discovery['safe_order_candidate_count']}`",
            "- blocked_only_by_missing_runtime_fields_count: "
            f"`{discovery['blocked_only_by_missing_runtime_fields_count']}`",
            "- blocked_only_by_configurable_thresholds_count: "
            f"`{discovery['blocked_only_by_configurable_thresholds_count']}`",
            "- would_pass_if_sbc_fallback_selected_count: "
            f"`{discovery['would_pass_if_sbc_fallback_selected_count']}`",
            "- fundamentally_unsafe_count: "
            f"`{discovery['fundamentally_unsafe_count']}`",
            "- why_simulated_allowed_order_count_zero: "
            f"`{report['why_simulated_allowed_order_count_zero']}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _v8_execution_runtime_field_coverage_markdown(report: dict[str, Any]) -> str:
    classifications = report["classification_counts"]
    return "\n".join(
        [
            "# O v8 Execution Runtime Field Coverage",
            "",
            f"- diagnostic_only: `{str(report['diagnostic_only']).lower()}`",
            f"- simulation_only: `{str(report['simulation_only']).lower()}`",
            "- uses_validation_outcomes_for_tuning: "
            f"`{str(report['uses_validation_outcomes_for_tuning']).lower()}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            f"- backfill_rules_applied: `{str(report['backfill_rules_applied']).lower()}`",
            "- applied_runtime_field_backfill_count: "
            f"`{report['applied_runtime_field_backfill_count']}`",
            "- applied_runtime_field_backfill_rule_counts: "
            f"`{report['applied_runtime_field_backfill_rule_counts']}`",
            "- runtime_field_backfill_provenance_validity_summary: "
            f"`{report['runtime_field_backfill_provenance_validity_summary']}`",
            "- mutates_o_model_predicted_score: "
            f"`{str(report['mutates_o_model_predicted_score']).lower()}`",
            f"- decision_count: `{report['decision_count']}`",
            "- missing_runtime_field_decision_count: "
            f"`{report['missing_runtime_field_decision_count']}`",
            "- missing_runtime_field_occurrence_count: "
            f"`{report['missing_runtime_field_occurrence_count']}`",
            "- primary_missing_runtime_fields: "
            f"`{report['primary_missing_runtime_fields']}`",
            "- true_data_coverage_gap_count: "
            f"`{classifications['true_data_coverage_gap']}`",
            "- derived_backfill_from_existing_handoff_fields_count: "
            f"`{classifications['derived_backfill_from_existing_handoff_fields']}`",
            "- safe_backfill_candidate_count: "
            f"`{report['safe_backfill_candidate_count']}`",
            "- existing_handoff_backfill_candidate_count: "
            f"`{report['existing_handoff_backfill_candidate_count']}`",
            "- decision_time_data_join_backfill_candidate_count: "
            f"`{report['decision_time_data_join_backfill_candidate_count']}`",
            "- required_field_policy_relaxation_candidate_count: "
            f"`{report['required_field_policy_relaxation_candidate_count']}`",
            "- optional_for_no_trade_count: "
            f"`{classifications['optional_for_no_trade']}`",
            "- too_strict_for_simulation_only_mode_count: "
            f"`{classifications['too_strict_for_simulation_only_mode']}`",
            "- runtime_field_policy_findings: "
            f"`{report['runtime_field_policy_findings']}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
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
        "- relaxed_diagnostic_source_candidate: "
        f"`{str(report['relaxed_diagnostic_source_candidate']).lower()}`",
        "- relaxed_diagnostic_no_paper_live_unlock: "
        f"`{str(report['relaxed_diagnostic_no_paper_live_unlock']).lower()}`",
        "- v8_action_rank_candidate_eligible: "
        f"`{str(report['v8_action_rank_candidate_eligible']).lower()}`",
        "- v8_execution_handoff_allowed: "
        f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
        "- model_layer_regret_risk_selection_deferred_to_issue: "
        f"`{report['model_layer_regret_risk_selection_deferred_to_issue']}`",
        f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
        f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
        "",
        "| candidate | score_source | scope | top1 | mean_regret | strict_calibration | relaxed_diagnostic | v8_action_rank | handoff_allowed | eligible | excluded_reason |",
        "|---|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            "| {name} | {source} | {scope} | {top1:.4f} | {regret:.6f} | {strict} | {relaxed} | {v8} | {handoff} | {eligible} | {reason} |".format(
                name=row["candidate_name"],
                source=row["ranking_score_source"],
                scope=row["ranking_metric_scope"],
                top1=float(row["top1_realized_best_action_hit_rate"]),
                regret=float(row["mean_regret"]),
                strict=str(row["strict_calibration_quality_passed"]).lower(),
                relaxed=str(row["relaxed_diagnostic_source_candidate"]).lower(),
                v8=str(row["v8_action_rank_candidate_eligible"]).lower(),
                handoff=str(row["v8_execution_handoff_allowed"]).lower(),
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
            "- strict_calibration_quality_passed: "
            f"`{str(report['strict_calibration_quality_passed']).lower()}`",
            "- relaxed_diagnostic_calibration_quality_passed: "
            f"`{str(report['relaxed_diagnostic_calibration_quality_passed']).lower()}`",
            "- relaxed_diagnostic_source_candidate: "
            f"`{str(report['relaxed_diagnostic_source_candidate']).lower()}`",
            "- relaxed_diagnostic_no_paper_live_unlock: "
            f"`{str(report['relaxed_diagnostic_no_paper_live_unlock']).lower()}`",
            "- v8_action_rank_quality_passed: "
            f"`{str(report['v8_action_rank_quality_passed']).lower()}`",
            "- v8_action_rank_candidate_eligible: "
            f"`{str(report['v8_action_rank_candidate_eligible']).lower()}`",
            "- v8_execution_risk_control_required: "
            f"`{str(report['v8_execution_risk_control_required']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            "- model_layer_regret_risk_selection_deferred_to_issue: "
            f"`{report['model_layer_regret_risk_selection_deferred_to_issue']}`",
            "- strict_source_gate_remains_failed: "
            f"`{str(report['strict_source_gate_remains_failed']).lower()}`",
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
            "- top1_miss_count: "
            f"`{report['top1_miss_diagnostics']['top1_miss_count']}`",
            "- top1_miss_regret_sum: "
            f"`{report['top1_miss_diagnostics']['top1_miss_regret_sum']}`",
            "- gate_reason_code_consistency_passed: "
            f"`{str(report['gate_reason_code_consistency_passed']).lower()}`",
            "- NO_TRADE_selection_rate: "
            f"`{report['NO_TRADE_selection_rate']}`",
            "- strict_vs_relaxed_gate_summary: "
            f"`{report['strict_vs_relaxed_gate_summary']}`",
            f"- ineligible_reason_codes: `{report['ineligible_reason_codes']}`",
            "- relaxed_diagnostic_reason_codes: "
            f"`{report['relaxed_diagnostic_reason_codes']}`",
            f"- v8_action_rank_reason_codes: `{report['v8_action_rank_reason_codes']}`",
            "- v8_execution_handoff_blocking_reason_codes: "
            f"`{report['v8_execution_handoff_blocking_reason_codes']}`",
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
            "- relaxed_diagnostic_source_candidate: "
            f"`{str(report['relaxed_diagnostic_source_candidate']).lower()}`",
            "- relaxed_diagnostic_no_freeze_unlock: "
            f"`{str(report['relaxed_diagnostic_no_freeze_unlock']).lower()}`",
            "- v8_action_rank_candidate_eligible: "
            f"`{str(report['v8_action_rank_candidate_eligible']).lower()}`",
            "- v8_execution_handoff_allowed: "
            f"`{str(report['v8_execution_handoff_allowed']).lower()}`",
            "- model_layer_regret_risk_selection_deferred_to_issue: "
            f"`{report['model_layer_regret_risk_selection_deferred_to_issue']}`",
            "- strict_source_gate_remains_failed: "
            f"`{str(report['strict_source_gate_remains_failed']).lower()}`",
            "- strict_calibration_quality_passed: "
            f"`{str(report['strict_calibration_quality_passed']).lower()}`",
            "- relaxed_diagnostic_calibration_quality_passed: "
            f"`{str(report['relaxed_diagnostic_calibration_quality_passed']).lower()}`",
            f"- model_sha256: `{report['model_sha256']}`",
            f"- model_manifest_sha256: `{report['model_manifest_sha256']}`",
            f"- training_data_hash: `{report['training_data_hash']}`",
            f"- label_grid_hash: `{report['label_grid_hash']}`",
            f"- feature_schema_hash: `{report['feature_schema_hash']}`",
            f"- split_hash: `{report['split_hash']}`",
            f"- candidate_config_hash: `{report['candidate_config_hash']}`",
            "- freeze_blocking_reason_codes: "
            f"`{report['freeze_blocking_reason_codes']}`",
            f"- v8_action_rank_reason_codes: `{report['v8_action_rank_reason_codes']}`",
            "- v8_execution_handoff_blocking_reason_codes: "
            f"`{report['v8_execution_handoff_blocking_reason_codes']}`",
            "- future_unseen_holdout_required: "
            f"`{str(report['future_unseen_holdout_required']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )


def _hts_p_up_confidently_wrong_feature_diagnostic_markdown(
    report: dict[str, Any],
) -> str:
    validation = report["split_summaries"]["validation"]
    alternatives = report["alternative_comparison_summary"]
    return "\n".join(
        [
            "# O HTS p_up Confidently-Wrong Feature Diagnostic",
            "",
            f"- candidate_name: `{report['candidate_name']}`",
            f"- case_count: `{report['case_count']}`",
            f"- validation_case_count: `{report['validation_case_count']}`",
            f"- validation_regret_sum: `{validation['regret_sum']}`",
            f"- validation_regret_mean: `{validation['regret_mean']}`",
            f"- validation_regret_max: `{validation['regret_max']}`",
            "- existing_features_insufficient: "
            f"`{str(report['feature_coverage_summary']['existing_features_insufficient']).lower()}`",
            "- recommended_next_action: "
            f"`{report['recommended_next_action']}`",
            "- recommendation_reason_codes: "
            f"`{report['recommendation_reason_codes']}`",
            "- opposite_hts_better_return_count: "
            f"`{alternatives['opposite_hts_side']['better_return_count']}`",
            "- best_sbc_better_return_count: "
            f"`{alternatives['best_sell_before_close_by_return']['better_return_count']}`",
            "- no_trade_better_return_count: "
            f"`{alternatives['NO_TRADE']['better_return_count']}`",
            "- uses_validation_labels_for_tuning: "
            f"`{str(report['uses_validation_labels_for_tuning']).lower()}`",
            "- source_model_candidate_eligible: "
            f"`{str(report['source_model_candidate_eligible']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
        ]
    )
