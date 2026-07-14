"""Fresh public-data paper-only loop for the v8 O candidate."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb

from bigan.execution.position_manager import PositionManager
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.corpus.contracts import BTC_UPDOWN_MARKET_HORIZONS_MS
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig
from bigan.v8.polymarket.recorder.public_provider import (
    BTC_UPDOWN_FAMILY_BY_SLUG,
    BTC_UPDOWN_SLUG_PATTERN,
    PolymarketPublicHTTPRealCorpusProvider,
    RealCorpusPublicProviderError,
)
from bigan.v8.polymarket.training.contracts import (
    POLYMARKET_POLICY_TRAINING_PHASE,
    compact_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    _calibrated_expected_return_source,
    _load_frozen_ev_calibration_artifact,
)
from bigan.v8.polymarket.training.o_v8_paper_candidate_unlock import (
    _sha256_file as _sha256_file_existing,
)
from bigan.v8.polymarket.training.post_freeze_o_replay_aligned_source_ranking import (
    O_DEPLOYABLE_MODEL_FEATURE_NAMES,
    O_REQUIRED_DECISION_ACTION_FAMILIES,
    _action_family,
    _apply_o_shadow_ranking_correction,
    _deployable_model_feature_map,
    _side_from_action,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
)

O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-run-v1"
)
O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-fill-simulation-v1"
)
O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-runtime-safety-v1"
)
O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-monitoring-v1"
)
O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-cumulative-monitoring-v1"
)
O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-loop-manifest-v1"
)
O_V8_PAPER_FRESH_NO_TRADE_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-no-trade-diagnostic-v1"
)
O_V8_PAPER_FRESH_SCORE_DECOMPOSITION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-score-decomposition-v1"
)
O_V8_PAPER_FRESH_PROVIDER_FEATURE_COVERAGE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-provider-feature-coverage-v1"
)
O_V8_PAPER_FRESH_CANONICAL_SCORER_ALIGNMENT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-canonical-scorer-alignment-v1"
)
O_V8_PAPER_FRESH_CANONICAL_FEATURE_MAPPING_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-canonical-feature-mapping-v1"
)
O_V8_PAPER_FRESH_CANONICAL_SCORER_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-canonical-scorer-v1"
)
O_V8_PAPER_FRESH_SCORER_COMPARISON_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-scorer-comparison-v1"
)
O_V8_PAPER_FRESH_SIGNAL_TRACE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-signal-trace-v1"
)
O_V8_PAPER_FRESH_TIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-time-window-diagnostic-v1"
)
O_V8_PAPER_FRESH_LEGACY_POSITION_POLICY_AUDIT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-legacy-position-policy-audit-v1"
)
O_V8_PAPER_FRESH_POSITION_STATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-position-state-v1"
)
O_V8_PAPER_FRESH_EXIT_SIGNAL_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-exit-signal-v1"
)
O_V8_PAPER_FRESH_EXIT_LEDGER_UPDATE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-o-v8-paper-fresh-exit-ledger-update-v1"
)
EXECUTION_LAYER_V2_PAPER_REMAP_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-paper-remap-v1"
)

PINNED_ISSUE_160_RUN_ID = "o-v8-paper-candidate-unlock-20260703T073000Z"
PINNED_ISSUE_160_MANIFEST_SHA256 = (
    "a7bbe5c6128e4471ee48ea0765d4305acb0d0c5722226b7556c9fd4a8f648815"
)
O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS: tuple[str, ...] = (
    "realized_pnl",
    "realized_trade_pnl",
    "settlement_pnl",
    "settlement_label",
    "oracle_action",
    "oracle_side",
    "future_return",
    "future_price",
    "future_outcome",
    "total_polymarket_pnl",
)
O_V8_PAPER_FRESH_EXIT_FORBIDDEN_OUTCOME_FIELDS: tuple[str, ...] = (
    *O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS,
    "oracle_side",
    "oracle_label",
    "winning_outcome",
    "settlement_result",
    "settlement_status",
    "resolved_outcome",
    "future_return_net",
    "future_last_price",
    "label_return",
    "action_return_target",
)
O_V8_PAPER_FRESH_EXIT_EDGE_THRESHOLD = -0.02
O_V8_PAPER_FRESH_EXIT_PROFIT_TARGET = 0.05
O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE = 60.0
O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME = "paper_only_adapter_heuristic_v1"
O_V8_PAPER_FRESH_EXIT_DECISION_POLICY_SOURCE = "paper_only_adapter_heuristic_v1"
O_V8_PAPER_FRESH_HTS_REMAP_EV_THRESHOLD = 0.02
O_V8_PAPER_FRESH_HTS_REMAP_SCORE_TO_EXPECTED_NET_RETURN_WEIGHT = 0.02
O_V8_PAPER_FRESH_HTS_REMAP_EXECUTION_COST = 0.001
O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER = "read_only_public_provider"
O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE = "snapshot_fixture"
O_V8_PAPER_FRESH_PUBLIC_DATA_SOURCES = (
    O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
    O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
)
O_V8_PAPER_FRESH_COMPARISON_RUN_IDS = (
    "o-v8-paper-fresh-loop-20260703T095844Z",
    "o-v8-paper-fresh-loop-20260703T094653Z",
    "o-v8-paper-fresh-loop-20260703T081500Z",
)

_FALSE_SAFETY_FIELDS = (
    "capital_at_risk",
    "polymarket_write_enabled",
    "wallet_signing_enabled",
    "v8_execution_handoff_allowed",
    "source_model_candidate_eligible",
    "freeze_ready",
    "promotion_evidence_eligible",
    "#134_resume_allowed",
    "#146_start_allowed",
)

_CHAINLINK_DECISION_TIME_FIELDS = (
    "chainlink_price_at_decision",
    "chainlink_reference_price_at_market_start",
    "chainlink_reference_distance_at_decision",
    "chainlink_momentum_30s",
    "chainlink_momentum_60s",
    "chainlink_momentum_120s",
    "chainlink_realized_volatility_120s",
    "chainlink_vs_btc_feature_price_gap",
    "chainlink_regime_feature_provenance",
)


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopConfig:
    """Configuration for one fresh public-data paper-only loop run."""

    run_id: str
    output_dir: Path | str
    paper_candidate_unlock_dir: Path | str
    loop_mode: Literal["single_cycle", "bounded_recurring"] = "single_cycle"
    max_cycles: int = 1
    sleep_seconds: float = 0.0
    public_data_cycles: tuple[tuple[dict[str, Any], ...], ...] | None = None
    public_data_source: str = O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    public_provider: Any | None = None
    chainlink_rtds_price_rows: tuple[dict[str, Any], ...] = ()
    chainlink_rtds_persist_price_rows: tuple[dict[str, Any], ...] | None = None
    initial_paper_position_rows: tuple[dict[str, Any], ...] = ()
    canonical_o_source_manifest_path: Path | str | None = None
    frozen_ev_calibration_artifact_path: Path | str | None = None
    expected_paper_candidate_unlock_manifest_sha256: str | None = (
        PINNED_ISSUE_160_MANIFEST_SHA256
    )
    overwrite_existing: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.loop_mode not in {"single_cycle", "bounded_recurring"}:
            raise ValueError("loop_mode must be single_cycle or bounded_recurring")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.loop_mode == "single_cycle" and self.max_cycles != 1:
            raise ValueError("single_cycle mode requires max_cycles=1")
        if self.sleep_seconds < 0.0:
            raise ValueError("sleep_seconds must be non-negative")
        if self.public_data_source not in O_V8_PAPER_FRESH_PUBLIC_DATA_SOURCES:
            raise ValueError(
                "public_data_source must be read_only_public_provider or snapshot_fixture"
            )
        if (
            self.public_data_source == O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
            and self.public_data_cycles is None
        ):
            raise ValueError("snapshot_fixture mode requires public_data_cycles")
        if self.paper_only is not True:
            raise ValueError("paper_only must be true")
        if self.capital_at_risk is not False:
            raise ValueError("capital_at_risk must be false")
        if self.polymarket_write_enabled is not False:
            raise ValueError("polymarket_write_enabled must be false")
        if self.wallet_signing_enabled is not False:
            raise ValueError("wallet_signing_enabled must be false")
        if not isinstance(self.initial_paper_position_rows, tuple):
            object.__setattr__(
                self,
                "initial_paper_position_rows",
                tuple(dict(row) for row in self.initial_paper_position_rows),
            )
        if not isinstance(self.chainlink_rtds_price_rows, tuple):
            object.__setattr__(
                self,
                "chainlink_rtds_price_rows",
                tuple(dict(row) for row in self.chainlink_rtds_price_rows),
            )
        if self.chainlink_rtds_persist_price_rows is not None and not isinstance(
            self.chainlink_rtds_persist_price_rows, tuple
        ):
            object.__setattr__(
                self,
                "chainlink_rtds_persist_price_rows",
                tuple(dict(row) for row in self.chainlink_rtds_persist_price_rows),
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )
        if self.canonical_o_source_manifest_path is not None and not isinstance(
            self.canonical_o_source_manifest_path,
            Path,
        ):
            object.__setattr__(
                self,
                "canonical_o_source_manifest_path",
                Path(self.canonical_o_source_manifest_path),
            )
        if self.frozen_ev_calibration_artifact_path is not None and not isinstance(
            self.frozen_ev_calibration_artifact_path,
            Path,
        ):
            object.__setattr__(
                self,
                "frozen_ev_calibration_artifact_path",
                Path(self.frozen_ev_calibration_artifact_path),
            )


@dataclass(frozen=True, slots=True)
class PolymarketOV8PaperFreshLoopResult:
    """Generated fresh paper loop bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    fresh_loop_run_report: dict[str, Any]
    fill_simulation_report: dict[str, Any]
    runtime_safety_report: dict[str, Any]
    monitoring_report: dict[str, Any]
    cumulative_monitoring_report: dict[str, Any]
    no_trade_diagnostic_report: dict[str, Any]
    score_decomposition_report: dict[str, Any]
    provider_feature_coverage_report: dict[str, Any]
    canonical_feature_mapping_report: dict[str, Any]
    canonical_action_rows: list[dict[str, Any]]
    canonical_scorer_report: dict[str, Any]
    scorer_comparison_report: dict[str, Any]
    canonical_scorer_alignment_report: dict[str, Any]
    signal_trace_report: dict[str, Any]
    time_window_diagnostic_report: dict[str, Any]
    legacy_position_policy_audit_report: dict[str, Any]
    paper_position_state_report: dict[str, Any]
    paper_exit_signal_report: dict[str, Any]
    paper_sell_position_intents: list[dict[str, Any]]
    synthetic_ledger_update_report: dict[str, Any]
    execution_layer_v2_paper_remap_report: dict[str, Any]
    manifest: dict[str, Any]


def run_polymarket_o_v8_paper_fresh_loop(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> PolymarketOV8PaperFreshLoopResult:
    """Run a bounded paper-only loop over fresh public read-only rows."""

    output_dir = Path(config.output_dir) / config.run_id
    if output_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"paper fresh loop output_dir already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    unlock_evidence = _verify_paper_candidate_unlock(config)
    unlock_verified = bool(unlock_evidence["paper_candidate_unlock_verified"])
    public_data = _resolve_public_data_cycles(config, unlock_evidence)
    public_cycles = public_data["public_data_cycles"]
    public_data_collection_report = public_data["public_data_collection_report"]
    raw_provider_payloads = public_data["raw_provider_payloads"]
    canonical_context = _fresh_canonical_scorer_context(
        config=config,
        unlock_evidence=unlock_evidence,
    )
    canonical_feature_mapping_report, canonical_action_rows = (
        _fresh_canonical_feature_mapping_report(
            config=config,
            public_cycles=public_cycles,
            public_data_collection_report=public_data_collection_report,
            canonical_context=canonical_context,
        )
    )
    canonical_scorer_report = _fresh_canonical_scorer_report(
        config=config,
        canonical_context=canonical_context,
        canonical_feature_mapping_report=canonical_feature_mapping_report,
        canonical_action_rows=canonical_action_rows,
    )
    scorer_comparison_report = _fresh_scorer_comparison_report(
        config=config,
        public_cycles=public_cycles,
        public_data_collection_report=public_data_collection_report,
        canonical_scorer_report=canonical_scorer_report,
    )
    ev_calibration_artifact = _load_frozen_ev_calibration_artifact(
        config.frozen_ev_calibration_artifact_path
    )
    execution_cycles = _fresh_execution_cycles_from_canonical_scorer(
        public_cycles=public_cycles,
        canonical_scorer_report=canonical_scorer_report,
    )
    execution_result = _execute_fresh_public_cycles(
        config=config,
        public_cycles=execution_cycles,
        public_data_source=public_data_collection_report["public_data_source"],
        unlock_verified=unlock_verified,
        ev_calibration_artifact=ev_calibration_artifact,
    )
    intents = execution_result["paper_order_intents"]
    fills = _fresh_paper_fills_from_intents(intents)
    ledger_rows = _fresh_paper_ledger_from_fills(fills)
    execution_layer_v2_paper_remap_report = (
        _execution_layer_v2_paper_remap_report(
            config=config,
            execution_result=execution_result,
            intents=intents,
        )
    )

    run_report = _fresh_loop_run_report(
        config=config,
        unlock_evidence=unlock_evidence,
        public_data_collection_report=public_data_collection_report,
        public_cycles=execution_cycles,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    fill_report = _fresh_fill_simulation_report(config=config, fills=fills)
    no_trade_report = _fresh_no_trade_diagnostic_report(
        config=config,
        public_cycles=execution_cycles,
        public_data_collection_report=public_data_collection_report,
        execution_result=execution_result,
        run_report=run_report,
    )
    score_decomposition_report = _fresh_score_decomposition_report(
        config=config,
        public_cycles=execution_cycles,
        public_data_collection_report=public_data_collection_report,
    )
    provider_feature_coverage_report = _fresh_provider_feature_coverage_report(
        config=config,
        public_cycles=public_cycles,
        public_data_collection_report=public_data_collection_report,
        run_report=run_report,
    )
    canonical_scorer_alignment_report = _fresh_canonical_scorer_alignment_report(
        config=config,
        public_cycles=public_cycles,
        public_data_collection_report=public_data_collection_report,
        canonical_context=canonical_context,
        canonical_feature_mapping_report=canonical_feature_mapping_report,
        canonical_scorer_report=canonical_scorer_report,
        scorer_comparison_report=scorer_comparison_report,
    )
    signal_trace_report = _fresh_signal_trace_report(
        config=config,
        public_cycles=public_cycles,
        public_data_collection_report=public_data_collection_report,
        canonical_scorer_report=canonical_scorer_report,
        scorer_comparison_report=scorer_comparison_report,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
    )
    time_window_diagnostic_report = _fresh_time_window_diagnostic_report(
        config=config,
        signal_trace_report=signal_trace_report,
    )
    legacy_position_policy_audit_report = (
        _fresh_legacy_position_policy_audit_report(config=config)
    )
    exit_adapter_bundle = _fresh_paper_exit_adapter_bundle(
        config=config,
        signal_trace_report=signal_trace_report,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    paper_position_state_report = exit_adapter_bundle["paper_position_state_report"]
    paper_exit_signal_report = exit_adapter_bundle["paper_exit_signal_report"]
    paper_sell_position_intents = exit_adapter_bundle["paper_sell_position_intents"]
    synthetic_ledger_update_report = exit_adapter_bundle[
        "synthetic_ledger_update_report"
    ]
    synthetic_ledger_update_rows = synthetic_ledger_update_report[
        "synthetic_ledger_update_rows"
    ]
    safety_report = _fresh_runtime_safety_report(
        config=config,
        run_report=run_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
        sell_position_intents=paper_sell_position_intents,
        synthetic_exit_ledger_rows=synthetic_ledger_update_rows,
    )
    monitoring_report = _fresh_monitoring_report(
        config=config,
        run_report=run_report,
        execution_result=execution_result,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )
    cumulative_report = _fresh_cumulative_monitoring_report(
        config=config,
        run_report=run_report,
        monitoring_report=monitoring_report,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
    )

    artifact_paths = {
        "fresh_loop_run_report": output_dir / "o_v8_paper_fresh_loop_run_report.json",
        "fresh_loop_run_summary": output_dir / "o_v8_paper_fresh_loop_run_report.md",
        "fresh_order_intent_log": output_dir
        / "o_v8_paper_fresh_order_intent_log.jsonl",
        "fresh_fill_simulation_report": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.json",
        "fresh_fill_simulation_summary": output_dir
        / "o_v8_paper_fresh_fill_simulation_report.md",
        "fresh_fill_log": output_dir / "o_v8_paper_fresh_fill_log.jsonl",
        "fresh_ledger_log": output_dir / "o_v8_paper_fresh_ledger_log.jsonl",
        "fresh_runtime_safety_report": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.json",
        "fresh_runtime_safety_summary": output_dir
        / "o_v8_paper_fresh_runtime_safety_report.md",
        "fresh_monitoring_report": output_dir
        / "o_v8_paper_fresh_monitoring_report.json",
        "fresh_monitoring_summary": output_dir
        / "o_v8_paper_fresh_monitoring_report.md",
        "fresh_cumulative_monitoring_report": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.json",
        "fresh_cumulative_monitoring_summary": output_dir
        / "o_v8_paper_fresh_cumulative_monitoring_report.md",
        "fresh_no_trade_diagnostic_report": output_dir
        / "o_v8_paper_fresh_no_trade_diagnostic.json",
        "fresh_no_trade_diagnostic_summary": output_dir
        / "o_v8_paper_fresh_no_trade_diagnostic.md",
        "fresh_score_decomposition_report": output_dir
        / "o_v8_paper_fresh_score_decomposition_report.json",
        "fresh_score_decomposition_summary": output_dir
        / "o_v8_paper_fresh_score_decomposition_report.md",
        "fresh_provider_feature_coverage_report": output_dir
        / "o_v8_paper_fresh_provider_feature_coverage_report.json",
        "fresh_provider_feature_coverage_summary": output_dir
        / "o_v8_paper_fresh_provider_feature_coverage_report.md",
        "fresh_canonical_feature_mapping_report": output_dir
        / "o_v8_paper_fresh_canonical_feature_mapping_report.json",
        "fresh_canonical_feature_mapping_summary": output_dir
        / "o_v8_paper_fresh_canonical_feature_mapping_report.md",
        "fresh_canonical_action_rows": output_dir
        / "o_v8_paper_fresh_canonical_action_rows.jsonl",
        "fresh_canonical_scorer_report": output_dir
        / "o_v8_paper_fresh_canonical_scorer_report.json",
        "fresh_canonical_scorer_summary": output_dir
        / "o_v8_paper_fresh_canonical_scorer_report.md",
        "fresh_scorer_comparison_report": output_dir
        / "o_v8_paper_fresh_scorer_comparison_report.json",
        "fresh_scorer_comparison_summary": output_dir
        / "o_v8_paper_fresh_scorer_comparison_report.md",
        "fresh_canonical_scorer_alignment_report": output_dir
        / "o_v8_paper_fresh_canonical_scorer_alignment_report.json",
        "fresh_canonical_scorer_alignment_summary": output_dir
        / "o_v8_paper_fresh_canonical_scorer_alignment_report.md",
        "fresh_signal_trace_report": output_dir
        / "o_v8_paper_fresh_signal_trace.json",
        "fresh_signal_trace_summary": output_dir
        / "o_v8_paper_fresh_signal_trace.md",
        "fresh_time_window_diagnostic_report": output_dir
        / "o_v8_paper_fresh_time_window_diagnostic.json",
        "fresh_time_window_diagnostic_summary": output_dir
        / "o_v8_paper_fresh_time_window_diagnostic.md",
        "fresh_legacy_position_policy_audit_report": output_dir
        / "o_v8_paper_fresh_legacy_position_policy_audit.json",
        "fresh_legacy_position_policy_audit_summary": output_dir
        / "o_v8_paper_fresh_legacy_position_policy_audit.md",
        "fresh_paper_position_state_report": output_dir
        / "o_v8_paper_fresh_position_state_report.json",
        "fresh_paper_position_state_summary": output_dir
        / "o_v8_paper_fresh_position_state_report.md",
        "fresh_paper_exit_signal_report": output_dir
        / "o_v8_paper_fresh_exit_signal_report.json",
        "fresh_paper_exit_signal_summary": output_dir
        / "o_v8_paper_fresh_exit_signal_report.md",
        "fresh_paper_sell_position_intent_log": output_dir
        / "o_v8_paper_fresh_sell_position_intent_log.jsonl",
        "fresh_synthetic_ledger_update_report": output_dir
        / "o_v8_paper_fresh_synthetic_ledger_update_report.json",
        "fresh_synthetic_ledger_update_summary": output_dir
        / "o_v8_paper_fresh_synthetic_ledger_update_report.md",
        "execution_layer_v2_paper_remap_report": output_dir
        / "execution_layer_v2_paper_remap_report.json",
        "execution_layer_v2_paper_remap_summary": output_dir
        / "execution_layer_v2_paper_remap_report.md",
        "raw_polymarket_markets": output_dir / "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks": output_dir
        / "raw_polymarket_orderbooks.jsonl",
        "raw_polymarket_trades": output_dir / "raw_polymarket_trades.jsonl",
        "raw_btc_feature_candles": output_dir
        / "raw_btc_feature_candles.jsonl",
        "raw_polymarket_chainlink_prices": output_dir
        / "raw_polymarket_chainlink_prices.jsonl",
        "chainlink_rtds_collection_report": output_dir
        / "chainlink_rtds_collection_report.json",
        "manifest": output_dir / "o_v8_paper_fresh_loop_manifest.json",
    }
    _write_json(artifact_paths["fresh_loop_run_report"], run_report)
    _write_text(artifact_paths["fresh_loop_run_summary"], _fresh_loop_run_md(run_report))
    _write_jsonl(artifact_paths["fresh_order_intent_log"], intents)
    _write_json(artifact_paths["fresh_fill_simulation_report"], fill_report)
    _write_text(
        artifact_paths["fresh_fill_simulation_summary"],
        _fresh_fill_simulation_md(fill_report),
    )
    _write_jsonl(artifact_paths["fresh_fill_log"], fills)
    _write_jsonl(artifact_paths["fresh_ledger_log"], ledger_rows)
    _write_json(artifact_paths["fresh_runtime_safety_report"], safety_report)
    _write_text(
        artifact_paths["fresh_runtime_safety_summary"],
        _fresh_runtime_safety_md(safety_report),
    )
    _write_json(artifact_paths["fresh_monitoring_report"], monitoring_report)
    _write_text(
        artifact_paths["fresh_monitoring_summary"],
        _fresh_monitoring_md(monitoring_report),
    )
    _write_json(
        artifact_paths["fresh_cumulative_monitoring_report"],
        cumulative_report,
    )
    _write_text(
        artifact_paths["fresh_cumulative_monitoring_summary"],
        _fresh_cumulative_monitoring_md(cumulative_report),
    )
    _write_json(artifact_paths["fresh_no_trade_diagnostic_report"], no_trade_report)
    _write_text(
        artifact_paths["fresh_no_trade_diagnostic_summary"],
        _fresh_no_trade_diagnostic_md(no_trade_report),
    )
    _write_json(
        artifact_paths["fresh_score_decomposition_report"],
        score_decomposition_report,
    )
    _write_text(
        artifact_paths["fresh_score_decomposition_summary"],
        _fresh_score_decomposition_md(score_decomposition_report),
    )
    _write_json(
        artifact_paths["fresh_provider_feature_coverage_report"],
        provider_feature_coverage_report,
    )
    _write_text(
        artifact_paths["fresh_provider_feature_coverage_summary"],
        _fresh_provider_feature_coverage_md(provider_feature_coverage_report),
    )
    _write_json(
        artifact_paths["fresh_canonical_feature_mapping_report"],
        canonical_feature_mapping_report,
    )
    _write_text(
        artifact_paths["fresh_canonical_feature_mapping_summary"],
        _fresh_canonical_feature_mapping_md(canonical_feature_mapping_report),
    )
    _write_jsonl(artifact_paths["fresh_canonical_action_rows"], canonical_action_rows)
    _write_json(
        artifact_paths["fresh_canonical_scorer_report"],
        canonical_scorer_report,
    )
    _write_text(
        artifact_paths["fresh_canonical_scorer_summary"],
        _fresh_canonical_scorer_md(canonical_scorer_report),
    )
    _write_json(
        artifact_paths["fresh_scorer_comparison_report"],
        scorer_comparison_report,
    )
    _write_text(
        artifact_paths["fresh_scorer_comparison_summary"],
        _fresh_scorer_comparison_md(scorer_comparison_report),
    )
    _write_json(
        artifact_paths["fresh_canonical_scorer_alignment_report"],
        canonical_scorer_alignment_report,
    )
    _write_text(
        artifact_paths["fresh_canonical_scorer_alignment_summary"],
        _fresh_canonical_scorer_alignment_md(canonical_scorer_alignment_report),
    )
    _write_json(artifact_paths["fresh_signal_trace_report"], signal_trace_report)
    _write_text(
        artifact_paths["fresh_signal_trace_summary"],
        _fresh_signal_trace_md(signal_trace_report),
    )
    _write_json(
        artifact_paths["fresh_time_window_diagnostic_report"],
        time_window_diagnostic_report,
    )
    _write_text(
        artifact_paths["fresh_time_window_diagnostic_summary"],
        _fresh_time_window_diagnostic_md(time_window_diagnostic_report),
    )
    _write_json(
        artifact_paths["fresh_legacy_position_policy_audit_report"],
        legacy_position_policy_audit_report,
    )
    _write_text(
        artifact_paths["fresh_legacy_position_policy_audit_summary"],
        _fresh_legacy_position_policy_audit_md(
            legacy_position_policy_audit_report
        ),
    )
    _write_json(
        artifact_paths["fresh_paper_position_state_report"],
        paper_position_state_report,
    )
    _write_text(
        artifact_paths["fresh_paper_position_state_summary"],
        _fresh_paper_position_state_md(paper_position_state_report),
    )
    _write_json(
        artifact_paths["fresh_paper_exit_signal_report"],
        paper_exit_signal_report,
    )
    _write_text(
        artifact_paths["fresh_paper_exit_signal_summary"],
        _fresh_paper_exit_signal_md(paper_exit_signal_report),
    )
    _write_jsonl(
        artifact_paths["fresh_paper_sell_position_intent_log"],
        paper_sell_position_intents,
    )
    _write_json(
        artifact_paths["fresh_synthetic_ledger_update_report"],
        synthetic_ledger_update_report,
    )
    _write_text(
        artifact_paths["fresh_synthetic_ledger_update_summary"],
        _fresh_synthetic_ledger_update_md(synthetic_ledger_update_report),
    )
    _write_json(
        artifact_paths["execution_layer_v2_paper_remap_report"],
        execution_layer_v2_paper_remap_report,
    )
    _write_text(
        artifact_paths["execution_layer_v2_paper_remap_summary"],
        _execution_layer_v2_paper_remap_md(
            execution_layer_v2_paper_remap_report
        ),
    )
    _write_jsonl(
        artifact_paths["raw_polymarket_markets"],
        raw_provider_payloads["markets"],
    )
    _write_jsonl(
        artifact_paths["raw_polymarket_orderbooks"],
        raw_provider_payloads["orderbooks"],
    )
    _write_jsonl(
        artifact_paths["raw_polymarket_trades"],
        raw_provider_payloads["trades"],
    )
    _write_jsonl(
        artifact_paths["raw_btc_feature_candles"],
        raw_provider_payloads["btc_feature_candles"],
    )
    _write_jsonl(
        artifact_paths["raw_polymarket_chainlink_prices"],
        raw_provider_payloads["chainlink_rtds_prices"],
    )
    _write_json(
        artifact_paths["chainlink_rtds_collection_report"],
        {
            "report_type": "polymarket_chainlink_rtds_fresh_loop_context",
            "source_type": "polymarket_rtds_chainlink",
            "raw_price_row_count": len(
                raw_provider_payloads["chainlink_rtds_prices"]
            ),
            "timestamp_causality_violation_count": sum(
                1
                for row in raw_provider_payloads["chainlink_rtds_prices"]
                if row.get("timestamp_causality_valid") is not True
            ),
            "collector_lifecycle_owned_by_parent_runner": True,
            "decision_critical": False,
            "fail_closed_when_feature_unavailable": True,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "v8_execution_handoff_allowed": False,
        },
    )

    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "manifest"
    }
    manifest = _fresh_loop_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        unlock_evidence=unlock_evidence,
        public_data_collection_report=public_data_collection_report,
        run_report=run_report,
        fill_report=fill_report,
        safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_report=cumulative_report,
        no_trade_report=no_trade_report,
        score_decomposition_report=score_decomposition_report,
        provider_feature_coverage_report=provider_feature_coverage_report,
        canonical_feature_mapping_report=canonical_feature_mapping_report,
        canonical_scorer_report=canonical_scorer_report,
        scorer_comparison_report=scorer_comparison_report,
        canonical_scorer_alignment_report=canonical_scorer_alignment_report,
        signal_trace_report=signal_trace_report,
        time_window_diagnostic_report=time_window_diagnostic_report,
        legacy_position_policy_audit_report=legacy_position_policy_audit_report,
        paper_position_state_report=paper_position_state_report,
        paper_exit_signal_report=paper_exit_signal_report,
        paper_sell_position_intents=paper_sell_position_intents,
        synthetic_ledger_update_report=synthetic_ledger_update_report,
        execution_layer_v2_paper_remap_report=execution_layer_v2_paper_remap_report,
    )
    _write_json(artifact_paths["manifest"], manifest)
    artifact_hashes["manifest"] = _sha256_file(artifact_paths["manifest"])

    return PolymarketOV8PaperFreshLoopResult(
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        fresh_loop_run_report=run_report,
        fill_simulation_report=fill_report,
        runtime_safety_report=safety_report,
        monitoring_report=monitoring_report,
        cumulative_monitoring_report=cumulative_report,
        no_trade_diagnostic_report=no_trade_report,
        score_decomposition_report=score_decomposition_report,
        provider_feature_coverage_report=provider_feature_coverage_report,
        canonical_feature_mapping_report=canonical_feature_mapping_report,
        canonical_action_rows=canonical_action_rows,
        canonical_scorer_report=canonical_scorer_report,
        scorer_comparison_report=scorer_comparison_report,
        canonical_scorer_alignment_report=canonical_scorer_alignment_report,
        signal_trace_report=signal_trace_report,
        time_window_diagnostic_report=time_window_diagnostic_report,
        legacy_position_policy_audit_report=legacy_position_policy_audit_report,
        paper_position_state_report=paper_position_state_report,
        paper_exit_signal_report=paper_exit_signal_report,
        paper_sell_position_intents=paper_sell_position_intents,
        synthetic_ledger_update_report=synthetic_ledger_update_report,
        execution_layer_v2_paper_remap_report=execution_layer_v2_paper_remap_report,
        manifest=manifest,
    )


def _verify_paper_candidate_unlock(
    config: PolymarketOV8PaperFreshLoopConfig,
) -> dict[str, Any]:
    unlock_dir = Path(config.paper_candidate_unlock_dir)
    manifest_path = unlock_dir / "o_v8_paper_candidate_unlock_manifest.json"
    observed_manifest_sha = _sha256_file(manifest_path) if manifest_path.exists() else ""
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    expected_manifest_sha = config.expected_paper_candidate_unlock_manifest_sha256
    manifest_hash_passed = (
        expected_manifest_sha is None or observed_manifest_sha == expected_manifest_sha
    )
    artifact_hash_rows: list[dict[str, Any]] = []
    artifact_hashes_match = True
    for name, expected_hash in sorted((manifest.get("artifact_hashes") or {}).items()):
        artifact_path = _resolve_unlock_artifact_path(
            unlock_dir,
            (manifest.get("artifact_paths") or {}).get(name) or "",
        )
        observed_hash = _sha256_file(artifact_path) if artifact_path.exists() else ""
        passed = observed_hash == expected_hash
        artifact_hashes_match = artifact_hashes_match and passed
        artifact_hash_rows.append(
            {
                "artifact_name": name,
                "artifact_path": str(artifact_path),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "passed": passed,
            }
        )
    required_flags = {
        "paper_candidate_allowed": True,
        "paper_internal_execution_loop_enabled": True,
        "v8_paper_internal_handoff_allowed": True,
        "v8_execution_handoff_allowed": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    flag_rows = {
        field_name: {
            "expected": expected,
            "observed": manifest.get(field_name),
            "passed": manifest.get(field_name) is expected,
        }
        for field_name, expected in required_flags.items()
    }
    blocking_reason_codes = []
    if not manifest_path.exists():
        blocking_reason_codes.append("paper_candidate_unlock_manifest_missing")
    if not manifest_hash_passed:
        blocking_reason_codes.append("paper_candidate_unlock_manifest_hash_mismatch")
    if not artifact_hashes_match:
        blocking_reason_codes.append("paper_candidate_unlock_artifact_hash_mismatch")
    if any(row["passed"] is not True for row in flag_rows.values()):
        blocking_reason_codes.append("paper_candidate_unlock_safety_flags_invalid")
    return {
        "paper_candidate_unlock_dir": str(unlock_dir),
        "paper_candidate_unlock_manifest_path": str(manifest_path),
        "expected_manifest_sha256": expected_manifest_sha,
        "observed_manifest_sha256": observed_manifest_sha,
        "manifest_hash_passed": manifest_hash_passed,
        "artifact_hash_rows": artifact_hash_rows,
        "artifact_hashes_match": artifact_hashes_match,
        "required_flag_checks": flag_rows,
        "paper_candidate_unlock_verified": blocking_reason_codes == [],
        "paper_candidate_unlock_blocking_reason_codes": sorted(blocking_reason_codes),
        "unlock_manifest": manifest,
    }


def _resolve_public_data_cycles(
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    if config.public_data_cycles is not None:
        cycles = [
            [dict(row) for row in cycle] for cycle in config.public_data_cycles
        ]
        return {
            "public_data_cycles": cycles,
            "public_data_collection_report": _snapshot_fixture_collection_report(
                config=config,
                cycles=cycles,
                unlock_evidence=unlock_evidence,
            ),
            "raw_provider_payloads": _empty_raw_provider_payloads(),
        }
    return _collect_read_only_public_provider_cycles(
        config=config,
        unlock_evidence=unlock_evidence,
    )


def _snapshot_fixture_collection_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    cycles: list[list[dict[str, Any]]],
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    row_count = sum(len(cycle) for cycle in cycles)
    return {
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
        "public_data_collection_mode": "offline_snapshot_fixture",
        "public_provider_class": None,
        "public_provider_read_only": True,
        "paper_fresh_provider_collection_failed": False,
        "public_data_collection_reason_codes": [],
        "public_data_cycle_count": len(cycles),
        "public_data_row_count": row_count,
        "public_market_count": len(
            {str(row.get("market_id")) for cycle in cycles for row in cycle}
        ),
        "public_orderbook_row_count": None,
        "orderbook_source_type_distribution": {},
        "orderbook_rest_fallback_row_count": 0,
        "orderbook_fallback_reason_distribution": {},
        "public_trade_row_count": None,
        "public_btc_feature_candle_row_count": None,
        "public_feature_row_count": row_count,
        "frozen_o_action_rank_reference_source": "issue_160_paper_candidate_unlock_manifest",
        "frozen_o_action_rank_reference_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "scoring_rule_id": "snapshot_fixture_pre_scored_rows",
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
    }


def _collect_read_only_public_provider_cycles(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    provider = config.public_provider or PolymarketPublicHTTPRealCorpusProvider()
    provider_class = provider.__class__.__name__
    provider_safety = _public_provider_safety(provider)
    base_report = {
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_data_collection_mode": "read_only_public_provider_live_snapshot",
        "public_provider_class": provider_class,
        **provider_safety,
        "frozen_o_action_rank_reference_source": "issue_160_paper_candidate_unlock_manifest",
        "frozen_o_action_rank_reference_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_frozen_o_scorer_used": False,
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
    }
    if not provider_safety["public_provider_safety_passed"]:
        report = {
            **base_report,
            "paper_fresh_provider_collection_failed": True,
            "public_data_collection_reason_codes": [
                "read_only_public_provider_safety_flags_invalid"
            ],
            "public_data_cycle_count": config.max_cycles,
            "public_data_row_count": 0,
            "public_market_count": 0,
            "public_orderbook_row_count": 0,
            "orderbook_source_type_distribution": {},
            "orderbook_rest_fallback_row_count": 0,
            "orderbook_fallback_reason_distribution": {},
            "public_trade_row_count": 0,
            "public_btc_feature_candle_row_count": 0,
            "public_feature_row_count": 0,
            "provider_exception_type": None,
            "provider_exception_message": None,
        }
        return {
            "public_data_cycles": [[] for _ in range(config.max_cycles)],
            "public_data_collection_report": report,
            "raw_provider_payloads": _empty_raw_provider_payloads(),
        }

    recorder_config = PolymarketRealCorpusRecorderConfig(
        run_id=f"{config.run_id}-fresh-public-provider",
        output_dir=Path(config.output_dir) / config.run_id / "_public_provider_input",
        market_families=tuple(BTC_UPDOWN_MARKET_HORIZONS_MS),
        mock_public_data=False,
        build_phase2_corpus=False,
    )
    stage_statuses: dict[str, dict[str, Any]] = {}
    markets = _call_public_provider_stage(
        stage_name="market_discovery",
        decision_critical=True,
        callback=lambda: provider.market_rows(recorder_config),
        stage_statuses=stage_statuses,
    )
    if markets:
        orderbooks = _call_public_provider_stage(
            stage_name="orderbook_collection",
            decision_critical=True,
            callback=lambda: provider.orderbook_rows(markets, recorder_config),
            stage_statuses=stage_statuses,
        )
        trades = _call_public_provider_stage(
            stage_name="trade_collection",
            decision_critical=False,
            callback=lambda: provider.trade_rows(markets, recorder_config),
            stage_statuses=stage_statuses,
        )
        btc_candles = _call_public_provider_stage(
            stage_name="btc_feature_candle_collection",
            decision_critical=True,
            callback=lambda: provider.btc_feature_candle_rows(
                markets, recorder_config
            ),
            stage_statuses=stage_statuses,
        )
    else:
        orderbooks = []
        trades = []
        btc_candles = []
        for stage_name, decision_critical in (
            ("orderbook_collection", True),
            ("trade_collection", False),
            ("btc_feature_candle_collection", True),
        ):
            stage_statuses[stage_name] = {
                "stage_name": stage_name,
                "decision_critical": decision_critical,
                "attempted": False,
                "passed": False,
                "row_count": 0,
                "reason_codes": ["provider_stage_skipped_missing_markets"],
                "exception_type": None,
                "exception_message": None,
            }
    rows = _fresh_public_rows_from_provider_payloads(
        run_id=config.run_id,
        markets=markets,
        orderbooks=orderbooks,
        trades=trades,
        btc_candles=btc_candles,
        chainlink_rtds_prices=[
            dict(row) for row in config.chainlink_rtds_price_rows
        ],
    )
    persisted_chainlink_rows = (
        [dict(row) for row in config.chainlink_rtds_persist_price_rows]
        if config.chainlink_rtds_persist_price_rows is not None
        else [dict(row) for row in config.chainlink_rtds_price_rows]
    )
    stage_statuses["decision_feature_build"] = {
        "stage_name": "decision_feature_build",
        "decision_critical": True,
        "attempted": True,
        "passed": bool(rows),
        "row_count": len(rows),
        "reason_codes": []
        if rows
        else ["read_only_public_provider_no_decision_feature_rows"],
        "exception_type": None,
        "exception_message": None,
    }
    collection_failed = not rows
    critical_stage_failures = [
        status
        for status in stage_statuses.values()
        if status["decision_critical"] is True and status["passed"] is not True
    ]
    optional_stage_failures = [
        status
        for status in stage_statuses.values()
        if status["decision_critical"] is False and status["passed"] is not True
    ]
    reason_codes = sorted(
        {
            str(reason)
            for status in critical_stage_failures
            for reason in status["reason_codes"]
        }
    )
    degradation_reason_codes = sorted(
        {
            str(reason)
            for status in optional_stage_failures
            for reason in status["reason_codes"]
        }
    )
    first_critical_exception = next(
        (
            status
            for status in critical_stage_failures
            if status.get("exception_type")
        ),
        None,
    )
    exception_type = (
        None
        if first_critical_exception is None
        else first_critical_exception["exception_type"]
    )
    exception_message = (
        None
        if first_critical_exception is None
        else first_critical_exception["exception_message"]
    )

    cycles = _partition_public_rows(rows, config.max_cycles)
    orderbook_source_counter = Counter(
        str(row.get("orderbook_source_type") or "unknown") for row in orderbooks
    )
    orderbook_fallback_reason_counter: Counter[str] = Counter()
    for row in orderbooks:
        orderbook_fallback_reason_counter.update(
            str(reason)
            for reason in row.get("orderbook_fallback_reason_codes") or []
        )
    report = {
        **base_report,
        "paper_fresh_provider_collection_failed": collection_failed,
        "public_data_collection_reason_codes": sorted(set(reason_codes)),
        "public_data_degraded": bool(optional_stage_failures),
        "public_data_degradation_reason_codes": degradation_reason_codes,
        "provider_stage_statuses": stage_statuses,
        "decision_critical_provider_failure": collection_failed,
        "decision_optional_provider_failure": bool(optional_stage_failures),
        "public_data_cycle_count": len(cycles),
        "public_data_row_count": len(rows),
        "public_market_count": len(markets),
        "public_orderbook_row_count": len(orderbooks),
        "orderbook_source_type_distribution": dict(
            sorted(orderbook_source_counter.items())
        ),
        "orderbook_rest_fallback_row_count": sum(
            1 for row in orderbooks if row.get("orderbook_rest_fallback_used") is True
        ),
        "orderbook_fallback_reason_distribution": dict(
            sorted(orderbook_fallback_reason_counter.items())
        ),
        "public_trade_row_count": len(trades),
        "public_btc_feature_candle_row_count": len(btc_candles),
        "public_chainlink_rtds_price_row_count": len(
            config.chainlink_rtds_price_rows
        ),
        "public_chainlink_rtds_persisted_price_row_count": len(
            persisted_chainlink_rows
        ),
        "chainlink_rtds_feature_available": bool(config.chainlink_rtds_price_rows),
        "chainlink_rtds_decision_critical": False,
        "public_feature_row_count": len(rows),
        "provider_exception_type": exception_type,
        "provider_exception_message": exception_message,
    }
    return {
        "public_data_cycles": cycles,
        "public_data_collection_report": report,
        "raw_provider_payloads": {
            "markets": [dict(row) for row in markets],
            "orderbooks": [dict(row) for row in orderbooks],
            "trades": [dict(row) for row in trades],
            "btc_feature_candles": [dict(row) for row in btc_candles],
            "chainlink_rtds_prices": persisted_chainlink_rows,
        },
    }


def _call_public_provider_stage(
    *,
    stage_name: str,
    decision_critical: bool,
    callback: Any,
    stage_statuses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        rows = callback()
    except RealCorpusPublicProviderError as exc:
        reason_codes = list(exc.reason_codes) or [
            f"{stage_name}_public_provider_failed"
        ]
        stage_statuses[stage_name] = {
            "stage_name": stage_name,
            "decision_critical": decision_critical,
            "attempted": True,
            "passed": False,
            "row_count": 0,
            "reason_codes": reason_codes,
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
        }
        return []
    except Exception as exc:  # noqa: BLE001
        stage_statuses[stage_name] = {
            "stage_name": stage_name,
            "decision_critical": decision_critical,
            "attempted": True,
            "passed": False,
            "row_count": 0,
            "reason_codes": [f"{stage_name}_public_provider_failed"],
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
        }
        return []
    normalized_rows = [dict(row) for row in rows]
    stage_statuses[stage_name] = {
        "stage_name": stage_name,
        "decision_critical": decision_critical,
        "attempted": True,
        "passed": bool(normalized_rows),
        "row_count": len(normalized_rows),
        "reason_codes": []
        if normalized_rows
        else [f"{stage_name}_returned_no_rows"],
        "exception_type": None,
        "exception_message": None,
    }
    return normalized_rows


def _empty_raw_provider_payloads() -> dict[str, list[dict[str, Any]]]:
    return {
        "markets": [],
        "orderbooks": [],
        "trades": [],
        "btc_feature_candles": [],
        "chainlink_rtds_prices": [],
    }


def _public_provider_safety(provider: Any) -> dict[str, Any]:
    required = {
        "read_only": True,
        "write_capable": False,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    checks = {
        field_name: {
            "expected": expected,
            "observed": getattr(provider, field_name, None),
            "passed": getattr(provider, field_name, None) is expected,
        }
        for field_name, expected in required.items()
    }
    return {
        "public_provider_read_only": checks["read_only"]["passed"],
        "public_provider_safety_passed": all(
            row["passed"] is True for row in checks.values()
        ),
        "public_provider_safety_checks": checks,
    }


def _fresh_public_rows_from_provider_payloads(
    *,
    run_id: str,
    markets: list[dict[str, Any]],
    orderbooks: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    btc_candles: list[dict[str, Any]],
    chainlink_rtds_prices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    del trades
    markets_by_id = {str(market.get("market_id")): dict(market) for market in markets}
    books_by_market = _latest_public_books_by_market(orderbooks)
    candles = sorted(btc_candles, key=lambda row: int(row.get("available_at_ts") or row.get("ts") or 0))
    chainlink_prices = sorted(
        chainlink_rtds_prices or [],
        key=lambda row: (
            int(row.get("available_at_ts") or 0),
            int(row.get("source_ts") or 0),
        ),
    )
    rows: list[dict[str, Any]] = []
    for market_id, pair in sorted(books_by_market.items()):
        market = markets_by_id.get(market_id)
        if market is None or "UP" not in pair or "DOWN" not in pair:
            continue
        up = pair["UP"]
        down = pair["DOWN"]
        book_decision_ts = max(
            _book_available_at(up),
            _book_available_at(down),
            int(market.get("market_start_ts") or 0),
        )
        latest_chainlink = _latest_chainlink_price_available_at(
            chainlink_prices,
            available_at_or_before=max(
                book_decision_ts,
                max(
                    (
                        int(row.get("available_at_ts") or 0)
                        for row in chainlink_prices
                    ),
                    default=0,
                ),
            ),
        )
        decision_ts = max(
            book_decision_ts,
            int((latest_chainlink or {}).get("available_at_ts") or 0),
        )
        market_end_ts = int(market.get("market_end_ts") or 0)
        if decision_ts <= 0 or market_end_ts <= decision_ts:
            continue
        candle = _latest_public_btc_candle(candles, decision_ts)
        if candle is None:
            continue
        reference_candle = _market_start_reference_public_btc_candle(
            candles,
            market_start_ts=int(market.get("market_start_ts") or 0),
            decision_ts=decision_ts,
        )
        rows.append(
            _fresh_public_row_from_provider_feature_context(
                run_id=run_id,
                row_index=len(rows),
                market=market,
                up=up,
                down=down,
                candle=candle,
                reference_candle=reference_candle,
                chainlink_rtds_prices=chainlink_prices,
                decision_ts=decision_ts,
            )
        )
    return rows


def _latest_public_books_by_market(
    orderbooks: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in orderbooks:
        market_id = str(row.get("market_id") or "")
        outcome = str(row.get("outcome") or "").upper()
        if market_id == "" or outcome not in {"UP", "DOWN"}:
            continue
        previous = grouped.setdefault(market_id, {}).get(outcome)
        if previous is None or _book_available_at(row) >= _book_available_at(previous):
            grouped[market_id][outcome] = dict(row)
    return grouped


def _latest_public_btc_candle(
    candles: list[dict[str, Any]],
    decision_ts: int,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in candles:
        available_at = int(row.get("available_at_ts") or row.get("ts") or 0)
        if available_at <= decision_ts:
            latest = dict(row)
        if available_at > decision_ts:
            break
    return latest


def _latest_chainlink_price_available_at(
    rows: list[dict[str, Any]],
    *,
    available_at_or_before: int,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in rows:
        available_at = int(row.get("available_at_ts") or 0)
        if available_at <= available_at_or_before:
            latest = dict(row)
        if available_at > available_at_or_before:
            break
    return latest


def _market_start_reference_public_btc_candle(
    candles: list[dict[str, Any]],
    *,
    market_start_ts: int,
    decision_ts: int,
) -> dict[str, Any] | None:
    if market_start_ts <= 0:
        return None
    latest: dict[str, Any] | None = None
    for row in candles:
        available_at = int(row.get("available_at_ts") or row.get("ts") or 0)
        close_time = int(row.get("close_time") or available_at)
        if available_at <= decision_ts and close_time <= market_start_ts:
            latest = dict(row)
        if close_time > market_start_ts and available_at > decision_ts:
            break
    return latest


def _fresh_public_row_from_provider_feature_context(
    *,
    run_id: str,
    row_index: int,
    market: dict[str, Any],
    up: dict[str, Any],
    down: dict[str, Any],
    candle: dict[str, Any],
    reference_candle: dict[str, Any] | None = None,
    chainlink_rtds_prices: list[dict[str, Any]] | None = None,
    decision_ts: int,
) -> dict[str, Any]:
    p_up = _public_p_up(up=up, down=down)
    p_down = 1.0 - p_up
    scores = _provider_action_scores(p_up=p_up, p_down=p_down, up=up, down=down)
    ranking = _provider_full_action_ranking(
        scores=scores,
        p_up=p_up,
        up=up,
        down=down,
        market=market,
        decision_ts=decision_ts,
    )
    selected = ranking[0]
    selected_action = str(selected["selected_action"])
    selected_side = _side_from_action(selected_action)
    regime_features = _provider_decision_time_regime_features(
        market=market,
        candle=candle,
        reference_candle=reference_candle,
        chainlink_rtds_prices=chainlink_rtds_prices or [],
        decision_ts=decision_ts,
        ranking=ranking,
        selected_action=selected_action,
    )
    reference_provenance = dict(
        regime_features.get("reference_price_to_beat_distance_provenance")
        or _provider_reference_price_provenance(
            market=market,
            candle=candle,
            decision_ts=decision_ts,
        )
    )
    max_input_ts = max(
        _book_available_at(up),
        _book_available_at(down),
        int(candle.get("available_at_ts") or candle.get("ts") or 0),
        int(reference_provenance.get("max_input_ts") or 0),
        int(regime_features.get("decision_time_regime_feature_max_input_ts") or 0),
    )
    market_start_ts = int(market.get("market_start_ts") or 0)
    market_end_ts = int(market.get("market_end_ts") or 0)
    horizon_ms = int(
        market.get("horizon_ms") or max(0, market_end_ts - market_start_ts)
    )
    market_schedule_provenance = {
        "source_type": "normalized_public_market_metadata",
        "source_fields_used": [
            "raw_polymarket_markets.slug",
            "raw_polymarket_markets.market_start_ts",
            "raw_polymarket_markets.market_end_ts",
            "raw_polymarket_markets.horizon_ms",
        ],
        "raw_market_sha256": market.get("raw_market_sha256"),
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "provenance_valid": bool(
            market_start_ts > 0
            and market_end_ts > market_start_ts
            and horizon_ms == market_end_ts - market_start_ts
        ),
        "warning_reason_codes": [],
    }
    return {
        "decision_group_id": (
            f"{run_id}|read-only-public-provider|{market.get('market_id')}|"
            f"{decision_ts}"
        ),
        "market_id": str(market.get("market_id") or ""),
        "condition_id": str(market.get("condition_id") or ""),
        "slug": str(market.get("slug") or ""),
        "market_family": str(market.get("market_family") or ""),
        "market_start_ts": market_start_ts,
        "market_end_ts": market_end_ts,
        "horizon_ms": horizon_ms,
        "market_schedule_source_type": "normalized_public_market_metadata",
        "market_schedule_provenance": market_schedule_provenance,
        "up_token_id": str(market.get("up_token_id") or up.get("token_id") or ""),
        "down_token_id": str(market.get("down_token_id") or down.get("token_id") or ""),
        "reference_price_source": str(market.get("reference_price_source") or ""),
        "reference_price_start": market.get("reference_price_start"),
        "reference_price_at_start": market.get("reference_price_at_start"),
        "reference_price_to_beat_at_decision": regime_features.get(
            "reference_price_to_beat_at_decision"
        ),
        "raw_market_sha256": market.get("raw_market_sha256"),
        "decision_ts": decision_ts,
        "selected_action": selected_action,
        "selected_side": selected_side,
        "selected_action_family": _action_family(selected_action),
        "corrected_model_score": _float(selected.get("corrected_model_score")),
        "raw_model_score": _float(selected.get("raw_model_score")),
        "high_score_flag": _float(selected.get("corrected_model_score")) >= 0.02,
        "p_up": p_up,
        "p_down": p_down,
        "p_up_action_disagreement": _p_up_action_disagreement(
            action=selected_action,
            p_up=p_up,
        ),
        "microstructure_snapshot": _provider_microstructure_for_action(
            action=selected_action,
            up=up,
            down=down,
            market=market,
            decision_ts=decision_ts,
        ),
        "reference_price_feature_provenance": reference_provenance,
        **regime_features,
        "decision_time_feature_max_input_ts": max_input_ts,
        "full_5_action_ranking": ranking,
        "score_components": {
            "scoring_rule_id": "fresh_provider_simplified_score",
            "canonical_frozen_o_scorer_used": False,
            "p_up": p_up,
            "p_down": p_down,
            "btc_mid_price": _float(candle.get("close_price")),
            "reference_price_to_beat": _float(
                regime_features.get("reference_price_to_beat_at_decision")
            ),
            "btc_momentum": regime_features.get("btc_momentum"),
            "reference_price_to_beat_distance_at_decision": regime_features.get(
                "reference_price_to_beat_distance_at_decision"
            ),
            "time_since_market_start_seconds": regime_features.get(
                "time_since_market_start_seconds"
            ),
            "action_score_margin": regime_features.get("action_score_margin"),
            "side_specific_action_score_margin": regime_features.get(
                "side_specific_action_score_margin"
            ),
            "chainlink_price_at_decision": regime_features.get(
                "chainlink_price_at_decision"
            ),
            "chainlink_reference_price_at_market_start": regime_features.get(
                "chainlink_reference_price_at_market_start"
            ),
            "chainlink_reference_distance_at_decision": regime_features.get(
                "chainlink_reference_distance_at_decision"
            ),
            "chainlink_momentum_30s": regime_features.get("chainlink_momentum_30s"),
            "chainlink_momentum_60s": regime_features.get("chainlink_momentum_60s"),
            "chainlink_momentum_120s": regime_features.get(
                "chainlink_momentum_120s"
            ),
            "chainlink_realized_volatility_120s": regime_features.get(
                "chainlink_realized_volatility_120s"
            ),
            "chainlink_vs_btc_feature_price_gap": regime_features.get(
                "chainlink_vs_btc_feature_price_gap"
            ),
            "max_input_ts": max_input_ts,
        },
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_provider_row_index": row_index,
        "public_provider_feature_builder_rule_id": (
            "public_provider_market_orderbook_btc_chainlink_to_decision_features_v2"
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _provider_action_scores(
    *,
    p_up: float,
    p_down: float,
    up: dict[str, Any],
    down: dict[str, Any],
) -> dict[str, float]:
    return {
        action: _float(
            _provider_score_decomposition(
                action=action,
                p_up=p_up,
                p_down=p_down,
                up=up,
                down=down,
            )["corrected_score"]
        )
        for action in O_REQUIRED_DECISION_ACTION_FAMILIES
    }


def _provider_full_action_ranking(
    *,
    scores: dict[str, float],
    p_up: float,
    up: dict[str, Any],
    down: dict[str, Any],
    market: dict[str, Any],
    decision_ts: int,
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
        side = _side_from_action(action)
        score = _float(scores.get(action))
        decomposition = _provider_score_decomposition(
            action=action,
            p_up=p_up,
            p_down=1.0 - p_up,
            up=up,
            down=down,
        )
        ranking.append(
            {
                "selected_action": action,
                "selected_side": side,
                "selected_action_family": _action_family(action),
                "corrected_model_score": score,
                "raw_model_score": score,
                "score_decomposition": decomposition,
                "scoring_rule_id": "fresh_provider_simplified_score",
                "canonical_frozen_o_scorer_used": False,
                "p_up_action_disagreement": _p_up_action_disagreement(
                    action=action,
                    p_up=p_up,
                ),
                "microstructure_snapshot": _provider_microstructure_for_action(
                    action=action,
                    up=up,
                    down=down,
                    market=market,
                    decision_ts=decision_ts,
                ),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            _float(row.get("corrected_model_score")),
            1 if row.get("selected_action") != "NO_TRADE" else 0,
            str(row.get("selected_action")),
        ),
        reverse=True,
    )


def _provider_score_decomposition(
    *,
    action: str,
    p_up: float,
    p_down: float,
    up: dict[str, Any],
    down: dict[str, Any],
) -> dict[str, Any]:
    if action == "NO_TRADE":
        return {
            "scoring_rule_id": "fresh_provider_simplified_score",
            "canonical_frozen_o_scorer_used": False,
            "p_side_contribution": 0.0,
            "p_up_contribution": 0.0,
            "p_down_contribution": 0.0,
            "ask_contribution": 0.0,
            "bid_contribution": 0.0,
            "spread_penalty": 0.0,
            "queue_fill_term": 0.0,
            "book_staleness_term": 0.0,
            "time_to_close_term": 0.0,
            "queue_fill_term_used": False,
            "book_staleness_term_used": False,
            "time_to_close_term_used": False,
            "raw_score": 0.0,
            "corrected_score": 0.0,
            "is_buy_action": False,
            "is_sell_before_close": False,
            "is_hold_to_settlement": False,
            "is_no_trade": True,
        }
    side = _side_from_action(action)
    book = up if side == "UP" else down
    p_side = p_up if side == "UP" else p_down
    ask = _float(book.get("ask_price"))
    bid = _float(book.get("bid_price"))
    spread_penalty = _provider_spread_bps(book) / 10_000.0
    family = _action_family(action)
    p_side_contribution = p_side if family == "HOLD_TO_SETTLEMENT" else 0.0
    bid_contribution = bid if family == "SELL_BEFORE_CLOSE" else 0.0
    ask_contribution = -ask
    corrected = p_side_contribution + bid_contribution + ask_contribution - spread_penalty
    return {
        "scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_frozen_o_scorer_used": False,
        "p_side_contribution": p_side_contribution,
        "p_up_contribution": (
            p_up if side == "UP" and family == "HOLD_TO_SETTLEMENT" else 0.0
        ),
        "p_down_contribution": p_down
        if side == "DOWN" and family == "HOLD_TO_SETTLEMENT"
        else 0.0,
        "ask_contribution": ask_contribution,
        "bid_contribution": bid_contribution,
        "spread_penalty": spread_penalty,
        "queue_fill_term": 0.0,
        "book_staleness_term": 0.0,
        "time_to_close_term": 0.0,
        "queue_fill_term_used": False,
        "book_staleness_term_used": False,
        "time_to_close_term_used": False,
        "raw_score": corrected,
        "corrected_score": corrected,
        "is_buy_action": True,
        "is_sell_before_close": family == "SELL_BEFORE_CLOSE",
        "is_hold_to_settlement": family == "HOLD_TO_SETTLEMENT",
        "is_no_trade": False,
    }


def _provider_microstructure_for_action(
    *,
    action: str,
    up: dict[str, Any],
    down: dict[str, Any],
    market: dict[str, Any],
    decision_ts: int,
) -> dict[str, Any]:
    if action == "NO_TRADE":
        return {}
    book = up if _side_from_action(action) == "UP" else down
    return {
        "entry_ask": _float(book.get("ask_price")),
        "executable_exit_bid_proxy": _float(book.get("bid_price")),
        "spread_bps": _provider_spread_bps(book),
        "book_staleness_ms": max(0, decision_ts - _book_available_at(book)),
        "queue_fill_proxy": _provider_queue_fill_proxy(book),
        "time_to_close_seconds": max(
            0.0,
            (int(market.get("market_end_ts") or decision_ts) - decision_ts)
            / 1000.0,
        ),
    }


def _provider_reference_price_provenance(
    *,
    market: dict[str, Any],
    candle: dict[str, Any],
    decision_ts: int,
) -> dict[str, Any]:
    reference_ts = int(market.get("market_start_ts") or 0)
    candle_ts = int(candle.get("available_at_ts") or candle.get("ts") or 0)
    max_input_ts = max(reference_ts, candle_ts)
    return {
        "provenance_valid": max_input_ts <= decision_ts,
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "source_fields_used": [
            "raw_polymarket_markets.reference_price_start",
            "raw_btc_feature_candles.close_price",
        ],
        "source_field_name": "read_only_public_provider_reference_and_btc_candle",
        "source_timestamp": max_input_ts,
    }


def _provider_decision_time_regime_features(
    *,
    market: dict[str, Any],
    candle: dict[str, Any],
    reference_candle: dict[str, Any] | None,
    chainlink_rtds_prices: list[dict[str, Any]],
    decision_ts: int,
    ranking: list[dict[str, Any]],
    selected_action: str,
) -> dict[str, Any]:
    btc_momentum, btc_provenance = _provider_btc_momentum_feature(
        candle=candle,
        decision_ts=decision_ts,
    )
    chainlink_features = _provider_chainlink_regime_features(
        rows=chainlink_rtds_prices,
        market_start_ts=int(market.get("market_start_ts") or 0),
        decision_ts=decision_ts,
        comparison_btc_price=_float(candle.get("close_price")),
    )
    (
        reference_price_to_beat,
        reference_distance,
        reference_provenance,
    ) = _provider_reference_distance_feature(
        market=market,
        candle=candle,
        reference_candle=reference_candle,
        chainlink_features=chainlink_features,
        decision_ts=decision_ts,
    )
    time_since_start, time_provenance = _provider_time_since_market_start_feature(
        market=market,
        decision_ts=decision_ts,
    )
    margin_features = _provider_action_score_margin_features(
        ranking=ranking,
        selected_action=selected_action,
        decision_ts=decision_ts,
    )
    required_provenances = [
        btc_provenance,
        reference_provenance,
        time_provenance,
        margin_features["action_score_margin_provenance"],
    ]
    optional_provenances = [
        margin_features["side_specific_action_score_margin_provenance"],
    ]
    provenances = [*required_provenances, *optional_provenances]
    valid_provenances = [
        provenance for provenance in provenances if provenance.get("provenance_valid")
    ]
    max_input_ts = max(
        [int(provenance.get("max_input_ts") or 0) for provenance in valid_provenances]
        or [decision_ts]
    )
    return {
        "btc_momentum": btc_momentum,
        "btc_momentum_provenance": btc_provenance,
        "reference_price_to_beat_at_decision": reference_price_to_beat,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "reference_price_to_beat_distance_provenance": reference_provenance,
        **chainlink_features,
        "time_since_market_start_seconds": time_since_start,
        "time_since_market_start_provenance": time_provenance,
        "action_score_margin": margin_features["action_score_margin"],
        "action_score_margin_provenance": margin_features[
            "action_score_margin_provenance"
        ],
        "side_specific_action_score_margin": margin_features[
            "side_specific_action_score_margin"
        ],
        "side_specific_action_score_margin_provenance": margin_features[
            "side_specific_action_score_margin_provenance"
        ],
        "decision_time_regime_feature_provenance": {
            "provenance_valid": all(
                bool(provenance.get("provenance_valid"))
                for provenance in required_provenances
            ),
            "decision_ts": decision_ts,
            "max_input_ts": max_input_ts,
            "source_fields_used": sorted(
                {
                    str(source)
                    for provenance in provenances
                    for source in provenance.get("source_fields_used", [])
                }
            ),
            "source_field_name": "decision_time_hts_regime_feature_block_v1",
            "source_timestamp": max_input_ts,
            "unavailable_reason_codes": sorted(
                {
                    str(reason)
                    for provenance in provenances
                    for reason in provenance.get("unavailable_reason_codes", [])
                }
            ),
        },
        "decision_time_regime_feature_max_input_ts": max_input_ts,
    }


def _provider_btc_momentum_feature(
    *,
    candle: dict[str, Any],
    decision_ts: int,
) -> tuple[float | None, dict[str, Any]]:
    available_at = int(candle.get("available_at_ts") or candle.get("ts") or 0)
    open_price = _float(candle.get("open_price"))
    close_price = _float(candle.get("close_price"))
    reason_codes: list[str] = []
    if available_at > decision_ts:
        reason_codes.append("btc_candle_not_decision_time_available")
    if open_price <= 0.0:
        reason_codes.append("btc_candle_open_price_missing_or_non_positive")
    if close_price <= 0.0:
        reason_codes.append("btc_candle_close_price_missing_or_non_positive")
    value = (
        (close_price - open_price) / open_price
        if not reason_codes
        else None
    )
    return value, {
        "provenance_valid": not reason_codes,
        "decision_ts": decision_ts,
        "max_input_ts": available_at,
        "source_fields_used": [
            "raw_btc_feature_candles.open_price",
            "raw_btc_feature_candles.close_price",
            "raw_btc_feature_candles.available_at_ts",
        ],
        "source_field_name": "read_only_public_provider_btc_candle_momentum",
        "source_timestamp": available_at,
        "unavailable_reason_codes": reason_codes,
    }


def _provider_reference_distance_feature(
    *,
    market: dict[str, Any],
    candle: dict[str, Any],
    reference_candle: dict[str, Any] | None,
    chainlink_features: dict[str, Any],
    decision_ts: int,
) -> tuple[float | None, float | None, dict[str, Any]]:
    reference = _float(
        market.get("reference_price_start")
        if market.get("reference_price_start") is not None
        else market.get("reference_price_at_start")
    )
    reference_source_type = "polymarket_market_metadata_price_to_beat"
    reference_source_timestamp = decision_ts if reference > 0.0 else 0
    reference_warning_reason_codes: list[str] = []
    source_fields_used = [
        "raw_polymarket_markets.reference_price_start",
        "raw_polymarket_markets.reference_price_at_start",
    ]
    current_price = _float(candle.get("close_price"))
    current_source_timestamp = int(
        candle.get("available_at_ts") or candle.get("ts") or 0
    )
    chainlink_reference = _float(
        chainlink_features.get("chainlink_reference_price_at_market_start")
    )
    chainlink_current = _float(
        chainlink_features.get("chainlink_price_at_decision")
    )
    chainlink_provenance = dict(
        chainlink_features.get("chainlink_regime_feature_provenance") or {}
    )
    if reference <= 0.0 and chainlink_reference > 0.0 and chainlink_current > 0.0:
        reference = chainlink_reference
        current_price = chainlink_current
        reference_source_timestamp = int(
            chainlink_provenance.get("max_input_ts") or 0
        )
        current_source_timestamp = reference_source_timestamp
        reference_source_type = "polymarket_rtds_chainlink_market_start_proxy"
        source_fields_used.extend(
            [
                "raw_polymarket_chainlink_prices.price",
                "raw_polymarket_chainlink_prices.source_ts",
                "raw_polymarket_chainlink_prices.available_at_ts",
            ]
        )
        reference_warning_reason_codes.append(
            "official_price_to_beat_metadata_unavailable_chainlink_rtds_market_start_proxy_used"
        )
    if reference <= 0.0 and reference_candle is not None:
        reference = _float(reference_candle.get("close_price"))
        reference_source_timestamp = int(
            reference_candle.get("available_at_ts")
            or reference_candle.get("close_time")
            or reference_candle.get("ts")
            or 0
        )
        reference_source_type = "btc_feature_candle_market_start_proxy"
        source_fields_used.extend(
            [
                "raw_btc_feature_candles.close_price",
                "raw_btc_feature_candles.available_at_ts",
                "raw_btc_feature_candles.close_time",
            ]
        )
        reference_warning_reason_codes.append(
            "official_polymarket_price_to_beat_unavailable_btc_feature_candle_proxy_used"
        )
    close_price = current_price
    candle_ts = current_source_timestamp
    reason_codes: list[str] = []
    if candle_ts > decision_ts:
        reason_codes.append("btc_candle_not_decision_time_available")
    if reference <= 0.0:
        reason_codes.append("reference_price_to_beat_missing_or_non_positive")
    if reference_source_timestamp > decision_ts:
        reason_codes.append("reference_price_to_beat_not_decision_time_available")
    if close_price <= 0.0:
        reason_codes.append("btc_candle_close_price_missing_or_non_positive")
    value = (close_price - reference) / reference if not reason_codes else None
    max_input_ts = max(candle_ts, reference_source_timestamp)
    return reference if not reason_codes else None, value, {
        "provenance_valid": not reason_codes and max_input_ts <= decision_ts,
        "decision_ts": decision_ts,
        "max_input_ts": max_input_ts,
        "source_fields_used": source_fields_used,
        "source_field_name": "read_only_public_provider_reference_distance",
        "reference_price_to_beat_source_type": reference_source_type,
        "reference_price_to_beat_at_decision": reference if reference > 0.0 else None,
        "source_timestamp": max_input_ts,
        "unavailable_reason_codes": reason_codes,
        "warning_reason_codes": reference_warning_reason_codes,
    }


def _provider_chainlink_regime_features(
    *,
    rows: list[dict[str, Any]],
    market_start_ts: int,
    decision_ts: int,
    comparison_btc_price: float,
) -> dict[str, Any]:
    causal_rows = [
        dict(row)
        for row in rows
        if int(row.get("available_at_ts") or 0) <= decision_ts
        and int(row.get("source_ts") or 0) <= int(row.get("available_at_ts") or 0)
        and _float(row.get("price")) > 0.0
    ]
    causal_rows.sort(
        key=lambda row: (
            int(row.get("source_ts") or 0),
            int(row.get("available_at_ts") or 0),
        )
    )
    current = causal_rows[-1] if causal_rows else None
    reference_candidates = [
        row
        for row in causal_rows
        if market_start_ts > 0 and int(row.get("source_ts") or 0) <= market_start_ts
    ]
    reference = reference_candidates[-1] if reference_candidates else None
    unavailable_reason_codes: list[str] = []
    if current is None:
        unavailable_reason_codes.append("chainlink_rtds_current_price_unavailable")
    if reference is None:
        unavailable_reason_codes.append(
            "chainlink_rtds_market_start_reference_unavailable"
        )
    current_price = _float((current or {}).get("price")) or None
    reference_price = _float((reference or {}).get("price")) or None
    distance = (
        (float(current_price) - float(reference_price)) / float(reference_price)
        if current_price is not None and reference_price is not None
        else None
    )
    momentum_by_horizon: dict[str, float | None] = {}
    if current is None:
        momentum_by_horizon = {
            "30s": None,
            "60s": None,
            "120s": None,
        }
    else:
        current_source_ts = int(current["source_ts"])
        for horizon_seconds in (30, 60, 120):
            baseline = _latest_chainlink_source_row_at_or_before(
                causal_rows,
                source_ts=current_source_ts - horizon_seconds * 1000,
            )
            baseline_price = _float((baseline or {}).get("price"))
            momentum_by_horizon[f"{horizon_seconds}s"] = (
                (float(current_price) - baseline_price) / baseline_price
                if current_price is not None and baseline_price > 0.0
                else None
            )
    recent_rows = []
    if current is not None:
        lower_bound = int(current["source_ts"]) - 120_000
        recent_rows = [row for row in causal_rows if int(row["source_ts"]) >= lower_bound]
    returns = [
        (_float(right.get("price")) - _float(left.get("price")))
        / _float(left.get("price"))
        for left, right in zip(recent_rows, recent_rows[1:], strict=False)
        if _float(left.get("price")) > 0.0
    ]
    realized_volatility = (
        (sum(value * value for value in returns) / len(returns)) ** 0.5
        if returns
        else None
    )
    max_input_ts = max(
        (
            int(row.get("available_at_ts") or 0)
            for row in (current, reference)
            if row is not None
        ),
        default=0,
    )
    provenance_valid = (
        not unavailable_reason_codes
        and max_input_ts > 0
        and max_input_ts <= decision_ts
    )
    return {
        "chainlink_price_at_decision": current_price,
        "chainlink_reference_price_at_market_start": reference_price,
        "chainlink_reference_distance_at_decision": distance,
        "chainlink_momentum_30s": momentum_by_horizon["30s"],
        "chainlink_momentum_60s": momentum_by_horizon["60s"],
        "chainlink_momentum_120s": momentum_by_horizon["120s"],
        "chainlink_realized_volatility_120s": realized_volatility,
        "chainlink_vs_btc_feature_price_gap": (
            (float(current_price) - comparison_btc_price) / comparison_btc_price
            if current_price is not None and comparison_btc_price > 0.0
            else None
        ),
        "chainlink_regime_feature_provenance": {
            "provenance_valid": provenance_valid,
            "decision_ts": decision_ts,
            "max_input_ts": max_input_ts,
            "current_source_ts": (current or {}).get("source_ts"),
            "reference_source_ts": (reference or {}).get("source_ts"),
            "source_fields_used": [
                "raw_polymarket_chainlink_prices.price",
                "raw_polymarket_chainlink_prices.source_ts",
                "raw_polymarket_chainlink_prices.available_at_ts",
            ],
            "source_field_name": "polymarket_rtds_chainlink_btc_usd",
            "source_timestamp": max_input_ts,
            "unavailable_reason_codes": unavailable_reason_codes,
            "warning_reason_codes": [
                "market_start_value_is_same_source_decision_time_proxy_not_embedded_market_price_to_beat"
            ]
            if reference is not None
            else [],
        },
    }


def _latest_chainlink_source_row_at_or_before(
    rows: list[dict[str, Any]],
    *,
    source_ts: int,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for row in rows:
        if int(row.get("source_ts") or 0) <= source_ts:
            latest = row
        else:
            break
    return latest


def _provider_time_since_market_start_feature(
    *,
    market: dict[str, Any],
    decision_ts: int,
) -> tuple[float | None, dict[str, Any]]:
    market_start_ts = int(market.get("market_start_ts") or 0)
    reason_codes: list[str] = []
    if market_start_ts <= 0:
        reason_codes.append("market_start_ts_missing")
    value = (
        (decision_ts - market_start_ts) / 1000.0
        if not reason_codes
        else None
    )
    # Market schedule metadata is read at decision time from the public market row.
    source_ts = decision_ts if not reason_codes else 0
    return value, {
        "provenance_valid": not reason_codes and source_ts <= decision_ts,
        "decision_ts": decision_ts,
        "max_input_ts": source_ts,
        "source_fields_used": ["raw_polymarket_markets.market_start_ts"],
        "source_field_name": "read_only_public_provider_market_schedule",
        "source_timestamp": source_ts,
        "unavailable_reason_codes": reason_codes,
    }


def _provider_action_score_margin_features(
    *,
    ranking: list[dict[str, Any]],
    selected_action: str,
    decision_ts: int,
) -> dict[str, Any]:
    selected = _ranking_action(ranking, selected_action)
    selected_score = (
        _float(selected.get("corrected_model_score")) if selected else None
    )
    other_scores = [
        _float(row.get("corrected_model_score"))
        for row in ranking
        if row.get("selected_action") != selected_action
    ]
    action_margin = (
        selected_score - max(other_scores)
        if selected_score is not None and other_scores
        else None
    )
    selected_side = _side_from_action(selected_action)
    opposite_side = (
        "DOWN" if selected_side == "UP" else "UP" if selected_side == "DOWN" else ""
    )
    opposite_scores = [
        _float(row.get("corrected_model_score"))
        for row in ranking
        if _side_from_action(str(row.get("selected_action") or "")) == opposite_side
    ]
    side_margin = (
        selected_score - max(opposite_scores)
        if selected_score is not None and opposite_scores
        else None
    )
    common_provenance = {
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "source_fields_used": [
            "full_5_action_ranking.selected_action",
            "full_5_action_ranking.corrected_model_score",
        ],
        "source_timestamp": decision_ts,
    }
    return {
        "action_score_margin": action_margin,
        "action_score_margin_provenance": {
            **common_provenance,
            "provenance_valid": action_margin is not None,
            "source_field_name": "full_5_action_ranking_top_action_margin",
            "unavailable_reason_codes": []
            if action_margin is not None
            else ["full_5_action_ranking_missing_second_action"],
        },
        "side_specific_action_score_margin": side_margin,
        "side_specific_action_score_margin_provenance": {
            **common_provenance,
            "provenance_valid": side_margin is not None,
            "source_field_name": "full_5_action_ranking_side_margin",
            "unavailable_reason_codes": []
            if side_margin is not None
            else ["full_5_action_ranking_missing_opposite_side_action"],
        },
    }


def _partition_public_rows(
    rows: list[dict[str, Any]],
    max_cycles: int,
) -> list[list[dict[str, Any]]]:
    cycles: list[list[dict[str, Any]]] = [[] for _ in range(max_cycles)]
    for index, row in enumerate(rows):
        cycles[index % max_cycles].append(row)
    return cycles


def _public_p_up(*, up: dict[str, Any], down: dict[str, Any]) -> float:
    up_mid = _float(up.get("mid_price"))
    down_mid = _float(down.get("mid_price"))
    total = up_mid + down_mid
    if total <= 0.0:
        return 0.5
    return max(0.0, min(1.0, up_mid / total))


def _provider_spread_bps(book: dict[str, Any]) -> float:
    bid = _float(book.get("bid_price"))
    ask = _float(book.get("ask_price"))
    mid = _float(book.get("mid_price")) or (bid + ask) / 2.0
    if mid <= 0.0:
        return 10_000.0
    return max(0.0, (ask - bid) / mid * 10_000.0)


def _provider_queue_fill_proxy(book: dict[str, Any]) -> float:
    bid_notional = _float(book.get("bid_price")) * _float(book.get("bid_size"))
    ask_notional = _float(book.get("ask_price")) * _float(book.get("ask_size"))
    depth_score = min(1.0, _float(book.get("liquidity_depth")) / 2.0)
    notional_score = min(1.0, max(bid_notional, ask_notional))
    spread_score = max(0.0, 1.0 - _provider_spread_bps(book) / 2_000.0)
    return max(
        0.0,
        min(1.0, 0.55 * notional_score + 0.35 * depth_score + 0.10 * spread_score),
    )


def _book_available_at(book: dict[str, Any]) -> int:
    return int(book.get("available_at_ts") or book.get("ts") or 0)


def _p_up_action_disagreement(*, action: str, p_up: float) -> bool:
    side = _side_from_action(action)
    if side == "UP":
        return p_up < 0.50
    if side == "DOWN":
        return p_up > 0.50
    return False


def _execute_fresh_public_cycles(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_source: str,
    unlock_verified: bool,
    ev_calibration_artifact: dict[str, Any],
) -> dict[str, Any]:
    runtime_state = _initial_fresh_runtime_state()
    guard_config = _v8_execution_guard_config()
    all_guard_rows: list[dict[str, Any]] = []
    all_intents: list[dict[str, Any]] = []
    all_remap_rows: list[dict[str, Any]] = []
    cycle_reports: list[dict[str, Any]] = []
    cycle_failure_count = 0

    for cycle_index in range(config.max_cycles):
        cycle_rows = list(public_cycles[cycle_index] if cycle_index < len(public_cycles) else [])
        cycle_id = f"{config.run_id}-cycle-{cycle_index + 1:06d}"
        cycle_guard_rows: list[dict[str, Any]] = []
        cycle_intents: list[dict[str, Any]] = []
        cycle_forbidden_rows = _rows_with_forbidden_fields(cycle_rows)
        cycle_failed = not unlock_verified or bool(cycle_forbidden_rows)
        if not cycle_failed:
            for row_index, public_row in enumerate(cycle_rows):
                guard_input = _guard_input_from_public_row(
                    public_row=public_row,
                    cycle_id=cycle_id,
                    row_index=row_index,
                )
                pre_state = _compact_runtime_state(runtime_state)
                guard_row = _v8_execution_guard_decision(
                    guard_input,
                    guard_config=guard_config,
                    runtime_state=runtime_state,
                    runtime_mode="simulated_runtime_state",
                )
                guard_row["cycle_id"] = cycle_id
                guard_row["public_data_source"] = public_data_source
                guard_row["pre_decision_exposure_state"] = pre_state
                guard_row, remap_row = _paper_hts_time_window_remap_guard_row(
                    original_guard_row=guard_row,
                    guard_input=guard_input,
                    guard_config=guard_config,
                    runtime_state=runtime_state,
                    runtime_mode="simulated_runtime_state",
                    cycle_id=cycle_id,
                    public_data_source=public_data_source,
                    pre_state=pre_state,
                    ev_calibration_artifact=ev_calibration_artifact,
                )
                if remap_row is not None:
                    all_remap_rows.append(remap_row)
                if guard_row.get("order_allowed") is True:
                    guard_row["simulated_order_id"] = (
                        f"{config.run_id}-fresh-sim-{len(all_intents) + 1:06d}"
                    )
                    _apply_guard_row_to_runtime_state(runtime_state, guard_row)
                else:
                    runtime_state["blocked_simulated_order_count"] = int(
                        runtime_state["blocked_simulated_order_count"]
                    ) + 1
                    guard_row["simulated_order_id"] = None
                guard_row["post_decision_exposure_state"] = _compact_runtime_state(
                    runtime_state
                )
                cycle_guard_rows.append(guard_row)
                if guard_row.get("order_allowed") is True:
                    intent = _fresh_order_intent_from_guard_row(
                        config=config,
                        cycle_id=cycle_id,
                        guard_row=guard_row,
                        intent_index=len(all_intents) + 1,
                    )
                    cycle_intents.append(intent)
                    all_intents.append(intent)
        else:
            cycle_failure_count += 1
        cycle_reports.append(
            _cycle_monitoring_row(
                cycle_id=cycle_id,
                cycle_index=cycle_index,
                public_rows=cycle_rows,
                guard_rows=cycle_guard_rows,
                intents=cycle_intents,
                cycle_failed=cycle_failed,
                cycle_forbidden_rows=cycle_forbidden_rows,
                public_data_source=public_data_source,
                runtime_state=runtime_state,
            )
        )
        all_guard_rows.extend(cycle_guard_rows)
        if cycle_index < config.max_cycles - 1 and config.sleep_seconds > 0.0:
            time.sleep(config.sleep_seconds)

    return {
        "guard_config": guard_config,
        "cycle_monitoring_rows": cycle_reports,
        "guard_decision_rows": all_guard_rows,
        "paper_order_intents": all_intents,
        "paper_remap_rows": all_remap_rows,
        "final_runtime_state": _compact_runtime_state(runtime_state),
        "cycle_failure_count": cycle_failure_count,
    }


def _paper_hts_time_window_remap_guard_row(
    *,
    original_guard_row: dict[str, Any],
    guard_input: dict[str, Any],
    guard_config: dict[str, Any],
    runtime_state: dict[str, Any],
    runtime_mode: str,
    cycle_id: str,
    public_data_source: str,
    pre_state: dict[str, Any],
    ev_calibration_artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not _paper_hts_time_window_remap_applicable(
        original_guard_row=original_guard_row,
        guard_config=guard_config,
    ):
        original_guard_row.setdefault("hts_time_window_remap_applied", False)
        original_guard_row.setdefault("remap_reason_codes", [])
        return original_guard_row, None

    original_action = str(original_guard_row.get("source_selected_action") or "")
    side = str(original_guard_row.get("source_selected_side") or _side_from_action(original_action))
    remapped_action = f"BUY_{side}_SELL_BEFORE_CLOSE"
    reason_codes: list[str] = ["hts_time_window_blocked_original_action"]
    candidate = _paper_same_side_sbc_candidate(
        full_ranking=list(guard_input.get("full_5_action_ranking") or []),
        remapped_action=remapped_action,
    )
    if candidate is None:
        return original_guard_row, _paper_remap_row(
            original_guard_row=original_guard_row,
            remapped_guard_row=None,
            remapped_action=remapped_action,
            applied=False,
            candidate=False,
            calibrated_ev=None,
            calibrated_ev_source="missing_same_side_sbc_candidate",
            reason_codes=[*reason_codes, "same_side_sbc_alternative_missing"],
        )
    reason_codes.append("same_side_sbc_alternative_available")

    calibrated_ev, ev_source = _paper_remap_calibrated_ev(
        guard_input=guard_input,
        candidate=candidate,
        side=side,
        ev_calibration_artifact=ev_calibration_artifact,
    )
    if calibrated_ev is None:
        return original_guard_row, _paper_remap_row(
            original_guard_row=original_guard_row,
            remapped_guard_row=None,
            remapped_action=remapped_action,
            applied=False,
            candidate=False,
            calibrated_ev=None,
            calibrated_ev_source=ev_source,
            reason_codes=[*reason_codes, "same_side_sbc_calibrated_ev_missing"],
        )
    if calibrated_ev < O_V8_PAPER_FRESH_HTS_REMAP_EV_THRESHOLD:
        return original_guard_row, _paper_remap_row(
            original_guard_row=original_guard_row,
            remapped_guard_row=None,
            remapped_action=remapped_action,
            applied=False,
            candidate=False,
            calibrated_ev=calibrated_ev,
            calibrated_ev_source=ev_source,
            reason_codes=[
                *reason_codes,
                "same_side_sbc_calibrated_ev_below_threshold",
            ],
        )

    remap_input = _paper_remap_guard_input(
        guard_input=guard_input,
        candidate=candidate,
        remapped_action=remapped_action,
        side=side,
    )
    remapped_guard_row = _v8_execution_guard_decision(
        remap_input,
        guard_config=guard_config,
        runtime_state=runtime_state,
        runtime_mode=runtime_mode,
    )
    remapped_guard_row["cycle_id"] = cycle_id
    remapped_guard_row["public_data_source"] = public_data_source
    remapped_guard_row["pre_decision_exposure_state"] = pre_state
    remapped_guard_row["original_action"] = original_action
    remapped_guard_row["original_family"] = original_guard_row.get(
        "source_selected_family"
    )
    remapped_guard_row["original_side"] = side
    remapped_guard_row["remapped_action"] = remapped_action
    remapped_guard_row["remapped_family"] = "SELL_BEFORE_CLOSE"
    remapped_guard_row["remapped_side"] = side
    remapped_guard_row["source_selected_action"] = original_action
    remapped_guard_row["source_selected_family"] = original_guard_row.get(
        "source_selected_family"
    )
    remapped_guard_row["source_selected_side"] = side
    remapped_guard_row["source_model_score"] = original_guard_row.get(
        "source_model_score"
    )
    remapped_guard_row["source_raw_model_score"] = original_guard_row.get(
        "source_raw_model_score"
    )
    remapped_guard_row["original_execution_blocking_reason_codes"] = list(
        original_guard_row.get("execution_blocking_reason_codes") or []
    )
    remapped_guard_row["original_execution_guard_reason_codes"] = list(
        original_guard_row.get("execution_guard_reason_codes") or []
    )
    remapped_guard_row["original_order_allowed"] = bool(
        original_guard_row.get("order_allowed")
    )
    remapped_guard_row["hts_time_window_remap_calibrated_ev"] = calibrated_ev
    remapped_guard_row["hts_time_window_remap_calibrated_ev_source"] = ev_source
    remapped_guard_row["hts_time_window_remap_ev_threshold"] = (
        O_V8_PAPER_FRESH_HTS_REMAP_EV_THRESHOLD
    )
    remapped_guard_row["hts_time_window_remap_applied"] = (
        remapped_guard_row.get("order_allowed") is True
    )
    remap_reasons = [
        *reason_codes,
        "same_side_sbc_calibrated_ev_threshold_passed",
    ]
    if remapped_guard_row.get("order_allowed") is True:
        remap_reasons.append("same_side_sbc_guard_passed")
    else:
        remap_reasons.extend(
            list(remapped_guard_row.get("execution_blocking_reason_codes") or [])
            or ["same_side_sbc_guard_blocked"]
        )
    remapped_guard_row["remap_reason_codes"] = sorted(set(remap_reasons))

    return (
        remapped_guard_row if remapped_guard_row.get("order_allowed") is True else original_guard_row,
        _paper_remap_row(
            original_guard_row=original_guard_row,
            remapped_guard_row=remapped_guard_row,
            remapped_action=remapped_action,
            applied=remapped_guard_row.get("order_allowed") is True,
            candidate=True,
            calibrated_ev=calibrated_ev,
            calibrated_ev_source=ev_source,
            reason_codes=remapped_guard_row["remap_reason_codes"],
        ),
    )


def _paper_hts_time_window_remap_applicable(
    *,
    original_guard_row: dict[str, Any],
    guard_config: dict[str, Any],
) -> bool:
    action = str(original_guard_row.get("source_selected_action") or "")
    if action not in {"BUY_UP_HOLD_TO_SETTLEMENT", "BUY_DOWN_HOLD_TO_SETTLEMENT"}:
        return False
    if original_guard_row.get("order_allowed") is True:
        return False
    blocking = set(original_guard_row.get("execution_blocking_reason_codes") or [])
    if blocking != {"execution_time_to_close_unsafe"}:
        return False
    micro = dict(original_guard_row.get("microstructure_snapshot") or {})
    time_to_close = _trace_float_or_none(micro.get("time_to_close_seconds"))
    return bool(
        time_to_close is not None
        and time_to_close >= float(guard_config["min_time_to_close_seconds"])
        and time_to_close < float(guard_config["min_hts_time_to_close_seconds"])
    )


def _paper_same_side_sbc_candidate(
    *,
    full_ranking: list[dict[str, Any]],
    remapped_action: str,
) -> dict[str, Any] | None:
    for row in full_ranking:
        if str(row.get("selected_action") or "") == remapped_action:
            return dict(row)
    return None


def _paper_remap_calibrated_ev(
    *,
    guard_input: dict[str, Any],
    candidate: dict[str, Any],
    side: str,
    ev_calibration_artifact: dict[str, Any],
) -> tuple[float | None, str]:
    explicit = candidate.get(
        "calibrated_action_expected_net_return",
        guard_input.get("calibrated_action_expected_net_return"),
    )
    if explicit is not None:
        return _float(explicit), "input_calibrated_action_expected_net_return"
    micro = {
        **dict(guard_input.get("microstructure_snapshot") or {}),
        **dict(candidate.get("microstructure_snapshot") or {}),
    }
    if ev_calibration_artifact.get("path") is not None:
        source_score = candidate.get(
            "corrected_model_score",
            guard_input.get("corrected_model_score"),
        )
        expected_return, _, ev_source, _, reasons = _calibrated_expected_return_source(
            input_expected_return=None,
            input_expected_return_field=None,
            canonical_score=_trace_float_or_none(source_score),
            canonical_score_field="same_side_sbc.corrected_model_score",
            canonical_raw_score=_trace_float_or_none(
                candidate.get("raw_model_score", guard_input.get("raw_model_score"))
            ),
            canonical_raw_score_field="same_side_sbc.raw_model_score",
            execution_price=_trace_float_or_none(micro.get("entry_ask")),
            execution_price_field="microstructure_snapshot.entry_ask",
            executable_exit_bid_proxy=_trace_float_or_none(
                micro.get("executable_exit_bid_proxy")
            ),
            spread_bps=_trace_float_or_none(micro.get("spread_bps")),
            queue_fill_proxy=_trace_float_or_none(micro.get("queue_fill_proxy")),
            book_staleness_ms=_trace_float_or_none(micro.get("book_staleness_ms")),
            time_to_close=_trace_float_or_none(micro.get("time_to_close_seconds")),
            family="SELL_BEFORE_CLOSE",
            side=side,
            execution_cost=O_V8_PAPER_FRESH_HTS_REMAP_EXECUTION_COST,
            ev_calibration_artifact=ev_calibration_artifact,
        )
        if expected_return is None:
            reason_suffix = ",".join(reasons) if reasons else "unknown"
            return None, f"{ev_source}:{reason_suffix}"
        return expected_return, ev_source
    source_score = guard_input.get("corrected_model_score")
    if source_score is None:
        return None, "missing_source_model_score_for_frozen_ev_mapping"
    return (
        _float(source_score)
        * O_V8_PAPER_FRESH_HTS_REMAP_SCORE_TO_EXPECTED_NET_RETURN_WEIGHT
        - O_V8_PAPER_FRESH_HTS_REMAP_EXECUTION_COST,
        "paper_fresh_frozen_score_to_expected_net_return_v1",
    )


def _paper_remap_guard_input(
    *,
    guard_input: dict[str, Any],
    candidate: dict[str, Any],
    remapped_action: str,
    side: str,
) -> dict[str, Any]:
    micro = {
        **dict(guard_input.get("microstructure_snapshot") or {}),
        **dict(candidate.get("microstructure_snapshot") or {}),
    }
    return {
        **dict(guard_input),
        "selected_action": remapped_action,
        "selected_side": side,
        "selected_action_family": "SELL_BEFORE_CLOSE",
        "microstructure_snapshot": micro,
        "p_up_action_disagreement": _p_up_action_disagreement(
            action=remapped_action,
            p_up=_float(guard_input.get("p_up")),
        ),
    }


def _paper_remap_row(
    *,
    original_guard_row: dict[str, Any],
    remapped_guard_row: dict[str, Any] | None,
    remapped_action: str,
    applied: bool,
    candidate: bool,
    calibrated_ev: float | None,
    calibrated_ev_source: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "decision_group_id": original_guard_row.get("decision_group_id"),
        "market_id": original_guard_row.get("market_id"),
        "decision_ts": original_guard_row.get("decision_ts"),
        "original_action": original_guard_row.get("source_selected_action"),
        "original_family": original_guard_row.get("source_selected_family"),
        "original_side": original_guard_row.get("source_selected_side"),
        "remapped_action": remapped_action,
        "remapped_family": "SELL_BEFORE_CLOSE",
        "remapped_side": _side_from_action(remapped_action),
        "hts_time_window_remap_applied": applied,
        "remap_candidate": candidate,
        "calibrated_ev": calibrated_ev,
        "calibrated_ev_source": calibrated_ev_source,
        "calibrated_ev_threshold": O_V8_PAPER_FRESH_HTS_REMAP_EV_THRESHOLD,
        "original_order_allowed": bool(original_guard_row.get("order_allowed")),
        "remapped_order_allowed": bool(
            remapped_guard_row and remapped_guard_row.get("order_allowed") is True
        ),
        "original_execution_blocking_reason_codes": list(
            original_guard_row.get("execution_blocking_reason_codes") or []
        ),
        "original_execution_guard_reason_codes": list(
            original_guard_row.get("execution_guard_reason_codes") or []
        ),
        "remapped_execution_blocking_reason_codes": list(
            (remapped_guard_row or {}).get("execution_blocking_reason_codes") or []
        ),
        "remapped_execution_guard_reason_codes": list(
            (remapped_guard_row or {}).get("execution_guard_reason_codes") or []
        ),
        "remap_reason_codes": sorted(set(reason_codes)),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "uses_settlement_pnl_or_outcome_labels": False,
        "uses_oracle_actions_or_future_returns": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
    }


def _execution_layer_v2_paper_remap_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(execution_result.get("paper_remap_rows") or [])
    applied_rows = [
        row for row in rows if row.get("hts_time_window_remap_applied") is True
    ]
    intent_remaps = [
        intent for intent in intents if intent.get("hts_time_window_remap_applied") is True
    ]
    report = {
        "schema_version": EXECUTION_LAYER_V2_PAPER_REMAP_SCHEMA_VERSION,
        "report_type": "execution_layer_v2_paper_remap",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "execution_layer_v2_paper_remap_enabled": True,
        "paper_only_intent_path": True,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        "hts_time_window_blocked_count": len(rows),
        "same_side_sbc_alternative_available_count": sum(
            1
            for row in rows
            if "same_side_sbc_alternative_available"
            in set(row.get("remap_reason_codes") or [])
        ),
        "same_side_sbc_calibrated_ev_available_count": sum(
            1 for row in rows if row.get("calibrated_ev") is not None
        ),
        "same_side_sbc_guard_passed_count": len(applied_rows),
        "remap_candidate_count": sum(
            1 for row in rows if row.get("remap_candidate") is True
        ),
        "remap_guard_passed_count": len(applied_rows),
        "paper_intent_remap_applied_count": len(intent_remaps),
        "original_action_distribution": _counter_from_rows(rows, "original_action"),
        "remapped_action_distribution": _counter_from_rows(rows, "remapped_action"),
        "remap_reason_distribution": _counter_from_rows(rows, "remap_reason_codes"),
        "remap_failure_reason_distribution": _counter_from_rows(
            [row for row in rows if row.get("hts_time_window_remap_applied") is not True],
            "remap_reason_codes",
        ),
        "paper_intent_remap_ids": [
            str(intent.get("paper_fresh_order_intent_id")) for intent in intent_remaps
        ],
        "remap_rows": rows,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "promotion_evidence": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "execution_layer_v2_paper_remap_report_id")


def _execution_layer_v2_paper_remap_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Execution Layer v2 Paper HTS Time-Window Remap",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_only_intent_path: `{report['paper_only_intent_path']}`",
            f"- hts_time_window_blocked_count: `{report['hts_time_window_blocked_count']}`",
            f"- remap_candidate_count: `{report['remap_candidate_count']}`",
            f"- remap_guard_passed_count: `{report['remap_guard_passed_count']}`",
            f"- paper_intent_remap_applied_count: `{report['paper_intent_remap_applied_count']}`",
            f"- same_side_sbc_alternative_available_count: `{report['same_side_sbc_alternative_available_count']}`",
            f"- same_side_sbc_calibrated_ev_available_count: `{report['same_side_sbc_calibrated_ev_available_count']}`",
            f"- same_side_sbc_guard_passed_count: `{report['same_side_sbc_guard_passed_count']}`",
            f"- v8_execution_handoff_allowed: `{report['v8_execution_handoff_allowed']}`",
            f"- capital_at_risk: `{report['capital_at_risk']}`",
            f"- polymarket_write_enabled: `{report['polymarket_write_enabled']}`",
            f"- wallet_signing_enabled: `{report['wallet_signing_enabled']}`",
            "",
            "## Remap Reason Distribution",
            "",
            "```json",
            json.dumps(report["remap_reason_distribution"], indent=2, sort_keys=True),
            "```",
            "",
            "This report promotes the HTS time-window remap only into the local "
            "paper intent path. It does not mutate O/source scores and does not "
            "unlock live, paper promotion, wallet signing, or Polymarket writes.",
            "",
        ]
    )


def _fresh_loop_run_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
    public_data_collection_report: dict[str, Any],
    public_cycles: list[list[dict[str, Any]]],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    guard_rows = execution_result["guard_decision_rows"]
    remap_rows = list(execution_result.get("paper_remap_rows") or [])
    blockers = list(unlock_evidence["paper_candidate_unlock_blocking_reason_codes"])
    canonical_scorer_used = any(
        row.get("canonical_frozen_o_scorer_used") is True
        for row in _flatten_public_rows(public_cycles)
    )
    if execution_result["cycle_failure_count"]:
        blockers.append("paper_fresh_public_data_cycle_failed")
    if public_data_collection_report["paper_fresh_provider_collection_failed"]:
        blockers.append("paper_fresh_public_provider_collection_failed")
        blockers.extend(public_data_collection_report["public_data_collection_reason_codes"])
    report = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_RUN_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_run",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_loop_enabled": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_fresh_loop_mode": config.loop_mode,
        "paper_fresh_loop_cycle_count": len(public_cycles),
        "paper_fresh_loop_max_cycles": config.max_cycles,
        "paper_fresh_loop_sleep_seconds": config.sleep_seconds,
        "paper_fresh_loop_public_data_source": public_data_collection_report[
            "public_data_source"
        ],
        "public_data_collection_report": public_data_collection_report,
        "paper_fresh_provider_collection_failed": public_data_collection_report[
            "paper_fresh_provider_collection_failed"
        ],
        "public_data_collection_reason_codes": public_data_collection_report[
            "public_data_collection_reason_codes"
        ],
        "scoring_rule_id": (
            "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
            if canonical_scorer_used
            else "fresh_provider_simplified_score"
        ),
        "canonical_frozen_o_scorer_used": canonical_scorer_used,
        "uses_paper_intent_logs_as_fresh_public_data": public_data_collection_report[
            "uses_paper_intent_logs_as_fresh_public_data"
        ],
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_blocking_reason_codes": unlock_evidence[
            "paper_candidate_unlock_blocking_reason_codes"
        ],
        "paper_fresh_loop_blocking_reason_codes": sorted(set(blockers)),
        "public_data_cycle_input_count": sum(len(cycle) for cycle in public_cycles),
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is True
        ),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "execution_layer_v2_paper_remap_enabled": True,
        "execution_layer_v2_paper_remap_candidate_count": sum(
            1 for row in remap_rows if row.get("remap_candidate") is True
        ),
        "execution_layer_v2_paper_remap_applied_count": sum(
            1 for row in remap_rows if row.get("hts_time_window_remap_applied") is True
        ),
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "final_runtime_state": execution_result["final_runtime_state"],
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_paper_internal_handoff_allowed": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_loop_run_report_id")


def _fresh_fill_simulation_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_FILL_SIMULATION_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_fill_simulation",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_fill_simulation_enabled": bool(fills),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_filled_size_sum": sum(_float(row.get("filled_size")) for row in fills),
        "paper_fresh_total_synthetic_execution_cost": sum(
            _float(row.get("total_execution_cost")) for row in fills
        ),
        "fill_simulation_rule_ids": sorted(
            {str(row.get("fill_simulation_rule_id")) for row in fills}
        ),
        "outcome_pnl_used": False,
        "realized_pnl_used": False,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "capital_at_risk": False,
        "paper_only": True,
    }
    return _with_report_id(report, "o_v8_paper_fresh_fill_simulation_report_id")


def _fresh_runtime_safety_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    sell_position_intents: list[dict[str, Any]],
    synthetic_exit_ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        *intents,
        *fills,
        *ledger_rows,
        *sell_position_intents,
        *synthetic_exit_ledger_rows,
    ]
    safety_checks = {
        "paper_only_true": _check(
            passed=config.paper_only is True
            and all(row.get("paper_only") is True for row in rows),
            reason_code="paper_fresh_runtime_not_paper_only",
            observed=True,
            required=True,
        ),
        "capital_at_risk_false": _check(
            passed=config.capital_at_risk is False
            and all(row.get("capital_at_risk") is False for row in rows),
            reason_code="paper_fresh_runtime_capital_at_risk",
            observed=False,
            required=False,
        ),
        "polymarket_writes_disabled": _check(
            passed=config.polymarket_write_enabled is False
            and all(row.get("polymarket_write_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_polymarket_write_enabled",
            observed=False,
            required=False,
        ),
        "wallet_signing_disabled": _check(
            passed=config.wallet_signing_enabled is False
            and all(row.get("wallet_signing_enabled") is False for row in rows),
            reason_code="paper_fresh_runtime_wallet_signing_enabled",
            observed=False,
            required=False,
        ),
        "ledger_updates_only_accepted_intents": _check(
            passed=len(ledger_rows) == len(intents)
            and {row["paper_fresh_order_intent_id"] for row in ledger_rows}
            == {row["paper_fresh_order_intent_id"] for row in intents},
            reason_code="paper_fresh_ledger_updates_unaccepted_intents",
            observed={
                "intent_count": len(intents),
                "ledger_entry_count": len(ledger_rows),
            },
            required="ledger ids equal accepted fresh intent ids",
        ),
        "exit_ledger_updates_only_accepted_sell_position_intents": _check(
            passed=len(synthetic_exit_ledger_rows) == len(sell_position_intents)
            and {
                row["paper_sell_position_intent_id"]
                for row in synthetic_exit_ledger_rows
            }
            == {
                row["paper_sell_position_intent_id"]
                for row in sell_position_intents
            },
            reason_code="paper_fresh_exit_ledger_updates_unaccepted_intents",
            observed={
                "sell_position_intent_count": len(sell_position_intents),
                "synthetic_exit_ledger_entry_count": len(synthetic_exit_ledger_rows),
            },
            required="exit ledger ids equal accepted sell position intent ids",
        ),
        "live_handoff_remains_blocked": _check(
            passed=run_report["v8_execution_handoff_allowed"] is False
            and run_report["#134_resume_allowed"] is False
            and run_report["#146_start_allowed"] is False,
            reason_code="paper_fresh_live_handoff_unexpectedly_unlocked",
            observed={
                "v8_execution_handoff_allowed": run_report[
                    "v8_execution_handoff_allowed"
                ],
                "#134_resume_allowed": run_report["#134_resume_allowed"],
                "#146_start_allowed": run_report["#146_start_allowed"],
            },
            required=False,
        ),
        "no_threshold_tuning_or_forbidden_outcomes": _check(
            passed=run_report["thresholds_tuned"] is False
            and run_report["forbidden_outcome_fields_used"] == [],
            reason_code="paper_fresh_threshold_or_forbidden_outcome_usage",
            observed={
                "thresholds_tuned": run_report["thresholds_tuned"],
                "forbidden_outcome_fields_used": run_report[
                    "forbidden_outcome_fields_used"
                ],
            },
            required={"thresholds_tuned": False, "forbidden_outcome_fields_used": []},
        ),
    }
    blockers = _blocking_reason_codes(safety_checks)
    report = {
        "schema_version": O_V8_PAPER_FRESH_RUNTIME_SAFETY_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_runtime_safety",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_runtime_safety_checks": safety_checks,
        "paper_fresh_runtime_safety_blocking_reason_codes": blockers,
        "paper_fresh_runtime_safety_passed": blockers == [],
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "paper_sell_position_intent_count": len(sell_position_intents),
        "synthetic_exit_ledger_entry_count": len(synthetic_exit_ledger_rows),
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_runtime_safety_report_id")


def _fresh_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "paper_fresh_monitoring_passed": run_report[
            "paper_fresh_loop_blocking_reason_codes"
        ]
        == [],
        "cycle_monitoring_reports": execution_result["cycle_monitoring_rows"],
        "cycle_count": len(execution_result["cycle_monitoring_rows"]),
        "cycle_failure_count": execution_result["cycle_failure_count"],
        "candidate_decision_count": run_report["candidate_decision_count"],
        "guard_allowed_decision_count": run_report["guard_allowed_decision_count"],
        "guard_blocked_decision_count": run_report["guard_blocked_decision_count"],
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "block_reason_distribution": run_report["block_reason_distribution"],
        "action_distribution": run_report["action_distribution"],
        "side_distribution": run_report["side_distribution"],
        "family_distribution": run_report["family_distribution"],
        "final_runtime_state": run_report["final_runtime_state"],
        "safety_flags": compact_safety_fields(),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_monitoring_report_id")


def _fresh_cumulative_monitoring_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    run_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    for intent in intents:
        size = _float(intent.get("paper_fresh_order_size"))
        exposure_by_market[str(intent.get("market_id"))] += size
        exposure_by_side[str(intent.get("execution_guarded_side"))] += size
    report = {
        "schema_version": O_V8_PAPER_FRESH_CUMULATIVE_MONITORING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_cumulative_monitoring",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "total_cycles": monitoring_report["cycle_count"],
        "total_paper_intents": len(intents),
        "total_paper_fills": len(fills),
        "total_blocked_decisions": run_report["guard_blocked_decision_count"],
        "cumulative_block_reason_distribution": run_report[
            "block_reason_distribution"
        ],
        "cumulative_action_distribution": run_report["action_distribution"],
        "cumulative_side_distribution": run_report["side_distribution"],
        "cumulative_family_distribution": run_report["family_distribution"],
        "cumulative_simulated_exposure_by_market": dict(sorted(exposure_by_market.items())),
        "cumulative_simulated_exposure_by_side": dict(sorted(exposure_by_side.items())),
        "cycle_failure_count": monitoring_report["cycle_failure_count"],
        "safety_violation_count": 0
        if run_report["paper_fresh_loop_blocking_reason_codes"] == []
        else len(run_report["paper_fresh_loop_blocking_reason_codes"]),
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "ledger_updates_only_accepted_intents": len(ledger_rows) == len(intents),
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_cumulative_monitoring_report_id"
    )


def _fresh_no_trade_diagnostic_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    execution_result: dict[str, Any],
    run_report: dict[str, Any],
) -> dict[str, Any]:
    guard_rows = list(execution_result["guard_decision_rows"])
    public_rows = _flatten_public_rows(public_cycles)
    public_by_group = {
        str(row.get("decision_group_id")): row for row in public_rows
    }
    decision_rows = []
    for guard_row in guard_rows:
        public_row = public_by_group.get(str(guard_row.get("decision_group_id")), {})
        ranking = _ranking_from_public_or_guard(public_row=public_row, guard_row=guard_row)
        no_trade = _ranking_action(ranking, "NO_TRADE")
        buy_actions = [
            row for row in ranking if str(row.get("selected_action")) != "NO_TRADE"
        ]
        best_buy = max(
            buy_actions,
            key=lambda row: _float(row.get("corrected_model_score")),
            default={},
        )
        selected_action = str(guard_row.get("source_selected_action") or "")
        selected_micro = dict(guard_row.get("microstructure_snapshot") or {})
        execution_blocked = bool(guard_row.get("execution_blocking_reason_codes"))
        decision_rows.append(
            {
                "decision_group_id": guard_row.get("decision_group_id"),
                "market_id": guard_row.get("market_id"),
                "decision_ts": guard_row.get("decision_ts"),
                "selected_action": selected_action,
                "selected_side": guard_row.get("source_selected_side"),
                "selected_family": guard_row.get("source_selected_family"),
                "execution_guarded_action": guard_row.get("execution_guarded_action"),
                "execution_guarded_side": guard_row.get("execution_guarded_side"),
                "execution_guarded_family": guard_row.get("execution_guarded_family"),
                "full_5_action_ranking": ranking,
                "no_trade_score": _float(no_trade.get("corrected_model_score")),
                "best_buy_action": best_buy.get("selected_action"),
                "best_buy_score": _float(best_buy.get("corrected_model_score")),
                "no_trade_score_minus_best_buy_score": _float(
                    no_trade.get("corrected_model_score")
                )
                - _float(best_buy.get("corrected_model_score")),
                "top_action_margin": _top_action_margin(ranking),
                "p_up": guard_row.get("p_up"),
                "p_down": guard_row.get("p_down"),
                "p_up_side_disagreement": guard_row.get("p_up_action_disagreement"),
                "entry_ask": selected_micro.get("entry_ask"),
                "exit_bid_proxy": selected_micro.get("executable_exit_bid_proxy"),
                "spread_bps": selected_micro.get("spread_bps"),
                "queue_fill_proxy": selected_micro.get("queue_fill_proxy"),
                "book_staleness_ms": selected_micro.get("book_staleness_ms"),
                "time_to_close_seconds": selected_micro.get("time_to_close_seconds"),
                "high_score_flag": guard_row.get("source_high_score_flag"),
                "rank_blocked_by_no_trade": selected_action == "NO_TRADE",
                "execution_guard_blocked": execution_blocked,
                "execution_guard_blocking_reasons": list(
                    guard_row.get("execution_blocking_reason_codes") or []
                ),
                "execution_guard_reason_codes": list(
                    guard_row.get("execution_guard_reason_codes") or []
                ),
                "missing_runtime_fields": list(
                    guard_row.get("missing_runtime_field_codes") or []
                ),
                "provenance_violations": list(
                    guard_row.get("runtime_field_backfill_provenance_violations")
                    or []
                ),
            }
        )
    rank_blocked_count = sum(
        1 for row in decision_rows if row["rank_blocked_by_no_trade"] is True
    )
    execution_blocked_count = sum(
        1 for row in decision_rows if row["execution_guard_blocked"] is True
    )
    conclusion = _fresh_no_trade_conclusion(
        run_report=run_report,
        rank_blocked_count=rank_blocked_count,
        execution_blocked_count=execution_blocked_count,
        public_data_collection_report=public_data_collection_report,
    )
    report = {
        "schema_version": O_V8_PAPER_FRESH_NO_TRADE_DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_no_trade_diagnostic",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "scoring_rule_id": run_report["scoring_rule_id"],
        "canonical_frozen_o_scorer_used": run_report[
            "canonical_frozen_o_scorer_used"
        ],
        "candidate_decision_count": len(decision_rows),
        "rank_blocked_by_no_trade_count": rank_blocked_count,
        "execution_guard_blocked_count": execution_blocked_count,
        "paper_fresh_order_intent_count": run_report["paper_fresh_order_intent_count"],
        "selected_action_distribution": _counter_from_rows(
            decision_rows, "selected_action"
        ),
        "best_buy_action_distribution": _counter_from_rows(
            decision_rows, "best_buy_action"
        ),
        "no_trade_gap_summary": _numeric_summary(
            [
                row["no_trade_score_minus_best_buy_score"]
                for row in decision_rows
            ]
        ),
        "decision_rows": decision_rows,
        "historical_run_comparison_rows": _fresh_historical_run_comparison_rows(
            current_run_report=run_report,
            output_dir=Path(config.output_dir),
        ),
        "zero_intent_behavior_classification": conclusion,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_no_trade_diagnostic_report_id")


def _fresh_score_decomposition_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
) -> dict[str, Any]:
    action_rows = []
    for public_row in _flatten_public_rows(public_cycles):
        ranking = list(public_row.get("full_5_action_ranking") or [])
        for rank, action_row in enumerate(ranking, start=1):
            action = str(action_row.get("selected_action") or "")
            decomposition = dict(action_row.get("score_decomposition") or {})
            action_rows.append(
                {
                    "decision_group_id": public_row.get("decision_group_id"),
                    "market_id": public_row.get("market_id"),
                    "decision_ts": public_row.get("decision_ts"),
                    "action": action,
                    "side": action_row.get("selected_side") or _side_from_action(action),
                    "family": action_row.get("selected_action_family")
                    or _action_family(action),
                    "rank": action_row.get("rank") or rank,
                    "p_up": public_row.get("p_up"),
                    "p_down": public_row.get("p_down"),
                    "p_side_contribution": _float(
                        decomposition.get("p_side_contribution")
                    ),
                    "p_up_contribution": _float(
                        decomposition.get("p_up_contribution")
                    ),
                    "p_down_contribution": _float(
                        decomposition.get("p_down_contribution")
                    ),
                    "ask_contribution": _float(
                        decomposition.get("ask_contribution")
                    ),
                    "bid_contribution": _float(
                        decomposition.get("bid_contribution")
                    ),
                    "spread_penalty": _float(decomposition.get("spread_penalty")),
                    "queue_fill_term": _float(decomposition.get("queue_fill_term")),
                    "book_staleness_term": _float(
                        decomposition.get("book_staleness_term")
                    ),
                    "time_to_close_term": _float(
                        decomposition.get("time_to_close_term")
                    ),
                    "queue_fill_term_used": bool(
                        decomposition.get("queue_fill_term_used")
                    ),
                    "book_staleness_term_used": bool(
                        decomposition.get("book_staleness_term_used")
                    ),
                    "time_to_close_term_used": bool(
                        decomposition.get("time_to_close_term_used")
                    ),
                    "raw_score": _float(
                        decomposition.get("raw_score")
                        if decomposition
                        else action_row.get("raw_model_score")
                    ),
                    "corrected_score": _float(
                        decomposition.get("corrected_score")
                        if decomposition
                        else action_row.get("corrected_model_score")
                    ),
                    "is_buy_action": action != "NO_TRADE",
                    "is_sell_before_close": _action_family(action)
                    == "SELL_BEFORE_CLOSE",
                    "is_hold_to_settlement": _action_family(action)
                    == "HOLD_TO_SETTLEMENT",
                    "is_no_trade": action == "NO_TRADE",
                    "scoring_rule_id": decomposition.get("scoring_rule_id")
                    or "fresh_provider_simplified_score",
                    "canonical_frozen_o_scorer_used": bool(
                        decomposition.get("canonical_frozen_o_scorer_used")
                    ),
                }
            )
    canonical_scorer_used = any(
        row.get("canonical_frozen_o_scorer_used") is True for row in action_rows
    )
    report = {
        "schema_version": O_V8_PAPER_FRESH_SCORE_DECOMPOSITION_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_score_decomposition",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "scoring_rule_id": (
            "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
            if canonical_scorer_used
            else "fresh_provider_simplified_score"
        ),
        "canonical_frozen_o_scorer_used": canonical_scorer_used,
        "score_decomposition_action_row_count": len(action_rows),
        "score_decomposition_rows": action_rows,
        "score_summary_by_action": _score_summary_by_action(action_rows),
        "simplified_provider_score_terms": [
            "p_side_contribution",
            "ask_contribution",
            "bid_contribution",
            "spread_penalty",
        ],
        "unused_decision_time_quality_terms": [
            "queue_fill_term",
            "book_staleness_term",
            "time_to_close_term",
        ],
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_score_decomposition_report_id"
    )


def _fresh_provider_feature_coverage_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    run_report: dict[str, Any],
) -> dict[str, Any]:
    rows = _flatten_public_rows(public_cycles)
    rows_per_cycle = [len(cycle) for cycle in public_cycles]
    missing_micro = [
        missing
        for row in rows
        for missing in _missing_required_microstructure_fields(row)
    ]
    invalid_provenance = [
        row
        for row in rows
        if (row.get("reference_price_feature_provenance") or {}).get(
            "provenance_valid"
        )
        is not True
    ]
    chainlink_available_rows = [
        row for row in rows if row.get("chainlink_price_at_decision") is not None
    ]
    chainlink_reference_rows = [
        row
        for row in rows
        if row.get("chainlink_reference_price_at_market_start") is not None
    ]
    chainlink_missing_reasons = Counter(
        str(reason)
        for row in rows
        for reason in (
            row.get("chainlink_regime_feature_provenance") or {}
        ).get("unavailable_reason_codes", [])
    )
    sparse = len(rows) < max(5, len(public_cycles))
    report = {
        "schema_version": O_V8_PAPER_FRESH_PROVIDER_FEATURE_COVERAGE_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_provider_feature_coverage",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "public_market_count": public_data_collection_report["public_market_count"],
        "public_feature_row_count": len(rows),
        "unique_market_count": len({str(row.get("market_id")) for row in rows}),
        "cycle_count": len(public_cycles),
        "cycles_with_rows": sum(1 for count in rows_per_cycle if count > 0),
        "idle_cycles": sum(1 for count in rows_per_cycle if count == 0),
        "rows_per_cycle": rows_per_cycle,
        "missing_book_side_count": sum(
            1
            for row in rows
            if len(row.get("full_5_action_ranking") or [])
            < len(O_REQUIRED_DECISION_ACTION_FAMILIES)
        ),
        "missing_btc_candle_count": 0
        if rows
        else int(
            public_data_collection_report["public_btc_feature_candle_row_count"] == 0
        ),
        "missing_required_microstructure_field_count": len(missing_micro),
        "missing_required_microstructure_field_distribution": dict(
            sorted(Counter(missing_micro).items())
        ),
        "missing_runtime_field_count": run_report["runtime_field_missing_count"],
        "provenance_invalid_count": len(invalid_provenance),
        "provider_collection_failures": int(
            public_data_collection_report["paper_fresh_provider_collection_failed"]
        ),
        "public_orderbook_row_count": public_data_collection_report[
            "public_orderbook_row_count"
        ],
        "public_trade_row_count": public_data_collection_report[
            "public_trade_row_count"
        ],
        "public_btc_feature_candle_row_count": public_data_collection_report[
            "public_btc_feature_candle_row_count"
        ],
        "public_chainlink_rtds_price_row_count": int(
            public_data_collection_report.get("public_chainlink_rtds_price_row_count")
            or 0
        ),
        "chainlink_price_at_decision_available_count": len(
            chainlink_available_rows
        ),
        "chainlink_market_start_reference_available_count": len(
            chainlink_reference_rows
        ),
        "chainlink_reference_distance_available_count": sum(
            1
            for row in rows
            if row.get("chainlink_reference_distance_at_decision") is not None
        ),
        "chainlink_feature_provenance_violation_count": sum(
            1
            for row in rows
            if row.get("chainlink_price_at_decision") is not None
            and (row.get("chainlink_regime_feature_provenance") or {}).get(
                "provenance_valid"
            )
            is not True
        ),
        "chainlink_feature_missing_reason_distribution": dict(
            sorted(chainlink_missing_reasons.items())
        ),
        "sparse_provider_row_flag": sparse,
        "sparse_provider_row_reason_codes": [
            "provider_feature_rows_below_minimum_diagnostic_density"
        ]
        if sparse
        else [],
        "historical_run_comparison_rows": _fresh_historical_run_comparison_rows(
            current_run_report=run_report,
            output_dir=Path(config.output_dir),
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_provider_feature_coverage_report_id"
    )


def _fresh_canonical_scorer_context(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_manifest_path = _fresh_canonical_source_manifest_path(
        config=config,
        unlock_evidence=unlock_evidence,
    )
    ranking_objective_report_path: Path | None = None
    model_summary: dict[str, Any] = {}
    if source_manifest_path is None or not source_manifest_path.exists():
        reason_codes.append("missing_frozen_model_summary")
    else:
        source_manifest = _read_json(source_manifest_path)
        ranking_objective_report_path = _resolve_fresh_canonical_artifact_path(
            source_manifest_path.parent,
            str(
                (source_manifest.get("artifact_paths") or {}).get(
                    "ranking_objective_report"
                )
                or ""
            ),
        )
        if ranking_objective_report_path is None or not ranking_objective_report_path.exists():
            reason_codes.append("missing_frozen_model_summary")
        else:
            ranking_objective_report = _read_json(ranking_objective_report_path)
            model_summary = dict(
                ranking_objective_report.get("o_model_training_summary") or {}
            )
    feature_names = list(model_summary.get("feature_names") or [])
    coefficients_by_feature = dict(model_summary.get("coefficients_by_feature") or {})
    ranking_correction_config = dict(
        model_summary.get("ranking_correction_config") or {}
    )
    if not feature_names:
        reason_codes.append("missing_feature_schema")
    if not coefficients_by_feature or any(
        feature_name not in coefficients_by_feature for feature_name in feature_names
    ):
        reason_codes.append("missing_coefficients")
    if not ranking_correction_config:
        reason_codes.append("missing_ranking_correction_config")
    correction_hash = str(ranking_correction_config.get("correction_config_hash") or "")
    summary_hash = str(model_summary.get("correction_config_hash") or "")
    correction_hash_verified = bool(correction_hash) and correction_hash == summary_hash
    if ranking_correction_config and not correction_hash_verified:
        reason_codes.append("ranking_correction_config_hash_mismatch")
    return {
        "source_manifest_path": str(source_manifest_path) if source_manifest_path else None,
        "ranking_objective_report_path": (
            str(ranking_objective_report_path)
            if ranking_objective_report_path is not None
            else None
        ),
        "source_manifest_sha256": _sha256_file(source_manifest_path)
        if source_manifest_path is not None and source_manifest_path.exists()
        else None,
        "ranking_objective_report_sha256": _sha256_file(ranking_objective_report_path)
        if ranking_objective_report_path is not None
        and ranking_objective_report_path.exists()
        else None,
        "model_summary_available": bool(model_summary),
        "feature_names": feature_names,
        "feature_schema_hash": canonical_json_sha256(feature_names)
        if feature_names
        else None,
        "coefficients_by_feature": coefficients_by_feature,
        "coefficient_count": len(coefficients_by_feature),
        "ranking_correction_config": ranking_correction_config,
        "ranking_correction_config_hash": correction_hash or None,
        "ranking_correction_config_hash_verified": correction_hash_verified,
        "selected_feature_set_name": model_summary.get("selected_feature_set_name"),
        "selected_correction_policy_name": model_summary.get(
            "selected_correction_policy_name"
        ),
        "selected_high_score_threshold_profile_name": model_summary.get(
            "selected_high_score_threshold_profile_name"
        ),
        "deployable_model_score_available": bool(
            model_summary.get("deployable_model_score_available")
        ),
        "canonical_input_blocking_reason_codes": sorted(set(reason_codes)),
        "canonical_inputs_available": not reason_codes,
    }


def _fresh_canonical_source_manifest_path(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    unlock_evidence: dict[str, Any],
) -> Path | None:
    if config.canonical_o_source_manifest_path is not None:
        return Path(config.canonical_o_source_manifest_path)
    unlock_manifest = dict(unlock_evidence.get("unlock_manifest") or {})
    raw_path = (
        (unlock_manifest.get("input_artifact_paths") or {}).get("source_manifest")
        or (unlock_manifest.get("pinned_artifact_paths") or {}).get("source_manifest")
    )
    if not raw_path:
        return None
    return _resolve_fresh_canonical_artifact_path(
        Path(unlock_evidence["paper_candidate_unlock_dir"]),
        str(raw_path),
    )


def _resolve_fresh_canonical_artifact_path(
    base_dir: Path,
    raw_path: str,
) -> Path | None:
    if not raw_path:
        return None
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute() or artifact_path.exists():
        return artifact_path
    return base_dir / artifact_path


def _fresh_canonical_feature_mapping_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    canonical_context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _flatten_public_rows(public_cycles)
    feature_names = list(
        canonical_context.get("feature_names") or O_DEPLOYABLE_MODEL_FEATURE_NAMES
    )
    canonical_action_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = list(
        canonical_context.get("canonical_input_blocking_reason_codes") or []
    )
    for row in rows:
        ranking = list(row.get("full_5_action_ranking") or [])
        available_actions = sorted(
            str(action_row.get("selected_action") or "") for action_row in ranking
        )
        missing_actions = sorted(
            set(O_REQUIRED_DECISION_ACTION_FAMILIES).difference(available_actions)
        )
        group_reason_codes = []
        if missing_actions:
            group_reason_codes.append("incomplete_action_row_normalization")
            reason_codes.append("incomplete_action_row_normalization")
        action_mapping_rows = []
        for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
            action_row = _fresh_canonical_action_row(
                public_row=row,
                action=action,
                feature_names=tuple(feature_names),
            )
            canonical_action_rows.append(action_row)
            action_mapping_rows.append(
                {
                    "action": action,
                    "canonical_action_row_hash": action_row[
                        "canonical_action_row_hash"
                    ],
                    "mapped_feature_count": len(action_row["canonical_feature_values"]),
                    "missing_canonical_features": action_row[
                        "missing_canonical_features"
                    ],
                    "default_backfilled_features": action_row[
                        "default_backfilled_features"
                    ],
                    "provenance_valid": action_row[
                        "canonical_feature_mapping_provenance_valid"
                    ],
                    "provenance_violation_reason_codes": action_row[
                        "canonical_feature_mapping_provenance_violation_reason_codes"
                    ],
                }
            )
            if action_row["missing_canonical_features"]:
                reason_codes.append("missing_feature_backfill_mapping")
            if not action_row["canonical_feature_mapping_provenance_valid"]:
                reason_codes.append("provenance_invalid_for_mapped_features")
        mapping_rows.append(
            {
                "decision_group_id": row.get("decision_group_id"),
                "market_id": row.get("market_id"),
                "decision_ts": row.get("decision_ts"),
                "available_action_families": [
                    _action_family(action) for action in available_actions if action
                ],
                "available_actions": available_actions,
                "missing_action_families": [
                    _action_family(action) for action in missing_actions
                ],
                "missing_actions": missing_actions,
                "decision_group_action_row_normalization_complete": (
                    not missing_actions
                ),
                "decision_group_mapping_reason_codes": group_reason_codes,
                "canonical_action_mapping_rows": action_mapping_rows,
            }
        )
    provider_features = _fresh_provider_feature_names(rows)
    missing_canonical_features = sorted(
        feature
        for feature in feature_names
        if not all(
            feature in row["canonical_feature_values"]
            for row in canonical_action_rows
        )
    )
    if missing_canonical_features:
        reason_codes.append("missing_feature_backfill_mapping")
    provenance_invalid_count = sum(
        1
        for row in canonical_action_rows
        if not row["canonical_feature_mapping_provenance_valid"]
    )
    complete = (
        canonical_context.get("canonical_inputs_available") is True
        and not missing_canonical_features
        and provenance_invalid_count == 0
        and len(canonical_action_rows)
        == len(rows) * len(O_REQUIRED_DECISION_ACTION_FAMILIES)
    )
    if rows and not canonical_action_rows:
        reason_codes.append("unsupported_fresh_provider_row_shape")
    report = {
        "schema_version": O_V8_PAPER_FRESH_CANONICAL_FEATURE_MAPPING_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_canonical_feature_mapping",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "canonical_source_manifest_path": canonical_context.get(
            "source_manifest_path"
        ),
        "canonical_ranking_objective_report_path": canonical_context.get(
            "ranking_objective_report_path"
        ),
        "fresh_provider_scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_feature_mapping_complete": complete,
        "canonical_feature_mapping_blocking_reason_codes": sorted(set(reason_codes)),
        "provider_feature_names": provider_features,
        "canonical_feature_names": feature_names,
        "canonical_feature_count": len(feature_names),
        "mapped_feature_count": len(feature_names) - len(missing_canonical_features),
        "missing_canonical_feature_names": missing_canonical_features,
        "default_backfilled_feature_distribution": dict(
            sorted(
                Counter(
                    feature
                    for action_row in canonical_action_rows
                    for feature in action_row["default_backfilled_features"]
                ).items()
            )
        ),
        "provenance_invalid_count": provenance_invalid_count,
        "decision_group_count": len(rows),
        "canonical_action_row_count": len(canonical_action_rows),
        "expected_canonical_action_row_count": len(rows)
        * len(O_REQUIRED_DECISION_ACTION_FAMILIES),
        "mapping_decision_rows": mapping_rows,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        **compact_safety_fields(),
    }
    return (
        _with_report_id(
            report, "o_v8_paper_fresh_canonical_feature_mapping_report_id"
        ),
        canonical_action_rows,
    )


def _fresh_canonical_action_row(
    *,
    public_row: dict[str, Any],
    action: str,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    decision_ts = int(public_row.get("decision_ts") or 0)
    ranking = list(public_row.get("full_5_action_ranking") or [])
    provider_action_row = _ranking_action(ranking, action)
    micro = {
        **dict(public_row.get("microstructure_snapshot") or {}),
        **dict(provider_action_row.get("microstructure_snapshot") or {}),
    }
    side = _side_from_action(action)
    family = _action_family(action)
    score_components = dict(public_row.get("score_components") or {})
    reference_distance, reference_backfilled = _fresh_reference_distance(public_row)
    default_backfilled_features: list[str] = []
    entry_ask = _fresh_feature_value(
        micro.get("entry_ask"),
        default=0.0,
        feature_name="entry_quality_ask",
        defaults=default_backfilled_features,
    )
    spread_bps = _fresh_feature_value(
        micro.get("spread_bps"),
        default=0.0,
        feature_name="entry_exit_quality_spread_bps",
        defaults=default_backfilled_features,
    )
    queue_fill = _fresh_feature_value(
        micro.get("queue_fill_proxy"),
        default=0.0,
        feature_name="entry_exit_quality_queue_fill",
        defaults=default_backfilled_features,
    )
    staleness_ms = _fresh_feature_value(
        micro.get("book_staleness_ms"),
        default=0.0,
        feature_name="entry_exit_quality_book_staleness_ms",
        defaults=default_backfilled_features,
    )
    time_to_close_seconds = _fresh_feature_value(
        micro.get("time_to_close_seconds"),
        default=0.0,
        feature_name="entry_exit_quality_time_to_close_seconds",
        defaults=default_backfilled_features,
    )
    if reference_backfilled:
        default_backfilled_features.append(
            "reference_price_to_beat_distance_at_decision"
        )
    default_backfilled_features.extend(
        [
            "recent_reference_price_momentum_30s",
            "recent_reference_price_momentum_60s",
            "recent_reference_price_momentum_120s",
            "side_book_depth_imbalance",
            "side_book_update_velocity",
            "p_up_calibration_residual_by_time_spread_queue_bucket",
        ]
    )
    reference_provenance = dict(
        public_row.get("reference_price_feature_provenance") or {}
    )
    max_input_ts = int(
        reference_provenance.get("max_input_ts")
        or public_row.get("decision_time_feature_max_input_ts")
        or decision_ts
    )
    provenance_valid = bool(
        reference_provenance.get("provenance_valid", True)
    ) and max_input_ts <= decision_ts
    row = {
        "decision_group_id": public_row.get("decision_group_id"),
        "market_id": public_row.get("market_id"),
        "decision_ts": decision_ts,
        "action": action,
        "selected_action": action,
        "selected_side": side,
        "action_family": family,
        "selected_action_family": family,
        "p_up": _float(public_row.get("p_up")),
        "p_down": _float(public_row.get("p_down")),
        "p_up_action_disagreement": _p_up_action_disagreement(
            action=action,
            p_up=_float(public_row.get("p_up")),
        ),
        "entry_quality_ask": entry_ask,
        "entry_exit_quality_spread_bps": spread_bps,
        "entry_exit_quality_queue_fill": queue_fill,
        "entry_exit_quality_book_staleness_ms": staleness_ms,
        "entry_exit_quality_time_to_close_seconds": time_to_close_seconds,
        "reference_price_to_beat_distance_at_decision": reference_distance,
        "recent_reference_price_momentum_30s": 0.0,
        "recent_reference_price_momentum_60s": 0.0,
        "recent_reference_price_momentum_120s": 0.0,
        "reference_price_feature_available": reference_distance is not None,
        "side_book_depth_imbalance": 0.0,
        "side_book_update_velocity": 0.0,
        "side_book_staleness_ms": staleness_ms,
        "opposite_book_staleness_ms": _fresh_opposite_book_staleness(
            ranking=ranking,
            action=action,
            default=staleness_ms,
        ),
        "side_spread_bps": spread_bps,
        "side_queue_fill_proxy": queue_fill,
        "hts_vs_sell_before_close_exit_value_gap_proxy": _fresh_hts_sbc_gap_proxy(
            ranking=ranking,
            action=action,
        ),
        "p_up_calibration_residual_by_time_spread_queue_bucket": 0.0,
        "book_pressure_feature_available": bool(micro),
        "canonical_feature_mapping_rule_id": (
            "fresh_provider_row_to_frozen_o_deployable_features_v1"
        ),
        "canonical_feature_mapping_source_fields": [
            "full_5_action_ranking.microstructure_snapshot",
            "score_components.p_up",
            "score_components.p_down",
            "score_components.btc_mid_price",
            "score_components.reference_price_to_beat",
            "reference_price_feature_provenance",
        ],
        "canonical_feature_mapping_max_input_ts": max_input_ts,
        "canonical_feature_mapping_provenance_valid": provenance_valid,
        "canonical_feature_mapping_provenance_violation_reason_codes": []
        if provenance_valid
        else ["provenance_invalid_for_mapped_features"],
        "canonical_feature_mapping_provenance": {
            "decision_ts": decision_ts,
            "max_input_ts": max_input_ts,
            "source_timestamp": max_input_ts,
            "source_field_name": "fresh_provider_decision_time_features",
            "source_fields_used": [
                "full_5_action_ranking",
                "score_components",
                "reference_price_feature_provenance",
            ],
            "deterministic_rule_id": (
                "fresh_provider_row_to_frozen_o_deployable_features_v1"
            ),
            "provenance_valid": provenance_valid,
        },
        "score_components_from_simplified_provider": score_components,
    }
    feature_map = _deployable_model_feature_map(row)
    row["canonical_feature_values"] = {
        feature_name: float(feature_map.get(feature_name, 0.0))
        for feature_name in feature_names
        if feature_name in feature_map
    }
    row["canonical_feature_names"] = list(feature_names)
    row["missing_canonical_features"] = [
        feature_name for feature_name in feature_names if feature_name not in feature_map
    ]
    row["default_backfilled_features"] = sorted(set(default_backfilled_features))
    row["canonical_action_row_hash"] = canonical_json_sha256(row)
    return row


def _fresh_canonical_scorer_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    canonical_context: dict[str, Any],
    canonical_feature_mapping_report: dict[str, Any],
    canonical_action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_codes = list(canonical_context.get("canonical_input_blocking_reason_codes") or [])
    if canonical_feature_mapping_report["canonical_feature_mapping_complete"] is not True:
        reason_codes.extend(
            canonical_feature_mapping_report[
                "canonical_feature_mapping_blocking_reason_codes"
            ]
        )
    scored_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    invoked = not reason_codes
    if invoked:
        feature_names = list(canonical_context["feature_names"])
        coefficients = dict(canonical_context["coefficients_by_feature"])
        raw_rows = []
        for row in canonical_action_rows:
            raw_score = sum(
                float(coefficients.get(feature_name, 0.0))
                * float(row["canonical_feature_values"].get(feature_name, 0.0))
                for feature_name in feature_names
            )
            raw_rows.append({**row, "o_raw_ridge_model_score": raw_score})
        scored_rows = _apply_o_shadow_ranking_correction(
            rows=raw_rows,
            deployable_available=True,
            ranking_correction=dict(canonical_context["ranking_correction_config"]),
        )
        high_score_threshold = float(
            canonical_context["ranking_correction_config"]["high_score_calibration"][
                "high_score_threshold"
            ]
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored_rows:
            row["canonical_raw_model_score"] = row["o_raw_ridge_model_score"]
            row["canonical_corrected_model_score"] = row["o_model_predicted_score"]
            row["raw_model_score"] = row["o_raw_ridge_model_score"]
            row["corrected_model_score"] = row["o_model_predicted_score"]
            row["high_score_flag"] = (
                float(row["o_model_predicted_score"]) >= high_score_threshold
            )
            row["ranking_score_source"] = "canonical_frozen_o_model_predicted_score"
            row["canonical_frozen_o_scorer_used"] = True
            row["canonical_scored_action_row_hash"] = canonical_json_sha256(row)
            grouped[str(row["decision_group_id"])].append(row)
        ranked_rows: list[dict[str, Any]] = []
        for group_rows in grouped.values():
            ordered = sorted(
                group_rows,
                key=lambda row: (
                    float(row["canonical_corrected_model_score"]),
                    1 if row["action"] != "NO_TRADE" else 0,
                    str(row["action"]),
                ),
                reverse=True,
            )
            for rank, row in enumerate(ordered, start=1):
                row["canonical_rank"] = rank
                ranked_rows.append(row)
            selected_rows.append(ordered[0])
        scored_rows = sorted(
            ranked_rows,
            key=lambda row: (str(row["decision_group_id"]), int(row["canonical_rank"])),
        )
    report = {
        "schema_version": O_V8_PAPER_FRESH_CANONICAL_SCORER_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_canonical_scorer",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "canonical_frozen_o_scorer_invoked": invoked,
        "canonical_frozen_o_scorer_used": invoked,
        "canonical_scorer_diagnostic_status": "passed"
        if invoked
        else "blocked_fail_closed",
        "canonical_scorer_blocking_reason_codes": sorted(set(reason_codes)),
        "canonical_source_manifest_path": canonical_context.get(
            "source_manifest_path"
        ),
        "canonical_ranking_objective_report_path": canonical_context.get(
            "ranking_objective_report_path"
        ),
        "feature_names": canonical_context.get("feature_names") or [],
        "feature_schema_hash": canonical_context.get("feature_schema_hash"),
        "coefficient_count": canonical_context.get("coefficient_count"),
        "ranking_correction_config_hash": canonical_context.get(
            "ranking_correction_config_hash"
        ),
        "ranking_correction_config_hash_verified": canonical_context.get(
            "ranking_correction_config_hash_verified"
        ),
        "selected_feature_set_name": canonical_context.get("selected_feature_set_name"),
        "selected_correction_policy_name": canonical_context.get(
            "selected_correction_policy_name"
        ),
        "selected_high_score_threshold_profile_name": canonical_context.get(
            "selected_high_score_threshold_profile_name"
        ),
        "canonical_action_row_count": len(canonical_action_rows),
        "canonical_scored_action_row_count": len(scored_rows),
        "canonical_selected_decision_count": len(selected_rows),
        "canonical_selected_action_distribution": _counter_from_rows(
            selected_rows, "action"
        ),
        "canonical_scored_action_rows": scored_rows,
        "canonical_selected_decision_rows": selected_rows,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_canonical_scorer_report_id")


def _fresh_execution_cycles_from_canonical_scorer(
    *,
    public_cycles: list[list[dict[str, Any]]],
    canonical_scorer_report: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if canonical_scorer_report.get("canonical_frozen_o_scorer_used") is not True:
        return public_cycles
    scored_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_scorer_report.get("canonical_scored_action_rows", []):
        scored_by_group[str(row.get("decision_group_id"))].append(dict(row))
    selected_by_group = {
        str(row.get("decision_group_id")): dict(row)
        for row in canonical_scorer_report.get("canonical_selected_decision_rows", [])
    }
    execution_cycles: list[list[dict[str, Any]]] = []
    for cycle in public_cycles:
        execution_rows = []
        for public_row in cycle:
            group_id = str(public_row.get("decision_group_id"))
            selected = selected_by_group.get(group_id)
            group_rows = sorted(
                scored_by_group.get(group_id, []),
                key=lambda row: int(row.get("canonical_rank") or 999),
            )
            if selected is None or not group_rows:
                execution_rows.append(dict(public_row))
                continue
            canonical_ranking = [
                _fresh_public_ranking_row_from_canonical(row) for row in group_rows
            ]
            selected_ranking_row = _fresh_public_ranking_row_from_canonical(selected)
            action = str(selected.get("action") or "")
            execution_rows.append(
                {
                    **dict(public_row),
                    "selected_action": action,
                    "selected_side": selected.get("selected_side")
                    or _side_from_action(action),
                    "selected_action_family": selected.get("selected_action_family")
                    or _action_family(action),
                    "corrected_model_score": selected.get(
                        "canonical_corrected_model_score"
                    ),
                    "raw_model_score": selected.get("canonical_raw_model_score"),
                    "high_score_flag": bool(selected.get("high_score_flag")),
                    "p_up_action_disagreement": bool(
                        selected.get("p_up_action_disagreement")
                    ),
                    "microstructure_snapshot": selected_ranking_row[
                        "microstructure_snapshot"
                    ],
                    "full_5_action_ranking": canonical_ranking,
                    "canonical_frozen_o_scorer_used": True,
                    "ranking_score_source": (
                        "canonical_frozen_o_model_predicted_score"
                    ),
                    "fresh_provider_simplified_selected_action": public_row.get(
                        "selected_action"
                    ),
                    "fresh_provider_simplified_corrected_model_score": public_row.get(
                        "corrected_model_score"
                    ),
                }
            )
        execution_cycles.append(execution_rows)
    return execution_cycles


def _fresh_public_ranking_row_from_canonical(row: dict[str, Any]) -> dict[str, Any]:
    action = str(row.get("action") or "")
    score_components = dict(row.get("o_model_score_components") or {})
    return {
        "selected_action": action,
        "selected_side": row.get("selected_side") or _side_from_action(action),
        "selected_action_family": row.get("selected_action_family")
        or _action_family(action),
        "corrected_model_score": row.get("canonical_corrected_model_score"),
        "raw_model_score": row.get("canonical_raw_model_score"),
        "rank": row.get("canonical_rank"),
        "high_score_flag": bool(row.get("high_score_flag")),
        "microstructure_snapshot": {
            "entry_ask": row.get("entry_quality_ask"),
            "executable_exit_bid_proxy": row.get(
                "hts_vs_sell_before_close_exit_value_gap_proxy"
            ),
            "spread_bps": row.get("entry_exit_quality_spread_bps"),
            "book_staleness_ms": row.get(
                "entry_exit_quality_book_staleness_ms"
            ),
            "queue_fill_proxy": row.get("entry_exit_quality_queue_fill"),
            "time_to_close_seconds": row.get(
                "entry_exit_quality_time_to_close_seconds"
            ),
        },
        "score_decomposition": {
            **score_components,
            "raw_score": row.get("canonical_raw_model_score"),
            "corrected_score": row.get("canonical_corrected_model_score"),
            "scoring_rule_id": (
                "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
            ),
            "canonical_frozen_o_scorer_used": True,
        },
        "canonical_rank": row.get("canonical_rank"),
        "canonical_scored_action_row_hash": row.get(
            "canonical_scored_action_row_hash"
        ),
    }


def _fresh_scorer_comparison_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    canonical_scorer_report: dict[str, Any],
) -> dict[str, Any]:
    public_rows = _flatten_public_rows(public_cycles)
    canonical_by_group_action = {
        (str(row.get("decision_group_id")), str(row.get("action"))): row
        for row in canonical_scorer_report.get("canonical_scored_action_rows", [])
    }
    reason_codes = list(canonical_scorer_report["canonical_scorer_blocking_reason_codes"])
    comparison_rows = []
    for public_row in public_rows:
        ranking = [dict(row) for row in public_row.get("full_5_action_ranking") or []]
        simplified_rank_by_action = _rank_by_action(
            ranking,
            score_field="corrected_model_score",
            action_field="selected_action",
        )
        simplified_selected = str(
            public_row.get("selected_action")
            or (ranking[0].get("selected_action") if ranking else "")
        )
        group_id = str(public_row.get("decision_group_id"))
        canonical_group_rows = [
            canonical_by_group_action[(group_id, action)]
            for action in O_REQUIRED_DECISION_ACTION_FAMILIES
            if (group_id, action) in canonical_by_group_action
        ]
        canonical_selected = (
            sorted(
                canonical_group_rows,
                key=lambda row: int(row.get("canonical_rank") or 999),
            )[0]
            if canonical_group_rows
            else {}
        )
        canonical_rank_by_action = {
            str(row.get("action")): int(row.get("canonical_rank") or 999)
            for row in canonical_group_rows
        }
        action_deltas = []
        for action in O_REQUIRED_DECISION_ACTION_FAMILIES:
            simplified_action_row = _ranking_action(ranking, action)
            canonical_action_row = canonical_by_group_action.get((group_id, action), {})
            action_deltas.append(
                {
                    "action": action,
                    "simplified_score": _float(
                        simplified_action_row.get("corrected_model_score")
                    ),
                    "canonical_raw_score": canonical_action_row.get(
                        "canonical_raw_model_score"
                    ),
                    "canonical_corrected_score": canonical_action_row.get(
                        "canonical_corrected_model_score"
                    ),
                    "simplified_rank": simplified_rank_by_action.get(action),
                    "canonical_rank": canonical_rank_by_action.get(action),
                    "rank_difference_simplified_minus_canonical": (
                        simplified_rank_by_action[action]
                        - canonical_rank_by_action[action]
                        if action in simplified_rank_by_action
                        and action in canonical_rank_by_action
                        else None
                    ),
                }
            )
        canonical_selected_action = canonical_selected.get("action")
        comparison_rows.append(
            {
                "decision_group_id": public_row.get("decision_group_id"),
                "market_id": public_row.get("market_id"),
                "decision_ts": public_row.get("decision_ts"),
                "simplified_provider_selected_action": simplified_selected,
                "canonical_selected_action": canonical_selected_action,
                "selected_action_agrees": simplified_selected
                == canonical_selected_action,
                "no_trade_selection_agrees": (
                    (simplified_selected == "NO_TRADE")
                    == (canonical_selected_action == "NO_TRADE")
                    if canonical_selected_action
                    else None
                ),
                "simplified_top_action_margin": _top_action_margin(ranking),
                "canonical_top_action_margin": _top_action_margin(
                    [
                        {
                            "corrected_model_score": row.get(
                                "canonical_corrected_model_score"
                            ),
                            "selected_action": row.get("action"),
                        }
                        for row in canonical_group_rows
                    ]
                ),
                "score_rank_differences_by_action": action_deltas,
                "comparison_status": "passed"
                if canonical_selected_action
                else "blocked_fail_closed",
                "comparison_reason_codes": []
                if canonical_selected_action
                else ["canonical_frozen_o_scorer_not_invoked"],
            }
        )
    comparison_complete = bool(public_rows) and all(
        row["comparison_status"] == "passed" for row in comparison_rows
    )
    selected_action_agreement_count = sum(
        1 for row in comparison_rows if row["selected_action_agrees"] is True
    )
    no_trade_agreement_count = sum(
        1 for row in comparison_rows if row["no_trade_selection_agrees"] is True
    )
    report = {
        "schema_version": O_V8_PAPER_FRESH_SCORER_COMPARISON_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_scorer_comparison",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "fresh_provider_scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_scoring_rule_id": (
            "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
        ),
        "scorer_comparison_complete": comparison_complete,
        "scorer_comparison_blocking_reason_codes": sorted(set(reason_codes))
        if not comparison_complete
        else [],
        "decision_group_count": len(public_rows),
        "comparison_decision_rows": comparison_rows,
        "selected_action_agreement_count": selected_action_agreement_count,
        "selected_action_disagreement_count": len(comparison_rows)
        - selected_action_agreement_count,
        "no_trade_agreement_count": no_trade_agreement_count,
        "no_trade_disagreement_count": len(comparison_rows) - no_trade_agreement_count,
        "simplified_selected_action_distribution": _counter_from_rows(
            comparison_rows, "simplified_provider_selected_action"
        ),
        "canonical_selected_action_distribution": _counter_from_rows(
            comparison_rows, "canonical_selected_action"
        ),
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_scorer_comparison_report_id")


def _fresh_canonical_scorer_alignment_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    canonical_context: dict[str, Any],
    canonical_feature_mapping_report: dict[str, Any],
    canonical_scorer_report: dict[str, Any],
    scorer_comparison_report: dict[str, Any],
) -> dict[str, Any]:
    rows = _flatten_public_rows(public_cycles)
    provider_features = _fresh_provider_feature_names(rows)
    feature_names = list(
        canonical_context.get("feature_names") or O_DEPLOYABLE_MODEL_FEATURE_NAMES
    )
    reason_codes = []
    reason_codes.extend(
        canonical_context.get("canonical_input_blocking_reason_codes") or []
    )
    reason_codes.extend(
        canonical_feature_mapping_report[
            "canonical_feature_mapping_blocking_reason_codes"
        ]
    )
    reason_codes.extend(
        canonical_scorer_report["canonical_scorer_blocking_reason_codes"]
    )
    reason_codes.extend(
        scorer_comparison_report["scorer_comparison_blocking_reason_codes"]
    )
    passed = (
        canonical_feature_mapping_report["canonical_feature_mapping_complete"] is True
        and canonical_scorer_report["canonical_frozen_o_scorer_used"] is True
        and scorer_comparison_report["scorer_comparison_complete"] is True
        and not reason_codes
    )
    report = {
        "schema_version": O_V8_PAPER_FRESH_CANONICAL_SCORER_ALIGNMENT_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_canonical_scorer_alignment",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "canonical_frozen_o_scorer_invoked": canonical_scorer_report[
            "canonical_frozen_o_scorer_invoked"
        ],
        "canonical_frozen_o_scorer_used": canonical_scorer_report[
            "canonical_frozen_o_scorer_used"
        ],
        "canonical_alignment_diagnostic_status": "passed"
        if passed
        else "blocked_fail_closed",
        "canonical_alignment_blocking_reason_codes": sorted(set(reason_codes)),
        "fresh_provider_scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_scoring_rule_id": (
            "canonical_frozen_o_model_predicted_score_with_frozen_shadow_correction"
        ),
        "feature_schema_matches_canonical_scorer_requirements": (
            canonical_feature_mapping_report["canonical_feature_mapping_complete"]
        ),
        "provider_feature_names": provider_features,
        "canonical_feature_names": feature_names,
        "missing_canonical_feature_names": canonical_feature_mapping_report[
            "missing_canonical_feature_names"
        ],
        "extra_provider_only_feature_names": sorted(
            set(provider_features).difference(feature_names)
        ),
        "canonical_feature_mapping_complete": canonical_feature_mapping_report[
            "canonical_feature_mapping_complete"
        ],
        "canonical_action_row_count": canonical_feature_mapping_report[
            "canonical_action_row_count"
        ],
        "canonical_scored_action_row_count": canonical_scorer_report[
            "canonical_scored_action_row_count"
        ],
        "scorer_comparison_complete": scorer_comparison_report[
            "scorer_comparison_complete"
        ],
        "selected_action_agreement_count": scorer_comparison_report[
            "selected_action_agreement_count"
        ],
        "no_trade_agreement_count": scorer_comparison_report[
            "no_trade_agreement_count"
        ],
        "alignment_decision_rows": scorer_comparison_report[
            "comparison_decision_rows"
        ],
        "source_o_score_mutated": False,
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_canonical_scorer_alignment_report_id"
    )


def _fresh_signal_trace_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    public_cycles: list[list[dict[str, Any]]],
    public_data_collection_report: dict[str, Any],
    canonical_scorer_report: dict[str, Any],
    scorer_comparison_report: dict[str, Any],
    execution_result: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    guard_config = dict(execution_result.get("guard_config") or {})
    provider_by_group: dict[str, dict[str, Any]] = {}
    for cycle_index, cycle in enumerate(public_cycles, start=1):
        cycle_id = f"{config.run_id}-cycle-{cycle_index:06d}"
        for row_index, public_row in enumerate(cycle):
            group_id = str(public_row.get("decision_group_id"))
            provider_by_group[group_id] = {
                **dict(public_row),
                "cycle_id": cycle_id,
                "cycle_index": cycle_index,
                "cycle_row_index": row_index,
            }
    canonical_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_scorer_report.get("canonical_scored_action_rows", []):
        canonical_rows_by_group[str(row.get("decision_group_id"))].append(dict(row))
    canonical_selected_by_group = {
        str(row.get("decision_group_id")): dict(row)
        for row in canonical_scorer_report.get("canonical_selected_decision_rows", [])
    }
    comparison_by_group = {
        str(row.get("decision_group_id")): dict(row)
        for row in scorer_comparison_report.get("comparison_decision_rows", [])
    }
    guard_by_group = {
        str(row.get("decision_group_id")): dict(row)
        for row in execution_result.get("guard_decision_rows", [])
    }
    intent_by_group = {
        str(row.get("decision_group_id")): dict(row) for row in intents
    }
    fill_by_intent = {
        str(row.get("paper_fresh_order_intent_id")): dict(row) for row in fills
    }
    trace_rows: list[dict[str, Any]] = []
    for group_id, provider_row in provider_by_group.items():
        guard_row = guard_by_group.get(group_id, {})
        canonical_selected = canonical_selected_by_group.get(group_id, {})
        canonical_group_rows = sorted(
            canonical_rows_by_group.get(group_id, []),
            key=lambda row: int(row.get("canonical_rank") or 999),
        )
        comparison_row = comparison_by_group.get(group_id, {})
        intent_row = intent_by_group.get(group_id, {})
        fill_row = fill_by_intent.get(str(intent_row.get("paper_fresh_order_intent_id")), {})
        selected_action = str(
            canonical_selected.get("action")
            or guard_row.get("source_selected_action")
            or provider_row.get("selected_action")
            or ""
        )
        selected_family = _action_family(selected_action)
        micro = dict(guard_row.get("microstructure_snapshot") or {})
        if not micro:
            micro = dict(provider_row.get("microstructure_snapshot") or {})
        decision_ts = int(provider_row.get("decision_ts") or guard_row.get("decision_ts") or 0)
        market_schedule = _trace_market_schedule(
            provider_row=provider_row,
            decision_ts=decision_ts,
            micro=micro,
        )
        market_start_ts = market_schedule["market_start_ts"]
        market_end_ts = market_schedule["market_end_ts"]
        time_to_close = _trace_time_to_close_seconds(
            decision_ts=decision_ts,
            market_end_ts=market_end_ts,
            micro=micro,
        )
        elapsed = (
            (decision_ts - market_start_ts) / 1000.0
            if market_start_ts is not None and decision_ts
            else None
        )
        required_min = _trace_required_min_time_to_close_seconds(
            action=selected_action,
            guard_config=guard_config,
        )
        shortfall = (
            max(0.0, required_min - time_to_close)
            if time_to_close is not None
            else required_min
        )
        time_gate_passed = (
            True
            if selected_action == "NO_TRADE"
            else bool(time_to_close is not None and time_to_close >= required_min)
        )
        lifecycle = _trace_lifecycle_window(
            elapsed_since_market_start_seconds=elapsed,
            time_to_close_seconds=time_to_close,
        )
        blocking_reasons = list(guard_row.get("execution_blocking_reason_codes") or [])
        paper_intent_id = intent_row.get("paper_fresh_order_intent_id")
        ranking_for_trace = (
            guard_row.get("top_k_action_ranking")
            or provider_row.get("full_5_action_ranking")
            or []
        )
        action_score_margin = (
            guard_row.get("action_score_margin")
            if guard_row.get("action_score_margin") is not None
            else provider_row.get("action_score_margin")
        )
        if action_score_margin is None:
            action_score_margin = _top_action_margin(ranking_for_trace)
        side_score_margin = (
            guard_row.get("side_specific_action_score_margin")
            if guard_row.get("side_specific_action_score_margin") is not None
            else provider_row.get("side_specific_action_score_margin")
        )
        if side_score_margin is None:
            side_score_margin = _top_action_side_margin(
                ranking_for_trace,
                selected_action=selected_action,
            )
        trace_row = {
            "run_id": config.run_id,
            "cycle_id": provider_row.get("cycle_id"),
            "cycle_index": provider_row.get("cycle_index"),
            "cycle_row_index": provider_row.get("cycle_row_index"),
            "market_id": provider_row.get("market_id"),
            "condition_id": provider_row.get("condition_id"),
            "slug": provider_row.get("slug"),
            "decision_group_id": group_id,
            "market_start_ts": market_start_ts,
            "decision_ts": decision_ts,
            "market_end_ts": market_end_ts,
            "market_schedule_source_type": market_schedule["source_type"],
            "market_schedule_provenance": market_schedule["provenance"],
            "market_schedule_warning_reason_codes": market_schedule[
                "warning_reason_codes"
            ],
            "elapsed_since_market_start_seconds": elapsed,
            "time_since_market_start_seconds": guard_row.get(
                "time_since_market_start_seconds",
                provider_row.get("time_since_market_start_seconds", elapsed),
            ),
            "time_since_market_start_provenance": dict(
                guard_row.get("time_since_market_start_provenance")
                or provider_row.get("time_since_market_start_provenance")
                or {}
            ),
            "time_to_close_seconds": time_to_close,
            "provider_row_source": provider_row.get("provider_row_source")
            or public_data_collection_report.get("public_data_collection_mode"),
            "public_data_source": public_data_collection_report["public_data_source"],
            "simplified_selected_action": provider_row.get("selected_action"),
            "simplified_selected_side": provider_row.get("selected_side"),
            "simplified_selected_family": provider_row.get("selected_action_family"),
            "simplified_corrected_score": provider_row.get("corrected_model_score"),
            "simplified_full_5_action_ranking_summary": (
                _trace_ranking_summary(provider_row.get("full_5_action_ranking") or [])
            ),
            "canonical_scorer_invoked": canonical_scorer_report[
                "canonical_frozen_o_scorer_invoked"
            ],
            "canonical_scorer_used": canonical_scorer_report[
                "canonical_frozen_o_scorer_used"
            ],
            "canonical_selected_action": canonical_selected.get("action"),
            "canonical_selected_side": canonical_selected.get("selected_side"),
            "canonical_selected_family": canonical_selected.get(
                "selected_action_family"
            ),
            "canonical_raw_score": canonical_selected.get("canonical_raw_model_score"),
            "canonical_corrected_score": canonical_selected.get(
                "canonical_corrected_model_score"
            ),
            "canonical_rank": canonical_selected.get("canonical_rank"),
            "canonical_full_5_action_ranking_summary": _trace_canonical_ranking_summary(
                canonical_group_rows
            ),
            "simplified_canonical_selected_action_agrees": comparison_row.get(
                "selected_action_agrees"
            ),
            "simplified_canonical_no_trade_agrees": comparison_row.get(
                "no_trade_selection_agrees"
            ),
            "selected_action_family": selected_family,
            "selected_action_is_hts": selected_family == "HOLD_TO_SETTLEMENT",
            "required_min_time_to_close_seconds": required_min,
            "time_to_close_shortfall_seconds": shortfall,
            "time_to_close_gate_passed": time_gate_passed,
            "lifecycle_window": lifecycle,
            "is_in_hts_allowed_window": bool(
                time_to_close is not None
                and time_to_close
                >= float(guard_config.get("min_hts_time_to_close_seconds") or 120.0)
            ),
            "is_in_sbc_allowed_window": bool(
                time_to_close is not None
                and time_to_close
                >= float(guard_config.get("min_time_to_close_seconds") or 60.0)
            ),
            "is_in_final_no_trade_window": lifecycle == "final_no_trade_window",
            "spread_bps": micro.get("spread_bps"),
            "queue_fill_proxy": micro.get("queue_fill_proxy"),
            "book_staleness_ms": micro.get("book_staleness_ms"),
            "entry_ask": micro.get("entry_ask"),
            "executable_exit_bid_proxy": micro.get("executable_exit_bid_proxy"),
            "p_up": guard_row.get("p_up", provider_row.get("p_up")),
            "p_down": guard_row.get("p_down", provider_row.get("p_down")),
            "p_up_action_disagreement": guard_row.get(
                "p_up_action_disagreement",
                provider_row.get("p_up_action_disagreement"),
            ),
            "btc_momentum": guard_row.get(
                "btc_momentum",
                provider_row.get("btc_momentum"),
            ),
            "btc_momentum_provenance": dict(
                guard_row.get("btc_momentum_provenance")
                or provider_row.get("btc_momentum_provenance")
                or {}
            ),
            "reference_price_to_beat_at_decision": guard_row.get(
                "reference_price_to_beat_at_decision",
                provider_row.get("reference_price_to_beat_at_decision"),
            ),
            "reference_price_to_beat_distance_at_decision": guard_row.get(
                "reference_price_to_beat_distance_at_decision",
                provider_row.get("reference_price_to_beat_distance_at_decision"),
            ),
            "reference_price_to_beat_distance_provenance": dict(
                guard_row.get("reference_price_to_beat_distance_provenance")
                or provider_row.get("reference_price_to_beat_distance_provenance")
                or {}
            ),
            "high_score_flag": guard_row.get(
                "source_high_score_flag",
                canonical_selected.get(
                    "high_score_flag",
                    provider_row.get("high_score_flag"),
                ),
            ),
            "action_score_margin": action_score_margin,
            "score_margin": action_score_margin,
            "action_score_margin_provenance": dict(
                guard_row.get("action_score_margin_provenance")
                or provider_row.get("action_score_margin_provenance")
                or {}
            ),
            "side_specific_action_score_margin": side_score_margin,
            "side_specific_action_score_margin_provenance": dict(
                guard_row.get("side_specific_action_score_margin_provenance")
                or provider_row.get("side_specific_action_score_margin_provenance")
                or {}
            ),
            "decision_time_regime_feature_provenance": dict(
                guard_row.get("decision_time_regime_feature_provenance")
                or provider_row.get("decision_time_regime_feature_provenance")
                or {}
            ),
            "decision_time_regime_feature_max_input_ts": guard_row.get(
                "decision_time_regime_feature_max_input_ts",
                provider_row.get("decision_time_regime_feature_max_input_ts"),
            ),
            **_chainlink_decision_time_field_payload(guard_row, provider_row),
            "execution_guarded_action": guard_row.get("execution_guarded_action"),
            "execution_guarded_side": guard_row.get("execution_guarded_side"),
            "execution_guarded_family": guard_row.get("execution_guarded_family"),
            "execution_guarded_score": guard_row.get("execution_guarded_score"),
            "original_action": guard_row.get("original_action")
            or guard_row.get("source_selected_action"),
            "remapped_action": guard_row.get("remapped_action"),
            "remap_reason_codes": list(guard_row.get("remap_reason_codes") or []),
            "hts_time_window_remap_applied": bool(
                guard_row.get("hts_time_window_remap_applied")
            ),
            "order_allowed": bool(guard_row.get("order_allowed")),
            "proposed_order_size": guard_row.get("proposed_order_size"),
            "paper_intent_id": paper_intent_id,
            "paper_fill_id": fill_row.get("paper_fresh_fill_id"),
            "execution_blocking_reason_codes": blocking_reasons,
            "execution_guard_reason_codes": list(
                guard_row.get("execution_guard_reason_codes") or []
            ),
            "exposure_reason_codes": list(guard_row.get("exposure_reason_codes") or []),
            "missing_runtime_field_codes": list(
                guard_row.get("missing_runtime_field_codes") or []
            ),
            "provenance_violations": list(
                guard_row.get("runtime_field_backfill_provenance_violations") or []
            ),
            "rank_blocked_by_no_trade": selected_action == "NO_TRADE",
            "execution_guard_blocked": bool(blocking_reasons),
            "time_window_blocked": (
                "execution_time_to_close_unsafe" in blocking_reasons
                or (selected_action != "NO_TRADE" and not time_gate_passed)
            ),
            "fail_closed": bool(blocking_reasons) or selected_action == "NO_TRADE",
            "signal_outcome_classification": _trace_signal_outcome_classification(
                selected_action=selected_action,
                order_allowed=bool(guard_row.get("order_allowed")),
                blocking_reasons=blocking_reasons,
            ),
        }
        trace_row["o_v8_paper_fresh_signal_trace_row_hash"] = canonical_json_sha256(
            trace_row
        )
        trace_rows.append(trace_row)
    trace_rows = sorted(
        trace_rows,
        key=lambda row: (
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
            str(row.get("decision_group_id") or ""),
        ),
    )
    aggregate = _fresh_signal_trace_aggregate(
        trace_rows=trace_rows,
        canonical_scorer_report=canonical_scorer_report,
        intents=intents,
        fills=fills,
    )
    report = {
        "schema_version": O_V8_PAPER_FRESH_SIGNAL_TRACE_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_signal_trace",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "public_data_source": public_data_collection_report["public_data_source"],
        "trace_row_count": len(trace_rows),
        "trace_rows_sorted_by_decision_ts": _trace_rows_sorted(trace_rows),
        "trace_rows": trace_rows,
        **aggregate,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_signal_trace_report_id")


def _fresh_time_window_diagnostic_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    signal_trace_report: dict[str, Any],
) -> dict[str, Any]:
    trace_rows = list(signal_trace_report.get("trace_rows") or [])
    report = {
        "schema_version": O_V8_PAPER_FRESH_TIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_time_window_diagnostic",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "trace_row_count": len(trace_rows),
        "rows_by_lifecycle_window": _counter_from_rows(trace_rows, "lifecycle_window"),
        "rows_by_selected_action_family": _counter_from_rows(
            trace_rows, "selected_action_family"
        ),
        "rows_blocked_by_time_to_close": sum(
            1 for row in trace_rows if row.get("time_window_blocked") is True
        ),
        "hts_selected_after_hts_window_expired_count": sum(
            1
            for row in trace_rows
            if row.get("selected_action_is_hts") is True
            and row.get("is_in_hts_allowed_window") is not True
        ),
        "real_action_selected_inside_executable_window_count": sum(
            1
            for row in trace_rows
            if row.get("rank_blocked_by_no_trade") is not True
            and row.get("time_to_close_gate_passed") is True
        ),
        "time_to_close_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="time_to_close_seconds",
        ),
        "time_to_close_shortfall_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="time_to_close_shortfall_seconds",
        ),
        "provider_collection_too_late_for_selected_action_family": (
            signal_trace_report[
                "provider_collection_too_late_for_selected_action_family"
            ]
        ),
        "zero_intent_explanation": signal_trace_report["zero_intent_explanation"],
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_time_window_diagnostic_report_id"
    )


def _fresh_exit_threshold_profile() -> dict[str, Any]:
    return {
        "exit_threshold_profile_name": O_V8_PAPER_FRESH_EXIT_THRESHOLD_PROFILE_NAME,
        "exit_threshold_source": (
            "static_code_constants_for_paper_only_diagnostic_adapter_not_legacy_tuned"
        ),
        "exit_thresholds_tuned": False,
        "exit_threshold_values": {
            "exit_edge_threshold": O_V8_PAPER_FRESH_EXIT_EDGE_THRESHOLD,
            "exit_profit_target": O_V8_PAPER_FRESH_EXIT_PROFIT_TARGET,
            "exit_force_seconds_to_close": (
                O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE
            ),
        },
        "exit_policy_kind": "paper_only_diagnostic_exit_adapter",
        "exit_decision_policy_source": O_V8_PAPER_FRESH_EXIT_DECISION_POLICY_SOURCE,
        "legacy_state_manager_reused": True,
        "legacy_decision_policy_reused": False,
        "exit_thresholds_are_live_execution_thresholds": False,
        "exit_thresholds_are_model_training_thresholds": False,
    }


def _fresh_exit_policy_contract_fields() -> dict[str, Any]:
    profile = _fresh_exit_threshold_profile()
    return {
        "legacy_state_manager_reused": profile["legacy_state_manager_reused"],
        "legacy_decision_policy_reused": profile["legacy_decision_policy_reused"],
        "exit_decision_policy_source": profile["exit_decision_policy_source"],
        "exit_threshold_profile_name": profile["exit_threshold_profile_name"],
        "exit_threshold_source": profile["exit_threshold_source"],
        "exit_thresholds_tuned": profile["exit_thresholds_tuned"],
        "exit_threshold_values": profile["exit_threshold_values"],
        "exit_policy_kind": profile["exit_policy_kind"],
    }


def _fresh_legacy_position_policy_audit_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
) -> dict[str, Any]:
    discovered = [
        {
            "module_or_script": "src/bigan/execution/position_manager.py",
            "functions_or_classes": [
                "PositionManager.open_position",
                "PositionManager.update_price",
                "PositionManager.close_position",
                "PositionManager.get_all_open",
            ],
            "classification": "state_tracking",
            "safe_to_reuse_in_v8_paper_only": True,
            "touches_live_write_wallet_capital": False,
            "uses_settlement_oracle_future_return_fields": False,
            "uses_realized_pnl_labels_for_tuning": False,
            "current_adapter_reuse": "direct_state_manager_reuse",
        },
        {
            "module_or_script": "scripts/replay_v7_settlement_position_manager.py",
            "functions_or_classes": [
                "_replay_positions",
                "_take_profit_candidate",
                "_convergence_evaluation",
                "Decision",
                "SimPosition",
            ],
            "classification": "offline_replay_decision_policy",
            "safe_to_reuse_in_v8_paper_only": False,
            "touches_live_write_wallet_capital": False,
            "uses_settlement_oracle_future_return_fields": True,
            "uses_realized_pnl_labels_for_tuning": False,
            "current_adapter_reuse": "audited_not_reused_script_replay_coupled",
        },
        {
            "module_or_script": "scripts/polymarket_phase4_live_champion_executor.py",
            "functions_or_classes": [
                "_maybe_exit",
                "_sell_settlement_policy_exit",
                "_sell_position",
                "_maybe_v7_settlement_position_adjustment",
            ],
            "classification": "live_execution_decision_policy",
            "safe_to_reuse_in_v8_paper_only": False,
            "touches_live_write_wallet_capital": True,
            "uses_settlement_oracle_future_return_fields": True,
            "uses_realized_pnl_labels_for_tuning": False,
            "current_adapter_reuse": "not_reused_live_write_path",
        },
        {
            "module_or_script": "src/bigan/execution/reconciliation.py",
            "functions_or_classes": ["reconcile_stale_open_positions"],
            "classification": "position_reconciliation",
            "safe_to_reuse_in_v8_paper_only": False,
            "touches_live_write_wallet_capital": False,
            "uses_settlement_oracle_future_return_fields": True,
            "uses_realized_pnl_labels_for_tuning": False,
            "current_adapter_reuse": "not_reused_settlement_reconciliation_path",
        },
        {
            "module_or_script": (
                "src/bigan/v8/polymarket/training/"
                "sell_before_close_exit_reliability.py"
            ),
            "functions_or_classes": [
                "_open_position_row",
                "_exit_reason_code",
                "sell-before-close replay attribution helpers",
            ],
            "classification": "training_replay_label_diagnostic",
            "safe_to_reuse_in_v8_paper_only": False,
            "touches_live_write_wallet_capital": False,
            "uses_settlement_oracle_future_return_fields": True,
            "uses_realized_pnl_labels_for_tuning": True,
            "current_adapter_reuse": "not_reused_training_replay_label_path",
        },
    ]
    report = {
        "schema_version": O_V8_PAPER_FRESH_LEGACY_POSITION_POLICY_AUDIT_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_legacy_position_policy_audit",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "audit_search_terms": [
            "position",
            "exit",
            "close",
            "sell",
            "hold",
            "take_profit",
            "stop_loss",
            "unrealized",
            "drawdown",
            "PositionManager",
        ],
        "discovered_modules_and_functions": discovered,
        "reusable_legacy_decision_policy_found": False,
        "legacy_decision_policy_reused": False,
        "legacy_state_manager_reused": True,
        "exit_decision_policy_source": O_V8_PAPER_FRESH_EXIT_DECISION_POLICY_SOURCE,
        "exit_policy_kind": "paper_only_diagnostic_exit_adapter",
        "exit_threshold_profile": _fresh_exit_threshold_profile(),
        "exit_thresholds_tuned": False,
        "audit_conclusion": (
            "Only PositionManager state lifecycle is reused directly; legacy "
            "decision policies are script/live/replay coupled and remain audit-only."
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(
        report, "o_v8_paper_fresh_legacy_position_policy_audit_report_id"
    )


def _fresh_paper_exit_adapter_bundle(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    signal_trace_report: dict[str, Any],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_rows = list(signal_trace_report.get("trace_rows") or [])
    forbidden_rows = [
        *_rows_with_forbidden_exit_fields(list(config.initial_paper_position_rows)),
        *_rows_with_forbidden_exit_fields(trace_rows),
        *_rows_with_forbidden_exit_fields(fills),
        *_rows_with_forbidden_exit_fields(ledger_rows),
    ]
    conn = duckdb.connect(":memory:")
    manager = PositionManager(conn=conn)
    try:
        opened_rows = _fresh_open_legacy_positions(
            config=config,
            manager=manager,
            fills=fills,
            forbidden_rows=forbidden_rows,
        )
        position_open_failure_rows = opened_rows["position_open_failure_rows"]
        exit_signal_rows, sell_intents, ledger_update_rows = (
            _fresh_evaluate_legacy_position_exits(
                config=config,
                manager=manager,
                trace_rows=trace_rows,
                forbidden_rows=forbidden_rows,
                position_open_failure_rows=position_open_failure_rows,
            )
        )
        position_rows = _fresh_position_state_rows(
            manager=manager,
            opened_rows=opened_rows["opened_position_rows"],
            trace_rows=trace_rows,
        )
    finally:
        conn.close()
    position_report = _fresh_paper_position_state_report(
        config=config,
        opened_rows=opened_rows["opened_position_rows"],
        position_rows=position_rows,
        ledger_rows=ledger_rows,
        forbidden_rows=forbidden_rows,
        position_open_failure_rows=position_open_failure_rows,
    )
    exit_signal_report = _fresh_paper_exit_signal_report(
        config=config,
        trace_rows=trace_rows,
        exit_signal_rows=exit_signal_rows,
        sell_intents=sell_intents,
        forbidden_rows=forbidden_rows,
    )
    ledger_report = _fresh_synthetic_exit_ledger_update_report(
        config=config,
        sell_intents=sell_intents,
        ledger_update_rows=ledger_update_rows,
    )
    return {
        "paper_position_state_report": position_report,
        "paper_exit_signal_report": exit_signal_report,
        "paper_sell_position_intents": sell_intents,
        "synthetic_ledger_update_report": ledger_report,
    }


def _fresh_open_legacy_positions(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    manager: PositionManager,
    fills: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if forbidden_rows:
        return {"opened_position_rows": [], "position_open_failure_rows": []}
    opened_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for index, row in enumerate(config.initial_paper_position_rows, start=1):
        opened, failure = _fresh_open_legacy_position_from_row(
            manager=manager,
            row=dict(row),
            row_index=index,
            source="initial_paper_position_rows",
        )
        if opened is not None:
            opened_rows.append(opened)
        if failure is not None:
            failure_rows.append(failure)
    for index, fill in enumerate(fills, start=1):
        opened, failure = _fresh_open_legacy_position_from_row(
                manager=manager,
            row=_fresh_initial_position_row_from_fill(fill),
                row_index=index,
            source="accepted_paper_entry_fill",
        )
        if opened is not None:
            opened_rows.append(opened)
        if failure is not None:
            failure_rows.append(failure)
    return {
        "opened_position_rows": opened_rows,
        "position_open_failure_rows": failure_rows,
    }


def _fresh_open_legacy_position_from_row(
    *,
    manager: PositionManager,
    row: dict[str, Any],
    row_index: int,
    source: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    market_id = str(row.get("market_id") or row.get("event_id") or "")
    side = str(row.get("side") or row.get("execution_guarded_side") or "").upper()
    entry_time = int(row.get("entry_time") or row.get("decision_ts") or 0)
    entry_price = _float(row.get("entry_price") or row.get("fill_price"))
    size = _float(row.get("size") or row.get("filled_size"))
    event_id = str(row.get("event_id") or f"{source}-{market_id}-{side}-{row_index:06d}")
    symbol = str(row.get("symbol") or f"POLYMARKET:{market_id}:{side}")
    fill_price = row.get("fill_price")
    validation_reasons = _fresh_validate_legacy_position_open_row(
        row=row,
        market_id=market_id,
        event_id=event_id,
        side=side,
        entry_price=entry_price,
        size=size,
    )
    if validation_reasons:
        return None, _fresh_position_open_failure_row(
            row=row,
            row_index=row_index,
            source=source,
            market_id=market_id,
            event_id=event_id,
            side=side,
            reason_codes=validation_reasons,
            error_message=None,
        )
    try:
        manager.open_position(
            event_id=event_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            order_id=str(row.get("order_id") or f"{source}-order-{row_index:06d}"),
            sleeve=str(row.get("sleeve") or "volatility"),
            entry_time=entry_time,
            fill_price=_float(fill_price) if fill_price is not None else entry_price,
        )
    except ValueError as exc:
        reason_codes = ["position_open_failed_exception"]
        if "already exists" in str(exc):
            reason_codes.append("position_open_duplicate_event_id")
        return None, _fresh_position_open_failure_row(
            row=row,
            row_index=row_index,
            source=source,
            market_id=market_id,
            event_id=event_id,
            side=side,
            reason_codes=reason_codes,
            error_message=str(exc),
        )
    opened_row = {
        "legacy_position_event_id": event_id,
        "position_source": source,
        "market_id": market_id,
        "side": side,
        "symbol": symbol,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "fill_price": _float(fill_price) if fill_price is not None else entry_price,
        "size": size,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    opened_row["paper_position_open_row_hash"] = canonical_json_sha256(opened_row)
    return opened_row, None


def _fresh_initial_position_row_from_fill(fill: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": f"fresh-fill-{fill.get('paper_fresh_fill_id')}",
        "market_id": fill.get("market_id"),
        "symbol": (
            f"POLYMARKET:{fill.get('market_id')}:{fill.get('execution_guarded_side')}"
        ),
        "side": fill.get("execution_guarded_side"),
        "entry_time": fill.get("decision_ts"),
        "entry_price": fill.get("paper_fill_price"),
        "fill_price": fill.get("paper_fill_price"),
        "size": fill.get("filled_size"),
        "order_id": fill.get("paper_fresh_order_intent_id"),
        "sleeve": "volatility",
    }


def _fresh_validate_legacy_position_open_row(
    *,
    row: dict[str, Any],
    market_id: str,
    event_id: str,
    side: str,
    entry_price: float,
    size: float,
) -> list[str]:
    reasons: list[str] = []
    if not str(market_id).strip() and not str(event_id).strip():
        reasons.append("position_open_missing_market_or_event_id")
    if (
        row.get("event_id") is None
        and row.get("market_id") is None
        and row.get("condition_id") is None
    ):
        reasons.append("position_open_missing_market_or_event_id")
    if side not in {"UP", "DOWN"}:
        reasons.append("position_open_invalid_side")
    if entry_price <= 0.0:
        reasons.append("position_open_non_positive_entry_price")
    if size <= 0.0:
        reasons.append("position_open_non_positive_size")
    return sorted(set(reasons))


def _fresh_position_open_failure_row(
    *,
    row: dict[str, Any],
    row_index: int,
    source: str,
    market_id: str,
    event_id: str,
    side: str,
    reason_codes: list[str],
    error_message: str | None,
) -> dict[str, Any]:
    failure = {
        "position_open_failure_row_id": f"position-open-failure-{row_index:06d}",
        "position_source": source,
        "row_index": row_index,
        "market_id": market_id,
        "event_id": event_id,
        "side": side,
        "position_open_failure_reason_codes": sorted(set(reason_codes)),
        "position_open_error_message": error_message,
        "raw_row_hash": canonical_json_sha256(row),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    failure["position_open_failure_row_hash"] = canonical_json_sha256(failure)
    return failure


def _fresh_evaluate_legacy_position_exits(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    manager: PositionManager,
    trace_rows: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
    position_open_failure_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    open_positions = manager.get_all_open()
    if not open_positions:
        reason_codes = ["no_open_paper_position"]
        if forbidden_rows:
            reason_codes.append("forbidden_outcome_field_present_fail_closed")
        if position_open_failure_rows:
            reason_codes.append("position_open_failure_present_fail_closed")
        no_exit = {
            "paper_exit_signal_row_id": f"{config.run_id}-exit-signal-none",
            "run_id": config.run_id,
            "paper_exit_decision": "NO_EXIT",
            "exit_decision": "NO_EXIT",
            "exit_reason_codes": sorted(set(reason_codes)),
            "accepted_for_paper_exit_intent": False,
            "legacy_position_manager_operations": [],
            **_fresh_exit_policy_contract_fields(),
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        no_exit["paper_exit_signal_row_hash"] = canonical_json_sha256(no_exit)
        return [no_exit], [], []
    signal_rows: list[dict[str, Any]] = []
    sell_intents: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trace_by_market = _fresh_trace_rows_by_market(trace_rows)
    for position in open_positions:
        position_payload = position.to_row()
        trace_row = _fresh_latest_post_entry_trace_for_position(
            position_payload=position_payload,
            trace_by_market=trace_by_market,
        )
        signal_row = _fresh_legacy_exit_signal_row(
            config=config,
            manager=manager,
            position_payload=position_payload,
            trace_row=trace_row,
            forbidden_rows=forbidden_rows,
            position_open_failure_rows=position_open_failure_rows,
            signal_index=len(signal_rows) + 1,
        )
        signal_rows.append(signal_row)
        if signal_row["paper_exit_decision"] == "SELL_POSITION":
            intent = _fresh_sell_position_intent_from_signal(
                config=config,
                signal_row=signal_row,
                intent_index=len(sell_intents) + 1,
            )
            sell_intents.append(intent)
            ledger_rows.append(
                _fresh_synthetic_exit_ledger_update_from_intent(
                    intent=intent,
                    row_index=len(ledger_rows) + 1,
                )
            )
    return signal_rows, sell_intents, ledger_rows


def _fresh_trace_rows_by_market(
    trace_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[str(row.get("market_id") or "")].append(dict(row))
    return {
        market_id: sorted(rows, key=lambda row: int(row.get("decision_ts") or 0))
        for market_id, rows in grouped.items()
    }


def _fresh_latest_post_entry_trace_for_position(
    *,
    position_payload: dict[str, Any],
    trace_by_market: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    market_id = _fresh_market_id_from_position(position_payload)
    entry_time = int(position_payload.get("entry_time") or 0)
    side = str(position_payload.get("side") or "")
    candidates = [
        row
        for row in trace_by_market.get(market_id, [])
        if int(row.get("decision_ts") or 0) > entry_time
        and str(row.get("execution_guarded_side") or row.get("canonical_selected_side") or "")
        in {side, "NONE", ""}
    ]
    if not candidates:
        return None
    return candidates[-1]


def _fresh_legacy_exit_signal_row(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    manager: PositionManager,
    position_payload: dict[str, Any],
    trace_row: dict[str, Any] | None,
    forbidden_rows: list[dict[str, Any]],
    position_open_failure_rows: list[dict[str, Any]],
    signal_index: int,
) -> dict[str, Any]:
    position_side = str(position_payload.get("side") or "").upper()
    entry_price = _float(position_payload.get("fill_price") or position_payload.get("entry_price"))
    size = _float(position_payload.get("size"))
    reason_codes: list[str] = []
    decision = "HOLD_POSITION"
    exit_price = None
    p_side = None
    hold_edge = None
    time_to_close = None
    if forbidden_rows:
        reason_codes.append("forbidden_outcome_field_present_fail_closed")
    elif position_open_failure_rows:
        reason_codes.append("position_open_failure_present_fail_closed")
    elif trace_row is None:
        reason_codes.append("no_post_entry_signal_trace_for_open_position")
    else:
        exit_price = _fresh_exit_bid_from_trace(trace_row)
        time_to_close = _trace_float_or_none(trace_row.get("time_to_close_seconds"))
        p_side = _fresh_probability_for_position_side(trace_row, position_side)
        if exit_price is None:
            reason_codes.append("missing_decision_time_exit_bid_proxy")
        else:
            manager.update_price(str(position_payload["event_id"]), exit_price)
            hold_edge = None if p_side is None else p_side - exit_price
            sell, sell_reasons = _fresh_paper_adapter_sell_decision(
                exit_price=exit_price,
                entry_price=entry_price,
                p_side=p_side,
                time_to_close=time_to_close,
            )
            if sell:
                decision = "SELL_POSITION"
                reason_codes.extend(sell_reasons)
                closed = manager.close_position(
                    str(position_payload["event_id"]),
                    exit_price,
                    exit_time=int(trace_row.get("decision_ts") or 0),
                )
                position_payload = closed.to_row()
            else:
                reason_codes.extend(
                    sell_reasons or ["paper_adapter_hold_edge_above_exit_threshold"]
                )
    row = {
        "paper_exit_signal_row_id": f"{config.run_id}-exit-signal-{signal_index:06d}",
        "run_id": config.run_id,
        "legacy_adapter_rule_id": "v8_paper_legacy_position_manager_exit_adapter_v1",
        "legacy_position_manager_reused": True,
        **_fresh_exit_policy_contract_fields(),
        "legacy_position_manager_module": "bigan.execution.position_manager.PositionManager",
        "legacy_position_manager_operations": _fresh_legacy_operations_for_decision(
            decision=decision,
            trace_row=trace_row,
            exit_price=exit_price,
        ),
        "legacy_signal_logic_reference": "audited_not_directly_reused",
        "legacy_position_event_id": position_payload.get("event_id"),
        "market_id": _fresh_market_id_from_position(position_payload),
        "symbol": position_payload.get("symbol"),
        "side": position_side,
        "position_status_after_decision": position_payload.get("status"),
        "entry_time": position_payload.get("entry_time"),
        "entry_price": entry_price,
        "fill_price": _float(position_payload.get("fill_price") or entry_price),
        "size": size,
        "decision_ts": None if trace_row is None else trace_row.get("decision_ts"),
        "matched_signal_trace_row_hash": None
        if trace_row is None
        else trace_row.get("o_v8_paper_fresh_signal_trace_row_hash"),
        "paper_exit_decision": decision,
        "exit_decision": decision,
        "accepted_for_paper_exit_intent": decision == "SELL_POSITION",
        "exit_price": exit_price,
        "exit_size": size if decision == "SELL_POSITION" else 0.0,
        "current_mark_price": exit_price,
        "synthetic_unrealized_paper_pnl": None
        if exit_price is None
        else (exit_price - entry_price) * size,
        "position_side_probability": p_side,
        "position_hold_edge": hold_edge,
        "time_to_close_seconds": time_to_close,
        "exit_reason_codes": reason_codes,
        "legacy_consumed_position_fields": _fresh_legacy_position_fields_consumed(),
        "legacy_consumed_signal_fields": _fresh_legacy_signal_fields_consumed(),
        "mapped_signal_input": _fresh_mapped_legacy_signal_input(
            trace_row=trace_row,
            position_side=position_side,
            exit_price=exit_price,
            p_side=p_side,
            hold_edge=hold_edge,
        ),
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "outcome_pnl_used": False,
        "mutates_o_entry_scorer": False,
        "mutates_source_ranking_scores": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    row["paper_exit_signal_row_hash"] = canonical_json_sha256(row)
    return row


def _fresh_paper_adapter_sell_decision(
    *,
    exit_price: float,
    entry_price: float,
    p_side: float | None,
    time_to_close: float | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if (
        time_to_close is not None
        and time_to_close <= O_V8_PAPER_FRESH_EXIT_FORCE_SECONDS_TO_CLOSE
    ):
        reasons.append("paper_adapter_force_exit_window")
    if exit_price - entry_price >= O_V8_PAPER_FRESH_EXIT_PROFIT_TARGET:
        reasons.append("paper_adapter_profit_target_crossed")
    if p_side is not None and p_side - exit_price <= O_V8_PAPER_FRESH_EXIT_EDGE_THRESHOLD:
        reasons.append("paper_adapter_exit_edge_threshold_crossed")
    if reasons:
        return True, reasons
    return False, ["paper_adapter_hold_position"]


def _fresh_legacy_operations_for_decision(
    *,
    decision: str,
    trace_row: dict[str, Any] | None,
    exit_price: float | None,
) -> list[str]:
    operations = ["open_position"]
    if trace_row is not None and exit_price is not None:
        operations.append("update_price")
    if decision == "SELL_POSITION":
        operations.append("close_position")
    return operations


def _fresh_sell_position_intent_from_signal(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    signal_row: dict[str, Any],
    intent_index: int,
) -> dict[str, Any]:
    intent = {
        "paper_sell_position_intent_id": (
            f"{config.run_id}-sell-position-intent-{intent_index:06d}"
        ),
        "paper_exit_signal_row_id": signal_row["paper_exit_signal_row_id"],
        "legacy_position_event_id": signal_row["legacy_position_event_id"],
        "market_id": signal_row["market_id"],
        "decision_ts": signal_row["decision_ts"],
        "side": signal_row["side"],
        "paper_exit_decision": "SELL_POSITION",
        "sell_position_intent_status": "accepted_local_paper_exit_intent",
        "sell_size": _float(signal_row.get("exit_size")),
        "paper_limit_price": _float(signal_row.get("exit_price")),
        "synthetic_exit_notional": _float(signal_row.get("exit_size"))
        * _float(signal_row.get("exit_price")),
        "exit_reason_codes": list(signal_row.get("exit_reason_codes") or []),
        **_fresh_exit_policy_contract_fields(),
        "order_intent_contract": "local_paper_sell_position_intent_no_exchange_write_v1",
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "outcome_pnl_used": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    intent["paper_sell_position_intent_hash"] = canonical_json_sha256(intent)
    return intent


def _fresh_synthetic_exit_ledger_update_from_intent(
    *,
    intent: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    size = _float(intent.get("sell_size"))
    exit_price = _float(intent.get("paper_limit_price"))
    row = {
        "synthetic_exit_ledger_update_id": f"fresh-paper-exit-ledger-{row_index:06d}",
        "paper_sell_position_intent_id": intent["paper_sell_position_intent_id"],
        "paper_exit_signal_row_id": intent["paper_exit_signal_row_id"],
        "legacy_position_event_id": intent["legacy_position_event_id"],
        "market_id": intent["market_id"],
        "decision_ts": intent["decision_ts"],
        "side": intent["side"],
        "synthetic_cash_delta": size * exit_price,
        "synthetic_position_delta": -size,
        "paper_position_closed": True,
        "ledger_update_source": "accepted_local_paper_sell_position_intent",
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "outcome_pnl_used": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    row["synthetic_exit_ledger_update_hash"] = canonical_json_sha256(row)
    return row


def _fresh_position_state_rows(
    *,
    manager: PositionManager,
    opened_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_trace_by_market = {
        market_id: rows[-1]
        for market_id, rows in _fresh_trace_rows_by_market(trace_rows).items()
        if rows
    }
    source_by_event_id = {
        row["legacy_position_event_id"]: row for row in opened_rows
    }
    rows: list[dict[str, Any]] = []
    for index, position in enumerate(manager.list_positions(), start=1):
        payload = position.to_row()
        market_id = _fresh_market_id_from_position(payload)
        trace_row = latest_trace_by_market.get(market_id)
        row = {
            "paper_position_state_row_id": f"paper-position-state-{index:06d}",
            "legacy_position_event_id": payload["event_id"],
            "legacy_position_manager_reused": True,
            "position_source": source_by_event_id.get(payload["event_id"], {}).get(
                "position_source"
            ),
            "market_id": market_id,
            "symbol": payload["symbol"],
            "side": payload["side"],
            "sleeve": payload["sleeve"],
            "status": payload["status"],
            "entry_time": payload["entry_time"],
            "entry_price": payload["entry_price"],
            "fill_price": payload["fill_price"],
            "size": payload["size"],
            "current_price": payload["current_price"],
            "synthetic_unrealized_paper_pnl": payload["unrealized_pnl"],
            "exit_price": payload["exit_price"],
            "exit_time": payload["exit_time"],
            "matched_latest_signal_trace_row_hash": None
            if trace_row is None
            else trace_row.get("o_v8_paper_fresh_signal_trace_row_hash"),
            "eligible_for_exit_signal": bool(
                payload["status"] == "open"
                and trace_row is not None
                and int(trace_row.get("decision_ts") or 0) > int(payload["entry_time"])
            ),
            "uses_settlement_oracle_future_return_fields": False,
            "uses_input_realized_pnl_fields": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        row["paper_position_state_row_hash"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _fresh_paper_position_state_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    opened_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
    position_open_failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blocking_reason_codes = []
    if forbidden_rows:
        blocking_reason_codes.append("forbidden_outcome_field_present_fail_closed")
    if position_open_failure_rows:
        blocking_reason_codes.append("position_open_failure_present_fail_closed")
    report = {
        "schema_version": O_V8_PAPER_FRESH_POSITION_STATE_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_position_state",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "legacy_position_manager_reused": True,
        **_fresh_exit_policy_contract_fields(),
        "legacy_position_manager_module": "bigan.execution.position_manager.PositionManager",
        "legacy_position_manager_consumed_fields": _fresh_legacy_position_fields_consumed(),
        "paper_ledger_fields_consumed_for_position_mapping": (
            _fresh_paper_ledger_fields_consumed_for_position_mapping()
        ),
        "paper_ledger_row_count": len(ledger_rows),
        "paper_ledger_rows_checked_for_forbidden_outcomes": True,
        "opened_position_rows": opened_rows,
        "position_state_rows": position_rows,
        "position_open_failed_count": len(position_open_failure_rows),
        "position_open_failure_rows": position_open_failure_rows,
        "initial_paper_position_row_count": len(config.initial_paper_position_rows),
        "accepted_entry_fill_position_count": sum(
            1 for row in opened_rows if row.get("position_source") == "accepted_paper_entry_fill"
        ),
        "paper_position_count": len(position_rows),
        "open_paper_position_count": sum(
            1 for row in position_rows if row.get("status") == "open"
        ),
        "closed_paper_position_count": sum(
            1 for row in position_rows if row.get("status") == "closed"
        ),
        "eligible_for_exit_signal_count": sum(
            1 for row in position_rows if row.get("eligible_for_exit_signal") is True
        ),
        "forbidden_outcome_fields_present": bool(forbidden_rows),
        "forbidden_outcome_field_rows": forbidden_rows,
        "position_state_adapter_status": "blocked_fail_closed"
        if blocking_reason_codes
        else "passed",
        "position_state_blocking_reason_codes": sorted(set(blocking_reason_codes)),
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "mutates_o_entry_scorer": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_position_state_report_id")


def _fresh_paper_exit_signal_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    trace_rows: list[dict[str, Any]],
    exit_signal_rows: list[dict[str, Any]],
    sell_intents: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": O_V8_PAPER_FRESH_EXIT_SIGNAL_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_exit_signal",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "legacy_position_manager_reused": True,
        **_fresh_exit_policy_contract_fields(),
        "legacy_position_manager_module": "bigan.execution.position_manager.PositionManager",
        "legacy_signal_logic_reference": "audited_not_directly_reused",
        "legacy_consumed_signal_fields": _fresh_legacy_signal_fields_consumed(),
        "signal_trace_row_count": len(trace_rows),
        "paper_exit_signal_rows": exit_signal_rows,
        "paper_exit_signal_count": len(exit_signal_rows),
        "hold_no_exit_count": sum(
            1
            for row in exit_signal_rows
            if row.get("paper_exit_decision") in {"HOLD_POSITION", "NO_EXIT"}
        ),
        "sell_position_signal_count": sum(
            1 for row in exit_signal_rows if row.get("paper_exit_decision") == "SELL_POSITION"
        ),
        "sell_position_intent_count": len(sell_intents),
        "sell_position_intents_are_local_paper_only": all(
            row.get("paper_only") is True
            and row.get("capital_at_risk") is False
            and row.get("polymarket_write_enabled") is False
            and row.get("wallet_signing_enabled") is False
            for row in sell_intents
        ),
        "exit_reason_distribution": _counter_from_rows(
            exit_signal_rows, "exit_reason_codes"
        ),
        "forbidden_outcome_fields_present": bool(forbidden_rows),
        "forbidden_outcome_field_rows": forbidden_rows,
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "outcome_pnl_used": False,
        "mutates_o_entry_scorer": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_exit_signal_report_id")


def _fresh_synthetic_exit_ledger_update_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    sell_intents: list[dict[str, Any]],
    ledger_update_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_intent_ids = {
        str(row.get("paper_sell_position_intent_id")) for row in sell_intents
    }
    ledger_intent_ids = {
        str(row.get("paper_sell_position_intent_id")) for row in ledger_update_rows
    }
    report = {
        "schema_version": O_V8_PAPER_FRESH_EXIT_LEDGER_UPDATE_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_synthetic_ledger_update",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        **_fresh_exit_policy_contract_fields(),
        "synthetic_ledger_update_rows": ledger_update_rows,
        "synthetic_ledger_update_count": len(ledger_update_rows),
        "paper_sell_position_intent_count": len(sell_intents),
        "ledger_updates_only_for_accepted_paper_exit_intents": (
            accepted_intent_ids == ledger_intent_ids
        ),
        "synthetic_cash_delta_sum": sum(
            _float(row.get("synthetic_cash_delta")) for row in ledger_update_rows
        ),
        "synthetic_position_delta_sum": sum(
            _float(row.get("synthetic_position_delta")) for row in ledger_update_rows
        ),
        "uses_settlement_oracle_future_return_fields": False,
        "uses_input_realized_pnl_fields": False,
        "outcome_pnl_used": False,
        "mutates_o_entry_scorer": False,
        "mutates_source_ranking_scores": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_synthetic_ledger_update_report_id")


def _fresh_exit_bid_from_trace(trace_row: dict[str, Any]) -> float | None:
    for field_name in ("executable_exit_bid_proxy", "paper_limit_price", "entry_ask"):
        value = trace_row.get(field_name)
        if value is not None:
            return _float(value)
    return None


def _fresh_probability_for_position_side(
    trace_row: dict[str, Any],
    side: str,
) -> float | None:
    if side == "UP" and trace_row.get("p_up") is not None:
        return _float(trace_row.get("p_up"))
    if side == "DOWN" and trace_row.get("p_down") is not None:
        return _float(trace_row.get("p_down"))
    return None


def _fresh_mapped_legacy_signal_input(
    *,
    trace_row: dict[str, Any] | None,
    position_side: str,
    exit_price: float | None,
    p_side: float | None,
    hold_edge: float | None,
) -> dict[str, Any]:
    if trace_row is None:
        return {}
    return {
        "event_id": trace_row.get("decision_group_id"),
        "round_slug": trace_row.get("slug") or trace_row.get("market_id"),
        "side": position_side,
        "ts_ms": trace_row.get("decision_ts"),
        "created_at_ms": trace_row.get("decision_ts"),
        "round_end_ts_ms": trace_row.get("market_end_ts"),
        "token_probability": p_side,
        "p_up": trace_row.get("p_up"),
        "p_down": trace_row.get("p_down"),
        "polymarket_price": exit_price,
        "mispricing_edge": hold_edge,
    }


def _fresh_legacy_position_fields_consumed() -> list[str]:
    return [
        "event_id",
        "symbol",
        "side",
        "sleeve",
        "status",
        "entry_time",
        "entry_price",
        "fill_price",
        "size",
        "order_id",
        "current_price",
        "unrealized_pnl",
        "exit_price",
        "exit_time",
    ]


def _fresh_legacy_signal_fields_consumed() -> list[str]:
    return [
        "decision_group_id",
        "market_id",
        "slug",
        "decision_ts",
        "market_end_ts",
        "p_up",
        "p_down",
        "time_to_close_seconds",
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "executable_exit_bid_proxy",
        "entry_ask",
        "selected_action_family",
        "execution_guarded_action",
        "execution_guarded_side",
    ]


def _fresh_paper_ledger_fields_consumed_for_position_mapping() -> list[str]:
    return [
        "paper_fresh_ledger_entry_id",
        "paper_fresh_fill_id",
        "paper_fresh_order_intent_id",
        "market_id",
        "decision_ts",
        "execution_guarded_action",
        "execution_guarded_side",
        "synthetic_position_after",
        "total_exposure_after",
        "side_exposure_after",
    ]


def _fresh_market_id_from_position(position_payload: dict[str, Any]) -> str:
    symbol = str(position_payload.get("symbol") or "")
    parts = symbol.split(":")
    if len(parts) >= 3:
        return parts[1]
    event_id = str(position_payload.get("event_id") or "")
    if event_id.startswith("fresh-fill-"):
        return event_id.removeprefix("fresh-fill-")
    return str(position_payload.get("market_id") or event_id)


def _rows_with_forbidden_exit_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        present = sorted(
            field_name
            for field_name in O_V8_PAPER_FRESH_EXIT_FORBIDDEN_OUTCOME_FIELDS
            if _forbidden_field_present(row, field_name)
        )
        if present:
            failures.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts") or row.get("entry_time"),
                    "forbidden_fields": present,
                }
            )
    return failures


def _forbidden_field_present(row: dict[str, Any], field_name: str) -> bool:
    if field_name not in row:
        return False
    value = row.get(field_name)
    return value not in (None, "", [], {})


def _fresh_signal_trace_aggregate(
    *,
    trace_rows: list[dict[str, Any]],
    canonical_scorer_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_buy_rows = [
        row
        for row in trace_rows
        if row.get("canonical_selected_action")
        and row.get("canonical_selected_action") != "NO_TRADE"
    ]
    canonical_real_rows = [
        row for row in trace_rows if row.get("selected_action_family") != "NO_TRADE"
    ]
    executable_rows = [
        row for row in canonical_real_rows if row.get("time_to_close_gate_passed") is True
    ]
    hts_expired_rows = [
        row
        for row in canonical_real_rows
        if row.get("selected_action_is_hts") is True
        and row.get("is_in_hts_allowed_window") is not True
    ]
    time_blocked_rows = [
        row for row in trace_rows if row.get("time_window_blocked") is True
    ]
    aggregate = {
        "total_provider_decision_count": len(trace_rows),
        "canonical_selected_decision_count": canonical_scorer_report[
            "canonical_selected_decision_count"
        ],
        "paper_intent_count": len(intents),
        "fill_count": len(fills),
        "rows_by_lifecycle_window": _counter_from_rows(trace_rows, "lifecycle_window"),
        "rows_by_selected_action_family": _counter_from_rows(
            trace_rows, "selected_action_family"
        ),
        "rows_by_execution_blocking_reason": _counter_from_rows(
            trace_rows, "execution_blocking_reason_codes"
        ),
        "rows_blocked_by_time_to_close": len(time_blocked_rows),
        "rows_blocked_by_spread": _trace_block_count(
            trace_rows, "execution_spread_too_wide"
        ),
        "rows_blocked_by_staleness": _trace_block_count(
            trace_rows, "execution_book_stale"
        ),
        "rows_blocked_by_liquidity": _trace_block_count(
            trace_rows, "execution_liquidity_too_weak"
        ),
        "rows_blocked_by_p_up_disagreement": _trace_block_count(
            trace_rows, "execution_p_up_side_disagreement"
        ),
        "rows_blocked_by_exposure_cooldown_duplicate": sum(
            1
            for row in trace_rows
            if any(
                token in reason
                for reason in row.get("execution_blocking_reason_codes") or []
                for token in ("exposure", "cooldown", "duplicate")
            )
        ),
        "rows_with_missing_runtime_fields": sum(
            1 for row in trace_rows if row.get("missing_runtime_field_codes")
        ),
        "rows_with_provenance_violations": sum(
            1 for row in trace_rows if row.get("provenance_violations")
        ),
        "time_to_close_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="time_to_close_seconds",
        ),
        "score_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="canonical_corrected_score",
        ),
        "spread_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="spread_bps",
        ),
        "queue_fill_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="queue_fill_proxy",
        ),
        "book_staleness_by_action_family": _trace_numeric_summary_by_field(
            trace_rows,
            group_field="selected_action_family",
            value_field="book_staleness_ms",
        ),
        "proportion_canonical_signals_inside_executable_window": (
            len(executable_rows) / len(canonical_real_rows)
            if canonical_real_rows
            else 0.0
        ),
        "proportion_canonical_signals_after_hts_window_expired": (
            len(hts_expired_rows) / len(canonical_real_rows)
            if canonical_real_rows
            else 0.0
        ),
        "provider_collection_too_late_for_selected_action_family": bool(
            canonical_real_rows and len(executable_rows) == 0 and time_blocked_rows
        ),
        "zero_intent_explanation": {
            "canonical_buy_side_signal_count": len(canonical_buy_rows),
            "canonical_selected_action_family_distribution": _counter_from_rows(
                trace_rows, "selected_action_family"
            ),
            "signals_inside_allowed_execution_window_count": len(executable_rows),
            "time_to_close_blocked_count": len(time_blocked_rows),
            "non_time_guard_blocking_reason_distribution": _trace_non_time_blockers(
                trace_rows
            ),
            "likely_collection_cadence_or_market_window_issue": bool(
                canonical_buy_rows and len(executable_rows) == 0 and time_blocked_rows
            ),
        },
    }
    return aggregate


def _trace_ranking_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(row) for row in rows],
        key=lambda row: float(row.get("corrected_model_score") or 0.0),
        reverse=True,
    )
    return [
        {
            "rank": index,
            "action": row.get("selected_action") or row.get("action"),
            "side": row.get("selected_side"),
            "family": row.get("selected_action_family"),
            "corrected_score": row.get("corrected_model_score"),
            "raw_score": row.get("raw_model_score"),
        }
        for index, row in enumerate(ordered, start=1)
    ]


def _trace_canonical_ranking_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": row.get("canonical_rank"),
            "action": row.get("action"),
            "side": row.get("selected_side"),
            "family": row.get("selected_action_family"),
            "canonical_raw_score": row.get("canonical_raw_model_score"),
            "canonical_corrected_score": row.get("canonical_corrected_model_score"),
            "high_score_flag": row.get("high_score_flag"),
        }
        for row in sorted(rows, key=lambda item: int(item.get("canonical_rank") or 999))
    ]


def _trace_required_min_time_to_close_seconds(
    *,
    action: str,
    guard_config: dict[str, Any],
) -> float:
    family = _action_family(action)
    if family == "NO_TRADE":
        return 0.0
    if family == "HOLD_TO_SETTLEMENT":
        return float(guard_config.get("min_hts_time_to_close_seconds") or 120.0)
    return float(guard_config.get("min_time_to_close_seconds") or 60.0)


def _trace_lifecycle_window(
    *,
    elapsed_since_market_start_seconds: float | None,
    time_to_close_seconds: float | None,
) -> str:
    if time_to_close_seconds is None:
        return "pre_market_or_invalid_time"
    if time_to_close_seconds < 0:
        return "post_market_close"
    if elapsed_since_market_start_seconds is not None and elapsed_since_market_start_seconds < 0:
        return "pre_market_or_invalid_time"
    if time_to_close_seconds < 60.0:
        return "final_no_trade_window"
    if elapsed_since_market_start_seconds is not None and elapsed_since_market_start_seconds < 60.0:
        return "early_window"
    if time_to_close_seconds < 120.0:
        return "sbc_only_window"
    return "hts_allowed_window"


def _trace_signal_outcome_classification(
    *,
    selected_action: str,
    order_allowed: bool,
    blocking_reasons: list[str],
) -> str:
    if order_allowed:
        return "paper_intent_created"
    if selected_action == "NO_TRADE":
        return "rank_blocked_by_no_trade"
    if "execution_time_to_close_unsafe" in blocking_reasons:
        return "guard_blocked_time_window"
    if blocking_reasons:
        return "guard_blocked_other"
    return "no_guard_decision_available"


def _trace_market_schedule(
    *,
    provider_row: dict[str, Any],
    decision_ts: int,
    micro: dict[str, Any],
) -> dict[str, Any]:
    provided_start = _trace_int_or_none(
        provider_row.get("market_start_ts")
        or provider_row.get("market_start_timestamp")
        or provider_row.get("score_components", {}).get("market_start_ts")
    )
    provided_end = _trace_int_or_none(
        provider_row.get("market_end_ts")
        or provider_row.get("market_end_timestamp")
        or provider_row.get("score_components", {}).get("market_end_ts")
    )
    slug = str(provider_row.get("slug") or "")
    slug_schedule = _trace_market_schedule_from_slug(slug)
    warning_reason_codes: list[str] = []
    source_type = "market_schedule_unavailable"
    source_fields_used: list[str] = []
    market_start_ts: int | None = None
    market_end_ts: int | None = None
    if slug_schedule is not None:
        slug_start, slug_end = slug_schedule
        if provided_start == slug_start and provided_end == slug_end:
            market_start_ts = provided_start
            market_end_ts = provided_end
            source_type = "normalized_public_market_metadata"
            source_fields_used = [
                "provider_row.market_start_ts",
                "provider_row.market_end_ts",
                "provider_row.slug",
            ]
        else:
            market_start_ts = slug_start
            market_end_ts = slug_end
            source_type = "canonical_market_slug_schedule"
            source_fields_used = ["provider_row.slug"]
            warning_reason_codes.append(
                "market_schedule_backfilled_from_canonical_slug"
                if provided_start is None or provided_end is None
                else "provider_market_schedule_mismatch_canonical_slug"
            )
    elif (
        provided_start is not None
        and provided_end is not None
        and provided_start > 0
        and provided_end > provided_start
    ):
        market_start_ts = provided_start
        market_end_ts = provided_end
        source_type = "normalized_public_market_metadata_without_slug_schedule"
        source_fields_used = [
            "provider_row.market_start_ts",
            "provider_row.market_end_ts",
        ]
    else:
        warning_reason_codes.append("market_schedule_identity_unavailable")
        if micro.get("time_to_close_seconds") is not None:
            warning_reason_codes.append(
                "microstructure_time_to_close_not_used_for_market_identity"
            )
    provenance = {
        "source_type": source_type,
        "source_fields_used": source_fields_used,
        "slug": slug,
        "raw_market_sha256": provider_row.get("raw_market_sha256"),
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts,
        "provenance_valid": bool(
            decision_ts > 0
            and market_start_ts is not None
            and market_end_ts is not None
            and market_start_ts > 0
            and market_end_ts > market_start_ts
        ),
        "warning_reason_codes": sorted(set(warning_reason_codes)),
    }
    return {
        "market_start_ts": market_start_ts,
        "market_end_ts": market_end_ts,
        "source_type": source_type,
        "warning_reason_codes": provenance["warning_reason_codes"],
        "provenance": provenance,
    }


def _trace_market_schedule_from_slug(slug: str) -> tuple[int, int] | None:
    match = BTC_UPDOWN_SLUG_PATTERN.match(slug)
    if match is None:
        return None
    family = BTC_UPDOWN_FAMILY_BY_SLUG[match.group(1)]
    start_ts = int(match.group(2)) * 1000
    return start_ts, start_ts + BTC_UPDOWN_MARKET_HORIZONS_MS[family]


def _trace_time_to_close_seconds(
    *,
    decision_ts: int,
    market_end_ts: int | None,
    micro: dict[str, Any],
) -> float | None:
    if market_end_ts is not None and decision_ts:
        return (market_end_ts - decision_ts) / 1000.0
    if micro.get("time_to_close_seconds") is not None:
        return _float(micro.get("time_to_close_seconds"))
    return None


def _trace_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_rows_sorted(rows: list[dict[str, Any]]) -> bool:
    keys = [
        (
            int(row.get("decision_ts") or 0),
            str(row.get("market_id") or ""),
            str(row.get("decision_group_id") or ""),
        )
        for row in rows
    ]
    return keys == sorted(keys)


def _trace_block_count(rows: list[dict[str, Any]], reason_code: str) -> int:
    return sum(
        1
        for row in rows
        if reason_code in (row.get("execution_blocking_reason_codes") or [])
    )


def _trace_non_time_blockers(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("execution_blocking_reason_codes") or []:
            if reason != "execution_time_to_close_unsafe":
                counter[str(reason)] += 1
    return dict(sorted(counter.items()))


def _trace_numeric_summary_by_field(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    value_field: str,
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        grouped[str(row.get(group_field))].append(_float(value))
    return {
        group: _numeric_summary(values)
        for group, values in sorted(grouped.items())
    }


def _fresh_loop_manifest(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    unlock_evidence: dict[str, Any],
    public_data_collection_report: dict[str, Any],
    run_report: dict[str, Any],
    fill_report: dict[str, Any],
    safety_report: dict[str, Any],
    monitoring_report: dict[str, Any],
    cumulative_report: dict[str, Any],
    no_trade_report: dict[str, Any],
    score_decomposition_report: dict[str, Any],
    provider_feature_coverage_report: dict[str, Any],
    canonical_feature_mapping_report: dict[str, Any],
    canonical_scorer_report: dict[str, Any],
    scorer_comparison_report: dict[str, Any],
    canonical_scorer_alignment_report: dict[str, Any],
    signal_trace_report: dict[str, Any],
    time_window_diagnostic_report: dict[str, Any],
    execution_layer_v2_paper_remap_report: dict[str, Any],
    legacy_position_policy_audit_report: dict[str, Any],
    paper_position_state_report: dict[str, Any],
    paper_exit_signal_report: dict[str, Any],
    paper_sell_position_intents: list[dict[str, Any]],
    synthetic_ledger_update_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": O_V8_PAPER_FRESH_LOOP_MANIFEST_SCHEMA_VERSION,
        "report_type": "o_v8_paper_fresh_loop_manifest",
        "phase": POLYMARKET_POLICY_TRAINING_PHASE,
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "paper_candidate_unlock_dir": unlock_evidence["paper_candidate_unlock_dir"],
        "paper_candidate_unlock_manifest_sha256": unlock_evidence[
            "observed_manifest_sha256"
        ],
        "paper_candidate_unlock_verified": unlock_evidence[
            "paper_candidate_unlock_verified"
        ],
        "fresh_loop_run_report_id": run_report[
            "o_v8_paper_fresh_loop_run_report_id"
        ],
        "fresh_fill_simulation_report_id": fill_report[
            "o_v8_paper_fresh_fill_simulation_report_id"
        ],
        "fresh_runtime_safety_report_id": safety_report[
            "o_v8_paper_fresh_runtime_safety_report_id"
        ],
        "fresh_monitoring_report_id": monitoring_report[
            "o_v8_paper_fresh_monitoring_report_id"
        ],
        "fresh_cumulative_monitoring_report_id": cumulative_report[
            "o_v8_paper_fresh_cumulative_monitoring_report_id"
        ],
        "fresh_no_trade_diagnostic_report_id": no_trade_report[
            "o_v8_paper_fresh_no_trade_diagnostic_report_id"
        ],
        "fresh_score_decomposition_report_id": score_decomposition_report[
            "o_v8_paper_fresh_score_decomposition_report_id"
        ],
        "fresh_provider_feature_coverage_report_id": provider_feature_coverage_report[
            "o_v8_paper_fresh_provider_feature_coverage_report_id"
        ],
        "fresh_canonical_feature_mapping_report_id": canonical_feature_mapping_report[
            "o_v8_paper_fresh_canonical_feature_mapping_report_id"
        ],
        "fresh_canonical_scorer_report_id": canonical_scorer_report[
            "o_v8_paper_fresh_canonical_scorer_report_id"
        ],
        "fresh_scorer_comparison_report_id": scorer_comparison_report[
            "o_v8_paper_fresh_scorer_comparison_report_id"
        ],
        "fresh_canonical_scorer_alignment_report_id": canonical_scorer_alignment_report[
            "o_v8_paper_fresh_canonical_scorer_alignment_report_id"
        ],
        "fresh_signal_trace_report_id": signal_trace_report[
            "o_v8_paper_fresh_signal_trace_report_id"
        ],
        "fresh_time_window_diagnostic_report_id": time_window_diagnostic_report[
            "o_v8_paper_fresh_time_window_diagnostic_report_id"
        ],
        "execution_layer_v2_paper_remap_report_id": (
            execution_layer_v2_paper_remap_report[
                "execution_layer_v2_paper_remap_report_id"
            ]
        ),
        "execution_layer_v2_paper_remap_enabled": (
            execution_layer_v2_paper_remap_report[
                "execution_layer_v2_paper_remap_enabled"
            ]
        ),
        "execution_layer_v2_paper_remap_applied_count": (
            execution_layer_v2_paper_remap_report["remap_guard_passed_count"]
        ),
        "execution_layer_v2_paper_remap_candidate_count": (
            execution_layer_v2_paper_remap_report["remap_candidate_count"]
        ),
        "execution_layer_v2_paper_remap_reason_distribution": (
            execution_layer_v2_paper_remap_report["remap_reason_distribution"]
        ),
        "fresh_legacy_position_policy_audit_report_id": (
            legacy_position_policy_audit_report[
                "o_v8_paper_fresh_legacy_position_policy_audit_report_id"
            ]
        ),
        "fresh_paper_position_state_report_id": paper_position_state_report[
            "o_v8_paper_fresh_position_state_report_id"
        ],
        "fresh_paper_exit_signal_report_id": paper_exit_signal_report[
            "o_v8_paper_fresh_exit_signal_report_id"
        ],
        "fresh_synthetic_ledger_update_report_id": synthetic_ledger_update_report[
            "o_v8_paper_fresh_synthetic_ledger_update_report_id"
        ],
        "signal_trace_row_count": signal_trace_report["trace_row_count"],
        "signal_trace_rows_sorted_by_decision_ts": signal_trace_report[
            "trace_rows_sorted_by_decision_ts"
        ],
        "signal_trace_rows_by_lifecycle_window": signal_trace_report[
            "rows_by_lifecycle_window"
        ],
        "signal_trace_rows_by_execution_blocking_reason": signal_trace_report[
            "rows_by_execution_blocking_reason"
        ],
        "time_window_rows_blocked_by_time_to_close": time_window_diagnostic_report[
            "rows_blocked_by_time_to_close"
        ],
        "provider_collection_too_late_for_selected_action_family": (
            signal_trace_report["provider_collection_too_late_for_selected_action_family"]
        ),
        "canonical_feature_mapping_complete": canonical_feature_mapping_report[
            "canonical_feature_mapping_complete"
        ],
        "canonical_action_row_count": canonical_feature_mapping_report[
            "canonical_action_row_count"
        ],
        "canonical_scorer_invoked": canonical_scorer_report[
            "canonical_frozen_o_scorer_invoked"
        ],
        "canonical_scorer_scored_action_row_count": canonical_scorer_report[
            "canonical_scored_action_row_count"
        ],
        "scorer_comparison_complete": scorer_comparison_report[
            "scorer_comparison_complete"
        ],
        "simplified_canonical_no_trade_agreement_count": scorer_comparison_report[
            "no_trade_agreement_count"
        ],
        "simplified_canonical_selected_action_agreement_count": (
            scorer_comparison_report["selected_action_agreement_count"]
        ),
        "canonical_frozen_o_scorer_used": canonical_scorer_alignment_report[
            "canonical_frozen_o_scorer_used"
        ],
        "canonical_alignment_diagnostic_status": canonical_scorer_alignment_report[
            "canonical_alignment_diagnostic_status"
        ],
        "canonical_alignment_blocking_reason_codes": canonical_scorer_alignment_report[
            "canonical_alignment_blocking_reason_codes"
        ],
        "rank_blocked_by_no_trade_count": no_trade_report[
            "rank_blocked_by_no_trade_count"
        ],
        "sparse_provider_row_flag": provider_feature_coverage_report[
            "sparse_provider_row_flag"
        ],
        "public_chainlink_rtds_price_row_count": provider_feature_coverage_report[
            "public_chainlink_rtds_price_row_count"
        ],
        "chainlink_price_at_decision_available_count": (
            provider_feature_coverage_report[
                "chainlink_price_at_decision_available_count"
            ]
        ),
        "chainlink_market_start_reference_available_count": (
            provider_feature_coverage_report[
                "chainlink_market_start_reference_available_count"
            ]
        ),
        "chainlink_feature_provenance_violation_count": (
            provider_feature_coverage_report[
                "chainlink_feature_provenance_violation_count"
            ]
        ),
        "paper_fresh_loop_enabled": run_report["paper_fresh_loop_enabled"],
        "paper_fresh_loop_mode": run_report["paper_fresh_loop_mode"],
        "paper_fresh_loop_cycle_count": run_report["paper_fresh_loop_cycle_count"],
        "paper_fresh_loop_max_cycles": run_report["paper_fresh_loop_max_cycles"],
        "paper_fresh_loop_sleep_seconds": run_report[
            "paper_fresh_loop_sleep_seconds"
        ],
        "paper_fresh_loop_public_data_source": run_report[
            "paper_fresh_loop_public_data_source"
        ],
        "public_data_collection_report": public_data_collection_report,
        "paper_fresh_provider_collection_failed": run_report[
            "paper_fresh_provider_collection_failed"
        ],
        "public_data_collection_reason_codes": run_report[
            "public_data_collection_reason_codes"
        ],
        "uses_paper_intent_logs_as_fresh_public_data": run_report[
            "uses_paper_intent_logs_as_fresh_public_data"
        ],
        "paper_fresh_order_intent_count": run_report[
            "paper_fresh_order_intent_count"
        ],
        "paper_fresh_fill_count": run_report["paper_fresh_fill_count"],
        "paper_fresh_ledger_entry_count": run_report[
            "paper_fresh_ledger_entry_count"
        ],
        "open_paper_position_count": paper_position_state_report[
            "open_paper_position_count"
        ],
        "position_open_failed_count": paper_position_state_report[
            "position_open_failed_count"
        ],
        "position_state_adapter_status": paper_position_state_report[
            "position_state_adapter_status"
        ],
        "position_state_blocking_reason_codes": paper_position_state_report[
            "position_state_blocking_reason_codes"
        ],
        "legacy_state_manager_reused": paper_exit_signal_report[
            "legacy_state_manager_reused"
        ],
        "legacy_decision_policy_reused": paper_exit_signal_report[
            "legacy_decision_policy_reused"
        ],
        "exit_decision_policy_source": paper_exit_signal_report[
            "exit_decision_policy_source"
        ],
        "exit_threshold_profile_name": paper_exit_signal_report[
            "exit_threshold_profile_name"
        ],
        "exit_threshold_source": paper_exit_signal_report["exit_threshold_source"],
        "exit_thresholds_tuned": paper_exit_signal_report["exit_thresholds_tuned"],
        "exit_threshold_values": paper_exit_signal_report["exit_threshold_values"],
        "exit_policy_kind": paper_exit_signal_report["exit_policy_kind"],
        "paper_exit_signal_count": paper_exit_signal_report[
            "paper_exit_signal_count"
        ],
        "paper_sell_position_intent_count": len(paper_sell_position_intents),
        "synthetic_exit_ledger_update_count": synthetic_ledger_update_report[
            "synthetic_ledger_update_count"
        ],
        "ledger_updates_only_for_accepted_paper_exit_intents": (
            synthetic_ledger_update_report[
                "ledger_updates_only_for_accepted_paper_exit_intents"
            ]
        ),
        "paper_exit_adapter_uses_settlement_oracle_future_return_fields": (
            paper_exit_signal_report["uses_settlement_oracle_future_return_fields"]
        ),
        "paper_exit_adapter_mutates_o_entry_scorer": paper_exit_signal_report[
            "mutates_o_entry_scorer"
        ],
        "paper_fresh_monitoring_passed": monitoring_report[
            "paper_fresh_monitoring_passed"
        ],
        "v8_paper_internal_handoff_allowed": run_report[
            "v8_paper_internal_handoff_allowed"
        ],
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#146_start_allowed": False,
        "#134_resume_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(manifest, "o_v8_paper_fresh_loop_manifest_id")


def _chainlink_decision_time_field_payload(
    *rows: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _CHAINLINK_DECISION_TIME_FIELDS:
        value: Any = None
        for row in rows:
            if row.get(field_name) is not None:
                value = row[field_name]
                break
        payload[field_name] = (
            dict(value)
            if field_name == "chainlink_regime_feature_provenance"
            and isinstance(value, dict)
            else value
        )
    return payload


def _guard_input_from_public_row(
    *,
    public_row: dict[str, Any],
    cycle_id: str,
    row_index: int,
) -> dict[str, Any]:
    action = str(public_row.get("selected_action") or public_row.get("action") or "")
    side = str(public_row.get("selected_side") or _side_from_action(action))
    family = str(public_row.get("selected_action_family") or _action_family(action))
    score = _float(
        public_row.get("corrected_model_score")
        if public_row.get("corrected_model_score") is not None
        else public_row.get("source_model_score")
    )
    decision_ts = int(public_row.get("decision_ts") or 0)
    ranking = list(public_row.get("full_5_action_ranking") or [])
    if not ranking:
        ranking = [
            {
                "selected_action": candidate,
                "corrected_model_score": score if candidate == action else score - 0.1,
                "raw_model_score": _float(public_row.get("raw_model_score")),
            }
            for candidate in O_REQUIRED_DECISION_ACTION_FAMILIES
        ]
    return {
        "decision_group_id": public_row.get("decision_group_id")
        or f"{cycle_id}|{public_row.get('market_id')}|{decision_ts}|{row_index}",
        "market_id": public_row.get("market_id"),
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "full_5_action_ranking": ranking,
        "corrected_model_score": score,
        "raw_model_score": _float(public_row.get("raw_model_score")),
        "score_components": dict(public_row.get("score_components") or {}),
        "high_score_flag": bool(public_row.get("high_score_flag", True)),
        "p_up": _float(public_row.get("p_up")),
        "p_down": _float(public_row.get("p_down")),
        "p_up_action_disagreement": bool(public_row.get("p_up_action_disagreement")),
        "microstructure_snapshot": dict(public_row.get("microstructure_snapshot") or {}),
        "reference_price_feature_provenance": dict(
            public_row.get("reference_price_feature_provenance")
            or {
                "provenance_valid": True,
                "decision_ts": decision_ts,
                "max_input_ts": decision_ts,
                "source_fields_used": ["fresh_public_provider_fixture"],
            }
        ),
        "decision_time_feature_max_input_ts": public_row.get(
            "decision_time_feature_max_input_ts", decision_ts
        ),
        "btc_momentum": public_row.get("btc_momentum"),
        "btc_momentum_provenance": dict(
            public_row.get("btc_momentum_provenance") or {}
        ),
        "reference_price_to_beat_at_decision": public_row.get(
            "reference_price_to_beat_at_decision"
        ),
        "reference_price_to_beat_distance_at_decision": public_row.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "reference_price_to_beat_distance_provenance": dict(
            public_row.get("reference_price_to_beat_distance_provenance") or {}
        ),
        "time_since_market_start_seconds": public_row.get(
            "time_since_market_start_seconds"
        ),
        "time_since_market_start_provenance": dict(
            public_row.get("time_since_market_start_provenance") or {}
        ),
        "action_score_margin": public_row.get("action_score_margin"),
        "action_score_margin_provenance": dict(
            public_row.get("action_score_margin_provenance") or {}
        ),
        "side_specific_action_score_margin": public_row.get(
            "side_specific_action_score_margin"
        ),
        "side_specific_action_score_margin_provenance": dict(
            public_row.get("side_specific_action_score_margin_provenance") or {}
        ),
        "decision_time_regime_feature_provenance": dict(
            public_row.get("decision_time_regime_feature_provenance") or {}
        ),
        "decision_time_regime_feature_max_input_ts": public_row.get(
            "decision_time_regime_feature_max_input_ts",
            public_row.get("decision_time_feature_max_input_ts", decision_ts),
        ),
        **_chainlink_decision_time_field_payload(public_row),
    }

def _resolve_unlock_artifact_path(unlock_dir: Path, raw_path: str) -> Path:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute() or artifact_path.exists():
        return artifact_path
    return unlock_dir / artifact_path


def _fresh_order_intent_from_guard_row(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    cycle_id: str,
    guard_row: dict[str, Any],
    intent_index: int,
) -> dict[str, Any]:
    micro = dict(guard_row.get("microstructure_snapshot") or {})
    intent = {
        "paper_fresh_order_intent_id": f"{config.run_id}-fresh-intent-{intent_index:06d}",
        "cycle_id": cycle_id,
        "simulated_order_id": guard_row.get("simulated_order_id"),
        "decision_group_id": guard_row.get("decision_group_id"),
        "market_id": guard_row.get("market_id"),
        "decision_ts": guard_row.get("decision_ts"),
        "source_selected_action": guard_row.get("source_selected_action"),
        "source_selected_family": guard_row.get("source_selected_family"),
        "source_selected_side": guard_row.get("source_selected_side"),
        "original_action": guard_row.get("original_action")
        or guard_row.get("source_selected_action"),
        "original_family": guard_row.get("original_family")
        or guard_row.get("source_selected_family"),
        "original_side": guard_row.get("original_side")
        or guard_row.get("source_selected_side"),
        "remapped_action": guard_row.get("remapped_action"),
        "remapped_family": guard_row.get("remapped_family"),
        "remapped_side": guard_row.get("remapped_side"),
        "hts_time_window_remap_applied": bool(
            guard_row.get("hts_time_window_remap_applied")
        ),
        "remap_reason_codes": list(guard_row.get("remap_reason_codes") or []),
        "hts_time_window_remap_calibrated_ev": (
            _float(guard_row.get("hts_time_window_remap_calibrated_ev"))
            if guard_row.get("hts_time_window_remap_calibrated_ev") is not None
            else None
        ),
        "hts_time_window_remap_calibrated_ev_source": guard_row.get(
            "hts_time_window_remap_calibrated_ev_source"
        ),
        "original_execution_blocking_reason_codes": list(
            guard_row.get("original_execution_blocking_reason_codes") or []
        ),
        "original_execution_guard_reason_codes": list(
            guard_row.get("original_execution_guard_reason_codes") or []
        ),
        "execution_guarded_action": guard_row.get("execution_guarded_action"),
        "execution_guarded_family": guard_row.get("execution_guarded_family"),
        "execution_guarded_side": guard_row.get("execution_guarded_side"),
        "source_model_score": _float(guard_row.get("source_model_score")),
        "execution_guarded_score": _float(guard_row.get("execution_guarded_score")),
        "p_up": _float(guard_row.get("p_up")),
        "p_down": _float(guard_row.get("p_down")),
        "p_up_action_disagreement": bool(guard_row.get("p_up_action_disagreement")),
        "order_origin": "fresh_public_guard_allowed_action",
        "paper_fresh_order_size": _float(guard_row.get("proposed_order_size")),
        "paper_limit_price": _fill_price_from_microstructure(micro),
        "spread_bps": _float(micro.get("spread_bps")),
        "book_staleness_ms": _float(micro.get("book_staleness_ms")),
        "queue_fill_proxy": _float(micro.get("queue_fill_proxy")),
        "time_to_close_seconds": _float(micro.get("time_to_close_seconds")),
        "entry_ask": _float(micro.get("entry_ask")),
        "executable_exit_bid_proxy": _float(micro.get("executable_exit_bid_proxy")),
        "btc_momentum": guard_row.get("btc_momentum"),
        "btc_momentum_provenance": dict(guard_row.get("btc_momentum_provenance") or {}),
        "reference_price_to_beat_at_decision": guard_row.get(
            "reference_price_to_beat_at_decision"
        ),
        "reference_price_to_beat_distance_at_decision": guard_row.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "reference_price_to_beat_distance_provenance": dict(
            guard_row.get("reference_price_to_beat_distance_provenance") or {}
        ),
        "time_since_market_start_seconds": guard_row.get(
            "time_since_market_start_seconds"
        ),
        "time_since_market_start_provenance": dict(
            guard_row.get("time_since_market_start_provenance") or {}
        ),
        "action_score_margin": guard_row.get("action_score_margin"),
        "action_score_margin_provenance": dict(
            guard_row.get("action_score_margin_provenance") or {}
        ),
        "side_specific_action_score_margin": guard_row.get(
            "side_specific_action_score_margin"
        ),
        "side_specific_action_score_margin_provenance": dict(
            guard_row.get("side_specific_action_score_margin_provenance") or {}
        ),
        "decision_time_regime_feature_provenance": dict(
            guard_row.get("decision_time_regime_feature_provenance") or {}
        ),
        "decision_time_regime_feature_max_input_ts": guard_row.get(
            "decision_time_regime_feature_max_input_ts"
        ),
        **_chainlink_decision_time_field_payload(guard_row),
        "pre_decision_exposure_state": guard_row.get("pre_decision_exposure_state"),
        "post_decision_exposure_state": guard_row.get("post_decision_exposure_state"),
        "execution_guard_reason_codes": guard_row.get("execution_guard_reason_codes", []),
        "execution_blocking_reason_codes": guard_row.get(
            "execution_blocking_reason_codes", []
        ),
        "sizing_reason_codes": guard_row.get("sizing_reason_codes", []),
        "paper_fresh_order_intent_status": "accepted_for_fresh_paper_loop",
        "order_intent_contract": "fresh_public_local_paper_intent_no_exchange_write_v1",
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    intent["paper_fresh_order_intent_hash"] = canonical_json_sha256(intent)
    return intent


def _fresh_paper_fills_from_intents(
    intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    for index, intent in enumerate(intents, start=1):
        size = _float(intent.get("paper_fresh_order_size"))
        fill_price = _float(intent.get("paper_limit_price"))
        spread_cost = size * _float(intent.get("spread_bps")) / 10_000.0
        fill = {
            "paper_fresh_fill_id": f"fresh-paper-fill-{index:06d}",
            "paper_fresh_order_intent_id": intent["paper_fresh_order_intent_id"],
            "cycle_id": intent.get("cycle_id"),
            "simulated_order_id": intent.get("simulated_order_id"),
            "market_id": intent.get("market_id"),
            "decision_ts": intent.get("decision_ts"),
            "execution_guarded_action": intent.get("execution_guarded_action"),
            "execution_guarded_family": intent.get("execution_guarded_family"),
            "execution_guarded_side": intent.get("execution_guarded_side"),
            "fill_simulation_status": "paper_fresh_filled",
            "fill_simulation_rule_id": "fresh_deterministic_queue_fill_proxy_v1",
            "requested_size": size,
            "filled_size": size,
            "fill_probability": _float(intent.get("queue_fill_proxy")),
            "paper_fill_price": fill_price,
            "btc_momentum": intent.get("btc_momentum"),
            "btc_momentum_provenance": dict(intent.get("btc_momentum_provenance") or {}),
            "reference_price_to_beat_at_decision": intent.get(
                "reference_price_to_beat_at_decision"
            ),
            "reference_price_to_beat_distance_at_decision": intent.get(
                "reference_price_to_beat_distance_at_decision"
            ),
            "reference_price_to_beat_distance_provenance": dict(
                intent.get("reference_price_to_beat_distance_provenance") or {}
            ),
            "time_since_market_start_seconds": intent.get(
                "time_since_market_start_seconds"
            ),
            "time_since_market_start_provenance": dict(
                intent.get("time_since_market_start_provenance") or {}
            ),
            "action_score_margin": intent.get("action_score_margin"),
            "action_score_margin_provenance": dict(
                intent.get("action_score_margin_provenance") or {}
            ),
            "side_specific_action_score_margin": intent.get(
                "side_specific_action_score_margin"
            ),
            "side_specific_action_score_margin_provenance": dict(
                intent.get("side_specific_action_score_margin_provenance") or {}
            ),
            "decision_time_regime_feature_provenance": dict(
                intent.get("decision_time_regime_feature_provenance") or {}
            ),
            "decision_time_regime_feature_max_input_ts": intent.get(
                "decision_time_regime_feature_max_input_ts"
            ),
            **_chainlink_decision_time_field_payload(intent),
            "spread_cost": spread_cost,
            "fee_cost": 0.0,
            "slippage_cost": 0.0,
            "liquidity_impact_cost": 0.0,
            "total_execution_cost": spread_cost,
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "synthetic_paper_cash_delta": -(size * fill_price + spread_cost),
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        fill["paper_fresh_fill_hash"] = canonical_json_sha256(fill)
        fills.append(fill)
    return fills


def _fresh_paper_ledger_from_fills(
    fills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cash = 10_000.0
    exposure_by_market: dict[str, float] = defaultdict(float)
    exposure_by_side: dict[str, float] = defaultdict(float)
    ledger_rows: list[dict[str, Any]] = []
    for index, fill in enumerate(fills, start=1):
        market_id = str(fill.get("market_id"))
        side = str(fill.get("execution_guarded_side"))
        size = _float(fill.get("filled_size"))
        cash_before = cash
        cash += _float(fill.get("synthetic_paper_cash_delta"))
        exposure_by_market[market_id] += size
        exposure_by_side[side] += size
        row = {
            "paper_fresh_ledger_entry_id": f"fresh-paper-ledger-{index:06d}",
            "paper_fresh_fill_id": fill["paper_fresh_fill_id"],
            "paper_fresh_order_intent_id": fill["paper_fresh_order_intent_id"],
            "cycle_id": fill.get("cycle_id"),
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": side,
            "cash_before": cash_before,
            "cash_after": cash,
            "synthetic_position_after": exposure_by_market[market_id],
            "total_exposure_after": sum(exposure_by_market.values()),
            "side_exposure_after": exposure_by_side[side],
            "outcome_pnl_used": False,
            "realized_pnl_used": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["paper_fresh_ledger_entry_hash"] = canonical_json_sha256(row)
        ledger_rows.append(row)
    return ledger_rows


def _cycle_monitoring_row(
    *,
    cycle_id: str,
    cycle_index: int,
    public_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    cycle_failed: bool,
    cycle_forbidden_rows: list[dict[str, Any]],
    public_data_source: str,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    unique_markets = sorted({str(row.get("market_id")) for row in public_rows})
    return {
        "cycle_id": cycle_id,
        "cycle_index": cycle_index + 1,
        "cycle_failed": cycle_failed,
        "cycle_failure_reason_codes": [
            "paper_candidate_unlock_not_verified"
        ]
        if cycle_failed and not cycle_forbidden_rows
        else (["fresh_public_data_forbidden_outcome_fields_present"] if cycle_failed else []),
        "public_data_source": public_data_source,
        "public_data_freshness": "read_only_public_provider_snapshot"
        if public_data_source == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
        else "offline_snapshot_fixture",
        "market_count": len(public_rows),
        "unique_market_count": len(unique_markets),
        "unique_market_ids": unique_markets,
        "candidate_decision_count": len(guard_rows),
        "guard_allowed_paper_intent_count": len(intents),
        "guard_blocked_decision_count": sum(
            1 for row in guard_rows if row.get("order_allowed") is not True
        ),
        "block_reason_distribution": _counter_from_rows(
            guard_rows, "execution_blocking_reason_codes"
        ),
        "action_distribution": Counter(
            str(row.get("execution_guarded_action")) for row in guard_rows
        ),
        "side_distribution": Counter(
            str(row.get("execution_guarded_side")) for row in guard_rows
        ),
        "family_distribution": Counter(
            str(row.get("execution_guarded_family")) for row in guard_rows
        ),
        "runtime_field_missing_count": sum(
            len(row.get("missing_runtime_field_codes") or []) for row in guard_rows
        ),
        "provenance_violation_count": sum(
            len(row.get("runtime_field_backfill_provenance_violations") or [])
            for row in guard_rows
        ),
        "p_up_disagreement_count": sum(
            1 for row in guard_rows if row.get("p_up_action_disagreement") is True
        ),
        "exposure_state_after": _compact_runtime_state(runtime_state),
        "forbidden_public_data_rows": cycle_forbidden_rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _rows_with_forbidden_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        present = sorted(
            field_name
            for field_name in O_V8_PAPER_FRESH_FORBIDDEN_PUBLIC_DATA_FIELDS
            if field_name in row
        )
        if present:
            failures.append(
                {
                    "row_index": index,
                    "market_id": row.get("market_id"),
                    "decision_ts": row.get("decision_ts"),
                    "forbidden_fields": present,
                }
            )
    return failures


def _initial_fresh_runtime_state() -> dict[str, Any]:
    return {
        "risk_state_source": "fresh_public_paper_simulated_ledger",
        "runtime_state_validation_passed": True,
        "current_total_exposure": 0.0,
        "current_side_exposure_by_side": {"UP": 0.0, "DOWN": 0.0, "NONE": 0.0},
        "current_market_exposure_by_market_id": {},
        "open_position_by_market_id": {},
        "open_position_by_market_side": {},
        "cooldown_state": {},
        "executed_simulated_order_count": 0,
        "blocked_simulated_order_count": 0,
    }


def _apply_guard_row_to_runtime_state(
    runtime_state: dict[str, Any],
    guard_row: dict[str, Any],
) -> None:
    market_id = str(guard_row.get("market_id"))
    side = str(guard_row.get("execution_guarded_side"))
    size = _float(guard_row.get("proposed_order_size"))
    market_exposure = runtime_state["current_market_exposure_by_market_id"]
    side_exposure = runtime_state["current_side_exposure_by_side"]
    market_exposure[market_id] = _float(market_exposure.get(market_id)) + size
    side_exposure[side] = _float(side_exposure.get(side)) + size
    runtime_state["current_total_exposure"] = _float(
        runtime_state.get("current_total_exposure")
    ) + size
    position = {
        "market_id": market_id,
        "side": side,
        "action": guard_row.get("execution_guarded_action"),
        "notional": size,
        "simulated_order_id": guard_row.get("simulated_order_id"),
    }
    runtime_state["open_position_by_market_id"][market_id] = position
    runtime_state["open_position_by_market_side"][f"{market_id}|{side}"] = position
    runtime_state["executed_simulated_order_count"] = int(
        runtime_state.get("executed_simulated_order_count") or 0
    ) + 1


def _compact_runtime_state(runtime_state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(runtime_state, sort_keys=True))


def _fill_price_from_microstructure(micro: dict[str, Any]) -> float:
    entry_ask = _float(micro.get("entry_ask"))
    if entry_ask > 0.0:
        return entry_ask
    exit_bid = _float(micro.get("executable_exit_bid_proxy"))
    return exit_bid if exit_bid > 0.0 else 1.0


def _counter_from_rows(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        elif value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _flatten_public_rows(
    public_cycles: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [dict(row) for cycle in public_cycles for row in cycle]


def _ranking_from_public_or_guard(
    *,
    public_row: dict[str, Any],
    guard_row: dict[str, Any],
) -> list[dict[str, Any]]:
    ranking = list(public_row.get("full_5_action_ranking") or [])
    if ranking:
        return [dict(row) for row in ranking]
    return [dict(row) for row in guard_row.get("top_k_action_ranking") or []]


def _ranking_action(
    ranking: list[dict[str, Any]],
    action: str,
) -> dict[str, Any]:
    for row in ranking:
        if row.get("selected_action") == action:
            return dict(row)
    return {}


def _top_action_margin(ranking: list[dict[str, Any]]) -> float | None:
    if len(ranking) < 2:
        return None
    scores = sorted(
        [_float(row.get("corrected_model_score")) for row in ranking],
        reverse=True,
    )
    return scores[0] - scores[1]


def _top_action_side_margin(
    ranking: list[dict[str, Any]],
    *,
    selected_action: str,
) -> float | None:
    selected = _ranking_action(ranking, selected_action)
    if not selected:
        return None
    selected_side = _side_from_action(selected_action)
    if selected_side not in {"UP", "DOWN"}:
        return None
    opposite_side = "DOWN" if selected_side == "UP" else "UP"
    opposite_scores = [
        _float(row.get("corrected_model_score"))
        for row in ranking
        if _side_from_action(str(row.get("selected_action") or "")) == opposite_side
    ]
    if not opposite_scores:
        return None
    return _float(selected.get("corrected_model_score")) - max(opposite_scores)


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    midpoint = len(clean) // 2
    median = (
        clean[midpoint]
        if len(clean) % 2 == 1
        else (clean[midpoint - 1] + clean[midpoint]) / 2.0
    )
    return {
        "count": len(clean),
        "min": clean[0],
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
        "median": median,
    }


def _score_summary_by_action(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("action"))].append(_float(row.get("corrected_score")))
    return {
        action: _numeric_summary(values)
        for action, values in sorted(grouped.items())
    }


def _missing_required_microstructure_fields(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for action_row in row.get("full_5_action_ranking") or []:
        action = str(action_row.get("selected_action") or "")
        if action == "NO_TRADE":
            continue
        micro = dict(action_row.get("microstructure_snapshot") or {})
        for field_name in (
            "entry_ask",
            "executable_exit_bid_proxy",
            "spread_bps",
            "book_staleness_ms",
            "queue_fill_proxy",
            "time_to_close_seconds",
        ):
            if micro.get(field_name) is None:
                missing.append(f"{action}.{field_name}")
    return missing


def _fresh_no_trade_conclusion(
    *,
    run_report: dict[str, Any],
    rank_blocked_count: int,
    execution_blocked_count: int,
    public_data_collection_report: dict[str, Any],
) -> str:
    if public_data_collection_report["paper_fresh_provider_collection_failed"]:
        return "provider_collection_failed"
    if run_report["paper_fresh_order_intent_count"] == 0 and rank_blocked_count:
        if execution_blocked_count == 0:
            return "rank_blocked_by_no_trade_under_simplified_provider_score"
        return "mixed_no_trade_rank_and_execution_guard_blocking"
    if run_report["paper_fresh_order_intent_count"] == 0 and execution_blocked_count:
        return "execution_guard_blocked_buy_actions"
    if run_report["paper_fresh_order_intent_count"] > 0:
        return "paper_intents_generated_when_buy_actions_rank_above_no_trade"
    return "no_provider_decisions_available"


def _fresh_historical_run_comparison_rows(
    *,
    current_run_report: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = [
        _fresh_comparison_row_from_report(
            run_id=str(current_run_report["run_id"]),
            report=current_run_report,
            report_path=None,
            comparison_source="current_run",
        )
    ]
    for run_id in O_V8_PAPER_FRESH_COMPARISON_RUN_IDS:
        report_path = output_dir / run_id / "o_v8_paper_fresh_loop_run_report.json"
        if not report_path.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "comparison_source": "historical_run",
                    "report_path": str(report_path),
                    "report_available": False,
                    "missing_reason_code": "historical_fresh_loop_report_missing",
                }
            )
            continue
        rows.append(
            _fresh_comparison_row_from_report(
                run_id=run_id,
                report=_read_json(report_path),
                report_path=report_path,
                comparison_source="historical_run",
            )
        )
    return rows


def _fresh_comparison_row_from_report(
    *,
    run_id: str,
    report: dict[str, Any],
    report_path: Path | None,
    comparison_source: str,
) -> dict[str, Any]:
    collection = dict(report.get("public_data_collection_report") or {})
    return {
        "run_id": run_id,
        "comparison_source": comparison_source,
        "report_path": str(report_path) if report_path is not None else None,
        "report_available": True,
        "provider_market_count": collection.get("public_market_count"),
        "provider_feature_row_count": collection.get("public_feature_row_count"),
        "candidate_decision_count": report.get("candidate_decision_count"),
        "selected_action_distribution": report.get("action_distribution") or {},
        "buy_action_score_distribution": (
            report.get("buy_action_score_distribution")
            or "unavailable_pre_162_score_diagnostic"
        ),
        "no_trade_gap_distribution": (
            report.get("no_trade_gap_distribution")
            or "unavailable_pre_162_score_diagnostic"
        ),
        "p_up_p_down_distribution": (
            report.get("p_up_p_down_distribution")
            or "unavailable_pre_162_score_diagnostic"
        ),
        "microstructure_summary": (
            report.get("microstructure_summary")
            or "unavailable_pre_162_score_diagnostic"
        ),
        "guard_block_reason_distribution": report.get("block_reason_distribution")
        or {},
        "paper_intent_count": report.get("paper_fresh_order_intent_count"),
        "paper_fill_count": report.get("paper_fresh_fill_count"),
        "runtime_missing_field_count": report.get("runtime_field_missing_count"),
        "provenance_violation_count": report.get("provenance_violation_count"),
    }


def _fresh_provider_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row.keys())
        names.update(f"score_components.{key}" for key in row.get("score_components") or {})
        names.update(
            f"microstructure_snapshot.{key}"
            for key in row.get("microstructure_snapshot") or {}
        )
        for action_row in row.get("full_5_action_ranking") or []:
            names.update(
                f"full_5_action_ranking.microstructure_snapshot.{key}"
                for key in action_row.get("microstructure_snapshot") or {}
            )
    return sorted(names)


def _fresh_feature_value(
    value: Any,
    *,
    default: float,
    feature_name: str,
    defaults: list[str],
) -> float:
    if value is None:
        defaults.append(feature_name)
        return default
    return _float(value)


def _fresh_reference_distance(row: dict[str, Any]) -> tuple[float | None, bool]:
    if row.get("reference_price_to_beat_distance_at_decision") is not None:
        return _float(row.get("reference_price_to_beat_distance_at_decision")), False
    score_components = dict(row.get("score_components") or {})
    btc_mid = _float(score_components.get("btc_mid_price"))
    reference = _float(score_components.get("reference_price_to_beat"))
    if reference > 0.0:
        return (btc_mid - reference) / reference, False
    return 0.0, True


def _fresh_opposite_book_staleness(
    *,
    ranking: list[dict[str, Any]],
    action: str,
    default: float,
) -> float:
    side = _side_from_action(action)
    if side == "UP":
        opposite_action = "BUY_DOWN_HOLD_TO_SETTLEMENT"
    elif side == "DOWN":
        opposite_action = "BUY_UP_HOLD_TO_SETTLEMENT"
    else:
        return default
    opposite = _ranking_action(ranking, opposite_action)
    micro = dict(opposite.get("microstructure_snapshot") or {})
    return _float(micro.get("book_staleness_ms")) if micro else default


def _fresh_hts_sbc_gap_proxy(
    *,
    ranking: list[dict[str, Any]],
    action: str,
) -> float:
    side = _side_from_action(action)
    if side not in {"UP", "DOWN"}:
        return 0.0
    hts = _ranking_action(ranking, f"BUY_{side}_HOLD_TO_SETTLEMENT")
    sbc = _ranking_action(ranking, f"BUY_{side}_SELL_BEFORE_CLOSE")
    if not hts or not sbc:
        return 0.0
    return _float(sbc.get("corrected_model_score")) - _float(
        hts.get("corrected_model_score")
    )


def _rank_by_action(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    action_field: str,
) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _float(row.get(score_field)),
            1 if row.get(action_field) != "NO_TRADE" else 0,
            str(row.get(action_field)),
        ),
        reverse=True,
    )
    return {
        str(row.get(action_field)): rank for rank, row in enumerate(ordered, start=1)
    }


def _check(
    *,
    passed: bool,
    reason_code: str,
    observed: Any,
    required: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reason_code": reason_code,
        "observed": observed,
        "required": required,
    }


def _blocking_reason_codes(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        str(check["reason_code"])
        for check in checks.values()
        if check.get("passed") is not True
    )


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    report = dict(payload)
    report[field_name] = canonical_json_sha256(report)
    return report


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _trace_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fresh_loop_run_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Loop Run",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_loop_enabled: `{str(report['paper_fresh_loop_enabled']).lower()}`",
            f"- paper_fresh_loop_mode: `{report['paper_fresh_loop_mode']}`",
            f"- public_data_source: `{report['paper_fresh_loop_public_data_source']}`",
            f"- provider_collection_failed: `{str(report['paper_fresh_provider_collection_failed']).lower()}`",
            f"- paper_fresh_loop_cycle_count: `{report['paper_fresh_loop_cycle_count']}`",
            f"- paper_fresh_order_intent_count: `{report['paper_fresh_order_intent_count']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_ledger_entry_count: `{report['paper_fresh_ledger_entry_count']}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_loop_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_fill_simulation_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Fill Simulation",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_fill_count: `{report['paper_fresh_fill_count']}`",
            f"- paper_fresh_filled_size_sum: `{report['paper_fresh_filled_size_sum']}`",
            f"- total_synthetic_execution_cost: `{report['paper_fresh_total_synthetic_execution_cost']}`",
            f"- outcome_pnl_used: `{str(report['outcome_pnl_used']).lower()}`",
            f"- realized_pnl_used: `{str(report['realized_pnl_used']).lower()}`",
            "",
        ]
    )


def _fresh_runtime_safety_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Runtime Safety",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_runtime_safety_passed: `{str(report['paper_fresh_runtime_safety_passed']).lower()}`",
            f"- v8_paper_internal_handoff_allowed: `{str(report['v8_paper_internal_handoff_allowed']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_only: `{str(report['paper_only']).lower()}`",
            f"- capital_at_risk: `{str(report['capital_at_risk']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["paper_fresh_runtime_safety_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_fresh_monitoring_passed: `{str(report['paper_fresh_monitoring_passed']).lower()}`",
            f"- cycle_count: `{report['cycle_count']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- candidate_decision_count: `{report['candidate_decision_count']}`",
            f"- guard_allowed_decision_count: `{report['guard_allowed_decision_count']}`",
            f"- guard_blocked_decision_count: `{report['guard_blocked_decision_count']}`",
            "",
        ]
    )


def _fresh_cumulative_monitoring_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Cumulative Monitoring",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- total_cycles: `{report['total_cycles']}`",
            f"- total_paper_intents: `{report['total_paper_intents']}`",
            f"- total_paper_fills: `{report['total_paper_fills']}`",
            f"- total_blocked_decisions: `{report['total_blocked_decisions']}`",
            f"- cycle_failure_count: `{report['cycle_failure_count']}`",
            f"- safety_violation_count: `{report['safety_violation_count']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
        ]
    )


def _fresh_no_trade_diagnostic_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh NO_TRADE Diagnostic",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- scoring_rule_id: `{report['scoring_rule_id']}`",
            f"- canonical_frozen_o_scorer_used: `{str(report['canonical_frozen_o_scorer_used']).lower()}`",
            f"- candidate_decision_count: `{report['candidate_decision_count']}`",
            f"- rank_blocked_by_no_trade_count: `{report['rank_blocked_by_no_trade_count']}`",
            f"- execution_guard_blocked_count: `{report['execution_guard_blocked_count']}`",
            f"- paper_fresh_order_intent_count: `{report['paper_fresh_order_intent_count']}`",
            f"- zero_intent_behavior_classification: `{report['zero_intent_behavior_classification']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- source_model_candidate_eligible: `{str(report['source_model_candidate_eligible']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
            "## Selected Action Distribution",
            "",
            *_markdown_dict(report["selected_action_distribution"]),
            "",
            "## Best Buy Action Distribution",
            "",
            *_markdown_dict(report["best_buy_action_distribution"]),
            "",
        ]
    )


def _fresh_score_decomposition_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Score Decomposition",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- scoring_rule_id: `{report['scoring_rule_id']}`",
            f"- canonical_frozen_o_scorer_used: `{str(report['canonical_frozen_o_scorer_used']).lower()}`",
            f"- score_decomposition_action_row_count: `{report['score_decomposition_action_row_count']}`",
            f"- thresholds_tuned: `{str(report['thresholds_tuned']).lower()}`",
            f"- uses_realized_pnl_or_labels_for_analysis: `{str(report['uses_realized_pnl_or_labels_for_analysis']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Score Summary By Action",
            "",
            *_markdown_summary_dict(report["score_summary_by_action"]),
            "",
        ]
    )


def _fresh_provider_feature_coverage_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Provider Feature Coverage",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- public_market_count: `{report['public_market_count']}`",
            f"- public_feature_row_count: `{report['public_feature_row_count']}`",
            f"- unique_market_count: `{report['unique_market_count']}`",
            f"- cycle_count: `{report['cycle_count']}`",
            f"- cycles_with_rows: `{report['cycles_with_rows']}`",
            f"- idle_cycles: `{report['idle_cycles']}`",
            f"- missing_required_microstructure_field_count: `{report['missing_required_microstructure_field_count']}`",
            f"- missing_runtime_field_count: `{report['missing_runtime_field_count']}`",
            f"- provenance_invalid_count: `{report['provenance_invalid_count']}`",
            f"- provider_collection_failures: `{report['provider_collection_failures']}`",
            f"- sparse_provider_row_flag: `{str(report['sparse_provider_row_flag']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Sparse Provider Row Reason Codes",
            "",
            *_markdown_list(report["sparse_provider_row_reason_codes"]),
            "",
            "## Missing Microstructure Fields",
            "",
            *_markdown_dict(
                report["missing_required_microstructure_field_distribution"]
            ),
            "",
        ]
    )


def _fresh_canonical_feature_mapping_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Canonical Feature Mapping",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- canonical_feature_mapping_complete: `{str(report['canonical_feature_mapping_complete']).lower()}`",
            f"- decision_group_count: `{report['decision_group_count']}`",
            f"- canonical_action_row_count: `{report['canonical_action_row_count']}`",
            f"- expected_canonical_action_row_count: `{report['expected_canonical_action_row_count']}`",
            f"- canonical_feature_count: `{report['canonical_feature_count']}`",
            f"- mapped_feature_count: `{report['mapped_feature_count']}`",
            f"- provenance_invalid_count: `{report['provenance_invalid_count']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["canonical_feature_mapping_blocking_reason_codes"]),
            "",
            "## Default Backfilled Feature Distribution",
            "",
            *_markdown_dict(report["default_backfilled_feature_distribution"]),
            "",
        ]
    )


def _fresh_canonical_scorer_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Canonical Scorer",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- canonical_frozen_o_scorer_invoked: `{str(report['canonical_frozen_o_scorer_invoked']).lower()}`",
            f"- canonical_frozen_o_scorer_used: `{str(report['canonical_frozen_o_scorer_used']).lower()}`",
            f"- canonical_scorer_diagnostic_status: `{report['canonical_scorer_diagnostic_status']}`",
            f"- canonical_action_row_count: `{report['canonical_action_row_count']}`",
            f"- canonical_scored_action_row_count: `{report['canonical_scored_action_row_count']}`",
            f"- canonical_selected_decision_count: `{report['canonical_selected_decision_count']}`",
            f"- selected_feature_set_name: `{report['selected_feature_set_name']}`",
            f"- selected_correction_policy_name: `{report['selected_correction_policy_name']}`",
            f"- ranking_correction_config_hash_verified: `{str(report['ranking_correction_config_hash_verified']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["canonical_scorer_blocking_reason_codes"]),
            "",
            "## Selected Action Distribution",
            "",
            *_markdown_dict(report["canonical_selected_action_distribution"]),
            "",
        ]
    )


def _fresh_scorer_comparison_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Scorer Comparison",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- scorer_comparison_complete: `{str(report['scorer_comparison_complete']).lower()}`",
            f"- decision_group_count: `{report['decision_group_count']}`",
            f"- selected_action_agreement_count: `{report['selected_action_agreement_count']}`",
            f"- selected_action_disagreement_count: `{report['selected_action_disagreement_count']}`",
            f"- no_trade_agreement_count: `{report['no_trade_agreement_count']}`",
            f"- no_trade_disagreement_count: `{report['no_trade_disagreement_count']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["scorer_comparison_blocking_reason_codes"]),
            "",
            "## Simplified Selected Action Distribution",
            "",
            *_markdown_dict(report["simplified_selected_action_distribution"]),
            "",
            "## Canonical Selected Action Distribution",
            "",
            *_markdown_dict(report["canonical_selected_action_distribution"]),
            "",
        ]
    )


def _fresh_canonical_scorer_alignment_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Canonical Scorer Alignment",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- canonical_frozen_o_scorer_invoked: `{str(report['canonical_frozen_o_scorer_invoked']).lower()}`",
            f"- canonical_frozen_o_scorer_used: `{str(report['canonical_frozen_o_scorer_used']).lower()}`",
            f"- canonical_alignment_diagnostic_status: `{report['canonical_alignment_diagnostic_status']}`",
            f"- fresh_provider_scoring_rule_id: `{report['fresh_provider_scoring_rule_id']}`",
            f"- feature_schema_matches_canonical_scorer_requirements: `{str(report['feature_schema_matches_canonical_scorer_requirements']).lower()}`",
            f"- source_o_score_mutated: `{str(report['source_o_score_mutated']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- #146_start_allowed: `{str(report['#146_start_allowed']).lower()}`",
            f"- #134_resume_allowed: `{str(report['#134_resume_allowed']).lower()}`",
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["canonical_alignment_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_signal_trace_md(report: dict[str, Any]) -> str:
    zero = report["zero_intent_explanation"]
    return "\n".join(
        [
            "# O v8 Paper Fresh Signal Trace",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- public_data_source: `{report['public_data_source']}`",
            f"- trace_row_count: `{report['trace_row_count']}`",
            f"- trace_rows_sorted_by_decision_ts: `{str(report['trace_rows_sorted_by_decision_ts']).lower()}`",
            f"- canonical_selected_decision_count: `{report['canonical_selected_decision_count']}`",
            f"- paper_intent_count: `{report['paper_intent_count']}`",
            f"- fill_count: `{report['fill_count']}`",
            f"- rows_blocked_by_time_to_close: `{report['rows_blocked_by_time_to_close']}`",
            f"- provider_collection_too_late_for_selected_action_family: `{str(report['provider_collection_too_late_for_selected_action_family']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            f"- paper_only: `{str(report['paper_only']).lower()}`",
            f"- capital_at_risk: `{str(report['capital_at_risk']).lower()}`",
            "",
            "## Zero Intent Explanation",
            "",
            f"- canonical_buy_side_signal_count: `{zero['canonical_buy_side_signal_count']}`",
            f"- signals_inside_allowed_execution_window_count: `{zero['signals_inside_allowed_execution_window_count']}`",
            f"- time_to_close_blocked_count: `{zero['time_to_close_blocked_count']}`",
            f"- likely_collection_cadence_or_market_window_issue: `{str(zero['likely_collection_cadence_or_market_window_issue']).lower()}`",
            "",
            "## Rows By Lifecycle Window",
            "",
            *_markdown_dict(report["rows_by_lifecycle_window"]),
            "",
            "## Rows By Execution Blocking Reason",
            "",
            *_markdown_dict(report["rows_by_execution_blocking_reason"]),
            "",
            "## Rows By Selected Action Family",
            "",
            *_markdown_dict(report["rows_by_selected_action_family"]),
            "",
        ]
    )


def _fresh_time_window_diagnostic_md(report: dict[str, Any]) -> str:
    zero = report["zero_intent_explanation"]
    return "\n".join(
        [
            "# O v8 Paper Fresh Time Window Diagnostic",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- trace_row_count: `{report['trace_row_count']}`",
            f"- rows_blocked_by_time_to_close: `{report['rows_blocked_by_time_to_close']}`",
            f"- hts_selected_after_hts_window_expired_count: `{report['hts_selected_after_hts_window_expired_count']}`",
            f"- real_action_selected_inside_executable_window_count: `{report['real_action_selected_inside_executable_window_count']}`",
            f"- provider_collection_too_late_for_selected_action_family: `{str(report['provider_collection_too_late_for_selected_action_family']).lower()}`",
            f"- likely_collection_cadence_or_market_window_issue: `{str(zero['likely_collection_cadence_or_market_window_issue']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Rows By Lifecycle Window",
            "",
            *_markdown_dict(report["rows_by_lifecycle_window"]),
            "",
            "## Time To Close By Action Family",
            "",
            *_markdown_summary_dict(report["time_to_close_by_action_family"]),
            "",
        ]
    )


def _fresh_paper_position_state_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Position State",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- legacy_position_manager_reused: `{str(report['legacy_position_manager_reused']).lower()}`",
            f"- initial_paper_position_row_count: `{report['initial_paper_position_row_count']}`",
            f"- accepted_entry_fill_position_count: `{report['accepted_entry_fill_position_count']}`",
            f"- position_open_failed_count: `{report['position_open_failed_count']}`",
            f"- position_state_adapter_status: `{report['position_state_adapter_status']}`",
            f"- exit_decision_policy_source: `{report['exit_decision_policy_source']}`",
            f"- exit_thresholds_tuned: `{str(report['exit_thresholds_tuned']).lower()}`",
            f"- open_paper_position_count: `{report['open_paper_position_count']}`",
            f"- eligible_for_exit_signal_count: `{report['eligible_for_exit_signal_count']}`",
            f"- forbidden_outcome_fields_present: `{str(report['forbidden_outcome_fields_present']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Consumed Legacy Position Fields",
            "",
            *_markdown_list(report["legacy_position_manager_consumed_fields"]),
            "",
            "## Consumed Paper Ledger Fields",
            "",
            *_markdown_list(report["paper_ledger_fields_consumed_for_position_mapping"]),
            "",
            "## Blocking Reason Codes",
            "",
            *_markdown_list(report["position_state_blocking_reason_codes"]),
            "",
        ]
    )


def _fresh_legacy_position_policy_audit_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Legacy Position Policy Audit",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- reusable_legacy_decision_policy_found: `{str(report['reusable_legacy_decision_policy_found']).lower()}`",
            f"- legacy_state_manager_reused: `{str(report['legacy_state_manager_reused']).lower()}`",
            f"- legacy_decision_policy_reused: `{str(report['legacy_decision_policy_reused']).lower()}`",
            f"- exit_decision_policy_source: `{report['exit_decision_policy_source']}`",
            f"- exit_thresholds_tuned: `{str(report['exit_thresholds_tuned']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Discovered Modules",
            "",
            *[
                f"- `{row['module_or_script']}`: `{row['classification']}`, reuse `{row['current_adapter_reuse']}`"
                for row in report["discovered_modules_and_functions"]
            ],
            "",
        ]
    )


def _fresh_paper_exit_signal_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Exit Signal",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_exit_signal_count: `{report['paper_exit_signal_count']}`",
            f"- hold_no_exit_count: `{report['hold_no_exit_count']}`",
            f"- sell_position_signal_count: `{report['sell_position_signal_count']}`",
            f"- sell_position_intent_count: `{report['sell_position_intent_count']}`",
            f"- sell_position_intents_are_local_paper_only: `{str(report['sell_position_intents_are_local_paper_only']).lower()}`",
            f"- legacy_state_manager_reused: `{str(report['legacy_state_manager_reused']).lower()}`",
            f"- legacy_decision_policy_reused: `{str(report['legacy_decision_policy_reused']).lower()}`",
            f"- exit_decision_policy_source: `{report['exit_decision_policy_source']}`",
            f"- exit_threshold_profile_name: `{report['exit_threshold_profile_name']}`",
            f"- exit_thresholds_tuned: `{str(report['exit_thresholds_tuned']).lower()}`",
            f"- forbidden_outcome_fields_present: `{str(report['forbidden_outcome_fields_present']).lower()}`",
            f"- mutates_o_entry_scorer: `{str(report['mutates_o_entry_scorer']).lower()}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
            "## Exit Reason Distribution",
            "",
            *_markdown_dict(report["exit_reason_distribution"]),
            "",
            "## Consumed Legacy Signal Fields",
            "",
            *_markdown_list(report["legacy_consumed_signal_fields"]),
            "",
        ]
    )


def _fresh_synthetic_ledger_update_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# O v8 Paper Fresh Synthetic Ledger Update",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- paper_sell_position_intent_count: `{report['paper_sell_position_intent_count']}`",
            f"- synthetic_ledger_update_count: `{report['synthetic_ledger_update_count']}`",
            f"- ledger_updates_only_for_accepted_paper_exit_intents: `{str(report['ledger_updates_only_for_accepted_paper_exit_intents']).lower()}`",
            f"- exit_decision_policy_source: `{report['exit_decision_policy_source']}`",
            f"- exit_thresholds_tuned: `{str(report['exit_thresholds_tuned']).lower()}`",
            f"- synthetic_cash_delta_sum: `{report['synthetic_cash_delta_sum']}`",
            f"- synthetic_position_delta_sum: `{report['synthetic_position_delta_sum']}`",
            f"- v8_execution_handoff_allowed: `{str(report['v8_execution_handoff_allowed']).lower()}`",
            "",
        ]
    )


def _markdown_list(rows: list[str]) -> list[str]:
    return ["- none"] if not rows else [f"- `{row}`" for row in rows]


def _markdown_dict(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: `{value}`" for key, value in sorted(values.items())]


def _markdown_summary_dict(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [
        f"- `{key}`: count `{summary.get('count')}`, mean `{summary.get('mean')}`, min `{summary.get('min')}`, max `{summary.get('max')}`"
        for key, summary in sorted(values.items())
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return _sha256_file_existing(path) if path.exists() else ""
