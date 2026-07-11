"""One-hour paper-only goal diagnostics for Execution Layer v2 remaps."""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.recorder.contracts import PolymarketRealCorpusRecorderConfig
from bigan.v8.polymarket.recorder.public_provider import (
    PolymarketPublicHTTPRealCorpusProvider,
    RealCorpusPublicProviderError,
)
from bigan.v8.polymarket.training.contracts import compact_safety_fields
from bigan.v8.polymarket.training.o_v8_paper_fresh_loop import (
    O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
    O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE,
    PINNED_ISSUE_160_MANIFEST_SHA256,
    PolymarketOV8PaperFreshLoopConfig,
    _action_family,
    _apply_guard_row_to_runtime_state,
    _fresh_order_intent_from_guard_row,
    _fresh_paper_fills_from_intents,
    _fresh_paper_ledger_from_fills,
    _guard_input_from_public_row,
    _initial_fresh_runtime_state,
    _side_from_action,
    _v8_execution_guard_config,
    _v8_execution_guard_decision,
    run_polymarket_o_v8_paper_fresh_loop,
)

ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-paper-goal-v1"
)
ONE_HOUR_REMAP_PAPER_GOAL_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-paper-goal-manifest-v1"
)
ONE_HOUR_REMAP_ROUND_COVERAGE_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-round-coverage-v1"
)
ONE_HOUR_REMAP_EXECUTION_REPORT_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-remap-execution-v1"
)
ONE_HOUR_REMAP_SETTLEMENT_RESOLUTION_SCHEMA_VERSION = (
    "bigan-v8-polymarket-execution-layer-v2-one-hour-settlement-resolution-v1"
)

GUARD_JUSTIFIED_NO_BET_ALLOWED_BLOCKER_CATEGORIES = {
    "time_to_close",
    "spread",
    "p_up_disagreement",
    "exposure",
    "missing_runtime_field",
    "missing_candidate_or_metadata",
    "guard_blocked_other",
}

DEFAULT_ONE_HOUR_UNLOCK_DIR = Path(
    "examples/v8/polymarket_runs/o-v8-paper-candidate-unlock-20260703T073000Z"
)
DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_frozen_ev_calibration_v1.json"
)


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2OneHourRemapPaperGoalConfig:
    """Configuration for the one-hour paper-only remap goal runner."""

    run_id: str
    output_dir: Path | str
    duration_seconds: int = 3600
    poll_interval_seconds: float = 60.0
    paper_candidate_unlock_dir: Path | str = DEFAULT_ONE_HOUR_UNLOCK_DIR
    expected_paper_candidate_unlock_manifest_sha256: str | None = (
        PINNED_ISSUE_160_MANIFEST_SHA256
    )
    frozen_ev_calibration_artifact_path: Path | str = (
        DEFAULT_FROZEN_EV_CALIBRATION_ARTIFACT
    )
    canonical_o_source_manifest_path: Path | str | None = None
    public_data_cycles: tuple[tuple[dict[str, Any], ...], ...] | None = None
    public_provider: Any | None = None
    settlement_evaluation_rows: tuple[dict[str, Any], ...] = ()
    settlement_poll_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_consecutive_orderbook_failure_rounds: int = 3
    allow_short_diagnostic_run: bool = False
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.duration_seconds < 3600 and not self.allow_short_diagnostic_run:
            raise ValueError("duration_seconds must be at least 3600")
        if self.poll_interval_seconds < 0.0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.settlement_poll_max_wait_seconds < 0.0:
            raise ValueError("settlement_poll_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0.0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
        if self.max_consecutive_orderbook_failure_rounds <= 0:
            raise ValueError("max_consecutive_orderbook_failure_rounds must be positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "paper_candidate_unlock_dir", Path(self.paper_candidate_unlock_dir)
        )
        object.__setattr__(
            self,
            "frozen_ev_calibration_artifact_path",
            Path(self.frozen_ev_calibration_artifact_path),
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
        if not isinstance(self.settlement_evaluation_rows, tuple):
            object.__setattr__(
                self,
                "settlement_evaluation_rows",
                tuple(dict(row) for row in self.settlement_evaluation_rows),
            )


@dataclass(frozen=True, slots=True)
class ExecutionLayerV2OneHourRemapPaperGoalResult:
    """Generated one-hour remap goal bundle."""

    output_dir: Path
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    goal_report: dict[str, Any]
    round_coverage_report: dict[str, Any]
    remap_execution_report: dict[str, Any]
    settlement_resolution_report: dict[str, Any]
    manifest: dict[str, Any]


def _run_incremental_read_only_provider_fresh_loop(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    goal_dir: Path,
    max_cycles: int,
    base_fresh_loop_config: PolymarketOV8PaperFreshLoopConfig,
) -> SimpleNamespace:
    """Collect read-only provider data once per poll cycle and fail fast on book loss."""

    aggregate_dir = goal_dir / "incremental_fresh_loop"
    cycle_output_dir = goal_dir / "incremental_fresh_loop_cycles"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    cycle_output_dir.mkdir(parents=True, exist_ok=True)

    all_trace_rows: list[dict[str, Any]] = []
    all_intents: list[dict[str, Any]] = []
    all_fills: list[dict[str, Any]] = []
    all_ledger_rows: list[dict[str, Any]] = []
    all_remap_rows: list[dict[str, Any]] = []
    all_raw_market_rows: list[dict[str, Any]] = []
    all_raw_orderbook_rows: list[dict[str, Any]] = []
    all_raw_trade_rows: list[dict[str, Any]] = []
    all_raw_btc_candle_rows: list[dict[str, Any]] = []
    cycle_status_rows: list[dict[str, Any]] = []
    cycle_results: list[Any] = []
    reason_counter: Counter[str] = Counter()
    block_counter: Counter[str] = Counter()
    consecutive_orderbook_failures = 0
    max_consecutive_failures = config.max_consecutive_orderbook_failure_rounds
    fail_fast_stop_triggered = False
    fail_fast_reason_codes: list[str] = []

    for cycle_index in range(max_cycles):
        cycle_run_id = f"{base_fresh_loop_config.run_id}-cycle-{cycle_index + 1:06d}"
        cycle_config = PolymarketOV8PaperFreshLoopConfig(
            run_id=cycle_run_id,
            output_dir=cycle_output_dir,
            paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                config.expected_paper_candidate_unlock_manifest_sha256
            ),
            loop_mode="single_cycle",
            max_cycles=1,
            sleep_seconds=0.0,
            public_data_cycles=None,
            public_data_source=O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
            public_provider=config.public_provider,
            canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
            frozen_ev_calibration_artifact_path=(
                config.frozen_ev_calibration_artifact_path
            ),
            overwrite_existing=config.overwrite_existing,
        )
        cycle_result = run_polymarket_o_v8_paper_fresh_loop(cycle_config)
        cycle_results.append(cycle_result)

        public_report = cycle_result.fresh_loop_run_report[
            "public_data_collection_report"
        ]
        orderbook_count = int(public_report.get("public_orderbook_row_count") or 0)
        feature_count = int(public_report.get("public_feature_row_count") or 0)
        orderbook_failure = orderbook_count <= 0 or feature_count <= 0
        if orderbook_failure:
            consecutive_orderbook_failures += 1
        else:
            consecutive_orderbook_failures = 0

        reason_codes = list(public_report.get("public_data_collection_reason_codes") or [])
        reason_counter.update(reason_codes)
        block_counter.update(
            cycle_result.fresh_loop_run_report.get("block_reason_distribution") or {}
        )
        cycle_trace_rows = list(cycle_result.signal_trace_report.get("trace_rows") or [])
        cycle_intents = _read_jsonl(cycle_result.artifact_paths["fresh_order_intent_log"])
        cycle_fills = _read_jsonl(cycle_result.artifact_paths["fresh_fill_log"])
        cycle_ledger_rows = _read_jsonl(cycle_result.artifact_paths["fresh_ledger_log"])
        cycle_raw_market_rows = _read_jsonl(
            cycle_result.artifact_paths["raw_polymarket_markets"]
        )
        cycle_raw_orderbook_rows = _read_jsonl(
            cycle_result.artifact_paths["raw_polymarket_orderbooks"]
        )
        cycle_raw_trade_rows = _read_jsonl(
            cycle_result.artifact_paths["raw_polymarket_trades"]
        )
        cycle_raw_btc_candle_rows = _read_jsonl(
            cycle_result.artifact_paths["raw_btc_feature_candles"]
        )
        all_trace_rows.extend(cycle_trace_rows)
        all_intents.extend(cycle_intents)
        all_fills.extend(cycle_fills)
        all_ledger_rows.extend(cycle_ledger_rows)
        all_raw_market_rows.extend(cycle_raw_market_rows)
        all_raw_orderbook_rows.extend(cycle_raw_orderbook_rows)
        all_raw_trade_rows.extend(cycle_raw_trade_rows)
        all_raw_btc_candle_rows.extend(cycle_raw_btc_candle_rows)
        all_remap_rows.extend(
            cycle_result.execution_layer_v2_paper_remap_report.get("remap_rows", [])
        )

        cycle_status = {
            "cycle_index": cycle_index,
            "cycle_run_id": cycle_run_id,
            "provider_collection_failed": public_report[
                "paper_fresh_provider_collection_failed"
            ],
            "orderbook_failure": orderbook_failure,
            "consecutive_orderbook_failure_count": consecutive_orderbook_failures,
            "max_consecutive_orderbook_failure_rounds": max_consecutive_failures,
            "public_data_collection_reason_codes": reason_codes,
            "public_market_count": public_report.get("public_market_count"),
            "public_orderbook_row_count": orderbook_count,
            "orderbook_source_type_distribution": public_report.get(
                "orderbook_source_type_distribution", {}
            ),
            "orderbook_rest_fallback_row_count": int(
                public_report.get("orderbook_rest_fallback_row_count") or 0
            ),
            "orderbook_fallback_reason_distribution": public_report.get(
                "orderbook_fallback_reason_distribution", {}
            ),
            "public_trade_row_count": public_report.get("public_trade_row_count"),
            "public_btc_feature_candle_row_count": public_report.get(
                "public_btc_feature_candle_row_count"
            ),
            "public_feature_row_count": feature_count,
            "signal_trace_row_count": len(cycle_trace_rows),
            "paper_intent_count": len(cycle_intents),
            "paper_fill_count": len(cycle_fills),
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "v8_execution_handoff_allowed": False,
        }
        cycle_status_rows.append(cycle_status)
        _write_jsonl(
            aggregate_dir / "provider_cycle_status.jsonl",
            cycle_status_rows,
        )

        _write_per_round_bet_artifacts(
            goal_dir=goal_dir,
            intents=all_intents,
            fills=all_fills,
            ledger_rows=all_ledger_rows,
            trace_rows=all_trace_rows,
            raw_market_rows=all_raw_market_rows,
            raw_orderbook_rows=all_raw_orderbook_rows,
            raw_trade_rows=all_raw_trade_rows,
            raw_btc_candle_rows=all_raw_btc_candle_rows,
        )

        if consecutive_orderbook_failures >= max_consecutive_failures:
            fail_fast_stop_triggered = True
            fail_fast_reason_codes = [
                "consecutive_orderbook_collection_failures_exceeded_limit"
            ]
            break
        if cycle_index < max_cycles - 1 and config.poll_interval_seconds > 0.0:
            time.sleep(config.poll_interval_seconds)

    attempted_cycle_count = len(cycle_status_rows)
    aggregate_public_report = _aggregate_public_collection_report(
        config=config,
        cycle_status_rows=cycle_status_rows,
        cycle_results=cycle_results,
        reason_counter=reason_counter,
        fail_fast_stop_triggered=fail_fast_stop_triggered,
        fail_fast_reason_codes=fail_fast_reason_codes,
    )
    remap_report = _aggregate_incremental_remap_report(
        run_id=base_fresh_loop_config.run_id,
        cycle_results=cycle_results,
        remap_rows=all_remap_rows,
        intents=all_intents,
    )
    run_report = _aggregate_incremental_fresh_loop_run_report(
        config=base_fresh_loop_config,
        attempted_cycle_count=attempted_cycle_count,
        max_cycles=max_cycles,
        public_collection_report=aggregate_public_report,
        trace_rows=all_trace_rows,
        intents=all_intents,
        fills=all_fills,
        ledger_rows=all_ledger_rows,
        block_counter=block_counter,
    )
    runtime_safety_report = _aggregate_incremental_runtime_safety_report(
        run_id=base_fresh_loop_config.run_id,
        cycle_results=cycle_results,
    )
    signal_trace_report = _aggregate_incremental_signal_trace_report(
        run_id=base_fresh_loop_config.run_id,
        trace_rows=all_trace_rows,
    )

    artifact_paths = {
        "manifest": aggregate_dir / "o_v8_paper_fresh_loop_manifest.json",
        "fresh_loop_run_report": aggregate_dir / "o_v8_paper_fresh_loop_run_report.json",
        "fresh_order_intent_log": aggregate_dir / "o_v8_paper_fresh_order_intent_log.jsonl",
        "fresh_fill_log": aggregate_dir / "o_v8_paper_fresh_fill_log.jsonl",
        "fresh_ledger_log": aggregate_dir / "o_v8_paper_fresh_ledger_log.jsonl",
        "provider_cycle_status": aggregate_dir / "provider_cycle_status.jsonl",
        "execution_layer_v2_paper_remap_report": aggregate_dir
        / "execution_layer_v2_paper_remap_report.json",
        "signal_trace_report": aggregate_dir / "o_v8_paper_fresh_signal_trace.json",
        "runtime_safety_report": aggregate_dir
        / "o_v8_paper_fresh_runtime_safety_report.json",
        "raw_polymarket_markets": aggregate_dir / "raw_polymarket_markets.jsonl",
        "raw_polymarket_orderbooks": aggregate_dir
        / "raw_polymarket_orderbooks.jsonl",
        "raw_polymarket_trades": aggregate_dir / "raw_polymarket_trades.jsonl",
        "raw_btc_feature_candles": aggregate_dir
        / "raw_btc_feature_candles.jsonl",
    }
    _write_jsonl(artifact_paths["fresh_order_intent_log"], all_intents)
    _write_jsonl(artifact_paths["fresh_fill_log"], all_fills)
    _write_jsonl(artifact_paths["fresh_ledger_log"], all_ledger_rows)
    _write_json(artifact_paths["fresh_loop_run_report"], run_report)
    _write_json(artifact_paths["execution_layer_v2_paper_remap_report"], remap_report)
    _write_json(artifact_paths["signal_trace_report"], signal_trace_report)
    _write_json(artifact_paths["runtime_safety_report"], runtime_safety_report)
    _write_jsonl(artifact_paths["raw_polymarket_markets"], all_raw_market_rows)
    _write_jsonl(artifact_paths["raw_polymarket_orderbooks"], all_raw_orderbook_rows)
    _write_jsonl(artifact_paths["raw_polymarket_trades"], all_raw_trade_rows)
    _write_jsonl(artifact_paths["raw_btc_feature_candles"], all_raw_btc_candle_rows)
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "manifest"
    }
    manifest = _aggregate_incremental_fresh_loop_manifest(
        run_id=base_fresh_loop_config.run_id,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        run_report=run_report,
        remap_report=remap_report,
        runtime_safety_report=runtime_safety_report,
    )
    _write_json(artifact_paths["manifest"], manifest)
    artifact_hashes["manifest"] = _sha256_file(artifact_paths["manifest"])
    manifest["artifact_hashes"] = dict(artifact_hashes)
    _write_json(artifact_paths["manifest"], manifest)

    return SimpleNamespace(
        artifact_paths=artifact_paths,
        fresh_loop_run_report=run_report,
        runtime_safety_report=runtime_safety_report,
        execution_layer_v2_paper_remap_report=remap_report,
        signal_trace_report=signal_trace_report,
        manifest=manifest,
    )


def run_execution_layer_v2_one_hour_remap_paper_goal(
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
) -> ExecutionLayerV2OneHourRemapPaperGoalResult:
    """Run the paper-only remap path and write one-hour goal diagnostics."""

    goal_dir = Path(config.output_dir) / config.run_id
    if goal_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"one-hour remap goal output already exists: {goal_dir}")
        shutil.rmtree(goal_dir)
    goal_dir.mkdir(parents=True)

    public_data_source = (
        O_V8_PUBLIC_DATA_SOURCE_SNAPSHOT_FIXTURE
        if config.public_data_cycles is not None
        else O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
    )
    max_cycles = (
        len(config.public_data_cycles)
        if config.public_data_cycles is not None
        else max(1, int(config.duration_seconds / max(config.poll_interval_seconds, 1.0)))
    )
    fresh_loop_config = PolymarketOV8PaperFreshLoopConfig(
        run_id=f"{config.run_id}-fresh-loop",
        output_dir=goal_dir,
        paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
        expected_paper_candidate_unlock_manifest_sha256=(
            config.expected_paper_candidate_unlock_manifest_sha256
        ),
        loop_mode="bounded_recurring" if max_cycles > 1 else "single_cycle",
        max_cycles=max_cycles,
        sleep_seconds=0.0
        if config.public_data_cycles is not None
        else config.poll_interval_seconds,
        public_data_cycles=config.public_data_cycles,
        public_data_source=public_data_source,
        public_provider=config.public_provider,
        canonical_o_source_manifest_path=config.canonical_o_source_manifest_path,
        frozen_ev_calibration_artifact_path=(
            config.frozen_ev_calibration_artifact_path
        ),
        overwrite_existing=config.overwrite_existing,
    )
    if config.public_data_cycles is None:
        fresh_result = _run_incremental_read_only_provider_fresh_loop(
            config=config,
            goal_dir=goal_dir,
            max_cycles=max_cycles,
            base_fresh_loop_config=fresh_loop_config,
        )
    else:
        fresh_result = run_polymarket_o_v8_paper_fresh_loop(fresh_loop_config)

    base_intents = _read_jsonl(fresh_result.artifact_paths["fresh_order_intent_log"])
    trace_rows = list(fresh_result.signal_trace_report.get("trace_rows") or [])
    forced_coverage = _forced_coverage_attempt_report(
        config=config,
        fresh_loop_config=PolymarketOV8PaperFreshLoopConfig(
            run_id=f"{config.run_id}-fresh-loop",
            output_dir=goal_dir,
            paper_candidate_unlock_dir=config.paper_candidate_unlock_dir,
            expected_paper_candidate_unlock_manifest_sha256=(
                config.expected_paper_candidate_unlock_manifest_sha256
            ),
            public_data_source=public_data_source,
            public_data_cycles=(),
        ),
        trace_rows=trace_rows,
        base_intents=base_intents,
    )
    intents = [*base_intents, *forced_coverage["forced_coverage_intents"]]
    fills = _fresh_paper_fills_from_intents(intents)
    ledger_rows = _fresh_paper_ledger_from_fills(fills)
    settlement_resolution = _settlement_resolution_report(
        config=config,
        fills=fills,
        trace_rows=trace_rows,
        settlement_evaluation_rows=list(config.settlement_evaluation_rows),
    )
    settlement_rows = _settlement_pnl_rows(
        fills=fills,
        settlement_evaluation_rows=[
            *list(config.settlement_evaluation_rows),
            *settlement_resolution["settlement_evaluation_rows"],
        ],
    )
    round_artifacts = _write_per_round_artifacts(
        goal_dir=goal_dir,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
        settlement_rows=settlement_rows,
        trace_rows=trace_rows,
        raw_market_rows=_read_optional_jsonl(
            fresh_result.artifact_paths.get("raw_polymarket_markets")
        ),
        raw_orderbook_rows=_read_optional_jsonl(
            fresh_result.artifact_paths.get("raw_polymarket_orderbooks")
        ),
        raw_trade_rows=_read_optional_jsonl(
            fresh_result.artifact_paths.get("raw_polymarket_trades")
        ),
        raw_btc_candle_rows=_read_optional_jsonl(
            fresh_result.artifact_paths.get("raw_btc_feature_candles")
        ),
    )
    round_coverage = _round_coverage_report(
        run_id=config.run_id,
        trace_rows=trace_rows,
        intents=intents,
        forced_coverage=forced_coverage,
    )
    remap_execution = _remap_execution_report(
        run_id=config.run_id,
        fresh_remap_report=fresh_result.execution_layer_v2_paper_remap_report,
        intents=intents,
        fills=fills,
        forced_coverage=forced_coverage,
    )
    goal_report = _one_hour_goal_report(
        config=config,
        fresh_result=fresh_result,
        round_coverage=round_coverage,
        remap_execution=remap_execution,
        intents=intents,
        fills=fills,
        settlement_rows=settlement_rows,
        settlement_resolution=settlement_resolution,
        round_artifacts=round_artifacts,
    )

    artifact_paths = {
        "one_hour_remap_paper_goal_report": goal_dir
        / "one_hour_remap_paper_goal_report.json",
        "one_hour_remap_paper_goal_summary": goal_dir
        / "one_hour_remap_paper_goal_report.md",
        "one_hour_remap_paper_goal_manifest": goal_dir
        / "one_hour_remap_paper_goal_manifest.json",
        "round_coverage_report": goal_dir / "round_coverage_report.json",
        "round_coverage_summary": goal_dir / "round_coverage_report.md",
        "remap_execution_report": goal_dir / "remap_execution_report.json",
        "remap_execution_summary": goal_dir / "remap_execution_report.md",
        "settlement_resolution_report": goal_dir
        / "settlement_resolution_report.json",
        "settlement_resolution_summary": goal_dir
        / "settlement_resolution_report.md",
        "settlement_pnl_rows": goal_dir / "settlement_pnl_rows.jsonl",
        "paper_intent_log": goal_dir / "one_hour_paper_intent_log.jsonl",
        "paper_fill_log": goal_dir / "one_hour_paper_fill_log.jsonl",
        "paper_ledger_log": goal_dir / "one_hour_paper_ledger_log.jsonl",
        "per_round_artifact_manifest": Path(round_artifacts["manifest_path"]),
        "paper_fresh_loop_manifest": fresh_result.artifact_paths["manifest"],
        "paper_remap_report": fresh_result.artifact_paths[
            "execution_layer_v2_paper_remap_report"
        ],
    }
    _write_jsonl(artifact_paths["paper_intent_log"], intents)
    _write_jsonl(artifact_paths["paper_fill_log"], fills)
    _write_jsonl(artifact_paths["paper_ledger_log"], ledger_rows)
    _write_json(artifact_paths["one_hour_remap_paper_goal_report"], goal_report)
    _write_text(
        artifact_paths["one_hour_remap_paper_goal_summary"],
        _one_hour_goal_md(goal_report),
    )
    _write_json(artifact_paths["round_coverage_report"], round_coverage)
    _write_text(
        artifact_paths["round_coverage_summary"],
        _round_coverage_md(round_coverage),
    )
    _write_json(artifact_paths["remap_execution_report"], remap_execution)
    _write_text(
        artifact_paths["remap_execution_summary"],
        _remap_execution_md(remap_execution),
    )
    _write_json(artifact_paths["settlement_resolution_report"], settlement_resolution)
    _write_text(
        artifact_paths["settlement_resolution_summary"],
        _settlement_resolution_md(settlement_resolution),
    )
    _write_jsonl(artifact_paths["settlement_pnl_rows"], settlement_rows)
    artifact_hashes = {
        name: _sha256_file(path)
        for name, path in sorted(artifact_paths.items())
        if name != "one_hour_remap_paper_goal_manifest"
    }
    manifest = _one_hour_goal_manifest(
        config=config,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        goal_report=goal_report,
        round_coverage=round_coverage,
        remap_execution=remap_execution,
        settlement_resolution=settlement_resolution,
        fresh_manifest=fresh_result.manifest,
    )
    _write_json(artifact_paths["one_hour_remap_paper_goal_manifest"], manifest)
    artifact_hashes["one_hour_remap_paper_goal_manifest"] = _sha256_file(
        artifact_paths["one_hour_remap_paper_goal_manifest"]
    )
    manifest["artifact_hashes"] = dict(artifact_hashes)
    _write_json(artifact_paths["one_hour_remap_paper_goal_manifest"], manifest)
    artifact_hashes["one_hour_remap_paper_goal_manifest"] = _sha256_file(
        artifact_paths["one_hour_remap_paper_goal_manifest"]
    )

    return ExecutionLayerV2OneHourRemapPaperGoalResult(
        output_dir=goal_dir,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        goal_report=goal_report,
        round_coverage_report=round_coverage,
        remap_execution_report=remap_execution,
        settlement_resolution_report=settlement_resolution,
        manifest=manifest,
    )


def _round_coverage_report(
    *,
    run_id: str,
    trace_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    forced_coverage: dict[str, Any],
) -> dict[str, Any]:
    complete_rounds = sorted({str(row.get("market_id")) for row in trace_rows if row.get("market_id")})
    bet_rounds = sorted({str(row.get("market_id")) for row in intents if row.get("market_id")})
    missing = sorted(set(complete_rounds) - set(bet_rounds))
    missing_classifications = _missing_bet_round_classifications(
        missing_round_ids=missing,
        forced_coverage=forced_coverage,
    )
    classification_by_market = {
        row["market_id"]: row for row in missing_classifications["classification_rows"]
    }
    rows = [
        {
            "market_id": market_id,
            "complete_round": True,
            "paper_bet_created": market_id in set(bet_rounds),
            "paper_bet_count": sum(
                1 for intent in intents if str(intent.get("market_id")) == market_id
            ),
            "missing_bet_classification": classification_by_market.get(
                market_id,
                {
                    "missing_bet_round": market_id in set(missing),
                    "guard_justified_no_bet": False,
                    "unjustified_missing_bet": market_id in set(missing),
                    "missing_bet_classification_reason_codes": [],
                },
            ),
        }
        for market_id in complete_rounds
    ]
    report = {
        "schema_version": ONE_HOUR_REMAP_ROUND_COVERAGE_SCHEMA_VERSION,
        "report_type": "one_hour_remap_round_coverage",
        "run_id": run_id,
        "complete_round_count": len(complete_rounds),
        "complete_rounds_with_bet_count": len(bet_rounds),
        "missing_bet_round_count": len(missing),
        "missing_bet_round_ids": missing,
        "guard_justified_no_bet_round_count": missing_classifications[
            "guard_justified_no_bet_round_count"
        ],
        "guard_justified_no_bet_round_ids": missing_classifications[
            "guard_justified_no_bet_round_ids"
        ],
        "unjustified_missing_bet_round_count": missing_classifications[
            "unjustified_missing_bet_round_count"
        ],
        "unjustified_missing_bet_round_ids": missing_classifications[
            "unjustified_missing_bet_round_ids"
        ],
        "guard_justified_no_bet_blocker_category_distribution": (
            missing_classifications[
                "guard_justified_no_bet_blocker_category_distribution"
            ]
        ),
        "unjustified_missing_bet_reason_distribution": missing_classifications[
            "unjustified_missing_bet_reason_distribution"
        ],
        "round_coverage_rows": rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    return _with_report_id(report, "round_coverage_report_id")


def _missing_bet_round_classifications(
    *,
    missing_round_ids: list[str],
    forced_coverage: dict[str, Any],
) -> dict[str, Any]:
    attempt_by_market = {
        str(row.get("market_id")): row
        for row in forced_coverage.get("forced_coverage_attempt_rows", [])
    }
    classification_rows = [
        _classify_missing_bet_round(
            market_id=market_id,
            attempt_row=attempt_by_market.get(market_id),
        )
        for market_id in missing_round_ids
    ]
    justified_ids = sorted(
        row["market_id"]
        for row in classification_rows
        if row["guard_justified_no_bet"] is True
    )
    unjustified_ids = sorted(
        row["market_id"]
        for row in classification_rows
        if row["unjustified_missing_bet"] is True
    )
    category_counter = Counter(
        category
        for row in classification_rows
        if row["guard_justified_no_bet"] is True
        for category in row["blocker_categories"]
    )
    unjustified_reason_counter = Counter(
        reason
        for row in classification_rows
        if row["unjustified_missing_bet"] is True
        for reason in row["missing_bet_classification_reason_codes"]
    )
    return {
        "classification_rows": classification_rows,
        "guard_justified_no_bet_round_count": len(justified_ids),
        "guard_justified_no_bet_round_ids": justified_ids,
        "unjustified_missing_bet_round_count": len(unjustified_ids),
        "unjustified_missing_bet_round_ids": unjustified_ids,
        "guard_justified_no_bet_blocker_category_distribution": dict(
            sorted(category_counter.items())
        ),
        "unjustified_missing_bet_reason_distribution": dict(
            sorted(unjustified_reason_counter.items())
        ),
    }


def _classify_missing_bet_round(
    *,
    market_id: str,
    attempt_row: dict[str, Any] | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if attempt_row is None:
        reason_codes.append("forced_coverage_attempt_missing")
        return _missing_bet_round_classification_row(
            market_id=market_id,
            attempt_row={},
            guard_justified=False,
            reason_codes=reason_codes,
        )
    candidate_rows = list(attempt_row.get("forced_coverage_candidate_attempt_rows") or [])
    candidate_attempt_count = int(
        attempt_row.get("forced_coverage_candidate_attempt_count") or 0
    )
    blocker_categories = set(attempt_row.get("forced_coverage_blocker_categories") or [])
    if attempt_row.get("coverage_forced_attempted") is not True:
        reason_codes.append("forced_coverage_not_attempted")
    if candidate_attempt_count != len(candidate_rows):
        reason_codes.append("forced_coverage_candidate_attempt_detail_mismatch")
    if candidate_attempt_count > 0 and not all(
        "order_allowed" in row for row in candidate_rows
    ):
        reason_codes.append("forced_coverage_candidates_not_checked_through_guard")
    if any(row.get("order_allowed") is True for row in candidate_rows):
        reason_codes.append("guard_passing_candidate_existed")
    if attempt_row.get("forced_coverage_candidate_search_found_guard_passed") is True:
        reason_codes.append("guard_passing_candidate_search_flag_true")
    if attempt_row.get("forced_coverage_guard_passed") is True:
        reason_codes.append("forced_coverage_guard_passed_without_bet")
    if not blocker_categories:
        reason_codes.append("missing_blocker_category_diagnostics")
    disallowed_categories = sorted(
        blocker_categories - GUARD_JUSTIFIED_NO_BET_ALLOWED_BLOCKER_CATEGORIES
    )
    if disallowed_categories:
        reason_codes.extend(
            f"disallowed_blocker_category:{category}"
            for category in disallowed_categories
        )
    if attempt_row.get("uses_settlement_pnl_or_outcome_labels_in_decision_logic") is True:
        reason_codes.append("outcome_or_pnl_used_in_decision_logic")
    if any(
        row.get("uses_settlement_pnl_or_outcome_labels_in_decision_logic") is True
        for row in candidate_rows
    ):
        reason_codes.append("candidate_used_outcome_or_pnl_in_decision_logic")
    return _missing_bet_round_classification_row(
        market_id=market_id,
        attempt_row=attempt_row,
        guard_justified=not reason_codes,
        reason_codes=reason_codes,
    )


def _missing_bet_round_classification_row(
    *,
    market_id: str,
    attempt_row: dict[str, Any],
    guard_justified: bool,
    reason_codes: list[str],
) -> dict[str, Any]:
    row = {
        "market_id": market_id,
        "missing_bet_round": True,
        "guard_justified_no_bet": guard_justified,
        "unjustified_missing_bet": not guard_justified,
        "missing_bet_classification_reason_codes": []
        if guard_justified
        else sorted(set(reason_codes or ["unjustified_missing_bet"])),
        "forced_coverage_attempted": bool(
            attempt_row.get("coverage_forced_attempted")
        ),
        "forced_coverage_guard_passed": bool(
            attempt_row.get("forced_coverage_guard_passed")
        ),
        "forced_coverage_candidate_search_found_guard_passed": bool(
            attempt_row.get("forced_coverage_candidate_search_found_guard_passed")
        ),
        "forced_coverage_candidate_attempt_count": int(
            attempt_row.get("forced_coverage_candidate_attempt_count") or 0
        ),
        "forced_coverage_source_trace_count": int(
            attempt_row.get("forced_coverage_source_trace_count") or 0
        ),
        "selected_forced_coverage_candidate_action": attempt_row.get(
            "forced_coverage_selected_action"
        ),
        "execution_guarded_action": attempt_row.get(
            "forced_coverage_execution_guarded_action"
        ),
        "blocker_categories": list(
            attempt_row.get("forced_coverage_blocker_categories") or []
        ),
        "blocking_reason_codes": list(
            attempt_row.get("forced_coverage_blocking_reason_codes") or []
        ),
        "missing_trace_metadata_codes": list(
            attempt_row.get("forced_coverage_missing_trace_metadata_codes") or []
        ),
        "missing_runtime_field_codes": list(
            attempt_row.get("forced_coverage_missing_runtime_field_codes") or []
        ),
        "uses_settlement_pnl_or_outcome_labels_in_decision_logic": bool(
            attempt_row.get(
                "uses_settlement_pnl_or_outcome_labels_in_decision_logic"
            )
        ),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    row["missing_bet_round_classification_hash"] = canonical_json_sha256(row)
    return row


def _aggregate_public_collection_report(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    cycle_status_rows: list[dict[str, Any]],
    cycle_results: list[Any],
    reason_counter: Counter[str],
    fail_fast_stop_triggered: bool,
    fail_fast_reason_codes: list[str],
) -> dict[str, Any]:
    public_reports = [
        result.fresh_loop_run_report["public_data_collection_report"]
        for result in cycle_results
    ]
    reason_codes = sorted(
        {
            reason
            for report in public_reports
            for reason in report.get("public_data_collection_reason_codes", [])
        }
    )
    if fail_fast_stop_triggered:
        reason_codes.extend(
            reason
            for reason in fail_fast_reason_codes
            if reason not in reason_codes
        )
    provider_class = (
        public_reports[-1].get("public_provider_class") if public_reports else None
    )
    orderbook_source_counter: Counter[str] = Counter()
    orderbook_fallback_reason_counter: Counter[str] = Counter()
    for report in public_reports:
        orderbook_source_counter.update(
            {
                str(source): int(count)
                for source, count in (
                    report.get("orderbook_source_type_distribution") or {}
                ).items()
            }
        )
        orderbook_fallback_reason_counter.update(
            {
                str(reason): int(count)
                for reason, count in (
                    report.get("orderbook_fallback_reason_distribution") or {}
                ).items()
            }
        )
    return {
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_data_collection_mode": "read_only_public_provider_live_polling",
        "public_provider_class": provider_class,
        "public_provider_read_only": True,
        "public_provider_safety_passed": all(
            bool(report.get("public_provider_safety_passed")) for report in public_reports
        )
        if public_reports
        else False,
        "paper_fresh_provider_collection_failed": any(
            bool(report.get("paper_fresh_provider_collection_failed"))
            for report in public_reports
        ),
        "public_data_collection_reason_codes": reason_codes,
        "public_data_collection_reason_distribution": dict(
            sorted(reason_counter.items())
        ),
        "public_data_cycle_count": len(cycle_status_rows),
        "public_data_row_count": sum(
            int(report.get("public_data_row_count") or 0) for report in public_reports
        ),
        "public_market_count": sum(
            int(report.get("public_market_count") or 0) for report in public_reports
        ),
        "public_orderbook_row_count": sum(
            int(report.get("public_orderbook_row_count") or 0)
            for report in public_reports
        ),
        "orderbook_source_type_distribution": dict(
            sorted(orderbook_source_counter.items())
        ),
        "orderbook_rest_fallback_row_count": sum(
            int(report.get("orderbook_rest_fallback_row_count") or 0)
            for report in public_reports
        ),
        "orderbook_fallback_reason_distribution": dict(
            sorted(orderbook_fallback_reason_counter.items())
        ),
        "public_trade_row_count": sum(
            int(report.get("public_trade_row_count") or 0) for report in public_reports
        ),
        "public_btc_feature_candle_row_count": sum(
            int(report.get("public_btc_feature_candle_row_count") or 0)
            for report in public_reports
        ),
        "public_feature_row_count": sum(
            int(report.get("public_feature_row_count") or 0)
            for report in public_reports
        ),
        "provider_exception_type": None,
        "provider_exception_message": None,
        "provider_fail_fast_stop_triggered": fail_fast_stop_triggered,
        "provider_fail_fast_reason_codes": list(fail_fast_reason_codes),
        "max_consecutive_orderbook_failure_rounds": (
            config.max_consecutive_orderbook_failure_rounds
        ),
        "consecutive_orderbook_failure_count_at_stop": (
            cycle_status_rows[-1]["consecutive_orderbook_failure_count"]
            if cycle_status_rows
            else 0
        ),
        "cycle_status_rows": cycle_status_rows,
        "frozen_o_action_rank_reference_source": (
            public_reports[-1].get("frozen_o_action_rank_reference_source")
            if public_reports
            else "issue_160_paper_candidate_unlock_manifest"
        ),
        "frozen_o_action_rank_reference_sha256": (
            public_reports[-1].get("frozen_o_action_rank_reference_sha256")
            if public_reports
            else ""
        ),
        "scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_frozen_o_scorer_used": any(
            bool(report.get("canonical_frozen_o_scorer_used"))
            for report in public_reports
        ),
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "uses_validation_outcomes_for_tuning": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "thresholds_tuned": False,
        "forbidden_outcome_fields_used": [],
    }


def _aggregate_incremental_remap_report(
    *,
    run_id: str,
    cycle_results: list[Any],
    remap_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    remap_intents = [
        intent for intent in intents if intent.get("hts_time_window_remap_applied") is True
    ]
    report = {
        "schema_version": "bigan-v8-polymarket-execution-layer-v2-paper-remap-v1",
        "report_type": "execution_layer_v2_paper_remap",
        "phase": "polymarket_policy_training",
        "run_id": run_id,
        "paper_only_intent_path": True,
        "execution_layer_v2_paper_remap_enabled": True,
        "hts_time_window_blocked_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("hts_time_window_blocked_count") or 0)
            for result in cycle_results
        ),
        "same_side_sbc_alternative_available_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("same_side_sbc_alternative_available_count") or 0)
            for result in cycle_results
        ),
        "same_side_sbc_calibrated_ev_available_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("same_side_sbc_calibrated_ev_available_count") or 0)
            for result in cycle_results
        ),
        "same_side_sbc_guard_passed_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("same_side_sbc_guard_passed_count") or 0)
            for result in cycle_results
        ),
        "remap_candidate_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("remap_candidate_count") or 0)
            for result in cycle_results
        ),
        "remap_guard_passed_count": sum(
            int(result.execution_layer_v2_paper_remap_report.get("remap_guard_passed_count") or 0)
            for result in cycle_results
        ),
        "paper_intent_remap_applied_count": len(remap_intents),
        "paper_intent_remap_ids": [
            str(intent.get("paper_fresh_order_intent_id")) for intent in remap_intents
        ],
        "original_action_distribution": dict(
            sorted(Counter(str(row.get("original_action")) for row in remap_rows).items())
        ),
        "remapped_action_distribution": dict(
            sorted(Counter(str(row.get("remapped_action")) for row in remap_rows).items())
        ),
        "remap_reason_distribution": _reason_distribution(remap_rows, "remap_reason_codes"),
        "remap_failure_reason_distribution": _reason_distribution(
            [
                row
                for row in remap_rows
                if row.get("hts_time_window_remap_applied") is not True
            ],
            "remap_reason_codes",
        ),
        "remap_rows": remap_rows,
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


def _aggregate_incremental_fresh_loop_run_report(
    *,
    config: PolymarketOV8PaperFreshLoopConfig,
    attempted_cycle_count: int,
    max_cycles: int,
    public_collection_report: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    block_counter: Counter[str],
) -> dict[str, Any]:
    blockers = []
    if public_collection_report["paper_fresh_provider_collection_failed"]:
        blockers.append("paper_fresh_public_provider_collection_failed")
        blockers.extend(public_collection_report["public_data_collection_reason_codes"])
    if public_collection_report["provider_fail_fast_stop_triggered"]:
        blockers.extend(public_collection_report["provider_fail_fast_reason_codes"])
    report = {
        "schema_version": "bigan-v8-polymarket-o-v8-paper-fresh-loop-run-v1",
        "report_type": "o_v8_paper_fresh_loop_run",
        "phase": "polymarket_policy_training",
        "run_id": config.run_id,
        "paper_fresh_loop_enabled": True,
        "paper_fresh_loop_mode": "bounded_recurring_live_polling",
        "paper_fresh_loop_cycle_count": attempted_cycle_count,
        "paper_fresh_loop_max_cycles": max_cycles,
        "paper_fresh_loop_sleep_seconds": config.sleep_seconds,
        "paper_fresh_loop_public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "public_data_collection_report": public_collection_report,
        "paper_fresh_provider_collection_failed": public_collection_report[
            "paper_fresh_provider_collection_failed"
        ],
        "public_data_collection_reason_codes": public_collection_report[
            "public_data_collection_reason_codes"
        ],
        "provider_fail_fast_stop_triggered": public_collection_report[
            "provider_fail_fast_stop_triggered"
        ],
        "provider_fail_fast_reason_codes": public_collection_report[
            "provider_fail_fast_reason_codes"
        ],
        "max_consecutive_orderbook_failure_rounds": public_collection_report[
            "max_consecutive_orderbook_failure_rounds"
        ],
        "consecutive_orderbook_failure_count_at_stop": public_collection_report[
            "consecutive_orderbook_failure_count_at_stop"
        ],
        "scoring_rule_id": "fresh_provider_simplified_score",
        "canonical_frozen_o_scorer_used": False,
        "uses_paper_intent_logs_as_fresh_public_data": False,
        "paper_candidate_unlock_verified": True,
        "paper_candidate_unlock_manifest_sha256": public_collection_report[
            "frozen_o_action_rank_reference_sha256"
        ],
        "paper_candidate_unlock_blocking_reason_codes": [],
        "paper_fresh_loop_blocking_reason_codes": sorted(set(blockers)),
        "public_data_cycle_input_count": len(trace_rows),
        "candidate_decision_count": len(trace_rows),
        "guard_allowed_decision_count": len(intents),
        "guard_blocked_decision_count": max(0, len(trace_rows) - len(intents)),
        "execution_layer_v2_paper_remap_enabled": True,
        "execution_layer_v2_paper_remap_candidate_count": 0,
        "execution_layer_v2_paper_remap_applied_count": sum(
            1 for intent in intents if intent.get("hts_time_window_remap_applied") is True
        ),
        "paper_fresh_order_intent_count": len(intents),
        "paper_fresh_fill_count": len(fills),
        "paper_fresh_ledger_entry_count": len(ledger_rows),
        "runtime_field_missing_count": 0,
        "provenance_violation_count": 0,
        "block_reason_distribution": dict(sorted(block_counter.items())),
        "action_distribution": dict(
            sorted(Counter(str(row.get("selected_action")) for row in trace_rows).items())
        ),
        "family_distribution": dict(
            sorted(Counter(str(row.get("selected_action_family")) for row in trace_rows).items())
        ),
        "side_distribution": dict(
            sorted(Counter(str(row.get("selected_side")) for row in trace_rows).items())
        ),
        "v8_paper_internal_handoff_allowed": True,
        "v8_execution_handoff_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_loop_run_report_id")


def _aggregate_incremental_runtime_safety_report(
    *,
    run_id: str,
    cycle_results: list[Any],
) -> dict[str, Any]:
    safety_passed = all(
        result.runtime_safety_report.get("paper_fresh_runtime_safety_passed") is True
        for result in cycle_results
    )
    report = {
        "schema_version": "bigan-v8-polymarket-o-v8-paper-fresh-runtime-safety-v1",
        "report_type": "o_v8_paper_fresh_runtime_safety",
        "phase": "polymarket_policy_training",
        "run_id": run_id,
        "paper_fresh_runtime_safety_passed": safety_passed,
        "paper_fresh_runtime_safety_blocking_reason_codes": sorted(
            {
                reason
                for result in cycle_results
                for reason in result.runtime_safety_report.get(
                    "paper_fresh_runtime_safety_blocking_reason_codes", []
                )
            }
        ),
        "paper_fresh_loop_enabled": True,
        "paper_fresh_order_intent_count": sum(
            int(result.runtime_safety_report.get("paper_fresh_order_intent_count") or 0)
            for result in cycle_results
        ),
        "paper_fresh_fill_count": sum(
            int(result.runtime_safety_report.get("paper_fresh_fill_count") or 0)
            for result in cycle_results
        ),
        "paper_fresh_ledger_entry_count": sum(
            int(result.runtime_safety_report.get("paper_fresh_ledger_entry_count") or 0)
            for result in cycle_results
        ),
        "v8_paper_internal_handoff_allowed": True,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_runtime_safety_report_id")


def _aggregate_incremental_signal_trace_report(
    *,
    run_id: str,
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "schema_version": "bigan-v8-polymarket-o-v8-paper-fresh-signal-trace-v1",
        "report_type": "o_v8_paper_fresh_signal_trace",
        "phase": "polymarket_policy_training",
        "run_id": run_id,
        "public_data_source": O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER,
        "trace_row_count": len(trace_rows),
        "total_provider_decision_count": len(trace_rows),
        "canonical_selected_decision_count": len(trace_rows),
        "trace_rows": trace_rows,
        "trace_rows_sorted_by_decision_ts": True,
        "paper_intent_count": 0,
        "fill_count": 0,
        "uses_validation_outcomes_for_tuning": False,
        "thresholds_tuned": False,
        "uses_realized_pnl_or_labels_for_analysis": False,
        "uses_oracle_actions_for_analysis": False,
        "forbidden_outcome_fields_used": [],
        "mutates_o_model_predicted_score": False,
        "mutates_source_ranking_scores": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "o_v8_paper_fresh_signal_trace_report_id")


def _aggregate_incremental_fresh_loop_manifest(
    *,
    run_id: str,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    run_report: dict[str, Any],
    remap_report: dict[str, Any],
    runtime_safety_report: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": "bigan-v8-polymarket-o-v8-paper-fresh-loop-manifest-v1",
        "report_type": "o_v8_paper_fresh_loop_manifest",
        "phase": "polymarket_policy_training",
        "run_id": run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "fresh_loop_run_report_id": run_report["o_v8_paper_fresh_loop_run_report_id"],
        "fresh_runtime_safety_report_id": runtime_safety_report[
            "o_v8_paper_fresh_runtime_safety_report_id"
        ],
        "execution_layer_v2_paper_remap_report_id": remap_report[
            "execution_layer_v2_paper_remap_report_id"
        ],
        "paper_fresh_loop_public_data_source": run_report[
            "paper_fresh_loop_public_data_source"
        ],
        "paper_fresh_loop_cycle_count": run_report["paper_fresh_loop_cycle_count"],
        "paper_fresh_provider_collection_failed": run_report[
            "paper_fresh_provider_collection_failed"
        ],
        "provider_fail_fast_stop_triggered": run_report[
            "provider_fail_fast_stop_triggered"
        ],
        "provider_fail_fast_reason_codes": run_report[
            "provider_fail_fast_reason_codes"
        ],
        "max_consecutive_orderbook_failure_rounds": run_report[
            "max_consecutive_orderbook_failure_rounds"
        ],
        "consecutive_orderbook_failure_count_at_stop": run_report[
            "consecutive_orderbook_failure_count_at_stop"
        ],
        "paper_fresh_order_intent_count": run_report[
            "paper_fresh_order_intent_count"
        ],
        "paper_fresh_fill_count": run_report["paper_fresh_fill_count"],
        "paper_fresh_runtime_safety_passed": runtime_safety_report[
            "paper_fresh_runtime_safety_passed"
        ],
        **compact_safety_fields(),
    }
    return _with_report_id(manifest, "o_v8_paper_fresh_loop_manifest_id")


def _remap_execution_report(
    *,
    run_id: str,
    fresh_remap_report: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    forced_coverage: dict[str, Any],
) -> dict[str, Any]:
    remap_intents = [
        intent for intent in intents if intent.get("hts_time_window_remap_applied") is True
    ]
    forced_intents = [
        intent for intent in intents if intent.get("coverage_forced_paper_bet") is True
    ]
    report = {
        "schema_version": ONE_HOUR_REMAP_EXECUTION_REPORT_SCHEMA_VERSION,
        "report_type": "one_hour_remap_execution",
        "run_id": run_id,
        "hts_time_window_blocked_count": fresh_remap_report[
            "hts_time_window_blocked_count"
        ],
        "same_side_sbc_remap_guard_passed_count": fresh_remap_report[
            "remap_guard_passed_count"
        ],
        "remap_paper_bet_count": len(remap_intents),
        "normal_policy_bet_count": len(intents) - len(remap_intents) - len(forced_intents),
        "paper_fill_count": len(fills),
        "forced_coverage_bet_count": len(forced_intents),
        "forced_coverage_round_ids": forced_coverage["forced_coverage_round_ids"],
        "forced_coverage_guard_passed_count": forced_coverage[
            "forced_coverage_guard_passed_count"
        ],
        "forced_coverage_guard_blocked_count": forced_coverage[
            "forced_coverage_guard_blocked_count"
        ],
        "forced_coverage_blocking_reason_distribution": forced_coverage[
            "forced_coverage_blocking_reason_distribution"
        ],
        "forced_coverage_blocker_category_distribution": forced_coverage[
            "forced_coverage_blocker_category_distribution"
        ],
        "forced_coverage_candidate_selection_rule_id": forced_coverage[
            "forced_coverage_candidate_selection_rule_id"
        ],
        "forced_coverage_candidate_attempt_count": forced_coverage[
            "forced_coverage_candidate_attempt_count"
        ],
        "forced_coverage_attempt_rows": forced_coverage[
            "forced_coverage_attempt_rows"
        ],
        "remap_reason_distribution": fresh_remap_report[
            "remap_reason_distribution"
        ],
        "remap_execution_rows": fresh_remap_report.get("remap_rows", []),
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
    }
    return _with_report_id(report, "remap_execution_report_id")


def _write_per_round_bet_artifacts(
    *,
    goal_dir: Path,
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]] | None = None,
    raw_market_rows: list[dict[str, Any]] | None = None,
    raw_orderbook_rows: list[dict[str, Any]] | None = None,
    raw_trade_rows: list[dict[str, Any]] | None = None,
    raw_btc_candle_rows: list[dict[str, Any]] | None = None,
) -> None:
    trace_rows = trace_rows or []
    raw_market_rows = raw_market_rows or []
    raw_orderbook_rows = raw_orderbook_rows or []
    raw_trade_rows = raw_trade_rows or []
    raw_btc_candle_rows = raw_btc_candle_rows or []
    rounds_dir = goal_dir / "round_artifacts"
    market_ids = sorted(
        {
            _artifact_market_id(row)
            for row in [
                *intents,
                *fills,
                *ledger_rows,
                *trace_rows,
                *raw_market_rows,
                *raw_orderbook_rows,
                *raw_trade_rows,
            ]
            if _artifact_market_id(row)
        }
    )
    for market_id in market_ids:
        market_dir = rounds_dir / _safe_path_component(market_id)
        _write_jsonl(
            market_dir / "paper_bets.jsonl",
            [row for row in intents if str(row.get("market_id")) == market_id],
        )
        _write_jsonl(
            market_dir / "paper_fills.jsonl",
            [row for row in fills if str(row.get("market_id")) == market_id],
        )
        _write_jsonl(
            market_dir / "paper_ledger.jsonl",
            [row for row in ledger_rows if str(row.get("market_id")) == market_id],
        )
        market_trace_rows = _rows_for_market(trace_rows, market_id)
        market_raw_rows = _rows_for_market(raw_market_rows, market_id)
        _write_jsonl(market_dir / "signal_trace.jsonl", market_trace_rows)
        _write_jsonl(market_dir / "raw_polymarket_markets.jsonl", market_raw_rows)
        _write_jsonl(
            market_dir / "raw_polymarket_orderbooks.jsonl",
            _rows_for_market(raw_orderbook_rows, market_id),
        )
        _write_jsonl(
            market_dir / "raw_polymarket_trades.jsonl",
            _rows_for_market(raw_trade_rows, market_id),
        )
        _write_jsonl(
            market_dir / "raw_btc_feature_candles.jsonl",
            _btc_candles_for_market(
                raw_btc_candle_rows,
                market_rows=market_raw_rows,
                trace_rows=market_trace_rows,
            ),
        )


def _artifact_market_id(row: dict[str, Any]) -> str:
    return str(row.get("market_id") or row.get("condition_id") or "")


def _rows_for_market(
    rows: list[dict[str, Any]], market_id: str
) -> list[dict[str, Any]]:
    return [row for row in rows if _artifact_market_id(row) == market_id]


def _btc_candles_for_market(
    rows: list[dict[str, Any]],
    *,
    market_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    starts = [
        _float(row.get("market_start_ts"))
        for row in [*market_rows, *trace_rows]
        if row.get("market_start_ts") is not None
    ]
    decisions = [
        _float(row.get("decision_ts"))
        for row in trace_rows
        if row.get("decision_ts") is not None
    ]
    if not starts or not decisions:
        return []
    lower_bound = min(starts) - 120_000.0
    upper_bound = max(decisions)
    selected: list[dict[str, Any]] = []
    for row in rows:
        available_at_ts = row.get("available_at_ts")
        source_ts = (
            available_at_ts
            if available_at_ts is not None
            else row.get("close_time", row.get("ts"))
        )
        if source_ts is None:
            continue
        if lower_bound <= _float(source_ts) <= upper_bound:
            selected.append(row)
    return selected


def _settlement_pnl_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled_rows = [row for row in rows if row.get("settlement_status") == "settled"]
    unresolved_rows = [
        row for row in rows if row.get("settlement_status") != "settled"
    ]
    settled_pnl = sum(_float(row.get("settlement_pnl")) for row in rows)
    unresolved_pnl = sum(_float(row.get("unresolved_pnl")) for row in rows)
    return {
        "settlement_row_count": len(rows),
        "settled_fill_count": len(settled_rows),
        "unresolved_fill_count": len(unresolved_rows),
        "winning_fill_count": sum(
            1 for row in settled_rows if _float(row.get("settlement_pnl")) > 0.0
        ),
        "losing_fill_count": sum(
            1 for row in settled_rows if _float(row.get("settlement_pnl")) < 0.0
        ),
        "flat_fill_count": sum(
            1 for row in settled_rows if _float(row.get("settlement_pnl")) == 0.0
        ),
        "settled_pnl": settled_pnl,
        "unresolved_pnl": unresolved_pnl,
        "pnl_by_side": _pnl_by_field(rows, "execution_guarded_side"),
        "pnl_by_action": _pnl_by_field(rows, "execution_guarded_action"),
        "resolved_outcome_distribution": _counter_from_field(
            settled_rows, "resolved_outcome"
        ),
    }


def _pnl_by_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        field_value = str(row.get(field_name) or "UNKNOWN")
        totals[field_value] = totals.get(field_value, 0.0) + _float(
            row.get("settlement_pnl")
        )
    return dict(sorted(totals.items()))


def _counter_from_field(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get(field_name) or "UNKNOWN")] += 1
    return dict(sorted(counter.items()))


def _write_per_round_artifacts(
    *,
    goal_dir: Path,
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    settlement_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    raw_market_rows: list[dict[str, Any]],
    raw_orderbook_rows: list[dict[str, Any]],
    raw_trade_rows: list[dict[str, Any]],
    raw_btc_candle_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _write_per_round_bet_artifacts(
        goal_dir=goal_dir,
        intents=intents,
        fills=fills,
        ledger_rows=ledger_rows,
        trace_rows=trace_rows,
        raw_market_rows=raw_market_rows,
        raw_orderbook_rows=raw_orderbook_rows,
        raw_trade_rows=raw_trade_rows,
        raw_btc_candle_rows=raw_btc_candle_rows,
    )
    rounds_dir = goal_dir / "round_artifacts"
    market_ids = sorted(
        {
            _artifact_market_id(row)
            for row in [
                *intents,
                *fills,
                *ledger_rows,
                *settlement_rows,
                *trace_rows,
                *raw_market_rows,
                *raw_orderbook_rows,
                *raw_trade_rows,
            ]
            if _artifact_market_id(row)
        }
    )
    round_rows: list[dict[str, Any]] = []
    for market_id in market_ids:
        market_dir = rounds_dir / _safe_path_component(market_id)
        settlement_for_market = [
            row for row in settlement_rows if str(row.get("market_id")) == market_id
        ]
        outcome_path = market_dir / "round_outcome.json"
        pnl_summary = _settlement_pnl_summary(settlement_for_market)
        if settlement_for_market:
            outcome_payload = {
                "market_id": market_id,
                "settlement_rows": settlement_for_market,
                "settlement_status": (
                    "settled"
                    if all(
                        row.get("settlement_status") == "settled"
                        for row in settlement_for_market
                    )
                    else "unresolved"
                ),
                **pnl_summary,
                "paper_only": True,
                "capital_at_risk": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
                "v8_execution_handoff_allowed": False,
            }
            _write_json(outcome_path, outcome_payload)
        paths = {
            "paper_bets": market_dir / "paper_bets.jsonl",
            "paper_fills": market_dir / "paper_fills.jsonl",
            "paper_ledger": market_dir / "paper_ledger.jsonl",
            "round_outcome": outcome_path,
            "signal_trace": market_dir / "signal_trace.jsonl",
            "raw_polymarket_markets": market_dir
            / "raw_polymarket_markets.jsonl",
            "raw_polymarket_orderbooks": market_dir
            / "raw_polymarket_orderbooks.jsonl",
            "raw_polymarket_trades": market_dir / "raw_polymarket_trades.jsonl",
            "raw_btc_feature_candles": market_dir
            / "raw_btc_feature_candles.jsonl",
        }
        row = {
            "market_id": market_id,
            "round_artifact_dir": str(market_dir),
            "paper_bet_artifact_exists": paths["paper_bets"].exists(),
            "paper_fill_artifact_exists": paths["paper_fills"].exists(),
            "paper_ledger_artifact_exists": paths["paper_ledger"].exists(),
            "round_outcome_artifact_exists": paths["round_outcome"].exists(),
            "paper_bet_count": sum(
                1 for intent in intents if str(intent.get("market_id")) == market_id
            ),
            "paper_fill_count": sum(
                1 for fill in fills if str(fill.get("market_id")) == market_id
            ),
            "signal_trace_row_count": len(_rows_for_market(trace_rows, market_id)),
            "raw_market_row_count": len(_rows_for_market(raw_market_rows, market_id)),
            "raw_orderbook_row_count": len(
                _rows_for_market(raw_orderbook_rows, market_id)
            ),
            "raw_trade_row_count": len(_rows_for_market(raw_trade_rows, market_id)),
            "raw_btc_feature_candle_row_count": len(
                _read_jsonl(paths["raw_btc_feature_candles"])
            ),
            **pnl_summary,
            "artifact_paths": {
                name: str(path)
                for name, path in paths.items()
                if path.exists()
            },
            "artifact_hashes": {
                name: _sha256_file(path)
                for name, path in paths.items()
                if path.exists()
            },
        }
        round_rows.append(row)
    manifest = {
        "schema_version": (
            "bigan-v8-polymarket-execution-layer-v2-per-round-artifacts-v1"
        ),
        "report_type": "one_hour_remap_per_round_artifacts",
        "per_round_async_artifact_flush_enabled": True,
        "per_round_bet_artifact_count": sum(
            1 for row in round_rows if row["paper_bet_artifact_exists"]
        ),
        "per_round_outcome_artifact_count": sum(
            1 for row in round_rows if row["round_outcome_artifact_exists"]
        ),
        "per_round_raw_orderbook_artifact_count": sum(
            1
            for row in round_rows
            if row["artifact_paths"].get("raw_polymarket_orderbooks")
        ),
        "per_round_signal_trace_artifact_count": sum(
            1 for row in round_rows if row["artifact_paths"].get("signal_trace")
        ),
        "settled_pnl": sum(_float(row["settled_pnl"]) for row in round_rows),
        "unresolved_pnl": sum(_float(row["unresolved_pnl"]) for row in round_rows),
        "settled_fill_count": sum(int(row["settled_fill_count"]) for row in round_rows),
        "unresolved_fill_count": sum(
            int(row["unresolved_fill_count"]) for row in round_rows
        ),
        "winning_fill_count": sum(int(row["winning_fill_count"]) for row in round_rows),
        "losing_fill_count": sum(int(row["losing_fill_count"]) for row in round_rows),
        "round_artifact_rows": round_rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    manifest = _with_report_id(manifest, "per_round_artifact_manifest_id")
    manifest_path = rounds_dir / "round_artifacts_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def _forced_coverage_attempt_report(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    fresh_loop_config: PolymarketOV8PaperFreshLoopConfig,
    trace_rows: list[dict[str, Any]],
    base_intents: list[dict[str, Any]],
) -> dict[str, Any]:
    bet_rounds = {str(intent.get("market_id")) for intent in base_intents}
    trace_by_market: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        market_id = str(row.get("market_id") or "")
        if market_id:
            trace_by_market.setdefault(market_id, []).append(dict(row))
    missing_market_ids = sorted(set(trace_by_market) - bet_rounds)
    runtime_state = _runtime_state_from_intents(base_intents)
    guard_config = _v8_execution_guard_config()
    forced_intents: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for market_id in missing_market_ids:
        candidate_attempt = _forced_coverage_candidate_attempt(
            run_id=config.run_id,
            market_id=market_id,
            trace_rows=trace_by_market[market_id],
            guard_config=guard_config,
            runtime_state=runtime_state,
        )
        source_trace = candidate_attempt["source_trace"]
        coverage_public_row = candidate_attempt["coverage_public_row"]
        guard_row = candidate_attempt["guard_row"]
        if coverage_public_row is None or guard_row is None:
            attempt_rows.append(
                _forced_coverage_attempt_row(
                    market_id=market_id,
                    source_trace=source_trace,
                    guard_row=None,
                    intent=None,
                    reason_codes=["forced_coverage_no_trade_candidate_available"],
                    candidate_attempt=candidate_attempt,
                )
            )
            continue
        guard_row["cycle_id"] = f"{config.run_id}-forced-coverage"
        guard_row["public_data_source"] = coverage_public_row["public_data_source"]
        guard_row["coverage_forced_attempted"] = True
        guard_row["coverage_forced_round_id"] = market_id
        guard_row["coverage_forced_reason_codes"] = [
            "complete_round_missing_paper_bet",
            "forced_coverage_full_execution_guard_checked",
            "forced_coverage_guard_compatible_candidate_search",
        ]
        if guard_row.get("order_allowed") is True:
            guard_row["coverage_forced_paper_bet"] = True
            guard_row["simulated_order_id"] = (
                f"{fresh_loop_config.run_id}-forced-coverage-sim-"
                f"{len(base_intents) + len(forced_intents) + 1:06d}"
            )
            guard_row["proposed_order_size"] = min(
                float(guard_row.get("proposed_order_size") or 0.0),
                float(guard_config["base_order_size"]),
            )
            guard_row["sizing_reason_codes"] = sorted(
                {
                    *list(guard_row.get("sizing_reason_codes") or []),
                    "forced_coverage_smallest_size_applied",
                }
            )
            _apply_guard_row_to_runtime_state(runtime_state, guard_row)
            guard_row["post_decision_exposure_state"] = dict(runtime_state)
            intent = _fresh_order_intent_from_guard_row(
                config=fresh_loop_config,
                cycle_id=f"{config.run_id}-forced-coverage",
                guard_row=guard_row,
                intent_index=len(base_intents) + len(forced_intents) + 1,
            )
            intent["coverage_forced_paper_bet"] = True
            intent["coverage_forced_round_id"] = market_id
            intent["coverage_forced_reason_codes"] = list(
                guard_row["coverage_forced_reason_codes"]
            )
            intent["order_origin"] = "forced_coverage_full_guard_paper_only"
            intent["paper_fresh_order_intent_hash"] = canonical_json_sha256(intent)
            forced_intents.append(intent)
        else:
            guard_row["coverage_forced_paper_bet"] = False
            runtime_state["blocked_simulated_order_count"] = int(
                runtime_state["blocked_simulated_order_count"]
            ) + 1
            guard_row["simulated_order_id"] = None
            guard_row["post_decision_exposure_state"] = dict(runtime_state)
            intent = None
        attempt_rows.append(
            _forced_coverage_attempt_row(
                market_id=market_id,
                source_trace=source_trace,
                guard_row=guard_row,
                intent=intent,
                reason_codes=_forced_coverage_guard_reason_codes(guard_row),
                candidate_attempt=candidate_attempt,
            )
        )
    block_reasons = Counter(
        reason
        for row in attempt_rows
        if row["forced_coverage_guard_passed"] is not True
        for reason in row["forced_coverage_blocking_reason_codes"]
    )
    block_categories = Counter(
        category
        for row in attempt_rows
        if row["forced_coverage_guard_passed"] is not True
        for category in row["forced_coverage_blocker_categories"]
    )
    return {
        "forced_coverage_round_ids": sorted(
            {
                str(intent.get("market_id"))
                for intent in forced_intents
                if intent.get("market_id")
            }
        ),
        "forced_coverage_intents": forced_intents,
        "forced_coverage_attempt_rows": attempt_rows,
        "forced_coverage_guard_passed_count": len(forced_intents),
        "forced_coverage_guard_blocked_count": len(attempt_rows) - len(forced_intents),
        "forced_coverage_blocking_reason_distribution": dict(sorted(block_reasons.items())),
        "forced_coverage_blocker_category_distribution": dict(
            sorted(block_categories.items())
        ),
        "forced_coverage_candidate_selection_rule_id": (
            "best_non_no_trade_ranked_candidate_full_guard_search"
        ),
        "forced_coverage_candidate_attempt_count": sum(
            int(row["forced_coverage_candidate_attempt_count"])
            for row in attempt_rows
        ),
    }


def _runtime_state_from_intents(intents: list[dict[str, Any]]) -> dict[str, Any]:
    state = _initial_fresh_runtime_state()
    for intent in intents:
        market_id = str(intent.get("market_id") or "")
        side = str(intent.get("execution_guarded_side") or "NONE")
        size = _float(intent.get("paper_fresh_order_size"))
        if not market_id or size <= 0.0:
            continue
        state["current_market_exposure_by_market_id"][market_id] = (
            _float(state["current_market_exposure_by_market_id"].get(market_id)) + size
        )
        state["current_side_exposure_by_side"][side] = (
            _float(state["current_side_exposure_by_side"].get(side)) + size
        )
        state["current_total_exposure"] = _float(state["current_total_exposure"]) + size
        position = {
            "market_id": market_id,
            "side": side,
            "action": intent.get("execution_guarded_action"),
            "notional": size,
            "simulated_order_id": intent.get("simulated_order_id"),
        }
        state["open_position_by_market_id"][market_id] = position
        state["open_position_by_market_side"][f"{market_id}|{side}"] = position
        state["executed_simulated_order_count"] = int(
            state["executed_simulated_order_count"]
        ) + 1
    return state


def _forced_coverage_candidate_attempt(
    *,
    run_id: str,
    market_id: str,
    trace_rows: list[dict[str, Any]],
    guard_config: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    source_traces = _forced_coverage_source_trace_candidates(trace_rows)
    first_source_trace = source_traces[0] if source_traces else {"market_id": market_id}
    candidate_attempt_rows: list[dict[str, Any]] = []
    first_candidate: dict[str, Any] | None = None
    for source_trace in source_traces:
        for candidate in _forced_coverage_non_no_trade_candidates(source_trace):
            coverage_public_row = _forced_coverage_public_row(
                run_id=run_id,
                source_trace=source_trace,
                candidate=candidate,
            )
            if coverage_public_row is None:
                continue
            guard_input = _guard_input_from_public_row(
                public_row=coverage_public_row,
                cycle_id=f"{run_id}-forced-coverage",
                row_index=len(candidate_attempt_rows),
            )
            guard_row = _v8_execution_guard_decision(
                guard_input,
                guard_config=guard_config,
                runtime_state=runtime_state,
                runtime_mode="simulated_runtime_state",
            )
            candidate_attempt_rows.append(
                _forced_coverage_candidate_attempt_row(
                    source_trace=source_trace,
                    candidate=candidate,
                    guard_row=guard_row,
                )
            )
            candidate_result = {
                "source_trace": source_trace,
                "coverage_public_row": coverage_public_row,
                "guard_row": guard_row,
                "candidate_attempt_rows": list(candidate_attempt_rows),
            }
            if first_candidate is None:
                first_candidate = candidate_result
            if guard_row.get("order_allowed") is True:
                return {
                    **candidate_result,
                    "candidate_attempt_count": len(candidate_attempt_rows),
                    "source_trace_count": len(source_traces),
                    "candidate_selection_rule_id": (
                        "best_non_no_trade_ranked_candidate_full_guard_search"
                    ),
                    "candidate_search_found_guard_passed": True,
                }
    selected = first_candidate or {
        "source_trace": first_source_trace,
        "coverage_public_row": None,
        "guard_row": None,
        "candidate_attempt_rows": candidate_attempt_rows,
    }
    return {
        **selected,
        "candidate_attempt_rows": candidate_attempt_rows,
        "candidate_attempt_count": len(candidate_attempt_rows),
        "source_trace_count": len(source_traces),
        "candidate_selection_rule_id": (
            "best_non_no_trade_ranked_candidate_full_guard_search"
        ),
        "candidate_search_found_guard_passed": False,
    }


def _forced_coverage_source_trace_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("order_allowed")),
            _float(
                row.get("canonical_corrected_score")
                or row.get("simplified_corrected_score")
            ),
            _float(row.get("time_to_close_seconds")),
            int(row.get("decision_ts") or 0),
        ),
        reverse=True,
    )


def _forced_coverage_source_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _forced_coverage_source_trace_candidates(rows)[0]


def _forced_coverage_non_no_trade_candidates(
    source_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _forced_coverage_full_ranking(source_trace)
        if str(row.get("selected_action") or "") != "NO_TRADE"
    ]


def _forced_coverage_public_row(
    *,
    run_id: str,
    source_trace: dict[str, Any],
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    ranking = _forced_coverage_full_ranking(source_trace)
    candidate = candidate or next(
        (row for row in ranking if str(row.get("selected_action") or "") != "NO_TRADE"),
        None,
    )
    if candidate is None:
        return None
    action = str(candidate["selected_action"])
    side = str(candidate.get("selected_side") or _side_from_action(action))
    family = str(candidate.get("selected_action_family") or _action_family(action))
    decision_ts = int(source_trace.get("decision_ts") or 0)
    return {
        "decision_group_id": (
            f"{run_id}|forced-coverage|{source_trace.get('market_id')}|{decision_ts}"
        ),
        "market_id": source_trace.get("market_id"),
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "selected_action_family": family,
        "full_5_action_ranking": ranking,
        "corrected_model_score": _float(candidate.get("corrected_model_score")),
        "raw_model_score": _float(candidate.get("raw_model_score")),
        "high_score_flag": bool(source_trace.get("high_score_flag", True)),
        "p_up": _float(source_trace.get("p_up")),
        "p_down": _float(source_trace.get("p_down")),
        "p_up_action_disagreement": _coverage_p_up_action_disagreement(
            action=action,
            p_up=_float(source_trace.get("p_up")),
        ),
        "microstructure_snapshot": dict(candidate.get("microstructure_snapshot") or {})
        or {
            "entry_ask": source_trace.get("entry_ask"),
            "executable_exit_bid_proxy": source_trace.get("executable_exit_bid_proxy"),
            "spread_bps": source_trace.get("spread_bps"),
            "book_staleness_ms": source_trace.get("book_staleness_ms"),
            "queue_fill_proxy": source_trace.get("queue_fill_proxy"),
            "time_to_close_seconds": source_trace.get("time_to_close_seconds"),
        },
        "reference_price_feature_provenance": {
            "provenance_valid": True,
            "decision_ts": decision_ts,
            "max_input_ts": source_trace.get("decision_time_feature_max_input_ts")
            or decision_ts,
            "source_fields_used": ["one_hour_signal_trace_decision_time_fields"],
        },
        "decision_time_feature_max_input_ts": source_trace.get(
            "decision_time_feature_max_input_ts", decision_ts
        ),
        "btc_momentum": source_trace.get("btc_momentum"),
        "btc_momentum_provenance": dict(
            source_trace.get("btc_momentum_provenance") or {}
        ),
        "reference_price_to_beat_distance_at_decision": source_trace.get(
            "reference_price_to_beat_distance_at_decision"
        ),
        "reference_price_to_beat_distance_provenance": dict(
            source_trace.get("reference_price_to_beat_distance_provenance") or {}
        ),
        "time_since_market_start_seconds": source_trace.get(
            "time_since_market_start_seconds",
            source_trace.get("elapsed_since_market_start_seconds"),
        ),
        "time_since_market_start_provenance": dict(
            source_trace.get("time_since_market_start_provenance") or {}
        ),
        "action_score_margin": source_trace.get(
            "action_score_margin",
            source_trace.get("score_margin"),
        ),
        "action_score_margin_provenance": dict(
            source_trace.get("action_score_margin_provenance") or {}
        ),
        "side_specific_action_score_margin": source_trace.get(
            "side_specific_action_score_margin"
        ),
        "side_specific_action_score_margin_provenance": dict(
            source_trace.get("side_specific_action_score_margin_provenance") or {}
        ),
        "decision_time_regime_feature_provenance": dict(
            source_trace.get("decision_time_regime_feature_provenance") or {}
        ),
        "decision_time_regime_feature_max_input_ts": source_trace.get(
            "decision_time_regime_feature_max_input_ts",
            source_trace.get("decision_time_feature_max_input_ts", decision_ts),
        ),
        "public_data_source": source_trace.get("public_data_source"),
        "coverage_forced_candidate_source": "one_hour_signal_trace_ranking_summary",
    }


def _forced_coverage_full_ranking(source_trace: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = (
        source_trace.get("canonical_full_5_action_ranking_summary")
        if source_trace.get("canonical_scorer_used") is True
        else source_trace.get("simplified_full_5_action_ranking_summary")
    ) or []
    rows = []
    for row in raw_rows:
        action = str(row.get("action") or row.get("selected_action") or "")
        if not action:
            continue
        score = row.get("canonical_corrected_score", row.get("corrected_score"))
        raw_score = row.get("canonical_raw_score", row.get("raw_score"))
        rows.append(
            {
                "selected_action": action,
                "selected_side": row.get("side") or _side_from_action(action),
                "selected_action_family": row.get("family") or _action_family(action),
                "corrected_model_score": _float(score),
                "raw_model_score": _float(raw_score),
                "microstructure_snapshot": dict(
                    row.get("microstructure_snapshot") or {}
                ),
            }
        )
    if not rows:
        action = str(
            source_trace.get("canonical_selected_action")
            or source_trace.get("simplified_selected_action")
            or ""
        )
        if action:
            rows.append(
                {
                    "selected_action": action,
                    "selected_side": source_trace.get("canonical_selected_side")
                    or source_trace.get("simplified_selected_side")
                    or _side_from_action(action),
                    "selected_action_family": source_trace.get(
                        "canonical_selected_family"
                    )
                    or source_trace.get("simplified_selected_family")
                    or _action_family(action),
                    "corrected_model_score": _float(
                        source_trace.get("canonical_corrected_score")
                        or source_trace.get("simplified_corrected_score")
                    ),
                    "raw_model_score": _float(source_trace.get("canonical_raw_score")),
                }
            )
    return sorted(
        rows,
        key=lambda row: _float(row.get("corrected_model_score")),
        reverse=True,
    )


def _coverage_p_up_action_disagreement(*, action: str, p_up: float) -> bool:
    side = _side_from_action(action)
    if side == "UP":
        return p_up < 0.50
    if side == "DOWN":
        return p_up > 0.50
    return False


def _forced_coverage_candidate_attempt_row(
    *,
    source_trace: dict[str, Any],
    candidate: dict[str, Any],
    guard_row: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = _forced_coverage_guard_reason_codes(guard_row)
    micro = dict(guard_row.get("microstructure_snapshot") or {})
    row = {
        "market_id": guard_row.get("market_id"),
        "decision_ts": guard_row.get("decision_ts"),
        "candidate_action": candidate.get("selected_action"),
        "candidate_side": candidate.get("selected_side"),
        "candidate_family": candidate.get("selected_action_family"),
        "candidate_rank_score": _float(candidate.get("corrected_model_score")),
        "candidate_raw_model_score": _float(candidate.get("raw_model_score")),
        "order_allowed": bool(guard_row.get("order_allowed")),
        "execution_guarded_action": guard_row.get("execution_guarded_action"),
        "execution_guarded_side": guard_row.get("execution_guarded_side"),
        "blocking_reason_codes": [] if guard_row.get("order_allowed") else reason_codes,
        "blocker_categories": []
        if guard_row.get("order_allowed")
        else _forced_coverage_blocker_categories(reason_codes),
        "time_to_close_seconds": micro.get("time_to_close_seconds"),
        "required_min_time_to_close_seconds": (
            guard_row.get("runtime_limits", {}).get("min_hts_time_to_close_seconds")
            if candidate.get("selected_action_family") == "HOLD_TO_SETTLEMENT"
            else guard_row.get("runtime_limits", {}).get("min_time_to_close_seconds")
        ),
        "spread_bps": micro.get("spread_bps"),
        "book_staleness_ms": micro.get("book_staleness_ms"),
        "queue_fill_proxy": micro.get("queue_fill_proxy"),
        "p_up": guard_row.get("p_up"),
        "p_up_action_disagreement": guard_row.get("p_up_action_disagreement"),
        "exposure_reason_codes": list(guard_row.get("exposure_reason_codes") or []),
        "missing_runtime_field_codes": list(
            guard_row.get("missing_runtime_field_codes") or []
        ),
        "missing_trace_metadata_codes": _forced_coverage_missing_trace_metadata(
            source_trace
        ),
        "uses_settlement_pnl_or_outcome_labels_in_decision_logic": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    row["forced_coverage_candidate_attempt_hash"] = canonical_json_sha256(row)
    return row


def _forced_coverage_guard_reason_codes(guard_row: dict[str, Any]) -> list[str]:
    reason_codes = {
        *list(guard_row.get("execution_blocking_reason_codes") or []),
        *[
            reason
            for reason in list(guard_row.get("execution_guard_reason_codes") or [])
            if reason.startswith("execution_hts_")
            or reason == "execution_score_margin_too_close"
        ],
        *[
            reason
            for reason in list(guard_row.get("exposure_reason_codes") or [])
            if reason.endswith("_blocked")
            or reason.endswith("_exceeded")
            or reason.startswith("execution_duplicate")
        ],
        *list(guard_row.get("missing_runtime_field_codes") or []),
    }
    return sorted(reason_codes or {"forced_coverage_guard_blocked"})


def _forced_coverage_blocker_categories(reason_codes: list[str]) -> list[str]:
    categories: set[str] = set()
    for reason in reason_codes:
        if "time_to_close" in reason or "time_window" in reason:
            categories.add("time_to_close")
        if "spread" in reason:
            categories.add("spread")
        if "p_up" in reason or "side_disagreement" in reason:
            categories.add("p_up_disagreement")
        if "exposure" in reason or "duplicate" in reason:
            categories.add("exposure")
        if "runtime_field" in reason:
            categories.add("missing_runtime_field")
        if "no_trade_candidate" in reason or "missing" in reason:
            categories.add("missing_candidate_or_metadata")
    return sorted(categories or {"guard_blocked_other"})


def _forced_coverage_missing_trace_metadata(source_trace: dict[str, Any]) -> list[str]:
    required_fields = [
        "decision_ts",
        "p_up",
        "spread_bps",
        "book_staleness_ms",
        "queue_fill_proxy",
        "time_to_close_seconds",
    ]
    return [
        f"missing_trace_field:{field}"
        for field in required_fields
        if source_trace.get(field) is None
    ]


def _forced_coverage_attempt_row(
    *,
    market_id: str,
    source_trace: dict[str, Any],
    guard_row: dict[str, Any] | None,
    intent: dict[str, Any] | None,
    reason_codes: list[str],
    candidate_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    passed = intent is not None
    blocking = sorted(set(reason_codes or ["forced_coverage_guard_blocked"]))
    candidate_attempt = candidate_attempt or {}
    micro = dict(guard_row.get("microstructure_snapshot") or {}) if guard_row else {}
    blocker_categories = _forced_coverage_blocker_categories(blocking)
    row = {
        "market_id": market_id,
        "decision_ts": source_trace.get("decision_ts"),
        "coverage_forced_attempted": True,
        "coverage_forced_paper_bet": passed,
        "forced_coverage_guard_passed": passed,
        "forced_coverage_full_guard_order_allowed": bool(
            guard_row.get("order_allowed") if guard_row is not None else False
        ),
        "forced_coverage_blocking_reason_codes": [] if passed else blocking,
        "forced_coverage_blocker_categories": [] if passed else blocker_categories,
        "forced_coverage_selected_action": (
            guard_row.get("source_selected_action") if guard_row is not None else None
        ),
        "forced_coverage_selected_side": (
            guard_row.get("source_selected_side") if guard_row is not None else None
        ),
        "forced_coverage_selected_family": (
            guard_row.get("source_selected_family") if guard_row is not None else None
        ),
        "forced_coverage_execution_guarded_action": (
            guard_row.get("execution_guarded_action") if guard_row is not None else None
        ),
        "forced_coverage_execution_guarded_side": (
            guard_row.get("execution_guarded_side") if guard_row is not None else None
        ),
        "forced_coverage_time_to_close_seconds": micro.get("time_to_close_seconds"),
        "forced_coverage_required_min_time_to_close_seconds": (
            guard_row.get("runtime_limits", {}).get("min_hts_time_to_close_seconds")
            if guard_row is not None
            and guard_row.get("source_selected_family") == "HOLD_TO_SETTLEMENT"
            else guard_row.get("runtime_limits", {}).get("min_time_to_close_seconds")
            if guard_row is not None
            else None
        ),
        "forced_coverage_spread_bps": micro.get("spread_bps"),
        "forced_coverage_book_staleness_ms": micro.get("book_staleness_ms"),
        "forced_coverage_queue_fill_proxy": micro.get("queue_fill_proxy"),
        "forced_coverage_p_up_action_disagreement": (
            guard_row.get("p_up_action_disagreement") if guard_row is not None else None
        ),
        "forced_coverage_exposure_reason_codes": (
            list(guard_row.get("exposure_reason_codes") or [])
            if guard_row is not None
            else []
        ),
        "forced_coverage_missing_runtime_field_codes": (
            list(guard_row.get("missing_runtime_field_codes") or [])
            if guard_row is not None
            else []
        ),
        "forced_coverage_missing_trace_metadata_codes": (
            _forced_coverage_missing_trace_metadata(source_trace)
        ),
        "forced_coverage_candidate_selection_rule_id": candidate_attempt.get(
            "candidate_selection_rule_id",
            "best_non_no_trade_ranked_candidate_full_guard_search",
        ),
        "forced_coverage_candidate_attempt_count": int(
            candidate_attempt.get("candidate_attempt_count") or 0
        ),
        "forced_coverage_source_trace_count": int(
            candidate_attempt.get("source_trace_count") or 0
        ),
        "forced_coverage_candidate_search_found_guard_passed": bool(
            candidate_attempt.get("candidate_search_found_guard_passed")
        ),
        "forced_coverage_candidate_attempt_rows": list(
            candidate_attempt.get("candidate_attempt_rows") or []
        ),
        "paper_fresh_order_intent_id": (
            intent.get("paper_fresh_order_intent_id") if intent is not None else None
        ),
        "uses_settlement_pnl_or_outcome_labels_in_decision_logic": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    row["forced_coverage_attempt_row_hash"] = canonical_json_sha256(row)
    return row


def _settlement_resolution_report(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    fills: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    settlement_evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    explicit_settled_market_ids = {
        str(row.get("market_id"))
        for row in settlement_evaluation_rows
        if "settlement_pnl" in row
    }
    unresolved_fills = [
        fill
        for fill in fills
        if str(fill.get("market_id")) not in explicit_settled_market_ids
    ]
    market_rows = _settlement_market_rows_from_trace(trace_rows=trace_rows, fills=fills)
    market_rows = [
        row
        for row in market_rows
        if str(row.get("market_id")) in {str(fill.get("market_id")) for fill in unresolved_fills}
    ]
    reason_codes: list[str] = []
    evaluation_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    provider = config.public_provider
    if provider is None and config.public_data_cycles is None and unresolved_fills:
        provider = PolymarketPublicHTTPRealCorpusProvider()
    provider_class = provider.__class__.__name__ if provider is not None else None
    provider_read_only = bool(getattr(provider, "read_only", False)) if provider else False
    provider_safe = _settlement_provider_safe(provider) if provider is not None else False
    attempt_count = 0
    if not unresolved_fills:
        reason_codes.append("settlement_poll_skipped_no_unresolved_fills")
    elif not market_rows:
        reason_codes.append("settlement_poll_skipped_missing_market_metadata")
    elif provider is None:
        reason_codes.append("settlement_poll_skipped_no_public_provider")
    elif not provider_safe:
        reason_codes.append("settlement_poll_blocked_unsafe_public_provider")
    else:
        deadline = time.monotonic() + config.settlement_poll_max_wait_seconds
        while True:
            attempt_count += 1
            request_timeout = min(
                config.settlement_poll_interval_seconds,
                max(0.001, deadline - time.monotonic()),
            )
            try:
                resolution_rows = _provider_resolution_rows_with_timeout(
                    provider=provider,
                    market_rows=market_rows,
                    recorder_config=_settlement_recorder_config(config=config),
                    timeout_seconds=request_timeout,
                )
            except RealCorpusPublicProviderError as exc:
                reason_codes.extend(
                    ["settlement_resolution_provider_error", *list(exc.reason_codes)]
                )
                break
            except TimeoutError:
                reason_codes.extend(
                    [
                        "settlement_resolution_provider_timeout",
                        "settlement_resolution_http_timeout",
                    ]
                )
                break
            except Exception as exc:  # pragma: no cover - defensive provider boundary
                reason_codes.extend(
                    [
                        "settlement_resolution_provider_unexpected_error",
                        exc.__class__.__name__,
                    ]
                )
                break
            by_market = {
                str(row.get("market_id")): dict(row)
                for row in resolution_rows
                if _resolved_outcome_from_resolution(row) in {"UP", "DOWN"}
            }
            if by_market:
                evaluation_rows = _settlement_evaluation_rows_from_resolutions(
                    fills=unresolved_fills,
                    resolutions_by_market=by_market,
                )
            if len(evaluation_rows) >= len(unresolved_fills):
                reason_codes.append("settlement_resolution_all_fills_resolved")
                break
            if time.monotonic() >= deadline:
                reason_codes.append("settlement_resolution_max_wait_elapsed")
                break
            time.sleep(
                min(
                    config.settlement_poll_interval_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
    resolved_fill_ids = {
        str(row.get("paper_fresh_order_intent_id")) for row in evaluation_rows
    }
    unresolved_fill_ids = [
        str(fill.get("paper_fresh_order_intent_id"))
        for fill in unresolved_fills
        if str(fill.get("paper_fresh_order_intent_id")) not in resolved_fill_ids
    ]
    report = {
        "schema_version": ONE_HOUR_REMAP_SETTLEMENT_RESOLUTION_SCHEMA_VERSION,
        "report_type": "one_hour_remap_settlement_resolution",
        "run_id": config.run_id,
        "settlement_polling_enabled": True,
        "settlement_poll_max_wait_seconds": config.settlement_poll_max_wait_seconds,
        "settlement_poll_interval_seconds": config.settlement_poll_interval_seconds,
        "settlement_poll_attempt_count": attempt_count,
        "settlement_provider_class": provider_class,
        "settlement_provider_read_only": provider_read_only,
        "settlement_provider_safety_passed": provider_safe,
        "unresolved_fill_count_before_poll": len(unresolved_fills),
        "settlement_market_metadata_count": len(market_rows),
        "raw_resolution_row_count": len(resolution_rows),
        "settlement_evaluation_row_count": len(evaluation_rows),
        "resolved_fill_count": len(evaluation_rows),
        "unresolved_fill_count_after_poll": len(unresolved_fill_ids),
        "unresolved_paper_fresh_order_intent_ids": unresolved_fill_ids,
        "settlement_evaluation_rows": evaluation_rows,
        "resolution_rows": resolution_rows,
        "settlement_resolution_reason_codes": sorted(set(reason_codes)),
        "uses_settlement_pnl_for_decision_time_logic": False,
        "uses_oracle_actions_or_future_returns_for_decision_time_logic": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    return _with_report_id(report, "settlement_resolution_report_id")


def _provider_resolution_rows_with_timeout(
    *,
    provider: Any,
    market_rows: list[dict[str, Any]],
    recorder_config: PolymarketRealCorpusRecorderConfig,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(
                ("result", list(provider.resolution_rows(market_rows, recorder_config)))
            )
        except Exception as exc:  # noqa: BLE001
            result_queue.put(("exception", exc))

    thread = threading.Thread(
        target=_target,
        name="v8-settlement-resolution-provider",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=max(0.001, timeout_seconds))
    if thread.is_alive():
        raise TimeoutError("settlement resolution provider request timed out")
    kind, payload = result_queue.get_nowait()
    if kind == "exception":
        raise payload
    return list(payload)


def _settlement_provider_safe(provider: Any) -> bool:
    return bool(
        getattr(provider, "read_only", False) is True
        and getattr(provider, "write_capable", True) is False
        and getattr(provider, "paper_only", False) is True
        and getattr(provider, "capital_at_risk", True) is False
        and getattr(provider, "polymarket_write_enabled", True) is False
        and getattr(provider, "wallet_signing_enabled", True) is False
    )


def _settlement_recorder_config(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
) -> PolymarketRealCorpusRecorderConfig:
    return PolymarketRealCorpusRecorderConfig(
        run_id=f"{config.run_id}-settlement-resolution",
        output_dir=Path(config.output_dir) / config.run_id / "_settlement_resolution",
        mock_public_data=False,
        build_phase2_corpus=False,
    )


def _settlement_market_rows_from_trace(
    *,
    trace_rows: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_market: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        market_id = str(row.get("market_id") or "")
        if not market_id:
            continue
        by_market.setdefault(market_id, row)
    rows = []
    for fill in fills:
        market_id = str(fill.get("market_id") or "")
        source = by_market.get(market_id, {})
        if not source:
            continue
        rows.append(
            {
                "market_id": market_id,
                "condition_id": str(source.get("condition_id") or market_id),
                "slug": str(source.get("slug") or ""),
                "market_family": str(source.get("market_family") or "btc_updown_5m"),
                "market_start_ts": source.get("market_start_ts"),
                "market_end_ts": source.get("market_end_ts"),
                "settlement_ts": source.get("settlement_ts"),
                "up_token_id": source.get("up_token_id"),
                "down_token_id": source.get("down_token_id"),
                "reference_price_source": str(
                    source.get("reference_price_source")
                    or "polymarket_official_btc_usd_reference"
                ),
                "reference_price_start": source.get("reference_price_start"),
                "reference_price_at_start": source.get("reference_price_at_start"),
                "raw_market_sha256": source.get("raw_market_sha256") or "0" * 64,
                "paper_only": True,
                "capital_at_risk": False,
                "broker_exchange_write_enabled": False,
                "live_exchange_write_enabled": False,
                "polymarket_write_enabled": False,
                "wallet_signing_enabled": False,
            }
        )
    return rows


def _settlement_evaluation_rows_from_resolutions(
    *,
    fills: list[dict[str, Any]],
    resolutions_by_market: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for fill in fills:
        market_id = str(fill.get("market_id"))
        resolution = resolutions_by_market.get(market_id)
        if resolution is None:
            continue
        outcome = _resolved_outcome_from_resolution(resolution)
        if outcome not in {"UP", "DOWN"}:
            continue
        side = str(fill.get("execution_guarded_side") or "").upper()
        payout = 1.0 if side == outcome else 0.0
        fill_price = _float(fill.get("paper_fill_price"))
        filled_size = _float(fill.get("filled_size"))
        execution_cost = _float(fill.get("total_execution_cost"))
        pnl = (filled_size * (payout - fill_price)) - execution_cost
        row = {
            "market_id": market_id,
            "paper_fresh_order_intent_id": fill.get("paper_fresh_order_intent_id"),
            "paper_fresh_fill_id": fill.get("paper_fresh_fill_id"),
            "resolved_outcome": outcome,
            "resolution_status": resolution.get("resolution_status", "normal"),
            "resolution_source_type": resolution.get("resolution_source_type"),
            "settlement_pnl": pnl,
            "settlement_calculation_rule_id": (
                "paper_fill_price_size_minus_execution_cost_v1"
            ),
            "paper_only": True,
            "capital_at_risk": False,
            "uses_settlement_pnl_for_decision_time_logic": False,
        }
        row["settlement_evaluation_row_hash"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _resolved_outcome_from_resolution(row: dict[str, Any]) -> str | None:
    outcome = str(row.get("resolved_outcome") or row.get("winning_outcome") or "").upper()
    if outcome in {"UP", "DOWN"}:
        return outcome
    payout_up = row.get("payout_up")
    payout_down = row.get("payout_down")
    if payout_up is not None and payout_down is not None:
        up = _float(payout_up)
        down = _float(payout_down)
        if up > down:
            return "UP"
        if down > up:
            return "DOWN"
    return None


def _settlement_pnl_rows(
    *,
    fills: list[dict[str, Any]],
    settlement_evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_order = {
        str(row.get("paper_fresh_order_intent_id")): dict(row)
        for row in settlement_evaluation_rows
        if row.get("paper_fresh_order_intent_id")
    }
    by_market = {
        str(row.get("market_id")): dict(row) for row in settlement_evaluation_rows
    }
    rows = []
    for fill in fills:
        market_id = str(fill.get("market_id"))
        order_id = str(fill.get("paper_fresh_order_intent_id"))
        evaluation = by_order.get(order_id, by_market.get(market_id, {}))
        settled = "settlement_pnl" in evaluation
        pnl = _float(evaluation.get("settlement_pnl")) if settled else 0.0
        row = {
            "market_id": market_id,
            "decision_ts": fill.get("decision_ts"),
            "paper_fresh_fill_id": fill.get("paper_fresh_fill_id"),
            "paper_fresh_order_intent_id": fill.get("paper_fresh_order_intent_id"),
            "execution_guarded_action": fill.get("execution_guarded_action"),
            "execution_guarded_side": fill.get("execution_guarded_side"),
            "settlement_status": "settled" if settled else "unresolved",
            "resolved_outcome": evaluation.get("resolved_outcome"),
            "resolution_source_type": evaluation.get("resolution_source_type"),
            "settlement_pnl": pnl,
            "unresolved_pnl": 0.0 if settled else 0.0,
            "paper_only": True,
            "capital_at_risk": False,
            "uses_settlement_pnl_for_decision_time_logic": False,
        }
        row["settlement_pnl_row_hash"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _one_hour_goal_report(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    fresh_result: Any,
    round_coverage: dict[str, Any],
    remap_execution: dict[str, Any],
    intents: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    settlement_rows: list[dict[str, Any]],
    settlement_resolution: dict[str, Any],
    round_artifacts: dict[str, Any],
) -> dict[str, Any]:
    settled_pnl = sum(_float(row.get("settlement_pnl")) for row in settlement_rows)
    unresolved_pnl = sum(_float(row.get("unresolved_pnl")) for row in settlement_rows)
    unresolved_count = sum(
        1 for row in settlement_rows if row.get("settlement_status") != "settled"
    )
    pnl_summary = _settlement_pnl_summary(settlement_rows)
    duration_requirement_passed = config.duration_seconds >= 3600
    runtime_safety_passed = (
        fresh_result.runtime_safety_report["paper_fresh_runtime_safety_passed"] is True
    )
    live_write_wallet_capital_blocked = True
    blockers = []
    if not duration_requirement_passed:
        blockers.append("duration_requirement_not_met")
    if round_coverage["complete_round_count"] <= 0:
        blockers.append("no_complete_rounds_observed")
    if round_coverage["unjustified_missing_bet_round_count"] > 0:
        blockers.append("complete_rounds_unjustified_missing_paper_bets")
    if settled_pnl <= 0.0:
        blockers.append("settled_pnl_not_positive")
    if unresolved_count:
        blockers.append("unresolved_round_pnl_remaining")
    if not runtime_safety_passed:
        blockers.append("paper_fresh_runtime_safety_failed")
    if not live_write_wallet_capital_blocked:
        blockers.append("paper_live_write_wallet_capital_safety_failed")
    provider_fail_fast_stop_triggered = bool(
        fresh_result.fresh_loop_run_report.get("provider_fail_fast_stop_triggered")
    )
    if provider_fail_fast_stop_triggered:
        blockers.append("orderbook_collection_failed_consecutive_limit")
    final_success = blockers == []
    report = {
        "schema_version": ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION,
        "report_type": "one_hour_remap_paper_goal",
        "run_id": config.run_id,
        "duration_seconds": config.duration_seconds,
        "duration_requirement_passed": duration_requirement_passed,
        "short_diagnostic_run_allowed": config.allow_short_diagnostic_run,
        "public_data_source": fresh_result.manifest["paper_fresh_loop_public_data_source"],
        "read_only_public_provider_required_for_real_run": True,
        "uses_read_only_public_provider_only": (
            fresh_result.manifest["paper_fresh_loop_public_data_source"]
            == O_V8_PUBLIC_DATA_SOURCE_READ_ONLY_PROVIDER
        ),
        "frozen_ev_calibration_artifact_path": str(
            config.frozen_ev_calibration_artifact_path
        ),
        "complete_round_count": round_coverage["complete_round_count"],
        "complete_rounds_with_bet_count": round_coverage[
            "complete_rounds_with_bet_count"
        ],
        "missing_bet_round_count": round_coverage["missing_bet_round_count"],
        "guard_justified_no_bet_round_count": round_coverage[
            "guard_justified_no_bet_round_count"
        ],
        "guard_justified_no_bet_round_ids": round_coverage[
            "guard_justified_no_bet_round_ids"
        ],
        "unjustified_missing_bet_round_count": round_coverage[
            "unjustified_missing_bet_round_count"
        ],
        "unjustified_missing_bet_round_ids": round_coverage[
            "unjustified_missing_bet_round_ids"
        ],
        "guard_justified_no_bet_blocker_category_distribution": round_coverage[
            "guard_justified_no_bet_blocker_category_distribution"
        ],
        "unjustified_missing_bet_reason_distribution": round_coverage[
            "unjustified_missing_bet_reason_distribution"
        ],
        "round_coverage_classification_rows": round_coverage[
            "round_coverage_rows"
        ],
        "normal_policy_bet_count": remap_execution["normal_policy_bet_count"],
        "remap_paper_bet_count": remap_execution["remap_paper_bet_count"],
        "forced_coverage_bet_count": remap_execution["forced_coverage_bet_count"],
        "forced_coverage_round_ids": remap_execution["forced_coverage_round_ids"],
        "forced_coverage_guard_passed_count": remap_execution[
            "forced_coverage_guard_passed_count"
        ],
        "forced_coverage_guard_blocked_count": remap_execution[
            "forced_coverage_guard_blocked_count"
        ],
        "forced_coverage_blocking_reason_distribution": remap_execution[
            "forced_coverage_blocking_reason_distribution"
        ],
        "forced_coverage_blocker_category_distribution": remap_execution[
            "forced_coverage_blocker_category_distribution"
        ],
        "forced_coverage_candidate_selection_rule_id": remap_execution[
            "forced_coverage_candidate_selection_rule_id"
        ],
        "forced_coverage_candidate_attempt_count": remap_execution[
            "forced_coverage_candidate_attempt_count"
        ],
        "paper_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "settled_pnl": settled_pnl,
        "unresolved_pnl": unresolved_pnl,
        "unresolved_settlement_count": unresolved_count,
        "settled_fill_count": pnl_summary["settled_fill_count"],
        "winning_fill_count": pnl_summary["winning_fill_count"],
        "losing_fill_count": pnl_summary["losing_fill_count"],
        "flat_fill_count": pnl_summary["flat_fill_count"],
        "pnl_by_side": pnl_summary["pnl_by_side"],
        "pnl_by_action": pnl_summary["pnl_by_action"],
        "resolved_outcome_distribution": pnl_summary[
            "resolved_outcome_distribution"
        ],
        "settlement_polling_enabled": settlement_resolution["settlement_polling_enabled"],
        "settlement_poll_attempt_count": settlement_resolution[
            "settlement_poll_attempt_count"
        ],
        "settlement_resolution_reason_codes": settlement_resolution[
            "settlement_resolution_reason_codes"
        ],
        "settlement_evaluation_row_count": settlement_resolution[
            "settlement_evaluation_row_count"
        ],
        "provider_fail_fast_stop_triggered": provider_fail_fast_stop_triggered,
        "provider_fail_fast_reason_codes": fresh_result.fresh_loop_run_report.get(
            "provider_fail_fast_reason_codes", []
        ),
        "max_consecutive_orderbook_failure_rounds": (
            fresh_result.fresh_loop_run_report.get(
                "max_consecutive_orderbook_failure_rounds"
            )
        ),
        "consecutive_orderbook_failure_count_at_stop": (
            fresh_result.fresh_loop_run_report.get(
                "consecutive_orderbook_failure_count_at_stop"
            )
        ),
        "per_round_async_artifact_flush_enabled": round_artifacts[
            "per_round_async_artifact_flush_enabled"
        ],
        "per_round_bet_artifact_count": round_artifacts[
            "per_round_bet_artifact_count"
        ],
        "per_round_outcome_artifact_count": round_artifacts[
            "per_round_outcome_artifact_count"
        ],
        "per_round_artifact_manifest_path": round_artifacts["manifest_path"],
        "per_round_artifact_manifest_sha256": round_artifacts["manifest_sha256"],
        "final_goal_success": final_success,
        "goal_failure_reason_codes": blockers,
        "losing_action_distribution": _action_distribution_by_pnl(
            settlement_rows, positive=False
        ),
        "winning_action_distribution": _action_distribution_by_pnl(
            settlement_rows, positive=True
        ),
        "guard_blocking_reason_distribution": fresh_result.fresh_loop_run_report[
            "block_reason_distribution"
        ],
        "live_write_wallet_capital_blocked": live_write_wallet_capital_blocked,
        "uses_settlement_pnl_or_outcome_labels_in_decision_logic": False,
        "uses_oracle_actions_or_future_returns": False,
        "source_scores_mutated": False,
        "o_score_mutated": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        **compact_safety_fields(),
    }
    return _with_report_id(report, "one_hour_remap_paper_goal_report_id")


def _one_hour_goal_manifest(
    *,
    config: ExecutionLayerV2OneHourRemapPaperGoalConfig,
    artifact_paths: dict[str, Path],
    artifact_hashes: dict[str, str],
    goal_report: dict[str, Any],
    round_coverage: dict[str, Any],
    remap_execution: dict[str, Any],
    settlement_resolution: dict[str, Any],
    fresh_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": ONE_HOUR_REMAP_PAPER_GOAL_MANIFEST_SCHEMA_VERSION,
        "report_type": "one_hour_remap_paper_goal_manifest",
        "run_id": config.run_id,
        "artifact_paths": {
            name: str(path) for name, path in sorted(artifact_paths.items())
        },
        "artifact_hashes": dict(artifact_hashes),
        "one_hour_remap_paper_goal_report_id": goal_report[
            "one_hour_remap_paper_goal_report_id"
        ],
        "round_coverage_report_id": round_coverage["round_coverage_report_id"],
        "remap_execution_report_id": remap_execution["remap_execution_report_id"],
        "settlement_resolution_report_id": settlement_resolution[
            "settlement_resolution_report_id"
        ],
        "paper_fresh_loop_manifest_id": fresh_manifest[
            "o_v8_paper_fresh_loop_manifest_id"
        ],
        "duration_seconds": goal_report["duration_seconds"],
        "complete_round_count": goal_report["complete_round_count"],
        "complete_rounds_with_bet_count": goal_report[
            "complete_rounds_with_bet_count"
        ],
        "missing_bet_round_count": goal_report["missing_bet_round_count"],
        "guard_justified_no_bet_round_count": goal_report[
            "guard_justified_no_bet_round_count"
        ],
        "guard_justified_no_bet_round_ids": goal_report[
            "guard_justified_no_bet_round_ids"
        ],
        "unjustified_missing_bet_round_count": goal_report[
            "unjustified_missing_bet_round_count"
        ],
        "unjustified_missing_bet_round_ids": goal_report[
            "unjustified_missing_bet_round_ids"
        ],
        "guard_justified_no_bet_blocker_category_distribution": goal_report[
            "guard_justified_no_bet_blocker_category_distribution"
        ],
        "unjustified_missing_bet_reason_distribution": goal_report[
            "unjustified_missing_bet_reason_distribution"
        ],
        "normal_policy_bet_count": goal_report["normal_policy_bet_count"],
        "remap_paper_bet_count": goal_report["remap_paper_bet_count"],
        "forced_coverage_bet_count": goal_report["forced_coverage_bet_count"],
        "forced_coverage_round_ids": goal_report["forced_coverage_round_ids"],
        "forced_coverage_guard_passed_count": goal_report[
            "forced_coverage_guard_passed_count"
        ],
        "forced_coverage_guard_blocked_count": goal_report[
            "forced_coverage_guard_blocked_count"
        ],
        "forced_coverage_blocking_reason_distribution": goal_report[
            "forced_coverage_blocking_reason_distribution"
        ],
        "forced_coverage_blocker_category_distribution": goal_report[
            "forced_coverage_blocker_category_distribution"
        ],
        "forced_coverage_candidate_selection_rule_id": goal_report[
            "forced_coverage_candidate_selection_rule_id"
        ],
        "forced_coverage_candidate_attempt_count": goal_report[
            "forced_coverage_candidate_attempt_count"
        ],
        "settled_pnl": goal_report["settled_pnl"],
        "unresolved_pnl": goal_report["unresolved_pnl"],
        "settled_fill_count": goal_report["settled_fill_count"],
        "winning_fill_count": goal_report["winning_fill_count"],
        "losing_fill_count": goal_report["losing_fill_count"],
        "flat_fill_count": goal_report["flat_fill_count"],
        "pnl_by_side": goal_report["pnl_by_side"],
        "pnl_by_action": goal_report["pnl_by_action"],
        "resolved_outcome_distribution": goal_report[
            "resolved_outcome_distribution"
        ],
        "settlement_poll_attempt_count": goal_report["settlement_poll_attempt_count"],
        "settlement_evaluation_row_count": goal_report["settlement_evaluation_row_count"],
        "settlement_resolution_reason_codes": goal_report[
            "settlement_resolution_reason_codes"
        ],
        "provider_fail_fast_stop_triggered": goal_report[
            "provider_fail_fast_stop_triggered"
        ],
        "provider_fail_fast_reason_codes": goal_report[
            "provider_fail_fast_reason_codes"
        ],
        "max_consecutive_orderbook_failure_rounds": goal_report[
            "max_consecutive_orderbook_failure_rounds"
        ],
        "consecutive_orderbook_failure_count_at_stop": goal_report[
            "consecutive_orderbook_failure_count_at_stop"
        ],
        "per_round_async_artifact_flush_enabled": goal_report[
            "per_round_async_artifact_flush_enabled"
        ],
        "per_round_bet_artifact_count": goal_report[
            "per_round_bet_artifact_count"
        ],
        "per_round_outcome_artifact_count": goal_report[
            "per_round_outcome_artifact_count"
        ],
        "final_goal_success": goal_report["final_goal_success"],
        "goal_failure_reason_codes": goal_report["goal_failure_reason_codes"],
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "paper_run_resume_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    return _with_report_id(manifest, "one_hour_remap_paper_goal_manifest_id")


def _one_hour_goal_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# One-Hour Remap Paper Goal",
            "",
            f"- run_id: `{report['run_id']}`",
            f"- duration_seconds: `{report['duration_seconds']}`",
            f"- complete_round_count: `{report['complete_round_count']}`",
            f"- complete_rounds_with_bet_count: `{report['complete_rounds_with_bet_count']}`",
            f"- missing_bet_round_count: `{report['missing_bet_round_count']}`",
            f"- guard_justified_no_bet_round_count: `{report['guard_justified_no_bet_round_count']}`",
            f"- unjustified_missing_bet_round_count: `{report['unjustified_missing_bet_round_count']}`",
            f"- normal_policy_bet_count: `{report['normal_policy_bet_count']}`",
            f"- remap_paper_bet_count: `{report['remap_paper_bet_count']}`",
            f"- forced_coverage_bet_count: `{report['forced_coverage_bet_count']}`",
            f"- forced_coverage_guard_passed_count: `{report['forced_coverage_guard_passed_count']}`",
            f"- forced_coverage_guard_blocked_count: `{report['forced_coverage_guard_blocked_count']}`",
            f"- settled_pnl: `{report['settled_pnl']}`",
            f"- unresolved_pnl: `{report['unresolved_pnl']}`",
            f"- settled_fill_count: `{report['settled_fill_count']}`",
            f"- winning_fill_count: `{report['winning_fill_count']}`",
            f"- losing_fill_count: `{report['losing_fill_count']}`",
            f"- pnl_by_side: `{report['pnl_by_side']}`",
            f"- pnl_by_action: `{report['pnl_by_action']}`",
            f"- settlement_poll_attempt_count: `{report['settlement_poll_attempt_count']}`",
            f"- settlement_evaluation_row_count: `{report['settlement_evaluation_row_count']}`",
            f"- provider_fail_fast_stop_triggered: `{report['provider_fail_fast_stop_triggered']}`",
            f"- max_consecutive_orderbook_failure_rounds: `{report['max_consecutive_orderbook_failure_rounds']}`",
            f"- consecutive_orderbook_failure_count_at_stop: `{report['consecutive_orderbook_failure_count_at_stop']}`",
            f"- per_round_async_artifact_flush_enabled: `{report['per_round_async_artifact_flush_enabled']}`",
            f"- per_round_bet_artifact_count: `{report['per_round_bet_artifact_count']}`",
            f"- per_round_outcome_artifact_count: `{report['per_round_outcome_artifact_count']}`",
            f"- final_goal_success: `{report['final_goal_success']}`",
            "",
            "## Settlement Resolution",
            "",
            *[
                f"- `{reason}`"
                for reason in report["settlement_resolution_reason_codes"]
            ],
            "",
            "## Forced Coverage",
            "",
            f"- forced_coverage_round_ids: `{report['forced_coverage_round_ids']}`",
            f"- forced_coverage_blocking_reason_distribution: `{report['forced_coverage_blocking_reason_distribution']}`",
            f"- forced_coverage_blocker_category_distribution: `{report['forced_coverage_blocker_category_distribution']}`",
            f"- forced_coverage_candidate_attempt_count: `{report['forced_coverage_candidate_attempt_count']}`",
            f"- guard_justified_no_bet_round_ids: `{report['guard_justified_no_bet_round_ids']}`",
            f"- unjustified_missing_bet_round_ids: `{report['unjustified_missing_bet_round_ids']}`",
            "",
            "## Failure Reasons",
            "",
            *[f"- `{reason}`" for reason in report["goal_failure_reason_codes"]],
            "",
        ]
    )


def _round_coverage_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round Coverage",
            "",
            f"- complete_round_count: `{report['complete_round_count']}`",
            f"- complete_rounds_with_bet_count: `{report['complete_rounds_with_bet_count']}`",
            f"- missing_bet_round_count: `{report['missing_bet_round_count']}`",
            f"- guard_justified_no_bet_round_count: `{report['guard_justified_no_bet_round_count']}`",
            f"- unjustified_missing_bet_round_count: `{report['unjustified_missing_bet_round_count']}`",
            f"- guard_justified_no_bet_blocker_category_distribution: `{report['guard_justified_no_bet_blocker_category_distribution']}`",
            "",
        ]
    )


def _remap_execution_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remap Execution",
            "",
            f"- hts_time_window_blocked_count: `{report['hts_time_window_blocked_count']}`",
            f"- same_side_sbc_remap_guard_passed_count: `{report['same_side_sbc_remap_guard_passed_count']}`",
            f"- remap_paper_bet_count: `{report['remap_paper_bet_count']}`",
            f"- normal_policy_bet_count: `{report['normal_policy_bet_count']}`",
            f"- forced_coverage_bet_count: `{report['forced_coverage_bet_count']}`",
            f"- forced_coverage_guard_passed_count: `{report['forced_coverage_guard_passed_count']}`",
            f"- forced_coverage_guard_blocked_count: `{report['forced_coverage_guard_blocked_count']}`",
            f"- forced_coverage_blocker_category_distribution: `{report['forced_coverage_blocker_category_distribution']}`",
            "",
        ]
    )


def _settlement_resolution_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Settlement Resolution",
            "",
            f"- settlement_polling_enabled: `{report['settlement_polling_enabled']}`",
            f"- settlement_provider_class: `{report['settlement_provider_class']}`",
            f"- settlement_provider_safety_passed: `{report['settlement_provider_safety_passed']}`",
            f"- settlement_poll_attempt_count: `{report['settlement_poll_attempt_count']}`",
            f"- unresolved_fill_count_before_poll: `{report['unresolved_fill_count_before_poll']}`",
            f"- raw_resolution_row_count: `{report['raw_resolution_row_count']}`",
            f"- settlement_evaluation_row_count: `{report['settlement_evaluation_row_count']}`",
            f"- unresolved_fill_count_after_poll: `{report['unresolved_fill_count_after_poll']}`",
            f"- uses_settlement_pnl_for_decision_time_logic: `{report['uses_settlement_pnl_for_decision_time_logic']}`",
            "",
            "## Reason Codes",
            "",
            *[
                f"- `{reason}`"
                for reason in report["settlement_resolution_reason_codes"]
            ],
            "",
        ]
    )


def _action_distribution_by_pnl(
    rows: list[dict[str, Any]],
    *,
    positive: bool,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        pnl = _float(row.get("settlement_pnl"))
        if positive and pnl <= 0.0:
            continue
        if not positive and pnl >= 0.0:
            continue
        counter[str(row.get("execution_guarded_action"))] += 1
    return dict(sorted(counter.items()))


def _reason_distribution(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row.get(field_name) or []
        if isinstance(values, str):
            values = [values]
        counter.update(str(value) for value in values)
    return dict(sorted(counter.items()))


def _safe_path_component(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    ).strip("._")
    return cleaned or "unknown_market"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path is not None else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_report_id(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    copied = dict(payload)
    copied[field_name] = canonical_json_sha256(copied)
    return copied


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
