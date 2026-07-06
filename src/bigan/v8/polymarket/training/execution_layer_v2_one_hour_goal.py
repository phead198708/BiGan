"""One-hour paper-only goal diagnostics for Execution Layer v2 remaps."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
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
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if self.duration_seconds < 3600:
            raise ValueError("duration_seconds must be at least 3600")
        if self.poll_interval_seconds < 0.0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.settlement_poll_max_wait_seconds < 0.0:
            raise ValueError("settlement_poll_max_wait_seconds must be non-negative")
        if self.settlement_poll_interval_seconds <= 0.0:
            raise ValueError("settlement_poll_interval_seconds must be positive")
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
    fresh_result = run_polymarket_o_v8_paper_fresh_loop(
        PolymarketOV8PaperFreshLoopConfig(
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
    )

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
    round_coverage = _round_coverage_report(
        run_id=config.run_id,
        trace_rows=trace_rows,
        intents=intents,
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
) -> dict[str, Any]:
    complete_rounds = sorted({str(row.get("market_id")) for row in trace_rows if row.get("market_id")})
    bet_rounds = sorted({str(row.get("market_id")) for row in intents if row.get("market_id")})
    missing = sorted(set(complete_rounds) - set(bet_rounds))
    rows = [
        {
            "market_id": market_id,
            "complete_round": True,
            "paper_bet_created": market_id in set(bet_rounds),
            "paper_bet_count": sum(
                1 for intent in intents if str(intent.get("market_id")) == market_id
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
        "round_coverage_rows": rows,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
    }
    return _with_report_id(report, "round_coverage_report_id")


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
        source_trace = _forced_coverage_source_trace(trace_by_market[market_id])
        coverage_public_row = _forced_coverage_public_row(
            run_id=config.run_id,
            source_trace=source_trace,
        )
        if coverage_public_row is None:
            attempt_rows.append(
                _forced_coverage_attempt_row(
                    market_id=market_id,
                    source_trace=source_trace,
                    guard_row=None,
                    intent=None,
                    reason_codes=["forced_coverage_no_trade_candidate_available"],
                )
            )
            continue
        guard_input = _guard_input_from_public_row(
            public_row=coverage_public_row,
            cycle_id=f"{config.run_id}-forced-coverage",
            row_index=len(attempt_rows),
        )
        guard_row = _v8_execution_guard_decision(
            guard_input,
            guard_config=guard_config,
            runtime_state=runtime_state,
            runtime_mode="simulated_runtime_state",
        )
        guard_row["cycle_id"] = f"{config.run_id}-forced-coverage"
        guard_row["public_data_source"] = coverage_public_row["public_data_source"]
        guard_row["coverage_forced_attempted"] = True
        guard_row["coverage_forced_round_id"] = market_id
        guard_row["coverage_forced_reason_codes"] = [
            "complete_round_missing_paper_bet",
            "forced_coverage_full_execution_guard_checked",
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
                reason_codes=list(guard_row.get("execution_blocking_reason_codes") or []),
            )
        )
    block_reasons = Counter(
        reason
        for row in attempt_rows
        if row["forced_coverage_guard_passed"] is not True
        for reason in row["forced_coverage_blocking_reason_codes"]
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


def _forced_coverage_source_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("order_allowed")),
            _float(
                row.get("canonical_corrected_score")
                or row.get("simplified_corrected_score")
            ),
            int(row.get("decision_ts") or 0),
        ),
        reverse=True,
    )[0]


def _forced_coverage_public_row(
    *,
    run_id: str,
    source_trace: dict[str, Any],
) -> dict[str, Any] | None:
    ranking = _forced_coverage_full_ranking(source_trace)
    candidate = next(
        (
            row
            for row in ranking
            if str(row.get("selected_action") or "") != "NO_TRADE"
        ),
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
        "microstructure_snapshot": {
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


def _forced_coverage_attempt_row(
    *,
    market_id: str,
    source_trace: dict[str, Any],
    guard_row: dict[str, Any] | None,
    intent: dict[str, Any] | None,
    reason_codes: list[str],
) -> dict[str, Any]:
    passed = intent is not None
    blocking = sorted(set(reason_codes or ["forced_coverage_guard_blocked"]))
    row = {
        "market_id": market_id,
        "decision_ts": source_trace.get("decision_ts"),
        "coverage_forced_attempted": True,
        "coverage_forced_paper_bet": passed,
        "forced_coverage_guard_passed": passed,
        "forced_coverage_blocking_reason_codes": [] if passed else blocking,
        "forced_coverage_selected_action": (
            guard_row.get("source_selected_action") if guard_row is not None else None
        ),
        "forced_coverage_execution_guarded_action": (
            guard_row.get("execution_guarded_action") if guard_row is not None else None
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
            try:
                resolution_rows = list(
                    provider.resolution_rows(
                        market_rows,
                        _settlement_recorder_config(config=config),
                    )
                )
            except RealCorpusPublicProviderError as exc:
                reason_codes.extend(
                    ["settlement_resolution_provider_error", *list(exc.reason_codes)]
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
) -> dict[str, Any]:
    settled_pnl = sum(_float(row.get("settlement_pnl")) for row in settlement_rows)
    unresolved_pnl = sum(_float(row.get("unresolved_pnl")) for row in settlement_rows)
    unresolved_count = sum(
        1 for row in settlement_rows if row.get("settlement_status") != "settled"
    )
    blockers = []
    if round_coverage["complete_round_count"] <= 0:
        blockers.append("no_complete_rounds_observed")
    if round_coverage["missing_bet_round_count"] > 0:
        blockers.append("complete_rounds_missing_paper_bets")
    if settled_pnl <= 0.0:
        blockers.append("settled_pnl_not_positive")
    if unresolved_count:
        blockers.append("unresolved_round_pnl_remaining")
    if fresh_result.runtime_safety_report["paper_fresh_runtime_safety_passed"] is not True:
        blockers.append("paper_fresh_runtime_safety_failed")
    final_success = blockers == []
    report = {
        "schema_version": ONE_HOUR_REMAP_PAPER_GOAL_SCHEMA_VERSION,
        "report_type": "one_hour_remap_paper_goal",
        "run_id": config.run_id,
        "duration_seconds": config.duration_seconds,
        "duration_requirement_passed": config.duration_seconds >= 3600,
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
        "paper_intent_count": len(intents),
        "paper_fill_count": len(fills),
        "settled_pnl": settled_pnl,
        "unresolved_pnl": unresolved_pnl,
        "unresolved_settlement_count": unresolved_count,
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
        "settled_pnl": goal_report["settled_pnl"],
        "unresolved_pnl": goal_report["unresolved_pnl"],
        "settlement_poll_attempt_count": goal_report["settlement_poll_attempt_count"],
        "settlement_evaluation_row_count": goal_report["settlement_evaluation_row_count"],
        "settlement_resolution_reason_codes": goal_report[
            "settlement_resolution_reason_codes"
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
            f"- normal_policy_bet_count: `{report['normal_policy_bet_count']}`",
            f"- remap_paper_bet_count: `{report['remap_paper_bet_count']}`",
            f"- forced_coverage_bet_count: `{report['forced_coverage_bet_count']}`",
            f"- forced_coverage_guard_passed_count: `{report['forced_coverage_guard_passed_count']}`",
            f"- forced_coverage_guard_blocked_count: `{report['forced_coverage_guard_blocked_count']}`",
            f"- settled_pnl: `{report['settled_pnl']}`",
            f"- unresolved_pnl: `{report['unresolved_pnl']}`",
            f"- settlement_poll_attempt_count: `{report['settlement_poll_attempt_count']}`",
            f"- settlement_evaluation_row_count: `{report['settlement_evaluation_row_count']}`",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
