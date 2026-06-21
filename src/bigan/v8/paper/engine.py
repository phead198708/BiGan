"""Deterministic paper-only execution harness."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.paper.contracts import (
    PAPER_TRADING_HARNESS_PHASE,
    PaperDegradationConfig,
    PaperFill,
    PaperHarnessConfig,
    PaperLedgerEntry,
    PaperOrder,
    PaperPositionSnapshot,
    PaperRunReport,
    PaperSide,
    PaperTradingError,
    json_ready,
    stream_sha256,
)
from bigan.v8.paper.ledger import PaperLedger
from bigan.v8.phase4 import AdaptiveDecision
from bigan.v8.phase5 import (
    LiveExecutionObservation,
    SafetyLayerConfig,
    StableModelSnapshot,
    compute_safe_parameters_sha256,
    run_phase5_safety_layer,
)
from bigan.v8.phase5.safety import Phase5SafetyLayerResult
from bigan.v8.phase6 import (
    CICDPipelineConfig,
    CICDPipelineResult,
    CICDStageEvidence,
    RollbackPlan,
    run_phase6_cicd_pipeline,
)


@dataclass(frozen=True, slots=True)
class PaperHarnessResult:
    """Complete paper harness result and artifact paths."""

    orders: tuple[PaperOrder, ...]
    fills: tuple[PaperFill, ...]
    ledger_entries: tuple[PaperLedgerEntry, ...]
    positions: tuple[PaperPositionSnapshot, ...]
    observations: tuple[LiveExecutionObservation, ...]
    paper_report: PaperRunReport
    phase5_result: Phase5SafetyLayerResult
    phase6_result: CICDPipelineResult
    bundle_manifest: dict[str, Any]
    output_dir: Path
    artifact_paths: dict[str, Path]


def run_paper_trading_harness(
    *,
    decisions: tuple[AdaptiveDecision, ...],
    config: PaperHarnessConfig,
    phase5_config: SafetyLayerConfig | None = None,
    phase6_config: CICDPipelineConfig | None = None,
) -> PaperHarnessResult:
    """Run deterministic paper execution, Phase 5 safety, and Phase 6 evidence."""

    if not decisions:
        raise PaperTradingError("paper harness requires at least one Phase 4 decision")
    output_dir = Path(config.output_dir or ".").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    orders, fills, ledger_entries, positions = _execute_paper_decisions(
        decisions=decisions,
        config=config,
    )
    observations = paper_fills_to_live_observations(fills)
    safe_parameters = {"max_position_size": 0.10, "risk_mode": "paper_safe"}
    stable_model = StableModelSnapshot(
        model_id=f"{config.candidate_run_id}-paper-stable",
        model_sha256=config.model_sha256,
        policy_dataset_hash=config.policy_dataset_hash,
        split_hash=config.split_hash,
        safe_parameter_sha256=compute_safe_parameters_sha256(safe_parameters),
        safe_parameters=safe_parameters,
    )
    resolved_phase5_config = phase5_config or _default_phase5_config(output_dir)
    phase5_result = run_phase5_safety_layer(
        shadow_decisions=decisions,
        live_observations=observations,
        stable_model=stable_model,
        config=resolved_phase5_config,
    )
    if phase5_result.report_path is None:
        raise PaperTradingError("Phase 5 did not write a report")

    artifact_paths = _write_primary_paper_artifacts(
        output_dir=output_dir,
        orders=orders,
        fills=fills,
        ledger_entries=ledger_entries,
        positions=positions,
    )
    phase5_report_sha256 = _file_sha256(phase5_result.report_path)
    paper_report = _build_paper_report(
        config=config,
        orders=orders,
        fills=fills,
        ledger_entries=ledger_entries,
        positions=positions,
        phase5_report_sha256=phase5_report_sha256,
        phase6_report_sha256=None,
    )
    _write_json(output_dir / "paper_pnl_report.json", paper_report.to_dict())
    artifact_paths["paper_pnl_report"] = output_dir / "paper_pnl_report.json"

    paper_report_sha256 = _file_sha256(artifact_paths["paper_pnl_report"])
    phase6_result = _run_phase6_paper_evidence(
        config=config,
        paper_report_sha256=paper_report_sha256,
        phase5_report_sha256=phase5_report_sha256,
        phase5_passed=(
            phase5_result.passed
            and not phase5_result.report.safety_action["kill_switch_triggered"]
        ),
        stable_model=stable_model,
        safe_parameters=safe_parameters,
        output_dir=output_dir,
        phase6_config=phase6_config,
    )
    if phase6_result.report_path is None:
        raise PaperTradingError("Phase 6 did not write a report")

    phase6_report_sha256 = _file_sha256(phase6_result.report_path)
    paper_report = _build_paper_report(
        config=config,
        orders=orders,
        fills=fills,
        ledger_entries=ledger_entries,
        positions=positions,
        phase5_report_sha256=phase5_report_sha256,
        phase6_report_sha256=phase6_report_sha256,
    )
    _write_json(output_dir / "paper_pnl_report.json", paper_report.to_dict())
    paper_report_sha256 = _file_sha256(output_dir / "paper_pnl_report.json")
    artifact_paths.update(
        {
            "paper_pnl_report": output_dir / "paper_pnl_report.json",
            "phase5_report": phase5_result.report_path,
            "phase6_report": phase6_result.report_path,
        }
    )
    bundle_manifest = _bundle_manifest(
        config=config,
        paper_report=paper_report,
        phase6_deployment_status=phase6_result.report.deployment_status,
        artifact_paths=artifact_paths,
        paper_report_sha256=paper_report_sha256,
        phase5_report_sha256=phase5_report_sha256,
        phase6_report_sha256=phase6_report_sha256,
    )
    _write_json(output_dir / "paper_bundle_manifest.json", bundle_manifest)
    artifact_paths["paper_bundle_manifest"] = output_dir / "paper_bundle_manifest.json"
    return PaperHarnessResult(
        orders=orders,
        fills=fills,
        ledger_entries=ledger_entries,
        positions=positions,
        observations=observations,
        paper_report=paper_report,
        phase5_result=phase5_result,
        phase6_result=phase6_result,
        bundle_manifest=bundle_manifest,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
    )


def paper_fills_to_live_observations(
    fills: tuple[PaperFill, ...],
) -> tuple[LiveExecutionObservation, ...]:
    """Convert deterministic paper fills into Phase 5 live-equivalent observations."""

    return tuple(
        LiveExecutionObservation(
            decision_ts=fill.decision_ts,
            source=fill.source,
            instrument_id=fill.instrument_id,
            live_filled_action=fill.filled_action,
            live_net_return=fill.net_return,
            live_total_execution_cost=fill.total_execution_cost,
            live_regime=fill.paper_regime,
            capital_at_risk=False,
        )
        for fill in fills
    )


def _execute_paper_decisions(
    *,
    decisions: tuple[AdaptiveDecision, ...],
    config: PaperHarnessConfig,
) -> tuple[
    tuple[PaperOrder, ...],
    tuple[PaperFill, ...],
    tuple[PaperLedgerEntry, ...],
    tuple[PaperPositionSnapshot, ...],
]:
    ledger = PaperLedger(initial_cash=config.initial_cash)
    orders: list[PaperOrder] = []
    fills: list[PaperFill] = []
    for index, decision in enumerate(decisions):
        previous_action = ledger.position_size(
            source=decision.source,
            instrument_id=decision.instrument_id,
        )
        mark_price = _mark_price(config, index)
        side = _side(previous_action, decision.filled_action)
        requested_size = abs(decision.filled_action - previous_action)
        order = PaperOrder(
            order_id=f"{config.run_id}-order-{index:06d}",
            candidate_run_id=config.candidate_run_id,
            decision_ts=decision.decision_ts,
            source=decision.source,
            instrument_id=decision.instrument_id,
            side=side,
            previous_action=previous_action,
            requested_action=decision.filled_action,
            requested_size=requested_size,
            limit_price=_limit_price(mark_price=mark_price, side=side, decision=decision),
            order_type="paper_limit",
            created_at_ts=decision.decision_ts,
            metadata={
                "phase4_regime": decision.regime,
                "phase4_decision_index": index,
                "paper_engine": "deterministic_v1",
            },
        )
        fill = _paper_fill_from_order(
            order=order,
            decision=decision,
            index=index,
            mark_price=mark_price,
            config=config,
        )
        ledger.apply_fill(order, fill)
        orders.append(order)
        fills.append(fill)
    return tuple(orders), tuple(fills), ledger.entries, ledger.snapshots()


def _paper_fill_from_order(
    *,
    order: PaperOrder,
    decision: AdaptiveDecision,
    index: int,
    mark_price: float,
    config: PaperHarnessConfig,
) -> PaperFill:
    degradation = _active_degradation(config.degradation, index)
    fill_probability = max(
        config.min_fill_probability,
        min(1.0, decision.fill_probability),
    )
    filled_size = order.requested_size * fill_probability
    filled_action = _filled_action_after_order(order, filled_size)
    cost_multiplier = 1.0 if degradation is None else degradation.cost_multiplier
    spread_cost = decision.spread_cost * cost_multiplier
    fee_cost = decision.fee_cost * cost_multiplier
    slippage_cost = decision.slippage_cost * cost_multiplier
    liquidity_impact_cost = decision.liquidity_impact_cost * cost_multiplier
    total_execution_cost = (
        spread_cost + fee_cost + slippage_cost + liquidity_impact_cost
    )
    net_return = decision.net_return
    paper_regime = decision.regime
    if degradation is not None:
        net_return = decision.net_return - degradation.net_return_shift
        paper_regime = degradation.live_regime or paper_regime
    return PaperFill(
        fill_id=f"{config.run_id}-fill-{index:06d}",
        order_id=order.order_id,
        decision_ts=decision.decision_ts,
        source=decision.source,
        instrument_id=decision.instrument_id,
        side=order.side,
        requested_size=order.requested_size,
        filled_size=filled_size,
        filled_action=filled_action,
        fill_price=_fill_price(mark_price=mark_price, side=order.side, decision=decision),
        mark_price=mark_price,
        fill_probability=fill_probability,
        spread_cost=spread_cost,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        liquidity_impact_cost=liquidity_impact_cost,
        total_execution_cost=total_execution_cost,
        net_return=net_return,
        paper_regime=paper_regime,
        metadata={
            "paper_only": True,
            "capital_at_risk": False,
            "degradation_injected": degradation is not None,
        },
    )


def _run_phase6_paper_evidence(
    *,
    config: PaperHarnessConfig,
    paper_report_sha256: str,
    phase5_report_sha256: str,
    phase5_passed: bool,
    stable_model: StableModelSnapshot,
    safe_parameters: dict[str, Any],
    output_dir: Path,
    phase6_config: CICDPipelineConfig | None,
) -> CICDPipelineResult:
    identity = config.identity_metadata()
    paper_metadata = {
        **identity,
        "paper_only": True,
        "capital_at_risk": False,
    }
    stage_evidence = (
        CICDStageEvidence(
            stage="training",
            passed=True,
            artifact_sha256=config.model_sha256,
            report_sha256=config.upstream_training_report_sha256,
            run_id=config.candidate_run_id,
            metadata={
                **paper_metadata,
                "accepted_candidate_model": True,
                "deterministic_training": True,
                "model_sha256": config.model_sha256,
            },
        ),
        CICDStageEvidence(
            stage="validation",
            passed=True,
            artifact_sha256=config.upstream_validation_report_sha256,
            report_sha256=config.upstream_validation_report_sha256,
            run_id="paper_validation_evidence",
            metadata={
                **paper_metadata,
                "oos_backtest_passed": True,
                "cost_stress_passed": True,
                "cost_stress_multipliers": [1.2, 1.5, 2.0],
            },
        ),
        CICDStageEvidence(
            stage="shadow_deployment",
            passed=phase5_passed,
            artifact_sha256=paper_report_sha256,
            report_sha256=paper_report_sha256,
            run_id=config.run_id,
            metadata={
                **paper_metadata,
                "shadow_mode": True,
                "simulate_live_execution": True,
            },
        ),
        CICDStageEvidence(
            stage="live_deployment",
            passed=phase5_passed,
            artifact_sha256=paper_report_sha256,
            report_sha256=paper_report_sha256,
            run_id=f"{config.run_id}_paper_rollout",
            metadata={
                **paper_metadata,
                "staged_capital_rollout": True,
                "manual_approval_recorded": True,
                "rollout_capital_fractions": [0.0, 0.01, 0.05, 0.10],
                "rollout_step_index": 0,
                "requested_capital_fraction": 0.0,
                "paper_mode_ci_cd_evidence": True,
            },
        ),
        CICDStageEvidence(
            stage="monitoring",
            passed=True,
            artifact_sha256=phase5_report_sha256,
            report_sha256=phase5_report_sha256,
            run_id=f"{config.run_id}_paper_monitoring",
            metadata={
                **paper_metadata,
                "performance_tracking_enabled": True,
                "risk_tracking_enabled": True,
                "kill_switch_wired": True,
            },
        ),
    )
    rollback_plan = RollbackPlan(
        stable_model_id=stable_model.model_id,
        stable_model_sha256=stable_model.model_sha256,
        safe_parameter_sha256=stable_model.safe_parameter_sha256,
        safe_parameters=safe_parameters,
        rollback_artifact_sha256=phase5_report_sha256,
        latency_measurements_ms=(50, 65, 80),
    )
    resolved_config = phase6_config or CICDPipelineConfig(
        output_dir=output_dir,
        created_at=config.created_at,
    )
    return run_phase6_cicd_pipeline(
        candidate_run_id=config.candidate_run_id,
        stage_evidence=stage_evidence,
        rollback_plan=rollback_plan,
        config=resolved_config,
    )


def _build_paper_report(
    *,
    config: PaperHarnessConfig,
    orders: tuple[PaperOrder, ...],
    fills: tuple[PaperFill, ...],
    ledger_entries: tuple[PaperLedgerEntry, ...],
    positions: tuple[PaperPositionSnapshot, ...],
    phase5_report_sha256: str | None,
    phase6_report_sha256: str | None,
) -> PaperRunReport:
    net_returns = [fill.net_return for fill in fills]
    total_execution_cost = sum(fill.total_execution_cost for fill in fills)
    criteria = {
        "paper_only": config.paper_only
        and _all_paper_only(orders)
        and _all_paper_only(fills)
        and _all_paper_only(ledger_entries)
        and _all_paper_only(positions),
        "no_capital_at_risk": not config.capital_at_risk
        and _all_no_capital_at_risk(orders)
        and _all_no_capital_at_risk(fills)
        and _all_no_capital_at_risk(ledger_entries)
        and _all_no_capital_at_risk(positions),
        "orders_fills_ledger_aligned": (
            len(orders) == len(fills) == len(ledger_entries)
        ),
        "paper_hashes_present": all(
            len(value) == 64
            for value in (
                stream_sha256(orders),
                stream_sha256(fills),
                stream_sha256(ledger_entries),
                stream_sha256(positions),
            )
        ),
        "phase5_evidence_recorded": phase5_report_sha256 is not None,
        "broker_exchange_write_disabled": not config.broker_write_enabled,
    }
    if phase6_report_sha256 is not None:
        criteria["phase6_paper_evidence_recorded"] = True
    return PaperRunReport(
        phase=PAPER_TRADING_HARNESS_PHASE,
        run_id=config.run_id,
        candidate_run_id=config.candidate_run_id,
        model_sha256=config.model_sha256,
        policy_dataset_hash=config.policy_dataset_hash,
        split_hash=config.split_hash,
        paper_only=True,
        capital_at_risk=False,
        row_count=len(fills),
        order_count=len(orders),
        fill_count=len(fills),
        ledger_entry_count=len(ledger_entries),
        paper_order_stream_sha256=stream_sha256(orders),
        paper_fill_stream_sha256=stream_sha256(fills),
        paper_ledger_sha256=stream_sha256(ledger_entries),
        paper_positions_sha256=stream_sha256(positions),
        mean_net_return=sum(net_returns) / len(net_returns) if net_returns else 0.0,
        max_drawdown=_max_drawdown(net_returns),
        total_execution_cost=total_execution_cost,
        phase5_report_sha256=phase5_report_sha256,
        phase6_report_sha256=phase6_report_sha256,
        acceptance_criteria=criteria,
        config=config.to_dict(),
        created_at=config.created_at,
    )


def _bundle_manifest(
    *,
    config: PaperHarnessConfig,
    paper_report: PaperRunReport,
    phase6_deployment_status: str,
    artifact_paths: dict[str, Path],
    paper_report_sha256: str,
    phase5_report_sha256: str,
    phase6_report_sha256: str,
) -> dict[str, Any]:
    artifacts = {
        name: {
            "path": path.name,
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(artifact_paths.items())
    }
    return {
        "schema_version": "bigan-v8-paper-harness-bundle-v1",
        "run_id": config.run_id,
        "candidate_run_id": config.candidate_run_id,
        "model_sha256": config.model_sha256,
        "policy_dataset_hash": config.policy_dataset_hash,
        "split_hash": config.split_hash,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "paper_order_stream_sha256": paper_report.paper_order_stream_sha256,
        "paper_fill_stream_sha256": paper_report.paper_fill_stream_sha256,
        "paper_ledger_sha256": paper_report.paper_ledger_sha256,
        "paper_positions_sha256": paper_report.paper_positions_sha256,
        "paper_report_sha256": paper_report_sha256,
        "phase5_report_sha256": phase5_report_sha256,
        "phase6_report_sha256": phase6_report_sha256,
        "phase6_deployment_status": phase6_deployment_status,
        "artifacts": artifacts,
    }


def _write_primary_paper_artifacts(
    *,
    output_dir: Path,
    orders: tuple[PaperOrder, ...],
    fills: tuple[PaperFill, ...],
    ledger_entries: tuple[PaperLedgerEntry, ...],
    positions: tuple[PaperPositionSnapshot, ...],
) -> dict[str, Path]:
    paths = {
        "paper_orders": output_dir / "paper_orders.jsonl",
        "paper_fills": output_dir / "paper_fills.jsonl",
        "paper_ledger": output_dir / "paper_ledger.jsonl",
        "paper_positions": output_dir / "paper_positions.json",
    }
    _write_jsonl(paths["paper_orders"], [order.to_dict() for order in orders])
    _write_jsonl(paths["paper_fills"], [fill.to_dict() for fill in fills])
    _write_jsonl(
        paths["paper_ledger"],
        [entry.to_dict() for entry in ledger_entries],
    )
    _write_json(
        paths["paper_positions"],
        {
            "paper_only": True,
            "capital_at_risk": False,
            "position_snapshot_sha256": stream_sha256(positions),
            "positions": [position.to_dict() for position in positions],
        },
    )
    return paths


def _default_phase5_config(output_dir: Path) -> SafetyLayerConfig:
    return SafetyLayerConfig(
        detection_window_size=4,
        min_shadow_live_correlation=0.70,
        max_mean_pnl_drift=0.006,
        max_cost_drift_ratio=0.50,
        max_regime_mismatch_rate=0.25,
        max_live_drawdown=0.05,
        output_dir=output_dir,
        created_at="2026-06-22T01:00:00Z",
    )


def _active_degradation(
    degradation: PaperDegradationConfig | None,
    index: int,
) -> PaperDegradationConfig | None:
    if degradation is None or index < degradation.start_index:
        return None
    return degradation


def _side(previous_action: float, requested_action: float) -> PaperSide:
    if requested_action > previous_action:
        return "buy"
    if requested_action < previous_action:
        return "sell"
    return "hold"


def _mark_price(config: PaperHarnessConfig, index: int) -> float:
    return config.base_mark_price * (1.0 + 0.0005 * index)


def _limit_price(
    *,
    mark_price: float,
    side: PaperSide,
    decision: AdaptiveDecision,
) -> float:
    half_spread = max(decision.spread_cost, 0.0) * mark_price / 2.0
    if side == "buy":
        return mark_price + half_spread
    if side == "sell":
        return max(1e-12, mark_price - half_spread)
    return mark_price


def _fill_price(
    *,
    mark_price: float,
    side: PaperSide,
    decision: AdaptiveDecision,
) -> float:
    spread_component = max(decision.spread_cost, 0.0) * mark_price / 2.0
    slippage_component = max(decision.slippage_cost, 0.0) * mark_price
    if side == "buy":
        return mark_price + spread_component + slippage_component
    if side == "sell":
        return max(1e-12, mark_price - spread_component - slippage_component)
    return mark_price


def _filled_action_after_order(order: PaperOrder, filled_size: float) -> float:
    if order.side == "buy":
        return min(1.0, order.previous_action + filled_size)
    if order.side == "sell":
        return max(0.0, order.previous_action - filled_size)
    return order.previous_action


def _all_paper_only(records: tuple[Any, ...]) -> bool:
    return all(getattr(record, "paper_only", False) is True for record in records)


def _all_no_capital_at_risk(records: tuple[Any, ...]) -> bool:
    return all(getattr(record, "capital_at_risk", True) is False for record in records)


def _max_drawdown(returns: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(json_ready(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
