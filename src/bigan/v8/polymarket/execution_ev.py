"""Expected-value execution layer for trained Polymarket policy predictions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from bigan.v8.polymarket.ledger import PolymarketPositionLedger
from bigan.v8.polymarket.rules import (
    build_btc_updown_resolution_rule,
    payout_for_resolved_outcome,
    resolve_polymarket_rule,
)
from bigan.v8.polymarket.training.contracts import (
    PolicyAction,
    PolicyOutcome,
    PolymarketPolicyDataset,
    PolymarketPolicyPrediction,
    PolymarketPolicyTrainingConfig,
    compact_safety_fields,
)


@dataclass(frozen=True, slots=True)
class PolymarketEVDecision:
    """Paper-only EV decision derived from trained P(UP) output."""

    market_id: str
    condition_id: str
    slug: str
    market_family: str
    horizon_ms: int
    decision_ts: int
    action: PolicyAction
    selected_outcome: PolicyOutcome
    estimated_up_probability: float
    confidence: float
    ev_buy_up: float
    ev_buy_down: float
    execution_price: float
    used_price_side: str
    paper_notional: float
    reason_codes: tuple[str, ...]
    trained_model_used: bool = True
    policy_signal_source: str = "trained_model"
    synthetic_fixture_signal_used: bool = False
    paper_only: bool = True
    capital_at_risk: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        if self.action not in ("BUY_UP", "BUY_DOWN", "SELL_UP", "SELL_DOWN", "HOLD", "NO_TRADE"):
            raise ValueError("unsupported EV action")
        if self.selected_outcome not in ("UP", "DOWN", "NO_TRADE"):
            raise ValueError("unsupported selected_outcome")
        if not 0.0 <= self.estimated_up_probability <= 1.0:
            raise ValueError("estimated_up_probability must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        for field_name in ("ev_buy_up", "ev_buy_down", "execution_price", "paper_notional"):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        if self.execution_price < 0.0 or self.paper_notional < 0.0:
            raise ValueError("execution_price and paper_notional must be non-negative")
        if not self.reason_codes:
            raise ValueError("reason_codes are required")
        for field_name, expected in compact_safety_fields().items():
            if getattr(self, field_name) is not expected:
                raise ValueError(f"{field_name} must be {expected}")
        if self.trained_model_used is not True or self.synthetic_fixture_signal_used is not False:
            raise ValueError("EV decisions must use the trained-model path")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def decide_polymarket_ev_action(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    existing_position_up: float = 0.0,
    existing_position_down: float = 0.0,
) -> PolymarketEVDecision:
    """Convert one probability prediction into an EV paper action."""

    features = prediction.features
    up_bid = float(features["up_bid"])
    up_ask = float(features["up_ask"])
    down_bid = float(features["down_bid"])
    down_ask = float(features["down_ask"])
    cost = _execution_cost(features, config)
    probability = prediction.estimated_up_probability
    ev_buy_up = probability - up_ask - cost
    ev_buy_down = (1.0 - probability) - down_ask - cost
    if prediction.confidence < config.min_confidence:
        return _decision(
            prediction=prediction,
            action="NO_TRADE",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("low_confidence", "paper_only_guard"),
        )
    if existing_position_up > 0.0:
        if probability + cost < up_bid:
            return _decision(
                prediction=prediction,
                action="SELL_UP",
                selected_outcome="UP",
                ev_buy_up=ev_buy_up,
                ev_buy_down=ev_buy_down,
                execution_price=up_bid,
                used_price_side="bid",
                paper_notional=existing_position_up * up_bid,
                reason_codes=("sell_up_ev_deteriorated", "bid_price_execution"),
            )
        return _decision(
            prediction=prediction,
            action="HOLD",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("existing_up_position", "hold_threshold_not_met"),
        )
    if existing_position_down > 0.0:
        down_probability = 1.0 - probability
        if down_probability + cost < down_bid:
            return _decision(
                prediction=prediction,
                action="SELL_DOWN",
                selected_outcome="DOWN",
                ev_buy_up=ev_buy_up,
                ev_buy_down=ev_buy_down,
                execution_price=down_bid,
                used_price_side="bid",
                paper_notional=existing_position_down * down_bid,
                reason_codes=("sell_down_ev_deteriorated", "bid_price_execution"),
            )
        return _decision(
            prediction=prediction,
            action="HOLD",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("existing_down_position", "hold_threshold_not_met"),
        )
    if ev_buy_up >= config.ev_threshold and ev_buy_up >= ev_buy_down:
        return _decision(
            prediction=prediction,
            action="BUY_UP",
            selected_outcome="UP",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=up_ask,
            used_price_side="ask",
            paper_notional=_paper_notional(ev_buy_up, config),
            reason_codes=("positive_ev_buy_up", "ask_price_execution"),
        )
    if ev_buy_down >= config.ev_threshold:
        return _decision(
            prediction=prediction,
            action="BUY_DOWN",
            selected_outcome="DOWN",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=down_ask,
            used_price_side="ask",
            paper_notional=_paper_notional(ev_buy_down, config),
            reason_codes=("positive_ev_buy_down", "ask_price_execution"),
        )
    return _decision(
        prediction=prediction,
        action="NO_TRADE",
        selected_outcome="NO_TRADE",
        ev_buy_up=ev_buy_up,
        ev_buy_down=ev_buy_down,
        execution_price=0.0,
        used_price_side="none",
        paper_notional=0.0,
        reason_codes=("ev_threshold_not_met", "paper_only_guard"),
    )


def build_polymarket_ev_decisions(
    *,
    predictions: tuple[PolymarketPolicyPrediction, ...],
    config: PolymarketPolicyTrainingConfig,
) -> tuple[PolymarketEVDecision, ...]:
    """Build sequential EV decisions while tracking paper-only open positions."""

    positions: dict[str, dict[str, float]] = {}
    decisions: list[PolymarketEVDecision] = []
    for prediction in sorted(predictions, key=lambda row: (row.decision_ts, row.market_id)):
        position = positions.setdefault(prediction.market_id, {"UP": 0.0, "DOWN": 0.0})
        decision = decide_polymarket_ev_action(
            prediction=prediction,
            config=config,
            existing_position_up=position["UP"],
            existing_position_down=position["DOWN"],
        )
        if decision.action == "BUY_UP":
            position["UP"] += decision.paper_notional / decision.execution_price
        elif decision.action == "BUY_DOWN":
            position["DOWN"] += decision.paper_notional / decision.execution_price
        elif decision.action == "SELL_UP":
            position["UP"] = 0.0
        elif decision.action == "SELL_DOWN":
            position["DOWN"] = 0.0
        decisions.append(decision)
    return tuple(decisions)


def ev_threshold_report(
    decisions: tuple[PolymarketEVDecision, ...],
    *,
    replay_split: str = "shadow",
) -> dict[str, Any]:
    """Summarize why EV execution traded or skipped."""

    _validate_replay_split(replay_split)
    action_counts = Counter(decision.action for decision in decisions)
    reason_counts: Counter[str] = Counter()
    for decision in decisions:
        reason_counts.update(decision.reason_codes)
    return {
        "schema_version": "bigan-v8-polymarket-ev-threshold-report-v1",
        "replay_split": replay_split,
        "out_of_sample_replay": True,
        "decision_count": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "trained_model_used": True,
        "synthetic_fixture_signal_used": False,
        "policy_signal_source": "trained_model",
        "max_ev_buy_up": max((decision.ev_buy_up for decision in decisions), default=0.0),
        "max_ev_buy_down": max((decision.ev_buy_down for decision in decisions), default=0.0),
        **compact_safety_fields(),
    }


def run_polymarket_policy_replay(
    *,
    dataset: PolymarketPolicyDataset,
    decisions: tuple[PolymarketEVDecision, ...],
    config: PolymarketPolicyTrainingConfig,
    calibration_error: float,
    calibration_split: str = "validation",
    replay_split: str = "shadow",
    prediction_count: int | None = None,
) -> dict[str, Any]:
    """Replay EV decisions through Phase 1 ledger and settlement primitives."""

    _validate_replay_split(calibration_split)
    _validate_replay_split(replay_split)
    replay_examples = _dataset_examples_for_split(dataset, replay_split)
    replay_keys = {(example.market_id, example.decision_ts) for example in replay_examples}
    decision_keys = {(decision.market_id, decision.decision_ts) for decision in decisions}
    if len(decision_keys) != len(decisions):
        raise ValueError("replay decisions must not contain duplicate market_id/decision_ts")
    if not decision_keys.issubset(replay_keys):
        raise ValueError("replay decisions must come from the selected out-of-sample split")
    replay_prediction_count = len(replay_examples) if prediction_count is None else prediction_count
    if replay_prediction_count != len(decisions):
        raise ValueError("replay prediction count must match replay decision count")
    ledgers = {
        market_id: PolymarketPositionLedger(
            market_id=market_id,
            condition_id=metadata["condition_id"],
            slug=metadata["slug"],
            up_token_id=metadata["up_token_id"],
            down_token_id=metadata["down_token_id"],
        )
        for market_id, metadata in dataset.market_metadata.items()
    }
    for decision in sorted(decisions, key=lambda row: (row.decision_ts, row.market_id)):
        ledger = ledgers[decision.market_id]
        _apply_replay_decision(ledger=ledger, decision=decision, config=config)
    settlement_events = []
    settlement_source_counts: Counter[str] = Counter()
    for market_id, ledger in sorted(ledgers.items()):
        metadata = dataset.market_metadata[market_id]
        resolution = dataset.resolution_events[market_id]
        rule = build_btc_updown_resolution_rule(
            market_id=market_id,
            condition_id=metadata["condition_id"],
            slug=metadata["slug"],
            market_family=metadata["market_family"],
            resolution_source=metadata["reference_price_source"],
            candle_open_ts=metadata["market_start_ts"],
            candle_close_ts=metadata["market_end_ts"],
            raw_rule_text=metadata["settlement_rule"],
        )
        payout_up, payout_down, settlement_source = _settlement_payout_vector(
            rule=rule,
            resolution=resolution,
        )
        settlement_source_counts[settlement_source] += 1
        event = ledger.settle(
            ts=metadata["settlement_ts"],
            payout_up=payout_up,
            payout_down=payout_down,
            reason_codes=("phase1_settlement_engine", "paper_settlement", settlement_source),
        )
        settlement_events.append(event.to_dict())
    all_events = [event.to_dict() for ledger in ledgers.values() for event in ledger.events]
    total_realized = sum(ledger.realized_trade_pnl for ledger in ledgers.values())
    total_settlement = sum(ledger.settlement_pnl for ledger in ledgers.values())
    total_complete_set = sum(ledger.complete_set_pnl for ledger in ledgers.values())
    total_fees = sum(ledger.fees for ledger in ledgers.values())
    total_slippage = sum(ledger.slippage for ledger in ledgers.values())
    total_pnl = total_realized + total_settlement + total_complete_set - total_fees - total_slippage
    cumulative = 0.0
    max_drawdown = 0.0
    peak = 0.0
    for event in sorted(all_events, key=lambda row: (row["ts"], row["market_id"], row["action"])):
        cumulative += float(event["cash_delta"]) - float(event["fees"]) - float(event["slippage"])
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
    return {
        "schema_version": "bigan-v8-polymarket-policy-replay-v1",
        "calibration_split": calibration_split,
        "replay_split": replay_split,
        "out_of_sample_replay": True,
        "replay_prediction_count": replay_prediction_count,
        "replay_decision_count": len(decisions),
        "replay_min_ts": min((decision.decision_ts for decision in decisions), default=None),
        "replay_max_ts": max((decision.decision_ts for decision in decisions), default=None),
        "trade_count": sum(decision.action.startswith(("BUY", "SELL")) for decision in decisions),
        "no_trade_count": sum(decision.action == "NO_TRADE" for decision in decisions),
        "settled_position_count": sum(event["qty"] > 0.0 for event in settlement_events),
        "realized_trade_pnl": total_realized,
        "settlement_pnl": total_settlement,
        "complete_set_pnl": total_complete_set,
        "fees": total_fees,
        "slippage": total_slippage,
        "total_polymarket_pnl": total_pnl,
        "max_drawdown": abs(max_drawdown),
        "calibration_error": calibration_error,
        "critical_alert_count": 0,
        "ledger_event_count": len(all_events),
        "settlement_event_count": len(settlement_events),
        "phase1_position_ledger_used": True,
        "phase1_settlement_engine_used": True,
        "settlement_resolution_source_counts": dict(sorted(settlement_source_counts.items())),
        "settlement_engine_components": [
            "PolymarketPositionLedger",
            "build_btc_updown_resolution_rule",
            "resolve_polymarket_rule",
            "verified_outcome_payout_vector",
        ],
        **compact_safety_fields(),
    }


def _validate_replay_split(replay_split: str) -> None:
    if replay_split not in ("validation", "shadow"):
        raise ValueError("replay_split must be validation or shadow")


def _dataset_examples_for_split(
    dataset: PolymarketPolicyDataset,
    replay_split: str,
) -> tuple[Any, ...]:
    if replay_split == "validation":
        return dataset.validation_examples
    if replay_split == "shadow":
        return dataset.shadow_examples
    raise ValueError("replay_split must be validation or shadow")


def _settlement_payout_vector(
    *,
    rule: Any,
    resolution: dict[str, Any],
) -> tuple[float, float, str]:
    reference_price_start = resolution.get("reference_price_start")
    reference_price_end = resolution.get("reference_price_end")
    if reference_price_start is not None and reference_price_end is not None:
        resolved = resolve_polymarket_rule(
            rule,
            reference_price_start=float(reference_price_start),
            reference_price_end=float(reference_price_end),
            resolution_status=resolution["resolution_status"],
        )
        return resolved.payout_up, resolved.payout_down, "reference_price_rule_resolution"

    resolved_outcome = str(resolution.get("resolved_outcome") or "")
    if resolved_outcome not in ("UP", "DOWN", "UNKNOWN_50_50"):
        raise ValueError("resolution requires reference prices or verified resolved_outcome")
    if resolution.get("resolution_status") not in ("normal", "unknown_50_50"):
        raise ValueError("unsupported resolution_status")
    expected_payout_up, expected_payout_down = payout_for_resolved_outcome(
        resolved_outcome,  # type: ignore[arg-type]
    )
    payout_up = float(resolution.get("payout_up", expected_payout_up))
    payout_down = float(resolution.get("payout_down", expected_payout_down))
    if not (
        math.isclose(payout_up, expected_payout_up, abs_tol=1e-12)
        and math.isclose(payout_down, expected_payout_down, abs_tol=1e-12)
    ):
        raise ValueError("resolution payout vector does not match resolved_outcome")
    return payout_up, payout_down, "verified_outcome_payout_vector"


def _apply_replay_decision(
    *,
    ledger: PolymarketPositionLedger,
    decision: PolymarketEVDecision,
    config: PolymarketPolicyTrainingConfig,
) -> None:
    fees = decision.paper_notional * config.fee_rate
    slippage = decision.paper_notional * config.slippage_rate
    if decision.action == "BUY_UP":
        ledger.buy(
            ts=decision.decision_ts,
            outcome="UP",
            qty=decision.paper_notional / decision.execution_price,
            ask_price=decision.execution_price,
            fees=fees,
            slippage=slippage,
            reason_codes=decision.reason_codes,
        )
        return
    if decision.action == "BUY_DOWN":
        ledger.buy(
            ts=decision.decision_ts,
            outcome="DOWN",
            qty=decision.paper_notional / decision.execution_price,
            ask_price=decision.execution_price,
            fees=fees,
            slippage=slippage,
            reason_codes=decision.reason_codes,
        )
        return
    if decision.action == "SELL_UP":
        qty = ledger.position_snapshot()["position_up"]
        if qty > 0.0:
            ledger.sell(
                ts=decision.decision_ts,
                outcome="UP",
                qty=qty,
                bid_price=decision.execution_price,
                fees=fees,
                slippage=slippage,
                reason_codes=decision.reason_codes,
            )
        return
    if decision.action == "SELL_DOWN":
        qty = ledger.position_snapshot()["position_down"]
        if qty > 0.0:
            ledger.sell(
                ts=decision.decision_ts,
                outcome="DOWN",
                qty=qty,
                bid_price=decision.execution_price,
                fees=fees,
                slippage=slippage,
                reason_codes=decision.reason_codes,
            )
        return
    if decision.action == "HOLD":
        ledger.hold(ts=decision.decision_ts, reason_codes=decision.reason_codes)
        return
    ledger.no_trade(ts=decision.decision_ts, reason_codes=decision.reason_codes)


def _decision(
    *,
    prediction: PolymarketPolicyPrediction,
    action: PolicyAction,
    selected_outcome: PolicyOutcome,
    ev_buy_up: float,
    ev_buy_down: float,
    execution_price: float,
    used_price_side: str,
    paper_notional: float,
    reason_codes: tuple[str, ...],
) -> PolymarketEVDecision:
    return PolymarketEVDecision(
        market_id=prediction.market_id,
        condition_id=prediction.condition_id,
        slug=prediction.slug,
        market_family=prediction.market_family,
        horizon_ms=prediction.horizon_ms,
        decision_ts=prediction.decision_ts,
        action=action,
        selected_outcome=selected_outcome,
        estimated_up_probability=prediction.estimated_up_probability,
        confidence=prediction.confidence,
        ev_buy_up=ev_buy_up,
        ev_buy_down=ev_buy_down,
        execution_price=execution_price,
        used_price_side=used_price_side,
        paper_notional=paper_notional,
        reason_codes=(*reason_codes, "trained_model_used", "paper_only_guard"),
    )


def _execution_cost(
    features: dict[str, float],
    config: PolymarketPolicyTrainingConfig,
) -> float:
    liquidity = max(
        1.0,
        float(features.get("up_liquidity_depth", 0.0))
        + float(features.get("down_liquidity_depth", 0.0)),
    )
    return (
        config.fee_rate
        + config.slippage_rate
        + config.liquidity_impact_rate * (config.max_paper_notional / liquidity)
    )


def _paper_notional(ev: float, config: PolymarketPolicyTrainingConfig) -> float:
    return min(config.max_paper_notional, max(0.01, ev * config.max_paper_notional * 5.0))
