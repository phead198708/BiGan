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
    ACTION_VALUE_LABEL_ACTIONS,
    PRIMARY_POLICY_TARGET_ACTION_VALUE,
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
    p_up_auxiliary: float | None = None
    expected_return_by_action: dict[str, float] | None = None
    best_policy_action: str | None = None
    best_action_expected_return: float | None = None
    second_best_action_expected_return: float | None = None
    best_action_margin: float | None = None
    policy_confidence: float | None = None
    entry_policy_action: str | None = None
    intended_exit_policy: str = "none"
    planned_exit_before_ts: int | None = None
    policy_exit_reason: str | None = None
    action_value_head_used: bool = False
    action_value_model_family: str = "resolved_up_probability_only"
    feature_conditioned_action_value_model_used: bool = False
    probability_ev_fallback_used: bool = True
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
        if self.action_value_head_used:
            if not self.action_value_model_family.strip():
                raise ValueError("action_value_model_family is required")
            if self.best_policy_action not in ACTION_VALUE_LABEL_ACTIONS:
                raise ValueError("best_policy_action must be present for action-value decisions")
            if not self.expected_return_by_action:
                raise ValueError("expected_return_by_action is required for action-value decisions")
            missing = set(ACTION_VALUE_LABEL_ACTIONS) - set(self.expected_return_by_action)
            if missing:
                raise ValueError(
                    "expected_return_by_action missing actions: " + ", ".join(sorted(missing))
                )
            for field_name in (
                "best_action_expected_return",
                "second_best_action_expected_return",
                "best_action_margin",
                "policy_confidence",
            ):
                value = getattr(self, field_name)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"{field_name} must be finite for action-value decisions")
        if self.intended_exit_policy not in ("none", "hold_to_settlement", "sell_before_close"):
            raise ValueError("unsupported intended_exit_policy")
        if self.entry_policy_action is not None and self.entry_policy_action not in ACTION_VALUE_LABEL_ACTIONS:
            raise ValueError("entry_policy_action must be a supported label action")
        if self.intended_exit_policy == "sell_before_close" and self.planned_exit_before_ts is None:
            raise ValueError("sell_before_close decisions require planned_exit_before_ts")
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
    existing_up_entry_policy_action: str | None = None,
    existing_down_entry_policy_action: str | None = None,
    existing_up_intended_exit_policy: str = "none",
    existing_down_intended_exit_policy: str = "none",
    existing_up_planned_exit_before_ts: int | None = None,
    existing_down_planned_exit_before_ts: int | None = None,
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
    if _planned_exit_due(
        prediction=prediction,
        existing_position=existing_position_up,
        intended_exit_policy=existing_up_intended_exit_policy,
        planned_exit_before_ts=existing_up_planned_exit_before_ts,
    ):
        return _decision(
            prediction=prediction,
            action="SELL_UP",
            selected_outcome="UP",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=up_bid,
            used_price_side="bid",
            paper_notional=existing_position_up * up_bid,
            reason_codes=("planned_sell_before_close_exit", "bid_price_execution"),
            entry_policy_action=existing_up_entry_policy_action,
            intended_exit_policy="sell_before_close",
            planned_exit_before_ts=existing_up_planned_exit_before_ts,
            policy_exit_reason="planned_sell_before_close",
            action_value_head_used=prediction.action_value_head_enabled,
            probability_ev_fallback_used=False,
        )
    if _planned_exit_due(
        prediction=prediction,
        existing_position=existing_position_down,
        intended_exit_policy=existing_down_intended_exit_policy,
        planned_exit_before_ts=existing_down_planned_exit_before_ts,
    ):
        return _decision(
            prediction=prediction,
            action="SELL_DOWN",
            selected_outcome="DOWN",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=down_bid,
            used_price_side="bid",
            paper_notional=existing_position_down * down_bid,
            reason_codes=("planned_sell_before_close_exit", "bid_price_execution"),
            entry_policy_action=existing_down_entry_policy_action,
            intended_exit_policy="sell_before_close",
            planned_exit_before_ts=existing_down_planned_exit_before_ts,
            policy_exit_reason="planned_sell_before_close",
            action_value_head_used=prediction.action_value_head_enabled,
            probability_ev_fallback_used=False,
        )
    if (
        prediction.action_value_head_enabled
        and prediction.expected_return_by_action
        and existing_position_up <= 0.0
        and existing_position_down <= 0.0
    ):
        return _action_value_decision(
            prediction=prediction,
            config=config,
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            up_ask=up_ask,
            down_ask=down_ask,
        )
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
                entry_policy_action=existing_up_entry_policy_action,
                intended_exit_policy=existing_up_intended_exit_policy,
                planned_exit_before_ts=existing_up_planned_exit_before_ts,
                policy_exit_reason="probability_ev_deteriorated",
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
            entry_policy_action=existing_up_entry_policy_action,
            intended_exit_policy=existing_up_intended_exit_policy,
            planned_exit_before_ts=existing_up_planned_exit_before_ts,
            policy_exit_reason="hold_until_exit_condition",
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
                entry_policy_action=existing_down_entry_policy_action,
                intended_exit_policy=existing_down_intended_exit_policy,
                planned_exit_before_ts=existing_down_planned_exit_before_ts,
                policy_exit_reason="probability_ev_deteriorated",
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
            entry_policy_action=existing_down_entry_policy_action,
            intended_exit_policy=existing_down_intended_exit_policy,
            planned_exit_before_ts=existing_down_planned_exit_before_ts,
            policy_exit_reason="hold_until_exit_condition",
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

    engine = StatefulPolymarketDecisionEngine(config=config)
    return engine.decide_many(predictions)


class StatefulPolymarketDecisionEngine:
    """Stateful paper decision lifecycle shared by batch replay and live streaming."""

    def __init__(self, *, config: PolymarketPolicyTrainingConfig) -> None:
        self._config = config
        self._positions: dict[str, dict[str, Any]] = {}
        self._decisions: list[PolymarketEVDecision] = []
        self._seen_keys: set[tuple[int, str]] = set()
        self._last_key: tuple[int, str] | None = None

    @property
    def decisions(self) -> tuple[PolymarketEVDecision, ...]:
        return tuple(self._decisions)

    def decide_many(
        self,
        predictions: tuple[PolymarketPolicyPrediction, ...],
    ) -> tuple[PolymarketEVDecision, ...]:
        decisions = []
        for prediction in sorted(predictions, key=lambda row: (row.decision_ts, row.market_id)):
            decisions.append(self.decide(prediction))
        return tuple(decisions)

    def decide(self, prediction: PolymarketPolicyPrediction) -> PolymarketEVDecision:
        key = (int(prediction.decision_ts), str(prediction.market_id))
        if key in self._seen_keys:
            raise ValueError("duplicate_prediction_decision_key")
        if self._last_key is not None and key < self._last_key:
            raise ValueError("decision_state_out_of_order")
        position = self._positions.setdefault(
            prediction.market_id,
            _empty_position_state(),
        )
        decision = decide_polymarket_ev_action(
            prediction=prediction,
            config=self._config,
            existing_position_up=float(position["UP"]),
            existing_position_down=float(position["DOWN"]),
            existing_up_entry_policy_action=position["UP_entry_policy_action"],
            existing_down_entry_policy_action=position["DOWN_entry_policy_action"],
            existing_up_intended_exit_policy=str(position["UP_intended_exit_policy"]),
            existing_down_intended_exit_policy=str(position["DOWN_intended_exit_policy"]),
            existing_up_planned_exit_before_ts=position["UP_planned_exit_before_ts"],
            existing_down_planned_exit_before_ts=position["DOWN_planned_exit_before_ts"],
        )
        self._apply_position_state(position=position, decision=decision)
        self._seen_keys.add(key)
        self._last_key = key
        self._decisions.append(decision)
        return decision

    @staticmethod
    def _apply_position_state(
        *,
        position: dict[str, Any],
        decision: PolymarketEVDecision,
    ) -> None:
        if decision.action == "BUY_UP":
            _open_position_state(position, "UP", decision)
        elif decision.action == "BUY_DOWN":
            _open_position_state(position, "DOWN", decision)
        elif decision.action == "SELL_UP":
            _close_position_state(position, "UP")
        elif decision.action == "SELL_DOWN":
            _close_position_state(position, "DOWN")


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
        "primary_policy_target": (
            PRIMARY_POLICY_TARGET_ACTION_VALUE
            if any(decision.action_value_head_used for decision in decisions)
            else "resolved_up_probability_only"
        ),
        "action_value_model_family": _action_value_model_family(decisions),
        "feature_conditioned_action_value_model_used": any(
            decision.feature_conditioned_action_value_model_used for decision in decisions
        ),
        "action_value_head_enabled": any(decision.action_value_head_used for decision in decisions),
        "action_value_decision_count": sum(
            decision.action_value_head_used for decision in decisions
        ),
        "probability_ev_fallback_decision_count": sum(
            decision.probability_ev_fallback_used for decision in decisions
        ),
        "decision_count": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "intended_exit_policy_counts": dict(
            sorted(Counter(decision.intended_exit_policy for decision in decisions).items())
        ),
        "planned_sell_before_close_exit_count": sum(
            "planned_sell_before_close_exit" in decision.reason_codes
            for decision in decisions
        ),
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
        "outcome_calibration_error": calibration_error,
        "action_value_policy_metrics": _action_value_policy_metrics(decisions),
        "intended_exit_policy_counts": dict(
            sorted(Counter(decision.intended_exit_policy for decision in decisions).items())
        ),
        "planned_sell_before_close_exit_count": sum(
            "planned_sell_before_close_exit" in decision.reason_codes
            for decision in decisions
        ),
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
    entry_policy_action: str | None = None,
    intended_exit_policy: str = "none",
    planned_exit_before_ts: int | None = None,
    policy_exit_reason: str | None = None,
    action_value_head_used: bool = False,
    probability_ev_fallback_used: bool = True,
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
        p_up_auxiliary=prediction.p_up_auxiliary or prediction.estimated_up_probability,
        expected_return_by_action=dict(prediction.expected_return_by_action),
        best_policy_action=prediction.best_policy_action,
        best_action_expected_return=prediction.best_action_expected_return,
        second_best_action_expected_return=prediction.second_best_action_expected_return,
        best_action_margin=prediction.best_action_margin,
        policy_confidence=prediction.policy_confidence,
        entry_policy_action=entry_policy_action,
        intended_exit_policy=intended_exit_policy,
        planned_exit_before_ts=planned_exit_before_ts,
        policy_exit_reason=policy_exit_reason,
        action_value_head_used=action_value_head_used,
        action_value_model_family=prediction.action_value_model_family,
        feature_conditioned_action_value_model_used=(
            prediction.feature_conditioned_action_value_model_enabled
            and action_value_head_used
        ),
        probability_ev_fallback_used=probability_ev_fallback_used,
    )


def _action_value_decision(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    ev_buy_up: float,
    ev_buy_down: float,
    up_ask: float,
    down_ask: float,
) -> PolymarketEVDecision:
    confidence = (
        prediction.policy_confidence
        if prediction.policy_confidence is not None
        else prediction.confidence
    )
    best_action = str(prediction.best_policy_action)
    best_return = float(prediction.best_action_expected_return or 0.0)
    if confidence < config.min_confidence:
        return _decision(
            prediction=prediction,
            action="NO_TRADE",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("low_policy_confidence", "action_value_head_used"),
            action_value_head_used=True,
            probability_ev_fallback_used=False,
        )
    if best_action == "NO_TRADE":
        return _decision(
            prediction=prediction,
            action="NO_TRADE",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("action_value_no_trade_selected", "action_value_head_used"),
            action_value_head_used=True,
            probability_ev_fallback_used=False,
        )
    if best_return < config.ev_threshold:
        return _decision(
            prediction=prediction,
            action="NO_TRADE",
            selected_outcome="NO_TRADE",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=0.0,
            used_price_side="none",
            paper_notional=0.0,
            reason_codes=("action_value_threshold_not_met", "action_value_head_used"),
            action_value_head_used=True,
            probability_ev_fallback_used=False,
        )
    if best_action.startswith("BUY_UP_"):
        intended_exit_policy = _intended_exit_policy(best_action)
        return _decision(
            prediction=prediction,
            action="BUY_UP",
            selected_outcome="UP",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=up_ask,
            used_price_side="ask",
            paper_notional=_paper_notional(best_return, config),
            reason_codes=(
                "positive_action_value_buy_up",
                _policy_reason(best_action),
                "ask_price_execution",
                "action_value_head_used",
            ),
            entry_policy_action=best_action,
            intended_exit_policy=intended_exit_policy,
            planned_exit_before_ts=_planned_exit_before_ts(
                prediction=prediction,
                config=config,
                intended_exit_policy=intended_exit_policy,
            ),
            policy_exit_reason=intended_exit_policy,
            action_value_head_used=True,
            probability_ev_fallback_used=False,
        )
    if best_action.startswith("BUY_DOWN_"):
        intended_exit_policy = _intended_exit_policy(best_action)
        return _decision(
            prediction=prediction,
            action="BUY_DOWN",
            selected_outcome="DOWN",
            ev_buy_up=ev_buy_up,
            ev_buy_down=ev_buy_down,
            execution_price=down_ask,
            used_price_side="ask",
            paper_notional=_paper_notional(best_return, config),
            reason_codes=(
                "positive_action_value_buy_down",
                _policy_reason(best_action),
                "ask_price_execution",
                "action_value_head_used",
            ),
            entry_policy_action=best_action,
            intended_exit_policy=intended_exit_policy,
            planned_exit_before_ts=_planned_exit_before_ts(
                prediction=prediction,
                config=config,
                intended_exit_policy=intended_exit_policy,
            ),
            policy_exit_reason=intended_exit_policy,
            action_value_head_used=True,
            probability_ev_fallback_used=False,
        )
    raise ValueError(f"unsupported best_policy_action: {best_action}")


def _policy_reason(best_action: str) -> str:
    return "policy_" + best_action.lower()


def _intended_exit_policy(best_action: str) -> str:
    if best_action.endswith("_SELL_BEFORE_CLOSE"):
        return "sell_before_close"
    if best_action.endswith("_HOLD_TO_SETTLEMENT"):
        return "hold_to_settlement"
    return "none"


def _planned_exit_before_ts(
    *,
    prediction: PolymarketPolicyPrediction,
    config: PolymarketPolicyTrainingConfig,
    intended_exit_policy: str,
) -> int | None:
    if intended_exit_policy != "sell_before_close":
        return None
    time_to_close_ms = max(
        0,
        int(float(prediction.features.get("time_to_close_seconds", 0.0)) * 1000),
    )
    market_end_ts = prediction.decision_ts + time_to_close_ms
    exit_buffer_ms = int(config.sell_before_close_exit_buffer_seconds * 1000)
    return max(prediction.decision_ts, market_end_ts - exit_buffer_ms)


def _planned_exit_due(
    *,
    prediction: PolymarketPolicyPrediction,
    existing_position: float,
    intended_exit_policy: str,
    planned_exit_before_ts: int | None,
) -> bool:
    return (
        existing_position > 0.0
        and intended_exit_policy == "sell_before_close"
        and planned_exit_before_ts is not None
        and prediction.decision_ts >= planned_exit_before_ts
    )


def _empty_position_state() -> dict[str, Any]:
    return {
        "UP": 0.0,
        "DOWN": 0.0,
        "UP_entry_policy_action": None,
        "DOWN_entry_policy_action": None,
        "UP_intended_exit_policy": "none",
        "DOWN_intended_exit_policy": "none",
        "UP_planned_exit_before_ts": None,
        "DOWN_planned_exit_before_ts": None,
    }


def _open_position_state(
    position: dict[str, Any],
    outcome: str,
    decision: PolymarketEVDecision,
) -> None:
    position[outcome] = float(position[outcome]) + decision.paper_notional / decision.execution_price
    position[f"{outcome}_entry_policy_action"] = decision.entry_policy_action
    position[f"{outcome}_intended_exit_policy"] = decision.intended_exit_policy
    position[f"{outcome}_planned_exit_before_ts"] = decision.planned_exit_before_ts


def _close_position_state(position: dict[str, Any], outcome: str) -> None:
    position[outcome] = 0.0
    position[f"{outcome}_entry_policy_action"] = None
    position[f"{outcome}_intended_exit_policy"] = "none"
    position[f"{outcome}_planned_exit_before_ts"] = None


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


def _action_value_policy_metrics(
    decisions: tuple[PolymarketEVDecision, ...],
) -> dict[str, Any]:
    action_value_decisions = tuple(
        decision for decision in decisions if decision.action_value_head_used
    )
    best_returns = [
        float(decision.best_action_expected_return)
        for decision in action_value_decisions
        if decision.best_action_expected_return is not None
    ]
    margins = [
        float(decision.best_action_margin)
        for decision in action_value_decisions
        if decision.best_action_margin is not None
    ]
    return {
        "schema_version": "bigan-v8-polymarket-action-value-policy-metrics-v1",
        "primary_policy_target": PRIMARY_POLICY_TARGET_ACTION_VALUE,
        "sample_count": len(action_value_decisions),
        "action_value_head_used_count": len(action_value_decisions),
        "mean_best_action_expected_return": _mean(best_returns),
        "mean_best_action_margin": _mean(margins),
        "best_policy_action_counts": dict(
            sorted(Counter(decision.best_policy_action for decision in action_value_decisions).items())
        ),
        "action_value_model_family": _action_value_model_family(decisions),
        "feature_conditioned_action_value_model_used": any(
            decision.feature_conditioned_action_value_model_used for decision in action_value_decisions
        ),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _action_value_model_family(decisions: tuple[PolymarketEVDecision, ...]) -> str:
    families = {
        decision.action_value_model_family
        for decision in decisions
        if decision.action_value_head_used
    }
    if not families:
        return "resolved_up_probability_only"
    if len(families) != 1:
        return "mixed_action_value_model_family"
    return next(iter(families))
