"""Reconcile stale execution positions from account history or settlement."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .cashflow import PolymarketCashFlow, reconcile_cash_flows
from .position_manager import Position, PositionManager, PositionSide

StaleReconcileAction = Literal[
    "unchanged",
    "closed_from_sell",
    "settled_from_redeem",
    "settled_from_provider",
]


@dataclass(frozen=True, slots=True)
class StalePositionReconciliation:
    """Outcome from reconciling one stale open execution row."""

    event_id: str
    round_slug: str
    prior_status: str
    new_status: str
    action: StaleReconcileAction
    exit_price: float | None
    settlement_result: PositionSide | None
    account_cash_pnl: float | None
    theoretical_pnl: float | None


def reconcile_stale_open_positions(
    position_manager: PositionManager,
    cash_flows: list[PolymarketCashFlow],
    *,
    settlement_results: dict[str, str] | None = None,
) -> list[StalePositionReconciliation]:
    """Close or settle DB rows still marked open after account history shows resolution."""

    open_positions = position_manager.get_all_open()
    if not open_positions:
        return []

    reconciled = reconcile_cash_flows(open_positions, cash_flows)
    flows_by_event = {
        record.event_id: _flows_from_json(record.cash_flows_json)
        for record in reconciled
        if record.match_status == "matched"
    }
    results: list[StalePositionReconciliation] = []
    provider = settlement_results or {}
    for position in open_positions:
        flows = flows_by_event.get(position.event_id, [])
        result = _reconcile_one(
            position_manager,
            position,
            flows,
            settlement_result=_normalise_side(provider.get(position.event_id)),
        )
        if result is not None:
            results.append(result)
    return results


def _reconcile_one(
    position_manager: PositionManager,
    position: Position,
    flows: list[dict[str, Any]],
    *,
    settlement_result: PositionSide | None,
) -> StalePositionReconciliation | None:
    if position.status != "open":
        return None

    round_slug = _round_slug_from_position(position)
    sells = [flow for flow in flows if flow.get("action") == "SELL"]
    redeems = [flow for flow in flows if flow.get("action") == "REDEEM"]
    account_pnl = sum(float(flow.get("cash_delta", 0.0)) for flow in flows) if flows else None

    if sells:
        last = sells[-1]
        token_amount = float(last.get("token_amount") or 0.0)
        usdc_amount = float(last.get("usdc_amount") or 0.0)
        exit_price = usdc_amount / token_amount if token_amount > 0 else 0.0
        leg_ts = int(last.get("timestamp") or 0)
        updated = position_manager.close_position(
            position.event_id,
            exit_price,
            exit_time=leg_ts * 1000 if leg_ts else None,
        )
        return StalePositionReconciliation(
            event_id=position.event_id,
            round_slug=round_slug,
            prior_status="open",
            new_status=updated.status,
            action="closed_from_sell",
            exit_price=exit_price,
            settlement_result=None,
            account_cash_pnl=account_pnl,
            theoretical_pnl=updated.realized_pnl,
        )

    if redeems and not sells:
        updated = position_manager.settle_position(position.event_id, position.side)
        return StalePositionReconciliation(
            event_id=position.event_id,
            round_slug=round_slug,
            prior_status="open",
            new_status=updated.status,
            action="settled_from_redeem",
            exit_price=updated.exit_price,
            settlement_result=updated.settlement_result,
            account_cash_pnl=account_pnl,
            theoretical_pnl=updated.realized_pnl,
        )

    if settlement_result is not None:
        updated = position_manager.settle_position(position.event_id, settlement_result)
        return StalePositionReconciliation(
            event_id=position.event_id,
            round_slug=round_slug,
            prior_status="open",
            new_status=updated.status,
            action="settled_from_provider",
            exit_price=updated.exit_price,
            settlement_result=updated.settlement_result,
            account_cash_pnl=account_pnl,
            theoretical_pnl=updated.realized_pnl,
        )

    return StalePositionReconciliation(
        event_id=position.event_id,
        round_slug=round_slug,
        prior_status="open",
        new_status="open",
        action="unchanged",
        exit_price=None,
        settlement_result=None,
        account_cash_pnl=account_pnl,
        theoretical_pnl=None,
    )


def _flows_from_json(cash_flows_json: str) -> list[dict[str, Any]]:
    import json

    payload = json.loads(cash_flows_json)
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload]


def _round_slug_from_position(position: Position) -> str:
    match = re.search(r"(btc-updown-15m-\d+)", position.event_id)
    if match:
        return match.group(1)
    parts = position.symbol.split(":")
    if len(parts) >= 2:
        return parts[-2]
    raise ValueError(f"cannot derive round slug from position {position.event_id}")


def _normalise_side(value: str | None) -> PositionSide | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"UP", "DOWN"}:
        return text  # type: ignore[return-value]
    return None
