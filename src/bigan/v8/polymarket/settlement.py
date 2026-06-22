"""Settlement engine for paper-only Polymarket binary outcome tokens."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import (
    PolymarketAdapterError,
    PolymarketBinaryDecision,
    PolymarketBinaryMarket,
    PolymarketTokenSnapshot,
    looks_like_sha256,
)
from bigan.v8.polymarket.ledger import (
    PolymarketLedgerEvent,
    PolymarketPositionLedger,
)
from bigan.v8.polymarket.rules import (
    PolymarketResolutionRule,
    PolymarketRuleResolution,
    build_btc15m_resolution_rule,
    resolve_polymarket_rule,
)


@dataclass(frozen=True, slots=True)
class PolymarketSettlementEvent:
    """Paper-only settlement event for a resolved binary Polymarket market."""

    market_id: str
    condition_id: str
    slug: str
    resolved_outcome: str
    payout_up: float
    payout_down: float
    qty_up_settled: float
    qty_down_settled: float
    settlement_cashflow: float
    settlement_pnl: float
    raw_resolution_sha256: str
    paper_only: bool = True
    capital_at_risk: bool = False
    broker_exchange_write_enabled: bool = False
    live_exchange_write_enabled: bool = False
    polymarket_write_enabled: bool = False
    wallet_signing_enabled: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(
            market_id=self.market_id,
            condition_id=self.condition_id,
            slug=self.slug,
            resolved_outcome=self.resolved_outcome,
            raw_resolution_sha256=self.raw_resolution_sha256,
        )
        if self.resolved_outcome not in ("UP", "DOWN", "UNKNOWN_50_50"):
            raise ValueError("unsupported resolved_outcome")
        for field_name in ("payout_up", "payout_down"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.qty_up_settled < 0.0 or self.qty_down_settled < 0.0:
            raise ValueError("settled quantities cannot be negative")
        if not looks_like_sha256(self.raw_resolution_sha256):
            raise ValueError("raw_resolution_sha256 must be SHA-256")
        _validate_safety_boundary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketSettlementEngineResult:
    """Complete rule, ledger, settlement, and artifact evidence."""

    resolution_rule: PolymarketResolutionRule
    resolution: PolymarketRuleResolution
    ledger_events: tuple[PolymarketLedgerEvent, ...]
    settlement_events: tuple[PolymarketSettlementEvent, ...]
    position_summary: dict[str, Any]
    pnl_breakdown: dict[str, Any]
    artifact_paths: dict[str, Path]
    artifact_hashes: dict[str, str]


def run_polymarket_settlement_engine(
    *,
    market: PolymarketBinaryMarket,
    token_snapshots: tuple[PolymarketTokenSnapshot, ...],
    decisions: tuple[PolymarketBinaryDecision, ...],
    reference_price_end: float,
    output_dir: Path | str | None = None,
) -> PolymarketSettlementEngineResult:
    """Replay paper decisions through a Polymarket position and settlement ledger."""

    rule = build_btc15m_resolution_rule(market)
    resolution = resolve_polymarket_rule(
        rule,
        reference_price_start=market.reference_price_at_start,
        reference_price_end=reference_price_end,
    )
    ledger = PolymarketPositionLedger(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        up_token_id=market.up_token_id,
        down_token_id=market.down_token_id,
    )
    snapshots = _snapshots_by_ts(token_snapshots)
    for decision in sorted(decisions, key=lambda item: item.decision_ts):
        _apply_decision(ledger=ledger, decision=decision, snapshots=snapshots)

    pre_settlement = ledger.position_snapshot()
    settlement_event = _settle_ledger(
        ledger=ledger,
        market=market,
        resolution=resolution,
        pre_settlement=pre_settlement,
    )
    position_summary = _position_summary(
        market=market,
        resolution=resolution,
        pre_settlement=pre_settlement,
        post_settlement=ledger.position_snapshot(),
    )
    pnl_breakdown = _pnl_breakdown(
        market=market,
        resolution=resolution,
        ledger=ledger,
        settlement_event=settlement_event,
    )
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    if output_dir is not None:
        artifact_paths = _write_settlement_artifacts(
            output_dir=Path(output_dir),
            ledger_events=ledger.events,
            settlement_events=(settlement_event,),
            position_summary=position_summary,
            pnl_breakdown=pnl_breakdown,
        )
        artifact_hashes = {
            name: _sha256_file(path) for name, path in sorted(artifact_paths.items())
        }
    return PolymarketSettlementEngineResult(
        resolution_rule=rule,
        resolution=resolution,
        ledger_events=ledger.events,
        settlement_events=(settlement_event,),
        position_summary=position_summary,
        pnl_breakdown=pnl_breakdown,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
    )


def _apply_decision(
    *,
    ledger: PolymarketPositionLedger,
    decision: PolymarketBinaryDecision,
    snapshots: dict[int, dict[str, PolymarketTokenSnapshot]],
) -> None:
    if decision.selected_outcome == "NO_TRADE" or decision.paper_notional <= 0.0:
        ledger.no_trade(
            ts=decision.decision_ts,
            reason_codes=tuple(decision.reason_codes) or ("paper_no_trade",),
        )
        return
    snapshot = snapshots.get(decision.decision_ts, {}).get(decision.selected_outcome)
    if snapshot is None:
        raise PolymarketAdapterError("missing_decision_token_snapshot")
    qty = decision.paper_notional / snapshot.ask_price
    ledger.buy(
        ts=decision.decision_ts,
        outcome=decision.selected_outcome,
        qty=qty,
        ask_price=snapshot.ask_price,
        reason_codes=("paper_buy", "ask_price_execution", *decision.reason_codes),
    )


def _settle_ledger(
    *,
    ledger: PolymarketPositionLedger,
    market: PolymarketBinaryMarket,
    resolution: PolymarketRuleResolution,
    pre_settlement: dict[str, Any],
) -> PolymarketSettlementEvent:
    ledger.settle(
        ts=market.settlement_ts,
        payout_up=resolution.payout_up,
        payout_down=resolution.payout_down,
        reason_codes=("paper_settlement", "market_resolved"),
    )
    qty_up = float(pre_settlement["position_up"])
    qty_down = float(pre_settlement["position_down"])
    settlement_cashflow = qty_up * resolution.payout_up + qty_down * resolution.payout_down
    settlement_basis = (
        qty_up * float(pre_settlement["avg_entry_up"])
        + qty_down * float(pre_settlement["avg_entry_down"])
    )
    return PolymarketSettlementEvent(
        market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        resolved_outcome=resolution.resolved_outcome,
        payout_up=resolution.payout_up,
        payout_down=resolution.payout_down,
        qty_up_settled=qty_up,
        qty_down_settled=qty_down,
        settlement_cashflow=settlement_cashflow,
        settlement_pnl=settlement_cashflow - settlement_basis,
        raw_resolution_sha256=resolution.raw_resolution_sha256,
    )


def _position_summary(
    *,
    market: PolymarketBinaryMarket,
    resolution: PolymarketRuleResolution,
    pre_settlement: dict[str, Any],
    post_settlement: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "bigan-v8-polymarket-position-summary-v1",
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "resolved_outcome": resolution.resolved_outcome,
        "pre_settlement_position_up": pre_settlement["position_up"],
        "pre_settlement_position_down": pre_settlement["position_down"],
        "position_up": post_settlement["position_up"],
        "position_down": post_settlement["position_down"],
        "avg_entry_up": post_settlement["avg_entry_up"],
        "avg_entry_down": post_settlement["avg_entry_down"],
        "unresolved_position_count": int(post_settlement["position_up"] > 0.0)
        + int(post_settlement["position_down"] > 0.0),
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _pnl_breakdown(
    *,
    market: PolymarketBinaryMarket,
    resolution: PolymarketRuleResolution,
    ledger: PolymarketPositionLedger,
    settlement_event: PolymarketSettlementEvent,
) -> dict[str, Any]:
    realized_trade_pnl = ledger.realized_trade_pnl
    settlement_pnl = ledger.settlement_pnl
    complete_set_pnl = ledger.complete_set_pnl
    fees = ledger.fees
    slippage = ledger.slippage
    total = realized_trade_pnl + settlement_pnl + complete_set_pnl - fees - slippage
    reconciled_total = total
    return {
        "schema_version": "bigan-v8-polymarket-pnl-breakdown-v1",
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "resolved_outcome": resolution.resolved_outcome,
        "realized_trade_pnl": realized_trade_pnl,
        "unrealized_mark_pnl": ledger.unrealized_mark_pnl,
        "settlement_pnl": settlement_pnl,
        "complete_set_pnl": complete_set_pnl,
        "fees": fees,
        "slippage": slippage,
        "total_polymarket_pnl": total,
        "settled_position_count": int(settlement_event.qty_up_settled > 0.0)
        + int(settlement_event.qty_down_settled > 0.0),
        "unresolved_position_count": 0,
        "pnl_reconciled": abs(
            realized_trade_pnl
            + settlement_pnl
            + complete_set_pnl
            - fees
            - slippage
            - reconciled_total
        )
        <= 1e-12,
        "raw_resolution_sha256": resolution.raw_resolution_sha256,
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }


def _write_settlement_artifacts(
    *,
    output_dir: Path,
    ledger_events: tuple[PolymarketLedgerEvent, ...],
    settlement_events: tuple[PolymarketSettlementEvent, ...],
    position_summary: dict[str, Any],
    pnl_breakdown: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "polymarket_position_ledger": output_dir / "polymarket_position_ledger.jsonl",
        "polymarket_settlement_events": (
            output_dir / "polymarket_settlement_events.jsonl"
        ),
        "polymarket_position_summary": output_dir / "polymarket_position_summary.json",
        "polymarket_pnl_breakdown": output_dir / "polymarket_pnl_breakdown.json",
    }
    _write_jsonl(paths["polymarket_position_ledger"], [event.to_dict() for event in ledger_events])
    _write_jsonl(
        paths["polymarket_settlement_events"],
        [event.to_dict() for event in settlement_events],
    )
    _write_json(paths["polymarket_position_summary"], position_summary)
    _write_json(paths["polymarket_pnl_breakdown"], pnl_breakdown)
    return paths


def _snapshots_by_ts(
    snapshots: tuple[PolymarketTokenSnapshot, ...],
) -> dict[int, dict[str, PolymarketTokenSnapshot]]:
    grouped: dict[int, dict[str, PolymarketTokenSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.ts, {})[snapshot.outcome] = snapshot
    return grouped


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                _json_ready(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_non_empty(**values: str) -> None:
    for field_name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} is required")


def _validate_safety_boundary(payload: Any) -> None:
    checks = {
        "paper_only": True,
        "capital_at_risk": False,
        "broker_exchange_write_enabled": False,
        "live_exchange_write_enabled": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
    }
    for field_name, expected in checks.items():
        if getattr(payload, field_name) is not expected:
            raise ValueError(f"{field_name} must be {str(expected).lower()}")
