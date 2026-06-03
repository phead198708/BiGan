"""Terminal-friendly champion signal views from prediction_events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .market_quality import tradable_market_implied_probability


@dataclass(frozen=True, slots=True)
class ChampionSignalRow:
    """One terminal-displayable signal derived from a prediction event."""

    created_at: int
    ts: int
    event_id: str
    model_version: str
    source_symbol: str
    canonical_symbol: str
    outcome_side: str
    prob_up_15m: float
    market_implied_prob: float | None
    edge: float | None
    signal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PositionSignalState:
    """Minimal long/flat state for paper signal routing."""

    position_open: bool = False
    outcome_side: str | None = None
    entry_price: float | None = None
    entry_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class PositionSignalDecision:
    """One state-machine decision and the resulting state."""

    signal: str
    state: PositionSignalState
    reason: str | None = None


def evaluate_position_signal(
    state: PositionSignalState,
    *,
    edge: float | None,
    market_implied_prob: float | None,
    outcome_side: str,
    event_id: str,
    edge_threshold: float = 0.30,
    exit_edge_threshold: float = 0.05,
    profit_target: float = 0.15,
    current_ts: int | None = None,
    round_end_ts: int | None = None,
    near_round_end_ms: int = 60_000,
) -> PositionSignalDecision:
    """Advance the paper signal state for BUY_UP/BUY_DOWN/HOLD/SELL.

    SELL is emitted only when a position is already open; a low edge while flat
    remains HOLD/NO_SIGNAL instead of becoming an impossible exit.
    """

    side = _normalise_outcome_side_value(outcome_side)
    if state.position_open:
        if state.outcome_side is not None and side != state.outcome_side:
            return PositionSignalDecision(signal="HOLD", state=state, reason="opposite_side")
        unrealized_pnl = _unrealized_pnl(
            market_implied_prob=market_implied_prob,
            entry_price=state.entry_price,
        )
        if unrealized_pnl is not None and unrealized_pnl >= profit_target:
            return PositionSignalDecision(
                signal="SELL",
                state=PositionSignalState(),
                reason="profit_target",
            )
        if (
            edge is not None
            and edge <= exit_edge_threshold
            and market_implied_prob is not None
        ):
            return PositionSignalDecision(
                signal="SELL",
                state=PositionSignalState(),
                reason="edge_reversal",
            )
        if _should_lock_round_end_profit(
            unrealized_pnl=unrealized_pnl,
            current_ts=current_ts,
            round_end_ts=round_end_ts,
            near_round_end_ms=near_round_end_ms,
        ):
            return PositionSignalDecision(
                signal="SELL",
                state=PositionSignalState(),
                reason="round_end_profit",
            )
        return PositionSignalDecision(signal="HOLD", state=state, reason="open_position")

    if edge is None:
        return PositionSignalDecision(signal="NO_SIGNAL", state=state, reason="missing_edge")
    if edge < edge_threshold or market_implied_prob is None:
        return PositionSignalDecision(signal="HOLD", state=state, reason="below_entry_edge")

    signal = f"BUY_{side}" if side in {"UP", "DOWN"} else "BUY_TOKEN"
    return PositionSignalDecision(
        signal=signal,
        state=PositionSignalState(
            position_open=True,
            outcome_side=side,
            entry_price=market_implied_prob,
            entry_event_id=event_id,
        ),
        reason="entry_edge",
    )


def _unrealized_pnl(
    *,
    market_implied_prob: float | None,
    entry_price: float | None,
) -> float | None:
    if market_implied_prob is None or entry_price is None:
        return None
    return market_implied_prob - entry_price


def _should_lock_round_end_profit(
    *,
    unrealized_pnl: float | None,
    current_ts: int | None,
    round_end_ts: int | None,
    near_round_end_ms: int,
) -> bool:
    if unrealized_pnl is None or unrealized_pnl <= 0.0:
        return False
    if current_ts is None or round_end_ts is None:
        return False
    return 0 <= round_end_ts - current_ts <= near_round_end_ms


def read_recent_signal_rows(
    db_path: Path | str,
    *,
    model_version: str,
    edge_threshold: float = 0.30,
    outcome_side: str | None = None,
    limit: int = 20,
) -> tuple[ChampionSignalRow, ...]:
    """Read recent signal rows without holding a long-lived DuckDB lock."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return ()
    with _connect_read_only(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT created_at, ts, event_id, model_version, prob_up_15m,
                   feature_snapshot_json
            FROM prediction_events
            WHERE model_version = ?
              {_outcome_side_sql(outcome_side)}
            ORDER BY created_at DESC, event_id DESC
            LIMIT ?
            """,
            [model_version, *_outcome_side_params(outcome_side), limit],
        ).fetchall()
    return tuple(
        _signal_from_event_row(row, edge_threshold=edge_threshold)
        for row in reversed(rows)
    )


def read_signal_rows_after(
    db_path: Path | str,
    *,
    model_version: str,
    after_created_at: int,
    after_event_id: str,
    edge_threshold: float = 0.30,
    outcome_side: str | None = None,
    limit: int = 100,
) -> tuple[ChampionSignalRow, ...]:
    """Read signal rows newer than a composite created_at/event_id cursor."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    with _connect_read_only(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT created_at, ts, event_id, model_version, prob_up_15m,
                   feature_snapshot_json
            FROM prediction_events
            WHERE model_version = ?
              {_outcome_side_sql(outcome_side)}
              AND (
                    created_at > ?
                 OR (created_at = ? AND event_id > ?)
              )
            ORDER BY created_at ASC, event_id ASC
            LIMIT ?
            """,
            [
                model_version,
                *_outcome_side_params(outcome_side),
                after_created_at,
                after_created_at,
                after_event_id,
                limit,
            ],
        ).fetchall()
    return tuple(
        _signal_from_event_row(row, edge_threshold=edge_threshold)
        for row in rows
    )


def latest_signal_cursor(
    db_path: Path | str,
    *,
    model_version: str,
    outcome_side: str | None = None,
) -> tuple[int, str]:
    """Return the latest composite cursor for one model."""

    with _connect_read_only(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT created_at, event_id
            FROM prediction_events
            WHERE model_version = ?
              {_outcome_side_sql(outcome_side)}
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """,
            [model_version, *_outcome_side_params(outcome_side)],
        ).fetchone()
    if row is None:
        return 0, ""
    return int(row[0]), str(row[1])


def format_signal_row(row: ChampionSignalRow) -> str:
    """Format a signal row for a plain terminal stream."""

    ts_text = datetime.fromtimestamp(row.ts / 1_000, tz=UTC).isoformat()
    market_text = (
        "NA" if row.market_implied_prob is None else f"{row.market_implied_prob:.4f}"
    )
    edge_text = "NA" if row.edge is None else f"{row.edge:.4f}"
    return (
        f"{ts_text} {row.signal:9s} "
        f"prob={row.prob_up_15m:.4f} market={market_text} edge={edge_text} "
        f"side={row.outcome_side} symbol={row.source_symbol} event={row.event_id}"
    )


def _connect_read_only(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    from bigan.mlops.registry import connect_mlops_db

    return connect_mlops_db(db_path, read_only=True)


def _signal_from_event_row(
    row: Sequence[Any],
    *,
    edge_threshold: float,
) -> ChampionSignalRow:
    created_at, ts, event_id, model_version, probability, snapshot_json = row
    snapshot = _parse_snapshot(snapshot_json)
    source_symbol = _snapshot_text(
        snapshot,
        ("source_symbol", "symbol", "canonical_symbol"),
        fallback="unknown",
    )
    canonical_symbol = _snapshot_text(snapshot, ("canonical_symbol", "symbol"), fallback="")
    outcome_side = _outcome_side(canonical_symbol)
    event_ts = int(ts)
    market = _tradable_market_implied_probability(snapshot, event_ts=event_ts)
    prob = float(probability)
    edge = None if market is None else _token_probability(prob, outcome_side) - market
    signal = _signal_from_edge(
        edge,
        edge_threshold=edge_threshold,
        outcome_side=outcome_side,
    )
    return ChampionSignalRow(
        created_at=int(created_at),
        ts=event_ts,
        event_id=str(event_id),
        model_version=str(model_version),
        source_symbol=source_symbol,
        canonical_symbol=canonical_symbol,
        outcome_side=outcome_side,
        prob_up_15m=prob,
        market_implied_prob=market,
        edge=edge,
        signal=signal,
    )


def _parse_snapshot(snapshot_json: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(snapshot_json))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tradable_market_implied_probability(
    snapshot: Mapping[str, Any],
    *,
    event_ts: int,
) -> float | None:
    return tradable_market_implied_probability(snapshot, event_ts=event_ts)


def _snapshot_text(
    snapshot: Mapping[str, Any],
    keys: Sequence[str],
    *,
    fallback: str,
) -> str:
    for key in keys:
        value = snapshot.get(key)
        if value is not None and str(value):
            return str(value)
    return fallback


def _signal_from_edge(
    edge: float | None,
    *,
    edge_threshold: float,
    outcome_side: str,
) -> str:
    if edge is None:
        return "NO_SIGNAL"
    if edge < edge_threshold:
        return "HOLD"
    if outcome_side in {"UP", "DOWN"}:
        return f"BUY_{outcome_side}"
    return "BUY_TOKEN"


def _outcome_side(canonical_symbol: str) -> str:
    text = canonical_symbol.strip().upper()
    if text.endswith("-UP-15M"):
        return "UP"
    if text.endswith("-DOWN-15M"):
        return "DOWN"
    side = text.rsplit(":", 1)[-1] if text else ""
    return side if side in {"UP", "DOWN"} else "UNKNOWN"


def _normalise_outcome_side_value(outcome_side: str) -> str:
    side = str(outcome_side).strip().upper()
    return side if side in {"UP", "DOWN"} else "UNKNOWN"


def _outcome_side_sql(outcome_side: str | None) -> str:
    side = _normalise_outcome_side_filter(outcome_side)
    if side is None:
        return ""
    return (
        "AND ("
        "upper(json_extract_string(feature_snapshot_json, '$.canonical_symbol')) LIKE ? "
        "OR upper(json_extract_string(feature_snapshot_json, '$.canonical_symbol')) LIKE ?"
        ")"
    )


def _outcome_side_params(outcome_side: str | None) -> list[str]:
    side = _normalise_outcome_side_filter(outcome_side)
    return [] if side is None else [f"%:{side}", f"%-{side}-15M"]


def _normalise_outcome_side_filter(outcome_side: str | None) -> str | None:
    if outcome_side is None:
        return None
    side = str(outcome_side).strip().upper()
    if not side or side == "ANY":
        return None
    if side not in {"UP", "DOWN"}:
        raise ValueError("outcome_side must be UP, DOWN, ANY, or None")
    return side


def _token_probability(prob_up_15m: float, outcome_side: str) -> float:
    return 1.0 - prob_up_15m if outcome_side == "DOWN" else prob_up_15m
