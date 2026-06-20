"""Live terminal dashboard for champion signals and round-level paper PnL."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .market_quality import tradable_market_implied_probability

HORIZON_MS = 15 * 60_000


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Standard signal event schema for dashboard and monitoring consumers."""

    timestamp: int
    signal_type: str
    prob: float
    market_implied_prob: float | None
    edge: float | None
    symbol: str
    event_id: str
    position: str
    entry_price: float | None
    unrealized_pnl: float | None
    source_market: str | None
    round_id: str
    outcome_side: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundSettlementEvent:
    """Standard round settlement schema for paper-trading PnL rows."""

    round_id: str
    event_id: str | None
    entry_signal: str
    entry_price: float | None
    exit_price: float | None
    result: str
    realized_pnl: float
    settled_at: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundDashboardRow:
    """Current or historical round state shown by the terminal dashboard."""

    round_id: str
    latest_ts: int
    signal_count: int
    signals: tuple[SignalEvent, ...]
    latest_signal: SignalEvent
    entry_event_id: str | None
    entry_price: float | None
    exit_price: float | None
    position: str
    unrealized_pnl: float | None
    settlement: RoundSettlementEvent | None

    @property
    def is_settled(self) -> bool:
        return self.settlement is not None


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """A complete terminal-renderable dashboard state."""

    model_version: str
    generated_at: int
    current_round: RoundDashboardRow | None
    history: tuple[RoundDashboardRow, ...]
    session_total_pnl: float
    session_win_count: int
    session_loss_count: int
    session_trade_count: int
    settled_round_count: int
    edge_trigger_rate_1h: float | None
    alerts: tuple[str, ...]
    v7_pm_monitoring: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["current_round"] = (
            None if self.current_round is None else asdict(self.current_round)
        )
        payload["history"] = [asdict(row) for row in self.history]
        return payload


@dataclass(frozen=True, slots=True)
class _EventRecord:
    created_at: int
    ts: int
    event_id: str
    model_version: str
    prob: float
    source_symbol: str
    source_market: str | None
    canonical_symbol: str
    market_implied_prob: float | None
    realized_label: bool | None
    realized_return: float | None
    outcome_ts: int | None

    @property
    def outcome_side(self) -> str:
        return _outcome_side(self.canonical_symbol)

    @property
    def round_id(self) -> str:
        if self.canonical_symbol and ":" in self.canonical_symbol:
            return self.canonical_symbol.rsplit(":", 1)[0]
        return self.source_market or self.canonical_symbol or self.source_symbol


def read_dashboard_snapshot(
    db_path: Path | str,
    *,
    model_version: str,
    edge_threshold: float = 0.30,
    exit_edge_threshold: float = 0.10,
    outcome_side: str | None = "UP",
    lookback_hours: float = 6.0,
    limit: int = 1_000,
    now_ms: int | None = None,
    phase4_summary_path: Path | str | None = None,
) -> DashboardSnapshot:
    """Read recent monitoring events and build a paper-trading dashboard snapshot."""

    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    run_now_ms = _now_ms() if now_ms is None else int(now_ms)
    since_ms = run_now_ms - int(lookback_hours * 60 * 60 * 1000)
    records = _read_event_records(
        db_path,
        model_version=model_version,
        outcome_side=outcome_side,
        since_ms=since_ms,
        limit=limit,
    )
    rounds = _round_rows(
        records,
        edge_threshold=edge_threshold,
        exit_edge_threshold=exit_edge_threshold,
    )
    current = max(
        rounds,
        key=lambda row: (row.latest_ts, row.latest_signal.timestamp, row.latest_signal.event_id),
        default=None,
    )
    history = tuple(
        row
        for row in sorted(rounds, key=lambda item: item.latest_ts, reverse=True)
        if row.is_settled
    )
    settled_with_trades = [
        row
        for row in history
        if row.settlement is not None and row.settlement.entry_price is not None
    ]
    total_pnl = sum(row.settlement.realized_pnl for row in settled_with_trades if row.settlement)
    wins = sum(1 for row in settled_with_trades if row.settlement and row.settlement.realized_pnl > 0)
    losses = sum(1 for row in settled_with_trades if row.settlement and row.settlement.realized_pnl < 0)
    edge_trigger_rate = _edge_trigger_rate_1h(
        records,
        edge_threshold=edge_threshold,
        now_ms=run_now_ms,
    )
    alerts = _alerts(
        history,
        edge_trigger_rate_1h=edge_trigger_rate,
    )
    return DashboardSnapshot(
        model_version=model_version,
        generated_at=run_now_ms,
        current_round=current,
        history=history[:10],
        session_total_pnl=total_pnl,
        session_win_count=wins,
        session_loss_count=losses,
        session_trade_count=len(settled_with_trades),
        settled_round_count=len(history),
        edge_trigger_rate_1h=edge_trigger_rate,
        alerts=tuple(alerts),
        v7_pm_monitoring=_read_v7_pm_monitoring(phase4_summary_path),
    )


def render_dashboard(snapshot: DashboardSnapshot, *, max_signals: int = 10) -> str:
    """Render a dashboard snapshot as an ANSI-clearable plain terminal table."""

    lines: list[str] = []
    lines.append(
        "LIVE CHAMPION DASHBOARD "
        f"model={snapshot.model_version} updated={_format_ts(snapshot.generated_at)}"
    )
    lines.append("=" * 100)
    lines.extend(_render_current_round(snapshot.current_round, snapshot.generated_at, max_signals=max_signals))
    lines.append("")
    lines.extend(_render_history(snapshot.history))
    lines.append("")
    lines.extend(_render_session(snapshot))
    if snapshot.v7_pm_monitoring:
        lines.append("")
        lines.extend(_render_v7_pm_monitoring(snapshot.v7_pm_monitoring))
    if snapshot.alerts:
        lines.append("")
        lines.append("ALERTS")
        lines.extend(f"! {alert}" for alert in snapshot.alerts)
    return "\n".join(lines)


def _read_v7_pm_monitoring(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    monitoring = payload.get("v7_pm_monitoring")
    return monitoring if isinstance(monitoring, dict) else None


def _read_event_records(
    db_path: Path | str,
    *,
    model_version: str,
    outcome_side: str | None,
    since_ms: int,
    limit: int,
) -> tuple[_EventRecord, ...]:
    side_clause, side_params = _outcome_side_filter_sql(outcome_side)
    from bigan.mlops.registry import connect_mlops_db

    with connect_mlops_db(db_path, read_only=True) as conn:
        rows = conn.execute(
            f"""
            WITH recent AS (
                SELECT
                    e.created_at,
                    e.ts,
                    e.event_id,
                    e.model_version,
                    e.prob_up_15m,
                    json_extract_string(e.feature_snapshot_json, '$.source_symbol') AS source_symbol,
                    json_extract_string(e.feature_snapshot_json, '$.source_market') AS source_market,
                    json_extract_string(e.feature_snapshot_json, '$.canonical_symbol') AS canonical_symbol,
                    try_cast(
                        json_extract_string(e.feature_snapshot_json, '$.market_implied_prob')
                        AS DOUBLE
                    ) AS market_implied_prob,
                    e.feature_snapshot_json,
                    o.realized_label,
                    o.realized_return,
                    o.outcome_ts
                FROM prediction_events e
                LEFT JOIN prediction_outcomes o USING (event_id)
                WHERE e.model_version = ?
                  AND e.ts >= ?
                  {side_clause}
                ORDER BY e.created_at DESC, e.event_id DESC
                LIMIT ?
            )
            SELECT *
            FROM recent
            ORDER BY ts ASC, created_at ASC, event_id ASC
            """,
            [model_version, since_ms, *side_params, limit],
        ).fetchall()
    return tuple(
        _EventRecord(
            created_at=int(row[0]),
            ts=int(row[1]),
            event_id=str(row[2]),
            model_version=str(row[3]),
            prob=float(row[4]),
            source_symbol=str(row[5] or "unknown"),
            source_market=None if row[6] is None else str(row[6]),
            canonical_symbol=str(row[7] or ""),
            market_implied_prob=_tradable_market_implied_probability(
                row[9],
                event_ts=int(row[1]),
                fallback=None if row[8] is None else float(row[8]),
            ),
            realized_label=None if row[10] is None else bool(row[10]),
            realized_return=None if row[11] is None else float(row[11]),
            outcome_ts=None if row[12] is None else int(row[12]),
        )
        for row in rows
    )


def _round_rows(
    records: Iterable[_EventRecord],
    *,
    edge_threshold: float,
    exit_edge_threshold: float,
) -> tuple[RoundDashboardRow, ...]:
    grouped: dict[str, list[_EventRecord]] = defaultdict(list)
    for record in records:
        grouped[record.round_id].append(record)

    rows = [
        _round_row(
            round_id,
            sorted(items, key=lambda item: (item.ts, item.created_at, item.event_id)),
            edge_threshold=edge_threshold,
            exit_edge_threshold=exit_edge_threshold,
        )
        for round_id, items in grouped.items()
        if items
    ]
    rows.sort(key=lambda item: (item.latest_ts, item.round_id))
    return tuple(rows)


def _round_row(
    round_id: str,
    records: Sequence[_EventRecord],
    *,
    edge_threshold: float,
    exit_edge_threshold: float,
) -> RoundDashboardRow:
    position_open = False
    position_closed = False
    entry_price: float | None = None
    entry_event_id: str | None = None
    entry_outcome_side: str | None = None
    exit_price: float | None = None
    exit_event_id: str | None = None
    signal_events: list[SignalEvent] = []

    for record in records:
        edge = _edge(record)
        signal_type = "HOLD"
        unrealized_pnl = None
        position = "-"
        is_entry_side = entry_outcome_side is None or record.outcome_side == entry_outcome_side

        if position_open and entry_price is not None and is_entry_side:
            unrealized_pnl = _pnl(record.market_implied_prob, entry_price)
            position = f"{entry_price:.2f} x 1"
        elif position_open and entry_price is not None:
            position = f"{entry_price:.2f} x 1"

        if (
            not position_open
            and not position_closed
            and edge is not None
            and edge >= edge_threshold
            and record.market_implied_prob is not None
        ):
            signal_type = "BUY_UP" if record.outcome_side == "UP" else f"BUY_{record.outcome_side}"
            position_open = True
            entry_price = record.market_implied_prob
            entry_event_id = record.event_id
            entry_outcome_side = record.outcome_side
            unrealized_pnl = 0.0
            position = f"{entry_price:.2f} x 1"
        elif (
            position_open
            and record.outcome_side == entry_outcome_side
            and edge is not None
            and edge <= exit_edge_threshold
            and record.market_implied_prob is not None
        ):
            signal_type = "SELL"
            exit_price = record.market_implied_prob
            exit_event_id = record.event_id
            unrealized_pnl = _pnl(exit_price, entry_price)
            position_open = False
            position_closed = True
            position = f"sold {exit_price:.2f}"

        signal_events.append(
            SignalEvent(
                timestamp=record.ts,
                signal_type=signal_type,
                prob=record.prob,
                market_implied_prob=record.market_implied_prob,
                edge=edge,
                symbol=record.source_symbol,
                event_id=record.event_id,
                position=position,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                source_market=record.source_market,
                round_id=round_id,
                outcome_side=record.outcome_side,
            )
        )

    settlement = _settlement(
        round_id,
        records,
        entry_event_id=entry_event_id,
        entry_price=entry_price,
        entry_outcome_side=entry_outcome_side,
        exit_event_id=exit_event_id,
        exit_price=exit_price,
    )
    latest_signal = signal_events[-1]
    return RoundDashboardRow(
        round_id=round_id,
        latest_ts=records[-1].ts,
        signal_count=len(signal_events),
        signals=tuple(signal_events),
        latest_signal=latest_signal,
        entry_event_id=entry_event_id,
        entry_price=entry_price,
        exit_price=exit_price,
        position=latest_signal.position,
        unrealized_pnl=latest_signal.unrealized_pnl,
        settlement=settlement,
    )


def _settlement(
    round_id: str,
    records: Sequence[_EventRecord],
    *,
    entry_event_id: str | None,
    entry_price: float | None,
    entry_outcome_side: str | None,
    exit_event_id: str | None,
    exit_price: float | None,
) -> RoundSettlementEvent | None:
    by_event_id = {record.event_id: record for record in records}
    entry_signal = _entry_signal(entry_outcome_side)
    if exit_event_id is not None and entry_price is not None and exit_price is not None:
        return RoundSettlementEvent(
            round_id=round_id,
            event_id=exit_event_id,
            entry_signal=entry_signal,
            entry_price=entry_price,
            exit_price=exit_price,
            result="SELL",
            realized_pnl=exit_price - entry_price,
            settled_at=by_event_id[exit_event_id].ts,
        )
    if entry_event_id is not None:
        entry = by_event_id.get(entry_event_id)
        if entry is None or entry.realized_return is None or entry.realized_label is None:
            return None
        return RoundSettlementEvent(
            round_id=round_id,
            event_id=entry_event_id,
            entry_signal=entry_signal,
            entry_price=entry_price,
            exit_price=1.0 if entry.realized_label else 0.0,
            result="UP" if entry.realized_label else "DOWN",
            realized_pnl=entry.realized_return,
            settled_at=entry.outcome_ts,
        )

    settled = next((record for record in reversed(records) if record.realized_label is not None), None)
    if settled is None:
        return None
    return RoundSettlementEvent(
        round_id=round_id,
        event_id=None,
        entry_signal="HOLD",
        entry_price=None,
        exit_price=1.0 if settled.realized_label else 0.0,
        result="UP" if settled.realized_label else "DOWN",
        realized_pnl=0.0,
        settled_at=settled.outcome_ts,
    )


def _render_current_round(
    row: RoundDashboardRow | None,
    now_ms: int,
    *,
    max_signals: int,
) -> list[str]:
    if row is None:
        return ["CURRENT ROUND", "No recent champion signal events found."]

    remaining = _remaining_text(row.latest_ts, now_ms)
    signal_rows = [
        [
            _clock(signal.timestamp),
            signal.signal_type,
            _fmt(signal.prob),
            _signed(signal.edge),
            signal.position,
            _signed(signal.unrealized_pnl),
            shorten(signal.event_id, 14),
        ]
        for signal in row.signals[-max_signals:]
    ]
    out = [
        f"CURRENT ROUND {shorten(row.round_id, 52)} remaining={remaining}",
        _table(
            ["Time", "Signal", "Prob", "Edge", "Position", "Unrlzd", "Event"],
            signal_rows,
        ),
    ]
    return out


def _render_history(rows: Sequence[RoundDashboardRow]) -> list[str]:
    table_rows = []
    for row in rows[:10]:
        settlement = row.settlement
        table_rows.append(
            [
                shorten(row.round_id, 24),
                "HOLD" if row.entry_price is None else "BUY_UP",
                "-" if row.entry_price is None else f"{row.entry_price:.2f}",
                "-" if settlement is None else settlement.result,
                "-" if settlement is None else _signed(settlement.realized_pnl),
                str(row.signal_count),
            ]
        )
    return [
        "ROUND HISTORY",
        _table(
            ["Round", "Entry", "EntryPx", "Result", "PnL", "Signals"],
            table_rows or [["-", "-", "-", "-", "-", "-"]],
        ),
    ]


def _render_session(snapshot: DashboardSnapshot) -> list[str]:
    win_rate = (
        None
        if snapshot.session_trade_count == 0
        else snapshot.session_win_count / snapshot.session_trade_count
    )
    trigger = (
        "NA"
        if snapshot.edge_trigger_rate_1h is None
        else f"{snapshot.edge_trigger_rate_1h:.1%}"
    )
    return [
        "SESSION",
        (
            f"PnL={_signed(snapshot.session_total_pnl)} "
            f"WinRate={_pct(win_rate)} "
            f"Trades={snapshot.session_trade_count} "
            f"SettledRounds={snapshot.settled_round_count} "
            f"EdgeTrigger1h={trigger}"
        ),
    ]


def _render_v7_pm_monitoring(monitoring: dict[str, Any]) -> list[str]:
    hold_edge = monitoring.get("divergence_reduce_hold_edge") or {}
    take_profit = monitoring.get("take_profit_candidates") or {}
    reason_counts = take_profit.get("reason_counts") or {}
    reason_text = (
        "-"
        if not reason_counts
        else ", ".join(f"{key}={value}" for key, value in sorted(reason_counts.items()))
    )
    return [
        "V7 PM MONITORING",
        (
            "DivergenceReduceHoldEdge "
            f"count={hold_edge.get('count', 0)} "
            f"p50={_fmt_optional_num(hold_edge.get('p50'))} "
            f"p90={_fmt_optional_num(hold_edge.get('p90'))}"
        ),
        (
            "TakeProfitCandidates "
            f"evaluations={take_profit.get('evaluations', 0)} "
            f"exits={take_profit.get('exits', 0)} "
            f"unexecuted={take_profit.get('unexecuted', 0)} "
            f"reasons={reason_text}"
        ),
    ]


def _fmt_optional_num(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header = " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(headers))
    sep = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _edge(record: _EventRecord) -> float | None:
    if record.market_implied_prob is None:
        return None
    return _token_probability(record.prob, record.outcome_side) - record.market_implied_prob


def _tradable_market_implied_probability(
    snapshot_json: Any,
    *,
    event_ts: int,
    fallback: float | None,
) -> float | None:
    try:
        snapshot = json.loads(str(snapshot_json))
    except json.JSONDecodeError:
        snapshot = {}
    if isinstance(snapshot, dict):
        return tradable_market_implied_probability(snapshot, event_ts=event_ts)
    return fallback


def _pnl(mark_price: float | None, entry_price: float | None) -> float | None:
    if mark_price is None or entry_price is None:
        return None
    return mark_price - entry_price


def _entry_signal(outcome_side: str | None) -> str:
    if outcome_side is None:
        return "BUY_TOKEN"
    return "BUY_UP" if outcome_side == "UP" else f"BUY_{outcome_side}"


def _edge_trigger_rate_1h(
    records: Sequence[_EventRecord],
    *,
    edge_threshold: float,
    now_ms: int,
) -> float | None:
    recent = [record for record in records if record.ts >= now_ms - 60 * 60 * 1000]
    if not recent:
        return None
    triggered = sum(
        1
        for record in recent
        if (edge := _edge(record)) is not None and edge >= edge_threshold
    )
    return triggered / len(recent)


def _alerts(
    history: Sequence[RoundDashboardRow],
    *,
    edge_trigger_rate_1h: float | None,
) -> list[str]:
    alerts: list[str] = []
    traded = [
        row
        for row in history
        if row.settlement is not None and row.settlement.entry_price is not None
    ]
    if len(traded) >= 3 and all(
        row.settlement is not None and row.settlement.realized_pnl < 0
        for row in traded[:3]
    ):
        alerts.append("3 consecutive losing traded rounds")
    if edge_trigger_rate_1h == 0.0:
        alerts.append("edge_trigger_rate was 0 over the last hour")
    return alerts


def _outcome_side_filter_sql(outcome_side: str | None) -> tuple[str, list[str]]:
    side = _normalise_outcome_side_filter(outcome_side)
    if side is None:
        return "", []
    return (
        "AND ("
        "upper(json_extract_string(e.feature_snapshot_json, '$.canonical_symbol')) LIKE ? "
        "OR upper(json_extract_string(e.feature_snapshot_json, '$.canonical_symbol')) LIKE ?"
        ")",
        [f"%:{side}", f"%-{side}-15M"],
    )


def _normalise_outcome_side_filter(outcome_side: str | None) -> str | None:
    if outcome_side is None:
        return None
    side = str(outcome_side).strip().upper()
    if not side or side == "ANY":
        return None
    if side not in {"UP", "DOWN"}:
        raise ValueError("outcome_side must be UP, DOWN, ANY, or None")
    return side


def _outcome_side(canonical_symbol: str) -> str:
    text = canonical_symbol.strip().upper()
    if text.endswith("-UP-15M"):
        return "UP"
    if text.endswith("-DOWN-15M"):
        return "DOWN"
    side = text.rsplit(":", 1)[-1] if text else ""
    return side if side in {"UP", "DOWN"} else "UNKNOWN"


def _remaining_text(latest_ts: int, now_ms: int) -> str:
    remaining_ms = latest_ts + HORIZON_MS - now_ms
    if remaining_ms <= 0:
        return "settling"
    seconds = remaining_ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat(timespec="seconds")


def _clock(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%H:%M:%S")


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _signed(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.3f}"


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def shorten(value: str, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _now_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def _token_probability(prob_up_15m: float, outcome_side: str) -> float:
    return 1.0 - prob_up_15m if outcome_side == "DOWN" else prob_up_15m
