from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "champion_signal_bridge.py"
EXECUTOR_SCRIPT = REPO_ROOT / "scripts" / "polymarket_phase4_live_champion_executor.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bridge = _load_script(BRIDGE_SCRIPT, "champion_signal_bridge")
executor = _load_script(EXECUTOR_SCRIPT, "polymarket_phase4_live_champion_executor")


def test_bridge_signal_from_prediction_event_row() -> None:
    snapshot = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "source_symbol": "token-up",
        "market_implied_prob": 0.47,
    }

    signal = bridge._bridge_signal_from_row(
        ("pred-1", 1_779_773_900_000, 1_779_773_910_000, 0.98, json.dumps(snapshot)),
        model_version="xgboost-v4",
    )

    assert signal is not None
    assert signal.event_id == "pred-1"
    assert signal.round_slug == "btc-updown-15m-1779774300"
    assert signal.round_end_ts == 1_779_775_200_000
    assert signal.outcome_side == "UP"
    assert signal.token_id == "token-up"
    assert signal.token_probability == 0.98
    assert signal.edge == 0.51
    assert signal.bridged_at > 0


def test_executor_reads_bridged_signal_jsonl(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    payload = {
        "event_id": "pred-1",
        "ts": 1_779_773_900_000,
        "created_at": 1_779_773_910_000,
        "model_version": "xgboost-v4",
        "prob_up_15m": 0.98,
        "canonical_symbol": "BTC-15M:btc-updown-15m-1779774300:UP",
        "token_id": "token-up",
        "outcome_side": "UP",
        "round_slug": "btc-updown-15m-1779774300",
        "round_end_ts": 1_779_775_200_000,
        "market_implied_prob": 0.47,
        "token_probability": 0.98,
        "edge": 0.51,
        "bridged_at": 1_779_773_912_000,
    }
    queue.write_text(json.dumps(payload) + "\nnot-json\n\n", encoding="utf-8")

    events, cursor = executor._read_signal_jsonl_after(
        queue,
        after_line_number=0,
        model_version="xgboost-v4",
        limit=10,
    )

    assert cursor == 3
    assert len(events) == 1
    assert events[0].event_id == "pred-1"
    assert events[0].edge == 0.51
    assert events[0].bridged_at == 1_779_773_912_000


def test_executor_latency_helpers() -> None:
    event = executor.SignalEvent(
        event_id="pred-1",
        ts=1_000,
        created_at=2_000,
        prob_up_15m=0.98,
        canonical_symbol="BTC-15M:btc-updown-15m-1779774300:UP",
        token_id="token-up",
        outcome_side="UP",
        round_slug="btc-updown-15m-1779774300",
        round_end_ts=1_779_775_200_000,
        market_implied_prob=0.47,
        token_probability=0.98,
        edge=0.51,
        bridged_at=3_000,
    )

    assert executor._signal_latency_ms(event, 5_500) == {
        "event_ts_to_at_ms": 4_500,
        "signal_created_to_at_ms": 3_500,
        "bridge_to_at_ms": 2_500,
    }


def test_executor_starts_jsonl_cursor_at_tail(tmp_path: Path) -> None:
    queue = tmp_path / "signals.jsonl"
    queue.write_text("{}\n{}\n", encoding="utf-8")

    assert executor._latest_signal_jsonl_cursor(queue, start="tail") == 2
    assert executor._latest_signal_jsonl_cursor(queue, start="beginning") == 0


def test_executor_exit_holds_when_orderbook_is_unavailable(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()
    position = _position()

    result = executor._maybe_exit(
        client=_UnavailableBookClient(),
        position_manager=object(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    assert result is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "exit_hold"
    assert row["reason"] == "orderbook_unavailable"
    assert row["token_id"] == "token-up"
    assert row["error_type"] == "RuntimeError"


def test_executor_shutdown_close_skips_unavailable_orderbook(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    positions = {"round-1": _position()}

    closed, pending, pnl = executor._close_remaining_positions(
        client=_UnavailableBookClient(),
        position_manager=object(),
        positions=positions,
        log_path=log_path,
        sell_slippage=0.01,
    )

    assert closed == 0
    assert pending == 0
    assert pnl == 0.0
    assert "round-1" in positions
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "shutdown_close_skipped"
    assert row["reason"] == "orderbook_unavailable"
    assert row["token_id"] == "token-up"


def test_round_lifecycle_only_confirmed_fill_locks_round() -> None:
    state = executor.RoundLifecycleState()
    signal = _signal()

    assert state.mark_event_seen(signal.event_id) is True
    assert state.mark_event_seen(signal.event_id) is False

    state.mark_entry_attempted(signal.event_id)
    state.mark_entry_result(signal, None)

    assert signal.event_id in state.attempted_entry_event_ids
    assert signal.round_slug not in state.filled_rounds
    assert signal.round_slug not in state.open_positions

    position = _position()
    state.mark_entry_result(signal, position)

    assert signal.round_slug in state.filled_rounds
    assert state.open_positions[signal.round_slug] == position


def test_executor_fill_requires_confirmed_trade(monkeypatch) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    client = _TradeClient(
        [
            {
                "taker_order_id": "order-1",
                "side": "BUY",
                "status": "FAILED",
                "price": "0.50",
                "size": "2",
            },
            {
                "taker_order_id": "order-1",
                "side": "BUY",
                "status": "CONFIRMED",
                "price": "0.51",
                "size": "1.96",
            },
        ]
    )

    fill = executor._fill_for_order(client, "order-1", wanted_side="BUY")

    assert fill["status"] == "CONFIRMED"
    assert fill["price"] == "0.51"


def test_executor_fok_post_failure_does_not_create_position(tmp_path: Path) -> None:
    log_path = tmp_path / "phase4.jsonl"
    signal = _signal()

    position = executor._try_entry(
        client=_BuyPostFailureClient(),
        position_manager=_PositionManager(),
        signal=signal,
        log_path=log_path,
        max_position_size_usdc=1.0,
        edge_threshold=0.45,
        buy_slippage=0.02,
    )

    assert position is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "entry_order_post_failed"
    assert row["error_type"] == "_FokKilledError"
    assert row["signal"]["event_id"] == signal.event_id
    assert row["worst_price"] == 0.52


def test_executor_opposite_signal_can_exit_without_reverse_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position(size=1.960783)
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        edge=0.52,
        round_end_ts=300_000,
    )
    client = _SellClient()
    manager = _PositionManager()

    sell_result = executor._maybe_exit_opposite_correction(
        client=client,
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        opposite_exit_edge_threshold=0.45,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    posted = next(row for row in rows if row["event"] == "exit_order_posted")
    filled = next(row for row in rows if row["event"] == "exit_filled")

    assert sell_result == executor.SellResult(status="filled", realized_pnl=0.10)
    assert client.created_orders == [{"token_id": "token-up", "side": "SELL", "amount": 1.96, "price": 0.59}]
    assert manager.closed == [("phase4-round-1-UP", 0.60)]
    assert posted["reason"] == "opposite_side_exit_correction"
    assert posted["signal"]["event_id"] == "pred-strong-down"
    assert posted["sell_size"] == 1.96
    assert posted["dust_amount"] > 0
    assert filled["reason"] == "opposite_side_exit_correction"
    assert filled["account_cashflow_reconciliation_required"] is True


def test_executor_matched_sell_without_trade_confirmation_is_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(edge=-0.31)
    manager = _PositionManager()

    sell_result = executor._maybe_exit(
        client=_SellPendingClient(),
        position_manager=manager,
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    pending = next(row for row in rows if row["event"] == "exit_pending_confirmation")

    assert sell_result == executor.SellResult(status="pending_confirmation")
    assert manager.closed == []
    assert pending["position_assumed_closed_to_prevent_duplicate_sell"] is True
    assert pending["account_cashflow_reconciliation_required"] is True


def test_executor_sell_post_failure_does_not_raise(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(edge=-0.31)

    sell_result = executor._maybe_exit(
        client=_SellPostFailureClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        exit_edge_threshold=0.10,
        profit_target=0.15,
        sell_slippage=0.01,
    )

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert sell_result is None
    assert row["event"] == "exit_order_post_failed"
    assert row["error_type"] == "_FokKilledError"


def test_executor_opposite_signal_holds_near_expiry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor, "_now_ms", lambda: 10_000)
    log_path = tmp_path / "phase4.jsonl"
    position = _position()
    signal = _signal(
        event_id="pred-strong-down",
        side="DOWN",
        token_id="token-down",
        edge=0.55,
        round_end_ts=100_000,
    )

    pnl = executor._maybe_exit_opposite_correction(
        client=_SellClient(),
        position_manager=_PositionManager(),
        position=position,
        signal=signal,
        log_path=log_path,
        opposite_exit_edge_threshold=0.45,
        opposite_exit_min_seconds_to_expiry=120.0,
        sell_slippage=0.01,
    )

    assert pnl is None
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "opposite_exit_hold"
    assert row["reason"] == "insufficient_time_remaining"
    assert row["seconds_to_expiry"] == 90.0


class _UnavailableBookClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        raise RuntimeError(f"No orderbook exists for the requested token id: {token_id}")


class _TradeClient:
    def __init__(self, trades):
        self.trades = trades

    def get_trades(self):  # noqa: ANN201
        return self.trades


class _FokKilledError(Exception):
    pass


class _BuyPostFailureClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.49"}], "asks": [{"price": "0.50"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False

    def create_market_order(self, *, order_args, options):  # noqa: ANN001, ANN201, ARG002
        assert order_args.token_id == "token-up"
        assert order_args.amount == 1.0
        assert order_args.price == 0.52
        return {"order": "signed"}

    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        raise _FokKilledError("order could not be fully filled")


class _SellClient:
    def __init__(self) -> None:
        self.created_orders = []

    def get_order_book(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return {"bids": [{"price": "0.60"}], "asks": [{"price": "0.61"}]}

    def get_tick_size(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return "0.01"

    def get_neg_risk(self, token_id: str):  # noqa: ANN201
        assert token_id == "token-up"
        return False

    def create_market_order(self, *, order_args, options):  # noqa: ANN001, ANN201, ARG002
        self.created_orders.append(
            {
                "token_id": order_args.token_id,
                "side": order_args.side,
                "amount": order_args.amount,
                "price": order_args.price,
            }
        )
        return {"order": "signed"}

    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        return {"success": True, "status": "matched", "orderID": "order-sell"}

    def get_trades(self):  # noqa: ANN201
        return [
            {
                "taker_order_id": "order-sell",
                "side": "SELL",
                "status": "MINED",
                "price": "0.60",
                "size": "1.96",
            }
        ]


class _SellPendingClient(_SellClient):
    def get_trades(self):  # noqa: ANN201
        return []


class _SellPostFailureClient(_SellClient):
    def post_order(self, order, order_type):  # noqa: ANN001, ANN201, ARG002
        raise _FokKilledError("not enough balance / allowance")


class _PositionManager:
    def __init__(self) -> None:
        self.closed = []

    def close_position(self, event_id: str, exit_price: float):  # noqa: ANN201
        self.closed.append((event_id, exit_price))
        return SimpleNamespace(realized_pnl=0.10)


def _signal(
    *,
    event_id: str = "pred-1",
    side: str = "UP",
    token_id: str = "token-up",
    edge: float = 0.51,
    round_end_ts: int = 1_779_775_200_000,
):
    return executor.SignalEvent(
        event_id=event_id,
        ts=1_000,
        created_at=2_000,
        prob_up_15m=0.98,
        canonical_symbol=f"BTC-15M:round-1:{side}",
        token_id=token_id,
        outcome_side=side,
        round_slug="round-1",
        round_end_ts=round_end_ts,
        market_implied_prob=0.47,
        token_probability=0.98,
        edge=edge,
        bridged_at=3_000,
    )


def _position(*, size: float = 2.0):
    return executor.LivePosition(
        event_id="phase4-round-1-UP",
        round_slug="round-1",
        side="UP",
        token_id="token-up",
        entry_price=0.50,
        fill_price=0.50,
        size=size,
        order_id="order-1",
        opened_at=4_000,
        entry_signal_event_id="pred-1",
        entry_signal_ts=1_000,
        entry_signal_created_at=2_000,
        entry_signal_bridged_at=3_000,
        entry_order_posted_at=3_500,
    )
