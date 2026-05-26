from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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

    closed, pnl = executor._close_remaining_positions(
        client=_UnavailableBookClient(),
        position_manager=object(),
        positions=positions,
        log_path=log_path,
        sell_slippage=0.01,
    )

    assert closed == 0
    assert pnl == 0.0
    assert "round-1" in positions
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "shutdown_close_skipped"
    assert row["reason"] == "orderbook_unavailable"
    assert row["token_id"] == "token-up"


class _UnavailableBookClient:
    def get_order_book(self, token_id: str):  # noqa: ANN201
        raise RuntimeError(f"No orderbook exists for the requested token id: {token_id}")


def _signal():
    return executor.SignalEvent(
        event_id="pred-1",
        ts=1_000,
        created_at=2_000,
        prob_up_15m=0.98,
        canonical_symbol="BTC-15M:round-1:UP",
        token_id="token-up",
        outcome_side="UP",
        round_slug="round-1",
        round_end_ts=1_779_775_200_000,
        market_implied_prob=0.47,
        token_probability=0.98,
        edge=0.51,
        bridged_at=3_000,
    )


def _position():
    return executor.LivePosition(
        event_id="phase4-round-1-UP",
        round_slug="round-1",
        side="UP",
        token_id="token-up",
        entry_price=0.50,
        fill_price=0.50,
        size=2.0,
        order_id="order-1",
        opened_at=4_000,
        entry_signal_event_id="pred-1",
        entry_signal_ts=1_000,
        entry_signal_created_at=2_000,
        entry_signal_bridged_at=3_000,
        entry_order_posted_at=3_500,
    )
