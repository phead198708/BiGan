from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from pytest import approx

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "analyze_phase4_signal_opportunities.py"

spec = importlib.util.spec_from_file_location("analyze_phase4_signal_opportunities", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_analyze_signal_opportunities_labels_volatility_and_settlement(tmp_path: Path) -> None:
    signals_path = tmp_path / "signals.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    base = 1_780_000_000_000
    round_end = base + 900_000
    signals = [
        {
            "event_id": "pred-wrong-but-tradable",
            "ts": base + 60_000,
            "created_at": base + 61_000,
            "bridged_at": base + 62_000,
            "model_version": "xgboost-v5",
            "canonical_symbol": "BTC-15M:round-1:UP",
            "token_id": "up-token",
            "outcome_side": "UP",
            "round_slug": "round-1",
            "round_end_ts": round_end,
            "market_implied_prob": 0.40,
            "token_probability": 0.90,
            "edge": 0.50,
        },
        {
            "event_id": "pred-settlement-only",
            "ts": base + 120_000,
            "created_at": base + 121_000,
            "bridged_at": base + 122_000,
            "model_version": "xgboost-v5",
            "canonical_symbol": "BTC-15M:round-1:DOWN",
            "token_id": "down-token",
            "outcome_side": "DOWN",
            "round_slug": "round-1",
            "round_end_ts": round_end,
            "market_implied_prob": 0.55,
            "token_probability": 0.95,
            "edge": 0.40,
        },
        {
            "event_id": "pred-blocked-but-tradable",
            "ts": base + 180_000,
            "created_at": base + 181_000,
            "bridged_at": base + 182_000,
            "model_version": "xgboost-v5",
            "canonical_symbol": "BTC-15M:round-2:UP",
            "token_id": "cheap-token",
            "outcome_side": "UP",
            "round_slug": "round-2",
            "round_end_ts": round_end + 300_000,
            "market_implied_prob": 0.10,
            "token_probability": 0.80,
            "edge": 0.70,
        },
    ]
    signals_path.write_text(
        "".join(json.dumps(signal) + "\n" for signal in signals),
        encoding="utf-8",
    )
    raw_rows = [
        _top_of_book(base + 62_000, "BTC-15M:round-1:UP", 0.39, 0.40),
        _top_of_book(base + 300_000, "BTC-15M:round-1:UP", 0.60, 0.61),
        _top_of_book(base + 122_000, "BTC-15M:round-1:DOWN", 0.54, 0.55),
        _top_of_book(base + 300_000, "BTC-15M:round-1:DOWN", 0.57, 0.58),
        _top_of_book(base + 182_000, "BTC-15M:round-2:UP", 0.08, 0.10),
        _top_of_book(base + 600_000, "BTC-15M:round-2:UP", 0.33, 0.34),
    ]
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )

    loaded_signals = module.load_signals_jsonl(signals_path)
    quotes = module.load_quotes([raw_path], signals=loaded_signals)
    rows = module.analyze_signals(
        loaded_signals,
        quotes_by_symbol=quotes,
        outcomes_by_event_id={
            "pred-wrong-but-tradable": False,
            "pred-settlement-only": True,
        },
        max_entry_wait_ms=60_000,
        min_exit_seconds_before_expiry=300.0,
        min_exit_gain=0.15,
        edge_threshold=0.45,
        min_entry_price=0.35,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        buy_slippage=0.02,
        sell_slippage=0.02,
        soft_exit_before_expiry_seconds=240.0,
        hard_exit_before_expiry_seconds=120.0,
        infer_settlement_from_final_book=True,
        settlement_win_bid_threshold=0.98,
        settlement_loss_ask_threshold=0.02,
    )
    by_event = {row.event_id: row for row in rows}

    assert by_event["pred-wrong-but-tradable"].volatility_exit_opportunity is True
    assert by_event["pred-wrong-but-tradable"].realized_label is False
    assert by_event["pred-wrong-but-tradable"].first_profitable_exit_seconds == 238.0
    assert by_event["pred-wrong-but-tradable"].max_drawdown_before_profit == approx(0.05)
    assert (
        by_event["pred-wrong-but-tradable"].opportunity_class
        == "wrong_outcome_but_volatility_exit"
    )
    assert by_event["pred-settlement-only"].settlement_hold_opportunity is True
    assert by_event["pred-settlement-only"].volatility_exit_opportunity is False
    assert by_event["pred-settlement-only"].policy_gate == "below_edge_threshold"
    assert by_event["pred-blocked-but-tradable"].policy_gate == "entry_price_below_min"
    assert by_event["pred-blocked-but-tradable"].volatility_exit_opportunity is True

    summary = module.summarize(rows, edge_thresholds=[0.45, 0.30])
    assert summary["volatility_exit_opportunities"] == 2
    assert summary["settlement_hold_opportunities"] == 1
    assert summary["wrong_outcome_but_volatility_exit"] == 1
    assert summary["blocked_by_policy_but_volatility_exit"] == 1
    assert (
        summary["policy_gate_table"]["below_edge_threshold"]["settlement_hold_opportunities"]
        == 1
    )
    assert (
        summary["gating_confusion_by_opportunity_type"]["volatility_exit"][
            "false_negative_blocked_opportunity"
        ]
        == 1
    )
    assert (
        summary["gating_confusion_by_opportunity_type"]["settlement_hold"][
            "false_negative_blocked_opportunity"
        ]
        == 1
    )
    assert summary["edge_threshold_sweep"][0]["volatility_opportunities_allowed"] == 1
    assert summary["edge_threshold_sweep"][0]["settlement_opportunities_allowed"] == 0
    assert summary["edge_threshold_sweep"][1]["settlement_opportunities_allowed"] == 1


def test_analyze_signal_opportunities_can_infer_settlement_from_final_book(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    base = 1_780_000_000_000
    round_end = base + 900_000
    signal = module.Signal(
        event_id="pred-inferred-win",
        ts=base + 60_000,
        created_at=base + 61_000,
        bridged_at=base + 62_000,
        model_version="xgboost-v5",
        canonical_symbol="BTC-15M:round-1:DOWN",
        token_id="down-token",
        outcome_side="DOWN",
        round_slug="round-1",
        round_end_ts=round_end,
        market_implied_prob=0.40,
        token_probability=0.90,
        edge=0.50,
    )
    raw_rows = [
        _top_of_book(base + 62_000, signal.canonical_symbol, 0.39, 0.40),
        _top_of_book(round_end - 1_000, signal.canonical_symbol, 0.99, 1.00),
    ]
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )

    quotes = module.load_quotes([raw_path], signals=[signal])
    rows = module.analyze_signals(
        [signal],
        quotes_by_symbol=quotes,
        outcomes_by_event_id={},
        max_entry_wait_ms=60_000,
        min_exit_seconds_before_expiry=300.0,
        min_exit_gain=0.15,
        edge_threshold=0.45,
        min_entry_price=0.35,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        buy_slippage=0.02,
        sell_slippage=0.02,
        soft_exit_before_expiry_seconds=240.0,
        hard_exit_before_expiry_seconds=120.0,
        infer_settlement_from_final_book=True,
        settlement_win_bid_threshold=0.98,
        settlement_loss_ask_threshold=0.02,
    )

    assert rows[0].inferred_settlement_label is True
    assert rows[0].effective_settlement_label is True
    assert rows[0].settlement_hold_opportunity is True
    assert rows[0].soft_exit_profitable is True


def test_load_quotes_reads_ws_market_lines(tmp_path: Path) -> None:
    raw_path = tmp_path / "ws.ndjson"
    signal = module.Signal(
        event_id="pred-ws",
        ts=1000,
        created_at=1000,
        bridged_at=1000,
        model_version="xgboost-v5",
        canonical_symbol="BTC-15M:round-1:DOWN",
        token_id="down-token",
        outcome_side="DOWN",
        round_slug="round-1",
        round_end_ts=10_000,
        market_implied_prob=0.40,
        token_probability=0.90,
        edge=0.50,
    )
    raw_rows = [
        {
            "raw": {
                "event_type": "book",
                "timestamp": "1000",
                "asset_id": "down-token",
                "bids": [{"price": "0.30"}, {"price": "0.35"}],
                "asks": [{"price": "0.44"}, {"price": "0.42"}],
            }
        },
        {
            "raw": {
                "event_type": "price_change",
                "timestamp": "2000",
                "price_changes": [
                    {
                        "asset_id": "down-token",
                        "best_bid": "0.50",
                        "best_ask": "0.52",
                    }
                ],
            }
        },
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in raw_rows), encoding="utf-8")

    quotes = module.load_quotes([raw_path], signals=[signal])

    assert [(quote.ts, quote.bid, quote.ask) for quote in quotes[signal.canonical_symbol]] == [
        (1000, 0.35, 0.42),
        (2000, 0.50, 0.52),
    ]


def _top_of_book(ts: int, canonical_symbol: str, bid: float, ask: float) -> dict:
    return {
        "table": "raw_top_of_book",
        "published_at_ms": ts,
        "row": {
            "ts": ts,
            "message_ts": ts,
            "ingest_ts": ts,
            "source": "polymarket",
            "source_symbol": canonical_symbol.rsplit(":", 1)[-1],
            "source_market": "0xmarket",
            "canonical_symbol": canonical_symbol,
            "bid_price": bid,
            "ask_price": ask,
            "spread": ask - bid,
        },
    }
