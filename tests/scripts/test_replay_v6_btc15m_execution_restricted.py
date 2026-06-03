from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v6_btc15m_execution_restricted.py"

spec = importlib.util.spec_from_file_location("replay_v6_btc15m_execution_restricted", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_execution_restricted_allows_cheap_settlement_when_cost_edge_clears(
    tmp_path: Path,
) -> None:
    cheap_row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-1000:UP",
        "feature_ts": 1_100_000,
        "round_end_ts": 1_900_000,
        "entry_ask_price": 0.30,
        "label_settlement_3way": "UP",
        "max_exit_gain_up": 0.20,
        "max_exit_gain_down": 0.20,
        "market_implied_prob": 0.30,
    }
    allowed_row = {
        **cheap_row,
        "canonical_symbol": "BTC-15M:btc-updown-15m-2000:UP",
        "feature_ts": 2_100_000,
        "round_end_ts": 2_900_000,
        "entry_ask_price": 0.40,
    }
    payload = {
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.05,
        "p_vol_up": 0.90,
        "p_vol_down": 0.10,
    }
    joint_rule = {
        "settlement_threshold": 0.50,
        "neutral_cap": 0.25,
        "volatility_threshold": 0.50,
    }
    policy = module.Phase4EntryPolicy(min_entry_price=0.35)
    gate_counts: dict[str, int] = {}
    trades = module._collect_execution_restricted_trades(
        [cheap_row, allowed_row],
        [payload, payload],
        joint_rule=joint_rule,
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=policy,
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts=gate_counts,
    )
    assert len(trades) == 2
    assert "entry_price_below_min" not in gate_counts
    assert trades[0].entry_worst_price == 0.32


def test_execution_restricted_admits_profitable_round(tmp_path: Path) -> None:
    row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-2000:UP",
        "feature_ts": 2_100_000,
        "round_end_ts": 2_900_000,
        "entry_ask_price": 0.40,
        "label_settlement_3way": "UP",
        "max_exit_gain_up": 0.20,
        "market_implied_prob": 0.40,
    }
    payload = {
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.05,
        "p_vol_up": 0.90,
        "p_vol_down": 0.10,
    }
    trades = module._collect_execution_restricted_trades(
        [row],
        [payload],
        joint_rule={
            "settlement_threshold": 0.50,
            "neutral_cap": 0.25,
            "volatility_threshold": 0.50,
        },
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=module.Phase4EntryPolicy(min_entry_price=0.35),
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts={},
    )
    assert len(trades) == 1
    assert trades[0].pnl > 0.0


def test_execution_restricted_normalizes_down_token_payload() -> None:
    row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-2000:DOWN",
        "feature_ts": 2_100_000,
        "round_end_ts": 2_900_000,
        "entry_ask_price": 0.20,
        "label_settlement_3way": "DOWN",
        "market_implied_prob": 0.20,
    }
    payload = {
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.05,
        "p_vol_up": 0.80,
        "p_vol_down": 0.10,
    }

    trades = module._collect_execution_restricted_trades(
        [row],
        [payload],
        joint_rule={
            "settlement_threshold": 0.50,
            "neutral_cap": 0.25,
            "volatility_threshold": 0.50,
        },
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=module.Phase4EntryPolicy(settlement_edge_threshold=0.082),
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts={},
    )

    assert len(trades) == 1
    assert trades[0].side == "DOWN"
    assert trades[0].entry_worst_price == 0.22
    assert trades[0].pnl > 0.0


def test_execution_restricted_blocks_cost_negative_settlement_edge() -> None:
    row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-2500:UP",
        "feature_ts": 2_600_000,
        "round_end_ts": 3_400_000,
        "entry_ask_price": 0.80,
        "label_settlement_3way": "UP",
        "market_implied_prob": 0.80,
    }
    payload = {
        "p_up": 0.85,
        "p_down": 0.10,
        "p_neutral": 0.05,
        "p_vol_up": 0.01,
        "p_vol_down": 0.99,
    }
    gate_counts: dict[str, int] = {}
    trades = module._collect_execution_restricted_trades(
        [row],
        [payload],
        joint_rule={
            "settlement_threshold": 0.50,
            "neutral_cap": 0.25,
            "volatility_threshold": 0.60,
        },
        round_trip_cost=0.072,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=module.Phase4EntryPolicy(
            min_entry_price=0.35,
            settlement_edge_threshold=0.082,
        ),
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts=gate_counts,
    )

    assert trades == []
    assert gate_counts["fresh_edge_below_threshold"] == 1


def test_execution_restricted_settlement_gate_ignores_volatility_head() -> None:
    row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-3000:UP",
        "feature_ts": 3_100_000,
        "round_end_ts": 3_900_000,
        "entry_ask_price": 0.40,
        "label_settlement_3way": "UP",
        "market_implied_prob": 0.40,
    }
    payload = {
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.95,
        "p_vol_up": 0.01,
        "p_vol_down": 0.99,
    }
    gate_counts: dict[str, int] = {}
    trades = module._collect_execution_restricted_trades(
        [row],
        [payload],
        joint_rule={
            "settlement_threshold": 0.50,
            "neutral_cap": 0.25,
            "volatility_threshold": 0.60,
        },
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=module.Phase4EntryPolicy(min_entry_price=0.35),
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts=gate_counts,
    )

    assert len(trades) == 1
    assert gate_counts == {}


def test_round_first_blocks_second_trade_in_same_round() -> None:
    row = {
        "canonical_symbol": "BTC-15M:btc-updown-15m-4000:UP",
        "feature_ts": 4_100_000,
        "round_end_ts": 4_900_000,
        "entry_ask_price": 0.40,
        "label_settlement_3way": "UP",
        "max_exit_gain_up": 0.20,
        "market_implied_prob": 0.40,
    }
    payload = {
        "p_up": 0.90,
        "p_down": 0.05,
        "p_neutral": 0.05,
        "p_vol_up": 0.90,
        "p_vol_down": 0.10,
    }
    gate_counts: dict[str, int] = {}
    trades = module._collect_execution_restricted_trades(
        [row, row],
        [payload, payload],
        joint_rule={
            "settlement_threshold": 0.50,
            "neutral_cap": 0.25,
            "volatility_threshold": 0.50,
        },
        round_trip_cost=0.04,
        ev_margin=0.01,
        gain_priors={"up": 0.30, "down": 0.30},
        policy=module.Phase4EntryPolicy(min_entry_price=0.35),
        buy_slippage=0.02,
        min_seconds_to_expiry=300.0,
        max_seconds_to_expiry=1200.0,
        no_new_entry_before_expiry_seconds=300.0,
        gate_counts=gate_counts,
    )
    assert len(trades) == 1
    assert gate_counts["round_first_blocked"] == 1
