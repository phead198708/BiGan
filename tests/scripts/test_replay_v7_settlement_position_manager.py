from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "replay_v7_settlement_position_manager.py"

spec = importlib.util.spec_from_file_location("replay_v7_settlement_position_manager", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def _args(**overrides: float | int) -> argparse.Namespace:
    defaults: dict[str, float | int] = {
        "round_cap_usdc": 1.0,
        "add_edge_min": 0.08,
        "full_add_edge": 0.20,
        "weak_hold_edge": 0.02,
        "reduce_fraction": 0.5,
        "exit_hold_edge": -0.02,
        "exit_hysteresis_bars": 2,
        "reversal_min_confidence": 0.75,
        "reversal_min_edge": 0.04,
        "reversal_hysteresis_bars": 2,
        "min_rebalance_usdc": 0.05,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _entry(*, round_slug: str = "btc-updown-15m-1", side: str = "UP") -> module.ActualEntry:
    return module.ActualEntry(
        round_slug=round_slug,
        side=side,
        canonical_symbol=f"{round_slug}-{side}",
        event_id=f"paper-{round_slug}-{side}",
        entry_price=0.50,
        shares=1.0,
        cost_basis_usdc=0.50,
        entry_signal_event_id="entry-signal",
        entry_signal_ts_ms=1_000,
        entry_signal_created_at_ms=1_100,
        opened_at_ms=1_200,
        p_up=0.80 if side == "UP" else 0.10,
        p_down=0.10 if side == "UP" else 0.80,
        p_neutral=0.10,
    )


def _signal(
    *,
    event_id: str,
    round_slug: str = "btc-updown-15m-1",
    side: str = "UP",
    ts_ms: int,
    created_at_ms: int,
    token_probability: float,
) -> module.SignalRow:
    p_up = token_probability if side == "UP" else 1.0 - token_probability
    p_down = token_probability if side == "DOWN" else 1.0 - token_probability
    return module.SignalRow(
        line_number=1,
        event_id=event_id,
        round_slug=round_slug,
        canonical_symbol=f"{round_slug}-{side}",
        side=side,
        ts_ms=ts_ms,
        created_at_ms=created_at_ms,
        bridged_at_ms=created_at_ms,
        round_end_ts_ms=ts_ms + 900_000,
        token_probability=token_probability,
        p_up=p_up,
        p_down=p_down,
        p_neutral=0.0,
        market_implied_prob=None,
        selected_expected_edge=None,
    )


def test_positive_edge_adds_toward_round_cap() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="strong-hold",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.90,
        )
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.60, ask=0.70),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(),
    )

    assert decisions[-1].action == "ADD"
    assert decisions[-1].reason == "positive_add_edge"
    assert decisions[-1].target_cost_basis_usdc > entry.cost_basis_usdc
    assert positions[0]["open"] is True
    assert positions[0]["remaining_cost_basis_usdc"] > entry.cost_basis_usdc


def test_weak_hold_edge_reduces_after_hysteresis() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="weak-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.52,
        ),
        _signal(
            event_id="weak-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.52,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.51, ask=0.54),
            module.Quote(ts_ms=3_050, bid=0.51, ask=0.54),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE"]
    assert decisions[-1].reason == "weak_hold_edge_reduce"
    assert positions[0]["open"] is True
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.25)


def test_confirmed_opposite_ev_reversal_exits_after_hysteresis() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="down-1",
            side="DOWN",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.82,
        ),
        _signal(
            event_id="down-2",
            side="DOWN",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.83,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.49, ask=0.52),
            module.Quote(ts_ms=3_050, bid=0.48, ask=0.51),
        ],
        "btc-updown-15m-1-DOWN": [
            module.Quote(ts_ms=2_050, bid=0.20, ask=0.70),
            module.Quote(ts_ms=3_050, bid=0.21, ask=0.70),
        ],
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "EXIT"]
    assert decisions[-1].reason == "confirmed_opposite_ev_reversal"
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(-0.02)
