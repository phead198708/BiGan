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


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "round_cap_usdc": 1.0,
        "add_edge_min": 0.08,
        "full_add_edge": 0.20,
        "weak_hold_edge": 0.02,
        "reduce_fraction": 0.5,
        "divergence_reduce_max_hold_edge": 0.08,
        "exit_hold_edge": -0.02,
        "exit_hysteresis_bars": 2,
        "reversal_min_confidence": 0.75,
        "reversal_min_edge": 0.04,
        "reversal_hysteresis_bars": 2,
        "min_rebalance_usdc": 0.05,
        "convergence_price_tolerance": 0.02,
        "convergence_model_decay_tolerance": 0.10,
        "divergence_hysteresis_bars": 2,
        "add_cooldown_after_divergence_reduce_seconds": 120.0,
        "take_profit_enabled": False,
        "take_profit_hold_edge": 0.03,
        "take_profit_residual_ratio": 0.40,
        "take_profit_price_convergence_move": 0.10,
        "take_profit_price_convergence_hold_edge_ratio": 0.50,
        "take_profit_force_exit_seconds": 180.0,
        "take_profit_hysteresis_bars": 2,
        "take_profit_up_hold_edge_tighten": 0.01,
        "take_profit_min_profit_delta": 0.10,
        "take_profit_min_profit_return": 0.35,
        "adverse_confidence_decay_enabled": False,
        "adverse_confidence_price_delta_start": 0.10,
        "adverse_confidence_base_allowed_decay": 0.08,
        "adverse_confidence_price_decay_slope": 0.30,
        "adverse_confidence_min_allowed_decay": 0.015,
        "adverse_confidence_max_required_probability": 0.97,
        "adverse_confidence_exit_probability_buffer": 0.03,
        "adverse_confidence_full_exit_min_model_decay": 0.06,
        "adverse_confidence_full_exit_max_hold_edge": 0.25,
        "adverse_confidence_reduce_min_model_decay": 0.06,
        "adverse_confidence_dust_exit_max_cost": 0.15,
        "adverse_confidence_dust_exit_min_candidate_count": 3,
        "adverse_confidence_hysteresis_bars": 2,
        "adverse_confidence_max_reduces": 0,
        "adverse_confidence_post_reduce_full_exit_enabled": False,
        "adverse_confidence_post_reduce_full_exit_bars": 1,
        "adverse_confidence_post_reduce_full_exit_min_model_decay": 0.06,
        "adverse_confidence_post_reduce_full_exit_max_hold_edge": -1.0,
        "block_add_after_adverse_confidence_reduce": False,
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
        entry_model_probability=0.80 if side == "UP" else 0.80,
        entry_polymarket_price=0.50,
        entry_mispricing_edge=0.30,
    )


def _signal(
    *,
    event_id: str,
    round_slug: str = "btc-updown-15m-1",
    side: str = "UP",
    ts_ms: int,
    created_at_ms: int,
    token_probability: float,
    polymarket_price: float | None = None,
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
        model_probability=token_probability,
        polymarket_price=polymarket_price,
        mispricing_edge=(
            None if polymarket_price is None else token_probability - polymarket_price
        ),
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


def test_price_convergence_take_profit_exits_before_add() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="converged-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.82,
            polymarket_price=0.62,
        ),
        _signal(
            event_id="converged-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.82,
            polymarket_price=0.62,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.69, ask=0.71),
            module.Quote(ts_ms=3_050, bid=0.69, ask=0.71),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(
            take_profit_enabled=True,
            take_profit_min_profit_delta=0.0,
            take_profit_min_profit_return=0.0,
        ),
    )

    assert decisions[-1].action == "EXIT"
    assert decisions[-1].reason == "convergence_price_move_take_profit"
    assert decisions[-1].take_profit_count == 2
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(0.19)


def test_profit_protect_take_profit_exits_even_when_hold_edge_is_high() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="profit-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.90,
        ),
        _signal(
            event_id="profit-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.90,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.62, ask=0.64),
            module.Quote(ts_ms=3_050, bid=0.62, ask=0.64),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(take_profit_enabled=True),
    )

    assert decisions[-1].action == "EXIT"
    assert decisions[-1].reason == "profit_protect_take_profit"
    assert decisions[-1].hold_edge == pytest.approx(0.28)
    assert decisions[-1].take_profit_count == 2
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(0.12)


def test_late_force_exit_reason_marks_loss_salvage() -> None:
    candidate, reason = module._take_profit_candidate(
        side="UP",
        hold_edge=0.25,
        hold_bid=0.20,
        avg_price=0.50,
        convergence={"available": True},
        seconds_to_expiry=120.0,
        args=_args(take_profit_enabled=True),
    )

    assert candidate is True
    assert reason == "convergence_loss_salvage_before_expiry"


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


def test_residual_divergence_blocks_reduce_when_hold_edge_healthy() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="diverge-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.37,
            polymarket_price=0.16,
        ),
        _signal(
            event_id="diverge-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.37,
            polymarket_price=0.16,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.15, ask=0.16),
            module.Quote(ts_ms=3_050, bid=0.15, ask=0.16),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(),
    )

    assert decisions[0].reason == "positive_add_edge_blocked_by_residual_divergence"
    assert decisions[0].convergence["price_diverged"] is True
    assert [decision.action for decision in decisions] == ["HOLD", "HOLD"]
    assert decisions[-1].reason == "residual_divergence_reduce_blocked_by_hold_edge"
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.50)


def test_adverse_confidence_decay_reduces_when_hold_edge_is_artificially_high() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="adverse-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.70,
            polymarket_price=0.20,
        ),
        _signal(
            event_id="adverse-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.70,
            polymarket_price=0.20,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.20, ask=0.21),
            module.Quote(ts_ms=3_050, bid=0.20, ask=0.21),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(adverse_confidence_decay_enabled=True),
    )

    assert decisions[0].action == "HOLD"
    assert decisions[0].reason == "adverse_confidence_decay_hysteresis_wait"
    assert decisions[0].hold_edge == pytest.approx(0.50)
    assert decisions[0].adverse_confidence["required_p_side"] == pytest.approx(0.785)
    assert decisions[-1].action == "REDUCE"
    assert decisions[-1].reason == "adverse_confidence_decay_reduce"
    assert decisions[-1].adverse_confidence["full_exit_allowed"] is False
    assert decisions[-1].adverse_confidence_count == 2
    assert positions[0]["open"] is True
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.25)
    assert positions[0]["realized_pnl"] == pytest.approx(-0.15)


def test_adverse_confidence_reduce_waits_for_material_model_decay() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="small-decay-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.76,
            polymarket_price=0.20,
        ),
        _signal(
            event_id="small-decay-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.76,
            polymarket_price=0.20,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.20, ask=0.21),
            module.Quote(ts_ms=3_050, bid=0.20, ask=0.21),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(adverse_confidence_decay_enabled=True),
    )

    assert decisions[0].reason == "adverse_confidence_decay_hysteresis_wait"
    assert decisions[-1].action == "HOLD"
    assert decisions[-1].reason == "adverse_confidence_reduce_blocked_by_model_decay"
    assert decisions[-1].adverse_confidence["model_decay"] == pytest.approx(0.04)
    assert decisions[-1].adverse_confidence["reduce_model_decay_allowed"] is False
    assert decisions[-1].adverse_confidence_reduce_allowed is False
    assert positions[0]["open"] is True
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.50)
    assert positions[0]["realized_pnl"] == pytest.approx(0.0)


def test_adverse_confidence_max_reduces_blocks_repeated_loss_slicing() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id=f"adverse-{idx}",
            side="UP",
            ts_ms=2_000 + idx * 1_000,
            created_at_ms=2_100 + idx * 1_000,
            token_probability=0.70,
            polymarket_price=0.20,
        )
        for idx in range(4)
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050 + idx * 1_000, bid=0.20, ask=0.21)
            for idx in range(4)
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(
            adverse_confidence_decay_enabled=True,
            adverse_confidence_dust_exit_max_cost=0.0,
            adverse_confidence_max_reduces=1,
        ),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE", "HOLD", "HOLD"]
    assert decisions[2].reason == "adverse_confidence_reduce_blocked_by_max_reduces"
    assert decisions[2].adverse_confidence_reduce_allowed is False
    assert positions[0]["adverse_confidence_reduce_count"] == 1
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.25)


def test_adverse_confidence_post_reduce_upgrade_full_exits() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id=f"adverse-{idx}",
            side="UP",
            ts_ms=2_000 + idx * 1_000,
            created_at_ms=2_100 + idx * 1_000,
            token_probability=0.70,
            polymarket_price=0.20,
        )
        for idx in range(3)
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050 + idx * 1_000, bid=0.20, ask=0.21)
            for idx in range(3)
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(
            adverse_confidence_decay_enabled=True,
            adverse_confidence_dust_exit_max_cost=0.0,
            adverse_confidence_max_reduces=1,
            adverse_confidence_post_reduce_full_exit_enabled=True,
        ),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE", "EXIT"]
    assert decisions[-1].reason == "adverse_confidence_post_reduce_full_exit"
    assert decisions[-1].adverse_confidence["post_reduce_full_exit_allowed"] is True
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(-0.30)


def test_adverse_confidence_reduce_blocks_later_add() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="adverse-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.70,
            polymarket_price=0.20,
        ),
        _signal(
            event_id="adverse-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.70,
            polymarket_price=0.20,
        ),
        _signal(
            event_id="post-adverse-add",
            side="UP",
            ts_ms=63_000,
            created_at_ms=63_100,
            token_probability=0.90,
            polymarket_price=0.70,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.20, ask=0.21),
            module.Quote(ts_ms=3_050, bid=0.20, ask=0.21),
            module.Quote(ts_ms=63_050, bid=0.60, ask=0.70),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(
            adverse_confidence_decay_enabled=True,
            block_add_after_adverse_confidence_reduce=True,
        ),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE", "HOLD"]
    assert decisions[-1].reason == "positive_add_edge_blocked_by_adverse_confidence_reduce"
    assert decisions[-1].adverse_confidence_add_blocked is True
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.25)


def test_adverse_confidence_dust_exit_cleans_repeated_reduce_tail() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id=f"adverse-{idx}",
            side="UP",
            ts_ms=2_000 + idx * 1_000,
            created_at_ms=2_100 + idx * 1_000,
            token_probability=0.70,
            polymarket_price=0.20,
        )
        for idx in range(4)
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050 + idx * 1_000, bid=0.20, ask=0.21)
            for idx in range(4)
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(adverse_confidence_decay_enabled=True),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE", "EXIT"]
    assert decisions[-1].reason == "adverse_confidence_dust_exit"
    assert decisions[-1].adverse_confidence["dust_exit_allowed"] is True
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(-0.30)


def test_adverse_confidence_decay_full_exits_when_model_decay_is_large_and_edge_is_weak() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="adverse-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.55,
            polymarket_price=0.40,
        ),
        _signal(
            event_id="adverse-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.55,
            polymarket_price=0.40,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.40, ask=0.41),
            module.Quote(ts_ms=3_050, bid=0.40, ask=0.41),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(adverse_confidence_decay_enabled=True),
    )

    assert decisions[-1].action == "EXIT"
    assert decisions[-1].reason == "adverse_confidence_decay_exit"
    assert decisions[-1].hold_edge == pytest.approx(0.15)
    assert decisions[-1].adverse_confidence["full_exit_allowed"] is True
    assert positions[0]["open"] is False
    assert positions[0]["realized_pnl"] == pytest.approx(-0.10)


def test_residual_divergence_reduces_when_hold_edge_is_weak() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="diverge-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.20,
            polymarket_price=0.16,
        ),
        _signal(
            event_id="diverge-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.20,
            polymarket_price=0.16,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.15, ask=0.16),
            module.Quote(ts_ms=3_050, bid=0.15, ask=0.16),
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
    assert decisions[-1].reason == "residual_divergence_reduce"
    assert decisions[-1].divergence_reduce_allowed is True
    assert positions[0]["remaining_cost_basis_usdc"] == pytest.approx(0.25)


def test_add_cooldown_blocks_add_after_divergence_reduce() -> None:
    entry = _entry()
    signals = [
        _signal(
            event_id="diverge-1",
            side="UP",
            ts_ms=2_000,
            created_at_ms=2_100,
            token_probability=0.20,
            polymarket_price=0.16,
        ),
        _signal(
            event_id="diverge-2",
            side="UP",
            ts_ms=3_000,
            created_at_ms=3_100,
            token_probability=0.20,
            polymarket_price=0.16,
        ),
        _signal(
            event_id="post-reduce-add",
            side="UP",
            ts_ms=63_000,
            created_at_ms=63_100,
            token_probability=0.90,
            polymarket_price=0.70,
        ),
    ]
    quotes = {
        "btc-updown-15m-1-UP": [
            module.Quote(ts_ms=2_050, bid=0.15, ask=0.16),
            module.Quote(ts_ms=3_050, bid=0.15, ask=0.16),
            module.Quote(ts_ms=63_050, bid=0.60, ask=0.70),
        ]
    }

    decisions, positions = module._replay_positions(
        entries=[entry],
        signals=signals,
        quotes=quotes,
        outcomes={},
        args=_args(),
    )

    assert [decision.action for decision in decisions] == ["HOLD", "REDUCE", "HOLD"]
    assert decisions[-1].reason == "positive_add_edge_blocked_by_divergence_reduce_cooldown"
    assert decisions[-1].add_cooldown_remaining_seconds == pytest.approx(60.0)
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
