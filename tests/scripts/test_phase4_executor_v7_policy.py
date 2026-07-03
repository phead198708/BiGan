from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "polymarket_phase4_live_champion_executor.py"

spec = importlib.util.spec_from_file_location("polymarket_phase4_live_champion_executor", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def _position() -> module.LivePosition:
    return module.LivePosition(
        event_id="position-1",
        round_slug="btc-updown-15m-1",
        side="UP",
        token_id="up-token",
        entry_price=0.45,
        fill_price=0.45,
        size=1.0,
        order_id="order-1",
        opened_at=1_000,
        entry_signal_event_id="entry-signal",
        entry_signal_ts=900,
        entry_signal_created_at=950,
        entry_signal_bridged_at=960,
        entry_order_posted_at=1_000,
        entry_p_up=0.62,
        entry_p_down=0.38,
        entry_model_probability=0.80,
        entry_polymarket_price=0.45,
        entry_mispricing_edge=0.35,
        paper=True,
    )


def _signal(
    *,
    token_probability: float,
    p_up: float,
    event_id: str = "candidate",
    ts: int = 2_000,
    created_at: int = 2_100,
    price: float = 0.40,
    edge: float | None = None,
    selected_confidence_score: float | None = None,
    selected_hit_5c_before_loss_10c: float | None = None,
    selected_hit_10c_before_loss_10c: float | None = None,
    selected_loss_10c_before_hit_5c: float | None = None,
) -> module.SignalEvent:
    if edge is None:
        edge = token_probability - price
    return module.SignalEvent(
        event_id=event_id,
        ts=ts,
        created_at=created_at,
        bridged_at=created_at,
        prob_up_15m=p_up,
        canonical_symbol="btc-updown-15m-1-UP",
        token_id="up-token",
        opposite_token_id="down-token",
        outcome_side="UP",
        selected_side="UP",
        round_slug="btc-updown-15m-1",
        round_end_ts=902_000,
        market_implied_prob=price,
        polymarket_price=price,
        token_probability=token_probability,
        model_probability=token_probability,
        edge=edge,
        p_up=p_up,
        p_down=1.0 - p_up,
        selected_confidence_score=selected_confidence_score,
        selected_hit_5c_before_loss_10c=selected_hit_5c_before_loss_10c,
        selected_hit_10c_before_loss_10c=selected_hit_10c_before_loss_10c,
        selected_loss_10c_before_hit_5c=selected_loss_10c_before_hit_5c,
    )


def test_post_take_profit_reentry_requires_stronger_confirmation() -> None:
    lifecycle = module.RoundLifecycleState()
    previous = _position()
    lifecycle.mark_entry_result(
        _signal(token_probability=0.80, p_up=0.62),
        previous,
    )
    lifecycle.mark_position_closed(
        previous.round_slug,
        "settlement",
        position=previous,
        reason="profit_protect_take_profit",
        realized_pnl=0.12,
        closed_at=3_000,
    )
    config = module.V7SettlementPositionConfig(
        post_take_profit_reentry_quality_enabled=True,
        post_take_profit_reentry_min_model_probability_improvement=0.03,
        post_take_profit_reentry_min_raw_probability_improvement=0.02,
    )

    skip_payload = module._v7_post_take_profit_reentry_skip_payload(
        lifecycle=lifecycle,
        signal=_signal(token_probability=0.81, p_up=0.63),
        config=config,
        seconds_to_expiry=600.0,
    )
    assert skip_payload is not None
    assert skip_payload["reason"] == "post_take_profit_reentry_quality_below_threshold"

    assert (
        module._v7_post_take_profit_reentry_skip_payload(
            lifecycle=lifecycle,
            signal=_signal(token_probability=0.81, p_up=0.65),
            config=config,
            seconds_to_expiry=600.0,
        )
        is None
    )


def test_low_confidence_scalp_take_profit_uses_near_profit_delta() -> None:
    config = module.V7SettlementPositionConfig(
        convergence_take_profit_enabled=True,
        take_profit_min_profit_delta=0.10,
        take_profit_min_profit_return=0.35,
        low_confidence_scalp_enabled=True,
        low_confidence_scalp_max_confidence_score=0.0,
        low_confidence_scalp_take_profit_min_profit_delta=0.05,
        low_confidence_scalp_take_profit_min_profit_return=0.10,
    )
    low_confidence = module._v7_low_confidence_scalp_profile(
        _signal(
            token_probability=0.72,
            p_up=0.58,
            selected_confidence_score=-0.20,
        ),
        config,
    )

    candidate, reason = module._v7_take_profit_exit_candidate(
        config=config,
        side="UP",
        hold_edge=0.20,
        hold_bid=0.55,
        avg_price=0.50,
        convergence={"available": False},
        seconds_to_expiry=600.0,
        low_confidence_scalp=low_confidence,
    )

    assert low_confidence["active"] is True
    assert candidate is True
    assert reason == "low_confidence_scalp_take_profit"


def test_low_confidence_scalp_adverse_can_upgrade_reduce_to_full_exit() -> None:
    config = module.V7SettlementPositionConfig(
        low_confidence_scalp_enabled=True,
        low_confidence_scalp_max_confidence_score=0.0,
        low_confidence_scalp_adverse_full_exit_enabled=True,
        adverse_confidence_reduce_min_model_decay=0.06,
    )
    low_confidence = module._v7_low_confidence_scalp_profile(
        _signal(
            token_probability=0.72,
            p_up=0.58,
            selected_confidence_score=-0.10,
        ),
        config,
    )
    high_confidence = module._v7_low_confidence_scalp_profile(
        _signal(
            token_probability=0.72,
            p_up=0.58,
            selected_confidence_score=0.20,
        ),
        config,
    )

    assert module._v7_low_confidence_adverse_full_exit_allowed(
        low_confidence_scalp=low_confidence,
        model_decay=0.06,
        config=config,
    )
    assert not module._v7_low_confidence_adverse_full_exit_allowed(
        low_confidence_scalp=low_confidence,
        model_decay=0.04,
        config=config,
    )
    assert not module._v7_low_confidence_adverse_full_exit_allowed(
        low_confidence_scalp=high_confidence,
        model_decay=0.06,
        config=config,
    )


def test_v7_entry_candidate_buffer_rejects_outside_price_band() -> None:
    buffer = module.V7EntryCandidateBuffer(
        module.V7EntryCandidateBufferConfig(
            enabled=True,
            min_price=0.40,
            max_price=0.70,
            min_edge=0.04,
        )
    )

    action = buffer.observe(
        _signal(token_probability=0.80, p_up=0.60, price=0.16),
        now_ms=2_000,
        seconds_to_expiry=600.0,
    )

    assert action.action == "skipped"
    assert action.reason == "v7_entry_candidate_price_below_band"


def test_v7_entry_candidate_buffer_releases_best_confidence_after_wait() -> None:
    buffer = module.V7EntryCandidateBuffer(
        module.V7EntryCandidateBufferConfig(
            enabled=True,
            max_wait_seconds=30.0,
            min_price=0.40,
            max_price=0.70,
            min_edge=0.04,
        )
    )
    weaker = _signal(
        token_probability=0.75,
        p_up=0.58,
        event_id="weaker",
        created_at=2_000,
        price=0.45,
        selected_confidence_score=-0.10,
        selected_hit_5c_before_loss_10c=0.45,
        selected_loss_10c_before_hit_5c=0.55,
    )
    stronger = _signal(
        token_probability=0.72,
        p_up=0.57,
        event_id="stronger",
        created_at=25_000,
        price=0.48,
        selected_confidence_score=0.25,
        selected_hit_5c_before_loss_10c=0.70,
        selected_loss_10c_before_hit_5c=0.45,
    )

    first_action = buffer.observe(weaker, now_ms=2_000, seconds_to_expiry=600.0)
    release_action = buffer.observe(stronger, now_ms=33_000, seconds_to_expiry=569.0)

    assert first_action.action == "buffered"
    assert release_action.action == "released"
    assert release_action.reason == "v7_entry_candidate_wait_elapsed"
    assert release_action.entry_event is stronger


def test_v7_entry_candidate_buffer_releases_best_fresh_candidate_after_wait() -> None:
    buffer = module.V7EntryCandidateBuffer(
        module.V7EntryCandidateBufferConfig(
            enabled=True,
            max_wait_seconds=30.0,
            min_price=0.40,
            max_price=0.70,
            min_edge=0.04,
        )
    )
    stale_best = _signal(
        token_probability=0.80,
        p_up=0.58,
        event_id="stale-best",
        ts=2_000,
        created_at=2_000,
        price=0.45,
        selected_confidence_score=0.90,
        selected_hit_5c_before_loss_10c=0.95,
        selected_loss_10c_before_hit_5c=0.05,
    )
    fresh_fallback = _signal(
        token_probability=0.72,
        p_up=0.57,
        event_id="fresh-fallback",
        ts=31_000,
        created_at=31_000,
        price=0.48,
        selected_confidence_score=0.20,
        selected_hit_5c_before_loss_10c=0.60,
        selected_loss_10c_before_hit_5c=0.40,
    )

    first_action = buffer.observe(
        stale_best,
        now_ms=2_000,
        seconds_to_expiry=600.0,
        max_signal_age_seconds=30.0,
    )
    release_action = buffer.observe(
        fresh_fallback,
        now_ms=33_000,
        seconds_to_expiry=569.0,
        max_signal_age_seconds=30.0,
    )

    assert first_action.action == "buffered"
    assert release_action.action == "released"
    assert release_action.reason == "v7_entry_candidate_wait_elapsed"
    assert release_action.entry_event is fresh_fallback
    assert release_action.stale_candidate_count == 1
    assert release_action.best_event_id == "fresh-fallback"


def test_v7_entry_candidate_buffer_skips_when_all_candidates_are_stale() -> None:
    buffer = module.V7EntryCandidateBuffer(
        module.V7EntryCandidateBufferConfig(
            enabled=True,
            max_wait_seconds=30.0,
            min_price=0.40,
            max_price=0.70,
            min_edge=0.04,
        )
    )

    action = buffer.observe(
        _signal(
            token_probability=0.80,
            p_up=0.58,
            event_id="stale-only",
            ts=2_000,
            created_at=2_000,
            price=0.45,
            selected_confidence_score=0.90,
        ),
        now_ms=40_000,
        seconds_to_expiry=600.0,
        max_signal_age_seconds=30.0,
    )

    assert action.action == "skipped"
    assert action.reason == "v7_entry_candidate_signal_age_above_threshold"
    assert action.entry_event is None
    assert action.stale_candidate_count == 1
    assert buffer.buckets == {}


def test_v7_entry_candidate_score_falls_back_to_side_specific_heads() -> None:
    event = module.SignalEvent(
        event_id="fallback",
        ts=2_000,
        created_at=2_100,
        prob_up_15m=0.60,
        canonical_symbol="btc-updown-15m-1-UP",
        token_id="up-token",
        outcome_side="UP",
        selected_side="UP",
        round_slug="btc-updown-15m-1",
        round_end_ts=902_000,
        market_implied_prob=0.45,
        token_probability=0.70,
        model_probability=0.70,
        edge=0.25,
        p_up=0.60,
        p_down=0.40,
        p_up_hit_5c_before_loss_10c=0.62,
        p_up_loss_10c_before_hit_5c=0.35,
    )

    assert module._v7_entry_candidate_score(event) == pytest.approx(0.27)


def test_v7_signal_payload_preserves_confidence_heads() -> None:
    event = module._event_from_signal_payload(
        {
            "event_id": "signal-1",
            "ts": 2_000,
            "created_at": 2_100,
            "model_version": "xgboost-v7",
            "prob_up_15m": 0.61,
            "canonical_symbol": "BTC-15M:btc-updown-15m-1:UP",
            "token_id": "up-token",
            "outcome_side": "UP",
            "round_slug": "btc-updown-15m-1",
            "round_end_ts": 902_000,
            "market_implied_prob": 0.45,
            "model_probability": 0.74,
            "polymarket_price": 0.45,
            "mispricing_edge": 0.29,
            "p_up": 0.61,
            "p_down": 0.39,
            "selected_hit_5c_before_loss_10c": 0.68,
            "selected_hit_10c_before_loss_10c": 0.52,
            "selected_loss_10c_before_hit_5c": 0.31,
            "selected_confidence_score": 0.37,
        },
        model_version="xgboost-v7",
    )

    assert event is not None
    assert event.selected_hit_5c_before_loss_10c == pytest.approx(0.68)
    assert event.selected_hit_10c_before_loss_10c == pytest.approx(0.52)
    assert event.selected_loss_10c_before_hit_5c == pytest.approx(0.31)
    assert event.selected_confidence_score == pytest.approx(0.37)
