from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "polymarket_phase1_safety_check.py"

spec = importlib.util.spec_from_file_location("polymarket_phase1_safety_check", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def _book(*, bid: float | None, ask: float | None):
    return module.BookSummary(token_id="token", best_bid=bid, best_ask=ask)


def test_rejects_near_expiry_markets() -> None:
    decision = module.evaluate_buy_safety(
        side="BUY_UP",
        proposed_price=0.10,
        same_outcome=_book(bid=0.50, ask=0.60),
        complement_outcome=_book(bid=0.30, ask=0.40),
        tick=0.01,
        tick_buffer=3,
        complement_buffer=3,
        seconds_to_expiry=120,
        min_seconds_to_expiry=600,
    )

    assert decision.status == "FAIL"
    assert "expires" in decision.reason


def test_rejects_price_that_can_match_complementary_buy_liquidity() -> None:
    decision = module.evaluate_buy_safety(
        side="BUY_UP",
        proposed_price=0.95,
        same_outcome=_book(bid=0.96, ask=0.97),
        complement_outcome=_book(bid=0.05, ask=0.06),
        tick=0.01,
        tick_buffer=3,
        complement_buffer=3,
        seconds_to_expiry=900,
        min_seconds_to_expiry=600,
    )

    assert decision.status == "FAIL"
    assert decision.complement_buy_ceiling == 0.92
    assert "complementary" in decision.reason


def test_accepts_price_below_same_and_complementary_ceiling() -> None:
    decision = module.evaluate_buy_safety(
        side="BUY_UP",
        proposed_price=0.10,
        same_outcome=_book(bid=0.50, ask=0.60),
        complement_outcome=_book(bid=0.70, ask=0.80),
        tick=0.01,
        tick_buffer=3,
        complement_buffer=3,
        seconds_to_expiry=900,
        min_seconds_to_expiry=600,
    )

    assert decision.status == "PASS"
    assert decision.safe_ceiling == 0.27
