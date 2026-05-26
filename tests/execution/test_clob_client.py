"""CLOB execution client tests for issue #72."""

from __future__ import annotations

import pytest

from bigan.execution import (
    ClobExecutionClient,
    ClobExecutionConfig,
    InsufficientBalanceError,
    OrderStatus,
)


def _config(**overrides: object) -> ClobExecutionConfig:
    return ClobExecutionConfig(
        dry_run=False,
        max_retries=3,
        retry_base_delay_seconds=0.0,
        min_request_interval_seconds=0.0,
        **overrides,
    )


def test_dry_run_place_cancel_status_never_calls_real_api() -> None:
    client = ClobExecutionClient(config=ClobExecutionConfig(dry_run=True))

    order_id = client.place_limit_order("token-1", "BUY", 0.51, 2.0)

    assert order_id.startswith("dryrun-")
    assert client.dry_run_orders[0]["token_id"] == "token-1"
    assert client.cancel_order(order_id) is True
    assert client.get_order_status(order_id) == OrderStatus(order_id=order_id, status="dry_run")
    assert client.get_best_bid_ask("token-1") == (None, None)


def test_place_limit_order_retries_transient_timeout() -> None:
    class Api:
        def __init__(self) -> None:
            self.calls = 0

        def place_limit_order(self, *, token_id: str, side: str, price: float, size: float) -> dict[str, str]:
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("temporary timeout")
            return {"order_id": f"{token_id}-{side}-{price}-{size}"}

    api = Api()
    client = ClobExecutionClient(api, config=_config())

    order_id = client.place_limit_order("token-2", "SELL", 0.42, 1.5)

    assert order_id == "token-2-SELL-0.42-1.5"
    assert api.calls == 3


def test_insufficient_balance_is_not_retried() -> None:
    class Api:
        def __init__(self) -> None:
            self.calls = 0

        def place_limit_order(self, *args: object, **kwargs: object) -> dict[str, str]:
            self.calls += 1
            raise RuntimeError("insufficient balance for order")

    api = Api()
    client = ClobExecutionClient(api, config=_config())

    with pytest.raises(InsufficientBalanceError):
        client.place_limit_order("token-3", "BUY", 0.50, 1.0)

    assert api.calls == 1


def test_status_and_best_bid_ask_are_normalized_from_client_payloads() -> None:
    class Api:
        def get_order(self, order_id: str) -> dict[str, object]:
            return {
                "id": order_id,
                "status": "partially_filled",
                "filled_size": "0.75",
                "remaining_size": "0.25",
                "avg_fill_price": "0.49",
            }

        def get_order_book(self, token_id: str) -> dict[str, object]:
            return {
                "bids": [{"price": "0.47"}, {"price": "0.48"}],
                "asks": [{"price": "0.52"}, {"price": "0.51"}],
            }

    client = ClobExecutionClient(Api(), config=_config())

    status = client.get_order_status("order-1")
    assert status.status == "partially_filled"
    assert status.filled_size == pytest.approx(0.75)
    assert status.remaining_size == pytest.approx(0.25)
    assert status.avg_fill_price == pytest.approx(0.49)
    assert client.get_best_bid_ask("token-4") == (0.48, 0.51)
