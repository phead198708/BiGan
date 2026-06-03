"""CLOB execution client tests for issue #72."""

from __future__ import annotations

import sys
import types

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


def test_builds_v2_client_with_proxy_signature_and_derived_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    class SignatureTypeV2(int):
        EOA = 0
        POLY_PROXY = 1
        POLY_GNOSIS_SAFE = 2
        POLY_1271 = 3

    class ApiCreds:
        def __init__(self, api_key: str, api_secret: str, api_passphrase: str) -> None:
            self.api_key = api_key
            self.api_secret = api_secret
            self.api_passphrase = api_passphrase

    class FakeClobClient:
        instances: list[FakeClobClient] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.derived = False
            self.creds: ApiCreds | None = None
            FakeClobClient.instances.append(self)

        def create_or_derive_api_key(self) -> ApiCreds:
            self.derived = True
            return ApiCreds("derived-key", "derived-secret", "derived-passphrase")

        def set_api_creds(self, creds: ApiCreds) -> None:
            self.creds = creds

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        types.SimpleNamespace(ClobClient=FakeClobClient, SignatureTypeV2=SignatureTypeV2),
    )
    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2.clob_types",
        types.SimpleNamespace(ApiCreds=ApiCreds),
    )
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xprivate")
    monkeypatch.setenv("POLYMARKET_FUNDER", "0xfunder")
    monkeypatch.setenv("POLYMARKET_API_KEY", "builder-key-that-should-not-be-used")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "builder-secret-that-should-not-be-used")
    monkeypatch.setenv("POLYMARKET_API_PASSPHRASE", "builder-pass-that-should-not-be-used")

    ClobExecutionClient(config=_config())

    fake = FakeClobClient.instances[-1]
    assert fake.kwargs["signature_type"] == SignatureTypeV2.POLY_PROXY
    assert fake.kwargs["funder"] == "0xfunder"
    assert fake.derived is True
    assert fake.creds is not None
    assert fake.creds.api_key == "derived-key"


def test_v2_static_env_auth_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    class SignatureTypeV2(int):
        POLY_PROXY = 1

    class ApiCreds:
        def __init__(self, api_key: str, api_secret: str, api_passphrase: str) -> None:
            self.api_key = api_key
            self.api_secret = api_secret
            self.api_passphrase = api_passphrase

    class FakeClobClient:
        instances: list[FakeClobClient] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.derived = False
            self.creds: ApiCreds | None = None
            FakeClobClient.instances.append(self)

        def create_or_derive_api_key(self) -> ApiCreds:
            self.derived = True
            return ApiCreds("derived-key", "derived-secret", "derived-passphrase")

        def set_api_creds(self, creds: ApiCreds) -> None:
            self.creds = creds

    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2",
        types.SimpleNamespace(ClobClient=FakeClobClient, SignatureTypeV2=SignatureTypeV2),
    )
    monkeypatch.setitem(
        sys.modules,
        "py_clob_client_v2.clob_types",
        types.SimpleNamespace(ApiCreds=ApiCreds),
    )
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xprivate")
    monkeypatch.setenv("POLYMARKET_CLOB_AUTH_MODE", "env")
    monkeypatch.setenv("POLYMARKET_CLOB_API_KEY", "clob-key")
    monkeypatch.setenv("POLYMARKET_CLOB_API_SECRET", "clob-secret")
    monkeypatch.setenv("POLYMARKET_CLOB_API_PASSPHRASE", "clob-pass")

    ClobExecutionClient(
        config=_config(
            api_key_env="POLYMARKET_CLOB_API_KEY",
            api_secret_env="POLYMARKET_CLOB_API_SECRET",
            api_passphrase_env="POLYMARKET_CLOB_API_PASSPHRASE",
        )
    )

    fake = FakeClobClient.instances[-1]
    assert fake.derived is False
    assert fake.creds is not None
    assert fake.creds.api_key == "clob-key"
