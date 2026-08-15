"""Concrete pinned-CLOB contract tests for the production gateway boundary."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from eth_account import Account

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.residual_promotion_execution_gateway import (
    ExecutionGatewayError,
    PolymarketClobV2VenueBoundary,
    VenueOrderOutcome,
    _validated_venue_runtime_binding,
)

PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
SECOND_PRIVATE_KEY = (
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
)
ORDER_HASH = "0x" + "ab" * 32
MARKET_ID = "0x" + "cd" * 32


def _private_file(path: Path, content: str) -> str:
    path.write_text(content, encoding="ascii")
    path.chmod(0o600)
    return str(path)


def _boundary(
    tmp_path: Path,
    *,
    private_key: str = PRIVATE_KEY,
    funder: str | None = None,
    api_key: str = "gateway-api-key",
    host: str = "https://clob.polymarket.com",
) -> PolymarketClobV2VenueBoundary:
    tmp_path.mkdir(parents=True, exist_ok=True)
    signer = Account.from_key(private_key).address
    key_path = _private_file(tmp_path / f"key-{api_key}.txt", private_key)
    credentials = {
        "api_key": api_key,
        "api_secret": base64.urlsafe_b64encode(b"secret-secret-secret").decode(),
        "api_passphrase": "gateway-passphrase",
    }
    credentials_path = tmp_path / f"credentials-{api_key}.json"
    credentials_path.write_text(
        json.dumps(credentials, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    credentials_path.chmod(0o600)
    return PolymarketClobV2VenueBoundary(
        private_key_path=key_path,
        api_credentials_path=str(credentials_path),
        host=host,
        chain_id=137,
        signature_type=0,
        funder=funder or signer,
        maximum_call_duration_ms=250,
    )


def _request() -> dict[str, Any]:
    return {
        "authorization_id": "a" * 64,
        "client_order_id": "b" * 64,
        "market_id": MARKET_ID,
        "token_id": "123",
        "quantity": "10",
        "limit_price": "0.5",
    }


def _prepared(request: dict[str, Any]) -> dict[str, Any]:
    signer = Account.from_key(PRIVATE_KEY).address
    return {
        "schema_version": (
            "bigan-btc-15m-residual-promotion-polymarket-clob-v2-boundary-v2"
        ),
        "client_order_id": request["client_order_id"],
        "exact_request_sha256": canonical_json_sha256(request),
        "order_hash": ORDER_HASH,
        "signed_order": {
            "salt": "1",
            "maker": signer,
            "signer": signer,
            "tokenId": request["token_id"],
            "makerAmount": "5000000",
            "takerAmount": "10000000",
            "side": 0,
            "signatureType": 0,
            "timestamp": "1752500000000",
            "metadata": "0x" + "00" * 32,
            "builder": "0x" + "00" * 32,
            "expiration": "0",
            "signature": "0x01",
        },
        "order_type": "GTC",
    }


@pytest.mark.parametrize(
    "venue_status",
    (
        pytest.param("live", id="resting-order"),
        pytest.param("matched", id="matched-order"),
    ),
)
def test_concrete_boundary_maps_successful_pinned_submission_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    venue_status: str,
) -> None:
    boundary = _boundary(tmp_path)
    request = _request()
    raw = {
        "success": True,
        "errorMsg": "",
        "orderID": ORDER_HASH,
        "status": venue_status,
        "takingAmount": "10000000",
        "makingAmount": "5000000",
    }
    monkeypatch.setattr(boundary._client, "post_order", lambda *_: raw)
    outcome = boundary.submit_prepared(_prepared(request), request)
    assert isinstance(outcome, VenueOrderOutcome)
    normalized = json.loads(outcome.normalized_response)
    assert normalized["status"] == "ACCEPTED"
    assert normalized["exchange_order_id"] == ORDER_HASH
    assert json.loads(outcome.raw_venue_response) == raw


def test_concrete_boundary_rejects_contradictory_and_maps_rejected_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _boundary(tmp_path)
    request = _request()
    rejected = {
        "success": False,
        "errorMsg": "insufficient balance",
        "orderID": "",
        "status": "rejected",
    }
    monkeypatch.setattr(boundary._client, "post_order", lambda *_: rejected)
    outcome = boundary.submit_prepared(_prepared(request), request)
    assert json.loads(outcome.normalized_response)["status"] == "REJECTED"

    contradictory = {**rejected, "success": True, "status": "live"}
    monkeypatch.setattr(boundary._client, "post_order", lambda *_: contradictory)
    with pytest.raises(ExecutionGatewayError, match="contradictory"):
        boundary.submit_prepared(_prepared(request), request)


def test_concrete_boundary_post_recovery_cancel_and_fill_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _boundary(tmp_path)
    request = _request()
    live = {
        "success": True,
        "errorMsg": "",
        "orderID": ORDER_HASH,
        "status": "live",
        "takingAmount": "10000000",
        "makingAmount": "5000000",
    }
    lookup = {
        "id": ORDER_HASH,
        "status": "live",
        "original_size": "10",
        "size_matched": "4",
        "associate_trades": ["trade-1"],
    }
    trade = {
        "id": "trade-1",
        "taker_order_id": ORDER_HASH,
        "market": MARKET_ID,
        "asset_id": request["token_id"],
        "side": "BUY",
        "size": "4",
        "fee_rate_bps": "100",
        "price": "0.5",
        "status": "MATCHED",
        "match_time": "1752500000",
        "maker_orders": [],
    }
    monkeypatch.setattr(boundary._client, "post_order", lambda *_: live)
    monkeypatch.setattr(boundary._client, "get_order", lambda *_: lookup)
    monkeypatch.setattr(boundary._client, "get_trades", lambda *_: [trade])
    monkeypatch.setattr(
        boundary._client,
        "cancel_order",
        lambda *_: {"canceled": [ORDER_HASH], "not_canceled": {}},
    )

    placed = boundary.submit_prepared(_prepared(request), request)
    recovered = boundary.lookup_submission(_prepared(request), request)
    assert json.loads(placed.normalized_response)["status"] == "ACCEPTED"
    assert recovered is not None
    assert json.loads(recovered.normalized_response)["status"] == "ACCEPTED"
    canceled = json.loads(
        boundary.cancel(
            {
                "client_order_id": request["client_order_id"],
                "exchange_order_id": ORDER_HASH,
            }
        )
    )
    assert canceled["status"] == "CANCEL_REQUESTED"

    cursor_request = {
        "authorization_id": request["authorization_id"],
        "execution_service_binding_sha256": "e" * 64,
        "request_started_at_ts_ms": 1_752_500_001_000,
        "client_order_id": request["client_order_id"],
        "exchange_order_id": ORDER_HASH,
        "market_id": request["market_id"],
        "token_id": request["token_id"],
    }
    partial = json.loads(boundary.read_fill_cursor(cursor_request))
    assert partial["status"] == "OPEN"
    assert partial["cumulative_filled_quantity"] == "4"
    assert partial["fill_events"][0]["fee_usd"] == "0.02"

    lookup.update({"status": "matched", "size_matched": "10"})
    trade["size"] = "10"
    full = json.loads(boundary.read_fill_cursor(cursor_request))
    assert full["status"] == "FILLED"
    assert full["fill_delivery_complete"] is True

    lookup.update({"status": "canceled", "size_matched": "4"})
    trade["size"] = "4"
    terminal_cancel = json.loads(boundary.read_fill_cursor(cursor_request))
    assert terminal_cancel["status"] == "CANCELED"
    assert terminal_cancel["fill_delivery_complete"] is True


def test_concrete_runtime_binding_rejects_every_identity_drift_before_service(
    tmp_path: Path,
) -> None:
    boundary = _boundary(tmp_path / "correct")
    binding = dict(boundary.runtime_binding)
    authority = {
        "service_identity_sha256": binding["service_identity_sha256"],
        "exchange_endpoint_sha256": binding["exchange_endpoint_sha256"],
        "exchange_account_sha256": binding["exchange_account_sha256"],
        "signer_identity_sha256": binding["signer_identity_sha256"],
    }
    assert _validated_venue_runtime_binding(boundary, authority) == binding

    class DriftedVenue:
        def __init__(self, value: dict[str, Any]) -> None:
            self.runtime_binding = value

    for field in (
        "gateway_implementation_sha256",
        "venue_configuration_sha256",
        "api_credentials_identity_sha256",
        "exchange_endpoint_sha256",
        "exchange_account_sha256",
        "signer_identity_sha256",
    ):
        drifted = copy.deepcopy(binding)
        drifted[field] = "0" * 64
        with pytest.raises(ExecutionGatewayError, match="identity is mismatched"):
            _validated_venue_runtime_binding(DriftedVenue(drifted), authority)

    wrong_key = _boundary(
        tmp_path / "wrong-key",
        private_key=SECOND_PRIVATE_KEY,
    )
    wrong_funder = _boundary(
        tmp_path / "wrong-funder",
        funder="0x" + "11" * 20,
    )
    wrong_api = _boundary(tmp_path / "wrong-api", api_key="other-api-key")
    for drifted_boundary in (wrong_key, wrong_funder, wrong_api):
        with pytest.raises(ExecutionGatewayError, match="identity is mismatched"):
            _validated_venue_runtime_binding(drifted_boundary, authority)

    with pytest.raises(ExecutionGatewayError, match="configuration is invalid"):
        _boundary(
            tmp_path / "wrong-endpoint",
            host="https://not-polymarket.invalid",
        )
