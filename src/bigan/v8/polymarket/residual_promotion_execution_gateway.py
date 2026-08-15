"""Deployment-owned execution gateway for the residual promotion lineage.

This module is deliberately separate from the model/executor process.  It is
the only production service allowed to own Polymarket credentials and receipt
signing material.  The client-facing RPC route is authenticated AF_UNIX; every
request is strict JSON, signed-session bound, and routed to the exact backend
defined here.  Tests may replace only the outer venue boundary.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import hashlib
import json
import multiprocessing.connection
import os
import stat
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from bigan.v8.polymarket import residual_promotion_micro_live_executor as executor

SERVICE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-gateway-service-v1"
)
STATE_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-execution-gateway-state-v1"
)
VENUE_BOUNDARY_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-polymarket-clob-v2-boundary-v1"
)


class ExecutionGatewayError(executor.MicroLiveExecutionError):
    """Raised when the deployment-owned gateway must fail closed."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ExecutionGatewayError(f"{label} is not a SHA-256")
    return value


def _decode_rpc_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"bytes_base64"}:
            encoded = value["bytes_base64"]
            if not isinstance(encoded, str):
                raise ExecutionGatewayError("gateway RPC byte value is invalid")
            try:
                return base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ExecutionGatewayError(
                    "gateway RPC byte value is invalid"
                ) from exc
        return {str(key): _decode_rpc_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_rpc_value(item) for item in value]
    return value


def _private_file(path: Path | str, label: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        raise ExecutionGatewayError(f"{label} path must be absolute")
    resolved = selected.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ExecutionGatewayError(f"{label} is not a regular file")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ExecutionGatewayError(f"{label} ownership/mode must be current uid/0600")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionGatewayServiceConfig:
    """Exact service, risk-authority, credential, and signer configuration."""

    endpoint: str
    rpc_credential_path: str
    receipt_private_exponent_path: str
    state_path: str
    execution_authority: Mapping[str, Any]
    risk_lease_id: str
    risk_service_identity_sha256: str
    risk_tenant_id: str
    risk_key_identity_sha256: str
    risk_public_key_modulus_hex: str
    risk_public_key_exponent: int

    def validated(self) -> ExecutionGatewayServiceConfig:
        endpoint = Path(self.endpoint)
        state_path = Path(self.state_path)
        if not endpoint.is_absolute() or not state_path.is_absolute():
            raise ExecutionGatewayError("gateway endpoint/state paths must be absolute")
        credential = _private_file(
            self.rpc_credential_path,
            "gateway RPC credential",
        ).read_bytes()
        _private_file(
            self.receipt_private_exponent_path,
            "gateway receipt private exponent",
        )
        authority = dict(self.execution_authority)
        required = {
            "service_identity_sha256",
            "adapter_implementation_sha256",
            "configuration_sha256",
            "route_mode",
            "route_binding_sha256",
            "exchange_endpoint_sha256",
            "exchange_account_sha256",
            "signer_identity_sha256",
            "cursor_key_identity_sha256",
            "clock_identity_sha256",
            "settlement_authority_identity_sha256",
            "public_key_modulus_hex",
            "public_key_exponent",
            "signature_algorithm",
            "maximum_clock_skew_ms",
            "maximum_call_duration_ms",
            "deployment_runtime_lock_sha256",
            "deployment_requirements_lock_sha256",
            "deployment_image_manifest_digest",
        }
        if set(authority) != required:
            raise ExecutionGatewayError("gateway execution authority is not exact")
        for field in {
            "service_identity_sha256",
            "adapter_implementation_sha256",
            "configuration_sha256",
            "route_binding_sha256",
            "exchange_endpoint_sha256",
            "exchange_account_sha256",
            "signer_identity_sha256",
            "cursor_key_identity_sha256",
            "clock_identity_sha256",
            "settlement_authority_identity_sha256",
            "deployment_runtime_lock_sha256",
            "deployment_requirements_lock_sha256",
        }:
            _require_sha256(authority[field], f"execution authority {field}")
        for field in (
            "risk_service_identity_sha256",
            "risk_key_identity_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not (
            isinstance(self.risk_lease_id, str)
            and self.risk_lease_id
            and isinstance(self.risk_tenant_id, str)
            and self.risk_tenant_id
            and isinstance(self.risk_public_key_modulus_hex, str)
            and len(self.risk_public_key_modulus_hex) >= 512
            and isinstance(self.risk_public_key_exponent, int)
            and self.risk_public_key_exponent > 1
            and isinstance(authority["public_key_modulus_hex"], str)
            and len(authority["public_key_modulus_hex"]) >= 512
            and isinstance(authority["public_key_exponent"], int)
            and authority["public_key_exponent"] > 1
            and authority["route_mode"] == executor.EXECUTION_GATEWAY_ROUTE_MODE
            and authority["signature_algorithm"]
            == executor.RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM
            and isinstance(authority["maximum_clock_skew_ms"], int)
            and authority["maximum_clock_skew_ms"] >= 0
            and isinstance(authority["maximum_call_duration_ms"], int)
            and authority["maximum_call_duration_ms"] > 0
            and isinstance(authority["deployment_image_manifest_digest"], str)
            and authority["deployment_image_manifest_digest"].startswith("sha256:")
        ):
            raise ExecutionGatewayError("gateway verification key is invalid")
        adapter_implementation_sha256 = (
            executor.deployment_owned_execution_gateway_adapter_implementation_sha256()
        )
        configuration_sha256 = (
            executor.deployment_owned_execution_gateway_configuration_sha256(
                endpoint=self.endpoint,
                credential=credential,
                service_identity_sha256=authority["service_identity_sha256"],
                exchange_endpoint_sha256=authority["exchange_endpoint_sha256"],
                exchange_account_sha256=authority["exchange_account_sha256"],
                signer_identity_sha256=authority["signer_identity_sha256"],
                cursor_key_identity_sha256=authority[
                    "cursor_key_identity_sha256"
                ],
                clock_identity_sha256=authority["clock_identity_sha256"],
                settlement_authority_identity_sha256=authority[
                    "settlement_authority_identity_sha256"
                ],
            )
        )
        route_binding_sha256 = (
            executor.deployment_owned_execution_gateway_route_binding_sha256(
                endpoint=self.endpoint,
                credential=credential,
                service_identity_sha256=authority["service_identity_sha256"],
                exchange_endpoint_sha256=authority["exchange_endpoint_sha256"],
                exchange_account_sha256=authority["exchange_account_sha256"],
                signer_identity_sha256=authority["signer_identity_sha256"],
                cursor_key_identity_sha256=authority[
                    "cursor_key_identity_sha256"
                ],
                clock_identity_sha256=authority["clock_identity_sha256"],
                settlement_authority_identity_sha256=authority[
                    "settlement_authority_identity_sha256"
                ],
                adapter_implementation_sha256=adapter_implementation_sha256,
                configuration_sha256=configuration_sha256,
            )
        )
        if not (
            authority["adapter_implementation_sha256"]
            == adapter_implementation_sha256
            and authority["configuration_sha256"] == configuration_sha256
            and authority["route_binding_sha256"] == route_binding_sha256
        ):
            raise ExecutionGatewayError(
                "gateway client implementation/configuration/route is mismatched"
            )
        return self

    @property
    def configuration_sha256(self) -> str:
        self.validated()
        credential = _private_file(
            self.rpc_credential_path,
            "gateway RPC credential",
        ).read_bytes()
        exponent = _private_file(
            self.receipt_private_exponent_path,
            "gateway receipt private exponent",
        ).read_bytes()
        return executor.canonical_json_sha256(
            {
                "schema_version": SERVICE_SCHEMA_VERSION,
                "endpoint": self.endpoint,
                "rpc_credential_sha256": _sha256(credential),
                "receipt_private_exponent_identity_sha256": _sha256(exponent),
                "state_path": self.state_path,
                "execution_authority": dict(self.execution_authority),
                "risk_lease_id": self.risk_lease_id,
                "risk_service_identity_sha256": self.risk_service_identity_sha256,
                "risk_tenant_id": self.risk_tenant_id,
                "risk_key_identity_sha256": self.risk_key_identity_sha256,
                "risk_public_key_modulus_hex": self.risk_public_key_modulus_hex,
                "risk_public_key_exponent": self.risk_public_key_exponent,
            }
        )


@dataclass(frozen=True, slots=True)
class PolymarketVenueConfig:
    """Credential-owned final venue configuration (never serialized over RPC)."""

    private_key_path: str
    api_credentials_path: str
    host: str
    chain_id: int
    signature_type: int
    funder: str

    def validated(self) -> PolymarketVenueConfig:
        _private_file(self.private_key_path, "Polymarket private key")
        _private_file(self.api_credentials_path, "Polymarket API credentials")
        if not (
            self.host == "https://clob.polymarket.com"
            and self.chain_id == 137
            and isinstance(self.signature_type, int)
            and self.signature_type >= 0
            and isinstance(self.funder, str)
            and self.funder.startswith("0x")
            and len(self.funder) == 42
        ):
            raise ExecutionGatewayError("Polymarket venue configuration is invalid")
        return self

    @property
    def configuration_sha256(self) -> str:
        self.validated()
        return executor.canonical_json_sha256(
            {
                "schema_version": VENUE_BOUNDARY_SCHEMA_VERSION,
                "private_key_identity_sha256": _sha256(
                    _private_file(
                        self.private_key_path,
                        "Polymarket private key",
                    ).read_bytes()
                ),
                "api_credentials_identity_sha256": _sha256(
                    _private_file(
                        self.api_credentials_path,
                        "Polymarket API credentials",
                    ).read_bytes()
                ),
                "host": self.host,
                "chain_id": self.chain_id,
                "signature_type": self.signature_type,
                "funder": self.funder,
            }
        )


class RsaSha256ReceiptSigner:
    """Final file-owned PKCS#1 v1.5 SHA-256 receipt signer."""

    __slots__ = ("_private_exponent", "_public_exponent", "_public_modulus")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError("gateway receipt signer is final")

    def __init__(
        self,
        *,
        private_exponent_path: Path | str,
        public_modulus_hex: str,
        public_exponent: int,
    ) -> None:
        raw = _private_file(
            private_exponent_path,
            "gateway receipt private exponent",
        ).read_text(encoding="ascii")
        try:
            private_exponent = int(raw.strip(), 16)
            public_modulus = int(public_modulus_hex, 16)
        except ValueError as exc:
            raise ExecutionGatewayError("gateway receipt RSA key is invalid") from exc
        if not (
            private_exponent > 1
            and public_modulus > 1
            and isinstance(public_exponent, int)
            and public_exponent > 1
        ):
            raise ExecutionGatewayError("gateway receipt RSA key is invalid")
        self._private_exponent = private_exponent
        self._public_modulus = public_modulus
        self._public_exponent = public_exponent
        self_test = 2
        if (
            pow(
                pow(self_test, private_exponent, public_modulus),
                public_exponent,
                public_modulus,
            )
            != self_test
        ):
            raise ExecutionGatewayError(
                "gateway receipt private/public RSA identity is mismatched"
            )

    def sign(self, core: Mapping[str, Any]) -> bytes:
        signed_bytes = _json_bytes(core)
        digest_info = executor._RSA_SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(
            signed_bytes
        ).digest()
        encoded_size = (self._public_modulus.bit_length() + 7) // 8
        if encoded_size < len(digest_info) + 11:
            raise ExecutionGatewayError("gateway receipt RSA modulus is too small")
        encoded = (
            b"\x00\x01"
            + b"\xff" * (encoded_size - len(digest_info) - 3)
            + b"\x00"
            + digest_info
        )
        signature = pow(
            int.from_bytes(encoded, "big"),
            self._private_exponent,
            self._public_modulus,
        ).to_bytes(encoded_size, "big")
        return _json_bytes(
            {
                **dict(core),
                "signature_algorithm": (
                    executor.RISK_DOMAIN_RECEIPT_SIGNATURE_ALGORITHM
                ),
                "signature_hex": signature.hex(),
            }
        )


class ExactVenueBoundary(Protocol):
    """Only mockable boundary: exact prepared writes and authoritative reads."""

    def prepare_submission(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def submit_prepared(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> bytes: ...

    def lookup_submission(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> bytes | None: ...

    def cancel(self, request: Mapping[str, Any]) -> bytes: ...

    def lookup(self, request: Mapping[str, Any]) -> bytes: ...

    def read_fill_cursor(self, request: Mapping[str, Any]) -> bytes: ...


class PolymarketClobV2VenueBoundary:
    """Concrete credential-owning py-clob-client-v2 venue implementation.

    A signed order is prepared before the dispatch side effect and is persisted
    by the gateway.  Recovery uses its deterministic EIP-712 order hash and is
    lookup-only; it never calls ``post_order``.
    """

    __slots__ = ("_client", "_order_type")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError("Polymarket CLOB v2 venue boundary is final")

    def __init__(
        self,
        *,
        private_key_path: Path | str,
        api_credentials_path: Path | str,
        host: str,
        chain_id: int,
        signature_type: int,
        funder: str,
    ) -> None:
        private_key = _private_file(
            private_key_path,
            "Polymarket private key",
        ).read_text(encoding="ascii").strip()
        raw_credentials = _private_file(
            api_credentials_path,
            "Polymarket API credentials",
        ).read_bytes()
        credentials, _, _ = executor._raw_json_object(
            raw_credentials,
            "Polymarket API credentials",
        )
        if set(credentials) != {"api_key", "api_secret", "api_passphrase"}:
            raise ExecutionGatewayError("Polymarket API credentials are not exact")
        try:
            from py_clob_client_v2 import ClobClient
            from py_clob_client_v2.clob_types import ApiCreds, OrderType
        except ImportError as exc:
            raise ExecutionGatewayError(
                "pinned py-clob-client-v2 runtime is unavailable"
            ) from exc
        self._client = ClobClient(
            host=host,
            chain_id=chain_id,
            key=private_key,
            creds=ApiCreds(**credentials),
            signature_type=signature_type,
            funder=funder,
            retry_on_error=False,
        )
        self._order_type = OrderType.GTC

    def prepare_submission(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            from py_clob_client_v2.clob_types import OrderArgsV2
            from py_clob_client_v2.config import get_contract_config
            from py_clob_client_v2.order_utils.exchange_order_builder_v2 import (
                ExchangeOrderBuilderV2,
            )
        except ImportError as exc:
            raise ExecutionGatewayError(
                "pinned py-clob-client-v2 order runtime is unavailable"
            ) from exc
        metadata = "0x" + hashlib.sha256(
            str(request["client_order_id"]).encode("ascii")
        ).hexdigest()
        signed_order = self._client.create_order(
            OrderArgsV2(
                token_id=str(request["token_id"]),
                price=float(request["limit_price"]),
                size=float(request["quantity"]),
                side="BUY",
                metadata=metadata,
            )
        )
        contract = get_contract_config(self._client.signer.get_chain_id())
        neg_risk = self._client.get_neg_risk(str(request["token_id"]))
        exchange_address = (
            contract.neg_risk_exchange_v2 if neg_risk else contract.exchange_v2
        )
        builder = ExchangeOrderBuilderV2(
            exchange_address,
            self._client.signer.get_chain_id(),
            self._client.signer,
        )
        typed_data = builder.build_order_typed_data(signed_order)
        order_hash = builder.build_order_hash(typed_data)
        serialized = dataclasses.asdict(signed_order)
        serialized["side"] = int(serialized["side"])
        serialized["signatureType"] = int(serialized["signatureType"])
        return {
            "schema_version": VENUE_BOUNDARY_SCHEMA_VERSION,
            "client_order_id": request["client_order_id"],
            "exact_request_sha256": executor.canonical_json_sha256(request),
            "order_hash": order_hash,
            "signed_order": serialized,
            "order_type": "GTC",
        }

    def _signed_order(self, prepared: Mapping[str, Any]) -> Any:
        try:
            from py_clob_client_v2.order_utils.model.order_data_v2 import (
                SignedOrderV2,
            )
            from py_clob_client_v2.order_utils.model.side import Side
            from py_clob_client_v2.order_utils.model.signature_type_v2 import (
                SignatureTypeV2,
            )
        except ImportError as exc:
            raise ExecutionGatewayError(
                "pinned py-clob-client-v2 order runtime is unavailable"
            ) from exc
        values = dict(prepared["signed_order"])
        values["side"] = Side(values["side"])
        values["signatureType"] = SignatureTypeV2(values["signatureType"])
        return SignedOrderV2(**values)

    @staticmethod
    def _normalized_order_response(
        raw: Mapping[str, Any],
        request: Mapping[str, Any],
        *,
        order_hash: str,
    ) -> bytes:
        exchange_order_id = raw.get("orderID") or raw.get("order_id") or raw.get("id")
        if exchange_order_id is not None and str(exchange_order_id) != order_hash:
            raise ExecutionGatewayError("venue order identity is mismatched")
        status = str(raw.get("status") or "ACCEPTED").upper()
        return _json_bytes(
            {
                "client_order_id": request["client_order_id"],
                "exchange_order_id": order_hash,
                "status": status,
                "market_id": request["market_id"],
                "token_id": request["token_id"],
                "accepted_quantity": request["quantity"],
                "limit_price": request["limit_price"],
            }
        )

    def submit_prepared(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> bytes:
        if prepared.get("exact_request_sha256") != executor.canonical_json_sha256(
            request
        ):
            raise ExecutionGatewayError("prepared venue request is mismatched")
        raw = self._client.post_order(self._signed_order(prepared), self._order_type)
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue submission response is invalid")
        return self._normalized_order_response(
            raw,
            request,
            order_hash=str(prepared["order_hash"]),
        )

    def lookup_submission(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> bytes | None:
        try:
            raw = self._client.get_order(str(prepared["order_hash"]))
        except BaseException as exc:  # third-party exception taxonomy is unstable
            if "not found" in str(exc).lower() or "404" in str(exc):
                return None
            raise ExecutionGatewayError("venue recovery lookup failed closed") from exc
        if not isinstance(raw, Mapping) or not raw:
            return None
        return self._normalized_order_response(
            raw,
            request,
            order_hash=str(prepared["order_hash"]),
        )

    def cancel(self, request: Mapping[str, Any]) -> bytes:
        try:
            from py_clob_client_v2.clob_types import OrderPayload
        except ImportError as exc:
            raise ExecutionGatewayError("pinned CLOB cancel runtime is unavailable") from exc
        raw = self._client.cancel_order(
            OrderPayload(orderID=str(request["exchange_order_id"]))
        )
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue cancel response is invalid")
        return _json_bytes(
            {
                "client_order_id": request["client_order_id"],
                "exchange_order_id": request["exchange_order_id"],
                "status": "CANCEL_REQUESTED",
            }
        )

    def lookup(self, request: Mapping[str, Any]) -> bytes:
        raw = self._client.get_order(str(request["exchange_order_id"]))
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue order lookup response is invalid")
        return self._normalized_order_response(
            raw,
            request,
            order_hash=str(request["exchange_order_id"]),
        )

    def read_fill_cursor(self, request: Mapping[str, Any]) -> bytes:
        raw = self._client.get_order(str(request["exchange_order_id"]))
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue fill cursor response is invalid")
        status = str(raw.get("status") or "OPEN").upper()
        filled = Decimal(str(raw.get("size_matched") or raw.get("filled_size") or "0"))
        fill_events = self._authoritative_fill_events(request, raw)
        cumulative = (
            Decimal(str(fill_events[-1]["cumulative_filled_quantity"]))
            if fill_events
            else Decimal("0")
        )
        if cumulative != filled:
            raise ExecutionGatewayError(
                "authoritative venue fills do not reconcile to order size_matched"
            )
        terminal = status in {"FILLED", "CANCELED", "EXPIRED"}
        observed_at = int(request["request_started_at_ts_ms"])
        response: dict[str, Any] = {
            "schema_version": executor.EXECUTION_CURSOR_SCHEMA_VERSION,
            "authorization_id": request["authorization_id"],
            "execution_service_binding_sha256": request[
                "execution_service_binding_sha256"
            ],
            "request_started_at_ts_ms": request["request_started_at_ts_ms"],
            "event_type": "ORDER_FILL_CURSOR",
            "client_order_id": request["client_order_id"],
            "exchange_order_id": request["exchange_order_id"],
            "market_id": request["market_id"],
            "token_id": request["token_id"],
            "status": status,
            "observed_at_ts_ms": observed_at,
            "effective_at_ts_ms": observed_at if terminal else None,
            "cumulative_filled_quantity": str(cumulative),
            "final_fill_event_sequence": len(fill_events),
            "final_fill_count": len(fill_events),
            "final_fill_watermark": None,
            "fill_delivery_complete": terminal,
            "fill_events": fill_events,
        }
        if terminal:
            response["final_fill_watermark"] = executor._expected_final_fill_watermark(
                response
            )
        return _json_bytes(response)

    def _authoritative_fill_events(
        self,
        request: Mapping[str, Any],
        order: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        trade_ids = order.get("associate_trades") or order.get("trade_ids") or []
        if not isinstance(trade_ids, list) or not all(
            isinstance(value, str) and value for value in trade_ids
        ):
            raise ExecutionGatewayError("venue associated trade identity is invalid")
        if not trade_ids:
            return []
        try:
            from py_clob_client_v2.clob_types import TradeParams
        except ImportError as exc:
            raise ExecutionGatewayError("pinned CLOB trade runtime is unavailable") from exc
        selected: list[tuple[int, str, Decimal, Decimal, Decimal]] = []
        order_id = str(request["exchange_order_id"])
        for trade_id in trade_ids:
            rows = self._client.get_trades(TradeParams(id=trade_id))
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise ExecutionGatewayError("authoritative venue trade lookup is ambiguous")
            trade = rows[0]
            if str(trade.get("id")) != trade_id:
                raise ExecutionGatewayError("authoritative venue trade identity is mismatched")
            leg: Mapping[str, Any] = trade
            if str(trade.get("taker_order_id") or "") != order_id:
                maker_orders = trade.get("maker_orders")
                if not isinstance(maker_orders, list):
                    raise ExecutionGatewayError("venue maker trade legs are absent")
                matches = [
                    row
                    for row in maker_orders
                    if isinstance(row, Mapping)
                    and str(row.get("order_id") or row.get("orderID") or "")
                    == order_id
                ]
                if len(matches) != 1:
                    raise ExecutionGatewayError("venue maker trade leg is ambiguous")
                leg = matches[0]
            token_id = str(
                leg.get("asset_id")
                or leg.get("token_id")
                or trade.get("asset_id")
                or ""
            )
            market_id = str(trade.get("market") or trade.get("market_id") or "")
            if token_id != str(request["token_id"]) or (
                market_id and market_id != str(request["market_id"])
            ):
                raise ExecutionGatewayError("authoritative venue trade population is mismatched")
            try:
                quantity = Decimal(
                    str(
                        leg.get("matched_amount")
                        or leg.get("size")
                        or trade.get("size")
                    )
                )
                price = Decimal(str(leg.get("price") or trade.get("price")))
                fee = Decimal(
                    str(
                        leg.get("fee_usd")
                        or leg.get("fee")
                        or trade.get("fee_usd")
                        or trade.get("fee")
                    )
                )
            except Exception as exc:
                raise ExecutionGatewayError(
                    "authoritative venue trade economics are incomplete"
                ) from exc
            if quantity <= 0 or not Decimal("0") < price <= Decimal("1") or fee < 0:
                raise ExecutionGatewayError(
                    "authoritative venue trade economics are invalid"
                )
            timestamp = (
                leg.get("match_time")
                or leg.get("executed_at")
                or trade.get("match_time")
                or trade.get("executed_at")
                or trade.get("timestamp")
            )
            executed_at = self._venue_timestamp_ms(timestamp)
            selected.append((executed_at, trade_id, quantity, price, fee))
        selected.sort(key=lambda row: (row[0], row[1]))
        cumulative = Decimal("0")
        events: list[dict[str, Any]] = []
        for sequence, (executed_at, trade_id, quantity, price, fee) in enumerate(
            selected,
            start=1,
        ):
            cumulative += quantity
            events.append(
                {
                    "event_type": "FILL",
                    "client_order_id": request["client_order_id"],
                    "exchange_order_id": order_id,
                    "fill_id": trade_id,
                    "market_id": request["market_id"],
                    "token_id": request["token_id"],
                    "quantity": str(quantity),
                    "price": str(price),
                    "fee_usd": str(fee),
                    "executed_at_ts_ms": executed_at,
                    "fill_event_sequence": sequence,
                    "cumulative_filled_quantity": str(cumulative),
                    "cumulative_fill_count": sequence,
                }
            )
        return events

    @staticmethod
    def _venue_timestamp_ms(value: Any) -> int:
        if isinstance(value, bool):
            raise ExecutionGatewayError("venue trade timestamp is invalid")
        if isinstance(value, (int, float)):
            number = int(value)
            return number if number >= 1_000_000_000_000 else number * 1_000
        if isinstance(value, str) and value:
            if value.isdigit():
                number = int(value)
                return number if number >= 1_000_000_000_000 else number * 1_000
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ExecutionGatewayError("venue trade timestamp is invalid") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return int(parsed.timestamp() * 1_000)
        raise ExecutionGatewayError("venue trade timestamp is invalid")


class _DurableGatewayState:
    """Single-writer, atomic, fsync-backed service state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ExecutionGatewayError("gateway state path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if self.path.exists():
            raw = self.path.read_bytes()
            value, _, _ = executor._raw_json_object(raw, "gateway durable state")
            if value.get("schema_version") != STATE_SCHEMA_VERSION:
                raise ExecutionGatewayError("gateway durable state schema is invalid")
            self.value = value
        else:
            self.value = {
                "schema_version": STATE_SCHEMA_VERSION,
                "sessions": {},
                "requests": {},
                "prepared": {},
                "venue_outcomes_base64": {},
                "dispatch_receipts_base64": {},
                "terminal_receipts_base64": {},
            }
            self.flush()

    def flush(self) -> None:
        with self._lock:
            raw = _json_bytes(self.value)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


class DeploymentOwnedExecutionGatewayBackend:
    """Exact service-side backend; only its outer venue may be mocked."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError("deployment-owned execution gateway backend is final")

    def __init__(
        self,
        config: ExecutionGatewayServiceConfig,
        *,
        venue: ExactVenueBoundary,
        state: _DurableGatewayState,
        signer: RsaSha256ReceiptSigner,
    ) -> None:
        self.config = config.validated()
        self.venue = venue
        self.state = state
        self.signer = signer
        self._dispatch_begin: Callable[..., bytes] | None = None
        self._dispatch_recover: Callable[..., bytes] | None = None
        self._dispatch_complete: Callable[..., bytes] | None = None
        self._dispatch_fence: Callable[..., bytes] | None = None
        self._identity: tuple[str, str, str, int] | None = None

    @property
    def authority(self) -> dict[str, Any]:
        authority = dict(self.config.execution_authority)
        authority["execution_service_binding_sha256"] = (
            executor.canonical_json_sha256(authority)
        )
        return authority

    def bind_execution_dispatch_authority(
        self,
        begin: Callable[..., bytes],
        recover: Callable[..., bytes],
        complete: Callable[..., bytes],
        fence: Callable[..., bytes],
        *,
        authorization_id: str,
        risk_domain_id: str,
        risk_domain_authority_binding_sha256: str,
        authorization_expires_at_ts_ms: int,
    ) -> None:
        identity = (
            authorization_id,
            risk_domain_id,
            risk_domain_authority_binding_sha256,
            authorization_expires_at_ts_ms,
        )
        if self._identity is not None and self._identity != identity:
            raise ExecutionGatewayError("execution dispatch authority was rebound")
        self._identity = identity
        self._dispatch_begin = begin
        self._dispatch_recover = recover
        self._dispatch_complete = complete
        self._dispatch_fence = fence

    def _signed(self, core: Mapping[str, Any]) -> bytes:
        return self.signer.sign(core)

    def attest_execution_binding(self, request: Mapping[str, Any]) -> bytes:
        identity = (
            str(request["authorization_id"]),
            str(request["risk_domain_id"]),
            str(request["risk_domain_authority_binding_sha256"]),
            int(request["authorization_expires_at_ts_ms"]),
        )
        if self._identity is not None and self._identity != identity:
            raise ExecutionGatewayError("execution gateway identity was rebound")
        self._identity = identity
        authority = self.authority
        core = {
            "schema_version": executor.EXECUTION_BINDING_ATTESTATION_SCHEMA_VERSION,
            "authorization_id": request["authorization_id"],
            "client_session_sha256": request["client_session_sha256"],
            "client_session_binding": request["client_session_binding"],
            "challenge_sha256": request["challenge_sha256"],
            "execution_service_binding_sha256": authority[
                "execution_service_binding_sha256"
            ],
            "service_identity_sha256": authority["service_identity_sha256"],
            "adapter_implementation_sha256": authority[
                "adapter_implementation_sha256"
            ],
            "configuration_sha256": authority["configuration_sha256"],
            "execution_gateway_route_mode": request["execution_gateway_route_mode"],
            "execution_gateway_route_binding_sha256": request[
                "execution_gateway_route_binding_sha256"
            ],
            "exchange_endpoint_sha256": authority["exchange_endpoint_sha256"],
            "exchange_account_sha256": authority["exchange_account_sha256"],
            "signer_identity_sha256": authority["signer_identity_sha256"],
            "cursor_key_identity_sha256": authority["cursor_key_identity_sha256"],
            "clock_identity_sha256": authority["clock_identity_sha256"],
            "settlement_authority_identity_sha256": authority[
                "settlement_authority_identity_sha256"
            ],
            "risk_domain_id": request["risk_domain_id"],
            "risk_domain_authority_binding_sha256": request[
                "risk_domain_authority_binding_sha256"
            ],
            "authorization_expires_at_ts_ms": request[
                "authorization_expires_at_ts_ms"
            ],
        }
        for field in (
            "execution_fence_protocol_schema_version",
            "execution_acceptance_protocol_schema_version",
            "execution_dispatch_protocol_schema_version",
            "execution_outbox_recovery_protocol_schema_version",
            "submission_recovery_operation",
            "submission_recovery_semantics",
            "submission_recovery_lookup_only_enforced",
            "execution_transport_operation_inventory_schema_version",
            "required_execution_transport_operations",
            "required_execution_transport_operations_sha256",
            "cancellation_operation",
            "cancellation_semantics",
            "terminal_cursor_operation",
            "terminal_cursor_semantics",
            "venue_idempotency_key_field",
            "venue_idempotency_scope",
            "venue_idempotency_semantics",
            "venue_idempotency_enforced",
            "deployment_runtime_lock_sha256",
            "deployment_requirements_lock_sha256",
            "deployment_image_manifest_digest",
        ):
            core[field] = copy.deepcopy(request[field])
        return self._signed(core)

    def read_trusted_time(self, request: Mapping[str, Any]) -> bytes:
        # Never report a completion before either the service wall clock or
        # the signed invocation's start.  Choosing the later value is
        # fail-closed for every executor deadline/expiry comparison.
        completed_at_ts_ms = max(
            int(time.time() * 1000),
            int(request["request_started_at_ts_ms"]),
        )
        return self._signed(
            {
                "schema_version": executor.TRUSTED_TIME_RECEIPT_SCHEMA_VERSION,
                **copy.deepcopy(dict(request)),
                "request_completed_at_ts_ms": completed_at_ts_ms,
            }
        )

    def _proof(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._identity is None:
            raise ExecutionGatewayError("gateway dispatch authority is unbound")
        return executor.verify_dispatchable_outbox_request(
            request,
            authorization_id=self._identity[0],
            risk_domain_id=self._identity[1],
            risk_domain_authority_binding_sha256=self._identity[2],
            lease_id=self.config.risk_lease_id,
            service_identity_sha256=self.config.risk_service_identity_sha256,
            tenant_id=self.config.risk_tenant_id,
            key_identity_sha256=self.config.risk_key_identity_sha256,
            public_key_modulus_hex=self.config.risk_public_key_modulus_hex,
            public_key_exponent=self.config.risk_public_key_exponent,
        )

    def _complete(
        self,
        key: str,
        dispatch_receipt: bytes,
        raw_response: bytes,
    ) -> bytes:
        if self._dispatch_complete is None:
            raise ExecutionGatewayError("gateway dispatch completion is unbound")
        terminal = self._dispatch_complete(dispatch_receipt, raw_response)
        self.state.value["terminal_receipts_base64"][key] = base64.b64encode(
            terminal
        ).decode("ascii")
        self.state.value["venue_outcomes_base64"][key] = base64.b64encode(
            raw_response
        ).decode("ascii")
        self.state.flush()
        return terminal

    def _authenticated_response(
        self,
        request: Mapping[str, Any],
        raw_response: bytes,
    ) -> bytes:
        authentication = dict(request["execution_authentication"])
        submit = authentication["operation"] == "submit_order"
        terminal = None
        if submit:
            encoded = self.state.value["terminal_receipts_base64"].get(
                request["client_order_id"]
            )
            if not isinstance(encoded, str):
                raise ExecutionGatewayError("dispatch terminal receipt is absent")
            terminal = base64.b64decode(encoded, validate=True)
        return self._signed(
            {
                "schema_version": executor.EXECUTION_OPERATION_RECEIPT_SCHEMA_VERSION,
                "authorization_id": authentication["authorization_id"],
                "execution_service_binding_sha256": authentication[
                    "execution_service_binding_sha256"
                ],
                "exchange_endpoint_sha256": authentication[
                    "exchange_endpoint_sha256"
                ],
                "exchange_account_sha256": authentication[
                    "exchange_account_sha256"
                ],
                "signer_identity_sha256": authentication[
                    "signer_identity_sha256"
                ],
                "operation": authentication["operation"],
                "request_nonce_sha256": authentication["request_nonce_sha256"],
                "request_sha256": executor.canonical_json_sha256(request),
                "response_sha256": _sha256(raw_response),
                "raw_response_json": raw_response.decode("utf-8"),
                "execution_invocation_fence_receipt_sha256": authentication[
                    "execution_invocation_fence_receipt_sha256"
                ],
                "execution_invocation_fence_status": (
                    "DISPATCHED" if submit else "NOT_APPLICABLE"
                ),
                "execution_outbox_command_sha256": (
                    authentication["execution_outbox_command_sha256"]
                    if submit
                    else None
                ),
                "raw_execution_outbox_acceptance_receipt_json": (
                    authentication["raw_execution_outbox_acceptance_receipt_json"]
                    if submit
                    else None
                ),
                "execution_outbox_acceptance_receipt_sha256": (
                    authentication["execution_outbox_acceptance_receipt_sha256"]
                    if submit
                    else None
                ),
                "raw_execution_dispatch_terminal_receipt_json": (
                    terminal.decode("utf-8") if terminal is not None else None
                ),
                "execution_dispatch_terminal_receipt_sha256": (
                    _sha256(terminal) if terminal is not None else None
                ),
            }
        )

    def submit_order(self, request: Mapping[str, Any]) -> bytes:
        proof = self._proof(request)
        if self._dispatch_begin is None:
            raise ExecutionGatewayError("gateway dispatch begin is unbound")
        key = str(request["client_order_id"])
        raw_dispatch = self._dispatch_begin(
            proof["outbox_acceptance_receipt_json"].encode("utf-8"),
            venue_idempotency_key=key,
            venue_idempotency_scope=executor.VENUE_IDEMPOTENCY_SCOPE,
            dispatch_deadline_ts_ms=request["execution_authentication"][
                "dispatch_deadline_ts_ms"
            ],
            authorization_expires_at_ts_ms=request["execution_authentication"][
                "authorization_expires_at_ts_ms"
            ],
        )
        dispatch, _, _ = executor._raw_json_object(
            raw_dispatch,
            "gateway dispatch receipt",
        )
        if dispatch["status"] == "DISPATCHED":
            outcome = dispatch["raw_outcome_json"].encode("utf-8")
            self.state.value["terminal_receipts_base64"][key] = base64.b64encode(
                raw_dispatch
            ).decode("ascii")
            self.state.flush()
            return self._authenticated_response(request, outcome)
        if dispatch["status"] != "DISPATCHING":
            raise ExecutionGatewayError("gateway dispatch is not permitted")
        self.state.value["requests"][key] = copy.deepcopy(dict(request))
        self.state.value["dispatch_receipts_base64"][key] = base64.b64encode(
            raw_dispatch
        ).decode("ascii")
        prepared = self.state.value["prepared"].get(key)
        if prepared is None:
            prepared = copy.deepcopy(dict(self.venue.prepare_submission(request)))
            self.state.value["prepared"][key] = prepared
        self.state.flush()  # prepared signed bytes precede the network side effect
        raw_response = self.venue.submit_prepared(prepared, request)
        self._complete(key, raw_dispatch, raw_response)
        return self._authenticated_response(request, raw_response)

    def recover_order_submission(self, request: Mapping[str, Any]) -> bytes:
        self._proof(request)
        key = str(request["client_order_id"])
        prepared = self.state.value["prepared"].get(key)
        if not isinstance(prepared, Mapping):
            raise executor.SubmissionRecoveryOutcomeNotFoundError(
                "gateway has no durable prepared submission"
            )
        raw_response = self.venue.lookup_submission(prepared, request)
        if raw_response is None:
            raise executor.SubmissionRecoveryOutcomeNotFoundError(
                "venue idempotency lookup found no outcome"
            )
        raw_response = executor.verify_recovered_submission_outcome(
            request,
            raw_response,
        )
        if self._dispatch_recover is None:
            raise ExecutionGatewayError("gateway dispatch recovery is unbound")
        authentication = request["execution_authentication"]
        terminal = self._dispatch_recover(
            transport_invocation_id=request["transport_invocation_id"],
            outbox_command_sha256=authentication[
                "execution_outbox_command_sha256"
            ],
            raw_outcome=raw_response,
        )
        self.state.value["terminal_receipts_base64"][key] = base64.b64encode(
            terminal
        ).decode("ascii")
        self.state.value["venue_outcomes_base64"][key] = base64.b64encode(
            raw_response
        ).decode("ascii")
        self.state.flush()
        return self._authenticated_response(request, raw_response)

    def cancel_order(self, request: Mapping[str, Any]) -> bytes:
        return self._authenticated_response(request, self.venue.cancel(request))

    def lookup_order(self, request: Mapping[str, Any]) -> bytes:
        return self._authenticated_response(request, self.venue.lookup(request))

    def read_order_fill_cursor(self, request: Mapping[str, Any]) -> bytes:
        raw = self.venue.read_fill_cursor(request)
        value, _, _ = executor._raw_json_object(raw, "venue fill cursor")
        payload_sha256 = executor.canonical_json_sha256(value)
        return self._signed({**value, "cursor_payload_sha256": payload_sha256})

    def fence_order_invocation(self, request: Mapping[str, Any]) -> bytes:
        key = str(request["client_order_id"])
        submitted = self.state.value["requests"].get(key)
        if not isinstance(submitted, Mapping) or self._dispatch_fence is None:
            raise ExecutionGatewayError("gateway dispatch fence is unbound")
        raw_fence = self._dispatch_fence(
            transport_invocation_id=request["transport_invocation_id"],
            outbox_command_sha256=submitted["execution_authentication"][
                "execution_outbox_command_sha256"
            ],
        )
        fence, _, _ = executor._raw_json_object(raw_fence, "dispatch fence")
        response = _json_bytes(
            {
                "authorization_id": request["authorization_id"],
                "client_order_id": request["client_order_id"],
                "transport_invocation_id": request["transport_invocation_id"],
                "side_effects_fenced": fence["status"] in {"FENCED", "DISPATCHED"},
            }
        )
        return self._authenticated_response(request, response)


def _rpc_response(
    *, raw_response: bytes | None = None, error_code: str | None = None
) -> bytes:
    if raw_response is not None and error_code is None:
        return _json_bytes(
            {
                "schema_version": executor.EXECUTION_GATEWAY_RPC_SCHEMA_VERSION,
                "status": "OK",
                "raw_response_base64": base64.b64encode(raw_response).decode("ascii"),
            }
        )
    return _json_bytes(
        {
            "schema_version": executor.EXECUTION_GATEWAY_RPC_SCHEMA_VERSION,
            "status": "ERROR",
            "error_code": error_code or "OPERATION_FAILED",
        }
    )


class DeploymentOwnedExecutionGatewayServer:
    """Final authenticated AF_UNIX server with a durable session registry."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError("deployment-owned execution gateway server is final")

    def __init__(
        self,
        config: ExecutionGatewayServiceConfig,
        *,
        venue: ExactVenueBoundary,
    ) -> None:
        self.config = config.validated()
        self.state = _DurableGatewayState(config.state_path)
        self.signer = RsaSha256ReceiptSigner(
            private_exponent_path=config.receipt_private_exponent_path,
            public_modulus_hex=str(
                config.execution_authority["public_key_modulus_hex"]
            ),
            public_exponent=int(
                config.execution_authority["public_key_exponent"]
            ),
        )
        self.backend = DeploymentOwnedExecutionGatewayBackend(
            config,
            venue=venue,
            state=self.state,
            signer=self.signer,
        )

    def serve_forever(
        self,
        *,
        ready: Any | None = None,
        stop: Any | None = None,
        audit: Any | None = None,
    ) -> None:
        endpoint = Path(self.config.endpoint)
        if endpoint.exists():
            raise ExecutionGatewayError(
                "gateway RPC endpoint already exists; stale-path removal is operator-only"
            )
        credential = _private_file(
            self.config.rpc_credential_path,
            "gateway RPC credential",
        ).read_bytes()
        listener = multiprocessing.connection.Listener(
            str(endpoint),
            family="AF_UNIX",
            backlog=16,
            authkey=credential,
        )
        if ready is not None:
            ready.set()
        try:
            while True:
                connection = listener.accept()
                if stop is not None and stop.is_set():
                    connection.close()
                    break
                operation = "invalid"
                try:
                    raw_request = connection.recv_bytes(
                        executor.MAX_EXECUTION_TRANSPORT_EVENT_BYTES
                    )
                    request, _, _ = executor._raw_json_object(
                        raw_request,
                        "execution gateway RPC request",
                        maximum_bytes=executor.MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                    )
                    verified = executor.verify_execution_gateway_rpc_session(
                        request,
                        expected_route_binding_sha256=str(
                            self.config.execution_authority["route_binding_sha256"]
                        ),
                        public_key_modulus_hex=str(
                            self.config.execution_authority[
                                "public_key_modulus_hex"
                            ]
                        ),
                        public_key_exponent=int(
                            self.config.execution_authority["public_key_exponent"]
                        ),
                    )
                    operation = str(verified["operation"])
                    session_sha256 = str(verified["client_session_sha256"])
                    payload = _decode_rpc_value(verified["payload"])
                    sessions = self.state.value["sessions"]
                    if operation == "attest_execution_binding":
                        if not (
                            payload.get("client_session_sha256") == session_sha256
                            and payload.get("client_session_binding")
                            == verified["client_session_binding"]
                        ):
                            raise ExecutionGatewayError(
                                "gateway attestation session is mismatched"
                            )
                        raw_response = self.backend.attest_execution_binding(payload)
                        sessions[session_sha256] = {
                            "client_session_binding": copy.deepcopy(
                                verified["client_session_binding"]
                            ),
                            "attestation_sha256": _sha256(raw_response),
                        }
                        self.state.flush()
                    else:
                        session = sessions.get(session_sha256)
                        if not (
                            isinstance(session, Mapping)
                            and session.get("client_session_binding")
                            == verified["client_session_binding"]
                            and session.get("attestation_sha256")
                            == verified["client_session_attestation_sha256"]
                        ):
                            raise ExecutionGatewayError(
                                "gateway signed session is not registered"
                            )
                        if operation == "bind_execution_dispatch_authority":
                            executor.bind_execution_backend_to_dispatch_authority(
                                self.backend,
                                **payload,
                            )
                            raw_response = b'{"bound":true}'
                        else:
                            method = getattr(self.backend, operation)
                            raw_response = method(payload)
                    connection.send_bytes(_rpc_response(raw_response=raw_response))
                    if audit is not None:
                        audit.put({"operation": operation, "status": "OK"})
                except executor.SubmissionRecoveryOutcomeNotFoundError:
                    connection.send_bytes(
                        _rpc_response(
                            error_code="SUBMISSION_RECOVERY_OUTCOME_NOT_FOUND"
                        )
                    )
                    if audit is not None:
                        audit.put({"operation": operation, "status": "NOT_FOUND"})
                except BaseException as exc:
                    with contextlib.suppress(BaseException):
                        connection.send_bytes(
                            _rpc_response(error_code="OPERATION_FAILED")
                        )
                    if audit is not None:
                        audit.put(
                            {
                                "operation": operation,
                                "status": "ERROR",
                                "error": exc.__class__.__name__,
                            }
                        )
                finally:
                    connection.close()
        finally:
            listener.close()
            if endpoint.exists() and stat.S_ISSOCK(endpoint.lstat().st_mode):
                endpoint.unlink()


def production_gateway_implementation_sha256() -> str:
    """Bind the exact final server/backend/venue/signer module bytes."""

    return executor.canonical_json_sha256(
        {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "module": __name__,
            "module_sha256": _sha256(Path(__file__).read_bytes()),
            "server": "DeploymentOwnedExecutionGatewayServer",
            "backend": "DeploymentOwnedExecutionGatewayBackend",
            "venue": "PolymarketClobV2VenueBoundary",
            "signer": "RsaSha256ReceiptSigner",
            "rpc_schema_version": executor.EXECUTION_GATEWAY_RPC_SCHEMA_VERSION,
            "required_operations_sha256": (
                executor.REQUIRED_EXECUTION_TRANSPORT_OPERATIONS_SHA256
            ),
        }
    )


def run_production_execution_gateway(
    service_config: ExecutionGatewayServiceConfig,
    venue_config: PolymarketVenueConfig,
    *,
    ready: Any | None = None,
    stop: Any | None = None,
    audit: Any | None = None,
) -> None:
    """Production entrypoint: no injectable backend, signer, wallet, or client."""

    venue_config.validated()
    venue = PolymarketClobV2VenueBoundary(
        private_key_path=venue_config.private_key_path,
        api_credentials_path=venue_config.api_credentials_path,
        host=venue_config.host,
        chain_id=venue_config.chain_id,
        signature_type=venue_config.signature_type,
        funder=venue_config.funder,
    )
    DeploymentOwnedExecutionGatewayServer(
        service_config,
        venue=venue,
    ).serve_forever(ready=ready, stop=stop, audit=audit)
