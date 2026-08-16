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
import queue
import select
import stat
import struct
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
    "bigan-btc-15m-residual-promotion-execution-gateway-state-v3"
)
VENUE_BOUNDARY_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-polymarket-clob-v2-boundary-v2"
)
VENUE_RUNTIME_BINDING_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-venue-runtime-binding-v1"
)


class ExecutionGatewayError(executor.MicroLiveExecutionError):
    """Raised when the deployment-owned gateway must fail closed."""


class ExecutionGatewayDeadlineExceeded(ExecutionGatewayError):
    """Raised when the signed service deadline is exhausted."""


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


def polymarket_exchange_endpoint_sha256(*, host: str, chain_id: int) -> str:
    """Canonical identity of the exact CLOB endpoint and chain."""

    return executor.canonical_json_sha256(
        {"host": host, "chain_id": chain_id}
    )


def polymarket_exchange_account_sha256(
    *,
    funder: str,
    signature_type: int,
) -> str:
    """Canonical identity of the account that the CLOB client writes for."""

    return executor.canonical_json_sha256(
        {
            "funder": funder.lower(),
            "signature_type": signature_type,
        }
    )


def polymarket_signer_identity_sha256(*, signer_address: str) -> str:
    """Canonical public wallet identity derived from the loaded private key."""

    return executor.canonical_json_sha256(
        {"signer_address": signer_address.lower()}
    )


def production_execution_service_identity_sha256(
    *,
    gateway_implementation_sha256: str,
    venue_configuration_sha256: str,
    api_credentials_identity_sha256: str,
    exchange_endpoint_sha256: str,
    exchange_account_sha256: str,
    signer_identity_sha256: str,
) -> str:
    """Bind the signed service identity to the concrete credential owner."""

    values = {
        "gateway_implementation_sha256": gateway_implementation_sha256,
        "venue_configuration_sha256": venue_configuration_sha256,
        "api_credentials_identity_sha256": api_credentials_identity_sha256,
        "exchange_endpoint_sha256": exchange_endpoint_sha256,
        "exchange_account_sha256": exchange_account_sha256,
        "signer_identity_sha256": signer_identity_sha256,
    }
    for field, value in values.items():
        _require_sha256(value, field)
    return executor.canonical_json_sha256(
        {
            "schema_version": VENUE_RUNTIME_BINDING_SCHEMA_VERSION,
            **values,
        }
    )


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


@dataclass(frozen=True, slots=True)
class VenueOrderOutcome:
    """Executor response plus the exact upstream lifecycle bytes."""

    normalized_response: bytes
    raw_venue_response: bytes


class ExactVenueBoundary(Protocol):
    """Only mockable boundary: exact prepared writes and authoritative reads."""

    @property
    def runtime_binding(self) -> Mapping[str, Any]: ...

    def prepare_submission(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def submit_prepared(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> VenueOrderOutcome: ...

    def lookup_submission(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> VenueOrderOutcome | None: ...

    def cancel(self, request: Mapping[str, Any]) -> bytes: ...

    def lookup(self, request: Mapping[str, Any]) -> bytes: ...

    def read_fill_cursor(self, request: Mapping[str, Any]) -> bytes: ...


class PolymarketClobV2VenueBoundary:
    """Concrete credential-owning py-clob-client-v2 venue implementation.

    A signed order is prepared before the dispatch side effect and is persisted
    by the gateway.  Recovery uses its deterministic EIP-712 order hash and is
    lookup-only; it never calls ``post_order``.
    """

    __slots__ = ("_client", "_order_type", "_runtime_binding")

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
        maximum_call_duration_ms: int,
    ) -> None:
        venue_config = PolymarketVenueConfig(
            private_key_path=str(private_key_path),
            api_credentials_path=str(api_credentials_path),
            host=host,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        ).validated()
        if not (
            isinstance(maximum_call_duration_ms, int)
            and not isinstance(maximum_call_duration_ms, bool)
            and maximum_call_duration_ms > 0
        ):
            raise ExecutionGatewayError("venue HTTP deadline is invalid")
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
        try:
            import httpx
            from py_clob_client_v2.http_helpers import helpers as http_helpers
        except ImportError as exc:
            raise ExecutionGatewayError(
                "pinned py-clob-client-v2 HTTP runtime is unavailable"
            ) from exc
        old_http_client = http_helpers._http_client
        http_helpers._http_client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(maximum_call_duration_ms / 1_000),
        )
        with contextlib.suppress(BaseException):
            old_http_client.close()
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
        signer_address = str(self._client.signer.address())
        endpoint_sha256 = polymarket_exchange_endpoint_sha256(
            host=host,
            chain_id=chain_id,
        )
        account_sha256 = polymarket_exchange_account_sha256(
            funder=funder,
            signature_type=signature_type,
        )
        signer_sha256 = polymarket_signer_identity_sha256(
            signer_address=signer_address,
        )
        api_credentials_identity_sha256 = _sha256(_json_bytes(credentials))
        gateway_implementation_sha256 = production_gateway_implementation_sha256()
        venue_configuration_sha256 = venue_config.configuration_sha256
        self._runtime_binding = {
            "schema_version": VENUE_RUNTIME_BINDING_SCHEMA_VERSION,
            "gateway_implementation_sha256": gateway_implementation_sha256,
            "venue_configuration_sha256": venue_configuration_sha256,
            "api_credentials_identity_sha256": api_credentials_identity_sha256,
            "exchange_endpoint_sha256": endpoint_sha256,
            "exchange_account_sha256": account_sha256,
            "signer_identity_sha256": signer_sha256,
            "service_identity_sha256": (
                production_execution_service_identity_sha256(
                    gateway_implementation_sha256=gateway_implementation_sha256,
                    venue_configuration_sha256=venue_configuration_sha256,
                    api_credentials_identity_sha256=(
                        api_credentials_identity_sha256
                    ),
                    exchange_endpoint_sha256=endpoint_sha256,
                    exchange_account_sha256=account_sha256,
                    signer_identity_sha256=signer_sha256,
                )
            ),
        }

    @property
    def runtime_binding(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._runtime_binding)

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
    def _executor_order_response(
        *,
        request: Mapping[str, Any],
        order_hash: str,
        status: str,
    ) -> bytes:
        if status not in {"ACCEPTED", "REJECTED"}:
            raise ExecutionGatewayError("executor order disposition is invalid")
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

    @classmethod
    def _normalized_submission_response(
        cls,
        raw: Mapping[str, Any],
        request: Mapping[str, Any],
        *,
        order_hash: str,
    ) -> VenueOrderOutcome:
        required = {"success", "errorMsg", "orderID", "status"}
        if not required.issubset(raw):
            raise ExecutionGatewayError("venue submission response schema is invalid")
        success = raw["success"]
        error_message = raw["errorMsg"]
        exchange_order_id = raw["orderID"]
        venue_status = str(raw["status"]).lower()
        if not (
            isinstance(success, bool)
            and isinstance(error_message, str)
            and isinstance(exchange_order_id, str)
        ):
            raise ExecutionGatewayError("venue submission response types are invalid")
        if exchange_order_id and exchange_order_id != order_hash:
            raise ExecutionGatewayError("venue order identity is mismatched")
        if success:
            if not (
                exchange_order_id == order_hash
                and error_message == ""
                and venue_status in {"live", "matched"}
            ):
                raise ExecutionGatewayError(
                    "successful venue submission response is contradictory"
                )
            disposition = "ACCEPTED"
        else:
            if not (
                error_message
                and venue_status in {"", "failed", "rejected"}
                and exchange_order_id in {"", order_hash}
            ):
                raise ExecutionGatewayError(
                    "rejected venue submission response is contradictory"
                )
            disposition = "REJECTED"
        return VenueOrderOutcome(
            normalized_response=cls._executor_order_response(
                request=request,
                order_hash=order_hash,
                status=disposition,
            ),
            raw_venue_response=_json_bytes(raw),
        )

    @classmethod
    def _normalized_lookup_response(
        cls,
        raw: Mapping[str, Any],
        request: Mapping[str, Any],
        *,
        order_hash: str,
    ) -> VenueOrderOutcome:
        exchange_order_id = raw.get("id") or raw.get("orderID")
        venue_status = str(raw.get("status") or "").lower()
        if not (
            isinstance(exchange_order_id, str)
            and exchange_order_id == order_hash
            and venue_status
            in {"live", "matched", "canceled", "cancelled", "expired"}
        ):
            raise ExecutionGatewayError("venue lookup response is invalid")
        return VenueOrderOutcome(
            normalized_response=cls._executor_order_response(
                request=request,
                order_hash=order_hash,
                status="ACCEPTED",
            ),
            raw_venue_response=_json_bytes(raw),
        )

    def submit_prepared(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> VenueOrderOutcome:
        if prepared.get("exact_request_sha256") != executor.canonical_json_sha256(
            request
        ):
            raise ExecutionGatewayError("prepared venue request is mismatched")
        raw = self._client.post_order(self._signed_order(prepared), self._order_type)
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue submission response is invalid")
        return self._normalized_submission_response(
            raw,
            request,
            order_hash=str(prepared["order_hash"]),
        )

    def lookup_submission(
        self,
        prepared: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> VenueOrderOutcome | None:
        try:
            raw = self._client.get_order(str(prepared["order_hash"]))
        except BaseException as exc:  # third-party exception taxonomy is unstable
            if "not found" in str(exc).lower() or "404" in str(exc):
                return None
            raise ExecutionGatewayError("venue recovery lookup failed closed") from exc
        if not isinstance(raw, Mapping) or not raw:
            return None
        return self._normalized_lookup_response(
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
        canceled = raw.get("canceled")
        not_canceled = raw.get("not_canceled")
        if not (
            isinstance(canceled, list)
            and canceled == [str(request["exchange_order_id"])]
            and isinstance(not_canceled, Mapping)
            and not not_canceled
        ):
            raise ExecutionGatewayError("venue cancel acknowledgement is invalid")
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
        return self._normalized_lookup_response(
            raw,
            request,
            order_hash=str(request["exchange_order_id"]),
        ).normalized_response

    def read_fill_cursor(self, request: Mapping[str, Any]) -> bytes:
        raw = self._client.get_order(str(request["exchange_order_id"]))
        if not isinstance(raw, Mapping):
            raise ExecutionGatewayError("venue fill cursor response is invalid")
        venue_status = str(raw.get("status") or "").upper()
        if venue_status not in {
            "LIVE",
            "MATCHED",
            "CANCELED",
            "CANCELLED",
            "EXPIRED",
        }:
            raise ExecutionGatewayError("venue order lifecycle status is invalid")
        try:
            filled = Decimal(
                str(raw.get("size_matched") or raw.get("filled_size") or "0")
            )
            original_size = Decimal(
                str(raw.get("original_size") or raw.get("size") or "0")
            )
        except Exception as exc:
            raise ExecutionGatewayError("venue order sizes are invalid") from exc
        if filled < 0 or original_size <= 0 or filled > original_size:
            raise ExecutionGatewayError("venue order sizes are invalid")
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
        if filled == original_size:
            status = "FILLED"
        elif venue_status in {"CANCELED", "CANCELLED"}:
            status = "CANCELED"
        elif venue_status == "EXPIRED":
            status = "EXPIRED"
        else:
            status = "OPEN"
        terminal = status != "OPEN"
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
            if str(trade.get("status") or "").upper() not in {
                "MATCHED",
                "MINED",
                "CONFIRMED",
                "TRADE_STATUS_MATCHED",
                "TRADE_STATUS_MINED",
                "TRADE_STATUS_CONFIRMED",
            }:
                raise ExecutionGatewayError(
                    "authoritative venue trade is not an executed fill"
                )
            is_taker = str(trade.get("taker_order_id") or "") == order_id
            leg: Mapping[str, Any] = trade
            if not is_taker:
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
                fee_value = leg.get("fee_usd")
                if fee_value is None:
                    fee_value = leg.get("fee")
                if is_taker and fee_value is None:
                    fee_value = trade.get("fee_usd")
                if is_taker and fee_value is None:
                    fee_value = trade.get("fee")
                if fee_value is not None:
                    fee = Decimal(str(fee_value))
                elif not is_taker:
                    # The active V2 market contract declares taker-only fees.
                    # A maker leg must not inherit the top-level taker fee.
                    fee = Decimal("0")
                else:
                    # `fee_rate_bps` alone is insufficient for the current V2
                    # dynamic-fee contract: role, market fee descriptor, the
                    # p*(1-p) term, and five-decimal venue rounding are all
                    # material.  Until the frozen candidate/cost transition is
                    # separately authorized, require the authoritative
                    # absolute taker fee instead of fabricating economics.
                    raise ValueError(
                        "absolute taker fee is required by the frozen contract"
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
                "preparation_reservations": {},
                "prepared": {},
                "venue_lifecycle_base64": {},
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


def _validated_venue_runtime_binding(
    venue: ExactVenueBoundary,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        binding = dict(venue.runtime_binding)
    except Exception as exc:
        raise ExecutionGatewayError("venue runtime binding is unavailable") from exc
    required = {
        "schema_version",
        "gateway_implementation_sha256",
        "venue_configuration_sha256",
        "api_credentials_identity_sha256",
        "exchange_endpoint_sha256",
        "exchange_account_sha256",
        "signer_identity_sha256",
        "service_identity_sha256",
    }
    if set(binding) != required or binding.get("schema_version") != (
        VENUE_RUNTIME_BINDING_SCHEMA_VERSION
    ):
        raise ExecutionGatewayError("venue runtime binding is not exact")
    for field in required - {"schema_version"}:
        _require_sha256(binding[field], f"venue runtime binding {field}")
    expected_service_identity = production_execution_service_identity_sha256(
        gateway_implementation_sha256=production_gateway_implementation_sha256(),
        venue_configuration_sha256=binding["venue_configuration_sha256"],
        api_credentials_identity_sha256=binding[
            "api_credentials_identity_sha256"
        ],
        exchange_endpoint_sha256=binding["exchange_endpoint_sha256"],
        exchange_account_sha256=binding["exchange_account_sha256"],
        signer_identity_sha256=binding["signer_identity_sha256"],
    )
    if not (
        binding["gateway_implementation_sha256"]
        == production_gateway_implementation_sha256()
        and binding["service_identity_sha256"] == expected_service_identity
        and authority["service_identity_sha256"] == expected_service_identity
        and authority["exchange_endpoint_sha256"]
        == binding["exchange_endpoint_sha256"]
        and authority["exchange_account_sha256"]
        == binding["exchange_account_sha256"]
        and authority["signer_identity_sha256"]
        == binding["signer_identity_sha256"]
    ):
        raise ExecutionGatewayError(
            "concrete gateway venue/wallet identity is mismatched"
        )
    return copy.deepcopy(binding)


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
        self.venue_runtime_binding = _validated_venue_runtime_binding(
            venue,
            self.config.execution_authority,
        )
        self.venue = venue
        self.state = state
        self.signer = signer
        self._state_lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._dispatch_transition_lock = threading.Lock()
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
        with self._state_lock:
            self.state.value["terminal_receipts_base64"][key] = base64.b64encode(
                terminal
            ).decode("ascii")
            self.state.value["venue_outcomes_base64"][key] = base64.b64encode(
                raw_response
            ).decode("ascii")
            self.state.flush()
        return terminal

    def _persist_venue_lifecycle(
        self,
        key: str,
        outcome: VenueOrderOutcome,
    ) -> bytes:
        if not (
            isinstance(outcome, VenueOrderOutcome)
            and isinstance(outcome.normalized_response, bytes)
            and isinstance(outcome.raw_venue_response, bytes)
        ):
            raise ExecutionGatewayError("venue outcome contract is invalid")
        _, canonical_lifecycle, _ = executor._raw_json_object(
            outcome.raw_venue_response,
            "raw venue lifecycle",
            maximum_bytes=executor.MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        if outcome.raw_venue_response != canonical_lifecycle.encode("utf-8"):
            raise ExecutionGatewayError("raw venue lifecycle is not canonical")
        executor._raw_json_object(
            outcome.normalized_response,
            "normalized venue outcome",
            maximum_bytes=executor.MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
        )
        with self._state_lock:
            self.state.value["venue_lifecycle_base64"][key] = base64.b64encode(
                outcome.raw_venue_response
            ).decode("ascii")
            self.state.flush()
        return outcome.normalized_response

    def _authenticated_response(
        self,
        request: Mapping[str, Any],
        raw_response: bytes,
    ) -> bytes:
        authentication = dict(request["execution_authentication"])
        submit = authentication["operation"] == "submit_order"
        terminal = None
        if submit:
            with self._state_lock:
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
        with self._submit_lock:
            proof = self._proof(request)
            if self._dispatch_begin is None:
                raise ExecutionGatewayError("gateway dispatch begin is unbound")
            key = str(request["client_order_id"])
            exact_request = copy.deepcopy(dict(request))
            exact_request_sha256 = executor.canonical_json_sha256(exact_request)
            with self._state_lock:
                existing_request = self.state.value["requests"].get(key)
                if existing_request is not None and existing_request != exact_request:
                    raise ExecutionGatewayError(
                        "gateway preparation reservation identity conflicts"
                    )
                reservation = self.state.value["preparation_reservations"].get(key)
                if reservation is None:
                    reservation = {
                        "exact_request_sha256": exact_request_sha256,
                        "status": "PREPARING",
                    }
                    self.state.value["preparation_reservations"][key] = reservation
                if not (
                    isinstance(reservation, Mapping)
                    and reservation.get("exact_request_sha256")
                    == exact_request_sha256
                    and reservation.get("status")
                    in {"PREPARING", "PREPARED", "DISPATCHING"}
                ):
                    raise ExecutionGatewayError(
                        "gateway preparation reservation is not dispatchable"
                    )
                self.state.value["requests"][key] = exact_request
                prepared = copy.deepcopy(self.state.value["prepared"].get(key))
                self.state.flush()

            # Order construction may query the venue or wallet.  It must never
            # hold the state lock or consume the externally fenceable dispatch
            # grant while it is in progress.
            if prepared is None:
                candidate = self.venue.prepare_submission(request)
                if not isinstance(candidate, Mapping):
                    raise ExecutionGatewayError(
                        "prepared venue submission is invalid"
                    )
                prepared = copy.deepcopy(dict(candidate))
                if not (
                    prepared.get("client_order_id") == key
                    and prepared.get("exact_request_sha256")
                    == exact_request_sha256
                    and isinstance(prepared.get("order_hash"), str)
                    and prepared["order_hash"]
                ):
                    raise ExecutionGatewayError(
                        "prepared venue submission identity is mismatched"
                    )
                with self._state_lock:
                    reservation = self.state.value[
                        "preparation_reservations"
                    ].get(key)
                    if not (
                        isinstance(reservation, Mapping)
                        and reservation.get("exact_request_sha256")
                        == exact_request_sha256
                        and reservation.get("status") == "PREPARING"
                    ):
                        raise ExecutionGatewayError(
                            "gateway preparation was fenced before dispatch"
                        )
                    self.state.value["prepared"][key] = prepared
                    reservation["status"] = "PREPARED"
                    self.state.flush()  # signed bytes precede every venue write
            elif not (
                isinstance(prepared, Mapping)
                and prepared.get("client_order_id") == key
                and prepared.get("exact_request_sha256") == exact_request_sha256
                and isinstance(prepared.get("order_hash"), str)
                and prepared["order_hash"]
            ):
                raise ExecutionGatewayError(
                    "durable prepared venue submission is mismatched"
                )

            # Serialize the final local fence check with consumption of the
            # authority dispatch grant.  A fence arriving while preparation is
            # stalled wins; after the grant is consumed it honestly reports an
            # in-progress side effect instead of claiming a false fence.
            with self._dispatch_transition_lock:
                with self._state_lock:
                    reservation = self.state.value[
                        "preparation_reservations"
                    ].get(key)
                    if not (
                        isinstance(reservation, Mapping)
                        and reservation.get("exact_request_sha256")
                        == exact_request_sha256
                        and reservation.get("status")
                        in {"PREPARED", "DISPATCHING"}
                    ):
                        raise ExecutionGatewayError(
                            "gateway preparation is not dispatchable"
                        )
                raw_dispatch = self._dispatch_begin(
                    proof["outbox_acceptance_receipt_json"].encode("utf-8"),
                    venue_idempotency_key=key,
                    venue_idempotency_scope=executor.VENUE_IDEMPOTENCY_SCOPE,
                    dispatch_deadline_ts_ms=request["execution_authentication"][
                        "dispatch_deadline_ts_ms"
                    ],
                    authorization_expires_at_ts_ms=request[
                        "execution_authentication"
                    ]["authorization_expires_at_ts_ms"],
                )
                dispatch, _, _ = executor._raw_json_object(
                    raw_dispatch,
                    "gateway dispatch receipt",
                )
                if dispatch["status"] == "DISPATCHED":
                    raw_response = dispatch["raw_outcome_json"].encode("utf-8")
                    with self._state_lock:
                        self.state.value["terminal_receipts_base64"][key] = (
                            base64.b64encode(raw_dispatch).decode("ascii")
                        )
                        reservation["status"] = "DISPATCHING"
                        self.state.flush()
                    return self._authenticated_response(request, raw_response)
                if dispatch["status"] == "IN_PROGRESS":
                    raise ExecutionGatewayError(
                        "gateway dispatch is in progress; recovery is required"
                    )
                if dispatch["status"] != "DISPATCHING":
                    with self._state_lock:
                        reservation["status"] = "FENCED"
                        self.state.flush()
                    raise ExecutionGatewayError(
                        "gateway dispatch is not permitted"
                    )
                with self._state_lock:
                    self.state.value["dispatch_receipts_base64"][key] = (
                        base64.b64encode(raw_dispatch).decode("ascii")
                    )
                    reservation["status"] = "DISPATCHING"
                    self.state.flush()
            outcome = self.venue.submit_prepared(prepared, request)
            raw_response = self._persist_venue_lifecycle(key, outcome)
            self._complete(key, raw_dispatch, raw_response)
            return self._authenticated_response(request, raw_response)

    def recover_order_submission(self, request: Mapping[str, Any]) -> bytes:
        self._proof(request)
        key = str(request["client_order_id"])
        with self._state_lock:
            prepared = copy.deepcopy(self.state.value["prepared"].get(key))
        if not isinstance(prepared, Mapping):
            raise executor.SubmissionRecoveryOutcomeNotFoundError(
                "gateway has no durable prepared submission"
            )
        outcome = self.venue.lookup_submission(prepared, request)
        if outcome is None:
            raise executor.SubmissionRecoveryOutcomeNotFoundError(
                "venue idempotency lookup found no outcome"
            )
        raw_response = self._persist_venue_lifecycle(key, outcome)
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
        with self._state_lock:
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
        key = str(request["client_order_id"])
        with self._state_lock:
            original = copy.deepcopy(self.state.value["requests"].get(key))
            prepared = copy.deepcopy(self.state.value["prepared"].get(key))
        if not isinstance(original, Mapping) or not isinstance(prepared, Mapping):
            raise ExecutionGatewayError("gateway lookup identity is not durable")
        venue_request = {
            **dict(request),
            "exchange_order_id": prepared["order_hash"],
            "quantity": original["quantity"],
            "limit_price": original["limit_price"],
        }
        return self._authenticated_response(
            request,
            self.venue.lookup(venue_request),
        )

    def read_order_fill_cursor(self, request: Mapping[str, Any]) -> bytes:
        raw = self.venue.read_fill_cursor(request)
        value, _, _ = executor._raw_json_object(raw, "venue fill cursor")
        payload_sha256 = executor.canonical_json_sha256(value)
        return self._signed({**value, "cursor_payload_sha256": payload_sha256})

    def fence_order_invocation(self, request: Mapping[str, Any]) -> bytes:
        key = str(request["client_order_id"])
        with self._dispatch_transition_lock:
            with self._state_lock:
                submitted = copy.deepcopy(self.state.value["requests"].get(key))
                reservation = self.state.value["preparation_reservations"].get(key)
                if not (
                    isinstance(submitted, Mapping)
                    and isinstance(reservation, Mapping)
                    and reservation.get("exact_request_sha256")
                    == executor.canonical_json_sha256(submitted)
                    and reservation.get("status")
                    in {"PREPARING", "PREPARED", "DISPATCHING", "FENCED"}
                    and self._dispatch_fence is not None
                ):
                    raise ExecutionGatewayError("gateway dispatch fence is unbound")
                if reservation["status"] in {"PREPARING", "PREPARED"}:
                    reservation["status"] = "FENCED"
                    self.state.flush()
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

    _MAXIMUM_CONCURRENT_CONNECTIONS = 64
    _READ_LANE_CAPACITY = 16
    _CONTROL_LANE_OPERATIONS = frozenset(
        {
            "attest_execution_binding",
            "bind_execution_dispatch_authority",
        }
    )
    _READ_LANE_OPERATIONS = frozenset(
        {
            "read_trusted_time",
            "lookup_order",
            "read_order_fill_cursor",
        }
    )
    _RECOVERY_LANE_OPERATIONS = frozenset({"recover_order_submission"})
    _CANCEL_LANE_OPERATIONS = frozenset({"cancel_order"})
    _FENCE_LANE_OPERATIONS = frozenset({"fence_order_invocation"})

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
        self.venue_runtime_binding = _validated_venue_runtime_binding(
            venue,
            self.config.execution_authority,
        )
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
        self._connection_slots = threading.BoundedSemaphore(
            self._MAXIMUM_CONCURRENT_CONNECTIONS
        )
        self._submit_lane = threading.BoundedSemaphore(1)
        self._control_lane = threading.BoundedSemaphore(4)
        self._read_lane = threading.BoundedSemaphore(self._READ_LANE_CAPACITY)
        self._recovery_lane = threading.BoundedSemaphore(4)
        # Fence and cancel admission is reserved.  Neither operation can be
        # queued behind network-backed lookup/recovery work.
        self._cancel_lane = threading.BoundedSemaphore(1)
        self._fence_lane = threading.BoundedSemaphore(1)
        self._active_handlers: set[threading.Thread] = set()
        self._active_handlers_lock = threading.Lock()

    @property
    def _maximum_call_duration_seconds(self) -> float:
        return (
            int(self.config.execution_authority["maximum_call_duration_ms"])
            / 1_000
        )

    @staticmethod
    def _remaining_seconds(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionGatewayDeadlineExceeded(
                "execution gateway service deadline exceeded"
            )
        return remaining

    def _lane_for_operation(self, operation: str) -> threading.BoundedSemaphore:
        if operation == "submit_order":
            return self._submit_lane
        if operation in self._CONTROL_LANE_OPERATIONS:
            return self._control_lane
        if operation in self._READ_LANE_OPERATIONS:
            return self._read_lane
        if operation in self._RECOVERY_LANE_OPERATIONS:
            return self._recovery_lane
        if operation in self._CANCEL_LANE_OPERATIONS:
            return self._cancel_lane
        if operation in self._FENCE_LANE_OPERATIONS:
            return self._fence_lane
        raise ExecutionGatewayError("execution gateway operation has no lane")

    def _invoke_with_deadline(
        self,
        *,
        operation: str,
        deadline: float,
        call: Callable[[], bytes],
    ) -> bytes:
        result: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        lane = self._lane_for_operation(operation)
        acquired = lane.acquire(timeout=self._remaining_seconds(deadline))
        if not acquired:
            raise ExecutionGatewayDeadlineExceeded(
                "execution gateway service lane is unavailable"
            )

        caller_reclaims_lane = operation in self._READ_LANE_OPERATIONS

        def invoke() -> None:
            try:
                value = call()
                with contextlib.suppress(queue.Full):
                    result.put_nowait(("OK", value))
            except BaseException as exc:
                with contextlib.suppress(queue.Full):
                    result.put_nowait(("ERROR", exc))
            finally:
                if not caller_reclaims_lane:
                    lane.release()

        started = False
        try:
            worker = threading.Thread(
                target=invoke,
                name=f"execution-gateway-{operation}",
                daemon=True,
            )
            worker.start()
            started = True
            try:
                status, value = result.get(
                    timeout=self._remaining_seconds(deadline)
                )
            except queue.Empty as exc:
                raise ExecutionGatewayDeadlineExceeded(
                    "execution gateway service operation timed out"
                ) from exc
        finally:
            if caller_reclaims_lane or not started:
                # Read-only admission belongs to the deadline-bound caller,
                # not to an abandoned backend worker.  Mutating operations
                # retain their isolated lane until the worker exits, so a late
                # side effect cannot overlap another same-class mutation.
                lane.release()
        if status != "OK":
            raise value
        if not isinstance(value, bytes):
            raise ExecutionGatewayError("gateway backend returned non-byte response")
        return value

    def _authenticate_connection_with_deadline(
        self,
        connection: Any,
        *,
        credential: bytes,
        deadline: float,
    ) -> None:
        result: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

        def authenticate() -> None:
            try:
                multiprocessing.connection.deliver_challenge(
                    connection,
                    credential,
                )
                multiprocessing.connection.answer_challenge(
                    connection,
                    credential,
                )
            except BaseException as exc:
                with contextlib.suppress(queue.Full):
                    result.put_nowait(exc)
            else:
                with contextlib.suppress(queue.Full):
                    result.put_nowait(None)

        threading.Thread(
            target=authenticate,
            name="execution-gateway-authentication",
            daemon=True,
        ).start()
        try:
            outcome = result.get(timeout=self._remaining_seconds(deadline))
        except queue.Empty as exc:
            # Closing the connection interrupts the abandoned authentication
            # worker without allowing it to occupy an admission slot forever.
            connection.close()
            raise ExecutionGatewayDeadlineExceeded(
                "execution gateway authentication timed out"
            ) from exc
        if outcome is not None:
            raise ExecutionGatewayError(
                "execution gateway authentication failed closed"
            ) from outcome

    def _read_exact_with_deadline(
        self,
        connection: Any,
        *,
        byte_count: int,
        deadline: float,
    ) -> bytes:
        chunks: list[bytes] = []
        remaining_bytes = byte_count
        descriptor = connection.fileno()
        while remaining_bytes:
            try:
                readable, _, _ = select.select(
                    [descriptor],
                    [],
                    [],
                    self._remaining_seconds(deadline),
                )
            except InterruptedError:
                continue
            if not readable:
                raise ExecutionGatewayDeadlineExceeded(
                    "execution gateway request payload timed out"
                )
            chunk = os.read(descriptor, remaining_bytes)
            if not chunk:
                raise EOFError("execution gateway request payload ended early")
            chunks.append(chunk)
            remaining_bytes -= len(chunk)
        return b"".join(chunks)

    def _recv_bytes_with_deadline(
        self,
        connection: Any,
        *,
        maximum_bytes: int,
        deadline: float,
    ) -> bytes:
        """Read one complete multiprocessing frame under the RPC deadline."""

        header = self._read_exact_with_deadline(
            connection,
            byte_count=4,
            deadline=deadline,
        )
        (message_size,) = struct.unpack("!i", header)
        if message_size == -1:
            extended_header = self._read_exact_with_deadline(
                connection,
                byte_count=8,
                deadline=deadline,
            )
            (message_size,) = struct.unpack("!Q", extended_header)
        if message_size < 0 or message_size > maximum_bytes:
            raise ExecutionGatewayError(
                "execution gateway request payload size is invalid"
            )
        return self._read_exact_with_deadline(
            connection,
            byte_count=message_size,
            deadline=deadline,
        )

    def _handle_connection(
        self,
        connection: Any,
        audit: Any | None,
        *,
        deadline: float,
    ) -> None:
        operation = "invalid"
        try:
            raw_request = self._recv_bytes_with_deadline(
                connection,
                maximum_bytes=executor.MAX_EXECUTION_TRANSPORT_EVENT_BYTES,
                deadline=deadline,
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
                    self.config.execution_authority["public_key_modulus_hex"]
                ),
                public_key_exponent=int(
                    self.config.execution_authority["public_key_exponent"]
                ),
            )
            operation = str(verified["operation"])
            session_sha256 = str(verified["client_session_sha256"])
            payload = _decode_rpc_value(verified["payload"])
            if operation == "attest_execution_binding":
                if not (
                    payload.get("client_session_sha256") == session_sha256
                    and payload.get("client_session_binding")
                    == verified["client_session_binding"]
                ):
                    raise ExecutionGatewayError(
                        "gateway attestation session is mismatched"
                    )
                raw_response = self._invoke_with_deadline(
                    operation=operation,
                    deadline=deadline,
                    call=lambda: self.backend.attest_execution_binding(payload),
                )
                with self.backend._state_lock:
                    self.state.value["sessions"][session_sha256] = {
                        "client_session_binding": copy.deepcopy(
                            verified["client_session_binding"]
                        ),
                        "attestation_sha256": _sha256(raw_response),
                    }
                    self.state.flush()
            else:
                with self.backend._state_lock:
                    session = copy.deepcopy(
                        self.state.value["sessions"].get(session_sha256)
                    )
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

                    def bind() -> bytes:
                        executor.bind_execution_backend_to_dispatch_authority(
                            self.backend,
                            **payload,
                        )
                        return b'{"bound":true}'

                    raw_response = self._invoke_with_deadline(
                        operation=operation,
                        deadline=deadline,
                        call=bind,
                    )
                else:
                    method = getattr(self.backend, operation)
                    raw_response = self._invoke_with_deadline(
                        operation=operation,
                        deadline=deadline,
                        call=lambda: method(payload),
                    )
            connection.send_bytes(_rpc_response(raw_response=raw_response))
            if audit is not None:
                audit.put({"operation": operation, "status": "OK"})
        except executor.SubmissionRecoveryOutcomeNotFoundError:
            with contextlib.suppress(BaseException):
                connection.send_bytes(
                    _rpc_response(error_code="SUBMISSION_RECOVERY_OUTCOME_NOT_FOUND")
                )
            if audit is not None:
                audit.put({"operation": operation, "status": "NOT_FOUND"})
        except ExecutionGatewayDeadlineExceeded:
            with contextlib.suppress(BaseException):
                connection.send_bytes(_rpc_response(error_code="DEADLINE_EXCEEDED"))
            if audit is not None:
                audit.put({"operation": operation, "status": "DEADLINE_EXCEEDED"})
        except BaseException as exc:
            with contextlib.suppress(BaseException):
                connection.send_bytes(_rpc_response(error_code="OPERATION_FAILED"))
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

    def _run_connection_handler(
        self,
        connection: Any,
        audit: Any | None,
        credential: bytes,
    ) -> None:
        deadline = time.monotonic() + self._maximum_call_duration_seconds
        try:
            self._authenticate_connection_with_deadline(
                connection,
                credential=credential,
                deadline=deadline,
            )
            self._handle_connection(
                connection,
                audit,
                deadline=deadline,
            )
        except ExecutionGatewayDeadlineExceeded:
            if audit is not None:
                audit.put(
                    {"operation": "authentication", "status": "DEADLINE_EXCEEDED"}
                )
        except BaseException as exc:
            if audit is not None:
                audit.put(
                    {
                        "operation": "authentication",
                        "status": "ERROR",
                        "error": exc.__class__.__name__,
                    }
                )
        finally:
            with contextlib.suppress(BaseException):
                connection.close()
            with self._active_handlers_lock:
                self._active_handlers.discard(threading.current_thread())
            self._connection_slots.release()

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
            # Authentication runs in the admitted handler so a raw client
            # cannot block the single accept loop before a deadline exists.
            authkey=None,
        )
        raw_socket = listener._listener._socket
        raw_socket.settimeout(
            min(0.1, self._maximum_call_duration_seconds)
        )
        if ready is not None:
            ready.set()
        try:
            while True:
                if stop is not None and stop.is_set():
                    break
                try:
                    connection = listener.accept()
                except TimeoutError:
                    continue
                if stop is not None and stop.is_set():
                    connection.close()
                    break
                if not self._connection_slots.acquire(blocking=False):
                    connection.close()
                    continue
                handler = threading.Thread(
                    target=self._run_connection_handler,
                    args=(connection, audit, credential),
                    name="execution-gateway-connection",
                    daemon=True,
                )
                with self._active_handlers_lock:
                    self._active_handlers.add(handler)
                handler.start()
        finally:
            listener.close()
            shutdown_deadline = (
                time.monotonic() + self._maximum_call_duration_seconds
            )
            with self._active_handlers_lock:
                handlers = tuple(self._active_handlers)
            for handler in handlers:
                handler.join(timeout=max(0, shutdown_deadline - time.monotonic()))
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

    service_config.validated()
    venue_config.validated()
    venue = PolymarketClobV2VenueBoundary(
        private_key_path=venue_config.private_key_path,
        api_credentials_path=venue_config.api_credentials_path,
        host=venue_config.host,
        chain_id=venue_config.chain_id,
        signature_type=venue_config.signature_type,
        funder=venue_config.funder,
        maximum_call_duration_ms=int(
            service_config.execution_authority["maximum_call_duration_ms"]
        ),
    )
    DeploymentOwnedExecutionGatewayServer(
        service_config,
        venue=venue,
    ).serve_forever(ready=ready, stop=stop, audit=audit)
