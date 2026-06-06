"""Polymarket CLOB execution client wrapper.

The wrapper is intentionally dry-run-first. Production code can inject an
already configured py-clob-client instance, while tests and paper execution can
use the same public interface without importing py-clob-client or placing real
orders.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

OrderSide = Literal["BUY", "SELL"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClobExecutionConfig:
    """Runtime configuration for CLOB execution."""

    dry_run: bool = True
    host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    signature_type: str = "POLY_PROXY"
    signature_type_env: str = "POLYMARKET_SIGNATURE_TYPE"
    api_auth_mode: str = "derive"
    api_auth_mode_env: str = "POLYMARKET_CLOB_AUTH_MODE"
    api_key_env: str = "POLYMARKET_API_KEY"
    api_secret_env: str = "POLYMARKET_API_SECRET"
    api_passphrase_env: str = "POLYMARKET_API_PASSPHRASE"
    private_key_env: str = "POLYMARKET_PRIVATE_KEY"
    funder_env: str = "POLYMARKET_FUNDER"
    default_order_type: str = "GTC"
    max_retries: int = 3
    retry_base_delay_seconds: float = 0.25
    min_request_interval_seconds: float = 0.10


@dataclass(frozen=True, slots=True)
class OrderStatus:
    """Normalized CLOB order status."""

    order_id: str
    status: str
    filled_size: float = 0.0
    remaining_size: float | None = None
    avg_fill_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ClobExecutionError(RuntimeError):
    """Base class for execution-layer failures."""


class InsufficientBalanceError(ClobExecutionError):
    """Raised when CLOB rejects an order for insufficient balance/collateral."""


class RateLimitError(ClobExecutionError):
    """Raised when CLOB rate limits requests after retry attempts are exhausted."""


class ClobExecutionClient:
    """Small production wrapper around Polymarket CLOB order operations."""

    def __init__(
        self,
        api_client: Any | None = None,
        *,
        config: ClobExecutionConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or ClobExecutionConfig()
        self._client = api_client
        self._logger = logger or LOGGER
        self._last_request_at = 0.0
        self.dry_run_orders: list[dict[str, Any]] = []
        if self._client is None and not self.config.dry_run:
            self._client = self._build_py_clob_client()

    def place_limit_order(
        self,
        token_id: str,
        side: OrderSide | str,
        price: float,
        size: float,
    ) -> str:
        """Place a limit order and return its order id."""

        side_text = _normalise_side(side)
        _validate_order(token_id=token_id, price=price, size=size)
        if self.config.dry_run:
            order_id = _dry_run_order_id(token_id, side_text, price, size, len(self.dry_run_orders))
            self.dry_run_orders.append(
                {
                    "order_id": order_id,
                    "token_id": token_id,
                    "side": side_text,
                    "price": float(price),
                    "size": float(size),
                    "order_type": self.config.default_order_type,
                    "created_at": _now_ms(),
                }
            )
            self._logger.info(
                "dry-run CLOB order skipped",
                extra={
                    "order_id": order_id,
                    "token_id": token_id,
                    "side": side_text,
                    "price": price,
                    "size": size,
                },
            )
            return order_id

        def op() -> Any:
            assert self._client is not None
            if hasattr(self._client, "place_limit_order"):
                return self._client.place_limit_order(
                    token_id=token_id,
                    side=side_text,
                    price=price,
                    size=size,
                )
            if hasattr(self._client, "place_order"):
                return self._client.place_order(token_id, side_text, price, size)
            if hasattr(self._client, "create_order") and hasattr(self._client, "post_order"):
                return self._post_py_clob_limit_order(token_id, side_text, price, size)
            raise ClobExecutionError("api_client does not expose a supported order placement method")

        response = self._call_with_retry(op)
        return _extract_order_id(response)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""

        _require_text("order_id", order_id)
        if self.config.dry_run:
            self._logger.info("dry-run CLOB cancel skipped", extra={"order_id": order_id})
            return True

        def op() -> Any:
            assert self._client is not None
            if hasattr(self._client, "cancel_order"):
                return self._client.cancel_order(order_id)
            if hasattr(self._client, "cancel"):
                return self._client.cancel(order_id)
            raise ClobExecutionError("api_client does not expose cancel_order/cancel")

        return _truthy_success(self._call_with_retry(op))

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return normalized status for an order."""

        _require_text("order_id", order_id)
        if self.config.dry_run:
            return OrderStatus(order_id=order_id, status="dry_run")

        def op() -> Any:
            assert self._client is not None
            if hasattr(self._client, "get_order_status"):
                return self._client.get_order_status(order_id)
            if hasattr(self._client, "get_order"):
                return self._client.get_order(order_id)
            raise ClobExecutionError("api_client does not expose get_order_status/get_order")

        return _normalise_order_status(order_id, self._call_with_retry(op))

    def get_best_bid_ask(self, token_id: str) -> tuple[float | None, float | None]:
        """Return best bid and ask for a token."""

        _require_text("token_id", token_id)
        if self.config.dry_run and self._client is None:
            return None, None

        def op() -> Any:
            assert self._client is not None
            if hasattr(self._client, "get_best_bid_ask"):
                return self._client.get_best_bid_ask(token_id)
            if hasattr(self._client, "get_order_book"):
                return self._client.get_order_book(token_id)
            if hasattr(self._client, "get_price"):
                bid = self._client.get_price(token_id, side="SELL")
                ask = self._client.get_price(token_id, side="BUY")
                return {"bid": bid, "ask": ask}
            raise ClobExecutionError("api_client does not expose a quote method")

        return _normalise_bid_ask(self._call_with_retry(op))

    def _call_with_retry(self, operation: Any) -> Any:
        last_exc: BaseException | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            self._respect_rate_limit()
            try:
                return operation()
            except BaseException as exc:  # noqa: BLE001 - classify third-party client errors.
                if _is_insufficient_balance(exc):
                    self._logger.warning("CLOB order skipped for insufficient balance: %s", exc)
                    raise InsufficientBalanceError(str(exc)) from exc
                last_exc = exc
                if attempt == attempts - 1 or not _is_retryable(exc):
                    break
                time.sleep(self.config.retry_base_delay_seconds * (2**attempt))
        if last_exc is None:
            raise ClobExecutionError("CLOB operation failed")
        if _is_rate_limit(last_exc):
            raise RateLimitError(str(last_exc)) from last_exc
        raise ClobExecutionError(str(last_exc)) from last_exc

    def _respect_rate_limit(self) -> None:
        delay = max(0.0, self.config.min_request_interval_seconds)
        if delay == 0:
            return
        now = time.monotonic()
        wait_seconds = self._last_request_at + delay - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.monotonic()

    def _build_py_clob_client(self) -> Any:
        private_key = os.getenv(self.config.private_key_env)
        if not private_key:
            raise ClobExecutionError(
                f"{self.config.private_key_env} is required when dry_run is false"
            )
        try:
            return self._build_py_clob_client_v2(private_key)
        except ImportError:
            return self._build_py_clob_client_v1(private_key)

    def _build_py_clob_client_v2(self, private_key: str) -> Any:
        try:
            from py_clob_client_v2 import (  # type: ignore[import-not-found]
                ClobClient,
                SignatureTypeV2,
            )
            from py_clob_client_v2.clob_types import ApiCreds  # type: ignore[import-not-found]
        except ImportError as exc:
            raise exc

        signature_type = _resolve_signature_type(
            SignatureTypeV2,
            os.getenv(self.config.signature_type_env, self.config.signature_type),
        )
        kwargs: dict[str, Any] = {
            "host": os.getenv("POLYMARKET_HOST", self.config.host),
            "key": private_key,
            "chain_id": self.config.chain_id,
            "signature_type": signature_type,
        }
        funder = os.getenv(self.config.funder_env) or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        if funder:
            kwargs["funder"] = funder
        client = ClobClient(**kwargs)
        self._configure_v2_api_creds(client, ApiCreds)
        return client

    def _configure_v2_api_creds(self, client: Any, api_creds_type: Any) -> None:
        mode = os.getenv(self.config.api_auth_mode_env, self.config.api_auth_mode)
        mode = str(mode).strip().lower()
        if mode in {"", "none", "disabled"}:
            return
        if mode in {"derive", "derived"}:
            if not hasattr(client, "create_or_derive_api_key") or not hasattr(client, "set_api_creds"):
                raise ClobExecutionError("py-clob-client-v2 does not support derived API creds")
            client.set_api_creds(client.create_or_derive_api_key())
            return
        if mode in {"env", "static"}:
            values = {
                "api_key": os.getenv(self.config.api_key_env),
                "api_secret": os.getenv(self.config.api_secret_env),
                "api_passphrase": os.getenv(self.config.api_passphrase_env),
            }
            missing = [
                env_name
                for env_name, value in (
                    (self.config.api_key_env, values["api_key"]),
                    (self.config.api_secret_env, values["api_secret"]),
                    (self.config.api_passphrase_env, values["api_passphrase"]),
                )
                if not value
            ]
            if missing:
                raise ClobExecutionError(
                    "static CLOB API auth requested but missing env vars: "
                    + ", ".join(missing)
                )
            if not hasattr(client, "set_api_creds"):
                raise ClobExecutionError("py-clob-client-v2 does not support set_api_creds")
            client.set_api_creds(api_creds_type(**values))
            return
        raise ClobExecutionError(
            f"unsupported {self.config.api_auth_mode_env}/{self.config.api_auth_mode}: {mode}"
        )

    def _build_py_clob_client_v1(self, private_key: str) -> Any:
        try:
            from py_clob_client.client import ClobClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ClobExecutionError(
                "py-clob-client-v2 or py-clob-client is required for live CLOB execution"
            ) from exc

        kwargs: dict[str, Any] = {
            "host": os.getenv("POLYMARKET_HOST", self.config.host),
            "key": private_key,
            "chain_id": self.config.chain_id,
        }
        funder = os.getenv(self.config.funder_env) or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        if funder:
            kwargs["funder"] = funder
        return ClobClient(**kwargs)

    def _post_py_clob_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Any:
        assert self._client is not None
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore[import-not-found]  # noqa: I001
        except ImportError as exc:
            raise ClobExecutionError(
                "py-clob-client order types are required for live CLOB execution"
            ) from exc
        order_type_name = self.config.default_order_type.upper()
        if order_type_name == "LIMIT":
            order_type_name = "GTC"
        order_type = getattr(OrderType, order_type_name, self.config.default_order_type)
        order_args = OrderArgs(
            price=float(price),
            size=float(size),
            side=side,
            token_id=str(token_id),
        )
        signed_order = self._client.create_order(order_args)
        return self._client.post_order(signed_order, order_type)


def _normalise_side(side: OrderSide | str) -> str:
    text = str(side).strip().upper()
    if text not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    return text


def _resolve_signature_type(signature_type_enum: Any, value: Any) -> Any:
    if value is None:
        return signature_type_enum.POLY_PROXY
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return signature_type_enum.POLY_PROXY
        if text.isdigit():
            return signature_type_enum(int(text))
        return getattr(signature_type_enum, text.upper())
    return signature_type_enum(value)


def _validate_order(*, token_id: str, price: float, size: float) -> None:
    _require_text("token_id", token_id)
    if not 0 < float(price) <= 1:
        raise ValueError("price must be in (0, 1]")
    if float(size) <= 0:
        raise ValueError("size must be positive")


def _require_text(field_name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} is required")


def _dry_run_order_id(token_id: str, side: str, price: float, size: float, sequence: int) -> str:
    digest = hashlib.sha1(f"{token_id}:{side}:{price:.6f}:{size:.6f}:{sequence}".encode()).hexdigest()
    return f"dryrun-{digest[:20]}"


def _extract_order_id(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("order_id", "id", "orderID", "orderId"):
            value = response.get(key)
            if value:
                return str(value)
    for attr in ("order_id", "id", "orderID", "orderId"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    raise ClobExecutionError(f"could not extract order id from response: {response!r}")


def _normalise_order_status(order_id: str, response: Any) -> OrderStatus:
    if isinstance(response, OrderStatus):
        return response
    raw = response if isinstance(response, dict) else _object_dict(response)
    return OrderStatus(
        order_id=str(raw.get("order_id") or raw.get("id") or order_id),
        status=str(raw.get("status") or raw.get("state") or "unknown"),
        filled_size=_optional_float(raw.get("filled_size") or raw.get("filledSize")) or 0.0,
        remaining_size=_optional_float(raw.get("remaining_size") or raw.get("remainingSize")),
        avg_fill_price=_optional_float(raw.get("avg_fill_price") or raw.get("averagePrice")),
        raw=dict(raw),
    )


def _normalise_bid_ask(response: Any) -> tuple[float | None, float | None]:
    if isinstance(response, tuple) and len(response) == 2:
        return _optional_float(response[0]), _optional_float(response[1])
    raw = response if isinstance(response, dict) else _object_dict(response)
    bid = _optional_float(raw.get("bid") or raw.get("best_bid") or raw.get("bestBid"))
    ask = _optional_float(raw.get("ask") or raw.get("best_ask") or raw.get("bestAsk"))
    if bid is not None or ask is not None:
        return bid, ask
    bids = raw.get("bids")
    asks = raw.get("asks")
    return _best_price(bids, want_max=True), _best_price(asks, want_max=False)


def _best_price(levels: Any, *, want_max: bool) -> float | None:
    prices: list[float] = []
    if not isinstance(levels, list):
        return None
    for level in levels:
        value = level.get("price") if isinstance(level, dict) else getattr(level, "price", None)
        parsed = _optional_float(value)
        if parsed is not None:
            prices.append(parsed)
    if not prices:
        return None
    return max(prices) if want_max else min(prices)


def _truthy_success(response: Any) -> bool:
    if isinstance(response, bool):
        return response
    if isinstance(response, dict):
        for key in ("success", "cancelled", "canceled"):
            if key in response:
                return bool(response[key])
    return True


def _object_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    return _is_rate_limit(exc) or any(
        token in str(exc).lower()
        for token in ("timeout", "temporarily", "connection", "503", "502", "504")
    )


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


def _is_insufficient_balance(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "insufficient balance",
            "insufficient collateral",
            "not enough balance",
            "not enough collateral",
        )
    )


def _now_ms() -> int:
    return int(time.time() * 1000)
