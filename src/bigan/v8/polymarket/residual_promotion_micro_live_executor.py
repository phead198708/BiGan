"""Capability-gated BTC-15M micro-live execution core.

There is intentionally no Polymarket client, network session, wallet, key, or
credential implementation here.  A transport capability must be injected, and
it is never called unless the separately verified future authorization passes.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_micro_live_authorization import (
    VerifiedMicroLiveAuthorization,
    authorization_capability_is_verified,
    verify_micro_live_authorization,
)
from bigan.v8.polymarket.residual_promotion_v1 import CANDIDATE_ID, LINEAGE_ID

STATE_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-micro-live-state-v1"
SIGNAL_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-execution-signal-v1"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
)
GENESIS = "GENESIS"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^btc-updown-15m-[1-9][0-9]*$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[1-9][0-9]*$")
_EVENT_TYPES = {
    "ORDER_PREPARED",
    "ORDER_ACKNOWLEDGED",
    "ORDER_REJECTED",
    "ORDER_SUBMISSION_UNKNOWN",
    "FILL_RECORDED",
    "ORDER_CANCELED",
    "ORDER_EXPIRED",
    "ORDER_CANCEL_UNKNOWN",
    "SETTLEMENT_RECORDED",
    "KILL_SWITCH_ENGAGED",
}


class MicroLiveExecutionError(RuntimeError):
    """Raised whenever execution or reconciliation cannot remain deterministic."""


class MicroLiveOrderTransport(Protocol):
    """Minimal injected write capability; no concrete implementation is bundled."""

    def submit_order(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Submit one idempotent order request."""

    def cancel_order(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Cancel one previously acknowledged order."""


def create_micro_live_executor(
    *,
    authorization: Mapping[str, Any],
    repository_root: Path | str,
    evidence_root: Path | str,
    now_ts_ms: int,
    transport: MicroLiveOrderTransport,
) -> MicroLiveExecutor:
    """Create an executor only after the complete future graph verifies."""

    verified = verify_micro_live_authorization(
        authorization,
        repository_root=repository_root,
        evidence_root=evidence_root,
        now_ts_ms=now_ts_ms,
    )
    return MicroLiveExecutor._from_verified_authorization(verified, transport=transport)


class MicroLiveExecutor:
    """Append-only execution state machine behind a verified authorization."""

    def __init__(
        self,
        authorization: VerifiedMicroLiveAuthorization,
        *,
        transport: MicroLiveOrderTransport,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not authorization_capability_is_verified(authorization):
            raise MicroLiveExecutionError("micro-live authorization capability is unverified")
        if transport is None:
            raise MicroLiveExecutionError("micro-live transport capability is missing")
        self.authorization = authorization
        self.transport = transport
        self._events = [copy.deepcopy(dict(event)) for event in events]
        self._verify_event_chain()
        self._reconcile_view()

    @classmethod
    def _from_verified_authorization(
        cls,
        authorization: VerifiedMicroLiveAuthorization,
        *,
        transport: MicroLiveOrderTransport,
    ) -> MicroLiveExecutor:
        return cls(authorization, transport=transport)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

    def submit_signal(
        self,
        *,
        signal_payload: Mapping[str, Any],
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> dict[str, Any]:
        """Submit one authorized intent or return an explicit NO_TRADE block."""

        signal = _validated_candidate_signal(
            signal_payload,
            expected_candidate_bundle_sha256=self.authorization.candidate_bundle_sha256,
        )
        decision_ts_ms = int(signal["decision_ts_ms"])
        self._validate_clock(
            now_ts_ms,
            operator_heartbeat_ts_ms,
            decision_ts_ms,
            int(signal["observed_at_ts_ms"]),
        )
        view = self._reconcile_view()
        if view["kill_switch_active"]:
            return _blocked("kill_switch_active")
        market_id = str(signal["market_id"])
        slug = str(signal["slug"])
        market_family = str(signal["market_family"])
        candidate_bundle_sha256 = str(signal["candidate_bundle_sha256"])
        selected_action = str(signal["selected_action"])
        if market_family not in self.authorization.market_allowlist:
            return _blocked("market_not_allowlisted")
        if signal["fail_closed"] is True or signal["model_scored"] is not True:
            return _blocked("signal_failed_closed")
        if selected_action == "NO_TRADE":
            return _blocked("signal_selected_no_trade")
        if selected_action not in self.authorization.allowed_actions:
            return _blocked("action_not_allowlisted")
        token_side = "UP" if selected_action == "BUY_UP_HOLD" else "DOWN"
        token_id = str(signal[f"{token_side.lower()}_token_id"])
        price = _positive_decimal(
            dict(signal["executable_asks"])[token_side],
            "limit price",
        )
        qty = Decimal("1")
        notional = price * qty
        signal_payload_sha256 = canonical_json_sha256(signal)
        business_key = canonical_json_sha256(
            {
                "authorization_id": self.authorization.authorization_id,
                "candidate_bundle_sha256": candidate_bundle_sha256,
                "market_id": market_id,
            }
        )
        intent_core = {
            "authorization_id": self.authorization.authorization_id,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "business_key": business_key,
            "market_id": market_id,
            "slug": slug,
            "market_family": market_family,
            "decision_ts_ms": decision_ts_ms,
            "selected_action": selected_action,
            "token_id": token_id,
            "token_side": token_side,
            "limit_price": str(price),
            "quantity": str(qty),
            "notional_usd": str(notional),
            "signal_payload_sha256": signal_payload_sha256,
            "signal_payload": signal,
        }
        intent_id = canonical_json_sha256(intent_core)
        client_order_id = intent_id
        existing = view["orders_by_business_key"].get(business_key)
        if existing is not None:
            if existing["intent_id"] != intent_id:
                self.engage_kill_switch(
                    reason="conflicting_duplicate_intent",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate intent failed closed")
            if existing.get("acknowledgement") is not None:
                return {
                    "status": "IDEMPOTENT_REPLAY",
                    "client_order_id": client_order_id,
                    "exchange_order_id": existing["acknowledgement"]["exchange_order_id"],
                    "transport_called": False,
                }
            return _blocked("existing_order_requires_reconciliation", client_order_id)
        if view["submitted_notional_usd"] + notional > self.authorization.maximum_notional_usd:
            return _blocked("authorization_notional_cap_exceeded")
        if view["open_order_count"] >= self.authorization.maximum_open_orders:
            return _blocked("maximum_open_orders_reached")

        request = {
            **intent_core,
            "intent_id": intent_id,
            "client_order_id": client_order_id,
            "authorization_payload_sha256": self.authorization.authorization_payload_sha256,
        }
        self._append_event("ORDER_PREPARED", request)
        try:
            response = dict(self.transport.submit_order(copy.deepcopy(request)))
            disposition = self._validate_submission_response(request, response)
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_UNKNOWN",
                {
                    "client_order_id": client_order_id,
                    "error_type": exc.__class__.__name__,
                },
            )
            self.engage_kill_switch(
                reason="order_submission_unknown",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("order submission became unknown; kill switch engaged") from exc
        if disposition["status"] == "REJECTED":
            self._append_event("ORDER_REJECTED", disposition)
            return {
                "status": "ORDER_REJECTED",
                "client_order_id": client_order_id,
                "transport_called": True,
            }
        self._append_event("ORDER_ACKNOWLEDGED", disposition)
        self._reconcile_view()
        return {
            "status": "ORDER_ACKNOWLEDGED",
            "client_order_id": client_order_id,
            "exchange_order_id": disposition["exchange_order_id"],
            "transport_called": True,
        }

    def record_fill(
        self,
        *,
        client_order_id: str,
        fill_id: str,
        now_ts_ms: int,
        quantity: str,
        price: str,
        fee_usd: str,
        transport_event_sha256: str,
    ) -> dict[str, Any]:
        """Record one externally observed fill and reconcile cash/position."""

        _require_positive_timestamp(now_ts_ms, "fill observation")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("fill has no acknowledged order")
        if order.get("closed_status") is not None or order.get("settlement") is not None:
            self.engage_kill_switch(reason="fill_after_terminal_state", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill arrived after order close or settlement")
        if not isinstance(fill_id, str) or not fill_id:
            raise MicroLiveExecutionError("fill identity is invalid")
        existing_fill = view["fills"].get(fill_id)
        payload = {
            "client_order_id": client_order_id,
            "fill_id": fill_id,
            "quantity": str(_positive_decimal(quantity, "fill quantity")),
            "price": str(_positive_decimal(price, "fill price")),
            "fee_usd": str(_nonnegative_decimal(fee_usd, "fill fee")),
            "transport_event_sha256": _require_sha256(
                transport_event_sha256, "fill transport event"
            ),
        }
        if existing_fill is not None:
            if existing_fill != payload:
                self.engage_kill_switch(
                    reason="conflicting_duplicate_fill",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate fill failed closed")
            return {"status": "IDEMPOTENT_FILL_REPLAY", "fill_id": fill_id}
        fill_qty = Decimal(payload["quantity"])
        fill_price = Decimal(payload["price"])
        fee = Decimal(payload["fee_usd"])
        if fill_price > Decimal(order["prepared"]["limit_price"]):
            self.engage_kill_switch(reason="fill_price_above_limit", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("buy fill price exceeds authorized limit")
        if fill_qty > order["remaining_quantity"]:
            self.engage_kill_switch(reason="fill_quantity_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill quantity exceeds remaining order")
        if fill_qty * fill_price + fee > view["cash_usd"]:
            self.engage_kill_switch(reason="fill_cash_cap_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill would make authorized cash negative")
        self._append_event("FILL_RECORDED", payload)
        snapshot = self.reconciliation_snapshot()
        return {"status": "FILL_RECORDED", "fill_id": fill_id, "snapshot": snapshot}

    def record_order_closed(
        self,
        *,
        client_order_id: str,
        status: str,
        now_ts_ms: int,
        transport_event_sha256: str,
    ) -> dict[str, Any]:
        """Record an externally observed CANCELED or EXPIRED terminal state."""

        _require_positive_timestamp(now_ts_ms, "order close observation")
        if status not in {"CANCELED", "EXPIRED"}:
            raise MicroLiveExecutionError("order close status is invalid")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("order close has no acknowledged order")
        if order.get("settlement") is not None:
            self.engage_kill_switch(reason="order_close_after_settlement", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("order close arrived after settlement")
        existing = order.get("closed_status")
        if existing is not None:
            if existing != status:
                self.engage_kill_switch(
                    reason="conflicting_order_close",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting order close status")
            return {"status": "IDEMPOTENT_ORDER_CLOSE", "client_order_id": client_order_id}
        event_type = "ORDER_CANCELED" if status == "CANCELED" else "ORDER_EXPIRED"
        self._append_event(
            event_type,
            {
                "client_order_id": client_order_id,
                "transport_event_sha256": _require_sha256(
                    transport_event_sha256, "order close transport event"
                ),
            },
        )
        return {"status": event_type, "client_order_id": client_order_id}

    def record_settlement(
        self,
        *,
        client_order_id: str,
        settlement_id: str,
        now_ts_ms: int,
        payout_per_token: str,
        official_settlement_sha256: str,
    ) -> dict[str, Any]:
        """Record one official finalized settlement after external ingestion."""

        _require_positive_timestamp(now_ts_ms, "settlement observation")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("settlement has no acknowledged order")
        payout = _nonnegative_decimal(payout_per_token, "settlement payout")
        if payout not in {Decimal("0"), Decimal("1")}:
            raise MicroLiveExecutionError("settlement payout must be official binary")
        if not isinstance(settlement_id, str) or not settlement_id:
            raise MicroLiveExecutionError("settlement identity is invalid")
        payload = {
            "client_order_id": client_order_id,
            "settlement_id": settlement_id,
            "payout_per_token": str(payout),
            "official_settlement_sha256": _require_sha256(
                official_settlement_sha256, "official settlement"
            ),
        }
        existing = order.get("settlement")
        settlement_elsewhere = view["settlements"].get(settlement_id)
        if settlement_elsewhere is not None and settlement_elsewhere != payload:
            self.engage_kill_switch(
                reason="settlement_identity_reused",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("settlement identity is reused across orders")
        if existing is not None:
            if existing != payload:
                self.engage_kill_switch(
                    reason="conflicting_duplicate_settlement",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate settlement")
            return {"status": "IDEMPOTENT_SETTLEMENT_REPLAY", "settlement_id": settlement_id}
        if order["filled_quantity"] <= 0:
            raise MicroLiveExecutionError("unfilled order cannot settle")
        if order["is_open"]:
            self.engage_kill_switch(
                reason="settlement_while_order_open",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("open order cannot settle before cancellation")
        self._append_event("SETTLEMENT_RECORDED", payload)
        return {
            "status": "SETTLEMENT_RECORDED",
            "settlement_id": settlement_id,
            "snapshot": self.reconciliation_snapshot(),
        }

    def engage_kill_switch(self, *, reason: str, now_ts_ms: int) -> dict[str, Any]:
        """Irreversibly stop submissions and best-effort cancel open orders."""

        if not isinstance(reason, str) or not reason:
            raise MicroLiveExecutionError("kill-switch reason is invalid")
        if isinstance(now_ts_ms, bool) or not isinstance(now_ts_ms, int) or now_ts_ms <= 0:
            raise MicroLiveExecutionError("kill-switch timestamp is invalid")
        view = self._reconcile_view()
        if not view["kill_switch_active"]:
            self._append_event(
                "KILL_SWITCH_ENGAGED",
                {"reason": reason, "engaged_at_ts_ms": now_ts_ms},
            )
            view = self._reconcile_view()
        canceled: list[str] = []
        unknown: list[str] = []
        for order in view["orders"].values():
            if not order["is_open"]:
                continue
            request = {
                "authorization_id": self.authorization.authorization_id,
                "client_order_id": order["prepared"]["client_order_id"],
                "exchange_order_id": order["acknowledgement"]["exchange_order_id"],
                "reason": reason,
            }
            try:
                response = dict(self.transport.cancel_order(copy.deepcopy(request)))
                if set(response) != {"client_order_id", "exchange_order_id", "status"} or not (
                    response.get("client_order_id") == request["client_order_id"]
                    and response.get("exchange_order_id") == request["exchange_order_id"]
                    and response.get("status") == "CANCELED"
                ):
                    raise MicroLiveExecutionError("cancel response contract mismatch")
                self._append_event(
                    "ORDER_CANCELED",
                    {
                        "client_order_id": request["client_order_id"],
                        "transport_event_sha256": canonical_json_sha256(response),
                    },
                )
                canceled.append(str(request["client_order_id"]))
            except Exception as exc:
                self._append_event(
                    "ORDER_CANCEL_UNKNOWN",
                    {
                        "client_order_id": request["client_order_id"],
                        "error_type": exc.__class__.__name__,
                    },
                )
                unknown.append(str(request["client_order_id"]))
        return {
            "status": "KILL_SWITCH_ENGAGED",
            "reason": reason,
            "canceled_client_order_ids": canceled,
            "unknown_cancel_client_order_ids": unknown,
        }

    def reconciliation_snapshot(self) -> dict[str, Any]:
        """Return a deterministic cash/order/position reconciliation snapshot."""

        view = self._reconcile_view()
        return {
            "authorization_id": self.authorization.authorization_id,
            "event_count": len(self._events),
            "kill_switch_active": view["kill_switch_active"],
            "kill_switch_reason": view["kill_switch_reason"],
            "cash_usd": str(view["cash_usd"]),
            "positions": {key: str(value) for key, value in sorted(view["positions"].items())},
            "submitted_notional_usd": str(view["submitted_notional_usd"]),
            "open_order_count": view["open_order_count"],
            "order_count": len(view["orders"]),
            "fill_count": len(view["fills"]),
            "settlement_count": sum(
                order.get("settlement") is not None for order in view["orders"].values()
            ),
            "reconciled": True,
            "state_sha256": canonical_json_sha256(self.export_state(include_state_sha=False)),
        }

    def export_state(self, *, include_state_sha: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "authorization_id": self.authorization.authorization_id,
            "authorization_payload_sha256": self.authorization.authorization_payload_sha256,
            "candidate_bundle_sha256": self.authorization.candidate_bundle_sha256,
            "events": copy.deepcopy(self._events),
        }
        return (
            {**payload, "state_sha256": canonical_json_sha256(payload)}
            if include_state_sha
            else payload
        )

    @classmethod
    def restore(
        cls,
        *,
        authorization: VerifiedMicroLiveAuthorization,
        transport: MicroLiveOrderTransport,
        state: Mapping[str, Any],
    ) -> MicroLiveExecutor:
        if set(state) != {
            "schema_version",
            "authorization_id",
            "authorization_payload_sha256",
            "candidate_bundle_sha256",
            "events",
            "state_sha256",
        }:
            raise MicroLiveExecutionError("micro-live state schema is invalid")
        payload = {key: copy.deepcopy(value) for key, value in state.items() if key != "state_sha256"}
        events = payload.get("events")
        if not (
            payload.get("schema_version") == STATE_SCHEMA_VERSION
            and payload.get("authorization_id") == authorization.authorization_id
            and payload.get("authorization_payload_sha256")
            == authorization.authorization_payload_sha256
            and payload.get("candidate_bundle_sha256") == authorization.candidate_bundle_sha256
            and isinstance(events, list)
            and state.get("state_sha256") == canonical_json_sha256(payload)
        ):
            raise MicroLiveExecutionError("micro-live state identity or SHA-256 mismatch")
        return cls(authorization, transport=transport, events=events)

    def _validate_clock(
        self,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
        decision_ts_ms: int,
        signal_observed_at_ts_ms: int,
    ) -> None:
        values = (
            now_ts_ms,
            operator_heartbeat_ts_ms,
            decision_ts_ms,
            signal_observed_at_ts_ms,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise MicroLiveExecutionError("execution timestamp is invalid")
        if now_ts_ms >= self.authorization.expires_at_ts_ms:
            self.engage_kill_switch(reason="authorization_expired", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live authorization expired")
        if not decision_ts_ms <= now_ts_ms or (
            now_ts_ms - decision_ts_ms > self.authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal is stale")
        if not decision_ts_ms <= signal_observed_at_ts_ms <= now_ts_ms or (
            now_ts_ms - signal_observed_at_ts_ms
            > self.authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_observation_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal observation is stale")
        if not operator_heartbeat_ts_ms <= now_ts_ms or (
            now_ts_ms - operator_heartbeat_ts_ms
            > self.authorization.maximum_operator_heartbeat_age_ms
        ):
            self.engage_kill_switch(reason="operator_heartbeat_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live operator heartbeat is stale")

    def _validate_submission_response(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_keys = {
            "client_order_id",
            "exchange_order_id",
            "status",
            "market_id",
            "token_id",
            "accepted_quantity",
            "limit_price",
        }
        if set(response) != expected_keys:
            raise MicroLiveExecutionError("order response schema mismatch")
        if not (
            response.get("client_order_id") == request["client_order_id"]
            and response.get("market_id") == request["market_id"]
            and response.get("token_id") == request["token_id"]
            and response.get("accepted_quantity") == request["quantity"]
            and response.get("limit_price") == request["limit_price"]
            and response.get("status") in {"ACCEPTED", "REJECTED"}
            and isinstance(response.get("exchange_order_id"), str)
            and response["exchange_order_id"]
        ):
            raise MicroLiveExecutionError("order response identity mismatch")
        return dict(response)

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if event_type not in _EVENT_TYPES:
            raise MicroLiveExecutionError("micro-live event type is invalid")
        core = {
            "sequence": len(self._events) + 1,
            "previous_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else GENESIS
            ),
            "event_type": event_type,
            "payload": copy.deepcopy(dict(payload)),
        }
        self._events.append({**core, "event_sha256": canonical_json_sha256(core)})

    def _verify_event_chain(self) -> None:
        previous = GENESIS
        for expected_sequence, event in enumerate(self._events, start=1):
            if set(event) != {
                "sequence",
                "previous_event_sha256",
                "event_type",
                "payload",
                "event_sha256",
            }:
                raise MicroLiveExecutionError("micro-live event schema is invalid")
            core = {key: value for key, value in event.items() if key != "event_sha256"}
            if not (
                event["sequence"] == expected_sequence
                and event["previous_event_sha256"] == previous
                and event["event_type"] in _EVENT_TYPES
                and isinstance(event["payload"], Mapping)
                and event["event_sha256"] == canonical_json_sha256(core)
            ):
                raise MicroLiveExecutionError("micro-live event chain is invalid")
            self._validate_event_payload(str(event["event_type"]), event["payload"])
            previous = str(event["event_sha256"])

    def _validate_event_payload(
        self,
        event_type: str,
        payload_value: Mapping[str, Any],
    ) -> None:
        payload = dict(payload_value)
        if event_type == "ORDER_PREPARED":
            expected = {
                "authorization_id",
                "authorization_payload_sha256",
                "candidate_bundle_sha256",
                "business_key",
                "market_id",
                "slug",
                "market_family",
                "decision_ts_ms",
                "selected_action",
                "token_id",
                "token_side",
                "limit_price",
                "quantity",
                "notional_usd",
                "signal_payload_sha256",
                "signal_payload",
                "intent_id",
                "client_order_id",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("prepared order payload schema is invalid")
            price = _positive_decimal(payload.get("limit_price"), "stored limit price")
            quantity = _positive_decimal(payload.get("quantity"), "stored quantity")
            if (
                price >= 1
                or quantity != Decimal("1")
                or payload.get("notional_usd") != str(price * quantity)
            ):
                raise MicroLiveExecutionError("prepared order economics are invalid")
            signal = _validated_candidate_signal(
                payload.get("signal_payload"),
                expected_candidate_bundle_sha256=self.authorization.candidate_bundle_sha256,
            )
            market_id = payload.get("market_id")
            business_key = canonical_json_sha256(
                {
                    "authorization_id": self.authorization.authorization_id,
                    "candidate_bundle_sha256": self.authorization.candidate_bundle_sha256,
                    "market_id": market_id,
                }
            )
            intent_core = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "authorization_payload_sha256",
                    "intent_id",
                    "client_order_id",
                }
            }
            intent_id = canonical_json_sha256(intent_core)
            if not (
                payload.get("authorization_id") == self.authorization.authorization_id
                and payload.get("authorization_payload_sha256")
                == self.authorization.authorization_payload_sha256
                and payload.get("candidate_bundle_sha256")
                == self.authorization.candidate_bundle_sha256
                and isinstance(market_id, str)
                and market_id
                and isinstance(payload.get("slug"), str)
                and _SLUG.fullmatch(payload["slug"]) is not None
                and payload.get("market_family") in self.authorization.market_allowlist
                and isinstance(payload.get("decision_ts_ms"), int)
                and not isinstance(payload.get("decision_ts_ms"), bool)
                and payload["decision_ts_ms"] > 0
                and payload.get("selected_action") in self.authorization.allowed_actions
                and payload.get("selected_action") == signal["selected_action"]
                and payload.get("market_id") == signal["market_id"]
                and payload.get("slug") == signal["slug"]
                and payload.get("market_family") == signal["market_family"]
                and payload.get("decision_ts_ms") == signal["decision_ts_ms"]
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and payload.get("token_side")
                == ("UP" if payload.get("selected_action") == "BUY_UP_HOLD" else "DOWN")
                and payload.get("token_id")
                == signal[f"{str(payload.get('token_side')).lower()}_token_id"]
                and payload.get("limit_price")
                == dict(signal["executable_asks"])[payload["token_side"]]
                and payload.get("signal_payload_sha256") == canonical_json_sha256(signal)
                and payload.get("business_key") == business_key
                and payload.get("intent_id") == intent_id
                and payload.get("client_order_id") == intent_id
            ):
                raise MicroLiveExecutionError("prepared order identity is invalid")
            return
        if event_type in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}:
            if set(payload) != {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
            } or payload.get("status") != (
                "ACCEPTED" if event_type == "ORDER_ACKNOWLEDGED" else "REJECTED"
            ):
                raise MicroLiveExecutionError("order disposition payload is invalid")
            return
        if event_type in {"ORDER_SUBMISSION_UNKNOWN", "ORDER_CANCEL_UNKNOWN"}:
            if set(payload) != {"client_order_id", "error_type"} or not (
                isinstance(payload.get("client_order_id"), str)
                and payload["client_order_id"]
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError("unknown transport event payload is invalid")
            return
        if event_type == "FILL_RECORDED":
            if set(payload) != {
                "client_order_id",
                "fill_id",
                "quantity",
                "price",
                "fee_usd",
                "transport_event_sha256",
            }:
                raise MicroLiveExecutionError("fill payload schema is invalid")
            _positive_decimal(payload.get("quantity"), "stored fill quantity")
            fill_price = _positive_decimal(payload.get("price"), "stored fill price")
            _nonnegative_decimal(payload.get("fee_usd"), "stored fill fee")
            if fill_price >= 1 or not _is_sha256(payload.get("transport_event_sha256")):
                raise MicroLiveExecutionError("fill payload values are invalid")
            return
        if event_type in {"ORDER_CANCELED", "ORDER_EXPIRED"}:
            if set(payload) != {"client_order_id", "transport_event_sha256"} or not (
                isinstance(payload.get("client_order_id"), str)
                and payload["client_order_id"]
                and _is_sha256(payload.get("transport_event_sha256"))
            ):
                raise MicroLiveExecutionError("order close payload is invalid")
            return
        if event_type == "SETTLEMENT_RECORDED":
            payout = _nonnegative_decimal(
                payload.get("payout_per_token"), "stored settlement payout"
            )
            if set(payload) != {
                "client_order_id",
                "settlement_id",
                "payout_per_token",
                "official_settlement_sha256",
            } or not (
                isinstance(payload.get("settlement_id"), str)
                and payload["settlement_id"]
                and payout in {Decimal("0"), Decimal("1")}
                and _is_sha256(payload.get("official_settlement_sha256"))
            ):
                raise MicroLiveExecutionError("settlement payload is invalid")
            return
        if event_type == "KILL_SWITCH_ENGAGED":
            if set(payload) != {"reason", "engaged_at_ts_ms"} or not (
                isinstance(payload.get("reason"), str)
                and payload["reason"]
                and isinstance(payload.get("engaged_at_ts_ms"), int)
                and not isinstance(payload.get("engaged_at_ts_ms"), bool)
                and payload["engaged_at_ts_ms"] > 0
            ):
                raise MicroLiveExecutionError("kill-switch payload is invalid")
            return
        raise MicroLiveExecutionError("unhandled micro-live event payload")

    def _reconcile_view(self) -> dict[str, Any]:
        self._verify_event_chain()
        orders: dict[str, dict[str, Any]] = {}
        business_keys: dict[str, dict[str, Any]] = {}
        fills: dict[str, dict[str, Any]] = {}
        settlements: dict[str, dict[str, Any]] = {}
        kill_switch_active = False
        kill_switch_reason: str | None = None
        for event in self._events:
            event_type = event["event_type"]
            payload = dict(event["payload"])
            client_order_id = payload.get("client_order_id")
            if event_type == "ORDER_PREPARED":
                if (
                    not isinstance(client_order_id, str)
                    or client_order_id in orders
                    or payload.get("intent_id") != client_order_id
                    or payload.get("authorization_id") != self.authorization.authorization_id
                    or payload.get("candidate_bundle_sha256")
                    != self.authorization.candidate_bundle_sha256
                ):
                    raise MicroLiveExecutionError("prepared order identity is invalid")
                business_key = payload.get("business_key")
                if not isinstance(business_key, str) or business_key in business_keys:
                    raise MicroLiveExecutionError("prepared order business identity is duplicated")
                order = {
                    "intent_id": client_order_id,
                    "prepared": payload,
                    "acknowledgement": None,
                    "submission_unknown": False,
                    "closed_status": None,
                    "fills": [],
                    "settlement": None,
                }
                orders[client_order_id] = order
                business_keys[business_key] = order
            elif event_type in {
                "ORDER_ACKNOWLEDGED",
                "ORDER_REJECTED",
                "ORDER_SUBMISSION_UNKNOWN",
                "ORDER_CANCELED",
                "ORDER_EXPIRED",
                "ORDER_CANCEL_UNKNOWN",
            }:
                order = orders.get(str(client_order_id))
                if order is None:
                    raise MicroLiveExecutionError("order lifecycle event lacks preparation")
                if event_type == "ORDER_ACKNOWLEDGED":
                    prepared = order["prepared"]
                    if (
                        order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or order["submission_unknown"]
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                    ):
                        raise MicroLiveExecutionError("order acknowledgement is duplicated")
                    order["acknowledgement"] = payload
                elif event_type == "ORDER_REJECTED":
                    if (
                        order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or order["submission_unknown"]
                    ):
                        raise MicroLiveExecutionError("order rejection lifecycle is invalid")
                    order["closed_status"] = "REJECTED"
                elif event_type == "ORDER_SUBMISSION_UNKNOWN":
                    if order["acknowledgement"] is not None or order["closed_status"] is not None:
                        raise MicroLiveExecutionError("unknown submission lifecycle is invalid")
                    order["submission_unknown"] = True
                elif event_type == "ORDER_CANCELED":
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                    ):
                        raise MicroLiveExecutionError("order cancellation lifecycle is invalid")
                    order["closed_status"] = "CANCELED"
                elif event_type == "ORDER_EXPIRED":
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                    ):
                        raise MicroLiveExecutionError("order expiration lifecycle is invalid")
                    order["closed_status"] = "EXPIRED"
                else:
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                    ):
                        raise MicroLiveExecutionError("unknown cancellation lifecycle is invalid")
                    order["cancel_unknown"] = True
            elif event_type == "FILL_RECORDED":
                order = orders.get(str(client_order_id))
                fill_id = payload.get("fill_id")
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["closed_status"] is not None
                    or order["settlement"] is not None
                    or not isinstance(fill_id, str)
                    or fill_id in fills
                ):
                    raise MicroLiveExecutionError("fill event identity is invalid")
                fills[fill_id] = payload
                order["fills"].append(payload)
            elif event_type == "SETTLEMENT_RECORDED":
                order = orders.get(str(client_order_id))
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["settlement"] is not None
                    or not order["fills"]
                ):
                    raise MicroLiveExecutionError("settlement event identity is invalid")
                settlement_id = payload.get("settlement_id")
                if not isinstance(settlement_id, str) or settlement_id in settlements:
                    raise MicroLiveExecutionError("settlement identity is duplicated")
                settlements[settlement_id] = payload
                order["settlement"] = payload
            elif event_type == "KILL_SWITCH_ENGAGED":
                if kill_switch_active:
                    raise MicroLiveExecutionError("kill switch event is duplicated")
                kill_switch_active = True
                kill_switch_reason = str(payload.get("reason") or "")

        cash = self.authorization.maximum_notional_usd
        positions = {"UP": Decimal("0"), "DOWN": Decimal("0")}
        submitted_notional = Decimal("0")
        open_order_count = 0
        for order in orders.values():
            prepared = order["prepared"]
            requested_qty = Decimal(prepared["quantity"])
            submitted_notional += Decimal(prepared["notional_usd"])
            filled_qty = Decimal("0")
            for fill in order["fills"]:
                qty = Decimal(fill["quantity"])
                price = Decimal(fill["price"])
                fee = Decimal(fill["fee_usd"])
                filled_qty += qty
                cash -= qty * price + fee
                side = "UP" if prepared["selected_action"] == "BUY_UP_HOLD" else "DOWN"
                positions[side] += qty
            if filled_qty > requested_qty or cash < 0:
                raise MicroLiveExecutionError("fill ledger exceeds quantity or cash")
            order["filled_quantity"] = filled_qty
            order["remaining_quantity"] = requested_qty - filled_qty
            order["is_open"] = bool(
                order["acknowledgement"] is not None
                and order["closed_status"] is None
                and order["settlement"] is None
                and order["remaining_quantity"] > 0
            )
            if order["is_open"]:
                open_order_count += 1
            settlement = order.get("settlement")
            if settlement is not None:
                if filled_qty < requested_qty and order["closed_status"] is None:
                    raise MicroLiveExecutionError(
                        "partially filled order settled before terminal close"
                    )
                payout = Decimal(settlement["payout_per_token"])
                cash += filled_qty * payout
                side = "UP" if prepared["selected_action"] == "BUY_UP_HOLD" else "DOWN"
                positions[side] -= filled_qty
        if submitted_notional > self.authorization.maximum_notional_usd:
            raise MicroLiveExecutionError("submitted notional exceeds authorization")
        if open_order_count > self.authorization.maximum_open_orders:
            raise MicroLiveExecutionError("open order count exceeds authorization")
        return {
            "orders": orders,
            "orders_by_business_key": business_keys,
            "fills": fills,
            "settlements": settlements,
            "cash_usd": cash,
            "positions": positions,
            "submitted_notional_usd": submitted_notional,
            "open_order_count": open_order_count,
            "kill_switch_active": kill_switch_active,
            "kill_switch_reason": kill_switch_reason,
        }


def _blocked(reason: str, client_order_id: str | None = None) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NO_TRADE",
        "reason": reason,
        "client_order_id": client_order_id,
        "transport_called": False,
    }


def _validated_candidate_signal(
    value: Any,
    *,
    expected_candidate_bundle_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError("candidate signal payload is not an object")
    signal = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "lineage_id",
        "candidate_id",
        "candidate_bundle_sha256",
        "market_id",
        "slug",
        "market_family",
        "decision_ts_ms",
        "observed_at_ts_ms",
        "action_values",
        "executable_asks",
        "up_token_id",
        "down_token_id",
        "selected_action",
        "model_scored",
        "fail_closed",
        "fail_closed_reasons",
        "decision_influenced_collection",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
        "safety",
    }
    if set(signal) != expected_keys:
        raise MicroLiveExecutionError("candidate signal schema is not exact")
    if not (
        signal.get("schema_version") == SIGNAL_SCHEMA_VERSION
        and signal.get("lineage_id") == LINEAGE_ID
        and signal.get("candidate_id") == CANDIDATE_ID
        and signal.get("candidate_bundle_sha256")
        == expected_candidate_bundle_sha256
        and isinstance(signal.get("market_id"), str)
        and _CONDITION_ID.fullmatch(signal["market_id"]) is not None
        and isinstance(signal.get("slug"), str)
        and _SLUG.fullmatch(signal["slug"]) is not None
        and signal.get("market_family") == "BTC-15M"
        and isinstance(signal.get("up_token_id"), str)
        and _TOKEN_ID.fullmatch(signal["up_token_id"]) is not None
        and isinstance(signal.get("down_token_id"), str)
        and _TOKEN_ID.fullmatch(signal["down_token_id"]) is not None
        and signal["up_token_id"] != signal["down_token_id"]
        and isinstance(signal.get("decision_ts_ms"), int)
        and not isinstance(signal.get("decision_ts_ms"), bool)
        and signal["decision_ts_ms"] > 0
        and isinstance(signal.get("observed_at_ts_ms"), int)
        and not isinstance(signal.get("observed_at_ts_ms"), bool)
        and signal["observed_at_ts_ms"] >= signal["decision_ts_ms"]
        and signal.get("decision_influenced_collection") is False
        and signal.get("outcomes_accessed") is False
        and signal.get("settlement_accessed") is False
        and signal.get("pnl_accessed") is False
        and dict(signal.get("safety") or {}) == SAFETY
    ):
        raise MicroLiveExecutionError("candidate signal identity or safety is invalid")
    market_start_ts_ms = int(str(signal["slug"]).rsplit("-", maxsplit=1)[1]) * 1_000
    if signal["decision_ts_ms"] not in {
        market_start_ts_ms + 300_000,
        market_start_ts_ms + 600_000,
    }:
        raise MicroLiveExecutionError("candidate signal decision is off the frozen schedule")
    action_values = signal.get("action_values")
    executable_asks = signal.get("executable_asks")
    fail_reasons = signal.get("fail_closed_reasons")
    if not isinstance(action_values, Mapping) or set(action_values) != {
        "NO_TRADE",
        "BUY_UP_HOLD",
        "BUY_DOWN_HOLD",
    }:
        raise MicroLiveExecutionError("candidate signal action values are invalid")
    if not isinstance(executable_asks, Mapping) or set(executable_asks) != {"UP", "DOWN"}:
        raise MicroLiveExecutionError("candidate signal executable asks are invalid")
    asks = {
        side: _positive_decimal(executable_asks[side], f"{side} executable ask")
        for side in ("UP", "DOWN")
    }
    if any(value >= 1 for value in asks.values()):
        raise MicroLiveExecutionError("candidate signal executable ask is outside binary range")
    if not isinstance(fail_reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in fail_reasons
    ):
        raise MicroLiveExecutionError("candidate signal fail-closed reasons are invalid")
    model_scored = signal.get("model_scored")
    failed_closed = signal.get("fail_closed")
    if model_scored is True and failed_closed is False:
        values = dict(action_values)
        if (
            isinstance(values.get("NO_TRADE"), bool)
            or values.get("NO_TRADE") != 0.0
            or any(
            isinstance(values.get(action), bool)
            or not isinstance(values.get(action), (int, float))
            or not math.isfinite(float(values[action]))
            for action in ("BUY_UP_HOLD", "BUY_DOWN_HOLD")
            )
        ):
            raise MicroLiveExecutionError("candidate signal scored values are invalid")
        up = float(values["BUY_UP_HOLD"])
        down = float(values["BUY_DOWN_HOLD"])
        expected_action = (
            ("BUY_UP_HOLD" if up >= down else "BUY_DOWN_HOLD")
            if max(up, down) > 0.0
            else "NO_TRADE"
        )
        if signal.get("selected_action") != expected_action or fail_reasons:
            raise MicroLiveExecutionError("candidate signal zero-threshold decision mismatches")
    elif model_scored is False and failed_closed is True:
        if not (
            dict(action_values)
            == {"NO_TRADE": 0.0, "BUY_UP_HOLD": None, "BUY_DOWN_HOLD": None}
            and signal.get("selected_action") == "NO_TRADE"
            and fail_reasons
        ):
            raise MicroLiveExecutionError("candidate signal fail-closed state is invalid")
    else:
        raise MicroLiveExecutionError("candidate signal scoring state is invalid")
    return signal


def _positive_decimal(value: Any, label: str) -> Decimal:
    parsed = _decimal(value, label)
    if parsed <= 0:
        raise MicroLiveExecutionError(f"{label} must be positive")
    return parsed


def _nonnegative_decimal(value: Any, label: str) -> Decimal:
    parsed = _decimal(value, label)
    if parsed < 0:
        raise MicroLiveExecutionError(f"{label} must be nonnegative")
    return parsed


def _decimal(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise MicroLiveExecutionError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise MicroLiveExecutionError(f"{label} is invalid") from exc
    if not parsed.is_finite() or str(parsed) != value:
        raise MicroLiveExecutionError(f"{label} is not canonical")
    return parsed


def _require_positive_timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MicroLiveExecutionError(f"{label} timestamp is invalid")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise MicroLiveExecutionError(f"{label} SHA-256 is invalid")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "IMPLEMENTATION_REPOSITORY_PATH",
    "MicroLiveExecutionError",
    "MicroLiveExecutor",
    "MicroLiveOrderTransport",
    "SIGNAL_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "create_micro_live_executor",
]
