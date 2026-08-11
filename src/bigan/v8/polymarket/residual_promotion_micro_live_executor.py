"""Capability-gated BTC-15M micro-live execution core.

There is intentionally no Polymarket client, network session, wallet, key, or
credential implementation here.  A transport capability must be injected, and
it is never called unless the separately verified future authorization passes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    ResidualPromotionRuntime,
)

STATE_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-micro-live-state-v3"
SIGNAL_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-execution-signal-v2"
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_executor.py"
)
GENESIS = "GENESIS"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^btc-updown-15m-[1-9][0-9]*$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[1-9][0-9]*$")
_FORBIDDEN_FEATURE_KEY_TOKENS = ("outcome", "settlement", "resolution", "pnl")
FROZEN_EXECUTION_FEE_PER_UNIT_USD = Decimal("0.0002")
_EVENT_TYPES = {
    "SIGNAL_REJECTED",
    "SIGNAL_EVALUATED",
    "ORDER_PREPARED",
    "ORDER_ACKNOWLEDGED",
    "ORDER_REJECTED",
    "ORDER_SUBMISSION_UNKNOWN",
    "ORDER_SUBMISSION_RECONCILED",
    "ORDER_SUBMISSION_RECONCILIATION_FAILED",
    "ORDER_CANCEL_RECONCILED",
    "ORDER_CANCEL_RECONCILIATION_FAILED",
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

    def lookup_order(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Read one order by its exact client/business identity."""


def _runtime_integrity_sha256(runtime: ResidualPromotionRuntime) -> str:
    residual_model_sha256 = hashlib.sha256(
        bytes(runtime.residual_booster.save_raw(raw_format="ubj"))
    ).hexdigest()
    logit_model_sha256 = hashlib.sha256(
        bytes(runtime.logit_booster.save_raw(raw_format="ubj"))
    ).hexdigest()
    if not (
        residual_model_sha256 == runtime.residual_model_sha256
        and logit_model_sha256 == runtime.logit_model_sha256
    ):
        raise MicroLiveExecutionError("bound runtime model bytes are mismatched")
    return canonical_json_sha256(
        {
            "candidate_id": runtime.candidate_id,
            "lineage_id": runtime.lineage_id,
            "manifest_sha256": runtime.manifest_sha256,
            "residual_model_sha256": runtime.residual_model_sha256,
            "logit_model_sha256": runtime.logit_model_sha256,
            "adapter_sha256": runtime.adapter_sha256,
            "maximum_decision_lag_ms": runtime.maximum_decision_lag_ms,
            "maximum_source_age_ms": runtime.maximum_source_age_ms,
            "coefficients": list(runtime.coefficients),
            "loaded_residual_model_sha256": residual_model_sha256,
            "loaded_logit_model_sha256": logit_model_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class _BoundAuthorization:
    """Immutable operational snapshot of one verified capability."""

    authorization_id: str
    authorization_payload_sha256: str
    candidate_bundle_sha256: str
    capital_base_usd: Decimal
    maximum_notional_usd: Decimal
    maximum_realized_loss_usd: Decimal
    maximum_open_orders: int
    authorized_at_ts_ms: int
    expires_at_ts_ms: int
    maximum_signal_age_ms: int
    maximum_operator_heartbeat_age_ms: int
    market_allowlist: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    runtime: ResidualPromotionRuntime
    runtime_integrity_sha256: str

    @classmethod
    def from_verified(
        cls,
        authorization: VerifiedMicroLiveAuthorization,
    ) -> _BoundAuthorization:
        runtime = copy.deepcopy(authorization.runtime)
        return cls(
            authorization_id=authorization.authorization_id,
            authorization_payload_sha256=authorization.authorization_payload_sha256,
            candidate_bundle_sha256=authorization.candidate_bundle_sha256,
            capital_base_usd=authorization.capital_base_usd,
            maximum_notional_usd=authorization.maximum_notional_usd,
            maximum_realized_loss_usd=authorization.maximum_realized_loss_usd,
            maximum_open_orders=authorization.maximum_open_orders,
            authorized_at_ts_ms=authorization.authorized_at_ts_ms,
            expires_at_ts_ms=authorization.expires_at_ts_ms,
            maximum_signal_age_ms=authorization.maximum_signal_age_ms,
            maximum_operator_heartbeat_age_ms=(
                authorization.maximum_operator_heartbeat_age_ms
            ),
            market_allowlist=authorization.market_allowlist,
            allowed_actions=authorization.allowed_actions,
            runtime=runtime,
            runtime_integrity_sha256=_runtime_integrity_sha256(runtime),
        )

    def matches_verified(self, authorization: VerifiedMicroLiveAuthorization) -> bool:
        return (
            self.authorization_id == authorization.authorization_id
            and self.authorization_payload_sha256
            == authorization.authorization_payload_sha256
            and self.candidate_bundle_sha256 == authorization.candidate_bundle_sha256
            and self.capital_base_usd == authorization.capital_base_usd
            and self.maximum_notional_usd == authorization.maximum_notional_usd
            and self.maximum_realized_loss_usd
            == authorization.maximum_realized_loss_usd
            and self.maximum_open_orders == authorization.maximum_open_orders
            and self.authorized_at_ts_ms == authorization.authorized_at_ts_ms
            and self.expires_at_ts_ms == authorization.expires_at_ts_ms
            and self.maximum_signal_age_ms == authorization.maximum_signal_age_ms
            and self.maximum_operator_heartbeat_age_ms
            == authorization.maximum_operator_heartbeat_age_ms
            and self.market_allowlist == authorization.market_allowlist
            and self.allowed_actions == authorization.allowed_actions
            and self.runtime_integrity_sha256
            == _runtime_integrity_sha256(authorization.runtime)
            == _runtime_integrity_sha256(self.runtime)
        )


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
        if not (
            authorization.runtime.lineage_id == LINEAGE_ID
            and authorization.runtime.candidate_id == CANDIDATE_ID
            and authorization.runtime.manifest_sha256
            == authorization.candidate_bundle_sha256
        ):
            raise MicroLiveExecutionError("micro-live runtime capability is mismatched")
        self.authorization = authorization
        self._authorization = _BoundAuthorization.from_verified(authorization)
        self.transport = transport
        self._events = [copy.deepcopy(dict(event)) for event in events]
        self._verify_event_chain()
        view = self._reconcile_view()
        if _realized_loss_limit_reached(view, self._authorization) and not view[
            "kill_switch_active"
        ]:
            raise MicroLiveExecutionError(
                "micro-live state crossed realized-loss limit without kill switch"
            )
        if (
            view["loss_budget_consumed_usd"]
            > self._authorization.maximum_realized_loss_usd
            and not view["kill_switch_active"]
        ):
            raise MicroLiveExecutionError(
                "micro-live state exceeded loss budget without kill switch"
            )

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

    def _require_authorization_integrity(
        self,
        *,
        now_ts_ms: int,
        reconciliation_only: bool = False,
    ) -> None:
        if authorization_capability_is_verified(self.authorization):
            try:
                if self._authorization.matches_verified(self.authorization):
                    return
            except Exception:
                pass
        now_is_valid = (
            isinstance(now_ts_ms, int)
            and not isinstance(now_ts_ms, bool)
            and now_ts_ms > 0
        )
        event_ts_ms = (
            now_ts_ms
            if now_is_valid
            else (
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else self._authorization.authorized_at_ts_ms
            )
        )
        try:
            self.engage_kill_switch(
                reason="authorization_capability_integrity_failed",
                now_ts_ms=event_ts_ms,
            )
        except Exception as exc:
            raise MicroLiveExecutionError(
                "micro-live authorization capability changed and kill persistence failed"
            ) from exc
        if reconciliation_only:
            return
        raise MicroLiveExecutionError(
            "micro-live authorization capability changed after executor construction"
        )

    def enforce_runtime_safety(
        self,
        *,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> dict[str, Any]:
        """Enforce time/loss safety even when no new model signal arrives.

        The future operator must call this watchdog independently of signal
        production.  A stale heartbeat or expired authorization therefore
        cancels acknowledged open orders instead of waiting for another market
        decision to exercise the signal-submission checks.
        """

        self._require_authorization_integrity(now_ts_ms=now_ts_ms)
        if isinstance(now_ts_ms, bool) or not isinstance(now_ts_ms, int) or now_ts_ms <= 0:
            fallback_ts_ms = (
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else self._authorization.authorized_at_ts_ms
            )
            self.engage_kill_switch(
                reason="runtime_watchdog_clock_invalid",
                now_ts_ms=fallback_ts_ms,
            )
            raise MicroLiveExecutionError("runtime watchdog timestamp is invalid")
        if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
            self.engage_kill_switch(
                reason="runtime_watchdog_clock_regression",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("runtime watchdog clock regressed")

        view = self._reconcile_view()
        if view["kill_switch_active"]:
            return {
                "status": "KILL_SWITCH_ALREADY_ACTIVE",
                "reason": view["kill_switch_reason"],
                "transport_called": False,
            }
        reason: str | None = None
        if now_ts_ms < self._authorization.authorized_at_ts_ms:
            reason = "runtime_clock_before_authorization"
        elif now_ts_ms >= self._authorization.expires_at_ts_ms:
            reason = "authorization_expired"
        elif (
            isinstance(operator_heartbeat_ts_ms, bool)
            or not isinstance(operator_heartbeat_ts_ms, int)
            or operator_heartbeat_ts_ms <= 0
            or operator_heartbeat_ts_ms > now_ts_ms
            or now_ts_ms - operator_heartbeat_ts_ms
            > self._authorization.maximum_operator_heartbeat_age_ms
        ):
            reason = "operator_heartbeat_stale"
        elif _realized_loss_limit_reached(view, self._authorization):
            reason = "maximum_realized_loss_reached"
        elif (
            view["loss_budget_consumed_usd"]
            > self._authorization.maximum_realized_loss_usd
        ):
            reason = "maximum_loss_budget_exceeded"

        if reason is not None:
            return self.engage_kill_switch(reason=reason, now_ts_ms=now_ts_ms)
        return {
            "status": "RUNTIME_SAFETY_OK",
            "checked_at_ts_ms": now_ts_ms,
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "transport_called": False,
        }

    def submit_signal(
        self,
        *,
        signal_payload: Mapping[str, Any],
        feature_row: Mapping[str, Any],
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
        market_identity_evidence: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        """Submit one authorized intent or return an explicit NO_TRADE block."""

        self._require_authorization_integrity(now_ts_ms=now_ts_ms)
        _require_positive_timestamp(now_ts_ms, "signal submission")
        try:
            signal = _validated_candidate_signal(
                signal_payload,
                expected_candidate_bundle_sha256=(
                    self._authorization.candidate_bundle_sha256
                ),
            )
            _validate_market_identity_evidence(
                signal=signal,
                evidence=market_identity_evidence,
            )
            features = _validated_runtime_binding(
                signal=signal,
                feature_row=feature_row,
                runtime=self._authorization.runtime,
            )
            decision_ts_ms = int(signal["decision_ts_ms"])
            self._validate_clock(
                now_ts_ms,
                operator_heartbeat_ts_ms,
                decision_ts_ms,
                int(signal["observed_at_ts_ms"]),
                int(dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]),
            )
        except MicroLiveExecutionError as exc:
            rejection_ts_ms = max(
                now_ts_ms,
                int(self._events[-1]["event_ts_ms"])
                if self._events
                else now_ts_ms,
            )
            self._append_event(
                "SIGNAL_REJECTED",
                {
                    "authorization_id": self._authorization.authorization_id,
                    "candidate_bundle_sha256": (
                        self._authorization.candidate_bundle_sha256
                    ),
                    "reason": "signal_validation_or_clock_failed",
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=rejection_ts_ms,
            )
            self.engage_kill_switch(
                reason="signal_validation_or_clock_failed",
                now_ts_ms=now_ts_ms,
            )
            raise
        view = self._reconcile_view()
        if _realized_loss_limit_reached(view, self._authorization) and not view[
            "kill_switch_active"
        ]:
            self.engage_kill_switch(
                reason="maximum_realized_loss_reached",
                now_ts_ms=now_ts_ms,
            )
            view = self._reconcile_view()
        if view["kill_switch_active"]:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="kill_switch_active",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("kill_switch_active")
        market_id = str(signal["market_id"])
        slug = str(signal["slug"])
        market_family = str(signal["market_family"])
        candidate_bundle_sha256 = str(signal["candidate_bundle_sha256"])
        selected_action = str(signal["selected_action"])
        if market_family not in self._authorization.market_allowlist:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="market_not_allowlisted",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("market_not_allowlisted")
        if signal["fail_closed"] is True or signal["model_scored"] is not True:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="signal_failed_closed",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("signal_failed_closed")
        if selected_action == "NO_TRADE":
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="signal_selected_no_trade",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("signal_selected_no_trade")
        if selected_action not in self._authorization.allowed_actions:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="action_not_allowlisted",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("action_not_allowlisted")
        token_side = "UP" if selected_action == "BUY_UP_HOLD" else "DOWN"
        token_id = str(signal[f"{token_side.lower()}_token_id"])
        price = _positive_decimal(
            dict(signal["executable_asks"])[token_side],
            "limit price",
        )
        qty = Decimal("1")
        notional = price * qty
        maximum_fee = FROZEN_EXECUTION_FEE_PER_UNIT_USD * qty
        maximum_loss = notional + maximum_fee
        signal_payload_sha256 = canonical_json_sha256(signal)
        business_key = canonical_json_sha256(
            {
                "authorization_id": self._authorization.authorization_id,
                "candidate_bundle_sha256": candidate_bundle_sha256,
                "market_id": market_id,
            }
        )
        intent_core = {
            "authorization_id": self._authorization.authorization_id,
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
            "maximum_fee_usd": str(maximum_fee),
            "maximum_loss_usd": str(maximum_loss),
            "signal_payload_sha256": signal_payload_sha256,
            "signal_payload": signal,
            "market_identity_sha256": canonical_json_sha256(
                dict(signal["market_identity"])
            ),
            "market_identity": dict(signal["market_identity"]),
            "feature_row_sha256": canonical_json_sha256(features),
            "feature_row": features,
        }
        intent_id = canonical_json_sha256(intent_core)
        client_order_id = intent_id
        existing = view["orders_by_business_key"].get(business_key)
        if existing is not None:
            if existing["intent_id"] != intent_id:
                if existing["prepared"]["decision_ts_ms"] != decision_ts_ms:
                    self._audit_signal_decision(
                        signal=signal,
                        features=features,
                        disposition="BLOCKED_NO_TRADE",
                        reason="one_trade_maximum_per_market",
                        now_ts_ms=now_ts_ms,
                        operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                    )
                    return _blocked("one_trade_maximum_per_market")
                self._audit_signal_decision(
                    signal=signal,
                    features=features,
                    disposition="REJECTED_CONFLICT",
                    reason="conflicting_duplicate_intent",
                    now_ts_ms=now_ts_ms,
                    operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                )
                self.engage_kill_switch(
                    reason="conflicting_duplicate_intent",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting duplicate intent failed closed")
            if existing.get("acknowledgement") is not None:
                self._audit_signal_decision(
                    signal=signal,
                    features=features,
                    disposition="IDEMPOTENT_REPLAY",
                    reason="existing_identical_intent",
                    now_ts_ms=now_ts_ms,
                    operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
                )
                return {
                    "status": "IDEMPOTENT_REPLAY",
                    "client_order_id": client_order_id,
                    "exchange_order_id": existing["acknowledgement"]["exchange_order_id"],
                    "transport_called": False,
                }
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="existing_order_requires_reconciliation",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("existing_order_requires_reconciliation", client_order_id)
        if view["submitted_notional_usd"] + notional > self._authorization.maximum_notional_usd:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="authorization_notional_cap_exceeded",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("authorization_notional_cap_exceeded")
        if view["open_order_count"] >= self._authorization.maximum_open_orders:
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="maximum_open_orders_reached",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("maximum_open_orders_reached")
        if (
            view["loss_budget_consumed_usd"] + maximum_loss
            > self._authorization.maximum_realized_loss_usd
        ):
            self._audit_signal_decision(
                signal=signal,
                features=features,
                disposition="BLOCKED_NO_TRADE",
                reason="maximum_loss_reservation_exceeded",
                now_ts_ms=now_ts_ms,
                operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
            )
            return _blocked("maximum_loss_reservation_exceeded")

        self._audit_signal_decision(
            signal=signal,
            features=features,
            disposition="EXECUTION_INTENT",
            reason=None,
            now_ts_ms=now_ts_ms,
            operator_heartbeat_ts_ms=operator_heartbeat_ts_ms,
        )
        prepared = {
            **intent_core,
            "intent_id": intent_id,
            "client_order_id": client_order_id,
            "submitted_at_ts_ms": now_ts_ms,
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "authorization_payload_sha256": self._authorization.authorization_payload_sha256,
        }
        self._append_event("ORDER_PREPARED", prepared, event_ts_ms=now_ts_ms)
        transport_request = {
            key: prepared[key]
            for key in (
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
                "maximum_fee_usd",
                "maximum_loss_usd",
                "signal_payload_sha256",
                "market_identity_sha256",
                "market_identity",
                "intent_id",
                "client_order_id",
                "submitted_at_ts_ms",
            )
        }
        try:
            response = dict(self.transport.submit_order(copy.deepcopy(transport_request)))
            disposition = self._validate_submission_response(transport_request, response)
            if (
                disposition["status"] == "ACCEPTED"
                and disposition["exchange_order_id"]
                in view["orders_by_exchange_id"]
            ):
                raise MicroLiveExecutionError(
                    "exchange order identity was reused across client orders"
                )
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_UNKNOWN",
                {
                    "client_order_id": client_order_id,
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=now_ts_ms,
            )
            self.engage_kill_switch(
                reason="order_submission_unknown",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("order submission became unknown; kill switch engaged") from exc
        if disposition["status"] == "REJECTED":
            self._append_event("ORDER_REJECTED", disposition, event_ts_ms=now_ts_ms)
            return {
                "status": "ORDER_REJECTED",
                "client_order_id": client_order_id,
                "transport_called": True,
            }
        self._append_event("ORDER_ACKNOWLEDGED", disposition, event_ts_ms=now_ts_ms)
        self._reconcile_view()
        return {
            "status": "ORDER_ACKNOWLEDGED",
            "client_order_id": client_order_id,
            "exchange_order_id": disposition["exchange_order_id"],
            "transport_called": True,
        }

    def reconcile_unknown_submission(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        """Resolve one unknown submission through a read-only transport lookup.

        Reconciliation never clears the persistent kill switch and never
        resubmits an order.  An accepted lookup result is immediately exposed
        to the existing kill-switch cancellation path.
        """

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        _require_positive_timestamp(now_ts_ms, "submission reconciliation")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if not (
            order is not None
            and order.get("submission_unknown") is True
            and order.get("acknowledgement") is None
            and order.get("closed_status") is None
            and view["kill_switch_active"] is True
        ):
            self.engage_kill_switch(
                reason="submission_reconciliation_precondition_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "submission reconciliation requires one killed unknown order"
            )
        prepared = dict(order["prepared"])
        lookup_request = {
            "authorization_id": self._authorization.authorization_id,
            "client_order_id": client_order_id,
            "business_key": prepared["business_key"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
        }
        try:
            response = dict(self.transport.lookup_order(copy.deepcopy(lookup_request)))
            disposition = self._validate_submission_response(prepared, response)
            exchange_order_id = disposition["exchange_order_id"]
            if (
                disposition["status"] == "ACCEPTED"
                and exchange_order_id in view["orders_by_exchange_id"]
            ):
                raise MicroLiveExecutionError(
                    "reconciled exchange order identity was reused"
                )
        except Exception as exc:
            self._append_event(
                "ORDER_SUBMISSION_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": canonical_json_sha256(lookup_request),
                    "error_type": exc.__class__.__name__,
                },
                event_ts_ms=max(
                    now_ts_ms,
                    int(self._events[-1]["event_ts_ms"]),
                ),
            )
            self.engage_kill_switch(
                reason="submission_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "unknown submission reconciliation failed closed"
            ) from exc
        self._append_event(
            "ORDER_SUBMISSION_RECONCILED",
            {
                **disposition,
                "lookup_response_sha256": canonical_json_sha256(response),
            },
            event_ts_ms=now_ts_ms,
        )
        cancel_result = self.engage_kill_switch(
            reason="submission_reconciled_after_unknown",
            now_ts_ms=now_ts_ms,
        )
        self._reconcile_view()
        return {
            "status": f"ORDER_SUBMISSION_RECONCILED_{disposition['status']}",
            "client_order_id": client_order_id,
            "exchange_order_id": disposition["exchange_order_id"],
            "kill_switch_active": True,
            "cancel_result": cancel_result,
        }

    def reconcile_unknown_cancellation(
        self,
        *,
        client_order_id: str,
        now_ts_ms: int,
    ) -> dict[str, Any]:
        """Resolve one ambiguous cancel through identity-bound read-only lookup.

        This method never submits or cancels an order and never clears the
        persistent kill switch.  It closes the local order only when lookup
        proves CANCELED or EXPIRED; OPEN/FILLED/invalid results remain audited
        and unresolved.
        """

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        _require_positive_timestamp(now_ts_ms, "cancel reconciliation")
        if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
            raise MicroLiveExecutionError("cancel reconciliation timestamp regressed")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if not (
            order is not None
            and order.get("acknowledgement") is not None
            and order.get("cancel_unknown") is True
            and order.get("closed_status") is None
            and order.get("settlement") is None
            and view["kill_switch_active"] is True
        ):
            self.engage_kill_switch(
                reason="cancel_reconciliation_precondition_failed",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "cancel reconciliation requires one killed cancel-unknown order"
            )
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        lookup_request = {
            "authorization_id": self._authorization.authorization_id,
            "client_order_id": client_order_id,
            "business_key": prepared["business_key"],
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "lookup_purpose": "cancel_reconciliation",
        }
        response: dict[str, Any] | None = None
        try:
            response = dict(self.transport.lookup_order(copy.deepcopy(lookup_request)))
            disposition = self._validate_cancel_lookup_response(
                order=order,
                response=response,
                now_ts_ms=now_ts_ms,
            )
        except Exception as exc:
            self._append_event(
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": canonical_json_sha256(lookup_request),
                    "error_type": exc.__class__.__name__,
                    "observed_status": None,
                    "lookup_response_sha256": None,
                    "raw_lookup_response_json": None,
                },
                event_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError(
                "unknown cancellation reconciliation failed closed"
            ) from exc
        raw_response_json = json.dumps(
            response,
            separators=(",", ":"),
            sort_keys=True,
        )
        lookup_response_sha256 = hashlib.sha256(
            raw_response_json.encode("utf-8")
        ).hexdigest()
        if disposition["status"] not in {"CANCELED", "EXPIRED"}:
            self._append_event(
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                {
                    "client_order_id": client_order_id,
                    "lookup_request_sha256": canonical_json_sha256(lookup_request),
                    "error_type": None,
                    "observed_status": disposition["status"],
                    "lookup_response_sha256": lookup_response_sha256,
                    "raw_lookup_response_json": raw_response_json,
                },
                event_ts_ms=now_ts_ms,
            )
            return {
                "status": f"ORDER_CANCEL_RECONCILIATION_{disposition['status']}",
                "client_order_id": client_order_id,
                "kill_switch_active": True,
                "order_closed": False,
                "lookup_called": True,
                "write_transport_called": False,
            }
        self._append_event(
            "ORDER_CANCEL_RECONCILED",
            {
                **disposition,
                "lookup_response_sha256": lookup_response_sha256,
                "raw_lookup_response_json": raw_response_json,
            },
            event_ts_ms=now_ts_ms,
        )
        self._reconcile_view()
        return {
            "status": f"ORDER_CANCEL_RECONCILED_{disposition['status']}",
            "client_order_id": client_order_id,
            "kill_switch_active": True,
            "order_closed": True,
            "lookup_called": True,
            "write_transport_called": False,
        }

    def _audit_signal_decision(
        self,
        *,
        signal: Mapping[str, Any],
        features: Mapping[str, Any],
        disposition: str,
        reason: str | None,
        now_ts_ms: int,
        operator_heartbeat_ts_ms: int,
    ) -> None:
        """Append the complete causal signal/input/decision audit row."""

        signal_copy = copy.deepcopy(dict(signal))
        feature_copy = copy.deepcopy(dict(features))
        core = {
            "authorization_id": self._authorization.authorization_id,
            "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
            "market_id": signal_copy["market_id"],
            "decision_ts_ms": signal_copy["decision_ts_ms"],
            "operator_heartbeat_ts_ms": operator_heartbeat_ts_ms,
            "signal_payload_sha256": canonical_json_sha256(signal_copy),
            "signal_payload": signal_copy,
            "market_identity_sha256": canonical_json_sha256(
                dict(signal_copy["market_identity"])
            ),
            "market_identity": copy.deepcopy(dict(signal_copy["market_identity"])),
            "feature_row_sha256": canonical_json_sha256(feature_copy),
            "feature_row": feature_copy,
            "disposition": disposition,
            "reason": reason,
        }
        self._append_event(
            "SIGNAL_EVALUATED",
            {**core, "decision_audit_sha256": canonical_json_sha256(core)},
            event_ts_ms=now_ts_ms,
        )

    def record_fill(
        self,
        *,
        client_order_id: str,
        fill_id: str,
        now_ts_ms: int,
        quantity: str,
        price: str,
        fee_usd: str,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        """Record one fill; every trusted-time reconciliation error kills."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        _require_positive_timestamp(now_ts_ms, "fill observation")
        try:
            if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
                raise MicroLiveExecutionError("fill observation timestamp regressed")
            return self._record_fill(
                client_order_id=client_order_id,
                fill_id=fill_id,
                now_ts_ms=now_ts_ms,
                quantity=quantity,
                price=price,
                fee_usd=fee_usd,
                raw_transport_event=raw_transport_event,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="fill_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _record_fill(
        self,
        *,
        client_order_id: str,
        fill_id: str,
        now_ts_ms: int,
        quantity: str,
        price: str,
        fee_usd: str,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        """Apply one externally observed fill to the append-only ledger."""

        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("fill has no acknowledged order")
        if order.get("settlement") is not None:
            self.engage_kill_switch(reason="fill_after_terminal_state", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill arrived after settlement")
        if not isinstance(fill_id, str) or not fill_id:
            raise MicroLiveExecutionError("fill identity is invalid")
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        transport_event, raw_json, transport_event_sha256 = _raw_json_object(
            raw_transport_event,
            "fill transport event",
        )
        expected_transport_keys = {
            "event_type",
            "client_order_id",
            "exchange_order_id",
            "fill_id",
            "market_id",
            "token_id",
            "quantity",
            "price",
            "fee_usd",
            "executed_at_ts_ms",
        }
        executed_at_ts_ms = transport_event.get("executed_at_ts_ms")
        if not (
            set(transport_event) == expected_transport_keys
            and transport_event.get("event_type") == "FILL"
            and transport_event.get("client_order_id") == client_order_id
            and transport_event.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and transport_event.get("fill_id") == fill_id
            and transport_event.get("market_id") == prepared["market_id"]
            and transport_event.get("token_id") == prepared["token_id"]
            and transport_event.get("quantity") == quantity
            and transport_event.get("price") == price
            and transport_event.get("fee_usd") == fee_usd
            and isinstance(executed_at_ts_ms, int)
            and not isinstance(executed_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"] <= executed_at_ts_ms <= now_ts_ms
        ):
            raise MicroLiveExecutionError("fill transport identity is mismatched")
        close_event = order.get("close_event")
        if close_event is not None and executed_at_ts_ms > int(
            close_event["effective_at_ts_ms"]
        ):
            self.engage_kill_switch(reason="fill_after_terminal_state", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill executed after order close")
        existing_fill = view["fills"].get(fill_id)
        payload = {
            "client_order_id": client_order_id,
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "fill_id": fill_id,
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "quantity": str(_positive_decimal(quantity, "fill quantity")),
            "price": str(_positive_decimal(price, "fill price")),
            "fee_usd": str(_nonnegative_decimal(fee_usd, "fill fee")),
            "executed_at_ts_ms": executed_at_ts_ms,
            "transport_event_sha256": transport_event_sha256,
            "raw_transport_event_json": raw_json,
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
        if fee > fill_qty * FROZEN_EXECUTION_FEE_PER_UNIT_USD:
            self.engage_kill_switch(
                reason="fill_fee_above_frozen_contract",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("fill fee exceeds frozen execution contract")
        if fill_price > Decimal(order["prepared"]["limit_price"]):
            self.engage_kill_switch(reason="fill_price_above_limit", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("buy fill price exceeds authorized limit")
        if fill_qty > order["remaining_quantity"]:
            self.engage_kill_switch(reason="fill_quantity_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill quantity exceeds remaining order")
        if fill_qty * fill_price + fee > view["cash_usd"]:
            self.engage_kill_switch(reason="fill_cash_cap_exceeded", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("fill would make authorized cash negative")
        self._append_event("FILL_RECORDED", payload, event_ts_ms=now_ts_ms)
        snapshot = self.reconciliation_snapshot()
        return {"status": "FILL_RECORDED", "fill_id": fill_id, "snapshot": snapshot}

    def record_order_closed(
        self,
        *,
        client_order_id: str,
        status: str,
        now_ts_ms: int,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        """Record an order close; any reconciliation ambiguity kills."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        _require_positive_timestamp(now_ts_ms, "order close observation")
        try:
            if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
                raise MicroLiveExecutionError("order close observation timestamp regressed")
            return self._record_order_closed(
                client_order_id=client_order_id,
                status=status,
                now_ts_ms=now_ts_ms,
                raw_transport_event=raw_transport_event,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="order_close_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _record_order_closed(
        self,
        *,
        client_order_id: str,
        status: str,
        now_ts_ms: int,
        raw_transport_event: bytes,
    ) -> dict[str, Any]:
        if status not in {"CANCELED", "EXPIRED"}:
            raise MicroLiveExecutionError("order close status is invalid")
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("order close has no acknowledged order")
        if order.get("settlement") is not None:
            self.engage_kill_switch(reason="order_close_after_settlement", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("order close arrived after settlement")
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        transport_event, raw_json, transport_event_sha256 = _raw_json_object(
            raw_transport_event,
            "order close transport event",
        )
        expected_transport_keys = {
            "event_type",
            "client_order_id",
            "exchange_order_id",
            "market_id",
            "token_id",
            "status",
            "effective_at_ts_ms",
        }
        effective_at_ts_ms = transport_event.get("effective_at_ts_ms")
        if not (
            set(transport_event) == expected_transport_keys
            and transport_event.get("event_type") == "ORDER_CLOSED"
            and transport_event.get("client_order_id") == client_order_id
            and transport_event.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and transport_event.get("market_id") == prepared["market_id"]
            and transport_event.get("token_id") == prepared["token_id"]
            and transport_event.get("status") == status
            and isinstance(effective_at_ts_ms, int)
            and not isinstance(effective_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"] <= effective_at_ts_ms <= now_ts_ms
        ):
            raise MicroLiveExecutionError("order close transport identity is mismatched")
        payload = {
            "client_order_id": client_order_id,
            "exchange_order_id": acknowledgement["exchange_order_id"],
            "market_id": prepared["market_id"],
            "token_id": prepared["token_id"],
            "effective_at_ts_ms": effective_at_ts_ms,
            "transport_event_sha256": transport_event_sha256,
            "raw_transport_event_json": raw_json,
        }
        existing = order.get("close_event")
        if existing is not None:
            if existing != payload or order.get("closed_status") != status:
                self.engage_kill_switch(
                    reason="conflicting_order_close",
                    now_ts_ms=now_ts_ms,
                )
                raise MicroLiveExecutionError("conflicting order close status")
            return {"status": "IDEMPOTENT_ORDER_CLOSE", "client_order_id": client_order_id}
        event_type = "ORDER_CANCELED" if status == "CANCELED" else "ORDER_EXPIRED"
        self._append_event(
            event_type,
            payload,
            event_ts_ms=now_ts_ms,
        )
        return {"status": event_type, "client_order_id": client_order_id}

    def record_settlement(
        self,
        *,
        client_order_id: str,
        settlement_id: str,
        now_ts_ms: int,
        payout_per_token: str,
        raw_official_settlement_event: bytes,
    ) -> dict[str, Any]:
        """Record official settlement; any reconciliation ambiguity kills."""

        self._require_authorization_integrity(
            now_ts_ms=now_ts_ms,
            reconciliation_only=True,
        )
        _require_positive_timestamp(now_ts_ms, "settlement observation")
        try:
            if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
                raise MicroLiveExecutionError("settlement observation timestamp regressed")
            return self._record_settlement(
                client_order_id=client_order_id,
                settlement_id=settlement_id,
                now_ts_ms=now_ts_ms,
                payout_per_token=payout_per_token,
                raw_official_settlement_event=raw_official_settlement_event,
            )
        except MicroLiveExecutionError:
            self.engage_kill_switch(
                reason="settlement_reconciliation_failed",
                now_ts_ms=now_ts_ms,
            )
            raise

    def _record_settlement(
        self,
        *,
        client_order_id: str,
        settlement_id: str,
        now_ts_ms: int,
        payout_per_token: str,
        raw_official_settlement_event: bytes,
    ) -> dict[str, Any]:
        view = self._reconcile_view()
        order = view["orders"].get(client_order_id)
        if order is None or order.get("acknowledgement") is None:
            raise MicroLiveExecutionError("settlement has no acknowledged order")
        payout = _nonnegative_decimal(payout_per_token, "settlement payout")
        if payout not in {Decimal("0"), Decimal("1")}:
            raise MicroLiveExecutionError("settlement payout must be official binary")
        if not isinstance(settlement_id, str) or not settlement_id:
            raise MicroLiveExecutionError("settlement identity is invalid")
        prepared = dict(order["prepared"])
        signal = dict(prepared["signal_payload"])
        official, raw_json, official_settlement_sha256 = _raw_json_object(
            raw_official_settlement_event,
            "official settlement event",
        )
        expected_official_keys = {
            "event_type",
            "settlement_id",
            "market_id",
            "slug",
            "winning_token_id",
            "payout_per_token",
            "finalized_at_ts_ms",
        }
        winning_token_id = official.get("winning_token_id")
        expected_winning_token_id = (
            prepared["token_id"]
            if payout == Decimal("1")
            else (
                signal["down_token_id"]
                if prepared["token_id"] == signal["up_token_id"]
                else signal["up_token_id"]
            )
        )
        finalized_at_ts_ms = official.get("finalized_at_ts_ms")
        if not (
            set(official) == expected_official_keys
            and official.get("event_type") == "OFFICIAL_SETTLEMENT"
            and official.get("settlement_id") == settlement_id
            and official.get("market_id") == prepared["market_id"]
            and official.get("slug") == prepared["slug"]
            and winning_token_id == expected_winning_token_id
            and official.get("payout_per_token") == payout_per_token
            and isinstance(finalized_at_ts_ms, int)
            and not isinstance(finalized_at_ts_ms, bool)
            and finalized_at_ts_ms >= int(
                dict(signal["market_identity"])["market_end_ts_ms"]
            )
            and finalized_at_ts_ms <= now_ts_ms
        ):
            raise MicroLiveExecutionError("official settlement identity is mismatched")
        payload = {
            "client_order_id": client_order_id,
            "settlement_id": settlement_id,
            "market_id": prepared["market_id"],
            "slug": prepared["slug"],
            "token_id": prepared["token_id"],
            "winning_token_id": winning_token_id,
            "payout_per_token": str(payout),
            "finalized_at_ts_ms": finalized_at_ts_ms,
            "official_settlement_sha256": official_settlement_sha256,
            "raw_official_settlement_json": raw_json,
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
        self._append_event("SETTLEMENT_RECORDED", payload, event_ts_ms=now_ts_ms)
        settled_view = self._reconcile_view()
        if _realized_loss_limit_reached(settled_view, self._authorization):
            self.engage_kill_switch(
                reason="maximum_realized_loss_reached",
                now_ts_ms=now_ts_ms,
            )
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
        event_ts_ms = max(
            now_ts_ms,
            int(self._events[-1]["event_ts_ms"]) if self._events else now_ts_ms,
        )
        view = self._reconcile_view()
        if not view["kill_switch_active"]:
            self._append_event(
                "KILL_SWITCH_ENGAGED",
                {"reason": reason, "engaged_at_ts_ms": event_ts_ms},
                event_ts_ms=event_ts_ms,
            )
            view = self._reconcile_view()
        canceled: list[str] = []
        unknown: list[str] = []
        for order in view["orders"].values():
            if not order["is_open"]:
                continue
            request = {
                "authorization_id": self._authorization.authorization_id,
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
                        "exchange_order_id": request["exchange_order_id"],
                        "market_id": order["prepared"]["market_id"],
                        "token_id": order["prepared"]["token_id"],
                        "effective_at_ts_ms": event_ts_ms,
                        "transport_event_sha256": canonical_json_sha256(response),
                        "raw_transport_event_json": json.dumps(
                            response,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                    event_ts_ms=event_ts_ms,
                )
                canceled.append(str(request["client_order_id"]))
            except Exception as exc:
                self._append_event(
                    "ORDER_CANCEL_UNKNOWN",
                    {
                        "client_order_id": request["client_order_id"],
                        "error_type": exc.__class__.__name__,
                    },
                    event_ts_ms=event_ts_ms,
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
            "authorization_id": self._authorization.authorization_id,
            "event_count": len(self._events),
            "kill_switch_active": view["kill_switch_active"],
            "kill_switch_reason": view["kill_switch_reason"],
            "cash_usd": str(view["cash_usd"]),
            "realized_pnl_usd": str(view["realized_pnl_usd"]),
            "maximum_realized_loss_usd": str(
                self._authorization.maximum_realized_loss_usd
            ),
            "unsettled_maximum_loss_usd": str(
                view["unsettled_maximum_loss_usd"]
            ),
            "loss_budget_consumed_usd": str(view["loss_budget_consumed_usd"]),
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
            "authorization_id": self._authorization.authorization_id,
            "authorization_payload_sha256": self._authorization.authorization_payload_sha256,
            "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
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
        identity_revalidated_at_ts_ms: int,
    ) -> None:
        values = (
            now_ts_ms,
            operator_heartbeat_ts_ms,
            decision_ts_ms,
            signal_observed_at_ts_ms,
            identity_revalidated_at_ts_ms,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise MicroLiveExecutionError("execution timestamp is invalid")
        if self._events and now_ts_ms < int(self._events[-1]["event_ts_ms"]):
            self.engage_kill_switch(
                reason="event_clock_regression",
                now_ts_ms=now_ts_ms,
            )
            raise MicroLiveExecutionError("micro-live event clock regressed")
        if now_ts_ms >= self._authorization.expires_at_ts_ms:
            self.engage_kill_switch(reason="authorization_expired", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live authorization expired")
        if not decision_ts_ms <= now_ts_ms or (
            now_ts_ms - decision_ts_ms > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal is stale")
        if not decision_ts_ms <= signal_observed_at_ts_ms <= now_ts_ms or (
            now_ts_ms - signal_observed_at_ts_ms
            > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="signal_observation_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live signal observation is stale")
        if not operator_heartbeat_ts_ms <= now_ts_ms or (
            now_ts_ms - operator_heartbeat_ts_ms
            > self._authorization.maximum_operator_heartbeat_age_ms
        ):
            self.engage_kill_switch(reason="operator_heartbeat_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live operator heartbeat is stale")
        if not identity_revalidated_at_ts_ms <= signal_observed_at_ts_ms or (
            now_ts_ms - identity_revalidated_at_ts_ms
            > self._authorization.maximum_signal_age_ms
        ):
            self.engage_kill_switch(reason="market_identity_stale", now_ts_ms=now_ts_ms)
            raise MicroLiveExecutionError("micro-live market identity is stale")

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

    def _validate_cancel_lookup_response(
        self,
        *,
        order: Mapping[str, Any],
        response: Mapping[str, Any],
        now_ts_ms: int,
    ) -> dict[str, Any]:
        expected_keys = {
            "client_order_id",
            "exchange_order_id",
            "status",
            "market_id",
            "token_id",
            "accepted_quantity",
            "limit_price",
            "observed_at_ts_ms",
            "effective_at_ts_ms",
        }
        if set(response) != expected_keys:
            raise MicroLiveExecutionError("cancel lookup response schema mismatch")
        prepared = dict(order["prepared"])
        acknowledgement = dict(order["acknowledgement"])
        status = response.get("status")
        observed_at_ts_ms = response.get("observed_at_ts_ms")
        effective_at_ts_ms = response.get("effective_at_ts_ms")
        if not (
            response.get("client_order_id") == prepared["client_order_id"]
            and response.get("exchange_order_id")
            == acknowledgement["exchange_order_id"]
            and response.get("market_id") == prepared["market_id"]
            and response.get("token_id") == prepared["token_id"]
            and response.get("accepted_quantity") == prepared["quantity"]
            and response.get("limit_price") == prepared["limit_price"]
            and status in {"OPEN", "FILLED", "CANCELED", "EXPIRED"}
            and isinstance(observed_at_ts_ms, int)
            and not isinstance(observed_at_ts_ms, bool)
            and prepared["submitted_at_ts_ms"] <= observed_at_ts_ms <= now_ts_ms
            and (
                (
                    status in {"CANCELED", "EXPIRED"}
                    and isinstance(effective_at_ts_ms, int)
                    and not isinstance(effective_at_ts_ms, bool)
                    and prepared["submitted_at_ts_ms"]
                    <= effective_at_ts_ms
                    <= observed_at_ts_ms
                )
                or (
                    status in {"OPEN", "FILLED"}
                    and effective_at_ts_ms is None
                )
            )
        ):
            raise MicroLiveExecutionError("cancel lookup response identity mismatch")
        return dict(response)

    def _append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_ts_ms: int,
    ) -> None:
        if event_type not in _EVENT_TYPES:
            raise MicroLiveExecutionError("micro-live event type is invalid")
        _require_positive_timestamp(event_ts_ms, "event")
        if self._events and event_ts_ms < self._events[-1]["event_ts_ms"]:
            raise MicroLiveExecutionError("micro-live event timestamp regressed")
        core = {
            "sequence": len(self._events) + 1,
            "event_ts_ms": event_ts_ms,
            "previous_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else GENESIS
            ),
            "event_type": event_type,
            "payload": copy.deepcopy(dict(payload)),
        }
        self._events.append({**core, "event_sha256": canonical_json_sha256(core)})

    def _verify_event_chain(self) -> None:
        previous = GENESIS
        previous_ts_ms = 0
        for expected_sequence, event in enumerate(self._events, start=1):
            if set(event) != {
                "sequence",
                "event_ts_ms",
                "previous_event_sha256",
                "event_type",
                "payload",
                "event_sha256",
            }:
                raise MicroLiveExecutionError("micro-live event schema is invalid")
            core = {key: value for key, value in event.items() if key != "event_sha256"}
            if not (
                event["sequence"] == expected_sequence
                and isinstance(event["event_ts_ms"], int)
                and not isinstance(event["event_ts_ms"], bool)
                and event["event_ts_ms"] > 0
                and event["event_ts_ms"] >= previous_ts_ms
                and event["previous_event_sha256"] == previous
                and event["event_type"] in _EVENT_TYPES
                and isinstance(event["payload"], Mapping)
                and event["event_sha256"] == canonical_json_sha256(core)
            ):
                raise MicroLiveExecutionError("micro-live event chain is invalid")
            self._validate_event_payload(
                str(event["event_type"]),
                event["payload"],
                event_ts_ms=int(event["event_ts_ms"]),
            )
            previous = str(event["event_sha256"])
            previous_ts_ms = int(event["event_ts_ms"])

    def _validate_event_payload(
        self,
        event_type: str,
        payload_value: Mapping[str, Any],
        *,
        event_ts_ms: int,
    ) -> None:
        payload = dict(payload_value)
        if event_type == "SIGNAL_REJECTED":
            if set(payload) != {
                "authorization_id",
                "candidate_bundle_sha256",
                "reason",
                "error_type",
            } or not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and payload.get("reason") == "signal_validation_or_clock_failed"
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError("rejected signal audit payload is invalid")
            return
        if event_type == "SIGNAL_EVALUATED":
            expected = {
                "authorization_id",
                "candidate_bundle_sha256",
                "market_id",
                "decision_ts_ms",
                "operator_heartbeat_ts_ms",
                "signal_payload_sha256",
                "signal_payload",
                "market_identity_sha256",
                "market_identity",
                "feature_row_sha256",
                "feature_row",
                "disposition",
                "reason",
                "decision_audit_sha256",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("evaluated signal audit schema is invalid")
            signal = _validated_candidate_signal(
                payload.get("signal_payload"),
                expected_candidate_bundle_sha256=(
                    self._authorization.candidate_bundle_sha256
                ),
            )
            feature_row = _validated_runtime_binding(
                signal=signal,
                feature_row=payload.get("feature_row"),
                runtime=self._authorization.runtime,
            )
            disposition = payload.get("disposition")
            reason = payload.get("reason")
            audit_core = {
                key: value
                for key, value in payload.items()
                if key != "decision_audit_sha256"
            }
            if not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and payload.get("market_id") == signal["market_id"]
                and payload.get("decision_ts_ms") == signal["decision_ts_ms"]
                and isinstance(payload.get("operator_heartbeat_ts_ms"), int)
                and not isinstance(payload.get("operator_heartbeat_ts_ms"), bool)
                and signal["decision_ts_ms"] <= event_ts_ms
                and event_ts_ms - signal["decision_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload["operator_heartbeat_ts_ms"] <= event_ts_ms
                and event_ts_ms - payload["operator_heartbeat_ts_ms"]
                <= self._authorization.maximum_operator_heartbeat_age_ms
                and dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]
                <= signal["observed_at_ts_ms"]
                and event_ts_ms
                - dict(signal["market_identity"])["clob_revalidated_at_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload.get("signal_payload_sha256")
                == canonical_json_sha256(signal)
                and payload.get("market_identity") == signal["market_identity"]
                and payload.get("market_identity_sha256")
                == canonical_json_sha256(dict(signal["market_identity"]))
                and payload.get("feature_row_sha256")
                == canonical_json_sha256(feature_row)
                and disposition
                in {
                    "EXECUTION_INTENT",
                    "BLOCKED_NO_TRADE",
                    "IDEMPOTENT_REPLAY",
                    "REJECTED_CONFLICT",
                }
                and (
                    (disposition == "EXECUTION_INTENT" and reason is None)
                    or (
                        disposition != "EXECUTION_INTENT"
                        and isinstance(reason, str)
                        and reason
                    )
                )
                and payload.get("decision_audit_sha256")
                == canonical_json_sha256(audit_core)
            ):
                raise MicroLiveExecutionError("evaluated signal audit payload is invalid")
            return
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
                "submitted_at_ts_ms",
                "operator_heartbeat_ts_ms",
                "selected_action",
                "token_id",
                "token_side",
                "limit_price",
                "quantity",
                "notional_usd",
                "maximum_fee_usd",
                "maximum_loss_usd",
                "signal_payload_sha256",
                "signal_payload",
                "market_identity_sha256",
                "market_identity",
                "feature_row_sha256",
                "feature_row",
                "intent_id",
                "client_order_id",
            }
            if set(payload) != expected:
                raise MicroLiveExecutionError("prepared order payload schema is invalid")
            price = _positive_decimal(payload.get("limit_price"), "stored limit price")
            quantity = _positive_decimal(payload.get("quantity"), "stored quantity")
            maximum_fee = _nonnegative_decimal(
                payload.get("maximum_fee_usd"), "stored maximum fee"
            )
            maximum_loss = _positive_decimal(
                payload.get("maximum_loss_usd"), "stored maximum loss"
            )
            if (
                price >= 1
                or quantity != Decimal("1")
                or payload.get("notional_usd") != str(price * quantity)
                or maximum_fee != quantity * FROZEN_EXECUTION_FEE_PER_UNIT_USD
                or maximum_loss != price * quantity + maximum_fee
            ):
                raise MicroLiveExecutionError("prepared order economics are invalid")
            signal = _validated_candidate_signal(
                payload.get("signal_payload"),
                expected_candidate_bundle_sha256=self._authorization.candidate_bundle_sha256,
            )
            feature_row = _validated_runtime_binding(
                signal=signal,
                feature_row=payload.get("feature_row"),
                runtime=self._authorization.runtime,
            )
            market_id = payload.get("market_id")
            business_key = canonical_json_sha256(
                {
                    "authorization_id": self._authorization.authorization_id,
                    "candidate_bundle_sha256": self._authorization.candidate_bundle_sha256,
                    "market_id": market_id,
                }
            )
            intent_core = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "authorization_payload_sha256",
                    "submitted_at_ts_ms",
                    "operator_heartbeat_ts_ms",
                    "intent_id",
                    "client_order_id",
                }
            }
            intent_id = canonical_json_sha256(intent_core)
            if not (
                payload.get("authorization_id") == self._authorization.authorization_id
                and payload.get("authorization_payload_sha256")
                == self._authorization.authorization_payload_sha256
                and payload.get("candidate_bundle_sha256")
                == self._authorization.candidate_bundle_sha256
                and isinstance(market_id, str)
                and market_id
                and isinstance(payload.get("slug"), str)
                and _SLUG.fullmatch(payload["slug"]) is not None
                and payload.get("market_family") in self._authorization.market_allowlist
                and isinstance(payload.get("decision_ts_ms"), int)
                and not isinstance(payload.get("decision_ts_ms"), bool)
                and payload["decision_ts_ms"] > 0
                and payload.get("selected_action") in self._authorization.allowed_actions
                and payload.get("selected_action") == signal["selected_action"]
                and payload.get("market_id") == signal["market_id"]
                and payload.get("slug") == signal["slug"]
                and payload.get("market_family") == signal["market_family"]
                and payload.get("decision_ts_ms") == signal["decision_ts_ms"]
                and payload.get("submitted_at_ts_ms") == event_ts_ms
                and isinstance(payload.get("operator_heartbeat_ts_ms"), int)
                and not isinstance(payload.get("operator_heartbeat_ts_ms"), bool)
                and self._authorization.authorized_at_ts_ms
                <= event_ts_ms
                < self._authorization.expires_at_ts_ms
                and signal["decision_ts_ms"] <= event_ts_ms
                and event_ts_ms - signal["decision_ts_ms"]
                <= self._authorization.maximum_signal_age_ms
                and payload["operator_heartbeat_ts_ms"] <= event_ts_ms
                and event_ts_ms - payload["operator_heartbeat_ts_ms"]
                <= self._authorization.maximum_operator_heartbeat_age_ms
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and payload.get("token_side")
                == ("UP" if payload.get("selected_action") == "BUY_UP_HOLD" else "DOWN")
                and payload.get("token_id")
                == signal[f"{str(payload.get('token_side')).lower()}_token_id"]
                and payload.get("limit_price")
                == dict(signal["executable_asks"])[payload["token_side"]]
                and payload.get("signal_payload_sha256") == canonical_json_sha256(signal)
                and payload.get("market_identity") == signal["market_identity"]
                and payload.get("market_identity_sha256")
                == canonical_json_sha256(dict(signal["market_identity"]))
                and payload.get("feature_row_sha256")
                == canonical_json_sha256(feature_row)
                and payload.get("business_key") == business_key
                and payload.get("intent_id") == intent_id
                and payload.get("client_order_id") == intent_id
            ):
                raise MicroLiveExecutionError("prepared order identity is invalid")
            return
        if event_type in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}:
            expected_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
            }
            if set(payload) != expected_keys:
                raise MicroLiveExecutionError("order disposition payload is invalid")
            accepted_quantity = _positive_decimal(
                payload.get("accepted_quantity"), "stored accepted quantity"
            )
            limit_price = _positive_decimal(
                payload.get("limit_price"), "stored disposition limit price"
            )
            if not (
                payload.get("status")
                == ("ACCEPTED" if event_type == "ORDER_ACKNOWLEDGED" else "REJECTED")
                and _is_sha256(payload.get("client_order_id"))
                and isinstance(payload.get("exchange_order_id"), str)
                and payload["exchange_order_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and accepted_quantity == Decimal("1")
                and limit_price < 1
            ):
                raise MicroLiveExecutionError("order disposition payload is invalid")
            return
        if event_type == "ORDER_SUBMISSION_RECONCILED":
            response_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
            }
            if set(payload) != {*response_keys, "lookup_response_sha256"}:
                raise MicroLiveExecutionError(
                    "submission reconciliation payload is invalid"
                )
            response = {key: payload[key] for key in response_keys}
            accepted_quantity = _positive_decimal(
                response.get("accepted_quantity"),
                "reconciled accepted quantity",
            )
            limit_price = _positive_decimal(
                response.get("limit_price"),
                "reconciled limit price",
            )
            if not (
                response.get("status") in {"ACCEPTED", "REJECTED"}
                and _is_sha256(response.get("client_order_id"))
                and isinstance(response.get("exchange_order_id"), str)
                and response["exchange_order_id"]
                and isinstance(response.get("market_id"), str)
                and _CONDITION_ID.fullmatch(response["market_id"]) is not None
                and isinstance(response.get("token_id"), str)
                and _TOKEN_ID.fullmatch(response["token_id"]) is not None
                and accepted_quantity == Decimal("1")
                and limit_price < 1
                and payload.get("lookup_response_sha256")
                == canonical_json_sha256(response)
            ):
                raise MicroLiveExecutionError(
                    "submission reconciliation payload is invalid"
                )
            return
        if event_type == "ORDER_SUBMISSION_RECONCILIATION_FAILED":
            if set(payload) != {
                "client_order_id",
                "lookup_request_sha256",
                "error_type",
            } or not (
                _is_sha256(payload.get("client_order_id"))
                and _is_sha256(payload.get("lookup_request_sha256"))
                and isinstance(payload.get("error_type"), str)
                and payload["error_type"]
            ):
                raise MicroLiveExecutionError(
                    "failed submission reconciliation audit is invalid"
                )
            return
        if event_type == "ORDER_CANCEL_RECONCILED":
            response_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
                "market_id",
                "token_id",
                "accepted_quantity",
                "limit_price",
                "observed_at_ts_ms",
                "effective_at_ts_ms",
            }
            if set(payload) != {
                *response_keys,
                "lookup_response_sha256",
                "raw_lookup_response_json",
            }:
                raise MicroLiveExecutionError(
                    "cancel reconciliation payload is invalid"
                )
            response = _stored_raw_json_object(
                payload.get("raw_lookup_response_json"),
                payload.get("lookup_response_sha256"),
                "stored cancel lookup response",
            )
            if not (
                set(response) == response_keys
                and all(response.get(key) == payload.get(key) for key in response_keys)
                and payload.get("status") in {"CANCELED", "EXPIRED"}
                and isinstance(payload.get("effective_at_ts_ms"), int)
                and not isinstance(payload.get("effective_at_ts_ms"), bool)
                and isinstance(payload.get("observed_at_ts_ms"), int)
                and not isinstance(payload.get("observed_at_ts_ms"), bool)
                and 0
                < payload["effective_at_ts_ms"]
                <= payload["observed_at_ts_ms"]
                <= event_ts_ms
            ):
                raise MicroLiveExecutionError(
                    "cancel reconciliation payload is invalid"
                )
            return
        if event_type == "ORDER_CANCEL_RECONCILIATION_FAILED":
            if set(payload) != {
                "client_order_id",
                "lookup_request_sha256",
                "error_type",
                "observed_status",
                "lookup_response_sha256",
                "raw_lookup_response_json",
            } or not (
                _is_sha256(payload.get("client_order_id"))
                and _is_sha256(payload.get("lookup_request_sha256"))
            ):
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
            error_type = payload.get("error_type")
            observed_status = payload.get("observed_status")
            if isinstance(error_type, str) and error_type:
                if any(
                    payload.get(key) is not None
                    for key in (
                        "observed_status",
                        "lookup_response_sha256",
                        "raw_lookup_response_json",
                    )
                ):
                    raise MicroLiveExecutionError(
                        "failed cancel reconciliation audit is invalid"
                    )
                return
            if error_type is not None or observed_status not in {"OPEN", "FILLED"}:
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
            response = _stored_raw_json_object(
                payload.get("raw_lookup_response_json"),
                payload.get("lookup_response_sha256"),
                "stored unresolved cancel lookup response",
            )
            if not (
                response.get("status") == observed_status
                and response.get("effective_at_ts_ms") is None
                and response.get("observed_at_ts_ms") <= event_ts_ms
            ):
                raise MicroLiveExecutionError(
                    "failed cancel reconciliation audit is invalid"
                )
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
                "exchange_order_id",
                "fill_id",
                "market_id",
                "token_id",
                "quantity",
                "price",
                "fee_usd",
                "executed_at_ts_ms",
                "transport_event_sha256",
                "raw_transport_event_json",
            }:
                raise MicroLiveExecutionError("fill payload schema is invalid")
            fill_quantity = _positive_decimal(
                payload.get("quantity"), "stored fill quantity"
            )
            fill_price = _positive_decimal(payload.get("price"), "stored fill price")
            fill_fee = _nonnegative_decimal(payload.get("fee_usd"), "stored fill fee")
            transport_event = _stored_raw_json_object(
                payload.get("raw_transport_event_json"),
                payload.get("transport_event_sha256"),
                "stored fill transport event",
            )
            if (
                fill_price >= 1
                or fill_fee > fill_quantity * FROZEN_EXECUTION_FEE_PER_UNIT_USD
                or set(transport_event)
                != {
                    "event_type",
                    "client_order_id",
                    "exchange_order_id",
                    "fill_id",
                    "market_id",
                    "token_id",
                    "quantity",
                    "price",
                    "fee_usd",
                    "executed_at_ts_ms",
                }
                or transport_event.get("event_type") != "FILL"
                or any(
                    transport_event.get(key) != payload.get(key)
                    for key in (
                        "client_order_id",
                        "exchange_order_id",
                        "fill_id",
                        "market_id",
                        "token_id",
                        "quantity",
                        "price",
                        "fee_usd",
                        "executed_at_ts_ms",
                    )
                )
                or not isinstance(payload.get("executed_at_ts_ms"), int)
                or isinstance(payload.get("executed_at_ts_ms"), bool)
                or payload["executed_at_ts_ms"] > event_ts_ms
            ):
                raise MicroLiveExecutionError("fill payload values are invalid")
            return
        if event_type in {"ORDER_CANCELED", "ORDER_EXPIRED"}:
            if set(payload) != {
                "client_order_id",
                "exchange_order_id",
                "market_id",
                "token_id",
                "effective_at_ts_ms",
                "transport_event_sha256",
                "raw_transport_event_json",
            } or not (
                isinstance(payload.get("client_order_id"), str)
                and payload["client_order_id"]
                and isinstance(payload.get("exchange_order_id"), str)
                and payload["exchange_order_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and isinstance(payload.get("effective_at_ts_ms"), int)
                and not isinstance(payload.get("effective_at_ts_ms"), bool)
                and 0 < payload["effective_at_ts_ms"] <= event_ts_ms
                and _is_sha256(payload.get("transport_event_sha256"))
            ):
                raise MicroLiveExecutionError("order close payload is invalid")
            transport_event = _stored_raw_json_object(
                payload.get("raw_transport_event_json"),
                payload.get("transport_event_sha256"),
                "stored order close transport event",
            )
            external_keys = {
                "event_type",
                "client_order_id",
                "exchange_order_id",
                "market_id",
                "token_id",
                "status",
                "effective_at_ts_ms",
            }
            direct_cancel_keys = {
                "client_order_id",
                "exchange_order_id",
                "status",
            }
            expected_status = "CANCELED" if event_type == "ORDER_CANCELED" else "EXPIRED"
            if set(transport_event) == external_keys:
                valid_transport = (
                    transport_event.get("event_type") == "ORDER_CLOSED"
                    and transport_event.get("status") == expected_status
                    and all(
                        transport_event.get(key) == payload.get(key)
                        for key in (
                            "client_order_id",
                            "exchange_order_id",
                            "market_id",
                            "token_id",
                            "effective_at_ts_ms",
                        )
                    )
                )
            else:
                valid_transport = (
                    event_type == "ORDER_CANCELED"
                    and set(transport_event) == direct_cancel_keys
                    and transport_event.get("status") == "CANCELED"
                    and transport_event.get("client_order_id")
                    == payload.get("client_order_id")
                    and transport_event.get("exchange_order_id")
                    == payload.get("exchange_order_id")
                )
            if not valid_transport:
                raise MicroLiveExecutionError("order close transport identity is invalid")
            return
        if event_type == "SETTLEMENT_RECORDED":
            payout = _nonnegative_decimal(
                payload.get("payout_per_token"), "stored settlement payout"
            )
            if set(payload) != {
                "client_order_id",
                "settlement_id",
                "market_id",
                "slug",
                "token_id",
                "winning_token_id",
                "payout_per_token",
                "finalized_at_ts_ms",
                "official_settlement_sha256",
                "raw_official_settlement_json",
            } or not (
                isinstance(payload.get("settlement_id"), str)
                and payload["settlement_id"]
                and isinstance(payload.get("market_id"), str)
                and _CONDITION_ID.fullmatch(payload["market_id"]) is not None
                and isinstance(payload.get("slug"), str)
                and _SLUG.fullmatch(payload["slug"]) is not None
                and isinstance(payload.get("token_id"), str)
                and _TOKEN_ID.fullmatch(payload["token_id"]) is not None
                and isinstance(payload.get("winning_token_id"), str)
                and _TOKEN_ID.fullmatch(payload["winning_token_id"]) is not None
                and payout in {Decimal("0"), Decimal("1")}
                and isinstance(payload.get("finalized_at_ts_ms"), int)
                and not isinstance(payload.get("finalized_at_ts_ms"), bool)
                and 0 < payload["finalized_at_ts_ms"] <= event_ts_ms
                and _is_sha256(payload.get("official_settlement_sha256"))
            ):
                raise MicroLiveExecutionError("settlement payload is invalid")
            official = _stored_raw_json_object(
                payload.get("raw_official_settlement_json"),
                payload.get("official_settlement_sha256"),
                "stored official settlement event",
            )
            if set(official) != {
                "event_type",
                "settlement_id",
                "market_id",
                "slug",
                "winning_token_id",
                "payout_per_token",
                "finalized_at_ts_ms",
            } or not (
                official.get("event_type") == "OFFICIAL_SETTLEMENT"
                and all(
                    official.get(key) == payload.get(key)
                    for key in (
                        "settlement_id",
                        "market_id",
                        "slug",
                        "winning_token_id",
                        "payout_per_token",
                        "finalized_at_ts_ms",
                    )
                )
            ):
                raise MicroLiveExecutionError("settlement transport identity is invalid")
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
        exchange_order_ids: dict[str, dict[str, Any]] = {}
        fills: dict[str, dict[str, Any]] = {}
        settlements: dict[str, dict[str, Any]] = {}
        kill_switch_active = False
        kill_switch_reason: str | None = None
        pending_execution_audit: dict[str, Any] | None = None
        for event in self._events:
            event_type = event["event_type"]
            payload = dict(event["payload"])
            client_order_id = payload.get("client_order_id")
            if pending_execution_audit is not None and event_type != "ORDER_PREPARED":
                raise MicroLiveExecutionError(
                    "execution-intent audit is not followed by order preparation"
                )
            if event_type == "SIGNAL_EVALUATED":
                if payload.get("disposition") == "EXECUTION_INTENT":
                    pending_execution_audit = payload
                continue
            if event_type == "SIGNAL_REJECTED":
                continue
            if event_type == "ORDER_PREPARED":
                if (
                    pending_execution_audit is None
                    or not isinstance(client_order_id, str)
                    or client_order_id in orders
                    or payload.get("intent_id") != client_order_id
                    or payload.get("authorization_id") != self._authorization.authorization_id
                    or payload.get("candidate_bundle_sha256")
                    != self._authorization.candidate_bundle_sha256
                ):
                    raise MicroLiveExecutionError("prepared order identity is invalid")
                if not (
                    pending_execution_audit.get("market_id") == payload.get("market_id")
                    and pending_execution_audit.get("decision_ts_ms")
                    == payload.get("decision_ts_ms")
                    and pending_execution_audit.get("signal_payload_sha256")
                    == payload.get("signal_payload_sha256")
                    and pending_execution_audit.get("market_identity_sha256")
                    == payload.get("market_identity_sha256")
                    and pending_execution_audit.get("feature_row_sha256")
                    == payload.get("feature_row_sha256")
                ):
                    raise MicroLiveExecutionError(
                        "order preparation does not reconcile with execution-intent audit"
                    )
                pending_execution_audit = None
                business_key = payload.get("business_key")
                if not isinstance(business_key, str) or business_key in business_keys:
                    raise MicroLiveExecutionError("prepared order business identity is duplicated")
                order = {
                    "intent_id": client_order_id,
                    "prepared": payload,
                    "acknowledgement": None,
                    "submission_unknown": False,
                    "cancel_unknown": False,
                    "closed_status": None,
                    "close_event": None,
                    "fills": [],
                    "settlement": None,
                }
                orders[client_order_id] = order
                business_keys[business_key] = order
            elif event_type in {
                "ORDER_ACKNOWLEDGED",
                "ORDER_REJECTED",
                "ORDER_SUBMISSION_UNKNOWN",
                "ORDER_SUBMISSION_RECONCILED",
                "ORDER_SUBMISSION_RECONCILIATION_FAILED",
                "ORDER_CANCEL_RECONCILED",
                "ORDER_CANCEL_RECONCILIATION_FAILED",
                "ORDER_CANCELED",
                "ORDER_EXPIRED",
                "ORDER_CANCEL_UNKNOWN",
            }:
                order = orders.get(str(client_order_id))
                if order is None:
                    raise MicroLiveExecutionError("order lifecycle event lacks preparation")
                if event_type == "ORDER_ACKNOWLEDGED":
                    prepared = order["prepared"]
                    exchange_order_id = payload.get("exchange_order_id")
                    if (
                        order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or order["submission_unknown"]
                        or not isinstance(exchange_order_id, str)
                        or exchange_order_id in exchange_order_ids
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                    ):
                        raise MicroLiveExecutionError("order acknowledgement is duplicated")
                    order["acknowledgement"] = payload
                    exchange_order_ids[exchange_order_id] = order
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
                elif event_type == "ORDER_SUBMISSION_RECONCILED":
                    exchange_order_id = payload.get("exchange_order_id")
                    prepared = order["prepared"]
                    if (
                        order["submission_unknown"] is not True
                        or order["acknowledgement"] is not None
                        or order["closed_status"] is not None
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                    ):
                        raise MicroLiveExecutionError(
                            "submission reconciliation lifecycle is invalid"
                        )
                    order["submission_unknown"] = False
                    if payload.get("status") == "ACCEPTED":
                        if (
                            not isinstance(exchange_order_id, str)
                            or exchange_order_id in exchange_order_ids
                        ):
                            raise MicroLiveExecutionError(
                                "reconciled exchange order identity is duplicated"
                            )
                        acknowledgement = {
                            key: value
                            for key, value in payload.items()
                            if key != "lookup_response_sha256"
                        }
                        order["acknowledgement"] = acknowledgement
                        exchange_order_ids[exchange_order_id] = order
                    elif payload.get("status") == "REJECTED":
                        order["closed_status"] = "REJECTED"
                    else:
                        raise MicroLiveExecutionError(
                            "submission reconciliation status is invalid"
                        )
                elif event_type == "ORDER_SUBMISSION_RECONCILIATION_FAILED":
                    prepared = order["prepared"]
                    expected_lookup_request = {
                        "authorization_id": self._authorization.authorization_id,
                        "client_order_id": client_order_id,
                        "business_key": prepared["business_key"],
                        "market_id": prepared["market_id"],
                        "token_id": prepared["token_id"],
                    }
                    if (
                        order["submission_unknown"] is not True
                        or payload.get("lookup_request_sha256")
                        != canonical_json_sha256(expected_lookup_request)
                    ):
                        raise MicroLiveExecutionError(
                            "failed submission reconciliation lifecycle is invalid"
                        )
                elif event_type == "ORDER_CANCEL_RECONCILED":
                    prepared = order["prepared"]
                    acknowledgement = order["acknowledgement"]
                    if (
                        acknowledgement is None
                        or order["cancel_unknown"] is not True
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                        or payload.get("exchange_order_id")
                        != acknowledgement["exchange_order_id"]
                        or payload.get("market_id") != prepared["market_id"]
                        or payload.get("token_id") != prepared["token_id"]
                        or payload.get("accepted_quantity") != prepared["quantity"]
                        or payload.get("limit_price") != prepared["limit_price"]
                        or payload.get("status") not in {"CANCELED", "EXPIRED"}
                        or int(payload["effective_at_ts_ms"])
                        < int(prepared["submitted_at_ts_ms"])
                    ):
                        raise MicroLiveExecutionError(
                            "cancel reconciliation lifecycle is invalid"
                        )
                    order["cancel_unknown"] = False
                    order["closed_status"] = payload["status"]
                    order["close_event"] = payload
                elif event_type == "ORDER_CANCEL_RECONCILIATION_FAILED":
                    prepared = order["prepared"]
                    acknowledgement = order["acknowledgement"]
                    if acknowledgement is None:
                        raise MicroLiveExecutionError(
                            "failed cancel reconciliation lifecycle is invalid"
                        )
                    expected_lookup_request = {
                        "authorization_id": self._authorization.authorization_id,
                        "client_order_id": client_order_id,
                        "business_key": prepared["business_key"],
                        "exchange_order_id": acknowledgement["exchange_order_id"],
                        "market_id": prepared["market_id"],
                        "token_id": prepared["token_id"],
                        "lookup_purpose": "cancel_reconciliation",
                    }
                    if (
                        order["cancel_unknown"] is not True
                        or order["closed_status"] is not None
                        or payload.get("lookup_request_sha256")
                        != canonical_json_sha256(expected_lookup_request)
                    ):
                        raise MicroLiveExecutionError(
                            "failed cancel reconciliation lifecycle is invalid"
                        )
                    if payload.get("observed_status") is not None:
                        response = _stored_raw_json_object(
                            payload["raw_lookup_response_json"],
                            payload["lookup_response_sha256"],
                            "stored unresolved cancel lookup response",
                        )
                        self._validate_cancel_lookup_response(
                            order=order,
                            response=response,
                            now_ts_ms=int(event["event_ts_ms"]),
                        )
                elif event_type == "ORDER_CANCELED":
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                        or payload.get("exchange_order_id")
                        != order["acknowledgement"]["exchange_order_id"]
                        or payload.get("market_id")
                        != order["prepared"]["market_id"]
                        or payload.get("token_id")
                        != order["prepared"]["token_id"]
                        or int(payload["effective_at_ts_ms"])
                        < int(order["prepared"]["submitted_at_ts_ms"])
                    ):
                        raise MicroLiveExecutionError("order cancellation lifecycle is invalid")
                    order["cancel_unknown"] = False
                    order["closed_status"] = "CANCELED"
                    order["close_event"] = payload
                elif event_type == "ORDER_EXPIRED":
                    if (
                        order["acknowledgement"] is None
                        or order["closed_status"] is not None
                        or order["settlement"] is not None
                        or payload.get("exchange_order_id")
                        != order["acknowledgement"]["exchange_order_id"]
                        or payload.get("market_id")
                        != order["prepared"]["market_id"]
                        or payload.get("token_id")
                        != order["prepared"]["token_id"]
                        or int(payload["effective_at_ts_ms"])
                        < int(order["prepared"]["submitted_at_ts_ms"])
                    ):
                        raise MicroLiveExecutionError("order expiration lifecycle is invalid")
                    order["cancel_unknown"] = False
                    order["closed_status"] = "EXPIRED"
                    order["close_event"] = payload
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
                close_event = None if order is None else order.get("close_event")
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["settlement"] is not None
                    or not isinstance(fill_id, str)
                    or fill_id in fills
                    or payload.get("exchange_order_id")
                    != order["acknowledgement"]["exchange_order_id"]
                    or payload.get("market_id") != order["prepared"]["market_id"]
                    or payload.get("token_id") != order["prepared"]["token_id"]
                    or int(payload["executed_at_ts_ms"])
                    < int(order["prepared"]["submitted_at_ts_ms"])
                    or (
                        close_event is not None
                        and int(payload["executed_at_ts_ms"])
                        > int(close_event["effective_at_ts_ms"])
                    )
                ):
                    raise MicroLiveExecutionError("fill event identity is invalid")
                fills[fill_id] = payload
                order["fills"].append(payload)
            elif event_type == "SETTLEMENT_RECORDED":
                order = orders.get(str(client_order_id))
                prepared = None if order is None else dict(order["prepared"])
                signal = (
                    None
                    if prepared is None
                    else dict(prepared["signal_payload"])
                )
                if (
                    order is None
                    or order["acknowledgement"] is None
                    or order["settlement"] is not None
                    or not order["fills"]
                    or payload.get("market_id") != prepared["market_id"]
                    or payload.get("slug") != prepared["slug"]
                    or payload.get("token_id") != prepared["token_id"]
                    or int(payload["finalized_at_ts_ms"])
                    < int(dict(signal["market_identity"])["market_end_ts_ms"])
                    or payload.get("winning_token_id")
                    != (
                        prepared["token_id"]
                        if Decimal(payload["payout_per_token"]) == Decimal("1")
                        else (
                            signal["down_token_id"]
                            if prepared["token_id"] == signal["up_token_id"]
                            else signal["up_token_id"]
                        )
                    )
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

        if pending_execution_audit is not None:
            raise MicroLiveExecutionError(
                "execution-intent audit is missing order preparation"
            )

        cash = self._authorization.maximum_notional_usd
        realized_pnl = Decimal("0")
        unsettled_maximum_loss = Decimal("0")
        positions = {"UP": Decimal("0"), "DOWN": Decimal("0")}
        submitted_notional = Decimal("0")
        open_order_count = 0
        for order in orders.values():
            prepared = order["prepared"]
            requested_qty = Decimal(prepared["quantity"])
            submitted_notional += Decimal(prepared["notional_usd"])
            filled_qty = Decimal("0")
            filled_cost = Decimal("0")
            for fill in order["fills"]:
                qty = Decimal(fill["quantity"])
                price = Decimal(fill["price"])
                fee = Decimal(fill["fee_usd"])
                filled_qty += qty
                filled_cost += qty * price + fee
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
            if settlement is None:
                if order["submission_unknown"] or (
                    order["acknowledgement"] is None
                    and order["closed_status"] is None
                ):
                    unsettled_maximum_loss += Decimal(
                        prepared["maximum_loss_usd"]
                    )
                elif order["closed_status"] != "REJECTED":
                    unsettled_maximum_loss += filled_cost
                    if order["is_open"]:
                        remaining = order["remaining_quantity"]
                        unsettled_maximum_loss += remaining * (
                            Decimal(prepared["limit_price"])
                            + FROZEN_EXECUTION_FEE_PER_UNIT_USD
                        )
            if settlement is not None:
                if filled_qty < requested_qty and order["closed_status"] is None:
                    raise MicroLiveExecutionError(
                        "partially filled order settled before terminal close"
                    )
                payout = Decimal(settlement["payout_per_token"])
                cash += filled_qty * payout
                realized_cost = sum(
                    Decimal(fill["quantity"]) * Decimal(fill["price"])
                    + Decimal(fill["fee_usd"])
                    for fill in order["fills"]
                )
                realized_pnl += filled_qty * payout - realized_cost
                side = "UP" if prepared["selected_action"] == "BUY_UP_HOLD" else "DOWN"
                positions[side] -= filled_qty
        if submitted_notional > self._authorization.maximum_notional_usd:
            raise MicroLiveExecutionError("submitted notional exceeds authorization")
        if open_order_count > self._authorization.maximum_open_orders:
            raise MicroLiveExecutionError("open order count exceeds authorization")
        loss_budget_consumed = max(-realized_pnl, Decimal("0")) + (
            unsettled_maximum_loss
        )
        return {
            "orders": orders,
            "orders_by_business_key": business_keys,
            "orders_by_exchange_id": exchange_order_ids,
            "fills": fills,
            "settlements": settlements,
            "cash_usd": cash,
            "realized_pnl_usd": realized_pnl,
            "unsettled_maximum_loss_usd": unsettled_maximum_loss,
            "loss_budget_consumed_usd": loss_budget_consumed,
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


def _realized_loss_limit_reached(
    view: Mapping[str, Any],
    authorization: VerifiedMicroLiveAuthorization | _BoundAuthorization,
) -> bool:
    realized = view.get("realized_pnl_usd")
    if not isinstance(realized, Decimal):
        raise MicroLiveExecutionError("realized PnL reconciliation is invalid")
    return realized <= -authorization.maximum_realized_loss_usd


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
        "market_identity",
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
    _validated_market_identity(signal)
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


def _validated_market_identity(signal: Mapping[str, Any]) -> None:
    identity_value = signal.get("market_identity")
    if not isinstance(identity_value, Mapping):
        raise MicroLiveExecutionError("candidate market identity is absent")
    identity = dict(identity_value)
    expected_keys = {
        "source_type",
        "condition_id",
        "slug",
        "market_family",
        "market_start_ts_ms",
        "market_end_ts_ms",
        "up_token_id",
        "down_token_id",
        "gamma_fetched_at_ts_ms",
        "clob_revalidated_at_ts_ms",
        "raw_gamma_payload_sha256",
        "clob_revalidation_payload_sha256",
        "clob_revalidation_passed",
        "outcomes_accessed",
        "settlement_accessed",
        "pnl_accessed",
    }
    if set(identity) != expected_keys:
        raise MicroLiveExecutionError("candidate market identity schema is not exact")
    start = int(str(signal["slug"]).rsplit("-", maxsplit=1)[1]) * 1_000
    fetched_at = identity.get("gamma_fetched_at_ts_ms")
    revalidated_at = identity.get("clob_revalidated_at_ts_ms")
    if not (
        identity.get("source_type") == "gamma_primary_plus_live_clob_revalidation"
        and identity.get("condition_id") == signal["market_id"]
        and identity.get("slug") == signal["slug"]
        and identity.get("market_family") == "btc_updown_15m"
        and identity.get("market_start_ts_ms") == start
        and identity.get("market_end_ts_ms") == start + 900_000
        and identity.get("up_token_id") == signal["up_token_id"]
        and identity.get("down_token_id") == signal["down_token_id"]
        and isinstance(fetched_at, int)
        and not isinstance(fetched_at, bool)
        and fetched_at > 0
        and fetched_at <= signal["observed_at_ts_ms"]
        and isinstance(revalidated_at, int)
        and not isinstance(revalidated_at, bool)
        and fetched_at <= revalidated_at <= signal["observed_at_ts_ms"]
        and _is_sha256(identity.get("raw_gamma_payload_sha256"))
        and _is_sha256(identity.get("clob_revalidation_payload_sha256"))
        and identity.get("clob_revalidation_passed") is True
        and identity.get("outcomes_accessed") is False
        and identity.get("settlement_accessed") is False
        and identity.get("pnl_accessed") is False
    ):
        raise MicroLiveExecutionError("candidate market identity binding is invalid")


def _validate_market_identity_evidence(
    *,
    signal: Mapping[str, Any],
    evidence: Any,
) -> None:
    """Verify the exact provider bytes behind the signal identity hashes."""

    if not isinstance(evidence, Mapping) or set(evidence) != {
        "raw_gamma_payload",
        "raw_clob_revalidation_payload",
    }:
        raise MicroLiveExecutionError("market identity evidence schema is not exact")
    gamma_raw = evidence.get("raw_gamma_payload")
    clob_raw = evidence.get("raw_clob_revalidation_payload")
    if not isinstance(gamma_raw, bytes) or not gamma_raw:
        raise MicroLiveExecutionError("raw Gamma market identity bytes are invalid")
    if not isinstance(clob_raw, bytes) or not clob_raw:
        raise MicroLiveExecutionError("raw CLOB revalidation bytes are invalid")
    identity = dict(signal["market_identity"])
    if not (
        hashlib.sha256(gamma_raw).hexdigest()
        == identity["raw_gamma_payload_sha256"]
        and hashlib.sha256(clob_raw).hexdigest()
        == identity["clob_revalidation_payload_sha256"]
    ):
        raise MicroLiveExecutionError("market identity raw-byte SHA-256 mismatch")
    gamma = _decode_provider_json(gamma_raw, "Gamma market identity")
    clob = _decode_provider_json(clob_raw, "CLOB market identity")
    slug = str(signal["slug"])
    market_id = str(signal["market_id"])
    expected_tokens = {
        "UP": str(signal["up_token_id"]),
        "DOWN": str(signal["down_token_id"]),
    }
    gamma_rows = _gamma_market_rows(gamma)
    exact_gamma = [
        row
        for row in gamma_rows
        if str(row.get("slug") or row.get("market_slug") or "") == slug
    ]
    if len(exact_gamma) != 1:
        raise MicroLiveExecutionError("Gamma identity does not resolve exact slug once")
    gamma_row = exact_gamma[0]
    gamma_condition = str(
        gamma_row.get("conditionId") or gamma_row.get("condition_id") or ""
    )
    gamma_tokens = _gamma_up_down_tokens(gamma_row)
    if gamma_condition != market_id or gamma_tokens != expected_tokens:
        raise MicroLiveExecutionError("Gamma condition or token identity mismatches")
    if not isinstance(clob, Mapping):
        raise MicroLiveExecutionError("CLOB market identity payload is not an object")
    clob_row = dict(clob)
    clob_condition = str(
        clob_row.get("condition_id")
        or clob_row.get("conditionId")
        or clob_row.get("market")
        or ""
    )
    clob_slug = str(clob_row.get("market_slug") or clob_row.get("slug") or "")
    clob_tokens = _clob_up_down_tokens(clob_row.get("tokens"))
    if not (
        clob_condition == market_id
        and clob_slug == slug
        and clob_tokens == expected_tokens
    ):
        raise MicroLiveExecutionError("CLOB condition, slug, or token identity mismatches")


def _decode_provider_json(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
        _validate_finite_json_tree(value)
        return value
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise MicroLiveExecutionError(f"{label} raw bytes are not strict JSON") from exc


def _raw_json_object(raw: Any, label: str) -> tuple[dict[str, Any], str, str]:
    if not isinstance(raw, bytes) or not raw:
        raise MicroLiveExecutionError(f"{label} raw bytes are invalid")
    value = _decode_provider_json(raw, label)
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError(f"{label} is not a JSON object")
    return dict(value), raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _stored_raw_json_object(raw_json: Any, expected_sha256: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw_json, str) or not raw_json:
        raise MicroLiveExecutionError(f"{label} raw JSON is invalid")
    raw = raw_json.encode("utf-8")
    if not _is_sha256(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MicroLiveExecutionError(f"{label} SHA-256 mismatch")
    value = _decode_provider_json(raw, label)
    if not isinstance(value, Mapping):
        raise MicroLiveExecutionError(f"{label} is not a JSON object")
    return dict(value)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_finite_json_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decoded JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_json_tree(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json_tree(item)


def _gamma_market_rows(value: Any) -> list[dict[str, Any]]:
    rows: Any
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        wrapper = dict(value)
        if isinstance(wrapper.get("data"), list):
            rows = wrapper["data"]
        elif isinstance(wrapper.get("markets"), list):
            rows = wrapper["markets"]
        else:
            rows = [wrapper]
    else:
        raise MicroLiveExecutionError("Gamma market identity payload shape is invalid")
    if not all(isinstance(row, Mapping) for row in rows):
        raise MicroLiveExecutionError("Gamma market identity row is not an object")
    return [dict(row) for row in rows]


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
            _validate_finite_json_tree(decoded)
        except (ValueError, json.JSONDecodeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _gamma_up_down_tokens(row: Mapping[str, Any]) -> dict[str, str] | None:
    outcomes = _json_array(row.get("outcomes"))
    token_ids = _json_array(row.get("clobTokenIds"))
    if len(outcomes) != len(token_ids):
        return None
    return _up_down_tokens(zip(outcomes, token_ids, strict=True))


def _clob_up_down_tokens(value: Any) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    pairs: list[tuple[Any, Any]] = []
    for token in value:
        if not isinstance(token, Mapping):
            return None
        row = dict(token)
        outcome = row.get("outcome") or row.get("name") or row.get("label")
        token_id = row.get("token_id") or row.get("tokenId") or row.get("asset_id")
        pairs.append((outcome, token_id))
    return _up_down_tokens(pairs)


def _up_down_tokens(pairs: Any) -> dict[str, str] | None:
    tokens: dict[str, str] = {}
    for outcome_value, token_value in pairs:
        outcome = str(outcome_value or "").strip().upper()
        token_id = str(token_value or "")
        if outcome not in {"UP", "DOWN"} or _TOKEN_ID.fullmatch(token_id) is None:
            return None
        if outcome in tokens:
            return None
        tokens[outcome] = token_id
    return tokens if set(tokens) == {"UP", "DOWN"} else None


def _validated_runtime_binding(
    *,
    signal: Mapping[str, Any],
    feature_row: Any,
    runtime: ResidualPromotionRuntime,
) -> dict[str, Any]:
    if not isinstance(feature_row, Mapping):
        raise MicroLiveExecutionError("candidate feature row is not an object")
    features = copy.deepcopy(dict(feature_row))
    _reject_forbidden_feature_keys(features)
    if not (
        features.get("market_id") == signal["market_id"]
        and features.get("market_family") == "btc_updown_15m"
        and features.get("horizon_ms") == 900_000
        and features.get("decision_ts") == signal["decision_ts_ms"]
    ):
        raise MicroLiveExecutionError("candidate feature row identity is invalid")
    raw = features.get("features")
    if not isinstance(raw, Mapping):
        raise MicroLiveExecutionError("candidate raw feature mapping is absent")
    try:
        raw_asks = {
            "UP": _numeric_decimal(raw.get("up_ask"), "UP feature ask"),
            "DOWN": _numeric_decimal(raw.get("down_ask"), "DOWN feature ask"),
        }
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MicroLiveExecutionError("candidate raw executable asks are invalid") from exc
    signal_asks = {
        side: _positive_decimal(
            dict(signal["executable_asks"])[side],
            f"{side} signal ask",
        )
        for side in ("UP", "DOWN")
    }
    if raw_asks != signal_asks:
        raise MicroLiveExecutionError("candidate signal ask does not match feature row")
    result = runtime.score_feature_row(
        features,
        observed_at_ts=int(signal["observed_at_ts_ms"]),
    )
    supplied_projection = {
        "action_values": signal["action_values"],
        "selected_action": signal["selected_action"],
        "model_scored": signal["model_scored"],
        "fail_closed": signal["fail_closed"],
        "fail_closed_reasons": signal["fail_closed_reasons"],
    }
    expected_projection = {
        name: result[name]
        for name in (
            "action_values",
            "selected_action",
            "model_scored",
            "fail_closed",
            "fail_closed_reasons",
        )
    }
    if not (
        canonical_json_sha256(supplied_projection)
        == canonical_json_sha256(expected_projection)
        and result.get("lineage_id") == LINEAGE_ID
        and result.get("candidate_id") == CANDIDATE_ID
        and result.get("market_id") == signal["market_id"]
        and result.get("decision_ts") == signal["decision_ts_ms"]
        and result.get("observed_at_ts") == signal["observed_at_ts_ms"]
        and result.get("manifest_sha256") == signal["candidate_bundle_sha256"]
        and result.get("outcomes_accessed") is False
        and result.get("settlement_accessed") is False
        and result.get("pnl_accessed") is False
    ):
        raise MicroLiveExecutionError("candidate signal does not match frozen runtime")
    return features


def _reject_forbidden_feature_keys(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MicroLiveExecutionError("candidate feature key is not a string")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_FEATURE_KEY_TOKENS):
                raise MicroLiveExecutionError(
                    f"candidate feature row contains forbidden field: {path}{key}"
                )
            _reject_forbidden_feature_keys(nested, f"{path}{key}.")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_feature_keys(nested, f"{path}{index}.")


def _numeric_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MicroLiveExecutionError(f"{label} is invalid")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0 or parsed >= 1:
        raise MicroLiveExecutionError(f"{label} is outside the binary range")
    return parsed


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
