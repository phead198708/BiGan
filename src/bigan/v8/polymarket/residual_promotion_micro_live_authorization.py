"""Fail-closed verifier for a future explicitly approved 1% micro-live record.

This module never creates an authorization and never talks to an exchange or
wallet.  The currently committed template is intentionally rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.residual_promotion_release_readiness_v7 import (
    ASSESSMENT_SCHEMA_VERSION,
    CONTRACT_REPOSITORY_PATH,
    assess_micro_live_preapproval_v7,
    validate_release_readiness_contract_v7,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    ResidualPromotionError,
    ResidualPromotionRuntime,
    load_residual_promotion_runtime,
)

AUTHORIZATION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-explicit-micro-live-authorization-v2"
)
HUMAN_ATTESTATION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-human-micro-live-attestation-v2"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_micro_live_authorization.py"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
CANDIDATE_BUNDLE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/candidate_bundle/bundle_manifest.json"
)
AUTHORIZATION_TEMPLATE_REPOSITORY_PATH = (
    f"{CONFIG_REPOSITORY_PATH}/micro_live_authorization_template_v7.json"
)
MAXIMUM_INITIAL_CAPITAL_FRACTION = Decimal("0.01")
MAXIMUM_AUTHORIZATION_DURATION_MS = 86_400_000
MAXIMUM_SIGNAL_AGE_MS = 5_000
MAXIMUM_OPERATOR_HEARTBEAT_AGE_MS = 5_000
MARKET_ALLOWLIST = ("BTC-15M",)
ALLOWED_ACTIONS = ("BUY_UP_HOLD", "BUY_DOWN_HOLD")
ISSUE_NUMBER = 264
TRUSTED_APPROVER_LOGINS = ("phead198708",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMENT_URL = re.compile(
    r"^https://github\.com/phead198708/BiGan/issues/264"
    r"#issuecomment-[1-9][0-9]*$"
)
_VERIFICATION_SEAL = object()
_EVIDENCE_NAME_MAP = {
    "fresh_evaluation_manifest": "evaluation_manifest",
    "phase6_release_manifest": "phase6_report",
    "phase6_zero_capital_authorization": "phase6_authorization",
    "operational_rollback_report": "operational_rollback",
    "independent_security_review_report": "security_review",
}


class MicroLiveAuthorizationError(ValueError):
    """Raised when a future micro-live authorization is incomplete or forged."""


@dataclass(frozen=True, slots=True, weakref_slot=True)
class VerifiedMicroLiveAuthorization:
    """Capability returned only after the full authorization graph validates."""

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
    _capability_sha256: str
    _seal: object


_VERIFIED_CAPABILITIES: dict[
    int,
    tuple[weakref.ReferenceType[VerifiedMicroLiveAuthorization], str],
] = {}


def verify_micro_live_authorization(
    authorization: Mapping[str, Any],
    *,
    repository_root: Path | str,
    evidence_root: Path | str,
    now_ts_ms: int,
) -> VerifiedMicroLiveAuthorization:
    """Verify the exact evidence graph and explicit human 1% approval."""

    root = Path(repository_root).resolve()
    evidence_base = Path(evidence_root).resolve()
    if isinstance(now_ts_ms, bool) or not isinstance(now_ts_ms, int) or now_ts_ms <= 0:
        raise MicroLiveAuthorizationError("authorization verification time is invalid")
    expected_keys = {
        "schema_version",
        "lineage_id",
        "candidate_id",
        "created_at",
        "authorization_id",
        "supersedes_template",
        "candidate_bundle",
        "preapproval_contract",
        "required_evidence",
        "evidence_payload_sha256",
        "human_approval",
        "capital_base_usd",
        "requested_initial_capital_fraction",
        "maximum_notional_usd",
        "maximum_realized_loss_usd",
        "maximum_open_orders",
        "market_allowlist",
        "allowed_actions",
        "one_trade_maximum_per_market",
        "authorized_at_ts_ms",
        "expires_at_ts_ms",
        "maximum_signal_age_ms",
        "maximum_operator_heartbeat_age_ms",
        "explicit_human_approval_recorded",
        "micro_live_authorized",
        "micro_live_started",
        "live_trading_allowed",
        "wallet_signing_allowed",
        "polymarket_write_allowed",
        "capital_at_risk",
        "automatic_launch_allowed",
        "capital_increase_allowed",
        "executable",
    }
    if set(authorization) != expected_keys:
        raise MicroLiveAuthorizationError("micro-live authorization schema is not exact")
    if not (
        authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
        and authorization.get("lineage_id") == LINEAGE_ID
        and authorization.get("candidate_id") == CANDIDATE_ID
    ):
        raise MicroLiveAuthorizationError("micro-live authorization identity is invalid")

    template = _verified_repository_json(root, AUTHORIZATION_TEMPLATE_REPOSITORY_PATH)
    contract = _verified_repository_json(root, CONTRACT_REPOSITORY_PATH)
    validate_release_readiness_contract_v7(contract, repository_root=root)
    _verify_repository_descriptor(
        root,
        dict(authorization.get("supersedes_template") or {}),
        expected_path=AUTHORIZATION_TEMPLATE_REPOSITORY_PATH,
    )
    _verify_repository_descriptor(
        root,
        dict(authorization.get("candidate_bundle") or {}),
        expected_path=CANDIDATE_BUNDLE_REPOSITORY_PATH,
    )
    _verify_repository_descriptor(
        root,
        dict(authorization.get("preapproval_contract") or {}),
        expected_path=CONTRACT_REPOSITORY_PATH,
    )
    candidate_sha = dict(authorization["candidate_bundle"])["sha256"]
    if not (
        dict(contract.get("candidate_bundle") or {})
        == dict(authorization["candidate_bundle"])
        and contract.get("lineage_id") == LINEAGE_ID
        and contract.get("candidate_id") == CANDIDATE_ID
        and dict(template.get("preapproval_contract") or {})
        == dict(authorization["preapproval_contract"])
        and template.get("candidate_id") == CANDIDATE_ID
        and template.get("micro_live_authorized") is False
        and template.get("executable") is False
    ):
        raise MicroLiveAuthorizationError("micro-live template binding is invalid")
    try:
        runtime = load_residual_promotion_runtime(
            manifest_path=CANDIDATE_BUNDLE_REPOSITORY_PATH,
            expected_manifest_sha256=str(candidate_sha),
            repository_root=root,
        )
    except ResidualPromotionError as exc:
        raise MicroLiveAuthorizationError(
            "micro-live frozen runtime failed to load"
        ) from exc
    if not (
        runtime.lineage_id == LINEAGE_ID
        and runtime.candidate_id == CANDIDATE_ID
        and runtime.manifest_sha256 == candidate_sha
    ):
        raise MicroLiveAuthorizationError("micro-live frozen runtime binding is invalid")

    capital_base = _positive_decimal(authorization.get("capital_base_usd"), "capital base")
    fraction = _positive_decimal(
        authorization.get("requested_initial_capital_fraction"),
        "initial capital fraction",
    )
    maximum_notional = _positive_decimal(
        authorization.get("maximum_notional_usd"), "maximum notional"
    )
    maximum_realized_loss = _positive_decimal(
        authorization.get("maximum_realized_loss_usd"), "maximum realized loss"
    )
    maximum_open_orders = authorization.get("maximum_open_orders")
    authorized_at = authorization.get("authorized_at_ts_ms")
    expires_at = authorization.get("expires_at_ts_ms")
    if not (
        fraction == MAXIMUM_INITIAL_CAPITAL_FRACTION
        and maximum_notional == capital_base * fraction
        and maximum_realized_loss <= maximum_notional
        and isinstance(maximum_open_orders, int)
        and not isinstance(maximum_open_orders, bool)
        and 1 <= maximum_open_orders <= 10
        and isinstance(authorized_at, int)
        and not isinstance(authorized_at, bool)
        and isinstance(expires_at, int)
        and not isinstance(expires_at, bool)
        and authorized_at > 0
        and authorized_at <= now_ts_ms < expires_at
        and expires_at - authorized_at <= MAXIMUM_AUTHORIZATION_DURATION_MS
        and authorization.get("maximum_signal_age_ms") == MAXIMUM_SIGNAL_AGE_MS
        and authorization.get("maximum_operator_heartbeat_age_ms")
        == MAXIMUM_OPERATOR_HEARTBEAT_AGE_MS
        and authorization.get("market_allowlist") == list(MARKET_ALLOWLIST)
        and authorization.get("allowed_actions") == list(ALLOWED_ACTIONS)
        and authorization.get("one_trade_maximum_per_market") is True
        and _parse_utc_ts_ms(authorization.get("created_at")) == authorized_at
    ):
        raise MicroLiveAuthorizationError("micro-live limits or validity window are invalid")

    required = _load_and_reconcile_evidence(
        authorization=authorization,
        contract=contract,
        evidence_base=evidence_base,
        repository_root=root,
    )
    evidence_payload = {
        name: descriptor["sha256"] for name, descriptor in sorted(required.items())
    }
    evidence_payload_sha = canonical_json_sha256(evidence_payload)
    if authorization.get("evidence_payload_sha256") != evidence_payload_sha:
        raise MicroLiveAuthorizationError("micro-live evidence payload SHA-256 mismatch")

    identity = {
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_bundle_sha256": candidate_sha,
        "evidence_payload_sha256": evidence_payload_sha,
        "capital_base_usd": str(capital_base),
        "requested_initial_capital_fraction": str(fraction),
        "maximum_notional_usd": str(maximum_notional),
        "maximum_realized_loss_usd": str(maximum_realized_loss),
        "maximum_open_orders": maximum_open_orders,
        "market_allowlist": list(MARKET_ALLOWLIST),
        "allowed_actions": list(ALLOWED_ACTIONS),
        "authorized_at_ts_ms": authorized_at,
        "expires_at_ts_ms": expires_at,
        "maximum_signal_age_ms": MAXIMUM_SIGNAL_AGE_MS,
        "maximum_operator_heartbeat_age_ms": MAXIMUM_OPERATOR_HEARTBEAT_AGE_MS,
        "approval_issue_number": ISSUE_NUMBER,
    }
    authorization_id = canonical_json_sha256(identity)
    if authorization.get("authorization_id") != authorization_id:
        raise MicroLiveAuthorizationError("micro-live authorization identity SHA-256 mismatch")
    _validate_human_approval(
        dict(authorization.get("human_approval") or {}),
        evidence_base=evidence_base,
        authorization_id=authorization_id,
        authorized_at_ts_ms=int(authorized_at),
        expires_at_ts_ms=int(expires_at),
        capital_base_usd=capital_base,
        maximum_notional_usd=maximum_notional,
        maximum_realized_loss_usd=maximum_realized_loss,
        maximum_open_orders=int(maximum_open_orders),
    )
    if not (
        authorization.get("explicit_human_approval_recorded") is True
        and authorization.get("micro_live_authorized") is True
        and authorization.get("micro_live_started") is False
        and authorization.get("live_trading_allowed") is True
        and authorization.get("wallet_signing_allowed") is True
        and authorization.get("polymarket_write_allowed") is True
        and authorization.get("capital_at_risk") is True
        and authorization.get("automatic_launch_allowed") is False
        and authorization.get("capital_increase_allowed") is False
        and authorization.get("executable") is True
    ):
        raise MicroLiveAuthorizationError("micro-live authorization state is not explicit")

    capability = VerifiedMicroLiveAuthorization(
        authorization_id=authorization_id,
        authorization_payload_sha256=canonical_json_sha256(dict(authorization)),
        candidate_bundle_sha256=str(candidate_sha),
        capital_base_usd=capital_base,
        maximum_notional_usd=maximum_notional,
        maximum_realized_loss_usd=maximum_realized_loss,
        maximum_open_orders=int(maximum_open_orders),
        authorized_at_ts_ms=int(authorized_at),
        expires_at_ts_ms=int(expires_at),
        maximum_signal_age_ms=MAXIMUM_SIGNAL_AGE_MS,
        maximum_operator_heartbeat_age_ms=MAXIMUM_OPERATOR_HEARTBEAT_AGE_MS,
        market_allowlist=MARKET_ALLOWLIST,
        allowed_actions=ALLOWED_ACTIONS,
        runtime=runtime,
        _capability_sha256="",
        _seal=_VERIFICATION_SEAL,
    )
    capability_sha256 = _capability_integrity_sha256(capability)
    object.__setattr__(capability, "_capability_sha256", capability_sha256)
    _register_verified_capability(capability, capability_sha256)
    return capability


def authorization_capability_is_verified(value: VerifiedMicroLiveAuthorization) -> bool:
    """Return whether a capability came from this verifier."""

    if not (
        isinstance(value, VerifiedMicroLiveAuthorization)
        and value._seal is _VERIFICATION_SEAL
    ):
        return False
    registered = _VERIFIED_CAPABILITIES.get(id(value))
    if registered is None or registered[0]() is not value:
        return False
    try:
        actual_sha256 = _capability_integrity_sha256(value)
    except Exception:
        return False
    return (
        value._capability_sha256 == registered[1]
        and actual_sha256 == registered[1]
    )


def _register_verified_capability(
    capability: VerifiedMicroLiveAuthorization,
    capability_sha256: str,
) -> None:
    capability_id = id(capability)

    def discard(reference: weakref.ReferenceType[VerifiedMicroLiveAuthorization]) -> None:
        registered = _VERIFIED_CAPABILITIES.get(capability_id)
        if registered is not None and registered[0] is reference:
            _VERIFIED_CAPABILITIES.pop(capability_id, None)

    reference = weakref.ref(capability, discard)
    _VERIFIED_CAPABILITIES[capability_id] = (reference, capability_sha256)


def _capability_integrity_sha256(
    capability: VerifiedMicroLiveAuthorization,
) -> str:
    runtime = capability.runtime
    residual_model_bytes = bytes(runtime.residual_booster.save_raw(raw_format="ubj"))
    logit_model_bytes = bytes(runtime.logit_booster.save_raw(raw_format="ubj"))
    loaded_residual_model_sha256 = hashlib.sha256(residual_model_bytes).hexdigest()
    loaded_logit_model_sha256 = hashlib.sha256(logit_model_bytes).hexdigest()
    if not (
        loaded_residual_model_sha256 == runtime.residual_model_sha256
        and loaded_logit_model_sha256 == runtime.logit_model_sha256
    ):
        raise ValueError("loaded micro-live model bytes do not match the frozen runtime")
    payload = {
        "schema_version": "verified-micro-live-authorization-capability-v1",
        "authorization_id": capability.authorization_id,
        "authorization_payload_sha256": capability.authorization_payload_sha256,
        "candidate_bundle_sha256": capability.candidate_bundle_sha256,
        "capital_base_usd": str(capability.capital_base_usd),
        "maximum_notional_usd": str(capability.maximum_notional_usd),
        "maximum_realized_loss_usd": str(capability.maximum_realized_loss_usd),
        "maximum_open_orders": capability.maximum_open_orders,
        "authorized_at_ts_ms": capability.authorized_at_ts_ms,
        "expires_at_ts_ms": capability.expires_at_ts_ms,
        "maximum_signal_age_ms": capability.maximum_signal_age_ms,
        "maximum_operator_heartbeat_age_ms": (
            capability.maximum_operator_heartbeat_age_ms
        ),
        "market_allowlist": list(capability.market_allowlist),
        "allowed_actions": list(capability.allowed_actions),
        "runtime_object_id": id(runtime),
        "runtime": {
            "candidate_id": runtime.candidate_id,
            "lineage_id": runtime.lineage_id,
            "manifest_sha256": runtime.manifest_sha256,
            "residual_model_sha256": runtime.residual_model_sha256,
            "logit_model_sha256": runtime.logit_model_sha256,
            "adapter_sha256": runtime.adapter_sha256,
            "maximum_decision_lag_ms": runtime.maximum_decision_lag_ms,
            "maximum_source_age_ms": runtime.maximum_source_age_ms,
            "coefficients": list(runtime.coefficients),
            "loaded_residual_model_sha256": loaded_residual_model_sha256,
            "loaded_logit_model_sha256": loaded_logit_model_sha256,
        },
    }
    return canonical_json_sha256(payload)


def _load_and_reconcile_evidence(
    *,
    authorization: Mapping[str, Any],
    contract: Mapping[str, Any],
    evidence_base: Path,
    repository_root: Path,
) -> dict[str, dict[str, str]]:
    required_value = authorization.get("required_evidence")
    if not isinstance(required_value, Mapping):
        raise MicroLiveAuthorizationError("micro-live required evidence is invalid")
    required = {name: dict(value) for name, value in required_value.items()}
    expected_names = {"preapproval_assessment", *_EVIDENCE_NAME_MAP}
    if set(required) != expected_names:
        raise MicroLiveAuthorizationError("micro-live required evidence set is invalid")
    loaded_required = {
        name: _verified_evidence_json(evidence_base, descriptor)
        for name, descriptor in required.items()
    }
    assessment = loaded_required["preapproval_assessment"]
    if assessment.get("schema_version") != ASSESSMENT_SCHEMA_VERSION:
        raise MicroLiveAuthorizationError("micro-live preapproval assessment schema is invalid")
    assessment_descriptors_value = assessment.get("evidence_file_descriptors")
    if not isinstance(assessment_descriptors_value, Mapping):
        raise MicroLiveAuthorizationError("micro-live preapproval evidence graph is absent")
    assessment_descriptors = {
        name: dict(value) for name, value in assessment_descriptors_value.items()
    }
    evidence_payloads = {
        name: _verified_evidence_json(evidence_base, descriptor)
        for name, descriptor in assessment_descriptors.items()
    }
    for authorization_name, assessment_name in _EVIDENCE_NAME_MAP.items():
        if (
            assessment_name not in assessment_descriptors
            or required[authorization_name] != assessment_descriptors[assessment_name]
            or loaded_required[authorization_name] != evidence_payloads[assessment_name]
        ):
            raise MicroLiveAuthorizationError(
                f"micro-live evidence binding mismatch: {authorization_name}"
            )
    expected = assess_micro_live_preapproval_v7(
        contract=contract,
        evidence=evidence_payloads,
        repository_root=repository_root,
        created_at=str(assessment.get("created_at") or ""),
    )
    supplied_core = {
        key: value for key, value in assessment.items() if key != "evidence_file_descriptors"
    }
    if supplied_core != expected:
        raise MicroLiveAuthorizationError("micro-live preapproval assessment does not recompute")
    if not (
        assessment.get("ready_to_request_micro_live_approval") is True
        and assessment.get("status")
        == "READY_TO_REQUEST_HUMAN_1_PERCENT_MICRO_LIVE_GO_NO_GO"
        and assessment.get("security_review_independent_and_exact_head") is True
        and assessment.get("phase6_zero_capital_pipeline_passed") is True
        and all(dict(assessment.get("technical_checks") or {}).values())
        and assessment.get("explicit_human_approval_recorded") is False
        and assessment.get("micro_live_authorized") is False
        and assessment.get("automatic_live_unlock") is False
        and assessment.get("wallet_signing_allowed") is False
        and assessment.get("polymarket_write_allowed") is False
        and assessment.get("capital_at_risk") is False
    ):
        raise MicroLiveAuthorizationError("micro-live preapproval has not passed every gate")
    return required


def _validate_human_approval(
    approval: Mapping[str, Any],
    *,
    evidence_base: Path,
    authorization_id: str,
    authorized_at_ts_ms: int,
    expires_at_ts_ms: int,
    capital_base_usd: Decimal,
    maximum_notional_usd: Decimal,
    maximum_realized_loss_usd: Decimal,
    maximum_open_orders: int,
) -> None:
    if set(approval) != {
        "github_login",
        "issue_number",
        "comment_id",
        "comment_url",
        "approved_at_ts_ms",
        "github_comment_payload",
        "attestation",
    }:
        raise MicroLiveAuthorizationError("human micro-live approval schema is invalid")
    login = approval.get("github_login")
    comment_id = approval.get("comment_id")
    comment_url = approval.get("comment_url")
    approved_at = approval.get("approved_at_ts_ms")
    if not (
        isinstance(login, str)
        and login in TRUSTED_APPROVER_LOGINS
        and approval.get("issue_number") == ISSUE_NUMBER
        and isinstance(comment_id, int)
        and not isinstance(comment_id, bool)
        and comment_id > 0
        and isinstance(comment_url, str)
        and comment_url
        == (
            f"https://github.com/phead198708/BiGan/issues/{ISSUE_NUMBER}"
            f"#issuecomment-{comment_id}"
        )
        and _COMMENT_URL.fullmatch(comment_url) is not None
        and isinstance(approved_at, int)
        and not isinstance(approved_at, bool)
        and approved_at > 0
        and approved_at == authorized_at_ts_ms
    ):
        raise MicroLiveAuthorizationError("human micro-live approval provenance is invalid")
    expected_command = (
        f"APPROVE {LINEAGE_ID} MICRO-LIVE authorization_id={authorization_id} "
        f"capital_base_usd={capital_base_usd} "
        f"maximum_notional_usd={maximum_notional_usd} "
        f"maximum_realized_loss_usd={maximum_realized_loss_usd} "
        f"maximum_open_orders={maximum_open_orders} capital_fraction=0.01 "
        f"expires_at_ts_ms={expires_at_ts_ms}"
    )
    github = _verified_evidence_json(
        evidence_base, dict(approval.get("github_comment_payload") or {})
    )
    user = dict(github.get("user") or {})
    if not (
        set(github) == {"id", "html_url", "created_at", "body", "user"}
        and set(user) == {"login"}
        and github.get("id") == comment_id
        and github.get("html_url") == comment_url
        and user.get("login") == login
        and github.get("body") == expected_command
        and isinstance(github.get("created_at"), str)
        and _parse_utc_ts_ms(github["created_at"]) == approved_at
    ):
        raise MicroLiveAuthorizationError("GitHub human approval payload is invalid")
    attestation = _verified_evidence_json(
        evidence_base, dict(approval.get("attestation") or {})
    )
    if not (
        set(attestation)
        == {
            "schema_version",
            "github_login",
            "issue_number",
            "comment_id",
            "comment_url",
            "authorization_id",
            "approved_at_ts_ms",
            "attestation_statement",
        }
        and attestation.get("schema_version") == HUMAN_ATTESTATION_SCHEMA_VERSION
        and attestation.get("github_login") == login
        and attestation.get("issue_number") == ISSUE_NUMBER
        and attestation.get("comment_id") == comment_id
        and attestation.get("comment_url") == comment_url
        and attestation.get("authorization_id") == authorization_id
        and attestation.get("approved_at_ts_ms") == approved_at
        and attestation.get("attestation_statement") == expected_command
    ):
        raise MicroLiveAuthorizationError("human micro-live approval attestation is invalid")


def _verified_repository_json(root: Path, repository_path: str) -> dict[str, Any]:
    path = (root / repository_path).resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise MicroLiveAuthorizationError("frozen repository artifact sidecar mismatch")
    return _load_json(path)


def _verify_repository_descriptor(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    expected_path: str,
) -> None:
    if set(descriptor) != {"path", "sha256"} or descriptor.get("path") != expected_path:
        raise MicroLiveAuthorizationError("repository authorization descriptor is invalid")
    path = (root / expected_path).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or not _is_sha256(descriptor.get("sha256"))
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise MicroLiveAuthorizationError("repository authorization descriptor mismatch")


def _verified_evidence_json(
    evidence_base: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    if set(descriptor) != {"path", "sha256"} or not _is_sha256(descriptor.get("sha256")):
        raise MicroLiveAuthorizationError("micro-live evidence descriptor is invalid")
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise MicroLiveAuthorizationError("micro-live evidence path is invalid")
    path = (evidence_base / path_value).resolve()
    if (
        not path.is_relative_to(evidence_base)
        or not path.is_file()
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise MicroLiveAuthorizationError("micro-live evidence path or SHA-256 mismatch")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
        _validate_finite_json_tree(value)
    except (OSError, ValueError) as exc:
        raise MicroLiveAuthorizationError("micro-live JSON evidence is invalid") from exc
    if not isinstance(value, dict):
        raise MicroLiveAuthorizationError("micro-live JSON evidence root must be an object")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_finite_json_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("decoded JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_json_tree(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json_tree(item)


def _positive_decimal(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise MicroLiveAuthorizationError(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise MicroLiveAuthorizationError(f"{label} is invalid") from exc
    if not parsed.is_finite() or parsed <= 0 or str(parsed) != value:
        raise MicroLiveAuthorizationError(f"{label} is not positive canonical decimal")
    return parsed


def _parse_utc_ts_ms(value: Any) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MicroLiveAuthorizationError("micro-live timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise MicroLiveAuthorizationError("micro-live timestamp is invalid") from exc
    if parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != value:
        raise MicroLiveAuthorizationError("micro-live timestamp is not canonical")
    return int(parsed.timestamp() * 1_000)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "ALLOWED_ACTIONS",
    "AUTHORIZATION_SCHEMA_VERSION",
    "HUMAN_ATTESTATION_SCHEMA_VERSION",
    "IMPLEMENTATION_REPOSITORY_PATH",
    "MARKET_ALLOWLIST",
    "MAXIMUM_INITIAL_CAPITAL_FRACTION",
    "MAXIMUM_OPERATOR_HEARTBEAT_AGE_MS",
    "MAXIMUM_SIGNAL_AGE_MS",
    "MicroLiveAuthorizationError",
    "TRUSTED_APPROVER_LOGINS",
    "VerifiedMicroLiveAuthorization",
    "authorization_capability_is_verified",
    "verify_micro_live_authorization",
]
