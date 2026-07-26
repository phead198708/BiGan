"""Versioned semantic serialization for hash-bound v8 decision artifacts.

This module deliberately does not replace legacy file hashes.  It adds a
separate canonical semantic hash that can be compared across runtimes without
depending on JSON whitespace, object insertion order, or equivalent numeric
spelling.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

CANONICAL_PAYLOAD_CONTRACT_VERSION = "bigan-v8-canonical-payload-v1"
DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION = "bigan-v8-decision-feature-payload-v1"
MAX_TIMESTAMP_MS = 253_402_300_799_999
MIN_SIGNED_64 = -(2**63)
MAX_SIGNED_64 = 2**63 - 1
TIMESTAMP_FIELD_SUFFIXES = ("_ts", "_ts_ms", "_timestamp_ms")


class CanonicalPayloadError(ValueError):
    """Raised when a payload cannot satisfy the canonical contract."""


@dataclass(frozen=True, slots=True)
class CanonicalHashDescriptor:
    """Raw audit bytes plus the independent canonical semantic descriptor."""

    canonical_contract_version: str
    payload_schema_version: str
    raw_payload_sha256: str
    raw_payload_byte_length: int
    raw_payload_base64: str
    raw_payload_origin: str
    canonical_payload_sha256: str
    canonical_payload_byte_length: int
    canonical_payload_utf8_base64: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalPayloadComparison:
    """Fail-closed equality result for a frozen and settled payload."""

    canonical_contract_version: str
    frozen_payload_schema_version: str
    settled_payload_schema_version: str
    frozen_descriptor: CanonicalHashDescriptor | None
    settled_descriptor: CanonicalHashDescriptor | None
    canonical_hash_match: bool
    approved_source_lineage: bool
    settlement_evaluation_eligible: bool
    reason_codes: tuple[str, ...]
    semantic_diff: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_contract_version": self.canonical_contract_version,
            "frozen_payload_schema_version": self.frozen_payload_schema_version,
            "settled_payload_schema_version": self.settled_payload_schema_version,
            "frozen_descriptor": (
                self.frozen_descriptor.to_dict() if self.frozen_descriptor else None
            ),
            "settled_descriptor": (
                self.settled_descriptor.to_dict() if self.settled_descriptor else None
            ),
            "canonical_hash_match": self.canonical_hash_match,
            "approved_source_lineage": self.approved_source_lineage,
            "settlement_evaluation_eligible": self.settlement_evaluation_eligible,
            "reason_codes": list(self.reason_codes),
            "semantic_diff": list(self.semantic_diff),
            "paper_candidate_unlocked": False,
            "promotion_unlocked": False,
            "live_unlocked": False,
            "write_enabled": False,
            "wallet_enabled": False,
            "capital_at_risk": False,
        }


def canonical_payload_bytes(
    payload: Any,
    *,
    payload_schema_version: str,
) -> bytes:
    """Return the contract envelope encoded as canonical UTF-8 JSON."""

    schema_version = _normalize_schema_version(payload_schema_version)
    _validate_embedded_schema_version(payload, expected=schema_version)
    envelope = {
        "canonical_contract_version": CANONICAL_PAYLOAD_CONTRACT_VERSION,
        "payload": payload,
        "payload_schema_version": schema_version,
    }
    return _encode_value(envelope, path="").encode("utf-8")


def canonical_payload_sha256(
    payload: Any,
    *,
    payload_schema_version: str,
) -> str:
    """Calculate SHA-256 over the canonical UTF-8 contract envelope."""

    return hashlib.sha256(
        canonical_payload_bytes(payload, payload_schema_version=payload_schema_version)
    ).hexdigest()


def describe_canonical_payload(
    payload: Any,
    *,
    payload_schema_version: str,
    raw_payload_bytes: bytes | None = None,
) -> CanonicalHashDescriptor:
    """Build separate raw and canonical descriptors without rewriting input bytes."""

    if raw_payload_bytes is not None and not isinstance(raw_payload_bytes, bytes):
        raise TypeError("raw_payload_bytes must be bytes when supplied")
    raw_origin = "caller_preserved_bytes"
    raw_bytes = raw_payload_bytes
    if raw_bytes is None:
        raw_origin = "generated_audit_serialization"
        try:
            raw_bytes = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CanonicalPayloadError(f"raw payload is not valid JSON: {exc}") from exc
    canonical_bytes = canonical_payload_bytes(
        payload,
        payload_schema_version=payload_schema_version,
    )
    return CanonicalHashDescriptor(
        canonical_contract_version=CANONICAL_PAYLOAD_CONTRACT_VERSION,
        payload_schema_version=_normalize_schema_version(payload_schema_version),
        raw_payload_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_payload_byte_length=len(raw_bytes),
        raw_payload_base64=base64.b64encode(raw_bytes).decode("ascii"),
        raw_payload_origin=raw_origin,
        canonical_payload_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        canonical_payload_byte_length=len(canonical_bytes),
        canonical_payload_utf8_base64=base64.b64encode(canonical_bytes).decode("ascii"),
    )


def compare_canonical_payloads(
    frozen_payload: Any,
    settled_payload: Any,
    *,
    frozen_payload_schema_version: str,
    settled_payload_schema_version: str,
    approved_source_lineage: bool,
    frozen_raw_payload_bytes: bytes | None = None,
    settled_raw_payload_bytes: bytes | None = None,
) -> CanonicalPayloadComparison:
    """Compare independently canonicalized payloads and fail closed on any defect."""

    reasons: list[str] = []
    diffs: list[dict[str, Any]] = []
    frozen_descriptor: CanonicalHashDescriptor | None = None
    settled_descriptor: CanonicalHashDescriptor | None = None
    frozen_schema = _normalize_schema_version(frozen_payload_schema_version)
    settled_schema = _normalize_schema_version(settled_payload_schema_version)
    if frozen_schema != settled_schema:
        reasons.append("canonical_payload_schema_version_mismatch")
        diffs.append(
            {
                "path": "/$payload_schema_version",
                "kind": "value_mismatch",
                "frozen": frozen_schema,
                "settled": settled_schema,
            }
        )
    if approved_source_lineage is not True:
        reasons.append("canonical_payload_source_lineage_not_approved")
    try:
        frozen_descriptor = describe_canonical_payload(
            frozen_payload,
            payload_schema_version=frozen_schema,
            raw_payload_bytes=frozen_raw_payload_bytes,
        )
    except (CanonicalPayloadError, TypeError) as exc:
        reasons.append("frozen_payload_canonicalization_failed")
        diffs.append(
            {
                "path": "",
                "kind": "canonicalization_error",
                "side": "frozen",
                "error": str(exc),
            }
        )
    try:
        settled_descriptor = describe_canonical_payload(
            settled_payload,
            payload_schema_version=settled_schema,
            raw_payload_bytes=settled_raw_payload_bytes,
        )
    except (CanonicalPayloadError, TypeError) as exc:
        reasons.append("settled_payload_canonicalization_failed")
        diffs.append(
            {
                "path": "",
                "kind": "canonicalization_error",
                "side": "settled",
                "error": str(exc),
            }
        )
    hashes_match = bool(
        frozen_descriptor
        and settled_descriptor
        and frozen_descriptor.canonical_payload_sha256
        == settled_descriptor.canonical_payload_sha256
    )
    if frozen_descriptor and settled_descriptor and not hashes_match:
        reasons.append("canonical_payload_hash_mismatch")
        diffs.extend(_semantic_diff(frozen_payload, settled_payload))
    eligible = (
        not reasons
        and hashes_match
        and approved_source_lineage is True
        and frozen_schema == settled_schema
    )
    return CanonicalPayloadComparison(
        canonical_contract_version=CANONICAL_PAYLOAD_CONTRACT_VERSION,
        frozen_payload_schema_version=frozen_schema,
        settled_payload_schema_version=settled_schema,
        frozen_descriptor=frozen_descriptor,
        settled_descriptor=settled_descriptor,
        canonical_hash_match=hashes_match,
        approved_source_lineage=approved_source_lineage is True,
        settlement_evaluation_eligible=eligible,
        reason_codes=tuple(sorted(set(reasons))),
        semantic_diff=tuple(
            sorted(diffs, key=lambda item: (str(item.get("path")), str(item.get("kind"))))
        ),
    )


def _encode_value(value: Any, *, path: str, field_name: str | None = None) -> str:
    if _is_timestamp_field(field_name):
        _validate_timestamp(value, path=path)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value)
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        if value < MIN_SIGNED_64 or value > MAX_SIGNED_64:
            raise CanonicalPayloadError(f"{path or '/'}: integer is outside signed 64-bit range")
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalPayloadError(f"{path or '/'}: NaN and infinity are forbidden")
        return _canonical_number(value)
    if isinstance(value, list):
        return "[" + ",".join(
            _encode_value(item, path=f"{path}/{index}") for index, item in enumerate(value)
        ) + "]"
    if isinstance(value, dict):
        normalized_items: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise CanonicalPayloadError(f"{path or '/'}: object keys must be strings")
            key = unicodedata.normalize("NFC", raw_key)
            if key in seen:
                raise CanonicalPayloadError(
                    f"{path or '/'}: object keys collide after NFC normalization"
                )
            seen.add(key)
            normalized_items.append((key, item))
        normalized_items.sort(key=lambda pair: pair[0].encode("utf-8"))
        parts = []
        for key, item in normalized_items:
            encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            child_path = f"{path}/{_pointer_escape(key)}"
            parts.append(
                encoded_key
                + ":"
                + _encode_value(item, path=child_path, field_name=key)
            )
        return "{" + ",".join(parts) + "}"
    raise CanonicalPayloadError(
        f"{path or '/'}: unsupported type {type(value).__name__}; "
        "allowed types are null, boolean, string, signed-64 integer, finite float, list, object"
    )


def _canonical_number(value: float) -> str:
    if value == 0:
        return "0"
    decimal_value = Decimal(repr(value))
    rendered = format(decimal_value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _normalize_schema_version(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalPayloadError("payload_schema_version must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise CanonicalPayloadError("payload_schema_version must already be NFC normalized")
    return normalized


def _validate_embedded_schema_version(payload: Any, *, expected: str) -> None:
    if isinstance(payload, dict) and "schema_version" in payload:
        actual = payload["schema_version"]
        if not isinstance(actual, str) or actual != expected:
            raise CanonicalPayloadError(
                "embedded schema_version does not match payload_schema_version"
            )


def _is_timestamp_field(field_name: str | None) -> bool:
    return bool(
        field_name
        and (
            field_name == "timestamp"
            or any(field_name.endswith(suffix) for suffix in TIMESTAMP_FIELD_SUFFIXES)
        )
    )


def _validate_timestamp(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalPayloadError(
            f"{path or '/'}: timestamps must be UTC Unix epoch milliseconds as integers"
        )
    if value < 0 or value > MAX_TIMESTAMP_MS:
        raise CanonicalPayloadError(f"{path or '/'}: timestamp is outside the supported UTC range")


def _semantic_diff(frozen: Any, settled: Any, *, path: str = "") -> list[dict[str, Any]]:
    try:
        frozen_node = _semantic_node(frozen, path=path)
        settled_node = _semantic_node(settled, path=path)
    except CanonicalPayloadError:
        return []
    return _diff_nodes(frozen_node, settled_node, path=path)


def _semantic_node(value: Any, *, path: str, field_name: str | None = None) -> Any:
    if _is_timestamp_field(field_name):
        _validate_timestamp(value, path=path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return ("string", unicodedata.normalize("NFC", value))
    if isinstance(value, int):
        if value < MIN_SIGNED_64 or value > MAX_SIGNED_64:
            raise CanonicalPayloadError("integer outside signed 64-bit range")
        return ("number", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalPayloadError("non-finite number")
        return ("number", _canonical_number(value))
    if isinstance(value, list):
        return [
            _semantic_node(item, path=f"{path}/{index}") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise CanonicalPayloadError("non-string object key")
            key = unicodedata.normalize("NFC", raw_key)
            if key in result:
                raise CanonicalPayloadError("normalized object key collision")
            result[key] = _semantic_node(
                item,
                path=f"{path}/{_pointer_escape(key)}",
                field_name=key,
            )
        return result
    raise CanonicalPayloadError("unsupported type")


def _diff_nodes(frozen: Any, settled: Any, *, path: str) -> list[dict[str, Any]]:
    if isinstance(frozen, dict) and isinstance(settled, dict):
        output: list[dict[str, Any]] = []
        for key in sorted(set(frozen) | set(settled), key=lambda item: item.encode("utf-8")):
            child_path = f"{path}/{_pointer_escape(key)}"
            if key not in frozen:
                output.append(
                    {
                        "path": child_path,
                        "kind": "missing_from_frozen",
                        "settled": _display_node(settled[key]),
                    }
                )
            elif key not in settled:
                output.append(
                    {
                        "path": child_path,
                        "kind": "missing_from_settled",
                        "frozen": _display_node(frozen[key]),
                    }
                )
            else:
                output.extend(_diff_nodes(frozen[key], settled[key], path=child_path))
        return output
    if isinstance(frozen, list) and isinstance(settled, list):
        output = []
        common = min(len(frozen), len(settled))
        for index in range(common):
            output.extend(
                _diff_nodes(frozen[index], settled[index], path=f"{path}/{index}")
            )
        if len(frozen) != len(settled):
            output.append(
                {
                    "path": path,
                    "kind": "list_length_mismatch",
                    "frozen": len(frozen),
                    "settled": len(settled),
                }
            )
        return output
    if type(frozen) is not type(settled):
        return [
            {
                "path": path,
                "kind": "type_mismatch",
                "frozen": _display_node(frozen),
                "settled": _display_node(settled),
            }
        ]
    if frozen != settled:
        return [
            {
                "path": path,
                "kind": "value_mismatch",
                "frozen": _display_node(frozen),
                "settled": _display_node(settled),
            }
        ]
    return []


def _display_node(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        return value[1]
    return value


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


__all__ = [
    "CANONICAL_PAYLOAD_CONTRACT_VERSION",
    "DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION",
    "CanonicalHashDescriptor",
    "CanonicalPayloadComparison",
    "CanonicalPayloadError",
    "canonical_payload_bytes",
    "canonical_payload_sha256",
    "compare_canonical_payloads",
    "describe_canonical_payload",
]
