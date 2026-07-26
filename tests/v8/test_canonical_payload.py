from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.canonical_payload import (
    CANONICAL_PAYLOAD_CONTRACT_VERSION,
    DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
    CanonicalPayloadError,
    canonical_payload_bytes,
    canonical_payload_sha256,
    compare_canonical_payloads,
    describe_canonical_payload,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "examples/v8/polymarket_configs/canonical_payload_contract.json"
)
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(".sha256")
FIXTURE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/canonical_payload_cross_runtime_fixtures.json"
)


def test_contract_artifact_is_hash_pinned_and_safety_closed() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = CONTRACT_SHA_PATH.read_text(encoding="ascii").strip()
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == expected
    assert contract["canonical_contract_version"] == CANONICAL_PAYLOAD_CONTRACT_VERSION
    assert all(value is False for value in contract["safety"].values())


def test_key_order_whitespace_equivalent_numbers_unicode_and_negative_zero_match() -> None:
    frozen_raw = b'{"b":"e\\u0301","a":1,"zero":-0.0}'
    settled_raw = '{"zero":0,"a":1.0,"b":"é"}'.encode()
    frozen = json.loads(frozen_raw)
    settled = json.loads(settled_raw)
    comparison = compare_canonical_payloads(
        frozen,
        settled,
        frozen_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        settled_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        approved_source_lineage=True,
        frozen_raw_payload_bytes=frozen_raw,
        settled_raw_payload_bytes=settled_raw,
    )
    assert comparison.settlement_evaluation_eligible is True
    assert comparison.canonical_hash_match is True
    assert comparison.frozen_descriptor is not None
    assert comparison.settled_descriptor is not None
    assert (
        comparison.frozen_descriptor.raw_payload_sha256
        != comparison.settled_descriptor.raw_payload_sha256
    )
    assert (
        comparison.frozen_descriptor.canonical_payload_sha256
        == comparison.settled_descriptor.canonical_payload_sha256
    )


@pytest.mark.parametrize(
    ("mutator", "expected_path"),
    [
        (lambda payload: payload.__setitem__("score", 0.5000001), "/score"),
        (lambda payload: payload.pop("score"), "/score"),
        (lambda payload: payload.__setitem__("nullable", None), "/nullable"),
        (lambda payload: payload.__setitem__("actions", ["DOWN", "UP"]), "/actions/0"),
    ],
)
def test_material_semantic_differences_fail_closed(mutator, expected_path: str) -> None:
    frozen = {"score": 0.5, "actions": ["UP", "DOWN"]}
    settled = {"score": 0.5, "actions": ["UP", "DOWN"]}
    mutator(settled)
    comparison = compare_canonical_payloads(
        frozen,
        settled,
        frozen_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        settled_payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        approved_source_lineage=True,
    )
    assert comparison.settlement_evaluation_eligible is False
    assert "canonical_payload_hash_mismatch" in comparison.reason_codes
    assert any(item["path"] == expected_path for item in comparison.semantic_diff)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_fail_closed(invalid: float) -> None:
    with pytest.raises(CanonicalPayloadError, match="forbidden"):
        canonical_payload_bytes(
            {"score": invalid},
            payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        )


@pytest.mark.parametrize(
    "invalid_timestamp",
    [-1, 1.5, "2026-07-26T00:00:00Z", 253_402_300_800_000],
)
def test_invalid_timestamp_representation_fails_closed(invalid_timestamp) -> None:
    with pytest.raises(CanonicalPayloadError, match="timestamp"):
        canonical_payload_bytes(
            {"decision_ts": invalid_timestamp},
            payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        )


def test_schema_version_change_and_unapproved_lineage_fail_closed() -> None:
    comparison = compare_canonical_payloads(
        {"value": 1},
        {"value": 1},
        frozen_payload_schema_version="schema-v1",
        settled_payload_schema_version="schema-v2",
        approved_source_lineage=False,
    )
    assert comparison.canonical_hash_match is False
    assert comparison.settlement_evaluation_eligible is False
    assert comparison.reason_codes == (
        "canonical_payload_hash_mismatch",
        "canonical_payload_schema_version_mismatch",
        "canonical_payload_source_lineage_not_approved",
    )


def test_embedded_schema_must_match_envelope_schema() -> None:
    with pytest.raises(CanonicalPayloadError, match="embedded schema_version"):
        canonical_payload_bytes(
            {"schema_version": "wrong"},
            payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        )


def test_raw_bytes_are_preserved_separately_from_canonical_bytes() -> None:
    raw = b'{ "value": 1.0 }\n'
    descriptor = describe_canonical_payload(
        {"value": 1.0},
        payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        raw_payload_bytes=raw,
    )
    assert descriptor.raw_payload_origin == "caller_preserved_bytes"
    assert descriptor.raw_payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert descriptor.raw_payload_base64 != descriptor.canonical_payload_utf8_base64


def test_canonical_output_is_repeatable_and_matches_cross_runtime_fixtures() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for row in fixture["fixtures"]:
        actual_bytes = canonical_payload_bytes(
            row["payload"],
            payload_schema_version=fixture["payload_schema_version"],
        )
        assert actual_bytes == row["canonical_utf8"].encode("utf-8")
        assert (
            canonical_payload_sha256(
                row["payload"],
                payload_schema_version=fixture["payload_schema_version"],
            )
            == row["canonical_sha256"]
        )
        assert actual_bytes == canonical_payload_bytes(
            row["payload"],
            payload_schema_version=fixture["payload_schema_version"],
        )


def test_nested_key_normalization_collision_fails_closed() -> None:
    with pytest.raises(CanonicalPayloadError, match="collide"):
        canonical_payload_bytes(
            {"nested": {"é": 1, "e\u0301": 2}},
            payload_schema_version=DECISION_FEATURE_PAYLOAD_SCHEMA_VERSION,
        )
