from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.integration_closure import (
    IntegrationClosureError,
    verify_integration_closure_payload,
    verify_integration_closure_set,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "examples" / "v8" / "integration_closure"


def _manifest_paths() -> list[Path]:
    return sorted(MANIFEST_DIR.glob("issue264_*_integration_closure_manifest.json"))


def _payload() -> dict:
    manifests = _manifest_paths()
    assert manifests, "at least one issue #264 integration closure manifest is required"
    return json.loads(manifests[-1].read_text())


def test_committed_issue264_integration_closures_verify_exactly() -> None:
    manifests = _manifest_paths()
    assert manifests
    report = verify_integration_closure_set(manifests, root=ROOT)
    assert report["verification_passed"] is True
    assert report["entry_count"] > 0
    assert report["layer_count"] == len(manifests)


def test_duplicate_destination_fails_closed() -> None:
    payload = _payload()
    payload["entries"].append(copy.deepcopy(payload["entries"][0]))
    with pytest.raises(IntegrationClosureError, match="not sorted|duplicate"):
        verify_integration_closure_payload(payload, root=ROOT)


def test_byte_drift_fails_closed() -> None:
    payload = _payload()
    payload["entries"][0]["destination_sha256"] = "0" * 64
    with pytest.raises(IntegrationClosureError, match="destination_sha256 drift"):
        verify_integration_closure_payload(payload, root=ROOT)


def test_extra_changed_path_fails_closed() -> None:
    payload = _payload()
    removable = next(
        entry
        for entry in payload["entries"]
        if entry["destination_path"] == ".github/workflows/v8-phase0.yml"
    )
    payload["entries"].remove(removable)
    with pytest.raises(IntegrationClosureError, match="closure inventory mismatch"):
        verify_integration_closure_payload(payload, root=ROOT)


def test_missing_declared_path_fails_closed() -> None:
    payload = _payload()
    payload["self_paths"].append("examples/v8/integration_closure/not-present.json")
    with pytest.raises(IntegrationClosureError, match="self_paths must contain exactly"):
        verify_integration_closure_payload(payload, root=ROOT)


def test_frozen_evidence_and_candidate_bindings_are_explicit() -> None:
    payloads = [json.loads(path.read_text()) for path in _manifest_paths()]
    frozen = [
        binding
        for payload in payloads
        for binding in payload["frozen_evidence_bindings"]
    ]
    candidate_bindings = [
        binding
        for payload in payloads
        for binding in payload["candidate_implementation_bindings"]
    ]
    if any("cost-aware-market-residual-v6" in str(item) for item in payloads):
        assert any(binding["kind"] == "frozen_terminal" for binding in frozen)
        assert any(binding["kind"] == "frozen_reconciliation" for binding in frozen)
        assert candidate_bindings
        assert all(
            binding["descriptor_json_pointer"].endswith(
                ("/candidate_implementation", "/inputs/implementation")
            )
            for binding in candidate_bindings
        )


def test_safety_flags_remain_false() -> None:
    for path in _manifest_paths():
        payload = json.loads(path.read_text())
        assert payload["safety"]
        assert set(payload["safety"].values()) == {False}
