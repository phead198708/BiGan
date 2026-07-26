from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
MANIFEST_PATH = (
    CONFIG_DIR / "challenge_attempt_002_collection_execution_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_collection_execution_manifest_is_hash_pinned() -> None:
    sidecar = MANIFEST_PATH.with_suffix(".sha256")
    assert _sha256(MANIFEST_PATH) == sidecar.read_text(
        encoding="ascii"
    ).strip()
    manifest = _json(MANIFEST_PATH)
    assert manifest["schema_version"] == (
        "bigan-v8-challenge-attempt-002-collection-execution-manifest-v1"
    )
    assert manifest["attempt_id"] == "v8-1-challenger-future-attempt-002"
    assert manifest["model_version"] == "v8.1"
    assert manifest["safety"] == SAFE_FALSES


def test_collection_implementation_and_generic_collector_are_frozen() -> None:
    manifest = _json(MANIFEST_PATH)
    implementation = manifest["implementation"]
    paths = {
        "supervisor_module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/"
            "challenge_attempt_002_collection.py"
        ),
        "supervisor_runner_sha256": (
            ROOT / "examples/v8/run_challenge_attempt_002_collection.py"
        ),
        "supervisor_test_sha256": (
            ROOT / "tests/v8/test_challenge_attempt_002_collection.py"
        ),
        "generic_collector_runner_sha256": (
            ROOT
            / "examples/v8/"
            "run_execution_layer_v2_persistent_outcome_blind_collector.py"
        ),
        "generic_collector_module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/training/"
            "execution_layer_v2_persistent_outcome_blind_collector.py"
        ),
    }
    for field, path in paths.items():
        assert implementation[field] == _sha256(path)
    commit = implementation["commit"]
    assert subprocess.run(
        ["git", "rev-parse", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == commit


def test_collection_protocol_pins_and_limits_match_repository() -> None:
    manifest = _json(MANIFEST_PATH)
    lineage = manifest["lineage"]
    assert lineage["attempt_002_preregistration_sha256"] == _sha256(
        CONFIG_DIR / "challenge_attempt_002_preregistration.json"
    )
    assert lineage["collector_protocol_sha256"] == _sha256(
        CONFIG_DIR
        / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
    )
    assert lineage["feature_contract_sha256"] == _sha256(
        CONFIG_DIR
        / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
    )
    limits = manifest["bounded_collection"]
    assert limits == {
        "batch_market_count": 12,
        "exact_quality_valid_market_target": 120,
        "maximum_attempted_market_count": 180,
        "maximum_batch_count": 15,
        "stop_when_quality_valid_target_reached": True,
    }


def test_collection_remains_unstarted_and_requires_authorization() -> None:
    manifest = _json(MANIFEST_PATH)
    current = manifest["current_state"]
    service_root = (
        ROOT
        / "examples/v8/polymarket_live_runs/"
        "challenge-model-v8-1-attempt-002"
    )

    assert current["operator_authorization_present"] is False
    assert current["collection_started"] is False
    assert current["collector_pid"] is None
    assert current["attempted_market_count"] == 0
    assert current["quality_valid_market_count"] == 0
    assert current["outcomes_resolution_labels_or_pnl_opened"] is False
    assert current["service_root_created"] is False
    assert not service_root.exists()
    assert manifest["safety"] == SAFE_FALSES
