from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
MANIFEST_PATH = (
    CONFIG_DIR
    / "challenge_attempt_002_target_freeze_execution_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_target_freeze_execution_manifest_is_hash_pinned() -> None:
    sidecar = MANIFEST_PATH.with_suffix(".sha256")
    assert _sha256(MANIFEST_PATH) == sidecar.read_text(
        encoding="ascii"
    ).strip()
    manifest = _json(MANIFEST_PATH)
    assert manifest["schema_version"] == (
        "bigan-v8-challenge-attempt-002-"
        "target-freeze-execution-manifest-v1"
    )
    assert manifest["attempt_id"] == "v8-1-challenger-future-attempt-002"
    assert manifest["model_version"] == "v8.1"
    assert manifest["candidate_id"] == (
        "v8_1_entry_price_floor_0_30_sized_1_0"
    )
    assert manifest["safety"] == SAFE_FALSES


def test_target_freeze_implementation_matches_frozen_commit() -> None:
    manifest = _json(MANIFEST_PATH)
    implementation = manifest["implementation"]
    paths = {
        "module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/"
            "challenge_attempt_002_target_freeze.py"
        ),
        "runner_sha256": (
            ROOT
            / "examples/v8/run_challenge_attempt_002_target_freeze.py"
        ),
        "test_sha256": (
            ROOT
            / "tests/v8/test_challenge_attempt_002_target_freeze.py"
        ),
        "runbook_sha256": (
            ROOT / "docs/v8/challenge_attempt_002_runbook.md"
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


def test_target_freeze_manifest_binds_existing_lineage() -> None:
    manifest = _json(MANIFEST_PATH)
    lineage = manifest["lineage"]
    local = {
        "attempt_002_collection_execution_manifest_sha256": (
            "challenge_attempt_002_collection_execution_manifest.json"
        ),
        "attempt_002_preregistration_sha256": (
            "challenge_attempt_002_preregistration.json"
        ),
        "entry_price_floor_profile_sha256": (
            "challenge_v8_1_entry_price_floor_0_30_profile.json"
        ),
        "feature_contract_sha256": (
            "execution_layer_v2_pairwise_action_advantage_"
            "lcb_feature_contract_v1.json"
        ),
        "frozen_model_binding_sha256": (
            "parallel_frozen_v8_1_model_binding.json"
        ),
        "sizing_profile_sha256": (
            "challenge_v8_1_entry_price_floor_0_30_sized_1_0_profile.json"
        ),
        "v8_1_candidate_contract_sha256": (
            "parallel_candidate_v8_1_primary_no_fallback_contract.json"
        ),
    }
    for field, filename in local.items():
        assert lineage[field] == _sha256(CONFIG_DIR / filename)
    assert lineage["historical_v8_1_fit_manifest_sha256"] == (
        "3fff5785a53cb32fb26d839786e3f48c2ff2bd7cc9dcf84e801c916a6ebb0fb7"
    )
    assert lineage["v6_2_candidate_manifest_sha256"] == (
        "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9"
    )


def test_target_freeze_manifest_keeps_collection_unauthorized() -> None:
    manifest = _json(MANIFEST_PATH)

    assert manifest["current_state"] == {
        "collection_started": False,
        "collector_pid": None,
        "operator_authorization_present": False,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "quality_valid_market_count": 0,
        "service_root_created": False,
        "target_freeze_executed": False,
    }
    assert manifest["execution_contract"][
        "candidate_scoring_during_raw_capture"
    ] is False
    assert manifest["execution_contract"][
        "settlement_or_resolution_provider_called"
    ] is False
    assert manifest["execution_contract"][
        "target_access_claim_written"
    ] is False
    assert manifest["safety"] == SAFE_FALSES
