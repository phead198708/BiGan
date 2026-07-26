from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_attempt_002_promotion import (
    audit_attempt_002_promotion,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
MANIFEST_PATH = (
    CONFIG_DIR / "challenge_attempt_002_promotion_execution_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_promotion_execution_manifest_is_hash_pinned() -> None:
    sidecar = MANIFEST_PATH.with_suffix(".sha256")
    assert _sha256(MANIFEST_PATH) == sidecar.read_text(
        encoding="ascii"
    ).strip()
    manifest = _json(MANIFEST_PATH)
    assert manifest["schema_version"] == (
        "bigan-v8-challenge-attempt-002-promotion-execution-manifest-v1"
    )
    assert manifest["attempt_id"] == "v8-1-challenger-future-attempt-002"
    assert manifest["model_version"] == "v8.1"
    assert manifest["candidate_id"] == (
        "v8_1_entry_price_floor_0_30_sized_1_0"
    )
    assert manifest["safety"] == SAFE_FALSES


def test_promotion_implementation_hashes_match_frozen_commit() -> None:
    manifest = _json(MANIFEST_PATH)
    implementation = manifest["implementation"]
    paths = {
        "audit_module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/"
            "challenge_attempt_002_promotion.py"
        ),
        "audit_runner_sha256": (
            ROOT / "examples/v8/run_challenge_attempt_002_promotion_audit.py"
        ),
        "supplemental_module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/"
            "challenge_attempt_002_supplemental.py"
        ),
        "supplemental_runner_sha256": (
            ROOT / "examples/v8/run_challenge_attempt_002_supplemental.py"
        ),
        "test_sha256": (
            ROOT / "tests/v8/test_challenge_attempt_002_promotion.py"
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


def test_promotion_manifest_binds_existing_attempt_002_artifacts() -> None:
    manifest = _json(MANIFEST_PATH)
    lineage = manifest["lineage"]
    assert lineage["attempt_002_preregistration_sha256"] == _sha256(
        CONFIG_DIR / "challenge_attempt_002_preregistration.json"
    )
    assert lineage["attempt_002_execution_manifest_sha256"] == _sha256(
        CONFIG_DIR / "challenge_attempt_002_execution_manifest.json"
    )
    assert lineage["promotion_evidence_protocol_sha256"] == _sha256(
        CONFIG_DIR / "challenge_promotion_evidence_protocol.json"
    )


def test_repository_remains_blocked_without_real_future_evidence() -> None:
    manifest = _json(MANIFEST_PATH)
    report = audit_attempt_002_promotion(repository_root=ROOT)

    assert all(report["static_checks"].values())
    assert report["fresh_runtime_evidence_supplied"] is False
    assert report["decision"] == "BLOCKED"
    assert report["challenge_model_promotion_eligible"] is False
    assert report["selected_champion_candidate"] is None
    assert manifest["current_state"]["collection_started"] is False
    assert manifest["current_state"]["real_future_evidence_present"] is False
    assert manifest["current_state"]["promotion_ready"] is False
    assert manifest["current_state"]["sole_external_start_condition"] == (
        "explicit_operator_authorization_for_exact_120_market_collection"
    )
    assert manifest["safety"] == SAFE_FALSES
