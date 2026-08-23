"""Exact candidate identity and runtime-byte tests for promotion v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.residual_promotion_release_readiness_v7 import (
    CONTRACT_REPOSITORY_PATH,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    ResidualPromotionError,
    load_residual_promotion_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v7_contract_loads_the_exact_frozen_candidate_runtime() -> None:
    contract = _json(REPO_ROOT / CONTRACT_REPOSITORY_PATH)
    descriptor = dict(contract["candidate_bundle"])
    manifest = REPO_ROOT / descriptor["path"]
    assert sha256_file(manifest) == descriptor["sha256"]

    runtime = load_residual_promotion_runtime(
        manifest_path=descriptor["path"],
        expected_manifest_sha256=descriptor["sha256"],
        repository_root=REPO_ROOT,
    )
    assert runtime.lineage_id == LINEAGE_ID == contract["lineage_id"]
    assert runtime.candidate_id == CANDIDATE_ID == contract["candidate_id"]
    assert runtime.manifest_sha256 == descriptor["sha256"]


def test_candidate_manifest_or_expected_sha_drift_fails_closed(tmp_path: Path) -> None:
    contract = _json(REPO_ROOT / CONTRACT_REPOSITORY_PATH)
    descriptor = dict(contract["candidate_bundle"])
    manifest = REPO_ROOT / descriptor["path"]

    with pytest.raises(ResidualPromotionError, match="manifest SHA-256 mismatch"):
        load_residual_promotion_runtime(
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            repository_root=REPO_ROOT,
        )

    outside = tmp_path / "bundle_manifest.json"
    outside.write_bytes(manifest.read_bytes())
    with pytest.raises(ResidualPromotionError, match="inside the repository"):
        load_residual_promotion_runtime(
            manifest_path=outside,
            expected_manifest_sha256=sha256_file(outside),
            repository_root=REPO_ROOT,
        )


def test_frozen_candidate_runtime_has_no_execution_or_wallet_surface() -> None:
    contract = _json(REPO_ROOT / CONTRACT_REPOSITORY_PATH)
    descriptor = dict(contract["candidate_bundle"])
    runtime = load_residual_promotion_runtime(
        manifest_path=descriptor["path"],
        expected_manifest_sha256=descriptor["sha256"],
        repository_root=REPO_ROOT,
    )
    forbidden = {
        "submit_order",
        "cancel_order",
        "sign_transaction",
        "wallet",
        "settle",
    }
    assert forbidden.isdisjoint(dir(runtime))
