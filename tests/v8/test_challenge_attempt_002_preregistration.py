from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_attempt_002 import (
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
PROTOCOL_PATH = CONFIG_DIR / "challenge_attempt_002_preregistration.json"
DRY_RUN_PATH = CONFIG_DIR / "challenge_attempt_002_synthetic_dry_run.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(path: Path) -> str:
    return path.with_suffix(".sha256").read_text(encoding="ascii").strip()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_attempt_002_protocol_and_synthetic_dry_run_are_hash_pinned() -> None:
    assert _sha256(PROTOCOL_PATH) == _sidecar(PROTOCOL_PATH)
    assert _sha256(DRY_RUN_PATH) == _sidecar(DRY_RUN_PATH)

    protocol = _json(PROTOCOL_PATH)
    validate_attempt_002_preregistration(
        protocol,
        expected_lineage=protocol["lineage"],
    )
    dry_run = _json(DRY_RUN_PATH)
    assert dry_run["synthetic_only"] is True
    assert dry_run["real_collection_started"] is False
    assert (
        dry_run["real_labels_outcomes_settlement_or_pnl_opened"] is False
    )
    assert dry_run["result"]["all_future_success_criteria_passed"] is True
    assert dry_run["result"]["safety"] == SAFE_FALSES


def test_attempt_002_lineage_matches_exact_repository_bytes_and_commits() -> None:
    protocol = _json(PROTOCOL_PATH)
    lineage = protocol["lineage"]
    pinned_paths = {
        "attempt_001_closure_sha256": (
            CONFIG_DIR / "challenge_attempt_001_closure.json"
        ),
        "candidate_module_sha256": (
            ROOT
            / "src/bigan/v8/polymarket/"
            "challenge_v8_1_entry_price_floor_sizing.py"
        ),
        "candidate_profile_sha256": (
            CONFIG_DIR
            / "challenge_v8_1_entry_price_floor_0_30_sized_1_0_profile.json"
        ),
        "development_registry_sha256": (
            CONFIG_DIR / "challenge_historical_development_data_registry.json"
        ),
        "future_evaluator_module_sha256": (
            ROOT / "src/bigan/v8/polymarket/challenge_attempt_002.py"
        ),
        "future_evaluator_test_sha256": (
            ROOT / "tests/v8/test_challenge_attempt_002.py"
        ),
        "historical_iteration_003_entry_file_sha256": (
            CONFIG_DIR / "challenge_historical_development_iteration_003_entry.json"
        ),
        "historical_iteration_003_result_sha256": (
            CONFIG_DIR / "challenge_historical_development_iteration_003_result.json"
        ),
        "historical_success_standard_sha256": (
            CONFIG_DIR / "challenge_historical_development_success_standard.json"
        ),
        "synthetic_dry_run_sha256": DRY_RUN_PATH,
    }
    for field, path in pinned_paths.items():
        assert _sha256(path) == lineage[field]

    for field in (
        "candidate_implementation_commit",
        "future_evaluator_commit",
        "historical_evidence_commit",
    ):
        resolved = subprocess.run(
            ["git", "rev-parse", lineage[field]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert resolved == lineage[field]


def test_attempt_002_remains_preregistered_but_not_authorized() -> None:
    protocol = _json(PROTOCOL_PATH)
    window = protocol["future_window"]

    assert protocol["model_version"] == "v8.1"
    assert protocol["candidate_id"] == (
        "v8_1_entry_price_floor_0_30_sized_1_0"
    )
    assert window["exact_quality_valid_market_count"] == 120
    assert window["operator_collection_authorization_required"] is True
    assert window["operator_collection_authorization_granted"] is False
    assert window["collection_started"] is False
    assert window["collector_pid"] is None
    assert window["attempted_market_count"] == 0
    assert window["quality_valid_market_count"] == 0
    assert window["outcomes_resolution_labels_or_pnl_opened"] is False
    assert protocol["alpha_spending"]["attempt_consumed"] is False
    assert protocol["safety"] == SAFE_FALSES
