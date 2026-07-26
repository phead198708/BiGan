from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.challenge_historical_development import SAFE_FALSES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/v8/polymarket_configs"
MANIFEST_PATH = CONFIG_DIR / "challenge_attempt_002_execution_manifest.json"
DRY_RUN_PATH = (
    CONFIG_DIR / "challenge_attempt_002_pipeline_synthetic_dry_run.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(path: Path) -> str:
    return path.with_suffix(".sha256").read_text(encoding="ascii").strip()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_execution_manifest_and_pipeline_dry_run_are_hash_pinned() -> None:
    assert _sha256(MANIFEST_PATH) == _sidecar(MANIFEST_PATH)
    assert _sha256(DRY_RUN_PATH) == _sidecar(DRY_RUN_PATH)

    manifest = _json(MANIFEST_PATH)
    dry_run = _json(DRY_RUN_PATH)
    assert manifest["synthetic_pipeline_dry_run"]["sha256"] == _sha256(
        DRY_RUN_PATH
    )
    assert dry_run["all_future_success_criteria_passed"] is True
    assert dry_run["synthetic_only"] is True
    assert dry_run["real_future_evidence"] is False
    assert dry_run["promotion_evidence_eligible"] is False
    assert dry_run["collection_control_invoked"] is False
    assert dry_run["safety"] == SAFE_FALSES


def test_execution_manifest_lineage_matches_repository_bytes() -> None:
    manifest = _json(MANIFEST_PATH)
    pipeline = manifest["pipeline"]
    assert pipeline["module_sha256"] == _sha256(
        ROOT
        / "src/bigan/v8/polymarket/challenge_attempt_002_pipeline.py"
    )
    assert pipeline["runner_sha256"] == _sha256(
        ROOT / "examples/v8/run_challenge_attempt_002_pipeline.py"
    )
    assert pipeline["test_sha256"] == _sha256(
        ROOT / "tests/v8/test_challenge_attempt_002_pipeline.py"
    )
    assert subprocess.run(
        ["git", "rev-parse", pipeline["implementation_commit"]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == pipeline["implementation_commit"]

    issue_paths = {
        "issue_254_parallel_candidate_protocol_sha256": (
            "parallel_candidate_protocol.json"
        ),
        "issue_255_candidate_budget_protocol_sha256": (
            "candidate_budget_protocol.json"
        ),
        "issue_256_execution_policy_contract_sha256": (
            "execution_policy_contract.json"
        ),
        "issue_257_feature_missingness_contract_sha256": (
            "feature_missingness_contract.json"
        ),
        "issue_258_regime_definition_contract_sha256": (
            "regime_definition_contract.json"
        ),
        "issue_259_canonical_payload_contract_sha256": (
            "canonical_payload_contract.json"
        ),
        "promotion_evidence_protocol_sha256": (
            "challenge_promotion_evidence_protocol.json"
        ),
    }
    lineage = manifest["issue_contract_lineage"]
    for field, filename in issue_paths.items():
        assert lineage[field] == _sha256(CONFIG_DIR / filename)


def test_execution_manifest_keeps_collection_and_promotion_blocked() -> None:
    manifest = _json(MANIFEST_PATH)
    collection = manifest["collection_state"]
    promotion = manifest["promotion_state"]

    assert collection["operator_authorization_required"] is True
    assert collection["operator_authorization_artifact_present"] is False
    assert collection["collection_started"] is False
    assert collection["collector_pid"] is None
    assert collection["quality_valid_market_count"] == 0
    assert collection["outcomes_resolution_labels_or_pnl_opened"] is False
    assert promotion["promotion_evidence_present"] is False
    assert promotion["promotion_ready"] is False
    assert manifest["safety"] == SAFE_FALSES
