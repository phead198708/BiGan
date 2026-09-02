"""Zero-capital rollback drill tests for residual promotion v1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_rollback import (
    run_zero_capital_rollback_drill,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1/"
    "zero_capital_rollback_drill_report.json"
)


def test_zero_capital_rollback_drill_is_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = run_zero_capital_rollback_drill(
        repository_root=REPO_ROOT,
        output_path=first_path,
        created_at="2026-08-10T15:30:00+00:00",
    )
    second = run_zero_capital_rollback_drill(
        repository_root=REPO_ROOT,
        output_path=second_path,
        created_at="2026-08-10T15:30:00+00:00",
    )
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["technical_rollback_drill_passed"] is True
    assert first["deterministic_recovery_passed"] is True
    assert all(case["passed"] for case in first["rollback_cases"])
    assert {case["selected_action"] for case in first["rollback_cases"]} == {
        "NO_TRADE"
    }
    assert first["fresh_confirmation_passed"] is False
    assert first["phase6_passed"] is False
    assert first["ready_to_request_micro_live_approval"] is False
    assert first["micro_live_authorized"] is False
    assert first["automatic_live_unlock"] is False
    assert first["outcomes_accessed"] is False
    assert first["settlement_accessed"] is False
    assert first["pnl_accessed"] is False
    assert first["wallet_signing_attempted"] is False
    assert first["polymarket_write_attempted"] is False
    assert first["capital_exposed"] is False
    assert first["safety"] == SAFETY


def test_committed_rollback_report_reconciles() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sidecar = REPORT.with_suffix(REPORT.suffix + ".sha256")
    assert sidecar.read_text(encoding="utf-8").strip() == sha256_file(REPORT)
    assert report["technical_rollback_drill_passed"] is True
    assert report["ready_to_request_micro_live_approval"] is False
    assert report["safety"] == SAFETY


def test_bundle_sidecar_mismatch_fails_closed(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    shutil.copytree(REPO_ROOT, clone, ignore=shutil.ignore_patterns(".git"))
    sidecar = (
        clone
        / "examples/v8/polymarket_configs/"
        "BTC-15M-cost-aware-market-residual-promotion-v1/"
        "candidate_bundle/bundle_manifest.json.sha256"
    )
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        run_zero_capital_rollback_drill(
            repository_root=clone,
            output_path=tmp_path / "blocked.json",
            created_at="2026-08-10T15:30:00+00:00",
        )
