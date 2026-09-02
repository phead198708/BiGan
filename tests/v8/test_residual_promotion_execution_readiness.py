"""Non-executable execution-readiness tests for residual promotion v1."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_execution_readiness import (
    BUNDLE_REPOSITORY_PATH,
    ExecutionReadinessError,
    NonExecutableIntentLedger,
    build_execution_readiness_report,
    reconcile_synthetic_fill,
)
from bigan.v8.polymarket.residual_promotion_release_readiness import (
    _operational_rollback_passes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-promotion-v1"
)
REPORT = CONFIG / "execution_engineering_readiness_report.json"
OPERATIONAL_REPORT = CONFIG / "operational_rollback_drill_report.json"
PREAPPROVAL_CONTRACT = CONFIG / "micro_live_preapproval_contract_v5.json"
BUNDLE = REPO_ROOT / BUNDLE_REPOSITORY_PATH


def _projection() -> dict:
    parity = json.loads(
        (CONFIG / "candidate_bundle/offline_live_parity_report.json").read_text()
    )
    return dict(parity["live_projection"])


def _record(ledger: NonExecutableIntentLedger, projection: dict) -> dict:
    return ledger.record_projection(
        market_id="synthetic-market-001",
        decision_ts=1_786_406_400_000,
        market_family="BTC-15M",
        candidate_bundle_sha256=sha256_file(BUNDLE),
        projection=projection,
    )


def test_intent_identity_is_idempotent_and_restart_safe() -> None:
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    first = _record(ledger, _projection())
    duplicate = _record(ledger, _projection())
    recovered = NonExecutableIntentLedger.restore(ledger.export_state())
    replay = _record(recovered, _projection())
    assert first["status"] == "RECORDED_NON_EXECUTABLE"
    assert duplicate["status"] == replay["status"] == "IDEMPOTENT_REPLAY"
    assert first["intent_id"] == duplicate["intent_id"] == replay["intent_id"]
    assert len(ledger.entries) == len(recovered.entries) == 1
    assert first["executable"] is False
    assert first["paper_order_allowed"] is False
    assert first["live_order_allowed"] is False


def test_conflicting_duplicate_fails_closed_without_mutation() -> None:
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    projection = _projection()
    _record(ledger, projection)
    before = ledger.export_state()
    projection["selected_action"] = (
        "BUY_UP_HOLD"
        if projection["selected_action"] != "BUY_UP_HOLD"
        else "BUY_DOWN_HOLD"
    )
    with pytest.raises(ExecutionReadinessError, match="conflicting duplicate"):
        _record(ledger, projection)
    assert ledger.export_state() == before


def test_tampered_state_fails_closed() -> None:
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    _record(ledger, _projection())
    state = ledger.export_state()
    state["entries"][0]["selected_action"] = "BUY_UP_HOLD"
    with pytest.raises(ExecutionReadinessError, match="state SHA-256 mismatch"):
        NonExecutableIntentLedger.restore(state)


@pytest.mark.parametrize("forbidden", ["outcome", "settlement_price", "unit_pnl"])
def test_outcome_bearing_projection_fields_fail_closed(forbidden: str) -> None:
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    projection = _projection()
    projection[forbidden] = 1
    with pytest.raises(ExecutionReadinessError, match="forbidden data field"):
        _record(ledger, projection)
    assert ledger.entries == ()


def test_kill_switch_and_candidate_mismatch_emit_no_intent() -> None:
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    ledger.engage_kill_switch()
    blocked = _record(ledger, _projection())
    assert blocked["status"] == "BLOCKED_NO_TRADE"
    assert blocked["reason"] == "kill_switch_active"
    assert blocked["intent_id"] is None
    assert ledger.entries == ()

    mismatch = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    result = mismatch.record_projection(
        market_id="synthetic-market-002",
        decision_ts=1_786_406_400_000,
        market_family="BTC-15M",
        candidate_bundle_sha256="0" * 64,
        projection=_projection(),
    )
    assert result["status"] == "BLOCKED_NO_TRADE"
    assert result["reason"] == "candidate_bundle_mismatch"
    assert mismatch.entries == ()


def test_synthetic_fill_reconciles_without_settlement_or_execution() -> None:
    result = reconcile_synthetic_fill(
        intent_id="a" * 64,
        side="DOWN",
        quantity="2",
        price="0.40",
        fee="0.01",
    )
    assert result["position_delta"] == "2"
    assert result["cash_delta"] == "-0.81"
    assert result["order_fill_position_cash_reconciled"] is True
    assert result["settlement_reconciliation_verified"] is False
    assert result["executable"] is False


def test_report_is_deterministic_non_authorizing_and_frozen(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = build_execution_readiness_report(
        repository_root=REPO_ROOT,
        output_path=first_path,
        created_at="2026-08-10T20:15:00+00:00",
    )
    second = build_execution_readiness_report(
        repository_root=REPO_ROOT,
        output_path=second_path,
        created_at="2026-08-10T20:15:00+00:00",
    )
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["engineering_readiness_passed"] is True
    assert first["security_review_passed"] is False
    assert first["paper_candidate_allowed"] is False
    assert first["paper_run_started"] is False
    assert first["phase6_zero_capital_authorized"] is False
    assert first["micro_live_authorized"] is False
    assert first["order_submission_attempted"] is False
    assert first["fresh_outcomes_accessed"] is False
    assert first["outcomes_accessed"] is False
    assert first["settlement_accessed"] is False
    assert first["pnl_accessed"] is False
    assert first["safety"] == SAFETY


def test_committed_readiness_report_reconciles() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert REPORT.with_suffix(REPORT.suffix + ".sha256").read_text().strip() == sha256_file(
        REPORT
    )
    assert report["candidate_bundle"]["sha256"] == sha256_file(BUNDLE)
    for descriptor_name in ("implementation", "cli", "frozen_runtime_parity"):
        descriptor = report[descriptor_name]
        assert descriptor["sha256"] == sha256_file(REPO_ROOT / descriptor["path"])
    assert report["engineering_readiness_passed"] is True
    assert report["security_review_passed"] is False
    assert report["paper_run_started"] is False
    assert report["safety"] == SAFETY


def test_committed_operational_rollback_satisfies_frozen_contract() -> None:
    report = json.loads(OPERATIONAL_REPORT.read_text(encoding="utf-8"))
    contract = json.loads(PREAPPROVAL_CONTRACT.read_text(encoding="utf-8"))
    assert OPERATIONAL_REPORT.with_suffix(".json.sha256").read_text().strip() == sha256_file(
        OPERATIONAL_REPORT
    )
    assert _operational_rollback_passes(
        {"operational_rollback": report}, contract
    ) is True
    assert report["fresh_population_used"] is False
    assert report["fresh_outcomes_accessed"] is False
    assert report["phase6_zero_capital_authorized"] is False
    assert report["micro_live_authorized"] is False


def test_bundle_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    shutil.copytree(REPO_ROOT, clone, ignore=shutil.ignore_patterns(".git"))
    bundle = clone / BUNDLE_REPOSITORY_PATH
    payload = json.loads(bundle.read_text())
    payload["candidate_id"] = "tampered"
    bundle.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ExecutionReadinessError, match="sidecar mismatch"):
        build_execution_readiness_report(
            repository_root=clone,
            output_path=tmp_path / "blocked.json",
            created_at="2026-08-10T20:15:00+00:00",
        )


def test_report_payload_does_not_depend_on_mutable_inputs() -> None:
    projection = _projection()
    original = copy.deepcopy(projection)
    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    _record(ledger, projection)
    projection["selected_action"] = "NO_TRADE"
    assert ledger.entries[0]["decision_sha256"] != canonical_projection_sha(projection)
    assert original != projection


def test_rehashed_tampered_ledger_identity_still_fails_closed() -> None:
    from bigan.v8.polymarket.contracts import canonical_json_sha256

    ledger = NonExecutableIntentLedger(candidate_bundle_sha256=sha256_file(BUNDLE))
    _record(ledger, _projection())
    state = ledger.export_state()
    entry = state["entries"][0]
    entry["business_key"] = "b" * 64
    core = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = canonical_json_sha256(core)
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ExecutionReadinessError, match="chain or safety invariant"):
        NonExecutableIntentLedger.restore(state)


def canonical_projection_sha(projection: dict) -> str:
    from bigan.v8.polymarket.contracts import canonical_json_sha256

    return canonical_json_sha256(projection)
