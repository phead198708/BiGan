from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.execution_policy_framework import (
    ExecutionPolicyError,
    build_policy_safety_report,
    build_replay_parity_report,
    execution_policy_hash,
    run_execution_policy_replay,
    validate_execution_policy_contract,
    validate_policy_reconciliation,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/v8/polymarket_configs"
SOURCE_HASH = "1565116daeb2f5d4d8c33fefa507276f59251edd5ffb5f4f313041bcf9dbb0ec"


def _json(name: str):
    return json.loads((CONFIG / name).read_text())


def _compatibility():
    return _json("source_execution_compatibility_manifest.json")


def _input(index: int, **overrides):
    row = {
        "market_id": f"market-{index}",
        "decision_ts": 1_000_000 + index * 1_000,
        "source_model_hash": SOURCE_HASH,
        "source_action_scores": {
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.12,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.04,
            "NO_TRADE": 0.0,
        },
        "uncertainty": 0.1,
        "opportunity_window_id": "window-1",
        "fill_quality_score": 0.95,
        "provider_health_score": 1.0,
        "provider_features_complete": True,
        "kill_switch_active": False,
    }
    row.update(overrides)
    return row


def test_contract_and_candidate_manifest_are_hash_pinned_and_closed() -> None:
    contract_path = CONFIG / "execution_policy_contract.json"
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == (
        contract_path.with_suffix(".sha256").read_text().strip()
    )
    validate_execution_policy_contract(_json("execution_policy_contract.json"))
    assert all(value is False for value in _json("execution_policy_contract.json")["safety"].values())
    manifest_path = CONFIG / "policy_candidate_manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        manifest_path.with_suffix(".sha256").read_text().strip()
    )


def test_three_materially_distinct_policy_fixtures_are_hash_bound() -> None:
    manifest = _json("policy_candidate_manifest.json")
    assert len(manifest["candidate_fixtures"]) == 3
    hashes = set()
    for descriptor in manifest["candidate_fixtures"]:
        path = CONFIG / descriptor["path"]
        fixture = json.loads(path.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["raw_sha256"]
        assert execution_policy_hash(fixture) == descriptor["execution_policy_hash"]
        hashes.add(descriptor["execution_policy_hash"])
    assert len(hashes) == 3
    assert set(_compatibility()["allowed_execution_policy_hashes"]) == hashes


def test_identical_source_scores_produce_offline_paper_parity() -> None:
    policy = _json("execution_policy_high_signal_abstention_v1.json")
    inputs = [_input(index, opportunity_window_id=f"window-{index}") for index in range(3)]
    offline = run_execution_policy_replay(
        inputs=inputs,
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    paper = run_execution_policy_replay(
        inputs=inputs,
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="paper_runtime",
    )
    parity = build_replay_parity_report(
        offline_replay=offline,
        paper_runtime=paper,
    )
    assert parity["passed"] is True
    assert offline["decision_stream_sha256"] == paper["decision_stream_sha256"]
    assert build_policy_safety_report(offline)["passed"] is True


def test_no_trade_exposure_limit_and_opportunity_budget_have_attribution() -> None:
    policy = _json("execution_policy_risk_budgeted_v1.json")
    replay = run_execution_policy_replay(
        inputs=[
            _input(1),
            _input(2),
            _input(3, opportunity_window_id="window-2", provider_health_score=0.1),
        ],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert replay["decisions"][0]["selected_action"] != "NO_TRADE"
    assert replay["decisions"][1]["selected_action"] == "NO_TRADE"
    assert "opportunity_budget_exhausted" in replay["decisions"][1]["reason_codes"]
    assert replay["decisions"][2]["selected_action"] == "NO_TRADE"
    assert "provider_health_below_minimum" in replay["decisions"][2]["reason_codes"]
    assert all(row["rule_results"] for row in replay["decision_attribution"])


def test_duplicate_cooldown_and_replacement_behavior() -> None:
    policy = _json("execution_policy_quality_replacement_v1.json")
    first = _input(1, market_id="same", opportunity_window_id="w1")
    duplicate = _input(
        2,
        market_id="same",
        opportunity_window_id="w2",
        decision_ts=first["decision_ts"] + 1_000,
    )
    replacement = _input(
        3,
        market_id="same",
        opportunity_window_id="w3",
        decision_ts=first["decision_ts"] + 2_000,
        source_action_scores={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.05,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
            "NO_TRADE": 0.0,
        },
    )
    replay = run_execution_policy_replay(
        inputs=[first, duplicate, replacement],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert "duplicate_position" in replay["decisions"][1]["reason_codes"]
    assert replay["decisions"][2]["decision_effect"] == "replaced"
    assert replay["decisions"][2]["selected_side"] == "DOWN"
    assert len(replay["intents"]) == 3
    assert replay["reconciliation_report"]["passed"] is True


def test_missing_unsupported_kill_switch_or_future_input_fails_closed_no_trade() -> None:
    policy = _json("execution_policy_high_signal_abstention_v1.json")
    replay = run_execution_policy_replay(
        inputs=[
            _input(1, provider_features_complete=False),
            _input(2, kill_switch_active=True),
            _input(3, resolved_outcome="UP"),
        ],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert all(row["selected_action"] == "NO_TRADE" for row in replay["decisions"])
    assert "provider_features_incomplete" in replay["decisions"][0]["reason_codes"]
    assert "policy_kill_switch_active_or_missing" in replay["decisions"][1]["reason_codes"]
    assert "forbidden_target_or_future_input_present" in replay["decisions"][2]["reason_codes"]


def test_position_intent_fill_ledger_tamper_is_rejected() -> None:
    replay = run_execution_policy_replay(
        inputs=[_input(1)],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    tampered = copy.deepcopy(replay)
    tampered["ledger"][0]["notional_delta"] = 9.0
    with pytest.raises(ExecutionPolicyError, match="reconciliation"):
        validate_policy_reconciliation(tampered)
