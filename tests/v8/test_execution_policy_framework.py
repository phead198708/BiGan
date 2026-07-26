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
    validate_execution_policy_fixture,
    validate_execution_policy_future_validation_protocol,
    validate_execution_policy_replay,
    validate_policy_candidate_manifest,
    validate_policy_reconciliation,
    validate_source_execution_compatibility,
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
    for name in (
        "execution_policy_contract.json",
        "policy_candidate_manifest.json",
        "source_execution_compatibility_manifest.json",
        "decision_attribution.jsonl",
        "risk_budget_state.jsonl",
        "replay_parity_report.json",
        "policy_safety_report.json",
        "execution_policy_future_validation_protocol.template.json",
    ):
        path = CONFIG / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            path.with_suffix(".sha256").read_text().strip()
        )
    validate_execution_policy_contract(_json("execution_policy_contract.json"))
    assert all(
        value is False for value in _json("execution_policy_contract.json")["safety"].values()
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

    probe_inputs = [
        _input(
            index,
            opportunity_window_id="shared-window",
            source_action_scores={
                "BUY_UP_HOLD_TO_SETTLEMENT": 0.06,
                "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.03,
                "NO_TRADE": 0.0,
            },
            uncertainty=0.25,
            fill_quality_score=0.85,
            provider_health_score=0.85,
        )
        for index in range(1, 3)
    ]
    decision_streams = {
        run_execution_policy_replay(
            inputs=probe_inputs,
            policy=_json(descriptor["path"]),
            compatibility_manifest=_compatibility(),
            runtime_mode="offline_replay",
        )["decision_stream_sha256"]
        for descriptor in manifest["candidate_fixtures"]
    }
    assert len(decision_streams) == 3


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

    exposure_replay = run_execution_policy_replay(
        inputs=[_input(index, opportunity_window_id=f"window-{index}") for index in range(1, 4)],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert "exposure_budget_exhausted" in exposure_replay["decisions"][2]["reason_codes"]


def test_duplicate_cooldown_and_replacement_behavior() -> None:
    policy = _json("execution_policy_quality_replacement_v1.json")
    first = _input(1, market_id="same", opportunity_window_id="w1")
    duplicate = _input(
        2,
        market_id="same",
        opportunity_window_id="w2",
        decision_ts=first["decision_ts"] + 1_000,
    )
    cooldown = _input(
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
    replacement = _input(
        4,
        market_id="same",
        opportunity_window_id="w4",
        decision_ts=first["decision_ts"] + 301_000,
        source_action_scores={
            "BUY_UP_HOLD_TO_SETTLEMENT": 0.05,
            "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.20,
            "NO_TRADE": 0.0,
        },
    )
    replay = run_execution_policy_replay(
        inputs=[first, duplicate, cooldown, replacement],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert "duplicate_position" in replay["decisions"][1]["reason_codes"]
    assert "reentry_cooldown_active" in replay["decisions"][2]["reason_codes"]
    assert replay["decisions"][3]["decision_effect"] == "replaced"
    assert replay["decisions"][3]["selected_side"] == "DOWN"
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
    safety = build_policy_safety_report(replay)
    assert safety["checks"]["policy_kill_switch_separate_from_source_scores"] is True
    assert safety["passed"] is True


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


def test_all_static_policy_governance_is_semantically_validated() -> None:
    contract = _json("execution_policy_contract.json")
    manifest = _json("policy_candidate_manifest.json")
    compatibility = _compatibility()
    policy = _json("execution_policy_high_signal_abstention_v1.json")
    validate_execution_policy_contract(contract)
    validate_policy_candidate_manifest(manifest)
    validate_execution_policy_fixture(policy)
    validate_source_execution_compatibility(
        compatibility_manifest=compatibility,
        policy=policy,
        source_model_hash=SOURCE_HASH,
    )
    validate_execution_policy_future_validation_protocol(
        _json("execution_policy_future_validation_protocol.template.json")
    )

    weakened_contract = copy.deepcopy(contract)
    weakened_contract["safety"].pop("wallet_enabled")
    with pytest.raises(ExecutionPolicyError, match="safety"):
        validate_execution_policy_contract(weakened_contract)

    open_manifest = copy.deepcopy(manifest)
    open_manifest["open_ended_optimizer_enabled"] = True
    with pytest.raises(ExecutionPolicyError, match="open_ended_optimizer"):
        validate_policy_candidate_manifest(open_manifest)

    unsafe_policy = copy.deepcopy(policy)
    unsafe_policy["paper_only"] = False
    with pytest.raises(ExecutionPolicyError, match="paper_only"):
        validate_execution_policy_fixture(unsafe_policy)

    permissive_compatibility = copy.deepcopy(compatibility)
    permissive_compatibility["required_decision_time_inputs"].remove("kill_switch_active")
    with pytest.raises(ExecutionPolicyError, match="required_inputs"):
        validate_source_execution_compatibility(
            compatibility_manifest=permissive_compatibility,
            policy=policy,
            source_model_hash=SOURCE_HASH,
        )


def test_invalid_types_nonfinite_and_unknown_inputs_fail_closed() -> None:
    invalid_timestamp = _input(1, decision_ts="1000")
    nonfinite_score = _input(
        2,
        source_action_scores={
            "BUY_UP_HOLD_TO_SETTLEMENT": float("nan"),
            "NO_TRADE": 0.0,
        },
    )
    unknown_input = _input(3, benign_but_unsupported_context="x")
    missing_input = _input(4)
    missing_input.pop("fill_quality_score")
    replay = run_execution_policy_replay(
        inputs=[invalid_timestamp, nonfinite_score, unknown_input, missing_input],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    assert all(decision["selected_action"] == "NO_TRADE" for decision in replay["decisions"])
    assert "decision_ts_invalid" in replay["decisions"][0]["reason_codes"]
    assert "source_action_scores_invalid" in replay["decisions"][1]["reason_codes"]
    assert "unsupported_execution_input_present" in replay["decisions"][2]["reason_codes"]
    assert "required_execution_input_missing" in replay["decisions"][3]["reason_codes"]
    assert len({decision["source_input_sha256"] for decision in replay["decisions"]}) == 4
    assert replay["decisions"][1]["source_action_scores_sha256"] != "0" * 64
    validate_execution_policy_replay(replay)


def test_replay_integrity_recomputes_rows_streams_and_state_chain() -> None:
    replay = run_execution_policy_replay(
        inputs=[_input(1), _input(2, opportunity_window_id="window-2")],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="paper_runtime",
    )
    validate_execution_policy_replay(replay)

    claimed_stream_tamper = copy.deepcopy(replay)
    claimed_stream_tamper["decision_stream_sha256"] = "f" * 64
    with pytest.raises(ExecutionPolicyError, match="decision_stream_sha256"):
        validate_execution_policy_replay(claimed_stream_tamper)

    attribution_tamper = copy.deepcopy(replay)
    attribution_tamper["decision_attribution"][0]["selected_side"] = "DOWN"
    with pytest.raises(ExecutionPolicyError, match="attribution"):
        validate_execution_policy_replay(attribution_tamper)

    chain_tamper = copy.deepcopy(replay)
    chain_tamper["risk_budget_state"][1]["before"]["total_exposure"] = 99.0
    with pytest.raises(ExecutionPolicyError, match="risk_state"):
        validate_execution_policy_replay(chain_tamper)


def test_parity_uses_derived_content_not_caller_claims() -> None:
    policy = _json("execution_policy_high_signal_abstention_v1.json")
    offline = run_execution_policy_replay(
        inputs=[_input(1)],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    paper = run_execution_policy_replay(
        inputs=[
            _input(
                1,
                source_action_scores={
                    "BUY_UP_HOLD_TO_SETTLEMENT": 0.13,
                    "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.04,
                    "NO_TRADE": 0.0,
                },
            )
        ],
        policy=policy,
        compatibility_manifest=_compatibility(),
        runtime_mode="paper_runtime",
    )
    report = build_replay_parity_report(
        offline_replay=offline,
        paper_runtime=paper,
    )
    assert report["passed"] is False
    assert report["checks"]["source_input_stream_sha256_match"] is False
    assert report["checks"]["execution_output_sha256_match"] is False

    paper["decision_stream_sha256"] = offline["decision_stream_sha256"]
    with pytest.raises(ExecutionPolicyError, match="decision_stream_sha256"):
        build_replay_parity_report(
            offline_replay=offline,
            paper_runtime=paper,
        )


def test_reconciliation_is_per_market_per_side_and_identifier_bound() -> None:
    replay = run_execution_policy_replay(
        inputs=[
            _input(1, opportunity_window_id="window-1"),
            _input(
                2,
                opportunity_window_id="window-2",
                source_action_scores={
                    "BUY_UP_HOLD_TO_SETTLEMENT": 0.04,
                    "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.12,
                    "NO_TRADE": 0.0,
                },
            ),
        ],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    market_tamper = copy.deepcopy(replay)
    market_tamper["intents"][0]["market_id"] = "market-2"
    market_tamper["fills"][0]["market_id"] = "market-2"
    market_tamper["ledger"][0]["market_id"] = "market-2"
    with pytest.raises(ExecutionPolicyError, match="reconciliation"):
        validate_policy_reconciliation(market_tamper)

    identifier_tamper = copy.deepcopy(replay)
    identifier_tamper["ledger"][0]["ledger_entry_id"] = "ledger:forged"
    with pytest.raises(ExecutionPolicyError, match="reconciliation"):
        validate_policy_reconciliation(identifier_tamper)


def test_empty_replay_cannot_vacuously_pass_safety() -> None:
    replay = run_execution_policy_replay(
        inputs=[],
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="offline_replay",
    )
    report = build_policy_safety_report(replay)
    assert report["checks"]["decisions_present"] is False
    assert report["passed"] is False


def test_source_inputs_are_not_mutated_by_policy_runtime() -> None:
    inputs = [_input(1)]
    before = copy.deepcopy(inputs)
    run_execution_policy_replay(
        inputs=inputs,
        policy=_json("execution_policy_high_signal_abstention_v1.json"),
        compatibility_manifest=_compatibility(),
        runtime_mode="paper_runtime",
    )
    assert inputs == before
