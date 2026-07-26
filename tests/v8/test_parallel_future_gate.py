from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.parallel_future_gate import (
    ParallelFutureGateError,
    build_parallel_target_free_freeze,
    evaluate_parallel_future_gate,
    validate_parallel_candidate_protocol,
    validate_parallel_future_collection_plan,
)
from examples.v8.run_parallel_future_gate import (
    build_legacy_v8_3_smoke_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/v8/polymarket_configs"


def _json(name: str):
    return json.loads((CONFIG / name).read_text())


def _contracts():
    return {
        "v8_1_primary_no_fallback": _json(
            "parallel_candidate_v8_1_primary_no_fallback_contract.json"
        ),
        "v8_3_primary_with_fallback": _json(
            "parallel_candidate_v8_3_primary_with_fallback_contract.json"
        ),
        "matched_frozen_v6_7": _json(
            "parallel_candidate_matched_frozen_v6_7_contract.json"
        ),
    }


def _source_rows(count: int = 45):
    return [
        {
            "market_id": f"market-{index:03d}",
            "decision_ts": 1_000_000 + index,
            "feature_score": 0.5,
        }
        for index in range(count)
    ]


def _decision(row: dict, candidate_id: str, *, action: str, origin: str):
    return {
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "executed_action": action,
        "selected_side": "UP" if action != "NO_TRADE" else "NONE",
        "decision_origin": origin,
        "primary_abstained": action == "NO_TRADE",
        "fallback_used": "fallback" in origin,
        "execution_guard_order_allowed": action != "NO_TRADE",
        "proposed_order_size": 1.0 if action != "NO_TRADE" else 0.0,
        "target_used_as_decision_input": False,
        "v8_3_frozen_contract_reproduced": candidate_id
        == "v8_3_primary_with_fallback",
        "matched_baseline_frozen_contract_reproduced": candidate_id
        == "matched_frozen_v6_7",
    }


def _freeze(count: int = 45):
    rows = _source_rows(count)
    decisions = {
        "v8_1_primary_no_fallback": [
            _decision(row, "v8_1_primary_no_fallback", action="BUY_UP", origin="primary")
            for row in rows
        ],
        "v8_3_primary_with_fallback": [
            _decision(
                row,
                "v8_3_primary_with_fallback",
                action="BUY_UP",
                origin="fallback_v6_7" if index % 2 else "primary",
            )
            for index, row in enumerate(rows)
        ],
        "matched_frozen_v6_7": [
            _decision(row, "matched_frozen_v6_7", action="BUY_UP", origin="baseline")
            for row in rows
        ],
    }
    return build_parallel_target_free_freeze(
        protocol=_json("parallel_candidate_protocol.json"),
        candidate_contracts=_contracts(),
        source_rows=rows,
        decisions_by_candidate=decisions,
        decision_freeze_created_ts=900_000,
        target_access_started=False,
    )


def _targets(freeze):
    return [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "after_cost_pnl_per_notional_by_action": {
                "BUY_UP": 0.03,
                "NO_TRADE": 0.0,
            },
            "target_available_after_decision_freeze": True,
            "target_used_as_decision_input": False,
        }
        for row in freeze["shared_source_rows"]
    ]


def test_protocol_and_candidate_contracts_are_hash_pinned() -> None:
    names = [
        "parallel_candidate_protocol.json",
        "parallel_candidate_v8_1_primary_no_fallback_contract.json",
        "parallel_candidate_v8_3_primary_with_fallback_contract.json",
        "parallel_candidate_matched_frozen_v6_7_contract.json",
    ]
    for name in names:
        path = CONFIG / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(
            ".sha256"
        ).read_text().strip()
    validate_parallel_candidate_protocol(
        _json("parallel_candidate_protocol.json"),
        candidate_contracts=_contracts(),
    )


def test_fresh_collection_plan_is_hash_pinned_and_preregistered() -> None:
    plan_path = CONFIG / "parallel_future_collection_plan.json"
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_path.with_suffix(
        ".sha256"
    ).read_text().strip()
    candidate_hashes = {
        "v8_1_primary_no_fallback": hashlib.sha256(
            (
                CONFIG
                / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
            ).read_bytes()
        ).hexdigest(),
        "v8_3_primary_with_fallback": hashlib.sha256(
            (
                CONFIG
                / "parallel_candidate_v8_3_primary_with_fallback_contract.json"
            ).read_bytes()
        ).hexdigest(),
        "matched_frozen_v6_7": hashlib.sha256(
            (
                CONFIG
                / "parallel_candidate_matched_frozen_v6_7_contract.json"
            ).read_bytes()
        ).hexdigest(),
    }
    plan = _json("parallel_future_collection_plan.json")
    validate_parallel_future_collection_plan(
        plan,
        protocol_sha256=hashlib.sha256(
            (CONFIG / "parallel_candidate_protocol.json").read_bytes()
        ).hexdigest(),
        candidate_contract_sha256s=candidate_hashes,
        collector_protocol_sha256=hashlib.sha256(
            (
                CONFIG
                / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
            ).read_bytes()
        ).hexdigest(),
        feature_contract_sha256=hashlib.sha256(
            (
                CONFIG
                / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
            ).read_bytes()
        ).hexdigest(),
        historical_gate_contract_sha256=hashlib.sha256(
            (
                CONFIG / "historical_replay_superiority_contract.json"
            ).read_bytes()
        ).hexdigest(),
        historical_replay_report_sha256=hashlib.sha256(
            (
                CONFIG / "historical_replay_superiority_report.json"
            ).read_bytes()
        ).hexdigest(),
        historical_replay_report=_json(
            "historical_replay_superiority_report.json"
        ),
        collection_started_ts=plan["freeze_created_ts"] + 1,
    )


def test_collection_plan_rejects_target_access_and_non_later_start() -> None:
    plan = _json("parallel_future_collection_plan.json")
    hashes = plan["lineage"]
    plan["collection"]["resolution_provider_enabled_during_collection"] = True
    with pytest.raises(ParallelFutureGateError, match="resolution_provider"):
        validate_parallel_future_collection_plan(
            plan,
            protocol_sha256=hashes["parallel_candidate_protocol_sha256"],
            candidate_contract_sha256s=hashes["candidate_contract_sha256s"],
            collector_protocol_sha256=hashes[
                "persistent_collector_protocol_sha256"
            ],
            feature_contract_sha256=hashes["feature_contract_sha256"],
            historical_gate_contract_sha256=plan[
                "historical_replay_prerequisite"
            ]["gate_contract_sha256"],
            historical_replay_report_sha256=plan[
                "historical_replay_prerequisite"
            ]["report_sha256"],
            historical_replay_report=_json(
                "historical_replay_superiority_report.json"
            ),
            collection_started_ts=plan["freeze_created_ts"],
        )


def test_collection_plan_rejects_missing_historical_superiority() -> None:
    plan = _json("parallel_future_collection_plan.json")
    hashes = plan["lineage"]
    historical = plan["historical_replay_prerequisite"]
    plan["historical_replay_prerequisite"][
        "historical_superiority_gate_passed"
    ] = False
    with pytest.raises(ParallelFutureGateError, match="historical_superiority"):
        validate_parallel_future_collection_plan(
            plan,
            protocol_sha256=hashes["parallel_candidate_protocol_sha256"],
            candidate_contract_sha256s=hashes["candidate_contract_sha256s"],
            collector_protocol_sha256=hashes[
                "persistent_collector_protocol_sha256"
            ],
            feature_contract_sha256=hashes["feature_contract_sha256"],
            historical_gate_contract_sha256=historical[
                "gate_contract_sha256"
            ],
            historical_replay_report_sha256=historical["report_sha256"],
            historical_replay_report=_json(
                "historical_replay_superiority_report.json"
            ),
            collection_started_ts=plan["freeze_created_ts"] + 1,
        )


def test_same_target_free_window_is_frozen_for_all_candidates() -> None:
    freeze = _freeze()
    hashes = freeze["candidate_decision_streams"]
    assert freeze["shared_source_row_count"] == 45
    assert set(hashes) == set(_contracts())
    assert all(value["decision_count"] == 45 for value in hashes.values())
    assert freeze["outcomes_labels_settlement_returns_or_pnl_opened"] is False


def test_v8_1_abstention_cannot_be_silently_replaced_by_fallback() -> None:
    rows = _source_rows(1)
    decisions = {
        candidate_id: [
            _decision(
                rows[0],
                candidate_id,
                action="BUY_UP",
                origin="fallback_v6_7",
            )
        ]
        for candidate_id in _contracts()
    }
    with pytest.raises(ParallelFutureGateError, match="v8.1 no-fallback"):
        build_parallel_target_free_freeze(
            protocol=_json("parallel_candidate_protocol.json"),
            candidate_contracts=_contracts(),
            source_rows=rows,
            decisions_by_candidate=decisions,
            decision_freeze_created_ts=900_000,
            target_access_started=False,
        )


def test_target_field_before_freeze_fails_closed() -> None:
    rows = _source_rows(1)
    rows[0]["resolved_outcome"] = "UP"
    with pytest.raises(ParallelFutureGateError, match="target fields"):
        build_parallel_target_free_freeze(
            protocol=_json("parallel_candidate_protocol.json"),
            candidate_contracts=_contracts(),
            source_rows=rows,
            decisions_by_candidate={candidate: [] for candidate in _contracts()},
            decision_freeze_created_ts=900_000,
            target_access_started=False,
        )


def test_parallel_evaluation_is_single_use_and_reports_attribution() -> None:
    freeze = _freeze()
    result = evaluate_parallel_future_gate(
        protocol=_json("parallel_candidate_protocol.json"),
        freeze=freeze,
        settled_targets=_targets(freeze),
        evaluation_started_ts=2_000_000,
        consumed_freeze_sha256s=set(),
    )
    report = result["report"]
    assert report["candidate_metrics"]["v8_3_primary_with_fallback"]["fallback_count"] == 22
    assert report["candidate_gates"]["v8_1_primary_no_fallback"]["status"] == "evaluated"
    assert (
        report["candidate_gates"]["v8_1_primary_no_fallback"][
            "candidate_minus_baseline_largest_winner_removed_after_cost_pnl"
        ]
        == 0.0
    )
    assert report["multiplicity_aware_selected_candidate"] is None
    assert report["promotion_unlocked"] is False
    with pytest.raises(ParallelFutureGateError, match="already consumed"):
        evaluate_parallel_future_gate(
            protocol=_json("parallel_candidate_protocol.json"),
            freeze=freeze,
            settled_targets=_targets(freeze),
            evaluation_started_ts=2_000_000,
            consumed_freeze_sha256s={freeze["freeze_sha256"]},
        )


def test_freeze_hash_tamper_and_target_grid_change_fail_closed() -> None:
    freeze = _freeze()
    tampered = copy.deepcopy(freeze)
    tampered["shared_source_rows"][0]["feature_score"] = 0.6
    with pytest.raises(ParallelFutureGateError, match="hash mismatch"):
        evaluate_parallel_future_gate(
            protocol=_json("parallel_candidate_protocol.json"),
            freeze=tampered,
            settled_targets=_targets(freeze),
            evaluation_started_ts=2_000_000,
            consumed_freeze_sha256s=set(),
        )
    with pytest.raises(ParallelFutureGateError, match="target grid"):
        evaluate_parallel_future_gate(
            protocol=_json("parallel_candidate_protocol.json"),
            freeze=freeze,
            settled_targets=_targets(freeze)[:-1],
            evaluation_started_ts=2_000_000,
            consumed_freeze_sha256s=set(),
        )


def test_insufficient_support_is_explicit_and_does_not_unlock() -> None:
    freeze = _freeze(count=10)
    result = evaluate_parallel_future_gate(
        protocol=_json("parallel_candidate_protocol.json"),
        freeze=freeze,
        settled_targets=_targets(freeze),
        evaluation_started_ts=2_000_000,
        consumed_freeze_sha256s=set(),
    )
    for candidate_id in ("v8_1_primary_no_fallback", "v8_3_primary_with_fallback"):
        assert result["report"]["candidate_gates"][candidate_id]["status"] == (
            "insufficient_support"
        )
        assert result["report"]["candidate_gates"][candidate_id][
            "all_hard_gates_passed"
        ] is False


def test_legacy_consumed_window_adapter_separates_primary_and_fallback() -> None:
    overlays = [
        {
            "market_id": "fallback-market",
            "decision_ts": 1_000,
            "overlay_decision_id": "fallback",
            "original_v8_1_action": "NO_TRADE",
            "original_v8_1_side": "NONE",
            "original_v8_1_guard_allowed": False,
            "selected_action": "BUY_DOWN",
            "selected_side": "DOWN",
            "selection_source": "v6_7_non_risk_abstention_fallback",
            "execution_guard_order_allowed": True,
            "fallback_applied": True,
            "original_v6_7_action": "BUY_DOWN",
            "original_v6_7_side": "DOWN",
            "original_v6_7_guard_allowed": True,
        },
        {
            "market_id": "primary-market",
            "decision_ts": 2_000,
            "overlay_decision_id": "primary",
            "original_v8_1_action": "BUY_UP",
            "original_v8_1_side": "UP",
            "original_v8_1_guard_allowed": True,
            "selected_action": "BUY_UP",
            "selected_side": "UP",
            "selection_source": "v8_1_primary",
            "execution_guard_order_allowed": True,
            "fallback_applied": False,
            "original_v6_7_action": "BUY_DOWN",
            "original_v6_7_side": "DOWN",
            "original_v6_7_guard_allowed": True,
        },
    ]
    candidate_targets = [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "action": row["selected_action"],
            "paper_position_size": 1.0,
            "runtime_policy_after_cost_net_pnl_per_contract": 0.1,
        }
        for row in overlays
    ]
    baseline_targets = [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "action": "BUY_DOWN",
            "paper_position_size": 1.0,
            "runtime_policy_after_cost_net_pnl_per_contract": 0.1,
        }
        for row in overlays
    ]
    inputs = build_legacy_v8_3_smoke_inputs(
        overlay_rows=overlays,
        candidate_target_rows=candidate_targets,
        baseline_target_rows=baseline_targets,
    )
    freeze = build_parallel_target_free_freeze(
        protocol=_json("parallel_candidate_protocol.json"),
        candidate_contracts=_contracts(),
        source_rows=inputs["source_rows"],
        decisions_by_candidate=inputs["decisions_by_candidate"],
        decision_freeze_created_ts=900,
        target_access_started=False,
    )
    report = evaluate_parallel_future_gate(
        protocol=_json("parallel_candidate_protocol.json"),
        freeze=freeze,
        settled_targets=inputs["settled_targets"],
        evaluation_started_ts=3_000,
        consumed_freeze_sha256s=set(),
    )["report"]
    assert report["candidate_metrics"]["v8_1_primary_no_fallback"][
        "accepted_bet_count"
    ] == 1
    assert report["candidate_metrics"]["v8_3_primary_with_fallback"][
        "fallback_count"
    ] == 1
    assert report["candidate_metrics"]["matched_frozen_v6_7"][
        "accepted_bet_count"
    ] == 2
